"""D07 acceptance: the Elasticsearch loader (monthly indices, alias, template).

Needs the ``es`` compose profile (``python scripts/dev.py up es``)::

    python -m pytest tests/integration/test_seed_es.py --require-services -q

The fixture loads the **default** world (not the small one) on purpose: the late-arrival plant
is rare (1% of rows get a lag of up to :data:`mdqm.testing.world.WorldConfig.max_lag_seconds`,
and only the last hours of a month can cross into the next one), so the small world usually
contains none. With the default seed the world holds exactly the rows whose
``insertion_time`` month differs from their ``start_time`` month, and the test asserts the
cluster agrees with that number instead of asserting "some rows exist".

Everything asserted here is read back from the cluster: a count from the loader's own report
proves what the loader intended, not what Elasticsearch stored.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
import pytest

from mdqm.testing.loaders.elastic import (
    ALIAS,
    INDEX_MONTHS,
    INDEX_PREFIX,
    INDEX_TEMPLATE,
    EsAdmin,
    EsLoadReport,
    EsTarget,
    bulk_body,
    document_for,
    index_for,
    index_months,
    index_template_body,
    load,
    mismatched_events,
    month_start,
    shift_months,
    target_from_environ,
)
from mdqm.testing.world import ANCHOR, World, WorldConfig, build_world


@dataclass(frozen=True, slots=True)
class Seed:
    """One loading session: the world, the target, the two reports and what the cluster says."""

    target: EsTarget
    world: World
    first: EsLoadReport
    second: EsLoadReport
    counts_first: dict[str, int]
    counts_second: dict[str, int]
    template: dict[str, Any]
    alias_indices: tuple[str, ...]
    write_index: str | None


def _unavailable(request: pytest.FixtureRequest, reason: str) -> None:
    """Skip, or fail when ``--require-services`` was given."""
    if request.config.getoption("--require-services"):
        pytest.fail(f"required service unavailable: {reason}")
    pytest.skip(f"service unavailable: {reason}")


def _index_counts(admin: EsAdmin, indices: tuple[str, ...]) -> dict[str, int]:
    """``_count`` per monthly index, refreshed first so a just-bulk-loaded doc is visible."""
    admin.refresh()
    return {name: admin.count(name) for name in indices}


@pytest.fixture(scope="module")
def seed(request: pytest.FixtureRequest) -> Iterator[Seed]:
    """Load the default world twice (the second run is the idempotency check)."""
    try:
        target = target_from_environ()
    except Exception as exc:  # noqa: BLE001 - any settings failure means "not configured here"
        _unavailable(request, f"Elasticsearch connection settings: {exc}")
        raise AssertionError("unreachable") from exc

    try:
        response = httpx.get(f"{target.base_url}/", timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        _unavailable(request, f"Elasticsearch at {target.base_url}: {exc}")
        raise AssertionError("unreachable") from exc

    world = build_world(WorldConfig())
    first = load(target, world=world)
    counts_first = _index_counts(EsAdmin(target), first.indices)
    second = load(target, world=world)
    counts_second = _index_counts(EsAdmin(target), second.indices)

    with EsAdmin(target) as admin:
        template = admin.index_template(INDEX_TEMPLATE) or {}
        alias_indices = admin.alias_indices()
        write_index = admin.write_index()

    yield Seed(
        target=target,
        world=world,
        first=first,
        second=second,
        counts_first=counts_first,
        counts_second=counts_second,
        template=template,
        alias_indices=alias_indices,
        write_index=write_index,
    )

    with EsAdmin(target) as admin:
        admin.delete_indices(admin.existing_indices(f"{INDEX_PREFIX}-*"))


# --------------------------------------------------------------------------------------
# Acceptance 1: indices and alias exist; document counts match the world
# --------------------------------------------------------------------------------------


def _expected_month_counts(world: World) -> dict[str, int]:
    """Rows per event month, straight from the world (ground truth, not the loader)."""
    counts: dict[str, int] = {}
    for event in world.billing.usage_events:
        name = index_for(event.start_time)
        counts[name] = counts.get(name, 0) + 1
    return counts


def test_loader_twice_yields_identical_document_counts(seed: Seed) -> None:
    """Delete-and-reload plus stable ids: the same world gives the same counts every time."""
    assert seed.counts_first == seed.counts_second, "the second load changed the counts"
    assert seed.first.counts == seed.counts_first
    assert seed.first.total == seed.world.config.event_count
    assert seed.second.total == seed.world.config.event_count

    expected = _expected_month_counts(seed.world)
    assert seed.counts_first == {name: expected.get(name, 0) for name in seed.first.indices}
    assert set(expected) <= set(seed.counts_first), "a month with rows has no index"


def test_at_least_twelve_monthly_indices_spanning_the_window(seed: Seed) -> None:
    """12+ indices, contiguous, ending at the anchor month: the empty months exist too."""
    indices = seed.first.indices
    assert len(indices) >= INDEX_MONTHS
    expected = index_months(seed.world.billing.usage_events, seed.world.config.anchor)
    assert indices == expected
    # Contiguous: walking month by month from the first name reaches the last one.
    first_month = datetime.strptime(indices[0][len(INDEX_PREFIX) + 1 :], "%Y-%m")
    months: list[str] = []
    for offset in range(len(indices)):
        months.append(index_for(shift_months(first_month, offset)))
    assert tuple(months) == indices
    assert indices[-1] == index_for(seed.world.config.anchor)
    # The empty months really are empty, and they really do exist.
    assert seed.counts_first[indices[0]] == 0


def test_template_exists_and_maps_the_fields_the_adapter_needs(seed: Seed) -> None:
    """The installed template maps dates, keywords and numerics explicitly (dynamic strict)."""
    assert seed.template, f"template {INDEX_TEMPLATE} is missing"
    templates: list[dict[str, Any]] = seed.template["index_templates"]
    assert len(templates) == 1
    assert templates[0]["name"] == INDEX_TEMPLATE
    body = templates[0]["index_template"]
    assert body["index_patterns"] == [f"{INDEX_PREFIX}-*"]
    mappings = body["template"]["mappings"]
    assert mappings["dynamic"] == "strict"
    properties = mappings["properties"]
    assert properties["start_time"] == {"type": "date"}
    assert properties["insertion_time"] == {"type": "date"}
    assert properties["client_no"] == {"type": "keyword"}
    assert properties["client_id"] == {"type": "keyword"}
    assert properties["event_id"] == {"type": "long"}
    assert properties["units"] == {"type": "integer"}
    # Elasticsearch normalises settings under `index` and renders the numbers as strings, so
    # assert the effective values rather than the bytes the loader sent.
    settings = body["template"]["settings"]["index"]
    assert int(settings["number_of_shards"]) == 1
    assert int(settings["number_of_replicas"]) == 0


def test_alias_points_at_every_month_with_one_write_index(seed: Seed) -> None:
    """``GET {alias}/_alias`` names every monthly index; exactly the newest is the write one."""
    assert seed.alias_indices == tuple(sorted(seed.first.indices))
    assert seed.write_index == seed.first.indices[-1]
    response = httpx.get(f"{seed.target.base_url}/{ALIAS}/_alias", timeout=10.0)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    assert tuple(sorted(payload)) == seed.alias_indices
    write_flags = [
        bool(meta["aliases"][ALIAS].get("is_write_index", False)) for meta in payload.values()
    ]
    assert sum(write_flags) == 1


# --------------------------------------------------------------------------------------
# Acceptance 2: a date-range query on a single month touches one index via index pattern
# --------------------------------------------------------------------------------------


def test_date_range_query_on_one_month_touches_one_index(seed: Seed) -> None:
    """A range on ``start_time`` inside one month reads one index, and returns all its rows."""
    index = "mdqm-usage-events-2025-11"
    month = datetime.strptime(index[len(INDEX_PREFIX) + 1 :], "%Y-%m").replace(tzinfo=ANCHOR.tzinfo)
    body: dict[str, Any] = {
        "size": 0,
        "track_total_hits": True,
        "query": {
            "range": {
                "start_time": {
                    "gte": month.isoformat(),
                    "lt": shift_months(month, 1).isoformat(),
                }
            }
        },
    }
    with EsAdmin(seed.target) as admin:
        hits = admin.search(body, index=index)
        pattern_hits = admin.search(body, index=f"{INDEX_PREFIX}-*")
        alias_hits = admin.search(body)
    total = int(hits["hits"]["total"]["value"])
    expected = _expected_month_counts(seed.world).get(index, 0)
    assert expected > 0
    assert total == expected
    # One explicit index name resolves to its single shard; the wildcard pattern expands to
    # every month, yet the range filter still returns exactly that month's rows.
    assert hits["_shards"]["total"] == 1
    assert int(pattern_hits["hits"]["total"]["value"]) == total
    assert int(alias_hits["hits"]["total"]["value"]) == total


# --------------------------------------------------------------------------------------
# Late arrivals: the document lives in its event month, its insertion time does not
# --------------------------------------------------------------------------------------


def _es_mismatch_count(admin: EsAdmin, months: tuple[str, ...]) -> int:
    """Count docs whose ``insertion_time`` is in a later month than their index month."""
    total = 0
    for name in months:
        month = datetime.strptime(name[len(INDEX_PREFIX) + 1 :], "%Y-%m").replace(
            tzinfo=ANCHOR.tzinfo
        )
        body: dict[str, Any] = {
            "size": 0,
            "track_total_hits": True,
            "query": {"range": {"insertion_time": {"gte": shift_months(month, 1).isoformat()}}},
        }
        total += int(admin.search(body, index=name)["hits"]["total"]["value"])
    return total


def test_late_arrivals_are_in_their_event_month(seed: Seed) -> None:
    """Late rows exist, live in the event-month index, and keep their later insertion time."""
    late = mismatched_events(seed.world.billing.usage_events)
    assert late, "the default world plants late-arriving rows; none were found"

    with EsAdmin(seed.target) as admin:
        assert _es_mismatch_count(admin, seed.first.indices) == len(late)
        for event in late:
            start_month = shift_months(month_start(event.start_time), 1)
            assert event.insertion_time >= start_month
            body: dict[str, Any] = {
                "size": 1,
                "query": {"term": {"event_id": event.event_id}},
            }
            hit = admin.search(body, index=index_for(event.start_time))
            assert hit["hits"]["hits"], f"event {event.event_id} is not in its event month"
            source = hit["hits"]["hits"][0]["_source"]
            assert hit["hits"]["hits"][0]["_index"] == index_for(event.start_time)
            assert source["insertion_time"] == event.insertion_time.isoformat()
            assert source["start_time"] == event.start_time.isoformat()
            assert source["insertion_time"][:7] != source["start_time"][:7]


def test_planted_orphan_client_numbers_are_loaded(seed: Seed) -> None:
    """The world's orphan ``client_no`` rows reach the cluster unchanged (no join on load)."""
    known = {account.client_no for account in seed.world.billing.accounts}
    orphans = [event for event in seed.world.billing.usage_events if event.client_no not in known]
    assert len(orphans) == seed.world.expected.orphan_event_count
    assert orphans
    body: dict[str, Any] = {
        "size": 0,
        "track_total_hits": True,
        "query": {"terms": {"client_no": sorted({event.client_no for event in orphans})}},
    }
    with EsAdmin(seed.target) as admin:
        found = admin.search(body)
    assert int(found["hits"]["total"]["value"]) == len(orphans)


def test_stable_document_ids_and_deterministic_body(seed: Seed) -> None:
    """The ``_id`` is the event id, so a reload overwrites instead of duplicating."""
    with EsAdmin(seed.target) as admin:
        for event in seed.world.billing.usage_events[:5]:
            body: dict[str, Any] = {"query": {"term": {"event_id": event.event_id}}}
            hit = admin.search(body, index=index_for(event.start_time))
            assert hit["hits"]["hits"][0]["_id"] == str(event.event_id)
    # The pure builder gives the same bytes for the same rows, in any input order.
    rows = seed.world.billing.usage_events[:20]
    assert bulk_body(rows) == bulk_body(tuple(reversed(rows)))
    assert document_for(rows[0])["event_id"] == rows[0].event_id
    assert index_template_body(seed.world.config.seed)["index_patterns"] == [f"{INDEX_PREFIX}-*"]


def test_search_view_cannot_write(seed: Seed) -> None:
    """The adapter's object has one operation; there is no write path to mistake for one."""
    view = seed.target.search_view()
    # `track_total_hits` is required for an exact total: without it Elasticsearch caps the
    # count at 10000 and a 50k-world assertion would silently pass on a wrong number.
    body: dict[str, Any] = {"size": 0, "track_total_hits": True, "query": {"match_all": {}}}
    try:
        payload = view.search(body)
    finally:
        view.close()
    assert int(payload["hits"]["total"]["value"]) == seed.world.config.event_count
    assert view.url.endswith(f"/{ALIAS}/_search")
    public = {name for name in dir(view) if not name.startswith("_")}
    assert public == {"search", "close", "url"}
    for forbidden in ("bulk", "index", "delete", "put_template", "update_alias", "create_index"):
        assert not hasattr(view, forbidden), f"the search view exposes {forbidden}"


def test_every_month_has_at_least_one_document_or_is_deliberately_empty(seed: Seed) -> None:
    """No index is silently missing: every month is either empty by design or counted."""
    assert set(seed.first.counts) == set(seed.first.indices)
    assert sum(seed.first.counts.values()) == seed.world.config.event_count
    window = {index_for(event.start_time) for event in seed.world.billing.usage_events}
    for name in seed.first.indices:
        if name in window:
            assert seed.first.counts[name] > 0
        else:
            assert seed.first.counts[name] == 0
