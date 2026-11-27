"""Unit tests for the Elasticsearch loader (D07): builders, targets, and the read-only view.

No Docker and no network: everything here is a pure function, a dataclass, or an httpx
``MockTransport``. The live acceptance checks live in ``tests/integration/test_seed_es.py``.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from mdqm.testing.loaders.elastic import (
    ALIAS,
    BULK_BATCH,
    INDEX_MONTHS,
    INDEX_PREFIX,
    EsAdmin,
    EsSearchView,
    EsTarget,
    LoaderError,
    batches,
    bulk_body,
    document_for,
    index_for,
    index_months,
    index_template_body,
    mismatched_events,
    month_start,
    read_dotenv,
    shift_months,
    target_from_environ,
)
from mdqm.testing.world import ANCHOR, UsageEvent, WorldConfig, build_world


def _event(
    *,
    event_id: int = 1,
    client_no: str = "0000000001",
    start: datetime = datetime(2025, 11, 30, 23, 0, tzinfo=UTC),
    lag: timedelta = timedelta(seconds=30),
    units: int = 7,
    corrects: int | None = None,
) -> UsageEvent:
    return UsageEvent(
        event_id=event_id,
        client_no=client_no,
        start_time=start,
        insertion_time=start + lag,
        units=units,
        corrects_event_id=corrects,
    )


# --------------------------------------------------------------------------------------
# Index naming and the monthly span
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (datetime(2025, 5, 1, tzinfo=UTC), "mdqm-usage-events-2025-05"),
        (datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC), "mdqm-usage-events-2025-12"),
        (datetime(2026, 1, 1, tzinfo=UTC), "mdqm-usage-events-2026-01"),
        # A non-UTC instant belongs to the UTC month, not to its local one.
        (datetime(2025, 6, 1, 2, 0, tzinfo=UTC).astimezone(), "mdqm-usage-events-2025-06"),
    ],
    ids=["may", "december-end", "january", "tz-aware"],
)
def test_index_for_uses_the_utc_event_month(moment: datetime, expected: str) -> None:
    assert index_for(moment) == expected


def test_index_for_converts_a_non_utc_instant_to_utc() -> None:
    """01:30 on 1 June in UTC+3 is still 31 May in UTC, so it belongs to the May index."""
    from datetime import timezone

    instant = datetime(2025, 6, 1, 1, 30, tzinfo=timezone(timedelta(hours=3)))
    assert instant.isoformat() == "2025-06-01T01:30:00+03:00"
    assert index_for(instant) == "mdqm-usage-events-2025-05"


def test_shift_months_crosses_a_year() -> None:
    assert shift_months(datetime(2026, 1, 1, tzinfo=UTC), -1) == datetime(2025, 12, 1, tzinfo=UTC)
    assert shift_months(datetime(2025, 11, 15, tzinfo=UTC), 3) == datetime(2026, 2, 15, tzinfo=UTC)


def test_index_months_is_a_contiguous_span_ending_at_the_anchor_month() -> None:
    names = index_months((_event(),), ANCHOR)
    assert len(names) == INDEX_MONTHS
    assert names == tuple(sorted(names))
    assert names[-1] == "mdqm-usage-events-2026-01"
    first = datetime.strptime(names[0][len(INDEX_PREFIX) + 1 :], "%Y-%m")
    assert tuple(index_for(shift_months(first, offset)) for offset in range(len(names))) == names
    assert "mdqm-usage-events-2025-11" in names


def test_index_months_keeps_a_data_month_outside_the_span() -> None:
    """A longer world window must never lose rows to the "empty months" span."""
    old = _event(start=datetime(2024, 3, 15, tzinfo=UTC))
    names = index_months((_event(), old), ANCHOR)
    assert "mdqm-usage-events-2024-03" in names
    assert len(names) == INDEX_MONTHS + 1


def test_index_months_rejects_a_nonsense_count() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        index_months((_event(),), ANCHOR, count=0)


def test_month_start_truncates_to_the_first_instant() -> None:
    assert month_start(datetime(2025, 12, 31, 18, 4, tzinfo=UTC)) == datetime(
        2025, 12, 1, tzinfo=UTC
    )


# --------------------------------------------------------------------------------------
# Template and document bodies
# --------------------------------------------------------------------------------------


def test_template_maps_dates_keywords_and_numbers() -> None:
    body = index_template_body(seed=7)
    assert body["index_patterns"] == [f"{INDEX_PREFIX}-*"]
    assert body["_meta"]["seed"] == 7
    template = body["template"]
    assert template["aliases"] == {ALIAS: {"is_write_index": False}}
    assert template["settings"]["number_of_shards"] == 1
    assert template["settings"]["number_of_replicas"] == 0
    properties = template["mappings"]["properties"]
    assert properties["start_time"] == {"type": "date"}
    assert properties["insertion_time"] == {"type": "date"}
    assert properties["client_no"] == {"type": "keyword"}
    assert properties["client_id"] == {"type": "keyword"}
    assert properties["event_id"] == {"type": "long"}
    assert properties["units"] == {"type": "integer"}
    assert template["mappings"]["dynamic"] == "strict"


def test_template_is_deterministic() -> None:
    assert json.dumps(index_template_body(1), sort_keys=True) == json.dumps(
        index_template_body(1), sort_keys=True
    )


def test_document_keeps_both_timestamps_apart() -> None:
    event = _event(lag=timedelta(hours=5), corrects=42)
    document = document_for(event)
    assert document["event_id"] == event.event_id
    assert document["client_no"] == event.client_no
    assert document["start_time"] == "2025-11-30T23:00:00+00:00"
    assert document["insertion_time"] == "2025-12-01T04:00:00+00:00"
    assert document["units"] == event.units
    assert document["corrects_event_id"] == 42
    assert set(document) == {
        "event_id",
        "client_no",
        "start_time",
        "insertion_time",
        "units",
        "corrects_event_id",
    }


# --------------------------------------------------------------------------------------
# Bulk NDJSON: deterministic bytes, stable ids, pairs
# --------------------------------------------------------------------------------------


def _bulk_lines(ndjson: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in ndjson.splitlines()]


def test_bulk_body_is_action_document_pairs_with_a_stable_id() -> None:
    events = (_event(event_id=2, start=datetime(2025, 12, 2, tzinfo=UTC)), _event(event_id=1))
    lines = _bulk_lines(bulk_body(events))
    assert len(lines) == 2 * len(events)
    for index, line in enumerate(lines):
        if index % 2 == 0:
            meta = line["index"]
            assert meta["_index"] == index_for(events[0].start_time) or meta["_index"].startswith(
                INDEX_PREFIX
            )
            assert meta["_id"] in {"1", "2"}
        else:
            assert line["event_id"] in {1, 2}
    assert [lines[0]["index"]["_id"], lines[2]["index"]["_id"]] == ["1", "2"]


def test_bulk_body_is_byte_identical_for_the_same_world_and_order_independent() -> None:
    """Same seed, same bytes; the input order is sorted away, so it cannot leak in."""
    world_a = build_world(WorldConfig.small())
    world_b = build_world(WorldConfig.small())
    events_a = world_a.billing.usage_events
    events_b = world_b.billing.usage_events
    assert bulk_body(events_a) == bulk_body(events_b)
    assert bulk_body(events_a) == bulk_body(tuple(reversed(events_a)))

    lines = _bulk_lines(bulk_body(events_a[:10]))
    assert lines[0]["index"]["_id"] == str(events_a[:10][0].event_id)
    for pair in range(0, len(lines), 2):
        action = lines[pair]["index"]
        document = lines[pair + 1]
        assert action["_index"] == index_for(datetime.fromisoformat(document["start_time"]))
        assert action["_id"] == str(document["event_id"])


def test_bulk_body_is_empty_for_no_rows() -> None:
    assert bulk_body(()) == ""


def test_batches_cover_every_row_exactly_once() -> None:
    events = tuple(_event(event_id=i) for i in range(1, 12))
    chunks = list(batches(events, size=5))
    assert [len(chunk) for chunk in chunks] == [5, 5, 1]
    assert tuple(event.event_id for chunk in chunks for event in chunk) == tuple(
        event.event_id for event in events
    )
    assert list(batches((), size=BULK_BATCH)) == []


def test_batches_rejects_a_nonsense_size() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        list(batches((_event(),), size=0))


# --------------------------------------------------------------------------------------
# Late arrivals
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lag", "expected"),
    [
        (timedelta(seconds=1), 0),
        (timedelta(hours=5), 1),
        (timedelta(seconds=0), 0),
    ],
    ids=["same-month", "crosses-into-december", "instant"],
)
def test_mismatched_events_counts_a_cross_month_lag(lag: timedelta, expected: int) -> None:
    event = _event(start=datetime(2025, 11, 30, 23, 0, tzinfo=UTC), lag=lag)
    assert len(mismatched_events((event,))) == expected


def test_mismatched_events_is_empty_when_every_row_is_prompt() -> None:
    events = (
        _event(event_id=1, start=datetime(2025, 11, 2, tzinfo=UTC)),
        _event(event_id=2, start=datetime(2025, 11, 20, 12, tzinfo=UTC)),
    )
    assert mismatched_events(events) == ()
    assert mismatched_events(()) == ()


# --------------------------------------------------------------------------------------
# Target and credentials
# --------------------------------------------------------------------------------------


def test_target_from_environment_prefers_the_process_environment(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("MDQM_ES_PORT=6000\nMDQM_ES_HOST=from-dotenv\n", encoding="utf-8")
    target = target_from_environ(
        environ={"MDQM_ES_PORT": "6123", "MDQM_ES_HOST": "from-env"}, dotenv_path=dotenv
    )
    assert target.port == 6123
    assert target.host == "from-env"
    assert target.base_url == "http://from-env:6123"
    assert target.admin_auth is None


def test_target_from_dotenv_and_optional_credentials(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "MDQM_ES_PORT=59200\n"
        "MDQM_ES_ADMIN_USER=admin\n"
        "MDQM_ES_ADMIN_PASSWORD='secret'\n"
        "MDQM_ES_SEARCH_USER=searcher\n",
        encoding="utf-8",
    )
    target = target_from_environ(environ={}, dotenv_path=dotenv)
    assert target.port == 59200
    assert target.admin_auth == ("admin", "secret")
    assert target.search_auth == ("searcher", "")


def test_target_rejects_a_non_integer_port() -> None:
    with pytest.raises(LoaderError, match="MDQM_ES_PORT must be an integer"):
        target_from_environ(environ={"MDQM_ES_PORT": "not-a-port"}, dotenv_path=Path("/none"))


def test_read_dotenv_ignores_comments_and_quotes(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("# comment\nA=1\nB = 'two'\n\nC=\"3\"\n", encoding="utf-8")
    assert read_dotenv(path) == {"A": "1", "B": "two", "C": "3"}
    assert read_dotenv(tmp_path / "missing.env") == {}


def test_search_view_is_search_only() -> None:
    """The adapter's object has one operation and no write path at all."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"hits": {"total": {"value": 3}}})

    view = EsSearchView("http://es.invalid:59200", ALIAS, transport=httpx.MockTransport(handler))
    try:
        payload = view.search({"size": 0})
        assert view.url == "http://es.invalid:59200/mdqm-usage-events/_search"
    finally:
        view.close()

    assert payload["hits"]["total"]["value"] == 3
    assert len(calls) == 1
    assert calls[0].method == "POST"
    assert calls[0].url.path == f"/{ALIAS}/_search"
    public = {name for name in dir(view) if not name.startswith("_")}
    assert public == {"search", "close", "url"}, public
    for forbidden in ("bulk", "index", "delete", "put_template", "create_index", "update_alias"):
        assert not hasattr(view, forbidden)


def test_search_view_raises_on_a_server_error() -> None:
    view = EsSearchView(
        "http://es.invalid:59200",
        ALIAS,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(400, json={"error": {"reason": "bad query"}})
        ),
    )
    try:
        with pytest.raises(LoaderError, match="HTTP 400"):
            view.search({"size": 0})
    finally:
        view.close()


def test_admin_bulk_raises_when_a_document_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "errors": True,
                "items": [
                    {
                        "index": {
                            "status": 400,
                            "error": {"type": "mapper_parsing_exception", "reason": "bad doc"},
                        }
                    }
                ],
            },
        )

    admin = EsAdmin(EsTarget(), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(LoaderError, match="mapper_parsing_exception"):
            admin.bulk('{"index": {"_index": "x", "_id": "1"}}\n{}\n')
    finally:
        admin.close()


def test_admin_counts_every_document_it_sent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"took": 1, "errors": False, "items": [{"index": {}}] * 3})

    admin = EsAdmin(EsTarget(), transport=httpx.MockTransport(handler))
    try:
        assert admin.bulk("{}\n{}\n{}\n{}\n{}\n{}\n") == 3
    finally:
        admin.close()


def test_admin_treats_a_missing_index_pattern_as_empty() -> None:
    admin = EsAdmin(
        EsTarget(), transport=httpx.MockTransport(lambda request: httpx.Response(404, json={}))
    )
    try:
        assert admin.existing_indices(f"{INDEX_PREFIX}-*") == ()
        assert admin.alias_indices() == ()
        assert admin.index_template() is None
    finally:
        admin.close()


def test_search_view_uses_the_search_account_not_the_admin_one() -> None:
    """Two accounts, kept in two fields: the view never sends the admin credential."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"hits": {"total": {"value": 0}}})

    target = EsTarget(
        admin_user="admin",
        admin_password="admin-secret",
        search_user="searcher",
        search_password="search-secret",
    )
    view = EsSearchView(
        target.base_url, ALIAS, auth=target.search_auth, transport=httpx.MockTransport(handler)
    )
    try:
        view.search({"size": 0})
    finally:
        view.close()
    assert target.admin_auth == ("admin", "admin-secret")
    assert target.search_auth == ("searcher", "search-secret")
    expected = "Basic " + base64.b64encode(b"searcher:search-secret").decode("ascii")
    assert seen[0].headers["authorization"] == expected
    assert "admin-secret" not in seen[0].headers["authorization"]
