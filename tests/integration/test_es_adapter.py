"""A09 acceptance: the Elasticsearch adapter against the live compose cluster.

Needs the ``es`` compose profile (``python scripts/dev.py up es``)::

    python -m pytest tests/integration/test_es_adapter.py --require-services -q

The fixture loads the **small** world itself, through the D07 loader, so this file proves the
adapter against a cluster it built rather than against whatever a previous run left behind. The
data is left in place afterwards: the loader is idempotent, and a loaded cluster is more useful to
the next task than an empty one.

Every assertion compares the adapter against something the adapter did not produce: the cluster's
own ``_count``, the matched total of the search response, the version ``GET /`` reports, or the
error Elasticsearch itself returns. A test that only re-read the adapter's own answer would pass
even if the adapter invented it.

The read-only proof is split in two layers, because "the write was refused" is only meaningful when
you say by whom:

* the **transport** (``SearchOnlyTransport``) refuses the method before a byte is sent — and the
  test then asks the cluster whether the index it tried to create exists (it does not);
* the **server** refuses ``GET /_query`` (ES|QL answers ``POST`` only, a real 405), which is why
  that one ``POST`` is the transport's single exception instead of a blanket "POST is fine".
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final, Protocol, runtime_checkable

import httpx
import pyarrow as pa
import pytest

from mdqm.adapters.base import OPTIONAL_METHODS, ArrowStream, NotSupported
from mdqm.adapters.elastic import (
    ESQL_PATH,
    ElasticAdapter,
    ElasticError,
    SearchOnlyTransport,
    month_of,
)
from mdqm.config.settings import SourceConfig
from mdqm.models import Fragment, Limits
from mdqm.testing.conformance import run_conformance
from mdqm.testing.loaders.elastic import (
    ALIAS,
    EsAdmin,
    EsLoadReport,
    EsTarget,
    load,
    target_from_environ,
)
from mdqm.testing.world import WorldConfig

LIMITS: Final = Limits(timeout_s=20.0, max_rows=2000, max_bytes=8_000_000, max_partitions=8)
"""Generous limits: only the caps under test are tight."""


@dataclass(frozen=True, slots=True)
class Cluster:
    """One loaded cluster plus the adapter pointed at it."""

    target: EsTarget
    report: EsLoadReport
    adapter: ElasticAdapter
    admin: EsAdmin

    def count(self, index: str) -> int:
        """The cluster's own document count for one index."""
        return self.admin.count(index)


@dataclass
class RecordingTransport(httpx.BaseTransport):
    """A real transport that remembers what was sent, so "no request" can be asserted."""

    requests: list[tuple[str, str]] = field(default_factory=list)
    _inner: httpx.HTTPTransport = field(default_factory=httpx.HTTPTransport)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()

    def searches(self) -> list[tuple[str, str]]:
        """Every request that would read or write documents."""
        return [
            (method, path)
            for method, path in self.requests
            if path.endswith("/_search") or path == ESQL_PATH
        ]


def _unavailable(request: pytest.FixtureRequest, reason: str) -> None:
    """Skip, or fail when ``--require-services`` was given."""
    if request.config.getoption("--require-services"):
        pytest.fail(f"required service unavailable: {reason}")
    pytest.skip(f"service unavailable: {reason}")


def _config(target: EsTarget, *, time_field: str = "start_time") -> SourceConfig:
    """The source config a deployment would use: one base URL and the field that decides a month."""
    return SourceConfig(
        name="usage_events_es",
        adapter="elastic",
        options={"base_url": target.base_url, "time_field": time_field},
    )


@pytest.fixture(scope="module")
def cluster(request: pytest.FixtureRequest) -> Iterator[Cluster]:
    """Load the small world through the D07 loader and point an adapter at the alias."""
    try:
        target = target_from_environ()
    except Exception as exc:
        _unavailable(request, f"Elasticsearch connection settings: {exc}")
        raise AssertionError("unreachable") from exc

    try:
        response = httpx.get(f"{target.base_url}/", timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        _unavailable(request, f"Elasticsearch at {target.base_url}: {exc}")
        raise AssertionError("unreachable") from exc

    report = load(target, WorldConfig.small())
    admin = EsAdmin(target)
    adapter = ElasticAdapter(_config(target))
    yield Cluster(target=target, report=report, adapter=adapter, admin=admin)
    adapter.close()
    admin.close()


def _next_month(moment: datetime) -> datetime:
    """First instant of the month after ``moment``."""
    return (
        moment.replace(year=moment.year + 1, month=1)
        if moment.month == 12
        else moment.replace(month=moment.month + 1)
    )


def _month_range(moment: datetime, *, months: int = 1) -> dict[str, Any]:
    """A Query DSL range over ``months`` whole calendar months starting at ``moment``."""
    end = moment
    for _ in range(months):
        end = _next_month(end)
    return {"range": {"start_time": {"gte": moment.isoformat(), "lt": end.isoformat()}}}


@runtime_checkable
class _EsRows(ArrowStream, Protocol):
    """What the Elasticsearch rows stream reports beyond the batches.

    ``ArrowStream`` -- the protocol ``execute`` declares -- carries batches and nothing
    else, but these tests check the pruning and the caps against what the cluster itself
    said about the request: the indices it touched and the total it matched.
    """

    index_target: str
    touched_indices: tuple[str, ...]
    total_hits: int
    dropped_rows: int
    esql: bool


def _es_rows(stream: ArrowStream) -> _EsRows:
    """Narrow one ``execute`` result to the provenance the adapter's stream also reports."""
    assert isinstance(stream, _EsRows), "the Elasticsearch stream must report its provenance"
    return stream


def _rows(stream: Any) -> int:
    """The row count of one ``execute`` result."""
    return sum(batch.num_rows for batch in stream)


def _filled(report: EsLoadReport) -> list[str]:
    """Indices that certainly hold documents (the empty leading months exist on purpose)."""
    filled = [name for name in report.indices if report.counts.get(name, 0) > 0]
    assert filled, f"the loader stored no documents at all: {report.counts}"
    return filled


def _month_fragment(adapter: ElasticAdapter, index: str, *, months: int = 1) -> Fragment:
    """A fragment reading one (or more) whole month(s) by time bound."""
    month = month_of(index)
    assert month is not None, f"{index} is not a monthly index"
    return adapter.fragment(ALIAS, query=_month_range(month, months=months))


def _esql_count(index: str) -> str:
    """An ES|QL statement counting one month, in the same window the DSL fragment uses."""
    month = month_of(index)
    assert month is not None, f"{index} is not a monthly index"
    end = _next_month(month)
    return (
        f'FROM "{ALIAS}" | WHERE start_time >= "{month.isoformat()}" '
        f'AND start_time < "{end.isoformat()}" | STATS n = COUNT(*)'
    )


# --------------------------------------------------------------------------------------
# Acceptance 1: the adapter is a real adapter
# --------------------------------------------------------------------------------------


def test_passes_the_conformance_kit(cluster: Cluster) -> None:
    """The one definition of "a real adapter" (``mdqm.testing.conformance``)."""
    adapter = cluster.adapter
    run_conformance(lambda: adapter, fragment=_month_fragment(adapter, _filled(cluster.report)[0]))


# --------------------------------------------------------------------------------------
# Acceptance 2: one month is one index, and the count is the cluster's own
# --------------------------------------------------------------------------------------


def test_one_month_filter_touches_exactly_one_index_and_matches_count(cluster: Cluster) -> None:
    """``_shards`` and the row count prove the pruning; ``_count`` proves the answer."""
    index = _filled(cluster.report)[0]
    expected = cluster.count(index)
    stream = _es_rows(cluster.adapter.execute(_month_fragment(cluster.adapter, index), LIMITS))
    rows = _rows(stream)

    assert stream.touched_indices == (index,), (
        f"a one-month window touched {stream.touched_indices}, not exactly {index}"
    )
    assert stream.index_target == index
    assert stream.total_hits == expected, "the cluster's own match count is the answer"
    assert rows == expected, f"search returned {rows} rows, _count says {expected}"
    assert rows > 0, "an empty month proves nothing"


def test_max_rows_truncates_exactly(cluster: Cluster) -> None:
    """``Limits.max_rows`` is enforced here, and the total still says what matched."""
    index = _filled(cluster.report)[0]
    expected = cluster.count(index)
    assert expected > 37, "pick a fixture month larger than the cap under test"

    tight = LIMITS.model_copy(update={"max_rows": 37})
    stream = _es_rows(cluster.adapter.execute(_month_fragment(cluster.adapter, index), tight))

    assert _rows(stream) == 37
    assert stream.dropped_rows == 0, "the cap is applied by the request size, not by trimming"
    assert stream.total_hits == expected


def test_request_against_the_alias_reads_across_monthly_indices(cluster: Cluster) -> None:
    """The alias is the table: a two-month window reads both months, and only those."""
    filled = _filled(cluster.report)
    assert len(filled) >= 2, "the small world must fill at least two months"
    first, second = filled[0], filled[1]
    expected = cluster.count(first) + cluster.count(second)

    stream = _es_rows(
        cluster.adapter.execute(_month_fragment(cluster.adapter, first, months=2), LIMITS)
    )

    assert stream.touched_indices == (first, second)
    assert _rows(stream) == expected == stream.total_hits


def test_no_time_bound_over_a_many_index_family_is_refused_before_any_request(
    cluster: Cluster,
) -> None:
    """An unbounded family read is refused with a fix, and not one search leaves the adapter."""
    recorder = RecordingTransport()
    adapter = ElasticAdapter(_config(cluster.target), transport=recorder)
    try:
        with pytest.raises(ElasticError) as caught:
            adapter.execute(adapter.fragment(ALIAS), LIMITS)
        assert caught.value.code == "PARTITION_SCAN_ALL"
        assert "start_time" in caught.value.response.fix
        assert recorder.searches() == [], "the refusal must happen before any search is sent"
    finally:
        adapter.close()


def test_a_range_over_too_many_months_is_refused(cluster: Cluster) -> None:
    """``max_partitions`` is a real gate: a year-long window over twelve indices is refused."""
    first = _filled(cluster.report)[0]
    window = _month_range(month_of(first) or datetime(2025, 1, 1, tzinfo=UTC), months=12)
    with pytest.raises(ElasticError) as caught:
        cluster.adapter.execute(cluster.adapter.fragment(ALIAS, query=window), LIMITS)
    assert caught.value.code == "COST_TOO_HIGH"


# --------------------------------------------------------------------------------------
# Acceptance 3: ES|QL when the server has it, DSL when it does not
# --------------------------------------------------------------------------------------


def test_probe_reports_the_version_and_esql_truthfully(cluster: Cluster) -> None:
    """The probe's version is the cluster's, and its ES|QL verdict matches ``POST /_query``."""
    version = httpx.get(f"{cluster.target.base_url}/", timeout=5.0).json()["version"]["number"]
    capabilities = cluster.adapter.probe()
    answered = httpx.post(
        f"{cluster.target.base_url}{ESQL_PATH}",
        content='{"query": "ROW mdqm_probe = 1"}',
        headers={"content-type": "application/json"},
        timeout=5.0,
    )
    features = cluster.adapter.features()

    assert capabilities.version == version
    assert capabilities.adapter == cluster.adapter.kind == "elastic"
    assert features.esql is (answered.status_code == httpx.codes.OK)
    assert features.aggregations is True
    # Only capability names from OPTIONAL_METHODS: an invented name would let a feature claim a
    # fallback that does not exist.
    assert set(capabilities.capabilities) <= set(OPTIONAL_METHODS.values())
    assert capabilities.supports(OPTIONAL_METHODS["extract_partitions"]) is True
    assert capabilities.supports(OPTIONAL_METHODS["mine_queries"]) is False
    assert capabilities.supports(OPTIONAL_METHODS["mutability_stats"]) is False


def test_esql_path_and_dsl_path_answer_the_same_count(cluster: Cluster) -> None:
    """With ``_query`` present the fragment's ES|QL runs; with it stubbed away the DSL does."""
    index = _filled(cluster.report)[0]
    expected = cluster.count(index)
    month = month_of(index)
    assert month is not None
    fragment = cluster.adapter.fragment(ALIAS, query=_month_range(month), esql=_esql_count(index))

    assert cluster.adapter.features().esql is True
    esql_stream = _es_rows(cluster.adapter.execute(fragment, LIMITS))
    batches = [batch.to_pydict() for batch in esql_stream]
    assert esql_stream.esql is True
    assert batches[0]["n"] == [expected], "ES|QL must count the same documents"

    dsl_only = ElasticAdapter(_config(cluster.target), esql_support=False)
    try:
        dsl_stream = _es_rows(dsl_only.execute(fragment, LIMITS))
        assert dsl_stream.esql is False, "a cluster without ES|QL must fall back to the DSL"
        assert _rows(dsl_stream) == expected
    finally:
        dsl_only.close()


def test_explain_reports_the_index_set_and_an_unknown_proof(cluster: Cluster) -> None:
    """No EXPLAIN on Elasticsearch: the plan is the index set, and the proof stays unknown."""
    index = _filled(cluster.report)[0]
    plan = cluster.adapter.explain(_month_fragment(cluster.adapter, index))

    assert plan.indexes_used == (index,)
    assert plan.partitions_scanned == 1
    assert plan.partitions_total == len(cluster.report.indices)
    assert plan.uses_index is True
    assert plan.pruning_proof.value == "unknown"


# --------------------------------------------------------------------------------------
# Acceptance 4: catalog and partitions
# --------------------------------------------------------------------------------------


def test_the_alias_is_the_table_and_the_monthly_indices_are_its_partitions(
    cluster: Cluster,
) -> None:
    """Twelve indices behind one alias: the agent sees one table with monthly partitions."""
    snapshot = cluster.adapter.extract_catalog()
    table = snapshot.table(ALIAS)

    assert table is not None, (
        f"the alias must be the user-facing table, got {[t.name for t in snapshot.tables]}"
    )
    assert table.row_estimate == cluster.report.total
    mapped = {column.name: column.native_type for column in table.columns}
    assert mapped["start_time"] == "date"
    assert mapped["client_no"] == "keyword"
    assert mapped["units"] == "integer"
    assert not [t for t in snapshot.tables if t.name.startswith(f"{ALIAS}-")], (
        "the monthly indices are partitions, not tables"
    )

    partitions = cluster.adapter.extract_partitions()
    assert len(partitions) == len(snapshot.tables)
    alias_partitioning = next(p for p in partitions if p.key)
    assert alias_partitioning.key == ("start_time",)
    assert alias_partitioning.granularity is not None
    assert alias_partitioning.granularity.value == "month"
    assert tuple(bound.name for bound in alias_partitioning.bounds) == cluster.report.indices
    assert all(
        bound.lower is not None and bound.upper is not None for bound in alias_partitioning.bounds
    )


def test_unimplemented_optional_methods_name_their_capability(cluster: Cluster) -> None:
    """No query store, no change counters: ``NotSupported`` names the capability it provides."""
    with pytest.raises(NotSupported) as queries:
        list(cluster.adapter.mine_queries(datetime(2025, 1, 1, tzinfo=UTC)))
    with pytest.raises(NotSupported) as counters:
        cluster.adapter.mutability_stats()

    assert queries.value.capability == OPTIONAL_METHODS["mine_queries"]
    assert counters.value.capability == OPTIONAL_METHODS["mutability_stats"]


def test_a_fragment_carrying_aggs_is_refused_with_a_fix(cluster: Cluster) -> None:
    """Aggregation bodies are not half-implemented: they are refused, naming the field."""
    base = _month_fragment(cluster.adapter, _filled(cluster.report)[0])
    fragment = base.model_copy(
        update={"dsl": {**(base.dsl or {}), "aggs": {"x": {"terms": {"field": "units"}}}}}
    )
    with pytest.raises(ElasticError) as caught:
        cluster.adapter.execute(fragment, LIMITS)
    assert caught.value.code == "INVALID_SPEC"
    assert "dsl.aggs" in caught.value.response.problem


# --------------------------------------------------------------------------------------
# Acceptance 5: a write is impossible, proven by the layer that refuses it
# --------------------------------------------------------------------------------------


def test_the_client_transport_refuses_every_write(cluster: Cluster) -> None:
    """The adapter's own client refuses the method, and the cluster never sees the request."""
    client = cluster.adapter._client
    assert client is not None
    assert isinstance(client._transport, SearchOnlyTransport)

    attempts = [
        ("PUT", f"/{ALIAS}-a09-write-probe"),
        ("DELETE", f"/{ALIAS}-2025-08"),
        ("POST", "/_bulk"),
        ("POST", f"/{ALIAS}/_delete_by_query"),
        ("POST", "/_aliases"),
        ("HEAD", "/"),
    ]
    for method, path in attempts:
        with pytest.raises(ElasticError) as refused:
            client.request(method, path, content="{}")
        assert refused.value.code == "WRITE_BLOCKED", f"{method} {path} was not refused"
        assert method in refused.value.response.problem

    created = httpx.get(f"{cluster.target.base_url}/{ALIAS}-a09-write-probe", timeout=5.0)
    assert created.status_code == httpx.codes.NOT_FOUND, "the refused PUT reached the cluster"
    still_there = httpx.get(f"{cluster.target.base_url}/{ALIAS}-2025-08/_count", timeout=5.0)
    assert still_there.status_code == httpx.codes.OK, "the refused DELETE reached the cluster"


def test_the_one_post_the_transport_allows_is_forced_by_the_server(cluster: Cluster) -> None:
    """ES|QL refuses ``GET`` (405), so ``POST /_query`` is the single documented exception."""
    response = httpx.request(
        "GET",
        f"{cluster.target.base_url}{ESQL_PATH}",
        content='{"query": "ROW mdqm_probe = 1"}',
        headers={"content-type": "application/json"},
        timeout=5.0,
    )
    assert response.status_code == httpx.codes.METHOD_NOT_ALLOWED
    assert "POST" in response.text, response.text

    client = cluster.adapter._client
    assert client is not None
    allowed = client.post(ESQL_PATH, content='{"query": "ROW mdqm_probe = 1"}')
    assert allowed.status_code == httpx.codes.OK
    assert allowed.json()["values"] == [[1]]


def test_a_stream_is_an_arrow_stream_of_record_batches(cluster: Cluster) -> None:
    """The result is Arrow, with a schema known before the first batch arrives."""
    index = _filled(cluster.report)[0]
    stream = _es_rows(cluster.adapter.execute(_month_fragment(cluster.adapter, index), LIMITS))
    assert isinstance(stream.schema, pa.Schema)
    assert stream.schema.names[:2] == ["client_id", "client_no"]
    batches = list(stream)
    assert batches, "a filled month must return at least one batch"
    assert all(isinstance(batch, pa.RecordBatch) for batch in batches)

    cluster.adapter.cancel(stream)
    cluster.adapter.cancel(stream)  # idempotent
