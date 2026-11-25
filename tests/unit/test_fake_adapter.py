"""Tests for the DuckDB-backed fake adapter (task F05).

The fake exists so execution, planning and the governor are testable without a container runtime,
so the tests here assert the two things that make it trustworthy: it passes the shared conformance
kit (:func:`~mdqm.testing.conformance.run_conformance`), and the ``World`` really round-trips into
DuckDB with every planted quirk intact for all three storage forms.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from mdqm.adapters.base import NotSupported
from mdqm.adapters.registry import AdapterRegistry
from mdqm.models import (
    Capability,
    Fragment,
    Limits,
    PartitionBound,
    Partitioning,
    PartitionStrategy,
    PlanSummary,
    PruningProof,
    RawQuery,
    TableStats,
)
from mdqm.testing.conformance import run_conformance
from mdqm.testing.fake_adapter import (
    FakeAdapter,
    FakeAdapterTimeout,
    FakeStream,
    table_from_rows,
    world_tables,
)
from mdqm.testing.world import ROUNDING_STEP, World, WorldConfig, build_world

EXEC_TS = datetime(2026, 1, 1, tzinfo=UTC)


def _fragment(sql: str, *tables: str) -> Fragment:
    return Fragment(source="fake", dialect="duckdb", sql=sql, tables=tables, exec_ts=EXEC_TS)


def _limits(**overrides: object) -> Limits:
    base = {"timeout_s": 5.0, "max_rows": 100, "max_bytes": 1_000_000, "max_partitions": 8}
    base.update(overrides)
    return Limits(**base)  # type: ignore[arg-type]


def _rows(adapter: FakeAdapter, sql: str, limits: Limits | None = None) -> list[dict[str, object]]:
    """Run one fragment and materialise its rows, the way a caller of the adapter would."""
    stream = adapter.execute(_fragment(sql), limits or _limits())
    try:
        batches = list(stream)
    finally:
        stream.close()
    return [row for batch in batches for row in batch.to_pylist()]


# --- the shared conformance kit -------------------------------------------------------------


def test_passes_conformance_with_every_optional_capability() -> None:
    """F05 acceptance 1: the fake is a real adapter under the F04 conformance kit."""
    world = build_world(WorldConfig.small())
    partitioning = {
        "usage_events": Partitioning(
            strategy=PartitionStrategy.RANGE,
            key=("start_time",),
            bounds=(
                PartitionBound(name="p1", lower=EXEC_TS - timedelta(days=30), row_estimate=500),
                PartitionBound(name="p2", lower=EXEC_TS, row_estimate=500),
            ),
        )
    }
    history = [RawQuery(text="SELECT 1", dialect="duckdb", last_seen=EXEC_TS)]
    stats = [TableStats(source="fake", table="accounts", captured_at=EXEC_TS, rows_updated=3)]

    run_conformance(lambda: _with_optionals(world, partitioning, history, stats))


def _with_optionals(
    world: World,
    partitioning: dict[str, Partitioning],
    history: list[RawQuery],
    stats: list[TableStats],
) -> FakeAdapter:
    return FakeAdapter(
        name="fake",
        tables=world_tables(world),
        partitioning=partitioning,
        query_history=history,
        mutability=stats,
    )


def test_passes_conformance_with_no_optional_capability() -> None:
    """The barer instance must also conform: optional features report themselves honestly."""
    run_conformance(lambda: FakeAdapter())


def test_conformance_proves_the_timeout_is_honoured() -> None:
    """A slow source must raise inside the deadline, not complete and be rejected afterwards."""
    adapter = FakeAdapter(latency_s=1.0)
    slow = _fragment("SELECT 1 AS one")
    run_conformance(
        lambda: FakeAdapter(latency_s=1.0),
        fragment=_fragment("SELECT 1 AS one"),
        slow_fragment=slow,
    )
    # The same instance used by the kit above is checked directly for the deadline rule.
    started = time.monotonic()
    with pytest.raises(FakeAdapterTimeout):
        stream = adapter.execute(slow, _limits(timeout_s=0.05))
        list(stream)
    assert time.monotonic() - started < 2.0


# --- probe reflects construction ------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {},
            {
                "partition_metadata": False,
                "explain_plan": True,
                "query_store": False,
                "mutability_counters": False,
            },
        ),
        (
            {"explain_enabled": False},
            {
                "partition_metadata": False,
                "explain_plan": False,
                "query_store": False,
                "mutability_counters": False,
            },
        ),
        (
            {
                "partitioning": {
                    "usage_events": Partitioning(
                        strategy=PartitionStrategy.RANGE,
                        key=("start_time",),
                        bounds=(PartitionBound(name="p1"),),
                    )
                },
                "query_history": (RawQuery(text="SELECT 1", dialect="duckdb"),),
            },
            {
                "partition_metadata": True,
                "explain_plan": True,
                "query_store": True,
                "mutability_counters": False,
            },
        ),
    ],
)
def test_probe_reflects_construction(kwargs: dict[str, object], expected: dict[str, bool]) -> None:
    """F05 acceptance 2: the probe says what the instance was actually given."""
    adapter = FakeAdapter(**kwargs)  # type: ignore[arg-type]
    capabilities = adapter.probe()
    for name, supported in expected.items():
        assert capabilities.get(name).supported is supported, name
    assert capabilities.adapter == "fake"
    assert capabilities.source == "fake"


def test_disabled_optional_methods_raise_not_supported() -> None:
    """The method and the probe must agree: absent feature means NotSupported, never garbage."""
    adapter = FakeAdapter(explain_enabled=False)
    for call, capability in (
        (lambda: adapter.extract_partitions(), "partition_metadata"),
        (lambda: adapter.explain(_fragment("SELECT 1")), "explain_plan"),
        (lambda: list(adapter.mine_queries(EXEC_TS)), "query_store"),
        (lambda: adapter.mutability_stats(), "mutability_counters"),
    ):
        with pytest.raises(NotSupported) as excinfo:
            call()
        assert excinfo.value.capability == capability


def test_extra_capabilities_are_reported() -> None:
    adapter = FakeAdapter(extra_capabilities={"row_level_security": Capability(supported=True)})
    assert adapter.probe().supports("row_level_security")


# --- execute: caps, calls, failure injection ------------------------------------------------


def test_execute_enforces_max_rows_exactly() -> None:
    """F05 acceptance 3: the row cap is the adapter's promise, exact and not batch-rounded."""
    adapter = FakeAdapter(tables={"t": pa.table({"a": list(range(500))})}, batch_size=64)
    rows = _rows(adapter, "SELECT a FROM t ORDER BY a", _limits(max_rows=10))
    assert [row["a"] for row in rows] == list(range(10))


def test_execute_enforces_max_bytes() -> None:
    """A byte cap stops the scan early instead of returning everything and reporting later."""
    adapter = FakeAdapter(tables={"t": pa.table({"a": list(range(2000))})}, batch_size=16)
    stream = adapter.execute(_fragment("SELECT a FROM t"), _limits(max_bytes=1024))
    try:
        batches = list(stream)
    finally:
        stream.close()
    assert len(batches) > 0
    assert sum(batch.nbytes for batch in batches) < 2000 * 8


def test_calls_records_the_fragment_sql() -> None:
    """F05 acceptance 3: ``calls`` is the audit trail cache/governor tests assert on."""
    adapter = FakeAdapter(tables={"t": pa.table({"a": [1, 2, 3]})})
    assert adapter.calls == []
    _rows(adapter, "SELECT count(*) AS n FROM t")
    assert adapter.calls == ["SELECT count(*) AS n FROM t"]


def test_injected_failure_raises_on_configured_call() -> None:
    """A failing source is testable: the governor/retry path needs a source that breaks."""
    boom = RuntimeError("connection reset")
    adapter = FakeAdapter(tables={"t": pa.table({"a": [1]})}, fail_with=boom)
    with pytest.raises(RuntimeError, match="connection reset"):
        adapter.execute(_fragment("SELECT a FROM t"), _limits())
    assert adapter.calls == ["SELECT a FROM t"]


def test_non_sql_fragment_is_rejected() -> None:
    adapter = FakeAdapter()
    dsl = Fragment(
        source="fake",
        dialect="duckdb",
        dsl={"query": {"match_all": {}}},
        exec_ts=EXEC_TS,
    )
    with pytest.raises(ValueError, match="runs SQL"):
        adapter.execute(dsl, _limits())


def test_cancel_is_idempotent_and_stops_the_scan() -> None:
    adapter = FakeAdapter(tables={"t": pa.table({"a": list(range(100))})}, batch_size=4)
    stream = adapter.execute(_fragment("SELECT a FROM t"), _limits())
    assert isinstance(stream, FakeStream)
    adapter.cancel(stream)
    adapter.cancel(stream)
    try:
        assert list(stream) == []
    finally:
        stream.close()


def test_unknown_sql_raises_a_duckdb_error() -> None:
    """The fake runs real SQL, so a bad fragment fails like the real source would."""
    adapter = FakeAdapter(tables={"t": pa.table({"a": [1]})})
    with pytest.raises(Exception, match="(?i)not found|does not exist|Catalog"):
        _rows(adapter, "SELECT * FROM no_such_table")


# --- catalog, partitions, explain, history ---------------------------------------------------


def test_catalog_describes_the_loaded_tables() -> None:
    world = build_world(WorldConfig.small())
    adapter = FakeAdapter.for_world(world)
    snapshot = adapter.extract_catalog()
    assert snapshot.source == adapter.probe().source
    names = {table.name for table in snapshot.tables}
    assert names == {"customers_crm", "accounts", "usage_events", "accounts_hist", "customers_erp"}
    crm = snapshot.table("customers_crm")
    assert crm is not None
    assert [c.name for c in crm.columns] == [
        "cust_id",
        "full_name",
        "status",
        "credit_limit",
        "region",
        "signup_date",
    ]
    assert crm.row_estimate == len(world.crm.customers)


def test_declared_partitioning_marks_the_table_and_is_returned() -> None:
    partitioning = Partitioning(
        strategy=PartitionStrategy.RANGE,
        key=("start_time",),
        granularity=None,
        bounds=(PartitionBound(name="p1", row_estimate=10), PartitionBound(name="p2")),
    )
    adapter = FakeAdapter(
        tables={"usage_events": pa.table({"start_time": [EXEC_TS]})},
        partitioning={"usage_events": partitioning},
    )
    assert adapter.extract_partitions() == [partitioning]
    table = adapter.extract_catalog().table("usage_events")
    assert table is not None
    assert table.kind.value == "partitioned_table"


@pytest.mark.parametrize(
    ("sql", "expected_scanned", "expected_proof"),
    [
        ("SELECT * FROM t WHERE start_time >= '2026-01-01'", 1, PruningProof.PROVEN),
        ("SELECT * FROM t", 2, PruningProof.UNPROVEN),
    ],
)
def test_explain_is_derived_from_declared_partitions(
    sql: str, expected_scanned: int, expected_proof: PruningProof
) -> None:
    """The explain gate's control flow is testable without a real plan parser."""
    partitioning = Partitioning(
        strategy=PartitionStrategy.RANGE,
        key=("start_time",),
        bounds=(PartitionBound(name="p1"), PartitionBound(name="p2")),
    )
    adapter = FakeAdapter(partitioning={"t": partitioning})
    summary = adapter.explain(_fragment(sql, "t"))
    assert isinstance(summary, PlanSummary)
    assert summary.partitions_total == 2
    assert summary.partitions_scanned == expected_scanned
    assert summary.pruning_proof is expected_proof
    assert "FAKE-SCAN" in (summary.raw or "")


def test_explain_without_declared_partitions_is_unknown() -> None:
    adapter = FakeAdapter()
    summary = adapter.explain(_fragment("SELECT 1"))
    assert summary.pruning_proof is PruningProof.UNKNOWN


def test_mine_queries_filters_by_since() -> None:
    old = RawQuery(text="SELECT old", dialect="duckdb", last_seen=EXEC_TS - timedelta(days=10))
    new = RawQuery(text="SELECT new", dialect="duckdb", last_seen=EXEC_TS)
    adapter = FakeAdapter(query_history=[old, new])
    assert [q.text for q in adapter.mine_queries(EXEC_TS - timedelta(days=1))] == ["SELECT new"]


# --- the synthetic world loads into DuckDB ---------------------------------------------------


def test_world_loads_and_all_three_storage_forms_return_correct_rows() -> None:
    """F05 acceptance 4: real rows for crm/billing/erp, with the planted quirks intact."""
    world = build_world(WorldConfig.small())
    adapter = FakeAdapter.for_world(world)

    crm = _rows(adapter, "SELECT count(*) AS n FROM customers_crm")
    assert crm == [{"n": len(world.crm.customers)}]

    erp = _rows(adapter, "SELECT count(*) AS n FROM customers_erp")
    assert erp == [{"n": len(world.erp.customers)}]

    billing = _rows(adapter, "SELECT count(*) AS n FROM accounts")
    assert billing == [{"n": len(world.billing.accounts)}]

    events = _rows(adapter, "SELECT count(*) AS n FROM usage_events")
    assert events == [{"n": len(world.billing.usage_events)}]

    hist = _rows(adapter, "SELECT count(*) AS n FROM accounts_hist")
    assert hist == [{"n": len(world.billing.accounts_hist)}]


def test_each_storage_form_keeps_its_own_representation() -> None:
    """The three forms must stay different, or the conversion tests would prove nothing."""
    world = build_world(WorldConfig.small())
    adapter = FakeAdapter.for_world(world)
    truth = world.expected.customers[0]

    crm_row = _rows(
        adapter, f"SELECT status, credit_limit FROM customers_crm WHERE cust_id = {truth.cust_id}"
    )[0]
    assert crm_row["credit_limit"] == truth.canonical_limit  # rounded half-up
    assert crm_row["status"] in {"A", "C", "S"}

    erp_row = _rows(
        adapter,
        f"SELECT status_code, limit_k FROM customers_erp WHERE cust_no = {truth.cust_id}",
    )[0]
    assert erp_row["limit_k"] == truth.canonical_limit // 10  # rounded, then /10
    assert erp_row["status_code"] in {1, 2, 3}

    billing_row = _rows(
        adapter,
        f"SELECT state, credit_limit FROM accounts WHERE client_no = '{truth.client_no}'",
    )[0]
    assert billing_row["credit_limit"] in (truth.raw_limit, None)  # raw, never rounded
    assert billing_row["state"] in {"Active", "Closed", "Hold"}


def test_planted_nulls_and_orphans_survive_the_load() -> None:
    """Planted NULLs and orphan keys are the point of the world: they must reach DuckDB."""
    world = build_world(WorldConfig.small())
    adapter = FakeAdapter.for_world(world)

    nulls = _rows(adapter, "SELECT count(*) AS n FROM accounts WHERE credit_limit IS NULL")
    assert nulls == [{"n": len([a for a in world.billing.accounts if a.credit_limit is None])}]
    assert world.expected.planted_nulls  # the world really planted at least one

    orphans = _rows(
        adapter,
        "SELECT count(*) AS n FROM usage_events u "
        "WHERE NOT EXISTS (SELECT 1 FROM accounts a WHERE a.client_no = u.client_no)",
    )
    assert orphans == [{"n": world.expected.orphan_event_count}]


def test_duplicate_erp_rows_survive_the_load() -> None:
    world = build_world(WorldConfig.small())
    adapter = FakeAdapter.for_world(world)
    assert world.expected.duplicate_cust_nos
    dup = world.expected.duplicate_cust_nos[0]
    count = _rows(adapter, f"SELECT count(*) AS n FROM customers_erp WHERE cust_no = {dup}")
    assert count[0]["n"] >= 2


def test_rounding_step_matches_the_world() -> None:
    """A cheap guard that the loaded crm limit is the world's rounded value, not the raw one."""
    world = build_world(WorldConfig.small())
    adapter = FakeAdapter.for_world(world)
    bad = _rows(
        adapter,
        "SELECT count(*) AS n FROM customers_crm c "
        "JOIN accounts a ON a.client_no = printf('%010d', c.cust_id) "
        "WHERE c.credit_limit % 5000 <> 0",
    )
    assert bad == [{"n": 0}]
    assert ROUNDING_STEP == 5000


def test_table_from_rows_matches_the_world_dataclasses() -> None:
    world = build_world(WorldConfig.small())
    arrow = table_from_rows(world.crm.customers)
    assert arrow.num_rows == len(world.crm.customers)
    assert set(arrow.schema.names) == {
        "cust_id",
        "full_name",
        "status",
        "credit_limit",
        "region",
        "signup_date",
    }


# --- registry integration --------------------------------------------------------------------


def test_registers_explicitly_with_the_registry() -> None:
    """Decision: no entry point; tests register the fake themselves (see progress/F05.md)."""
    world = build_world(WorldConfig.small())
    registry = AdapterRegistry()
    registry.register("fake", FakeAdapter)
    from mdqm.config.settings import SourceConfig

    adapter = registry.create(SourceConfig(name="crm", adapter="fake"))
    assert isinstance(adapter, FakeAdapter)
    assert adapter.probe().adapter == "fake"
    assert adapter.extract_catalog().tables  # the default table keeps it a valid source
    assert world.expected.customers
