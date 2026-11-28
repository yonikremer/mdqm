"""A08 acceptance: the SQL Server adapter against the live ``mssql`` compose profile.

Needs the profile up and the erp world loaded (task D06)::

    python scripts/dev.py up mssql
    python -m mdqm.testing.loaders.sqlserver --small
    python -m pytest tests/integration/test_mssql_adapter.py --require-services -q

Every assertion compares the adapter with something the adapter did not produce: the server's own
``sys.*`` rows, a direct ``mssql_python`` query with the same credentials, an actual execution plan
captured with ``SET STATISTICS XML ON``, or the error SQL Server itself raises.

The plan-XML question (AGENTS.md section 9, "SQL Server plan XML format for partition counting is
unverified") is answered here in two tests that must agree:

* :func:`test_explain_proves_pruning_from_the_estimated_plan` reads the estimated plan through the
  adapter and expects ``pruning_proof=proven`` with ``partitions_scanned`` from the ``PtnId1000``
  seek predicate;
* :func:`test_the_estimated_plan_range_matches_the_actual_partition_access` captures the *actual*
  plan of the same statement (``SET STATISTICS XML ON``, ``RunTimePartitionSummary``) and asserts
  the adapter counted the same partitions the engine really touched.

Both run over the D06 world, whose ``pf_usage_events_month``/``ps_usage_events_month`` function has
17 tuples and therefore 18 partitions.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import threading
import time
import tomllib
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pytest

from mdqm.adapters.base import OPTIONAL_METHODS
from mdqm.adapters.registry import AdapterRegistry
from mdqm.adapters.sqlserver import (
    SqlServerAdapter,
    SqlServerArrowStream,
    SqlServerCancelled,
    SqlServerConfigError,
    SqlServerTimeout,
    plan_summary,
)
from mdqm.config.settings import SourceConfig
from mdqm.models import (
    Fragment,
    Limits,
    PartitionGranularity,
    PartitionStrategy,
    PruningProof,
    TableKind,
)
from mdqm.sql.execution import StatementRefused
from mdqm.sql.guard import WRITE_BLOCKED
from mdqm.testing.conformance import run_conformance
from mdqm.testing.loaders.sqlserver import (
    PARTITION_FUNCTION,
    PARTITION_SCHEME,
    MssqlTarget,
    connect,
    target_from_environ,
)

pytestmark = pytest.mark.slow

NOW: Final = datetime(2026, 1, 1, tzinfo=UTC)
"""A fixed ``exec_ts``: the adapter never substitutes the clock, so a constant is honest."""

LIMITS: Final = Limits(timeout_s=20.0, max_rows=100_000, max_bytes=50_000_000, max_partitions=8)

#: A read that keeps the server busy past any deadline here. ``COUNT_BIG(*)`` of this cross join is
#: not enough: SQL Server rewrites it to the product of the three cardinalities and answers in
#: about half a second, which is the same size as ``timeout_s`` and flakes. A checksum of every
#: triple cannot be answered from the cardinalities, so the scan is still running when the
#: watchdog fires. It must be a *read* -- the execution chokepoint guards every statement, so only
#: a read can reach the driver, which is why this is not a ``WAITFOR DELAY``.
SLOW_SQL: Final = (
    "SELECT CHECKSUM_AGG(BINARY_CHECKSUM(a.event_id, b.event_id, c.event_id)) AS n "
    "FROM dbo.usage_events a CROSS JOIN dbo.usage_events b CROSS JOIN dbo.usage_events c"
)

#: A filter on the partition key as a ``datetime2`` literal: prunable to two of eighteen partitions.
PRUNED_SQL: Final = (
    "SELECT COUNT(*) AS n, client_no FROM dbo.usage_events "
    "WHERE start_time >= CONVERT(datetime2(6), '2025-06-01', 126) "
    "AND start_time < CONVERT(datetime2(6), '2025-08-01', 126) "
    "GROUP BY client_no"
)

#: The same filter written as a ``varchar`` literal: an implicit conversion removes elimination.
UNPRUNED_SQL: Final = (
    "SELECT COUNT(*) AS n, client_no FROM dbo.usage_events "
    "WHERE start_time >= '2025-06-01' GROUP BY client_no"
)

#: ~25 million rows (5001 x 5001): a stream that is still fetching when ``cancel`` arrives.
HUGE_SQL: Final = (
    "SELECT a.client_no, b.event_id FROM dbo.usage_events a CROSS JOIN dbo.usage_events b"
)
HUGE_LIMITS: Final = Limits(timeout_s=60.0, max_rows=25_000_000, max_bytes=10**10, max_partitions=8)


class _StaticSecrets:
    """A secret provider returning one throwaway dev password for any reference."""

    def __init__(self, password: str) -> None:
        self._password = password

    def get(self, name: str) -> str:
        """Resolve any reference to the throwaway dev password (the name is what config carries)."""
        assert name, "a secret reference must be a name, never an empty string"
        return self._password


def _unavailable(request: pytest.FixtureRequest, reason: str) -> None:
    """Skip, or fail when ``--require-services`` was given (a skipped test is not a pass)."""
    if request.config.getoption("--require-services"):
        pytest.fail(f"required service unavailable: {reason}")
    pytest.skip(f"service unavailable: {reason}")


def _config(name: str, target: MssqlTarget) -> SourceConfig:
    """A source config for the ``erp`` database, connected as the read-only login."""
    return SourceConfig(
        name=name,
        adapter="sqlserver",
        secret="erp_ro",
        options={
            "host": target.host,
            "port": target.port,
            "database": target.database,
            "user": target.ro_user,
            "connect_timeout": 5,
            # The compose server uses a self-signed certificate; production names its own CA.
            "trust_server_certificate": "yes",
        },
    )


def _fragment(
    source: str, statement: str, tables: tuple[str, ...] = ("dbo.usage_events",)
) -> Fragment:
    return Fragment(source=source, dialect="tsql", sql=statement, tables=tables, exec_ts=NOW)


def _drain(adapter: SqlServerAdapter, statement: str, limits: Limits = LIMITS) -> int:
    """Run a statement through the adapter and return the row count (closing the stream)."""
    stream = adapter.execute(_fragment(adapter.source, statement), limits)
    rows = 0
    try:
        for batch in stream:
            assert isinstance(batch, pa.RecordBatch)
            rows += batch.num_rows
    finally:
        stream.close()
    return rows


def _direct(target: MssqlTarget, statement: str, params: tuple[Any, ...] = ()) -> list[Any]:
    """Run one statement with the driver itself: the reference the adapter is compared against."""
    with connect(target, read_only=True, autocommit=True) as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(statement, params) if params else cursor.execute(statement)
            return [tuple(row) for row in cursor.fetchall()]
        finally:
            cursor.close()


@pytest.fixture(scope="module")
def target(request: pytest.FixtureRequest) -> MssqlTarget:
    """Where the compose SQL Server lives, from ``.env`` (never a hard-coded credential)."""
    try:
        resolved = target_from_environ()
    except Exception as exc:  # a missing .env is "no service", not a test failure
        _unavailable(request, f"no SQL Server credentials: {exc}")
        raise AssertionError("unreachable") from exc
    try:
        _direct(resolved, "SELECT 1 AS one")
    except Exception as exc:
        _unavailable(request, f"SQL Server at {resolved.server} is unreachable: {exc}")
    return resolved


@pytest.fixture(scope="module")
def adapter(target: MssqlTarget) -> Iterator[SqlServerAdapter]:
    """A connected adapter for ``erp`` as the read-only login (the D06 grants, nothing more)."""
    built = SqlServerAdapter(_config("erp_mssql", target), _StaticSecrets(target.ro_password))
    try:
        yield built
    finally:
        built.close()


# --- identity and the entry point -------------------------------------------------------------


def test_the_entry_point_declaration_builds_a_protocol_adapter() -> None:
    """The ``mdqm.adapters`` entry point names a factory the registry accepts (no database needed).

    Read from ``pyproject.toml`` rather than installed metadata: a git worktree installs no
    distribution of its own, so the installed entry points still describe the checkout it was
    installed from.
    """
    declared = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    value = declared["project"]["entry-points"]["mdqm.adapters"]["sqlserver"]
    loaded = importlib.metadata.EntryPoint(
        name="sqlserver", value=value, group="mdqm.adapters"
    ).load()
    assert loaded is SqlServerAdapter

    registry = AdapterRegistry(entry_points=lambda: (), secrets=_StaticSecrets("unused"))
    registry.register("sqlserver", loaded)
    config = SourceConfig(
        name="erp_mssql",
        adapter="sqlserver",
        secret="erp_ro",
        options={"host": "127.0.0.1", "database": "erp", "user": "mdqm_ro"},
    )
    built = registry.create(config)
    assert isinstance(built, SqlServerAdapter)
    assert built.kind == "sqlserver"
    assert built.dialect() == "tsql"
    built.close()


def test_config_is_refused_without_a_secret_reference() -> None:
    """A password in config would be a password in git, so a missing ``secret:`` is refused."""
    with pytest.raises(SqlServerConfigError, match="no secret reference"):
        SqlServerAdapter(
            SourceConfig(
                name="erp",
                adapter="sqlserver",
                options={"host": "127.0.0.1", "database": "erp", "user": "mdqm_ro"},
            ),
            _StaticSecrets("unused"),
        )


# --- conformance ------------------------------------------------------------------------------


def test_conformance(target: MssqlTarget) -> None:
    """The shared conformance kit, including the timeout path (``WAITFOR`` + the watchdog)."""

    def factory() -> SqlServerAdapter:
        return SqlServerAdapter(_config("erp_mssql", target), _StaticSecrets(target.ro_password))

    run_conformance(factory, slow_fragment=_fragment("erp_mssql", SLOW_SQL))


# --- probe ------------------------------------------------------------------------------------


def test_probe_names_the_server_and_claims_only_what_it_serves(adapter: SqlServerAdapter) -> None:
    """The probe reports the reached version and a verdict per optional method."""
    capabilities = adapter.probe()
    assert capabilities.source == "erp_mssql"
    assert capabilities.adapter == "sqlserver"
    assert capabilities.version is not None
    assert "16." in capabilities.version, capabilities.version

    for method, capability in OPTIONAL_METHODS.items():
        verdict = capabilities.get(capability)
        if verdict.supported is True:
            continue
        assert verdict.detail, f"{capability} is unsupported without saying why"
        assert callable(getattr(adapter, method, None)), (
            f"{capability} is unsupported but {method}() exists and could serve it"
        )
    assert capabilities.supports(OPTIONAL_METHODS["extract_partitions"])
    assert capabilities.supports(OPTIONAL_METHODS["explain"])
    assert capabilities.supports(OPTIONAL_METHODS["mine_queries"])


# --- catalog ----------------------------------------------------------------------------------


def test_catalog_matches_the_sys_views(adapter: SqlServerAdapter, target: MssqlTarget) -> None:
    """Every table, column and index the adapter reports is one SQL Server reports, and back."""
    snapshot = adapter.extract_catalog()
    assert snapshot.source == adapter.source

    server_tables = {
        (str(schema), str(name))
        for schema, name in _direct(
            target,
            "SELECT s.name, t.name FROM sys.tables AS t "
            "JOIN sys.schemas AS s ON s.schema_id = t.schema_id",
        )
    }
    reported = {(t.schema_name, t.name) for t in snapshot.tables if t.kind is not TableKind.VIEW}
    assert reported == server_tables

    for name in ("customers", "usage_events"):
        table = snapshot.table(f"dbo.{name}")
        assert table is not None, f"dbo.{name} is missing from the catalog"
        columns = [
            (str(column), str(type_name))
            for column, type_name in _direct(
                target,
                "SELECT c.name, ty.name FROM sys.columns AS c "
                "JOIN sys.tables AS t ON t.object_id = c.object_id "
                "JOIN sys.schemas AS s ON s.schema_id = t.schema_id "
                "JOIN sys.types AS ty ON ty.user_type_id = c.user_type_id "
                "WHERE s.name = 'dbo' AND t.name = ? ORDER BY c.column_id",
                (name,),
            )
        ]
        assert [(c.name, c.native_type.split("(")[0]) for c in table.columns] == columns

    usage = snapshot.table("dbo.usage_events")
    assert usage is not None
    assert usage.kind is TableKind.PARTITIONED_TABLE
    assert usage.primary_key == ("event_id", "start_time")
    assert "usage_events_pk" in {index.name for index in usage.indexes}
    assert usage.row_estimate == sum(
        int(row[0])
        for row in _direct(
            target,
            "SELECT p.rows FROM sys.partitions AS p "
            "JOIN sys.indexes AS i ON i.object_id = p.object_id AND i.index_id = p.index_id "
            "JOIN sys.tables AS t ON t.object_id = p.object_id "
            "WHERE t.name = 'usage_events' AND i.index_id IN (0, 1)",
        )
    )
    customers = snapshot.table("dbo.customers")
    assert customers is not None
    assert customers.kind is TableKind.TABLE
    assert customers.primary_key == (), "the world deliberately gives customers no key"


# --- partitions -------------------------------------------------------------------------------


def test_extract_partitions_reads_the_real_bound_values(
    adapter: SqlServerAdapter, target: MssqlTarget
) -> None:
    """The bounds are the function's own tuples, compared row for row with ``sys``."""
    found = adapter.extract_partitions()
    assert [p.key for p in found if p.strategy is not PartitionStrategy.NONE] == [("start_time",)]
    (partitioning,) = found
    assert partitioning.strategy is PartitionStrategy.RANGE
    assert partitioning.granularity is PartitionGranularity.MONTH

    tuples = [
        row[0]
        for row in _direct(
            target,
            "SELECT CAST(prv.value AS datetime2(6)) FROM sys.partition_range_values AS prv "
            "JOIN sys.partition_functions AS pf ON pf.function_id = prv.function_id "
            "WHERE pf.name = ? ORDER BY prv.boundary_id",
            (PARTITION_FUNCTION,),
        )
    ]
    fanout = int(
        _direct(
            target,
            "SELECT fanout FROM sys.partition_functions WHERE name = ?",
            (PARTITION_FUNCTION,),
        )[0][0]
    )
    assert fanout == len(tuples) + 1, "RANGE RIGHT: N tuples define N+1 partitions"
    assert len(partitioning.bounds) == fanout

    # RANGE RIGHT means the tuple is the *lower* bound of the partition to its right, so the
    # first partition is unbounded below and the last is unbounded above.
    assert partitioning.bounds[0].lower is None
    assert partitioning.bounds[0].upper == tuples[0].replace(tzinfo=UTC)
    assert partitioning.bounds[-1].upper is None
    for number, bound in enumerate(partitioning.bounds[1:-1], start=0):
        assert bound.lower == tuples[number].replace(tzinfo=UTC)
        assert bound.upper == tuples[number + 1].replace(tzinfo=UTC)

    # The scheme maps every partition to PRIMARY through one extra reserved destination, so the
    # destination list is one longer than the fanout: the adapter must not report that extra row.
    destinations = _direct(
        target,
        "SELECT COUNT(*) FROM sys.destination_data_spaces AS dds "
        "JOIN sys.partition_schemes AS ps ON ps.data_space_id = dds.partition_scheme_id "
        "WHERE ps.name = ?",
        (PARTITION_SCHEME,),
    )
    assert int(destinations[0][0]) == fanout + 1

    rows_by_partition = {
        int(number): int(count)
        for number, count in _direct(
            target,
            "SELECT p.partition_number, SUM(p.rows) FROM sys.partitions AS p "
            "JOIN sys.indexes AS i ON i.object_id = p.object_id AND i.index_id = p.index_id "
            "JOIN sys.tables AS t ON t.object_id = p.object_id "
            "WHERE t.name = 'usage_events' AND i.index_id IN (0, 1) "
            "GROUP BY p.partition_number",
        )
    }
    assert {
        number: bound.row_estimate for number, bound in enumerate(partitioning.bounds, start=1)
    } == rows_by_partition
    assert sum(rows_by_partition.values()) == 5001


# --- execute / limits -------------------------------------------------------------------------


def test_execute_matches_a_direct_query(adapter: SqlServerAdapter, target: MssqlTarget) -> None:
    """The stream carries exactly the rows the driver returns for the same statement."""
    statement = (
        "SELECT event_id, client_no, start_time, units FROM dbo.usage_events ORDER BY event_id"
    )
    expected = _direct(target, statement)
    stream = adapter.execute(_fragment(adapter.source, statement), LIMITS)
    assert isinstance(stream, SqlServerArrowStream)
    assert [field.name for field in stream.schema] == [
        "event_id",
        "client_no",
        "start_time",
        "units",
    ]
    columns: list[list[Any]] = [[] for _ in stream.schema]
    try:
        for batch in stream:
            assert isinstance(batch, pa.RecordBatch)
            for position in range(batch.num_columns):
                columns[position].extend(batch.column(position).to_pylist())
    finally:
        stream.close()
    assert len(columns[0]) == len(expected)
    assert columns[0] == [row[0] for row in expected]
    assert columns[1] == [row[1] for row in expected]
    # DATETIME2 is naive UTC in the database; Arrow carries it as UTC (never a local shift).
    assert columns[2] == [row[2].replace(tzinfo=UTC) for row in expected]
    assert columns[3] == [row[3] for row in expected]


def test_max_rows_truncates_exactly(adapter: SqlServerAdapter, target: MssqlTarget) -> None:
    """``max_rows`` is a hard cap the adapter enforces, not a hint the caller applies."""
    statement = "SELECT event_id FROM dbo.usage_events ORDER BY event_id"
    for cap in (1, 7, 1024, 1025):
        limits = LIMITS.model_copy(update={"max_rows": cap})
        assert _drain(adapter, statement, limits) == cap


def test_max_bytes_stops_the_stream(adapter: SqlServerAdapter) -> None:
    """``max_bytes`` stops the scan early instead of trimming a buffer already fetched."""
    statement = "SELECT event_id, client_no, start_time FROM dbo.usage_events ORDER BY event_id"
    limits = LIMITS.model_copy(update={"max_bytes": 4096})
    stream = adapter.execute(_fragment(adapter.source, statement), limits)
    try:
        total = 0
        for batch in stream:
            total += batch.nbytes
    finally:
        stream.close()
    assert 0 < total <= 4096 + 1024, total


def test_a_write_is_refused_by_the_server(adapter: SqlServerAdapter, target: MssqlTarget) -> None:
    """Two layers, both proved: the chokepoint refuses the write, and SQL Server would too.

    The adapter path is closed by the guard now -- which is the point -- so the server layer is
    proved through a **raw** connection as the same read-only login: that is how an injected
    statement would arrive if our guard were absent.
    """
    for statement in (
        "INSERT INTO dbo.customers (cust_no, status_code, limit_k, name) VALUES (1, 1, 1, 'x')",
        "DELETE FROM dbo.customers",
    ):
        with pytest.raises(StatementRefused) as refused:
            stream = adapter.execute(_fragment(adapter.source, statement), LIMITS)
            try:
                for _ in stream:
                    pass
            finally:
                stream.close()
        assert refused.value.result.error is not None
        assert refused.value.result.error.code == WRITE_BLOCKED

    # Layer 2: the same DML on a raw connection is refused by the server, not by us.
    with connect(target, read_only=True) as connection:
        cursor = connection.cursor()
        try:
            with pytest.raises(Exception) as server:
                cursor.execute(
                    "INSERT INTO dbo.customers (cust_no, status_code, limit_k, name) "
                    "VALUES (1, 1, 1, 'x')"
                )
        finally:
            cursor.close()
    message = str(server.value)
    assert "permission was denied" in message, message
    assert "INSERT" in message, message

    # The adapter itself is still healthy after both refusals.
    assert _drain(adapter, "SELECT COUNT(*) AS n FROM dbo.customers", LIMITS) == 1


# --- cancel and the deadline ------------------------------------------------------------------


def test_cancel_interrupts_and_the_session_survives(adapter: SqlServerAdapter) -> None:
    """``cancel()`` breaks a fetch in flight, and the connection behind it stays usable."""
    stream = adapter.execute(_fragment(adapter.source, HUGE_SQL), HUGE_LIMITS)
    started = time.monotonic()
    threading.Timer(0.3, adapter.cancel, args=(stream,)).start()
    with pytest.raises(SqlServerCancelled):
        for _ in stream:
            pass
    elapsed = time.monotonic() - started
    stream.close()
    adapter.cancel(stream)  # idempotent: a second cancel is a no-op, not an error
    assert elapsed < 20.0, f"cancel() took {elapsed:.1f}s to interrupt the stream"
    # The proof that cancellation is not a broken socket: the same adapter runs a query again.
    assert _drain(adapter, "SELECT COUNT(*) AS n FROM dbo.customers", LIMITS) == 1


def test_the_timeout_cancels_the_statement_and_the_session_survives(
    adapter: SqlServerAdapter,
) -> None:
    """``Limits.timeout_s`` is a real deadline: it raises, it stops the server work, it recovers."""
    limits = LIMITS.model_copy(update={"timeout_s": 0.5})
    started = time.monotonic()
    with pytest.raises(SqlServerTimeout):
        _drain(adapter, SLOW_SQL, limits)
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"the deadline took {elapsed:.1f}s for timeout_s=0.5"
    assert _drain(adapter, "SELECT COUNT(*) AS n FROM dbo.customers", LIMITS) == 1


# --- explain: the plan-XML finding ------------------------------------------------------------


def test_explain_proves_pruning_from_the_estimated_plan(adapter: SqlServerAdapter) -> None:
    """A literal key filter is proveably pruned, with a counted range, from ``SHOWPLAN_XML``."""
    summary = adapter.explain(_fragment(adapter.source, PRUNED_SQL))
    assert summary.pruning_proof is PruningProof.PROVEN
    assert summary.partitions_scanned == 2, summary.raw and summary.raw[:400]
    assert summary.partitions_total == 18
    assert "[usage_events_pk]" in summary.indexes_used
    assert summary.uses_index is True
    assert summary.est_cost is not None and summary.est_cost > 0
    assert summary.raw is not None and "PtnId1000" in summary.raw
    # The document an *estimated* plan never carries: that element is only in an actual plan.
    assert "RunTimePartitionSummary" not in summary.raw


def test_the_estimated_plan_range_matches_the_actual_partition_access(
    adapter: SqlServerAdapter, target: MssqlTarget
) -> None:
    """The count the adapter reads from the plan equals the partitions the engine really touched.

    This is the empirical answer to the open question in AGENTS.md section 9: the estimated plan's
    ``PtnId1000`` range and the actual plan's ``PartitionsAccessed/@PartitionCount`` must agree,
    and a ``varchar`` literal (implicit conversion) must show no elimination at all.
    """
    assert _actual_partition_count(target, PRUNED_SQL) == 2
    summary = adapter.explain(_fragment(adapter.source, PRUNED_SQL))
    assert summary.partitions_scanned == _actual_partition_count(target, PRUNED_SQL)

    assert _actual_partition_count(target, UNPRUNED_SQL) == 18
    unpruned = adapter.explain(_fragment(adapter.source, UNPRUNED_SQL))
    assert unpruned.pruning_proof is PruningProof.UNPROVEN
    assert unpruned.partitions_scanned is None


def test_explain_on_an_unpartitioned_table_has_nothing_to_prove(adapter: SqlServerAdapter) -> None:
    """``UNKNOWN`` is reserved for a plan with no partitioned table: there is nothing to prove."""
    summary = adapter.explain(
        _fragment(
            adapter.source,
            "SELECT status_code, AVG(limit_k) AS avg_k FROM dbo.customers GROUP BY status_code",
            tables=("dbo.customers",),
        )
    )
    assert summary.pruning_proof is PruningProof.UNKNOWN
    assert summary.partitions_total is None
    assert summary.partitions_scanned is None


def _actual_partition_count(target: MssqlTarget, statement: str) -> int:
    """``PartitionsAccessed/@PartitionCount`` from the statement's *actual* plan."""
    with connect(target, read_only=True, autocommit=True) as connection:
        cursor = connection.cursor()
        try:
            cursor.execute("SET STATISTICS XML ON")
            cursor.execute(statement)
            cursor.fetchall()
            while cursor.nextset():
                for row in cursor.fetchall():
                    text = "" if row[0] is None else str(row[0])
                    if text.lstrip().startswith("<ShowPlanXML") and "PartitionsAccessed" in text:
                        return _partition_count_attribute(text)
        finally:
            with contextlib.suppress(Exception):
                cursor.execute("SET STATISTICS XML OFF")
            cursor.close()
    raise AssertionError("no actual plan with PartitionsAccessed was captured")


def _partition_count_attribute(xml: str) -> int:
    """The integer in ``<PartitionsAccessed PartitionCount="N">``."""
    marker = 'PartitionCount="'
    start = xml.index(marker) + len(marker)
    return int(xml[start : xml.index('"', start)])


# --- query store and mutability ---------------------------------------------------------------


def test_mine_queries_reads_the_query_store(adapter: SqlServerAdapter) -> None:
    """Query Store rows come back with cumulative counters and no invented executor name."""
    queries = list(adapter.mine_queries(datetime(2000, 1, 1, tzinfo=UTC)))
    assert queries, "Query Store is READ_WRITE and the loader ran statements; it cannot be empty"
    for query in queries:
        assert query.dialect == "tsql"
        assert query.text
        assert query.user is None, "Query Store names no executor, so the adapter must not"
        assert query.calls is None or query.calls >= 1
        if query.calls and query.mean_time_ms is not None:
            assert query.mean_time_ms >= 0


def test_mutability_stats_agree_with_the_partition_statistics(adapter: SqlServerAdapter) -> None:
    """Row counts are the server's own, and the change counter is a stats counter, not a guess."""
    stats = {item.table: item for item in adapter.mutability_stats()}
    assert set(stats) == {"dbo.customers", "dbo.usage_events"}
    assert stats["dbo.usage_events"].row_estimate == 5001
    assert stats["dbo.customers"].row_estimate == 201
    for item in stats.values():
        assert item.rows_modified_since_analyze is not None
        assert item.rows_inserted is None, "SQL Server keeps no per-table insert counter"
        assert item.last_analyze is not None
        assert item.last_analyze.tzinfo is not None


def test_plan_summary_never_invents_a_partition_range() -> None:
    """A parameterised range expression is not a number: the verdict stays ``unproven``."""
    xml = (
        '<ShowPlanXML xmlns="http://schemas.microsoft.com/sqlserver/2004/07/showplan">'
        '<BatchSequence><Batch><Statements><StmtSimple StatementEstRows="1" '
        'StatementSubTreeCost="0.01"><QueryPlan><RelOp NodeId="0" '
        'PhysicalOp="Clustered Index Scan" Partitioned="1" EstimatedTotalSubtreeCost="0.01">'
        '<IndexScan><Object Table="[usage_events]" Index="[usage_events_pk]" '
        'IndexKind="Clustered"></Object><SeekPredicates><SeekPredicateNew><SeekKeys>'
        '<StartRange ScanType="GE"><RangeColumns>'
        '<ColumnReference Column="PtnId1000"></ColumnReference></RangeColumns>'
        "<RangeExpressions><ScalarOperator "
        "ScalarString=\"RangePartitionNew([@1],(1),'2024-11-01')\"></ScalarOperator>"
        "</RangeExpressions></StartRange>"
        "</SeekKeys></SeekPredicateNew></SeekPredicates></IndexScan></RelOp>"
        "</QueryPlan></StmtSimple></Statements></Batch></BatchSequence></ShowPlanXML>"
    )
    summary = plan_summary(xml, partitions_total=18)
    assert summary.pruning_proof is PruningProof.UNPROVEN
    assert summary.partitions_scanned is None
    assert summary.partitions_total == 18
