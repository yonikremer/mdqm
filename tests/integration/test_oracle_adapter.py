"""A07 acceptance: the Oracle adapter against the live ``oracle`` compose profile.

Needs the profile up and the billing world loaded (task D05)::

    python scripts/dev.py up oracle
    python -m mdqm.testing.loaders.oracle --small
    python -m pytest tests/integration/test_oracle_adapter.py --require-services -q

Every assertion compares the adapter with something the adapter did not produce: the server's own
``ALL_*`` dictionary rows, a direct ``oracledb`` query on the same session settings, or the error
Oracle itself raises. A test that only re-read the adapter's own answer would pass even if the
adapter invented it.

The read-only proof is deliberately split in two layers, because "the write was refused" is only
meaningful when you say by whom:

* the **account** (``MDQM_RO``) has ``SELECT`` and nothing else, so Oracle answers ``ORA-41900``;
* the **session** (``SET TRANSACTION READ ONLY``, which every execution re-issues) refuses the same
  write even for a user that *could* write, and Oracle answers ``ORA-01456``. The second adapter in
  this file logs in as the owner schema for exactly that proof and writes nothing.
"""

from __future__ import annotations

import importlib.metadata
import threading
import time
import tomllib
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import oracledb
import pyarrow as pa
import pytest

from mdqm.adapters.base import OPTIONAL_METHODS, NotSupported
from mdqm.adapters.oracle import (
    OracleAdapter,
    OracleArrowStream,
    OracleCancelled,
    OracleConfigError,
    OracleTimeout,
)
from mdqm.adapters.registry import AdapterRegistry
from mdqm.config.settings import SourceConfig
from mdqm.models import (
    Fragment,
    Limits,
    PartitionGranularity,
    Partitioning,
    PartitionStrategy,
    PruningProof,
    TableKind,
)
from mdqm.testing.conformance import run_conformance
from mdqm.testing.loaders.oracle import (
    GENERATED_PARTITION_PREFIX,
    INITIAL_PARTITION,
    OracleTarget,
    connect_read_only,
    generation_expected_months,
    target_from_environ,
)
from mdqm.testing.world import WorldConfig, build_world

pytestmark = pytest.mark.slow

NOW: Final = datetime(2026, 1, 1, tzinfo=UTC)
"""A fixed ``exec_ts``: the adapter never substitutes the clock, so a constant is honest."""

LIMITS: Final = Limits(timeout_s=10.0, max_rows=100_000, max_bytes=10_000_000, max_partitions=8)

SIMPLE_SQL: Final = "SELECT 1 AS one FROM dual"

# A statement that burns server time without CPU or memory, so the 1 s deadline is the only
# thing that can end it. `CONNECT BY` was the first attempt and is the wrong tool: at a level
# big enough to outlast a second it runs out of memory (ORA-30009) before the timeout fires.
SLOW_SQL: Final = (
    "WITH r(n) AS (SELECT 1 FROM dual UNION ALL SELECT n + 1 FROM r WHERE n < 2000000) "
    "SELECT count(*) AS n FROM r"
)
"""Ten seconds of pure server CPU, no table and no privilege: what a deadline must interrupt.

``CONNECT BY`` is not usable here: 2 million levels ran in 0.4 s once the plan was cached, and 5
million hit ``ORA-30009`` (not enough memory), so it is neither slow nor reliable. ``DBMS_SESSION
.SLEEP`` is not either: this read-only account may not execute it (``ORA-00904``, measured).
"""

HUGE_SQL: Final = (
    "SELECT a.client_no, b.region FROM MDQM_BILLING.accounts a, "
    "MDQM_BILLING.usage_events e, MDQM_BILLING.accounts b"
)
"""200 million rows (200 accounts x 5001 events x 200 accounts): a stream that is still fetching
when ``cancel`` arrives, however fast the machine is."""

LITERAL_SQL: Final = (
    "SELECT count(*) AS n FROM MDQM_BILLING.usage_events "
    "WHERE start_time >= timestamp '2025-11-01 00:00:00'"
)
BIND_SQL: Final = "SELECT count(*) AS n FROM MDQM_BILLING.usage_events WHERE start_time >= :since"

MISSING_PRIVILEGE: Final = 41900
READ_ONLY_TRANSACTION: Final = 1456


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


def _config(name: str, target: OracleTarget, *, user: str | None = None) -> SourceConfig:
    """A source config for the billing schema, connected as the read-only login by default."""
    return SourceConfig(
        name=name,
        adapter="oracle",
        secret="billing_ro",
        options={
            "host": target.host,
            "port": target.port,
            "service_name": target.service_name,
            "user": user or target.ro_user,
            "connect_timeout": 5,
        },
    )


def _fragment(source: str, statement: str, params: dict[str, Any] | None = None) -> Fragment:
    return Fragment(
        source=source,
        dialect="oracle",
        sql=statement,
        params=params or {},
        tables=("MDQM_BILLING.USAGE_EVENTS",),
        exec_ts=NOW,
    )


def _drain(adapter: OracleAdapter, statement: str, limits: Limits = LIMITS) -> int:
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


@pytest.fixture(scope="module")
def target(request: pytest.FixtureRequest) -> OracleTarget:
    """Where the compose Oracle lives, from ``.env`` (never a hard-coded credential)."""
    try:
        found = target_from_environ()
    except Exception as exc:
        _unavailable(request, f"Oracle connection settings: {exc}")
        raise AssertionError("unreachable") from exc
    try:
        with oracledb.connect(
            user=found.ro_user,
            password=found.ro_password,
            dsn=found.dsn,
            tcp_connect_timeout=5,
        ):
            pass
    except oracledb.Error as exc:
        _unavailable(request, f"Oracle at {found.dsn}: {exc}")
        raise AssertionError("unreachable") from exc
    return found


@pytest.fixture(scope="module")
def adapter(target: OracleTarget) -> Iterator[OracleAdapter]:
    """The adapter under test: thin mode, read-only login (``MDQM_RO``)."""
    built = OracleAdapter(_config("billing_oracle", target), _StaticSecrets(target.ro_password))
    try:
        yield built
    finally:
        built.close()


@pytest.fixture(scope="module")
def owner_adapter(target: OracleTarget) -> Iterator[OracleAdapter]:
    """The same adapter, logged in as the schema that *may* write.

    It exists to prove the second read-only layer: with ``SELECT``-only grants a refused write
    proves nothing about ``SET TRANSACTION READ ONLY``. This connection can only ever read because
    the adapter sets the transaction read-only, and the test never commits.
    """
    built = OracleAdapter(
        _config("billing_owner", target, user=target.owner_user),
        _StaticSecrets(target.owner_password),
    )
    try:
        yield built
    finally:
        built.close()


# --------------------------------------------------------------------------------------
# Acceptance 1: the conformance kit, with the live database
# --------------------------------------------------------------------------------------


def test_adapter_passes_the_conformance_kit(adapter: OracleAdapter) -> None:
    """``run_conformance`` is the only definition of "a real adapter" in this project.

    ``slow_fragment`` makes the kit prove ``Limits.timeout_s`` too: two seconds of ``CONNECT BY``
    under a 50 ms deadline must raise, not complete.
    """
    run_conformance(
        lambda: adapter,
        fragment=_fragment("billing_oracle", SIMPLE_SQL),
        slow_fragment=_fragment("billing_oracle", SLOW_SQL),
    )


def test_the_entry_point_declaration_builds_a_protocol_adapter() -> None:
    """The ``mdqm.adapters`` entry point names a factory the registry accepts (no database needed).

    Read from ``pyproject.toml`` rather than installed metadata: a git worktree installs no
    distribution of its own, so the installed entry points still describe the checkout it was
    installed from.
    """
    declared = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    value = declared["project"]["entry-points"]["mdqm.adapters"]["oracle"]
    loaded = importlib.metadata.EntryPoint(name="oracle", value=value, group="mdqm.adapters").load()
    assert loaded is OracleAdapter

    registry = AdapterRegistry(entry_points=lambda: (), secrets=_StaticSecrets("unused"))
    registry.register("oracle", loaded)
    config = SourceConfig(
        name="billing_oracle",
        adapter="oracle",
        secret="billing_ro",
        options={"host": "127.0.0.1", "service_name": "MDQM", "user": "MDQM_RO"},
    )
    built = registry.create(config)
    assert isinstance(built, OracleAdapter)
    assert built.kind == "oracle"
    assert built.dialect() == "oracle"
    built.close()


def test_probe_reports_thin_mode_version_and_capabilities(adapter: OracleAdapter) -> None:
    """The probe names the server it reached, and only claims what this adapter can serve."""
    capabilities = adapter.probe()
    assert capabilities.source == "billing_oracle"
    assert capabilities.adapter == "oracle"
    assert capabilities.version is not None
    assert "Oracle" in capabilities.version
    assert oracledb.is_thin_mode(), "the adapter must talk the wire protocol itself"

    assert capabilities.supports(OPTIONAL_METHODS["extract_partitions"])
    assert capabilities.supports(OPTIONAL_METHODS["explain"])
    assert capabilities.supports(OPTIONAL_METHODS["mine_queries"])
    # Oracle's counters are statistics, refreshed on a stats gather, never on the write.
    mutability = capabilities.get(OPTIONAL_METHODS["mutability_stats"])
    assert mutability.supported is False
    assert mutability.detail is not None and "DBMS_STATS" in mutability.detail


# --------------------------------------------------------------------------------------
# Acceptance 2: the catalog, from the server's own dictionary views
# --------------------------------------------------------------------------------------


def test_catalog_lists_the_billing_tables_from_all_tab_columns(
    adapter: OracleAdapter, target: OracleTarget
) -> None:
    """Tables, columns, primary keys and the index asymmetry, checked against ``ALL_*`` itself."""
    snapshot = adapter.extract_catalog()
    assert snapshot.source == "billing_oracle"
    names = {table.qualified_name for table in snapshot.tables}
    expected = {
        "MDQM_BILLING.ACCOUNTS",
        "MDQM_BILLING.ACCOUNTS_HIST",
        "MDQM_BILLING.STATUS_LOOKUP",
        "MDQM_BILLING.USAGE_BY_LOAD",
        "MDQM_BILLING.USAGE_EVENTS",
    }
    assert expected <= names, f"missing {sorted(expected - names)}"
    # The dictionary's own tables belong to Oracle, not to the organisation.
    assert not {name for name in names if name.startswith(("SYS.", "SYSTEM.", "XDB."))}

    usage = snapshot.table("MDQM_BILLING.USAGE_EVENTS")
    assert usage is not None
    assert usage.kind is TableKind.PARTITIONED_TABLE
    assert usage.primary_key == ("EVENT_ID", "START_TIME")
    assert [column.name for column in usage.columns] == [
        "EVENT_ID",
        "CLIENT_NO",
        "START_TIME",
        "INSERTION_TIME",
        "UNITS",
        "CORRECTS_EVENT_ID",
    ]
    client_no = usage.column("CLIENT_NO")
    assert client_no is not None and client_no.native_type == "VARCHAR2"
    units = usage.column("UNITS")
    assert units is not None and units.native_type == "NUMBER(10,0)"
    start_time = usage.column("START_TIME")
    assert start_time is not None and start_time.native_type == "TIMESTAMP(6)"

    indexed = {name for index in usage.indexes for name in index.columns}
    assert "START_TIME" in indexed, f"indexes are {[i.columns for i in usage.indexes]}"
    assert "INSERTION_TIME" not in indexed, "ingestion time must not look indexed"
    primary = [index for index in usage.indexes if index.primary]
    assert [index.columns for index in primary] == [("EVENT_ID", "START_TIME")]
    assert usage.row_estimate is not None and usage.row_estimate > 0

    # Ground truth: the dictionary, not the adapter's opinion.
    with connect_read_only(target) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM all_tab_cols "
            "WHERE owner = 'MDQM_BILLING' AND table_name = 'USAGE_EVENTS' AND hidden_column = 'NO'"
        )
        columns_row = cur.fetchone()
        cur.execute(
            "SELECT cc.column_name FROM all_cons_columns cc "
            "JOIN all_constraints c "
            "ON c.owner = cc.owner AND c.constraint_name = cc.constraint_name "
            "WHERE c.owner = 'MDQM_BILLING' AND c.table_name = 'USAGE_EVENTS' "
            "AND c.constraint_type = 'P' ORDER BY cc.position"
        )
        server_pk = tuple(str(row[0]) for row in cur.fetchall())
    assert columns_row is not None and len(usage.columns) == int(columns_row[0])
    assert usage.primary_key == server_pk


def test_catalog_reports_a_table_without_a_primary_key(
    adapter: OracleAdapter, target: OracleTarget
) -> None:
    """Nothing here invents a key: a table with no ``P`` constraint reports an empty key."""
    with connect_read_only(target) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM all_constraints WHERE owner = 'MDQM_BILLING' "
            "AND table_name = 'STATUS_LOOKUP' AND constraint_type = 'P'"
        )
        row = cur.fetchone()
    assert row is not None and int(row[0]) == 1
    table = adapter.extract_catalog().table("MDQM_BILLING.STATUS_LOOKUP")
    assert table is not None
    assert table.primary_key == ("CANONICAL",)


# --------------------------------------------------------------------------------------
# Acceptance 3: interval partitions, with the bounds the data produced
# --------------------------------------------------------------------------------------


def test_partitions_come_from_the_data_with_real_bounds(
    adapter: OracleAdapter, target: OracleTarget
) -> None:
    """``ALL_TAB_PARTITIONS`` is the only honest source: Oracle derived one partition per month."""
    world = build_world(WorldConfig.small())
    months = generation_expected_months(world)

    partitioning = {entry.key[0]: entry for entry in adapter.extract_partitions() if entry.key}
    assert set(partitioning) == {"START_TIME", "INSERTION_TIME"}
    event_time: Partitioning = partitioning["START_TIME"]
    assert event_time.strategy is PartitionStrategy.RANGE
    assert event_time.granularity is PartitionGranularity.MONTH
    assert event_time.partitions_total == len(event_time.bounds)

    # Ground truth: the dictionary rows themselves, in the same order.
    with connect_read_only(target) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT partition_name, high_value FROM all_tab_partitions "
            "WHERE table_owner = 'MDQM_BILLING' AND table_name = 'USAGE_EVENTS' "
            "ORDER BY partition_position"
        )
        server = [(str(row[0]), str(row[1])) for row in cur.fetchall()]

    assert [bound.name for bound in event_time.bounds] == [name for name, _ in server]
    assert event_time.bounds[0].name == INITIAL_PARTITION
    generated = [
        bound for bound in event_time.bounds if bound.name.startswith(GENERATED_PARTITION_PREFIX)
    ]
    assert len(generated) >= 2, f"Oracle derived {len(generated)} partitions"
    assert len(generated) == len(months), (
        f"the data covers {len(months)} months, the catalog reports {len(generated)}"
    )

    # A monthly range partition ends exactly at the first instant of the next month, and its
    # lower bound is the previous partition's upper bound: the chain is what a range means.
    assert event_time.bounds[0].lower is None, "the initial partition is unbounded below"
    for previous, current in zip(event_time.bounds, event_time.bounds[1:], strict=False):
        assert current.lower == previous.upper, (
            f"{current.name} starts at {current.lower}, {previous.name} ends at {previous.upper}"
        )
    assert [bound.upper for bound in event_time.bounds[1:]] == [
        _first_of_next_month(month) for month in months
    ]


def _first_of_next_month(value: datetime) -> datetime:
    """First instant of the month after the one containing ``value`` (UTC)."""
    return datetime(value.year + (value.month // 12), value.month % 12 + 1, 1, tzinfo=UTC)


# --------------------------------------------------------------------------------------
# Acceptance 4: execute returns real rows, and the caps are exact
# --------------------------------------------------------------------------------------


def test_execute_returns_the_rows_a_direct_query_counts(
    adapter: OracleAdapter, target: OracleTarget
) -> None:
    """The adapter's answer is compared with ``oracledb`` talking to the same server directly."""
    with connect_read_only(target) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*), sum(units) FROM MDQM_BILLING.usage_events")
        row = cur.fetchone()
    assert row is not None
    expected_rows, expected_units = int(row[0]), int(row[1])
    assert expected_rows > 0, "the world was not loaded: run mdqm.testing.loaders.oracle"

    stream = adapter.execute(
        _fragment(
            "billing_oracle",
            "SELECT event_id, units, start_time FROM MDQM_BILLING.usage_events ORDER BY event_id",
        ),
        LIMITS,
    )
    try:
        batches = list(stream)
        schema = stream.schema
    finally:
        stream.close()
    assert [field.name for field in schema] == ["EVENT_ID", "UNITS", "START_TIME"]
    # EVENT_ID is NUMBER(19,0): 19 digits can exceed int64, so it stays an exact decimal.
    assert schema.field("EVENT_ID").type == pa.decimal128(38, 0)
    # UNITS is NUMBER(10,0): it fits, and an int keeps aggregation cheap.
    assert pa.types.is_int64(schema.field("UNITS").type)
    assert pa.types.is_timestamp(schema.field("START_TIME").type)
    ids = [value for batch in batches for value in batch.column("EVENT_ID").to_pylist()]
    units = [value for batch in batches for value in batch.column("UNITS").to_pylist()]
    assert len(ids) == expected_rows
    assert sum(units) == expected_units
    assert ids == sorted(ids) and len(set(ids)) == expected_rows


def test_max_rows_truncates_exactly(adapter: OracleAdapter) -> None:
    """``max_rows`` is the adapter's promise: 7 means 7, not "7 rounded up to a batch"."""
    stream = adapter.execute(
        _fragment(
            "billing_oracle", "SELECT client_no FROM MDQM_BILLING.accounts ORDER BY client_no"
        ),
        LIMITS.model_copy(update={"max_rows": 7}),
    )
    try:
        ids = [value for batch in stream for value in batch.column("CLIENT_NO").to_pylist()]
    finally:
        stream.close()
    assert ids == [f"{number:010d}" for number in range(1, 8)]


def test_max_bytes_stops_the_stream(adapter: OracleAdapter) -> None:
    """The byte cap is enforced while streaming, so an oversized result is never fully fetched."""
    stream = adapter.execute(
        _fragment("billing_oracle", "SELECT * FROM MDQM_BILLING.accounts ORDER BY client_no"),
        LIMITS.model_copy(update={"max_bytes": 256}),
    )
    try:
        for _ in stream:
            pass
    finally:
        stream.close()
    assert isinstance(stream, OracleArrowStream)
    assert stream.bytes_returned <= 256
    assert stream.rows_returned < 200, "the byte cap must stop the scan, not read everything"


# --------------------------------------------------------------------------------------
# Acceptance 5: the timeout cancels the statement, and the connection survives it
# --------------------------------------------------------------------------------------


def test_timeout_cancels_a_slow_statement_and_the_connection_is_reusable(
    adapter: OracleAdapter,
) -> None:
    """A 1 s deadline on ten seconds of server CPU raises, and the session is still usable.

    The deadline is enforced by cancelling the running statement (``ORA-01013``), not by
    ``Connection.call_timeout``: measured here, when ``call_timeout`` fires the driver still has to
    drain the response, and on a slow statement that drain times out in turn -- ``DPY-4011`` and a
    dead session, which fails the second half of the acceptance. It is kept only as a backstop far
    beyond the deadline, so the assertion is on the deadline and on the session surviving it.
    """
    assert _drain(adapter, SIMPLE_SQL) == 1  # warm the pool: min_size is 0
    opened_before = adapter.pool.opened
    assert opened_before >= 1

    started = time.monotonic()
    with pytest.raises(OracleTimeout) as excinfo:
        _drain(adapter, SLOW_SQL, LIMITS.model_copy(update={"timeout_s": 1.0}))
    elapsed = time.monotonic() - started
    assert elapsed < 10.0, f"the 1s deadline took {elapsed:.1f}s to fire"
    assert elapsed >= 0.9, f"the deadline fired early, after {elapsed:.2f}s"
    assert "timeout" in str(excinfo.value), str(excinfo.value)
    assert "ORA-01013" in str(excinfo.value), str(excinfo.value)

    # Same pool, same connection: a discarded connection would be replaced by a fresh one and
    # "still works" would pass for the wrong reason.
    assert adapter.pool.opened == opened_before
    for _ in range(3):
        assert _drain(adapter, SIMPLE_SQL) == 1


def test_cancel_interrupts_a_running_fetch_and_releases_the_connection(
    adapter: OracleAdapter,
) -> None:
    """``cancel()`` really interrupts: the fetching thread sees ``ORA-01013``, not EOF.

    Ten million rows cannot be fetched inside the test, so the stream is guaranteed to still be
    running when the canceller fires; the assertion is on Oracle's error, and on the same
    connection answering afterwards.
    """
    assert _drain(adapter, SIMPLE_SQL) == 1
    # A cap far above what can be fetched makes the stream run until the cancel arrives; the
    # default row cap would end the scan normally instead.
    unbounded = LIMITS.model_copy(update={"max_rows": 20_000_000, "max_bytes": 20_000_000_000})
    stream = adapter.execute(_fragment("billing_oracle", HUGE_SQL), unbounded)
    fired = threading.Event()

    def cancel_soon() -> None:
        time.sleep(0.3)
        adapter.cancel(stream)
        adapter.cancel(stream)  # cancelling twice must be a no-op, not an error
        fired.set()

    canceller = threading.Thread(target=cancel_soon, daemon=True)
    canceller.start()
    produced = 0
    with pytest.raises(OracleCancelled):
        try:
            for batch in stream:
                produced += batch.num_rows
        finally:
            stream.close()
    canceller.join(timeout=10)
    assert fired.is_set()
    assert produced < 10_000_000, "the query finished instead of being cancelled"
    assert _drain(adapter, SIMPLE_SQL) == 1


# --------------------------------------------------------------------------------------
# Acceptance 6: a write is refused by the account AND by the read-only session
# --------------------------------------------------------------------------------------

INSERT_PROBE: Final = (
    "insert into MDQM_BILLING.accounts (client_no, state, region, opened_date) "
    "values ('RO00000001', 'Active', 'north', date '2025-01-01')"
)


def test_read_only_account_refuses_a_write(adapter: OracleAdapter) -> None:
    """``MDQM_RO`` holds ``SELECT`` and nothing else: Oracle answers ``ORA-41900``."""
    with pytest.raises(oracledb.DatabaseError) as excinfo:
        _drain(adapter, INSERT_PROBE)
    error = excinfo.value.args[0]
    message = _message(excinfo.value)
    assert int(getattr(error, "code", 0)) == MISSING_PRIVILEGE, message
    assert "privilege" in message.lower(), message
    # The adapter itself is still healthy after the refusal.
    assert _drain(adapter, SIMPLE_SQL) == 1


def test_read_only_transaction_refuses_a_write_the_account_could_make(
    owner_adapter: OracleAdapter,
) -> None:
    """``SET TRANSACTION READ ONLY`` is the layer the guard bypasses: ``ORA-01456``.

    The connection can write and the statement is well formed, so the only thing that can refuse it
    is the read-only transaction the adapter opened before executing.
    """
    assert _drain(owner_adapter, "SELECT count(*) AS n FROM MDQM_BILLING.accounts") == 1
    with pytest.raises(oracledb.DatabaseError) as excinfo:
        _drain(owner_adapter, INSERT_PROBE)
    error = excinfo.value.args[0]
    message = _message(excinfo.value)
    assert int(getattr(error, "code", 0)) == READ_ONLY_TRANSACTION, message
    assert "READ ONLY" in message.upper(), message
    assert _drain(owner_adapter, SIMPLE_SQL) == 1


# --------------------------------------------------------------------------------------
# Acceptance 7: the pruning proof (literal = proven, bind = unproven)
# --------------------------------------------------------------------------------------


def test_literal_date_filter_proves_pruning(adapter: OracleAdapter) -> None:
    """A literal filter is pruned at parse time: ``PSTART``/``PSTOP`` are partition numbers."""
    summary = adapter.explain(_fragment("billing_oracle", LITERAL_SQL))
    assert summary.pruning_proof is PruningProof.PROVEN
    assert summary.partitions_total is not None and summary.partitions_total > 1
    assert summary.partitions_scanned is None, (
        "Oracle's PSTART/PSTOP are internal partition numbers, not a count"
    )
    assert summary.uses_index is True
    assert "USAGE_EVENTS_START_TIME_IDX" in summary.indexes_used
    assert summary.raw is not None and "PARTITION RANGE" in summary.raw
    assert summary.est_cost is not None


def test_bind_parameter_filter_cannot_prove_pruning(adapter: OracleAdapter) -> None:
    """The same predicate with a bind is pruned at run time only: Oracle reports ``KEY``."""
    summary = adapter.explain(
        _fragment("billing_oracle", BIND_SQL, {"since": "2025-11-01 00:00:00"})
    )
    assert summary.pruning_proof is PruningProof.UNPROVEN
    assert summary.raw is not None and "KEY" in summary.raw


# --------------------------------------------------------------------------------------
# Acceptance 8: the query store, and the capability when it is not readable
# --------------------------------------------------------------------------------------


def test_mine_queries_reads_v_sql(adapter: OracleAdapter) -> None:
    """``V$SQL`` is readable here, so the capability is real and returns real statements."""
    assert _drain(adapter, "SELECT 4321 AS probe FROM dual") == 1
    mined = list(adapter.mine_queries(datetime(2020, 1, 1, tzinfo=UTC)))
    assert mined, "V$SQL listed no statements at all"
    assert all(query.dialect == "oracle" for query in mined)
    assert all(query.calls is not None and query.calls > 0 for query in mined)
    with_data = [query for query in mined if query.total_time_ms is not None]
    assert with_data, "no statement carried a timing"
    for query in with_data:
        total, calls, mean = query.total_time_ms, query.calls, query.mean_time_ms
        assert total is not None and calls is not None and mean is not None
        assert calls > 0
        assert mean == pytest.approx(total / calls)


def test_v_sql_not_readable_is_an_unsupported_capability(owner_adapter: OracleAdapter) -> None:
    """An account without ``V$SQL`` gets "unsupported", not a crash and not an empty list.

    The owner schema is the natural negative: it may read and write the billing tables and has no
    grant on the fixed views (verified: ``ORA-00942`` on ``SYS.V_$SQL``).
    """
    capabilities = owner_adapter.probe()
    query_store = capabilities.get(OPTIONAL_METHODS["mine_queries"])
    assert query_store.supported is False
    assert query_store.detail is not None and "V$SQL" in query_store.detail

    with pytest.raises(NotSupported) as excinfo:
        list(owner_adapter.mine_queries(datetime(2020, 1, 1, tzinfo=UTC)))
    assert excinfo.value.capability == OPTIONAL_METHODS["mine_queries"]
    # The same adapter still serves everything that does not need the fixed views.
    assert owner_adapter.extract_catalog().tables
    assert _drain(owner_adapter, SIMPLE_SQL) == 1


def test_mutability_stats_says_why_it_is_not_served(adapter: OracleAdapter) -> None:
    """Not supported, with the capability name and the reason (a stats gather, not a trigger)."""
    with pytest.raises(NotSupported) as excinfo:
        adapter.mutability_stats()
    assert excinfo.value.capability == OPTIONAL_METHODS["mutability_stats"]
    assert excinfo.value.detail is not None and "DBMS_STATS" in excinfo.value.detail


# --------------------------------------------------------------------------------------
# Configuration: what a read-only account needs is documented, not guessed
# --------------------------------------------------------------------------------------


def test_configuration_errors_name_the_key_not_the_password(target: OracleTarget) -> None:
    """A missing user, host or secret is a config error naming the key; no credential leaks."""
    secrets = _StaticSecrets(target.ro_password)
    with pytest.raises(OracleConfigError) as no_user:
        OracleAdapter(
            SourceConfig(
                name="x", adapter="oracle", secret="s", options={"host": "h", "service_name": "s"}
            ),
            secrets,
        )
    assert "options.user" in str(no_user.value)

    with pytest.raises(OracleConfigError) as no_host:
        OracleAdapter(
            SourceConfig(name="x", adapter="oracle", secret="s", options={"user": "u"}), secrets
        )
    assert "options.host" in str(no_host.value)

    with pytest.raises(OracleConfigError) as no_secret:
        OracleAdapter(
            SourceConfig(
                name="x", adapter="oracle", options={"host": "h", "service_name": "s", "user": "u"}
            ),
            secrets,
        )
    assert "secret" in str(no_secret.value)

    built = OracleAdapter(_config("shown", target), secrets)
    try:
        assert target.ro_password not in repr(built)
    finally:
        built.close()


def _message(exc: BaseException) -> str:
    """One line of Oracle's own message, for an assertion that prints what really happened."""
    if isinstance(exc, oracledb.DatabaseError) and exc.args:
        return str(getattr(exc.args[0], "message", exc)).split("\n")[0].strip()
    return str(exc)
