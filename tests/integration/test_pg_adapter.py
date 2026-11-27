"""A02 acceptance: the PostgreSQL adapter against the live compose database.

Needs the ``pg`` profile and the loaded world (docs/dev-env.md)::

    python scripts/dev.py up pg
    python -m mdqm.testing.loaders.postgres --small
    python -m pytest tests/integration/test_pg_adapter.py --require-services -q

Every assertion here compares the adapter against something the adapter did not produce: the
server's own ``pg_inherits`` and ``pg_indexes`` output, a direct ``psycopg`` query, or the error
PostgreSQL itself raises. A test that only re-read the adapter's own answer would pass even if the
adapter invented it.

The read-only proof is deliberately split in two layers, because "the write was refused" is only
meaningful when you say by whom:

* the **guard** (``mdqm.sql.guard``) refuses the statement before it is sent at all;
* the **server** refuses it anyway when the guard is bypassed, and the test shows PostgreSQL's own
  SQLSTATE and message for both a plain DML statement and a ``SELECT`` that would write.
"""

from __future__ import annotations

import importlib.metadata
import time
import tomllib
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import psycopg
import pyarrow as pa
import pytest

from mdqm.adapters.postgresql import (
    PostgresAdapter,
    PostgresCancelled,
    PostgresTimeout,
)
from mdqm.adapters.registry import AdapterRegistry
from mdqm.config.settings import SourceConfig
from mdqm.models import (
    Fragment,
    Limits,
    PartitionGranularity,
    Partitioning,
    PartitionStrategy,
    TableKind,
)
from mdqm.sql.guard import WRITE_BLOCKED, guard
from mdqm.testing.conformance import run_conformance
from mdqm.testing.loaders.postgres import PgTarget, target_from_environ

NOW: Final = datetime(2026, 1, 1, tzinfo=UTC)
"""A fixed ``exec_ts``: the adapter never substitutes the clock, so a constant is honest."""

LIMITS: Final = Limits(timeout_s=10.0, max_rows=100, max_bytes=1_000_000, max_partitions=8)

WRITE_PROBE_SEQUENCE: Final = "mdqm_a02_write_probe"
"""A sequence only this test creates, so ``nextval`` is a real write the server can judge."""

EVENT_PARTITIONS: Final = 14
"""Monthly ``usage_events`` partitions the loader creates (``PARTITION_MONTHS_BACK``)."""

HASH_PARTITIONS: Final = 4


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


def _config(name: str, database: str, target: PgTarget) -> SourceConfig:
    """A source config for one of the three databases, connected as the read-only role."""
    return SourceConfig(
        name=name,
        adapter="postgresql",
        secret=f"{name}_ro",
        options={
            "host": target.host,
            "port": target.port,
            "dbname": database,
            "user": target.ro_user,
            "connect_timeout": 5,
        },
    )


def _fragment(source: str, statement: str) -> Fragment:
    return Fragment(source=source, dialect="postgres", sql=statement, exec_ts=NOW)


def _drain(adapter: PostgresAdapter, statement: str, limits: Limits = LIMITS) -> int:
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
def target(request: pytest.FixtureRequest) -> PgTarget:
    """Where the compose PostgreSQL lives, from ``.env`` (never a hard-coded credential)."""
    try:
        found = target_from_environ()
    except Exception as exc:
        _unavailable(request, f"PostgreSQL connection settings: {exc}")
        raise AssertionError("unreachable") from exc
    try:
        with psycopg.connect(found.dsn("crm"), connect_timeout=5):
            pass
    except psycopg.Error as exc:
        _unavailable(request, f"PostgreSQL at {found.host}:{found.port}: {exc}")
        raise AssertionError("unreachable") from exc
    return found


@pytest.fixture(scope="module")
def secrets(target: PgTarget) -> _StaticSecrets:
    return _StaticSecrets(target.ro_password)


def _adapter(
    name: str, database: str, target: PgTarget, secrets: _StaticSecrets
) -> Iterator[PostgresAdapter]:
    adapter = PostgresAdapter(_config(name, database, target), secrets)
    try:
        yield adapter
    finally:
        adapter.close()


@pytest.fixture(scope="module")
def crm_adapter(target: PgTarget, secrets: _StaticSecrets) -> Iterator[PostgresAdapter]:
    yield from _adapter("crm", "crm", target, secrets)


@pytest.fixture(scope="module")
def billing_adapter(target: PgTarget, secrets: _StaticSecrets) -> Iterator[PostgresAdapter]:
    yield from _adapter("billing", "billing", target, secrets)


@pytest.fixture
def write_probe(target: PgTarget) -> Iterator[str]:
    """A sequence ``mdqm_ro`` may advance, so the read-only layer is what refuses it.

    Without the grant the server would answer ``42501 permission denied`` and prove nothing about
    the transaction; with it, only ``read-only transaction`` can stop the write.
    """
    with psycopg.connect(target.dsn("billing")) as conn, conn.cursor() as cur:
        # Drop first: a sequence left behind by an interrupted run would make CREATE raise a
        # UniqueViolation on pg_class rather than being silently reused.
        cur.execute(f"DROP SEQUENCE IF EXISTS public.{WRITE_PROBE_SEQUENCE}")
        cur.execute(f"CREATE SEQUENCE public.{WRITE_PROBE_SEQUENCE}")
        cur.execute(f"GRANT USAGE ON SEQUENCE public.{WRITE_PROBE_SEQUENCE} TO mdqm_ro")
        conn.commit()
    try:
        yield WRITE_PROBE_SEQUENCE
    finally:
        with psycopg.connect(target.dsn("billing")) as conn, conn.cursor() as cur:
            cur.execute(f"DROP SEQUENCE IF EXISTS public.{WRITE_PROBE_SEQUENCE}")
            conn.commit()


# --------------------------------------------------------------------------------------
# Acceptance 1: the conformance kit, with the live database
# --------------------------------------------------------------------------------------


def test_adapter_passes_the_conformance_kit(crm_adapter: PostgresAdapter) -> None:
    """``run_conformance`` is the only definition of "a real adapter" in this project.

    ``slow_fragment`` makes the kit prove ``Limits.timeout_s`` too: a ``SELECT pg_sleep(30)`` under
    a 50 ms deadline must raise, not complete.
    """
    run_conformance(
        lambda: crm_adapter,
        slow_fragment=_fragment("crm", "SELECT pg_sleep(30) AS slept"),
    )


def test_the_entry_point_declaration_builds_a_protocol_adapter() -> None:
    """The ``mdqm.adapters`` entry point names a factory the registry accepts (no database needed).

    Read from ``pyproject.toml`` rather than from installed metadata: a git worktree installs no
    distribution of its own, so the installed entry points still describe the checkout it was
    installed from.
    """
    declared = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    value = declared["project"]["entry-points"]["mdqm.adapters"]["postgresql"]
    loaded = importlib.metadata.EntryPoint(
        name="postgresql", value=value, group="mdqm.adapters"
    ).load()
    assert loaded is PostgresAdapter

    registry = AdapterRegistry(entry_points=lambda: ())
    registry.register("postgresql", loaded)
    config = SourceConfig(
        name="crm", adapter="postgresql", options={"user": "mdqm_ro", "dbname": "crm"}
    )
    adapter = registry.create(config)
    assert isinstance(adapter, PostgresAdapter)
    assert adapter.kind == "postgresql"
    assert adapter.dialect() == "postgres"
    adapter.close()


def test_probe_reports_a_version_and_the_partition_capability(
    billing_adapter: PostgresAdapter,
) -> None:
    """The probe names the server it reached, and only claims what this adapter can serve."""
    capabilities = billing_adapter.probe()
    assert capabilities.source == "billing"
    assert capabilities.adapter == "postgresql"
    assert capabilities.version is not None
    assert capabilities.version.startswith("PostgreSQL ")
    assert capabilities.supports("partition_metadata")
    # A03/A04 own EXPLAIN and query mining: unprobed, which blocks their features and says why.
    assert capabilities.is_unknown("explain_plan")
    assert capabilities.get("query_store").detail is not None


# --------------------------------------------------------------------------------------
# Acceptance 2: the catalog of billing, from the server's own catalogs
# --------------------------------------------------------------------------------------


def test_catalog_lists_usage_events_with_indexes_and_pk(
    billing_adapter: PostgresAdapter, target: PgTarget
) -> None:
    """``usage_events``: 14 monthly partitions, PK ``(event_id, start_time)``, start_time indexed.

    The index asymmetry is what the planner reads: event time is indexed, ingestion time is not.
    """
    snapshot = billing_adapter.extract_catalog()
    assert snapshot.source == "billing"
    assert snapshot.tables, "an empty catalog means a privilege problem, not a valid answer"

    partitioning = {entry.key: entry for entry in billing_adapter.extract_partitions()}
    event_time = partitioning.get(("start_time",))
    assert event_time is not None, f"no RANGE partition on start_time among {sorted(partitioning)}"
    assert isinstance(event_time, Partitioning)
    assert event_time.strategy is PartitionStrategy.RANGE
    assert len(event_time.bounds) == EVENT_PARTITIONS
    assert event_time.granularity is PartitionGranularity.MONTH

    # Ground truth: the server's own pg_inherits, not the adapter's answer.
    with psycopg.connect(target.dsn("billing")) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT child.relname FROM pg_inherits i "
            "JOIN pg_class parent ON parent.oid = i.inhparent "
            "JOIN pg_class child ON child.oid = i.inhrelid "
            "WHERE parent.relname = 'usage_events' ORDER BY child.relname"
        )
        server_partitions = [str(row[0]) for row in cur.fetchall()]
    assert len(server_partitions) == EVENT_PARTITIONS
    assert [bound.name for bound in event_time.bounds] == server_partitions

    table = snapshot.table("public.usage_events")
    assert table is not None
    assert table.kind is TableKind.PARTITIONED_TABLE
    assert table.primary_key == ("event_id", "start_time")
    indexed = {name for index in table.indexes for name in index.columns}
    assert "start_time" in indexed, f"indexes are {[i.columns for i in table.indexes]}"
    assert "insertion_time" not in indexed, "ingestion time must not look indexed"

    # The asymmetry the planner depends on, straight from pg_indexes.
    with psycopg.connect(target.dsn("billing")) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT indexdef FROM pg_indexes WHERE tablename = 'usage_events' ORDER BY indexname"
        )
        definitions = "\n".join(str(row[0]) for row in cur.fetchall())
    assert "start_time" in definitions
    assert "insertion_time" not in definitions

    # The hash table, so the catalog is not only reporting one table correctly.
    hashes = billing_adapter.extract_partitions()
    hash_entry = next(entry for entry in hashes if entry.strategy is PartitionStrategy.HASH)
    assert len(hash_entry.bounds) == HASH_PARTITIONS


def test_catalog_reports_erp_without_a_primary_key(
    target: PgTarget, secrets: _StaticSecrets
) -> None:
    """``erp.customers`` has no key on purpose: a catalog that invents one would break the joins."""
    adapter = PostgresAdapter(_config("erp", "erp", target), secrets)
    try:
        table = adapter.extract_catalog().table("public.customers")
        assert table is not None
        assert table.primary_key == ()
        assert table.row_estimate is not None and table.row_estimate > 0
    finally:
        adapter.close()


# --------------------------------------------------------------------------------------
# Acceptance 3: execute returns real rows, and the row cap is exact
# --------------------------------------------------------------------------------------


def test_execute_returns_the_rows_a_direct_query_counts(
    crm_adapter: PostgresAdapter, target: PgTarget
) -> None:
    """The adapter's answer is compared with ``psycopg`` talking to the same server directly."""
    with psycopg.connect(target.dsn("crm")) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM public.customers")
        row = cur.fetchone()
    assert row is not None
    expected = int(row[0])
    assert expected > 0, "the world was not loaded: run mdqm.testing.loaders.postgres"

    stream = crm_adapter.execute(
        _fragment(
            "crm", "SELECT cust_id, full_name, status FROM public.customers ORDER BY cust_id"
        ),
        LIMITS.model_copy(update={"max_rows": expected + 10}),
    )
    try:
        batches = list(stream)
        schema = stream.schema
    finally:
        stream.close()
    rows = [value for batch in batches for value in batch.column("cust_id").to_pylist()]
    assert len(rows) == expected
    assert rows == sorted(rows) and len(set(rows)) == expected
    assert [field.name for field in schema] == ["cust_id", "full_name", "status"]
    assert pa.types.is_int32(schema.field("cust_id").type)


def test_max_rows_truncates_exactly(crm_adapter: PostgresAdapter) -> None:
    """``max_rows`` is the adapter's promise: 7 means 7, not "7 rounded up to a batch"."""
    stream = crm_adapter.execute(
        _fragment("crm", "SELECT cust_id FROM public.customers ORDER BY cust_id"),
        LIMITS.model_copy(update={"max_rows": 7, "timeout_s": 10.0}),
    )
    try:
        ids = [value for batch in stream for value in batch.column("cust_id").to_pylist()]
    finally:
        stream.close()
    assert ids == [1, 2, 3, 4, 5, 6, 7]


def test_max_bytes_stops_the_stream(crm_adapter: PostgresAdapter) -> None:
    """The byte cap is enforced while streaming, so an oversized result is never fully fetched."""
    stream = crm_adapter.execute(
        _fragment("crm", "SELECT * FROM public.customers ORDER BY cust_id"),
        LIMITS.model_copy(update={"max_bytes": 256}),
    )
    try:
        for _ in stream:
            pass
    finally:
        stream.close()
    assert stream.bytes_returned <= 256
    assert stream.rows_returned < 200, "the byte cap must stop the scan, not read everything"


# --------------------------------------------------------------------------------------
# Acceptance 4: the timeout cancels the query, and the connection survives it
# --------------------------------------------------------------------------------------


def test_timeout_cancels_a_slow_query_and_the_connection_is_reusable(
    target: PgTarget, secrets: _StaticSecrets
) -> None:
    """A 1 s deadline on ``pg_sleep(30)`` raises, and the pooled connection is still usable.

    The proof that the connection was *reused* rather than discarded is the pool's connection
    count: it must not grow. A discarded connection would be replaced by a fresh one and the
    "still works" assertion would pass for the wrong reason.
    """
    adapter = PostgresAdapter(_config("billing", "billing", target), secrets)
    try:
        assert _drain(adapter, "SELECT 1 AS one") == 1  # warm the pool: min_size is 0
        before = adapter.pool.get_stats()["pool_size"]
        assert before >= 1
        started = time.monotonic()
        with pytest.raises(PostgresTimeout) as excinfo:
            _drain(
                adapter,
                "SELECT pg_sleep(30) AS slept",
                LIMITS.model_copy(update={"timeout_s": 1.0}),
            )
        elapsed = time.monotonic() - started
        assert elapsed < 10.0, f"the 1s deadline took {elapsed:.1f}s to fire"
        assert "timeout" in str(excinfo.value)

        # The same pool, the same connection: no new connection had to be opened.
        after = adapter.pool.get_stats()["pool_size"]
        assert after == before, f"the timed-out connection was discarded ({before} -> {after})"
        for _ in range(3):
            assert _drain(adapter, "SELECT 1 AS one") == 1
        assert adapter.pool.get_stats()["requests_waiting"] == 0
    finally:
        adapter.close()


def test_cancel_is_idempotent_and_releases_the_connection(billing_adapter: PostgresAdapter) -> None:
    """``cancel`` twice is a no-op; the stream then yields nothing and the pool is reused."""
    stream = billing_adapter.execute(
        _fragment("billing", "SELECT client_no FROM public.accounts ORDER BY client_no"), LIMITS
    )
    try:
        billing_adapter.cancel(stream)
        billing_adapter.cancel(stream)
        produced = 0
        try:
            for batch in stream:
                produced += batch.num_rows
        except (PostgresCancelled, PostgresTimeout):
            produced = 0  # reporting the cancellation by raising is allowed by the protocol
        assert produced == 0
    finally:
        stream.close()
    assert _drain(billing_adapter, "SELECT 1 AS one") == 1


# --------------------------------------------------------------------------------------
# Acceptance 5: a write is refused by the guard AND by the server
# --------------------------------------------------------------------------------------


def test_guard_refuses_writes_before_they_are_sent() -> None:
    """The AST layer: the statement never reaches a source, and the error carries a fix."""
    for statement in (
        "DELETE FROM public.accounts WHERE false",
        "UPDATE public.accounts SET state = 'Hold'",
        f"SELECT nextval('public.{WRITE_PROBE_SEQUENCE}')",
        "WITH d AS (DELETE FROM public.accounts_hist RETURNING hist_id) SELECT hist_id FROM d",
    ):
        result = guard(statement, "postgres")
        assert result.ok is False, statement
        assert result.error is not None
        assert result.error.code == WRITE_BLOCKED
        assert result.error.fix


def test_server_refuses_a_write_even_with_the_guard_bypassed(
    billing_adapter: PostgresAdapter,
) -> None:
    """Send DML straight to ``execute``: psycopg/the server refuses it, not our code.

    A server-side cursor cannot carry DML at all, so PostgreSQL answers with its own parser error
    (``42601``) instead of running it. That is the server refusing, which is what the acceptance
    asks for; the read-only-transaction layer is proved separately below, by a statement a cursor
    *can* carry.
    """
    with pytest.raises(psycopg.Error) as excinfo:
        stream = billing_adapter.execute(
            _fragment("billing", "DELETE FROM public.accounts WHERE false"), LIMITS
        )
        try:
            for _ in stream:
                pass
        finally:
            stream.close()
    message = str(excinfo.value).strip()
    assert excinfo.value.sqlstate == "42601", f"unexpected refusal: {message}"
    assert "DELETE" in message
    # The adapter itself is still healthy after the refusal.
    assert _drain(billing_adapter, "SELECT 1 AS one") == 1


def test_read_only_session_refuses_a_write_that_a_cursor_can_carry(
    billing_adapter: PostgresAdapter, write_probe: str
) -> None:
    """``SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY`` is the layer the guard bypasses.

    ``SELECT nextval(...)`` is a plain, cursor-able query that writes permanently, so only the
    transaction can refuse it. ``mdqm_ro`` is granted USAGE on the sequence for the duration of the
    test, otherwise the answer would be ``42501 permission denied`` and prove nothing.
    """
    statement = f"SELECT nextval('public.{write_probe}') AS n"
    assert guard(statement, "postgres").ok is False, "the guard must also refuse a sequence advance"
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction) as excinfo:
        stream = billing_adapter.execute(_fragment("billing", statement), LIMITS)
        try:
            for _ in stream:
                pass
        finally:
            stream.close()
    error = excinfo.value
    assert error.sqlstate == "25006"
    assert "read-only transaction" in str(error)
    assert _drain(billing_adapter, "SELECT 1 AS one") == 1


def test_read_only_role_is_read_only_at_the_session_level(
    billing_adapter: PostgresAdapter,
) -> None:
    """The session contract every pooled connection is configured with, as the server sees it."""
    stream = billing_adapter.execute(
        _fragment(
            billing_adapter.source,
            "SELECT current_setting('transaction_read_only') AS ro, "
            "current_setting('default_transaction_read_only') AS dro, "
            "current_setting('TimeZone') AS tz, current_setting('application_name') AS app",
        ),
        LIMITS,
    )
    try:
        batch = next(iter(stream))
        row = {name: batch.column(name).to_pylist()[0] for name in batch.schema.names}
    finally:
        stream.close()
    assert row["ro"] == "on"
    assert row["dro"] == "on"
    assert row["tz"] == "UTC"
    assert row["app"] == "mdqm:mdqm_ro"


def test_execute_rejects_a_non_sql_fragment(billing_adapter: PostgresAdapter) -> None:
    """A ``dsl``/``request`` fragment is a wiring mistake, and it says so instead of running."""
    fragment = Fragment(source="billing", dialect="postgres", dsl={"query": {}}, exec_ts=NOW)
    with pytest.raises(ValueError) as excinfo:
        billing_adapter.execute(fragment, LIMITS)
    assert "dsl" in str(excinfo.value)
