"""A04 acceptance: query-store mining and mutability counters against the live database.

Needs the ``pg`` profile and the loaded world (docs/dev-env.md)::

    python scripts/dev.py up pg
    python -m mdqm.testing.loaders.postgres --small
    python -m pytest tests/integration/test_pg_stats.py --require-services -q

Every number here is compared against something the adapter did not invent: statements this test
ran and then found in ``pg_stat_statements`` through a direct ``psycopg`` query, counters read
twice around a change this test made itself, and a database where the extension really is absent.

Two properties of ``pg_stat_statements`` are asserted rather than assumed, because the whole
feature depends on them:

* the counters are **cumulative**, so the miner reports deltas -- the same total twice would count
  one execution twice (stage D9 sums what it is handed);
* another role's statement text is masked as ``<insufficient privilege>`` unless the reader holds
  ``pg_read_all_stats``, which is why ``docker/pg/init/01-init.sh`` grants it to ``mdqm_ro``.
  Without that grant mining returns only our own reads and looks calm about it.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime

import psycopg
import pytest

from mdqm.adapters.base import NotSupported
from mdqm.adapters.postgresql import PostgresAdapter
from mdqm.adapters.postgresql_stats import MASKED_QUERY
from mdqm.config.settings import SourceConfig
from mdqm.models import Fragment, Limits, RawQuery, TableStats
from mdqm.sql.fingerprint import analyze, fingerprint
from mdqm.testing.loaders.postgres import PgTarget, target_from_environ

LIMITS = Limits(timeout_s=10.0, max_rows=100, max_bytes=1_000_000, max_partitions=8)

EXTENSIONLESS_DATABASE = "mdqm_a04_noext"
"""A database with no ``CREATE EXTENSION pg_stat_statements``: the "not installed" case."""

MUTABLE_TABLE = "mdqm_a04_mutable"
"""A table this test owns, so nothing else can move its counters while it is measured."""

UPDATED_ROWS = 3


class _StaticSecrets:
    """A secret provider returning the throwaway dev password for any reference."""

    def __init__(self, password: str) -> None:
        self._password = password

    def get(self, name: str) -> str:
        """Resolve any reference to the dev password (config carries the name, never the value)."""
        assert name, "a secret reference must be a name, never an empty string"
        return self._password


def _unavailable(request: pytest.FixtureRequest, reason: str) -> None:
    """Skip, or fail when ``--require-services`` was given (a skipped test is not a pass)."""
    if request.config.getoption("--require-services"):
        pytest.fail(f"required service unavailable: {reason}")
    pytest.skip(f"service unavailable: {reason}")


def _config(name: str, database: str, target: PgTarget) -> SourceConfig:
    """A source config for one database, connected as the read-only role."""
    return SourceConfig(
        name=name,
        adapter="postgresql",
        secret="pg-a04",
        options={
            "host": target.host,
            "port": target.port,
            "dbname": database,
            "user": target.ro_user,
            "connect_timeout": 5,
        },
    )


def _run(adapter: PostgresAdapter, statement: str) -> int:
    """Run a statement through the adapter and return its row count (closing the stream)."""
    stream = adapter.execute(
        Fragment(
            source=adapter.source, dialect="postgres", sql=statement, exec_ts=datetime.now(UTC)
        ),
        LIMITS,
    )
    rows = 0
    try:
        for batch in stream:
            rows += batch.num_rows
    finally:
        stream.close()
    return rows


def _fingerprint(text: str) -> str | None:
    """The mined/statement fingerprint, or ``None`` when that text does not parse."""
    return analyze(text, "postgres").fingerprint


def _find(mined: list[RawQuery], statement: str) -> list[RawQuery]:
    """Mined rows for ``statement``, matched by fingerprint (not by byte equality of the text)."""
    wanted = _fingerprint(statement)
    return [row for row in mined if _fingerprint(row.text) == wanted]


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


@pytest.fixture
def crm_adapter(target: PgTarget, secrets: _StaticSecrets) -> Iterator[PostgresAdapter]:
    """A fresh adapter per test: its query-store watermark starts empty (a baseline)."""
    adapter = PostgresAdapter(_config("crm-a04", "crm", target), secrets)
    try:
        yield adapter
    finally:
        adapter.close()


@pytest.fixture
def billing_adapter(target: PgTarget, secrets: _StaticSecrets) -> Iterator[PostgresAdapter]:
    adapter = PostgresAdapter(_config("billing-a04", "billing", target), secrets)
    try:
        yield adapter
    finally:
        adapter.close()


@pytest.fixture
def mutable_table(target: PgTarget) -> Iterator[str]:
    """A throwaway table in ``billing`` whose counters this test is the only writer of."""
    with psycopg.connect(target.dsn("billing")) as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS public.{MUTABLE_TABLE}")
        cur.execute(f"CREATE TABLE public.{MUTABLE_TABLE} (id int, note text)")
        conn.commit()
    try:
        yield MUTABLE_TABLE
    finally:
        with psycopg.connect(target.dsn("billing")) as conn, conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS public.{MUTABLE_TABLE}")
            conn.commit()


@pytest.fixture
def extensionless_database(target: PgTarget) -> Iterator[str]:
    """A real database on the same server where ``pg_stat_statements`` was never created.

    ``CREATE EXTENSION`` is per database, so this is exactly the "operator has not enabled it"
    case the capability report must survive. Created and dropped by the test; ``WITH (FORCE)``
    closes a connection a failed run left behind.
    """
    admin = target.dsn("crm")
    with psycopg.connect(admin, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {EXTENSIONLESS_DATABASE} WITH (FORCE)")
        cur.execute(f"CREATE DATABASE {EXTENSIONLESS_DATABASE} OWNER postgres")
        cur.execute(f"GRANT CONNECT ON DATABASE {EXTENSIONLESS_DATABASE} TO {target.ro_user}")
    try:
        yield EXTENSIONLESS_DATABASE
    finally:
        with psycopg.connect(admin, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {EXTENSIONLESS_DATABASE} WITH (FORCE)")


def _stats(adapter: PostgresAdapter) -> dict[str, TableStats]:
    """One reading of the counters, keyed by ``schema.table``."""
    return {item.table: item for item in adapter.mutability_stats()}


def _delta(after: TableStats | None, before: TableStats | None, field: str) -> int:
    """Growth of one counter between two readings (missing counter counts as zero)."""
    new = 0 if after is None else int(getattr(after, field) or 0)
    old = 0 if before is None else int(getattr(before, field) or 0)
    return new - old


def _await_counter(
    adapter: PostgresAdapter,
    table: str,
    field: str,
    at_least: int,
    before: TableStats | None,
) -> TableStats:
    """Read the counters until ``field`` grew by ``at_least``; PostgreSQL flushes them on commit.

    A backend flushes its pending statistics at most once per ``PGSTAT_MIN_INTERVAL`` (1 s), so a
    reading taken in the same millisecond as the owner's commit can legitimately lag behind.
    """
    deadline = time.monotonic() + 10.0
    while True:
        reading = _stats(adapter).get(table)
        if _delta(reading, before, field) >= at_least:
            assert reading is not None
            return reading
        if time.monotonic() > deadline:
            pytest.fail(
                f"{table}.{field} did not grow by {at_least} within 10 s: "
                f"before={before}, after={reading}"
            )
        time.sleep(0.1)


# --------------------------------------------------------------------------------------
# Acceptance 1: the probe claims query_store only when the view is really readable
# --------------------------------------------------------------------------------------


def test_probe_claims_query_store_and_mutability_counters(billing_adapter: PostgresAdapter) -> None:
    """Both claims are checked against a server where they hold, and the detail says where from."""
    capabilities = billing_adapter.probe()
    query_store = capabilities.get("query_store")
    assert query_store.supported is True, query_store.detail
    assert "pg_stat_statements" in (query_store.detail or "")
    assert "pg_stat_user_tables" in (capabilities.get("mutability_counters").detail or "")
    assert capabilities.supports("mutability_counters") is True


def test_the_read_only_role_reads_the_query_store_and_other_roles_text(target: PgTarget) -> None:
    """``mdqm_ro`` can select the view *and* is not limited to its own statements.

    This is the assertion the whole mining source rests on: a read-only login that sees only its
    own reads would mine nothing of the organisation's history.
    """
    with psycopg.connect(target.dsn("crm", read_only=True)) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT has_table_privilege('pg_stat_statements', 'SELECT'),"
            "       (SELECT count(*) FROM pg_stat_statements),"
            "       pg_has_role(current_user, 'pg_read_all_stats', 'MEMBER')"
        )
        row = cur.fetchone()
        assert row is not None
        readable, statements, all_users = row
        assert readable is True, "mdqm_ro cannot select pg_stat_statements"
        assert statements >= 1, "the view is empty: run the unit's queries first"
        assert all_users is True, (
            "mdqm_ro lacks pg_read_all_stats: another role's statement text is masked as "
            f"{MASKED_QUERY!r}, so mining would see only our own reads (see docker/pg/init)"
        )
        cur.execute(
            "SELECT count(*) FROM pg_stat_statements WHERE userid <> "
            "(SELECT oid FROM pg_roles WHERE rolname = current_user) AND query <> %s",
            (MASKED_QUERY,),
        )
        others = cur.fetchone()
        assert others is not None and others[0] >= 1


# --------------------------------------------------------------------------------------
# Acceptance 1: mining returns the statements the test just ran
# --------------------------------------------------------------------------------------


def test_mining_returns_the_statement_this_test_just_ran(crm_adapter: PostgresAdapter) -> None:
    """A statement run through the adapter comes back with its counters and its own text."""
    statement = "SELECT count(*) FROM customers"
    assert _run(crm_adapter, statement) == 1

    mined = list(crm_adapter.mine_queries(datetime.now(UTC)))
    found = _find(mined, statement)
    assert found, f"mining did not report {statement!r}; mined {len(mined)} statements"
    row = max(found, key=lambda item: item.calls or 0)
    assert row.calls is not None and row.calls >= 1
    assert row.total_time_ms is not None and row.total_time_ms >= 0.0
    assert row.mean_time_ms is not None
    assert row.rows is not None and row.rows >= 0
    assert row.user is not None, "the view reports whose statement it was"
    assert row.dialect == "postgres"
    # pg_stat_statements 1.10 keeps totals, not timestamps: a fabricated last_seen would be a lie.
    assert row.last_seen is None


def test_mined_text_is_fingerprint_ready(crm_adapter: PostgresAdapter) -> None:
    """``mdqm.sql.fingerprint`` (S08) parses the mined text and finds the table it reads.

    The point of unwrapping ``DECLARE ... CURSOR FOR``: without it the mined text would be the
    cursor declaration and every read through our own server-side cursor would fail to parse.
    """
    statement = "SELECT count(*) FROM customers"
    assert _run(crm_adapter, statement) == 1
    found = _find(list(crm_adapter.mine_queries(datetime.now(UTC))), statement)
    assert found
    facts = analyze(max(found, key=lambda item: item.calls or 0).text, "postgres")
    assert facts.ok, facts.error
    assert facts.tables == ("customers",)
    assert facts.fingerprint == fingerprint(statement, "postgres")


def test_the_same_statement_twice_is_one_row_with_two_calls(
    billing_adapter: PostgresAdapter,
) -> None:
    """Two executions are one mined row with ``calls == 2``, not two rows.

    The first call only establishes the baseline: this adapter reports *deltas*, so the two runs
    below are the change, and the totals the view already held are not re-delivered.
    """
    statement = "SELECT count(*) FROM accounts"
    list(billing_adapter.mine_queries(datetime.now(UTC)))

    assert _run(billing_adapter, statement) == 1
    assert _run(billing_adapter, statement) == 1

    found = _find(list(billing_adapter.mine_queries(datetime.now(UTC))), statement)
    assert len(found) == 1, [row.text for row in found]
    assert found[0].calls == 2, "cumulative counters must be handed over as deltas, once"


def test_a_second_reading_does_not_repeat_the_same_counters(
    billing_adapter: PostgresAdapter,
) -> None:
    """Cumulative totals handed over twice would double one execution (stage D9 sums them)."""
    statement = "SELECT count(*) FROM accounts"
    list(billing_adapter.mine_queries(datetime.now(UTC)))
    assert _run(billing_adapter, statement) == 1
    first = _find(list(billing_adapter.mine_queries(datetime.now(UTC))), statement)
    assert [row.calls for row in first] == [1]

    again = _find(list(billing_adapter.mine_queries(datetime.now(UTC))), statement)
    assert again == [], "no execution happened between the two readings"


def test_literal_values_are_already_normalised_by_the_view(crm_adapter: PostgresAdapter) -> None:
    """PostgreSQL reports ``$1`` instead of a literal, so two values are one statement."""
    first = "SELECT count(*) FROM customers WHERE cust_id > 10"
    second = "SELECT count(*) FROM customers WHERE cust_id > 20"
    assert _run(crm_adapter, first) == 1
    assert _run(crm_adapter, second) == 1

    mined = list(crm_adapter.mine_queries(datetime.now(UTC)))
    placeholders = [row for row in mined if "FROM customers WHERE cust_id > $" in row.text]
    assert placeholders, [row.text for row in mined if "customers" in row.text][:5]
    row = max(placeholders, key=lambda item: item.calls or 0)
    assert row.calls is not None and row.calls >= 2
    assert analyze(row.text, "postgres").ok, "a placeholder must still parse and fingerprint"
    assert "10" not in row.text and "20" not in row.text


# --------------------------------------------------------------------------------------
# Acceptance 3: the extension is absent -> unsupported, never a crash
# --------------------------------------------------------------------------------------


def test_missing_extension_is_unsupported_and_the_pipeline_falls_back(
    target: PgTarget, secrets: _StaticSecrets, extensionless_database: str
) -> None:
    """In a database where the extension was never created: unsupported, no crash, no empty calm."""
    adapter = PostgresAdapter(_config("a04-noext", extensionless_database, target), secrets)
    try:
        capabilities = adapter.probe()
        query_store = capabilities.get("query_store")
        assert query_store.supported is False
        assert "pg_stat_statements" in (query_store.detail or "")
        assert "shared_preload_libraries" in (query_store.detail or "")
        with pytest.raises(NotSupported) as raised:
            list(adapter.mine_queries(datetime.now(UTC)))
        assert raised.value.capability == "query_store"
        # The other counters come from a core view that exists in every database: still supported.
        assert capabilities.supports("mutability_counters") is True
        assert adapter.mutability_stats() == []
    finally:
        adapter.close()


# --------------------------------------------------------------------------------------
# Acceptance 2: the counters move, and they tell the two table shapes apart
# --------------------------------------------------------------------------------------


def test_mutability_counters_move_when_rows_change(
    billing_adapter: PostgresAdapter, target: PgTarget, mutable_table: str
) -> None:
    """Insert, update and delete rows as the owner; the counters grow by exactly that much."""
    table = f"public.{mutable_table}"
    before = _stats(billing_adapter).get(table)
    assert before is not None, "a freshly created table is listed with zero counters"
    assert before.rows_inserted == 0

    with psycopg.connect(target.dsn("billing")) as conn, conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO public.{mutable_table} (id, note) "
            "VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')"
        )
        cur.execute(f"UPDATE public.{mutable_table} SET note = 'x' WHERE id <= {UPDATED_ROWS}")
        cur.execute(f"DELETE FROM public.{mutable_table} WHERE id = 4")
        conn.commit()

    after = _await_counter(billing_adapter, table, "rows_inserted", 4, before)
    assert _delta(after, before, "rows_inserted") == 4
    assert _delta(after, before, "rows_updated") >= UPDATED_ROWS
    assert _delta(after, before, "rows_deleted") >= 1
    assert _delta(after, before, "rows_modified_since_analyze") >= 7


def test_counters_distinguish_an_append_only_table_from_an_updated_one(
    billing_adapter: PostgresAdapter, target: PgTarget
) -> None:
    """``usage_events`` only ever grows; ``accounts_hist`` collects updates -- measured here.

    The events were loaded by ``COPY``, so ``n_tup_ins`` is large and ``n_tup_upd`` is zero; the
    one update to ``accounts_hist`` below is ours, set to the same value so no data changes.
    """
    before = _stats(billing_adapter)
    events = [item for name, item in before.items() if name.startswith("public.usage_events_p")]
    assert events, "the world's usage_events partitions are missing: load the world first"
    assert sum(item.rows_inserted or 0 for item in events) > 0
    assert sum(item.rows_updated or 0 for item in events) == 0, "append-only: no updates"

    history = before.get("public.accounts_hist")
    assert history is not None
    with psycopg.connect(target.dsn("billing")) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE accounts_hist SET state = state WHERE hist_id IN "
            "(SELECT hist_id FROM accounts_hist ORDER BY hist_id LIMIT 5)"
        )
        assert cur.rowcount == 5
        conn.commit()

    after = _await_counter(billing_adapter, "public.accounts_hist", "rows_updated", 5, history)
    assert _delta(after, history, "rows_updated") == 5
    assert _delta(after, history, "rows_deleted") == 0
    # Nothing about the data changed: `state = state` writes the same value back.
    assert after.rows_inserted == history.rows_inserted
