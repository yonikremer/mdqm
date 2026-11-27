"""D03 acceptance: PostgreSQL schema + loader for the synthetic world.

Needs the ``pg`` compose profile (``python scripts/dev.py up pg``)::

    python -m pytest tests/integration/test_seed_pg.py --require-services -q

Why the time bounds in the pruning test are anchored to the world and not to ``now()``: the
world's event window ends at :data:`mdqm.testing.world.ANCHOR` (2026-01-01), while the machine
clock keeps moving past it. ``start_time >= now() - 30 days`` would prune to zero partitions
once the clock is past the anchor and would satisfy "<=" 2 vacuously, proving nothing. The
window's *last 30 days* (``anchor - 30 days``) touches exactly one event-time partition, and
the identical bound on ``insertion_time`` cannot prune at all: it scans every partition of
``usage_events``, because that table is partitioned on event time and has no index on
ingestion time.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import psycopg
import pytest
from psycopg import sql

from mdqm.testing.loaders.postgres import (
    HASH_PARTITIONS,
    LoadReport,
    PgTarget,
    load,
    monthly_partitions,
    target_from_environ,
)
from mdqm.testing.world import ANCHOR, World, WorldConfig, build_world

WINDOW_START: datetime = ANCHOR - timedelta(days=30)
"""First instant of the world window's last 30 days."""

READ_ONLY_SQLSTATES: frozenset[str] = frozenset(
    {
        psycopg.errors.ReadOnlySqlTransaction.sqlstate,
        psycopg.errors.InsufficientPrivilege.sqlstate,
    }
)
"""25006 (read-only transaction) and 42501 (missing privilege): the two read-only layers."""

LOADED_TABLES: dict[str, tuple[str, ...]] = {
    "crm": ("customers",),
    "billing": ("accounts", "accounts_hash", "accounts_hist", "usage_events", "usage_by_load"),
    "erp": ("customers",),
}


@dataclass(frozen=True, slots=True)
class Seed:
    """One loading run: the report, the world it came from, and the target it went to."""

    target: PgTarget
    world: World
    first: LoadReport
    second: LoadReport
    counts_first: dict[str, int]
    counts_second: dict[str, int]


def _unavailable(request: pytest.FixtureRequest, reason: str) -> None:
    """Skip, or fail when ``--require-services`` was given."""
    if request.config.getoption("--require-services"):
        pytest.fail(f"required service unavailable: {reason}")
    pytest.skip(f"service unavailable: {reason}")


def _server_counts(target: PgTarget, database: str, tables: Sequence[str]) -> dict[str, int]:
    """``SELECT count(*)`` per table, as the owner: proves COPY landed, not just that we tried."""
    out: dict[str, int] = {}
    with psycopg.connect(target.dsn(database)) as conn, conn.cursor() as cur:
        for table in tables:
            cur.execute(sql.SQL("SELECT count(*) FROM public.{}").format(sql.Identifier(table)))
            row = cur.fetchone()
            out[f"{database}.{table}"] = 0 if row is None else int(row[0])
    return out


@pytest.fixture(scope="module")
def seed(request: pytest.FixtureRequest) -> Iterator[Seed]:
    """Load the small world twice (the second run is the idempotency check)."""
    try:
        target = target_from_environ()
    except Exception as exc:
        _unavailable(request, f"PostgreSQL connection settings: {exc}")
        raise AssertionError("unreachable") from exc

    try:
        with psycopg.connect(target.dsn("crm"), connect_timeout=5):
            pass
    except psycopg.Error as exc:
        _unavailable(request, f"PostgreSQL at {target.host}:{target.port}: {exc}")
        raise AssertionError("unreachable") from exc

    world = build_world(WorldConfig.small())
    first = load(target, world=world)
    counts_first = {
        name: count
        for database, tables in LOADED_TABLES.items()
        for name, count in _server_counts(target, database, tables).items()
    }
    second = load(target, world=world)
    counts_second = {
        name: count
        for database, tables in LOADED_TABLES.items()
        for name, count in _server_counts(target, database, tables).items()
    }
    yield Seed(
        target=target,
        world=world,
        first=first,
        second=second,
        counts_first=counts_first,
        counts_second=counts_second,
    )


# --------------------------------------------------------------------------------------
# Acceptance 1: running the loader twice yields identical row counts
# --------------------------------------------------------------------------------------


def test_loader_twice_yields_identical_row_counts(seed: Seed) -> None:
    """Drop/create is idempotent: same counts, and they match the world the loader was given."""
    world = seed.world
    assert seed.counts_first == seed.counts_second
    assert seed.first.counts == seed.second.counts

    expected = {
        "crm.customers": len(world.crm.customers),
        "billing.accounts": len(world.billing.accounts),
        "billing.accounts_hash": len(world.billing.accounts),
        "billing.accounts_hist": len(world.billing.accounts_hist),
        "billing.usage_events": world.config.event_count,
        "billing.usage_by_load": world.config.event_count,
        "erp.customers": len(world.erp.customers),
    }
    assert seed.counts_first == expected

    # The planted shape survives the copy: duplicates in erp, NULL raw limits, orphan keys.
    assert len(world.erp.customers) > len(world.crm.customers), "duplicate erp rows are planted"
    assert world.expected.orphan_event_count > 0
    assert any(account.credit_limit is None for account in world.billing.accounts)


def test_erp_has_no_primary_key_and_preserves_duplicates(seed: Seed) -> None:
    """D03 owns a table the world deliberately duplicates, so it must not gain a key."""
    with psycopg.connect(seed.target.dsn("erp")) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM pg_constraint "
            "WHERE conrelid = 'public.customers'::regclass AND contype = 'p'"
        )
        assert cur.fetchone() == (0,)
        cur.execute("SELECT count(*) - count(DISTINCT cust_no) FROM public.customers")
        duplicates = cur.fetchone()
        assert duplicates is not None and int(duplicates[0]) > 0


def test_planted_nulls_and_orphans_survive_the_copy(seed: Seed) -> None:
    world = seed.world
    null_limits = sum(1 for account in world.billing.accounts if account.credit_limit is None)
    with psycopg.connect(seed.target.dsn("billing")) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM public.accounts WHERE credit_limit IS NULL")
        assert cur.fetchone() == (null_limits,)
        cur.execute(
            "SELECT count(*) FROM public.usage_events e "
            "LEFT JOIN public.accounts a USING (client_no) WHERE a.client_no IS NULL"
        )
        assert cur.fetchone() == (world.expected.orphan_event_count,)


# --------------------------------------------------------------------------------------
# Acceptance 2: mdqm_ro cannot INSERT/UPDATE/DDL - assert the actual error
# --------------------------------------------------------------------------------------

FORBIDDEN_STATEMENTS: tuple[tuple[str, str], ...] = (
    (
        "insert",
        "INSERT INTO public.accounts (client_no, state, region, opened_date) "
        "VALUES ('RO000001', 'Active', 'north', DATE '2025-01-01')",
    ),
    ("update", "UPDATE public.accounts SET state = 'Hold'"),
    ("delete", "DELETE FROM public.accounts_hist"),
    ("truncate", "TRUNCATE public.accounts"),
    ("ddl", "CREATE TABLE public.ro_probe (i INTEGER)"),
    ("drop", "DROP TABLE public.accounts_hist"),
)


@pytest.mark.parametrize(
    ("label", "statement"),
    FORBIDDEN_STATEMENTS,
    ids=[statement[0] for statement in FORBIDDEN_STATEMENTS],
)
def test_read_only_role_cannot_write(seed: Seed, label: str, statement: str) -> None:
    """A write as ``mdqm_ro`` must be rejected by PostgreSQL, both read-only layers.

    Layer one is the role's ``default_transaction_read_only=on`` (SQLSTATE 25006); layer two is
    the missing privilege (SQLSTATE 42501), which is still there when the transaction is forced
    read-write. Asserting both proves the account has neither the setting nor the grant, rather
    than just one of them by accident.
    """
    rejections: list[tuple[str, str]] = []
    for mode, force_read_write in (("default", False), ("forced_rw", True)):
        with (
            psycopg.connect(seed.target.dsn("billing", read_only=True)) as conn,
            conn.cursor() as cur,
        ):
            if force_read_write:
                cur.execute("SET default_transaction_read_only = off")
                conn.commit()  # the setting applies to the *next* transaction
            with pytest.raises(psycopg.errors.Error) as excinfo:
                cur.execute(statement)
            conn.rollback()
        error = excinfo.value
        message = str(error).strip()
        assert message, f"{label}/{mode}: rejected with an empty message"
        assert error.sqlstate in READ_ONLY_SQLSTATES, (
            f"{label}/{mode}: unexpected SQLSTATE {error.sqlstate}: {message}"
        )
        rejections.append((mode, f"{error.sqlstate} {message}"))

    assert len(rejections) == 2
    if rejections[0][0] == "default":
        assert rejections[0][1].startswith("25006"), (
            f"{label}: the read-only transaction layer did not fire first: {rejections[0][1]}"
        )
    assert rejections[1][1].startswith("42501"), (
        f"{label}: privilege layer missing when the transaction was forced read-write: "
        f"{rejections[1][1]}"
    )


def test_read_only_role_is_read_only_and_can_select(seed: Seed) -> None:
    """``default_transaction_read_only`` is on for the role, and plain SELECT works."""
    with psycopg.connect(seed.target.dsn("billing", read_only=True)) as conn, conn.cursor() as cur:
        cur.execute("SHOW default_transaction_read_only")
        assert cur.fetchone() == ("on",)
        cur.execute("SHOW transaction_read_only")
        assert cur.fetchone() == ("on",)
        cur.execute("SELECT count(*) FROM public.usage_events")
        row = cur.fetchone()
        assert row is not None and int(row[0]) == seed.world.config.event_count


# --------------------------------------------------------------------------------------
# Acceptance 3: 12+ monthly partitions, event time prunes, insertion time cannot
# --------------------------------------------------------------------------------------


def _plan_relations(target: PgTarget, database: str, query: sql.Composable) -> list[str]:
    """Base-relation names the planner will read, from ``EXPLAIN (FORMAT JSON)``."""
    with psycopg.connect(target.dsn(database)) as conn, conn.cursor() as cur:
        cur.execute(sql.SQL("EXPLAIN (FORMAT JSON) {}").format(query))
        row = cur.fetchone()
    assert row is not None
    document = row[0]
    if isinstance(document, str):
        document = json.loads(document)
    names: list[str] = []

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        name = node.get("Relation Name")
        if isinstance(name, str):
            names.append(name)
        for child in node.get("Plans", []) or []:
            walk(child)

    for entry in document:
        walk(entry.get("Plan") if isinstance(entry, dict) else None)
    return names


def _partition_names(target: PgTarget, database: str, parent: str) -> list[str]:
    with psycopg.connect(target.dsn(database)) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT child.relname FROM pg_inherits i "
            "JOIN pg_class parent ON parent.oid = i.inhparent "
            "JOIN pg_class child ON child.oid = i.inhrelid "
            "WHERE parent.relname = %s ORDER BY child.relname",
            (parent,),
        )
        return [str(row[0]) for row in cur.fetchall()]


def test_twelve_plus_monthly_partitions(seed: Seed) -> None:
    """The catalog shows the monthly set the acceptance criterion counts."""
    partitions = _partition_names(seed.target, "billing", "usage_events")
    assert len(partitions) >= 12, f"only {len(partitions)} usage_events partitions"
    assert len(partitions) == len(monthly_partitions(ANCHOR, forward=0))
    assert partitions == sorted(partitions)
    assert partitions[-1] == "usage_events_p2025_12"

    by_load = _partition_names(seed.target, "billing", "usage_by_load")
    assert len(by_load) >= len(partitions), "ingestion time needs at least as many partitions"
    assert len(_partition_names(seed.target, "billing", "accounts_hash")) == HASH_PARTITIONS


def test_planned_pruning_event_time_versus_insertion_time(seed: Seed) -> None:
    """Event time prunes to one partition; ingestion time scans all of them."""
    by_start = _plan_relations(
        seed.target,
        "billing",
        sql.SQL("SELECT count(*) FROM public.usage_events WHERE start_time >= {}").format(
            sql.Literal(WINDOW_START)
        ),
    )
    event_partitions = [name for name in by_start if name.startswith("usage_events_p")]
    assert len(event_partitions) <= 2, f"start_time touched {event_partitions}"

    by_insertion = _plan_relations(
        seed.target,
        "billing",
        sql.SQL("SELECT count(*) FROM public.usage_events WHERE insertion_time >= {}").format(
            sql.Literal(WINDOW_START)
        ),
    )
    insertion_partitions = [name for name in by_insertion if name.startswith("usage_events_p")]
    total = len(_partition_names(seed.target, "billing", "usage_events"))
    assert len(insertion_partitions) == total, (
        f"insertion_time scanned {len(insertion_partitions)} of {total} partitions"
    )


def test_insertion_time_is_not_indexed(seed: Seed) -> None:
    """The asymmetry is the point: event time is indexed, ingestion time is not."""
    with psycopg.connect(seed.target.dsn("billing")) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT indexdef FROM pg_indexes WHERE tablename IN "
            "('usage_events', 'usage_events_p2025_12')"
        )
        definitions = " ".join(str(row[0]) for row in cur.fetchall())
    assert "start_time" in definitions
    assert "insertion_time" not in definitions


# --------------------------------------------------------------------------------------
# Acceptance 4: pg_stat_statements view is readable by mdqm_ro
# --------------------------------------------------------------------------------------


def test_read_only_role_reads_pg_stat_statements(seed: Seed) -> None:
    """``mdqm_ro`` can read the statistics view D09 will mine, and it is not empty."""
    with psycopg.connect(seed.target.dsn("billing")) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM public.usage_events")  # produce one statistic
    with psycopg.connect(seed.target.dsn("billing", read_only=True)) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM pg_stat_statements")
        row = cur.fetchone()
        assert row is not None and int(row[0]) > 0
        cur.execute("SELECT count(*) FROM public.status_lookup")
        assert cur.fetchone() == (3,)
