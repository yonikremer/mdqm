"""D05 acceptance: Oracle billing dataset, interval partitioning, read-only account, PSTART/PSTOP.

Needs the ``oracle`` compose profile (``python scripts/dev.py up oracle``, slow first start)::

    python -m pytest tests/integration/test_seed_oracle.py --require-services -q

What the assertions are anchored to:

* the **world** (``WorldConfig.small()``) for row counts, raw credit limits and the months the
  events cover - never to ``now()``, because the world's window ends at
  :data:`mdqm.testing.world.ANCHOR` while the machine clock keeps moving;
* **Oracle's own catalog** (``USER_TAB_PARTITIONS``, ``USER_PART_TABLES``, ``USER_INDEXES``,
  ``PLAN_TABLE``) for the partitioning and pruning claims, so the test cannot pass on a loader's
  opinion of itself;
* **Oracle's own error numbers** for the read-only claims. On 23ai a missing DML privilege is
  ``ORA-41900`` (the older ``ORA-01031`` for DDL and ``ORA-01456`` for a read-only transaction are
  still what the engine raises), so the test accepts the family and prints the exact message.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import oracledb
import pytest

from mdqm.testing.loaders.oracle import (
    ALL_TABLES,
    GENERATED_PARTITION_PREFIX,
    INITIAL_PARTITION,
    LoadReport,
    OracleTarget,
    connect_admin,
    connect_owner,
    connect_read_only,
    generation_expected_months,
    load,
    partitions_of,
    target_from_environ,
    to_stored_timestamp,
)
from mdqm.testing.world import World, WorldConfig, build_world

pytestmark = pytest.mark.slow

MISSING_PRIVILEGE: Final = frozenset({41900, 1031})
"""``ORA-41900`` (23ai missing INSERT/UPDATE/DELETE privilege) and ``ORA-01031`` (DDL,
``TRUNCATE``, ``DROP``)."""

READ_ONLY_TRANSACTION: Final = 1456
"""``ORA-01456``: a write inside ``SET TRANSACTION READ ONLY`` - the second read-only layer."""


@dataclass(frozen=True, slots=True)
class Seed:
    """One loading run, twice, plus what the catalog says about the result."""

    target: OracleTarget
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


def _counts(target: OracleTarget) -> dict[str, int]:
    """``SELECT count(*)`` per table as the owner: proves the rows landed, not that we tried."""
    with connect_owner(target) as conn, conn.cursor() as cur:
        out: dict[str, int] = {}
        for table in ALL_TABLES:
            cur.execute(f"select count(*) from {table}")
            row = cur.fetchone()
            out[f"billing.{table}"] = 0 if row is None else int(row[0])
        return out


def _next_month(value: datetime) -> datetime:
    """First instant of the month after the one containing ``value``."""
    return datetime(value.year + (value.month // 12), value.month % 12 + 1, 1)


@pytest.fixture(scope="module")
def seed(request: pytest.FixtureRequest) -> Iterator[Seed]:
    """Load the small world twice into Oracle (the second run is the idempotency check)."""
    try:
        target = target_from_environ()
    except Exception as exc:
        _unavailable(request, f"Oracle connection settings: {exc}")
        raise AssertionError("unreachable") from exc

    try:
        with oracledb.connect(
            user="sys",
            password=target.admin_password,
            dsn=target.dsn,
            mode=oracledb.AUTH_MODE_SYSDBA,
            tcp_connect_timeout=5,
        ):
            pass
    except oracledb.Error as exc:
        _unavailable(request, f"Oracle at {target.dsn}: {exc}")
        raise AssertionError("unreachable") from exc

    world = build_world(WorldConfig.small())
    first = load(target, world=world)
    counts_first = _counts(target)
    second = load(target, world=world)
    counts_second = _counts(target)
    yield Seed(
        target=target,
        world=world,
        first=first,
        second=second,
        counts_first=counts_first,
        counts_second=counts_second,
    )


# --------------------------------------------------------------------------------------
# Acceptance 1: idempotent loader, row counts equal the world's
# --------------------------------------------------------------------------------------


def test_loader_twice_yields_identical_row_counts(seed: Seed) -> None:
    """Drop/recreate is idempotent, and the counts are the world's, not just "some" rows."""
    world = seed.world
    assert seed.counts_first == seed.counts_second
    assert seed.first.counts == seed.second.counts

    expected = {
        "billing.accounts": len(world.billing.accounts),
        "billing.accounts_hist": len(world.billing.accounts_hist),
        "billing.status_lookup": 3,
        "billing.usage_events": world.config.event_count,
        "billing.usage_by_load": world.config.event_count,
    }
    assert seed.counts_first == expected
    assert seed.first.counts == expected


def test_accounts_keep_the_raw_limit_and_the_padded_client_number(seed: Seed) -> None:
    """``credit_limit`` is unrounded and ``client_no`` is the zero-padded ``VARCHAR2(10)``."""
    world = seed.world
    with connect_owner(seed.target) as conn, conn.cursor() as cur:
        cur.execute("select client_no, credit_limit from accounts order by client_no")
        loaded = [(str(row[0]), None if row[1] is None else int(row[1])) for row in cur.fetchall()]
        cur.execute("select count(*) from accounts where regexp_like(client_no, '^[0-9]{10}$')")
        padded = cur.fetchone()
        cur.execute(
            "select data_length from user_tab_columns "
            "where table_name = 'ACCOUNTS' and column_name = 'CLIENT_NO'"
        )
        width = cur.fetchone()

    expected = sorted(
        (account.client_no, account.credit_limit) for account in world.billing.accounts
    )
    assert loaded == expected
    assert padded is not None and int(padded[0]) == len(world.billing.accounts)
    assert width is not None and int(width[0]) == 10
    assert any(limit is None for _, limit in loaded), "the world plants NULL raw limits"


def test_planted_orphans_and_hebrew_lookup_survive_the_load(seed: Seed) -> None:
    """Orphan ``client_no`` values and the Hebrew names arrive intact (AL32UTF8 database)."""
    world = seed.world
    orphan_clients = set(world.expected.orphan_client_nos)
    assert orphan_clients, "the world plants orphan event keys"
    with connect_owner(seed.target) as conn, conn.cursor() as cur:
        placeholders = ", ".join(f":{i}" for i in range(1, len(orphan_clients) + 1))
        cur.execute(
            f"select count(*) from usage_events where client_no in ({placeholders})",
            sorted(orphan_clients),
        )
        row = cur.fetchone()
        cur.execute("select name_he, name_en from status_lookup where canonical = 'active'")
        names = cur.fetchone()

    assert row is not None and int(row[0]) > 0
    assert names == ("פעיל", "active")


# --------------------------------------------------------------------------------------
# Acceptance 2: interval partitioning - Oracle derives the partitions from the data
# --------------------------------------------------------------------------------------


def test_the_loader_creates_exactly_one_partition(seed: Seed) -> None:
    """The DDL pre-creates one partition; every monthly partition below is Oracle's work."""
    assert [partition.name for partition in seed.first.partitions_before_insert] == [
        INITIAL_PARTITION
    ]


def test_catalog_reports_monthly_interval_partitioning(seed: Seed) -> None:
    """``USER_PART_TABLES`` says RANGE + ``NUMTOYMINTERVAL(1, 'MONTH')``: the data drives it."""
    with connect_owner(seed.target) as conn, conn.cursor() as cur:
        cur.execute(
            "select partitioning_type, interval, autolist from user_part_tables "
            "where table_name = 'USAGE_EVENTS'"
        )
        row = cur.fetchone()
        catalog = partitions_of(cur, "usage_events")

    assert row is not None
    partitioning_type, interval, autolist = str(row[0]), str(row[1]), str(row[2])
    assert partitioning_type == "RANGE"
    assert "NUMTOYMINTERVAL(1, 'MONTH')" in interval.upper()
    # No fixed partition list: the catalog exposes the interval, the count comes from the derived
    # partitions themselves (``partition_count`` reports the sentinel 1048575 for interval tables).
    assert autolist == "NO"
    assert len(catalog) >= 2
    assert len(catalog) == len(seed.first.partitions)


def test_every_partition_past_the_first_one_is_generated_from_the_data(seed: Seed) -> None:
    """One hand-made partition, N generated ``SYS_P*`` partitions, one per month in the data."""
    world = seed.world
    partitions = seed.first.partitions
    generated = [partition for partition in partitions if partition.generated]
    hand_made = [partition for partition in partitions if not partition.generated]

    assert [partition.name for partition in hand_made] == [INITIAL_PARTITION]
    assert len(generated) >= 2, f"Oracle derived {len(generated)} partitions"
    assert all(p.name.startswith(GENERATED_PARTITION_PREFIX) for p in generated)

    months = generation_expected_months(world)
    assert len(generated) == len(months), (
        f"data covers {len(months)} months, Oracle derived {len(generated)} partitions"
    )

    boundaries = [partition.month for partition in generated]
    assert all(month is not None for month in boundaries), "every HIGH_VALUE is a timestamp"
    # A monthly RANGE partition for month M ends exactly at the first instant of M+1.
    assert boundaries == [_next_month(month) for month in months]

    # The assertion the task really wants: the live catalog matches what the loader reported.
    with connect_owner(seed.target) as conn, conn.cursor() as cur:
        live = partitions_of(cur, "usage_events")
    assert [partition.month for partition in live] == [partition.month for partition in partitions]

    # A second load re-derives the same boundaries. The SYS_P numbers are dictionary sequence
    # values and legitimately differ per run, so compare the bounds, not the names.
    assert seed.second.partitions_before_insert == seed.first.partitions_before_insert
    assert [partition.month for partition in seed.second.partitions] == [
        partition.month for partition in partitions
    ]


def test_usage_by_load_is_partitioned_on_ingestion_time(seed: Seed) -> None:
    """The event-time versus insertion-time choice must exist in Oracle too (mirrors D03)."""
    with connect_owner(seed.target) as conn, conn.cursor() as cur:
        cur.execute(
            "select column_name from user_part_key_columns "
            "where name = 'USAGE_BY_LOAD' and object_type = 'TABLE'"
        )
        columns = [str(row[0]) for row in cur.fetchall()]
        by_load = partitions_of(cur, "usage_by_load")
    assert columns == ["INSERTION_TIME"]
    assert len(by_load) >= len(seed.first.partitions) - 1


# --------------------------------------------------------------------------------------
# Acceptance 3: index on start_time only (as in D03)
# --------------------------------------------------------------------------------------


def test_index_on_start_time_only(seed: Seed) -> None:
    """Event time is indexed (LOCAL, so the plan can prune); ingestion time is not."""
    with connect_owner(seed.target) as conn, conn.cursor() as cur:
        cur.execute(
            "select index_name, partitioned from user_indexes where table_name = 'USAGE_EVENTS'"
        )
        indexes = {str(row[0]): str(row[1]) for row in cur.fetchall()}
        cur.execute(
            "select index_name, column_name from user_ind_columns "
            "where table_name = 'USAGE_EVENTS' order by index_name, column_position"
        )
        columns: dict[str, list[str]] = {}
        for row in cur.fetchall():
            columns.setdefault(str(row[0]), []).append(str(row[1]))

    assert set(indexes) == {"USAGE_EVENTS_PK", "USAGE_EVENTS_START_TIME_IDX"}
    assert indexes["USAGE_EVENTS_START_TIME_IDX"] == "YES", "a global index would hide pruning"
    assert columns["USAGE_EVENTS_START_TIME_IDX"] == ["START_TIME"]
    assert not any("INSERTION_TIME" in names for names in columns.values())


# --------------------------------------------------------------------------------------
# Acceptance 4: the read-only account cannot write, and reads what the planner needs
# --------------------------------------------------------------------------------------

FORBIDDEN_STATEMENTS: tuple[tuple[str, str], ...] = (
    (
        "insert",
        "insert into MDQM_BILLING.accounts (client_no, state, region, opened_date) "
        "values ('RO00000001', 'Active', 'north', date '2025-01-01')",
    ),
    ("update", "update MDQM_BILLING.accounts set state = 'Hold'"),
    ("delete", "delete from MDQM_BILLING.accounts_hist"),
    ("truncate", "truncate table MDQM_BILLING.accounts"),
    ("create_table", "create table ro_probe (i number)"),
    ("drop", "drop table MDQM_BILLING.accounts_hist"),
)


@pytest.mark.parametrize(
    ("label", "statement"),
    FORBIDDEN_STATEMENTS,
    ids=[statement[0] for statement in FORBIDDEN_STATEMENTS],
)
def test_read_only_account_cannot_write(seed: Seed, label: str, statement: str) -> None:
    """Every write dies with Oracle's own error; the message is recorded, not assumed."""
    with connect_read_only(seed.target) as conn, conn.cursor() as cur:
        assert conn.username == seed.target.ro_user.upper()
        with pytest.raises(oracledb.DatabaseError) as excinfo:
            cur.execute(statement)
    error = excinfo.value.args[0]
    code = int(getattr(error, "code", 0))
    message = str(getattr(error, "message", excinfo.value)).strip()
    assert message, f"{label}: rejected with an empty message"
    assert code in MISSING_PRIVILEGE, f"{label}: unexpected error ORA-{code:05d}: {message}"
    assert "privilege" in message.lower(), (
        f"{label}: the rejection does not mention a privilege: {message}"
    )


def test_read_only_transaction_is_the_second_layer(seed: Seed) -> None:
    """Even the owner's connection cannot write inside ``SET TRANSACTION READ ONLY`` (ORA-01456)."""
    with connect_owner(seed.target) as conn, conn.cursor() as cur:
        cur.execute("set transaction read only")
        with pytest.raises(oracledb.DatabaseError) as excinfo:
            cur.execute(
                "insert into accounts (client_no, state, region, opened_date) "
                "values ('RO00000002', 'Active', 'north', date '2025-01-01')"
            )
        conn.rollback()
    error = excinfo.value.args[0]
    assert int(getattr(error, "code", 0)) == READ_ONLY_TRANSACTION, str(
        getattr(error, "message", excinfo.value)
    )


def test_read_only_account_holds_only_select(seed: Seed) -> None:
    """``CREATE SESSION`` plus ``SELECT`` on the loaded objects - no role, no write privilege."""
    with connect_read_only(seed.target) as conn, conn.cursor() as cur:
        cur.execute("select privilege from user_sys_privs")
        sys_privs = {str(row[0]) for row in cur.fetchall()}
        cur.execute("select granted_role from user_role_privs")
        roles = {str(row[0]) for row in cur.fetchall()}
        cur.execute("select distinct privilege from user_tab_privs")
        tab_privs = {str(row[0]) for row in cur.fetchall()}

    assert sys_privs == {"CREATE SESSION"}
    assert roles == set(), f"the image ships DB_DEVELOPER_ROLE, the loader must revoke it: {roles}"
    write_privileges = {
        privilege
        for privilege in tab_privs
        if privilege.upper().split()[0] in {"INSERT", "UPDATE", "DELETE", "ALTER", "CREATE"}
    }
    assert write_privileges == set(), f"write privileges granted: {write_privileges}"
    assert "SELECT" in tab_privs


def test_read_only_account_reads_the_dictionary_and_v_sql(seed: Seed) -> None:
    """The planner's inputs are readable by the account the adapter will use."""
    with connect_read_only(seed.target) as conn, conn.cursor() as cur:
        cur.execute("select count(*) from all_part_tables where table_name = 'USAGE_EVENTS'")
        tables = cur.fetchone()
        cur.execute("select column_name from all_part_key_columns where name = 'USAGE_EVENTS'")
        key_columns = [str(row[0]) for row in cur.fetchall()]
        cur.execute("select count(*) from v$sql")
        statistics = cur.fetchone()
        cur.execute("select count(*) from MDQM_BILLING.usage_events")
        events = cur.fetchone()
        cur.execute("select count(*) from MDQM_BILLING.accounts")
        accounts = cur.fetchone()

    assert tables is not None and int(tables[0]) == 1
    assert key_columns == ["START_TIME"]
    assert statistics is not None and int(statistics[0]) > 0
    assert events is not None and int(events[0]) == seed.world.config.event_count
    assert accounts is not None and int(accounts[0]) == len(seed.world.billing.accounts)


# --------------------------------------------------------------------------------------
# Acceptance 5: EXPLAIN PLAN exposes PSTART/PSTOP, a literal filter gives a numeric range
# --------------------------------------------------------------------------------------


def _plan_partition_range(target: OracleTarget, statement: str, binds: Sequence[Any]) -> list[Any]:
    """Rows of the partition operation in ``PLAN_TABLE`` for one ``EXPLAIN PLAN``."""
    with connect_owner(target) as conn, conn.cursor() as cur:
        cur.execute("delete from plan_table")
        cur.execute(f"explain plan for {statement}", list(binds))
        cur.execute(
            "select operation, partition_start, partition_stop from plan_table "
            "where operation like 'PARTITION RANGE%' order by id"
        )
        rows = cur.fetchall()
    return [tuple(row) for row in rows]


WINDOW_START: Final = datetime(2025, 11, 1)
"""First instant of a month inside the world window (the world runs May 2025 .. Jan 2026)."""

LIMIT_QUERY: Final = (
    "select count(*) from usage_events where start_time >= timestamp '2025-11-01 00:00:00'"
)
BIND_QUERY: Final = "select count(*) from usage_events where start_time >= :1"


def test_literal_date_filter_prunes_to_a_numeric_partition_range(seed: Seed) -> None:
    """A literal filter gives a numeric ``PSTART``: the optimizer pruned at parse time."""
    del seed
    rows = _plan_partition_range(_target(), LIMIT_QUERY, [])
    assert rows, "EXPLAIN PLAN shows no partition operation: the table is not pruned"
    for operation, start, stop in rows:
        assert "PARTITION RANGE" in str(operation)
        assert str(start).strip().isdigit(), f"PSTART is {start!r}, not a partition number"
        assert str(stop).strip().isdigit(), f"PSTOP is {stop!r}, not a partition number"


def test_bound_parameter_filter_shows_key_not_a_numeric_range(seed: Seed) -> None:
    """The same predicate with a bind cannot be pruned at parse time: Oracle reports ``KEY``."""
    del seed
    with connect_owner(_target()) as conn, conn.cursor() as cur:
        cur.execute("delete from plan_table")
        cur.execute(f"explain plan for {BIND_QUERY}", [WINDOW_START])
        cur.execute(
            "select partition_start, partition_stop from plan_table "
            "where operation like 'PARTITION RANGE%'"
        )
        rows = cur.fetchall()
    assert rows, "EXPLAIN PLAN shows no partition operation for the bound predicate"
    assert any(str(row[0]).strip().upper() == "KEY" for row in rows), (
        f"expected PSTART = KEY for a bind predicate, got {rows}"
    )


def _target() -> OracleTarget:
    """The configured target, for the two pruning tests (no fixture needed: no data written)."""
    return target_from_environ()


def test_usage_events_covers_the_world_window(seed: Seed) -> None:
    """Sanity check of the loaded window: min/max event time are the world's, to the second."""
    world = seed.world
    starts = [to_stored_timestamp(event.start_time) for event in world.billing.usage_events]
    with connect_owner(seed.target) as conn, conn.cursor() as cur:
        cur.execute("select min(start_time), max(start_time) from usage_events")
        row = cur.fetchone()
    assert row is not None
    assert row[0] == min(starts)
    assert row[1] == max(starts)


def test_grant_read_only_is_idempotent(seed: Seed) -> None:
    """Re-running grants (the second load did) must not fail or duplicate privileges."""
    with connect_admin(seed.target) as conn, conn.cursor() as cur:
        cur.execute(
            "select count(*) from dba_tab_privs "
            "where grantee = :1 and owner = :2 and privilege = 'SELECT'",
            [seed.target.ro_user.upper(), seed.target.owner_user.upper()],
        )
        row = cur.fetchone()
    assert row is not None and int(row[0]) == len(ALL_TABLES)
