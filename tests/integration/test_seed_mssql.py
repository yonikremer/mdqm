"""D06 acceptance: SQL Server erp dataset, partition function/scheme, read-only login, Query Store.

Needs the ``mssql`` compose profile (``python scripts/dev.py up mssql``, ~2 GB and a slow first
start)::

    python -m pytest tests/integration/test_seed_mssql.py --require-services -q

What the assertions are anchored to:

* the **world** (``WorldConfig.small()``) for row counts, the planted duplicate ``cust_no`` keys
  and the ``limit_k`` conversion — never to ``now()``, because the world's window ends at
  :data:`mdqm.testing.world.ANCHOR` while the machine clock keeps moving;
* **SQL Server's own catalog** (``sys.partition_range_values``, ``sys.partition_functions``,
  ``sys.partition_schemes``, ``sys.partitions``, ``sys.database_query_store_options``) for the
  partitioning and Query Store claims, so the test cannot pass on the loader's opinion of itself;
* **SQL Server's own error numbers** for the read-only claims: ``229`` for a denied DML/DDL
  permission on an object, ``262`` for ``CREATE ...`` denied in the database. The exact message
  the engine produced is printed, and asserted on its text too.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import pytest

from mdqm.testing.loaders.sqlserver import (
    PARTITION_FUNCTION,
    PARTITION_SCHEME,
    LoadReport,
    MssqlTarget,
    connect,
    duplicate_cust_nos,
    function_fanout,
    load,
    partition_boundaries,
    partition_tuples,
    query_store_state,
    table_partition_rows,
    target_from_environ,
)
from mdqm.testing.world import ERP_SCALE, World, WorldConfig, build_world

pytestmark = pytest.mark.slow

DENIED: Final = 229
"""``The INSERT/UPDATE/DELETE permission was denied on the object ...``"""

CREATE_DENIED: Final = 262
"""``CREATE TABLE permission denied in database 'erp'.``"""

STATUS_CODES: Final[dict[str, int]] = {"active": 1, "churned": 2, "suspended": 3}


@dataclass(frozen=True, slots=True)
class Seed:
    """Two loading runs and what SQL Server says about the result of each."""

    target: MssqlTarget
    world: World
    first: LoadReport
    second: LoadReport


def _unavailable(request: pytest.FixtureRequest, reason: str) -> None:
    """Skip, or fail when ``--require-services`` was given."""
    if request.config.getoption("--require-services"):
        pytest.fail(f"required service unavailable: {reason}")
    pytest.skip(f"service unavailable: {reason}")


def _counts(target: MssqlTarget) -> dict[str, int]:
    """``COUNT_BIG(*)`` per table as the owner: proves the rows landed, not that we tried."""
    with connect(target) as conn, conn.cursor() as cur:
        out: dict[str, int] = {}
        for table in ("customers", "usage_events"):
            cur.execute(f"SELECT COUNT_BIG(*) FROM dbo.{table}")  # noqa: S608 - module constant
            row = cur.fetchone()
            out[f"erp.{table}"] = 0 if row is None else int(row[0])
        return out


@pytest.fixture(scope="module")
def seed(request: pytest.FixtureRequest) -> Iterator[Seed]:
    """Load the small world twice into SQL Server (the second run is the idempotency check)."""
    try:
        target = target_from_environ()
    except Exception as exc:  # noqa: BLE001 - any settings failure means "not configured here"
        _unavailable(request, f"SQL Server connection settings: {exc}")
        raise AssertionError("unreachable") from exc

    try:
        # master, so the probe answers "is the server there" and not "does erp exist yet".
        with connect(target, database="master") as conn:
            conn.cursor().execute("SELECT 1")
    except Exception as exc:  # noqa: BLE001 - driver missing, container down, wrong password
        _unavailable(request, f"SQL Server at {target.server}: {type(exc).__name__}: {exc}")
        raise AssertionError("unreachable") from exc

    world = build_world(WorldConfig.small())
    first = load(target, world=world)
    second = load(target, world=world)
    yield Seed(target=target, world=world, first=first, second=second)


# --------------------------------------------------------------------------------------
# Rows: the erp storage form, duplicates included
# --------------------------------------------------------------------------------------


def test_row_counts_match_the_world(seed: Seed) -> None:
    """Every world row arrived, including the planted duplicate, and nothing else did."""
    expected_customers = seed.world.config.customers + len(seed.world.expected.duplicate_cust_nos)
    assert len(seed.world.erp.customers) == expected_customers
    assert _counts(seed.target) == {
        "erp.customers": expected_customers,
        "erp.usage_events": seed.world.config.event_count,
    }


def test_planted_duplicate_cust_no_rows(seed: Seed) -> None:
    """The duplicate keys are duplicates **in the database**, not only in the generator."""
    wanted = tuple((cust_no, 2) for cust_no in seed.world.expected.duplicate_cust_nos)
    with connect(seed.target) as conn, conn.cursor() as cur:
        assert duplicate_cust_nos(cur) == wanted
    assert seed.first.duplicates == wanted
    assert seed.first.duplicate_rows == len(wanted)


def test_limit_k_is_the_rounded_limit_over_ten(seed: Seed) -> None:
    """``limit_k`` is the canonical limit / 10, with only the planted exceptions off by 500."""
    exceptions = {row.cust_id: row.erp_limit_k for row in seed.world.expected.erp_exceptions}
    canonical = {row.cust_id: row.canonical_limit for row in seed.world.expected.customers}
    with connect(seed.target) as conn, conn.cursor() as cur:
        cur.execute("SELECT cust_no, status_code, limit_k FROM dbo.customers ORDER BY cust_no")
        rows = cur.fetchall()

    assert len(rows) == len(seed.world.erp.customers)
    for cust_no, _status_code, limit_k in rows:
        wanted = exceptions.get(cust_no, canonical[cust_no] // ERP_SCALE)
        assert limit_k == wanted, f"cust_no {cust_no}: limit_k {limit_k} != {wanted}"


def test_status_codes_are_the_erp_vocabulary(seed: Seed) -> None:
    """``1``/``2``/``3`` for active/churned/suspended, one code per customer."""
    canonical = {row.cust_id: row.canonical_status for row in seed.world.expected.customers}
    with connect(seed.target) as conn, conn.cursor() as cur:
        cur.execute("SELECT cust_no, status_code FROM dbo.customers")
        for cust_no, status_code in cur.fetchall():
            assert status_code == STATUS_CODES[canonical[cust_no]], f"cust_no {cust_no}"


def test_reload_is_idempotent(seed: Seed) -> None:
    """A second run writes the same rows, the same tuples and the same partition fill."""
    assert seed.first.counts == seed.second.counts
    assert seed.first.boundaries == seed.second.boundaries
    assert seed.first.partition_rows == seed.second.partition_rows
    assert _counts(seed.target) == seed.first.counts


# --------------------------------------------------------------------------------------
# Partitioning: SQL Server's catalog, not the loader's word
# --------------------------------------------------------------------------------------


def test_function_tuples_cover_the_data_window(seed: Seed) -> None:
    """The tuples are the monthly boundaries the loader computed, and SQL Server stored them."""
    wanted = partition_tuples(seed.world.config.anchor)
    assert seed.first.boundaries == wanted
    assert all(isinstance(value, datetime) for value in wanted)
    assert seed.first.fanout == len(wanted) + 1
    with connect(seed.target) as conn, conn.cursor() as cur:
        assert partition_boundaries(cur) == wanted
        assert function_fanout(cur) == len(wanted) + 1


def test_scheme_maps_every_partition(seed: Seed) -> None:
    """A scheme exists, is built on the function, and every destination is the PRIMARY filegroup."""
    with connect(seed.target) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT pf.name, pf.fanout, ds.name
            FROM sys.partition_schemes AS ps
            JOIN sys.partition_functions AS pf ON pf.function_id = ps.function_id
            JOIN sys.destination_data_spaces AS dds ON dds.partition_scheme_id = ps.data_space_id
            JOIN sys.data_spaces AS ds ON ds.data_space_id = dds.data_space_id
            WHERE ps.name = ?
            """,
            (PARTITION_SCHEME,),
        )
        rows = cur.fetchall()

    # "ALL TO" reserves one extra destination for the next split, so rows == fanout + 1.
    assert {row[0] for row in rows} == {PARTITION_FUNCTION}
    assert {row[1] for row in rows} == {seed.first.fanout}
    assert {row[2] for row in rows} == {"PRIMARY"}
    assert len(rows) == seed.first.fanout + 1


def test_rows_are_divided_across_the_monthly_partitions(seed: Seed) -> None:
    """``sys.partitions`` shows the rows in the window's months and none in the catch-alls."""
    with connect(seed.target) as conn, conn.cursor() as cur:
        rows = table_partition_rows(cur)
    assert rows == seed.first.partition_rows
    assert sum(count for _, count in rows) == seed.world.config.event_count
    assert len(rows) == len(seed.first.boundaries) + 1

    by_number = dict(rows)
    tuples = seed.first.boundaries
    # RANGE RIGHT: partition k+1 holds [tuple k, tuple k+1), so month m sits at index(m) + 2.
    window = seed.world.expected.events_per_month
    assert window, "the world must plant events, or this proves nothing"
    for month in window:
        start = datetime.fromisoformat(f"{month}-01")
        number = tuples.index(start) + 2
        assert by_number[number] > 0, f"month {month} is empty in partition {number}"
    # Both catch-alls stay empty: no row fell outside the created tuple range.
    assert by_number[1] == 0
    assert by_number[len(tuples) + 1] == 0


# --------------------------------------------------------------------------------------
# Read-only login: SQL Server's own refusal
# --------------------------------------------------------------------------------------


def test_read_only_login_can_read(seed: Seed) -> None:
    with connect(seed.target, read_only=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT_BIG(*) FROM dbo.usage_events")
        row = cur.fetchone()
    assert row is not None and row[0] == seed.world.config.event_count


@pytest.mark.parametrize(
    ("statement", "code", "needle"),
    [
        (
            "INSERT INTO dbo.customers (cust_no, status_code, limit_k, name) VALUES (1, 1, 1, 'x')",
            DENIED,
            "INSERT permission was denied on the object 'customers'",
        ),
        ("UPDATE dbo.customers SET limit_k = 1 WHERE cust_no = 1", DENIED, "UPDATE permission"),
        ("DELETE FROM dbo.usage_events", DENIED, "DELETE permission"),
        (
            "CREATE TABLE dbo.denied (a INT)",
            CREATE_DENIED,
            "CREATE TABLE permission denied in database 'erp'",
        ),
    ],
)
def test_read_only_login_cannot_write(seed: Seed, statement: str, code: int, needle: str) -> None:
    """The write fails with SQL Server's own error, and the message names the object."""
    with connect(seed.target, read_only=True) as conn, conn.cursor() as cur:
        with pytest.raises(Exception) as caught:
            cur.execute(statement)
        error: Any = caught.value
        with contextlib.suppress(Exception):
            # A failed statement may already have aborted the transaction; rollback is best effort.
            conn.rollback()

    # mssql_python exposes the native error number and SQLSTATE the engine sent.
    assert getattr(error, "native_error", None) == code, f"unexpected error: {error}"
    assert getattr(error, "sqlstate", None) in {"42000", "S0001"}, f"unexpected error: {error}"
    assert needle in str(error), f"unexpected message: {error}"


# --------------------------------------------------------------------------------------
# Query Store and catalog visibility for the read-only login
# --------------------------------------------------------------------------------------


def test_query_store_is_enabled(seed: Seed) -> None:
    """Query Store is READ_WRITE (A08 reads plans from it), and the login may read its views."""
    with connect(seed.target) as conn, conn.cursor() as cur:
        assert query_store_state(cur) == "READ_WRITE"
    assert seed.first.query_store == "READ_WRITE"

    with connect(seed.target, read_only=True) as conn, conn.cursor() as cur:
        assert query_store_state(cur) == "READ_WRITE"
        # VIEW DATABASE STATE is what makes these readable for a db_datareader login.
        cur.execute("SELECT TOP (1) query_id FROM sys.query_store_query ORDER BY query_id DESC")
        cur.fetchone()
        cur.execute("SELECT TOP (1) query_sql_text FROM sys.query_store_query_text")
        cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM sys.query_store_runtime_stats")
        cur.fetchone()


def test_read_only_login_sees_the_partition_catalog(seed: Seed) -> None:
    """The login the planner runs as can see the partition shape it has to prune on."""
    expected = len(partition_tuples(seed.world.config.anchor))
    with connect(seed.target, read_only=True) as conn, conn.cursor() as cur:
        assert len(partition_boundaries(cur)) == expected
        assert function_fanout(cur) == expected + 1
        assert len(table_partition_rows(cur)) == expected + 1


def test_permissions_are_reader_only(seed: Seed) -> None:
    """``db_datareader`` and nothing that can write: no db_owner, db_datawriter or ddl admin."""
    with connect(seed.target, read_only=True) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.name
            FROM sys.database_role_members AS m
            JOIN sys.database_principals AS r ON r.principal_id = m.role_principal_id
            JOIN sys.database_principals AS u ON u.principal_id = m.member_principal_id
            WHERE u.name = ?
            """,
            (seed.target.ro_user,),
        )
        roles: Sequence[str] = [row[0] for row in cur.fetchall()]
        cur.execute("SELECT IS_SRVROLEMEMBER('sysadmin')")
        row = cur.fetchone()
    assert set(roles) == {"db_datareader"}, roles
    assert row is not None and row[0] == 0
