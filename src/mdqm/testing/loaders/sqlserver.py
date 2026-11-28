"""Create the erp schema in SQL Server and load the synthetic world into it (task D06).

Same rows as :mod:`mdqm.testing.loaders.postgres` (task D03) and
:mod:`mdqm.testing.loaders.oracle` (task D05), fourth engine. The point of this one is the
**partition function / partition scheme** mechanism SQL Server uses instead of a
``PARTITION BY`` clause, plus Query Store, which A08 reads plans from::

    dbo.customers      cust_no INT, status_code INT, limit_k INT, name NVARCHAR(200)
                       deliberately **no** primary key: the world plants duplicate cust_no rows
    dbo.usage_events   the same rows as ``billing.usage_events``, RANGE RIGHT monthly on
                       ``start_time`` through ``pf_usage_events_month`` / ``ps_usage_events_month``

Why the schema looks like this:

* ``limit_k`` is the canonical rounded limit divided by 10 (``ERP_SCALE``), which is the
  conversion the planner must rediscover. ``status_code`` is ``1``/``2``/``3``.
* Partition **tuples** are the first instant of each month, ``anchor - 14`` .. ``anchor + 2``.
  RANGE RIGHT makes a tuple the *lower* bound of the partition to its right, so N tuples give
  N+1 partitions: a catch-all on the left and one on the right, and the world's window
  (``anchor - months`` .. ``anchor``) lands in the exact monthly partitions in between. Rows
  that arrive with an ingestion-derived bound past the window cannot spill past the last tuple
  and create a partition nobody counted.
* ``ALL TO ([PRIMARY])``: the scheme maps every partition to the single PRIMARY filegroup. A
  partition *function* can be materialised through any scheme the test box happens to have, and
  ``sys.partition_functions``/``sys.partition_schemes`` then describe the same shape as
  production without a second data file.
* ``start_time``/``insertion_time`` are ``DATETIME2(6)`` holding **naive UTC**: SQL Server has no
  timestamp-with-time-zone type, and DESIGN section 5 keeps one UTC ``exec_ts`` per execution, so
  the loader stores naive UTC and the adapter attaches UTC again on read. Same choice as D05.
* Query Store is enabled because A08 reads plan XML from it; ``VIEW DATABASE STATE`` is granted to
  the read-only login, which is what makes ``sys.query_store_*`` readable for it.

The loader is **idempotent by drop/recreate**: tables are dropped (children first), then the
partition scheme and function, then everything is created and filled again. Re-running produces
the same rows, byte for byte, because the world is seeded and has no clock. Seeding uses the
``sa`` connection from ``.env`` (``MDQM_MSSQL_PASSWORD``); ``mdqm_ro`` — the same login name every
adapter uses for the shared read-only account — gets ``db_datareader`` and nothing else.

Run from the repository root (the ``mssql`` compose profile must be up)::

    python -m mdqm.testing.loaders.sqlserver --small

Credentials and the port come from ``.env`` (``MDQM_MSSQL_PASSWORD``, ``MDQM_RO_PASSWORD``,
``MDQM_MSSQL_PORT``) or the same environment variables; nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from mdqm.testing.world import ANCHOR, World, WorldConfig, build_world

DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 51433
"""Host port published by the ``mssql`` compose profile (see .env.example)."""

DEFAULT_DATABASE: Final = "erp"
"""The database this loader owns. One database, one dataset: ``erp``."""

ADMIN_USER: Final = "sa"
"""(``MDQM_MSSQL_PASSWORD`` is this login's password.)"""

READ_ONLY_LOGIN: Final = "mdqm_ro"
"""Server login and database user the adapters read with: ``db_datareader`` only."""

PARTITION_FUNCTION: Final = "pf_usage_events_month"
PARTITION_SCHEME: Final = "ps_usage_events_month"

PARTITION_MONTHS_BACK: Final = 14
PARTITION_MONTHS_FORWARD: Final = 2
"""Monthly tuples created before/after the anchor month, inclusive of the anchor."""

INSERT_BATCH: Final = 5000
"""Rows per ``executemany`` call: rare round trips, small enough to commit in slices."""

DOTENV_NAME: Final = ".env"

TABLES: Final = ("customers", "usage_events")
"""Tables dropped and recreated on every run."""

USAGE_COLUMNS: Final = (
    "event_id",
    "client_no",
    "start_time",
    "insertion_time",
    "units",
    "corrects_event_id",
)

CUSTOMER_COLUMNS: Final = ("cust_no", "status_code", "limit_k", "name")


class LoaderError(RuntimeError):
    """The loader could not do its job; the message names what and where."""


# --------------------------------------------------------------------------------------
# Target and credentials
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MssqlTarget:
    """Where the database lives and who may seed it, versus who may only read."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    database: str = DEFAULT_DATABASE
    admin_user: str = ADMIN_USER
    admin_password: str = ""
    ro_user: str = READ_ONLY_LOGIN
    ro_password: str = ""

    @property
    def server(self) -> str:
        """``host,port`` as the .NET-style connection string spells it."""
        return f"{self.host},{self.port}"

    def connection_string(self, *, read_only: bool = False, database: str | None = None) -> str:
        """ODBC connection string, as the owner or as the read-only login.

        ``TrustServerCertificate=yes``: the container uses a self-signed certificate, and the
        alternative (turning encryption off) would make the dev stack differ from production.
        """
        user = self.ro_user if read_only else self.admin_user
        password = self.ro_password if read_only else self.admin_password
        name = self.database if database is None else database
        return (
            f"Server={self.server};Database={name};UID={user};PWD={password};"
            "Encrypt=yes;TrustServerCertificate=yes;APP=mdqm-loader"
        )


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines of a dotenv file. Missing file yields an empty mapping."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def find_dotenv(start: Path | None = None) -> Path | None:
    """Nearest credentials file walking up from ``start`` (the working directory by default)."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        path = candidate / DOTENV_NAME
        if path.is_file():
            return path
    return None


def target_from_environ(
    *, environ: dict[str, str] | None = None, dotenv_path: Path | None = None
) -> MssqlTarget:
    """Build a :class:`MssqlTarget` from the environment, falling back to the dotenv file."""
    path = dotenv_path if dotenv_path is not None else find_dotenv()
    values = read_dotenv(path) if path is not None else {}
    env: dict[str, str] = dict(os.environ) if environ is None else environ
    values.update({key: value for key, value in env.items() if key.startswith("MDQM_")})

    password = values.get("MDQM_MSSQL_PASSWORD", "")
    if not password:
        where = path if path is not None else "no dotenv file found"
        raise LoaderError(
            f"MDQM_MSSQL_PASSWORD is not set ({where}): copy .env.example to .env "
            "(`python scripts/dev.py env`) or export the variable"
        )
    return MssqlTarget(
        host=values.get("MDQM_MSSQL_HOST", DEFAULT_HOST),
        port=int(values.get("MDQM_MSSQL_PORT", DEFAULT_PORT)),
        database=values.get("MDQM_MSSQL_DATABASE", DEFAULT_DATABASE),
        admin_password=password,
        ro_password=values.get("MDQM_RO_PASSWORD", ""),
    )


def connect(
    target: MssqlTarget,
    *,
    read_only: bool = False,
    autocommit: bool = False,
    database: str | None = None,
) -> Any:
    """Open a DB-API connection as the owner or the read-only login.

    ``database`` overrides the target's database: server-level statements (create the database,
    create the login) must run in ``master``, and the driver reports "cannot open database" as a
    login failure, which is a confusing message for a database that does not exist yet.

    ``mssql_python`` (the ``mssql`` extra) is imported **here**, not at module level: the unit
    test suite must run on a machine that never installed a SQL Server driver.
    """
    try:
        import mssql_python
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the extra
        raise LoaderError(
            "the SQL Server driver is not installed: `pip install -e .[mssql]` (mssql-python)"
        ) from exc
    if read_only and not target.ro_password:
        raise LoaderError("MDQM_RO_PASSWORD is empty: the read-only login needs a password")
    return mssql_python.connect(
        target.connection_string(read_only=read_only, database=database),
        autocommit=autocommit,
        timeout=30,
    )


# --------------------------------------------------------------------------------------
# Partition tuples
# --------------------------------------------------------------------------------------


def _month_start(value: datetime) -> datetime:
    """First instant of the UTC month that contains ``value``, naive (SQL Server has no tz)."""
    aware = value.astimezone(UTC)
    return datetime(aware.year, aware.month, 1)


def _shift_months(value: datetime, months: int) -> datetime:
    """Shift a naive first-of-month value by whole months."""
    total = value.year * 12 + (value.month - 1) + months
    return datetime(total // 12, total % 12 + 1, 1)


def partition_tuples(
    anchor: datetime = ANCHOR,
    *,
    back: int = PARTITION_MONTHS_BACK,
    forward: int = PARTITION_MONTHS_FORWARD,
) -> tuple[datetime, ...]:
    """The RANGE RIGHT tuples: the first instant of every month from ``anchor - back`` to
    ``anchor + forward``, inclusive.

    Inclusive of the anchor because a RANGE RIGHT tuple is the **lower** bound of the partition
    to its right: the data ending exactly at the anchor month start must still fall inside a
    named partition instead of the right-hand catch-all.
    """
    first = _shift_months(_month_start(anchor), -back)
    last = _shift_months(_month_start(anchor), forward)
    out: list[datetime] = []
    cursor = first
    while cursor <= last:
        out.append(cursor)
        cursor = _shift_months(cursor, 1)
    return tuple(out)


def partition_count(tuples: Sequence[datetime]) -> int:
    """Partitions a RANGE RIGHT function with ``tuples`` has: one more than the tuple count."""
    return len(tuples) + 1


def to_stored_timestamp(value: datetime) -> datetime:
    """Aware UTC timestamp as the naive UTC value SQL Server stores."""
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware; store naive UTC only at the edge")
    return value.astimezone(UTC).replace(tzinfo=None)


def _tuple_literal(value: datetime) -> str:
    """``DATETIME2(6)`` literal, ISO 8601 with a ``T`` so no language setting can reorder it."""
    return f"CONVERT(datetime2(6), '{value.isoformat()}', 126)"


# --------------------------------------------------------------------------------------
# DDL
# --------------------------------------------------------------------------------------

_USAGE_DDL: Final = """
    event_id           BIGINT        NOT NULL,
    client_no          VARCHAR(10)   NOT NULL,
    start_time         DATETIME2(6)  NOT NULL,
    insertion_time     DATETIME2(6)  NOT NULL,
    units              INT           NOT NULL,
    corrects_event_id  BIGINT
"""


_DROP_SCHEME_SQL: Final = (
    "IF EXISTS (SELECT 1 FROM sys.partition_schemes WHERE name = ?) DROP PARTITION SCHEME ["  # noqa: S608 - constant name, value is bound
    + PARTITION_SCHEME
    + "]"
)

_DROP_FUNCTION_SQL: Final = (
    "IF EXISTS (SELECT 1 FROM sys.partition_functions WHERE name = ?) "  # noqa: S608
    "DROP PARTITION FUNCTION [" + PARTITION_FUNCTION + "]"
)


def drop_schema(cur: Any) -> None:
    """Drop the tables, then the partition scheme and function (in that order).

    A partition function cannot be dropped while a scheme uses it, and a scheme cannot be
    dropped while a table is built on it, so the order is load-bearing, not cosmetic.
    """
    # Every identifier below is a module constant or the loader's own target config, and SQL Server
    # cannot bind an identifier; no value is interpolated.
    for table in TABLES:
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
        cur.execute(f"IF OBJECT_ID('dbo.{table}', 'U') IS NOT NULL DROP TABLE dbo.{table}")
    # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
    cur.execute(_DROP_SCHEME_SQL, (PARTITION_SCHEME,))
    # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
    cur.execute(_DROP_FUNCTION_SQL, (PARTITION_FUNCTION,))


def create_partitioning(cur: Any, tuples: Sequence[datetime]) -> None:
    """Create the monthly RANGE RIGHT function and a scheme that puts every part on PRIMARY."""
    values = ", ".join(_tuple_literal(value) for value in tuples)
    cur.execute(
        f"CREATE PARTITION FUNCTION [{PARTITION_FUNCTION}] (datetime2(6)) "
        f"AS RANGE RIGHT FOR VALUES ({values})"
    )
    cur.execute(
        f"CREATE PARTITION SCHEME [{PARTITION_SCHEME}] AS PARTITION [{PARTITION_FUNCTION}] "
        "ALL TO ([PRIMARY])"
    )


def create_tables(cur: Any) -> None:
    """``dbo.customers`` (no key: duplicates are data here) and the partitioned usage table."""
    cur.execute(
        """
        CREATE TABLE dbo.customers (
            cust_no      INT           NOT NULL,
            status_code  INT           NOT NULL,
            limit_k      INT           NOT NULL,
            name         NVARCHAR(200) NOT NULL
        )
        """
    )
    # The clustered index must contain the partitioning column, and putting it on the scheme is
    # what makes the index partitioned rather than a single heap over every month.
    # The only interpolations are module constants; SQL Server cannot bind an identifier.
    # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
    cur.execute(
        f"""
        CREATE TABLE dbo.usage_events (
            {_USAGE_DDL},
            CONSTRAINT usage_events_pk PRIMARY KEY CLUSTERED (event_id, start_time)
                ON [{PARTITION_SCHEME}] (start_time)
        ) ON [{PARTITION_SCHEME}] (start_time)
        """
    )


# --------------------------------------------------------------------------------------
# Catalog reads (what SQL Server itself says, never what the loader believes)
# --------------------------------------------------------------------------------------


def partition_boundaries(cur: Any, function_name: str = PARTITION_FUNCTION) -> tuple[datetime, ...]:
    """The tuples of ``function_name``, read from ``sys.partition_range_values``."""
    cur.execute(
        """
        SELECT CAST(prv.value AS datetime2(6)) AS boundary
        FROM sys.partition_range_values AS prv
        JOIN sys.partition_functions AS pf ON pf.function_id = prv.function_id
        WHERE pf.name = ?
        ORDER BY prv.boundary_id
        """,
        (function_name,),
    )
    return tuple(row[0] for row in cur.fetchall())


def function_fanout(cur: Any, function_name: str = PARTITION_FUNCTION) -> int:
    """Partitions the function declares, from ``sys.partition_functions.fanout``.

    Not the count of ``sys.destination_data_spaces``: ``ALL TO ([PRIMARY])`` reserves one extra
    destination for the *next* split, so that view reports one more row than there are partitions.
    """
    cur.execute(
        "SELECT fanout FROM sys.partition_functions WHERE name = ?",
        (function_name,),
    )
    row = cur.fetchone()
    return 0 if row is None else int(row[0])


def table_partition_rows(cur: Any, table: str = "usage_events") -> tuple[tuple[int, int], ...]:
    """``(partition_number, rows)`` from ``sys.partitions``: proof the rows are divided."""
    cur.execute(
        """
        SELECT p.partition_number, SUM(p.rows) AS row_count
        FROM sys.partitions AS p
        JOIN sys.tables AS t ON t.object_id = p.object_id
        JOIN sys.indexes AS i ON i.object_id = p.object_id AND i.index_id = p.index_id
        WHERE t.name = ? AND i.index_id IN (0, 1)
        GROUP BY p.partition_number
        ORDER BY p.partition_number
        """,
        (table,),
    )
    return tuple((int(number), int(rows)) for number, rows in cur.fetchall())


def duplicate_cust_nos(cur: Any) -> tuple[tuple[int, int], ...]:
    """``(cust_no, occurrences)`` for every key that appears more than once, ascending."""
    cur.execute(
        """
        SELECT cust_no, COUNT(*) AS occurrences
        FROM dbo.customers
        GROUP BY cust_no
        HAVING COUNT(*) > 1
        ORDER BY cust_no
        """
    )
    return tuple((int(cust_no), int(occurrences)) for cust_no, occurrences in cur.fetchall())


def query_store_state(cur: Any) -> str:
    """``actual_state_desc`` from ``sys.database_query_store_options`` (``READ_WRITE`` when on)."""
    cur.execute("SELECT actual_state_desc FROM sys.database_query_store_options")
    row = cur.fetchone()
    return "" if row is None else str(row[0])


# --------------------------------------------------------------------------------------
# Database, logins, grants
# --------------------------------------------------------------------------------------


def ensure_database(target: MssqlTarget) -> None:
    """Create the database if it is missing and turn Query Store on.

    ``CREATE DATABASE`` and ``ALTER DATABASE ... SET QUERY_STORE`` cannot run inside a
    transaction, hence the autocommit connection to ``master``.
    """
    with connect(target, autocommit=True, database="master") as conn, conn.cursor() as cur:
        cur.execute("SELECT DB_ID(?)", (target.database,))
        row = cur.fetchone()
        if row is None or row[0] is None:
            # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
            cur.execute(f"CREATE DATABASE [{target.database}]")
        # READ_WRITE, not READ_ONLY: A08 reads plans, and a read-only store never fills with them.
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
        cur.execute(
            f"""
            ALTER DATABASE [{target.database}] SET QUERY_STORE = ON (
                OPERATION_MODE = READ_WRITE,
                QUERY_CAPTURE_MODE = AUTO,
                DATA_FLUSH_INTERVAL_SECONDS = 900
            )
            """
        )


def _quote_literal(value: str) -> str:
    """``N'...'`` with quotes doubled: ``ALTER LOGIN`` takes no parameter placeholders."""
    return "N'" + value.replace("'", "''") + "'"


def ensure_read_only_login(target: MssqlTarget) -> None:
    """Create (or repoint) the ``mdqm_ro`` login and its user in the database, read-only."""
    if not target.ro_password:
        raise LoaderError("MDQM_RO_PASSWORD is empty: refusing to create a passwordless login")
    with connect(target, autocommit=True, database="master") as conn, conn.cursor() as cur:
        cur.execute("SELECT name FROM sys.server_principals WHERE name = ?", (target.ro_user,))
        password = _quote_literal(target.ro_password)
        if cur.fetchone() is None:
            cur.execute(
                f"CREATE LOGIN [{target.ro_user}] WITH PASSWORD = {password}, "
                "CHECK_POLICY = OFF, CHECK_EXPIRATION = OFF"
            )
        else:
            # Keep the stored password in step with .env, and keep the login harmless.
            cur.execute(
                f"ALTER LOGIN [{target.ro_user}] WITH PASSWORD = {password}, "
                "CHECK_POLICY = OFF, CHECK_EXPIRATION = OFF"
            )
        # A login that exists here always started as a plain login, but drop-and-recreate of the
        # container is common in dev; make the demotion a no-op instead of an error.
        cur.execute(
            f"IF IS_SRVROLEMEMBER('sysadmin', '{target.ro_user}') = 1 "
            f"ALTER SERVER ROLE [sysadmin] DROP MEMBER [{target.ro_user}]"
        )
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
        cur.execute(f"ALTER LOGIN [{target.ro_user}] WITH DEFAULT_DATABASE = [{target.database}]")


def grant_read_only(target: MssqlTarget) -> None:
    """Give the login ``db_datareader`` + ``VIEW DATABASE STATE``; nothing that can write.

    ``VIEW DATABASE STATE`` is what makes ``sys.query_store_query`` (and the other
    ``sys.query_store_*`` views) readable for a non-admin login — the D06 acceptance item and the
    A08 plan reader. The grants are idempotent: re-running the loader must not fail on them.
    """
    with connect(target) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM sys.database_principals WHERE name = ? AND type IN ('S', 'U')",
            (target.ro_user,),
        )
        if cur.fetchone() is None:
            # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
            cur.execute(f"CREATE USER [{target.ro_user}] FOR LOGIN [{target.ro_user}]")
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
        cur.execute(f"ALTER ROLE [db_datareader] ADD MEMBER [{target.ro_user}]")
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
        cur.execute(f"GRANT VIEW DATABASE STATE TO [{target.ro_user}]")
        # SHOWPLAN is what makes SET SHOWPLAN_XML return a plan. Without it the read-only login
        # can turn the mode on and still be refused when the statement is compiled (error 229 is
        # the DML denial; SHOWPLAN's refusal is "SHOWPLAN permission denied in database").
        cur.execute(f"GRANT SHOWPLAN TO [{target.ro_user}]")
        conn.commit()


# --------------------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------------------


def _batched(rows: Sequence[Sequence[Any]], size: int = INSERT_BATCH) -> Iterator[list[Any]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def _insert(cur: Any, table: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> int:
    """``executemany`` in batches with bound values; returns the number of rows written."""
    placeholders = ", ".join("?" for _ in columns)
    # S608: table and columns are this module's constants; every value is bound.
    statement = f"INSERT INTO dbo.{table} ({', '.join(columns)}) VALUES ({placeholders})"  # noqa: S608
    written = 0
    for batch in _batched(rows, INSERT_BATCH):
        cur.executemany(statement, batch)
        written += len(batch)
    return written


def load(
    target: MssqlTarget | None = None,
    config: WorldConfig | None = None,
    *,
    world: World | None = None,
) -> LoadReport:
    """Create the erp schema and load the world into it; returns what was written.

    Idempotent by drop/recreate: a second run leaves the same rows, the same partition function
    and the same partition fill, because drop removes the scheme and function as well.
    """
    resolved = target if target is not None else target_from_environ()
    built = world if world is not None else build_world(config)
    tuples = partition_tuples(built.config.anchor)

    ensure_database(resolved)
    ensure_read_only_login(resolved)

    with connect(resolved) as conn, conn.cursor() as cur:
        drop_schema(cur)
        create_partitioning(cur, tuples)
        create_tables(cur)

        counts = {
            "erp.customers": _insert(
                cur,
                "customers",
                CUSTOMER_COLUMNS,
                [(c.cust_no, c.status_code, c.limit_k, c.name) for c in built.erp.customers],
            ),
            "erp.usage_events": _insert(
                cur,
                "usage_events",
                USAGE_COLUMNS,
                [
                    (
                        e.event_id,
                        e.client_no,
                        to_stored_timestamp(e.start_time),
                        to_stored_timestamp(e.insertion_time),
                        e.units,
                        e.corrects_event_id,
                    )
                    for e in built.billing.usage_events
                ],
            ),
        }
        # Statistics are not cosmetic: without them the optimizer has no cardinality to prune on
        # and A08's plan assertions would compare against a guess.
        for table in TABLES:
            # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
            cur.execute(f"UPDATE STATISTICS dbo.{table} WITH FULLSCAN")
        conn.commit()

        report = LoadReport(
            counts=counts,
            tuples=tuples,
            boundaries=partition_boundaries(cur),
            fanout=function_fanout(cur),
            partition_rows=table_partition_rows(cur),
            duplicates=duplicate_cust_nos(cur),
            query_store=query_store_state(cur),
        )

    # Grants come after the recreate: drop removed the objects they name.
    grant_read_only(resolved)
    return report


@dataclass(frozen=True, slots=True)
class LoadReport:
    """What one run wrote, plus the partition state as SQL Server reports it."""

    counts: dict[str, int] = field(default_factory=dict)
    tuples: tuple[datetime, ...] = ()
    boundaries: tuple[datetime, ...] = ()
    fanout: int = 0
    partition_rows: tuple[tuple[int, int], ...] = ()
    duplicates: tuple[tuple[int, int], ...] = ()
    query_store: str = ""

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def duplicate_rows(self) -> int:
        """Rows beyond the first for each duplicated ``cust_no`` (the intentionally dirty part)."""
        return sum(occurrences - 1 for _, occurrences in self.duplicates)

    @property
    def partitioned_rows(self) -> int:
        return sum(rows for _, rows in self.partition_rows)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: load the world and print counts, tuples and duplicates."""
    parser = argparse.ArgumentParser(
        prog="python -m mdqm.testing.loaders.sqlserver",
        description="Create the erp schema in SQL Server and load the synthetic world into it.",
    )
    parser.add_argument("--small", action="store_true", help="200 customers / ~5k events")
    parser.add_argument("--host", default=None, help=f"database host (default {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=None, help=f"port (default {DEFAULT_PORT})")
    parser.add_argument("--database", default=None, help=f"database (default {DEFAULT_DATABASE})")
    parser.add_argument("--months", type=int, default=None, help="event window in months")
    parser.add_argument("--quiet", action="store_true", help="print nothing but the counts")
    args = parser.parse_args(argv)

    config = WorldConfig.small() if args.small else WorldConfig()
    if args.months is not None:
        config = WorldConfig(
            seed=config.seed,
            customers=config.customers,
            months=args.months,
            events_per_customer=config.events_per_customer,
        )

    base = target_from_environ()
    target = MssqlTarget(
        host=base.host if args.host is None else args.host,
        port=base.port if args.port is None else args.port,
        database=base.database if args.database is None else args.database,
        admin_password=base.admin_password,
        ro_password=base.ro_password,
    )

    report = load(target, config)
    if not args.quiet:
        sys.stdout.write(
            f"loaded seed={config.seed} customers={config.customers} "
            f"events={config.event_count} into {target.server}/{target.database}\n"
        )
    width = max(len(name) for name in report.counts)
    for name in sorted(report.counts):
        sys.stdout.write(f"{name:<{width}}  {report.counts[name]:>8}\n")
    sys.stdout.write(f"{'total':<{width}}  {report.total:>8}\n")
    sys.stdout.write(
        f"partition function {PARTITION_FUNCTION}: {len(report.boundaries)} tuples -> "
        f"{report.fanout} partitions (scheme {PARTITION_SCHEME} "
        f"= {report.fanout} parts), query store = {report.query_store}\n"
    )
    for index, boundary in enumerate(report.boundaries, start=1):
        sys.stdout.write(f"  tuple {index:>2}  {boundary.isoformat(sep=' ')}\n")
    for number, rows in report.partition_rows:
        sys.stdout.write(f"  partition {number:>2}  rows={rows}\n")
    sys.stdout.write(
        f"duplicate cust_no keys: {len(report.duplicates)} "
        f"(+{report.duplicate_rows} rows): {report.duplicates}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
