"""Create the billing schema in Oracle Free and load the synthetic world into it (task D05).

Same rows as :mod:`mdqm.testing.loaders.postgres` (task D03), third engine, so the cross-source
conversions can be checked against a database that behaves differently:

* ``usage_events`` is **interval partitioned** on ``start_time``
  (``INTERVAL (NUMTOYMINTERVAL(1, 'MONTH'))``): Oracle creates one monthly partition per month
  that the data touches. The loader creates exactly **one** initial partition (``P_INIT``,
  ``< 2000-01-01``) and lets the database derive the rest, which is the point of the task: the
  partition set is a property of the data, not of the loader.
* ``start_time`` carries a **local** index (``LOCAL`` is not the default for a partitioned
  table in Oracle, and a global index would hide partition pruning from the plan); there is no
  index on ``insertion_time``, mirroring D03.
* ``credit_limit`` is the raw, unrounded value; ``client_no`` is a zero-padded ``VARCHAR2(10)``.
* Timestamps are ``TIMESTAMP(6)`` holding UTC. Oracle does not support interval partitioning on
  ``TIMESTAMP WITH TIME ZONE`` (``ORA-03001``, verified on 23ai Free), and DESIGN section 5 keeps
  one UTC ``exec_ts`` per execution, so the loader stores naive UTC and documents it here. The
  adapter layer attaches UTC again on read.

The loader is idempotent by drop/recreate: re-running produces the same rows and the same
partition set, because the world is seeded and has no clock. Per engine this also needs an owner
schema: ``MDQM_BILLING`` (created/refreshed by this loader) holds the tables. ``MDQM_RO`` receives
``SELECT`` on them plus ``SELECT`` on ``V$SQL``, and **every role it held is revoked**: the compose
image's ``APP_USER`` ships with ``DB_DEVELOPER_ROLE``, which lets it ``CREATE TABLE``, so the
read-only guarantee is enforced here and not assumed from the image. Plain ``CREATE SESSION`` is
granted back, because the login itself needs it. Seeding uses the ``SYS`` connection from ``.env``
(``MDQM_ORACLE_PASSWORD``); ``MDQM_RO`` is the read-only login the adapter uses.

Run from the repository root (the ``oracle`` compose profile must be up)::

    python -m mdqm.testing.loaders.oracle --small

Credentials and the port come from ``.env`` (``MDQM_ORACLE_PASSWORD``, ``MDQM_RO_PASSWORD``,
``MDQM_ORACLE_PORT``); nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import oracledb

from mdqm.testing.world import World, WorldConfig, build_world

DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 51521
"""Host port published by the ``oracle`` compose profile (see .env.example)."""

DEFAULT_SERVICE: Final = "MDQM"
"""Service name of the pluggable database the compose file creates (``ORACLE_DATABASE``)."""

OWNER_USER: Final = "MDQM_BILLING"
"""Schema that owns the billing tables. Created by this loader, because the image only makes
``MDQM_RO``."""

READ_ONLY_USER: Final = "MDQM_RO"
"""The compose image's ``APP_USER``: receives ``SELECT`` and nothing else."""

INITIAL_PARTITION: Final = "P_INIT"
"""The single partition the DDL creates; every other partition is derived from the data."""

INITIAL_BOUND: Final = "TIMESTAMP '2000-01-01 00:00:00'"
"""Upper bound of ``P_INIT``: before the world window, so every month in the data is derived."""

GENERATED_PARTITION_PREFIX: Final = "SYS_P"
"""Oracle names a partition it creates from the data ``SYS_P<number>``."""

PARTITION_CLAUSE: Final = (
    "partition by range (start_time)\n"
    "interval (numtoyminterval(1, 'month'))\n"
    f"(partition {INITIAL_PARTITION} values less than ({INITIAL_BOUND}))"
)

INSERT_BATCH: Final = 5000
"""Rows per ``executemany`` call: enough to keep the round trips rare, small enough to stay in
memory and to commit in slices on a slow first start."""

DOTENV_NAME: Final = ".env"

USAGE_COLUMNS: Final = (
    "event_id",
    "client_no",
    "start_time",
    "insertion_time",
    "units",
    "corrects_event_id",
)

STATUS_LOOKUP_ROWS: Final = (
    ("active", "A", "Active", 1, "פעיל", "active"),
    ("churned", "C", "Closed", 2, "עזב", "churned"),
    ("suspended", "S", "Hold", 3, "מושהה", "suspended"),
)

BILLING_TABLES: Final = ("usage_by_load", "usage_events", "accounts_hist", "status_lookup")
"""Tables dropped and recreated on every run, child-first order for readability."""

ALL_TABLES: Final = (*BILLING_TABLES, "accounts")


class LoaderError(RuntimeError):
    """The loader could not do its job; the message names what and where."""


# --------------------------------------------------------------------------------------
# Target and credentials
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OracleTarget:
    """Where the database lives and who may seed it, versus who may only read."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    service_name: str = DEFAULT_SERVICE
    admin_user: str = "sys"
    admin_password: str = ""
    owner_user: str = OWNER_USER
    ro_user: str = READ_ONLY_USER
    ro_password: str = ""

    @property
    def dsn(self) -> str:
        """EzConnect string of the pluggable database."""
        return f"{self.host}:{self.port}/{self.service_name}"

    @property
    def owner_password(self) -> str:
        """The owner schema logs in with the same throwaway dev password as ``SYS``.

        One password fewer in ``.env`` than a separate owner credential would need; both are
        local-only throwaway values bound to 127.0.0.1 (see .env.example).
        """
        return self.admin_password


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines of a dotenv file. Missing file yields an empty mapping.

    Deliberately tiny, and deliberately local to this loader: importing the PostgreSQL loader for
    it would pull ``psycopg`` (an optional extra) into an Oracle-only path.
    """
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
    """Nearest ``.env`` walking up from ``start``, so a worktree under ``.worktrees/`` reuses the
    checkout's credentials file."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        path = candidate / DOTENV_NAME
        if path.is_file():
            return path
    return None


def target_from_environ(
    *, environ: dict[str, str] | None = None, dotenv_path: Path | None = None
) -> OracleTarget:
    """Build an :class:`OracleTarget` from the environment, falling back to the dotenv file."""
    path = dotenv_path if dotenv_path is not None else find_dotenv()
    values = read_dotenv(path) if path is not None else {}
    env: dict[str, str] = dict(os.environ) if environ is None else environ
    values.update({key: value for key, value in env.items() if key.startswith("MDQM_")})

    password = values.get("MDQM_ORACLE_PASSWORD", "")
    if not password:
        where = path if path is not None else "no dotenv file found"
        raise LoaderError(
            f"MDQM_ORACLE_PASSWORD is not set ({where}): copy .env.example to .env "
            "(`python scripts/dev.py env`) or export the variable"
        )
    raw_port = values.get("MDQM_ORACLE_PORT", str(DEFAULT_PORT))
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise LoaderError(f"MDQM_ORACLE_PORT must be an integer, got {raw_port!r}") from exc
    return OracleTarget(
        host=env.get("MDQM_ORACLE_HOST", DEFAULT_HOST),
        port=port,
        service_name=env.get("MDQM_ORACLE_SERVICE", DEFAULT_SERVICE),
        admin_password=password,
        ro_password=values.get("MDQM_RO_PASSWORD", ""),
    )


def _quote_password(password: str) -> str:
    """Quote a password for ``IDENTIFIED BY`` and refuse one that would break the quoting."""
    if not password or '"' in password:
        raise LoaderError("the Oracle password must be non-empty and must not contain a quote")
    return f'"{password}"'


def connect_admin(target: OracleTarget) -> oracledb.Connection:
    """``SYS ... AS SYSDBA`` connection to the pluggable database."""
    return oracledb.connect(
        user="sys",
        password=target.admin_password,
        dsn=target.dsn,
        mode=oracledb.AUTH_MODE_SYSDBA,
    )


def connect_owner(target: OracleTarget) -> oracledb.Connection:
    """The owner schema's connection: the only one that may write."""
    return oracledb.connect(user=target.owner_user, password=target.owner_password, dsn=target.dsn)


def connect_read_only(target: OracleTarget) -> oracledb.Connection:
    """The read-only login the adapter uses (``MDQM_RO``)."""
    return oracledb.connect(user=target.ro_user, password=target.ro_password, dsn=target.dsn)


# --------------------------------------------------------------------------------------
# Timestamps and partitions
# --------------------------------------------------------------------------------------


def to_stored_timestamp(value: datetime) -> datetime:
    """Convert a world timestamp (aware UTC) to the naive UTC ``TIMESTAMP(6)`` Oracle stores."""
    if value.tzinfo is None:
        raise LoaderError(
            f"refusing to store a naive timestamp, it cannot be placed in UTC: {value}"
        )
    return value.astimezone(UTC).replace(tzinfo=None)


def month_start(value: datetime) -> datetime:
    """First instant of the UTC month that contains ``value``."""
    return datetime(value.year, value.month, 1)


def generation_expected_months(world: World) -> tuple[datetime, ...]:
    """Month starts the world's events cover: how many partitions Oracle must derive."""
    months = {
        month_start(to_stored_timestamp(event.start_time)) for event in world.billing.usage_events
    }
    return tuple(sorted(months))


@dataclass(frozen=True, slots=True)
class Partition:
    """One partition of a table, as ``USER_TAB_PARTITIONS`` reports it."""

    name: str
    high_value: str
    month: datetime | None
    """Upper bound parsed out of ``HIGH_VALUE``; ``None`` when it is not a timestamp."""

    @property
    def generated(self) -> bool:
        """True when Oracle created this partition from the data, not the loader."""
        return self.name.startswith(GENERATED_PARTITION_PREFIX)


def parse_high_value(text: str) -> datetime | None:
    """Parse ``TIMESTAMP' 2025-07-01 00:00:00'`` (the catalog's ``LONG`` form) into a datetime."""
    body = text.strip()
    if body.upper().startswith("TIMESTAMP'"):
        body = body.split("'", 1)[1].strip().rstrip("'").strip()
    try:
        return datetime.fromisoformat(body)
    except ValueError:
        return None


def partitions_of(cur: oracledb.Cursor, table: str) -> tuple[Partition, ...]:
    """Partitions of ``table`` in creation order, straight from ``USER_TAB_PARTITIONS``."""
    cur.execute(
        "SELECT partition_name, high_value FROM user_tab_partitions "
        "WHERE table_name = :1 ORDER BY partition_position",
        [table.upper()],
    )
    out: list[Partition] = []
    for row in cur.fetchall():
        name = str(row[0])
        high_value = "" if row[1] is None else str(row[1])
        out.append(Partition(name=name, high_value=high_value, month=parse_high_value(high_value)))
    return tuple(out)


# --------------------------------------------------------------------------------------
# DDL
# --------------------------------------------------------------------------------------


def ensure_owner(target: OracleTarget, admin: oracledb.Connection) -> None:
    """Create or refresh the owner schema, so the loader works on a fresh container too."""
    cur = admin.cursor()
    identifier = target.owner_user.upper()
    password = _quote_password(target.owner_password)
    try:
        # Oracle cannot bind an identifier or a password in DDL; both are escaped above.
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
        cur.execute(f"create user {identifier} identified by {password} quota unlimited on users")
    except oracledb.DatabaseError as exc:
        # ORA-01920: the user already exists - the loader is re-run on the same container.
        if "ORA-01920" not in str(exc):
            raise
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
        cur.execute(f"alter user {identifier} identified by {password} quota unlimited on users")
    # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
    cur.execute(f"grant create session, create table to {identifier}")
    admin.commit()


ROLE_NAME = re.compile(r"^[A-Z0-9_$#]+$")
"""A dictionary role name that is safe to interpolate into ``REVOKE``."""


def ensure_read_only_user(target: OracleTarget, admin: oracledb.Connection) -> None:
    """Make ``MDQM_RO`` what its name promises: login plus ``SELECT``, nothing else.

    The compose image creates ``APP_USER`` with ``DB_DEVELOPER_ROLE``, which carries ``CREATE
    TABLE`` (verified: without this revocation the account could create its own table). Every
    granted role is revoked and ``CREATE SESSION`` is granted directly instead, so the account
    cannot write anywhere - not even into its own schema.
    """
    cur = admin.cursor()
    identifier = target.ro_user.upper()
    try:
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
        cur.execute(f"create user {identifier} identified by {_quote_password(target.ro_password)}")
    except oracledb.DatabaseError as exc:
        # ORA-01920: the image already created it.
        if "ORA-01920" not in str(exc):
            raise
    cur.execute("select granted_role from dba_role_privs where grantee = :1", [identifier])
    for row in cur.fetchall():
        role = str(row[0]).upper()
        if not ROLE_NAME.match(role):
            raise LoaderError(f"refusing to revoke a role with an unexpected name: {role!r}")
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
        cur.execute(f"revoke {role} from {identifier}")
    # Oracle cannot bind an identifier in DDL; this one is a schema name the loader owns.
    # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
    cur.execute(f"grant create session to {identifier}")
    admin.commit()


def drop_tables(cur: oracledb.Cursor) -> None:
    """Drop what a previous run left. ``PURGE`` keeps the recycle bin from filling up."""
    for table in ALL_TABLES:
        try:
            # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
            cur.execute(f"drop table {table} purge")
        except oracledb.DatabaseError as exc:
            # ORA-00942: first run, nothing to drop yet.
            if "ORA-00942" not in str(exc):
                raise


def create_accounts(cur: oracledb.Cursor) -> None:
    """``accounts``: the billing storage form. ``credit_limit`` is raw, never rounded."""
    cur.execute(
        """
        create table accounts (
            client_no     varchar2(10) not null,
            state         varchar2(16) not null,
            credit_limit  number(12),
            region        varchar2(32) not null,
            opened_date   date         not null,
            constraint accounts_pk primary key (client_no),
            constraint accounts_state_ck check (state in ('Active', 'Closed', 'Hold'))
        )
        """
    )


def create_usage_tables(cur: oracledb.Cursor) -> None:
    """Both usage tables: interval monthly partitions, event time indexed, ingestion time not."""
    # The DDL is fixed text; Oracle cannot bind an identifier and none is interpolated here.
    # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
    cur.execute(
        f"""
        create table usage_events (
            event_id           number(19)   not null,
            client_no          varchar2(10) not null,
            start_time         timestamp(6) not null,
            insertion_time     timestamp(6) not null,
            units              number(10)   not null,
            corrects_event_id  number(19),
            constraint usage_events_pk primary key (event_id, start_time)
        )
        {PARTITION_CLAUSE}
        """
    )
    # LOCAL on purpose: a global index answers the query without touching a partition, which
    # would hide PSTART/PSTOP from EXPLAIN PLAN and defeat the point of this table.
    cur.execute("create index usage_events_start_time_idx on usage_events (start_time) local")
    cur.execute(
        """
        create table usage_by_load (
            event_id           number(19)   not null,
            client_no          varchar2(10) not null,
            start_time         timestamp(6) not null,
            insertion_time     timestamp(6) not null,
            units              number(10)   not null,
            corrects_event_id  number(19),
            constraint usage_by_load_pk primary key (event_id, insertion_time)
        )
        partition by range (insertion_time)
        interval (numtoyminterval(1, 'month'))
        (partition p_load_init values less than (timestamp '2000-01-01 00:00:00'))
        """
    )
    cur.execute(
        "create index usage_by_load_insertion_time_idx on usage_by_load (insertion_time) local"
    )


def create_history_and_lookup(cur: oracledb.Cursor) -> None:
    """``accounts_hist`` (frequently updated, mutable-critical) and the bilingual lookup."""
    cur.execute(
        """
        create table accounts_hist (
            hist_id       number(19)   not null,
            client_no     varchar2(10) not null,
            state         varchar2(16) not null,
            credit_limit  number(12)   not null,
            updated_at    timestamp(6) not null,
            constraint accounts_hist_pk primary key (hist_id)
        )
        """
    )
    cur.execute("create index accounts_hist_client_no_idx on accounts_hist (client_no)")
    cur.execute(
        """
        create table status_lookup (
            canonical     varchar2(16) not null,
            crm_code      char(1)      not null,
            billing_code  varchar2(16) not null,
            erp_code      number(3)    not null,
            name_he       varchar2(64) not null,
            name_en       varchar2(64) not null,
            constraint status_lookup_pk primary key (canonical)
        )
        """
    )


# --------------------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------------------


def _batched(
    rows: Sequence[Sequence[Any]], size: int = INSERT_BATCH
) -> Iterator[list[Sequence[Any]]]:
    """Slices of at most ``size`` rows, as lists because ``executemany`` requires a list."""
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def _insert(
    cur: oracledb.Cursor, table: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]
) -> int:
    """``executemany`` in batches; returns the number of rows written."""
    placeholders = ", ".join(f":{index}" for index in range(1, len(columns) + 1))
    # S608: table and columns are this module's own constants; every value is bound.
    statement = f"insert into {table} ({', '.join(columns)}) values ({placeholders})"  # noqa: S608
    written = 0
    for batch in _batched(rows, INSERT_BATCH):
        # pi-lens-ignore: python-sql-injection
        cur.executemany(statement, batch)
        written += len(batch)
    return written


def _usage_rows(world: World) -> list[tuple[Any, ...]]:
    return [
        (
            event.event_id,
            event.client_no,
            to_stored_timestamp(event.start_time),
            to_stored_timestamp(event.insertion_time),
            event.units,
            event.corrects_event_id,
        )
        for event in world.billing.usage_events
    ]


def load_billing(
    cur: oracledb.Cursor, world: World
) -> tuple[dict[str, int], tuple[Partition, ...], tuple[Partition, ...]]:
    """Create and fill the billing schema.

    Returns ``(counts, partitions_before_insert, partitions_after_insert)``: the two partition
    snapshots are the proof that the loader pre-created exactly one partition and Oracle derived
    the rest from the data.
    """
    drop_tables(cur)
    create_accounts(cur)
    create_usage_tables(cur)
    create_history_and_lookup(cur)
    before = partitions_of(cur, "usage_events")

    counts = {
        "billing.accounts": _insert(
            cur,
            "accounts",
            ("client_no", "state", "credit_limit", "region", "opened_date"),
            [
                (a.client_no, a.state, a.credit_limit, a.region, a.opened_date)
                for a in world.billing.accounts
            ],
        ),
        "billing.accounts_hist": _insert(
            cur,
            "accounts_hist",
            ("hist_id", "client_no", "state", "credit_limit", "updated_at"),
            [
                (
                    h.hist_id,
                    h.client_no,
                    h.state,
                    h.credit_limit,
                    to_stored_timestamp(h.updated_at),
                )
                for h in world.billing.accounts_hist
            ],
        ),
        "billing.status_lookup": _insert(
            cur,
            "status_lookup",
            ("canonical", "crm_code", "billing_code", "erp_code", "name_he", "name_en"),
            STATUS_LOOKUP_ROWS,
        ),
    }

    events = _usage_rows(world)
    counts["billing.usage_events"] = _insert(cur, "usage_events", USAGE_COLUMNS, events)
    counts["billing.usage_by_load"] = _insert(cur, "usage_by_load", USAGE_COLUMNS, events)

    # Statistics are not cosmetic here: without them the optimizer has no reason to prune, and
    # EXPLAIN PLAN would not expose PSTART/PSTOP (the D05 acceptance criterion).
    for table in ALL_TABLES:
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
        cur.execute(f"begin dbms_stats.gather_table_stats(user, '{table.upper()}'); end;")

    return counts, before, partitions_of(cur, "usage_events")


def grant_read_only(admin: oracledb.Connection, target: OracleTarget) -> None:
    """Give ``MDQM_RO`` ``SELECT`` on the loaded tables and on ``V$SQL``; nothing else.

    Oracle grants a ``V_$`` view like any other object, so one statement covers the plan/statistics
    inspection the adapter needs. The role keeps no ``INSERT``/``UPDATE``/``DELETE``/``CREATE``
    privilege, which is what the read-only test asserts.
    """
    cur = admin.cursor()
    owner = target.owner_user.upper()
    for table in ALL_TABLES:
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
        cur.execute(f"grant select on {owner}.{table.upper()} to {target.ro_user.upper()}")
    # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
    cur.execute(f"grant select on v_$sql to {target.ro_user.upper()}")
    admin.commit()


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadReport:
    """What one loader run wrote, plus the partition snapshots that prove how they were made."""

    counts: dict[str, int] = field(default_factory=dict)
    partitions_before_insert: tuple[Partition, ...] = ()
    partitions: tuple[Partition, ...] = ()

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def generated_partitions(self) -> tuple[Partition, ...]:
        """Partitions Oracle created from the data (``SYS_P*``), i.e. not the loader's."""
        return tuple(partition for partition in self.partitions if partition.generated)


def load(
    target: OracleTarget | None = None,
    config: WorldConfig | None = None,
    *,
    world: World | None = None,
) -> LoadReport:
    """Create the billing schema in the owner and load the world into it.

    Idempotent by drop/recreate: a second run leaves the same rows, the same indexes and the same
    partition set (drop removes the derived partitions with the table).
    """
    resolved = target if target is not None else target_from_environ()
    built = world if world is not None else build_world(config)

    with connect_admin(resolved) as admin:
        ensure_owner(resolved, admin)
        ensure_read_only_user(resolved, admin)

    with connect_owner(resolved) as owner:
        cur = owner.cursor()
        counts, before, after = load_billing(cur, built)
        owner.commit()

    # The grants must come after the recreate: drop removed the objects they name.
    with connect_admin(resolved) as admin:
        grant_read_only(admin, resolved)

    return LoadReport(counts=counts, partitions_before_insert=before, partitions=after)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: load the world, print the row counts and the derived partitions."""
    parser = argparse.ArgumentParser(
        prog="python -m mdqm.testing.loaders.oracle",
        description="Create the billing schema in Oracle and load the synthetic world into it.",
    )
    parser.add_argument("--small", action="store_true", help="200 customers / ~5k events")
    parser.add_argument("--host", default=None, help=f"database host (default {DEFAULT_HOST})")
    parser.add_argument(
        "--port", type=int, default=None, help=f"database port (default {DEFAULT_PORT})"
    )
    parser.add_argument("--service", default=None, help=f"service name (default {DEFAULT_SERVICE})")
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
    target = OracleTarget(
        host=base.host if args.host is None else args.host,
        port=base.port if args.port is None else args.port,
        service_name=base.service_name if args.service is None else args.service,
        admin_password=base.admin_password,
        ro_password=base.ro_password,
    )

    report = load(target, config)
    if not args.quiet:
        sys.stdout.write(
            f"loaded seed={config.seed} customers={config.customers} "
            f"events={config.event_count} into {target.dsn} "
            f"(owner {target.owner_user})\n"
        )
    width = max(len(name) for name in report.counts)
    for name in sorted(report.counts):
        sys.stdout.write(f"{name:<{width}}  {report.counts[name]:>8}\n")
    sys.stdout.write(f"{'total':<{width}}  {report.total:>8}\n")
    sys.stdout.write(
        f"usage_events partitions: {len(report.partitions_before_insert)} before the insert, "
        f"{len(report.partitions)} after ({len(report.generated_partitions)} derived by Oracle)\n"
    )
    for partition in report.partitions:
        marker = "derived" if partition.generated else "created"
        sys.stdout.write(f"  {partition.name:<12} {partition.high_value.strip():<40} {marker}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
