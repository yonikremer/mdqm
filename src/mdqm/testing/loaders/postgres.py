"""Create the PostgreSQL schema and load the synthetic world into it (task D03).

The same organisation in three databases, mirroring the three storage forms of the world:

======  ==========================================================================
crm     ``customers`` — PK ``cust_id``, status ``'A'``/``'C'``/``'S'``
billing ``accounts`` (PK ``client_no``), ``usage_events`` (RANGE monthly on
        ``start_time``), ``usage_by_load`` (RANGE monthly on ``insertion_time``),
        ``accounts_hash`` (HASH on ``client_no``, 4 partitions), ``accounts_hist``
        (frequently updated), ``status_lookup`` (small lookup table)
erp     ``customers`` — deliberately **no** primary key: the world plants duplicate rows
======  ==========================================================================

Why the schema looks like this:

* ``usage_events`` is partitioned on **event time** and carries an index on ``start_time``
  only. There is no index on ``insertion_time`` on purpose: a query filtered on ingestion time
  must scan every partition, which is exactly what the planner must be able to see.
* ``usage_by_load`` holds the same rows partitioned on **ingestion time**, so the "event time
  versus insertion time" choice in the planner has a real, measurable difference.
* The primary key of a partitioned table must contain the partition key, hence
  ``(event_id, start_time)``, not ``event_id`` alone.
* Partitions are pre-created (not auto-created) so the count is deterministic: 12+ monthly
  partitions of which only the world's window carries rows.

The loader is idempotent by drop/create: re-running it produces the same table contents, byte
for byte, because the world is seeded and has no clock. Seeding uses the owner connection; the
read-only role ``mdqm_ro`` only ever gets ``SELECT`` and ``default_transaction_read_only=on``.

Run from the repository root (the ``pg`` compose profile must be up)::

    python -m mdqm.testing.loaders.postgres --small

Credentials come from ``.env`` (``MDQM_PG_PASSWORD``, ``MDQM_RO_PASSWORD``, ``MDQM_PG_PORT``)
or from the same environment variables; nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import psycopg
from psycopg import conninfo, sql

from mdqm.testing.world import ANCHOR, World, WorldConfig, build_world

DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 55440
"""Host port published by the ``pg`` compose profile (see .env.example)."""

DATABASES: Final = ("crm", "billing", "erp")

READ_ONLY_ROLE: Final = "mdqm_ro"

PARTITION_MONTHS_BACK: Final = 14
"""Months of ``usage_events`` partitions created before the anchor month inclusive."""

PARTITION_MONTHS_FORWARD: Final = 2
"""Months created after the anchor for ``usage_by_load`` only (see :func:`monthly_partitions`)."""

HASH_PARTITIONS: Final = 4

DOTENV_NAME: Final = ".env"
"""Credentials file name, looked up by walking up from the working directory."""

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


class LoaderError(RuntimeError):
    """The loader could not do its job; the message names what and where."""


# --------------------------------------------------------------------------------------
# Target and credentials
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PgTarget:
    """Where the three databases live and who may seed them."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    admin_user: str = "postgres"
    admin_password: str = ""
    ro_user: str = READ_ONLY_ROLE
    ro_password: str = ""

    def dsn(self, database: str, *, read_only: bool = False) -> str:
        """Connection string for ``database``, as the owner or as the read-only role."""
        return conninfo.make_conninfo(
            host=self.host,
            port=self.port,
            dbname=database,
            user=self.ro_user if read_only else self.admin_user,
            password=self.ro_password if read_only else self.admin_password,
            connect_timeout=10,
        )


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines of a dotenv file. Missing file yields an empty mapping.

    Deliberately tiny: the loader needs three values and must not depend on a settings stack
    that would also demand sources, secrets and a model endpoint.
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
    """Nearest credentials file walking up from ``start`` (the working directory by default).

    Walking up rather than resolving the package location means a git worktree under
    ``.worktrees/<task>`` reuses the checkout's file instead of needing its own copy.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        path = candidate / DOTENV_NAME
        if path.is_file():
            return path
    return None


def target_from_environ(
    *, environ: dict[str, str] | None = None, dotenv_path: Path | None = None
) -> PgTarget:
    """Build a :class:`PgTarget` from the environment, falling back to the dotenv file."""
    path = dotenv_path if dotenv_path is not None else find_dotenv()
    values = read_dotenv(path) if path is not None else {}
    env: dict[str, str] = dict(os.environ) if environ is None else environ
    values.update({key: value for key, value in env.items() if key.startswith("MDQM_")})

    password = values.get("MDQM_PG_PASSWORD", "")
    if not password:
        where = path if path is not None else "no dotenv file found"
        raise LoaderError(
            f"MDQM_PG_PASSWORD is not set ({where}): copy .env.example to .env "
            "(`python scripts/dev.py env`) or export the variable"
        )
    return PgTarget(
        host=env.get("MDQM_PG_HOST", DEFAULT_HOST),
        port=int(values.get("MDQM_PG_PORT", DEFAULT_PORT)),
        admin_password=password,
        ro_password=values.get("MDQM_RO_PASSWORD", ""),
    )


# --------------------------------------------------------------------------------------
# Partitions
# --------------------------------------------------------------------------------------


def _month_start(value: datetime) -> datetime:
    """First instant of the UTC month that contains ``value``."""
    return datetime(value.year, value.month, 1, tzinfo=UTC)


def _shift_months(value: datetime, months: int) -> datetime:
    total = value.year * 12 + (value.month - 1) + months
    return datetime(total // 12, total % 12 + 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class Partition:
    """One monthly RANGE partition: ``[lower, upper)``."""

    suffix: str
    lower: datetime
    upper: datetime


def monthly_partitions(
    anchor: datetime = ANCHOR,
    *,
    back: int = PARTITION_MONTHS_BACK,
    forward: int = PARTITION_MONTHS_FORWARD,
) -> tuple[Partition, ...]:
    """Monthly partitions covering ``anchor - back months`` .. ``anchor + forward months``.

    The anchor month itself is included (``back`` counts from it), so the world window
    ``anchor - months`` .. ``anchor`` always lands inside the created range.

    ``usage_events`` is created with ``forward=0``: an open-ended lower bound
    (``start_time >= X``) cannot prune a partition that lies entirely after it, so extra
    future partitions would be scanned for no reason. ``usage_by_load`` needs them, because
    ingestion time may pass the anchor (lag <= ``max_lag_seconds``).
    """
    first = _shift_months(_month_start(anchor), -back)
    last = _shift_months(_month_start(anchor), forward)
    out: list[Partition] = []
    cursor = first
    while cursor < last:
        follow = _shift_months(cursor, 1)
        out.append(Partition(f"p{cursor.year:04d}_{cursor.month:02d}", cursor, follow))
        cursor = follow
    return tuple(out)


# --------------------------------------------------------------------------------------
# DDL
# --------------------------------------------------------------------------------------

_USAGE_EVENT_COLUMNS: Final = """
    event_id            BIGINT      NOT NULL,
    client_no           VARCHAR(10) NOT NULL,
    start_time          TIMESTAMPTZ NOT NULL,
    insertion_time      TIMESTAMPTZ NOT NULL,
    units               INTEGER     NOT NULL,
    corrects_event_id   BIGINT
"""


def _drop(cur: psycopg.Cursor[Any], tables: Sequence[str]) -> None:
    for table in tables:
        cur.execute(sql.SQL("DROP TABLE IF EXISTS public.{} CASCADE").format(sql.Identifier(table)))


def _drop_views(cur: psycopg.Cursor[Any], views: Sequence[str]) -> None:
    for view in views:
        cur.execute(sql.SQL("DROP VIEW IF EXISTS public.{} CASCADE").format(sql.Identifier(view)))


def _create_partition(cur: psycopg.Cursor[Any], parent: str, partition: Partition) -> None:
    cur.execute(
        sql.SQL(
            "CREATE TABLE public.{} PARTITION OF public.{} FOR VALUES FROM ({}) TO ({})"
        ).format(
            sql.Identifier(f"{parent}_{partition.suffix}"),
            sql.Identifier(parent),
            sql.Literal(partition.lower),
            sql.Literal(partition.upper),
        )
    )


def create_crm(cur: psycopg.Cursor[Any]) -> None:
    """``crm``: the CRM storage form. Status is the one-letter code."""
    _drop(cur, ("customers",))
    cur.execute(
        """
        CREATE TABLE public.customers (
            cust_id      INTEGER PRIMARY KEY,
            full_name    TEXT        NOT NULL,
            status       CHAR(1)     NOT NULL CHECK (status IN ('A', 'C', 'S')),
            credit_limit NUMERIC     NOT NULL,
            region       TEXT        NOT NULL,
            signup_date  DATE        NOT NULL
        )
        """
    )


def create_billing(
    cur: psycopg.Cursor[Any],
    event_partitions: Sequence[Partition],
    load_partitions: Sequence[Partition],
) -> None:
    """``billing``: accounts, both usage tables, the hash table, history and lookup."""
    _drop(
        cur,
        (
            "accounts",
            "usage_events",
            "usage_by_load",
            "accounts_hash",
            "accounts_hist",
            "status_lookup",
        ),
    )
    _drop_views(cur, ("accounts_current",))

    cur.execute(
        """
        CREATE TABLE public.accounts (
            client_no     VARCHAR(10) PRIMARY KEY,
            state         VARCHAR(16) NOT NULL CHECK (state IN ('Active', 'Closed', 'Hold')),
            credit_limit  NUMERIC,
            region        TEXT        NOT NULL,
            opened_date   DATE        NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE public.accounts_hash (
            client_no     VARCHAR(10) NOT NULL,
            state         VARCHAR(16) NOT NULL CHECK (state IN ('Active', 'Closed', 'Hold')),
            credit_limit  NUMERIC,
            region        TEXT        NOT NULL,
            opened_date   DATE        NOT NULL,
            PRIMARY KEY (client_no)
        ) PARTITION BY HASH (client_no)
        """
    )
    for remainder in range(HASH_PARTITIONS):
        cur.execute(
            sql.SQL(
                "CREATE TABLE public.{} PARTITION OF public.accounts_hash "
                "FOR VALUES WITH (MODULUS {}, REMAINDER {})"
            ).format(
                sql.Identifier(f"accounts_hash_p{remainder}"),
                sql.Literal(HASH_PARTITIONS),
                sql.Literal(remainder),
            )
        )

    cur.execute(
        f"""
        CREATE TABLE public.usage_events (
            {_USAGE_EVENT_COLUMNS},
            CONSTRAINT usage_events_pk PRIMARY KEY (event_id, start_time)
        ) PARTITION BY RANGE (start_time)
        """
    )
    cur.execute(
        f"""
        CREATE TABLE public.usage_by_load (
            {_USAGE_EVENT_COLUMNS},
            CONSTRAINT usage_by_load_pk PRIMARY KEY (event_id, insertion_time)
        ) PARTITION BY RANGE (insertion_time)
        """
    )
    for partition in event_partitions:
        _create_partition(cur, "usage_events", partition)
    for partition in load_partitions:
        _create_partition(cur, "usage_by_load", partition)

    # Event time is indexed; ingestion time is not, on purpose (see the module docstring).
    cur.execute("CREATE INDEX usage_events_start_time_idx ON public.usage_events (start_time)")
    cur.execute(
        "CREATE INDEX usage_by_load_insertion_time_idx ON public.usage_by_load (insertion_time)"
    )

    cur.execute(
        """
        CREATE TABLE public.accounts_hist (
            hist_id      BIGINT      NOT NULL,
            client_no    VARCHAR(10) NOT NULL,
            state        VARCHAR(16) NOT NULL,
            credit_limit NUMERIC     NOT NULL,
            updated_at   TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (hist_id)
        )
        """
    )
    cur.execute("CREATE INDEX accounts_hist_client_no_idx ON public.accounts_hist (client_no)")

    cur.execute(
        """
        CREATE TABLE public.status_lookup (
            canonical    TEXT      PRIMARY KEY,
            crm_code     CHAR(1)   NOT NULL,
            billing_code VARCHAR(16) NOT NULL,
            erp_code     SMALLINT  NOT NULL,
            name_he      TEXT      NOT NULL,
            name_en      TEXT      NOT NULL
        )
        """
    )

    # "One row per account, latest revision": the shape the mutable_critical table exists for.
    cur.execute(
        """
        CREATE VIEW public.accounts_current AS
        SELECT DISTINCT ON (client_no) client_no, state, credit_limit, updated_at
        FROM public.accounts_hist
        ORDER BY client_no, updated_at DESC, hist_id DESC
        """
    )


def create_erp(cur: psycopg.Cursor[Any]) -> None:
    """``erp``: the ERP storage form. No primary key: the world plants duplicate rows."""
    _drop(cur, ("customers",))
    cur.execute(
        """
        CREATE TABLE public.customers (
            cust_no     INTEGER     NOT NULL,
            name        TEXT        NOT NULL,
            status_code SMALLINT    NOT NULL CHECK (status_code IN (1, 2, 3)),
            limit_k     NUMERIC     NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX customers_cust_no_idx ON public.customers (cust_no)")


# --------------------------------------------------------------------------------------
# COPY
# --------------------------------------------------------------------------------------


def _copy(
    cur: psycopg.Cursor[Any], table: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]
) -> int:
    """COPY ``rows`` into ``table``; returns the number of rows written."""
    statement = sql.SQL("COPY {} ({}) FROM STDIN").format(
        sql.Identifier("public", table),
        sql.SQL(", ").join(sql.Identifier(column) for column in columns),
    )
    written = 0
    with cur.copy(statement) as copy:
        for row in rows:
            copy.write_row(row)
            written += 1
    return written


def _usage_rows(world: World) -> list[tuple[Any, ...]]:
    return [
        (
            event.event_id,
            event.client_no,
            event.start_time,
            event.insertion_time,
            event.units,
            event.corrects_event_id,
        )
        for event in world.billing.usage_events
    ]


def load_crm(cur: psycopg.Cursor[Any], world: World) -> dict[str, int]:
    """Load the CRM dataset; returns ``{table: rows}``."""
    create_crm(cur)
    rows = _copy(
        cur,
        "customers",
        ("cust_id", "full_name", "status", "credit_limit", "region", "signup_date"),
        (
            (c.cust_id, c.full_name, c.status, c.credit_limit, c.region, c.signup_date)
            for c in world.crm.customers
        ),
    )
    return {"crm.customers": rows}


def load_billing(
    cur: psycopg.Cursor[Any],
    world: World,
    event_partitions: Sequence[Partition],
    load_partitions: Sequence[Partition],
) -> dict[str, int]:
    """Load the billing dataset; returns ``{table: rows}``."""
    create_billing(cur, event_partitions, load_partitions)
    accounts = _copy(
        cur,
        "accounts",
        ("client_no", "state", "credit_limit", "region", "opened_date"),
        (
            (a.client_no, a.state, a.credit_limit, a.region, a.opened_date)
            for a in world.billing.accounts
        ),
    )
    hashed = _copy(
        cur,
        "accounts_hash",
        ("client_no", "state", "credit_limit", "region", "opened_date"),
        (
            (a.client_no, a.state, a.credit_limit, a.region, a.opened_date)
            for a in world.billing.accounts
        ),
    )
    history = _copy(
        cur,
        "accounts_hist",
        ("hist_id", "client_no", "state", "credit_limit", "updated_at"),
        (
            (h.hist_id, h.client_no, h.state, h.credit_limit, h.updated_at)
            for h in world.billing.accounts_hist
        ),
    )
    lookup = _copy(
        cur,
        "status_lookup",
        ("canonical", "crm_code", "billing_code", "erp_code", "name_he", "name_en"),
        STATUS_LOOKUP_ROWS,
    )
    events = _usage_rows(world)
    by_event = _copy(cur, "usage_events", USAGE_COLUMNS, events)
    by_load = _copy(cur, "usage_by_load", USAGE_COLUMNS, events)
    return {
        "billing.accounts": accounts,
        "billing.accounts_hash": hashed,
        "billing.accounts_hist": history,
        "billing.status_lookup": lookup,
        "billing.usage_events": by_event,
        "billing.usage_by_load": by_load,
    }


def load_erp(cur: psycopg.Cursor[Any], world: World) -> dict[str, int]:
    """Load the ERP dataset (duplicates and all); returns ``{table: rows}``."""
    create_erp(cur)
    rows = _copy(
        cur,
        "customers",
        ("cust_no", "name", "status_code", "limit_k"),
        ((c.cust_no, c.name, c.status_code, c.limit_k) for c in world.erp.customers),
    )
    return {"erp.customers": rows}


# --------------------------------------------------------------------------------------
# Read-only role
# --------------------------------------------------------------------------------------

_READ_ONLY_SETUP: Final = (
    "CREATE EXTENSION IF NOT EXISTS pg_stat_statements",
    sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(READ_ONLY_ROLE)),
    sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(
        sql.Identifier(READ_ONLY_ROLE)
    ),
    sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {}").format(
        sql.Identifier(READ_ONLY_ROLE)
    ),
    sql.SQL("GRANT SELECT ON pg_stat_statements TO {}").format(sql.Identifier(READ_ONLY_ROLE)),
    sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(
        sql.Identifier(READ_ONLY_ROLE)
    ),
)


def grant_read_only(cur: psycopg.Cursor[Any]) -> None:
    """Give ``mdqm_ro`` SELECT on everything loaded, and read-only transactions by default.

    PostgreSQL grants privileges on ``pg_stat_statements`` the same way as on a table, so the
    same statement covers acceptance item 4 (the view is readable by the read-only role).
    """
    for statement in _READ_ONLY_SETUP:
        cur.execute(statement)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadReport:
    """What one loader run wrote, per database-qualified table."""

    counts: dict[str, int] = field(default_factory=dict)
    partitions: tuple[Partition, ...] = ()

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def event_partitions(self) -> tuple[Partition, ...]:
        """Partitions of ``usage_events`` (event time), which do not run past the anchor."""
        return self.partitions


def load(
    target: PgTarget | None = None,
    config: WorldConfig | None = None,
    *,
    world: World | None = None,
) -> LoadReport:
    """Create the schema in all three databases and load the world into it.

    Idempotent: every table is dropped and recreated, so a second run leaves the same rows.
    """
    resolved = target if target is not None else target_from_environ()
    built = world if world is not None else build_world(config)
    anchor = built.config.anchor
    event_partitions = monthly_partitions(anchor, forward=0)
    all_partitions = monthly_partitions(anchor)

    counts: dict[str, int] = {}
    with psycopg.connect(resolved.dsn("crm"), autocommit=True) as conn, conn.cursor() as cur:
        counts.update(load_crm(cur, built))
        grant_read_only(cur)
        cur.execute("ANALYZE")

    with psycopg.connect(resolved.dsn("billing"), autocommit=True) as conn, conn.cursor() as cur:
        counts.update(load_billing(cur, built, event_partitions, all_partitions))
        grant_read_only(cur)
        cur.execute("ANALYZE")

    with psycopg.connect(resolved.dsn("erp"), autocommit=True) as conn, conn.cursor() as cur:
        counts.update(load_erp(cur, built))
        grant_read_only(cur)
        cur.execute("ANALYZE")

    return LoadReport(counts=counts, partitions=event_partitions)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: load the world, print the row counts per table."""
    parser = argparse.ArgumentParser(
        prog="python -m mdqm.testing.loaders.postgres",
        description="Create the crm/billing/erp schema and load the synthetic world into it.",
    )
    parser.add_argument("--small", action="store_true", help="200 customers / ~5k events")
    parser.add_argument("--host", default=None, help=f"database host (default {DEFAULT_HOST})")
    parser.add_argument(
        "--port", type=int, default=None, help=f"database port (default {DEFAULT_PORT})"
    )
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
    target = PgTarget(
        host=base.host if args.host is None else args.host,
        port=base.port if args.port is None else args.port,
        admin_user=base.admin_user,
        admin_password=base.admin_password,
        ro_user=base.ro_user,
        ro_password=base.ro_password,
    )

    report = load(target, config)
    if not args.quiet:
        sys.stdout.write(
            f"loaded seed={config.seed} customers={config.customers} "
            f"events={config.event_count} partitions={len(report.partitions)} "
            f"into {target.host}:{target.port}\n"
        )
    width = max(len(name) for name in report.counts)
    for name in sorted(report.counts):
        sys.stdout.write(f"{name:<{width}}  {report.counts[name]:>8}\n")
    sys.stdout.write(f"{'total':<{width}}  {report.total:>8}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
