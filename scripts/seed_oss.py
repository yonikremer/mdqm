"""Generate the open-source seed files for the `oss` compose profile (MariaDB + OpenSearch).

    .venv/Scripts/python.exe scripts/seed_oss.py --out docker/oss/
    .venv/Scripts/python.exe scripts/seed_oss.py --small --out docker/oss/
    .venv/Scripts/python.exe scripts/seed_oss.py --world data/world.json --out docker/oss/

The rows are never invented here: they come from the deterministic synthetic organisation in
``src/mdqm/testing/world.py`` (task D02), either rebuilt in process (``--seed``) or read back from
a world JSON that task wrote. Same seed, same bytes — the containers get the same data every run.

Four generated files, all consumed by ``docker-compose.yml`` (profile `oss`):

    docker/oss/maria/01-schema.sql              MariaDB DDL, MySQL dialect
    docker/oss/maria/02-data.sql                MariaDB INSERTs, batched
    docker/oss/opensearch/index-template.json   composable index template + the alias
    docker/oss/opensearch/usage-events.ndjson   `_bulk` action/doc pairs, monthly indices

Only this script (not ``docker/oss``) owns those paths, so the hand-written helpers next to them
(``maria/10-readonly-user.sh``, ``opensearch/load.sh``) survive a regeneration.

Type mapping, MariaDB side (a *second* dialect, not a replacement for Postgres):

    world row                  MariaDB table             note
    CrmCustomer                `crm_customers`           INT limit, CHAR(1) status
    BillingAccount             `billing_accounts`        DECIMAL(12,2), nullable limit
    UsageEvent                 `billing_usage_events`    DATETIME event + insertion time
    AccountsHistRow            `billing_accounts_hist`   ENGINE=InnoDB, updated often
    ErpCustomer                `erp_customers`           limit in thousands (DECIMAL(12,3))

Search side: one index per event month, ``mdqm-usage-events-YYYY-MM``, selected by the alias
``mdqm-usage-events``. The document id is the event id, so a reload overwrites instead of
duplicating. Documents carry both ``start_time`` (event time, the partition key) and
``insertion_time`` (ingestion time, not indexed).

Determinism: no clock, no floats, no dict iteration over unordered keys; JSON is emitted with
sorted keys and LF endings, SQL rows in the generator's own order.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from mdqm.testing.world import (
    AccountsHistRow,
    BillingAccount,
    CrmCustomer,
    ErpCustomer,
    UsageEvent,
    World,
    WorldConfig,
    build_world,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

MARIA_DATABASE = "mdqm"
"""The single MariaDB database the `oss` profile creates (MARIADB_DATABASE in compose)."""

INDEX_PREFIX = "mdqm-usage-events"
"""One index per event month is ``{INDEX_PREFIX}-YYYY-MM``; the alias is ``INDEX_PREFIX``."""

ALIAS = INDEX_PREFIX

INDEX_TEMPLATE = INDEX_PREFIX
"""Name of the composable index template the loader PUTs."""

INSERT_BATCH = 250
"""Rows per multi-row INSERT: one statement per batch keeps the dump small and the load fast."""


# --------------------------------------------------------------------------------------
# The seed: the five row collections, from a World or from a world JSON file
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OssSeed:
    """The rows both engines get, with no rendering and no I/O."""

    seed: int
    crm_customers: tuple[CrmCustomer, ...]
    accounts: tuple[BillingAccount, ...]
    usage_events: tuple[UsageEvent, ...]
    accounts_hist: tuple[AccountsHistRow, ...]
    erp_customers: tuple[ErpCustomer, ...]

    @classmethod
    def from_world(cls, world: World) -> OssSeed:
        """Take the rows straight out of a generated world."""
        return cls(
            seed=world.config.seed,
            crm_customers=world.crm.customers,
            accounts=world.billing.accounts,
            usage_events=world.billing.usage_events,
            accounts_hist=world.billing.accounts_hist,
            erp_customers=world.erp.customers,
        )

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> OssSeed:
        """Read ``render_json``/``python -m mdqm.testing.world`` output back into rows."""
        return cls(
            seed=int(document["config"]["seed"]),
            crm_customers=tuple(
                CrmCustomer(
                    cust_id=int(row["cust_id"]),
                    full_name=str(row["full_name"]),
                    status=str(row["status"]),
                    credit_limit=int(row["credit_limit"]),
                    region=str(row["region"]),
                    signup_date=_as_date(row["signup_date"]),
                )
                for row in document["crm"]["customers"]
            ),
            accounts=tuple(
                BillingAccount(
                    client_no=str(row["client_no"]),
                    state=str(row["state"]),
                    credit_limit=None if row["credit_limit"] is None else int(row["credit_limit"]),
                    region=str(row["region"]),
                    opened_date=_as_date(row["opened_date"]),
                )
                for row in document["billing"]["accounts"]
            ),
            usage_events=tuple(
                UsageEvent(
                    event_id=int(row["event_id"]),
                    client_no=str(row["client_no"]),
                    start_time=_as_datetime(row["start_time"]),
                    insertion_time=_as_datetime(row["insertion_time"]),
                    units=int(row["units"]),
                    corrects_event_id=(
                        None if row["corrects_event_id"] is None else int(row["corrects_event_id"])
                    ),
                )
                for row in document["billing"]["usage_events"]
            ),
            accounts_hist=tuple(
                AccountsHistRow(
                    hist_id=int(row["hist_id"]),
                    client_no=str(row["client_no"]),
                    state=str(row["state"]),
                    credit_limit=int(row["credit_limit"]),
                    updated_at=_as_datetime(row["updated_at"]),
                )
                for row in document["billing"]["accounts_hist"]
            ),
            erp_customers=tuple(
                ErpCustomer(
                    cust_no=int(row["cust_no"]),
                    status_code=int(row["status_code"]),
                    limit_k=int(row["limit_k"]),
                    name=str(row["name"]),
                )
                for row in document["erp"]["customers"]
            ),
        )

    def counts(self) -> dict[str, int]:
        """Row count per collection, for progress output and for the tests."""
        return {
            "crm_customers": len(self.crm_customers),
            "billing_accounts": len(self.accounts),
            "billing_usage_events": len(self.usage_events),
            "billing_accounts_hist": len(self.accounts_hist),
            "erp_customers": len(self.erp_customers),
        }


def _as_date(value: Any) -> date:
    return date.fromisoformat(str(value))


def _as_datetime(value: Any) -> datetime:
    """Parse the ISO-8601 form ``world.render_json`` writes; naive input is taken as UTC."""
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


# --------------------------------------------------------------------------------------
# MariaDB rendering (MySQL dialect: backticks, DATETIME, DECIMAL, ENGINE=InnoDB)
# --------------------------------------------------------------------------------------

SCHEMA_SQL = """\
-- Generated by scripts/seed_oss.py from src/mdqm/testing/world.py. Do not edit by hand:
-- regenerate with `python scripts/seed_oss.py --out docker/oss/`.
--
-- MariaDB seed for the `oss` compose profile (a second SQL dialect, NOT Oracle and NOT T-SQL).
-- The world's three datasets live in ONE MariaDB database: a single server is enough to prove
-- that the compiler and the loader can speak MySQL, and one database keeps the profile small.
-- All timestamps are UTC; MariaDB DATETIME carries no zone.
--
-- Read by /docker-entrypoint-initdb.d on the first start of an empty volume, so `mdqm` is
-- already selected and the tables below are created inside it.

CREATE TABLE IF NOT EXISTS `crm_customers` (
  `cust_id` INT NOT NULL,
  `full_name` VARCHAR(80) NOT NULL,
  `status` CHAR(1) NOT NULL,
  `credit_limit` INT NOT NULL,
  `region` VARCHAR(16) NOT NULL,
  `signup_date` DATE NOT NULL,
  PRIMARY KEY (`cust_id`)
);

CREATE TABLE IF NOT EXISTS `billing_accounts` (
  `client_no` VARCHAR(10) NOT NULL,
  `state` VARCHAR(16) NOT NULL,
  `credit_limit` DECIMAL(12,2) NULL,
  `region` VARCHAR(16) NOT NULL,
  `opened_date` DATE NOT NULL,
  PRIMARY KEY (`client_no`)
);

CREATE TABLE IF NOT EXISTS `billing_usage_events` (
  `event_id` BIGINT NOT NULL,
  `client_no` VARCHAR(10) NOT NULL,
  `start_time` DATETIME NOT NULL,
  `insertion_time` DATETIME NOT NULL,
  `units` INT NOT NULL,
  `corrects_event_id` BIGINT NULL,
  PRIMARY KEY (`event_id`),
  KEY `ix_usage_events_start_time` (`start_time`),
  KEY `ix_usage_events_insertion_time` (`insertion_time`)
);

CREATE TABLE IF NOT EXISTS `billing_accounts_hist` (
  `hist_id` BIGINT NOT NULL,
  `client_no` VARCHAR(10) NOT NULL,
  `state` VARCHAR(16) NOT NULL,
  `credit_limit` DECIMAL(12,2) NOT NULL,
  `updated_at` DATETIME NOT NULL,
  PRIMARY KEY (`hist_id`),
  KEY `ix_accounts_hist_client_updated` (`client_no`, `updated_at`)
) ENGINE=InnoDB;
-- ENGINE=InnoDB: this is the mutable_critical table (many revisions per account), so it needs
-- real row locking and a transaction log, not a lock-free read-only format.

CREATE TABLE IF NOT EXISTS `erp_customers` (
  `cust_no` INT NOT NULL,
  `status_code` INT NOT NULL,
  `limit_k` DECIMAL(12,3) NOT NULL,
  `name` VARCHAR(80) NOT NULL,
  KEY `ix_erp_customers_cust_no` (`cust_no`)
);
-- No primary key on purpose: the world plants duplicate `cust_no` rows so a naive AVG
-- double counts them.
"""


def _sql_str(value: str) -> str:
    r"""Quote a MariaDB string literal: ``\`` and ``'`` are escaped, then doubled."""
    return "'" + value.replace("\\", "\\\\").replace("'", "''") + "'"


def _sql_opt_int(value: int | None) -> str:
    return "NULL" if value is None else str(value)


def _sql_opt_datetime(value: datetime | None) -> str:
    return "NULL" if value is None else _sql_str(value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S"))


def _sql_date(value: date) -> str:
    return _sql_str(value.isoformat())


def _insert_statements(
    table: str, columns: Sequence[str], rows: Iterable[Sequence[str]]
) -> list[str]:
    """One multi-row ``INSERT`` per batch of ``INSERT_BATCH`` already-rendered rows."""
    header = f"INSERT INTO `{table}` ({', '.join(f'`{column}`' for column in columns)}) VALUES"
    statements: list[str] = []
    batch: list[str] = []
    for row in rows:
        batch.append("(" + ", ".join(row) + ")")
        if len(batch) == INSERT_BATCH:
            statements.append(header + "\n" + ",\n".join(batch) + ";\n")
            batch = []
    if batch:
        statements.append(header + "\n" + ",\n".join(batch) + ";\n")
    return statements


def render_data_sql(seed: OssSeed) -> str:
    """The ``INSERT`` half of the MariaDB seed, in the world's own row order."""
    parts = [
        "-- Generated by scripts/seed_oss.py from src/mdqm/testing/world.py. Do not edit by hand.\n"
        f"-- seed {seed.seed}; all DATETIME values are UTC; rows are in generator order.\n\n"
    ]
    parts.extend(
        _insert_statements(
            "crm_customers",
            ("cust_id", "full_name", "status", "credit_limit", "region", "signup_date"),
            (
                (
                    str(row.cust_id),
                    _sql_str(row.full_name),
                    _sql_str(row.status),
                    str(row.credit_limit),
                    _sql_str(row.region),
                    _sql_date(row.signup_date),
                )
                for row in seed.crm_customers
            ),
        )
    )
    parts.extend(
        _insert_statements(
            "billing_accounts",
            ("client_no", "state", "credit_limit", "region", "opened_date"),
            (
                (
                    _sql_str(row.client_no),
                    _sql_str(row.state),
                    _sql_opt_int(row.credit_limit),
                    _sql_str(row.region),
                    _sql_date(row.opened_date),
                )
                for row in seed.accounts
            ),
        )
    )
    parts.extend(
        _insert_statements(
            "billing_usage_events",
            ("event_id", "client_no", "start_time", "insertion_time", "units", "corrects_event_id"),
            (
                (
                    str(row.event_id),
                    _sql_str(row.client_no),
                    _sql_opt_datetime(row.start_time),
                    _sql_opt_datetime(row.insertion_time),
                    str(row.units),
                    _sql_opt_int(row.corrects_event_id),
                )
                for row in seed.usage_events
            ),
        )
    )
    parts.extend(
        _insert_statements(
            "billing_accounts_hist",
            ("hist_id", "client_no", "state", "credit_limit", "updated_at"),
            (
                (
                    str(row.hist_id),
                    _sql_str(row.client_no),
                    _sql_str(row.state),
                    str(row.credit_limit),
                    _sql_opt_datetime(row.updated_at),
                )
                for row in seed.accounts_hist
            ),
        )
    )
    parts.extend(
        _insert_statements(
            "erp_customers",
            ("cust_no", "status_code", "limit_k", "name"),
            (
                (
                    str(row.cust_no),
                    str(row.status_code),
                    str(row.limit_k),
                    _sql_str(row.name),
                )
                for row in seed.erp_customers
            ),
        )
    )
    return "".join(parts)


# --------------------------------------------------------------------------------------
# OpenSearch rendering: one index per event month, an alias, `_bulk` NDJSON
# --------------------------------------------------------------------------------------


def index_for(moment: datetime) -> str:
    """Index name for an event: ``mdqm-usage-events-YYYY-MM`` from its UTC event time."""
    return f"{INDEX_PREFIX}-{moment.astimezone(UTC).strftime('%Y-%m')}"


def render_index_template(seed: OssSeed) -> str:
    """The composable index template: settings + mappings + the alias every index gets."""
    template: dict[str, Any] = {
        "index_patterns": [f"{INDEX_PREFIX}-*"],
        "priority": 100,
        "_meta": {
            "generator": "scripts/seed_oss.py",
            "source": "src/mdqm/testing/world.py",
            "seed": seed.seed,
            "note": "one index per event month; query through the alias, never a bare index name",
        },
        "template": {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index.refresh_interval": "1s",
            },
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "event_id": {"type": "long"},
                    "client_no": {"type": "keyword"},
                    "start_time": {"type": "date"},
                    "insertion_time": {"type": "date"},
                    "units": {"type": "integer"},
                    "corrects_event_id": {"type": "long"},
                },
            },
            "aliases": {ALIAS: {"is_write_index": False}},
        },
    }
    return json.dumps(template, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def render_bulk_ndjson(seed: OssSeed) -> str:
    """``_bulk`` body: action line then document line, grouped by index, ordered by event id."""
    lines: list[str] = []
    ordered = sorted(seed.usage_events, key=lambda row: (index_for(row.start_time), row.event_id))
    for row in ordered:
        action = {"index": {"_index": index_for(row.start_time), "_id": str(row.event_id)}}
        document: dict[str, Any] = {
            "event_id": row.event_id,
            "client_no": row.client_no,
            "start_time": row.start_time.astimezone(UTC).isoformat(),
            "insertion_time": row.insertion_time.astimezone(UTC).isoformat(),
            "units": row.units,
            "corrects_event_id": row.corrects_event_id,
        }
        lines.append(json.dumps(action, sort_keys=True, ensure_ascii=False))
        lines.append(json.dumps(document, sort_keys=True, ensure_ascii=False))
    return "\n".join(lines) + "\n" if lines else ""


def months(seed: OssSeed) -> list[str]:
    """The monthly indices the bulk load creates, sorted."""
    return sorted({index_for(row.start_time) for row in seed.usage_events})


# --------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------


def write_all(seed: OssSeed, out_dir: Path) -> list[Path]:
    """Write the four generated files under ``out_dir`` (``docker/oss`` by default)."""
    maria_dir = out_dir / "maria"
    search_dir = out_dir / "opensearch"
    maria_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)
    written = [
        (maria_dir / "01-schema.sql", SCHEMA_SQL),
        (maria_dir / "02-data.sql", render_data_sql(seed)),
        (search_dir / "index-template.json", render_index_template(seed)),
        (search_dir / "usage-events.ndjson", render_bulk_ndjson(seed)),
    ]
    for path, text in written:
        path.write_text(text, encoding="utf-8", newline="\n")
    return [path for path, _ in written]


def load_seed(seed_number: int, small: bool, world_path: Path | None) -> OssSeed:
    """A world JSON when one is given, otherwise a world built in process from the seed."""
    if world_path is not None:
        document: dict[str, Any] = json.loads(world_path.read_text(encoding="utf-8"))
        return OssSeed.from_json(document)
    config = WorldConfig.small(seed=seed_number) if small else WorldConfig(seed=seed_number)
    return OssSeed.from_world(build_world(config))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: write the MariaDB and OpenSearch seed files, then report what was written."""
    parser = argparse.ArgumentParser(
        prog="python scripts/seed_oss.py",
        description="Generate the open-source (`oss` profile) MariaDB + OpenSearch seeds.",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docker" / "oss", help="output dir")
    parser.add_argument("--seed", type=int, default=1, help="world seed (default: 1)")
    parser.add_argument("--small", action="store_true", help="200 customers / 5k events")
    parser.add_argument(
        "--world",
        type=Path,
        default=None,
        help="read a world JSON written by `python -m mdqm.testing.world` instead of building one",
    )
    args = parser.parse_args(argv)

    try:
        seed = load_seed(args.seed, args.small, args.world)
    except (OSError, KeyError, ValueError) as error:
        sys.stderr.write(f"cannot read the world: {error}\n")
        return 1

    written = write_all(seed, args.out)
    counts = seed.counts()
    sys.stdout.write(
        "oss seed: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
        + f", indices={len(months(seed))}, seed {seed.seed}\n"
    )
    for path in written:
        sys.stdout.write(f"  {path} ({path.stat().st_size} bytes)\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
