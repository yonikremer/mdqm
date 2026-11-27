# Development environment (`docker-compose.yml`)

Read this first. It covers the local services, the profiles, the ports and the throwaway
credentials. `AGENTS.md` section 6 has the one-line summary; this file has the details.

Nothing here is a production configuration. Every service binds to `127.0.0.1` only, on a
non-default host port, with a throwaway password from `.env`.

## 1. Install a container runtime

Windows and macOS: **Docker Desktop** with the **WSL2 backend** and **Linux containers**
(the Windows-container mode cannot run any of these images).

1. Install Docker Desktop from <https://docs.docker.com/desktop/>.
2. Start it and wait until it reports "Engine running".
3. Check Compose is v2.20 or newer — `dev.py up` uses `docker compose up --wait`:

```powershell
docker compose version
```

If `docker` is not installed, `python scripts/dev.py` says so and exits 2. Everything that does
not need a database still works: `python scripts/check.py` (ruff, mypy, unit tests).

## 2. Create `.env`

```powershell
python scripts/dev.py env
```

That copies `.env.example` to `.env` once. `.env` is git-ignored; `.env.example` holds the only
credentials that may ever be in this repo, and all of them are throwaway dev passwords.

Compose needs those values: a missing password makes `docker compose` fail loudly instead of
starting a server with a blank one.

## 3. Profiles: start only what your task needs

Every service sits behind a Compose profile, so a task starts one engine, not five.

| Profile | Service | Image (pinned) | Host port | RAM | Used by |
|---|---|---|---|---|---|
| `pg` | PostgreSQL 16 + `pg_stat_statements`, databases `crm`/`billing`/`erp`, read-only role `mdqm_ro` | `postgres:16` | `127.0.0.1:55432` | ~1 GB | the MVP: most tasks, all Postgres integration tests, D02/D03, E03 |
| `oracle` | Oracle Free, Thin mode, slow first start (minutes) | `gvenzl/oracle-free:23-slim-faststart` | `127.0.0.1:51521` | ~3.5 GB | interval partitioning, `V$SQL`, PSTART/PSTOP (A08, probe) |
| `mssql` | SQL Server 2022 Developer | `mcr.microsoft.com/mssql/server:2022-CU14-ubuntu-22.04` | `127.0.0.1:51433` | ~2 GB | partition function/scheme, Query Store, plan XML (A08) |
| `es` | Elasticsearch 8.x, single node, security off, 1 GB heap | `docker.elastic.co/elasticsearch/elasticsearch:8.15.3` | `127.0.0.1:59200` | ~2 GB | monthly indices + alias, search-only adapter |
| `api` | the mock internal API from D04 (`src/mdqm/testing/mock_api.py`) | built here: `mdqm/mock-api:dev` | `127.0.0.1:58000` | ~0.2 GB | OpenAPI adapter tests that need a real port |
| `oss` | the whole open-source stack at once: `pg` + `mariadb` + `opensearch` + `api` (see section 7) | the images of those services | `55432`, `53306`, `59201`, `58000` | ~3 GB | anyone with no Oracle/SQL Server/Elasticsearch licence |

`pg` and `api` belong to **two** profiles (`pg`/`oss` and `api`/`oss`), which the Compose spec allows:
`--profile oss` therefore starts them together with MariaDB and OpenSearch and nothing else.

**Memory.** Oracle + SQL Server + Elasticsearch together need roughly **10–12 GB** and a slow first
start (Oracle alone can take several minutes). Do not bring them all up unless the task needs them:
`pg` plus `api` is under 1.5 GB and is what almost every task needs. On Windows, Docker Desktop's
WSL2 VM gets a share of RAM; if you do run the heavy three, raise that share in `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=12GB
```

then `wsl --shutdown` and restart Docker Desktop.

**Disk.** The Oracle image alone is a few GB compressed; `python scripts/dev.py down` reclaims the
data volumes, and `docker image prune` (not `docker system prune`) reclaims old builds.

## 4. Commands

```powershell
python scripts/dev.py up pg            # start PostgreSQL; exits 0 only once it is healthy
python scripts/dev.py up pg api        # several profiles at once
python scripts/dev.py status           # state of every service
python scripts/dev.py status pg        # state of one profile
python scripts/dev.py down             # stop everything and REMOVE the data volumes
python scripts/dev.py env              # create .env from .env.example if missing
```

`up` maps to `docker compose --profile <p> up --detach --wait`: it blocks until every healthcheck
passes and exits non-zero when a service never becomes healthy, so a green exit means "ready", not
"started". `down` maps to `docker compose down --volumes --remove-orphans`, so the next `up` runs
the init scripts again — which is what you want after changing them.

Logs and one-off shells (there is no wrapper for these; run them from the repo root):

```powershell
docker compose --profile pg logs -f pg
docker compose --profile pg exec pg psql -U postgres -d crm
```

Ports and a single wait, when Compose is not the thing you are testing:

```powershell
python scripts/wait_for.py tcp 127.0.0.1:55432 --timeout 120
python scripts/wait_for.py http http://127.0.0.1:58000/health --timeout 60
```

## 5. What the `pg` profile gives you

Created by `docker/pg/init/01-init.sh` on the first start of an empty volume:

- databases `crm`, `billing` and `erp` — the same data in three storage forms, which is how
  cross-source unit/rounding conversions are tested with Postgres alone (see `AGENTS.md` section 6);
- the read-only role `mdqm_ro`, with `default_transaction_read_only = on` and `SELECT` on all
  current and future tables in `public` (the read-only account is layer one of three);
- `pg_stat_statements` created in all three databases (the server is started with
  `-c shared_preload_libraries=pg_stat_statements -c pg_stat_statements.track=all`), which is where
  the query-mining stage reads its stats from, and `pg_read_all_stats` granted to `mdqm_ro` so the
  miner sees every role's statement text (the view masks other roles' text as
  `<insufficient privilege>` without it, and mining would silently see only our own reads);
- `track_io_timing=on`, so I/O timing is available to the planner work.

A volume created before the `pg_read_all_stats` grant needs the one line by hand:

```powershell
docker compose --profile pg exec pg psql -U postgres -d crm -c "GRANT pg_read_all_stats TO mdqm_ro"
```

Verify it yourself after the first `up pg`:

```powershell
docker compose --profile pg exec pg psql -U postgres -d crm -c "SELECT count(*) FROM pg_stat_statements"
docker compose --profile pg exec pg psql -U postgres -l
docker compose --profile pg exec pg psql -h 127.0.0.1 -U mdqm_ro -d crm -c "SELECT current_user"
```

The last one should answer as `mdqm_ro`; a write attempt through that role must fail, and that is
deliberate.

## 6. Integration tests

`tests/integration` needs the services and is marked `integration`. Without them it **skips**;
with `--require-services` it **fails** instead, which is how acceptance criteria are checked:

```powershell
python scripts/dev.py up pg
.venv/Scripts/python.exe -m pytest tests/integration -q --require-services
```

A skipped test is not a pass. If a service is up but the test skips anyway, check the port in the
error message against the table above.

## 7. The open-source-only stack (`oss` profile)

Everything in this profile is free and open source — there is no Oracle, no SQL Server and no
Elasticsearch in it — so it is the stack to use when you have no licence for the proprietary
engines. It adds to the profiles above; it never replaces them.

```powershell
docker compose --profile oss up -d
python scripts/dev.py up oss          # same thing, and it waits for healthchecks
```

That starts exactly five services:

| Service | Image (pinned, open source) | Host port | What it is |
|---|---|---|---|
| `pg` | `postgres:16` (PostgreSQL licence) | `127.0.0.1:55432` | the same `pg` profile as above: `crm`/`billing`/`erp` + `mdqm_ro` |
| `mariadb` | `mariadb:11.4.4` (GPL) | `127.0.0.1:53306` | one database `mdqm`, seeded, plus the read-only account `mdqm_ro` |
| `opensearch` | `opensearchproject/opensearch:2.17.1` (Apache-2.0) | `127.0.0.1:59201` | single node, security plugin off, 512 MB heap |
| `opensearch-init` | `curlimages/curl:8.11.1` | none | one-shot: loads the seed, prints what exists, exits |
| `api` | `mdqm/mock-api:dev` (built from this repo) | `127.0.0.1:58000` | the D04 mock OpenAPI service |

### What OpenSearch stands in for, and what it does not prove

OpenSearch serves the **Elasticsearch REST API** (`_bulk`, `_index_template`, `_cat/*`, `_search`),
and the repo's search adapter talks plain HTTP through `httpx` rather than an engine-specific
client. That is why the `es` profile can be swapped for `opensearch`: same requests, same
responses, open-source licence.

What this **does** prove: the adapter's request building, NDJSON bulk framing, index-template
handling, monthly-index-plus-alias addressing, and search-only access all work against a real
service.

What it does **not** prove: Elasticsearch-version-specific behaviour. OpenSearch is a fork of
Elasticsearch 7.10, so 8.x features (and anything the proprietary distribution adds) are not
exercised, and a bug that only appears on Elasticsearch 8 will not show up here. If a task
depends on Elasticsearch 8 semantics, use the `es` profile.

### What MariaDB stands in for, and what it does not prove

MariaDB gives the compiler and the loaders a **second SQL dialect**: backtick identifiers,
`DATETIME`, `DECIMAL`, `ENGINE=InnoDB`. It has the same logical datasets as `pg` in one database
(`crm_customers`, `billing_accounts`, `billing_usage_events`, `billing_accounts_hist`,
`erp_customers`), so cross-dialect translation is testable without a licence.

It is **not** Oracle and **not** T-SQL: no `VARCHAR2`, no `NUMBER`, no interval partitioning, no
`V$SQL`, no partition function/scheme, no Query Store, no plan XML. Tasks A08 and the
partition/plan work still need the `oracle` and `mssql` profiles.

### The seeds

The data is generated, never hand-written, from the same synthetic world as the `pg` profile
(`src/mdqm/testing/world.py`, task D02), so the ground truth stays consistent across engines:

```powershell
.venv/Scripts/python.exe scripts/seed_oss.py --small --out docker/oss/   # what is committed
.venv/Scripts/python.exe scripts/seed_oss.py --out docker/oss/           # full 2000-customer world
.venv/Scripts/python.exe scripts/seed_oss.py --world data/world.json --out docker/oss/
```

That writes four files, and they are committed so the compose init hooks have something to read:

| File | Consumed by |
|---|---|
| `docker/oss/maria/01-schema.sql` | MariaDB `/docker-entrypoint-initdb.d` (DDL) |
| `docker/oss/maria/02-data.sql` | MariaDB `/docker-entrypoint-initdb.d` (batched INSERTs) |
| `docker/oss/opensearch/index-template.json` | `opensearch-init`, PUT to `_index_template/mdqm-usage-events` |
| `docker/oss/opensearch/usage-events.ndjson` | `opensearch-init`, POSTed to `_bulk` |

Two hand-written helpers sit next to them and are never overwritten by the generator:
`docker/oss/maria/10-readonly-user.sh` (creates `mdqm_ro` with `SELECT` only, password from
`MDQM_RO_PASSWORD`) and `docker/oss/opensearch/load.sh` (the loader the one-shot service runs).

Same seed ⇒ byte-identical files: `tests/unit/test_seed_oss.py` regenerates them and fails if the
committed copies drift, so never edit them by hand.

**Getting the seeds applied.** Both engines seed themselves from a *volume*, so the seeds run once
per empty volume:

- MariaDB: on first start of an empty `mariadb-data` volume it runs `01-schema.sql`, `02-data.sql`,
  then `10-readonly-user.sh`. Change a seed file ⇒ `python scripts/dev.py down` (removes volumes),
  then `up oss` again.
- OpenSearch: `opensearch-init` waits for the node to be healthy, PUTs the template, bulk-loads,
  then prints the indices. It runs on every `up oss`, and a document id is its event id, so
  reloading overwrites instead of duplicating. Re-run it alone with:

```powershell
docker compose --profile oss up -d opensearch
docker compose --profile oss run --rm opensearch-init
```

Verify by hand (these are the only proofs that the seeds landed):

```powershell
docker compose --profile oss exec mariadb mariadb -u mdqm_ro -p"$env:MDQM_RO_PASSWORD" -D mdqm -e "SELECT COUNT(*) FROM billing_usage_events"
docker compose --profile oss logs opensearch-init
python scripts/wait_for.py http http://127.0.0.1:59201/_cluster/health --timeout 120
```

The read-only account must be able to `SELECT` and must fail on `INSERT` or `UPDATE` — that is the
same three-layer read-only rule as Postgres, not a convenience.

### Still there for licence holders

The `pg`, `oracle`, `mssql`, `es` and `api` profiles are unchanged and keep working exactly as
before. `oss` does not retarget or remove any of them.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `docker was not found on PATH` | No container runtime. Install Docker Desktop (section 1). Nothing else in the repo needs it. |
| `required variable MDQM_PG_PASSWORD is missing a value` | `.env` does not exist. Run `python scripts/dev.py env`. |
| `up` exits non-zero, one service `unhealthy` | Read its log: `docker compose --profile pg logs pg`. First start of Oracle/Elasticsearch is slow; check `docker compose ps` again after a minute. |
| `port is already allocated` | A local engine uses the default port. We already shifted ours (55432/51521/51433/59200/58000); override with `MDQM_*_PORT` in `.env`. |
| Elasticsearch restarts with `max virtual memory areas vm.max_map_count is too low` | WSL2 host setting; raise `vm.max_map_count` in the WSL2 VM. Only the `es` profile hits this. |
| `sqlcmd: command not found` in the mssql container | Expected: the 2022 image ships without mssql-tools. Use the `pymssql`/`mssql-python` client from Python, or the healthcheck's `/dev/tcp` trick. |
| `up` hangs on `api` | The image build needs network access once, on the developer machine. If pip cannot reach the index, the build fails; the service itself never contacts the internet. |
| Want a clean slate | `python scripts/dev.py down` (removes volumes), then `up` again. |

## תקציר (Hebrew)

- מריצים רק את מה שצריך: `python scripts/dev.py up pg` — מסד Postgres אחד עם שלושה בסיסי נתונים
  (`crm`/`billing`/`erp`) ומשתמש קריאה-בלבד `mdqm_ro`.
- כל הסיסמאות הן סיסמאות פיתוח זמניות ב-`.env` (העותק של `.env.example`), ואף אחת מהן אינה סוד אמיתי.
- כל הפורטים מאזינים ל-`127.0.0.1` בלבד ובפורטים לא שגרתיים, כדי לא להתנגש בהתקנות מקומיות.
- Oracle + SQL Server + Elasticsearch יחד דורשים כ-10–12 GB RAM והרצה ראשונה איטית.
- `python scripts/dev.py down` מוחק גם את נפחי הנתונים.
- סטאק בקוד פתוח בלבד (ללא Oracle/SQL Server/Elasticsearch): `python scripts/dev.py up oss` —
  Postgres + MariaDB + OpenSearch + ה-API המדומה, בפורטים 55432/53306/59201/58000.
- OpenSearch מחליף את Elasticsearch כי הוא מדבר את אותו REST API, אבל הוא לא מוכיח התנהגות של
  Elasticsearch 8. MariaDB הוא דיאלקט SQL שני — לא Oracle ולא T-SQL.
