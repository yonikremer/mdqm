status: partial
task: OSS-mocks (owner request; NOT in TASKS.md)
agent: pi-build-agent   branch: feature/oss-stack (commit landed on main)   commit: ae975e9 + progress commit
tests run:
  /.venv/Scripts/python.exe -m pytest tests/unit/test_seed_oss.py -q      -> 42 passed
  /.venv/Scripts/python.exe -m pytest tests/unit/test_dev_env.py -q      -> 28 passed
  /.venv/Scripts/python.exe scripts/check.py  (clean worktree at ae975e9) -> all checks passed
  /.venv/Scripts/python.exe scripts/check.py  (main worktree)            -> all checks passed
  .venv/Scripts/python.exe scripts/seed_oss.py --small --seed 1 --out docker/oss/
      -> crm_customers=200, billing_accounts=200, billing_usage_events=5001,
         billing_accounts_hist=401, erp_customers=201, indices=8
  .venv/Scripts/python.exe scripts/seed_oss.py --seed 1 --out <tmp>
      -> 2000 customers / 50006 events, 8 indices (used to check the size trade-off)
  command -v docker -> absent, twice (start and end of the task)

## Summary

Added an open-source-only local test stack next to the existing proprietary one, without removing
or retargeting Oracle, SQL Server or Elasticsearch. New Compose profile `oss` starts exactly
Postgres + MariaDB + OpenSearch + the mock API. MariaDB 11.4.4 (GPL) is a second SQL dialect;
OpenSearch 2.17.1 (Apache-2.0) is the OSS stand-in for Elasticsearch, speaking the same REST API the
repo's httpx-based adapter already uses. `scripts/seed_oss.py` renders the D02 synthetic world into
MariaDB DDL/INSERTs (backticks, DATETIME, DECIMAL, ENGINE=InnoDB on the mutable hist table) and into
an OpenSearch index template (monthly indices + alias) plus `_bulk` NDJSON, all deterministic and
committed so the init hooks have files to consume. `docker/oss/maria/10-readonly-user.sh` creates
`mdqm_ro` with SELECT only; `docker/oss/opensearch/load.sh` is the one-shot loader.

## Decisions

1. **Committed seeds are the `--small` world** (200 customers / ~5k events): 1.6 MB total instead of
   15 MB for the full world. Regenerate at full size with one command, documented in dev-env.md.
   A test regenerates the committed files and fails if they drift, so they are never hand-edited.
2. **One MariaDB database `mdqm`**, tables prefixed `crm_`/`billing_`/`erp_`: the task asked for one
   database, and a single server is enough to prove the second dialect.
3. **`accounts_hist` is the explicit `ENGINE=InnoDB` table** (task wording: the mutable/hist table);
   the other four keep MariaDB's default. `erp_customers` deliberately has no primary key, because
   the world plants duplicate `cust_no` rows.
4. **OpenSearch monthly indices are created by the `_bulk` load under the index template**, and the
   alias ships inside the composable template (`template.aliases`), so no extra index-list file is
   needed and a reload overwrites by `_id = event_id` instead of duplicating.
5. **The read-only MariaDB account reuses `MDQM_RO_PASSWORD`**, the same variable the Postgres and
   Oracle services use, so one throwaway dev password covers the read-only role everywhere.
6. **`pg` and `api` now belong to two profiles** (`["pg","oss"]`, `["api","oss"]`), which the Compose
   spec allows; assumed Compose file format = the spec (no top-level `version:` key) with Compose
   v2.20+, the same floor `dev.py` already requires for `up --wait`.
7. **`opensearch-init` is a one-shot `curlimages/curl:8.11.1` service** with no ports and
   `restart: "no"`; it waits for the node itself, so it also works via `run --rm`.

## Deviations

- `scripts/dev.py` gained one line (`PROFILES` now includes `oss`) so `python scripts/dev.py up oss`
  works; that file belongs to D01. Smallest possible edit, mentioned here as AGENTS.md requires.
- `tests/unit/test_dev_env.py` (D01's test) had to change: it asserted "one service = one profile"
  and an exact service set, which multi-profile membership and the three new services contradict.
  The assertions were **extended, not weakened**: a `ServiceSpec` table now declares each service's
  exact profile list, image prefix, container port and whether it has a healthcheck; `pg`/`api` two
  profiles are asserted explicitly; one-shot services are now required to publish no ports and to
  carry `restart: "no"`. `.env.example` gained the MariaDB password and two ports.
- The lens YAML rule that wants ≤80 columns in `docker-compose.yml` is **not this repo's
  convention**: ruff is configured with `line-length = 100` and lints Python only, and the same rule
  already fires on ~28 untouched lines of that file on `main` (plus hundreds in
  `tests/eval/questions.yaml`, `tests/data/semantic/*.yaml`). Full compliance is impossible here
  anyway: the existing `CMD-SHELL` healthchecks are 112 and 127 characters and cannot be wrapped
  without changing those services, and my owner restricted this file to additive changes. New lines
  were kept within the project's own 100-column limit; nothing was rewrapped.

## Dependencies

none. New images only: `mariadb:11.4.4` (GPL), `opensearchproject/opensearch:2.17.1` (Apache-2.0),
`curlimages/curl:8.11.1` (curl licence) — all pinned, no `latest`. No new Python package: the
generator is stdlib + the repo's own `mdqm.testing.world`.

## Not verified

Docker never appeared on PATH (checked at the start and at the end), so every runtime claim below is
unverified and no container was started:

- `docker compose --profile oss up -d` / `python scripts/dev.py up oss` — never executed.
- MariaDB: that the image starts, that `/docker-entrypoint-initdb.d` runs `01-schema.sql`,
  `02-data.sql` and `10-readonly-user.sh` in order, that the DDL is accepted by MariaDB 11.4
  (sqlglot's MySQL parser is the only check), that the batched INSERTs load, and that `mdqm_ro` can
  SELECT but not write.
- OpenSearch: that `DISABLE_SECURITY_PLUGIN`/`bootstrap.memory_lock`/512 MB heap start cleanly, that
  the healthcheck's `curl` exists in the image, that the composable template is accepted with the
  alias, that `_bulk` creates the 8 monthly indices, and that the alias resolves.
- `opensearch-init`: that `curlimages/curl` has `/bin/sh`, that `entrypoint: ["sh", ...]` runs the
  loader, and that the log prints indices.
- Host ports 53306/59201 answering.
- The compose file was only checked for *syntax via a YAML parser* here; `docker compose config` was
  never run, so Compose's own validation (multi-profile membership included) is unverified.

Green-and-verified: generation, determinism, MySQL dialect parsing, NDJSON shape, compose YAML
structure, `scripts/check.py` on a clean worktree of the merged commit.

## Plan issues

- **This work is NOT in TASKS.md** — it came from an owner request ("create mock databases and APIs
  locally for testing, open-source only"). There is no task ID, no `owns` list and no dependency
  entry for it, so `next_tasks.py` cannot claim/release it and the "one task at a time" bookkeeping
  does not apply. Consider adding a real task for the runtime verification above.
- **Shared working tree was clobbered mid-task by parallel sessions**: while this task ran, another
  session moved `feature/oss-stack` back to an old commit, checked out `fix/security-review`, and
  left uncommitted edits in `README.md`, `src/mdqm/adapters/openapi.py`, `src/mdqm/sql/guard.py` and
  others. The OSS commit (`ae975e9`) survived as a parent of `af99202`; it was merged into `main`
  with `git merge --ff-only ae975e9` in a throwaway worktree, because a plain `git checkout main`
  in the shared tree was refused (other sessions' uncommitted changes). This is exactly the hazard
  AGENTS.md section 8 warns about — branch refs on a shared tree are not private.
- `scripts/seed_oss.py` writes files under `docker/oss/`, which means a stale fixture is possible if
  someone changes the world generator and forgets to regenerate. `test_committed_fixtures_match_the_generator`
  is the guard; if it ever fails, rerun the documented command rather than editing the SQL.
