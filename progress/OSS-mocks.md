status: done
task: OSS-mocks (owner request; NOT in TASKS.md)
agent: claude   branch: feature/oss-stack   commit: ae975e9
tests run:
  `.venv/Scripts/python.exe -m pytest tests/unit/test_seed_oss.py tests/unit/test_dev_env.py -q`
    -> green (counts change; run the command)
  `.venv/Scripts/python.exe scripts/check.py` -> ruff + `mypy --strict` on `src` + all unit tests, green

## Summary
An open-source-only local stack, added ALONGSIDE the proprietary services rather than replacing them.
`docker compose --profile oss up -d` brings up exactly Postgres + MariaDB + OpenSearch + the mock
OpenAPI API, and nothing that needs a licence.

- `docker-compose.yml`: new `mariadb` and `opensearch` services (pinned tags, healthchecks, named
  volumes); the existing `pg` and `api` services also joined profile `oss`, so one profile is the whole
  open-source stack. Every credential still comes from `.env` with the fail-loudly `:?` form.
- `scripts/seed_oss.py` generates the second-dialect and search data FROM the D02 world generator, so
  the ground truth in E01 stays true: MySQL-dialect DDL + INSERTs for MariaDB
  (`docker/oss/maria/`), an index template, monthly indices, an alias and `_bulk` NDJSON for
  OpenSearch (`docker/oss/opensearch/`). Deterministic: same seed, byte-identical output.
- OpenSearch is the open-source stand-in for Elasticsearch and speaks the same REST API, so the
  httpx search path can be exercised without a licence. MariaDB is a second SQL dialect, but it is
  NOT Oracle and NOT T-SQL — that distinction is written in `docs/dev-env.md` rather than implied.
- The emitted fixtures are committed so `up` is self-contained, and a test regenerates them and
  asserts byte equality, so hand-edits or generator drift fail loudly.

## Decisions
- Additive, not replacing: the Oracle/SQL Server/Elasticsearch profiles and their loaders stay, because
  those are the product's actual targets; the OSS stack is what you run without licences.
- Fixtures are committed (not generated at container start) because `/docker-entrypoint-initdb.d` and
  the OpenSearch init service consume files at startup; the drift test is what keeps them honest.
- `pytest tests/unit` never needs Docker: the seed tests assert file content and determinism only.

## Deviations
none from the owner's request. Nothing in TASKS.md covers this; recorded here so the ledger is complete.

## Dependencies
none added. `scripts/seed_oss.py` uses stdlib + the existing `mdqm.testing.world` generator.

## Not verified
At commit time this machine had **no container runtime**, so the stack was never started. That gap is
now closed; see the section below.

## Verified after the merge, with a real container runtime
The stack was brought up with Docker Desktop 29.9.3 / Compose v5.7.1 after the merge:

- `docker/oss/maria/10-readonly-user.sh` needed `MDQM_RO_PASSWORD`, which the `mariadb` service never
  passed, so the container exited 2 mid-init. Fixed in `docker-compose.yml` (commit `a716d71`) and the
  stack then reached 5/5 healthy: `pg`, `mariadb`, `opensearch`, `opensearch-init`, `api`.
- MariaDB row counts after the seed: `crm_customers` 200, `billing_accounts` 200,
  `billing_usage_events` 5001, `erp_customers` 201 (the planted duplicates survive),
  `billing_accounts_hist` 401.
- MariaDB read-only account: `SELECT` returns rows; `CREATE TABLE` fails with
  `ERROR 1142 (42000): CREATE command denied to user 'mdqm_ro'` — the same guarantee Postgres gives.
- OpenSearch: monthly indices `mdqm-usage-events-2025-05 .. 2025-12` exist with document counts
  (613 / 606 / 651 / 643 ...) and the `mdqm-usage-events` alias points at them, so the httpx search
  path has a real REST API to run against.

Still not exercised: the OpenAPI adapter against the `api` container over a real port (its tests use an
in-process transport), and the generated fixtures on any platform other than this one.

## Plan issues
`docs/dev-env.md` is owned by D01; this task made a small additive edit to it (the OSS section) and
says so here rather than pretending the file was free.
