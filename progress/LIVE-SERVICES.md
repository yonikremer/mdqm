status: done
task: LIVE-SERVICES
agent: cursor branch: cursor/live-services-proof-4dea commit: see git log
tests run:
  `.venv/bin/python -m pytest tests/integration --require-services -q` -> 193 passed in 108.52s
  (PostgreSQL, Oracle, SQL Server, Elasticsearch, the mock API, the compose server on port 58765).
  After that run, MariaDB and OpenSearch were started and
  `.venv/bin/python -m pytest tests/integration/test_oss_live.py --require-services -q` -> 2 passed.
  The integration tree is 195 tests. `.venv/bin/python scripts/check.py` is the CI gate (ruff, mypy, unit tests).

## Summary
Docker 29.1.3 was installed on this VM and the compose stack was started. Every integration test
ran with `--require-services` against the live engines, not against DuckDB stand-ins. Bugs found
that way were fixed and the assertions were left as they were.

## What ran live
- PostgreSQL 16 (`postgres:16`) on 127.0.0.1:55440. Databases crm, billing, erp. Read-only role
  `mdqm_ro`: a write is SQLSTATE 25006 while `default_transaction_read_only` is on, and 42501
  after that is forced off.
- Oracle Free 23ai (`gvenzl/oracle-free:23-slim-faststart`) on 127.0.0.1:51521. Read-only account
  refuses DML with ORA-41900; `SET TRANSACTION READ ONLY` refuses a write the account could
  otherwise make with ORA-01456.
- SQL Server 2022 CU14 on 127.0.0.1:51433. Read-only login `mdqm_ro` is error 229 on
  INSERT/UPDATE/DELETE and 262 on CREATE TABLE. `GRANT SHOWPLAN` is now part of the loader, so
  `SET SHOWPLAN_XML` returns a plan document.
- Elasticsearch 8.15.3 on 127.0.0.1:59200. Search-only transport; a write method never leaves the
  client.
- Mock API container `mdqm/mock-api:dev` on 127.0.0.1:58000. `/health` is open, `/rates` requires
  the key, POST `/rates` is 405, and the OpenAPI document lists only `get` for `/rates`.
- Server image `mdqm/server:dev` from `docker build`. Compose runs it as uid 10001 gid 0, talks to
  the three Postgres databases, and `test_acceptance_e2e.py` is 5 passed (cross-source answer,
  cache hit with `fetched_at`, partition refusal, no way to express DELETE, budgeted context
  pack). A second container as uid 1000990000 gid 0 read a root:0 mode 0440 token file, answered
  `GET /healthz` on port 58766, and wrote a file under `/data`. The compose container created
  `/data/metadata.sqlite` as `mdqm:root`.
- MariaDB 11.4.4 on 127.0.0.1:53306 and OpenSearch 2.17.1 on 127.0.0.1:59201. `mdqm_ro` selects
  the small world (5001 usage events, 200 crm customers, more erp rows than crm because of the
  planted duplicate) and `CREATE TABLE` is error 1142. The alias `mdqm-usage-events` covers at
  least eight monthly indices and `_count` equals `WorldConfig.small().event_count`.

## Counts from the 193-passed run, plus the two OSS tests
| file | passed |
| --- | --- |
| test_acceptance_e2e.py | 5 |
| test_claims_e2e.py | 7 |
| test_e2e.py | 8 |
| test_es_adapter.py | 15 |
| test_mock_api_live.py | 1 |
| test_mssql_adapter.py | 18 |
| test_oracle_adapter.py | 19 |
| test_oss_live.py | 2 |
| test_pg_adapter.py | 15 |
| test_pg_partitions.py | 7 |
| test_pg_stats.py | 10 |
| test_postgres_e2e.py | 5 |
| test_seed_es.py | 10 |
| test_seed_mssql.py | 17 |
| test_seed_oracle.py | 21 |
| test_seed_pg.py | 14 |
| test_stage_catalog_live.py | 3 |
| test_stage_mutability.py | 6 |
| test_stage_propose_names_live.py | 4 |
| test_stats_postgres.py | 1 |
| test_tool_query_e2e.py | 3 |
| test_tool_query_federated_e2e.py | 4 |
| total | 195 |

## Bugs fixed
1. SQL Server's read-only login was never granted `SHOWPLAN`. `SET SHOWPLAN_XML ON` succeeded and
   the compiled statement was then refused, so `explain` could not run. `grant_read_only` now
   issues `GRANT SHOWPLAN`, and `test_read_only_login_can_capture_a_plan` requires a plan document.
2. `COUNT_BIG(*)` of the three-way `usage_events` cross join is rewritten by SQL Server to the
   product of the cardinalities and finishes in about half a second, which is `timeout_s`. The
   deadline test flaked. The slow statement is now a checksum over every triple, which cannot be
   answered from the cardinalities. Five repeats of cancel-then-timeout passed.
3. `size_stats` was added to `OPTIONAL_METHODS` and no probe reported it, so the SQL Server probe
   test failed with "unsupported without saying why". PostgreSQL probes it by running `sizes()`.
   Oracle, SQL Server and Elasticsearch report it unsupported with a reason, and each has a
   `sizes()` that raises `NotSupported` with that same reason. The plan keeps `estimate_source:
   catalog` for those three.
4. `mssql-python` 1.19 on Ubuntu links `libodbcinst.so.2` against `libltdl.so.7`, which the wheel
   does not contain. Every connection failed with "Failed to load the driver" until `libltdl7`
   was installed. The server image now installs that library at build time. `docs/mssql-driver.md`
   no longer says the host needs nothing.
5. The MariaDB service never received `MDQM_RO_PASSWORD`, so `10-readonly-user.sh` aborted init
   and the container exited 1. Compose now passes the variable. A unit test locks the wiring. The
   progress note that claimed this was fixed in commit `a716d71` does not match this history:
   that commit is not in the repository.
6. `python scripts/dev.py env` did not create `docker/server/tokens.json` or
   `docker/server/pg_ro_password`, which the server profile mounts and which the acceptance test
   tells the operator to create with that command. It creates both, once, and does not overwrite
   a file that is already there. The password is not printed.

## Decisions
- Container-to-container traffic was dropped because this VM's iptables-legacy `FORWARD` policy
  is `DROP` while Docker 29 writes its rules through iptables-nft. `iptables-legacy -P FORWARD
  ACCEPT` was set on the VM so the server container could reach Postgres. That is an environment
  fix, not a product change.
- OpenSearch in the `oss` profile has its security plugin off, so there is no read-only account
  to assert. The test checks the seed and a search. The search adapter's own write refusal is
  covered by the Elasticsearch suite.
- No new Python dependency. MariaDB is queried with the client inside its container so the
  password stays in the container environment.

## Deviations
none from the request to prove the live services and keep the assertions.

## Dependencies
none

## Not verified
- An image built with `--build-arg MDQM_EXTRAS=pg,mssql` logging into SQL Server. That is task
  `A10`. The rebuilt default image (extras `pg`) loads `libltdl.so.7` as uid 1000990000 and still
  answers `GET /healthz`. The `mssql` extra itself was not selected, so the driver was not
  imported inside the image.
- The on-prem model endpoint.
- Oracle, SQL Server and Elasticsearch still do not collect relation sizes. The probe says so.
  Implementing those collectors was not required to make the live suite pass.

## Plan issues
`progress/OSS-mocks.md` says the MariaDB password pass-through was fixed in `a716d71`. That commit
is not in this clone, and the compose file on `main` did not pass the variable. The unit test
added here is what stops that from regressing quietly.
