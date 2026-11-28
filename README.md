# mdqm — Multi DB Query MCP

An MCP server that lets a coding agent ask business questions across many data sources — PostgreSQL,
Oracle, SQL Server, Elasticsearch, internal REST APIs — and get one trustworthy answer back.

Built for **weak agents on desks, in Hebrew and English**, and for **primary databases with no replica**
that must not be hammered.

## The problems it solves

**The same number means different things in different systems.**
A credit limit is rounded to 5000 in one system, stored raw in another, and stored as *rounded then
divided by 10* in a third. Join them naively and you get a confident, wrong average. mdqm keeps a
semantic layer that knows the conversion per source, applies it in SQL explicitly, and reports every
conversion it applied.

**Agents write SQL that melts the primary.**
No partition filter, no time bound, `SELECT *` over a year of events. mdqm refuses a query with no
partition key, tells the agent exactly what to add and what it would cost, and never reaches the source
until the query is acceptable. Concurrency, quotas, timeouts and a circuit breaker sit in front of every
source.

**The same fetch happens twenty times.**
Agents retry, reformulate, re-ask. Identical concurrent calls collapse into one source call; results are
cached in partition-aligned segments, and every served piece carries its own `fetched_at` so mixed
freshness is visible instead of averaged into a lie.

**Late-arriving rows silently vanish.**
Event time and ingestion time are different questions. mdqm derives the ingestion-time bound from the
measured lag, keeps the original event-time predicate so no late row is dropped, and warns when a window
ends inside the lag.

**A weak model cannot choose between 40 tools or read a schema.**
There are five coarse tools. Queries are a structured spec, not free SQL. The context the agent needs
(names, measures, values in both languages, an approved example) is injected automatically, inside a
token budget, with what was left out reported.

**Every agent session needs a trail.**
Append-only, hash-chained audit of what was asked and what was refused. Query history that can be
re-run *frozen* (same window, same timestamp) or *relative* (last 7 days, today).

**Read-only is not a promise, it is three layers.**
A read-only account, an AST guard that also inspects SQL the server compiles, and a read-only
transaction. APIs: GET only, enforced at the transport. Elasticsearch: search only.

## How an answer is produced

```text
question → find_data (context pack) → spec → lint → compile per source → guard
        → governor (limits, breaker, cache) → execute → DuckDB join
        → summary + sample + handle + quality report
```

The quality report counts what each source returned, what the join matched, what it dropped and why.
A dropped row is never a silent row.

## Status: honest

This is an early build, and it is built so that every claim can be checked by running something.
The ledger for the live run is `progress/LIVE-SERVICES.md`.

**Proven against real services** (`python -m pytest tests/integration --require-services`, Docker
29.1.3, the commands in `docs/dev-env.md`):

| service | what was proven |
| --- | --- |
| PostgreSQL 16 | three databases, the read-only role (SQLSTATE 25006 and 42501), partitions, stats, the cross-source credit-limit answer through the MCP client, cache, the refused unbounded scan, column-share collection (`test_stats_postgres.py`) |
| Oracle Free 23ai | seed, interval partitions, read-only account (ORA-41900) and read-only transaction (ORA-01456), the adapter conformance kit |
| SQL Server 2022 | seed, partition function/scheme, read-only login (error 229), `SHOWPLAN`, Query Store, the adapter conformance kit |
| Elasticsearch 8.15.3 | monthly indices, alias, search-only adapter |
| Mock OpenAPI | the compose container on port 58000: open `/health`, key required, POST not served, planted USD rates |
| MariaDB 11.4 + OpenSearch 2.17 | the `oss` profile: `mdqm_ro` can `SELECT` and is refused `CREATE` (error 1142); the usage-events alias and document count match the small world |
| Server image | `docker build` succeeds; Compose runs it as uid 10001 / gid 0 against the three Postgres databases; `tests/integration/test_acceptance_e2e.py` is 5 passed on port 58765. The same image started as uid 1000990000 / gid 0 (OpenShift's arbitrary-uid pattern), read a mode-0440 group-0 token file, answered `GET /healthz`, and wrote under `/data` |

Read-only holds at the database on every dialect that was started: the account is denied the
privilege, and (where the engine has it) a read-only transaction is a second refusal. Elasticsearch
and the mock API refuse writes at the HTTP method. OpenSearch in the `oss` profile runs with
security off, so it has no read-only account; the search adapter still issues search requests only.

**Not proven on a live service in that run:**

- A server image built with the `mssql` extra actually logging into SQL Server (`A10`). The image
  that was built installs `libltdl7` (the library the driver needs on Debian) and loads it, and the
  host venv connected to SQL Server after the same library was installed. The extra itself was
  not selected for that image.
- The on-prem model endpoint. It has only been spoken to through a mock transport, so the label
  proposals A1+A2 now write are unverified in quality (`progress/LLM-WIRING.md`).
- The asynchronous query path (`query` with `mode: "async"` plus `get_more(query_id=…)`,
  `progress/ASYNC-QUERY.md`). The blocking path is the one the live runs above exercised.
- The Oracle bytes branch of the size reader: this compose Oracle grants the read-only account no
  segment view, so the live run proves the "not reported" answer, not a measured byte figure
  (`progress/SIZE-READERS-LIVE.md`). The other three engines' size readers are proven live.

`TASKS.md` holds the backlog and `progress/` holds one short report per finished task, including what
each one could not verify.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows. Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,pg]"
python scripts/check.py          # ruff + mypy --strict + unit tests
```

Open-source local stack (Postgres, MariaDB, OpenSearch, mock API) — needs Docker:

```bash
python scripts/dev.py env
docker compose --profile oss up -d
```

Deployment: `docs/deploy.md` (Docker and OpenShift, air-gapped). Configuration: `docs/dev-env.md`.

## Where to read next

| file | what it is |
| --- | --- |
| `AGENTS.md` | rules and the overnight protocol agents follow |
| `docs/DESIGN.md` | the design and its reasoning |
| `TASKS.md` | backlog, dependencies, acceptance criteria |
| `docs/deploy.md` | Docker and OpenShift deployment |
| `docs/dev-env.md` | local services, ports, profiles |
| `progress/` | one ledger per finished task, unverified items included |
