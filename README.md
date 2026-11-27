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

This is an early build. What is proven:

- The full path from a validated spec to executed SQL, including the 5000-rounding and the ÷10 form,
  runs end-to-end against three in-process DuckDB sources and reproduces ground truth that was
  computed independently of the code under test.
- Guard, planner, cache, governor, audit, history, semantic layer, MCP tools and the Claude Code hook
  have unit tests; `scripts/check.py` runs ruff, `mypy --strict` and the suite.

What is **not** proven:

- Nothing here has ever run against a real PostgreSQL, Oracle or SQL Server instance. No container
  runtime was available while it was built, so the compose stack and the Dockerfile are written and
  statically verified but never started. See `progress/D01.md` and `progress/M08.md` for the exact list.
- Oracle, SQL Server and Elasticsearch support is written, not exercised.
- The on-prem model endpoint has only been spoken to through a mock transport.

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
