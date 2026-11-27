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

**Verified against real services** (Docker Desktop, real containers; the commands are in
`docs/dev-env.md`, `docs/deployment.md` and each `progress/*.md`):

- **PostgreSQL**: the D03 schema loads (three databases, monthly range partitions, a read-only role
  that the server itself refuses to write through), the A02 adapter passes the conformance kit, and
  the canonical cross-source question -- average credit limit by status, with the 5000 half-up
  rounding and the erp division-by-ten form -- reproduces ground truth on the real database.
  `tests/integration/test_postgres_e2e.py` answers it three ways: the compiled spec, hand-written
  SQL, and E01's offline answer.
- **Elasticsearch**: monthly indices, the alias, the index template and the bulk load run against a
  real node; the search adapter is exercised against the same data.
- **The container**: `docker build` succeeds, the image runs as a non-root user with GID 0 (the
  OpenShift pattern), `GET /healthz` answers, `mdqm probe` reaches the database, and a first boot
  with no semantic layer starts and says so instead of crash-looping.
- **The claims**: `tests/integration/test_claims_e2e.py` asserts each problem this README opens with
  -- the rounding, the refused query with zero source calls, the cache hit with no new fetch, the
  late-arriving row kept by the derived bound, the audited refusal, the context pack inside its
  budget, and the governor's concurrency cap with a real cancellation.
- **Benchmarks**: `scripts/bench.py` measures each stage and asserts only invariants; the model
  benchmark (`scripts/bench_glm.py`) scores a real weak model's answers against E01's ground truth.

**Not proven yet** (each is a named task, not a guess):

- The MCP `query` tool that ties the compiled pipeline together (`M03`/`M04`) and the end-to-end
  acceptance that runs it over real PostgreSQL (`E03`) are still open. Until they land, "end to end"
  means the pipeline, not the tool surface.
- Oracle and SQL Server support is written and unexercised (`D05`/`D06`, `A07`/`A08`).
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
