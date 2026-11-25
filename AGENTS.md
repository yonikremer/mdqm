# AGENTS.md: Multi DB Query MCP (`mdqm`)

You are a coding agent working on `mdqm`. Read this file fully, then `docs/DESIGN.md`, then your task in `TASKS.md`
(`python scripts/next_tasks.py show <ID>`). The owner is Yoni, a data analyst. Nobody is awake to answer questions
overnight: when something is ambiguous, choose the conservative option, write the decision down in `progress/<ID>.md`, and continue.

## 1. Goal

A thin, central **MCP server in Python** that lets weak coding agents (Claude Code on user desktops, mid-size model, Hebrew and
English) query many heterogeneous data sources safely and cheaply: PostgreSQL, Oracle, SQL Server, Elasticsearch, and internal
REST APIs with an OpenAPI spec. It fixes: non-standard cross-source joins with unit/format/rounding conversions, index- and
partition-aware access, no repeated fetches (cache), organisational identity instead of shared passwords, query history and audit,
runtime control, a semantic dictionary (names and categorical values in two languages), schema drift, and agents that send many
inefficient queries to primary databases that have no replicas.

The semantic layer and schemas are built **autonomously**: as much as possible deterministic code; the rest agentic (a model proposes,
code validates, humans approve).

The target runs in an **air-gapped on-prem network**. We develop outside, with internet. The product must be **generic and
plug-and-play for other organisations**: no hard-coded versions, product assumptions, or organisation facts in code.

## 2. Principles (non-negotiable)

1. **Generic via capability detection.** A probe discovers what each source supports; each feature declares required capabilities and has a fallback. Org facts live in YAML/config, never in code.
2. **Adapters are plugins.** Core depends only on the `SourceAdapter` protocol. Adapters register through the entry-point group `mdqm.adapters`.
3. **Deterministic first.** If code can compute it, code computes it. Models only propose; validators check against data; humans approve (see DESIGN section 10).
4. **Weak-model friendly.** Few coarse tools; a structured JSON spec instead of free SQL; context injected automatically; errors carry an exact `fix`. Never assume the calling agent is smart, diligent, or reads documentation.
5. **Read-only, three layers**: read-only DB account, AST guard (also on SQL we compile), read-only transaction. APIs: GET only. ES: search only. Never write a code path that can modify a source.
6. **Protect primary databases.** Concurrency limits, timeouts, EXPLAIN gate, enforced partition pruning, quotas, row/byte caps.
7. **No runtime internet.** At runtime nothing may contact any host except configured sources and the configured model endpoint. No DuckDB `INSTALL`/extensions, no model or data downloads, no telemetry, no CDN assets, no `pip install` at runtime.
8. **All dependencies come from a PyPI mirror as wheels.** Prefer pure-Python or wheel-shipping packages. Before adding a dependency: is there a stdlib or existing-dependency way? Is it on PyPI with wheels for Windows, Linux, macOS? Is it maintained? Record each new dependency and why in `progress/<ID>.md`.
9. **OS agnostic.** Windows is the primary dev OS. No bash in shipped code or in test commands; no `&&` in documented commands (Windows PowerShell 5 lacks it); use `pathlib`; always pass `encoding="utf-8"` to `open`/`read_text`/`write_text`; no `fcntl`/`signal.SIGALRM`-only designs; LF line endings (see `.gitattributes`).
10. **Everything reported, nothing silent.** Auto-fixes appear in `applied_fixes`; fallbacks appear in the capability report; cache hits show `fetched_at`; join drops are counted.
11. **Secrets never in code, YAML, logs, tests fixtures (other than throwaway dev passwords in `.env.example`), or error messages.**

Out of scope (do not build): row/column permissions, PII masking, a second join engine (Trino), embeddings, dashboards, a web UI.

## 3. Tech stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ (develop on 3.12/3.13, CI 3.11 and 3.13) | type hints everywhere, `mypy --strict` on `src` |
| MCP | `mcp` **2.x** (`from mcp.server.mcpserver import MCPServer`) | **2.x renamed FastMCP to MCPServer** and changed APIs. Almost all online examples are 1.x. Read the installed package and https://py.sdk.modelcontextprotocol.io/v2/ before writing server code. `pyproject.toml` pins `mcp>=2,<3` and a smoke test guards it |
| Models/validation | `pydantic` v2, `pydantic-settings` | frozen models, `extra="forbid"` |
| SQL parsing/translation | `sqlglot` | `unsupported_level=RAISE`; golden tests; never trust best-effort translation silently |
| Local engine and cache | `duckdb` (+ `pyarrow`) | in-process; no extensions; set `memory_limit` and a spill dir |
| Metadata DB, FTS | SQLite via stdlib `sqlite3`, FTS5 | WAL; FTS5 availability is probed (Python builds can lack it) |
| YAML | `ruamel.yaml` | round-trip with comments and line numbers for curated files |
| HTTP | `httpx` | Elasticsearch REST and OpenAPI adapters use httpx directly (no `elasticsearch` package: its client/server version coupling hurts generic support) |
| PostgreSQL | `psycopg[binary]` 3 + `psycopg_pool` | extra `pg` |
| Oracle | `oracledb` Thin mode | extra `oracle`; no Oracle client; Thin works with 12c+ (do not require 23ai) |
| SQL Server | undecided: task A08 runs a driver spike | `mssql-python` vs `pymssql` vs `pyodbc` |
| CLI | `typer` | `mdqm probe`, `mdqm bootstrap`, `mdqm serve`, `mdqm audit`, `mdqm review`, `mdqm eval` |
| Tests | `pytest`, `pytest-asyncio`, `pytest-socket`, `hypothesis` | mock API with `fastapi` (dev extra only) |
| Lint/format | `ruff` (lint + format), `mypy` | `python scripts/check.py` runs everything |
| Packaging | `hatchling`, `pyproject.toml`, `src/` layout | single distribution `mdqm`; adapters are separate entry points |
| Time zones | `zoneinfo` + `tzdata` | tzdata is required on Windows |

Style of concurrency: the server is async (MCP SDK); adapters expose **synchronous** methods and the server runs them in a bounded thread pool behind the governor (semaphores, timeouts, cancellation via the driver's cancel API). Keep adapter code simple and blocking.

## 4. Repository layout

```
AGENTS.md  CLAUDE.md  TASKS.md  tasks.json  README.md  pyproject.toml  docker-compose.yml  .env.example
docs/            DESIGN.md (design), dev-env.md, tool-schemas.json (generated)
examples/        semantic_example.yaml, query_specs.json
scripts/         next_tasks.py integrate.py render_tasks.py tasks_def.py check.py dev.py wait_for.py
progress/        <TASK-ID>.md  one per finished or blocked task (this is the "done" ledger)
plugin/mdqm/     Claude Code plugin (client side): .claude-plugin/ .mcp.json hooks/ skills/
src/mdqm/
  models/        shared pydantic models (Capabilities, CatalogSnapshot, Partitioning, Fragment, ErrorResponse ...)
  config/        Settings, SourceConfig, secret providers
  adapters/      base.py registry.py postgresql.py openapi.py oracle.py sqlserver.py elastic.py
  sql/           guard.py exec_ts.py fingerprint.py          (pure sqlglot logic)
  spec/          query spec models, durations
  semantic/      YAML models + loader, value maps, measures
  planner/       time.py partitions.py
  compiler/      spec -> Fragment
  linter/        rules registry
  cache/         freshness.py segments.py singleflight.py
  governor/      semaphores, quotas, result handles
  executor/      federated execution over DuckDB
  results/       summaries
  store/         db.py (SQLite), audit.py, history.py, migrations/
  context/       FTS index, Context Pack
  probe/         capability probe
  pipeline/      core.py stages/ (D1..D10, A1..A2)
  review/        proposals queue, digest
  llm/           OpenAI-compatible client, FakeLLM
  server/        MCP server, tools/, context endpoint, auth
  eval/          phase-0 harness
  testing/       world.py (synthetic data), fake_adapter.py, conformance.py, loaders/, mock_api.py
tests/
  unit/          no network, no Docker. Fast (the whole folder < 60 s)
  integration/   needs docker-compose services; `--require-services` makes missing services fail
  data/          queries/ semantic/ specs/ plans/ golden/ world/
  eval/          questions.yaml
```

Modules are small and have one job. Pure logic (guard, exec_ts, planner, freshness, linter rules, compiler) must have **no I/O** so it is unit-testable with tables of cases.

## 5. Coding conventions

- Public functions and models are typed; `mypy --strict` passes on `src`. Prefer `Protocol` and dataclass/pydantic over inheritance trees.
- Pydantic models are `frozen=True, extra="forbid"`. Config/YAML errors must name the file, path and (for YAML) line.
- No `print` in library code (CLI output via typer). Use `logging` with structured `extra`. Never log secrets, API keys, tokens, or full result rows.
- Timestamps: timezone-aware UTC internally; convert at the edges. `exec_ts` is one value per execution.
- No global mutable state; pass dependencies explicitly (settings, secret provider, clock). Inject a `Clock` where time matters so tests don't sleep.
- Errors the agent can see are `ErrorResponse` values (code, problem, fix), never exceptions with stack traces. Internal exceptions are logged and converted to `INTERNAL_ERROR` with a correlation id.
- SQL is built through the sqlglot AST. **No string concatenation of user-supplied values into SQL.** Identifiers are validated against the catalog; values are bound or rendered through sqlglot literals.
- Hebrew: all files are UTF-8; tests include real Hebrew strings (value maps, synonyms, questions). Normalise with NFKC, strip niqqud, fold geresh/gershayim variants. HTML for Hebrew uses `dir="rtl"`.
- Keep functions short. Comments explain *why*. Docstrings on public API. No dead code, no TODOs without a task ID.
- Dependency additions: append to `pyproject.toml`, keep the diff minimal (several agents edit it; on merge conflict keep both sides).
- Commit messages: `T-ID: short summary` (e.g. `C06: measure normalisation with explicit rounding modes`).

## 6. Testing conventions

- `python -m pytest tests/unit -q` must always be green and fast. Unit tests never touch the network or Docker (`pytest-socket` blocks sockets for `tests/unit`; tests that need loopback, like the mock API and the in-process MCP client, opt in for 127.0.0.1 only).
- Integration tests live in `tests/integration`, are marked `integration` automatically, and use Docker Compose services. Without services they **skip**; with `--require-services` they **fail**. Acceptance criteria that mention integration tests always mean "ran with `--require-services` against real services". A skipped test is not a pass.
- Prefer table-driven tests with many cases over a few clever ones. Add `hypothesis` property tests for pure functions (freshness monotonicity, rewrite idempotence, derived time bounds never losing rows).
- Golden files for compiled SQL and rendered output: `tests/data/golden/`, via the `golden` fixture; update with `--update-golden` and review the diff before committing.
- Determinism: seeded randomness, injected clock, sorted outputs. A flaky test is a bug: fix or delete with a note, never retry-until-green.
- Every bug fix and every linter/guard rule gets a regression test first.
- Never make a test pass by weakening it, skipping it, or special-casing test inputs in `src`.

### Docker Compose (task D01; services behind profiles so you start only what you need)

| Profile | Service | Purpose |
|---|---|---|
| `pg` | PostgreSQL 16, databases `crm`, `billing`, `erp`, role `mdqm_ro`, `pg_stat_statements` | MVP target. The three logical datasets live in three databases so cross-source conversions are testable with Postgres alone |
| `oracle` | Oracle Free (slim-faststart) | later: interval partitioning, `V$SQL`, PSTART/PSTOP |
| `mssql` | SQL Server 2022 | later: partition function/scheme, Query Store |
| `es` | Elasticsearch 8.x single node, security off | later: monthly indices + alias |
| `api` | mock OpenAPI service (D04) | API adapter tests (also runs in-process) |

Use `python scripts/dev.py up pg` / `down` / `status` (stdlib wrapper around `docker compose`, works on Windows). Oracle, SQL Server and Elasticsearch together need roughly 10-12 GB RAM and a slow first start: do not bring them up unless your task needs them. If Docker is unavailable in your environment, do everything that does not need it, and report exactly which acceptance items you could not verify.

### Synthetic test data (task D02, seeded, deterministic, OS independent)

One generated "organisation" in three storage forms of the same customers:

| Dataset | Table | Status column | Credit limit |
|---|---|---|---|
| `crm` | `customers(cust_id INT, ...)` | `'A'/'C'/'S'` | **rounded half-up to 5000** |
| `billing` | `accounts(client_no VARCHAR(10), ...)` + `usage_events` | `'Active'/'Closed'/'Hold'` | **raw, not rounded** |
| `erp` | `customers(cust_no INT, ...)` | `1/2/3` | `limit_k` = rounded to 5000 **then divided by 10** (multiples of 500) |

So the canonical measure is `round_half_up(x/5000)*5000` and normalisations are identity / `ROUND(x/5000)*5000` / `x*10`. The generator plants: exact .5 boundary values (2500, 7500, 12500), NULLs, orphan keys, duplicates, and a few intentional exceptions listed in the ground truth. `usage_events` has `start_time` (event time, indexed, partition key, monthly) and `insertion_time` (ingestion time, not indexed) with lag up to a stated `max_lag`; `usage_by_load` is partitioned on `insertion_time`; one table is hash partitioned; `accounts_hist` is frequently updated (mutable_critical). Ground truth for every eval question is computed independently of the code under test.

## 7. Definition of done

**A task is done when all of these hold** (otherwise it is not done, and you write `status: blocked` or `status: partial`, never `done`):

1. Every acceptance criterion in `TASKS.md` for the task is met and has a test (or an explicit manual verification note for the few non-automatable ones).
2. The task's **test command** passes, and `python scripts/check.py` is green (ruff, mypy, all unit tests), on a rebased branch.
3. No new network access at runtime; no new dependency without a note; no secrets; code is OS agnostic.
4. Public API documented by docstrings; if behaviour changed, `docs/DESIGN.md` is updated in the same commit (only if the design truly changed: otherwise write the deviation in your progress file instead).
5. `progress/<ID>.md` exists (format below) and the branch has been merged by `python scripts/integrate.py <ID>`.

**MVP slice done (tonight's goal)**: all P0 tasks done, ending with E03 green: against Docker Postgres x3 and synthetic data, an MCP client gets the cross-source "average credit limit by status" answer equal to the ground truth through the 5 tools; late-arriving rows are handled via the derived bound; a query without the partition filter is rejected with a precise `fix` and no source call; a repeated query is served from cache with `fetched_at`; a DELETE attempt is blocked and audited; and the bootstrap stages rediscover the 5000 / 5000-then-/10 conversions from data.

## 8. Overnight operating protocol

Several agents may run in parallel (one task each). Use the scripts; they enforce the rules.

```
python scripts/next_tasks.py ready --priority P0,P1        # what can start now (deps done, not claimed)
python scripts/next_tasks.py show C06                      # full task text
python scripts/next_tasks.py claim C06 --agent <your-name> # atomic. If it fails, pick another task
git worktree add .worktrees/C06 -b task/C06 main           # your own working copy (cd into it, create/activate a venv or reuse the repo venv)
... implement + test ...
python scripts/check.py                                    # must be green
# write progress/C06.md (format below), commit
python scripts/integrate.py C06                            # lock, rebase on main, re-run checks, fast-forward main, release claim
python scripts/next_tasks.py ready --priority P0,P1        # loop
```

Worktree gotcha: the venv's editable install points at the main checkout. pytest is configured with `pythonpath = ["src"]`, so tests in your worktree import your worktree's code. For ad-hoc commands such as `python -m mdqm.testing.world`, set `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH="src"`).

Rules:

1. **One task at a time, only files the task owns** (the `owns` list in TASKS.md). Need a change in someone else's file? Make the smallest possible edit, mention it in your progress file. Do not refactor other lanes.
2. **Never edit** `TASKS.md`, `tasks.json`, `scripts/tasks_def.py`, or other tasks' progress files. If the plan is wrong, say so in your progress file (`## Plan issues`).
3. **Priority order**: P0 first, then P1, then P2. Prefer the lowest wave number. Do not start a task whose dependencies are not done, even if you could stub them.
4. **Time box**: if a task is still not green after ~90 minutes of effort, or you are blocked by something you cannot resolve (missing Docker, ambiguity that changes the design, a dependency that does not exist), stop: write `progress/<ID>.md` with `status: blocked`, what you tried, what you need, `python scripts/next_tasks.py release <ID>`, and take another task. Blocked is a fine outcome; faking green is not.
5. **Safety**: stay inside the repo directory and its Docker Compose project (`mdqm_*` containers/volumes). Do not touch other containers, global Python, system settings, or the user's other projects. Do not push to any remote, do not force-push, do not rewrite `main` history, do not delete branches that are not yours. Never `docker system prune`. No real credentials anywhere: only the throwaway ones in `.env.example`.
6. **Do not invent facts about the first organisation.** The unknowns below stay configurable/pluggable. Where you must pick a default, pick the generic, conservative one and record it.
7. Hebrew and English both matter. Examples, tests and docs for user-facing text include both.
8. When you finish, always leave `main` green: `python scripts/check.py` passes on `main` after your integrate.

`progress/<ID>.md` format (first line must be exactly one of `status: done`, `status: partial`, `status: blocked`):

```
status: done
task: C06
agent: <name>   branch: task/C06   commit: <sha>
tests run: <exact commands and results>
## Summary          what was built, in 3-8 lines
## Decisions        anything you chose that the task did not specify
## Deviations       differences from TASKS.md/DESIGN.md and why (or "none")
## Dependencies     new packages with justification (or "none")
## Not verified     acceptance items you could not verify, and why (or "none")
## Plan issues      mistakes or gaps you noticed in the backlog (or "none")
```

Only `status: done` unlocks dependent tasks. Use `partial` when useful code is merged but some acceptance items are unmet (dependents stay blocked until someone finishes it).

## 9. Known unknowns (do NOT guess; keep pluggable)

- **Identity** in the first org (directory/IdP, how personal tokens are issued). Build the `TokenVerifier` interface and a static-token dev implementation only.
- **Model endpoint**: an on-prem GLM-family model behind an API; assumed OpenAI-compatible chat completions, context ~500K (unverified), tool calling reliable (unverified). Everything is probed; never hard-code model behaviour.
- **Database versions**: Oracle is probably 23, Elasticsearch self-managed; the probe must discover versions, not require them.
- **Hosting platform** for the server (an internal cloud with unknown details): ship a container image and env/volume/secret-file configuration only.
- **Internal SMTP** probably exists: SMTP digest with a static-page fallback.
- Claude Code plugin/hook/skill formats and MCP SDK 2.x APIs change: verify against the installed package and current docs, not memory.
- SQL Server plan XML format for partition counting is **unverified**: verify empirically in A08.
- API rate limits/budgets for the three internal APIs are unknown: default to a conservative shared limiter.
