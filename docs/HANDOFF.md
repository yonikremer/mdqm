# Handoff: what is done, what is not, and exactly where to start

Written at `main = 9a17b42` (pushed, working tree clean). Everything below was measured by running
something, not by reading code — the commands are included so you can re-measure instead of trusting me.

Read this first: `progress/SERVER-E2E.md` (the same state, in the shorter progress-note format).

---

## 1. How to bring the stack up

```bash
export PATH="/c/Users/yonik/AppData/Local/Programs/DockerDesktop/resources/bin:$PATH"   # Docker Desktop
.venv/Scripts/python.exe scripts/dev.py up pg          # or: docker compose --profile pg --profile server up -d --build --wait
.venv/Scripts/python.exe scripts/dev.py status
```

* Server: `http://127.0.0.1:58765` — `GET /healthz` returns 200.
* MCP client for manual calls: `python scripts/mdqm_call.py call <tool> '<json args>'`
* Token for HTTP endpoints: `docker/server/tokens.json` (gitignored), first entry's `token`.
* **The CLI is `python -m mdqm.cli <group>`.** `python -m mdqm` does *not* exist (no `__main__.py`) —
  that cost me a debugging round.
* Metadata database inside the container: `/data/metadata.sqlite` (tables include `query_history`,
  `audit_log`, `feedback`). Inspect with `docker exec mdqm-server-1 python -c "…sqlite3…"`.

## 2. The gate you must keep green

```bash
.venv/Scripts/python.exe -m ruff check .          # clean
.venv/Scripts/python.exe -m mypy                  # clean, 143 files
.venv/Scripts/python.exe -m pytest tests/unit -q  # 4466 passed at 9a17b42
.venv/Scripts/python.exe -m pytest tests/integration --require-services -q   # 164 passed
.venv/Scripts/python.exe scripts/check.py         # ruff + mypy + unit, the CI gate
```

Also green at this commit, and worth re-running after changes to query paths:

```bash
.venv/Scripts/python.exe scripts/bench_glm.py score --answers bench/answers   # 15/15 vs E01 ground truth
.venv/Scripts/python.exe -m pytest tests/integration/test_acceptance_e2e.py --require-services -q   # 5 passed
```

## 3. Verified working end to end (through the running server)

| surface | how it was checked | result |
|---|---|---|
| `find_data` | `mdqm_call.py call find_data '{"question":"…"}'` | ok |
| `query` single source | avg credit limit by status | ok, rows + `result_handle` |
| **`query` federated** (`billing.usage_events` ⨝ `crm.customers`) | `group_by customer_status` over Nov–Dec | **1810474 / 956379 / 354603 = E01 ground truth** |
| `get_more` | with the handle from `query` | ok |
| `history` | `{"limit":3}` | ok — **the field is `runs`, not `items`** |
| `feedback` | `{"query_id":"232","ok":true}` | ok, record saved |
| `GET /healthz` | curl | 200 |
| `POST /context` | `{"prompt":"…"}` at top level + `Authorization: Bearer <token>` | 200, returns `additionalContext` |
| `mdqm audit verify` | inside container | "audit chain intact: 234 row(s) verified" |
| `mdqm review list` | inside container | "nothing to review" |
| start with a missing adapter driver | image installs `.[pg]` only | server starts, logs `adapter 'oracle' is not installed in this deployment … it is unavailable` |

## 4. The one big functional gap: the bootstrap pipeline runs nothing

```bash
docker exec mdqm-server-1 python -m mdqm.cli bootstrap run
#   source crm: skipped (no adapter registered for 'postgresql' (registered kinds: none) …)
#   no bootstrap stage is registered in this build; nothing to run
```

Two independent causes, both real:

1. `src/mdqm/pipeline/core.py::available_stages()` returns `()` on purpose ("nothing is registered
   without its source dependency"), so the runner has an empty list.
2. `src/mdqm/pipeline/cli.py` never asks the adapter registry to discover its entry points, hence
   *"registered kinds: none"*. `mdqm.adapters.registry.registry.discover()` is the call that fixes it.

The eight stage classes exist in `src/mdqm/pipeline/stages/`:
`CatalogStage`, `OpenApiStage`, `ConversionStage`, `JoinStage`, `ProfilingStage`, `MutabilityStage`,
`DriftStage`, `ProposeNamesStage`. **Each constructor requires real inputs** (mypy will tell you):
`CatalogStage(adapters)`, `OpenApiStage(specs)`, `ConversionStage(adapters, pairs, measure=…)`,
`JoinStage(adapters)`, `ProfilingStage(adapters)`, and so on.

Do **not** try `Stage()` with no arguments — I did, it produced 12 mypy missing-argument errors and was
reverted. The correct shape is a signature change plus a builder:

```python
def available_stages(*, adapters, specs, pairs, measure) -> tuple[Stage, ...]:   # run order
    return (CatalogStage(adapters), OpenApiStage(specs), ConversionStage(adapters, pairs, measure=measure), …)
```

and `pipeline/cli.py` builds those inputs from `Settings` (`build_adapters(settings)` already exists in
the wiring layer) before calling `run_pipeline`.

**Done looks like:** `mdqm bootstrap run` against the live pg profile executes D1+D2 → … → A1+A2 and
**rediscovers the 5000 / 5000-then-/10 conversions** from data. That is E01's last unproven acceptance
claim; the conversions stage's own unit tests pass, but nothing has run it end to end.

## 5. SQL-injection work: the design, and what is left

Design (agreed with the owner, do not re-litigate):

* **`src/mdqm/sql/execution.py` is the single chokepoint.** `prepare(text, dialect, params)` runs
  `mdqm.sql.guard.guard` and raises `StatementRefused` (carrying the guard's `ErrorResponse`);
  `PreparedStatement` has no other constructor. `execute_one(cursor, statement)` is the only function
  that hands dynamic source SQL to a driver(`run` is an alias). `wrapped(prepared, template=…)` is the
  only place statement text is assembled (EXPLAIN wrappers). `set_local(engine, name, value)` validates
  a local-engine setting name and value against a fixed table. `run_local(cursor, text, params)` is for
  our own metadata SQLite, which writes and which the read-only guard must not see.
* **`scripts/check_sql_sinks.py` is our own lint rule** (a data-flow check, replacing the naive
  built-in one). It flags interpolation (`f"…{v}"`), `+`, `%`, `.format()/.join()`, and conditional-built
  SQL reaching `execute`/`executemany`/`copy`; it passes literals, names bound only to literals, and
  calls to `prepare`/`execute_one`/`run_local`/`sql.SQL`. `TRUSTED_ASSEMBLERS` exempts
  `mdqm/sql/execution.py` **by path** — the list has exactly one entry on purpose.
* The naive rules (`python-sql-injection`, semgrep `sqlalchemy-execute-raw-query`) are **disabled in
  `.pi-lens.json`** and replaced by the above. Also disabled there: `no-ellipsis-body`,
  `unchecked-numeric-parse-python`, `line-length` (the owner asked for the last one explicitly).
* `29` adversarial protection tests live in `tests/unit/test_sql_injection_protection.py`; the chokepoint
  has its own tests in `tests/unit/test_sql_execution.py` (added by the adapters agent).

Remaining:

1. **`python scripts/check_sql_sinks.py src/mdqm` reports 42 sites.** Mostly `src/mdqm/testing/loaders/*`
   (DDL built from table-name constants) and two `src/mdqm/cache/segments.py` `CREATE OR REPLACE TABLE …`
   statements. Fix by inlining the literal at the call site, or by adding chokepoint helpers for the
   identifier cases. Note: I added `create_table_as` / `create_table_from_query` to `execution.py` in an
   attempt at the cache sites, then reverted the file — **check `grep -n "^def " src/mdqm/sql/execution.py`
   before assuming they exist.**
2. **The checker has no tests of its own.** It is a security tool; it needs cases proving: f-string → flagged,
   `+` → flagged, `%` → flagged, `.format`/`.join` → flagged, conditional → flagged; literal → clean,
   module constant → clean, `prepare(...)` → clean, `execute_one(...)` → clean; the trusted-assembler
   exemption honoured. Table-driven, in `tests/unit/`.
3. **Oracle / SQL Server / Elasticsearch**: their `execute` is wired through `execute_one` by the adapters
   agent (verified: 52 adapter integration tests green), Elasticsearch is documented as JSON-not-SQL.
   Re-check with `python scripts/check_sql_sinks.py src/mdqm/adapters`.

## 6. Deployment facts and gaps

* `Dockerfile` installs `.[pg]` only. Oracle and SQL Server need their extras (`.[oracle]`, `.[mssql]`);
  a missing driver is now survivable but the adapter is unavailable. Decide per deployment and document.
* Air-gap rules still apply: no runtime `pip install`, no DuckDB extensions, no network at hook time.
  `.pre-commit-config.yaml` is `repo: local` throughout for that reason, and every hook goes through
  `scripts/precommit_hook.py` (it resolves `.venv/Scripts/python.exe`; `language: system` alone runs the
  wrong interpreter — that is how `No module named 'ruamel'` happened).
* Pre-commit hooks: `ruff check --fix`, `ruff format`, `check_configs.py`, `check_secrets.py`,
  `sql-sinks` (our rule, scoped to runtime paths) on commit; `mypy --strict` and unit tests on push.
  All six verified passing at this commit.

## 7. Lint state (so you do not re-discover it)

* `ruff check .` → **0 findings**. Rule families enabled: `E,F,I,W,S,B,BLE,TRY,RUF,A,DTZ,PERF,C4,PIE,RET,UP,SIM,ASYNC`.
* Five rules are **ratcheted** in `pyproject.toml` with counts and reasons (`PERF401` 39, `BLE001` 15,
  `TRY004` 12, `RUF043` 7, `DTZ001` 56). `TRY003` (490) and `RUF001-3` (73) are off with reasons. Burn the
  ratchet down; do not add to it without a count and a reason.
* **pyright** still reports ~141 findings, almost all in `tests/`: `reportArgumentType` (literals where
  pydantic wants enums/typed models: `TimeMode`, `Duration`, `SpecInstant`, `QuietWindow.start: time`,
  `SecretStr`; and stub adapters not satisfying `SourceAdapter`), `reportReturnType` (fixtures annotated
  `-> X` but they are generators; a stub stream missing `rows_returned`/`bytes_returned`),
  `reportAttributeAccessIssue` (`ArrowStream` not declaring `cached`/`fetched_at`/`total_hits`; `sqlglot.errors`
  used without importing the submodule). None of this gates CI.

## 8. Traps that cost time (avoid repeating them)

1. **A delegated agent often produces no commits.** Observed on D03, S02, D05, E02, M04, E03, D06, A07,
   sqli-testing and others. Always `git log --oneline main..task/<ID>` before believing a completion
   report, and re-run the verification yourself. The good ones (store, adapters) did land.
2. **Heredocs mangle backslashes.** Writing a regex through `<<'PY'` turned `\b` into a literal backspace
   byte and the pattern silently matched nothing. Use the `edit` tool, or build the character with
   `chr(92)`.
3. **`git checkout -- <file>` cannot restore an untracked file.** Committing early is the only real undo.
4. **pyright honours `# type: ignore` too.** Deleting a mypy-only ignore in a test file un-suppressed real
   pyright errors (that is fine — fix them — but know why they appeared).
5. **`.ruff_cache` permission warnings** ("Failed to write cache file … Access is denied") are noise on
   this Windows setup.
6. **The lens scan caps at 100 files per call** out of 249 Python files, and its store is outside the
   workspace. Directory-by-directory scans are the only trustworthy count; a single workspace scan is not
   proof of "zero findings".
7. **Container rebuild required** after any `src/` change (`docker compose … up -d --build --wait server`);
   the running image does not pick up local files.

## 9. Suggested order for the next session

1. **Bootstrap pipeline** (§4) — the largest functional gap; ends with the conversion-rediscovery claim
   proven end to end.
2. **`/context` content**: it answers 200, but confirm `additionalContext` is *non-empty and useful* for a
   real question (the layer/index must be built for it; if the server started without a semantic layer it
   returns the no-layer fallback).
3. **Tests for `scripts/check_sql_sinks.py`** (§5.2), then clear the 42 sites (§5.1).
4. **pyright debt in `tests/`** (§7) — mechanical, one class at a time.
5. Deployment docs for extras (§6).

## 10. Where the history lives

`progress/<TASK-ID>.md` is the ledger: 69 backlog tasks are all `status: done`. That means the *task
board* is closed, not that every feature is proven end to end — §4 is the proof that the two are not the
same thing. `docs/DESIGN.md` is the design; `progress/SERVER-E2E.md` is this state in brief.
