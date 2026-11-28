# Handoff: what is done, what is not, and exactly where to start

Written at `main = a55d0ab` (working tree clean). Everything below was measured by running something,
not by reading code — the commands are included so you can re-measure instead of trusting me. Where a
number would go stale, the command is written instead: run it.

Ledger in `progress/`: `BOOTSTRAP-PIPELINE.md`, `SQLI-CHECKER.md`, `PYRIGHT-TESTS2.md`,
`HEBREW-LABELS.md`, `SIZES.md`. Each says what it verified and what it did not.

**The owner's current scope, in their words:** track table and partition sizes
(`min/max/p1/p25/p50/p75/p99`) and use them for query optimization; return feedback about a query
fast, before it completes. Everything else is secondary. Position of the three:

| ask | state |
|---|---|
| size profile → query optimization | **done and measured** (§4a) |
| fast feedback before completion | **not started** — design agreed: async handle + poll (§4b) |
| model endpoint (so A1+A2 can propose names, incl. Hebrew labels) | **not started** (§4c) |

---

## 1. How to bring the stack up

```bash
export PATH="/c/Users/yonik/AppData/Local/Programs/DockerDesktop/resources/bin:$PATH"   # Docker Desktop
.venv/Scripts/python.exe scripts/dev.py up pg          # or: docker compose --profile pg --profile server up -d --build --wait server
docker compose --profile oracle --profile mssql --profile es --profile api up -d --wait   # for the full integration suite
.venv/Scripts/python.exe scripts/dev.py status
```

* Server: `http://127.0.0.1:58765` — `GET /healthz` returns 200.
* MCP client for manual calls: `python scripts/mdqm_call.py call <tool> '<json args>'` (run it from the
  repo on the host, **not** inside the container: the script is not in the image).
* Token for HTTP endpoints: `docker/server/tokens.json` (gitignored), first entry's `token`.
* **The CLI is `python -m mdqm.cli <group>`.** `python -m mdqm` does *not* exist (no `__main__.py`).
* Metadata database inside the container: `/data/metadata.sqlite` — `query_history`, `audit_log`,
  `feedback`, `pipeline_run`, `review_proposals`, **`size_sample`**.
* Docker gotchas that cost time:
  * `pg` can fail to bind its host port with *"An attempt was made to access a socket in a way
    forbidden by its access permissions"* — Windows had the port as an outbound ephemeral socket.
    Test with `python -c "import socket; socket.socket().bind(('127.0.0.1', 55440))"`; it usually frees
    within a minute, otherwise override `MDQM_PG_PORT` (the env var beats `.env` for Compose **and** for
    the integration fixtures).
  * `pg_stat_user_tables` counters are **zero after an engine restart**, so `test_pg_stats` fails until
    the world is loaded again: `PYTHONPATH=src .venv/Scripts/python.exe -m mdqm.testing.loaders.postgres`.
  * A worktree has no `.env` (gitignored): pass `--env-file /c/Users/yonik/mdqm/.env` to Compose.

## 2. The gate you must keep green

```bash
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m mypy
.venv/Scripts/python.exe -m pyright                          # 0 errors, 0 warnings
.venv/Scripts/python.exe scripts/check_sql_sinks.py src/mdqm  # 0 findings
.venv/Scripts/python.exe -m pytest tests/unit -q
.venv/Scripts/python.exe -m pytest tests/integration --require-services -q
.venv/Scripts/python.exe scripts/check.py                     # ruff + mypy + unit, the CI gate
```

`scripts/bench_glm.py score --answers bench/answers` and
`tests/integration/test_acceptance_e2e.py --require-services` are worth running after a change to query
paths.

## 3. Verified working end to end

| surface | how it was checked | result |
| ------- | ------------------ | ------ |
| `find_data` | `mdqm_call.py call find_data '{"question":"…"}'` | ok |
| `query` single source | avg credit limit by status | ok, rows + `result_handle` |
| **`query` federated** (`billing.usage_events` ⨝ `crm.customers`) | `group_by customer_status` over Nov–Dec | **1810474 / 956379 / 354603 = E01 ground truth** |
| `get_more` | with the handle from `query` | ok |
| `history` | `{"limit":3}` | ok — **the field is `runs`, not `items`** |
| `feedback` | `{"query_id":"232","ok":true}` | ok, record saved |
| `POST /context` in Hebrew | `{"prompt":"מסגרת אשראי ממוצעת לפי סטטוס"}` | 200, 522 chars (was `""`) |
| `mdqm audit verify` | inside container | "audit chain intact" |
| `mdqm bootstrap run` | rebuilt image, live pg profile | every stage ran; layer loads (15 tables, 2 measures, 6 labels) |
| conversion rediscovery | `mdqm review export-patch <id>` | billing `ROUND({col}/5000)*5000` and erp `{col} * 10`, both `match_rate 1.00`, pending |
| `mdqm sizes measure` | live pg, 8 tables | recorded; `billing.usage_events` 14 partitions |
| plan uses the measurement | `query` with a 400-day window | `3 of 14` partitions, `est_bytes=2703360`, `estimate_source="profile"` |
| start with a missing adapter driver | image installs `.[pg]` only | server starts, logs `adapter 'oracle' is not installed … it is unavailable` |
| `--require-services` suite | all five services up | **182 passed** |

## 4. The owner's scope in detail

### 4a. Table and partition sizes — done

Why it existed: a plan's row estimate was a sum of the source's own `reltuples` — stale (it moves when
the source analyzes, not when rows arrive), byte-less, and shapeless (34 even months and one month
holding 20× the median sum to the same number).

What exists now:

* **Measure.** `SIZES_SQL` in `adapters/postgresql_plan.py` reads `pg_total_relation_size` (heap +
  indexes + TOAST) and `reltuples` per relation; `parse_sizes` turns those rows into
  `TableSize` with `PartitionSize` entries (`models/sizes.py`). A partitioned parent has no storage of
  its own, `reltuples = -1` (never analyzed) is unknown, a boolean is not a size.
* **Store.** `0009_sizes.sql` + `store/sizes.py`: one row per sample **per pass, append-only**, so the
  newest capture answers a plan and two captures show growth. `latest_all()` is the newest capture
  *per table* (each source records under its own instant). Deliberately **not** in the semantic layer:
  a nightly size would change the layer's version hash nightly and invalidate the context index and
  every cache key with it.
* **Quantiles.** `planner/sizes.py`: nearest-rank `min/p1/p25/p50/p75/p99/max` over rows *and* bytes,
  `spread`, and `fat_partitions()` (≥4× the median). Pure, table-tested. `None` means "not reported",
  never zero — an Oracle install reporting `num_rows` but no bytes must not drag the byte median down.
* **Use.** `PartitionEstimate` gains `bytes`, `estimated_from` (`profile`/`catalog`) and `incomplete`
  (a touched partition reported no size → `rows` is a lower bound). `QueryServices.sizes` → the tool
  hands the spec table's profile to `CompileContext.sizes` → `plan_partitions(sizes=…)`.
* **Expose.** `mdqm sizes measure | show | profile <source> <table>`; the answer's `plan` carries
  `est_bytes`, `estimate_source`, `estimate_incomplete`.

Two invariants worth keeping: **without sizes the behaviour is byte-for-byte the old one** (a test
pins it), and **a partition the profile does not know falls back to the catalog**, so a partition
created after last night's measurement is never estimated as free.

Open items, in the order they matter: the `PartitionNote.FAT_PARTITION` note is declared but **not
emitted** (one line in `plan_partitions`, needs the table's whole profile not just the touched
partitions); **nothing schedules** `sizes measure` (not in the bootstrap pipeline, no cron); **only
Postgres** reports sizes (Oracle/SQL Server/Elasticsearch return the capability as unsupported and the
plan falls back, saying `estimate_source: "catalog"`); and **no benchmark** yet compares plan cost with
and without the profile.

### 4b. Fast feedback before the query completes — not started

Today `query` blocks until the whole result is in; the only early exit is a `dry_run` spec (plan +
estimate, no rows). The agreed design (owner's choice):

```text
query(spec)      -> { status: "started", query_id: "q-42", plan: { partitions: 2, est_rows: 84000 } }
get_more(query_id) -> still running, 1 of 2 partitions
                   -> done: summary + sample + handle
```

Constraints to respect:

* `get_more` currently pages a **finished** result by `result_handle`; a job id is a second kind of
  reference. Decide one shape, do not overload `result_handle`.
* Deployment is **one replica** (a correctness rule, `docs/deploy.md` §8), so an in-process job
  registry is acceptable; status must still be visible across calls (a `query_job` row).
* The governor already runs adapters in a bounded thread pool with deadlines and `cancel` — a job is
  that call plus a place to record progress. Per-partition progress is only available where the
  executor knows the partitions (`PartitionPlan.estimate.partitions`).
* 5 acceptance tests in `tests/integration/test_acceptance_e2e.py` assert the blocking shape; they
  must keep passing (a `wait` mode) or be updated deliberately.

### 4c. Model endpoint — not started

A1+A2 runs but answers `no model client configured: nothing proposed`, so the semantic layer is still
built by deterministic stages plus human labels. `mdqm/llm/` already has an OpenAI-compatible client and
`FakeLLM`; what is missing is settings → client → `PipelineContext.llm` in `pipeline/cli.py`
(`context_for` does not set it). Unknowns from `AGENTS.md` §9 stay unknown: the on-prem GLM endpoint is
assumed OpenAI-compatible, its context window and tool-calling reliability unverified. There is **no
endpoint in this environment**, so this can only be verified against `FakeLLM` here; say so in the note.

Once wired, the natural first target is **labels**: `SemanticLayer.labels` exists (§6) but nothing
proposes into it, so Hebrew names are hand-written per thing.

## 5. Bootstrap pipeline — proven end to end

`available_stages()` orders what the caller built against one declared `STAGE_ORDER`
(`D1+D2, D4, D5, D6, D7, D8, D10, A1+A2`); core imports no stage. `build_stages()` in
`pipeline/cli.py` is the wiring: `registry.discover()` first, then adapters → catalogs → layer →
OpenAPI documents → stage list. Which stage appears is **derived, never invented**, and anything
undeliverable is named in the output.

```bash
docker compose --profile pg --profile server up -d --build --wait server
docker exec mdqm-server-1 python -m mdqm.cli bootstrap run
docker exec mdqm-server-1 python -m mdqm.cli review list
docker exec mdqm-server-1 python -m mdqm.cli review export-patch <id>
```

Measured: every stage ran, the layer loads, and D8 **rediscovers both storage forms** of the
credit-limit measure (billing `ROUND({col} / 5000) * 5000`, erp `{col} * 10`, `match_rate 1.00`,
pending approval).

**A rule this run earned:** a stage's evidence must not land where the layer reads. D5 wrote
`generated/profile_<source>.yaml` at the top level, the loader merges every top-level file there into
one document, so the first bootstrap made the whole layer unloadable — `/context` fell back, D6/D10
reported "the semantic layer did not load", and the next server start **refused to run**. Sections may
now name one directory level (`generated/profiles/<source>.yaml`), which the loader skips; `.`/`..`
segments are still refused.

Known limits: `A1+A2` gets `profiles=None` (no artifact handoff between stages, so a name proposal
carries catalog evidence only); a measure with no identity column makes D8 pick a reference and say so;
and with two or more measures declaring pairs, all `D8` instances share one stage name, so none can be
skipped as unchanged.

## 6. Semantic layer, `/context`, and Hebrew

`/context` returns measures, joins, value maps and table facts for an English prompt; a Hebrew prompt
returned `""` because `value_maps.synonyms` was the only section that could hold a Hebrew word.

`SemanticLayer.labels` (keyed `table:<source.table>`, `measure:<name>`, `column:<qualified>`, a list of
words per key) fixes it: the context index reads it from the layer itself, a caller can override per
key, a key naming nothing is refused at build time, and labels enter `version_hash`. Nothing is
language-specific — the same field carries a nickname or an abbreviation.

Measured on `examples/semantic_example.yaml`: `מסגרת אשראי ממוצעת לפי סטטוס` went 0 → 5 items; the
English question is unchanged; a prompt naming nothing still injects nothing.

Not done: nothing **proposes** labels (§4c), and A1+A2 has no target/validator for one.

## 7. SQL-injection work: the chokepoint and the lint rule

* **`src/mdqm/sql/execution.py` is the single chokepoint.** `prepare()` runs the guard and raises
  `StatementRefused`; `execute_one()` is the only function that hands dynamic source SQL to a driver;
  `run_local()` is for our own metadata SQLite and the local DuckDB engine; `set_local()` validates both
  halves of a `SET`; `local_ident()`/`local_idents()` validate and quote an identifier (the one thing a
  driver cannot bind).
* **`scripts/check_sql_sinks.py`** is our own data-flow rule, with 31 table-driven tests
  (`tests/unit/test_sql_sinks_checker.py`). Writing them found two real holes: a name bound to
  interpolated text was never reported (the two-line injection form), and `sql.SQL(…).format(
  sql.Identifier(col))` was flagged although psycopg quotes the identifier itself.
* **The limit, stated in the code:** a bare parameter is not traced. `def go(cursor, text):
  cursor.execute(text)` is not reported — the question there is "was the guard run", which
  `prepare()`/`execute_one()` answer at execution time.
* `TRUSTED_ASSEMBLERS` exempts `mdqm/sql/execution.py` by path (one entry); `TRUSTED_PREFIXES` exempts
  `mdqm/testing/` (the loaders emit DDL, which the read-only guard refuses, and the query path never
  runs them). A new exemption is the signal to use the chokepoint instead.
* 29 adversarial tests in `tests/unit/test_sql_injection_protection.py`; the chokepoint has its own in
  `tests/unit/test_sql_execution.py`.

## 8. Lint and type state

* `ruff check .` clean. Families: `E,F,I,W,S,B,BLE,TRY,RUF,A,DTZ,PERF,C4,PIE,RET,UP,SIM,ASYNC`.
* Five rules are **ratcheted** in `pyproject.toml` with counts and reasons; `TRY003` and `RUF001-3` are
  off with reasons. Burn the ratchet down; do not add to it without a count and a reason.
* **pyright reports nothing** (`python -m pyright` → 0 errors, 0 warnings). Getting there changed `src`
  in three ways worth knowing: `SourceAdapter` no longer requires the four optional methods (they are
  `OptionalSourceMethods`), `mdqm.testing.loaders` imports its engines lazily so a deployment without an
  extra can still import the package, and `ConversionStage.pairs` is a public read-only property.
* pyright is **not** in `scripts/check.py`, so the debt can grow back: adding it is one line plus a
  decision about latency.

## 9. Deployment facts

* `Dockerfile` takes `--build-arg MDQM_EXTRAS=pg` (default `pg`; `pg,oracle`, `pg,mssql`, or all three).
  `docs/deployment.md` §1a has the per-site matrix and the command that proves which drivers arrived. A
  missing driver makes its adapter unavailable; the server still runs.
* `/data`, `/cache` **and `/semantic`** are volumes. `/semantic` was not, and a recreate silently lost
  the whole generated layer.
* Air-gap rules: no runtime `pip install`, no DuckDB extensions, no network at hook time.
  `.pre-commit-config.yaml` is `repo: local` throughout, every hook goes through
  `scripts/precommit_hook.py`. Hooks: `ruff check --fix`, `ruff format`, `check_configs.py`,
  `check_secrets.py`, `sql-sinks` on commit; `mypy --strict` and unit tests on push.
* One replica, ever (`docs/deploy.md` §8) — a correctness rule, not a default.

## 10. Traps that cost time (avoid repeating them)

1. **A delegated agent often produces no commits.** Three subagent attempts this session burned ~15
   minutes of tokens each and landed nothing (a provider outage). Always
   `git log --oneline main..task/<ID>` before believing a completion report, and tell a delegate to
   create its worktree first and commit per step. A `subagents_run` worktree lives **outside** this
   workspace (under `P:\.pi\workflows\…`), so its commits cannot be read or merged here at all.
2. **`ruff format .` reformats unrelated files** — it left a stray blank line in
   `src/mdqm/adapters/registry.py` twice. Format only the files you touched.
3. **`run_local(cursor, text)` with the default `params=None` breaks sqlite3** (`parameters are of
   unsupported type`). Pass `()` for metadata SQLite; DuckDB tolerates `None`.
4. **Heredocs mangle backslashes.** Writing a regex through `<<'PY'` put a NUL byte into a Python
   source file. Use the `edit` tool, or build the character with `chr(92)`.
5. **A worktree needs its own `pyrightconfig.json`** (`venvPath` pointing at the main checkout), because
   the repo config excludes `.worktrees` and `venvPath="."` resolves to the worktree. Delete it before
   committing.
6. **pydantic models are not dataclasses:** `dataclasses.replace` raises `TypeError`. Use
   `model_copy(update={…})`.
7. **The lens diagnostics can name stale lines** after you edit a file it was already scanning; when it
   disagrees with `ruff`/`mypy`/`pyright`/the LSP probe on the same file, trust the tools and say so.
8. **`git checkout -- <file>` cannot restore an untracked file.** Commit early; that is the undo.
9. **A container rebuild is required after any `src/` change**; the running image does not pick up
   local files.

## 11. Suggested order for the next session

1. **Async query handle + poll** (§4b) — the owner's second ask, design already agreed.
2. **Model endpoint** (§4c) — the third ask; verify with `FakeLLM` here and say that the real endpoint
   is unverified.
3. **Emit the fat-partition note** (§4a) — the numbers exist; the note needs the table's whole profile.
4. **Schedule `sizes measure`** (§4a) — a bootstrap stage or a cron entry, so the profile is fresh.
5. **Size readers for Oracle / SQL Server / Elasticsearch** (§4a) — each is one query per engine.
6. **`pyright` in `scripts/check.py`** (§8) so the cleared backlog cannot grow back silently.
7. **A proposal nobody answers** — 30 join candidates and 2 measure conversions sit pending with no
   ageing or escalation.

## 12. Where the history lives

`progress/<TASK-ID>.md` is the ledger: 69 backlog tasks are `status: done`, plus
`BOOTSTRAP-PIPELINE.md`, `SQLI-CHECKER.md`, `PYRIGHT-TESTS2.md`, `HEBREW-LABELS.md` and `SIZES.md`
(`status: partial` — the measurement and the plan are wired; the note, the scheduler and the other
engines' readers are not). `docs/DESIGN.md` is the design. The task board being closed does **not** mean
every feature is proven end to end; §4b and §4c are features nobody has started.
