# Handoff: what is done, what is not, and exactly where to start

The live-service proof after this note is `progress/LIVE-SERVICES.md` and the Status section of
`README.md`. The paragraphs below were written at `main = a55d0ab` and still describe that tree;
they are not the current measurement.

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
| size profile → query optimization | **done and measured** (§4a); all four engines' size readers proven live (`progress/SIZE-READERS-LIVE.md`), the Oracle *bytes* branch excepted |
| fast feedback before completion | **done and proven live** — async job + poll, `progress/ASYNC-QUERY.md` and `progress/ASYNC-LIVE.md` (§4b) |
| model endpoint (so A1+A2 can propose names, incl. Hebrew labels) | **wired, labels proposed; the real endpoint unverified** — `progress/LLM-WIRING.md` (§4c) |

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

Closed since: `PartitionNote.FAT_PARTITION` is emitted (median over the table's whole measured profile,
names on `PartitionPlan.fat_partitions`); a `sizes` bootstrap stage schedules the measurement once a
day per source (heavy: quiet window + per-source gate); Oracle, SQL Server and Elasticsearch have size
readers behind the same `parse_sizes`. See `progress/SIZES-FOLLOWUP.md`.

Those three readers have now run against live engines
(`pytest tests/integration/test_size_readers_live.py --require-services`), which is how the Oracle one
turned out to be broken: 23ai has no `ALL_SEGMENTS`, so the query now probes `ALL_SEGMENTS` →
`DBA_SEGMENTS` → none and reports bytes as "not reported" when the account may read neither
(`progress/SIZE-READERS-LIVE.md`).

Open items, in the order they matter: the Oracle **bytes** branch (an install that grants
`DBA_SEGMENTS`) is the only size path nobody has executed end to end; `bootstrap run` with the new
`sizes` stage is unit-tested only; `PartitionNote.SIZE_PROFILE_APPLIED` is still never emitted; the fat
partitions' names reach `PartitionPlan.fat_partitions` but not yet `plan.notes` in
`server/tools/query.py`; and **no benchmark** compares plan cost with and without the profile.

## 4b. Fast feedback before the query completes — done and proven live (_ASYNC-QUERY_)

The paragraphs below are the agreement; the code now does it. `query` takes `mode: "blocking"`
(the default, unchanged) or `"async"`; the state lives in the `query_job` table (migration 0011) and
`progress/ASYNC-QUERY.md` + `progress/ASYNC-LIVE.md` record what is verified.

Measured against the rebuilt server image and real PostgreSQL:
`query(mode="async")` answers `{status: "started", query_id: "j_2", plan: {partitions_scanned: 3,
partitions_total: 14, est_rows: 18946}}` immediately, and `get_more(query_id="j_2")` later answers
`done` with the summary, the sample, `partitions_done: 3` and a `result_handle` that pages through
`get_more(result_handle=…)`. `tests/integration --require-services` is one green run over all five
services, acceptance suite included.

One honest nuance worth knowing: a job answered from the segment cache reports
`partitions_done: null`, because no source plan was asked for and no partition was scanned. `null`
means "not reported"; inventing `0` or `3` there would be a lie.

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

### 4c. Model endpoint — wired (_LLM-WIRING_), real endpoint unverified

`mdqm/llm/factory.py:model_client(settings)` is the one translation from `settings.model` to a client;
`pipeline/cli.py:context_for` puts it on `PipelineContext.llm`, and `bootstrap run` prints
`describe_model` **before** the stage list. Configure with `MDQM_MODEL__BASE_URL` / `MDQM_MODEL__MODEL`
(env or `mdqm.yaml`); the key is a secret *reference* (`MDQM_SECRET_MODEL_API_KEY` or `secrets.dir`),
resolved lazily on the first call, never logged, never in YAML. Nothing configured → no client, which is
a supported deployment; half configured → no client plus one line naming the missing field.

A1+A2 now proposes `SemanticLayer.labels` per table, column and measure (keyed `table:`/`column:`/
`measure:`), i.e. the payload whose approval actually loads and that answers a Hebrew prompt (§6).
Garbage words are refused (`RejectionCode.NOT_WORDS`), a per-key cap is counted in `applied_fixes`, and
a model failure fails the stage instead of producing a green run that hid an outage.

Unverified, and unverifiable here: there is no endpoint in this environment, so answer quality, context
window and `response_format`/tool-calling support stay guesses; every test drives `FakeLLM`. A
deployment must run `mdqm bootstrap run` against the real server and read the rejections. A
`name`/`synonym` proposal still cannot be approved — it targets a `names` section `SemanticLayer` does
not declare; that needs a design decision (see `progress/LLM-WIRING.md`, Deviations).

## 5. Bootstrap pipeline — proven end to end

`available_stages()` orders what the caller built against one declared `STAGE_ORDER`
(`D1+D2, D4, D5, D6, D7, D8, D10, A1+A2, sizes`); core imports no stage. `build_stages()` in
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
* **pyright is clean** — run `python -m pyright` for the current state, never this sentence. Getting
  there changed `src`
  in three ways worth knowing: `SourceAdapter` no longer requires the four optional methods (they are
  `OptionalSourceMethods`), `mdqm.testing.loaders` imports its engines lazily so a deployment without an
  extra can still import the package, and `ConversionStage.pairs` is a public read-only property.
* pyright is **in** `scripts/check.py` now, and that is why the debt has to stay at zero: it was not,
  and 17 errors (plus 3 `check_sql_sinks` findings) were sitting on `main` before it went in. Run
  `python scripts/check.py` rather than trusting this bullet; `--fast` skips both type checkers.

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

1. **The real model endpoint** (§4c) — wiring and label proposals are in, but only `FakeLLM` verified
   them; run `mdqm bootstrap run` against the on-prem server and read what the validator rejected.
   This is the only ask whose proof is missing.
2. **A `names` section** (§4c Deviations) — until the layer can hold a display name and a one-sentence
   description, A1's first two fields pay for model calls whose answer cannot be approved.
3. **Oracle bytes on a granting install** (§4a) — the one size path nobody has executed: a
   `DBA_SEGMENTS` grant would take it.
4. **`bootstrap run` with the `sizes` stage against live sources** (§4a) — the stage is unit-tested
   only; nothing has scheduled a real measurement yet.
5. **`plan.notes` carries the fat partitions** (§4a) — the names exist on
   `PartitionPlan.fat_partitions`; one line in `server/tools/query.py::_plan` exposes them.
6. **A proposal nobody answers** — 30 join candidates and 2 measure conversions sit pending with no
   ageing or escalation.

Done since the last session: the async job + poll path is proven live (`progress/ASYNC-LIVE.md`), the
three extra size readers are proven live and the Oracle one was fixed on the way
(`progress/SIZE-READERS-LIVE.md`), `A10` is closed (`progress/A10.md`, which also removed the base
image's `pip`), and the merged worktrees from earlier sessions were pruned (67 of them; the 10 whose
branches were not ancestors of `main` are still there on purpose).

## 12. Where the history lives

`progress/<TASK-ID>.md` is the ledger: **every backlog task is `status: done`**, plus the notes
written outside the board — `BOOTSTRAP-PIPELINE.md`, `SQLI-CHECKER.md`, `PYRIGHT-TESTS2.md`,
`HEBREW-LABELS.md`, `SIZES.md`, `ASYNC-QUERY.md`, `LLM-WIRING.md`, `SIZES-FOLLOWUP.md`, `A10.md`,
`ASYNC-LIVE.md`, `SIZE-READERS-LIVE.md` (the ones marked `partial` say so only where the proof needed a
service that does not exist here, and each names it). `docs/DESIGN.md` is the design.

The task board being closed is not the same as every path being executed: the honest gaps are the real
on-prem model endpoint (no endpoint here), the Oracle bytes branch (no segment grant here), `bootstrap
run` with the `sizes` stage against live sources, and the two design decisions listed in §11.
