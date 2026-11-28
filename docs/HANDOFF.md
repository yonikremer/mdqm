# Handoff: what is done, what is not, and exactly where to start

Written at `main = 2ced811` (working tree clean). Everything below was measured by running something,
not by reading code — the commands are included so you can re-measure instead of trusting me. Where a
number would go stale, the command is written instead: run it.

Read `progress/` for the ledger: `BOOTSTRAP-PIPELINE.md` and `SQLI-CHECKER.md` are the two notes from
this session, both honest about what they could not verify.

---

## 1. How to bring the stack up

```bash
export PATH="/c/Users/yonik/AppData/Local/Programs/DockerDesktop/resources/bin:$PATH"   # Docker Desktop
.venv/Scripts/python.exe scripts/dev.py up pg          # or: docker compose --profile pg --profile server up -d --build --wait server
.venv/Scripts/python.exe scripts/dev.py status
```

* Server: `http://127.0.0.1:58765` — `GET /healthz` returns 200.
* MCP client for manual calls: `python scripts/mdqm_call.py call <tool> '<json args>'`
* Token for HTTP endpoints: `docker/server/tokens.json` (gitignored), first entry's `token`.
* **The CLI is `python -m mdqm.cli <group>`.** `python -m mdqm` does *not* exist (no `__main__.py`).
* Metadata database inside the container: `/data/metadata.sqlite` (tables include `query_history`,
  `audit_log`, `feedback`, `pipeline_run`, `review_proposals`). Inspect with
  `docker exec mdqm-server-1 python -c "…sqlite3…"`.
* **Docker Desktop's engine died mid-session and could not be restarted from the agent's workspace**
  (`open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`). Every
  container-level claim below was measured *before* that; the two things it blocked are marked
  "unverified" in sections 4 and 5.
* Compose needs the checkout's `.env` for interpolation, and a worktree has none:
  `docker compose --env-file /c/Users/yonik/mdqm/.env …`.

## 2. The gate you must keep green

```bash
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m mypy
.venv/Scripts/python.exe scripts/check_sql_sinks.py src/mdqm    # 0 findings expected
.venv/Scripts/python.exe -m pytest tests/unit -q
.venv/Scripts/python.exe -m pytest tests/integration --require-services -q
.venv/Scripts/python.exe scripts/check.py                        # ruff + mypy + unit, the CI gate
```

Also worth running after a change to query paths:

```bash
.venv/Scripts/python.exe scripts/bench_glm.py score --answers bench/answers   # vs the E01 ground truth
.venv/Scripts/python.exe -m pytest tests/integration/test_acceptance_e2e.py --require-services -q
```

## 3. Verified working end to end (through the running server, before the engine died)

| surface | how it was checked | result |
| ------- | ------------------- | ------ |
| `find_data` | `mdqm_call.py call find_data '{"question":"…"}'` | ok |
| `query` single source | avg credit limit by status | ok, rows + `result_handle` |
| **`query` federated** (`billing.usage_events` ⨝ `crm.customers`) | `group_by customer_status` over Nov–Dec | **1810474 / 956379 / 354603 = E01 ground truth** |
| `get_more` | with the handle from `query` | ok |
| `history` | `{"limit":3}` | ok — **the field is `runs`, not `items`** |
| `feedback` | `{"query_id":"232","ok":true}` | ok, record saved |
| `GET /healthz` | curl | 200 |
| `POST /context` | `{"prompt":"…"}` at top level + bearer token | 200, returns `additionalContext` (see section 6 for the Hebrew caveat) |
| `mdqm audit verify` | inside container | "audit chain intact: 234 row(s) verified" |
| `mdqm review list` | inside container | "nothing to review" |
| start with a missing adapter driver | image installs `.[pg]` only | server starts, logs `adapter 'oracle' is not installed in this deployment … it is unavailable` |

## 4. Bootstrap pipeline: wired, but the end-to-end run is still unproven

The command that used to do nothing now has stages to run. What was wrong (both fixed):

1. `src/mdqm/pipeline/core.py::available_stages()` returned `()` on purpose. It now **orders** the
   stages a caller built against one declared `STAGE_ORDER`
   (`D1+D2, D4, D5, D6, D7, D8, D10, A1+A2`). Core still imports no stage: every stage imports core,
   so the construction cannot live there.
2. `src/mdqm/pipeline/cli.py` never asked the adapter registry to discover its entry points, so every
   source answered `no adapter registered for 'postgresql' (registered kinds: none)`. `build_stages()`
   now calls `registry.discover()` first and reports what it found.

Which stage appears is **derived, never invented**, and anything undeliverable is named in the run
output: `D1+D2`/`D5`/`D6`/`D7`/`D10` need only adapters, `D4` needs a `sources[].spec` document, `D8`
needs a measure declaring two or more readable source columns (the layer is the declaration —
`ColumnPair` is not a guess), `A1+A2` needs a catalog.

**Unverified, and it is E01's last acceptance claim:** `mdqm bootstrap run` inside the rebuilt image
executing those stages and **rediscovering the 5000 / 5000-then-/10 conversions** from data. Docker was
down before this could run. Finish it with:

```bash
docker compose --env-file /c/Users/yonik/mdqm/.env --profile pg --profile server up -d --build --wait server
docker exec mdqm-server-1 python -m mdqm.cli bootstrap run
docker exec mdqm-server-1 python -m mdqm.cli review list      # the conversion proposals land here
```

Known limits of the wiring, stated rather than hidden: `A1+A2` gets `profiles=None` (D5's profiles are
produced at run time and there is no artifact handoff between stages); a measure with no identity
column makes D8 pick a reference and say so; and with two or more measures declaring pairs, all `D8`
instances share one stage name, so none of them can be skipped as unchanged.

## 5. SQL-injection work: the chokepoint, the lint rule, and its limits

Design (agreed with the owner, do not re-litigate):

* **`src/mdqm/sql/execution.py` is the single chokepoint.** `prepare(text, dialect=…, params=…)` runs
  `mdqm.sql.guard.guard` and raises `StatementRefused`; `execute_one(cursor, statement)` is the only
  function that hands dynamic source SQL to a driver; `wrapped(...)` assembles EXPLAIN wrappers;
  `run_local(cursor, text, params)` is for our own metadata SQLite and the local DuckDB engine;
  `set_local(engine, name, value)` validates both halves of a `SET`; `local_ident(name)` /
  `local_idents(names)` validate and quote an identifier, which is the one thing a driver cannot bind.
* **`scripts/check_sql_sinks.py` is our own lint rule** and now has tests
  (`tests/unit/test_sql_sinks_checker.py`, table-driven, fixtures on disk). Writing them found two real
  holes: a name bound to interpolated text was never reported (the two-line injection form), and
  `sql.SQL(…).format(sql.Identifier(col))` was flagged although psycopg quotes the identifier itself.
  Resolution is provenance-based: literals and compiled objects (`Fragment`/`FederatedPlan`, by
  construction, annotation or the compiler's attribute) are safe; a name bound to anything else is a
  finding.
* **The limit, in the code and here:** a bare parameter is not traced. `def go(cursor, text):
  cursor.execute(text)` is not reported, because the question there is "was the guard run", which
  `prepare()`/`execute_one()` answer at execution time. The rule catches text being *built*.
* `TRUSTED_ASSEMBLERS` exempts `mdqm/sql/execution.py` by path (one entry), and `TRUSTED_PREFIXES`
  exempts `mdqm/testing/` (the synthetic-data loaders emit DDL, which the read-only guard refuses, and
  the query path never runs them). If you add a call site that needs an exemption, that is the signal
  to use the chokepoint instead.
* `29` adversarial protection tests live in `tests/unit/test_sql_injection_protection.py`; the
  chokepoint has its own in `tests/unit/test_sql_execution.py`.
* The pre-commit `sql-sinks` hook now scopes `pipeline/` and `eval/` as well — the old hand-written
  directory list silently missed both.

**Unverified:** `tests/integration --require-services`. The loaders' emitted DDL did not change (the
conversions were mechanical), but that is a claim until it runs against real services.

## 6. `/context` and Hebrew: a real gap, and it is a model question

The endpoint works: English prompts return 500–1100 characters of measures, joins, value maps and
table facts. Hebrew returns `""` — because the only place the layer can hold Hebrew today is
`value_maps.synonyms`. "פעיל" and "עזב" match; "מסגרת אשראי ממוצעת" matches nothing, since a measure
and a table have no name or label per language.

**Decision needed from you:** should `Measure` and `TableSemantics` carry per-language labels
(`he`/`en`) that the context index matches, or is a Hebrew question expected to go through value
synonyms and English identifiers only? The code cannot pick for you, and inventing a field name would
be a design change.

## 7. Deployment facts and gaps

* `Dockerfile` takes `--build-arg MDQM_EXTRAS=pg` (default `pg`; `pg,oracle`, `pg,mssql`, or all
  three). `docs/deployment.md` §1a has the per-site matrix and the command that proves which drivers
  actually arrived in the image. A missing driver makes its adapter unavailable; the server still runs.
* `/data` and `/cache` are volumes, and **`/semantic` now is too** (`server-semantic`). It was not:
  a container recreate silently lost the whole generated layer, after which `/context` fell back to
  curated-only without saying so.
* Air-gap rules still apply: no runtime `pip install`, no DuckDB extensions, no network at hook time.
  `.pre-commit-config.yaml` is `repo: local` throughout, and every hook goes through
  `scripts/precommit_hook.py` (`language: system` alone runs the wrong interpreter).
* Pre-commit hooks: `ruff check --fix`, `ruff format`, `check_configs.py`, `check_secrets.py`,
  `sql-sinks` on commit; `mypy --strict` and unit tests on push.

## 8. Lint state (so you do not re-discover it)

* `ruff check .` → clean. Rule families: `E,F,I,W,S,B,BLE,TRY,RUF,A,DTZ,PERF,C4,PIE,RET,UP,SIM,ASYNC`.
* Five rules are **ratcheted** in `pyproject.toml` with counts and reasons (`PERF401`, `BLE001`,
  `TRY004`, `RUF043`, `DTZ001`). `TRY003` and `RUF001-3` are off with reasons. Burn the ratchet down;
  do not add to it without a count and a reason.
* **pyright** still reports findings, almost all in `tests/`: literals where pydantic wants enums or
  typed models, fixtures annotated `-> X` that are generators, and doubles that do not satisfy their
  protocol. Measure with `python -m pyright`; none of it gates CI.
  `progress/PYRIGHT-TESTS2.md` records what was cleared and lists what is left, file by file. The one
  structural fix already landed: `SourceAdapter` no longer requires the four optional methods (they
  are `OptionalSourceMethods`), so the type model now matches `REQUIRED_METHODS` and the probe.
  A worktree needs its own `pyrightconfig.json` (`venvPath` pointing at the main checkout), because
  the repo config excludes `.worktrees` — delete that file before committing.

## 9. Traps that cost time (avoid repeating them)

1. **A delegated agent often produces no commits.** Observed repeatedly, including three attempts in
   this session that each burned ~15 minutes of tokens and landed nothing. Always
   `git log --oneline main..task/<ID>` before believing a completion report, and tell the delegate to
   create its worktree first and commit per step.
2. **`ruff format .` reformats unrelated files.** It left a stray blank line in
   `src/mdqm/adapters/registry.py` twice. Run the formatter on the files you touched.
3. **`run_local(cursor, text)` with the default `params=None` breaks sqlite3**
   (`ProgrammingError: parameters are of unsupported type`). Pass `()` for metadata SQLite; DuckDB
   tolerates `None`.
4. **Heredocs mangle backslashes and escape sequences.** Writing a regex through `<<'PY'` put a NUL
   byte into a Python source file (`source code string cannot contain null bytes`). Use the `edit`
   tool, or build the character with `chr(92)`.
5. **`git checkout -- <file>` cannot restore an untracked file.** Committing early is the only undo.
6. **pyright honours `# type: ignore` too.** Deleting a mypy-only ignore in a test un-suppresses real
   pyright errors — fix them, but know why they appeared.
7. **A container rebuild is required after any `src/` change**; the running image does not pick up
   local files.
8. **Compose in a worktree has no `.env`** (it is gitignored): pass `--env-file` pointing at the main
   checkout.

## 10. Suggested order for the next session

1. **Run the bootstrap end to end** (§4) — the last unproven acceptance claim, and the command is
   written out for you.
2. **Finish the pyright debt in `tests/`** (§8) — the remainder is listed per file in
   `progress/PYRIGHT-TESTS2.md`; every entry is the same shape as the classes already cleared.
3. **Decide the Hebrew labelling question** (§6), then either extend the layer models or document that
   Hebrew questions are matched through value synonyms only.
4. Re-run `tests/integration --require-services` (§5) once Docker is back, to turn the loader claim
   into a measurement.

## 11. Where the history lives

`progress/<TASK-ID>.md` is the ledger: 69 backlog tasks are `status: done`, plus
`BOOTSTRAP-PIPELINE.md` and `SQLI-CHECKER.md` from this session. The task board being closed does not
mean every feature is proven end to end — §4 and §5 are the proof that the two are not the same
thing. `docs/DESIGN.md` is the design.
