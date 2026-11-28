status: done
task: SIZES-FOLLOWUP (docs/HANDOFF.md §4a and §8, items 3-6 of §11)
agent: pi (sizes-followups session)
branch: task/sizes-followups
commits: 6ed1790 (fat-partition note), 5b66d38 (sizes stage), 730f3a9 (Oracle/SQL Server/ES readers),
         4caf3e7 (clear the 17 pyright errors main already had), 4fa03a4 (pyright in the gate),
         8949f9c (stats collect through the chokepoint), and the note commit that adds this file
tests run: see "Tests run" below (the exact commands, all green)

## Summary
The three size-profile follow-ups that were "open items" in `progress/SIZES.md` are closed, and the
gate cannot silently lose the type checking again:

1. **`PartitionNote.FAT_PARTITION` is emitted.** `plan_partitions` computes it over the table's
   **whole** measured profile (`fat_partitions(sizes)`), not over the touched partitions, and names
   the offenders on `PartitionPlan.fat_partitions`. Found and fixed a real hole on the way: the
   injected-`default_window` path dropped `sizes` entirely, so a filterless query on a table that
   allows one was estimated from the catalog even when a profile existed.
2. **`sizes measure` is scheduled.** A new bootstrap stage `sizes` (`pipeline/stages/sizes.py`) is
   declared in `STAGE_ORDER`, derives its source list from adapters that declare `sizes()`, is
   heavy (quiet-window gated, per-source gated), skips on an unchanged input hash (the day is part of
   the hash, exactly as D5/D6), and records every source under one `ctx.exec_ts` so a pass is one
   capture. No cron, no second scheduler.
3. **Oracle, SQL Server and Elasticsearch report sizes.** One statement each, pure parses, `None`
   for "not reported" (never 0), and the probe claims `size_stats` only when the read really answers
   (Oracle says so even when the account may not read segment bytes, and says which half is missing).
4. **`pyright` is in `scripts/check.py`**, with `tests/unit/test_check_gate.py` pinning the gate's
   composition so deleting the line fails a test. `pyright` is now a declared dev dependency.

## Tests run
* `python -m ruff check .` → all checks passed
* `python -m mypy` → success, 166 source files
* `python -m pyright` → 0 errors, 0 warnings (see "Not verified" for the worktree config used)
* `python scripts/check_sql_sinks.py src/mdqm` → 0 findings (was 3 on `main`)
* `python -m pytest tests/unit -q` → 4655 passed
* `python scripts/check.py` → all checks passed, 160 s wall clock
* `python -m pytest tests/unit/test_partition_planner.py tests/unit/test_stage_sizes.py
  tests/unit/test_size_readers.py tests/unit/test_sizes.py tests/unit/test_check_gate.py -q` → the
  new blocks pass on their own

pyright's cost in the gate: **~17.5 s** (three warm runs: 17.0 / 18.0 / 17.5 s) out of 160 s for the
whole `scripts/check.py`, i.e. about a tenth of it. Cold, it measured up to ~20 s. mypy is the other
slow type step; `--fast` skips both and keeps ruff + unit tests.

## Baseline this started from (main @ 9086472, measured, not assumed)
`docs/HANDOFF.md` §2/§8 claimed the gate was clean. It was not:
* `python -m pyright` → **17 errors, 0 warnings**: 8 in `src/mdqm/stats/postgres.py`
  (`psycopg.rows` is an unimported submodule; `cursor.execute(<str>)` no longer type-checks against
  psycopg 3.3's `QueryNoTemplate`), 5 in `tests/unit/test_dev_env.py` (assigning module constants on
  a `ModuleType`), 1 in `tests/unit/test_model_validators.py` (`Iterator[AST]` vs `Iterable[stmt]`),
  1 in `tests/unit/test_stats.py` (`"NULLIF" in <str | None>`), 2 in `tests/integration/test_e2e.py`.
* `python scripts/check_sql_sinks.py src/mdqm` → **3 findings**, all in the stats lane:
  `stats/collect.py:230`, `stats/postgres.py:99`, `stats/postgres.py:166`.

Both were fixed for real, not suppressed — see "Decisions" — so `pyright` could join the gate at zero
instead of at 17.

## Decisions
* **The fat note's median is the table's, not the window's.** Using only the touched partitions would
  compare a fat partition with itself and never report it. Pinned by a case whose window *is* the
  fat partition: touched-only would give no note, the whole profile does.
* **The note carries the note; the names live on the plan.** `PartitionPlan.notes` is a set of
  enum values, so `fat_partitions: tuple[str, ...]` carries which partitions. `CompiledQuery.notes`
  therefore says `fat_partition` today; putting the names in `plan.notes` needs one line in
  `src/mdqm/server/tools/query.py` (`_plan`), which belongs to the async-query agent — see Plan issues.
* **The stage is named `sizes`, not `D11`.** DESIGN §10's stage table ends at D10; claiming a number
  that the design does not define would be an invented fact. It sits last in `STAGE_ORDER` because it
  writes only to the metadata store and reads no stage output, so its position cannot matter.
* **A source that refuses (`NotSupported`) is a capability fact, not a failure.** It appears in the
  stage's `skipped`, the other sources still record, and the stage still reports `RAN`.
* **One shared seven-column row shape** (`src/mdqm/adapters/sizes.py`) for PostgreSQL and Oracle,
  because both engines name a relation per partition; SQL Server has no relation per partition, so it
  builds the `scheme.number (filegroup)` name in Python with the **same** function its partition
  bounds use (`partition_name`), and a test asserts the size names equal the bound names.
* **Oracle bytes = the table segment only.** Index segments are deliberately not summed: Oracle has no
  per-table total and inflating a *scan* estimate with index bytes would be wrong. A read-only account
  without a segment grant gets `None` (not reported), which the probe names explicitly.
* **The pyright/sinks fixes are typing-only.** `stats/postgres.py` now goes through
  `prepare()`/`execute_one()` (the project's chokepoint) instead of a bare `cursor.execute(<str>)`,
  and `stats/collect.py` uses `run_local()`: both are the intended shape rather than a suppression,
  and they cleared the three `check_sql_sinks` findings as a side effect. No `# type: ignore`, no
  file excluded from `pyrightconfig`, no rule added to the ratchet.

## Deviations
* `docs/HANDOFF.md` §8 had two stale claims; both were rewritten as *commands to run* (per the
  never-hard-code-a-number rule), not as a new pair of numbers.
* `pyproject.toml`: `pyright>=1.1` added to the `dev` extra. It was installed in the venv but never
  declared, so `scripts/check.py` would have depended on whatever a laptop happened to have.

## Dependencies
* `pyright>=1.1` (dev extra only). Justification: it is now a step of `scripts/check.py`; an
  undeclared, unpinned dev tool in the gate is a gate that depends on the machine. A wheel-shipping,
  pure-Python dev dependency; nothing is downloaded or contacted at runtime, and nothing ships.

## Not verified
* **The three size readers were never run against a live engine.** Docker was not running, so the
  statements in `adapters/oracle.py` (`ALL_TABLES` + `ALL_TAB_PARTITIONS` + `ALL_SEGMENTS`),
  `adapters/sqlserver.py` (`sys.dm_db_partition_stats` + `sys.destination_data_spaces`) and
  `adapters/elastic.py` (`_cat/indices?bytes=b`) are **unverified against real Oracle / SQL Server /
  Elasticsearch**. What *is* verified: each statement parses under its sqlglot dialect and passes the
  read-only guard, and each parse turns captured row shapes into the models (including the `None`,
  `Decimal`, `"-"`, missing-filegroup and closed-index cases). The privilege questions are the open
  ones: Oracle needs a segment grant for bytes, SQL Server needs `VIEW DATABASE STATE`, and an
  Elasticsearch alias only shows up as partitions if the alias is visible to the account.
* The `bootstrap run` with the new `sizes` stage was **not** run end to end against the compose stack
  (no Docker here). Its registration, skip, refusal and direct-call rules are unit-tested, and the
  stage calls exactly what `mdqm sizes measure` calls.
* pyright was run in this worktree with a temporary `pyrightconfig.json` whose `venvPath` points at
  the main checkout's `.venv` (HANDOFF §10 trap 5: the repo config resolves `venvPath="."` to the
  worktree, which has no `.venv`). That file was deleted before the last commit; in a normal checkout
  `pyproject.toml [tool.pyright]` applies and gives the same result.

## Plan issues
* **`PartitionNote.SIZE_PROFILE_APPLIED` is declared and still never emitted** (same class of bug as
  `FAT_PARTITION` was). It is one line — `estimate.estimated_from == "profile"` — but it would add a
  note to every plan that uses a profile, which is a behaviour change nobody asked for tonight.
  Worth a small task.
* **The names behind the fat note do not reach the agent yet.** `_plan()` in
  `src/mdqm/server/tools/query.py` maps `partition_plan.notes` to `note.value`; the names are on
  `PartitionPlan.fat_partitions`. One line there (owned by the async-query agent) finishes it.
* **`docs/HANDOFF.md` §4a still says "only Postgres reports sizes"** and its §11 list still shows
  items 3-6 as suggestions; §2's "gate you must keep green" list is now what `scripts/check.py`
  runs. Left alone on purpose: only §8 was in scope, and §4a is the handoff author's summary.
* **`README.md`'s Status section** may repeat the "Postgres only" claim; not checked, not edited.
* **`ruff format --check` disagrees with committed files** in `src/mdqm/planner/partitions.py` (four
  long `_no_filter(...)` one-liners that predate this work). `ruff check` is the gate; reformatting
  them would have touched lines another lane wrote, so they were left.
