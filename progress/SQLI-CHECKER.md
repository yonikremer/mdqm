status: done
task: sql-injection work, part 2 (docs/HANDOFF.md section 5.1 and 5.2)
agent: pi (main session)   branch: task/sql-sinks   commit: 474c1ea
tests run: `python -m pytest tests/unit/test_sql_sinks_checker.py -q` (31 passed); `python -m pytest tests/unit -q` (see below); `python -m ruff check .` clean; `python -m mypy` clean (143 files); `python scripts/check_sql_sinks.py src/mdqm` → 0 findings (was 47 after the rule fixes, 41 before).

## Summary
`scripts/check_sql_sinks.py` had no tests and two real holes in the rule itself; writing the cases first is what found them.
1. **A name bound to interpolated text was never reported.** The rule resolved string-literal names and then let every other bare name through, so `statement = f"...{table}"` followed by `cursor.execute(statement)` — the two-line form of the injection this tool exists to catch — was clean. Resolution is now provenance-based.
2. **`sql.SQL(...).format(sql.Identifier(col))` was reported as `.format()`**, although psycopg's composition quotes the identifier itself, so the false positive punished the safe idiom.
Both are covered by table-driven cases, and the fixtures are files on disk run through the command, so the tests pin the tool's real contract (findings + exit code), not its internals.
Then the sites those rules exposed moved onto the documented chokepoint instead of being suppressed: metadata SQLite and the local DuckDB engine go through `run_local()`, `SET` through `set_local()`, and identifier text through the new `local_ident()`/`local_idents()` in `execution.py`, which validate a name's shape before quoting it.

## Decisions
* **What "safe" means, and the limit, stated in the code.** Safe: a string literal, a name bound only to literals, a compiled object (`Fragment`/`FederatedPlan`, recognised by construction, annotation, or the compiler's own attribute), and calls to `prepare`/`execute_one`/`run_local`/`sql.SQL`. Reported: interpolation, `+`, `%`, `.format`/`.join`, a conditional, and a name bound to anything else. **Not traced: a bare parameter.** `def go(cursor, text): cursor.execute(text)` is not reported, because at that point the question is "was the guard run", which `prepare()`/`execute_one()` answer at execution time. A lint rule cannot answer it, and pretending otherwise would mean flagging every adapter call.
* **`mdqm/testing/` is exempt by path, in one line.** The synthetic-data loaders create, fill and drop tables: the read-only guard is the wrong tool for them and would refuse valid DDL. The server's query path never runs through those modules. Exempting the tree once is reviewable; exempting a dozen call sites is not. Everything else — including `pipeline/` and `eval/`, which the old hook skipped — is checked, and the pre-commit hook's `files:` now scopes both.
* `local_ident` lives in `sql/execution.py`, so the "trusted assembler" list still has exactly one entry.

## Deviations
* The handoff counted 42 sites; the honest number after fixing the rule's holes was 47, and the rule's own false positive on the adapter/executor port (`adapter.execute(fragment, limits)`, `executor.execute(plan, user=…)`) had to be resolved before the count meant anything. The final state is 0 findings over `src/mdqm`, 0 over `src/mdqm/adapters`.

## Dependencies
none

## Not verified
* `tests/integration --require-services`: Docker Desktop's engine went down mid-session and never came back (`open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`). The changed loaders and the cache's identifier quoting are exercised only by unit tests here. Re-run `python -m pytest tests/integration --require-services -q` once Docker is back — the loaders' emitted DDL is unchanged, but that is a claim, not a measurement, until it runs.
* The full unit suite on this branch: running when this note was committed; the result belongs here.

## Plan issues
* The pre-commit hook scoped the rule to a hand-written directory list, which silently missed `pipeline/` (a runtime path that builds source SQL) and `eval/`. A directory list that must be maintained by hand will drift again; a single "everything under src/mdqm except mdqm/testing/" default would be harder to forget.
* `set_local()` refused with `StatementRefused(guard("SELECT 1", "duckdb"))`, so a bad setting name produced a refusal that did not name the setting — and its `temp_directory` pattern rejected a *relative* spill path, which every test and dev run uses. Adopting it in the segment cache is what surfaced both; it now raises `ValueError` naming the setting and its rule, and the path rule is "1-512 characters, no quote, no backslash, no control character", which is the actual condition for breaking out of a quoted literal.
