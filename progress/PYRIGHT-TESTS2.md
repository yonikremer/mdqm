status: partial
task: pyright debt in tests/ (docs/HANDOFF.md section 7 / 8)
agent: pi (main session)   branch: task/pyright-tests2
tests run: `python -m pytest tests/unit -q` (see the line below); `python -m ruff check .` clean; `python -m mypy` clean (143 files); `python -m pyright` from `.worktrees/PYRIGHT`: 144 errors + 2 warnings at the branch point → **52 errors + 2 warnings** now.

## Summary
Two things landed, and one of them is not typing work at all.
1. **`SourceAdapter` declared four optional methods as required.** `REQUIRED_METHODS` excludes them, `OPTIONAL_METHODS` names them, the probe reports an absent one as an unsupported capability, and core code already reaches them through `supports_method`/`getattr` — but the *protocol* made them mandatory, so every test double had to implement four methods it existed to prove can be missing. They now live in `OptionalSourceMethods` (documentation, not a constraint) and `SourceAdapter` keeps the five required members. That exposed three real call sites that assumed the methods exist (`pipeline/cli.py`, `pipeline/stages/catalog.py`, `pipeline/stages/mutability.py`, `server/tools/query.py`), which now ask first — including `query.py`'s `explain`, a latent `AttributeError` for an adapter without it.
2. **Test doubles now satisfy the contracts they claim to.** Stream doubles gained `rows_returned`/`bytes_returned` (the governor enforces its caps with them, so a double without them was not a stream), clocks gained `monotonic`, adapter doubles renamed `execute(query=…)` to `execute(fragment=…)` and take `ArrowStream` in `cancel`, `kind` is a `ClassVar`, generator fixtures are annotated `Iterator[...]`, and `QuietWindow` is given real `time` objects instead of wire strings.

## What is left: 52 errors, and where
* `test_config.py` (6) — `Settings(server={...})` passes the wire dict where `ServerSettings` is declared; construct the sub-model (or `model_validate`).
* `test_partition_planner.py` (6) — two fixture generators annotated as their yielded connection, `tuple[expr.Expr]` where `Sequence[Expression]` is declared, two `Optional` bound accesses, one `int | None` comparison.
* `test_stage_drift.py` (4), `test_audit.py` (4) — helper keyword arguments receiving the wrong type (`sources`, `row_count`, `error_code`).
* `test_openapi_adapter.py` (3) — `cached`/`fetched_at` are the OpenAPI adapter's own counters, not members of the `ArrowStream` protocol; the assertions need a cast to the concrete stream type.
* `test_guard.py` (3) — `pytest.param` passed where a class is expected.
* `test_stage_mutability.py` (3), `test_stage_conversions.py` (3), `test_time_planner.py` (3) — enum literals (`granularity="month"`) and an `Optional` `replace`.
* The rest are one or two each: `test_registry`, `test_singleflight`, `test_query_corpus`, `test_linter`, `test_history`, `test_digest`, `test_compiler`, `test_fake_adapter`.

Every one of those is the same shape as the classes already cleared here: name the declared type instead of the wire value.

## Decisions
* **The protocol split is the one src change, and it is not cosmetic.** The type model contradicted the documented contract and the runtime; the alternative — patching 6 test doubles to grow four methods — would have made the tests lie about what an adapter must implement.
* A time spec in the wire form goes through `TimeSpec.model_validate(...)` via a tiny `wire()` helper in `test_time_planner.py`, which clears 45 findings in one honest move: those tests deliberately send the raw strings an agent sends, so pydantic's documented entry point for unparsed input is the right call, not a suppression.
* No `# type: ignore` or `# pyright: ignore` was added. None was removed either.

## Deviations
* The brief expected 4483 collected tests and 139 pyright errors; this tree collects 4473 at the branch point (main gained the bootstrap and sql-sinks tests, so 4524) and pyright counted 144 here against 139 on `main` — same code, different include set (`.worktrees` is excluded in the repo config, so a worktree needs its own `pyrightconfig.json`, which is not committed).
* `pyright` cannot be run inside a worktree with the repo config: `venvPath = "."` resolves to the worktree, which has no `.venv`, and `.worktrees` is in `exclude`. A throwaway `pyrightconfig.json` pointing `venvPath` at the main checkout is the workaround; it must be deleted before committing.

## Dependencies
none

## Not verified
* `tests/integration --require-services`: Docker is down (see `docs/HANDOFF.md` §1). The one edited integration double (`test_claims_e2e.py`) is type-level only — it now counts the batches it forwards.
* An earlier subagent completed the same task and reported 139 → 0, but its worktree lives outside this workspace (`P:\...`), so its commits could not be read or merged. This branch is an independent pass; if that patch is ever imported, expect conflicts in the same files.

## Plan issues
* `pyright` is not part of `scripts/check.py`, so this debt can grow back silently between sessions.
