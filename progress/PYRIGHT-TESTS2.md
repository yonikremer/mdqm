status: done
task: pyright debt in tests/ (docs/HANDOFF.md section 8)
agent: pi (main session)   branch: task/pyright-rest
tests run: `python -m pyright` 0 errors 0 warnings (was 139 errors + 2 warnings at the handoff);
`python -m ruff check .` clean; `python -m mypy` clean (143 files); `python -m pytest tests/unit -q`
4524 passed; `scripts/check.py` all checks passed.

## Summary
The whole pyright backlog is cleared, and three of the fixes are not typing work at all:
1. **`SourceAdapter` required four optional methods.** `REQUIRED_METHODS` excluded
   `extract_partitions`/`explain`/`mine_queries`/`mutability_stats`, the probe reports an absent one
   as an unsupported capability, and callers already reached them through `supports_method`. They now
   live in `OptionalSourceMethods`. Four call sites that assumed they exist ask first — including
   `query.py`'s `explain`, a latent `AttributeError` for any adapter without it.
2. **`mdqm.testing.loaders` imports its engines lazily** (module `__getattr__`, PEP 562). The package
   exported `elastic`/`postgres` in `__all__` without importing them; eager imports would have made
   `import mdqm.testing.loaders` fail on a deployment without the Elasticsearch extra.
3. **`ConversionStage.pairs` is a public read-only property.** The bootstrap test needed to assert
   what was declared and was reaching into `_pairs`; a stage's input is not an internal.

The rest is the work the note in the previous commit listed: fixtures that are generators are
annotated `Iterator[...]`, wire values (strings, dicts) go through `model_validate` where pydantic is
the parser, declared types (enums such as `NormalizedType`, `PartitionGranularity`, `PruningProof`,
`MeasureCanonical`, `SecretStr`, `Decimal`) are named instead of their wire forms, `Optional` reads
are asserted before use so the test says what it depends on, and doubles satisfy the protocol they
claim to (`ArrowStream` counters, `Clock.monotonic`, `execute(fragment=…)`, `cancel(ArrowStream)`,
`ClassVar` kinds).

## Decisions
* **No `# type: ignore` / `# pyright: ignore` was added.** Two existed in the tree already; one
  (`test_audit.py`) remains on a `**kwargs` case that deliberately passes a forbidden key, and is the
  documented way to test `extra="forbid"`.
* **A wrong-type test input keeps its wrong type.** `parse_spec(cast(Any, ["from", …]))` is cast
  because the *point* of the test is that a list is refused; changing the input to a dict would have
  tested a different thing and, measured, asserted `'from'` instead of `'(root)'`.
* **`pytest.ParameterSet` is not in the stubs**, so the guard's parametrisation helpers are typed
  `list[Any]` rather than reaching into `_pytest` internals.

## Deviations
* pyright cannot run inside a worktree with the repo config: `venvPath = "."` resolves to the
  worktree (which has no `.venv`) and `.worktrees` is in `exclude`. A throwaway
  `pyrightconfig.json` pointing `venvPath` at the main checkout is the workaround, and it is not
  committed. Worth adding to the handoff's traps.

## Dependencies
none

## Not verified
* `tests/integration --require-services`: Docker Desktop's engine is down (see `docs/HANDOFF.md` §1).
  The one edited integration double (`test_claims_e2e.py`) is type-level only: it counts the batches
  it forwards.

## Plan issues
* pyright still gate nothing (`scripts/check.py` runs ruff + mypy + unit only), so this debt can grow
  back between sessions. Adding `pyright` to the gate is a one-line change plus a decision about the
  latency it costs.
