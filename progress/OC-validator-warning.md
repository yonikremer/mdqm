status: done
task: OC-validator-warning
agent: opencode   branch: oc/validator-warning-20261128-162348   commit: d616095
tests run: `python -m pytest tests/unit/test_duckdb_spill.py tests/unit/test_model_validators.py tests/unit/test_semantic_loader.py -q -W default` -> 115 passed; `python scripts/check.py` -> all checks passed (4512 passed, 2 skipped: oracle extra absent)
## Summary
Fixed the pydantic after-model-validator warning: `SemanticLayer._fill_version_hash` was returning `self.model_copy(update={"version_hash": ...})`. It now sets the frozen field in place via `object.__setattr__` and returns `self`, so behaviour (an empty `version_hash` is filled with `compute_version_hash(self)`) is unchanged without the warning.
Added `tests/unit/test_model_validators.py`: a source scan that fails if any `@model_validator(mode="after")` in `src/mdqm` returns anything other than `self`, plus a runtime test constructing `SemanticLayer()` under `warnings.simplefilter("error")`.
Updated a stale comment in `tests/unit/test_stage_drift.py` that said the validator copies the model.
## Decisions
Used `object.__setattr__` rather than a `mode="before"` validator because the hash must be computed over the already-validated model. The frozen model stays frozen to callers; only the validator writes the field.
## Deviations
none
## Dependencies
none
## Not verified
Live-service (Docker) tests: Docker is not available on this machine, so no integration/e2e tests were run.
## Plan issues
none
