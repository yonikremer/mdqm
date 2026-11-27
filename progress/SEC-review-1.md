status: done
task: SEC-review-1
agent: security-fixer   branch: fix/security-review   commit: (this commit)
tests run:
  - `.venv/Scripts/python.exe -m pytest tests/unit/test_security_review.py -q` -> 53 passed
    (before the fix, on the reverted `src/`: 37 failed, 16 passed -- the 16 are the
    "must not be a blunt instrument" cases plus the fixture helpers)
  - `.venv/Scripts/python.exe -m pytest tests/unit/test_guard.py -q` -> 780 passed (unchanged)
  - `.venv/Scripts/python.exe -m pytest tests/unit/test_governor.py -q` -> 45 passed
  - `.venv/Scripts/python.exe -m pytest tests/unit/test_openapi_adapter.py tests/unit/test_stage_openapi.py -q` -> 97 passed
  - `.venv/Scripts/python.exe -m pytest tests/unit/test_stage_profiling.py -q` -> 52 passed
  - `.venv/Scripts/python.exe scripts/check.py` -> all checks passed (ruff, mypy --strict on `src`,
    all unit tests: 3978 passed)

## Summary

Seven findings from an owner-requested read-only security review, each fixed with a regression test
built from the reviewer's own repro. Every case lives in `tests/unit/test_security_review.py`.

1. **HIGH -- OpenAPI path key may be an absolute URL.** `read_endpoints`/`_operations` now refuse a
   `paths` key that is not exactly one leading `/` with no scheme or authority (`_require_relative_path`),
   raising `_SpecError` carrying the repo's `invalid_spec` `ErrorResponse` + `fix`. Belt and braces:
   `GetOnlyTransport` now takes `allowed_host` (the base URL's host) and refuses any request whose URL
   host differs, so `httpx` dropping `base_url` for an absolute URL can no longer send the source's
   auth header anywhere. The check lives in the transport, not in `_get`, so a subclass cannot skip it.
2. **HIGH -- guard denylist missed sequence advances.** `_forbidden` now refuses `exp.NextValueFor`
   (T-SQL `NEXT VALUE FOR`) and any `exp.Column` named `nextval` (Oracle `seq.NEXTVAL`, quoted or
   not) as `WRITE_BLOCKED`, and `pg_logical_emit_message` joined the denylist. `currval` stays allowed.
3. **MEDIUM -- DuckDB file readers passed the guard.** `read_` joined the denied prefixes (covering
   `read_text`/`read_csv`/`read_parquet`/`read_blob`/...), and `glob`, `query`, `query_table` plus the
   `*_scan` aliases, `st_read` and `which_secret` joined `DEFAULT_DENIED_FUNCTIONS`. The guard has one
   policy for every dialect, so `read_` is a global prefix; the trade-off (a user-defined `read_*` is
   refused too) is written in a comment next to the list.
4. **MEDIUM -- advisory locks allowed.** `pg_advisory_` and `pg_try_advisory_` joined the denied
   prefixes; all eleven names in the family are now refused, and ordinary reads stay allowed.
5. **MEDIUM -- a bearer token echoed into the startup error.** `StaticTokenVerifier.from_file` builds
   its message from `exc.errors()` (`loc` + `msg` only) via the new `_problems` helper, never from
   `str(exc)`, so pydantic's `input_value=` can no longer print the token. `git grep` over the tree
   found no other place interpolating a `ValidationError`; the others already read `loc`/`msg`.
6. **MEDIUM -- the guard was not on the outbound path.** `Governor.execute` now calls
   `_refuse_unreadable` first: any fragment carrying SQL is guarded and refused through
   `mdqm.errors.from_guard`, before a slot, a quota unit or the breaker is touched. The profiling
   stage's `_read` (the one place that hands a fragment straight to an adapter, and where
   `sample_sql()` returns raw plugin text) now guards too and reports a refusal in `skipped`.
7. **LOW -- plaintext `http://` base URLs.** `_base_url` validates both `options.base_url` and
   `servers[0].url` through `_checked_base_url`: `https://` or a loopback `http://` host only.

## Decisions

- **Where the base-URL check applies.** I validate the spec's `servers[0].url` as well as
  `options.base_url`, because the override was not the only way in. Three test fixtures that used
  `http://mock-api.invalid` as an OpenAPI base URL now use `https://`; the in-process ASGI bridge
  never resolves the host, so nothing else changed. This is a fixture adjustment, not a weakened test.
- **The `read_` prefix is global, not per-dialect.** `GuardPolicy` is dialect-independent, so a
  per-dialect list would mean adding dialect plumbing to a security-critical pure module. A refused
  user-defined `read_*` on an engine that has none is the conservative side of the trade-off, and it
  can never be a write the caller needed.
- **A column genuinely named `nextval` is refused.** Same reasoning: the guard cannot tell it from
  Oracle's sequence pseudo-column, and refusing a read is recoverable while allowing a sequence
  advance is not.
- **`Fragment` itself stays unvalidated.** A fragment may legitimately carry SQL for any dialect, and
  the model has no policy context. Validation belongs at the boundary, which is now the governor plus
  the compiler plus the profiling stage.
- **`_SpecError` gained an optional `error` attribute** so a spec refusal can carry the agent-facing
  `ErrorResponse` while staying a `ValueError` for every existing caller.

## Deviations

- The reviewer's FIX 1 note suggested refusing in `_operations`/`read_endpoints`; `_operations` is
  currently unused (dead code) but I routed it through the same check so the two cannot disagree.
- The reviewer's FIX 6 note said "one guard call in the governor's drain path". I put it in
  `execute()` *before* the breaker and the slot, which is strictly stronger: a write no longer even
  spends a quota unit. The drain path is still reached only after it passed.
- `docs/DESIGN.md` needed no edit: line 38 ("AST guard, also on SQL we compile ourselves") and line
  88 ("API key stays on the server") were already the intended guarantees -- finding 1 and finding 6
  were cases where the code did not honour them, and the fixes make the document true rather than
  changing it.

## Dependencies

None. Every fix uses stdlib or an existing dependency (`httpx`, `pydantic`, `sqlglot`).

## Not verified

- **No live database.** Every guard finding (2, 3, 4) is verified at the AST layer only, which is
  where the refusal happens. I did not confirm against a real Oracle/T-SQL/DuckDB server that the
  denied spellings are the only ones those engines accept; the AST is the choke point by design, so
  a spelling that parses to one of these nodes is covered, and one that does not parse is refused as
  `PARSE_ERROR`.
- **DuckDB `read_parquet` on a real `file://` URL** is asserted blocked at the guard level; the
  reviewer's end-to-end repro (`FakeAdapter` returning a file's contents as rows) is not re-run here
  because the guard now refuses before the adapter, which is the same boundary the repro crossed.
- **No Docker on this machine**, so no integration test was run or attempted (none of the seven
  findings needs one).

## Plan issues

- **This came from the owner-requested security review, not from `TASKS.md`.** It has no task ID, no
  `owns` list and no entry in `tasks.json`; the ID `SEC-review-1` is invented for the progress file
  only. Nothing in the backlog was changed.
- The review also flagged, without proof, that `Fragment` has no read-only validation of its own. I
  recorded the reasoning above rather than adding a validator to the model.
- Pre-existing, out of scope, noticed while working: `src/mdqm/adapters/openapi.py::_operations` is
  dead code (no caller in `src/` or `tests/`); `FakeAdapter.kind` is `Final` while
  `SourceAdapter.kind` is declared writable, which pyright cannot match (the conformance kit proves
  the adapter satisfies the protocol at runtime, and `scripts/check.py` does not run pyright).
