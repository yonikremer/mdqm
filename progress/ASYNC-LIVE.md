status: partial
task: the async query path on live services (follow-up to ASYNC-QUERY)
agent: pi (main session)   branch: main   commit: the commit that adds this file
tests run: `pytest tests/integration/test_async_query_e2e.py --require-services -q` -> 2 passed;
  `pytest tests/integration --require-services -q` -> see the commit's run.

## Summary
`progress/ASYNC-QUERY.md` shipped the job + poll shape with the live test never executed, because that
session had no Docker. This session ran it against the rebuilt server image and a real PostgreSQL. The
shape works: `query` with `mode: "async"` answers `{status: "started", query_id: "j_2",
plan: {partitions_scanned: 3, partitions_total: 14, est_rows: 18946}}` in milliseconds, and a later
`get_more({"query_id": "j_2"})` answers `done` with the summary, the sample, `partitions_done: 3` and a
`result_handle` that pages through `get_more({"result_handle": …})`. The blocking default is unchanged
and writes no job row.

## The one thing that was wrong: an assertion that depended on the cache
The live test asserted `partitions_done is not None` on whatever job it happened to create. In a full
suite run the first job was a **cache hit**, and a cache hit never asks the source for an EXPLAIN
(`_run` returns before the `hooks.plan(...)` call), so `partitions_done` stayed `None` — honest, since
no partition was scanned and inventing `0` or `3` would both be lies. The test therefore passed alone
and failed in the suite. It now forces `fresh: true` for the progress assertion, so it measures the
thing it claims to measure, and the docstring says why.

Not fixed, deliberately: a cache-hit job still reports `partitions_done: null`. Reporting `0` would be
tempting and wrong ("not reported" is not "none scanned"), and a `cache_hit` flag on the job row is a
design change nobody asked for.

## Decisions
* **The test asks for the work it measures.** `fresh: true` instead of reordering modules or warming
  the cache: an integration test that depends on a previous test's side effect is the flakiness
  `AGENTS.md` forbids.
* **No production change for this.** The alternative — making the cache-hit path write a synthetic
  partition count — would trade a true `null` for a false number.

## Deviations
none

## Dependencies
none

## Not verified
* Cancel against a long query on the live stack: the unit tests cover the flag and the adapter's
  `cancel`, the live test does not run a statement long enough to interrupt.
* A job that outlives a server restart: the row survives (`query_job`), the in-process outcome does
  not, and `get_more` then reports the row + the handle. Unit-tested with a fresh registry; not
  exercised by a real container restart.

## Plan issues
* The `partitions_done` semantics deserve a line in `docs/DESIGN.md` §9: "null while running, and null
  for a cache hit, because no source plan was asked for". Right now only the field description says it.
