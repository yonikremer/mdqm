status: partial
task: ASYNC-QUERY
agent: pi (async-query worktree)   branch: task/async-query   commit: 5c1f860, 098d2bb, 8536e54 (head)
tests run:
  `.venv/Scripts/python.exe -m pytest tests/unit -q` -> 4637 passed, 1 warning
  `.venv/Scripts/python.exe -m pytest tests/unit/test_jobs_async.py -q` -> 26 passed
  `.venv/Scripts/python.exe -m pytest tests/unit/test_errors.py -q` -> 131 passed
  `.venv/Scripts/python.exe -m ruff check <touched files>` -> clean
  `.venv/Scripts/python.exe -m mypy` -> Success: no issues found in 166 source files
  `.venv/Scripts/python.exe -m pytest tests/integration/test_async_query_e2e.py -q` -> 2 skipped
    (no Docker, no server); with `--require-services` -> 2 errors, so a skip is never counted as a pass
  `.venv/Scripts/python.exe -m pyright` -> 17 errors, all pre-existing (see Not verified)

## Summary

`query` now has two modes. Without `mode` it blocks exactly as before and writes no job row, which is
what `tests/integration/test_acceptance_e2e.py` and every existing caller depend on. With
`mode: "async"` it runs every check that needs no source inline (parse, lint, value gate, compile,
static partition plan), answers immediately with `status: "started"` + `query_id` + the plan, and
runs the same blocking call on a worker thread of the in-process registry. `get_more(query_id=…)`
reports the job from its `query_job` row (state, phase, measured `rows_read`, `partitions_total`/
`partitions_done`), and once done hands back the same `summary`, `sample` and `result_handle` a
blocking call returns. `get_more(query_id=…, cancel: true)` asks the governor to stop: it calls the
adapter's own `cancel` on the live stream and the job ends `cancelled` with code `CANCELLED`, which is
deliberately not `TIMEOUT` and deliberately not a breaker failure.

## Decisions

* **One reference kind per field, never overloaded.** `result_handle` (`h_…`) pages rows that exist;
  `query_id` (`j_…`) watches work still running; a history run id stays `q_<n>`/`<n>` for `feedback`.
  `get_more` takes them as two arguments and refuses both-at-once and neither with `INVALID_SPEC`.
  The job prefix is `j_`, not the `q-42` of the design sketch, because `feedback` resolves a `q_` id
  against `query_history`: one id space with two meanings would silently rate the wrong run.
* **Blocking is the default** (`mode: "blocking"`). Chosen conservatively: the five acceptance tests
  and every existing caller keep today's shape, and nothing changes silently.
* **The state lives in a row, the worker lives in the process.** `query_job` (migration 0011) is what
  a poll reads, so a poll works from any thread/message; the registry holds only the thread, the live
  counters and the finished summary. One replica (`docs/deploy.md` §8) is what makes that sound.
* **Progress is measured, never invented.** `rows_read`/`bytes_read` come from the governor's drain
  loop, `partitions_done` from the source's own plan (`PlanSummary.partitions_scanned`). Both are
  `null` until something reports them; `null` is never rendered as zero.
* **A stopped job is the caller's choice, not the source's failure.** New `ErrorCode.CANCELLED` and
  `GovernorCancelled` (a `GovernorRefusal` subclass) so `Governor._run_slot` does not record a breaker
  failure for it. `ErrorCode` is a published closed set, so `tests/unit/test_errors.py` grew the
  matching `EXPECTED_CODES` entry, `BUILDERS` entry and two `CASES`.
* **Two hooks on `Governor.execute`**, both optional and both `None` for a blocking run:
  `progress(rows, bytes)` after every drained batch and `cancelled()` asked before every batch. No
  new code path in the governor for the job; the job reuses the deadline/cancel machinery that was
  already there.
* **A lost job is reported as lost.** A row still `running` with no worker behind it (process restart)
  is marked `failed` with an `INTERNAL_ERROR` and a note the first time it is polled, instead of
  showing `running` for ever. This is the one write-on-read, and it is deliberate.
* **A build without a registry refuses with `NOT_SUPPORTED` and names the fallback** instead of
  accepting a job nobody will run. A full registry (`max_jobs`, default 8) refuses with
  `QUOTA_EXCEEDED`; that ceiling is the server's thread budget, separate from the governor's
  per-source semaphores.
* **A cross-source (federated) spec is deferred whole.** `federated_plan` is cheap and local, so the
  started answer carries a note instead of a partition count: several fragments have no single plan.
  Proven by a unit test that stubs the federated engine (which has its own tests) and asserts only
  this branch.

## Deviations

* The design sketch showed `plan: {partitions, est_rows}` in the started answer; it is there, as the
  full `PlanInfo` (so `est_bytes`/`estimate_source` from the size profile travel too). The finished
  answer does not repeat the plan: partitions are in `progress`, and `JobOutcome` deliberately does
  not carry a second copy (see `mdqm/server/jobs.py`).
* `get_more`'s page path answers `status: "ok"` as before; the job path answers the job's state in
  the same field (`queued`/`running`/`done`/`failed`/`cancelled`), so the widened `Answer.status`
  literal is shared by `history` and `feedback` even though only `get_more` produces the new values.
* `docs/HANDOFF.md` §4b/§11 and the summary table were edited to say this is built but not yet run
  against live services. The rest of that file is untouched.

## Dependencies

None. Everything used is already in the tree (`threading`, `sqlite3`, `pydantic`, `pyarrow`).

## Not verified

* **The integration test has never run.** Docker is not running in this environment
  (`docker ps` -> "the system cannot find the file specified"), so
  `tests/integration/test_async_query_e2e.py` skips; with `--require-services` it errors, as designed.
  Its two claims are unproven against a real server, and the image must be rebuilt before anyone tries.
* `tests/integration/test_acceptance_e2e.py` (the five blocking-shape claims) could not be run for the
  same reason. The blocking shape is pinned by unit tests (`test_the_blocking_default_is_unchanged_and_writes_no_job`
  and the whole existing `test_tool_query.py`), but that is not the same as the acceptance run.
* **`pyright` reports 17 errors** and every one is pre-existing on `main`: `src/mdqm/stats/postgres.py`,
  `src/mdqm/stats/collect.py`, `tests/unit/test_dev_env.py`, `tests/unit/test_model_validators.py`,
  `tests/unit/test_stats.py`. No finding names a file this task touched. pyright is not in
  `scripts/check.py`, so this is not a regression introduced here.
* `scripts/check_sql_sinks.py src/mdqm` reports **3 findings, all pre-existing on `main`**
  (`stats/collect.py`, `stats/postgres.py` x2). The new SQL in `store/jobs.py` is literal text at each
  call site, so it adds none.
* **Cancel latency is bounded by the adapter, not by us.** A `SourceAdapter.execute` that is blocking
  before its first batch cannot be interrupted; the stop is honoured between batches through
  `adapter.cancel(stream)`. Stated in the governor docstring and in the `cancel` field's description.

## Not done (deliberately)

* Retention/housekeeping for `query_job` rows. They are a few dozen bytes each and nothing schedules a
  sweep yet; `store/jobs.py` says so instead of shipping a job nobody triggers.
* No `cancel_query` tool: the surface stays the five tools, and `cancel: true` on the reference that
  names the job is enough.
* No cross-replica visibility: a second replica would see the rows but not the workers, which is exactly
  why the deployment rule is one replica. Nothing here changes that rule.
* `job_registry` is constructed in `serve()`; a worker-style deployment that polled from a different
  process would need the row-only path (`result_handle`), which already works.

## Plan issues

* `AGENTS.md` §9 lists "unknown identity", "unknown model endpoint" and friends as pluggable unknowns;
  "how many replicas" is not in that list even though `docs/deploy.md` §8 makes it a correctness rule
  the async registry depends on. Worth promoting into §9 so the next feature does not assume it.
* `scripts/check_sql_sinks.py` is documented in the handoff as reporting 0 findings; it reports 3 on
  `main` today. Either the handoff line or the code drifted.
