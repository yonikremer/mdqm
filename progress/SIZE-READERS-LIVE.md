status: done
task: size readers on live engines (follow-up to SIZES-FOLLOWUP)
agent: pi (main session)   branch: main   commit: the commit that adds this file
tests run: `pytest tests/integration/test_size_readers_live.py --require-services -q` -> 9 passed;
  `pytest tests/integration --require-services -q` -> see the commit's run.

## Summary
`progress/SIZES-FOLLOWUP.md` wrote size readers for Oracle, SQL Server and Elasticsearch and could only
parse-test them, because that session had no Docker. This session had Docker, ran them against the live
compose services, and found the Oracle one **broken**:

    oracledb.exceptions.DatabaseError: ORA-00942: table or view "MDQM_RO"."ALL_SEGMENTS" does not exist

Oracle Free 23ai has no `ALL_SEGMENTS` view at all (measured: `all_views` carries `DBA_SEGMENTS` and
`USER_SEGMENTS`, not `ALL_SEGMENTS`), so `sizes()` raised on the first real call — a conformance run
found it, and every unit test in the follow-up passed anyway, which is exactly the failure mode of a
parse-only test.

The fix is capability detection, not a new constant: `_segment_view(cur)` probes `ALL_SEGMENTS` and then
`DBA_SEGMENTS` with two literal statements (one per candidate — an identifier spelled by a loop is
still a statement assembled from a variable, and the SQL sink checker refuses those), and
`_sizes_sql(view)` renders the same query around the answer, or around `CAST(NULL AS NUMBER)` when the
account may read neither. `None` there reads as "not reported", never zero, so a scan never looks free.

`tests/integration/test_size_readers_live.py` is the missing half of the follow-up: one live test per
engine, asserting the shape a planner relies on against numbers the adapter did not produce (a planted
row count, a partition list that must equal `extract_partitions()`'s names, a positive byte figure).

## Decisions
* **Probe, do not assume the release.** The view name is discovered per connection rather than chosen
  by a version check: a 19c install with an object grant and a 23ai install without one must both work,
  and only a read attempt tells them apart.
* **Both variants stay parse-tested.** `None`, `ALL_SEGMENTS` and `DBA_SEGMENTS` each go through
  sqlglot and the read-only guard in `tests/unit/test_size_readers.py`, so the statement half is covered
  without a server.
* **A wrong view name raises.** `_sizes_sql("USER_SEGMENTS; DROP TABLE x")` is a `ValueError`, pinned by
  a test: the identifier comes from a closed set.
* **The live tests fail rather than skip under `--require-services`**, and the Elasticsearch module
  loads the world it needs (the `es` container has no data volume, so a restart empties the cluster —
  the first run of this file "passed" the empty cluster by reporting no tables, which is now a failed
  assertion).

## Deviations
* `_SIZES_SQL` (a module constant) became `_sizes_sql(view)` (a function). One call site, one test
  import; no public API changed.

## Dependencies
none

## Not verified
* **Oracle bytes on an install that grants a segment view.** This compose Oracle grants `mdqm_ro`
  neither `ALL_SEGMENTS` (absent) nor `DBA_SEGMENTS` (no grant), so the live run proves only the
  no-bytes branch; the bytes branch is parse-tested. An install that grants `DBA_SEGMENTS` would take a
  path nobody has executed end to end.
* SQL Server and Elasticsearch: `sizes()` is measured live, but the planner was not re-run end to end
  against their profiles (`bootstrap run` with the `sizes` stage) — only PostgreSQL has that proof.

## Plan issues
* `progress/SIZES-FOLLOWUP.md` says `status: done` for a reader that raised on every Oracle call. The
  ledger rule ("a skip is not a pass") applies to `status: done` too: parse-tested wiring is not a
  working reader, and `Not verified` said so while the status said done.
* `tests/integration/test_stage_propose_names_live.py` still pinned the pre-label proposal shape
  (`{"names"}`) after `LLM-WIRING` merged, so the full suite went red on a change that was correct. It
  now pins `{"names", "labels"}` and names the unapprovable `names` gap.
