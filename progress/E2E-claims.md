status: done
task: E2E-claims
agent: e2e-claims   branch: task/E2E-claims   commit: 3745092
tests run:
  - `.venv/Scripts/python.exe -m pytest tests/integration/test_claims_e2e.py -q` -> 7 passed in 6.7s
  - `.venv/Scripts/python.exe -m pytest tests/integration/ -q` -> 23 passed, 3 skipped (the 3 skips are the
    A02-dependent tests in test_postgres_e2e.py)
  - `.venv/Scripts/python.exe -m pytest tests/integration/test_postgres_e2e.py --require-services -q` -> 2 passed, 3 skipped
  - `MDQM_PG_PORT=59999 .venv/Scripts/python.exe -m pytest tests/integration/test_postgres_e2e.py -q` -> all skipped
    ("service unavailable: PostgreSQL at 127.0.0.1:59999"), and with `--require-services` -> 5 errors: a missing
    service fails as required
  - `.venv/Scripts/python.exe scripts/check.py` -> all checks passed (ruff, mypy --strict on src, 4033 unit tests)
  - `.venv/Scripts/python.exe scripts/bench.py --runs 20` -> table below; `--json` prints the same numbers as JSON

## Summary

Three deliverables, one per claim of `README.md` "The problems it solves".

1. `tests/integration/test_claims_e2e.py`: seven tests, one per claim, each driven through the product path
   (validated spec -> compiler -> guard -> governor -> real adapter -> federated executor / context index /
   audit store) over the D02 seeded world, checked against the E01 answers that were derived without the code
   under test. Every docstring names the README sentence and the thing that would make the assertion false.
2. `tests/integration/test_postgres_e2e.py`: the real-database twin (5 tests). Two run today against the live
   `mdqm-pg-1` (partition pruning, read-only); three skip with a precise reason until task A02 exports a
   `PostgresAdapter`.
3. `scripts/bench.py`: prints p50/p95 per stage (compile+plan, execute, cache miss, cache hit, context pack,
   federated join, 10k-value round-trip) with `--json`; asserts only invariants, needs no Docker.

Proof that the tests can fail: mutating the fixture changed the outcome as predicted.
`mode: half_even` in the layer -> claim (a) fails (`29671.22 != 29731.68`); `require_key_filter: false` ->
claim (b) fails (the query is accepted and answers 0 rows instead of being refused).

## Benchmarks as printed (`.venv/Scripts/python.exe scripts/bench.py --runs 20`)

```
mdqm benchmarks: seed 20260101, 20 runs
world: 200 customers, 5001 usage events

stage                              p50         p95     n  note
--------------------------  ----------  ----------  ----  ------------------------
compile+plan                   2.131ms     3.072ms    20  186 chars SQL, 2+0 predicates
execute (fake source)          1.015ms     1.085ms     6  1240 rows fetched per run
cache miss (fetch)            25.209ms    25.209ms     1  1 source call(s)
cache hit                     10.365ms    14.693ms    20  20 hits, 0 new calls
context pack build             0.688ms     0.965ms    20  303 tokens, 8 facts
federated join                24.613ms    69.943ms     6  3 groups, active first
round/normalise round-trip   488.926ms   524.158ms     3  10000 labels + 10000 roundings/pass, 48.89 us/value

read these as shapes, not targets: same machine, same seed, same work.
context pack: 303 tokens / 8 facts; values round-tripped: 30000
```

## Decisions

- **Claims (a)-(g) need no Docker.** They use the F05 fake adapters, so they live under `tests/integration`
  (they are the acceptance slice, not a unit test of one module) but they *run* everywhere instead of
  skipping. Only the PostgreSQL twin reports `service_unavailable`.
- **The semantic layer YAML is repeated inside the test module.** `tests/` is not a package, so an integration
  test cannot import `tests/unit/test_executor.py`'s fixture. It is the C17 federation layer (identity /
  `ROUND(x/5000)*5000` / `x*10`) plus five more measures on the same columns, because claim (f) needs a pack
  that genuinely overflows `MIN_PACK_BUDGET`.
- **erp is deduped in the fixture, and the gap is stated in the code.** A query spec cannot say "distinct
  entity" (documented in `progress/C09.md`), so claim (a)'s erp leg hands the compiler `E01`'s deduped view
  (`test_compiler.py` does the same). The cross-source federated leg uses the shipped semi join instead, which
  is the plan the product actually runs.
- **Claim (d) picks its own instant.** The shipped `new_since_resync` question happens to have no row
  straddling 2025-12-01 in this seeded world (`start_time >= since` and `insertion_time >= since` both count
  6316), so that question cannot show the difference. The test keeps the E01 assertion and then takes
  `since = min(insertion_time)` over the world's rows with a lag > 60s, which is deterministic and guarantees
  a late row, and asserts the naive filter loses exactly those rows against the same adapter.
- **Claim (e) proves guard + audit here, layers 1 and 3 on the real server.** The read-only *account* and the
  read-only *transaction* are PostgreSQL facts; they are asserted with the server's own SQLSTATEs in
  `test_postgres_e2e.py`. This module asserts layer 2 (AST guard, including on SQL we compiled) and that the
  refusal lands in the hash-chained audit log with a verifying chain.
- **Claim (g) uses a scripted clock, not a sleep.** A stream wrapper moves the injected clock past the deadline
  after the first batch, so "the adapter's `cancel` was called on the live stream" is observed with no waiting
  and no flakiness. The concurrency peak is the number of callers parked *inside* the adapter, which is the
  number of slots the governor actually granted.
- **`bench.py` asserts invariants only** (compiled SQL carries both conversions; a cache hit adds zero source
  calls; the pack stays inside budget and is byte-identical twice; the join answers identically twice; all
  30 000 round-trips are exact). No threshold on a duration, and no Docker.

## Deviations

- The bench numbers were produced with the repository venv, whose editable install points at the *main*
  checkout. This branch changes no `src/`, so the measured code is the same; a branch that does change `src/`
  must run bench with `PYTHONPATH=src` to measure itself.
- `docs/DESIGN.md` was not touched: nothing about the design changed, these are tests and a measuring tool.

## Dependencies

none (stdlib `statistics` replaces a hand-rolled percentile; `pyarrow`, `duckdb`, `psycopg` were already there).

## Not verified

- **Task A02's adapter.** Three tests in `test_postgres_e2e.py` skip: real-data conversions through the
  adapter, catalog + partition discovery, and the `pg_sleep` cancel. They build the adapter through
  `AdapterRegistry` + a secret provider that resolves a source's secret to its read-only DSN -- that wiring is
  an assumption about A02, so the first run after A02 lands may need a one-line fix (the class name is looked
  up as `PostgresAdapter` and the test skips with that exact message if it is spelled differently).
- **The `--require-services` hole:** a missing *service* fails, but a missing *adapter* still skips even with
  `--require-services`, because no environment can supply A02 today. That is the one acceptance item the flag
  cannot force.
- **Claim (g) on a real server** (cancel of a real `pg_sleep`) is written and skips; the scripted-clock proof
  covers the governor's behaviour today, but PostgreSQL's own statement cancellation is unproven until A02.
- **The "five coarse tools" half of claim (f)** is not asserted here: the pack half is, and the tool surface is
  already covered by `tests/unit/test_mock_*`/tool-schema tests. No test in this change counts the tools.
- **Cache-hit latency being ~10 ms** (see Plan issues) is reported, not explained.

## Plan issues

- The cache-hit path costs ~9.7 ms against a ~1.2 ms adapter drain and a ~24 ms miss: the hit is 2.5x faster
  than the miss, so it saves a source call, but the absolute cost is dominated by something other than the
  fetch. A hit re-materialises from the local DuckDB cache, and `federated join` p95 (63 ms) is 2.7x its p50.
  Both are worth a look in a performance task; neither is a correctness bug, and neither is in this change's
  scope.
- The E01 question `new_since_resync` cannot demonstrate the late-arrival claim for the seeded world (see
  Decisions). A question whose `since` sits on a planted late row would make the claim testable without
  inventing an instant inside the test.
- `README.md` "Status: honest" still says nothing has run against a real PostgreSQL. `tests/integration/`
  contradicts that today (D03's loader and this module's two psycopg tests run against `mdqm-pg-1`), so the
  README sentence is stale and the owner should decide its wording.
