status: partial
task: partition size profile for query optimization (owner's stated need)
agent: pi (main session)   branch: main
tests run: `python -m pytest tests/unit -q` 4557 passed; `ruff`/`mypy`/`pyright` clean.

## Summary
A plan's row estimate was a sum of the source's own `reltuples`: stale (it moves when the source
analyzes, not when rows arrive), byte-less, and shapeless -- 34 even months and one month holding
20x the median sum to the same number.

Three pieces, all measured live against the three Postgres databases:
1. **Measurement.** `SIZES_SQL` reads `pg_total_relation_size` (heap + indexes + TOAST) and
   `reltuples` per relation; `parse_sizes` turns those rows into tables with their partitions. A
   partitioned parent has no storage of its own, `reltuples = -1` (never analyzed) is unknown, and a
   boolean is not a size.
2. **Storage.** `0009_sizes.sql` + `store/sizes.py`: one row per sample per pass, append-only, so the
   newest capture answers a plan and two captures show growth. Deliberately **not** in the semantic
   layer: a nightly size would change the layer's version hash nightly and invalidate the context
   index and every cache key with it.
3. **Use.** `planner/sizes.py` (nearest-rank `min/p1/p25/p50/p75/p99/max` over rows *and* bytes, with
   spread and a fat-partition hint); `PartitionEstimate` gains `bytes`, `estimated_from`
   (`profile`/`catalog`) and `incomplete` (a touched partition reported no size, so rows is a lower
   bound); `QueryServices` carries the store and the tool hands the spec table's profile to
   `CompileContext`.

`mdqm sizes measure | show | profile` expose it.

## Measured live (billing.usage_events, 14 partitions)
    rows   min=0 p1=0 p25=0 p50=5916 p75=6316 p99=6434 max=6434 total=50006
    bytes  min=16KB p50=852KB p99=918KB  spread=1.088  unbalanced=false

## Decisions
* Rows **and** bytes, per table and per partition: bytes is what a scan costs on disk, rows is what
  the governor caps; a profile with only one of them cannot answer either question.
* `None` means "not reported", never zero -- an Oracle install reporting `num_rows` but no bytes must
  not drag the byte median to 0.
* A measured profile is preferred, but a partition it does not mention falls back to the catalog, so a
  partition created after last night's measurement is never estimated as free.
* No sizes measured means byte-for-byte the previous behaviour. A test pins that, because an
  optimization that silently changes every plan is not an improvement.

## Deviations
none

## Dependencies
none (sqlite + existing pydantic; no new package)

## Not verified
* Oracle, SQL Server and Elasticsearch size readers: only PostgreSQL has `sizes()`. The others report
  the capability as unsupported, and the planner falls back to their catalog estimates.
* Whether the profile actually makes a query cheaper end to end: the estimate is reported in the plan,
  but no benchmark compares plans with and without it yet.

## Plan issues
* The fat-partition *note* (`PartitionNote.FAT_PARTITION`) is declared but not emitted: the estimate
  carries the numbers, the note needs one line in `plan_partitions` that names the partition.
* Nothing schedules `mdqm sizes measure`; it is a command an operator runs (or a cron job), and the
  bootstrap pipeline does not include it.
* Async query and the model endpoint (the owner's other two asks) are not started.
