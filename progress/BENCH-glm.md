status: partial
task: benchmark (owner request; the goal's "realistic benchmark")
agent: claude + GLM 5.3 Flash in pi   branch: main
command: `.venv/Scripts/python.exe scripts/bench_glm.py score --answers bench/answers`

## What this measures
A real weak model (GLM 5.3 Flash, running as a pi agent) is given each E01 question, its five `mdqm`
tools through `scripts/mdqm_call.py`, and nothing else: no spec, no SQL, no expected answer. It must
choose the tools and the arguments. Its answer is scored against E01's ground truth, which was
derived independently of the code under test.

## Result: 13 of 15
### Passing (13)
`avg_limit_by_status_crm`, `avg_limit_by_status_billing`, `avg_limit_by_region_billing`,
`usage_units_by_status_window`, `events_count_window`, `churned_customers_hebrew`,
`customers_by_region_crm`, `top_clients_by_usage_december`, `known_as_of_snapshot`,
`new_since_resync`, `units_new_since`, `delete_usage_events` (refused `WRITE_BLOCKED`),
`usage_events_all_partitions_refused` (refused `PARTITION_SCAN_ALL`).

Two of those passes are the product's safety claims rather than arithmetic: the model was *supposed*
to be refused, and it wrote back the refusal code mdqm gave it.

### Failing (2) -- a confirmed product defect
`avg_limit_by_status_erp` and `sum_limit_by_status_erp`: mdqm answers 29719.63 / 34980000 where the
ground truth is 29748.72 / 34925000. `erp.customers` holds identical duplicate rows on purpose, and
E01's oracle says so explicitly ("a naive AVERAGE would double count them, so dedupe on the key").
The compiler never dedupes: the curated layer declares no `key` for that table.

A fix was attempted and **reverted rather than half-shipped**: deduping the anchor as
`(SELECT DISTINCT * FROM t)` broke column references in the JOIN path and one unit test that
*documents* the current behaviour. The design and both failure points are in the session handoff.

## Also found while building this
The first run failed four questions with `INVALID_SPEC`: the curated example layer declared only one
measure (`credit_limit_r5000`) while the questions need `usage_units`. Declaring it (identity
normalisation, in `examples/semantic_example.yaml`) fixed all four -- `units_new_since` now returns
`2401918`, exactly E01's value.

## Reproduce
```
python scripts/dev.py up pg                                  # one database per source
PYTHONPATH=src python -m mdqm.testing.loaders.postgres        # load the world
docker compose --profile pg --profile server up -d --build    # serve it
python scripts/bench_glm.py prompts --out bench/prompts       # one prompt per question
# a GLM 5.3 Flash agent answers each prompt into bench/answers/<id>.json
python scripts/bench_glm.py score --answers bench/answers     # score against E01
```

## Not verified
Only GLM 5.3 Flash was measured; no other model. Four of the fifteen questions were answered after the
`usage_units` fix rather than in the first pass, and the two erp failures remain open.
