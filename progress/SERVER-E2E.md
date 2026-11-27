status: in progress
task: server end-to-end
agent: claude

## Verified working through the running container (scripts/mdqm_call.py + curl)
- `find_data`, `query` (single source), `query` (FEDERATED: billing.usage_events JOIN crm.customers --
  returns E01's ground truth 1810474 / 956379 / 354603), `get_more`, `history` (answer field is
  `runs`), `feedback` (record saved for run 232).
- `GET /healthz` 200; `POST /context` 200 with `{"prompt": "..."}` (top-level, not nested) and a
  bearer token, returning `additionalContext`.
- `mdqm audit verify` -> "audit chain intact: 234 row(s) verified".
- `mdqm review list` -> "nothing to review" (queue reachable).
- Server starts even when an adapter's driver is absent: discovery logs
  "adapter 'oracle' is not installed in this deployment ... it is unavailable".

## Fixed
1. `ToolServices` (history/feedback/get_more stores) was never configured in `serve()`.
2. Two `ResultStore` instances (Governor built its own) -> handles were invisible to `get_more`;
   one instance is now shared by `query_services(handles=)` and `tool_services(handles=)`.
3. Discovery raised `AdapterRegistrationError` on a missing optional driver and stopped the server;
   a missing driver is now a capability fact (warning + `_unavailable`), not a fatal error.
4. `python -m mdqm` does not exist (no `__main__`); the CLI is `python -m mdqm.cli <group>`.

## Open: the bootstrap pipeline has no stages wired (measured, not guessed)
`mdqm bootstrap run` (inside the container) answers:

    source crm: skipped (no adapter registered for 'postgresql' (registered kinds: none) ...)
    no bootstrap stage is registered in this build; nothing to run

Two separate causes:
1. `available_stages()` in `src/mdqm/pipeline/core.py` returns `()` on purpose ("nothing is
   registered without its source dependency"), so the runner has an empty list.
2. `src/mdqm/pipeline/cli.py` never asks the adapter registry to `discover()`, so even a wired stage
   would find "registered kinds: none".

The eight stage classes exist (`CatalogStage`, `OpenApiStage`, `ConversionStage`, `JoinStage`,
`ProfilingStage`, `MutabilityStage`, `DriftStage`, `ProposeNamesStage`) but each needs real inputs
(`adapters`, `specs`, `pairs`, `measure`), so `available_stages()` cannot invent them: its signature
must take the wiring the CLI already builds from settings, and the CLI must discover adapters first.

An attempt to register `Stage()` with no arguments was reverted (mypy: 12 missing-argument errors;
the stage constructors demand their inputs). The correct shape is `available_stages(adapters=...,
specs=..., pairs=..., measure=...)` built by `pipeline/cli.py` from settings, then passed to
`run_pipeline`.

## Next
1. Wire `available_stages(...)` + adapter discovery so `mdqm bootstrap run` actually runs D1+D2..A1+A2
   against the live sources, and assert it rediscovers the 5000 / 5000-then-/10 conversions (E01's
   acceptance claim).
2. Exercising `/context` returned `additionalContext` -- check it is non-empty for a real question.
3. SQL-sink debt: `python scripts/check_sql_sinks.py src/mdqm` -> 42 statements built from values
   (mostly `testing/loaders/*` DDL and 2 `cache/segments.py` sites); the checker has no tests yet.
