# Multi DB Query MCP: design (condensed, for coding agents)

This is the English condensation of the architecture page the author (Yoni) approved. If code and this file disagree,
the file wins unless a progress note explains a deliberate deviation. Anything marked **UNVERIFIED** has not been
checked against a real system: verify empirically in code and write the finding down.

## 1. Problem

A data analyst has data spread over several databases (PostgreSQL, Oracle, SQL Server), Elasticsearch, and a few
internal APIs that have an OpenAPI spec. Some DBs have dozens of tables. Coding agents (Claude Code on desktops,
backed by a mid-size model that is NOT reliably good at SQL) query them and cause these problems:

- non-standard joins across sources, with format/unit/rounding conversions (also between tables of one DB)
- queries that ignore indexes and partitions and hammer primary databases (there are no replicas)
- the same data fetched again and again
- shared DB passwords instead of organisational identity
- no history of queries, no audit, no control of run times
- categorical values differ per source ('A' vs 'Active' vs 1) and per language (Hebrew/English)
- schema drift silently breaks joins and conversions

Target environment: an air-gapped on-prem network (no internet), up to ~100 users (not concurrent), agents in Hebrew and
English. Development happens outside, with internet. **The product must be generic and plug-and-play for other
organisations**: nothing about the first organisation's versions or products is hard-coded.

Out of scope: row/column-level permissions and PII masking. Low priority: cross-source time consistency (just
record `fetched_at`).

## 2. Principles

1. **Generic.** Capability detection (probe) at install time instead of assumptions. Every optional feature declares the
   capability it needs and has a fallback. Org-specific facts live in config/YAML, never in code.
2. **Deterministic first.** Anything computable in code is computed in code. A model may only *propose* what code cannot
   know (names, synonyms, value mappings, conversion hypotheses chosen from a closed set), every proposal is validated
   against data, and humans approve before it becomes truth.
3. **Weak-model friendly.** Assume the agent will not read docs, will not call helper tools on its own, and will write
   poor SQL. So: few coarse tools, a structured query *spec* (JSON with enums) instead of free SQL, context injected
   automatically (hook and in every response/error), errors that say exactly what to fix.
4. **Read-only, defence in depth.** DB account with SELECT only + AST guard (also on SQL we compile ourselves) + read-only
   transaction. APIs: GET only. Elasticsearch: search only.
5. **Protect the primary DB.** Per-source semaphores, timeouts, EXPLAIN gate, partition pruning enforcement, quotas, row/byte caps.
6. **Air-gapped target.** No runtime internet access of any kind. All dependencies come from a PyPI mirror as wheels.
   No DuckDB extension installs, no model downloads, no CDN assets, no telemetry.
7. **OS agnostic.** Windows, Linux, macOS. No bash in shipped code. Use `pathlib`, no shell pipelines, no `fcntl`-only APIs.
8. **Plugin adapters.** The core knows an interface, not engines. New sources arrive as separate packages through an entry point.

## 3. Components

```
Claude Code + plugin (hook, skill, .mcp.json)
        | HTTP (streamable)
   MCP server  --- identity, audit
        |-- Context Pack (SQLite FTS5 over the semantic layer)
        |-- Compiler + linter + planner (sqlglot, partitions, EXPLAIN gate)
        |-- Semantic layer (YAML in git: generated/ + curated/)
        |-- Capability map (generated/capabilities.yaml)
        |-- DuckDB: segment cache + local joins/normalisation
        '-- Adapters (plugins) -> sources (read-only accounts)
Bootstrap pipeline: probe -> deterministic stages D1-D10 -> LLM proposer (A1-A5) -> validators -> gate -> YAML + review queue
```

- Metadata DB: SQLite (history, audit, mined-query stats, FTS index, proposals).
- Join executor: DuckDB by default. A second executor (e.g. Trino) may be added later behind an interface; do not build it now.

## 4. Adapter interface

```python
class SourceAdapter(Protocol):
    kind: str  # "postgresql", "oracle", ...

    def dialect(self) -> str: ...  # a sqlglot dialect name, or "dsl" for ES / API
    def probe(self) -> Capabilities: ...
    def extract_catalog(self) -> CatalogSnapshot: ...
    def execute(self, fragment, limits) -> ArrowStream: ...
    def cancel(self, handle) -> None: ...
    # optional: unimplemented -> raise NotSupported -> the probe marks the capability unsupported -> fallback runs
    def extract_partitions(self) -> list[Partitioning]: ...
    def explain(self, fragment) -> PlanSummary: ...
    def mine_queries(self, since) -> Iterator[RawQuery]: ...
    def mutability_stats(self) -> list[TableStats]: ...
```

Registered through entry-point group `mdqm.adapters`. The adapter never substitutes `now()` itself: it receives
already-rewritten fragments (the server owns `exec_ts`).

Built-in adapters: PostgreSQL, Oracle, SQL Server, Elasticsearch, OpenAPI. Per-source specifics:

- SQL adapters: small pool, timeout, read-only transaction, cancel, tag the query with the user id where the engine supports it.
- Elasticsearch: plain HTTP. Prefer ES|QL or the SQL API when the probe finds them, otherwise Query DSL (always available).
- OpenAPI: each GET collection endpoint is a virtual table. Query parameters are its "indexes". API key stays on the server.
  Shared rate limiter, TTL cache, single-flight.

### Capabilities and fallbacks

| Feature | Needs | Fallback |
|---|---|---|
| Partition planning | partition_metadata | indexes + cost gate only |
| Pruning proof | explain_plan | inject filter without proof, warn, cap scan using statistics |
| Mining past queries | query_store | MCP history only |
| Elasticsearch | query_mode | ES\|QL or SQL API if present, else Query DSL |
| Mutability classification | mutability_counters | checksum over a sample, else `unknown` (= treated as mutable) |
| Profiling | sampling | catalog statistics + small timed fetch |
| Structured model output | structured_output | validate JSON in code, up to 2 retries |
| Joins | (default) | DuckDB; other executor only if probe finds supported connectors |

`generated/capabilities.yaml` is machine written; an admin may override entries in `overrides.yaml` (admin wins). Re-probe nightly and on version change.

## 5. Query spec (what the agent writes instead of SQL)

```jsonc
{
  "from": "accounts",
  "select": ["customer_status", { "measure": "credit_limit_r5000", "agg": "avg" }],
  "time": { "mode": "events_in_range", "last": "3mo" },      // or { "mode": "new_since", "since": "2026-11-25T14:00:00" }
  "group_by": ["customer_status"],
  "limit": 50
}
```

Free SQL exists only as an emergency exit and goes through the same linter and guard. The agent never chooses the time column:
it states an intent (`events_in_range` or `new_since`) and the server maps it.

### Linter: auto-fix only when the meaning is preserved; otherwise error with an exact `fix`

| Situation | Behaviour |
|---|---|
| missing LIMIT | add from `max_rows` |
| categorical label in Hebrew/English or source code | replace with the canonical value; ambiguous = error listing valid values |
| unknown table/column | error "did you mean ..." |
| join not approved | error with the approved join and its full ON clause; format conversion added automatically |
| raw column that has a measure | replace with the measure and per-source normalisation |
| filter on insertion_time without bound | derive a bound on the indexed event column, else error |
| no filter on the partition key | default window on the key, or reject (per table setting) |
| function/cast on the partition key | rewrite into a range on the raw column so pruning works |
| now(), SYSDATE, GETDATE(), ... | replace with one absolute `exec_ts` |
| DML, DDL, several statements | block |

### Error shape

```json
{
  "status": "error",
  "code": "PARTITION_SCAN_ALL",
  "problem": "usage_events is partitioned monthly on start_time and has no filter on it. The whole table would be scanned",
  "fix": "add time: { mode: 'events_in_range', last: '30d' }",
  "partitions_if_fixed": "2 of 60"
}
```

Optional fields: `valid_values`, `did_you_mean`. No stack traces, ever.

## 6. Time columns

A table may have an event time (`start_time`, indexed) and an ingestion time (`insertion_time`, not indexed). Business questions
use event time. "What is new since my last fetch" MUST use ingestion time, because a late-arriving row would be missed otherwise.
Filtering on `insertion_time` alone is a full scan, so the declared lag gives a derived bound on the indexed column:

```
insertion_time >= X   =>   start_time >= X - max_lag
```
```sql
WHERE insertion_time >= TIMESTAMP '2026-11-25 14:00:00'
  AND start_time     >= TIMESTAMP '2026-11-25 08:00:00'   -- derived: minus max_lag (6h)
```

The original predicate stays, so the result is correct; the derived one only narrows the scan (and partitions).

| Question type | Column |
|---|---|
| about a period | event time |
| what is new, sync, tail refresh | ingestion time with derived bound |
| what was known at T | ingestion time <= T plus an event-time range |
| load checks | ingestion time |

`lag` is measured by the pipeline, approved by a human, verified nightly (violation disables the derivation until reviewed).
Unknown lag: do not derive; require a narrow range. A filter ending inside the last `max_lag` window gets a "partial data possible" note.

## 7. Partitions

Scanning all partitions is direct load on the primary. Partition metadata is extracted deterministically when the probe finds support;
the compiler injects/requires a filter on the key; the planner verifies pruning before running.

| Key is | Behaviour |
|---|---|
| event time | filter on it prunes directly; filter on ingestion time prunes through the derived bound |
| ingestion time | event range [X, Y) becomes ingestion range [X + min_lag, Y + max_lag); needs verified lag |
| other key / hash | needs a filter on the key itself; hash does not prune ranges |
| expression | prune only if the predicate matches the expression; needs human approval |
| ES time-based indices | choose indices by a narrow date pattern, never a broad wildcard |
| none | indexes + cost gate |

Pruning proof before running, without executing: PostgreSQL = count partitions in `EXPLAIN (FORMAT JSON)` (literals are inlined so pruning is at plan time);
Oracle = `PSTART/PSTOP` (numbers are proof, `KEY` is decided at run time so warn); SQL Server = count partitions in the plan (**UNVERIFIED format**);
Elasticsearch = the list of selected indices (no EXPLAIN). Without a readable plan: inject the filter and cap the scan by statistics.
Per-table `max_partitions_scanned`; the response shows `partitions_scanned: 3/60`. A new partition is non-breaking; a key change is breaking and alerts.

## 8. Cache

A cross-source query is split into one fragment per source and cut along partitions. Each piece is a **segment** stored in DuckDB and never fetched again while fresh. Every response shows when each segment was fetched.

```
fresh  <=>  age <= max(min_ttl, theta * (range_to - range_from))   and   age <= max_ttl
```
Defaults: `theta = 1%`, `min_ttl = 30s`, `max_ttl = 7d`, overridable per source/table. A 1-year pull from yesterday is fresh; a 1-week pull from yesterday is stale.

Mutability class (measured, not declared; unclassified behaves as mutable):

| Class | Policy |
|---|---|
| append_only | tail refresh; sealed segments kept until max_ttl |
| append_rarely | tail refresh + nightly checksum on a sample; an unexpected change downgrades the class |
| mutable_critical | no cache or short TTL; always full refresh |
| unknown | same as mutable_critical until classified |

Sealed segments are bounded only by max_ttl and the nightly check; the hot (current) partition follows the ratio rule. Tail refresh = `insertion_time >= fetched_at`
with the derived bound, merged by key; only for append tables with a unique key. `fresh=true` forces a fetch. API sources have TTL caches and call coalescing to protect their shared budget.
Identical concurrent calls are coalesced (single flight).

## 9. Weak-agent governance

- Context Pack in every answer and error, and from the client hook: tables, values, approved joins, time column and partition key, approved example specs. bm25 ranked, default budget 1,500 tokens, relevance threshold so non-data prompts get nothing.
- Five coarse tools: `find_data`, `query`, `get_more`, `history`, `feedback`. `query` does most of the work.
- Long queries answer early (`mode: "async"`): `query` returns `status: "started"`, a `query_id` and the
  plan (partitions, estimated rows/bytes), the work continues on a job thread, `get_more(query_id=…)`
  reports phase/`rows_read`/`partitions_done` (null until the source's own plan said what it will
  scan) and, when done, the same `summary` + `sample` + `result_handle` a blocking call returns; the
  same call with `cancel: true` asks the governor to stop, which calls the adapter's own `cancel` on
  the live stream. Blocking stays the default. The two references are separate fields on purpose:
  `result_handle` pages rows that exist, `query_id` watches work that is still running (a `j_` id),
  and a history run id stays `q_`/`<n>`; each namespace keeps its own prefix.
- Default time window and enforced pruning; EXPLAIN gate (expensive query gets a suggested fix, not a run).
- Semaphores per source and user, quotas, row/byte caps; result = summary + sample + handle (`get_more` pages).
- Feedback to the agent: error with exact `fix`, `applied_fixes` on every auto-correction.
- Output size: warn above 10,000 tokens, hard cap 25,000.

Claude Code plugin (one-step install; **verify formats in the current docs**): MCP server entry (HTTP, central server), a `UserPromptSubmit` hook calling `/context`
and returning `additionalContext`, and a skill that documents the spec. Personal identity via `headersHelper` / OAuth / token from an env var; never keys in shared policy files.

`POST /context` wire contract (task M06, `mdqm.server.context_endpoint`): request `{"prompt": str}` (max 4,000 characters, unknown fields refused), success `200 {"additionalContext": str}` with `""` when nothing is relevant — an irrelevant prompt costs zero tokens and is not an error. Refusals are the agent-facing error shape: `401` from the transport for a bad token, `400` `INVALID_SPEC` for a body the hook should not send, `503` `NOT_SUPPORTED` when the deployment has no semantic layer yet. The block is `build_pack` over the same indexed layer `find_data` uses, so a result row can never travel through the hook.

## 10. Bootstrap pipeline (semantic layer built autonomously)

Autonomous, scheduled, idempotent (stage inputs are hashed; unchanged input => skip, including model calls). Heavy work (profiling, stats, mining)
only inside `quiet_windows` (config; e.g. 21:00-07:00 local in the first org), per-source concurrency 1-2, stop if latency rises.

Deterministic stages:

| Stage | Output |
|---|---|
| D1 catalog | tables, columns, types, keys, indexes, comments, row estimates |
| D2 partitions | strategy, key, expressions, bounds, granularity (ES: index patterns, data streams, ILM) |
| D3 ES mapping | fields, types, aliases, templates |
| D4 OpenAPI | GET endpoints, filter params, pagination, auth, rate-limit headers; generates an adapter |
| D5 profiling | null rate, distinct, min/max, top values, formats, numeric granularity |
| D6 mutability + lag | class per table; distribution of (ingestion - event) |
| D7 join candidates | name similarity, types, containment, uniqueness; format conversions |
| D8 conversion detection | scale, rounding granularity, order of operations; threshold + counter-examples |
| D9 mining | fingerprints, joins, filters, time columns, conversion expressions from past queries (sqlglot) |
| D10 drift | catalog diffs: breaking vs not, mapped to affected YAML |

Agentic stages (the model *proposes*, closed JSON output validated in code, up to 2 retries with the error, temperature 0, prompt versions + model name stored, double run in different order, confidence computed from validators not from the model):

A1 names/descriptions (he+en), A2 synonyms, A1+A2 also the `labels` words of a table, a measure or a column (section 12) -- the payload the layer declares, so it is the one approval that writes a loadable section; a `name`/`synonym` payload still waits for a `names` section the layer does not have, A3 categorical value mapping (derived mostly from value co-occurrence over shared keys), A4 conversion hypotheses (choose among D8 hypotheses or "none", never invents a formula), A5 entities and cross-source joins (always human-approved).

Human gate: **auto** for deterministic facts (catalog, indexes, partitions, endpoints, statistics, drift, joins with a declared FK and full containment); **human approval** for anything a model proposed, any conversion, value map, new join, non-conservative mutability; **never** a proposal without evidence or a rejected one (rejected is remembered as negative memory).
The data team gets a weekly digest (email via internal SMTP if available, else a page on the server): at most 10 questions by impact, short yes/no/other. Answers are recorded (who/when) and become a PR on the YAML. Breaking schema changes go to the admin immediately.
`generated/` is rewritten by the pipeline; `curated/` is edited by humans only and wins, so reruns never overwrite manual work.

### Data statistics (continuous)

D5 profiles a table once (null rate, distinct, min/max, top values, format, granularity) and the size store records partition rows and bytes. The statistics pass keeps that record current and adds what those two do not:

* per column, the share of NULL, numeric 0, `''` and whitespace-only values, the distinct count, and min/max for numeric and time columns;
* for a categorical column (low cardinality, or text whose distinct values are a small share of the rows), the top values with counts and an `other` tail, stored per partition or time bucket so a later reading can show drift;
* for candidate pairs, how a category of X moves Y: NULL/0/`''`/whitespace share of Y inside each category, Cramér's V and mutual information when both are categorical, and per-category mean and nearest-rank quantiles when Y is numeric;
* per partition, rows, bytes where the catalog reports them, and min/max of the time column.

Collection is deterministic, read-only and incremental: a partition is read again only when it is new or its row or byte count changed. It runs in the quiet window, default 21:00-07:00 Asia/Jerusalem, or in `quiet_windows` when the deployment sets them. `force` overrides the clock, not a disabled collector. Catalog statistics come first (`pg_stats` / `pg_class`, Oracle `ALL_TAB_COL_STATISTICS`, SQL Server stats DMVs, Elasticsearch `_stats`). Zeros and blank strings are not in those views, so a sample (`TABLESAMPLE` or the dialect's equivalent, with the sample size recorded) fills them, and a table with no statistics is measured from a capped read. Every row stores `collected_at` and `method` (`catalog`, `sample`, `exact`, `mixed`) in the metadata database, not in the semantic layer, so a nightly refresh does not change the layer hash.

The same numbers are what a weak agent sees, and only for columns the question already matched: one short line in the Context Pack and in `find_data`. No sixth tool. The linter then uses them:

* a filter on a categorical value the column never holds is a hard error (`VALUE_NOT_IN_DATA`) before any query runs. The `fix` lists the closest real values, each with its frequency. Matching folds case and whitespace, strips Hebrew points, accepts a consonantal transliteration, and follows the value map's synonyms. When the stats are not a complete reading of every partition (a sample, a tail, or a partition that appeared after `collected_at`), a bounded `SELECT 1 ... LIMIT 1` checks the value before the error is raised. If that lookup cannot run, the filter is not rejected.
* `''` and `0` that a sentinel file marks as missing are read through `NULLIF`. The compiler emits it and `applied_fixes` says so. The file is separate from the semantic layer so marking a sentinel does not change the version hash.
* a join key whose NULL share is at least 20% is reported: those rows will be dropped.
* measured partition sizes pick a window that stays under the row block, and a spec with no time filter that would read more than the block is refused with that window.
* an unaggregated spec whose estimated row count (`table rows × filter selectivity`) exceeds the block is refused with `COST_TOO_HIGH` and told to aggregate or narrow. Between the warn threshold and the block the spec runs and the estimate is reported.

## 11. Learning from past queries (deterministic, no LLM)

Sources decided by the probe: PostgreSQL `pg_stat_statements`, SQL Server Query Store, Oracle `V$SQL` (AWR needs a licence: not default), Elasticsearch slow logs if enabled, and always the MCP's own history (the only source of cross-source joins and of user identity).
Steps: pull by watermark -> parse with sqlglot per dialect (failures counted) -> normalise (strip literals, replace now-like with a marker, hash the AST = fingerprint) ->
extract tables, join edges (including conversions like `LPAD`), filter columns, time columns, GROUP BY, aggregations, conversion expressions (`ROUND(x/5000)*5000`) -> upsert counts per fingerprint.
`score = a*log(count) + b*0.5^(age/half_life) + c*distinct_users`. Ranking orders joins/examples in the Context Pack and suggests `use_by_default` columns; conversion expressions found in history raise the prior of D8 hypotheses.
"Common" is not "correct": ranking influences order and proposals only; it approves nothing.

## 12. Semantic layer file (YAML in git)

See `examples/semantic_example.yaml` (complete worked example). Key ideas: sources with load limits; `value_maps` (canonical values, synonyms in both languages, per-source code mappings);
`measures` (one canonical definition with rounding rule, plus a per-source `normalize` template: `{col}`, `ROUND({col} / 5000) * 5000`, `{col} * 10`);
`tables` (key, default window, partitioning, time columns with role/indexed/lag, mutability class and approval status, cache override); `approved_joins` (with cardinality and usage counts);
`labels` (the deployment's own words for a thing, keyed `table:<source.table>`, `measure:<name>` or `column:<qualified>`).
Labels are how one field serves every language: the identifiers stay as the source spells them, a curated file adds the words people type (Hebrew for a Hebrew question, an internal nickname, an abbreviation), and the context index matches them. A label key that names nothing in the layer or a catalog is refused at index time, because a silently ignored label is a search that mysteriously misses.
The version hash goes into cache keys and audit. Every item may carry `provenance` (who proposed, evidence, match rate, who approved).

### The motivating conversion case

One field must be rounded to a multiple of 5000. In most sources that is the stored form; in one source it is stored raw (unrounded); in another it is rounded to 5000 and then divided by 10.
So the canonical measure is `round_half_up(x / 5000) * 5000` and the three normalisations are identity, `ROUND(x/5000)*5000`, and `x * 10`. Rounding mode (half_up vs half_even) is a stated, human-approved property.
D8 must rediscover this from data; D9 hints (rounding expressions in old queries) raise its confidence.

## 13. Identity, security, audit

Identity is the user's, but sources are always accessed with a read-only service account held by the server; no user ever receives a DB password; API keys stay on the server; secrets in an encrypted store, never in YAML or code.
How personal identity is issued in the first org is **unknown** (probably an internal directory, to be checked by Yoni); therefore identity is a pluggable `TokenVerifier` and the dev implementation is static tokens.
Options to support later: `headersHelper` script returning a short-lived token, OAuth/OIDC, reverse proxy with Kerberos adding an identity header.
Every query is tagged with the user id at the source where supported. Shared cache/examples: if row-level permissions ever appear, cache keys need a permission scope (not now).

History: spec or original SQL, SQL with frozen `exec_ts`, per-source SQL, applied fixes, errors, partitions, cache, `fetched_at`; retention 1 year.
`exec_ts`: `now()`, `CURRENT_TIMESTAMP`, `SYSDATE`, `GETDATE()` and ES date math are replaced by absolute values; rerun can be `frozen` (same timestamp) or `relative` (against now).
Audit: user, tool, semantic version hash, sources, rows, blocked attempts; append-only with hash chaining; retention 1 year; readable by the network admin.
Observability: per-source/tool latency histograms, timeouts, cache hits, partitions scanned, first-try success, auto-fix rate, searches with no result.

## 14. Deployment

One container image; config via env + a volume; secrets from files (`/run/secrets`); TLS at the edge; single instance;
backup = metadata DB + the YAML git repo (cache is disposable). Rough size (estimate): 4 vCPU, 16 GB RAM, 200 GB SSD, DuckDB `memory_limit` ~60% of RAM.
Concurrency defaults: 8 queries in the server, 2 per user, 2-3 per source. The first org's cloud platform is unknown, so nothing may depend on Kubernetes or a specific VM.
Offline: wheels from the internal PyPI, FTS5 from Python's sqlite (checked by the probe), Oracle in Thin mode (no client), model reached through its internal API.

## 15. Risks to keep in mind while coding

DuckDB pulls whole fragments (cap them, prune, cost gate) | sqlglot translation is best effort (simple fragments, `unsupported_level=RAISE`, golden tests) | big context confuses (budget, relevance threshold, measure) |
auto-fix that changes meaning looks like a correct answer (only meaning-preserving, always reported) | wrong metadata makes enforcement frustrating (verify, default window, message with fix) |
the cache rule is a heuristic (mutability class, nightly checks, max_ttl, `fetched_at`) | probe lies (admin override, health check at startup) | the autonomous pipeline invents things (validation, 10-question cap, negative memory) |
Hebrew morphology hurts FTS (synonyms both sides, prefix variants, log empty searches).

## 16. Roadmap

0. Discovery: probe and D1-D5 read-only in quiet windows; an eval harness of 10-15 real questions against the model (spec vs direct SQL, with and without context); plugin check (connect, hook, tool search, token issuance). Exit: numeric baseline + capability map.
1. MVP: central HTTP server, 5 tools, one-step plugin; PostgreSQL + one more adapter; compiler, linter, `exec_ts`, planner with partitions and EXPLAIN; cache with classification; pipeline v0 (probe, D1, D2, D4-D6, D10, D8 in verify mode, A1-A2, digest, D9 from MCP history and `pg_stat_statements`).
2. Completion: remaining adapters, D3, D7, D8 discovery mode, D9 from all sources, A3-A5, tail refresh, semi-join, API rate limiter, OIDC if present, decision on a second executor.
3. Expansion: `query_metric`, admin dashboard, materialisation, embeddings only if measured recall is low, packaging for other orgs.
