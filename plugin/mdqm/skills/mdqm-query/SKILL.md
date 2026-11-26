---
name: mdqm-query
description: Query organisational data through the mdqm MCP server. Use whenever the user asks for a number, a list, a trend or a comparison that lives in a company database (customers, accounts, usage, billing, orders) instead of asking them for SQL or a table name.
when_to_use: The user asks about business data ("how many active customers", "average credit limit", "usage last month", "כמה לקוחות פעילים"), or mentions a table, column, measure or status they want aggregated or filtered.
allowed-tools: mcp__plugin_mdqm_mdqm__find_data mcp__plugin_mdqm_mdqm__query mcp__plugin_mdqm_mdqm__get_more mcp__plugin_mdqm_mdqm__history mcp__plugin_mdqm_mdqm__feedback
---

# Querying data with mdqm

The `mdqm` server gives you five tools. You never write SQL: you send a **JSON query spec** and get a
summary, a sample and a handle back.

| Tool | Call it when |
|---|---|
| `find_data` | You are not sure of a table, column, measure or status-value name. It returns the matching semantic facts and a spec to start from. |
| `query` | You know what to ask. Takes the spec. Returns a summary, a sample, a `handle`, and a run `id`. |
| `get_more` | You have a `handle` and need more rows. `limit` up to 1000. |
| `history` | You need your own earlier runs, or what a re-run would send. `rerun_id` asks what a re-run would send. |
| `feedback` | **Once per answered question**, after the user says the answer was right or wrong. `ok: false` needs `corrected`. |

**Always** call `find_data` first when the user uses a word you have not seen in an answer from this
server. Guessing a measure or a Hebrew status label costs a round trip and an error.

## The spec

```json
{
  "from": "accounts",
  "select": ["customer_status", {"measure": "credit_limit_r5000", "agg": "avg"}],
  "time": {"mode": "events_in_range", "last": "3mo"},
  "filters": [{"column": "customer_status", "op": "eq", "value": "פעיל"}],
  "group_by": ["customer_status"],
  "order_by": [{"by": "customer_status", "direction": "asc"}],
  "limit": 50,
  "joins": ["usage_to_customers"],
  "dry_run": false,
  "fresh": false
}
```

* `from` — one table, index or endpoint name from the catalog. Required.
* `select` — required. A plain column name, or `{"measure": ..., "agg": ...}`. `agg` is one of
  `avg`, `sum`, `min`, `max`, `count`, `count_distinct`; omit it to use the measure's own default.
  `alias` renames the output column.
* `time` — the window **intent**; the server picks the column. Never name a time column yourself.
  * `{"mode": "events_in_range", "last": "3mo"}` — trailing window, units `d`, `w`, `mo`, `y`.
  * `{"mode": "events_in_range", "from": "2026-06-01", "to": "2026-11-01"}` — absolute, `to` exclusive.
  * `{"mode": "new_since", "since": "2026-11-25T14:00:00"}` — what arrived since a previous fetch; the
    server uses ingestion time with a derived bound so late rows are not missed.
* `filters` — `op` is `eq`, `ne`, `in`, `contains`, `gt`, `gte`, `lt`, `lte`, `range`.
  `in` takes a list, `range` takes exactly two values `[low, high]`, everything else one value.
  Filter **values** may be the business label in Hebrew or English (`"פעיל"`); the server maps it to
  the source's own value (`'A'`). Values are never SQL.
* `group_by` — column, measure or alias names. `month` is available on time-based tables.
* `joins` — approved join **names** only, never a join condition. `find_data` and `query` errors
  list the ones that exist.
* `dry_run: true` — plan and lint, no rows. Cheap way to check a spec you are unsure about.
* `fresh: true` — skip the cache. Use only when the user asks for live numbers; a cached answer
  carries `fetched_at`, which you should mention instead.

A table with a partition key refuses a query without a `time` window, with a `fix` that names the
window. Do not fight it: add the window the user means.

## Three worked examples

**1. A grouped average over a trailing window.** "Average credit limit by status, last 3 months."

```json
{"from": "accounts",
 "select": ["customer_status", {"measure": "credit_limit_r5000", "agg": "avg"}],
 "time": {"mode": "events_in_range", "last": "3mo"},
 "group_by": ["customer_status"],
 "limit": 50}
```

**2. Late-arriving rows.** "What is new in usage events since my last fetch?"

```json
{"from": "usage_events",
 "select": ["event_id", "client_no", "units"],
 "time": {"mode": "new_since", "since": "2026-11-25T14:00:00"},
 "limit": 200}
```

**3. A Hebrew filter across an approved join.** "Events per month for active customers."

```json
{"from": "usage_events",
 "select": [{"measure": "event_count", "agg": "count"}],
 "filters": [{"column": "customer_status", "op": "eq", "value": "פעיל"}],
 "time": {"mode": "events_in_range", "from": "2026-06-01", "to": "2026-11-01"},
 "group_by": ["month"],
 "joins": ["usage_to_customers"],
 "limit": 50}
```

## When a call fails

Every failure is an error object with `code`, `problem` and `fix`. Read `fix` and do exactly that —
it names the valid options (`valid_values`, `did_you_mean`) and, for a partition problem,
`partitions_if_fixed`. Retry **once**. If it fails again, tell the user what the server asked for
instead of trying a different guess.

Never invent a table, measure, join or status value to get past an error.

## Reporting back

Quote the numbers with their units and the `exec_ts` the answer used, and say when the answer came
from cache. Then call `feedback` once with the run `id`. Anything derived from that rating is
reviewed by a human before it changes the semantic layer, so a `corrected` note helps and never
overwrites anything directly.
