-- 0003: the query history (H03): one row per run, so a question can be asked again exactly.
--
-- The distinction the schema exists for: a *relative* re-run ("last 7 days") means something
-- different today than when it first ran, while a *frozen* re-run must reproduce the original
-- window and the original exec_ts. Reproducing either needs both spellings of each payload: the
-- original (with its now-like functions, `original_sql` / the fragments' `original_*`) and the one
-- that actually ran (`frozen_sql`, with `exec_ts` substituted). Storing only the executed payload
-- would make a relative re-run silently return the old window under a new timestamp.
--
-- Rows hold identifiers, timestamps, and the query itself. Never a result row, never a credential:
-- every text column is scrubbed of credential shapes before it is written, the count lands in
-- `redactions`, and a scrubbed row is refused by `rerun` instead of returning a fragment that
-- differs from what ran (AGENTS.md principle 11, DESIGN section 13).
--
-- The FTS index is NOT created here. `CREATE VIRTUAL TABLE ... USING fts5` fails on a Python build
-- without FTS5, and a failing migration would take the audit log down with it; `history.py` creates
-- the index lazily, inside its own writer transaction, and falls back to a LIKE scan when the
-- capability probe says FTS5 is missing (AGENTS.md principle 1).
CREATE TABLE IF NOT EXISTS query_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    user_id       TEXT    NOT NULL,
    tool          TEXT    NOT NULL,
    mode          TEXT    NOT NULL,
    exec_ts       TEXT    NOT NULL,
    spec_json     TEXT,
    original_sql  TEXT,
    frozen_sql    TEXT,
    dialect       TEXT,
    fragments     TEXT    NOT NULL,
    applied_fixes TEXT    NOT NULL,
    plan_json     TEXT,
    cache_json    TEXT,
    duration_ms   INTEGER,
    error_json    TEXT,
    redactions    INTEGER NOT NULL DEFAULT 0,
    search_text   TEXT    NOT NULL,
    CHECK (mode IN ('frozen', 'relative')),
    CHECK (duration_ms IS NULL OR duration_ms >= 0),
    CHECK (redactions >= 0)
);
CREATE INDEX IF NOT EXISTS query_history_created_at_idx ON query_history (created_at);
CREATE INDEX IF NOT EXISTS query_history_user_idx ON query_history (user_id, created_at);
