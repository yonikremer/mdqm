-- 0007: the feedback queue the `feedback` tool writes to (M05).
--
-- Every row is a *claim the caller made*, never a fact about the data: nothing reads this table to
-- change the semantic layer on its own. A later consumer (the review queue / mining stages) reads
-- the rows, and a human approves anything derived from them (AGENTS.md principle 3).
--
-- No result row and no query text are stored: `query_id` is the caller-quoted id, `run_id` points at
-- query_history when the id resolved to one of the caller's own runs, and `corrected` is one
-- bounded sentence in which credential shapes are already replaced with [REDACTED].
CREATE TABLE IF NOT EXISTS query_feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT    NOT NULL,
    user_id    TEXT    NOT NULL,
    query_id   TEXT    NOT NULL,
    run_id     INTEGER,
    ok         INTEGER NOT NULL,
    corrected  TEXT,
    CHECK (length(user_id) > 0),
    CHECK (length(query_id) > 0),
    CHECK (ok IN (0, 1)),
    CHECK (run_id IS NULL OR run_id > 0)
);
-- The consumer's read path: oldest unread first, so nothing starves.
CREATE INDEX IF NOT EXISTS query_feedback_pending_idx ON query_feedback (id);
-- One caller's own history of ratings.
CREATE INDEX IF NOT EXISTS query_feedback_user_idx ON query_feedback (user_id, id DESC);
