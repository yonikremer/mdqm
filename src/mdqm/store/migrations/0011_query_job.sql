-- 0011: one row per asynchronous query job, so "how is it going?" survives the call that started it.
--
-- A job is a `query` run the caller did not wait for. The result itself never lives here: when the
-- run finishes, `result_handle` names the row in the governor's result store, exactly like a
-- blocking answer. This table holds only the bookkeeping a poll reads -- state, phase, measured
-- rows, how many partitions the plan said, and the error if it failed.
--
-- `state` is the field the agent branches on. `queued` -> `running` -> one of `done`, `failed`,
-- `cancelled`; the CHECK is the whole state machine, so a typo cannot invent a sixth state.
--
-- `partitions_done` is NULL while running: one fragment is pruned by the source, not iterated by
-- us, so per-partition progress is unknown until the plan reports what it scanned. NULL means
-- "not reported", never zero (the same rule the size profile keeps).
--
-- `query_id` is the opaque id the agent quotes back (`j_7`). It is a different namespace from a
-- history run id (`q_12`), which is why the prefix differs: `feedback` resolves a run id against
-- `query_history`, and one id space with two meanings would silently rate the wrong run.
CREATE TABLE IF NOT EXISTS query_job (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           TEXT    NOT NULL,
    query_id          TEXT    NOT NULL UNIQUE,
    state             TEXT    NOT NULL,
    source            TEXT    NOT NULL DEFAULT '',
    phase             TEXT    NOT NULL DEFAULT '',
    est_rows          REAL,
    partitions_total  INTEGER,
    partitions_done   INTEGER,
    rows_read         INTEGER NOT NULL DEFAULT 0,
    bytes_read        INTEGER NOT NULL DEFAULT 0,
    result_handle     TEXT,
    rows_total        INTEGER,
    error_code        TEXT,
    error_json        TEXT,
    note              TEXT    NOT NULL DEFAULT '',
    created_at        TEXT    NOT NULL,
    started_at        TEXT,
    finished_at       TEXT,
    CHECK (state IN ('queued', 'running', 'done', 'failed', 'cancelled')),
    CHECK (rows_read >= 0),
    CHECK (bytes_read >= 0),
    CHECK (est_rows IS NULL OR est_rows >= 0),
    CHECK (partitions_total IS NULL OR partitions_total >= 0),
    CHECK (partitions_done IS NULL OR partitions_done >= 0),
    CHECK (rows_total IS NULL OR rows_total >= 0)
);

CREATE INDEX IF NOT EXISTS query_job_owner_idx ON query_job (user_id, id DESC);

CREATE INDEX IF NOT EXISTS query_job_state_idx ON query_job (state, id DESC);
