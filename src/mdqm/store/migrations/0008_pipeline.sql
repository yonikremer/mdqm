-- 0008: the bootstrap pipeline ledger (S01): one row per stage attempt.
--
-- The rule DESIGN.md section 10 states is "stage inputs are hashed; unchanged input => skip": a row
-- here is the memory that makes the rule possible. `status` is the verdict of one attempt, so the
-- operator can see why last night did nothing:
--
--   ran      the stage executed and its output reached generated/
--   skipped  the input hash matched the last `ran` row (and its outputs were still on disk)
--   refused  a heavy stage outside its quiet window; nothing was run and nothing was written
--   failed   the stage raised; the attempt is kept so a rerun resumes from it
--
-- Only `ran` rows make a later run skip, which is why a failed attempt can never look like success.
-- `inputs_hash` is a sha256 over the stage's declarative inputs (never a result and never a secret);
-- the counters, the skips, the applied fixes and the output paths live in JSON columns because
-- nothing queries inside them -- a reader selects the whole row and prints it.
CREATE TABLE IF NOT EXISTS pipeline_run (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    stage          TEXT    NOT NULL,
    inputs_hash    TEXT    NOT NULL,
    status         TEXT    NOT NULL,
    started_at     TEXT    NOT NULL,
    finished_at    TEXT    NOT NULL,
    counts         TEXT    NOT NULL DEFAULT '{}',
    skipped        TEXT    NOT NULL DEFAULT '[]',
    applied_fixes  TEXT    NOT NULL DEFAULT '[]',
    outputs        TEXT    NOT NULL DEFAULT '[]',
    error          TEXT,
    forced         INTEGER NOT NULL DEFAULT 0,
    CHECK (length(inputs_hash) = 64),
    CHECK (status IN ('ran', 'skipped', 'refused', 'failed')),
    CHECK (forced IN (0, 1))
);
CREATE INDEX IF NOT EXISTS pipeline_run_stage_idx ON pipeline_run (stage, status, id DESC);
