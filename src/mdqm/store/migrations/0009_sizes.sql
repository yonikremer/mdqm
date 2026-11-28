-- 0009: measured table and partition sizes (the planner's input).
--
-- One row per sample per pass: a table, or one of its partitions, with the rows and bytes the source
-- reported. Rows accumulate on purpose -- the point is the *shape* of a table over time (the median
-- partition, the fat one) and its growth between passes, which a single mutable row could not show.
--
-- Not in the semantic layer: sizes change every night, and the layer's version hash covers every
-- fact in it, so a daily size would invalidate the context index and every cache key daily. They are
-- measurements, not declarations, and they belong beside the ledger.
--
-- `rows` and `bytes` are nullable because a source may report one and not the other (Oracle's
-- `num_rows` is an estimate, Elasticsearch reports docs and store bytes, an unanalyzed table reports
-- neither). NULL means "not reported", never zero.
CREATE TABLE IF NOT EXISTS size_sample (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT    NOT NULL,
    table_name   TEXT    NOT NULL,
    kind         TEXT    NOT NULL,
    name         TEXT    NOT NULL,
    rows         INTEGER,
    bytes        INTEGER,
    captured_at  TEXT    NOT NULL,
    CHECK (kind IN ('table', 'partition')),
    CHECK (rows IS NULL OR rows >= 0),
    CHECK (bytes IS NULL OR bytes >= 0)
);

-- The read the planner makes: the newest capture of one table, both kinds.
CREATE INDEX IF NOT EXISTS size_sample_latest_idx
    ON size_sample (source, table_name, captured_at DESC, id DESC);

-- The read an operator makes: everything from the last pass.
CREATE INDEX IF NOT EXISTS size_sample_capture_idx ON size_sample (captured_at DESC, id DESC);
