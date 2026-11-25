-- 0003: stage D9 mining (S08): per-fingerprint query counters, the ingested-batch ledger and the
-- per-source watermark that the next pull starts from.
--
-- Everything here is a *candidate*: counts of statements seen, never a fact about the data. The
-- normalised statement (literals already stripped by mdqm.sql.fingerprint) and the extracted facts
-- live in one JSON column per fingerprint -- one object with nested lists, not a table of records --
-- because nothing queries inside it; the ranked reads select whole rows.
--
-- No SQL text as it was written and no result row is stored: the normalised shape is enough for a
-- human to recognise the query, and it cannot carry a literal with a customer name in it. Parse
-- failures are not persisted either: the run that saw them reports them (``IngestReport.failures``)
-- with a hash of the text, so a statement we cannot parse leaves no trace in the store.
CREATE TABLE IF NOT EXISTS mined_query (
    source      TEXT    NOT NULL,
    fingerprint TEXT    NOT NULL,
    dialect     TEXT    NOT NULL,
    calls       INTEGER NOT NULL DEFAULT 0,
    statements  INTEGER NOT NULL DEFAULT 0,
    users       TEXT    NOT NULL DEFAULT '[]',
    first_seen  TEXT,
    last_seen   TEXT,
    facts       TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL,
    PRIMARY KEY (source, fingerprint),
    CHECK (calls >= 0),
    CHECK (statements >= 0),
    CHECK (length(fingerprint) = 64)
);
CREATE INDEX IF NOT EXISTS mined_query_rank_idx ON mined_query (source, calls DESC);
CREATE TABLE IF NOT EXISTS mined_ingest (
    source       TEXT    NOT NULL,
    batch_hash   TEXT    NOT NULL,
    mined_at     TEXT    NOT NULL,
    statements   INTEGER NOT NULL,
    fingerprints INTEGER NOT NULL,
    watermark    TEXT,
    PRIMARY KEY (source, batch_hash),
    CHECK (length(batch_hash) = 64)
);
CREATE TABLE IF NOT EXISTS mining_watermark (
    source     TEXT NOT NULL PRIMARY KEY,
    watermark  TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
