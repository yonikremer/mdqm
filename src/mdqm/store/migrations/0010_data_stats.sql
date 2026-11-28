-- 0010: column quality, categorical distributions, associations and partition extents.
--
-- Append-only, like size_sample. The newest collected_at of a column is what the linter reads;
-- older rows are how a category's mix drifting between partitions stays visible. NULL means the
-- source did not report that number. method records catalog / sample / exact / mixed.
--
-- Not in the semantic layer: a nightly refresh must not change the layer's version hash.
CREATE TABLE IF NOT EXISTS data_column_quality (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source            TEXT    NOT NULL,
    table_name        TEXT    NOT NULL,
    column_name       TEXT    NOT NULL,
    partition_name    TEXT    NOT NULL DEFAULT '',
    null_share        REAL,
    zero_share        REAL,
    empty_share       REAL,
    whitespace_share  REAL,
    distinct_count    INTEGER,
    minimum           TEXT,
    maximum           TEXT,
    rows_seen         INTEGER NOT NULL,
    sample_rows       INTEGER,
    method            TEXT    NOT NULL,
    collected_at      TEXT    NOT NULL,
    CHECK (method IN ('catalog', 'sample', 'exact', 'mixed')),
    CHECK (rows_seen >= 0),
    CHECK (null_share IS NULL OR (null_share >= 0 AND null_share <= 1)),
    CHECK (zero_share IS NULL OR (zero_share >= 0 AND zero_share <= 1)),
    CHECK (empty_share IS NULL OR (empty_share >= 0 AND empty_share <= 1)),
    CHECK (whitespace_share IS NULL OR (whitespace_share >= 0 AND whitespace_share <= 1))
);

CREATE INDEX IF NOT EXISTS data_column_quality_latest_idx
    ON data_column_quality (source, table_name, column_name, partition_name, collected_at DESC);

CREATE TABLE IF NOT EXISTS data_category_bucket (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    source                 TEXT    NOT NULL,
    table_name             TEXT    NOT NULL,
    column_name            TEXT    NOT NULL,
    bucket                 TEXT    NOT NULL DEFAULT '',
    payload                TEXT    NOT NULL,
    other_count            INTEGER NOT NULL,
    rows_seen              INTEGER NOT NULL,
    sample_rows            INTEGER,
    method                 TEXT    NOT NULL,
    distribution_complete  INTEGER NOT NULL,
    unprofiled             TEXT    NOT NULL,
    collected_at           TEXT    NOT NULL,
    CHECK (method IN ('catalog', 'sample', 'exact', 'mixed')),
    CHECK (other_count >= 0),
    CHECK (rows_seen >= 0),
    CHECK (distribution_complete IN (0, 1))
);

CREATE INDEX IF NOT EXISTS data_category_bucket_latest_idx
    ON data_category_bucket (source, table_name, column_name, bucket, collected_at DESC);

CREATE TABLE IF NOT EXISTS data_association (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT    NOT NULL,
    table_name   TEXT    NOT NULL,
    column_x     TEXT    NOT NULL,
    column_y     TEXT    NOT NULL,
    kind         TEXT    NOT NULL,
    score        REAL,
    payload      TEXT    NOT NULL,
    sample_rows  INTEGER NOT NULL,
    method       TEXT    NOT NULL,
    collected_at TEXT    NOT NULL,
    CHECK (kind IN ('cramers_v', 'mutual_info', 'numeric')),
    CHECK (method IN ('catalog', 'sample', 'exact', 'mixed')),
    CHECK (sample_rows >= 0),
    CHECK (score IS NULL OR score >= 0)
);

CREATE INDEX IF NOT EXISTS data_association_latest_idx
    ON data_association (source, table_name, column_x, column_y, collected_at DESC);

CREATE TABLE IF NOT EXISTS data_partition_extent (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT    NOT NULL,
    table_name      TEXT    NOT NULL,
    partition_name  TEXT    NOT NULL,
    rows            INTEGER,
    bytes           INTEGER,
    time_column     TEXT,
    time_min        TEXT,
    time_max        TEXT,
    method          TEXT    NOT NULL,
    collected_at    TEXT    NOT NULL,
    CHECK (method IN ('catalog', 'sample', 'exact', 'mixed')),
    CHECK (rows IS NULL OR rows >= 0),
    CHECK (bytes IS NULL OR bytes >= 0)
);

CREATE INDEX IF NOT EXISTS data_partition_extent_latest_idx
    ON data_partition_extent (source, table_name, partition_name, collected_at DESC);
