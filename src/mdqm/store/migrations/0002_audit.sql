-- 0002: the audit log (H02): one append-only table plus the checkpoint that survives a purge.
--
-- Append-only is enforced by the database, not only by this package: the two triggers below abort
-- any UPDATE or DELETE, so a mistake in another lane, or an ad-hoc sqlite3 edit, is refused at the
-- storage layer instead of silently rewriting history. An attacker with write access to the file can
-- drop a trigger (H02's purge does exactly that, in a transaction); the hash chain is what detects
-- that, which is why the triggers are defence in depth and not the guarantee.
--
-- Rows hold identifiers, codes and counts only. Never SQL text, never a result row, never a secret:
-- the audit is read by the network admin, not by the agent (AGENTS.md principle 11, DESIGN s.13).
CREATE TABLE IF NOT EXISTS audit_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               TEXT    NOT NULL,
    user_id          TEXT    NOT NULL,
    tool             TEXT    NOT NULL,
    version_hash     TEXT,
    sources          TEXT    NOT NULL,
    row_count        INTEGER NOT NULL,
    blocked_attempts INTEGER NOT NULL,
    outcome          TEXT    NOT NULL,
    error_code       TEXT,
    prev_hash        TEXT    NOT NULL,
    hash             TEXT    NOT NULL,
    CHECK (row_count >= 0),
    CHECK (blocked_attempts >= 0),
    CHECK (length(hash) = 64),
    CHECK (length(prev_hash) = 64)
);
CREATE INDEX IF NOT EXISTS audit_log_ts_idx ON audit_log (ts);
CREATE UNIQUE INDEX IF NOT EXISTS audit_log_hash_idx ON audit_log (hash);
CREATE TABLE IF NOT EXISTS audit_checkpoint (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    last_purged_id   INTEGER NOT NULL,
    last_purged_hash TEXT    NOT NULL,
    purged_at        TEXT    NOT NULL,
    purged_rows      INTEGER NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_log_no_update BEFORE UPDATE ON audit_log BEGIN SELECT RAISE(ABORT, 'audit_log is append-only: rows are never updated'); END;
CREATE TRIGGER IF NOT EXISTS audit_log_no_delete BEFORE DELETE ON audit_log BEGIN SELECT RAISE(ABORT, 'audit_log is append-only: rows are never deleted'); END;
