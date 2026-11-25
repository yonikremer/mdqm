-- 0001: the metadata store's own key/value table.
--
-- Small, generic, and the only table H01 needs: later tasks add their own tables through new
-- numbered migrations (audit, history, proposals). Values here are identifiers and hashes, never a
-- credential: secrets live in the secret provider, not in the metadata DB (AGENTS.md principle 11).
CREATE TABLE IF NOT EXISTS mdqm_meta (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
