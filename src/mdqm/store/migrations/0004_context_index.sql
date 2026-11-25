-- 0004: the context index (H04): the searchable view of the semantic layer.
--
-- A weak agent asks "average credit limit by status" in Hebrew or English and must be handed the
-- right measure, value map and join. The text it searches is `context_document`: one row per
-- table, column, measure, categorical value, approved join and approved example query, with the
-- four fields FTS5 ranks (title, names, labels, body) already normalised by the loaders
-- (`mdqm.semantic.text.normalize_label`), so the index cannot fold one way and a query another.
--
-- The FTS5 virtual table is NOT created here. `CREATE VIRTUAL TABLE ... USING fts5` fails on a
-- Python build whose SQLite lacks FTS5, and a failing migration would take the audit log and the
-- history down with it; the index module creates it lazily and, when the capability probe says
-- FTS5 is missing, answers the same query with a LIKE scan of `context_document` (AGENTS.md
-- principle 1, principle 10).
--
-- `context_synonym` holds the label groups both sides expand: a query word and the words that mean
-- the same thing in the other language, keyed so one lookup finds them.
--
-- `context_index_state` is the one row that makes a rebuild incremental: the semantic layer's
-- `version_hash` plus what was indexed at it. An unchanged hash is a rebuild that writes nothing.
--
-- `empty_searches` is the data team's queue: a question that found nothing is the only cheap
-- evidence that the semantic layer is missing a word (DESIGN section 15, "log empty searches").
CREATE TABLE IF NOT EXISTS context_document (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT    NOT NULL UNIQUE,
    kind   TEXT    NOT NULL,
    ref    TEXT    NOT NULL,
    title  TEXT    NOT NULL,
    names  TEXT    NOT NULL,
    labels TEXT    NOT NULL,
    body   TEXT    NOT NULL,
    CHECK (length(doc_id) > 0)
);
CREATE INDEX IF NOT EXISTS context_document_kind_idx ON context_document (kind);

CREATE TABLE IF NOT EXISTS context_synonym (
    label TEXT NOT NULL,
    grp   TEXT NOT NULL,
    PRIMARY KEY (label, grp)
);
CREATE INDEX IF NOT EXISTS context_synonym_grp_idx ON context_synonym (grp);

CREATE TABLE IF NOT EXISTS context_index_state (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    version_hash TEXT   NOT NULL,
    documents   INTEGER NOT NULL DEFAULT 0,
    fts5        INTEGER NOT NULL DEFAULT 0,
    indexed_at  TEXT    NOT NULL,
    CHECK (documents >= 0),
    CHECK (fts5 IN (0, 1))
);

CREATE TABLE IF NOT EXISTS empty_searches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    searched_at TEXT    NOT NULL,
    query       TEXT    NOT NULL,
    user_id     TEXT,
    documents   INTEGER NOT NULL DEFAULT 0,
    CHECK (documents >= 0)
);
CREATE INDEX IF NOT EXISTS empty_searches_searched_at_idx ON empty_searches (searched_at);
