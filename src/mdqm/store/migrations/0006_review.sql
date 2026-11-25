-- 0006: the review queue (S12): every proposal a human has not decided yet, and every decision.
--
-- Two properties the table exists to guarantee:
--
-- 1. Nothing here writes `curated/*.yaml`. A row is a *proposal*; only an explicit approval (by a
--    named human, recorded below) reaches the semantic layer, and `queue.py` is the single writer.
-- 2. A rejected proposal is remembered. `id` is the content hash of the proposed fact, so a rerun
--    that proposes the same fact with the same evidence matches this row and is suppressed instead
--    of asking a human the same question again. When the evidence changes, `evidence_hash` no
--    longer matches and the row is resurfaced for a fresh decision.
--
-- No free text: `kind`, `section`, `key`, `proposed_by` and `decided_by` are identifiers, and
-- `patch`/`evidence` hold the YAML fragment and the evidence lines the proposal carries (never a
-- credential). `decided_at` is NULL until a human decides, which is what makes `pending` cheap.
CREATE TABLE IF NOT EXISTS review_proposals (
    id            TEXT    PRIMARY KEY,
    kind          TEXT    NOT NULL,
    section       TEXT    NOT NULL,
    "key"         TEXT    NOT NULL,
    field         TEXT    NOT NULL,
    collection    INTEGER NOT NULL DEFAULT 0,
    patch         TEXT    NOT NULL,
    evidence      TEXT    NOT NULL,
    evidence_hash TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    status        TEXT    NOT NULL,
    proposed_by   TEXT    NOT NULL,
    proposed_at   TEXT    NOT NULL,
    decided_by    TEXT,
    decided_at    TEXT,
    CHECK (length(id) = 64),
    CHECK (length(evidence_hash) = 64),
    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CHECK (collection IN (0, 1)),
    CHECK (status IN ('pending', 'approved', 'rejected', 'never_ask')),
    CHECK (status = 'pending' OR decided_at IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS review_proposals_status_idx ON review_proposals (status, confidence);
CREATE INDEX IF NOT EXISTS review_proposals_item_idx ON review_proposals (section, "key");
