"""The review queue: proposals wait, a human decides, and only approval writes ``curated/*.yaml``.

The rule this module exists to enforce (``docs/DESIGN.md`` section 10): a model proposes, code
validates, a **human approves**. Nothing else in ``mdqm`` may write the curated layer, so every
question a machine wants to ask about the organisation comes through here and ends in one of four
states: ``pending`` (waiting), ``approved`` (written, with provenance), ``rejected`` (remembered,
never asked again with the same evidence) and ``never_ask`` (remembered, never asked again at all).

Two consequences that are easy to get wrong, so they are spelled out:

* **A rejection is data, not an absence.** The row stays, keeps its evidence, and the next run that
  proposes the same fact with the same evidence is suppressed silently. When the evidence changes,
  ``evidence_hash`` no longer matches and the proposal is resurfaced -- a human decided against the
  old evidence, not against the new one.
* **The id is the content, not the run.** ``id`` is the SHA-256 of the proposed fact (kind, section,
  target, patch), so a nightly rerun updates one row instead of filling the queue with duplicates.

Approval writes the file with ``ruamel.yaml``'s round-trip parser, so a human's comments survive
the machine's edit. Every approval and rejection is also appended to the hash-chained audit log
(task H02): who decided what, and which proposal the decision produced.
"""

from __future__ import annotations

import difflib
import hashlib
import io
import json
import os
import sqlite3
import tempfile
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from pydantic import Field
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import YAMLError

from mdqm.cache.freshness import Clock, SystemClock
from mdqm.models.base import JsonObject, MDQMModel, UtcDatetime
from mdqm.semantic.errors import SemanticError
from mdqm.semantic.loader import CURATED_DIR, load
from mdqm.store.audit import AuditEntry, AuditLog, AuditOutcome
from mdqm.store.db import Store

__all__ = [
    "Proposal",
    "ProposalKind",
    "ProposalStatus",
    "Question",
    "ReviewError",
    "ReviewQueue",
    "SubmitAction",
    "SubmitResult",
    "proposal_id",
]

#: The table holding every proposal and every decision.
PROPOSALS_TABLE: str = "review_proposals"

# The statements that name a table, built once from the constant above; values are bound.
_SELECT_PROPOSAL: Final[str] = f"SELECT * FROM {PROPOSALS_TABLE} WHERE id = ?"  # noqa: S608 - module constant

#: A section name is a file stem, never a path: the same rule as ``semantic.writer``.
_ALLOWED_SECTION_CHARS: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
)


class ReviewError(RuntimeError):
    """A review-queue problem the caller must not ignore (unknown id, bad transition, bad YAML)."""


class ProposalKind(StrEnum):
    """What kind of semantic fact a proposal adds.

    Deliberately open-ended: the mining pipeline (S08) proposes these, and a new kind must not
    need a schema change. ``OTHER`` is the honest label for anything the queue does not special
    case, and it is still validated by the loader when it is approved.
    """

    NAME = "name"
    SYNONYM = "synonym"
    VALUE_MAP = "value_map"
    MEASURE = "measure"
    JOIN = "join"
    MUTABILITY = "mutability"
    OTHER = "other"


class ProposalStatus(StrEnum):
    """Where a proposal stands. Only ``pending`` awaits a human; the rest are decisions."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEVER_ASK = "never_ask"


class SubmitAction(StrEnum):
    """What :meth:`ReviewQueue.submit` did with a proposal, so a caller can report it honestly."""

    CREATED = "created"
    #: The same fact is already waiting with the same evidence: nothing to ask twice.
    UNCHANGED = "unchanged"
    #: Already approved. Re-proposing it must not reopen the question.
    ALREADY_APPROVED = "already_approved"
    #: Rejected before, with the same evidence: the negative memory suppressed it.
    SUPPRESSED = "suppressed"
    #: Rejected before, but the evidence changed: the question is open again.
    RESURFACED = "resurfaced"


def _canonical_json(value: Any) -> str:
    """Stable JSON for hashing: sorted keys, no whitespace, UTF-8 so Hebrew hashes the same."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def proposal_id(kind: ProposalKind, section: str, key: str, field: str, patch: JsonObject) -> str:
    """The content hash of a proposed fact, which is also its primary key.

    The evidence is **not** part of the id: the same fact proposed with new evidence must land on
    the same row so the queue can notice that the evidence changed (see :class:`SubmitAction`).
    """
    payload = {
        "kind": kind.value,
        "section": section,
        "key": key,
        "field": field,
        "patch": patch,
    }
    return _digest(_canonical_json(payload))


def evidence_hash(evidence: tuple[str, ...]) -> str:
    """The content hash of the evidence lines, compared to decide whether to resurface."""
    return _digest(_canonical_json(sorted(evidence)))


class Proposal(MDQMModel):
    """One proposed semantic fact, and the decision made about it.

    ``section`` is the YAML top-level key and the curated file stem (``value_maps`` ->
    ``curated/value_maps.yaml``). ``key`` names the item inside it, verbatim: a table key such as
    ``billing.usage_events`` contains a dot of its own, so it can never be recovered by splitting a
    dotted path. ``field`` is the dotted path *inside* that item (``mutability``,
    ``synonyms.active``). ``collection`` says the section holds a *list* of named entries
    (``approved_joins``), matched by ``name``. ``evidence`` is what a human reads to decide.
    """

    id: str = Field(min_length=64, max_length=64)
    kind: ProposalKind
    section: str = Field(min_length=1)
    #: The item inside ``section``: a map key or a list entry's ``name``.
    key: str = ""
    #: Dotted path inside that item; empty when the patch *is* the item.
    field: str = ""
    collection: bool = False
    patch: JsonObject
    evidence: tuple[str, ...] = ()
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: ProposalStatus = ProposalStatus.PENDING
    proposed_by: str = "pipeline"
    proposed_at: UtcDatetime
    decided_by: str | None = None
    decided_at: UtcDatetime | None = None

    @property
    def evidence_digest(self) -> str:
        """The hash the queue compares to decide whether this proposal is genuinely new."""
        return evidence_hash(self.evidence)

    @property
    def location(self) -> tuple[str, ...]:
        """Where the patch goes: the section, the item key, then the dotted ``field``."""
        parts = tuple(part for part in self.field.split(".") if part)
        if not self.key:
            return (self.section, *parts)
        return (self.section, self.key, *parts)

    @property
    def target(self) -> str:
        """Human-readable location, for digests and for grouping a question's summary."""
        return ".".join(part for part in (self.key, self.field) if part)


class SubmitResult(MDQMModel):
    """The outcome of :meth:`ReviewQueue.submit`: what happened, and the row now stored."""

    action: SubmitAction
    proposal: Proposal

    @property
    def asks_a_human(self) -> bool:
        """True only when the queue gained a question nobody has answered yet."""
        return self.action in (SubmitAction.CREATED, SubmitAction.RESURFACED)


class Question(MDQMModel):
    """One thing to ask a human, possibly covering several proposals of the same confidence.

    ``proposals`` is a tuple of ids so a bulk item can be approved in one call. Bulk approval of
    high-confidence items counts as **one** question: the digest must not drown a reader in rows
    they would approve anyway.
    """

    summary: str = ""
    proposals: tuple[str, ...] = ()
    bulk: bool = False

    @property
    def count(self) -> int:
        return len(self.proposals)


def _row_to_proposal(row: sqlite3.Row) -> Proposal:
    """Rebuild the model from a stored row; a row that no longer parses is an error."""
    try:
        return Proposal(
            id=row["id"],
            kind=ProposalKind(row["kind"]),
            section=row["section"],
            key=row["key"],
            field=row["field"],
            collection=bool(row["collection"]),
            patch=json.loads(row["patch"]),
            evidence=tuple(json.loads(row["evidence"])),
            confidence=row["confidence"],
            status=ProposalStatus(row["status"]),
            proposed_by=row["proposed_by"],
            proposed_at=row["proposed_at"],
            decided_by=row["decided_by"],
            decided_at=row["decided_at"],
        )
    except (ValueError, TypeError, KeyError, IndexError) as exc:  # ValidationError is a ValueError
        raise ReviewError(f"proposal row {row['id']} is corrupt: {exc}") from exc


class ReviewQueue:
    """Every proposal and every decision, over one :class:`~mdqm.store.db.Store`.

    ``root`` is the semantic directory (the one holding ``curated/``). The queue is not a
    singleton: pass the store, the root and, in tests, the clock. It is the **only** writer of the
    curated layer; see :meth:`approve`.
    """

    def __init__(
        self,
        store: Store,
        root: Path | str,
        *,
        clock: Clock | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self._store = store
        self._root = Path(root)
        self._clock = clock if clock is not None else SystemClock()
        self._audit = audit
        store.initialize()  # idempotent: migration 0004 exists by the time a row is written

    # -- reading -------------------------------------------------------------------------

    def get(self, proposal: str) -> Proposal:
        """The proposal with this id, or :class:`ReviewError` if no such row exists."""
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
        # Module-constant SQL, values bound.
        # pi-lens-ignore: python-sql-injection
        row = self._store.connect().execute(_SELECT_PROPOSAL, (proposal,)).fetchone()
        if row is None:
            raise ReviewError(f"no proposal with id {proposal!r}")
        return _row_to_proposal(row)

    def pending(self) -> tuple[Proposal, ...]:
        """Every undecided proposal, highest confidence first, ordered stably for determinism."""
        rows = (
            self._store.connect()
            .execute(
                # S608: the table name is a module constant; the status value is bound.
                f"SELECT * FROM {PROPOSALS_TABLE} WHERE status = ?"  # noqa: S608
                " ORDER BY confidence DESC, id ASC",
                (ProposalStatus.PENDING.value,),
            )
            .fetchall()
        )
        return tuple(_row_to_proposal(row) for row in rows)

    def decided(self) -> tuple[Proposal, ...]:
        """Every proposal a human has decided on, oldest decision first."""
        rows = (
            self._store.connect()
            .execute(
                f"SELECT * FROM {PROPOSALS_TABLE} WHERE status != ?"  # noqa: S608
                " ORDER BY decided_at ASC, id ASC",
                (ProposalStatus.PENDING.value,),
            )
            .fetchall()
        )
        return tuple(_row_to_proposal(row) for row in rows)

    def questions(self, *, bulk_threshold: float = 0.9) -> tuple[Question, ...]:
        """What to ask a human: pending proposals, with high-confidence ones grouped into one.

        A digest that lists two hundred 0.99-confidence synonyms separately asks two hundred
        questions with one answer. Grouping them is what makes "bulk approval counts as one
        question" true in the only place it matters -- the reader's attention.
        """
        high: list[str] = []
        singles: list[Question] = []
        for proposal in self.pending():
            if proposal.confidence >= bulk_threshold:
                high.append(proposal.id)
            else:
                singles.append(
                    Question(
                        summary=f"{proposal.kind.value} {proposal.section}.{proposal.target}",
                        proposals=(proposal.id,),
                    )
                )
        if not high:
            return tuple(singles)
        grouped = Question(
            summary=f"{len(high)} high-confidence proposals (>= {bulk_threshold:.2f})",
            proposals=tuple(high),
            bulk=True,
        )
        return (grouped, *singles)

    # -- writing -------------------------------------------------------------------------

    def submit(self, proposal: Proposal) -> SubmitResult:
        """Store a proposed fact, applying the negative memory of past rejections.

        Idempotent by content: the same fact proposed twice is one row. A proposal rejected with
        the same evidence is suppressed; with different evidence it is resurfaced as pending,
        because the human said no to what they were shown.
        """
        existing = self._find(proposal.id)
        if existing is None:
            self._insert(proposal)
            return SubmitResult(action=SubmitAction.CREATED, proposal=proposal)

        if existing.status is ProposalStatus.APPROVED:
            return SubmitResult(action=SubmitAction.ALREADY_APPROVED, proposal=existing)

        if existing.status is ProposalStatus.NEVER_ASK:
            return SubmitResult(action=SubmitAction.SUPPRESSED, proposal=existing)

        if existing.status is ProposalStatus.PENDING:
            # Keep the newest evidence: the queue shows what is known now, not what was known then.
            if existing.evidence_digest == proposal.evidence_digest:
                return SubmitResult(action=SubmitAction.UNCHANGED, proposal=existing)
            self._replace(proposal.model_copy(update={"status": ProposalStatus.PENDING}))
            return SubmitResult(action=SubmitAction.UNCHANGED, proposal=proposal)

        if existing.evidence_digest == proposal.evidence_digest:
            return SubmitResult(action=SubmitAction.SUPPRESSED, proposal=existing)

        resurfaced = proposal.model_copy(
            update={"status": ProposalStatus.PENDING, "decided_by": None, "decided_at": None}
        )
        self._replace(resurfaced)
        return SubmitResult(action=SubmitAction.RESURFACED, proposal=resurfaced)

    def approve(self, proposal: str, *, by: str) -> Proposal:
        """Write an approved proposal into ``curated/*.yaml`` and record who decided.

        The patch is merged through the round-trip loader, so comments and key order elsewhere in
        the file survive. The resulting layer is then loaded and validated: an approval that would
        leave the semantic layer unloadable is rolled back and refused, because a nightly run that
        cannot read its own layer is worse than a proposal that stays pending.
        """
        current = self.get(proposal)
        if current.status is ProposalStatus.APPROVED:
            raise ReviewError(f"proposal {proposal!r} is already approved")
        at = self._now()
        path = self.curated_path(current.section)
        before = path.read_text(encoding="utf-8") if path.is_file() else None
        approved_patch = _approved_patch(current, by=by, at=at)
        after = _render(before, current, approved_patch)
        try:
            _write_text(path, after)
            self._validate_layer()
        except (SemanticError, ReviewError) as exc:
            if before is None:
                path.unlink(missing_ok=True)
            else:
                _write_text(path, before)
            raise ReviewError(f"approval refused: {exc}") from None
        decided = current.model_copy(
            update={"status": ProposalStatus.APPROVED, "decided_by": by, "decided_at": at}
        )
        self._replace(decided)
        self._record(decided, tool="review.approve")
        return decided

    def reject(self, proposal: str, *, by: str) -> Proposal:
        """Record a rejection. The curated layer is untouched; the row stays as negative memory."""
        return self._decide(proposal, by=by, status=ProposalStatus.REJECTED, tool="review.reject")

    def never_ask(self, proposal: str, *, by: str) -> Proposal:
        """Record "do not ask again, even with new evidence" for this exact fact."""
        return self._decide(
            proposal, by=by, status=ProposalStatus.NEVER_ASK, tool="review.never_ask"
        )

    def approve_high_confidence(
        self, *, by: str, threshold: float = 0.9, limit: int | None = None
    ) -> tuple[Proposal, ...]:
        """Approve every pending proposal at or above ``threshold``.

        This is the "bulk approval counts as one question" path: the reader answered once, so the
        digest asked a single question (see :meth:`questions`).
        """
        selected = [p for p in self.pending() if p.confidence >= threshold]
        if limit is not None:
            selected = selected[: max(limit, 0)]
        return tuple(self.approve(item.id, by=by) for item in selected)

    # -- the git-friendly PR flow --------------------------------------------------------

    def export_patch(self, proposal: str) -> str:
        """The approval as a unified diff, for the PR flow where a human reviews before merging.

        The same renderer as :meth:`approve`, so what the reviewer reads and what approval writes
        cannot drift apart. Paths are relative to the semantic root (``a/curated/...``), which is
        what ``git apply`` expects from the repository root.
        """
        current = self.get(proposal)
        if current.status is ProposalStatus.APPROVED:
            raise ReviewError(f"proposal {proposal!r} is already approved")
        path = self.curated_path(current.section)
        before = path.read_text(encoding="utf-8") if path.is_file() else ""
        after = _render(before or None, current, _approved_patch(current, by="", at=None))
        relative = f"{CURATED_DIR}/{path.name}"
        from_file = f"a/{relative}" if before else "/dev/null"
        return "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=from_file,
                tofile=f"b/{relative}",
                lineterm="\n",
            )
        )

    def curated_path(self, section: str) -> Path:
        """``<root>/curated/<section>.yaml``, refusing a section that could escape the directory."""
        if not section or not set(section) <= _ALLOWED_SECTION_CHARS:
            raise ReviewError(
                f"{section!r} is not a section name: use letters, digits, '_', '-' or '.'"
            )
        return self._root / CURATED_DIR / f"{section}.yaml"

    # -- internals -----------------------------------------------------------------------

    def _now(self) -> datetime:
        """One clock reading per decision, normalised to UTC so the row and audit agree."""
        now = self._clock.now()
        if now.tzinfo is None:
            raise ReviewError("the clock returned a naive datetime; decisions are stored in UTC")
        return now.astimezone(UTC)

    def _find(self, proposal: str) -> Proposal | None:
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
        # Module-constant SQL, values bound.
        # pi-lens-ignore: python-sql-injection
        row = self._store.connect().execute(_SELECT_PROPOSAL, (proposal,)).fetchone()
        return None if row is None else _row_to_proposal(row)

    def _insert(self, proposal: Proposal) -> None:
        with self._store.writing() as conn:
            conn.execute(
                f"INSERT INTO {PROPOSALS_TABLE} (id, kind, section, 'key', field, collection,"  # noqa: S608
                " patch, evidence, evidence_hash, confidence, status, proposed_by, proposed_at,"
                " decided_by, decided_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    proposal.id,
                    proposal.kind.value,
                    proposal.section,
                    proposal.key,
                    proposal.field,
                    int(proposal.collection),
                    _canonical_json(proposal.patch),
                    json.dumps(list(proposal.evidence), ensure_ascii=False),
                    proposal.evidence_digest,
                    proposal.confidence,
                    proposal.status.value,
                    proposal.proposed_by,
                    proposal.proposed_at.isoformat(),
                    proposal.decided_by,
                    proposal.decided_at.isoformat() if proposal.decided_at else None,
                ),
            )

    def _replace(self, proposal: Proposal) -> None:
        with self._store.writing() as conn:
            conn.execute(
                f"UPDATE {PROPOSALS_TABLE} SET evidence = ?, evidence_hash = ?, confidence = ?,"  # noqa: S608
                " status = ?, proposed_by = ?, proposed_at = ?, decided_by = ?, decided_at = ?"
                " WHERE id = ?",
                (
                    json.dumps(list(proposal.evidence), ensure_ascii=False),
                    proposal.evidence_digest,
                    proposal.confidence,
                    proposal.status.value,
                    proposal.proposed_by,
                    proposal.proposed_at.isoformat(),
                    proposal.decided_by,
                    proposal.decided_at.isoformat() if proposal.decided_at else None,
                    proposal.id,
                ),
            )

    def _decide(self, proposal: str, *, by: str, status: ProposalStatus, tool: str) -> Proposal:
        current = self.get(proposal)
        if current.status is ProposalStatus.APPROVED:
            raise ReviewError(f"proposal {proposal!r} is already approved and cannot be {status}")
        decided = current.model_copy(
            update={"status": status, "decided_by": by, "decided_at": self._now()}
        )
        self._replace(decided)
        self._record(decided, tool=tool)
        return decided

    def _record(self, proposal: Proposal, *, tool: str) -> None:
        """Append the decision to the hash-chained audit log, when the caller supplied one.

        The audit holds identifiers and counts only, so the proposal's **id** is recorded rather
        than the patch: an audit reader learns who decided what, and follows the id to the row,
        while no semantic content is duplicated into the log.
        """
        if self._audit is None:
            return
        self._audit.record(
            AuditEntry(
                user_id=proposal.decided_by or proposal.proposed_by,
                tool=tool,
                outcome=AuditOutcome.OK,
                version_hash=proposal.id,
                row_count=1,
            )
        )

    def _validate_layer(self) -> None:
        """Load the layer the approval just changed: an unloadable layer is a failed approval."""
        load(self._root)


def _approved_patch(proposal: Proposal, *, by: str, at: datetime | None) -> CommentedMap:
    """The patch to merge, with the human's decision attached in the shape the model expects.

    ``mutability`` is the one kind that records approval as a *status* rather than provenance
    (``MutabilityDecl``), and every other kind carries :class:`~mdqm.semantic.models.Provenance`.
    """
    patch = CommentedMap(proposal.patch)
    if proposal.kind is ProposalKind.MUTABILITY:
        patch["status"] = "approved"
        return patch
    provenance = patch.get("provenance")
    merged = CommentedMap(provenance) if isinstance(provenance, dict) else CommentedMap()
    if by:
        merged["approved_by"] = by
    if at is not None:
        merged["approved_at"] = at.isoformat()
    if merged:
        patch["provenance"] = merged
    return patch


def _load_document(text: str | None, path: Path) -> CommentedMap:
    """Parse one curated file for editing, keeping its comments; a missing file starts empty."""
    if text is None:
        return CommentedMap()
    try:
        document = YAML(typ="rt").load(text)
    except YAMLError as exc:
        raise ReviewError(f"{path}: invalid YAML: {type(exc).__name__}") from None
    if document is None:
        return CommentedMap()
    if not isinstance(document, CommentedMap):
        raise ReviewError(f"{path}: the top level of a semantic file must be a mapping")
    return document


def _descend(container: CommentedMap, part: str) -> CommentedMap:
    """The child mapping at ``part``, created when absent; a scalar there is a conflict."""
    child = container.get(part)
    if child is None:
        created = CommentedMap()
        container[part] = created
        return created
    if not isinstance(child, CommentedMap):
        raise ReviewError(f"{part!r} already holds a value that is not a mapping")
    return child


def _assign(document: CommentedMap, path: tuple[str, ...], value: CommentedMap) -> None:
    """Place ``value`` at ``path`` inside ``document``, merging rather than replacing.

    A mapping is merged key by key, so a curated file's other keys (and their comments) stay put.
    """
    container = document
    for part in path[:-1]:
        container = _descend(container, part)
    last = path[-1]
    existing = container.get(last)
    if isinstance(existing, CommentedMap):
        for key, nested in value.items():
            existing[key] = nested
        return
    container[last] = value


def _append_named(entries: CommentedSeq, value: CommentedMap) -> None:
    """Replace the list entry with a matching ``name``, or append: a join is identified by name."""
    name = value.get("name")
    for entry in entries:
        if isinstance(entry, CommentedMap) and entry.get("name") == name:
            for item_key, nested in value.items():
                entry[item_key] = nested
            return
    entries.append(value)


def _render(before: str | None, proposal: Proposal, value: CommentedMap) -> str:
    """The file text after merging ``value``: one path for approval and export, so they agree."""
    document = _load_document(before, Path(CURATED_DIR))
    if proposal.collection:
        section = document.get(proposal.section)
        if not isinstance(section, CommentedSeq):
            if section is not None:
                raise ReviewError(
                    f"{proposal.section!r} holds a mapping but this proposal expects a list"
                )
            section = CommentedSeq()
            document[proposal.section] = section
        _append_named(section, value)
    else:
        _assign(document, proposal.location, value)
    yaml = YAML(typ="rt")
    yaml.allow_unicode = True
    yaml.default_flow_style = False
    stream = io.StringIO()
    yaml.dump(document, stream)
    return stream.getvalue()


def _write_text(path: Path, text: str) -> None:
    """Replace the file atomically, so a crash never leaves a half-written semantic layer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
