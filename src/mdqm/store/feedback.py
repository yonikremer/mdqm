"""The feedback queue: what the caller said about one answer, stored and never auto-trusted (M05).

The ``feedback`` tool needs somewhere to put a rating. That place is a table
(``0007_feedback.sql``), not a file, and it is a **queue**: rows are read later by the review
pipeline and the mining stages, and a model may propose a change from them, but nothing in this
module edits the semantic layer. A rating is a claim by a caller who may be wrong (AGENTS.md
principle 3: models propose, code validates, humans approve).

Two properties matter:

* **A row is a record of the caller's words, bounded and scrubbed.** ``corrected`` is free text from
  an agent, so it goes through the same credential scrubber free text already uses
  (:func:`mdqm.store.history._redact`: one home for "what looks like a credential") and is then
  bounded. A rating must never be the way a token reaches the metadata file (principle 11).
* **Only the caller's own runs can be rated.** ``run_id`` is resolved by the tool against the
  caller's history and the owner check happens there; this module stores what it is given,
  ``run_id`` or not, so a rating that names no run of theirs cannot be written by accident.

No result row is stored here either: the row holds ids, a boolean and one sentence.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final

from pydantic import Field

from mdqm.models import MDQMModel, UtcDatetime
from mdqm.store.audit import DEFAULT_RETENTION_DAYS, _clean
from mdqm.store.db import Store, StoreError
from mdqm.store.history import _redact

__all__ = [
    "FEEDBACK_TABLE",
    "MAX_CORRECTED",
    "Clock",
    "FeedbackEntry",
    "FeedbackError",
    "FeedbackRecord",
    "FeedbackStore",
]

logger = logging.getLogger(__name__)

#: The table (see ``0007_feedback.sql``). Public because tests and tooling name it; every
#: statement below spells the name out, so the SQL at each call site is a literal.
FEEDBACK_TABLE: Final[str] = "query_feedback"

#: Longest ``corrected`` sentence kept. A rating is one line, not a report; the rest is dropped.
MAX_CORRECTED: Final[int] = 4000

#: A clock, injected so "when was this rated" is a two-line test with no sleeping.
Clock = Callable[[], datetime]


class FeedbackError(StoreError, ValueError):
    """A rating that must not be stored as it arrived."""


class FeedbackEntry(MDQMModel):
    """One rating, before it is written: who, about which run, and what they said."""

    user_id: str = Field(min_length=1, description="The caller who rated the answer.")
    query_id: str = Field(min_length=1, description="The id the caller quoted back.")
    run_id: int | None = Field(
        default=None, ge=1, description="The caller's own history row, when the id resolved to one."
    )
    ok: bool = Field(description="True when the answer was what the caller needed.")
    corrected: str | None = Field(
        default=None, description="When ok is false: what the answer should have been."
    )


class FeedbackRecord(FeedbackEntry):
    """A stored rating, as the queue and the tool return it."""

    id: int
    created_at: UtcDatetime


def _utc_now() -> datetime:
    """The default clock: timezone-aware UTC, never a naive local time."""
    return datetime.now(UTC)


class FeedbackStore:
    """Ratings over one :class:`~mdqm.store.db.Store`: pass the store and, in tests, the clock."""

    def __init__(
        self,
        store: Store,
        *,
        clock: Clock | None = None,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> None:
        if retention_days < 0:
            raise FeedbackError("retention_days must not be negative")
        self._store = store
        self._clock = clock if clock is not None else _utc_now
        self.retention_days = retention_days

    def record(self, entry: FeedbackEntry) -> FeedbackRecord:
        """Store one rating and return the row.

        Raises:
            FeedbackError: ``query_id`` is not a token, feeling like a credential, or empty (see
                :func:`mdqm.store.audit._clean`), or the clock returned a naive time.
        """
        # A quoted id travels into a log line and a review digest, so it must be a token.
        query_id = _clean("query_id", entry.query_id)
        user_id = _clean("user_id", entry.user_id)
        corrected = _scrubbed_corrected(entry.corrected)
        created_at = self._now()
        with self._store.writing() as conn:
            cursor = conn.execute(
                "INSERT INTO query_feedback"
                " (created_at, user_id, query_id, run_id, ok, corrected) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    created_at.isoformat(),
                    user_id,
                    query_id,
                    entry.run_id,
                    int(entry.ok),
                    corrected,
                ),
            )
            feedback_id = int(cursor.lastrowid or 0)
        logger.info(
            "feedback recorded",
            extra={"feedback_id": feedback_id, "run_id": entry.run_id, "ok": entry.ok},
        )
        return FeedbackRecord(
            id=feedback_id,
            created_at=created_at,
            user_id=user_id,
            query_id=query_id,
            run_id=entry.run_id,
            ok=entry.ok,
            corrected=corrected,
        )

    def recent(self, *, user_id: str | None = None, limit: int = 20) -> tuple[FeedbackRecord, ...]:
        """The newest ratings first, optionally only one caller's: the queue's read path."""
        # One statement, not a statement and an appended clause: a NULL ``user_id`` makes the
        # predicate true for every row, so "every user" needs no second SQL text.
        rows = (
            self._store.connect()
            .execute(
                "SELECT * FROM query_feedback WHERE COALESCE(?, user_id) = user_id"
                " ORDER BY id DESC LIMIT ?",
                (user_id, _bounded(limit)),
            )
            .fetchall()
        )
        return tuple(_to_record(row) for row in rows)

    def count(self) -> int:
        """How many ratings the queue holds."""
        row = self._store.connect().execute("SELECT COUNT(*) FROM query_feedback").fetchone()
        return int(row[0])

    def _now(self) -> datetime:
        """One clock reading per write, normalised to UTC (the store holds UTC only)."""
        moment = self._clock()
        if moment.tzinfo is None:
            raise FeedbackError("the clock returned a naive datetime; feedback stores UTC only")
        return moment.astimezone(UTC)


def _scrubbed_corrected(text: str | None) -> str | None:
    """Redact credential shapes out of the caller's sentence, then bound it.

    Over-redaction is accepted: a wrongly redacted rating is a worse review proposal, a missed one
    puts a credential in the metadata file.
    """
    if text is None:
        return None
    cleaned, _ = _redact(text)
    cleaned = " ".join(cleaned.split())
    return cleaned[:MAX_CORRECTED] or None


def _bounded(limit: int) -> int:
    """Clamp a page size into a range SQLite and the caller can both live with."""
    return max(1, min(limit, 200))


def _to_record(row: sqlite3.Row) -> FeedbackRecord:
    """Map one SQLite row onto the model; a row that no longer parses is an error, not a default."""
    try:
        return FeedbackRecord(
            id=int(row["id"]),
            created_at=_parse_ts(str(row["created_at"])),
            user_id=str(row["user_id"]),
            query_id=str(row["query_id"]),
            run_id=None if row["run_id"] is None else int(row["run_id"]),
            ok=bool(row["ok"]),
            corrected=None if row["corrected"] is None else str(row["corrected"]),
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise FeedbackError(f"stored feedback row is unreadable: {exc}") from exc


def _parse_ts(text: str) -> datetime:
    """Parse a stored UTC timestamp, refusing a naive one."""
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise FeedbackError("stored timestamp is naive; feedback stores UTC only")
    return moment.astimezone(UTC)
