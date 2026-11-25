"""The audit log: what ran, who ran it, and what was refused.

Three guarantees, in the order they matter:

* **Refusals are audited, not only successes.** The MVP acceptance item is that a blocked ``DELETE``
  leaves a record; a security log that only holds successful queries answers the wrong question.
  :meth:`AuditLog.record_refusal` takes the guard's ``ErrorResponse`` straight, so the code the
  agent saw is the code in the audit row.
* **Append-only, hash chained.** A row's ``hash`` is the SHA-256 of a canonical JSON rendering of
  the row's own fields *plus* the previous row's hash, so an edit, an insertion or the removal of a
  row in the middle breaks the chain at the first affected id. ``0002_audit.sql`` additionally holds
  ``BEFORE UPDATE``/``BEFORE DELETE`` triggers, because a chain proves tampering happened while a
  trigger stops the ordinary case of another lane rewriting history by accident.
* **No payload, ever.** The model has no field a result row, a SQL statement or a credential could
  occupy: identifiers, closed-set codes and counts only (AGENTS.md principle 11). Every text field
  is length-capped, refused if it holds whitespace, a quote or ``=``, and refused outright if it
  looks like a credential, so a token pasted into a "tool name" cannot reach the file.

The honest limit: a hash chain detects tampering, it does not prevent it. Someone who can write the
file can drop a trigger and rewrite every row from the edit onward; that is what the chain is for —
it makes the rewrite detectable. Two cheap holes are closed anyway: the id is inside the hash, so a
deletion leaves a gap, and the head hash is mirrored into ``mdqm_meta`` so truncating the tail is
caught too.

Retention: :meth:`AuditLog.purge` removes whole prefixes only and writes the last purged
``(id, hash)`` into ``audit_checkpoint``, so verification resumes from the checkpoint instead of
failing on a chain whose beginning was legally deleted (DESIGN section 13: retain one year).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Final

from pydantic import Field, model_validator

from mdqm.errors import ErrorCode
from mdqm.models import MDQMModel, UtcDatetime
from mdqm.models.errors import ErrorResponse
from mdqm.store.db import Store, StoreError

__all__ = [
    "AUDIT_CHECKPOINT_TABLE",
    "AUDIT_LOG_TABLE",
    "DEFAULT_RETENTION_DAYS",
    "GENESIS_HASH",
    "HEAD_KEY",
    "AuditEntry",
    "AuditError",
    "AuditLog",
    "AuditOutcome",
    "AuditRecord",
    "ChainVerification",
    "Clock",
    "PurgeReport",
]

logger = logging.getLogger(__name__)

#: The audit table. Named as a constant because the SQL below interpolates it (no caller input).
AUDIT_LOG_TABLE: Final[str] = "audit_log"
#: Holds the ``(id, hash)`` of the last row a purge removed, so the chain resumes there.
AUDIT_CHECKPOINT_TABLE: Final[str] = "audit_checkpoint"
#: ``prev_hash`` of the very first row: 64 zeros, so "no previous row" is a value and not NULL.
GENESIS_HASH: Final[str] = "0" * 64
#: ``mdqm_meta`` key holding the head hash, which makes a truncated tail detectable.
HEAD_KEY: Final[str] = "audit.head_hash"
#: DESIGN section 13: the audit is retained for a year.
DEFAULT_RETENTION_DAYS: Final[int] = 365

#: A clock, injected so chain ordering and timestamps are deterministic in tests.
Clock = Callable[[], datetime]

_MAX_TOKEN: Final[int] = 200

#: A row field is an identifier or a code: one token. Whitespace, quotes and ``=`` have no business
#: in one, and refusing them is what stops a pasted ``password=...`` from being stored.
_NOT_A_TOKEN: Final[re.Pattern[str]] = re.compile(r"[\s\"'`=;\\]")
#: Credential shapes refused even when they happen to be a single token.
_SECRET_SHAPES: Final[tuple[re.Pattern[str], ...]] = (
    # A JSON Web Token (the shape oauth2 and on-prem gateways issue).
    re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}"),
    re.compile(r"(?i)\b(?:bearer|basic)\s+\S"),
    re.compile(r"-----BEGIN"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(?:password|passwd|pwd|secret|token|api[_-]?key|authorization)\b\s*[:=]"),
    re.compile(r"://[^/\s:@]+:[^/\s@]+@"),
)


class AuditError(StoreError, ValueError):
    """A value that must never reach an audit row was refused (or the log is unusable).

    Also a ``ValueError`` so a pydantic validator that calls :func:`_clean` reports it as a normal
    field rejection instead of leaking an unexpected exception type out of ``AuditEntry(...)``.
    """


class AuditOutcome(StrEnum):
    """How a call ended. Three outcomes, so a reader can count refusals without parsing text."""

    OK = "ok"
    BLOCKED = "blocked"
    ERROR = "error"


def _utc_now() -> datetime:
    """The default clock: timezone-aware UTC, never a naive local time."""
    return datetime.now(UTC)


def _clean(field: str, value: str, *, limit: int = _MAX_TOKEN) -> str:
    """Refuse a value that is not a plain token, naming the field but never echoing the value.

    The refusal deliberately does not quote the offending text: quoting it would write the very
    credential being rejected into the exception, the log and (via a traceback) possibly further.
    """
    text = value if isinstance(value, str) else str(value)
    if not text or not text.strip():
        raise AuditError(f"audit field {field!r} is empty")
    if len(text) > limit:
        raise AuditError(f"audit field {field!r} is longer than {limit} characters")
    if _NOT_A_TOKEN.search(text):
        raise AuditError(
            f"audit field {field!r} holds whitespace, a quote, '=' or a backslash; an audit field "
            "is a single identifier or code, never free text"
        )
    for pattern in _SECRET_SHAPES:
        if pattern.search(text):
            raise AuditError(
                f"audit field {field!r} looks like a credential (its value is not repeated here); "
                "record an identifier, a code or a count instead"
            )
    return text


class AuditEntry(MDQMModel):
    """One appendable audit row, before the chain fields are computed.

    There is no field for a SQL statement, a result row or a request body, by design: this model
    cannot represent a payload, so no caller can leak one into the audit file.
    """

    user_id: str = Field(min_length=1, max_length=_MAX_TOKEN)
    tool: str = Field(min_length=1, max_length=_MAX_TOKEN)
    outcome: AuditOutcome
    version_hash: str | None = Field(default=None, max_length=_MAX_TOKEN)
    sources: tuple[str, ...] = ()
    row_count: int = Field(default=0, ge=0)
    blocked_attempts: int = Field(default=0, ge=0)
    error_code: ErrorCode | None = None

    @model_validator(mode="after")
    def _check(self) -> AuditEntry:
        """Every text field is a token, and a blocked row says how many attempts and why."""
        _clean("user_id", self.user_id)
        _clean("tool", self.tool)
        if self.version_hash is not None:
            _clean("version_hash", self.version_hash)
        for source in self.sources:
            _clean("sources", source)
        if self.outcome is AuditOutcome.BLOCKED:
            if self.blocked_attempts < 1:
                raise ValueError("a blocked audit row needs blocked_attempts >= 1")
            if self.error_code is None:
                raise ValueError("a blocked audit row needs the error_code the agent was given")
        return self


class AuditRecord(MDQMModel):
    """A stored audit row, chain fields included: what :meth:`AuditLog.rows` returns."""

    id: int
    ts: UtcDatetime
    user_id: str
    tool: str
    outcome: AuditOutcome
    version_hash: str | None
    sources: tuple[str, ...]
    row_count: int
    blocked_attempts: int
    error_code: ErrorCode | None
    prev_hash: str
    hash: str

    def payload(self) -> dict[str, object]:
        """The fields the hash covers, in canonical order: the id is inside, so a gap is a break."""
        return {
            "v": 1,
            "id": self.id,
            "ts": self.ts.isoformat(),
            "user_id": self.user_id,
            "tool": self.tool,
            "outcome": self.outcome.value,
            "version_hash": self.version_hash,
            "sources": list(self.sources),
            "row_count": self.row_count,
            "blocked_attempts": self.blocked_attempts,
            "error_code": self.error_code.value if self.error_code is not None else None,
            "prev_hash": self.prev_hash,
        }


def _digest(payload: dict[str, object]) -> str:
    """SHA-256 of the canonical JSON of ``payload``: sorted keys, no spaces, UTF-8."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ChainVerification(MDQMModel):
    """The verdict of :meth:`AuditLog.verify_chain`, with the first bad link named."""

    ok: bool
    checked: int
    first_bad_id: int | None = None
    problem: str | None = None
    last_hash: str = GENESIS_HASH
    checkpoint_id: int = 0

    def describe(self) -> str:
        """One line for a human or a CLI: the first bad link, or how much was verified."""
        if self.ok:
            return (
                f"audit chain intact: {self.checked} row(s) verified"
                f"{f' after checkpoint {self.checkpoint_id}' if self.checkpoint_id else ''}"
            )
        return f"audit chain broken at id {self.first_bad_id}: {self.problem}"


class PurgeReport(MDQMModel):
    """What :meth:`AuditLog.purge` removed, and where the chain now resumes."""

    purged_rows: int
    kept_rows: int
    checkpoint_id: int
    checkpoint_hash: str

    @property
    def purged(self) -> bool:
        """True when this run removed anything (an idempotent re-run removes nothing)."""
        return self.purged_rows > 0


class AuditLog:
    """Append-only, hash-chained audit over one :class:`~mdqm.store.db.Store`.

    Not a singleton: pass the store (and, in tests, the clock). Every write goes through
    :meth:`Store.writing`, so two threads cannot both read the current head and compute the same
    ``prev_hash``. Reads use the caller's connection and see committed rows only.
    """

    def __init__(
        self,
        store: Store,
        *,
        clock: Clock | None = None,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> None:
        if retention_days < 0:
            raise AuditError("retention_days must not be negative")
        self._store = store
        self._clock = clock if clock is not None else _utc_now
        self.retention_days = retention_days

    # -- writing -------------------------------------------------------------------------

    def record(self, entry: AuditEntry) -> AuditRecord:
        """Append ``entry`` and return the row, chain fields filled in."""
        ts = self._now()
        with self._store.writing() as conn:
            return self._append(conn, entry, ts)

    def _now(self) -> datetime:
        """One clock reading per write, normalised to UTC.

        Read once and threaded through, so a row's ``ts`` and the head mirror cannot disagree
        because the clock ticked between them (and a test clock stays deterministic).
        """
        ts = self._clock()
        if ts.tzinfo is None:
            raise AuditError("the clock returned a naive datetime; the audit stores UTC only")
        return ts.astimezone(UTC)

    def record_refusal(
        self,
        *,
        user_id: str,
        tool: str,
        error: ErrorResponse,
        version_hash: str | None = None,
        sources: Sequence[str] = (),
    ) -> AuditRecord:
        """Audit a refusal the guard (or the linter) produced: the agent's code, one attempt.

        This is the path the MVP acceptance item exercises: a blocked ``DELETE`` reaches the guard,
        the guard returns ``WRITE_BLOCKED``, and that refusal is what lands in the log.
        """
        return self.record(
            AuditEntry(
                user_id=user_id,
                tool=tool,
                outcome=AuditOutcome.BLOCKED,
                version_hash=version_hash,
                sources=tuple(sources),
                row_count=0,
                blocked_attempts=1,
                error_code=ErrorCode(error.code),
            )
        )

    def _append(
        self, conn: sqlite3.Connection, entry: AuditEntry, ts: datetime
    ) -> AuditRecord:
        """Compute the chain fields inside the writer's transaction and insert one row."""
        prev_hash, next_id = self._head(conn)
        record = AuditRecord(
            id=next_id,
            ts=ts,
            user_id=entry.user_id,
            tool=entry.tool,
            outcome=entry.outcome,
            version_hash=entry.version_hash,
            sources=entry.sources,
            row_count=entry.row_count,
            blocked_attempts=entry.blocked_attempts,
            error_code=entry.error_code,
            prev_hash=prev_hash,
            hash=GENESIS_HASH,  # replaced below, once the payload exists
        )
        digest = _digest(record.payload())
        conn.execute(
            # S608: the table name is a module constant, not caller input; values are bound.
            f"INSERT INTO {AUDIT_LOG_TABLE} (id, ts, user_id, tool, version_hash, sources,"  # noqa: S608
            " row_count, blocked_attempts, outcome, error_code, prev_hash, hash)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.ts.isoformat(),
                record.user_id,
                record.tool,
                record.version_hash,
                json.dumps(list(record.sources), separators=(",", ":"), ensure_ascii=False),
                record.row_count,
                record.blocked_attempts,
                record.outcome.value,
                record.error_code.value if record.error_code is not None else None,
                record.prev_hash,
                digest,
            ),
        )
        self._write_head(conn, digest, ts)
        return record.model_copy(update={"hash": digest})

    def _head(self, conn: sqlite3.Connection) -> tuple[str, int]:
        """The hash of the newest row and the id the next row must take.

        Falls back to the checkpoint when a purge emptied the table, so the chain continues across
        a legal prefix deletion instead of restarting at the genesis hash.
        """
        row = conn.execute(
            # S608: constant table name.
            f"SELECT id, hash FROM {AUDIT_LOG_TABLE} ORDER BY id DESC LIMIT 1"  # noqa: S608
        ).fetchone()
        if row is not None:
            return str(row["hash"]), int(row["id"]) + 1
        checkpoint = self._checkpoint(conn)
        if checkpoint is None:
            return GENESIS_HASH, 1
        return checkpoint[1], checkpoint[0] + 1

    def _write_head(self, conn: sqlite3.Connection, head_hash: str, ts: datetime) -> None:
        """Mirror the head hash into ``mdqm_meta`` so a truncated tail cannot pass verification."""
        conn.execute(
            "INSERT INTO mdqm_meta (key, value, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
            " updated_at = excluded.updated_at",
            (HEAD_KEY, head_hash, ts.isoformat()),
        )

    # -- reading -------------------------------------------------------------------------

    def rows(
        self, *, limit: int | None = None, since_id: int | None = None
    ) -> tuple[AuditRecord, ...]:
        """Committed rows in id order, optionally after ``since_id`` and capped at ``limit``."""
        sql = (
            f"SELECT * FROM {AUDIT_LOG_TABLE}"  # noqa: S608 - constant table name
            + (" WHERE id > ?" if since_id is not None else "")
            + " ORDER BY id"
            + (" LIMIT ?" if limit is not None else "")
        )
        params: list[object] = []
        if since_id is not None:
            params.append(since_id)
        if limit is not None:
            params.append(limit)
        return tuple(_to_record(row) for row in self._store.connect().execute(sql, params))

    def count(self) -> int:
        """How many audit rows the log holds."""
        row = self._store.connect().execute(
            f"SELECT COUNT(*) FROM {AUDIT_LOG_TABLE}"  # noqa: S608 - constant table name
        ).fetchone()
        return int(row[0])

    def _checkpoint(self, conn: sqlite3.Connection) -> tuple[int, str] | None:
        """``(last_purged_id, last_purged_hash)``, or ``None`` when nothing was ever purged."""
        row = conn.execute(
            # S608: constant table name.
            f"SELECT last_purged_id, last_purged_hash FROM {AUDIT_CHECKPOINT_TABLE} WHERE id = 1"  # noqa: S608
        ).fetchone()
        if row is None:
            return None
        return int(row["last_purged_id"]), str(row["last_purged_hash"])

    # -- verification --------------------------------------------------------------------

    def verify_chain(self) -> ChainVerification:
        """Recompute the whole chain and report the first link that does not hold.

        Checks, in order, per row: the ids are contiguous (a deletion leaves a gap), the row's own
        hash matches its fields (an edit), and ``prev_hash`` is the previous row's hash (an
        insertion or a reordering). A prefix removed by :meth:`purge` is not a break: verification
        starts at the checkpoint instead of at the genesis hash.
        """
        conn = self._store.connect()
        checkpoint = self._checkpoint(conn)
        if checkpoint is None:
            expected_prev, expected_id = GENESIS_HASH, 1
        else:
            expected_prev, expected_id = checkpoint[1], checkpoint[0] + 1
        rows = conn.execute(
            f"SELECT * FROM {AUDIT_LOG_TABLE} ORDER BY id"  # noqa: S608 - constant table name
        ).fetchall()
        checked = 0
        for row in rows:
            record = _to_record(row)
            if record.id != expected_id:
                return self._broken(
                    record.id,
                    checked,
                    f"expected id {expected_id} and found {record.id}: a row was deleted or "
                    "inserted",
                    expected_prev,
                    checkpoint,
                )
            if _digest(record.payload()) != record.hash:
                return self._broken(
                    record.id,
                    checked,
                    "the row's fields no longer hash to its stored hash: it was edited",
                    expected_prev,
                    checkpoint,
                )
            if record.prev_hash != expected_prev:
                return self._broken(
                    record.id,
                    checked,
                    "prev_hash is not the previous row's hash: a row was inserted or replaced",
                    expected_prev,
                    checkpoint,
                )
            checked += 1
            expected_prev, expected_id = record.hash, record.id + 1
        head = self._stored_head(conn)
        if head is not None and head != expected_prev:
            return self._broken(
                None,
                checked,
                "the newest row's hash is not the recorded head: the tail of the log was removed",
                expected_prev,
                checkpoint,
            )
        return ChainVerification(
            ok=True,
            checked=checked,
            last_hash=expected_prev,
            checkpoint_id=checkpoint[0] if checkpoint is not None else 0,
        )

    @staticmethod
    def _broken(
        first_bad_id: int | None,
        checked: int,
        problem: str,
        last_hash: str,
        checkpoint: tuple[int, str] | None,
    ) -> ChainVerification:
        return ChainVerification(
            ok=False,
            checked=checked,
            first_bad_id=first_bad_id,
            problem=problem,
            last_hash=last_hash,
            checkpoint_id=checkpoint[0] if checkpoint is not None else 0,
        )

    def _stored_head(self, conn: sqlite3.Connection) -> str | None:
        """The head hash mirrored in ``mdqm_meta``, or ``None`` for a log predating the mirror."""
        row = conn.execute("SELECT value FROM mdqm_meta WHERE key = ?", (HEAD_KEY,)).fetchone()
        return None if row is None else str(row[0])

    # -- retention -----------------------------------------------------------------------

    def purge(self, *, now: datetime | None = None) -> PurgeReport:
        """Remove the oldest rows that are past retention, as a whole prefix, and checkpoint.

        Only a prefix: a row is removed when every older row is removed too, so the survivors are
        still one contiguous chain and the checkpoint says exactly where it starts. Rows inside the
        retention window are never touched, and a second run with nothing to remove is a no-op.

        The ``BEFORE DELETE`` trigger is dropped and restored inside the same transaction, so the
        append-only guarantee holds again the moment this returns, and a failure rolls the drop back
        with it.
        """
        cutoff = (now if now is not None else self._clock()).astimezone(UTC) - timedelta(
            days=self.retention_days
        )
        with self._store.writing() as conn:
            stale = conn.execute(
                # S608: constant table name.
                f"SELECT id, ts, hash FROM {AUDIT_LOG_TABLE} ORDER BY id"  # noqa: S608
            ).fetchall()
            purge_upto: int | None = None
            checkpoint_hash = GENESIS_HASH
            for row in stale:
                if _parse_ts(str(row["ts"])) >= cutoff:
                    break  # the first row inside the window ends the prefix
                purge_upto = int(row["id"])
                checkpoint_hash = str(row["hash"])
            if purge_upto is None:
                return self._no_op(conn)
            trigger = self._drop_delete_trigger(conn)
            try:
                conn.execute(
                    # S608: constant table name; the bound value is an integer id.
                    f"DELETE FROM {AUDIT_LOG_TABLE} WHERE id <= ?",  # noqa: S608
                    (purge_upto,),
                )
            finally:
                if trigger is not None:
                    conn.execute(trigger)
            conn.execute(
                # S608: constant table name.
                f"INSERT INTO {AUDIT_CHECKPOINT_TABLE}"  # noqa: S608
                " (id, last_purged_id, last_purged_hash, purged_at, purged_rows)"
                " VALUES (1, ?, ?, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET last_purged_id = excluded.last_purged_id,"
                " last_purged_hash = excluded.last_purged_hash,"
                " purged_at = excluded.purged_at, purged_rows = excluded.purged_rows",
                (
                    purge_upto,
                    checkpoint_hash,
                    self._clock().astimezone(UTC).isoformat(),
                    purge_upto,
                ),
            )
            kept = self._count(conn)
        logger.info(
            "audit purge removed %d row(s) past %d-day retention",
            purge_upto,
            self.retention_days,
            extra={"purged_rows": purge_upto, "kept_rows": kept},
        )
        return PurgeReport(
            purged_rows=purge_upto,
            kept_rows=kept,
            checkpoint_id=purge_upto,
            checkpoint_hash=checkpoint_hash,
        )

    def _no_op(self, conn: sqlite3.Connection) -> PurgeReport:
        """Nothing is past retention: report the current checkpoint and remove nothing."""
        checkpoint = self._checkpoint(conn)
        return PurgeReport(
            purged_rows=0,
            kept_rows=self._count(conn),
            checkpoint_id=checkpoint[0] if checkpoint is not None else 0,
            checkpoint_hash=checkpoint[1] if checkpoint is not None else GENESIS_HASH,
        )

    @staticmethod
    def _count(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            f"SELECT COUNT(*) FROM {AUDIT_LOG_TABLE}"  # noqa: S608 - constant table name
        ).fetchone()
        return int(row[0])

    @staticmethod
    def _drop_delete_trigger(conn: sqlite3.Connection) -> str | None:
        """Drop the append-only DELETE trigger, returning the SQL that restores it.

        Read back from ``sqlite_master`` rather than duplicated here, so the trigger text has one
        home (the migration) and a purge can never restore a different rule.
        """
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = 'audit_log_no_delete'"
        ).fetchone()
        ddl = None if row is None else row["sql"]
        if ddl is None:
            # No trigger to restore: the guarantee is already gone, and saying so beats a silent
            # delete that leaves the table unprotected.
            raise AuditError(
                "the append-only DELETE trigger is missing from audit_log, so a purge would leave "
                "the table unprotected; re-run Store.initialize() or restore 0002_audit.sql"
            )
        conn.execute("DROP TRIGGER audit_log_no_delete")
        return str(ddl)


def _parse_ts(text: str) -> datetime:
    """Parse a stored timestamp; an unparseable one is a broken row, not a default."""
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise AuditError(f"an audit row holds an unparseable timestamp: {exc}") from exc
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _to_record(row: sqlite3.Row) -> AuditRecord:
    """Map one SQLite row onto the model, decoding the JSON ``sources`` list."""
    raw_sources = row["sources"]
    sources = tuple(json.loads(str(raw_sources))) if raw_sources else ()
    error_code = row["error_code"]
    return AuditRecord(
        id=int(row["id"]),
        ts=_parse_ts(str(row["ts"])),
        user_id=str(row["user_id"]),
        tool=str(row["tool"]),
        outcome=AuditOutcome(str(row["outcome"])),
        version_hash=None if row["version_hash"] is None else str(row["version_hash"]),
        sources=sources,
        row_count=int(row["row_count"]),
        blocked_attempts=int(row["blocked_attempts"]),
        error_code=None if error_code is None else ErrorCode(str(error_code)),
        prev_hash=str(row["prev_hash"]),
        hash=str(row["hash"]),
    )
