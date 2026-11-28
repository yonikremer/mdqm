"""The asynchronous query jobs: one row per run the caller did not wait for (task ASYNC-QUERY).

Why a table and not a dict: a job outlives the call that started it. The in-process registry
(``mdqm.server.jobs``) holds the worker thread and the measured progress, but the *state* has to be
readable by the next call -- ``get_more(query_id=...)`` arrives as a separate MCP message, possibly
on another worker thread -- and has to survive long enough to be honest after a restart. That is what
``0011_query_job.sql`` is for.

Three properties this module keeps:

* **The row never holds rows.** A finished job names its result with ``result_handle``, exactly like
  a blocking answer. The Arrow table stays in the governor's :class:`~mdqm.governor.handles.ResultStore`
  with its own TTL and ownership; this module holds counters.
* **Ownership is checked here, and a miss looks like a miss.** ``get`` takes the caller's id and
  returns ``None`` for another user's job, the same answer as an unknown one, so a job id cannot be
  used to probe for someone else's work (the rule ``ResultStore`` keeps for handles).
* **Nothing is invented.** ``partitions_done`` is ``None`` while the job runs, because one fragment
  is pruned by the source rather than iterated by us; ``rows_read`` is what the governor measured.
  ``None`` means "not reported", never zero (``docs/DESIGN.md`` section 7).

Retention is deliberately not implemented: a row is a few dozen bytes and there is no scheduler for
it yet, so the honest statement is that they accumulate. ``progress/ASYNC-QUERY.md`` says so.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

from pydantic import Field

from mdqm.models import ErrorResponse, MDQMModel, UtcDatetime
from mdqm.store.audit import _clean
from mdqm.store.db import Store, StoreError

__all__ = [
    "JOB_TABLE",
    "Clock",
    "JobState",
    "QueryJob",
    "QueryJobStore",
]

logger = logging.getLogger(__name__)

#: The table (see ``0011_query_job.sql``). Spelled out in every statement below, never interpolated.
JOB_TABLE: Final[str] = "query_job"

#: A clock, injected so "how long has it been running" is a test with no sleeping.
Clock = Callable[[], datetime]


class JobState(StrEnum):
    """The state machine the ``query_job.state`` CHECK allows. Nothing else can be stored."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: States a poll may still turn into a result.
LIVE_STATES: Final[frozenset[JobState]] = frozenset({JobState.QUEUED, JobState.RUNNING})


class QueryJob(MDQMModel):
    """One asynchronous run, as stored. What ``get_more(query_id=...)`` reads and reports."""

    id: int = Field(ge=1)
    query_id: str = Field(min_length=1, description="The opaque id the agent quotes back.")
    user_id: str = Field(min_length=1)
    state: JobState
    source: str = ""
    phase: str = Field(
        default="", description="What the worker is doing: planning, executing, finished."
    )
    est_rows: float | None = Field(default=None, ge=0)
    partitions_total: int | None = Field(default=None, ge=0)
    partitions_done: int | None = Field(
        default=None, ge=0, description="Unknown (null) until the run reports what it scanned."
    )
    rows_read: int = Field(default=0, ge=0, description="Rows the source has produced so far.")
    bytes_read: int = Field(default=0, ge=0)
    result_handle: str | None = Field(
        default=None,
        description="Page it with `get_more(result_handle=...)`, like a normal answer.",
    )
    rows_total: int | None = Field(default=None, ge=0)
    error_code: str | None = None
    error: ErrorResponse | None = Field(
        default=None, description="The same structured error a blocking run would have returned."
    )
    note: str = Field(default="", description="One line on anything unusual about this run.")
    created_at: UtcDatetime
    started_at: UtcDatetime | None = None
    finished_at: UtcDatetime | None = None

    @property
    def live(self) -> bool:
        """True while the run may still produce a result."""
        return self.state in LIVE_STATES


def _utc_now() -> datetime:
    """The default clock: timezone-aware UTC, never a naive local time."""
    return datetime.now(UTC)


class QueryJobStore:
    """Job rows over one :class:`~mdqm.store.db.Store`.

    Args:
        store: the migrated metadata database.
        clock: injected time source, so a test never sleeps to age a job.
    """

    def __init__(self, store: Store, *, clock: Clock | None = None) -> None:
        self._store = store
        self._clock = clock if clock is not None else _utc_now

    def create(
        self,
        *,
        user_id: str,
        source: str,
        est_rows: float | None = None,
        partitions_total: int | None = None,
    ) -> QueryJob:
        """Write the ``queued`` row and return it, id and ``query_id`` included.

        The id is assigned by SQLite, so the opaque ``query_id`` is written in the same transaction:
        a row that is visible with no id the agent could quote would be a job nobody can poll.
        """
        owner = _clean("user_id", user_id)
        created_at = self._now()
        with self._store.writing() as conn:
            cursor = conn.execute(
                "INSERT INTO query_job"
                " (user_id, query_id, state, source, phase, est_rows, partitions_total,"
                " created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    owner,
                    "",  # unique per row, replaced below: the id is what names the job
                    JobState.QUEUED.value,
                    source,
                    "queued",
                    est_rows,
                    partitions_total,
                    created_at.isoformat(),
                ),
            )
            row_id = int(cursor.lastrowid or 0)
            query_id = f"j_{row_id}"
            conn.execute("UPDATE query_job SET query_id = ? WHERE id = ?", (query_id, row_id))
        job = self.get(query_id, owner=owner)
        if job is None:  # pragma: no cover - a write we just committed and cannot read is a bug
            raise StoreError(f"query_job row {row_id} vanished inside its own transaction")
        return job

    def mark_running(
        self,
        query_id: str,
        *,
        phase: str,
        rows_read: int | None = None,
        bytes_read: int | None = None,
        partitions_total: int | None = None,
        est_rows: float | None = None,
    ) -> None:
        """Move a job to ``running`` and record what the worker just learned.

        ``None`` for a number means "leave what is stored": the static plan arrives later on some
        paths (a federated run has no single fragment), and over-writing a known total with NULL
        would throw away the only progress denominator the caller has.
        """
        started_at = self._now()
        with self._store.writing() as conn:
            conn.execute(
                "UPDATE query_job SET"
                " state = ?,"
                " phase = ?,"
                " started_at = COALESCE(started_at, ?),"
                " rows_read = COALESCE(?, rows_read),"
                " bytes_read = COALESCE(?, bytes_read),"
                " partitions_total = COALESCE(?, partitions_total),"
                " est_rows = COALESCE(?, est_rows)"
                " WHERE query_id = ?",
                (
                    JobState.RUNNING.value,
                    phase,
                    started_at.isoformat(),
                    rows_read,
                    bytes_read,
                    partitions_total,
                    est_rows,
                    query_id,
                ),
            )

    def mark_progress(self, query_id: str, *, rows_read: int, bytes_read: int) -> None:
        """Record measured progress. The registry throttles it; every call is one small UPDATE."""
        with self._store.writing() as conn:
            conn.execute(
                "UPDATE query_job SET rows_read = ?, bytes_read = ?"
                " WHERE query_id = ? AND state = ?",
                (max(0, rows_read), max(0, bytes_read), query_id, JobState.RUNNING.value),
            )

    def mark_progress_shape(
        self,
        query_id: str,
        *,
        partitions_done: int | None,
        partitions_total: int | None,
    ) -> None:
        """Record what the source's own plan says the run will scan.

        Called once per run, after the plan is read: the partition counts come from here, and
        ``None`` for either one means the source reported nothing, which is not zero.
        """
        with self._store.writing() as conn:
            conn.execute(
                "UPDATE query_job SET partitions_done = COALESCE(?, partitions_done),"
                " partitions_total = COALESCE(?, partitions_total)"
                " WHERE query_id = ? AND state = ?",
                (
                    partitions_done,
                    partitions_total,
                    query_id,
                    JobState.RUNNING.value,
                ),
            )

    def mark_finished(
        self,
        query_id: str,
        *,
        state: JobState,
        phase: str = "finished",
        rows_read: int | None = None,
        bytes_read: int | None = None,
        partitions_done: int | None = None,
        result_handle: str | None = None,
        rows_total: int | None = None,
        error: ErrorResponse | None = None,
        note: str = "",
    ) -> None:
        """Record the terminal state, with the handle or the structured error and nothing else.

        Raises:
            StoreError: ``state`` is one a running job cannot reach (``running``/``queued``).
        """
        if state in LIVE_STATES:
            raise StoreError(f"{state.value} is not a terminal state; use mark_running")
        finished_at = self._now()
        with self._store.writing() as conn:
            conn.execute(
                "UPDATE query_job SET"
                " state = ?,"
                " phase = ?,"
                " finished_at = ?,"
                " rows_read = COALESCE(?, rows_read),"
                " bytes_read = COALESCE(?, bytes_read),"
                " partitions_done = COALESCE(?, partitions_done),"
                " result_handle = COALESCE(?, result_handle),"
                " rows_total = COALESCE(?, rows_total),"
                " error_code = ?,"
                " error_json = ?,"
                " note = ?"
                " WHERE query_id = ?",
                (
                    state.value,
                    phase,
                    finished_at.isoformat(),
                    rows_read,
                    bytes_read,
                    partitions_done,
                    result_handle,
                    rows_total,
                    None if error is None else error.code,
                    None if error is None else error.model_dump_json(),
                    note,
                    query_id,
                ),
            )

    def get(self, query_id: str, *, owner: str) -> QueryJob | None:
        """The job, if ``owner`` started it. Another user's job answers exactly like a missing one."""
        row = (
            self._store.connect()
            .execute(
                "SELECT * FROM query_job WHERE query_id = ? AND user_id = ?", (query_id, owner)
            )
            .fetchone()
        )
        return None if row is None else _to_job(row)

    def count_live(self, *, owner: str | None = None) -> int:
        """How many jobs are still ``queued`` or ``running``, for the registry's ceiling."""
        row = (
            self._store.connect()
            .execute(
                "SELECT COUNT(*) FROM query_job WHERE state IN (?, ?)"
                " AND COALESCE(?, user_id) = user_id",
                (JobState.QUEUED.value, JobState.RUNNING.value, owner),
            )
            .fetchone()
        )
        return int(row[0]) if row is not None else 0

    def _now(self) -> datetime:
        """One clock reading per write, normalised to UTC (the store holds UTC only)."""
        moment = self._clock()
        if moment.tzinfo is None:
            raise StoreError("the clock returned a naive datetime; jobs store UTC only")
        return moment.astimezone(UTC)


def _to_job(row: sqlite3.Row) -> QueryJob:
    """Map one SQLite row onto the model; a row that no longer parses is an error, not a default."""
    error_text = row["error_json"]
    try:
        return QueryJob(
            id=int(row["id"]),
            query_id=str(row["query_id"]),
            user_id=str(row["user_id"]),
            state=JobState(str(row["state"])),
            source=str(row["source"]),
            phase=str(row["phase"]),
            est_rows=None if row["est_rows"] is None else float(row["est_rows"]),
            partitions_total=_opt_int(row["partitions_total"]),
            partitions_done=_opt_int(row["partitions_done"]),
            rows_read=int(row["rows_read"]),
            bytes_read=int(row["bytes_read"]),
            result_handle=None if row["result_handle"] is None else str(row["result_handle"]),
            rows_total=_opt_int(row["rows_total"]),
            error_code=None if row["error_code"] is None else str(row["error_code"]),
            error=None if error_text is None else ErrorResponse.model_validate_json(error_text),
            note=str(row["note"]),
            created_at=_parse_ts(str(row["created_at"])),
            started_at=_opt_ts(row["started_at"]),
            finished_at=_opt_ts(row["finished_at"]),
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise StoreError(f"stored query_job row is unreadable: {exc}") from exc


def _opt_int(value: object) -> int | None:
    return None if value is None else int(str(value))


def _opt_ts(value: object) -> datetime | None:
    return None if value is None else _parse_ts(str(value))


def _parse_ts(text: str) -> datetime:
    """Parse a stored UTC timestamp, refusing a naive one."""
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise StoreError("stored timestamp is naive; jobs store UTC only")
    return moment.astimezone(UTC)
