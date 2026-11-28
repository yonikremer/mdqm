"""The in-process registry of asynchronous query jobs: a worker thread and a place to look.

Deployment is one replica by rule (``docs/deploy.md`` section 8), so a registry in this process is
enough to *run* a job and to answer "how is it going". It is deliberately not enough on its own to
*see* the job: the state lives in the ``query_job`` table (:mod:`mdqm.store.jobs`), because
``get_more(query_id=...)`` arrives as a separate MCP message, possibly on another worker thread, and
a poll must never depend on which thread happens to hold the dict.

What this module adds on top of that table:

* **The worker.** One daemon thread per job, running exactly the call a blocking ``query`` would
  have run. Nothing here re-implements the pipeline: the work is a closure the query tool hands over
  together with the hooks it needs (:class:`JobHooks`).
* **Progress that is measured.** ``rows_read`` is what the governor drained, ``partitions_done`` is
  what the source's own plan said it would scan. Neither is invented: while a run is between
  batches the numbers simply do not move, and a value nobody reported stays ``None``.
* **A stop that reaches the driver.** ``cancel`` sets an event; the governor's drain loop sees it
  between batches and calls the adapter's own ``cancel`` on the live stream, so the source is told
  to stop rather than a thread being abandoned. A ``SourceAdapter.execute`` that is still blocking
  before its first batch cannot be interrupted this way; that limit is stated, not papered over.
* **A lost job reported as lost.** A row that still says ``running`` with no worker behind it (the
  process restarted) is marked ``failed`` with a precise reason the first time it is polled, instead
  of showing "running" for ever.

The ceiling is a count of live jobs in this process, and it is deliberately separate from the
governor's per-source semaphores: the governor protects the source, this protects the server's own
thread count. Both can refuse, and each refusal says which one did.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from pydantic import Field

from mdqm.errors import cancelled as cancelled_error
from mdqm.errors import from_exception, internal_error
from mdqm.governor.clock import Clock, SystemClock
from mdqm.models import ErrorResponse, JsonObject, MDQMModel, UtcDatetime
from mdqm.results.summary import ResultSummary
from mdqm.store.jobs import JobState, QueryJob, QueryJobStore

__all__ = [
    "DEFAULT_MAX_JOBS",
    "DEFAULT_PROGRESS_INTERVAL_S",
    "CancelCheck",
    "JobHooks",
    "JobOutcome",
    "JobProgress",
    "JobQueueFull",
    "JobRegistry",
    "PlanSink",
    "ProgressSink",
]

logger = logging.getLogger(__name__)

#: How many jobs one process runs at once before it refuses to start another. The governor's
#: semaphores protect the *source*; this protects the server's thread budget.
DEFAULT_MAX_JOBS: Final[int] = 8

#: How often measured progress is written to the row. Between writes the live numbers are still
#: served from memory, so a poll is never stale and the metadata DB is never hammered.
DEFAULT_PROGRESS_INTERVAL_S: Final[float] = 0.5

#: ``(rows drained so far, bytes so far)`` -- what the governor measured, after every batch.
ProgressSink = Callable[[int, int], None]

#: ``(partitions the plan will scan, partitions it knows about)`` -- both are ``None`` when the
#: source reported no plan, which is the honest answer, not zero.
PlanSink = Callable[[int | None, int | None], None]

#: Asked before every batch. True means "the caller wants this stopped".
CancelCheck = Callable[[], bool]


class JobQueueFull(RuntimeError):
    """This process already runs ``max_jobs`` queries asynchronously."""

    def __init__(self, max_jobs: int) -> None:
        self.max_jobs = max_jobs
        super().__init__(f"this server already runs {max_jobs} queries asynchronously")


@dataclass(frozen=True)
class JobHooks:
    """The three things a background run is given, so it can be watched and stopped.

    A blocking run passes ``None`` instead: there is nobody to report to, and inventing a sink that
    writes nowhere would be the kind of silent no-op this codebase refuses.
    """

    progress: ProgressSink
    plan: PlanSink
    cancelled: CancelCheck


class JobOutcome(MDQMModel):
    """What a finished job produced: the same summary/sample/handle a blocking answer carries.

    The plan is deliberately absent. It is reported live (``partitions_done``/``partitions_total``
    in :class:`JobProgress`) and the static estimate was already in the answer that started the job;
    keeping a second copy here would be a second thing to keep in step.
    """

    summary: ResultSummary | None = None
    sample: tuple[JsonObject, ...] = ()
    result_handle: str | None = Field(
        default=None, description="Page it with `get_more(result_handle=...)`."
    )
    rows_total: int = Field(default=0, ge=0)
    error: ErrorResponse | None = Field(
        default=None, description="Set when the run did not answer."
    )


class JobProgress(MDQMModel):
    """One job's live status, as ``get_more(query_id=...)`` reports it."""

    query_id: str
    state: JobState
    phase: str = Field(default="", description="planning, executing or finished.")
    elapsed_s: float = Field(default=0.0, ge=0, description="Wall clock since the job was created.")
    rows_read: int = Field(default=0, ge=0, description="Rows the source has produced so far.")
    bytes_read: int = Field(default=0, ge=0)
    est_rows: float | None = Field(default=None, ge=0, description="The plan's static estimate.")
    partitions_total: int | None = Field(
        default=None, ge=0, description="Partitions the window touches, when the plan knows."
    )
    partitions_done: int | None = Field(
        default=None,
        ge=0,
        description="Partitions the source's plan said it will scan; null when it said nothing.",
    )
    result_handle: str | None = Field(
        default=None,
        description="Set once the job answered: page it with `get_more(result_handle=...)`.",
    )
    rows_total: int | None = Field(default=None, ge=0)
    error: ErrorResponse | None = Field(
        default=None, description="Set when the job failed or was cancelled."
    )
    note: str = Field(default="", description="One line on anything unusual about this run.")
    created_at: UtcDatetime
    started_at: UtcDatetime | None = None
    finished_at: UtcDatetime | None = None

    @property
    def finished(self) -> bool:
        """True when this job will not change again."""
        return self.state not in {JobState.QUEUED, JobState.RUNNING}


class _Live:
    """The mutable half of a running job: never in the database, only in this process."""

    __slots__ = ("bytes_read", "event", "last_write_s", "note", "phase", "rows_read")

    def __init__(self, *, now_monotonic: float) -> None:
        self.event = threading.Event()
        self.rows_read = 0
        self.bytes_read = 0
        self.phase = "queued"
        self.note = ""
        self.last_write_s = now_monotonic


@dataclass
class _Entry:
    """One tracked job: its row, its live counters, its thread and (once finished) its answer."""

    job: QueryJob
    live: _Live
    thread: threading.Thread | None = None
    outcome: JobOutcome | None = field(default=None)


class JobRegistry:
    """Start, watch and stop asynchronous query jobs; see the module docstring.

    Args:
        store: the ``query_job`` table (already migrated).
        clock: injected time source, so elapsed time and the write throttle are testable.
        max_jobs: how many jobs may run at once in this process.
        progress_interval_s: how often measured progress is written to the row; 0 writes every
            batch (tests and very small results).
    """

    def __init__(
        self,
        store: QueryJobStore,
        *,
        clock: Clock | None = None,
        max_jobs: int = DEFAULT_MAX_JOBS,
        progress_interval_s: float = DEFAULT_PROGRESS_INTERVAL_S,
    ) -> None:
        if max_jobs < 1:
            raise ValueError(f"max_jobs must be at least 1, got {max_jobs}")
        if progress_interval_s < 0:
            raise ValueError(f"progress_interval_s must not be negative, got {progress_interval_s}")
        self._store = store
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._max_jobs = max_jobs
        self._interval = progress_interval_s
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}

    @property
    def max_jobs(self) -> int:
        """How many jobs this registry runs at once."""
        return self._max_jobs

    # --- starting ---------------------------------------------------------------------------

    def start(
        self,
        *,
        user_id: str,
        source: str,
        work: Callable[[JobHooks], JobOutcome],
        est_rows: float | None = None,
        partitions_total: int | None = None,
    ) -> QueryJob:
        """Record the job, then run ``work`` on its own thread.

        Returns:
            The ``queued`` row: its ``query_id`` is what the caller polls.

        Raises:
            JobQueueFull: this process already runs ``max_jobs`` jobs.
        """
        with self._lock:
            running = sum(1 for entry in self._entries.values() if entry.outcome is None)
            if running >= self._max_jobs:
                raise JobQueueFull(self._max_jobs)
            job = self._store.create(
                user_id=user_id,
                source=source,
                est_rows=est_rows,
                partitions_total=partitions_total,
            )
            entry = _Entry(job=job, live=_Live(now_monotonic=self._clock.monotonic()))
            self._entries[job.query_id] = entry
            thread = threading.Thread(
                target=self._work,
                args=(entry, work),
                name=f"mdqm-job-{job.query_id}",
                daemon=True,
            )
            entry.thread = thread
            thread.start()
        logger.info("asynchronous query job started", extra={"query_id": job.query_id})
        return job

    def _work(self, entry: _Entry, work: Callable[[JobHooks], JobOutcome]) -> None:
        """The thread body: run the blocking call with hooks, then record what happened."""
        job = entry.job
        hooks = JobHooks(
            progress=self._progress_sink(entry),
            plan=self._plan_sink(entry),
            cancelled=entry.live.event.is_set,
        )
        try:
            if entry.live.event.is_set():
                self._finish(
                    entry,
                    JobOutcome(error=cancelled_error(query_id=job.query_id)),
                    JobState.CANCELLED,
                )
                return
            entry.live.phase = "executing"
            self._store.mark_running(job.query_id, phase="executing")
            outcome = work(hooks)
        except Exception as exc:
            logger.exception("an asynchronous query job failed", extra={"query_id": job.query_id})
            self._finish(
                entry,
                JobOutcome(error=from_exception(exc, source=job.source or None)),
                JobState.FAILED,
            )
            return
        if entry.live.event.is_set() or _was_cancelled(outcome.error):
            state = JobState.CANCELLED
        elif outcome.error is not None:
            state = JobState.FAILED
        else:
            state = JobState.DONE
        if state is JobState.CANCELLED and entry.live.note == "":
            entry.live.note = "stopped at the caller's request"
        self._finish(entry, outcome, state)

    def _finish(self, entry: _Entry, outcome: JobOutcome, state: JobState) -> None:
        """Write the terminal state, then publish the outcome to this process's pollers."""
        try:
            self._store.mark_finished(
                entry.job.query_id,
                state=state,
                rows_read=entry.live.rows_read,
                bytes_read=entry.live.bytes_read,
                result_handle=outcome.result_handle,
                rows_total=None if outcome.error is not None else outcome.rows_total,
                error=outcome.error,
                note=entry.live.note,
            )
        except Exception:  # the caller's answer still exists; losing the row is what we report
            logger.exception(
                "could not record the end of an asynchronous query job",
                extra={"query_id": entry.job.query_id},
            )
        with self._lock:
            entry.outcome = outcome
            entry.live.phase = "finished"
        logger.info(
            "asynchronous query job finished",
            extra={"query_id": entry.job.query_id, "state": state.value},
        )

    def _progress_sink(self, entry: _Entry) -> ProgressSink:
        """Write measured progress, at most once per ``progress_interval_s``."""

        def sink(rows_read: int, bytes_read: int) -> None:
            entry.live.rows_read = rows_read
            entry.live.bytes_read = bytes_read
            now = self._clock.monotonic()
            if now - entry.live.last_write_s < self._interval:
                return
            entry.live.last_write_s = now
            self._write_progress(entry)

        return sink

    def _plan_sink(self, entry: _Entry) -> PlanSink:
        """Record what the source's plan said it will scan. Called at most once per run."""

        def sink(partitions_scanned: int | None, partitions_total: int | None) -> None:
            if partitions_scanned is None and partitions_total is None:
                return
            try:
                self._store.mark_progress_shape(
                    entry.job.query_id,
                    partitions_done=partitions_scanned,
                    partitions_total=partitions_total,
                )
            except Exception:  # progress is a courtesy; the run itself must not die for it
                logger.warning(
                    "could not record a job's plan", extra={"query_id": entry.job.query_id}
                )

        return sink

    def _write_progress(self, entry: _Entry) -> None:
        """One small UPDATE with the live counters, ignoring a failure to write it."""
        try:
            self._store.mark_progress(
                entry.job.query_id,
                rows_read=entry.live.rows_read,
                bytes_read=entry.live.bytes_read,
            )
        except Exception:  # a lost progress write must never kill a running query
            logger.warning("could not record job progress", extra={"query_id": entry.job.query_id})

    # --- watching ---------------------------------------------------------------------------

    def status(self, query_id: str, *, owner: str) -> JobProgress | None:
        """The job's live status for ``owner``, or ``None`` when they own no such job."""
        job = self._store.get(query_id, owner=owner)
        if job is None:
            return None
        entry = self._live_entry(query_id)
        if job.live and entry is None:
            job = self._reap(query_id, owner=owner) or job
        return _progress(job, entry, now=self._clock.now())

    def outcome(self, query_id: str, *, owner: str) -> JobOutcome | None:
        """The finished answer, when this process still holds it.

        ``None`` for a job that never ran here (a restart, or a row from another deployment) even
        when its state is ``done``: the handle in the row is the durable answer, and paging it does
        not need this object.
        """
        entry = self._live_entry(query_id)
        if entry is None or entry.outcome is None:
            return None
        owner_check = self._store.get(query_id, owner=owner)
        return None if owner_check is None else entry.outcome

    def cancel(self, query_id: str, *, owner: str) -> JobProgress | None:
        """Ask the worker to stop, and report the state immediately.

        The answer to this call is the state *now*: the governor checks the stop between batches and
        the adapter's own ``cancel`` is called on the live stream, so a later poll is what confirms
        it. A job that already finished is reported as it is, unchanged.
        """
        job = self._store.get(query_id, owner=owner)
        if job is None:
            return None
        entry = self._live_entry(query_id)
        if entry is not None and entry.outcome is None:
            entry.live.event.set()
            entry.live.note = "stopped at the caller's request"
        return _progress(job, entry, now=self._clock.now())

    def wait(self, query_id: str, *, timeout_s: float = 10.0) -> bool:
        """Block until the job's thread is done. For tests and for an orderly shutdown."""
        entry = self._live_entry(query_id)
        if entry is None or entry.thread is None:
            return True
        entry.thread.join(timeout_s)
        return not entry.thread.is_alive()

    def shutdown(self, *, timeout_s: float = 10.0) -> None:
        """Ask every job to stop and join it, so a test does not leak a running query."""
        with self._lock:
            entries = list(self._entries.values())
        for entry in entries:
            entry.live.event.set()
        for entry in entries:
            if entry.thread is not None:
                entry.thread.join(timeout_s)

    # --- internals --------------------------------------------------------------------------

    def _live_entry(self, query_id: str) -> _Entry | None:
        with self._lock:
            return self._entries.get(query_id)

    def _reap(self, query_id: str, *, owner: str) -> QueryJob | None:
        """Mark a row whose worker is gone as failed, and say exactly why.

        This is the one path that writes while reading, and it earns it: a poll must not show
        ``running`` for a job no thread will ever finish.
        """
        try:
            self._store.mark_finished(
                query_id,
                state=JobState.FAILED,
                error=internal_error(correlation_id=uuid.uuid4().hex[:16]),
                note="the server restarted (or the job was evicted) while this job was running, "
                "so its work was lost",
            )
        except Exception:
            logger.exception("could not reap a lost job", extra={"query_id": query_id})
            return None
        logger.warning("marked a lost query job as failed", extra={"query_id": query_id})
        return self._store.get(query_id, owner=owner)


def _was_cancelled(error: ErrorResponse | None) -> bool:
    """True when the run's own error says the caller stopped it."""
    return error is not None and error.code == "CANCELLED"


def _progress(job: QueryJob, entry: _Entry | None, *, now: datetime) -> JobProgress:
    """Merge the stored row with the live counters of a job this process is still running."""
    live = entry.live if entry is not None and entry.outcome is None else None
    end = job.finished_at if job.finished_at is not None else now
    return JobProgress(
        query_id=job.query_id,
        state=job.state,
        phase=live.phase if live is not None else job.phase,
        elapsed_s=max(0.0, (end - job.created_at).total_seconds()),
        rows_read=live.rows_read if live is not None else job.rows_read,
        bytes_read=live.bytes_read if live is not None else job.bytes_read,
        est_rows=job.est_rows,
        partitions_total=job.partitions_total,
        partitions_done=job.partitions_done,
        result_handle=job.result_handle,
        rows_total=job.rows_total,
        error=job.error if live is None else None,
        note=live.note if live is not None and live.note else job.note,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )
