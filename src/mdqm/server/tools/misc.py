"""The three lookup tools: page a stored result, recall your own runs, rate an answer (M05).

Each body is deliberately thin, because the behaviour already lives elsewhere and is tested there:
:class:`~mdqm.governor.handles.ResultStore` owns paging, ownership and TTL;
:class:`~mdqm.store.history.HistoryStore` owns search and the frozen/relative re-run;
:class:`~mdqm.store.feedback.FeedbackStore` owns the rating queue. This module's job is the part
those modules must not know about:

* **Who is asking.** The caller comes from the authenticated message
  (:func:`mdqm.server.context.current_context`), never from an argument. Every lookup passes that id
  down as the owner, so an answer can only ever be about the caller's own rows.
* **What the agent sees.** One answer shape with a ``status``, so a weak model branches on one field
  instead of guessing: ``ok`` with a payload, ``error`` with the structured
  :class:`~mdqm.models.ErrorResponse`, or ``not_implemented`` when the server was started without
  the stores. A refusal is also appended to the audit log, so "someone tried another user's handle"
  is a row, not a hunch.
* **That a refusal leaks nothing.** An unknown id, an expired one and another user's produce the
  *same* text (``NOT_FOUND``): the answer never confirms that someone else's result exists.

The stores are injected with :func:`configure_services` when the server is built. That is one
process-wide slot rather than a per-call argument because the tool signatures are the MCP wire
schema: adding a dependency parameter would leak into the JSON schema the agent reads. A call with
no services configured answers ``not_implemented`` and says exactly that, so a wiring mistake shows
up as itself instead of as an empty result.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field

from mdqm.errors import invalid_spec, not_found
from mdqm.governor.handles import HandleNotFound, Page, ResultStore
from mdqm.models import ErrorResponse, JsonObject, UtcDatetime
from mdqm.models.base import MDQMModel
from mdqm.results.summary import ResultSummary
from mdqm.server.context import current_context
from mdqm.server.jobs import JobProgress, JobRegistry
from mdqm.spec.models import QuerySpec
from mdqm.store.audit import AuditLog
from mdqm.store.feedback import FeedbackEntry, FeedbackRecord, FeedbackStore
from mdqm.store.history import (
    CacheInfo,
    HistoryError,
    HistoryRecord,
    HistoryStore,
    RerunPlan,
)

__all__ = [
    "Answer",
    "FeedbackAnswer",
    "GetMoreAnswer",
    "HistoryAnswer",
    "HistoryItem",
    "ServicesUnavailable",
    "ToolServices",
    "configure_services",
    "current_services",
    "feedback",
    "get_more",
    "history",
    "reset_services",
    "use_services",
]

logger = logging.getLogger(__name__)

#: Longest `corrected` sentence the review queue keeps. One line, not a report.
_MAX_CORRECTED: int = 4000


# --- what the tools are given -----------------------------------------------------------


@dataclass(frozen=True)
class ToolServices:
    """The stores the lookup tools read, built once by the server and injected.

    Args:
        handles: the result store the ``query`` tool also filled; paging happens here.
        history: the caller's stored runs, for ``history``.
        feedback: the rating queue, for ``feedback``.
        audit: where a refusal is appended, so a cross-user attempt leaves a row behind.
        jobs: the registry that runs ``mode="async"`` queries, so ``get_more`` can report on one.
            It must be the *same* object the query tool was given; ``None`` is a build without it.
    """

    handles: ResultStore
    history: HistoryStore
    feedback: FeedbackStore
    audit: AuditLog
    jobs: JobRegistry | None = None


class ServicesUnavailable(RuntimeError):
    """No stores were configured on this server: the lookup tools cannot answer."""


_services: ToolServices | None = None


def configure_services(services: ToolServices | None) -> None:
    """Hand the lookup tools their stores. Called once when the server is built; ``None`` clears it.

    Process-wide on purpose: one server process has one result store, one metadata database and
    one audit log, and the tool signatures are the MCP wire schema, so the dependency cannot travel
    as an argument. Tests call this (and :func:`reset_services`) in a fixture.
    """
    global _services
    _services = services


def reset_services() -> None:
    """Forget the configured stores. Tests use it; production configures once at startup."""
    configure_services(None)


def current_services() -> ToolServices:
    """The configured stores.

    Raises:
        ServicesUnavailable: the server was built without them. A tool reports that to the agent as
            ``not_implemented`` instead of raising, because a wiring bug is not the agent's to fix.
    """
    if _services is None:
        raise ServicesUnavailable(
            "the lookup tools are registered but no stores were configured: call "
            "configure_services() when building the server"
        )
    return _services


@contextmanager
def use_services(services: ToolServices) -> Iterator[ToolServices]:
    """Configure the stores for the duration of a block (tests, and a future per-request scope)."""
    previous = _services
    configure_services(services)
    try:
        yield services
    finally:
        configure_services(previous)


# --- what the agent sees ----------------------------------------------------------------


class Answer(MDQMModel):
    """One answer shape for all three lookup tools. A weak model branches on ``status`` only.

    ``status`` is ``ok`` with the payload, ``error`` with the structured ``ErrorResponse`` the agent
    must read, or ``not_implemented`` with ``problem``/``fix`` when the server has no stores.
    ``get_more`` adds the states of an asynchronous job -- ``queued``, ``running``, ``done``,
    ``failed``, ``cancelled`` -- so a poll reports the job's state in this same field rather than
    making the caller read a second one.
    """

    status: Literal[
        "ok",
        "error",
        "not_implemented",
        "queued",
        "running",
        "done",
        "failed",
        "cancelled",
    ] = "ok"
    surface: str = Field(description="Which tool answered.")
    user_id: str = Field(description="The authenticated caller the server saw.")
    error: ErrorResponse | None = Field(default=None, description="Set when status is 'error'.")
    problem: str | None = Field(default=None, description="What is missing (not_implemented).")
    fix: str | None = Field(default=None, description="What to do instead (not_implemented).")


class GetMoreAnswer(Answer):
    """A page of a stored result, or the state of a job and (once it is ``done``) its answer.

    The two references are separate fields on purpose: ``result_handle`` pages rows that already
    exist, ``query_id`` watches work that is still running. Send exactly one of them. When a job
    finishes, this answer carries the same ``summary``, ``sample`` and ``result_handle`` a blocking
    ``query`` would have returned, so the next step is the same page-the-handle call as always.
    """

    page: Page | None = Field(default=None, description="The rows, and where the next page starts.")
    progress: JobProgress | None = Field(
        default=None,
        description=(
            "The job's state, measured progress and (for a finished job) its `result_handle`. "
            "`partitions_done` is null until the source's own plan said how many it will scan."
        ),
    )
    summary: ResultSummary | None = Field(
        default=None, description="A finished job's bounded answer, like `query` returns."
    )
    sample: tuple[JsonObject, ...] = Field(
        default=(), description="The first rows of a finished job's result."
    )
    result_handle: str | None = Field(
        default=None, description="A finished job's handle: page it with this same tool."
    )
    rows_total: int = Field(default=0, ge=0)


class HistoryItem(MDQMModel):
    """One of the caller's past runs, in the shape a weak model can recognise.

    The spec or the SQL is included because that is what makes a run recognisable; result rows and
    credentials never are (``mdqm.store.history`` stores neither).
    """

    id: int = Field(ge=1, description="The run id, which `feedback` and `rerun_id` take.")
    created_at: UtcDatetime
    exec_ts: UtcDatetime = Field(description="The one timestamp that run's time window used.")
    tool: str
    mode: str = Field(description="'frozen' when the window did not move with the clock.")
    spec: QuerySpec | None = None
    sql: str | None = Field(default=None, description="The raw SQL, when the run was raw SQL.")
    cache: CacheInfo | None = None
    error_code: str | None = Field(default=None, description="Set when the run failed.")
    rerunnable: bool = Field(description="False when a credential was redacted out of the row.")


class HistoryAnswer(Answer):
    """The caller's runs, and/or the plan a re-run would execute."""

    runs: tuple[HistoryItem, ...] = ()
    rerun: RerunPlan | None = Field(
        default=None, description="What a re-run executes: one exec_ts and the payloads at it."
    )


class FeedbackAnswer(Answer):
    """The stored rating, as the queue holds it."""

    saved: FeedbackRecord | None = None


def _unavailable(surface: str, user_id: str, problem: str, fix: str) -> Answer:
    """The honest answer when the server has no stores: not a silent empty result."""
    return Answer(
        status="not_implemented", surface=surface, user_id=user_id, problem=problem, fix=fix
    )


def _refusal(*, surface: str, user_id: str, error: ErrorResponse) -> ErrorResponse:
    """Audit a refusal, then hand the same structured error to the agent."""
    try:
        current_services().audit.record_refusal(user_id=user_id, tool=surface, error=error)
    except Exception:
        # The refusal is the caller's answer; the failed audit is ours to log, never to swap for a
        # different code. Losing one audit row is bad, answering with the wrong error is worse.
        logger.exception("could not audit a %s refusal", surface, extra={"error_code": error.code})
    return error


def _item(record: HistoryRecord) -> HistoryItem:
    """The compact, agent-facing view of one stored run."""
    return HistoryItem(
        id=record.id,
        created_at=record.created_at,
        exec_ts=record.exec_ts,
        tool=record.tool,
        mode=record.mode.value,
        spec=record.spec,
        sql=record.original_sql if record.original_sql is not None else record.frozen_sql,
        cache=record.cache,
        error_code=None if record.error is None else record.error.code,
        rerunnable=record.rerunnable,
    )


def _run_id_of(query_id: str) -> int | None:
    """Read a run id out of a ``query_id``: ``"12"`` and ``"q_12"`` both mean run 12.

    Anything else is not an id this server issues, so it cannot name a run the caller owns.
    """
    text = query_id.strip()
    if text[:2] in {"q_", "Q_"}:
        text = text[2:]
    return int(text) if text.isdigit() and int(text) > 0 else None


# --- the three tools --------------------------------------------------------------------


def get_more(
    result_handle: Annotated[
        str | None,
        Field(
            min_length=1,
            description=(
                "A `result_handle` (or a summary's `handle`) from a previous answer: pages those "
                "rows. Send this OR `query_id`, never both."
            ),
        ),
    ] = None,
    offset: Annotated[
        int, Field(ge=0, description="Rows to skip; 0 for the next page in order.")
    ] = 0,
    limit: Annotated[int, Field(ge=1, le=1000, description="Rows to return.")] = 100,
    query_id: Annotated[
        str | None,
        Field(
            min_length=1,
            description=(
                'The `query_id` from a `query` answer with `mode: "async"`: reports that job\'s '
                "progress, and its summary/sample/`result_handle` once it is done."
            ),
        ),
    ] = None,
    cancel: Annotated[
        bool,
        Field(
            description=(
                "With `query_id`: ask the server to stop the job. The source is told to stop "
                "between batches, so poll once more with `cancel: false` to confirm."
            )
        ),
    ] = False,
) -> GetMoreAnswer:
    """Page the rest of a result, or watch a job you started asynchronously.

    With a `result_handle` from a previous answer it pages those cached rows; it never re-runs a
    query. Your own handles only; someone else's reads like one that expired.
    With a `query_id` (from `query` with `mode: "async"`) it reports progress: `status` stays
    `running` until the job finishes, then `done` with the `summary`, `sample` and `result_handle`
    a blocking call returns. `cancel: true` asks the job to stop.
    Example: get_more({"result_handle": "h_8f21"}) or get_more({"query_id": "j_7"})
    """
    context = current_context()
    services = _services
    if services is None:
        return GetMoreAnswer(
            status="not_implemented",
            surface="get_more",
            user_id=context.user_id,
            problem="no result store is configured on this server, so no handle can be paged",
            fix="re-run `query` once the server is configured; nothing is cached in this build",
        )
    if query_id is not None:
        if result_handle is not None:
            return GetMoreAnswer(
                status="error",
                surface="get_more",
                user_id=context.user_id,
                error=_refusal(
                    surface="get_more",
                    user_id=context.user_id,
                    error=invalid_spec(
                        field="query_id",
                        reason=(
                            "send either `result_handle` (rows that exist) or `query_id` (a job "
                            "to watch), not both: they are two different references"
                        ),
                        expected=("query_id", "result_handle"),
                    ),
                ),
            )
        return _job(services, context.user_id, query_id, cancel=cancel)
    if result_handle is None:
        return GetMoreAnswer(
            status="error",
            surface="get_more",
            user_id=context.user_id,
            error=_refusal(
                surface="get_more",
                user_id=context.user_id,
                error=invalid_spec(
                    field="result_handle",
                    reason="send `result_handle` to page rows, or `query_id` to watch a job",
                    expected=("result_handle", "query_id"),
                ),
            ),
        )
    try:
        page = services.handles.get_more(
            result_handle, owner=context.user_id, offset=offset, limit=limit
        )
    except HandleNotFound:
        error = _refusal(
            surface="get_more",
            user_id=context.user_id,
            error=not_found(
                what="result handle",
                fix="Call `query` again to get a fresh handle: handles are short-lived, and one "
                "issued to someone else is not readable. Never guess or reuse an id.",
            ),
        )
        return GetMoreAnswer(
            status="error", surface="get_more", user_id=context.user_id, error=error
        )
    return GetMoreAnswer(surface="get_more", user_id=context.user_id, page=page)


def _job(services: ToolServices, user_id: str, query_id: str, *, cancel: bool) -> GetMoreAnswer:
    """Report one asynchronous job, or stop it, from the registry and its stored row.

    A finished job answers with the same three things a blocking run would have returned. When this
    process no longer holds the run's summary (it restarted), the answer still carries the
    `result_handle` the row recorded, which is enough to page the rows: losing a summary is a
    smaller loss than pretending the job never finished.
    """
    registry = services.jobs
    if registry is None:
        return GetMoreAnswer(
            status="not_implemented",
            surface="get_more",
            user_id=user_id,
            problem=(
                "this server runs no query jobs, so a `query_id` cannot be watched here; only "
                "`result_handle` can be paged"
            ),
            fix="call `query` without `mode` and wait for its answer, then page its `result_handle`",
        )
    progress = (
        registry.cancel(query_id, owner=user_id)
        if cancel
        else registry.status(query_id, owner=user_id)
    )
    if progress is None:
        error = _refusal(
            surface="get_more",
            user_id=user_id,
            error=not_found(
                what="query id",
                fix='Use the `query_id` a `query` answer returned with `status: "started"`. '
                "Job ids are short-lived and one issued to someone else is not readable.",
            ),
        )
        return GetMoreAnswer(status="error", surface="get_more", user_id=user_id, error=error)
    if not progress.finished:
        return GetMoreAnswer(
            status=progress.state.value,
            surface="get_more",
            user_id=user_id,
            progress=progress,
        )
    outcome = registry.outcome(query_id, owner=user_id)
    return GetMoreAnswer(
        status=progress.state.value,
        surface="get_more",
        user_id=user_id,
        progress=progress,
        error=progress.error,
        summary=None if outcome is None else outcome.summary,
        sample=() if outcome is None else outcome.sample,
        result_handle=progress.result_handle,
        rows_total=progress.rows_total or 0,
    )


def history(
    text: Annotated[
        str | None,
        Field(description="Words to look for in your own past runs (Hebrew or English)."),
    ] = None,
    rerun_id: Annotated[
        int | None,
        Field(ge=1, description="A run `id` from this list: ask what a re-run would send."),
    ] = None,
    rerun_mode: Annotated[
        Literal["frozen", "relative"],
        Field(description="'frozen' repeats the original window; 'relative' moves it to now."),
    ] = "frozen",
    limit: Annotated[int, Field(ge=1, le=50, description="How many runs to list.")] = 5,
) -> HistoryAnswer:
    """List your own recent queries, search them, or show what a re-run would send.

    Call this when the user refers to something asked before, to get a run `id` for `feedback`, or
    with `rerun_id` to repeat an old question exactly: `rerun_mode` 'frozen' keeps the original time
    window, 'relative' moves it to now. Only your own runs are listed.
    Example: history({"text": "credit limit", "limit": 5}) or history({"rerun_id": 12,
    "rerun_mode": "relative"})
    """
    context = current_context()
    services = _services
    if services is None:
        return HistoryAnswer(
            status="not_implemented",
            surface="history",
            user_id=context.user_id,
            problem="no history store is configured on this server, so past runs cannot be listed",
            fix="restate the question and call `query`; history is unavailable in this build",
        )
    if rerun_id is not None:
        return _rerun(services, context.user_id, rerun_id, rerun_mode)
    try:
        records = (
            services.history.search(text, user_id=context.user_id, limit=limit)
            if text is not None
            else services.history.recent(user_id=context.user_id, limit=limit)
        )
    except HistoryError:
        error = _refusal(
            surface="history",
            user_id=context.user_id,
            error=invalid_spec(
                field="text",
                reason="it needs at least one word to look for",
            ),
        )
        return HistoryAnswer(
            status="error", surface="history", user_id=context.user_id, error=error
        )
    return HistoryAnswer(
        surface="history", user_id=context.user_id, runs=tuple(_item(r) for r in records)
    )


def _rerun(services: ToolServices, user_id: str, rerun_id: int, rerun_mode: str) -> HistoryAnswer:
    """Resolve the run (owner check first), then hand back the plan its re-run would execute."""
    try:
        record = services.history.get(rerun_id)
    except HistoryError:
        record = None
    # A run that is not the caller's answers exactly like one that does not exist.
    if record is None or record.user_id != user_id:
        error = _refusal(
            surface="history",
            user_id=user_id,
            error=not_found(
                what="run id",
                fix="Call `history` without arguments to list your own run ids, then pass one of "
                "those as `rerun_id`.",
            ),
        )
        return HistoryAnswer(status="error", surface="history", user_id=user_id, error=error)
    try:
        plan = services.history.rerun(rerun_id, mode=rerun_mode)
    except HistoryError:
        error = _refusal(
            surface="history",
            user_id=user_id,
            error=invalid_spec(
                field="rerun_id",
                reason="a credential was redacted out of that run, so a re-run would not be exact",
            ),
        )
        return HistoryAnswer(status="error", surface="history", user_id=user_id, error=error)
    return HistoryAnswer(surface="history", user_id=user_id, runs=(_item(record),), rerun=plan)


def feedback(
    query_id: Annotated[
        str,
        Field(
            min_length=1,
            description="The run `id` from `history` (`12` or `q_12`) of an answer.",
        ),
    ],
    ok: Annotated[
        bool, Field(description="True when the answer was what the caller needed, false when not.")
    ],
    corrected: Annotated[
        str | None,
        Field(description="When ok is false: one line on what the answer should have been."),
    ] = None,
) -> FeedbackAnswer:
    """Rate a previous answer so the semantic layer learns which tables and measures work.

    Call this once per answered question when the user says the answer was right or wrong. The
    rating is queued, not applied: a human approves anything derived from it.
    Example: feedback({"query_id": "12", "ok": false, "corrected": "status S means suspended"})
    """
    context = current_context()
    services = _services
    if services is None:
        return FeedbackAnswer(
            status="not_implemented",
            surface="feedback",
            user_id=context.user_id,
            problem="no feedback queue is configured on this server, so nothing can be stored",
            fix="carry on; the rating is dropped in this build",
        )
    run_id = _run_id_of(query_id)
    record = None
    if run_id is not None:
        try:
            record = services.history.get(run_id)
        except HistoryError:
            record = None
    # An id this server never issued and another user's run answer exactly alike: the caller learns
    # only that they own no such run, which is all they are allowed to learn.
    if run_id is None or record is None or record.user_id != context.user_id:
        error = _refusal(
            surface="feedback",
            user_id=context.user_id,
            error=not_found(
                what="run id",
                fix="Call `history` without arguments to list your own run ids, then rate one of "
                "those as `query_id`.",
            ),
        )
        return FeedbackAnswer(
            status="error", surface="feedback", user_id=context.user_id, error=error
        )
    saved = services.feedback.record(
        FeedbackEntry(
            user_id=context.user_id,
            query_id=query_id,
            run_id=run_id,
            ok=ok,
            corrected=None if corrected is None else corrected[:_MAX_CORRECTED],
        )
    )
    return FeedbackAnswer(surface="feedback", user_id=context.user_id, saved=saved)
