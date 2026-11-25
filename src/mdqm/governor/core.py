"""The governor: one call that takes a slot, honours a deadline and hands back a handle.

This is where the four protections meet, in a fixed order, so a weak agent cannot get past any of
them (:class:`~mdqm.governor.admission.Admission` ceilings and quota,
:class:`~mdqm.governor.breaker.CircuitBreaker`, the per-call deadline with a real ``cancel``, and
:class:`~mdqm.governor.handles.ResultStore`):

1. **The breaker** refuses a source already known to be down, before any slot is taken.
2. **The slots** bound global, per-user and per-source concurrency; a refused slot releases nothing
   and reports which ceiling bound it.
3. **The quota** spends one query from the user's window; an exhausted budget reports its reset.
4. **The deadline** is checked around every batch while the stream is drained. When it passes, the
   adapter's own ``cancel`` path is called on the live stream -- the driver is told to stop, rather
   than a thread being abandoned -- and the refusal is a ``TIMEOUT`` naming the source and limit.
5. **The handle** stores the full table and the agent receives summary + sample + handle.

Cancellation is the adapter's contract, not something the governor invents: ``SourceAdapter.cancel``
takes the stream ``execute`` returned, so the cancel point is the stream. An ``execute`` that blocks
for ever, before any stream exists, cannot be cancelled through that protocol; every adapter bounds
its own connect/execute time with ``Limits.timeout_s`` (the fake one does exactly that). The
governor's deadline covers the slow-stream case, which is the one where a primary database is
actually burning a worker.

The slot is released as soon as the refusal is raised (:class:`Admission`'s ``finally``), not when
the cancelled stream finishes unwinding, so one slow source cannot starve the others.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import pyarrow as pa
from pydantic import Field

from mdqm.adapters.base import ArrowStream, SourceAdapter
from mdqm.errors import from_exception
from mdqm.errors import timeout as timeout_error
from mdqm.governor.admission import Admission, Peak, Quota, Scope
from mdqm.governor.breaker import BreakerReport, CircuitBreaker
from mdqm.governor.clock import Clock, SystemClock
from mdqm.governor.handles import Page, ResultHandle, ResultStore
from mdqm.governor.refusal import GovernorRefusal
from mdqm.models import Fragment, Limits
from mdqm.models.base import MDQMModel
from mdqm.results.summary import (
    DEFAULT_PAGE_ROWS,
    DEFAULT_SAMPLE_ROWS,
    DEFAULT_TOKEN_BUDGET,
    ResultSummary,
    summarise,
)

__all__ = [
    "ExecutionOutcome",
    "Governance",
    "Governor",
    "GovernorReport",
]


class GovernorReport(MDQMModel):
    """What the governor did to this call. Every clamp and refusal is visible here.

    ``row_cap_hit``/``byte_cap_hit`` mean the result may be truncated at ``Limits``, which the agent
    must know about; ``waited_s`` is queue time, not query time.
    """

    source: str
    user: str
    waited_s: float = Field(ge=0, description="Seconds spent queueing for a slot.")
    row_cap_hit: bool = Field(description="True when the row cap stopped the scan.")
    byte_cap_hit: bool = Field(description="True when the byte cap stopped the scan.")
    rows: int = Field(ge=0)
    nbytes: int = Field(ge=0)
    source_peak: int = Field(ge=0, description="Most queries this source ran at once, so far.")
    source_limit: int = Field(ge=1)


class ExecutionOutcome(MDQMModel):
    """The agent's answer plus the governor's receipt."""

    summary: ResultSummary
    handle: ResultHandle
    report: GovernorReport


class Governance(MDQMModel):
    """A point-in-time view of every protection, for ``mdqm audit`` and for tests."""

    peaks: tuple[Peak, ...] = ()
    breakers: tuple[BreakerReport, ...] = ()


class _Drained(NamedTuple):
    """Internal: what draining learned, before the receipt is assembled with the peaks."""

    table: pa.Table
    row_cap_hit: bool
    byte_cap_hit: bool


class Governor:
    """Runs one fragment under every protection; see the module docstring.

    Args:
        adapters: source name -> adapter. Explicit, because a badly named source must fail loudly
            here rather than reach a database.
        admission: the concurrency ceilings; a default policy when omitted.
        quota: the per-user budget, or ``None`` for none.
        breaker: the per-source circuit breaker; a default policy when omitted.
        store: the result store; a default one when omitted.
        clock: injected time source, shared with the deadline check.
    """

    def __init__(
        self,
        adapters: Mapping[str, SourceAdapter],
        *,
        admission: Admission | None = None,
        quota: Quota | None = None,
        breaker: CircuitBreaker | None = None,
        store: ResultStore | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._adapters = dict(adapters)
        self._admission = admission if admission is not None else Admission(clock=self._clock)
        self._quota = quota
        self._breaker = breaker if breaker is not None else CircuitBreaker(clock=self._clock)
        self._store = store if store is not None else ResultStore(clock=self._clock)

    @property
    def store(self) -> ResultStore:
        """The result store, so the server can page a handle it handed out."""
        return self._store

    @property
    def breaker(self) -> CircuitBreaker:
        """The circuit breaker, so the server can report a source's state."""
        return self._breaker

    def execute(
        self,
        *,
        source: str,
        user: str,
        fragment: Fragment,
        limits: Limits,
        sample_rows: int = DEFAULT_SAMPLE_ROWS,
        page_rows: int = DEFAULT_PAGE_ROWS,
        budget_tokens: int = DEFAULT_TOKEN_BUDGET,
    ) -> ExecutionOutcome:
        """Run ``fragment`` against ``source`` as ``user`` and hand back a bounded answer.

        Returns:
            The summary, the handle and the receipt.

        Raises:
            GovernorRefusal: the breaker is open, a slot was not free in time, the quota is spent,
                or the deadline cancelled the stream. Each carries the ``ErrorResponse`` to return.
            KeyError: ``source`` is not one of the configured adapters.
        """
        adapter = self._adapters[source]
        self._breaker.ensure_closed(source)
        drained, waited_s = self._run_slot(
            adapter=adapter, source=source, user=user, fragment=fragment, limits=limits
        )
        # Slots are released by now: storing and summarising is our CPU work, not the source's.
        handle = self._store.put(owner=user, source=source, table=drained.table)
        summary = summarise(
            drained.table,
            sample_rows=sample_rows,
            page_rows=page_rows,
            budget_tokens=budget_tokens,
            handle=handle.handle,
        )
        peak, limit = self._source_peak(source)
        return ExecutionOutcome(
            summary=summary,
            handle=handle,
            report=GovernorReport(
                source=source,
                user=user,
                waited_s=waited_s,
                row_cap_hit=drained.row_cap_hit,
                byte_cap_hit=drained.byte_cap_hit,
                rows=drained.table.num_rows,
                nbytes=drained.table.nbytes,
                source_peak=peak,
                source_limit=limit,
            ),
        )

    def get_more(self, handle: str, *, user: str, offset: int, limit: int) -> Page:
        """Page a stored result for its owner; see :meth:`ResultStore.get_more`."""
        return self._store.get_more(handle, owner=user, offset=offset, limit=limit)

    def governance(self) -> Governance:
        """Every protection's current state, sorted, for the audit view and for tests."""
        return Governance(
            peaks=self._admission.peaks(),
            breakers=tuple(self._breaker.report(name) for name in sorted(self._adapters)),
        )

    def _run_slot(
        self,
        *,
        adapter: SourceAdapter,
        source: str,
        user: str,
        fragment: Fragment,
        limits: Limits,
    ) -> tuple[_Drained, float]:
        """Hold the slots, spend the quota, drain under the deadline, and classify the outcome."""
        with self._admission.hold(source=source, user=user) as lease:
            if self._quota is not None:
                self._quota.consume(user)
            try:
                drained = self._drain(
                    adapter=adapter, source=source, fragment=fragment, limits=limits
                )
            except GovernorRefusal:
                # A timeout is the source's failure: count it, so a chronically slow source opens.
                self._breaker.record_failure(source)
                raise
            except Exception as exc:
                self._breaker.record_failure(source)
                raise GovernorRefusal(
                    from_exception(exc, source=source, timeout_s=limits.timeout_s)
                ) from exc
            self._breaker.record_success(source)
            return drained, lease.waited_s

    def _drain(
        self,
        *,
        adapter: SourceAdapter,
        source: str,
        fragment: Fragment,
        limits: Limits,
    ) -> _Drained:
        """Consume the stream batch by batch, cancelling on the deadline.

        Raises:
            GovernorRefusal: the deadline passed; the adapter's ``cancel`` was called first.
        """
        deadline = self._clock.monotonic() + limits.timeout_s
        stream: ArrowStream = adapter.execute(fragment, limits)
        schema = stream.schema
        batches: list[pa.RecordBatch] = []
        rows = 0
        nbytes = 0
        iterator = iter(stream)
        try:
            while True:
                if self._clock.monotonic() > deadline:
                    # The driver's own stop path, not an abandoned thread.
                    adapter.cancel(stream)
                    raise GovernorRefusal(
                        timeout_error(seconds=limits.timeout_s, target=source)
                    )
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                batches.append(batch)
                rows += batch.num_rows
                nbytes += batch.nbytes
        finally:
            stream.close()
        return _Drained(
            table=pa.Table.from_batches(batches, schema=schema),
            row_cap_hit=rows >= limits.max_rows,
            byte_cap_hit=nbytes >= limits.max_bytes,
        )

    def _source_peak(self, source: str) -> tuple[int, int]:
        """This source's high-water mark and its limit, for the receipt."""
        for peak in self._admission.peaks():
            if peak.scope is Scope.SOURCE and peak.name == source:
                return peak.peak, peak.limit
        return 0, self._admission.policy.per_source
