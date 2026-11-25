"""Tests for the governor: concurrency, quotas, timeouts, breaker and result handles (C15).

Every test is deterministic. Time comes from :class:`ScriptedClock` and moves only when a test says
so; concurrency overlap comes from :class:`_Gate`, which holds callers inside the adapter until the
test releases them. Nothing here sleeps to wait for a coincidence, so a failure means a real bug.

The adapter is F05's DuckDB-backed :class:`~mdqm.testing.fake_adapter.FakeAdapter` wrapped by
:class:`HarnessSource`, not a mock: the governor is driven against a real stream that really
cancels, so "the timeout reached the adapter's cancel path" is an observed fact, not a stub's word.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Final

import pyarrow as pa
import pytest

from mdqm.adapters.base import ArrowStream
from mdqm.errors import ErrorCode
from mdqm.governor import (
    Admission,
    AdmissionPolicy,
    BreakerPolicy,
    BreakerState,
    CircuitBreaker,
    Governor,
    GovernorRefusal,
    HandleNotFound,
    Quota,
    QuotaPolicy,
    ResultStore,
    Scope,
)
from mdqm.models import (
    Capabilities,
    CatalogSnapshot,
    Fragment,
    Limits,
    Partitioning,
    PlanSummary,
    RawQuery,
    TableStats,
)
from mdqm.testing.fake_adapter import FakeAdapter

EXEC_TS: Final = datetime(2026, 1, 1, tzinfo=UTC)


# --- deterministic time and concurrency helpers ---------------------------------------------


class ScriptedClock:
    """A :class:`~mdqm.governor.Clock` that moves only when the test moves it."""

    def __init__(self, *, mono: float = 0.0, wall: datetime | None = None) -> None:
        self._mono = mono
        self._wall = wall if wall is not None else EXEC_TS
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        with self._lock:
            return self._mono

    def now(self) -> datetime:
        with self._lock:
            return self._wall

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._mono += seconds
            self._wall += timedelta(seconds=seconds)

    def jump(self, mono: float) -> None:
        with self._lock:
            self._mono = mono


class _Gate:
    """Holds every caller inside the adapter until the test releases them.

    This is what makes the concurrency assertion exact: the governor's slots are the only thing
    that can let a caller through, so the number of callers inside is the number of slots held.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._entered = 0
        self.peak = 0
        self._released = threading.Event()

    def enter(self) -> None:
        with self._cond:
            self._entered += 1
            self.peak = max(self.peak, self._entered)
            self._cond.notify_all()
        self._released.wait()

    def leave(self) -> None:
        with self._cond:
            self._entered -= 1

    def wait_entered(self, count: int, timeout: float = 10.0) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: self._entered >= count, timeout)

    @property
    def inside(self) -> int:
        with self._cond:
            return self._entered

    def release(self) -> None:
        self._released.set()


class _AdvancingStream:
    """A stream that moves the governor's clock past its deadline after the first batch."""

    def __init__(self, inner: ArrowStream, clock: ScriptedClock, jump_to: float) -> None:
        self.schema: pa.Schema = inner.schema
        self._inner = inner
        self._clock = clock
        self._jump_to = jump_to
        self.cancelled = False

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        for index, batch in enumerate(self._inner):
            yield batch
            if index == 0:
                self._clock.jump(self._jump_to)

    def cancel(self) -> None:
        self.cancelled = True
        self._inner.close()

    def close(self) -> None:
        self._inner.close()


class HarnessSource:
    """A real adapter plus the two hooks a deterministic governor test needs.

    ``gate`` holds callers inside ``execute`` so overlap is observable; ``advance_after_first``
    makes the deadline pass mid-drain; ``jump_inside`` makes it pass before the first batch. Every
    other call delegates to a real :class:`FakeAdapter`, so this is not a mock of the adapter
    contract -- it is the real one with a controllable clock and gate.

    ``factory`` exists because the fake holds one DuckDB connection, which is single-threaded by
    design: a load test that runs 50 queries at once builds a fresh adapter per call, so the
    concurrency being measured is the governor's admission, not DuckDB's connection mutex.
    """

    kind: str = "fake"

    def __init__(
        self,
        inner: FakeAdapter | None = None,
        *,
        factory: Callable[[], FakeAdapter] | None = None,
        gate: _Gate | None = None,
        clock: ScriptedClock | None = None,
        jump_inside: float | None = None,
        advance_after_first: float | None = None,
    ) -> None:
        if inner is None and factory is None:
            raise ValueError("HarnessSource needs an inner adapter or a factory")
        self.inner = inner
        self._factory = factory
        self._gate = gate
        self._clock = clock
        self._jump_inside = jump_inside
        self._advance_after_first = advance_after_first
        self.calls: list[str] = []
        self.last_stream: object | None = None

    def _adapter(self) -> FakeAdapter:
        return self._factory() if self._factory is not None else self.inner  # type: ignore[return-value]

    def dialect(self) -> str:
        return self._adapter().dialect()

    def probe(self) -> Capabilities:
        return self._adapter().probe()

    def extract_catalog(self) -> CatalogSnapshot:
        return self._adapter().extract_catalog()

    def execute(self, fragment: Fragment, limits: Limits) -> ArrowStream:
        if self._gate is not None:
            self._gate.enter()
        try:
            if self._clock is not None and self._jump_inside is not None:
                self._clock.jump(self._jump_inside)
            adapter = self._adapter()
            self.calls.append(fragment.sql or "")
            stream = adapter.execute(fragment, limits)
        finally:
            if self._gate is not None:
                self._gate.leave()
        if self._clock is not None and self._advance_after_first is not None:
            wrapped = _AdvancingStream(stream, self._clock, self._advance_after_first)
            self.last_stream = wrapped
            return wrapped
        self.last_stream = stream
        return stream

    def cancel(self, handle: ArrowStream) -> None:
        self._adapter().cancel(handle)

    def extract_partitions(self) -> list[Partitioning]:
        return self._adapter().extract_partitions()

    def explain(self, fragment: Fragment) -> PlanSummary:
        return self._adapter().explain(fragment)

    def mine_queries(self, since: datetime) -> Iterator[RawQuery]:
        return self._adapter().mine_queries(since)

    def mutability_stats(self) -> list[TableStats]:
        return self._adapter().mutability_stats()


def _fragment() -> Fragment:
    return Fragment(
        source="fake", dialect="duckdb", sql="SELECT * FROM t", tables=("t",), exec_ts=EXEC_TS
    )


def _limits(**overrides: object) -> Limits:
    base: dict[str, object] = {
        "timeout_s": 5.0,
        "max_rows": 100,
        "max_bytes": 1_000_000,
        "max_partitions": 8,
    }
    base.update(overrides)
    return Limits(**base)  # type: ignore[arg-type]


def _table(rows: int = 5) -> pa.Table:
    return pa.table({"n": list(range(rows))})


def _source(**kwargs: object) -> HarnessSource:
    return HarnessSource(FakeAdapter(name="fake", tables={"t": _table()}), **kwargs)  # type: ignore[arg-type]


def _governor(
    source: HarnessSource,
    *,
    clock: ScriptedClock,
    policy: AdmissionPolicy | None = None,
    quota: Quota | None = None,
    breaker: CircuitBreaker | None = None,
    store: ResultStore | None = None,
) -> Governor:
    return Governor(
        {"fake": source},
        admission=Admission(policy or AdmissionPolicy(max_wait_s=30.0), clock=clock),
        quota=quota,
        breaker=breaker,
        store=store,
        clock=clock,
    )


# --- acceptance 1: the per-source ceiling holds under load ----------------------------------


def test_fifty_concurrent_queries_never_exceed_per_source_concurrency() -> None:
    """C15 acceptance 1: 50 concurrent queries, and the source never sees more than the limit."""
    clock = ScriptedClock()
    gate = _Gate()
    source = HarnessSource(
        factory=lambda: FakeAdapter(name="fake", tables={"t": _table()}), gate=gate
    )
    admission = Admission(
        AdmissionPolicy(max_global=50, per_user=50, per_source=3, max_wait_s=30.0), clock=clock
    )
    governor = Governor({"fake": source}, admission=admission, clock=clock)

    results: list[object] = []
    failures: list[BaseException] = []
    started = threading.Barrier(50, timeout=10)

    def worker(index: int) -> None:
        try:
            started.wait()
            results.append(
                governor.execute(
                    source="fake", user=f"user{index}", fragment=_fragment(), limits=_limits()
                )
            )
        except BaseException as exc:  # noqa: BLE001 - a thread's failure must reach the test
            failures.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for thread in threads:
        thread.start()

    # Exactly the configured number are inside the source; everyone else is queued on a slot.
    assert gate.wait_entered(3), "the first three callers never reached the source"
    assert gate.inside == 3
    assert gate.peak == 3

    gate.release()
    for thread in threads:
        thread.join(timeout=30)

    assert not failures, f"unexpected failures: {failures}"
    assert len(results) == 50
    # The acceptance assertion: the observed maximum, not the configured one.
    assert gate.peak == 3
    assert len(source.calls) == 50

    peaks = {peak.name: peak for peak in admission.peaks() if peak.scope is Scope.SOURCE}
    assert peaks["fake"].peak == 3
    assert peaks["fake"].limit == 3


def test_global_and_per_user_ceilings_are_enforced() -> None:
    """The other two ceilings bind too, and a refusal names the one that bound it."""
    clock = ScriptedClock()
    admission = Admission(
        AdmissionPolicy(max_global=2, per_user=1, per_source=5, max_wait_s=0.0), clock=clock
    )
    # Two holds for one user: the second must be refused by the per_user ceiling.
    with (
        admission.hold(source="s", user="a"),
        pytest.raises(GovernorRefusal) as caught,
        admission.hold(source="s", user="a"),
    ):
        pass
    assert caught.value.response.code == ErrorCode.QUOTA_EXCEEDED.value
    assert "user" in caught.value.response.problem


# --- acceptance 2: handles are owned ---------------------------------------------------------


def test_user_a_cannot_read_user_bs_handle() -> None:
    """C15 acceptance 2: another user's handle is indistinguishable from one that never existed."""
    clock = ScriptedClock()
    governor = _governor(_source(), clock=clock)
    mine = governor.execute(source="fake", user="alice", fragment=_fragment(), limits=_limits())
    handle = mine.handle.handle

    page = governor.get_more(handle, user="alice", offset=0, limit=2)
    assert len(page.rows) == 2

    with pytest.raises(HandleNotFound) as stolen:
        governor.get_more(handle, user="bob", offset=0, limit=2)
    with pytest.raises(HandleNotFound) as missing:
        governor.get_more("h_does_not_exist", user="bob", offset=0, limit=2)
    # Same message: the store must not reveal that someone else's result exists.
    assert str(stolen.value) == str(missing.value)


def test_handle_pages_and_expires_on_the_injected_clock() -> None:
    """Paging is owner-only and the TTL is driven by the clock, never by sleeping."""
    clock = ScriptedClock()
    store = ResultStore(clock=clock, ttl_s=10.0)
    handle = store.put(owner="alice", source="fake", table=_table(5))

    first = store.get_more(handle.handle, owner="alice", offset=0, limit=2)
    assert [row["n"] for row in first.rows] == [0, 1]
    assert first.rows_total == 5
    assert first.next_offset == 2
    assert first.truncated is True

    last = store.get_more(handle.handle, owner="alice", offset=2, limit=10)
    assert [row["n"] for row in last.rows] == [2, 3, 4]
    assert last.next_offset is None
    assert last.truncated is False

    clock.advance(11)
    with pytest.raises(HandleNotFound):
        store.get_more(handle.handle, owner="alice", offset=0, limit=2)
    assert store.stats().expired == 1


def test_store_eviction_is_counted_not_silent() -> None:
    """The handle cap evicts the oldest and reports it, rather than growing without bound."""
    clock = ScriptedClock()
    store = ResultStore(clock=clock, max_handles=1)
    first = store.put(owner="alice", source="fake", table=_table(1))
    second = store.put(owner="alice", source="fake", table=_table(1))

    with pytest.raises(HandleNotFound):
        store.meta(first.handle, owner="alice")
    assert store.meta(second.handle, owner="alice").handle == second.handle
    assert store.stats().evicted == 1


# --- acceptance 3: a timeout cancels the adapter and frees the slot -------------------------


def test_timeout_cancels_the_adapter_stream_and_releases_the_slot() -> None:
    """C15 acceptance 3: the deadline calls the adapter's cancel path and frees the slot."""
    clock = ScriptedClock()
    # 3 rows at one batch per row gives the drain loop a batch boundary to notice the deadline.
    source = HarnessSource(
        FakeAdapter(name="fake", tables={"t": _table(3)}, batch_size=1),
        clock=clock,
        advance_after_first=1_000_000.0,
    )
    admission = Admission(AdmissionPolicy(max_wait_s=5.0), clock=clock)
    governor = Governor({"fake": source}, admission=admission, clock=clock)

    with pytest.raises(GovernorRefusal) as caught:
        governor.execute(
            source="fake", user="alice", fragment=_fragment(), limits=_limits(timeout_s=1.0)
        )

    response = caught.value.response
    assert response.code == ErrorCode.TIMEOUT.value
    # The limit that fired is named, so the agent knows what to change.
    assert "1s" in response.problem
    assert '"fake"' in response.problem

    # The cancellation really reached the adapter's stream, not just this thread's stack.
    stream = source.last_stream
    assert isinstance(stream, _AdvancingStream)
    assert stream.cancelled is True

    # And the slot came back: nothing is in use, and the next call is admitted.
    assert all(peak.in_use == 0 for peak in admission.peaks())
    recovered = governor.execute(
        source="fake", user="alice", fragment=_fragment(), limits=_limits()
    )
    assert recovered.report.rows == 3


def test_timeout_before_the_first_batch_also_cancels() -> None:
    """A deadline that has already passed when the stream arrives still cancels the stream."""
    clock = ScriptedClock()
    source = HarnessSource(
        FakeAdapter(name="fake", tables={"t": _table(3)}), clock=clock, jump_inside=1_000_000.0
    )
    admission = Admission(AdmissionPolicy(max_wait_s=5.0), clock=clock)
    governor = Governor({"fake": source}, admission=admission, clock=clock)

    with pytest.raises(GovernorRefusal) as caught:
        governor.execute(
            source="fake", user="alice", fragment=_fragment(), limits=_limits(timeout_s=1.0)
        )
    assert caught.value.response.code == ErrorCode.TIMEOUT.value
    assert getattr(source.last_stream, "cancelled", False) is True
    assert all(peak.in_use == 0 for peak in admission.peaks())


# --- acceptance 4: the quota reports its reset time -----------------------------------------


def test_quota_exceeded_names_the_reset_time_in_fix() -> None:
    """C15 acceptance 4: ``QUOTA_EXCEEDED`` with the reset in ``fix``, then usable again."""
    clock = ScriptedClock()
    quota = Quota(QuotaPolicy(max_queries=2, window_s=60.0), clock=clock)
    governor = _governor(_source(), clock=clock, quota=quota)

    for _ in range(2):
        governor.execute(source="fake", user="alice", fragment=_fragment(), limits=_limits())

    with pytest.raises(GovernorRefusal) as caught:
        governor.execute(source="fake", user="alice", fragment=_fragment(), limits=_limits())
    response = caught.value.response
    assert response.code == ErrorCode.QUOTA_EXCEEDED.value
    assert "in 60s" in response.fix

    # Another user has their own window.
    governor.execute(source="fake", user="bob", fragment=_fragment(), limits=_limits())

    clock.advance(61)
    governor.execute(source="fake", user="alice", fragment=_fragment(), limits=_limits())


# --- the circuit breaker --------------------------------------------------------------------


def test_breaker_opens_after_failures_and_stops_calling_the_source() -> None:
    """A failing source is refused before the adapter is touched, then trialled after cooldown."""
    clock = ScriptedClock()
    inner = FakeAdapter(name="fake", tables={"t": _table()}, fail_with=OSError("refused"))
    source = HarnessSource(inner)
    breaker = CircuitBreaker(BreakerPolicy(failure_threshold=2, cooldown_s=30.0), clock=clock)
    governor = _governor(source, clock=clock, breaker=breaker)

    for _ in range(2):
        with pytest.raises(GovernorRefusal) as caught:
            governor.execute(source="fake", user="alice", fragment=_fragment(), limits=_limits())
        assert caught.value.response.code == ErrorCode.SOURCE_UNAVAILABLE.value

    report = breaker.report("fake")
    assert report.state is BreakerState.OPEN
    assert report.opens_in_s == 30.0
    assert report.opened_count == 1

    calls_while_open = len(source.calls)
    with pytest.raises(GovernorRefusal) as caught:
        governor.execute(source="fake", user="alice", fragment=_fragment(), limits=_limits())
    assert caught.value.response.code == ErrorCode.SOURCE_UNAVAILABLE.value
    # The breaker short-circuited: the adapter was not called.
    assert len(source.calls) == calls_while_open

    # After the cooldown exactly one trial is admitted; it fails, so the breaker re-opens.
    clock.advance(31)
    with pytest.raises(GovernorRefusal):
        governor.execute(source="fake", user="alice", fragment=_fragment(), limits=_limits())
    assert len(source.calls) == calls_while_open + 1
    assert breaker.report("fake").state is BreakerState.OPEN


def test_breaker_closes_after_a_successful_trial() -> None:
    """A half-open trial that succeeds closes the breaker and resets the count."""
    clock = ScriptedClock()
    breaker = CircuitBreaker(BreakerPolicy(failure_threshold=1, cooldown_s=10.0), clock=clock)
    breaker.record_failure("fake")
    assert breaker.report("fake").state is BreakerState.OPEN

    clock.advance(11)
    breaker.ensure_closed("fake")  # the single trial
    breaker.record_success("fake")

    report = breaker.report("fake")
    assert report.state is BreakerState.CLOSED
    assert report.consecutive_failures == 0


# --- reporting: clamps and caps are visible -------------------------------------------------


def test_row_cap_is_reported_not_hidden() -> None:
    """A result stopped by the row cap says so, so the agent knows it saw a truncated answer."""
    clock = ScriptedClock()
    governor = _governor(_source(), clock=clock)
    outcome = governor.execute(
        source="fake",
        user="alice",
        fragment=_fragment(),
        limits=_limits(max_rows=2),
    )
    assert outcome.report.row_cap_hit is True
    assert outcome.report.rows == 2
    assert outcome.summary.rows_total == 2
    assert outcome.report.source_limit == 3
    assert outcome.report.waited_s >= 0.0


def test_unknown_source_fails_loudly() -> None:
    """A source the governor was not given is a KeyError here, never a call to a database."""
    clock = ScriptedClock()
    governor = _governor(_source(), clock=clock)
    with pytest.raises(KeyError):
        governor.execute(source="nope", user="alice", fragment=_fragment(), limits=_limits())
