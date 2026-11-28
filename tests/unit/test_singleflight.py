"""Tests for single-flight coalescing (task C13).

Two things are under test. :func:`~mdqm.cache.singleflight.flight_key` decides what "identical"
means; :class:`~mdqm.cache.singleflight.SingleFlight` turns equal keys into one execution. Both are
driven without sleeping: interleavings come from ``asyncio.Barrier``, ``asyncio.Event`` and an
injected :class:`FixedClock`, because a concurrency test that depends on timing is a bug here.

``asyncio.wait_for`` appears only as a hang guard (5 s): if the implementation deadlocks, the test
fails instead of blocking the suite. No test waits for time to pass.

``pytest.mark.enable_socket`` is required for the same reason as in ``test_server_skeleton``: on
Windows every asyncio event loop's self-pipe is a ``socket.socketpair()``, which pytest-socket
blocks in ``tests/unit``. No test here opens a network connection — the only "source" is the
in-process DuckDB fake.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import TypeVar

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mdqm.cache.singleflight import KEY_VERSION, SingleFlight, flight_key
from mdqm.models import Fragment, Limits
from mdqm.testing.fake_adapter import FakeAdapter
from mdqm.testing.world import WorldConfig, build_world

pytestmark = pytest.mark.enable_socket

_T = TypeVar("_T")

EXEC_TS = datetime(2026, 3, 1, tzinfo=UTC)
SQL = "SELECT cust_id, status FROM customers_crm ORDER BY cust_id"
HANG_GUARD_S = 5.0


class FixedClock:
    """Time under test control: each call to ``now`` can be told what to return."""

    def __init__(self, now: datetime) -> None:
        self.current = now
        self.reads = 0

    def now(self) -> datetime:
        self.reads += 1
        return self.current


@dataclass(frozen=True, slots=True)
class Segment:
    """A stand-in for one fetched result: the ``fetched_at`` makes coalescing visible."""

    fetched_at: datetime
    rows: int = 1


async def within(awaitable: Awaitable[_T], seconds: float = HANG_GUARD_S) -> _T:
    """Await with a hang guard, so a deadlock fails the test instead of stalling the suite."""
    return await asyncio.wait_for(awaitable, timeout=seconds)


# --- flight_key: what counts as the same query ----------------------------------------------

BASE: dict[str, object] = {
    "source": "crm",
    "query": SQL,
    "params": {"status": "A"},
    "exec_ts": EXEC_TS,
}


def key(**overrides: object) -> str:
    merged = dict(BASE)
    merged.update(overrides)
    return flight_key(**merged)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("label", "left", "right", "same"),
    [
        ("identical", {}, {}, True),
        ("other source", {}, {"source": "billing"}, False),
        ("other query", {}, {"query": SQL + " LIMIT 1"}, False),
        ("query whitespace", {"query": "SELECT 1"}, {"query": "SELECT  1"}, False),
        ("query case", {"query": "SELECT 1"}, {"query": "select 1"}, False),
        ("no params vs params", {"params": None}, {}, False),
        ("other param value", {"params": {"status": "C"}}, {}, False),
        ("extra param", {"params": {"status": "A", "x": 1}}, {}, False),
        ("param order", {"params": {"a": 1, "b": 2}}, {"params": {"b": 2, "a": 1}}, True),
        (
            "nested param order",
            {"params": {"f": {"y": 1, "x": 2}}},
            {"params": {"f": {"x": 2, "y": 1}}},
            True,
        ),
        ("list order", {"params": {"k": [1, 2]}}, {"params": {"k": [2, 1]}}, False),
        ("tuple equals list", {"params": {"k": (1, 2)}}, {"params": {"k": [1, 2]}}, True),
        ("no exec_ts vs exec_ts", {"exec_ts": None}, {}, False),
        ("exec_ts later", {}, {"exec_ts": EXEC_TS + timedelta(seconds=1)}, False),
        (
            "exec_ts same instant, other zone",
            {},
            {"exec_ts": EXEC_TS.astimezone(timezone(timedelta(hours=2)))},
            True,
        ),
        (
            "exec_ts west zone",
            {},
            {"exec_ts": EXEC_TS.astimezone(timezone(timedelta(hours=-5)))},
            True,
        ),
        ("Decimal vs float", {"params": {"k": Decimal("1.5")}}, {"params": {"k": 1.5}}, False),
        ("Decimal vs int", {"params": {"k": Decimal("1")}}, {"params": {"k": 1}}, False),
        ("True vs 1", {"params": {"k": True}}, {"params": {"k": 1}}, False),
        ("None param value", {"params": {"k": None}}, {"params": {}}, False),
        ("bytes vs str", {"params": {"k": b"ab"}}, {"params": {"k": "ab"}}, False),
        (
            "date vs datetime",
            {"params": {"k": datetime(2026, 3, 1, tzinfo=UTC)}},
            {"params": {"k": datetime(2026, 3, 1, tzinfo=UTC).date()}},
            False,
        ),
    ],
)
def test_key_identity_table(
    label: str, left: dict[str, object], right: dict[str, object], same: bool
) -> None:
    """Table of the cases that must share a key and the ones that must not."""
    left_key = key(**left)
    right_key = key(**right)
    if same:
        assert left_key == right_key, label
    else:
        assert left_key != right_key, label


def test_key_is_versioned_and_opaque() -> None:
    """The key is a versioned digest: readable provenance lives in the result, never in the key."""
    value = key()
    assert value.startswith(f"{KEY_VERSION}:")
    digest = value.split(":", 1)[1]
    assert len(digest) == 64
    assert SQL not in value
    assert set(digest) <= set("0123456789abcdef")


def test_key_matches_a_frozen_golden_digest() -> None:
    """Pins the exact bytes hashed.

    A literal digest is the cross-process stability check: any dependence on ``hash()``,
    ``id()`` or dict order would make this literal different on every run (PYTHONHASHSEED), and any
    silent change to the canonicalisation would break it deterministically.
    """
    assert key() == "v1:af6de3af1e3a5a1588392ad3448c10925def5f6c6c2e014f9bf0b925b79f51d1"


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"source": "", "query": SQL}, ValueError, "source"),
        ({"source": "crm", "query": ""}, ValueError, "query"),
        ({"source": "crm", "query": SQL, "exec_ts": datetime(2026, 3, 1)}, ValueError, "naive"),
        (
            {"source": "crm", "query": SQL, "params": {"k": datetime(2026, 3, 1)}},
            ValueError,
            "timezone-aware",
        ),
        ({"source": "crm", "query": SQL, "params": {1: "a"}}, TypeError, "str"),
        ({"source": "crm", "query": SQL, "params": {"k": object()}}, TypeError, "fingerprint"),
    ],
)
def test_key_rejects_ambiguous_input(
    kwargs: dict[str, object], error: type[Exception], match: str
) -> None:
    """A key we cannot build faithfully must raise, never fall back to a lossy identity."""
    with pytest.raises(error, match=match):
        flight_key(**kwargs)  # type: ignore[arg-type]


# --- flight_key: properties -----------------------------------------------------------------

_TEXT = st.text(min_size=1, max_size=40)
_PARAM_VALUE = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(),
    st.lists(st.integers(), max_size=3),
)
_PARAMS = st.dictionaries(st.text(min_size=1, max_size=8), _PARAM_VALUE, max_size=4)


@given(source=_TEXT, query=_TEXT, params=_PARAMS, offset_minutes=st.integers(-720, 720))
def test_property_equal_inputs_give_equal_key(
    source: str, query: str, params: dict[str, object], offset_minutes: int
) -> None:
    """Equal inputs always produce one key, whatever the parameter insertion order."""
    exec_ts = EXEC_TS.astimezone(timezone(timedelta(minutes=offset_minutes)))
    first = flight_key(source=source, query=query, params=params, exec_ts=exec_ts)
    reordered = dict(reversed(list(params.items())))
    second = flight_key(source=source, query=query, params=reordered, exec_ts=EXEC_TS)
    assert first == second


@given(query=_TEXT, other=_TEXT, source=_TEXT)
def test_property_a_different_question_gets_a_different_key(
    query: str, other: str, source: str
) -> None:
    """Distinct queries never collide, and neither do distinct sources."""
    if query != other:
        assert flight_key(source=source, query=query) != flight_key(source=source, query=other)
    assert flight_key(source=source, query=query) != flight_key(source=source + "x", query=query)


@given(seconds=st.integers(-10 * 365 * 24 * 3600, 10 * 365 * 24 * 3600))
def test_property_exec_ts_key_depends_only_on_the_instant(seconds: int) -> None:
    """The zone an ``exec_ts`` was written in is not part of the identity; the instant is."""
    instant = EXEC_TS + timedelta(seconds=seconds)
    zones = [UTC, timezone(timedelta(hours=2)), timezone(timedelta(hours=-9))]
    keys = {flight_key(source="crm", query=SQL, exec_ts=instant.astimezone(z)) for z in zones}
    assert len(keys) == 1


# --- SingleFlight: one execution for many callers --------------------------------------------


async def test_100_concurrent_identical_calls_execute_once() -> None:
    """Acceptance 1: 100 identical concurrent calls cost exactly one execution."""
    callers = 100
    clock = FixedClock(EXEC_TS)
    flight = SingleFlight()
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()
    barrier = asyncio.Barrier(callers)
    key_ = flight_key(source="crm", query=SQL, exec_ts=EXEC_TS)

    async def work() -> Segment:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return Segment(fetched_at=clock.now())

    async def caller() -> Segment:
        # The barrier makes the interleaving deterministic: every caller is past it before the
        # first one can start the work, so all 100 really are concurrent.
        await barrier.wait()
        return await flight.do(key_, work)

    async def run() -> list[Segment]:
        tasks = [asyncio.create_task(caller()) for _ in range(callers)]
        await within(started.wait())
        release.set()
        return await asyncio.gather(*tasks)

    results = await within(run())
    assert isinstance(results, list)
    assert calls == 1, "the source was called more than once"
    assert len(results) == callers
    assert len({id(result) for result in results}) == 1, "callers got different result objects"
    assert {result.fetched_at for result in results} == {EXEC_TS}
    assert clock.reads == 1, "each caller must see the leader's fetched_at, not its own"
    assert len(flight) == 0, "the key must be gone once the call finished"


async def test_a_later_call_runs_again() -> None:
    """The key is dropped on success, so a sequential repeat is a fresh call, not a stale hit."""
    flight = SingleFlight()
    calls: list[int] = []

    async def work() -> Segment:
        calls.append(1)
        return Segment(fetched_at=EXEC_TS, rows=len(calls))

    async def run() -> tuple[Segment, Segment]:
        return await flight.do("k", work), await flight.do("k", work)

    first, second = await within(run())
    assert len(calls) == 2
    assert first.rows == 1
    assert second.rows == 2
    assert len(flight) == 0


async def test_exception_reaches_every_waiter_and_the_next_call_retries() -> None:
    """Acceptance 2: the failure is shared, is not cached, and the next call starts over."""
    flight = SingleFlight()
    attempts = 0
    started = asyncio.Event()
    release = asyncio.Event()
    barrier = asyncio.Barrier(3)

    class SourceDown(RuntimeError):
        pass

    async def work() -> Segment:
        nonlocal attempts
        attempts += 1
        started.set()
        await release.wait()  # park so every waiter really joins this attempt
        raise SourceDown(f"attempt {attempts} failed")

    async def caller() -> Segment:
        await barrier.wait()
        return await flight.do("k", work)

    tasks = [asyncio.create_task(caller()) for _ in range(3)]
    await within(started.wait())
    fourth = asyncio.create_task(flight.do("k", work))  # joins the still-running attempt
    release.set()
    first_round = await within(asyncio.gather(*tasks, fourth, return_exceptions=True))

    assert isinstance(first_round, list)
    assert attempts == 1, "the failure was not shared with the waiters"
    assert [type(outcome) for outcome in first_round] == [SourceDown] * 4
    assert len(flight) == 0, "a failed call must not leave the key behind"

    succeeded = await within(flight.do("k", _segment))

    assert attempts == 1, "the retry must be the caller's own fresh call"
    assert succeeded.fetched_at == EXEC_TS


async def _segment() -> Segment:
    return Segment(fetched_at=EXEC_TS)


async def test_cancelling_one_waiter_leaves_the_work_running() -> None:
    """Acceptance 3: a cancelled waiter neither cancels nor aborts the shared call."""
    flight = SingleFlight()
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()
    arrived = [asyncio.Event(), asyncio.Event()]

    async def work() -> Segment:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return Segment(fetched_at=EXEC_TS, rows=7)

    async def waiter(index: int) -> Segment:
        arrived[index].set()
        return await flight.do("k", work)

    async def run() -> tuple[Segment, Segment]:
        leader = asyncio.create_task(flight.do("k", work))
        await within(started.wait())
        two = asyncio.create_task(waiter(0))
        three = asyncio.create_task(waiter(1))
        # ``set`` schedules the wakeup and the waiter keeps going into ``do`` in the same step, so
        # both waiters are parked on the flight once both events are observed.
        await within(asyncio.gather(arrived[0].wait(), arrived[1].wait()))
        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader
        release.set()
        return await asyncio.gather(two, three)

    survivors = await within(run())
    assert calls == 1
    assert [result.rows for result in survivors] == [7, 7]
    assert len(flight) == 0


async def test_cancelling_every_waiter_still_lets_the_call_finish() -> None:
    """Nobody left to receive the result is not a reason to abandon it; the key still clears."""
    flight = SingleFlight()
    calls = 0
    started = asyncio.Event()
    finished = asyncio.Event()
    release = asyncio.Event()

    async def work() -> Segment:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        finished.set()
        return Segment(fetched_at=EXEC_TS)

    async def run() -> None:
        leader = asyncio.create_task(flight.do("k", work))
        await within(started.wait())
        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader
        release.set()
        await within(finished.wait())
        # One loop turn for the completion callback that drops the key.
        await asyncio.sleep(0)
        assert len(flight) == 0
        assert calls == 1

    await within(run())


async def test_different_keys_run_concurrently() -> None:
    """Independent keys must not queue behind each other: both reach the barrier or it hangs."""
    flight = SingleFlight()
    both = asyncio.Barrier(2)

    async def work(tag: str) -> str:
        await both.wait()  # each call waits for the other: only concurrent execution gets past
        return tag

    async def run() -> list[str]:
        return list(
            await asyncio.gather(
                flight.do("a", lambda: work("a")),
                flight.do("b", lambda: work("b")),
            )
        )

    assert await within(run()) == ["a", "b"]
    assert len(flight) == 0


async def test_one_key_failing_does_not_affect_another() -> None:
    """Coalescing shares the failure with the callers of that key only."""
    flight = SingleFlight()

    async def boom() -> Segment:
        raise RuntimeError("crm is down")

    async def ok() -> Segment:
        return Segment(fetched_at=EXEC_TS, rows=3)

    async def run() -> tuple[object, Segment]:
        failed, fine = await asyncio.gather(
            flight.do("bad", boom), flight.do("good", ok), return_exceptions=True
        )
        assert isinstance(fine, Segment), "the failing key must not take the healthy one with it"
        return failed, fine

    failed, fine = await within(run())
    assert isinstance(failed, RuntimeError)
    assert isinstance(fine, Segment)
    assert fine.rows == 3


async def test_a_function_that_re_enters_its_own_key_is_refused_not_deadlocked() -> None:
    """The one real deadlock (a call waiting for itself) is reported as an error instead."""
    flight = SingleFlight()
    key_ = flight_key(source="crm", query=SQL, exec_ts=EXEC_TS)

    async def nested() -> Segment:
        return await flight.do(key_, _segment)

    with pytest.raises(RuntimeError, match="re-entry"):
        await within(flight.do(key_, nested))
    assert len(flight) == 0


async def test_a_blank_key_is_refused() -> None:
    """An empty key would coalesce unrelated callers, which must never happen silently."""
    flight = SingleFlight()
    with pytest.raises(ValueError, match="key"):
        await flight.do("", _segment)


# --- against a real adapter: one source call for 100 concurrent queries ----------------------

LIMITS = Limits(timeout_s=5.0, max_rows=1000, max_bytes=1_000_000, max_partitions=8)


def _fragment(sql: str) -> Fragment:
    return Fragment(
        source="crm", dialect="duckdb", sql=sql, tables=("customers_crm",), exec_ts=EXEC_TS
    )


def _run_fragment(adapter: FakeAdapter, sql: str) -> int:
    """One blocking source call, the way the server runs an adapter in its pool."""
    stream = adapter.execute(_fragment(sql), LIMITS)
    try:
        return sum(len(batch) for batch in stream)
    finally:
        stream.close()


async def test_100_concurrent_identical_queries_hit_the_source_once() -> None:
    """The acceptance criterion against a real adapter: ``calls`` holds one entry, not 100."""
    adapter = FakeAdapter.for_world(build_world(WorldConfig.small()), name="crm")
    clock = FixedClock(EXEC_TS)
    flight = SingleFlight()
    key_ = flight_key(source="crm", query=SQL, exec_ts=EXEC_TS)
    started = asyncio.Event()
    release = asyncio.Event()
    barrier = asyncio.Barrier(100)

    async def work() -> Segment:
        started.set()
        await release.wait()  # park inside the call so all 100 callers really join
        return Segment(fetched_at=clock.now(), rows=_run_fragment(adapter, SQL))

    async def caller() -> Segment:
        await barrier.wait()
        return await flight.do(key_, work)

    async def run() -> list[Segment]:
        tasks = [asyncio.create_task(caller()) for _ in range(100)]
        await within(started.wait())
        release.set()
        return await asyncio.gather(*tasks)

    results = await within(run())
    assert isinstance(results, list)
    assert adapter.calls == [SQL], "the fake adapter recorded more than one source call"
    assert len({id(result) for result in results}) == 1
    assert {result.fetched_at for result in results} == {EXEC_TS}
    assert results[0].rows > 0
    assert len(flight) == 0


async def test_a_different_exec_ts_is_a_different_call() -> None:
    """The same SQL at a different execution instant must not be served from another flight."""
    flight = SingleFlight()
    calls = 0

    async def work() -> Segment:
        nonlocal calls
        calls += 1
        return Segment(fetched_at=EXEC_TS)

    async def run() -> None:
        await flight.do(flight_key(source="crm", query=SQL, exec_ts=EXEC_TS), work)
        later = EXEC_TS + timedelta(hours=1)
        await flight.do(flight_key(source="crm", query=SQL, exec_ts=later), work)

    await within(run())
    assert calls == 2
