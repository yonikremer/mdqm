"""Tests for the token-bucket rate limiter (A05).

Every test drives a :class:`VirtualClock` instead of waiting: ``sleep`` records the delay and moves
the clock forward, so the assertions are on fake seconds and the whole file runs in milliseconds.
The two thread tests are deterministic too — one freezes the clock so the token arithmetic is
exact, the other checks the metering bound over the completion times.
"""

from __future__ import annotations

import random
import threading
from datetime import UTC, datetime, timedelta
from typing import Final

import pytest

from mdqm.config.settings import RateLimitSettings, Settings, SourceConfig
from mdqm.errors import ErrorCode
from mdqm.ratelimit import (
    DEFAULT_POOL,
    BucketState,
    Decision,
    DecisionReason,
    LimitPolicy,
    RateLimiter,
    SystemTime,
    denial_error,
    limits_for_sources,
    parse_retry_after,
)

#: Tolerance for float time comparisons on one scale (rps 10 gives 0.1 s waits).
EPS: Final[float] = 1e-9


class VirtualClock:
    """A ``TimeSource`` under test control: sleeping advances virtual time instead of waiting.

    ``advance=False`` freezes the clock, which makes the token arithmetic of a concurrent burst
    exact and is how the lost-update test works.
    """

    def __init__(self, start: float = 0.0, *, advance: bool = True) -> None:
        self._lock = threading.Lock()
        self._now = start
        self._advance = advance
        self._sleeps: list[float] = []

    def now(self) -> float:
        with self._lock:
            return self._now

    def sleep(self, seconds: float) -> None:
        if seconds <= 0.0:
            return
        with self._lock:
            self._sleeps.append(seconds)
            if self._advance:
                self._now += seconds

    def recorded(self) -> tuple[float, ...]:
        """Every positive sleep, in call order."""
        with self._lock:
            return tuple(self._sleeps)

    @property
    def current(self) -> float:
        return self.now()


def source(name: str, *, rps: float | None = None, shared: bool = True) -> SourceConfig:
    """A config source: no rate limit when ``rps`` is None, otherwise a ``RateLimitSettings``."""
    rate = None if rps is None else RateLimitSettings(shared_pool=shared, rps=rps)
    return SourceConfig(name=name, adapter="openapi", rate_limit=rate)


def limiter_for(
    policies: dict[str, LimitPolicy],
    sources: dict[str, tuple[str, ...]],
    clock: VirtualClock,
    *,
    seed: int = 0,
) -> RateLimiter:
    return RateLimiter(
        policies,
        sources,
        time_source=clock,
        rng=random.Random(seed),  # noqa: S311 - jitter is not a security decision
    )


# --- acceptance 1: 100 requests at rps=2 take 50 s of fake time ------------------------------


def test_acceptance_100_requests_at_2_rps_take_50_seconds() -> None:
    clock = VirtualClock()
    limiter = limiter_for({"p": LimitPolicy(rps=2)}, {"api": ("p",)}, clock)

    decisions = [limiter.acquire("api") for _ in range(100)]

    assert all(decision.granted for decision in decisions)
    # The bucket starts empty, so even the first request pays one interval.
    assert all(decision.reason is DecisionReason.THROTTLED for decision in decisions)
    assert [decision.wait_s for decision in decisions] == [pytest.approx(0.5)] * 100
    assert clock.current == pytest.approx(50.0, abs=EPS)
    assert sum(clock.recorded()) == pytest.approx(50.0, abs=EPS)


@pytest.mark.parametrize(
    ("rps", "count", "seconds"),
    [
        (2.0, 100, 50.0),
        (1.0, 10, 10.0),
        (4.0, 10, 2.5),
        (0.5, 4, 8.0),
        (10.0, 7, 0.7),
        (2.0, 1, 0.5),
    ],
)
def test_the_wait_is_requests_over_rate_for_any_policy(
    rps: float, count: int, seconds: float
) -> None:
    clock = VirtualClock()
    limiter = limiter_for({"p": LimitPolicy(rps=rps)}, {"api": ("p",)}, clock)

    for _ in range(count):
        limiter.acquire("api")

    assert clock.current == pytest.approx(seconds, abs=1e-6)


def test_a_burst_banks_tokens_but_never_past_its_size() -> None:
    clock = VirtualClock()
    limiter = limiter_for({"p": LimitPolicy(rps=2, burst=3)}, {"api": ("p",)}, clock)

    # 1 s of idle time at burst=3 banks 2 tokens, capped at 3: the third call is free, the
    # fourth pays one interval. reserve() does not sleep, so the debt keeps growing.
    clock.sleep(1.0)
    decisions = [limiter.reserve("api") for _ in range(4)]

    assert [round(d.wait_s, 6) for d in decisions] == [0.0, 0.0, 0.5, 1.0]
    assert limiter.snapshot()[0].tokens == pytest.approx(-2.0)


# --- acceptance 2: two sources sharing a pool never exceed the pool total ---------------------


def test_acceptance_two_sources_sharing_a_pool_never_exceed_the_total() -> None:
    clock = VirtualClock()
    limiter = limiter_for(
        {"shared": LimitPolicy(rps=2)}, {"a": ("shared",), "b": ("shared",)}, clock
    )

    stamps: list[float] = []
    for index in range(40):
        limiter.acquire("a" if index % 2 == 0 else "b")
        stamps.append(clock.current)

    # Together the two sources get exactly 2 requests per second: the k-th call lands at k/2.
    for position, stamp in enumerate(stamps, start=1):
        assert stamp == pytest.approx(position / 2, abs=EPS)
    assert stamps[-1] == pytest.approx(20.0, abs=EPS)
    state = limiter.snapshot()[0]
    assert state.name == "shared"
    assert state.granted == 40
    # The token the last call took is not re-earned until the next access, so the bucket is one
    # in flight: no free token was handed out beyond the rate.
    assert state.tokens == pytest.approx(-1.0)


def test_a_per_source_sub_limit_binds_below_a_generous_pool() -> None:
    clock = VirtualClock()
    limiter = limiter_for(
        {"shared": LimitPolicy(rps=10), "source:a": LimitPolicy(rps=1)},
        {"a": ("shared", "source:a")},
        clock,
    )

    for _ in range(5):
        limiter.acquire("a")

    # The pool would allow 0.1 s; the source's own 1 rps wins, and the report names it.
    assert clock.current == pytest.approx(5.0, abs=EPS)
    decision = limiter.reserve("a")
    assert decision.pool == "source:a"
    assert decision.limit_rps == pytest.approx(1.0)


def test_shared_pool_false_opt_out_meters_the_source_on_its_own() -> None:
    clock = VirtualClock()
    policies, members = limits_for_sources(
        [source("a", rps=2, shared=False), source("b", rps=2, shared=False)]
    )
    assert policies["source:a"].rps == pytest.approx(2.0)
    assert members == {"a": ("source:a",), "b": ("source:b",)}
    assert DEFAULT_POOL not in policies

    limiter = limiter_for(policies, members, clock)
    for _ in range(20):
        limiter.acquire("a" if len(clock.recorded()) % 2 == 0 else "b")

    # Each source keeps its own 2 rps, so 10 calls each finish in 5 s, not in 10 s.
    assert clock.current == pytest.approx(5.0, abs=EPS)


# --- acceptance 3: a 429 with Retry-After pauses every member of the pool ---------------------


def test_acceptance_429_pauses_all_members_of_the_pool() -> None:
    clock = VirtualClock()
    limiter = limiter_for(
        {"shared": LimitPolicy(rps=10), "other": LimitPolicy(rps=10)},
        {"a": ("shared",), "b": ("shared",), "c": ("other",)},
        clock,
    )
    limiter.acquire("a")  # t = 0.1

    paused = limiter.report_429("a", retry_after=30)

    assert paused == ("shared",)
    for member in ("a", "b"):
        decision = limiter.reserve(member)
        assert decision.reason is DecisionReason.RETRY_AFTER
        assert decision.wait_s == pytest.approx(30.0, abs=EPS)
        assert decision.retry_after_s == pytest.approx(30.0, abs=EPS)
    # A source in another pool is untouched by the pause; it banked a token while the first pool
    # was served, so it goes straight through.
    stranger = limiter.reserve("c")
    assert stranger.reason is DecisionReason.OK
    assert stranger.wait_s == 0.0
    assert stranger.retry_after_s is None
    # reserve() never sleeps; acquire() does, for the whole pause.
    assert clock.current == pytest.approx(0.1, abs=EPS)
    limiter.acquire("b")
    assert clock.current == pytest.approx(30.1, abs=EPS)


def test_the_pause_freezes_refill_so_the_debt_survives_it() -> None:
    clock = VirtualClock()
    limiter = limiter_for({"p": LimitPolicy(rps=1)}, {"api": ("p",)}, clock)
    for _ in range(3):
        limiter.acquire("api")  # t = 3, one token owed

    limiter.report_429("api", retry_after=5)
    waits = [limiter.acquire("api").wait_s for _ in range(2)]

    # Five calls need 5 s of credit; 3 s were spent before the 429 and the pause credits none, so
    # the fifth call lands at 10 s. The pause refunds nothing and freezes the refill.
    assert waits == [pytest.approx(5.0), pytest.approx(2.0)]
    assert clock.current == pytest.approx(10.0, abs=EPS)


def test_a_429_without_retry_after_pauses_one_burst_window() -> None:
    clock = VirtualClock()
    limiter = limiter_for({"p": LimitPolicy(rps=10, burst=3)}, {"api": ("p",)}, clock)

    limiter.report_429("api")

    decision = limiter.reserve("api")
    assert decision.reason is DecisionReason.RETRY_AFTER
    assert decision.wait_s == pytest.approx(0.3, abs=EPS)  # burst / rps


def test_reporting_a_429_for_an_unmetered_source_pauses_nothing() -> None:
    clock = VirtualClock()
    limiter = limiter_for({"p": LimitPolicy(rps=1)}, {"api": ("p",)}, clock)

    assert limiter.report_429("database", retry_after=60) == ()
    assert limiter.snapshot()[0].pause_remaining_s == 0.0


# --- denial, jitter, determinism -------------------------------------------------------------


def test_a_wait_above_the_cap_is_denied_and_consumes_nothing() -> None:
    clock = VirtualClock()
    limiter = limiter_for({"p": LimitPolicy(rps=0.5, max_wait_s=1.0)}, {"api": ("p",)}, clock)
    before = limiter.snapshot()[0]

    decision = limiter.acquire("api")

    assert decision.granted is False
    assert decision.reason is DecisionReason.DENIED
    assert decision.wait_s == pytest.approx(2.0, abs=EPS)
    assert decision.pool == "p"
    assert clock.recorded() == ()  # a denial never blocks
    after = limiter.snapshot()[0]
    assert after.tokens == before.tokens  # refunded
    assert (after.granted, after.denied) == (0, 1)

    error = denial_error(decision)
    assert error.code == ErrorCode.QUOTA_EXCEEDED.value
    assert '"p"' in error.problem
    assert "2.0s" in error.fix


@pytest.mark.parametrize(("rps", "max_wait_s", "granted"), [(0.5, 2.0, True), (0.5, 1.9, False)])
def test_the_cap_is_inclusive(rps: float, max_wait_s: float, granted: bool) -> None:
    clock = VirtualClock()
    limiter = limiter_for(
        {"p": LimitPolicy(rps=rps, max_wait_s=max_wait_s)}, {"api": ("p",)}, clock
    )

    assert limiter.reserve("api").granted is granted


def test_denial_error_refuses_a_granted_decision() -> None:
    clock = VirtualClock()
    limiter = limiter_for({"p": LimitPolicy(rps=1)}, {"api": ("p",)}, clock)

    with pytest.raises(ValueError, match="granted"):
        denial_error(limiter.reserve("api"))


def test_jitter_is_bounded_seeded_and_never_breaks_the_rate() -> None:
    def run(seed: int) -> tuple[list[Decision], float]:
        clock = VirtualClock()
        limiter = limiter_for(
            {"p": LimitPolicy(rps=2, jitter_s=0.5)}, {"api": ("p",)}, clock, seed=seed
        )
        decisions = [limiter.acquire("api") for _ in range(10)]
        return decisions, clock.current

    first, elapsed_a = run(7)
    second, elapsed_b = run(7)
    third, _ = run(8)

    assert all(0.0 <= decision.jitter_s <= 0.5 for decision in first)
    assert any(decision.jitter_s > 0.0 for decision in first)
    assert [decision.jitter_s for decision in first] != [decision.jitter_s for decision in third]
    assert [decision.model_dump() for decision in first] == [
        decision.model_dump() for decision in second
    ]
    assert elapsed_a == pytest.approx(elapsed_b, abs=EPS)
    # The metered wait is 5 s; jitter only ever adds, and never more than its bound.
    assert 5.0 <= elapsed_a <= 5.0 + 10 * 0.5


def test_the_same_schedule_twice_gives_identical_decisions() -> None:
    def run() -> tuple[list[dict[str, object]], float]:
        clock = VirtualClock()
        policies, members = limits_for_sources(
            [source("a", rps=2), source("b", rps=5), source("db")]
        )
        limiter = limiter_for(policies, members, clock, seed=11)
        plan = ["a", "b", "a", "db", "b", "a"] * 10
        decisions = [limiter.acquire(name) for name in plan]
        limiter.report_429("b", retry_after=3)
        decisions += [limiter.acquire(name) for name in ("a", "b", "db")]
        return [decision.model_dump() for decision in decisions], clock.current

    first, elapsed_a = run()
    second, elapsed_b = run()

    assert first == second
    assert elapsed_a == elapsed_b


# --- config ---------------------------------------------------------------------------------


def test_the_shared_pool_total_defaults_to_the_smallest_member_rate() -> None:
    policies, members = limits_for_sources([source("a", rps=5), source("b", rps=2)])

    assert policies[DEFAULT_POOL].rps == pytest.approx(2.0)
    assert members == {
        "a": (DEFAULT_POOL, "source:a"),
        "b": (DEFAULT_POOL, "source:b"),
    }
    assert policies["source:a"].rps == pytest.approx(5.0)


def test_an_operator_supplied_pool_total_wins() -> None:
    policies, _ = limits_for_sources([source("a", rps=5), source("b", rps=2)], shared_pool_rps=10)

    assert policies[DEFAULT_POOL].rps == pytest.approx(10.0)


def test_a_source_without_rate_limit_is_absent_and_unmetered() -> None:
    policies, members = limits_for_sources([source("db"), source("api", rps=3)])

    assert members == {"api": (DEFAULT_POOL, "source:api")}
    clock = VirtualClock()
    limiter = limiter_for(policies, members, clock)

    decision = limiter.acquire("db")

    assert decision.reason is DecisionReason.UNLIMITED
    assert decision.granted is True
    assert decision.wait_s == 0.0
    assert clock.recorded() == ()
    assert limiter.members("db") == ()


def test_from_settings_meters_the_configured_api_sources() -> None:
    settings = Settings(
        sources=[
            SourceConfig(name="crm_api", adapter="openapi", rate_limit=RateLimitSettings(rps=2)),
            SourceConfig(
                name="billing_api", adapter="openapi", rate_limit=RateLimitSettings(rps=2)
            ),
            SourceConfig(name="crm_db", adapter="postgresql"),
        ]
    )
    clock = VirtualClock()
    limiter = RateLimiter.from_settings(settings, jitter_s=0.0, time_source=clock)

    assert limiter.members("crm_api") == (DEFAULT_POOL, "source:crm_api")
    assert limiter.members("crm_db") == ()
    for index in range(100):
        limiter.acquire("crm_api" if index % 2 == 0 else "billing_api")

    assert clock.current == pytest.approx(50.0, abs=EPS)


def test_from_settings_honours_the_shared_pool_opt_out() -> None:
    settings = Settings(
        sources=[
            SourceConfig(
                name="crm_api",
                adapter="openapi",
                rate_limit=RateLimitSettings(shared_pool=False, rps=4),
            )
        ]
    )
    clock = VirtualClock()
    limiter = RateLimiter.from_settings(settings, time_source=clock)

    assert limiter.members("crm_api") == ("source:crm_api",)
    for _ in range(4):
        limiter.acquire("crm_api")
    assert clock.current == pytest.approx(1.0, abs=EPS)


def test_a_source_naming_an_undefined_pool_fails_loudly() -> None:
    clock = VirtualClock()

    with pytest.raises(ValueError, match="undefined pool"):
        limiter_for({"p": LimitPolicy(rps=1)}, {"api": ("nope",)}, clock)


def test_snapshot_is_sorted_and_reports_the_counters() -> None:
    clock = VirtualClock()
    limiter = limiter_for(
        {"b": LimitPolicy(rps=1), "a": LimitPolicy(rps=1)},
        {"api": ("b", "a")},
        clock,
    )
    limiter.acquire("api")

    states = limiter.snapshot()

    assert isinstance(states[0], BucketState)
    assert [state.name for state in states] == ["a", "b"]
    assert all(state.granted == 1 for state in states)
    assert all(state.tokens == pytest.approx(-1.0) for state in states)


# --- Retry-After parsing ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("5", 5.0),
        ("0", 0.0),
        ("-3", 0.0),
        ("1.5", 1.5),
        (" 7 ", 7.0),
        ("soon", None),
        ("inf", None),
        ("nan", None),
    ],
)
def test_parse_retry_after_reads_delta_seconds(value: str | None, expected: float | None) -> None:
    assert parse_retry_after(value) == expected


@pytest.mark.parametrize(
    ("value", "offset", "expected"),
    [
        ("Wed, 21 Oct 2015 07:28:00 GMT", -30, 30.0),
        ("Wed, 21 Oct 2015 07:28:00 GMT", 30, 0.0),
        ("wed, 21 oct 2015 07:28:00 GMT", 0, 0.0),
        ("not a date", 0, None),
        ("Wed, 21 Oct 2015 07:28:00", 0, None),
    ],
)
def test_parse_retry_after_reads_an_http_date(
    value: str, offset: int, expected: float | None
) -> None:
    now = datetime(2015, 10, 21, 7, 28, tzinfo=UTC) + timedelta(seconds=offset)

    result = parse_retry_after(value, now=now)

    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected, abs=1.0)


def test_parse_retry_after_rejects_a_naive_now() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", now=datetime(2015, 10, 21, 7, 28))


# --- thread safety (the server runs adapters in a thread pool) --------------------------------


def test_no_token_is_lost_when_threads_race_on_one_pool() -> None:
    clock = VirtualClock(advance=False)
    limiter = limiter_for({"p": LimitPolicy(rps=1)}, {"api": ("p",)}, clock)
    threads, per_thread = 8, 10

    def worker() -> None:
        for _ in range(per_thread):
            limiter.acquire("api")

    pool = [threading.Thread(target=worker) for _ in range(threads)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join(10.0)

    assert not any(thread.is_alive() for thread in pool)
    state = limiter.snapshot()[0]
    # Exactly one token per call, no increments lost: the frozen clock credits nothing.
    assert state.granted == threads * per_thread
    assert state.tokens == pytest.approx(-float(threads * per_thread))


def test_concurrent_callers_keep_the_metering_bound() -> None:
    clock = VirtualClock()
    limiter = limiter_for({"p": LimitPolicy(rps=1)}, {"api": ("p",)}, clock)
    stamps: list[float] = []
    stamps_lock = threading.Lock()

    def worker() -> None:
        for _ in range(5):
            limiter.acquire("api")
            with stamps_lock:
                stamps.append(clock.now())

    pool = [threading.Thread(target=worker) for _ in range(4)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join(10.0)

    assert len(stamps) == 20
    assert max(stamps) <= 20.0 + 1e-6  # the serial bound still holds
    for stamp in stamps:
        completed = sum(1 for other in stamps if other <= stamp + 1e-9)
        assert completed <= 1 + stamp + 1e-6  # burst + rps * elapsed


# --- the async face and the production clock -------------------------------------------------


#: The Windows proactor loop builds its self-pipe from a socketpair, so the async test needs
#: sockets enabled; it never connects anywhere (see AGENTS.md section 6 on loopback opt-in).
@pytest.mark.enable_socket
async def test_acquire_async_moves_the_wait_to_a_worker_thread() -> None:
    clock = VirtualClock()
    limiter = limiter_for({"p": LimitPolicy(rps=2)}, {"api": ("p",)}, clock)

    decision = await limiter.acquire_async("api")

    assert decision.granted is True
    assert decision.wait_s == pytest.approx(0.5, abs=EPS)
    assert clock.current == pytest.approx(0.5, abs=EPS)


def test_system_time_wraps_the_monotonic_clock() -> None:
    clock = SystemTime()

    first = clock.now()
    clock.sleep(0.0)  # a non-positive sleep returns at once
    clock.sleep(-1.0)
    second = clock.now()

    assert second >= first
