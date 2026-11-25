"""Token-bucket rate limiting for the API sources (A05).

The real budgets of the internal APIs are unknown (``AGENTS.md`` section 9), so the default is the
conservative one: **one pool shared by every API source**, with each source keeping its own
sub-limit on top of it. ``shared_pool: false`` in a source's config is the opt-out, and that source
then meters only against itself. Nothing here guesses an organisation's numbers: the rates come
from :class:`~mdqm.config.settings.RateLimitSettings`.

Three rules the module is built around:

* **Time is injected.** :class:`TimeSource` supplies monotonic seconds and the sleep, so a test
  drives a virtual clock and never waits. The bucket banks no free token at start-up: the first
  request already pays one interval, which is what "100 requests at 2 rps take 50 s" means.
* **Nothing is silent.** Every acquisition returns a :class:`Decision`: granted or denied, how long
  it waited, which pool bound it and why (throttled, a 429 pause, or the configured cap). A denial
  carries the agent-facing error through :func:`denial_error`.
* **A ``429`` means the whole group backs off.** :meth:`RateLimiter.report_429` pauses every bucket
  the source draws from, and the refill is frozen for the length of the pause, so the tokens spent
  before the pause are not handed back for free.

Thread safety: the server runs adapters in a bounded thread pool (``AGENTS.md`` section 3), so
:meth:`RateLimiter.acquire` takes one lock around the metering and sleeps *outside* it. The metering
is a reservation — the token is debited before the sleep — so concurrent callers queue up instead of
all passing at once. For the async server, :meth:`RateLimiter.acquire_async` moves that blocking
sleep to a worker thread.

The module has no I/O, no global mutable state and no wall-clock dependency.
"""

from __future__ import annotations

import asyncio
import math
import random
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Final, Protocol

from pydantic import Field

from mdqm.config.settings import RateLimitSettings, Settings, SourceConfig
from mdqm.errors import quota_exceeded
from mdqm.models import ErrorResponse
from mdqm.models.base import MDQMModel

__all__ = [
    "DEFAULT_POOL",
    "BucketState",
    "Decision",
    "DecisionReason",
    "LimitPolicy",
    "RateLimiter",
    "SystemTime",
    "TimeSource",
    "denial_error",
    "limits_for_sources",
    "parse_retry_after",
]

#: Name of the pool every ``shared_pool: true`` source joins, and the default pool total.
DEFAULT_POOL: Final[str] = "shared"

#: Prefix of the bucket that holds one source's own sub-limit (``source:crm_api``).
_SOURCE_PREFIX: Final[str] = "source:"


class TimeSource(Protocol):
    """Monotonic time and the sleep, injected so tests never wait and callers never read the
    wall clock (a rate limiter must not care about clock jumps or daylight saving)."""

    def now(self) -> float:
        """Seconds on a monotonic scale; only differences between two readings matter."""
        ...

    def sleep(self, seconds: float) -> None:
        """Block for ``seconds``; a non-positive value returns immediately."""
        ...


class SystemTime:
    """Production time source: :func:`time.monotonic` and :func:`time.sleep`."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0.0:
            time.sleep(seconds)


class LimitPolicy(MDQMModel):
    """One bucket's meter: ``rps`` tokens per second, at most ``burst`` tokens banked.

    ``jitter_s`` adds a random delay in ``[0, jitter_s]`` on top of a metered wait; it only ever
    slows a caller down, so it cannot break the rate, and the random source is seeded so a rerun
    repeats the same schedule. ``max_wait_s`` turns a wait longer than the cap into a denial: a
    caller that cannot afford to block gets an answer instead of a stuck thread.
    """

    rps: float = Field(default=1.0, gt=0, description="Sustained rate in requests per second.")
    burst: int = Field(default=1, ge=1, description="Tokens the bucket may bank: the burst size.")
    jitter_s: float = Field(default=0.0, ge=0, description="Upper bound of the random extra delay.")
    max_wait_s: float | None = Field(
        default=None, gt=0, description="Longest accepted wait; a longer one is denied."
    )


class DecisionReason(StrEnum):
    """Why an acquisition waited, or why it did not. Values are stable: a caller may branch."""

    #: The source declares no rate limit, so nothing is metered.
    UNLIMITED = "unlimited"
    #: A token was available right away.
    OK = "ok"
    #: The bucket had to be refilled first.
    THROTTLED = "throttled"
    #: A ``429``'s ``Retry-After`` is still running on one of the buckets.
    RETRY_AFTER = "retry_after"
    #: The wait was longer than the policy's ``max_wait_s``, so nothing was consumed.
    DENIED = "denied"


class Decision(MDQMModel):
    """The outcome of one acquisition: the caller's record of what the limiter did and why.

    ``wait_s`` is the metered wait and ``jitter_s`` the extra delay on top of it; together they are
    what was actually slept (:attr:`total_s`). ``pool`` names the bucket that produced the longest
    wait, so a throttle can be explained ("pool 'shared' at 1 req/s"). A denial sets
    ``granted=False`` and consumes nothing, and :func:`denial_error` renders it for the agent.
    """

    source: str
    granted: bool
    reason: DecisionReason
    wait_s: float = Field(ge=0, description="Metered wait in seconds.")
    jitter_s: float = Field(default=0.0, ge=0, description="Random delay added on top.")
    pool: str | None = Field(default=None, description="The bucket that bound this call.")
    limit_rps: float | None = Field(default=None, gt=0, description="That bucket's rate.")
    retry_after_s: float | None = Field(
        default=None, ge=0, description="A 429 pause still running on that bucket."
    )

    @property
    def total_s(self) -> float:
        """The delay actually slept: the metered wait plus the jitter."""
        return self.wait_s + self.jitter_s


class BucketState(MDQMModel):
    """A point-in-time view of one bucket. For logs, tests and the ``audit`` view.

    ``tokens`` may be negative: a reservation is debited before its sleep, so the debt is the
    number of calls still queued. ``pause_remaining_s`` is what a ``429`` still costs.
    """

    name: str
    rps: float
    burst: int
    tokens: float
    pause_remaining_s: float = Field(ge=0)
    granted: int = Field(ge=0)
    denied: int = Field(ge=0)
    waited_s: float = Field(ge=0)


class _Bucket:
    """Mutable state of one bucket. Private: it is only ever touched under the limiter's lock."""

    __slots__ = (
        "denied",
        "granted",
        "last",
        "name",
        "paused_until",
        "policy",
        "tokens",
        "waited_s",
    )

    def __init__(self, name: str, policy: LimitPolicy, now: float) -> None:
        self.name = name
        self.policy = policy
        # Start empty: bank nothing at start-up, so a burst of callers on a cold process cannot
        # exceed the budget (AGENTS.md principle 6: protect the source).
        self.tokens = 0.0
        self.last = now
        self.paused_until = now
        self.granted = 0
        self.denied = 0
        self.waited_s = 0.0

    def refill(self, now: float) -> None:
        """Credit the tokens elapsed time earned; ``last`` in the future (a pause) credits 0."""
        elapsed = now - self.last
        if elapsed > 0.0:
            self.tokens = min(float(self.policy.burst), self.tokens + elapsed * self.policy.rps)
            self.last = now

    def reserve(self, now: float) -> float:
        """Debit one token and report how long until it is actually available."""
        self.refill(now)
        self.tokens -= 1.0
        from_tokens = 0.0 if self.tokens >= 0.0 else -self.tokens / self.policy.rps
        return max(from_tokens, max(0.0, self.paused_until - now))

    def refund(self) -> None:
        """Give a reservation back (a denial consumes nothing)."""
        self.tokens += 1.0

    def pause(self, now: float, retry_after: float) -> None:
        """Freeze the bucket for ``retry_after`` seconds without refunding banked tokens."""
        self.refill(now)
        self.paused_until = max(self.paused_until, now + retry_after)
        # Refill starts again when the pause ends, so the debt survives the pause.
        self.last = max(self.last, self.paused_until)

    def state(self, now: float) -> BucketState:
        return BucketState(
            name=self.name,
            rps=self.policy.rps,
            burst=self.policy.burst,
            tokens=self.tokens,
            pause_remaining_s=max(0.0, self.paused_until - now),
            granted=self.granted,
            denied=self.denied,
            waited_s=self.waited_s,
        )


class RateLimiter:
    """Named token buckets shared by sources; see the module docstring for the rules.

    A source maps to the buckets it must draw one token from, in order. Two sources naming the same
    bucket share one budget, which is how ``shared_pool: true`` works: the pool bounds the group and
    each source's own bucket bounds that source. An unknown source is unmetered
    (``DecisionReason.UNLIMITED``), because a database source has no request budget.

    Args:
        policies: bucket name -> meter. A ``"source:<name>"`` bucket is one source's sub-limit.
        sources: source name -> the bucket names one call must draw from.
        time_source: monotonic time and sleep; ``SystemTime`` in production.
        rng: the random source for jitter; seeded in tests so a schedule repeats exactly.

    Raises:
        ValueError: a source names a bucket that has no policy, so a typo cannot silently unlimit
            a source.
    """

    def __init__(
        self,
        policies: Mapping[str, LimitPolicy],
        sources: Mapping[str, Iterable[str]],
        *,
        time_source: TimeSource | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._time: TimeSource = time_source if time_source is not None else SystemTime()
        # Jitter only spreads the gaps between calls; it is not a security decision (S311).
        self._rng = rng if rng is not None else random.Random()  # noqa: S311
        self._lock = threading.Lock()
        now = self._time.now()
        self._buckets: dict[str, _Bucket] = {
            name: _Bucket(name, policy, now) for name, policy in policies.items()
        }
        self._members: dict[str, tuple[str, ...]] = {}
        for source, names in sources.items():
            order = tuple(dict.fromkeys(names))
            unknown = [name for name in order if name not in self._buckets]
            if unknown:
                raise ValueError(
                    f"source {source!r} draws from undefined pool(s): {', '.join(sorted(unknown))}"
                )
            self._members[source] = order

    # --- building from config -----------------------------------------------------------

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        shared_pool_rps: float | None = None,
        jitter_s: float = 0.0,
        time_source: TimeSource | None = None,
        rng: random.Random | None = None,
    ) -> RateLimiter:
        """Build the limiter the configured sources share.

        ``shared_pool_rps`` is the pool's total when the operator knows it; without it the total is
        the smallest member's ``rps`` (see :func:`limits_for_sources`).
        """
        policies, sources = limits_for_sources(
            settings.sources, shared_pool_rps=shared_pool_rps, jitter_s=jitter_s
        )
        return cls(policies, sources, time_source=time_source, rng=rng)

    # --- using it -----------------------------------------------------------------------

    def members(self, source: str) -> tuple[str, ...]:
        """The buckets ``source`` draws from (empty when the source is unmetered)."""
        return self._members.get(source, ())

    def snapshot(self) -> tuple[BucketState, ...]:
        """Every bucket's state, sorted by name, for logs and tests."""
        with self._lock:
            now = self._time.now()
            return tuple(bucket.state(now) for _, bucket in sorted(self._buckets.items()))

    def acquire(self, source: str) -> Decision:
        """Take one token for ``source``, sleeping as long as the buckets demand.

        Returns the decision either way; a denial sleeps nothing and consumes nothing.
        """
        decision = self.reserve(source)
        if decision.granted and decision.total_s > 0.0:
            self._time.sleep(decision.total_s)
        return decision

    async def acquire_async(self, source: str) -> Decision:
        """``acquire`` for the async server: the blocking sleep runs in a worker thread."""
        return await asyncio.to_thread(self.acquire, source)

    def reserve(self, source: str) -> Decision:
        """Meter ``source`` without sleeping: the caller decides what to do with the wait.

        This is the whole logic of :meth:`acquire` minus the sleep, which makes it the right call
        for a test on a frozen clock and for a caller that would rather report the wait than block.
        """
        buckets = self._members.get(source, ())
        if not buckets:
            return Decision(source=source, granted=True, reason=DecisionReason.UNLIMITED, wait_s=0)
        with self._lock:
            now = self._time.now()
            waits = [
                (bucket, bucket.reserve(now)) for bucket in (self._buckets[n] for n in buckets)
            ]
            driver, wait_s = max(waits, key=lambda item: item[1])
            pause_s = max(0.0, driver.paused_until - now)
            overdue = [
                bucket
                for bucket, wait in waits
                if bucket.policy.max_wait_s is not None and wait > bucket.policy.max_wait_s
            ]
            if overdue:
                for bucket, _ in waits:
                    bucket.refund()
                for bucket in overdue:
                    bucket.denied += 1
                return Decision(
                    source=source,
                    granted=False,
                    reason=DecisionReason.DENIED,
                    wait_s=wait_s,
                    pool=driver.name,
                    limit_rps=driver.policy.rps,
                    retry_after_s=pause_s or None,
                )
            # Every bucket handed out a token; only the binding one is charged the wait.
            for bucket, _ in waits:
                bucket.granted += 1
            driver.waited_s += wait_s
            jitter_s = (
                self._rng.uniform(0.0, driver.policy.jitter_s)
                if driver.policy.jitter_s > 0
                else 0.0
            )
            return Decision(
                source=source,
                granted=True,
                reason=_reason(wait_s=wait_s, pause_s=pause_s),
                wait_s=wait_s,
                jitter_s=jitter_s,
                pool=driver.name,
                limit_rps=driver.policy.rps,
                retry_after_s=pause_s or None,
            )

    def report_429(self, source: str, retry_after: float | None = None) -> tuple[str, ...]:
        """A source answered ``429``: pause every bucket it draws from.

        Because members share one bucket, this pauses the whole pool, which is what the source is
        telling us to do. ``retry_after`` is the header's seconds (see :func:`parse_retry_after`);
        without one, the pause lasts one burst window (``burst / rps``), the shortest back-off that
        still refills the bucket.

        Returns:
            The names of the paused buckets, so the pause can be reported.
        """
        buckets = self._members.get(source, ())
        with self._lock:
            now = self._time.now()
            for name in buckets:
                bucket = self._buckets[name]
                pause_s = (
                    retry_after
                    if retry_after is not None
                    else bucket.policy.burst / (bucket.policy.rps)
                )
                bucket.pause(now, max(0.0, pause_s))
        return buckets


def _reason(*, wait_s: float, pause_s: float) -> DecisionReason:
    """Report the binding cause: a 429 pause outranks a plain refill."""
    if pause_s > 0.0 and pause_s >= wait_s:
        return DecisionReason.RETRY_AFTER
    return DecisionReason.THROTTLED if wait_s > 0.0 else DecisionReason.OK


def limits_for_sources(
    sources: Sequence[SourceConfig],
    *,
    shared_pool_rps: float | None = None,
    jitter_s: float = 0.0,
) -> tuple[dict[str, LimitPolicy], dict[str, tuple[str, ...]]]:
    """Turn the configured sources into bucket policies and a source -> buckets map.

    Every source with a ``rate_limit`` gets its own bucket (``source:<name>``) at its own ``rps``:
    that is its sub-limit. ``shared_pool: true`` (the default) additionally puts it in
    :data:`DEFAULT_POOL`, so the group is bounded by the pool total as well. ``shared_pool: false``
    meters the source on its own and nothing else.

    The pool total is ``shared_pool_rps`` when the operator knows the real budget; otherwise it is
    the **smallest** member's ``rps``, the conservative reading while the API budgets are unknown
    (``AGENTS.md`` section 9): the pool then never hands out more than any single member declared
    for itself. Configure ``shared_pool_rps`` explicitly once the real organisation total is known.

    Sources without a ``rate_limit`` are absent from the result and therefore unmetered.
    """
    policies: dict[str, LimitPolicy] = {}
    members: dict[str, tuple[str, ...]] = {}
    shared: list[tuple[str, RateLimitSettings]] = []
    for source in sources:
        policy = source.rate_limit
        if policy is None:
            continue
        own = f"{_SOURCE_PREFIX}{source.name}"
        policies[own] = LimitPolicy(rps=policy.rps, jitter_s=jitter_s)
        if policy.shared_pool:
            shared.append((source.name, policy))
        else:
            members[source.name] = (own,)
    if shared:
        total = (
            shared_pool_rps
            if shared_pool_rps is not None
            else min(policy.rps for _, policy in shared)
        )
        policies[DEFAULT_POOL] = LimitPolicy(rps=total, jitter_s=jitter_s)
        for name, _ in shared:
            members[name] = (DEFAULT_POOL, f"{_SOURCE_PREFIX}{name}")
    return policies, members


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """The seconds an HTTP ``Retry-After`` header asks for, or ``None`` when it says nothing usable.

    Both forms RFC 9110 allows: ``delta-seconds`` (``"5"``; a decimal such as ``"1.5"`` is accepted
    because sources send it) and an HTTP-date, which needs ``now`` to become a delay. A past date or
    a negative delta gives ``0.0``: never sleep a negative time. An infinite delta, an unparseable
    value, an empty header or a date without a timezone gives ``None``, so the caller falls back to
    :meth:`RateLimiter.report_429`'s default pause instead of guessing.

    Args:
        value: the raw header value, or ``None`` when the header is absent.
        now: the current instant, timezone-aware, for the HTTP-date form.

    Raises:
        ValueError: ``now`` is naive, which would make the delay depend on the machine's zone.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        seconds = None
    if seconds is not None:
        return max(0.0, seconds) if math.isfinite(seconds) else None
    if now is None:
        return None
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware to read an HTTP-date Retry-After")
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None or when.utcoffset() is None:
        return None
    return max(0.0, (when - now).total_seconds())


def denial_error(decision: Decision) -> ErrorResponse:
    """The agent-facing error for a denied decision (code ``QUOTA_EXCEEDED``).

    Raises:
        ValueError: the decision was granted, so there is no error to report.
    """
    if decision.granted:
        raise ValueError("a granted decision carries no error")
    pool = decision.pool or decision.source
    limit = f"{decision.limit_rps:g} requests/s" if decision.limit_rps is not None else "the budget"
    return quota_exceeded(
        quota=f'rate limit of "{pool}"',
        limit=limit,
        resets_at=f"in {decision.wait_s:.1f}s",
    )
