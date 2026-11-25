"""A circuit breaker per source, so a dead database is not asked again on every call.

A primary database with no replica fails hard when it fails: every retry costs it a connection
attempt and a plan. The breaker is the cheap, deterministic answer -- after a few consecutive
failures it *opens*, and the next calls are refused before the adapter is touched. After a cooldown
it goes *half-open*: exactly one trial query is allowed through, which proves the source is back
without a stampede.

This is a classic three-state machine, and the states are reported, never implied:

* ``CLOSED`` -- normal; failures are counted.
* ``OPEN`` -- refused; ``opens_in_s`` says when the cooldown ends.
* ``HALF_OPEN`` -- one trial in flight; a success closes, a failure re-opens with a fresh cooldown.

The clock is injected, so :class:`CircuitBreaker` is driven by a scripted clock in tests and never
sleeps. It is thread-safe: the server runs adapters in a bounded thread pool, so the state machine
takes one lock around each transition.
"""

from __future__ import annotations

import threading
from enum import StrEnum

from pydantic import Field

from mdqm.errors import source_unavailable
from mdqm.governor.clock import Clock, SystemClock
from mdqm.governor.refusal import GovernorRefusal
from mdqm.models.base import MDQMModel

__all__ = [
    "BreakerPolicy",
    "BreakerReport",
    "BreakerState",
    "CircuitBreaker",
]


class BreakerState(StrEnum):
    """The three states, in the vocabulary ``docs/DESIGN.md`` uses."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class BreakerPolicy(MDQMModel):
    """When to stop asking a source, and how long to wait before asking once more."""

    failure_threshold: int = Field(
        default=3, ge=1, description="Consecutive failures that open the breaker."
    )
    cooldown_s: float = Field(
        default=30.0, gt=0, description="How long an open breaker refuses before one trial."
    )


class BreakerReport(MDQMModel):
    """A source's breaker, as an operator or a test reads it."""

    source: str
    state: BreakerState
    consecutive_failures: int = Field(ge=0)
    opens_in_s: float | None = Field(
        default=None, ge=0, description="Seconds until the next trial; null when closed."
    )
    opened_count: int = Field(ge=0, description="How many times this breaker has opened.")


class _State:
    """One source's mutable breaker state. Private: only touched under the breaker lock."""

    __slots__ = ("consecutive_failures", "open_until", "opened_count", "state", "trial_in_flight")

    def __init__(self) -> None:
        self.state = BreakerState.CLOSED
        self.consecutive_failures = 0
        self.open_until = 0.0
        self.opened_count = 0
        self.trial_in_flight = False


class CircuitBreaker:
    """Breaker per source; see the module docstring.

    Args:
        policy: the failure threshold and the cooldown.
        clock: injected time source; ``SystemClock`` in production.
    """

    def __init__(self, policy: BreakerPolicy | None = None, *, clock: Clock | None = None) -> None:
        self._policy = policy if policy is not None else BreakerPolicy()
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._lock = threading.Lock()
        self._states: dict[str, _State] = {}

    @property
    def policy(self) -> BreakerPolicy:
        """The thresholds in force, for reporting."""
        return self._policy

    def ensure_closed(self, source: str) -> None:
        """Let a call through, or refuse it because the source is known to be down.

        A half-open breaker admits exactly one trial: the caller that receives the trial owns
        reporting its outcome through :meth:`record_success` or :meth:`record_failure`.

        Raises:
            GovernorRefusal: the breaker is open. The ``SOURCE_UNAVAILABLE`` response says nothing
                in the spec can fix it, which is the truth for a dead source.
        """
        with self._lock:
            state = self._state(source)
            now = self._clock.monotonic()
            if state.state is BreakerState.OPEN and now < state.open_until:
                raise GovernorRefusal(source_unavailable(source=source))
            if state.state is BreakerState.OPEN:
                state.state = BreakerState.HALF_OPEN
                state.trial_in_flight = True
                return
            if state.state is BreakerState.HALF_OPEN:
                if state.trial_in_flight:
                    # One trial at a time: the rest wait out the cooldown instead of stampeding.
                    raise GovernorRefusal(source_unavailable(source=source))
                state.trial_in_flight = True

    def record_success(self, source: str) -> None:
        """A call to ``source`` worked: close the breaker and reset the failure count."""
        with self._lock:
            state = self._state(source)
            state.state = BreakerState.CLOSED
            state.consecutive_failures = 0
            state.trial_in_flight = False

    def record_failure(self, source: str) -> None:
        """A call to ``source`` failed: count it, and open or re-open once the threshold is met."""
        with self._lock:
            state = self._state(source)
            state.consecutive_failures += 1
            state.trial_in_flight = False
            if state.consecutive_failures >= self._policy.failure_threshold:
                if state.state is not BreakerState.OPEN:
                    state.opened_count += 1
                state.state = BreakerState.OPEN
                state.open_until = self._clock.monotonic() + self._policy.cooldown_s

    def report(self, source: str) -> BreakerReport:
        """``source``'s breaker state, without changing it."""
        with self._lock:
            state = self._state(source)
            opens_in: float | None = None
            if state.state is BreakerState.OPEN:
                opens_in = max(0.0, state.open_until - self._clock.monotonic())
            return BreakerReport(
                source=source,
                state=state.state,
                consecutive_failures=state.consecutive_failures,
                opens_in_s=opens_in,
                opened_count=state.opened_count,
            )

    def _state(self, source: str) -> _State:
        """The source's state, created closed on first use. Caller holds the lock."""
        state = self._states.get(source)
        if state is None:
            state = _State()
            self._states[source] = state
        return state
