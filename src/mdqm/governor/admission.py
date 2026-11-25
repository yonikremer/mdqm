"""Bounded admission: how many queries may run at once, and how many one user may send.

This is the layer that keeps weak agents from hammering primary databases with no replicas
(``AGENTS.md`` principle 6). It enforces three nested ceilings around every execution:

* a **global** ceiling -- the server never runs more than ``max_global`` queries at a time;
* a **per-user** ceiling -- one noisy agent cannot occupy the whole server;
* a **per-source** ceiling -- one popular source never sees more than ``per_source`` at a time.

Plus a **query quota** per user over a fixed window, which is a budget rather than a concurrency
limit: 50 genuinely sequential queries still cost the source 50 plans, so the count matters even
when nothing overlaps.

Two rules shape the code:

* **Every scope is released, whatever happens.** Slots are taken in a fixed order (global, user,
  source) and released in reverse inside a ``finally``, so a timeout, an adapter exception or a
  refusal cannot leak a slot. Taking them in one order everywhere is what keeps two users from
  deadlocking on each other.
* **Nothing is silent.** A wait is measured and returned on the :class:`Lease`; a refusal carries
  the ``ErrorResponse`` that says which ceiling bound it and when it frees up. Peaks per scope are
  kept so an operator (or a load test) can see what actually happened, not what the config said.

Waiting is real (a semaphore timeout, not a sleep) and bounded by ``max_wait_s``: a caller that
cannot get a slot in that window is refused with ``QUOTA_EXCEEDED`` instead of blocking forever.
The clock is injected so the quota window and the breaker's cooldown are testable without sleeping.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum

from pydantic import Field

from mdqm.errors import quota_exceeded
from mdqm.governor.clock import Clock, SystemClock
from mdqm.governor.refusal import GovernorRefusal
from mdqm.models import ErrorResponse
from mdqm.models.base import MDQMModel

__all__ = [
    "Admission",
    "AdmissionPolicy",
    "Lease",
    "Peak",
    "Quota",
    "QuotaPolicy",
    "QuotaState",
    "Scope",
]


class AdmissionPolicy(MDQMModel):
    """The three concurrency ceilings and how long a caller may queue for a slot.

    The defaults are the generic product defaults (``docs/DESIGN.md``: 8 in the server, 2 per user,
    2-3 per source). An organisation overrides them from config; nothing here knows a real source's
    capacity.
    """

    max_global: int = Field(default=8, ge=1, description="Queries the server runs at once.")
    per_user: int = Field(default=2, ge=1, description="Queries one user may run at once.")
    per_source: int = Field(default=3, ge=1, description="Queries one source may run at once.")
    max_wait_s: float = Field(
        default=5.0,
        ge=0,
        description="Longest accepted queue for a slot; a caller that waits longer is refused.",
    )


class Scope(StrEnum):
    """Which ceiling refused a caller. Values are stable: a refusal names one of them."""

    GLOBAL = "global"
    USER = "user"
    SOURCE = "source"


class Peak(MDQMModel):
    """The high-water mark of one scope: what a load test asserts on, and what an operator reads."""

    scope: Scope
    name: str
    limit: int = Field(ge=1)
    peak: int = Field(ge=0, description="Most slots held at once since start-up.")
    in_use: int = Field(ge=0, description="Slots held right now.")
    waited_total_s: float = Field(ge=0)


class Lease(MDQMModel):
    """One holder's receipt: how long it queued, so the wait is reported not hidden."""

    source: str
    user: str
    waited_s: float = Field(ge=0)


class QuotaPolicy(MDQMModel):
    """A per-user budget of queries over a fixed window.

    A fixed window (not a sliding one) is deliberate: its reset time is a single instant the error
    can name, which is what makes ``QUOTA_EXCEEDED``'s ``fix`` actionable for a weak agent.
    """

    max_queries: int = Field(default=100, ge=1, description="Queries one user may run per window.")
    window_s: float = Field(default=60.0, gt=0, description="Length of the fixed window.")


class QuotaState(MDQMModel):
    """One user's window, for logs and tests."""

    user: str
    used: int = Field(ge=0)
    limit: int = Field(ge=1)
    resets_in_s: float = Field(ge=0, description="Seconds until the window rolls over.")
    granted: int = Field(ge=0)
    denied: int = Field(ge=0)


class _Counter:
    """One ceiling's live state. Private: only ever touched under the admission lock."""

    __slots__ = ("in_use", "limit", "name", "peak", "scope", "waited_total_s")

    def __init__(self, scope: Scope, name: str, limit: int) -> None:
        self.scope = scope
        self.name = name
        self.limit = limit
        self.in_use = 0
        self.peak = 0
        self.waited_total_s = 0.0

    def enter(self) -> None:
        self.in_use += 1
        self.peak = max(self.peak, self.in_use)

    def leave(self) -> None:
        self.in_use -= 1

    def state(self) -> Peak:
        return Peak(
            scope=self.scope,
            name=self.name,
            limit=self.limit,
            peak=self.peak,
            in_use=self.in_use,
            waited_total_s=self.waited_total_s,
        )


class _Window:
    """One user's quota window. Private: only touched under the quota lock."""

    __slots__ = ("denied", "granted", "started", "used")

    def __init__(self, started: float) -> None:
        self.started = started
        self.used = 0
        self.granted = 0
        self.denied = 0


class Admission:
    """The three bounded ceilings; see the module docstring.

    Args:
        policy: the ceilings and the longest accepted queue.
        clock: injected time source; ``SystemClock`` in production.
    """

    def __init__(
        self,
        policy: AdmissionPolicy | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._policy = policy if policy is not None else AdmissionPolicy()
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._lock = threading.Lock()
        self._global = _Counter(Scope.GLOBAL, "server", self._policy.max_global)
        self._sources: dict[str, _Counter] = {}
        self._users: dict[str, _Counter] = {}
        self._sem_global = threading.BoundedSemaphore(self._policy.max_global)
        self._sem_sources: dict[str, threading.BoundedSemaphore] = {}
        self._sem_users: dict[str, threading.BoundedSemaphore] = {}

    @property
    def policy(self) -> AdmissionPolicy:
        """The ceilings in force, for reporting."""
        return self._policy

    @contextmanager
    def hold(self, *, source: str, user: str) -> Iterator[Lease]:
        """Hold the global, user and source slots for one execution, in that fixed order.

        The slots are released in reverse order when the block exits, so a timeout, an adapter
        exception or a refusal cannot strand one.

        Raises:
            GovernorRefusal: a slot did not free up within ``max_wait_s``. The response's
                ``QUOTA_EXCEEDED`` names the binding ceiling and when it frees up.
        """
        started = self._clock.monotonic()
        acquired: list[tuple[threading.BoundedSemaphore, _Counter]] = []
        try:
            for semaphore, counter in self._slots(source, user):
                remaining = self._policy.max_wait_s - (self._clock.monotonic() - started)
                if not semaphore.acquire(timeout=max(0.0, remaining)):
                    waited = self._clock.monotonic() - started
                    raise GovernorRefusal(_busy_error(counter, waited))
                acquired.append((semaphore, counter))
                with self._lock:
                    counter.enter()
                    counter.waited_total_s += self._clock.monotonic() - started
            waited = max(0.0, self._clock.monotonic() - started)
            yield Lease(source=source, user=user, waited_s=waited)
        finally:
            for semaphore, counter in reversed(acquired):
                with self._lock:
                    counter.leave()
                semaphore.release()

    def peaks(self) -> tuple[Peak, ...]:
        """Every ceiling's high-water mark, sorted by scope then name. What a load test asserts."""
        with self._lock:
            counters = [self._global, *self._sources.values(), *self._users.values()]
            return tuple(
                counter.state() for counter in sorted(counters, key=lambda c: (c.scope, c.name))
            )

    def _slots(
        self, source: str, user: str
    ) -> list[tuple[threading.BoundedSemaphore, _Counter]]:
        """The three (semaphore, counter) pairs in acquisition order, created lazily."""
        with self._lock:
            source_counter = self._sources.get(source)
            if source_counter is None:
                source_counter = _Counter(Scope.SOURCE, source, self._policy.per_source)
                self._sources[source] = source_counter
                self._sem_sources[source] = threading.BoundedSemaphore(self._policy.per_source)
            user_counter = self._users.get(user)
            if user_counter is None:
                user_counter = _Counter(Scope.USER, user, self._policy.per_user)
                self._users[user] = user_counter
                self._sem_users[user] = threading.BoundedSemaphore(self._policy.per_user)
            return [
                (self._sem_global, self._global),
                (self._sem_users[user], user_counter),
                (self._sem_sources[source], source_counter),
            ]


class Quota:
    """A per-user fixed-window budget of queries; see :class:`QuotaPolicy`.

    Args:
        policy: the budget and its window.
        clock: injected time source; ``SystemClock`` in production.
    """

    def __init__(self, policy: QuotaPolicy, *, clock: Clock | None = None) -> None:
        self._policy = policy
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._lock = threading.Lock()
        self._windows: dict[str, _Window] = {}

    def consume(self, user: str) -> QuotaState:
        """Spend one query from ``user``'s budget.

        Raises:
            GovernorRefusal: the budget is used up. The ``QUOTA_EXCEEDED`` response carries the
                reset time (seconds until the window rolls) in its ``fix``.
        """
        with self._lock:
            window = self._window(user)
            remaining = self._policy.window_s - (self._clock.monotonic() - window.started)
            if window.used >= self._policy.max_queries:
                window.denied += 1
                raise GovernorRefusal(_quota_error(user, self._policy, remaining))
            window.used += 1
            window.granted += 1
            return _quota_state(user, self._policy, window, max(0.0, remaining))

    def state(self, user: str) -> QuotaState:
        """``user``'s window without spending from it."""
        with self._lock:
            window = self._window(user)
            remaining = self._policy.window_s - (self._clock.monotonic() - window.started)
            return _quota_state(user, self._policy, window, max(0.0, remaining))

    def _window(self, user: str) -> _Window:
        """The user's live window, rolled over when it has expired. Caller holds the lock."""
        now = self._clock.monotonic()
        window = self._windows.get(user)
        if window is None or now - window.started >= self._policy.window_s:
            window = _Window(now)
            self._windows[user] = window
        return window


def _busy_error(counter: _Counter, waited: float) -> ErrorResponse:
    """The refusal for a ceiling that held for the whole wait."""
    quota = {
        Scope.GLOBAL: "concurrent queries on the server",
        Scope.USER: f'concurrent queries for user "{counter.name}"',
        Scope.SOURCE: f'concurrent queries on source "{counter.name}"',
    }[counter.scope]
    return quota_exceeded(
        quota=quota,
        limit=f"{counter.limit} at a time",
        resets_at=f"in at most {waited:g}s, once a running query finishes",
    )


def _quota_error(user: str, policy: QuotaPolicy, remaining: float) -> ErrorResponse:
    """The refusal for an exhausted per-user budget, naming the reset."""
    return quota_exceeded(
        quota=f'query quota for user "{user}"',
        limit=f"{policy.max_queries} queries per {policy.window_s:g}s",
        resets_at=f"in {max(0.0, remaining):g}s",
    )


def _quota_state(user: str, policy: QuotaPolicy, window: _Window, remaining: float) -> QuotaState:
    return QuotaState(
        user=user,
        used=window.used,
        limit=policy.max_queries,
        resets_in_s=remaining,
        granted=window.granted,
        denied=window.denied,
    )
