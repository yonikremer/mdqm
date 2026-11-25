"""The governor's one time source, injected so no test ever waits and no call reads the wall clock.

Two readings, because the governor needs both and they answer different questions:

* :meth:`Clock.monotonic` -- seconds on a monotonic scale. Deadlines, waits, cooldowns and quota
  windows are all *differences* between two readings, so a clock jump or a daylight-saving change
  must not stretch or shrink a timeout.
* :meth:`Clock.now` -- the current instant, timezone-aware, for the two places a human reads a
  timestamp: a handle's expiry and a quota's reset time.

A test supplies a scripted clock and drives it, which is what makes the timeout, quota and breaker
tests deterministic instead of sleepy.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol

__all__ = ["Clock", "SystemClock"]


class Clock(Protocol):
    """Monotonic seconds plus the aware instant. See the module docstring."""

    def monotonic(self) -> float:
        """Seconds on a monotonic scale; only differences between two readings matter."""
        ...

    def now(self) -> datetime:
        """The current instant, timezone-aware, for TTLs and reset times."""
        ...


class SystemClock:
    """Production clock: :func:`time.monotonic` and an aware UTC instant."""

    def monotonic(self) -> float:
        return time.monotonic()

    def now(self) -> datetime:
        return datetime.now(UTC)
