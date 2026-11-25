"""Quiet windows: the hours in which heavy jobs (profiling, statistics, mining) may run.

Pure logic, no I/O, so it is table-testable. One window is a wall-clock range in a named
time zone; ``start: "21:00", end: "07:00"`` crosses midnight.
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, field_validator


class QuietWindow(BaseModel):
    """Wall-clock range in a time zone. ``start`` is inclusive, ``end`` exclusive."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tz: str = "UTC"
    start: time
    end: time

    @field_validator("tz")
    @classmethod
    def _known_zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            # zoneinfo reads the OS tz database; on Windows it needs the tzdata wheel.
            raise ValueError(
                f"unknown time zone {value!r}: use an IANA name such as 'Asia/Jerusalem' "
                "(zoneinfo needs the tzdata package on Windows)"
            ) from exc
        return value

    @field_validator("start", "end")
    @classmethod
    def _no_offset(cls, value: time) -> time:
        if value.tzinfo is not None:
            raise ValueError(
                "give a wall-clock time without an offset (e.g. '21:00'); the offset comes from tz"
            )
        return value

    @property
    def zone(self) -> ZoneInfo:
        """The window's zone (ZoneInfo caches instances, so this is cheap)."""
        return ZoneInfo(self.tz)

    def is_quiet(self, now: datetime) -> bool:
        """True when ``now`` falls inside the window.

        An aware datetime is converted to the window's zone. A naive datetime is read as
        wall-clock time in the window's zone. A zero-length window (start == end) is never
        quiet: it is the safe reading, heavy work then needs an explicit window.
        """
        local = now.astimezone(self.zone) if now.tzinfo is not None else now
        current = local.time()
        if self.start == self.end:
            return False
        if self.start < self.end:
            return self.start <= current < self.end
        return current >= self.start or current < self.end
