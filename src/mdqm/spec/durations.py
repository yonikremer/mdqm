"""Strict, locale-free durations: ``30d``, ``6h``, ``90m``, ``2w``, ``3mo``, ``1y`` and ISO-8601.

Two accepted forms, nothing else (a weak agent that guesses is told exactly what to send):

* compact, lowercase, one component: ``45s``, ``90m``, ``6h``, ``2w``, ``30d``, ``3mo``, ``1y``
* ISO-8601: ``P1M``, ``PT6H``, ``P1Y2M3DT4H5M6S``, ``P2W``

Rejected on purpose, because they are ambiguous or lossy: ``1D``/``1M`` (case decides whether ``M``
is minutes or months in ISO), ``1.5d`` (fractional), ``1 d`` (spaces), ``1min``, ``-1d``, ``1d12h``
(several components in the compact form: use ISO when you mean a sum).

Months are **calendar** months, so ``1mo`` is not 30 days. The one ambiguity a calendar leaves is
the month end. The rule here is *clamp*: keep the day of month, and if the target month is shorter,
use its last day. So 2026-01-31 + 1mo is 2026-02-28, and 2024-01-31 + 1mo is 2024-02-29.
Clamping is the conservative choice for a query bound: the window never grows past the asked-for
range (adding days instead would, and rolling over into March would skip February entirely).
"""

from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta
from typing import Annotated, Any, TypeAlias

from pydantic import BeforeValidator, Field, PlainSerializer, model_validator

from mdqm.models.base import MDQMModel

#: Largest count accepted in one component. Keeps every result inside datetime's 1..9999 years.
MAX_COMPONENT = 100_000

_ISO = re.compile(
    r"^P"
    r"(?!$)"
    r"(?:([0-9]{1,6})Y)?"
    r"(?:([0-9]{1,6})M)?"
    r"(?:([0-9]{1,6})W)?"
    r"(?:([0-9]{1,6})D)?"
    r"(?:T(?!$)(?:([0-9]{1,6})H)?(?:([0-9]{1,6})M)?(?:([0-9]{1,6})S)?)?"
    r"$"
)

_COMPACT: dict[str, tuple[str, int]] = {
    "s": ("seconds", 1),
    "m": ("seconds", 60),
    "h": ("seconds", 3600),
    "d": ("days", 1),
    "w": ("days", 7),
    "mo": ("months", 1),
    "y": ("months", 12),
}
_COMPACT_RE = re.compile(r"^([0-9]{1,6})(s|mo|m|h|d|w|y)$")

_COMPACT_HINT = "use '45s', '90m', '6h', '2w', '30d', '3mo' or '1y'"


class Duration(MDQMModel):
    """A calendar-aware span: ``months`` are calendar months, ``days``/``seconds`` are fixed.

    Serialises as its canonical text (``"3mo"``), so a spec survives a JSON round-trip unchanged.
    """

    months: int = Field(0, strict=True)
    days: int = Field(0, strict=True)
    seconds: int = Field(0, strict=True)

    @model_validator(mode="after")
    def _check_parts(self) -> Duration:
        for name, value in (
            ("months", self.months),
            ("days", self.days),
            ("seconds", self.seconds),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative, got {value}")
            if value > MAX_COMPONENT:
                raise ValueError(f"{name} must be at most {MAX_COMPONENT}, got {value}")
        return self

    @property
    def is_zero(self) -> bool:
        """True for a span that moves nothing: a lag lower bound may be zero, a window may not."""
        return self.months == self.days == self.seconds == 0

    def add_to(self, moment: datetime) -> datetime:
        """``moment`` shifted forward by this span, clamped to the last day of a short month."""
        shifted = _shift_months(moment, self.months)
        return shifted + timedelta(days=self.days, seconds=self.seconds)

    def subtract_from(self, moment: datetime) -> datetime:
        """``moment`` shifted backward by this span (same month-end clamp rule)."""
        shifted = _shift_months(moment, -self.months)
        return shifted - timedelta(days=self.days, seconds=self.seconds)

    def to_text(self) -> str:
        """Canonical text: the compact form when one component fits, else ISO-8601."""
        if self.days == 0 and self.seconds == 0:
            if self.months % 12 == 0 and self.months:
                return f"{self.months // 12}y"
            if self.months:
                return f"{self.months}mo"
        if self.months == 0:
            if self.seconds == 0:
                if self.days and self.days % 7 == 0:
                    return f"{self.days // 7}w"
                return f"{self.days}d"
            if self.days == 0:
                for unit, size in (("h", 3600), ("m", 60), ("s", 1)):
                    if self.seconds % size == 0:
                        return f"{self.seconds // size}{unit}"
        return _to_iso(self)

    def __str__(self) -> str:
        return self.to_text()

    @classmethod
    def parse(cls, text: str) -> Duration:
        """Parse compact or ISO-8601 text; ``ValueError`` names the reason and the valid forms."""
        if not isinstance(text, str):
            raise ValueError(f"a duration must be text, got {type(text).__name__}")
        raw = text.strip()
        if not raw:
            raise ValueError(f"empty duration; {_COMPACT_HINT} or ISO-8601 such as 'P3M'")
        if raw != text:
            raise ValueError(f"duration {text!r} has surrounding spaces; send {raw!r}")
        if raw.startswith("P"):
            return _parse_iso(raw)
        found = _COMPACT_RE.match(raw)
        if found is None:
            raise ValueError(_reject_compact(raw))
        count, unit = int(found.group(1)), found.group(2)
        field, size = _COMPACT[unit]
        return _from_parts({field: count * size})

    @classmethod
    def _validate(cls, value: Any) -> Duration:
        """Accept text, ``Duration`` parts, or a ``Duration`` itself (validator/serialiser)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.parse(value)
        if isinstance(value, dict):
            return _from_parts(value)
        if isinstance(value, timedelta):
            total = value.total_seconds()
            if total < 0 or total != int(total):
                raise ValueError(
                    "a timedelta duration must be a whole, non-negative number of "
                    f"seconds, got {value}"
                )
            return _from_parts({"seconds": int(total)})
        raise ValueError(f"a duration must be text such as '3mo' or ISO-8601 'P3M', got {value!r}")


#: A duration field in a spec: written as text (``"3mo"``), dumped as the same canonical text.
DurationField: TypeAlias = Annotated[
    Duration,
    BeforeValidator(Duration._validate, json_schema_input_type=str),
    PlainSerializer(Duration.to_text, return_type=str),
]


def _from_parts(parts: dict[str, Any]) -> Duration:
    """Build a ``Duration`` from its three parts, without re-entering the field validator."""
    unknown = sorted(set(parts) - {"months", "days", "seconds"})
    if unknown:
        raise ValueError(
            f"unknown duration part(s) {unknown}; a duration has months, days and seconds"
        )
    checked: dict[str, int] = {}
    for name in ("months", "days", "seconds"):
        raw = parts.get(name, 0)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"duration {name} must be a whole number, got {raw!r}")
        if raw < 0:
            raise ValueError(f"duration {name} must not be negative, got {raw}")
        if raw > MAX_COMPONENT:
            raise ValueError(f"duration {name} must be at most {MAX_COMPONENT}, got {raw}")
        checked[name] = raw
    return Duration.model_construct(
        months=checked["months"], days=checked["days"], seconds=checked["seconds"]
    )


def _reject_compact(raw: str) -> str:
    """Say exactly why a compact duration was refused, and what to send instead."""
    lowered = raw.lower()
    if lowered != raw:
        return (
            f"duration {raw!r} must be lowercase: 'm' is minutes and 'mo' is months, "
            "so case cannot be optional"
        )
    if _COMPACT_RE.match(lowered) and lowered != raw:
        return f"duration {raw!r} must be lowercase; send {lowered!r}"
    cut = re.match(r"^([0-9]+)([A-Za-z]+)$", raw)
    if cut is not None:
        unit = cut.group(2)
        if unit in {"M", "mo.", "mon", "month", "months", "min", "mins", "minute", "sec", "secs"}:
            return f"duration unit {unit!r} is ambiguous; {_COMPACT_HINT}"
        return f"unknown duration unit {unit!r}; {_COMPACT_HINT}"
    if re.match(r"^[0-9]", raw):
        return (
            f"duration {raw!r} is not a single '<number><unit>'; "
            "use ISO-8601 for several components, e.g. 'P1DT12H'"
        )
    return f"duration {raw!r} is not a duration; {_COMPACT_HINT} or ISO-8601 such as 'P3M'"


def _parse_iso(raw: str) -> Duration:
    found = _ISO.match(raw)
    if found is None:
        raise ValueError(
            f"duration {raw!r} is not valid ISO-8601; use e.g. 'P3M', 'P1Y2M', 'PT6H' or 'P1DT12H'"
        )
    years, months, weeks, days, hours, minutes, seconds = (
        int(part) if part is not None else 0 for part in found.groups()
    )
    return _from_parts(
        {
            "months": years * 12 + months,
            "days": weeks * 7 + days,
            "seconds": hours * 3600 + minutes * 60 + seconds,
        }
    )


def _to_iso(duration: Duration) -> str:
    years, months = divmod(duration.months, 12)
    text = "P"
    if years:
        text += f"{years}Y"
    if months:
        text += f"{months}M"
    if duration.days:
        text += f"{duration.days}D"
    hours, rest = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours or minutes or seconds:
        text += "T"
        if hours:
            text += f"{hours}H"
        if minutes:
            text += f"{minutes}M"
        if seconds:
            text += f"{seconds}S"
    return text


def _shift_months(moment: datetime, months: int) -> datetime:
    """Move by whole calendar months, clamping the day to the target month's last day."""
    if months == 0:
        return moment
    index = moment.month - 1 + months
    year = moment.year + index // 12
    month = index % 12 + 1
    if not 1 <= year <= 9999:
        raise ValueError(f"duration moves {moment.isoformat()} outside years 1..9999")
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def parse_duration(text: str) -> Duration:
    """Parse compact or ISO-8601 duration text; ``ValueError`` names what to send instead."""
    return Duration.parse(text)
