"""Durations in the semantic YAML: ``7d``, ``6h``, ``30s``.

Time windows, cache TTLs and ingestion lag are written the way a person writes them. Numbers are
ambiguous (seconds? days?), so a bare number is rejected; ISO 8601 (``P7D``) is accepted because
that is what a JSON round-trip of a :class:`datetime.timedelta` produces.
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Annotated, Any

from pydantic import BeforeValidator, TypeAdapter, ValidationError

__all__ = ["DURATION_HELP", "Duration", "parse_duration"]

#: Seconds in each unit suffix. ``m`` is minutes, ``ms`` milliseconds; there is no month/year
#: because a month is not a fixed number of seconds.
_UNITS: dict[str, float] = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "w": 604800.0,
}

_SHORT_FORM = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s|m|h|d|w)")

DURATION_HELP = "write a duration such as '30s', '6h', '7d' or an ISO 8601 duration like 'P7D'"

_ISO: TypeAdapter[timedelta] = TypeAdapter(timedelta)


def parse_duration(value: Any) -> timedelta:
    """Parse ``"30s"`` / ``"7d"`` / ISO 8601 into a timedelta.

    Raises :class:`ValueError` (never returns a silently wrong value) for a bare number, a
    negative duration, an unknown unit or an unparseable string.
    """
    if isinstance(value, timedelta):
        duration = value
    elif isinstance(value, str):
        text = value.strip()
        match = _SHORT_FORM.fullmatch(text)
        if match is not None:
            duration = timedelta(seconds=float(match.group("value")) * _UNITS[match.group("unit")])
        else:
            try:
                duration = _ISO.validate_python(text)
            except ValidationError:
                raise ValueError(f"{value!r} is not a duration: {DURATION_HELP}") from None
    else:
        raise ValueError(
            f"expected a duration string, got {type(value).__name__}: {DURATION_HELP} "
            "(a bare number would be ambiguous)"
        )
    if duration < timedelta(0):
        raise ValueError(f"a duration cannot be negative, got {value!r}")
    return duration


#: A :class:`datetime.timedelta` that the YAML writes as a duration string.
Duration = Annotated[timedelta, BeforeValidator(parse_duration)]
