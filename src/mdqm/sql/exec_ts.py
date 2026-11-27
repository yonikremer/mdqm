"""Rewrite every non-deterministic time function to one absolute ``exec_ts`` value.

AGENTS.md section 5: *exec_ts is one value per execution*. A source's own clock must never
decide the timestamp of a query, because a re-run has to see the same window and the audit has
to be reproducible (DESIGN.md section 13: ``frozen`` vs ``relative``).

This module is pure: no I/O and no clock read. The caller injects one timezone-aware
``exec_ts`` per execution (:func:`resolve_exec_ts` turns a stored value or an injected clock
into that value) and the source's *session* time zone, in which the literal is rendered.

Rewritten: ``now()``, ``CURRENT_TIMESTAMP``, ``CURRENT_DATE``, ``LOCALTIMESTAMP``,
``SYSDATE``, ``SYSTIMESTAMP``, ``GETDATE()``, ``GETUTCDATE()``, ``SYSDATETIME()``,
``SYSDATETIMEOFFSET()``, ``UTC_TIMESTAMP()`` and friends, plus the interval arithmetic on
them (``now() - interval '7 days'``, ``SYSDATE - 7``, ``SYSDATE - 1/24``,
``DATEADD(day, -7, GETDATE())``) which is folded into a single literal. Other wrappers
(``TRUNC``, ``ADD_MONTHS``, ``EOMONTH``, ...) keep their meaning: only the now-value inside
them is replaced.

Elasticsearch date math gets the same treatment on a Query DSL dict:
``now-7d/d`` becomes the absolute range bound, ``/d`` rounding included.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from fractions import Fraction
from typing import Any, Final, NamedTuple, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import sqlglot
from pydantic import BaseModel, ConfigDict, Field
from sqlglot import exp

#: sqlglot dialects this module knows the now-like function names of.
SUPPORTED_DIALECTS: Final[frozenset[str]] = frozenset(
    {"duckdb", "mysql", "oracle", "postgres", "tsql"}
)

#: Names sqlglot does not normalise into a dedicated node, per dialect. A name is only
#: rewritten when the target dialect actually has it; a same-named user function is left alone.
_NOW_NAMES: Final[Mapping[str, frozenset[str]]] = {
    "duckdb": frozenset(
        {"now", "current_date", "current_timestamp", "localtime", "localtimestamp"}
    ),
    "mysql": frozenset(
        {
            "now",
            "curdate",
            "curtime",
            "utc_date",
            "utc_timestamp",
            "current_date",
            "current_timestamp",
            "localtime",
            "localtimestamp",
        }
    ),
    "oracle": frozenset(
        {
            "sysdate",
            "systimestamp",
            "current_date",
            "current_timestamp",
            "localtime",
            "localtimestamp",
        }
    ),
    "postgres": frozenset(
        {"now", "current_date", "current_timestamp", "localtime", "localtimestamp"}
    ),
    "tsql": frozenset(
        {
            "getdate",
            "getutcdate",
            "sysdatetime",
            "sysdatetimeoffset",
            "sysutcdatetime",
            "current_timestamp",
        }
    ),
}

#: Names whose value is UTC, not the session's local wall clock.
_UTC_NAMES: Final[frozenset[str]] = frozenset(
    {"getutcdate", "sysutcdatetime", "utc_date", "utc_timestamp"}
)

#: SQL Server datetime2 keeps 100ns; MySQL DATETIME is second-precision without a fsp argument.
_TIMESTAMP_TYPE: Final[Mapping[str, str]] = {
    "duckdb": "TIMESTAMP",
    "mysql": "DATETIME",
    "oracle": "TIMESTAMP",
    "postgres": "TIMESTAMP",
    "tsql": "DATETIME2",
}

#: Dialects where ``date_column - <number>`` means "that many days earlier".
_NUMBER_IS_DAYS: Final[frozenset[str]] = frozenset({"oracle", "tsql"})

_MONTHS_PER_UNIT: Final[Mapping[str, int]] = {
    "QUARTER": 3,
    "QUARTERS": 3,
    "YEAR": 12,
    "YEARS": 12,
    "YY": 12,
    "YYYY": 12,
    "MONTH": 1,
    "MONTHS": 1,
    "MM": 1,
}

_MICROS_PER_UNIT: Final[Mapping[str, int]] = {
    "WEEK": 604_800 * 1_000_000,
    "WEEKS": 604_800 * 1_000_000,
    "WW": 604_800 * 1_000_000,
    "DAY": 86_400 * 1_000_000,
    "DAYS": 86_400 * 1_000_000,
    "DD": 86_400 * 1_000_000,
    "D": 86_400 * 1_000_000,
    "HOUR": 3_600 * 1_000_000,
    "HOURS": 3_600 * 1_000_000,
    "HH": 3_600 * 1_000_000,
    "H": 3_600 * 1_000_000,
    "MINUTE": 60 * 1_000_000,
    "MINUTES": 60 * 1_000_000,
    "MI": 60 * 1_000_000,
    "N": 60 * 1_000_000,
    "SECOND": 1_000_000,
    "SECONDS": 1_000_000,
    "SS": 1_000_000,
    "S": 1_000_000,
    "MILLISECOND": 1_000,
    "MILLISECONDS": 1_000,
    "MS": 1_000,
    "MICROSECOND": 1,
    "MICROSECONDS": 1,
    "US": 1,
}

_MICROS_PER_DAY: Final = 86_400 * 1_000_000

#: Determinism guard: sqlglot's generator is not always a fixed point on the first pass
#: (T-SQL ``EOMONTH`` grows a CAST on the second). Rendering until it stops changing makes
#: "rewrite twice is a no-op" hold on the text, not just on the AST.
_MAX_RENDER_PASSES: Final = 4


class ExecTsMode(StrEnum):
    """Which timestamp an execution uses (DESIGN.md section 13)."""

    FROZEN = "frozen"
    """Re-use the stored ``exec_ts`` of a past run, so a re-run sees the same window."""
    RELATIVE = "relative"
    """Take a fresh value from the injected clock."""


class Replacement(BaseModel):
    """One audited replacement: what was there, what it became, and where.

    ``original`` is the replaced expression as sqlglot renders it in the target dialect, which
    is the same expression but not always the same spelling (sqlglot normalises ``now()`` to
    ``CURRENT_TIMESTAMP`` and ``interval '30 days'`` to ``INTERVAL '30 DAYS'``).
    ``position`` is the character offset of the expression's first character in the input. It
    is ``None`` when the dialect's parser records no position for that expression (bare
    keywords such as ``CURRENT_DATE``, and spellings the dialect renames on the way in, such as
    T-SQL ``CURRENT_TIMESTAMP`` -> ``GETDATE()``), and for replacements in a structured document
    (an Elasticsearch Query DSL dict) rather than in text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    original: str = Field(min_length=1)
    replaced_with: str = Field(min_length=1)
    position: int | None = None


class RewriteResult(BaseModel):
    """Rewritten SQL plus the audit trail of every substitution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sql: str
    replacements: tuple[Replacement, ...] = ()


def resolve_exec_ts(
    mode: ExecTsMode,
    *,
    clock: Callable[[], datetime] | None = None,
    stored: datetime | None = None,
) -> datetime:
    """Return the one UTC-aware ``exec_ts`` of this execution.

    Args:
        mode: ``frozen`` re-uses ``stored``; ``relative`` asks the injected ``clock``.
        clock: Injected clock for ``relative`` mode. Never ``datetime.now`` in here: the
            module stays pure so tests and re-runs are deterministic.
        stored: The stored ``exec_ts`` of a past run, required for ``frozen`` mode.

    Raises:
        ValueError: the mode's input is missing, or a timestamp is naive.
    """
    if mode is ExecTsMode.FROZEN:
        if stored is None:
            raise ValueError(
                "frozen mode needs the stored exec_ts of a past run: pass stored=<datetime>"
            )
        return _require_aware(stored, "stored exec_ts")
    if clock is None:
        raise ValueError(
            "relative mode needs an injected clock: pass clock=<callable returning datetime>"
        )
    return _require_aware(clock(), "clock()")


def rewrite_sql(
    sql: str,
    *,
    dialect: str,
    exec_ts: datetime,
    session_time_zone: str = "UTC",
) -> RewriteResult:
    """Replace every now-like value in ``sql`` with a literal derived from ``exec_ts``.

    Args:
        sql: Exactly one statement, in ``dialect``.
        dialect: A sqlglot dialect name, one of :data:`SUPPORTED_DIALECTS`.
        exec_ts: The single UTC-aware execution timestamp.
        session_time_zone: The source's session time zone; the literal is rendered in it,
            because that is how the source itself would have read its own clock.

    Returns:
        The rewritten SQL and one :class:`Replacement` per substitution.

    Raises:
        ValueError: unsupported dialect, unknown time zone, naive ``exec_ts``, or the input is
            not exactly one statement.
        sqlglot.errors.SqlglotError: the SQL does not parse in ``dialect``.
    """
    timestamp = _require_aware(exec_ts, "exec_ts")
    zone = _zone(session_time_zone)
    if dialect not in SUPPORTED_DIALECTS:
        raise ValueError(
            f"unsupported dialect {dialect!r} for exec_ts rewriting: "
            f"known dialects are {sorted(SUPPORTED_DIALECTS)}"
        )
    parsed = sqlglot.parse(sql, read=dialect, error_level=sqlglot.ErrorLevel.RAISE)
    if len(parsed) != 1 or parsed[0] is None:
        raise ValueError(
            f"exec_ts rewriting expects exactly one statement, got {len(parsed)}: "
            "the read-only guard blocks multi-statement input"
        )
    found: list[tuple[str, str, int | None]] = []
    statement = cast("exp.Expression", parsed[0])
    tree = _rewrite(statement, dialect=dialect, exec_ts=timestamp, zone=zone, found=found)
    rendered = _stable_text(tree, dialect)
    return RewriteResult(sql=rendered, replacements=_locate(found))


def rewrite_es_date_math(
    expression: str,
    *,
    exec_ts: datetime,
    time_zone: str = "UTC",
) -> str:
    """Turn one Elasticsearch date-math expression into an absolute UTC instant.

    Handles the ``now`` anchor with ``+``/``-`` arithmetic and the trailing rounding
    operators, applied in that order and last, as Elasticsearch documents them
    (``now-7d/d`` = seven days ago, rounded down to the start of that day).

    Args:
        expression: Date math such as ``now``, ``now-7d``, ``now-7d/d``, ``now+1M/M``.
        exec_ts: The single UTC-aware execution timestamp that replaces ``now``.
        time_zone: Time zone used for both the arithmetic and the rounding, matching the
            ``time_zone`` of the surrounding range query.

    Returns:
        An absolute instant, ``YYYY-MM-DDTHH:MM:SS.mmmZ``.

    Raises:
        ValueError: ``expression`` is not date math, or ``time_zone`` is unknown.
    """
    timestamp = _require_aware(exec_ts, "exec_ts")
    zone = _zone(time_zone)
    match = _ES_DATE_MATH.fullmatch(expression)
    if match is None:
        raise ValueError(
            f"not Elasticsearch date math: {expression!r} (expected e.g. 'now', 'now-7d/d')"
        )
    moment = timestamp.astimezone(zone)
    for sign, digits, unit in _ES_ARITHMETIC.findall(match.group("math")):
        amount = int(digits) * (1 if sign == "+" else -1)
        moment = _shift(moment, _es_offset(unit, amount))
    for unit in _ES_ROUNDING.findall(match.group("round")):
        moment = _floor(moment, unit)
    return _es_instant(moment)


def rewrite_es_query(
    query: Mapping[str, Any],
    *,
    exec_ts: datetime,
    time_zone: str = "UTC",
) -> tuple[dict[str, Any], tuple[Replacement, ...]]:
    """Return ``query`` with every date-math range bound replaced by an absolute instant.

    Only the range bounds (``gt``, ``gte``, ``lt``, ``lte``, ``from``, ``to``) are touched and
    only when their value is date math, so filter text and ``_source`` stay untouched. A
    ``time_zone`` found in the same range object is used for that bound; the rest of the
    document is copied unchanged.
    """
    replacements: list[Replacement] = []
    rewritten = _walk_dsl(query, exec_ts=exec_ts, zone_name=time_zone, found=replacements)
    return rewritten, tuple(replacements)


# ---------------------------------------------------------------------------------------------
# SQL rewriting internals
# ---------------------------------------------------------------------------------------------


class _Kind(StrEnum):
    """How a now-like value is read: a date, the session's local wall clock, or UTC."""

    DATE = "date"
    LOCAL = "local"
    UTC = "utc"


class _Absolute(NamedTuple):
    kind: _Kind
    value: datetime


class _Offset(NamedTuple):
    months: int
    micros: int

    def __neg__(self) -> _Offset:
        return _Offset(-self.months, -self.micros)


def _rewrite(
    node: exp.Expression,
    *,
    dialect: str,
    exec_ts: datetime,
    zone: ZoneInfo,
    found: list[tuple[str, str, int | None]],
) -> exp.Expression:
    """Fold or replace ``node`` in place; returns the node to keep in its parent slot."""
    absolute = _fold(node, dialect=dialect, exec_ts=exec_ts, zone=zone)
    if absolute is not None:
        literal = _literal(dialect, absolute)
        found.append(
            (
                node.sql(dialect=dialect),
                literal.sql(dialect=dialect),
                _first_offset(node, dialect),
            )
        )
        return literal
    for key, value in node.args.items():
        if isinstance(value, exp.Expression):
            node.set(key, _rewrite(value, dialect=dialect, exec_ts=exec_ts, zone=zone, found=found))
        elif isinstance(value, list):
            node.set(
                key,
                [
                    _rewrite(item, dialect=dialect, exec_ts=exec_ts, zone=zone, found=found)
                    if isinstance(item, exp.Expression)
                    else item
                    for item in value
                ],
            )
    return node


def _fold(
    node: exp.Expression, *, dialect: str, exec_ts: datetime, zone: ZoneInfo
) -> _Absolute | None:
    """The absolute value ``node`` denotes, or ``None`` when it is not a now-value."""
    kind = _now_kind(node, dialect)
    if kind is not None:
        return _absolute(kind, exec_ts, zone)
    if isinstance(node, (exp.Sub, exp.Add)):
        offset = _operand_offset(node.expression, dialect=dialect)
        signed = None if offset is None else (-offset if isinstance(node, exp.Sub) else offset)
    else:  # DATEADD(unit, amount, now) and its other spellings
        offset = _operand_offset(node, dialect=dialect)
        signed = None if offset is None else (-offset if isinstance(node, exp.DateSub) else offset)
    if signed is None:
        return None
    base = _fold(node.this, dialect=dialect, exec_ts=exec_ts, zone=zone)
    if base is None:
        return None
    folded_kind = _Kind.LOCAL if base.kind is _Kind.DATE else base.kind
    return _Absolute(folded_kind, _shift(base.value, signed))


def _now_kind(node: exp.Expr, dialect: str) -> _Kind | None:
    """Which now-like value ``node`` is, if any."""
    if isinstance(node, exp.CurrentDate):
        return _Kind.DATE
    if isinstance(node, exp.UtcTimestamp):
        return _Kind.UTC
    if isinstance(
        node,
        (exp.CurrentTimestamp, exp.Localtimestamp, exp.Systimestamp, exp.CurrentTimestampLTZ),
    ):
        return _Kind.LOCAL
    if isinstance(node, exp.Anonymous) and node.name.lower() in _NOW_NAMES[dialect]:
        return _Kind.UTC if node.name.lower() in _UTC_NAMES else _Kind.LOCAL
    return None


def _absolute(kind: _Kind, exec_ts: datetime, zone: ZoneInfo) -> _Absolute:
    if kind is _Kind.UTC:
        return _Absolute(kind, exec_ts.astimezone(UTC))
    return _Absolute(kind, exec_ts.astimezone(zone))


def _operand_offset(node: exp.Expression, *, dialect: str) -> _Offset | None:
    """The literal offset a now-value is shifted by, or ``None`` if it is not literal.

    ``date - <number>`` is only read as days in the dialects that mean that by it, so an
    Oracle or T-SQL arithmetic expression is never guessed from a PostgreSQL reading.
    """
    if isinstance(node, exp.Interval):
        unit = _unit_name(node.args.get("unit"))
        amount = _number(node.this)
        return None if unit is None or amount is None else _offset_for(unit, amount)
    if isinstance(node, (exp.DateAdd, exp.DateSub)):
        unit = _unit_name(node.args.get("unit"))
        amount = _number(node.expression)
        return None if unit is None or amount is None else _offset_for(unit, amount)
    if dialect in _NUMBER_IS_DAYS:
        days = _days(node)
        if days is None:
            return None
        return _Offset(0, _micros(days * _MICROS_PER_DAY))
    return None


def _offset_for(unit: str, amount: Fraction | int) -> _Offset:
    """Calendar months for month/year units, exact microseconds for the rest."""
    upper = unit.upper()
    if upper in _MONTHS_PER_UNIT:
        return _Offset(_MONTHS_PER_UNIT[upper] * int(amount), 0)
    return _Offset(0, _micros(Fraction(amount) * _MICROS_PER_UNIT[upper]))


def _es_offset(unit: str, amount: int) -> _Offset:
    """Elasticsearch units: ``y`` and ``M`` are calendar, the rest are exact elapsed time."""
    if unit == "y":
        return _Offset(12 * amount, 0)
    if unit == "M":
        return _Offset(amount, 0)
    if unit == "w":
        return _Offset(0, amount * 7 * _MICROS_PER_DAY)
    if unit == "d":
        return _Offset(0, amount * _MICROS_PER_DAY)
    if unit in ("h", "H"):
        return _Offset(0, amount * _MICROS_PER_UNIT["HOUR"])
    if unit == "m":
        return _Offset(0, amount * _MICROS_PER_UNIT["MINUTE"])
    return _Offset(0, amount * _MICROS_PER_UNIT["SECOND"])


def _days(node: exp.Expression) -> Fraction | None:
    """A day count written as a number, a negative number, or a fraction such as ``1/24``."""
    if isinstance(node, exp.Div):
        left = _number(node.this)
        right = _number(node.expression)
        if left is None or right is None or right == 0:
            return None
        return left / right
    return _number(node)


def _number(node: exp.Expression | None) -> Fraction | None:
    if node is None:
        return None
    if isinstance(node, exp.Neg):
        inner = _number(node.this)
        return None if inner is None else -inner
    if isinstance(node, exp.Paren):
        return _number(node.this)
    if isinstance(node, exp.Literal):
        text = node.this if isinstance(node.this, str) else str(node.this)
        try:
            return Fraction(text)
        except (ValueError, ZeroDivisionError):
            return None
    return None


def _unit_name(node: exp.Expression | None) -> str | None:
    """``Var(HOUR)``, ``Literal('DAY')`` and ``Column(day)`` all name a unit."""
    if node is None:
        return None
    name = node.name.upper() if node.name else ""
    if name in _MICROS_PER_UNIT or name in _MONTHS_PER_UNIT:
        return name
    return None


def _micros(value: Fraction) -> int:
    """Exact microseconds for a Fraction; anything finer rounds to the nearest microsecond."""
    return round(value)


def _shift(value: datetime, offset: _Offset) -> datetime:
    """Calendar months on the wall clock, then elapsed time on the instant.

    Months keep the time of day the way SQL engines add them; hours and days are elapsed time,
    so a 24h step across a DST change really is 24h of elapsed time, not a 23 or 25 hour day.
    """
    shifted = _add_months(value, offset.months) if offset.months else value
    if offset.micros:
        zone = shifted.tzinfo
        shifted = (shifted.astimezone(UTC) + timedelta(microseconds=offset.micros)).astimezone(zone)
    return shifted


def _add_months(value: datetime, months: int) -> datetime:
    """Calendar months, clamping the day (31 January + 1 month is 28/29 February)."""
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, _days_in_month(year, month))
    return value.replace(year=year, month=month, day=day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (datetime(year, month + 1, 1) - datetime(year, month, 1)).days


def _literal(dialect: str, absolute: _Absolute) -> exp.Expression:
    """A literal of the same kind as the value it replaces, valid in ``dialect``."""
    if absolute.kind is _Kind.DATE:
        return exp.Cast(
            this=exp.Literal.string(absolute.value.date().isoformat()),
            to=exp.DataType.build("DATE"),
        )
    return exp.Cast(
        this=exp.Literal.string(_render_timestamp(absolute)),
        to=exp.DataType.build(_TIMESTAMP_TYPE[dialect], dialect=dialect),
    )


def _render_timestamp(absolute: _Absolute) -> str:
    naive = absolute.value.replace(tzinfo=None)
    text = naive.strftime("%Y-%m-%d %H:%M:%S")
    return f"{text}.{naive.microsecond:06d}" if naive.microsecond else text


def _stable_text(tree: exp.Expression, dialect: str) -> str:
    text = tree.sql(dialect=dialect)
    for _ in range(_MAX_RENDER_PASSES):
        again = sqlglot.parse_one(text, read=dialect, error_level=sqlglot.ErrorLevel.RAISE).sql(
            dialect=dialect
        )
        if again == text:
            break
        text = again
    return text


def _first_offset(node: exp.Expression, dialect: str) -> int | None:
    """Offset of the expression's first character, as recorded by sqlglot's parser.

    A binary arithmetic node has no position of its own, so the position of the now-like value
    it contains is used; that value is the leftmost thing the parser saw. Bare keywords
    (``CURRENT_DATE``, Oracle ``SYSDATE``) carry no position at all, and the caller then looks
    for the expression in the source text.
    """
    own = node.meta.get("start")
    if isinstance(own, int):
        return own
    for child in node.walk():
        if _now_kind(child, dialect) is None:
            continue
        start = child.meta.get("start")
        return start if isinstance(start, int) else None
    return None


def _locate(found: list[tuple[str, str, int | None]]) -> tuple[Replacement, ...]:
    """Turn the collected pairs into audit entries.

    ``position`` stays ``None`` when the dialect's parser recorded none for the expression
    (bare keywords such as ``CURRENT_DATE`` or Oracle ``SYSDATE``, and spellings the dialect
    renames on the way in, such as T-SQL ``CURRENT_TIMESTAMP`` -> ``GETDATE()``). Searching the
    source text instead was tried and dropped: with two identical functions in one statement it
    can point at the wrong one, and a wrong offset is worse than a missing one.
    """
    return tuple(
        Replacement(original=original, replaced_with=replaced_with, position=position)
        for original, replaced_with, position in found
    )


# ---------------------------------------------------------------------------------------------
# Elasticsearch date math
# ---------------------------------------------------------------------------------------------

_ES_DATE_MATH: Final = re.compile(
    r"now(?P<math>(?:[+\-]\d+[yMwdhHms])*)(?P<round>(?:/[yMwdhHms])*)"
)
_ES_ARITHMETIC: Final = re.compile(r"([+\-])(\d+)([yMwdhHms])")
_ES_ROUNDING: Final = re.compile(r"/([yMwdhHms])")
_ES_BOUND_KEYS: Final[frozenset[str]] = frozenset({"from", "gt", "gte", "lt", "lte", "to"})


def _es_instant(moment: datetime) -> str:
    utc = moment.astimezone(UTC)
    return f"{utc.strftime('%Y-%m-%dT%H:%M:%S')}.{utc.microsecond // 1000:03d}Z"


def _floor(moment: datetime, unit: str) -> datetime:
    """Round down in the session zone, the way Elasticsearch applies its rounding operators.

    The local wall-clock replace keeps the zone, so a zone that shifts at midnight floors to
    the same instant Elasticsearch picks (fold=0); no DST transition happens on a first-of-month.
    """
    if unit == "s":
        return moment.replace(microsecond=0)
    if unit == "m":
        return moment.replace(second=0, microsecond=0)
    if unit in ("h", "H"):
        return moment.replace(minute=0, second=0, microsecond=0)
    if unit == "d":
        return moment.replace(hour=0, minute=0, second=0, microsecond=0)
    if unit == "w":
        start_of_day = moment.replace(hour=0, minute=0, second=0, microsecond=0)
        return start_of_day - timedelta(days=start_of_day.weekday())
    if unit == "M":
        return moment.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return moment.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def _walk_dsl(
    node: Any,
    *,
    exec_ts: datetime,
    zone_name: str,
    found: list[Replacement],
) -> Any:
    if isinstance(node, Mapping):
        local_zone = node.get("time_zone")
        zone = local_zone if isinstance(local_zone, str) else zone_name
        rewritten: dict[str, Any] = {}
        for key, value in node.items():
            if (
                isinstance(key, str)
                and key in _ES_BOUND_KEYS
                and isinstance(value, str)
                and _ES_DATE_MATH.fullmatch(value)
            ):
                absolute = rewrite_es_date_math(value, exec_ts=exec_ts, time_zone=zone)
                found.append(Replacement(original=value, replaced_with=absolute, position=None))
                rewritten[key] = absolute
            else:
                rewritten[key] = _walk_dsl(value, exec_ts=exec_ts, zone_name=zone, found=found)
        return rewritten
    if isinstance(node, list):
        return [_walk_dsl(item, exec_ts=exec_ts, zone_name=zone_name, found=found) for item in node]
    return node


# ---------------------------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------------------------


def _require_aware(value: datetime, what: str) -> datetime:
    """Timestamps are UTC-aware internally (AGENTS.md section 5)."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{what} must be timezone-aware; timestamps are UTC-aware internally")
    return value.astimezone(UTC)


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError(f"unknown time zone {name!r}: {error}") from error
