"""The per-dialect rules the compiler needs on top of sqlglot (task C10).

``sqlglot`` translates most of the AST and then silently passes the rest through: for T-SQL it
emits ``LPAD(...)`` and drops ``NULLS FIRST``, for Oracle it emits ``TIMESTAMP_TRUNC(...)`` and
``TRUNC(x, 'WEEK')``. None of those exist, or mean the same thing, on the real engine, and a wrong
number is worse than a refusal (AGENTS.md section 2, principle 10).

So the compiler writes the handful of constructs it emits itself (:func:`timestamp_literal`,
:func:`trunc_expression`), rewrites the one function a semantic template uses that the target lacks
(:func:`rewrite`), and where no faithful translation exists the caller refuses: the rule functions
return ``None`` instead of guessing.

Every render goes through :func:`render`, which sets ``unsupported_level=RAISE``, so anything
sqlglot itself knows it cannot translate raises instead of degrading.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Final

from sqlglot import exp
from sqlglot.errors import ErrorLevel

__all__ = [
    "ORACLE_TRUNC_MASKS",
    "TSQL_TRUNC_UNITS",
    "render",
    "rewrite",
    "timestamp_literal",
    "trunc_expression",
]

#: Oracle ``TRUNC`` takes a format mask, not a unit name. ``'IW'`` is the ISO Monday week, which is
#: what PostgreSQL's ``date_trunc('week')`` and the time planner mean by a week.
ORACLE_TRUNC_MASKS: Final[Mapping[str, str]] = {
    "day": "DD",
    "week": "IW",
    "month": "MM",
    "quarter": "Q",
    "year": "YYYY",
}

#: Units T-SQL ``DATETRUNC`` truncates to a fixed boundary. ``week`` is left out on purpose: its
#: start comes from the session's ``DATEFIRST``, so the same statement means different weeks on two
#: connections and the caller must refuse it.
#: shortcut: DATETRUNC needs SQL Server 2022 (compatibility level 160). Serving an older instance
#: means a DATEADD/DATEDIFF form here, which is why the version belongs in the capability probe.
TSQL_TRUNC_UNITS: Final[frozenset[str]] = frozenset({"day", "month", "quarter", "year"})

#: Oracle's mask for a timestamp literal, matched by the fraction :func:`timestamp_literal` writes.
_ORACLE_TS_FORMAT: Final[str] = "YYYY-MM-DD HH24:MI:SS.FF6"

#: Dialects whose session settings decide how ``CAST('...' AS TIMESTAMP)`` is read.
_SESSION_FORMAT_DIALECTS: Final[frozenset[str]] = frozenset({"oracle", "tsql"})

#: The cast targets that mean "a point in time", as sqlglot names them.
_INSTANT_TYPES: Final[tuple[str, ...]] = ("datetime", "datetime2", "timestamp", "timestamptz")


def timestamp_literal(moment: datetime, dialect: str) -> exp.Expression:
    """A naive UTC instant as a literal the target reads the same way whatever its session settings.

    A cast is explicit because a bare string compares as text on Oracle and SQL Server. On Oracle
    the cast is not enough: ``CAST('...' AS TIMESTAMP)`` goes through ``NLS_TIMESTAMP_FORMAT``, so
    the same statement means a different instant (or fails) on a differently configured session,
    hence ``TO_TIMESTAMP`` with an explicit mask. T-SQL gets the ISO 8601
    ``yyyy-mm-ddThh:mm:ss`` form, which is the one ``SET DATEFORMAT`` cannot re-read.
    """
    if moment.tzinfo is not None:  # one instant, one spelling: UTC
        moment = moment.astimezone(UTC).replace(tzinfo=None)
    if dialect == "oracle":
        return exp.Anonymous(
            this="TO_TIMESTAMP",
            expressions=[
                exp.Literal.string(moment.strftime("%Y-%m-%d %H:%M:%S.%f")),
                exp.Literal.string(_ORACLE_TS_FORMAT),
            ],
        )
    if dialect == "tsql":
        text = moment.strftime("%Y-%m-%dT%H:%M:%S.%f")
    else:
        text = moment.isoformat(sep=" ")
    return exp.cast(exp.Literal.string(text), exp.DataType.build("TIMESTAMP"))


def trunc_expression(unit: str, column: exp.Expression, dialect: str) -> exp.Expression | None:
    """The start of ``unit`` for a time column, or ``None`` when the dialect has no faithful form.

    ``None`` is the refusal: T-SQL cannot start a week at a fixed day, so a week bucket there would
    silently group by a different week than the same query on PostgreSQL.
    """
    key = unit.strip().lower()
    if dialect == "oracle":
        mask = ORACLE_TRUNC_MASKS.get(key)
        if mask is None:
            return None
        # TRUNC, not TO_CHAR/EXTRACT: it keeps the value a date, so it compares and sorts like one.
        return exp.Anonymous(
            this="TRUNC",
            expressions=[column.copy(), exp.Literal.string(mask)],
        )
    if dialect == "tsql":
        if key not in TSQL_TRUNC_UNITS:
            return None
    elif dialect != "oracle":
        return exp.TimestampTrunc(unit=exp.var(key.upper()), this=column)
    # T-SQL keeps sqlglot's own node: it renders DATETRUNC, which the tsql generator handles.
    return exp.TimestampTrunc(unit=exp.var(key.upper()), this=column)


def rewrite(node: exp.Expression, dialect: str) -> exp.Expression:
    """Apply the rewrites this target dialect needs to the assembled statement, if any.

    The time planner and the partition planner (:mod:`mdqm.planner`) are dialect-free: they emit a
    generic ``CAST('<iso>' AS TIMESTAMP)``, which is exact on PostgreSQL and reads through a session
    format mask on Oracle. Turning that into the dialect's own literal is this module's job, so both
    planner paths are covered by one rule.
    """
    if dialect not in _SESSION_FORMAT_DIALECTS:
        return node

    def from_cast(expression: exp.Expression) -> exp.Expression:
        moment = _cast_instant(expression)
        # ``transform`` needs every callback to return a node: returning None is not "unchanged".
        return expression if moment is None else timestamp_literal(moment, dialect)

    rewritten = node.transform(from_cast, copy=True)
    if dialect != "tsql":
        return rewritten
    return rewritten.transform(_pad_to_tsql, copy=True)


def _cast_instant(node: exp.Expression) -> datetime | None:
    """The instant a ``CAST('<iso text>' AS TIMESTAMP)`` names, or ``None`` for anything else."""
    if not isinstance(node, exp.Cast) or not node.to.is_type(*_INSTANT_TYPES):
        return None
    text = node.this
    if not isinstance(text, exp.Literal) or not text.is_string:
        return None
    try:
        return datetime.fromisoformat(str(text.this))
    except ValueError:  # not an instant we wrote: leave the cast exactly as it was
        return None


def _pad_to_tsql(node: exp.Expression) -> exp.Expression:
    """``LPAD(x, n, fill)`` as the T-SQL it always meant:
    ``RIGHT(REPLICATE(fill, n - LEN(x)) + x, n)``.

    Semantic join templates pad a key to a fixed width (``LPAD(cust_id, 10, '0')``); T-SQL has
    neither ``LPAD`` nor ``RPAD``, and sqlglot passes the name through unchanged, which is a
    guaranteed "Invalid object name" at run time.

    shortcut: ``LEN`` ignores trailing spaces, so a key that ends in a space pads by that many
    characters less. Join keys are trimmed by their catalog, so this is a documented limit rather
    than a silent one.
    """
    if not isinstance(node, exp.Pad):
        return node
    text = node.this
    length = node.args.get("expression")
    if length is None:
        return node
    fill = node.args.get("fill_pattern") or exp.Literal.string(" ")
    padding = exp.Repeat(
        this=fill,
        times=exp.Sub(this=length.copy(), expression=exp.Length(this=text)),
    )
    if node.args.get("is_left", True) is False:
        return exp.Left(this=exp.DPipe(this=text, expression=padding), expression=length)
    return exp.Right(this=exp.DPipe(this=padding, expression=text), expression=length)


def render(query: exp.Expression, dialect: str) -> str:
    """Render the statement once in the target dialect; a construct it cannot express raises."""
    return query.sql(dialect=dialect, unsupported_level=ErrorLevel.RAISE)
