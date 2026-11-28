"""Partition planning: refuse an unpruned scan, rewrite a wrapped key, estimate what it reads.

Scanning every partition is direct load on a primary database that has no replica, so this module
answers three questions *before* anything runs (docs/DESIGN.md sections 5 and 7):

1. **Does the request filter the partition key at all?** No filter is either refused
   (``PARTITION_SCAN_ALL``, with the ``partitions_if_fixed`` estimate and a ``fix`` that says what
   to add) or answered with the table's ``default_window``, per ``partitioning.require_key_filter``.
2. **Can the source prune with what it was given?** ``CAST(start_time AS DATE) = ...``,
   ``TRUNC(start_time)`` and ``date_trunc('month', start_time) = ...`` all hide the raw column
   behind a function, and a function on the key defeats pruning. Each is turned into a predicate on
   the **raw** column; the original predicate is kept, so the added range is *implied* by it and the
   result is unchanged by construction (proved on the D02 world in DuckDB).
3. **How many partitions and rows would that read?** Counted from the declared ``bounds`` and
   compared with ``max_partitions`` (``COST_TOO_HIGH`` when it is over the cap).

Hash partitions need a filter on the key itself (a range does not prune); an expression key prunes
only when the predicate matches the declared expression, otherwise it goes to human review. Tables
with ``strategy=none`` have no partitions: the cost gate handles them.

Pure: no I/O, no clock read. ``exec_ts`` is injected once per execution (C02), so the injected
default window and the estimates are reproducible. Sibling of :mod:`mdqm.planner.time`, which
resolves *which* column a time intent filters; this module asks what that costs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Final

import sqlglot
from sqlglot import errors as sqlglot_errors
from sqlglot import exp

from mdqm.errors import cost_too_high, partition_scan_all
from mdqm.models import (
    ErrorResponse,
    PartitionBound,
    PartitionGranularity,
    Partitioning,
    PartitionStrategy,
    PlanSummary,
)
from mdqm.planner.sizes import fat_partitions
from mdqm.spec.durations import Duration

__all__ = [
    "PartitionEstimate",
    "PartitionNote",
    "PartitionPlan",
    "Range",
    "align_to_partitions",
    "estimate_partition_scan",
    "gate_plan",
    "plan_partitions",
]


class PartitionNote(StrEnum):
    """A caveat the caller must show the agent (principle 10: nothing silent)."""

    DEFAULT_WINDOW_APPLIED = "default_window_applied"
    """No key filter, and the table allows one: a window was added to the query."""
    KEY_FUNCTION_REWRITTEN = "key_function_rewritten"
    """A function or cast on the key was turned into a range on the raw column so pruning works."""
    NEEDS_HUMAN_APPROVAL = "needs_human_approval"
    """The key is filtered, but not in a shape this planner can prove prunable."""
    ESTIMATE_UNPROVEN = "estimate_unproven"
    """The partition count is an upper bound, not a proof (hash partitioning, multi-column key)."""
    SIZE_PROFILE_APPLIED = "size_profile_applied"
    """The row and byte estimate comes from a measured size profile, not the catalog's ``reltuples``."""
    FAT_PARTITION = "fat_partition"
    """A partition in the window is far bigger than the table's median: the window is cheaper if the
    filter is narrowed inside it. :attr:`PartitionPlan.fat_partitions` names them, largest first,
    so the agent can say which one to narrow."""


@dataclass(frozen=True, slots=True)
class Range:
    """A half-open UTC window ``[lower, upper)``: a partition scope, a cache segment."""

    lower: datetime
    upper: datetime

    def __post_init__(self) -> None:
        """Reject a naive or inverted window: both are programming bugs, not agent input."""
        for name, moment in (("lower", self.lower), ("upper", self.upper)):
            if moment.tzinfo is None:
                raise ValueError(f"Range.{name} must be timezone-aware UTC; naive is a bug")
        if self.upper < self.lower:
            raise ValueError(
                f"Range upper {self.upper.isoformat()} is before lower {self.lower.isoformat()}"
            )

    @property
    def is_empty(self) -> bool:
        """True when the window contains no instant."""
        return self.upper <= self.lower

    @property
    def length(self) -> timedelta:
        """How wide the window is (``timedelta(0)`` when empty)."""
        return self.upper - self.lower


@dataclass(frozen=True, slots=True)
class PartitionEstimate:
    """What a scan would read, counted from declared bounds and never from a run.

    ``partitions`` is ``None`` when the plan established no window at all; ``total`` is ``None``
    when the probe could not enumerate the table's partitions.
    """

    partitions: int | None = None
    total: int | None = None
    rows: int | None = None
    bytes: int | None = None
    """Bytes the window would read, when a measured size profile said so. ``None`` otherwise: a
    catalog estimate knows rows, never bytes."""
    estimated_from: str | None = None
    """``'profile'`` when a measured size answered, ``'catalog'`` when the declared ``reltuples`` did,
    ``None`` when nothing did. Reported so an estimate is never mistaken for a measurement."""
    incomplete: bool = False
    """True when a partition in the window reported no size at all, so ``rows`` is a lower bound.

    Kept separate from ``rows is None``: a partial answer is still useful, and hiding that it is
    partial is how an agent ends up trusting half a number."""

    @property
    def label(self) -> str | None:
        """``'2 of 60'``: the shape DESIGN section 5 puts in ``partitions_if_fixed``."""
        if self.partitions is None:
            return None
        total = "unknown" if self.total is None else str(self.total)
        return f"{self.partitions} of {total}"

    @property
    def summary(self) -> str:
        """One phrase for an error message: ``'2 of 60 partitions, ~1200 rows'``."""
        text = f"{self.label or 'an unknown number of'} partitions"
        return f"{text}, about {self.rows} rows" if self.rows else text


@dataclass(frozen=True, slots=True)
class PartitionPlan:
    """What the planner decided. ``predicates`` are *additional* sqlglot predicates to AND in.

    They are implied by the fragment's own predicates (or, for ``default_window_applied``, they are
    the window this plan chose), so adding them cannot change the answer; they exist so the source
    can prune. ``error`` is mutually exclusive with a runnable plan.
    """

    predicates: tuple[exp.Expression, ...] = ()
    notes: tuple[PartitionNote, ...] = ()
    applied_fixes: tuple[str, ...] = ()
    error: ErrorResponse | None = None
    estimate: PartitionEstimate | None = None
    scope: Range | None = None
    segments: tuple[Range, ...] = ()
    fat_partitions: tuple[str, ...] = ()
    """The partitions in the window that hold at least :data:`~mdqm.planner.sizes.FAT_RATIO`
    times the table's median: what :attr:`PartitionNote.FAT_PARTITION` names, largest first.

    Empty without a measured size profile -- the catalog's ``reltuples`` has no median to compare
    against -- so a plan that was never measured keeps the notes it always had.
    """

    @property
    def ok(self) -> bool:
        """True when nothing blocks the run. False means: do not call the source."""
        return self.error is None

    def where_sql(self, dialect: str = "duckdb") -> str:
        """The added predicates rendered as a SQL ``WHERE`` body (empty when there are none)."""
        return " AND ".join(predicate.sql(dialect=dialect) for predicate in self.predicates)


# --- granularity arithmetic -------------------------------------------------------------

_FIXED_WIDTH: Final[dict[PartitionGranularity, timedelta]] = {
    PartitionGranularity.HOUR: timedelta(hours=1),
    PartitionGranularity.DAY: timedelta(days=1),
    PartitionGranularity.WEEK: timedelta(days=7),
}
_MONTHS_PER: Final[dict[PartitionGranularity, int]] = {
    PartitionGranularity.MONTH: 1,
    PartitionGranularity.QUARTER: 3,
    PartitionGranularity.YEAR: 12,
}
_GRANULARITIES: Final[dict[str, PartitionGranularity]] = {
    granularity.value: granularity for granularity in PartitionGranularity
}
#: Every ``date_trunc``-family node. sqlglot picks one from the operand's type, so match them all.
_TRUNC_NODES: Final[tuple[type[exp.Expr], ...]] = (
    exp.TimestampTrunc,
    exp.DateTrunc,
    exp.DatetimeTrunc,
    exp.TimeTrunc,
)


def _floor(moment: datetime, granularity: PartitionGranularity) -> datetime:
    """The start of the partition ``moment`` falls in. Weeks start on Monday, as SQL does."""
    utc = moment.astimezone(UTC)
    midnight = utc.replace(hour=0, minute=0, second=0, microsecond=0)
    if granularity is PartitionGranularity.HOUR:
        return utc.replace(minute=0, second=0, microsecond=0)
    if granularity is PartitionGranularity.DAY:
        return midnight
    if granularity is PartitionGranularity.WEEK:
        return midnight - timedelta(days=utc.weekday())
    months_back = (utc.month - 1) % _MONTHS_PER[granularity]
    return _add_months(midnight.replace(day=1), -months_back)


def _next_boundary(moment: datetime, granularity: PartitionGranularity) -> datetime:
    """The first partition boundary strictly after ``moment``."""
    floored = _floor(moment, granularity)
    width = _FIXED_WIDTH.get(granularity)
    if width is not None:
        return floored + width
    return _add_months(floored, _MONTHS_PER[granularity])


def _add_months(moment: datetime, months: int) -> datetime:
    """Shift by whole calendar months. Only called on a month start, so no day clamping."""
    index = moment.year * 12 + (moment.month - 1) + months
    year, month = divmod(index, 12)
    return moment.replace(year=year, month=month + 1)


def align_to_partitions(
    window: Range, granularity: PartitionGranularity = PartitionGranularity.MONTH
) -> tuple[Range, ...]:
    """Split ``window`` into consecutive pieces whose inner boundaries are partition boundaries.

    The cache stores one segment per piece, so a segment never straddles two partitions and its
    freshness can be judged against the piece it covers. The pieces cover exactly ``window``: the
    first starts at ``window.lower`` and the last ends at ``window.upper``, so nothing outside the
    asked-for range is fetched and nothing inside it is dropped.
    """
    pieces: list[Range] = []
    cursor = window.lower
    while cursor < window.upper:
        end = min(_next_boundary(cursor, granularity), window.upper)
        pieces.append(Range(cursor, end))
        cursor = end
    return tuple(pieces)


# --- estimation ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Window:
    """A constraint on the key: an inclusive lower and an (in/ex)clusive upper, either optional."""

    lower: datetime | None = None
    upper: datetime | None = None
    upper_inclusive: bool = False

    @property
    def bounded(self) -> bool:
        """True when at least one end is known, i.e. the predicate prunes at all."""
        return self.lower is not None or self.upper is not None

    def touches(self, bound: PartitionBound) -> bool:
        """True when partition ``bound`` can hold a row the window allows."""
        if self.lower is not None and bound.upper is not None and bound.upper <= self.lower:
            return False
        if self.upper is None or bound.lower is None:
            return True
        if self.upper_inclusive:
            return bound.lower <= self.upper
        return bound.lower < self.upper


def _touched(partitioning: Partitioning, window: _Window | None) -> tuple[PartitionBound, ...]:
    """The declared partitions ``window`` can touch (``window is None``: all of them)."""
    if window is None:
        return partitioning.bounds
    return tuple(bound for bound in partitioning.bounds if window.touches(bound))


def _fat(
    partitioning: Partitioning,
    window: _Window | None,
    sizes: Mapping[str, tuple[int | None, int | None]] | None,
) -> tuple[str, ...]:
    """The touched partitions at least ``FAT_RATIO`` times the table's median, largest first.

    The median is taken over the **whole** measured profile, not only the touched partitions: a
    window over two partitions that both hold 40 000 rows has no fat one, and a window over the one
    partition holding 20x the median is the case the note exists for. Empty without a profile, so
    nothing is reported that was not measured.
    """
    if not sizes or not partitioning.bounds:
        return ()
    touched = {bound.name for bound in _touched(partitioning, window)}
    return tuple(name for name in fat_partitions(sizes) if name in touched)


def _estimate(
    partitioning: Partitioning,
    window: _Window | None,
    sizes: Mapping[str, tuple[int | None, int | None]] | None = None,
) -> PartitionEstimate:
    """Count the partitions (and their rows and bytes) that ``window`` can touch.

    ``sizes`` is a measured profile, ``{partition name: (rows, bytes)}``. It wins over the catalog's
    ``reltuples`` because it is ours, it is dated, and it knows bytes; a partition the profile does
    not mention falls back to the bound's own estimate, so a fresh partition is never estimated as
    free.
    """
    if not partitioning.bounds:
        return PartitionEstimate()
    touched = _touched(partitioning, window)
    measured = sizes or {}
    rows = 0
    total_bytes = 0
    incomplete = False
    counted = 0
    for bound in touched:
        from_profile = measured.get(bound.name)
        if from_profile is not None:
            if from_profile[0] is None:
                incomplete = True
                continue
            rows += from_profile[0]
            total_bytes += from_profile[1] or 0
            counted += 1
            continue
        if bound.row_estimate is None:
            incomplete = True
            continue
        rows += bound.row_estimate
        counted += 1
    source = None
    if counted and all(bound.name in measured for bound in touched):
        source = "profile"
    elif counted:
        source = "catalog"
    return PartitionEstimate(
        partitions=len(touched),
        total=partitioning.partitions_total,
        rows=rows or None,
        bytes=total_bytes or None,
        estimated_from=source,
        incomplete=incomplete,
    )


def estimate_partition_scan(
    partitioning: Partitioning,
    window: Range | None = None,
    sizes: Mapping[str, tuple[int | None, int | None]] | None = None,
) -> PartitionEstimate:
    """The static scan estimate for ``window`` (``None``: every declared partition)."""
    if window is None:
        return _estimate(partitioning, None, sizes)
    return _estimate(partitioning, _Window(lower=window.lower, upper=window.upper), sizes)


# --- reading the fragment's predicates ---------------------------------------------------

#: sqlglot node -> the comparison it expresses, as seen from the key's side.
_COMPARISONS: Final[dict[type[exp.Expr], str]] = {
    exp.GTE: "gte",
    exp.GT: "gt",
    exp.LTE: "lte",
    exp.LT: "lt",
    exp.EQ: "eq",
}
_FLIPPED: Final[dict[str, str]] = {"gte": "lte", "gt": "lt", "lte": "gte", "lt": "gt", "eq": "eq"}


@dataclass(frozen=True, slots=True)
class _Constraint:
    """One comparison against a constant, as it constrains the raw key column.

    ``unit`` is set when the predicate wrapped the key in a floor (``CAST(... AS DATE)``,
    ``date_trunc('month', ...)``, ``TRUNC(...)``); that is the case a rewrite is needed for.
    """

    column: str
    unit: PartitionGranularity | None
    op: str
    value: datetime


def _conjuncts(predicates: Sequence[exp.Expr]) -> tuple[exp.Expr, ...]:
    """Flatten ``a AND b`` into its terms, so each one can be read on its own."""
    out: list[exp.Expr] = []
    for predicate in predicates:
        node = predicate.unnest()
        if isinstance(node, exp.And):
            out.extend(_conjuncts((node.this, node.expression)))
        elif not isinstance(node, exp.Boolean):
            out.append(node)
    return tuple(out)


def _columns(node: exp.Expr) -> frozenset[str]:
    """The names of every column in ``node`` (qualifiers dropped: the fragment names one table)."""
    return frozenset(column.name for column in node.find_all(exp.Column))


def _mentions(node: exp.Expr, columns: frozenset[str]) -> bool:
    """True when ``node`` reads any of ``columns`` somewhere inside an expression."""
    return bool(_columns(node) & columns)


def _key_side(
    node: exp.Expr, columns: frozenset[str]
) -> tuple[str, PartitionGranularity | None] | None:
    """``(column, floor unit)`` when ``node`` is a key column, optionally floored; else ``None``.

    ``unit`` is ``None`` for the raw column. A cast to date and a bare ``TRUNC`` floor to a day; a
    ``date_trunc`` floors to its unit, and nesting floors take the coarser one, because
    ``date_trunc('month', CAST(x AS DATE))`` is a month floor.
    """
    target = node.unnest()
    if isinstance(target, exp.Column):
        return (target.name, None) if target.name in columns else None
    if isinstance(target, exp.Cast):
        inner = _key_side(target.this, columns)
        if inner is None:
            return None
        if target.to.is_type(exp.DataType.Type.DATE):
            return (inner[0], _coarser(inner[1], PartitionGranularity.DAY))
        if target.to.is_type(
            exp.DataType.Type.TIMESTAMP,
            exp.DataType.Type.TIMESTAMPTZ,
            exp.DataType.Type.TIMESTAMPLTZ,
            exp.DataType.Type.DATETIME,
        ):
            return inner
        return None
    if isinstance(target, _TRUNC_NODES):
        # sqlglot picks DateTrunc or TimestampTrunc from the operand's type: same meaning here.
        inner = _key_side(target.this, columns)
        unit = _granularity(target.args.get("unit"))
        return None if inner is None or unit is None else (inner[0], _coarser(inner[1], unit))
    if isinstance(target, exp.Trunc):
        inner = _key_side(target.this, columns)
        return None if inner is None else (inner[0], _coarser(inner[1], PartitionGranularity.DAY))
    return None


def _coarser(
    first: PartitionGranularity | None, second: PartitionGranularity
) -> PartitionGranularity:
    """The coarser of two floor units: ``date_trunc('month', CAST(x AS DATE))`` is a month."""
    if first is None:
        return second
    order = list(PartitionGranularity)
    return first if order.index(first) >= order.index(second) else second


def _granularity(node: exp.Expr | None) -> PartitionGranularity | None:
    """The granularity a ``date_trunc`` unit names, or ``None`` when it names something else."""
    return None if node is None else _GRANULARITIES.get(node.name.lower())


def _instant(node: exp.Expr) -> datetime | None:
    """The UTC instant a constant names, or ``None`` when it is not a date/timestamp literal."""
    target = node.unnest()
    while isinstance(target, (exp.Cast, exp.TsOrDsToDate)):
        target = target.this.unnest()
    if not isinstance(target, exp.Literal) or not target.is_string:
        return None
    try:
        moment = datetime.fromisoformat(target.this)
    except ValueError:
        return None
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _literal_text(node: exp.Expr) -> str | None:
    """The text of a constant side, as the declared partition values are written."""
    target = node.unnest()
    while isinstance(target, exp.Cast):
        target = target.this.unnest()
    return target.this if isinstance(target, exp.Literal) else None


def _constraints(node: exp.Expr, columns: frozenset[str]) -> tuple[_Constraint, ...]:
    """Every comparison in one conjunct that constrains a key column against a constant."""
    target = node.unnest()
    if isinstance(target, exp.Between):
        side = _key_side(target.this, columns)
        low = _instant(target.args["low"])
        high = _instant(target.args["high"])
        if side is None:
            return ()
        out: list[_Constraint] = []
        if low is not None:
            out.append(_Constraint(side[0], side[1], "gte", low))
        if high is not None:
            out.append(_Constraint(side[0], side[1], "lte", high))
        return tuple(out)
    if isinstance(target, exp.In):
        side = _key_side(target.this, columns)
        if side is None:
            return ()
        values = [_instant(item) for item in target.expressions]
        instants = [value for value in values if value is not None]
        if len(instants) != len(values) or not instants:
            return ()
        return (
            _Constraint(side[0], side[1], "gte", min(instants)),
            _Constraint(side[0], side[1], "lte", max(instants)),
        )
    op = _COMPARISONS.get(type(target))
    if op is None:
        return ()
    left = _key_side(target.this, columns)
    right = _key_side(target.expression, columns)
    if (left is None) == (right is None):
        # Neither side is the key, or both are: this conjunct compares two columns.
        return ()
    side, value_node = (left, target.expression) if left is not None else (right, target.this)
    if side is None:
        return ()
    if left is None:
        op = _FLIPPED[op]
    value = _instant(value_node)
    return () if value is None else (_Constraint(side[0], side[1], op, value),)


def _all_constraints(
    conjuncts: Sequence[exp.Expr], columns: frozenset[str]
) -> tuple[_Constraint, ...]:
    """The constraints of every conjunct, in fragment order."""
    return tuple(
        constraint for conjunct in conjuncts for constraint in _constraints(conjunct, columns)
    )


def _window(constraints: Sequence[_Constraint]) -> _Window:
    """The tightest window every constraint agrees on.

    A floored key needs care in one direction only: ``date_trunc('month', x) >= '2026-11-15'``
    implies ``x >= '2026-11-15'`` (weaker, safe), but ``date_trunc('month', x) < '2026-11-15'``
    implies ``x < '2026-12-01'``, not ``x < '2026-11-15'``. So an upper bound on a floored key is
    moved out to the next partition boundary and the weaker lower bound is kept as it is.
    """
    lowers: list[datetime] = []
    uppers: list[tuple[datetime, bool]] = []
    for constraint in constraints:
        unit = constraint.unit
        if unit is not None and constraint.op == "eq":
            if _floor(constraint.value, unit) != constraint.value:
                # floor(x) = a mid-boundary value matches nothing; the original filter stays exact.
                continue
            lowers.append(constraint.value)
            uppers.append((_next_boundary(constraint.value, unit), False))
            continue
        if constraint.op in ("gte", "gt"):
            lowers.append(constraint.value)
        elif constraint.op == "lt":
            uppers.append(
                (
                    constraint.value if unit is None else _next_boundary(constraint.value, unit),
                    False,
                )
            )
        elif constraint.op == "lte":
            if unit is None:
                uppers.append((constraint.value, True))
            else:
                uppers.append((_next_boundary(constraint.value, unit), False))
        else:  # eq on the raw column: one instant, an inclusive upper keeps the count honest
            lowers.append(constraint.value)
            uppers.append((constraint.value, True))
    upper: datetime | None = None
    upper_inclusive = False
    for value, inclusive in uppers:
        if upper is None or value < upper or (value == upper and not inclusive):
            upper, upper_inclusive = value, inclusive
    return _Window(
        lower=max(lowers) if lowers else None, upper=upper, upper_inclusive=upper_inclusive
    )


def _rewrite_predicates(
    constraints: Sequence[_Constraint], column: str
) -> tuple[exp.Expression, ...]:
    """Predicates on the raw column, implied by the constraints that used a function on it."""
    floored = tuple(constraint for constraint in constraints if constraint.unit is not None)
    if not floored:
        return ()
    window = _window(floored)
    out: list[exp.Expression] = []
    if window.lower is not None:
        out.append(_at_least(column, window.lower))
    if window.upper is not None:
        out.append(
            _at_most(column, window.upper)
            if window.upper_inclusive
            else _before(column, window.upper)
        )
    return tuple(out)


def _timestamp(moment: datetime) -> exp.Expression:
    """A timestamp literal, written as a cast so every dialect reads it as a time, not a string."""
    text = moment.astimezone(UTC).replace(tzinfo=None).isoformat(sep=" ")
    return exp.cast(exp.Literal.string(text), exp.DataType.build("TIMESTAMP"))


def _at_least(column: str, moment: datetime) -> exp.Expression:
    return exp.GTE(this=exp.column(column), expression=_timestamp(moment))


def _at_most(column: str, moment: datetime) -> exp.Expression:
    return exp.LTE(this=exp.column(column), expression=_timestamp(moment))


def _before(column: str, moment: datetime) -> exp.Expression:
    return exp.LT(this=exp.column(column), expression=_timestamp(moment))


# --- the public entry point --------------------------------------------------------------


def plan_partitions(
    partitioning: Partitioning | None,
    predicates: Sequence[exp.Expression],
    *,
    exec_ts: datetime,
    table: str = "",
    default_window: Duration | None = None,
    max_partitions: int | None = None,
    sizes: Mapping[str, tuple[int | None, int | None]] | None = None,
) -> PartitionPlan:
    """Plan one source's partition access from its metadata and the fragment's predicates.

    Args:
        partitioning: the table's partition metadata, or ``None``/``strategy=none`` when it has
            none (then nothing is pruned and nothing is refused).
        predicates: the sqlglot predicates the compiler will AND into the fragment's ``WHERE``.
        exec_ts: the single UTC execution timestamp; the injected default window ends here.
        table: the table's catalog name, used in error text.
        default_window: the table's ``default_window``; used for ``partitions_if_fixed`` and for
            the window injected when ``require_key_filter`` is off.
        max_partitions: the per-table scan cap (``Limits.max_partitions``). Over it, the plan
            refuses with ``COST_TOO_HIGH``.
        sizes: a measured size profile, ``{partition name: (rows, bytes)}`` (``mdqm sizes
            measure``). When given, the estimate counts what was measured for the partitions in the
            window -- rows *and* bytes -- instead of the catalog's own ``reltuples``, and a
            partition holding ``FAT_RATIO`` times the table's median is reported in
            :attr:`PartitionPlan.fat_partitions` with ``PartitionNote.FAT_PARTITION``.

    Returns:
        A :class:`PartitionPlan`. When ``plan.ok`` is false nothing may run and nothing is fetched
        from the source; the error is agent-facing and carries an exact ``fix``.
    """
    now = _require_aware(exec_ts, "exec_ts")
    if partitioning is None or partitioning.strategy is PartitionStrategy.NONE:
        return PartitionPlan()
    conjuncts = _conjuncts(predicates)
    match partitioning.strategy:
        case PartitionStrategy.RANGE:
            plan = _plan_range(
                partitioning,
                conjuncts,
                exec_ts=now,
                table=table,
                default_window=default_window,
                sizes=sizes,
            )
        case PartitionStrategy.EXPRESSION:
            plan = _plan_expression(
                partitioning,
                conjuncts,
                exec_ts=now,
                table=table,
                default_window=default_window,
                sizes=sizes,
            )
        case PartitionStrategy.LIST:
            plan = _plan_list(
                partitioning,
                conjuncts,
                exec_ts=now,
                table=table,
                default_window=default_window,
                sizes=sizes,
            )
        case _:
            plan = _plan_hash(
                partitioning,
                conjuncts,
                exec_ts=now,
                table=table,
                default_window=default_window,
                sizes=sizes,
            )
    return _apply_cap(plan, table=table, max_partitions=max_partitions)


def _require_aware(moment: datetime, name: str) -> datetime:
    if moment.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC; naive datetimes are a bug")
    return moment.astimezone(UTC)


# --- per-strategy plans ------------------------------------------------------------------


def _plan_range(
    partitioning: Partitioning,
    conjuncts: Sequence[exp.Expr],
    *,
    exec_ts: datetime,
    table: str,
    default_window: Duration | None,
    sizes: Mapping[str, tuple[int | None, int | None]] | None = None,
) -> PartitionPlan:
    """Range partitioning on a time column: a window on the key is exactly a set of partitions."""
    # Bind the key once, explicitly typed: pyright narrows the pydantic field to an empty tuple
    # from its `= ()` default and then reports the index below, which is a false positive.
    keys: tuple[str, ...] = partitioning.key
    if len(keys) == 0:
        # Declared range partitioning with no key column: there is nothing to window on, and a
        # partition count would be invented. Refuse to derive rather than index an empty tuple.
        return _no_filter(partitioning, exec_ts=exec_ts, table=table, default_window=default_window, sizes=sizes)
    if len(keys) > 1:
        # A multi-column range scheme gives no one-dimensional window: a count would be a guess.
        key_set = frozenset(keys)
        if all(
            any(_mentions(conjunct, frozenset({key})) for conjunct in conjuncts) for key in key_set
        ):
            return _unproven(partitioning, note=PartitionNote.ESTIMATE_UNPROVEN, sizes=sizes)
        return _no_filter(partitioning, exec_ts=exec_ts, table=table, default_window=default_window, sizes=sizes)
    return _plan_key_range(
        partitioning,
        conjuncts,
        column=keys[0],
        granularity=partitioning.granularity,
        exec_ts=exec_ts,
        table=table,
        default_window=default_window,
        sizes=sizes,
    )


def _plan_key_range(
    partitioning: Partitioning,
    conjuncts: Sequence[exp.Expr],
    *,
    column: str,
    granularity: PartitionGranularity | None,
    exec_ts: datetime,
    table: str,
    sizes: Mapping[str, tuple[int | None, int | None]] | None = None,
    default_window: Duration | None,
) -> PartitionPlan:
    """The shared core for a time-range key, also used for an expression key on one column."""
    columns = frozenset({column})
    constraints = _all_constraints(conjuncts, columns)
    window = _window(constraints)
    if not window.bounded:
        if any(_mentions(conjunct, columns) for conjunct in conjuncts):
            # The key *is* filtered, but not in a shape whose partitions can be counted.
            return _unproven(partitioning, note=PartitionNote.NEEDS_HUMAN_APPROVAL, sizes=sizes)
        return _no_filter(
            partitioning,
            exec_ts=exec_ts,
            table=table,
            default_window=default_window,
            injectable=len(partitioning.key) == 1,
            column=column,
            # The injected window is a window like any other: a measured profile answers it too.
            sizes=sizes,
        )
    predicates = _rewrite_predicates(constraints, column)
    fixes = tuple(
        f'rewrote the {constraint.unit.value if constraint.unit else "cast"} floor on "{column}" '
        "into a range on the raw column so the source can prune"
        for constraint in constraints
        if constraint.unit is not None
    )
    notes = (PartitionNote.KEY_FUNCTION_REWRITTEN,) if fixes else ()
    fat = _fat(partitioning, window, sizes)
    scope = (
        Range(window.lower, window.upper)
        if window.lower is not None and window.upper is not None and not window.upper_inclusive
        else None
    )
    return PartitionPlan(
        predicates=predicates,
        notes=notes + ((PartitionNote.FAT_PARTITION,) if fat else ()),
        applied_fixes=tuple(dict.fromkeys(fixes)),
        estimate=_estimate(partitioning, window, sizes),
        scope=scope,
        segments=align_to_partitions(scope, granularity or PartitionGranularity.MONTH)
        if scope is not None
        else (),
        fat_partitions=fat,
    )


def _plan_expression(
    partitioning: Partitioning,
    conjuncts: Sequence[exp.Expr],
    *,
    exec_ts: datetime,
    table: str,
    default_window: Duration | None,
    sizes: Mapping[str, tuple[int | None, int | None]] | None = None,
) -> PartitionPlan:
    """Expression partitioning: prune only through the declared expression (or its own column)."""
    declared = _parse(partitioning.expression or "")
    if declared is None:
        return _unproven(partitioning, note=PartitionNote.NEEDS_HUMAN_APPROVAL, sizes=sizes)
    side = _key_side(declared, _columns(declared))
    if side is None:
        # Not a floor of one column (e.g. region || state): only a human can judge what prunes.
        return _unproven(partitioning, note=PartitionNote.NEEDS_HUMAN_APPROVAL, sizes=sizes)
    column, unit = side
    return _plan_key_range(
        partitioning,
        conjuncts,
        column=column,
        granularity=unit or partitioning.granularity,
        exec_ts=exec_ts,
        table=table,
        default_window=default_window,
        sizes=sizes,
    )


def _plan_list(
    partitioning: Partitioning,
    conjuncts: Sequence[exp.Expr],
    *,
    exec_ts: datetime,
    table: str,
    default_window: Duration | None,
    sizes: Mapping[str, tuple[int | None, int | None]] | None = None,
) -> PartitionPlan:
    """List partitioning: a filter on the key selects partitions by their declared values."""
    values, mentioned = _set_values(conjuncts, frozenset(partitioning.key))
    if values is None:
        if mentioned:
            return _unproven(partitioning, note=PartitionNote.NEEDS_HUMAN_APPROVAL, sizes=sizes)
        return _no_filter(partitioning, exec_ts=exec_ts, table=table, default_window=default_window, sizes=sizes)
    wanted = set(values)
    touched = tuple(bound for bound in partitioning.bounds if wanted & set(bound.values))
    rows = sum(bound.row_estimate or 0 for bound in touched)
    return PartitionPlan(
        estimate=PartitionEstimate(
            partitions=len(touched), total=partitioning.partitions_total, rows=rows or None
        )
    )


def _plan_hash(
    partitioning: Partitioning,
    conjuncts: Sequence[exp.Expr],
    *,
    exec_ts: datetime,
    table: str,
    default_window: Duration | None,
    sizes: Mapping[str, tuple[int | None, int | None]] | None = None,
) -> PartitionPlan:
    """Hash partitioning: the key itself must be filtered; a range prunes nothing."""
    values, mentioned = _set_values(conjuncts, frozenset(partitioning.key))
    if values is None:
        if mentioned:
            # A range on the key does not let the source skip hash partitions.
            return _unproven(partitioning, note=PartitionNote.NEEDS_HUMAN_APPROVAL, sizes=sizes)
        return _no_filter(partitioning, exec_ts=exec_ts, table=table, default_window=default_window, sizes=sizes)
    total = partitioning.partitions_total
    # Each distinct value lands in exactly one partition, so the value count is an upper bound.
    partitions = len(values) if total is None else min(len(values), total)
    return PartitionPlan(
        notes=(PartitionNote.ESTIMATE_UNPROVEN,),
        estimate=PartitionEstimate(partitions=partitions, total=total),
    )


def _set_values(
    conjuncts: Sequence[exp.Expr], columns: frozenset[str]
) -> tuple[tuple[str, ...] | None, bool]:
    """The constants an ``=``/``IN`` filter on the key itself names, and whether the key appears.

    A wrapped key is deliberately not collected: ``date_trunc('month', region) = 'x'`` neither
    matches a declared value nor lets a list- or hash-partitioned source skip a partition.

    Conjuncts are ANDed, so their value sets are intersected: ``region = 'a' AND region = 'b'``
    matches nothing, and the estimate says ``0`` rather than counting both partitions. ``None``
    values means no ``=``/``IN`` filter on the key was read at all, which is a different answer
    from "the filter matches nothing" (``()``).
    """
    per_conjunct: list[tuple[str, ...]] = []
    mentioned = False
    for conjunct in conjuncts:
        node = conjunct.unnest()
        mentioned = mentioned or _mentions(node, columns)
        if isinstance(node, exp.EQ):
            side = _key_side(node.this, columns)
            if side is not None and side[1] is None and _key_side(node.expression, columns) is None:
                text = _literal_text(node.expression)
                if text is not None:
                    per_conjunct.append((text,))
        elif isinstance(node, exp.In):
            side = _key_side(node.this, columns)
            if side is None or side[1] is not None:
                continue
            texts = tuple(dict.fromkeys(_literal_text(item) for item in node.expressions))
            if None not in texts:
                per_conjunct.append(tuple(text for text in texts if text is not None))
    if not per_conjunct:
        return None, mentioned
    common = set(per_conjunct[0]).intersection(*per_conjunct[1:])
    return tuple(value for value in per_conjunct[0] if value in common), mentioned


# --- refusing, injecting, capping ---------------------------------------------------------


def _no_filter(
    partitioning: Partitioning,
    *,
    exec_ts: datetime,
    table: str,
    default_window: Duration | None,
    injectable: bool = False,
    column: str | None = None,
    sizes: Mapping[str, tuple[int | None, int | None]] | None = None,
) -> PartitionPlan:
    """Nothing filters the key: inject the table's default window, or refuse with an estimate."""
    inject = (
        injectable
        and column is not None
        and default_window is not None
        and not partitioning.require_key_filter
        and partitioning.strategy is PartitionStrategy.RANGE
        and partitioning.granularity is not None
    )
    window: Range | None = None
    if inject and default_window is not None:
        window = Range(default_window.subtract_from(exec_ts), exec_ts)
    if window is not None and column is not None and default_window is not None:
        granularity = partitioning.granularity or PartitionGranularity.MONTH
        injected = _Window(lower=window.lower, upper=window.upper)
        fat = _fat(partitioning, injected, sizes)
        return PartitionPlan(
            predicates=(_at_least(column, window.lower), _before(column, window.upper)),
            notes=(PartitionNote.DEFAULT_WINDOW_APPLIED,)
            + ((PartitionNote.FAT_PARTITION,) if fat else ()),
            applied_fixes=(
                f'added the table default window {default_window.to_text()} on "{column}" '
                "because this table allows a query without a key filter",
            ),
            estimate=estimate_partition_scan(partitioning, window, sizes),
            scope=window,
            segments=align_to_partitions(window, granularity),
            fat_partitions=fat,
        )
    fix_window = default_window or Duration.parse("30d")
    estimate_when_fixed = estimate_partition_scan(
        partitioning,
        Range(fix_window.subtract_from(exec_ts), exec_ts)
        if partitioning.strategy is PartitionStrategy.RANGE
        else None,
    )
    return PartitionPlan(
        error=partition_scan_all(
            table=table,
            key=", ".join(partitioning.key)
            or (partitioning.expression or partitioning.strategy.value),
            granularity=(
                partitioning.granularity.value
                if partitioning.granularity
                else partitioning.strategy.value
            ),
            default_window=default_window.to_text() if default_window is not None else None,
            partitions_if_fixed=estimate_when_fixed.label,
        )
    )


def _unproven(
    partitioning: Partitioning,
    *,
    note: PartitionNote,
    sizes: Mapping[str, tuple[int | None, int | None]] | None = None,
) -> PartitionPlan:
    """The key is filtered, but the partitions it touches cannot be counted from here.

    Not an error: the count is reported as every declared partition and the ``max_partitions`` cap
    decides. A note carries the reason, so the human review queue sees it (principle 10).
    """
    return PartitionPlan(notes=(note,), estimate=_estimate(partitioning, None, sizes))


def _apply_cap(plan: PartitionPlan, *, table: str, max_partitions: int | None) -> PartitionPlan:
    """Refuse the scan when the static estimate is over the table's partition cap."""
    if plan.error is not None or max_partitions is None or plan.estimate is None:
        return plan
    partitions = plan.estimate.partitions
    if partitions is None or partitions <= max_partitions:
        return plan
    return replace(
        plan,
        error=cost_too_high(
            table=table,
            estimated=plan.estimate.summary,
            limit=f"{max_partitions} partitions",
        ),
    )


def gate_plan(
    summary: PlanSummary,
    *,
    table: str,
    partitioning: Partitioning | None,
    max_partitions_scanned: int | None,
) -> ErrorResponse | None:
    """Refuse a plan that scans more partitions than ``max_partitions_scanned`` (C08's cap).

    :func:`plan_partitions` caps the *static* estimate, counted from the declared bounds. This gate
    applies the same cap to what the source's own planner did, which is the only authority on
    whether pruning actually happened: a filter written on the wrong column, or one a function
    hides, passes the static count and still reads every partition. The code is
    ``PARTITION_SCAN_ALL`` because from the source's point of view that is what happened.

    Returns:
        The agent-facing error, or ``None`` when the plan stays inside the cap (and whenever the
        source reported no partition count at all: an unproven count is not a refusal).
    """
    scanned = summary.partitions_scanned
    if max_partitions_scanned is None or scanned is None or scanned <= max_partitions_scanned:
        return None
    total = summary.partitions_total
    key = ", ".join(partitioning.key) if partitioning is not None else ""
    granularity = (
        partitioning.granularity.value
        if partitioning is not None and partitioning.granularity is not None
        else (partitioning.strategy.value if partitioning is not None else "by the source")
    )
    return partition_scan_all(
        table=table,
        key=key or "the partition key",
        granularity=granularity,
        scanned=f"{scanned} of {total}" if total is not None else str(scanned),
    )


def _parse(text: str) -> exp.Expr | None:
    """Parse a declared partition expression, or ``None`` when it is not one this module reads."""
    try:
        return sqlglot.parse_one(text, read="duckdb")
    except (sqlglot_errors.ParseError, sqlglot_errors.TokenError, ValueError):
        return None
