"""Time planning: turn a spec time *intent* into predicates on the right timestamp column.

The one mistake this module exists to prevent (docs/DESIGN.md section 6): **event time and
ingestion time are different questions**. ``start_time`` is the event time (indexed, often the
partition key); ``insertion_time`` is when the row landed (not indexed). A business question about
a period filters event time. "What is new since my last fetch" filters ingestion time, because a
row that arrived late is *not* in its event-time window yet.

Filtering ingestion time alone scans everything, so the declared lag gives a derived bound on the
indexed event column::

    insertion_time >= X   =>   start_time >= X - max_lag

The original predicate stays. The derived one is **implied** by it: a row with
``insertion_time >= X`` has ``start_time = insertion_time - lag >= X - max_lag`` whenever
``lag <= max_lag``. So adding the bound can only narrow the scan, never drop a row — that is the
"never loses rows" invariant, proved by ``tests/unit/test_time_planner.py`` with hypothesis over
the D02 world and DuckDB.

Pure: no I/O, no clock read. ``exec_ts`` is one value per execution (C02) and is injected here, so
a re-run sees the same window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from sqlglot import exp

from mdqm.errors import insertion_time_no_bound, invalid_spec
from mdqm.models import ErrorResponse
from mdqm.semantic.models import Lag, TableSemantics, TimeColumn, TimeRole
from mdqm.spec.models import TimeMode, TimeSpec

__all__ = [
    "TimeNote",
    "TimePlan",
    "event_column",
    "ingestion_column",
    "known_as_of",
    "plan_time",
]


class TimeNote(StrEnum):
    """A caveat the caller must show the agent (principle 10: nothing silent)."""

    PARTIAL_DATA_POSSIBLE = "partial_data_possible"


@dataclass(frozen=True, slots=True)
class TimePlan:
    """The predicates a time intent produced, plus notes, an error, or the derived bound.

    ``predicates`` are sqlglot expressions to AND into the fragment's WHERE. Empty means the
    intent added no bound (no ``time`` in the spec). ``error`` is mutually exclusive with a usable
    plan: when it is set, nothing may run.
    """

    predicates: tuple[exp.Expression, ...] = ()
    notes: tuple[TimeNote, ...] = ()
    error: ErrorResponse | None = None
    window: tuple[datetime, datetime] | None = None
    """The resolved half-open event window ``[start, end)``, when the intent named one."""

    derived_bound: datetime | None = None
    """The derived lower bound added to the indexed event column, when one was derived."""

    @property
    def ok(self) -> bool:
        """True when the intent produced a runnable plan (possibly with no predicates)."""
        return self.error is None

    def where_sql(self, dialect: str = "duckdb") -> str:
        """The predicates rendered as a SQL ``WHERE`` body (empty string when there are none)."""
        return " AND ".join(predicate.sql(dialect=dialect) for predicate in self.predicates)


# --- column selection -------------------------------------------------------------------


def event_column(table: TableSemantics) -> tuple[str, TimeColumn] | None:
    """The event-time column a period question filters, or ``None`` when the table has none.

    Preference: ``use_by_default`` (the curated answer), then an indexed event column. Both are
    deterministic, so the same semantic YAML always compiles to the same SQL.
    """
    candidates = [
        (name, column)
        for name, column in table.time_columns.items()
        if column.role is TimeRole.EVENT_TIME
    ]
    for name, column in candidates:
        if column.use_by_default:
            return name, column
    for name, column in candidates:
        if column.indexed:
            return name, column
    return candidates[0] if candidates else None


def ingestion_column(table: TableSemantics) -> tuple[str, TimeColumn] | None:
    """The ingestion-time column, or ``None``. One that declares ``derives_bound_on`` wins."""
    candidates = [
        (name, column)
        for name, column in table.time_columns.items()
        if column.role is TimeRole.INGESTION_TIME
    ]
    for name, column in candidates:
        if column.derives_bound_on is not None:
            return name, column
    return candidates[0] if candidates else None


def _lag_for(
    table: TableSemantics,
    event_name: str,
    *,
    lag_unverified: frozenset[str],
) -> Lag | None:
    """The lag of the ingestion column that derives its bound on ``event_name``, if trusted.

    A column named in ``lag_unverified`` had a nightly violation, so its lag is treated as
    unknown: the derivation is disabled until a human reviews it (DESIGN section 6).
    """
    for name, column in table.time_columns.items():
        if column.role is not TimeRole.INGESTION_TIME:
            continue
        if column.derives_bound_on != event_name:
            continue
        if lag_unverified and name in lag_unverified:
            return None
        return column.lag
    return None


def _max_lag_for(
    table: TableSemantics, event_name: str, *, lag_unverified: frozenset[str]
) -> timedelta | None:
    lag = _lag_for(table, event_name, lag_unverified=lag_unverified)
    return None if lag is None else lag.maximum


# --- expression helpers -----------------------------------------------------------------


def _timestamp(moment: datetime) -> exp.Expression:
    """A timestamp literal, rendered as a cast so every dialect reads it as a time, not a string."""
    text = moment.astimezone(UTC).replace(tzinfo=None).isoformat(sep=" ")
    return exp.cast(exp.Literal.string(text), exp.DataType.build("TIMESTAMP"))


def _at_least(column: str, moment: datetime) -> exp.Expression:
    return exp.GTE(this=exp.column(column), expression=_timestamp(moment))


def _before(column: str, moment: datetime) -> exp.Expression:
    return exp.LT(this=exp.column(column), expression=_timestamp(moment))


def _at_most(column: str, moment: datetime) -> exp.Expression:
    return exp.LTE(this=exp.column(column), expression=_timestamp(moment))


def _require_aware(moment: datetime, name: str) -> datetime:
    if moment.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC; naive datetimes are a bug")
    return moment.astimezone(UTC)


# --- the public entry point -------------------------------------------------------------


def plan_time(
    time: TimeSpec | None,
    table: TableSemantics,
    *,
    exec_ts: datetime,
    table_name: str = "",
    lag_unverified: frozenset[str] = frozenset(),
) -> TimePlan:
    """Plan one spec time intent against a table's time-column metadata.

    Args:
        time: the spec ``time`` block, or ``None`` (no time intent: no predicates).
        table: the table's semantic metadata, which owns the time-column roles and lag.
        exec_ts: the single UTC execution timestamp; ``last`` windows are relative to it.
        table_name: the table's catalog name, used in error text.
        lag_unverified: ingestion columns whose nightly lag verification failed. Their lag is
            treated as unknown, so no derived bound is produced.

    Returns:
        A :class:`TimePlan`. When ``plan.ok`` is false the intent cannot be answered safely and
        nothing may run; the error is agent-facing and carries an exact ``fix``.
    """
    now = _require_aware(exec_ts, "exec_ts")
    if time is None:
        return TimePlan()
    if time.mode is TimeMode.NEW_SINCE:
        return _plan_new_since(
            time, table, exec_ts=now, table_name=table_name, lag_unverified=lag_unverified
        )
    return _plan_events_in_range(
        time, table, exec_ts=now, table_name=table_name, lag_unverified=lag_unverified
    )


def _plan_events_in_range(
    time: TimeSpec,
    table: TableSemantics,
    *,
    exec_ts: datetime,
    table_name: str,
    lag_unverified: frozenset[str],
) -> TimePlan:
    """A period question: a half-open range on the event-time column.

    When the partition key *is* the ingestion column, the event range ``[X, Y)`` also yields an
    ingestion range ``[X + min_lag, Y + max_lag)``: every row whose event time is in
    ``[X, Y)`` was inserted inside that wider range, so the range prunes partitions without
    losing a row.
    """
    chosen = event_column(table)
    if chosen is None:
        return TimePlan(
            error=invalid_spec(field="time", reason="the table has no event-time column")
        )
    name, _column = chosen

    window = _resolve_window(time, exec_ts)
    if window is None:
        return TimePlan(
            error=invalid_spec(
                field="time",
                reason="it needs 'last', or both 'from' and 'to', to name a period",
            )
        )
    start, end = window
    predicates: list[exp.Expression] = [_at_least(name, start), _before(name, end)]

    max_lag = _max_lag_for(table, name, lag_unverified=lag_unverified)
    notes: list[TimeNote] = []
    if max_lag is not None and end > exec_ts - max_lag:
        # Rows whose event time is in the tail may not have been inserted yet.
        notes.append(TimeNote.PARTIAL_DATA_POSSIBLE)

    ingestion = ingestion_column(table)
    if ingestion is not None and _partitioned_on(table, ingestion[0]):
        widened = _ingestion_range(
            ingestion[0], ingestion[1], start, end, lag_unverified=lag_unverified
        )
        if widened is not None:
            predicates.extend(
                (_at_least(ingestion[0], widened[0]), _before(ingestion[0], widened[1]))
            )

    return TimePlan(predicates=tuple(predicates), notes=tuple(notes), window=window)


def _plan_new_since(
    time: TimeSpec,
    table: TableSemantics,
    *,
    exec_ts: datetime,
    table_name: str,
    lag_unverified: frozenset[str],
) -> TimePlan:
    """A sync question: ingestion time since ``X``, plus the implied event-time lower bound.

    Without a trusted lag there is no safe bound to add, so the answer is
    ``INSERTION_TIME_NO_BOUND`` with a ``fix`` that names the spec's ``time`` field.
    """
    if time.since is None:
        return TimePlan(
            error=invalid_spec(field="time.since", reason="it is required when mode is 'new_since'")
        )
    since = _require_aware(time.since, "time.since")
    chosen = ingestion_column(table)
    if chosen is None:
        return TimePlan(
            error=invalid_spec(field="time", reason="the table has no ingestion-time column")
        )
    name, column = chosen

    target = column.derives_bound_on
    lag = _lag_for(table, target, lag_unverified=lag_unverified) if target else None
    if target is None or lag is None:
        # No measured lag (or a failed verification): deriving would silently drop late rows.
        return TimePlan(
            error=insertion_time_no_bound(
                table=table_name,
                event_column=target or (event_column(table) or (name, column))[0],
            )
        )

    bound = since - lag.maximum
    predicates = (_at_least(name, since), _at_least(target, bound))
    notes: list[TimeNote] = []
    if since > exec_ts - lag.maximum:
        # The requested insertion window is shorter than the lag: a later run can return more.
        notes.append(TimeNote.PARTIAL_DATA_POSSIBLE)
    return TimePlan(predicates=predicates, notes=tuple(notes), derived_bound=bound)


def known_as_of(
    table: TableSemantics,
    *,
    as_of: datetime,
    start: datetime,
    end: datetime,
    exec_ts: datetime,
    lag_unverified: frozenset[str] = frozenset(),
) -> TimePlan:
    """What the table held *as known at* ``as_of``: rows inserted by then, events in the range.

    This answers "what did we know at T", which is neither a pure period question nor a pure sync
    question: the event range picks rows, the ingestion ceiling excludes what arrived later.
    """
    moment = _require_aware(as_of, "as_of")
    _require_aware(exec_ts, "exec_ts")
    events = event_column(table)
    ingestions = ingestion_column(table)
    if events is None or ingestions is None:
        return TimePlan(
            error=invalid_spec(
                field="time",
                reason="'known as of' needs both an event-time and an ingestion-time column",
            )
        )
    event_name, _event = events
    ingestion_name, _ingestion = ingestions
    if start >= end:
        return TimePlan(
            error=invalid_spec(field="time", reason="the event window must end after it starts")
        )
    predicates = (
        _at_least(event_name, _require_aware(start, "start")),
        _before(event_name, _require_aware(end, "end")),
        _at_most(ingestion_name, moment),
    )
    return TimePlan(predicates=predicates, window=(start, end))


# --- helpers ----------------------------------------------------------------------------


def _resolve_window(time: TimeSpec, exec_ts: datetime) -> tuple[datetime, datetime] | None:
    """``last`` or ``from``/``to`` as a half-open UTC range, or ``None`` when under-specified."""
    if time.last is not None:
        return time.last.subtract_from(exec_ts), exec_ts
    if time.from_time is None:
        return None
    start = _require_aware(time.from_time, "time.from")
    end = exec_ts if time.to is None else _require_aware(time.to, "time.to")
    return (start, end) if end > start else None


def _partitioned_on(table: TableSemantics, column: str) -> bool:
    """True when the table's partition key is this column (range/list partitioning on time)."""
    partitioning = table.partitioning
    return partitioning is not None and column in partitioning.key


def _ingestion_range(
    name: str,
    ingestion: TimeColumn,
    start: datetime,
    end: datetime,
    *,
    lag_unverified: frozenset[str],
) -> tuple[datetime, datetime] | None:
    """The ingestion range that surely contains every row whose event time is in ``[start, end)``.

    ``insertion = event + lag`` and ``min_lag <= lag <= max_lag``, so the row is inserted in
    ``[start + min_lag, end + max_lag)``. Unknown or failed lag means no range: correctness stands,
    pruning is left to C08.
    """
    if ingestion.derives_bound_on is None or ingestion.lag is None:
        return None
    if lag_unverified and name in lag_unverified:
        return None
    return start + ingestion.lag.minimum, end + ingestion.lag.maximum
