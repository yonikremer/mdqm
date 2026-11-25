"""The structured query spec: what an agent sends instead of SQL (docs/DESIGN.md section 5).

Shape (every field is explained by the docstring of its model; ``spec_schema()`` turns it into a
JSON Schema with the organisation's table/measure/join names filled in)::

    {
      "from": "accounts",
      "select": ["customer_status", {"measure": "credit_limit_r5000", "agg": "avg"}],
      "time": {"mode": "events_in_range", "last": "3mo"},
      "filters": [{"column": "customer_status", "op": "eq", "value": "פעיל"}],
      "group_by": ["customer_status"],
      "order_by": [{"by": "customer_status", "direction": "asc"}],
      "limit": 50,
      "joins": ["usage_to_customers"],
      "dry_run": false,
      "fresh": false
    }

Only structural rules live in the pydantic models (types, enums, ranges, per-operator value shapes),
so a bad spec fails while it is still JSON. :func:`parse_spec` then runs the cross-field rules of
:func:`check_spec` and raises :class:`SpecError`, which carries one or more :class:`SpecIssue`
values: an exact ``path``, a ``problem`` and a ``fix``. Nothing here touches a source, a file or a
clock.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Annotated, Any, TypeAlias

from pydantic import BeforeValidator, ConfigDict, Field, ValidationInfo, field_validator

from mdqm.models.base import MDQMModel
from mdqm.spec.durations import Duration, DurationField

#: A filter value or an order-by key: JSON scalars only (no objects, no nesting).
Scalar: TypeAlias = str | bool | int | float


class SpecModel(MDQMModel):
    """Base for spec models: the wire name is the alias (``from``), and both spellings validate."""

    model_config = ConfigDict(validate_by_name=True, serialize_by_alias=True)


class Agg(StrEnum):
    """Aggregation over a measure. ``None`` on a select item means the measure's own default."""

    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"


class FilterOp(StrEnum):
    """Comparison on one column: a list for ``in``/``range``, a scalar for everything else."""

    EQ = "eq"
    NE = "ne"
    IN = "in"
    CONTAINS = "contains"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    RANGE = "range"


class TimeMode(StrEnum):
    """The time intent. The agent says what it means; the server picks the column.

    ``events_in_range`` asks about a period (event time). ``new_since`` asks what is new since a
    previous fetch (ingestion time, with a derived bound so late rows are not missed).
    """

    EVENTS_IN_RANGE = "events_in_range"
    NEW_SINCE = "new_since"


class Direction(StrEnum):
    """Sort direction."""

    ASC = "asc"
    DESC = "desc"


def _instant(value: Any) -> datetime:
    """Parse an ISO-8601 date/datetime and normalise to UTC.

    A date-only value is midnight UTC; a value without an offset is *taken* as UTC, because every
    timestamp inside mdqm is UTC and guessing a local zone would silently move the window.
    """
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime(value.year, value.month, value.day, tzinfo=UTC)
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.strip())
        except ValueError:
            raise ValueError(
                f"{value!r} is not an ISO-8601 date or datetime; "
                "send '2026-06-01' or '2026-11-25T14:00:00' (UTC when no offset is given)"
            ) from None
    else:
        raise ValueError(f"expected an ISO-8601 date or datetime string, got {value!r}")
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


#: An ISO-8601 date or datetime, normalised to aware UTC (naive input means UTC).
SpecInstant: TypeAlias = Annotated[datetime, BeforeValidator(_instant)]


class MeasureSelect(SpecModel):
    """One aggregate over a measure: ``{"measure": "credit_limit_r5000", "agg": "avg"}``."""

    measure: str = Field(min_length=1, description="A measure name from the semantic layer.")
    agg: Agg | None = Field(default=None, description="Aggregation; omit for the measure default.")
    alias: str | None = Field(
        default=None, min_length=1, description="Output column name; defaults to measure_agg."
    )


#: A plain column name, or an aggregate over a measure.
SelectItem: TypeAlias = Annotated[str, Field(min_length=1)] | MeasureSelect


class Filter(SpecModel):
    """One predicate. Hebrew/English labels are fine: the server resolves them to source values."""

    column: str = Field(min_length=1)
    op: FilterOp
    # Untyped on purpose: one validator then reports the operator-aware rule at this exact path,
    # instead of a union's "expected string / expected number / expected list" branch errors.
    value: Any = Field(
        description="A single value, or a list for op 'in' (any length) and 'range' (exactly 2).",
        json_schema_extra={"type": ["string", "number", "boolean", "array"]},
    )

    @field_validator("value")
    @classmethod
    def _check_value_shape(cls, value: Any, info: ValidationInfo) -> Any:
        op: FilterOp | None = info.data.get("op")
        if op is None:
            return value  # op failed already; its own issue is clearer than a second one here
        items = list(value) if isinstance(value, (list, tuple)) else None
        if op in (FilterOp.IN, FilterOp.RANGE):
            if items is None:
                raise ValueError(
                    f"op '{op.value}' takes a list; send value: [..] "
                    f"({'at least one' if op is FilterOp.IN else 'exactly two'} entries)"
                )
            if op is FilterOp.IN and not items:
                raise ValueError("op 'in' needs at least one value")
            if op is FilterOp.RANGE and len(items) != 2:
                raise ValueError(f"op 'range' takes exactly two values, got {len(items)}")
            if op is FilterOp.RANGE:
                low, high = items
                numeric = all(
                    isinstance(i, (int, float)) and not isinstance(i, bool) for i in items
                )
                if numeric and float(low) >= float(high):
                    raise ValueError(f"op 'range' takes [low, high] with low < high, got {items!r}")
                if not numeric and not all(isinstance(i, str) for i in items):
                    raise ValueError("op 'range' takes two numbers or two dates, not a mix")
            for item in items:
                _check_scalar(item, op)
            return tuple(items)
        if items is not None:
            raise ValueError(
                f"op '{op.value}' takes a single value; use op 'in' for a list of allowed values "
                "or op 'range' for a low/high pair"
            )
        _check_scalar(value, op)
        return value


def _check_scalar(value: Any, op: FilterOp) -> None:
    """Reject values the source could never compare against, naming the operator."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"op '{op.value}' got a non-finite number; use a real number")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"op '{op.value}' got an empty string; send the value you mean")
        if any(character < " " for character in value):
            raise ValueError(f"op '{op.value}' got a control character; send plain text")
        if op in (FilterOp.GT, FilterOp.GTE, FilterOp.LT, FilterOp.LTE, FilterOp.RANGE):
            _check_instant_text(value, op)
        return
    if isinstance(value, bool):
        if op in (FilterOp.GT, FilterOp.GTE, FilterOp.LT, FilterOp.LTE):
            raise ValueError(f"op '{op.value}' needs a number or an ISO-8601 date, not a boolean")
        return
    if isinstance(value, (int, float)):
        if op is FilterOp.CONTAINS:
            raise ValueError("op 'contains' takes a text value to search for, not a number")
        return
    raise ValueError(
        f"value must be a string, number or boolean, got {type(value).__name__}; "
        "the spec has no way to express SQL in a filter value"
    )


def _check_instant_text(value: str, op: FilterOp) -> None:
    try:
        _instant(value)
    except ValueError:
        raise ValueError(
            f"op '{op.value}' needs a number or an ISO-8601 date/datetime, got {value!r}"
        ) from None


class OrderBy(SpecModel):
    """Sort key: a column, a measure name or a select alias, plus a direction."""

    by: str = Field(
        min_length=1, description="A selected column, measure or alias, or a group_by key."
    )
    direction: Direction = Direction.ASC


class TimeSpec(SpecModel):
    """The time window intent, never a column (the server maps it: DESIGN section 6).

    ``events_in_range`` needs ``last`` (a trailing window from the execution time) **or** an
    absolute ``from``/``to``. ``new_since`` needs ``since`` and is answered on the ingestion time
    with a derived bound, so nothing is missed when a row arrives late.
    """

    mode: TimeMode
    last: DurationField | None = Field(default=None, description="Trailing window, e.g. '3mo'.")
    from_time: SpecInstant | None = Field(default=None, alias="from", description="Window start.")
    to: SpecInstant | None = Field(default=None, description="Window end (exclusive).")
    since: SpecInstant | None = Field(default=None, description="Ingestion-time lower bound.")

    @field_validator("last")
    @classmethod
    def _check_last(cls, value: Duration | None) -> Duration | None:
        if value is not None and value.is_zero:
            raise ValueError("a window must be longer than zero; send e.g. '7d' or '3mo'")
        return value


class QuerySpec(SpecModel):
    """One query, as an agent sends it. Frozen: a validated spec never changes under the server."""

    from_table: str = Field(
        alias="from", min_length=1, description="Table (or endpoint/index) name from the catalog."
    )
    select: tuple[SelectItem, ...] = Field(
        min_length=1, description="Columns and aggregates to return."
    )
    filters: tuple[Filter, ...] = ()
    time: TimeSpec | None = None
    group_by: tuple[str, ...] = ()
    order_by: tuple[OrderBy, ...] = ()
    limit: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description="Row cap; the server adds its own cap when this is missing.",
    )
    joins: tuple[str, ...] = Field(
        default=(),
        description="Approved join names from the semantic layer; never a join condition.",
    )
    dry_run: bool = Field(default=False, description="Plan and lint, return no rows.")
    fresh: bool = Field(default=False, description="Skip the cache and fetch from the source now.")

    def selected_names(self) -> tuple[str, ...]:
        """Output names: aliases, qualified measures and plain columns, in select order."""
        names: list[str] = []
        for item in self.select:
            if isinstance(item, str):
                names.append(item)
            elif item.agg is None:
                names.append(item.alias or item.measure)
            else:
                names.append(item.alias or f"{item.measure}_{item.agg.value}")
        return tuple(names)
