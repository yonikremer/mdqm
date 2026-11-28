"""Column quality and categorical distributions, from values already in memory.

Pure: no I/O, no clock, no SQL. A collector (DuckDB, PostgreSQL, a catalog parser) hands rows
here. Shares use the whole column as the denominator, so a NULL share and a zero share are
comparable. ``False`` is not zero, and the string ``"0"`` is not zero: zero is a numeric 0.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Final

from mdqm.stats.models import CategoryColumnStats, CollectionMethod, ColumnQuality, ObservedValue

__all__ = [
    "DEFAULT_MAX_CARDINALITY",
    "DEFAULT_MAX_DISTINCT_RATIO",
    "DEFAULT_TOP_K",
    "LOW_CARDINALITY",
    "ColumnRequest",
    "display_value",
    "is_categorical",
    "measure_column",
    "measure_distribution",
    "top_values",
]

#: A column with at most this many distinct non-null values is categorical whatever its type.
LOW_CARDINALITY: Final[int] = 20

#: Above that, a column is categorical only while it stays under both of these.
DEFAULT_MAX_CARDINALITY: Final[int] = 50
DEFAULT_MAX_DISTINCT_RATIO: Final[float] = 0.05

#: How many values are stored before the rest collapse into ``other``. A categorical column
#: (cardinality <= 50) therefore keeps its whole distribution.
DEFAULT_TOP_K: Final[int] = 50


class ColumnRequest:
    """Which shares and bounds to compute for one column. Flags come from the catalog type."""

    __slots__ = ("name", "numeric", "temporal", "text")

    def __init__(
        self,
        name: str,
        *,
        numeric: bool = False,
        temporal: bool = False,
        text: bool = False,
    ) -> None:
        self.name = name
        self.numeric = numeric
        self.temporal = temporal
        self.text = text


def display_value(value: object) -> str:
    """The stable text form of a stored value. Booleans are words, so they do not become ``0``."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _distinct_key(value: object) -> str:
    """A key that keeps ``True`` distinct from ``1`` and ``1`` distinct from ``"1"``."""
    if isinstance(value, bool):
        return f"b:{value}"
    if isinstance(value, (int, float, Decimal)):
        return f"n:{value}"
    if isinstance(value, datetime):
        return f"t:{value.isoformat()}"
    if isinstance(value, date):
        return f"d:{value.isoformat()}"
    return f"s:{value}"


def _is_zero(value: object) -> bool:
    """Numeric zero only. ``False`` and ``"0"`` are values, not the zero sentinel."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return value == 0


def _share(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return count / total


def measure_column(
    values: Sequence[object],
    request: ColumnRequest,
    *,
    source: str,
    table: str,
    collected_at: datetime,
    method: CollectionMethod,
    partition: str = "",
    sample_rows: int | None = None,
) -> ColumnQuality:
    """Shares, distinct count and bounds of one column.

    An empty input leaves every share ``None``: a fraction of no rows is not zero.
    """
    total = len(values)
    nulls = zeros = empties = blanks = 0
    distinct: set[str] = set()
    numbers: list[Decimal] = []
    moments: list[str] = []
    for value in values:
        if value is None:
            nulls += 1
            continue
        distinct.add(_distinct_key(value))
        if request.numeric and _is_zero(value):
            zeros += 1
        if request.text and isinstance(value, str):
            if value == "":
                empties += 1
            elif value.strip() == "":
                blanks += 1
        if (
            request.numeric
            and isinstance(value, (int, float, Decimal))
            and not isinstance(value, bool)
        ):
            if isinstance(value, float) and not math.isfinite(value):
                continue
            numbers.append(Decimal(str(value)))
        elif request.temporal and isinstance(value, (datetime, date)):
            moments.append(display_value(value))
    minimum: str | None = None
    maximum: str | None = None
    if numbers:
        minimum = format(min(numbers), "f")
        maximum = format(max(numbers), "f")
    elif moments:
        ordered = sorted(moments)
        minimum, maximum = ordered[0], ordered[-1]
    return ColumnQuality(
        source=source,
        table=table,
        column=request.name,
        partition=partition,
        null_share=_share(nulls, total),
        zero_share=_share(zeros, total) if request.numeric else None,
        empty_share=_share(empties, total) if request.text else None,
        whitespace_share=_share(blanks, total) if request.text else None,
        distinct_count=len(distinct) if total else 0,
        minimum=minimum,
        maximum=maximum,
        rows_seen=total,
        sample_rows=sample_rows,
        method=method,
        collected_at=collected_at,
    )


def is_categorical(*, distinct: int, non_null: int) -> bool:
    """True when the column is a category: few distinct values, or a small share of the rows.

    Zero non-null rows is not a category. There is nothing to distribute.
    """
    if distinct <= 0 or non_null <= 0:
        return False
    if distinct <= LOW_CARDINALITY:
        return True
    if distinct > DEFAULT_MAX_CARDINALITY:
        return False
    return distinct / non_null <= DEFAULT_MAX_DISTINCT_RATIO


def top_values(counts: Mapping[str, int], k: int) -> tuple[tuple[ObservedValue, ...], int]:
    """The ``k`` most common values and the count of everything else.

    Ties break by the value, ascending, so the same counts always produce the same head.
    ``k < 1`` keeps nothing and reports the whole column as the tail.
    """
    if k < 1:
        return (), sum(counts.values())
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    head = tuple(ObservedValue(value=value, count=count) for value, count in ordered[:k])
    other = sum(count for _, count in ordered[k:])
    return head, other


def measure_distribution(
    values: Sequence[object],
    *,
    source: str,
    table: str,
    column: str,
    collected_at: datetime,
    method: CollectionMethod,
    top_k: int = DEFAULT_TOP_K,
    sample_rows: int | None = None,
    covers_table: bool = False,
    unprofiled_partitions: Sequence[str] = (),
) -> CategoryColumnStats:
    """Value counts for one column. NULL is not a category value; it is a quality share.

    The distribution is complete when this reading covered the table and every distinct value
    fitted in ``top_k``. A sample, or a tail, is not complete: a missing filter value might
    still exist, so the caller must look it up before rejecting it.
    """
    counts: Counter[str] = Counter()
    for value in values:
        if value is None:
            continue
        counts[display_value(value)] += 1
    head, other = top_values(counts, top_k)
    complete = covers_table and other == 0 and not unprofiled_partitions
    return CategoryColumnStats(
        source=source,
        table=table,
        column=column,
        values=head,
        other_count=other,
        rows_seen=len(values),
        sample_rows=sample_rows,
        collected_at=collected_at,
        method=method,
        distribution_complete=complete,
        unprofiled_partitions=tuple(sorted(unprofiled_partitions)),
    )
