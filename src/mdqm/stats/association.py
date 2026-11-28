"""How one column moves with another: Cramér's V, mutual information, conditional rates.

Pure. Candidate pairs are chosen before any of this runs, and the functions never look at more
rows than they are given, so the cost bound lives in the caller (a sample, a pair cap).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Final

from mdqm.stats.measure import display_value
from mdqm.stats.models import (
    Association,
    AssociationKind,
    CollectionMethod,
    ConditionalRate,
)

__all__ = [
    "MAX_PAIRS",
    "choose_pairs",
    "conditional_rates",
    "cramers_v",
    "measure_association",
    "mutual_information",
    "mutual_information_of",
    "nearest_rank",
]

#: How many pairs one table may be measured for. Co-used columns come first, so the cap drops
#: the pairs nobody has filtered together rather than a pair a query actually used.
MAX_PAIRS: Final[int] = 12


def choose_pairs(
    columns: Sequence[tuple[str, int, bool]],
    co_used: Sequence[tuple[str, str]] = (),
    *,
    max_pairs: int = MAX_PAIRS,
) -> tuple[tuple[str, str], ...]:
    """Ordered ``(x, y)`` pairs. ``x`` is categorical; ``y`` is another column of the same table.

    ``columns`` is ``(name, distinct, categorical)``. A pair that appears in ``co_used`` (from
    mined queries, order ignored) sorts ahead of a pair that does not. Ties break by column name,
    so the same inputs always yield the same pairs. ``max_pairs < 1`` yields nothing.
    """
    if max_pairs < 1:
        return ()
    preferred = {frozenset(pair) for pair in co_used if pair[0] != pair[1]}
    names = [name for name, _, _ in columns]
    categorical = [name for name, _, flag in columns if flag]
    candidates: list[tuple[int, str, str]] = []
    for x_name in categorical:
        for y_name in names:
            if y_name == x_name:
                continue
            rank = 0 if frozenset((x_name, y_name)) in preferred else 1
            candidates.append((rank, x_name, y_name))
    candidates.sort()
    return tuple((x_name, y_name) for _, x_name, y_name in candidates[:max_pairs])


def _contingency(pairs: Sequence[tuple[str, str]]) -> tuple[list[str], list[str], list[list[int]]]:
    """Rows, columns and counts, with both axes sorted so the table is deterministic."""
    xs = sorted({left for left, _ in pairs})
    ys = sorted({right for _, right in pairs})
    index_x = {value: index for index, value in enumerate(xs)}
    index_y = {value: index for index, value in enumerate(ys)}
    table = [[0 for _ in ys] for _ in xs]
    for left, right in pairs:
        table[index_x[left]][index_y[right]] += 1
    return xs, ys, table


def cramers_v(pairs: Sequence[tuple[str, str]]) -> float:
    """Cramér's V of two categorical columns. ``0`` is independence, ``1`` is a perfect link.

    The standard (not bias-corrected) statistic: ``sqrt(chi-square / (n * min(r - 1, c - 1)))``.
    A single-row or single-column table has no association to measure and returns ``0``.
    """
    if not pairs:
        return 0.0
    _, _, table = _contingency(pairs)
    row_sums = [sum(row) for row in table]
    col_sums = [sum(table[row][col] for row in range(len(table))) for col in range(len(table[0]))]
    n = sum(row_sums)
    if n == 0:
        return 0.0
    chi2 = 0.0
    for row, row_sum in enumerate(row_sums):
        for col, col_sum in enumerate(col_sums):
            expected = row_sum * col_sum / n
            if expected <= 0:
                continue
            observed = table[row][col]
            chi2 += (observed - expected) ** 2 / expected
    live_rows = sum(1 for total in row_sums if total > 0)
    live_cols = sum(1 for total in col_sums if total > 0)
    freedom = min(live_rows - 1, live_cols - 1)
    if freedom <= 0:
        return 0.0
    score = math.sqrt(chi2 / (n * freedom))
    return min(1.0, score)


def mutual_information(pairs: Sequence[tuple[str, str]]) -> float:
    """Mutual information in bits. ``0`` when the columns are independent on this sample."""
    if not pairs:
        return 0.0
    _, _, table = _contingency(pairs)
    row_sums = [sum(row) for row in table]
    col_sums = [sum(table[row][col] for row in range(len(table))) for col in range(len(table[0]))]
    n = sum(row_sums)
    if n == 0:
        return 0.0
    info = 0.0
    for row, row_sum in enumerate(row_sums):
        for col, col_sum in enumerate(col_sums):
            observed = table[row][col]
            if observed == 0 or row_sum == 0 or col_sum == 0:
                continue
            p_xy = observed / n
            info += p_xy * math.log2(p_xy / ((row_sum / n) * (col_sum / n)))
    return max(0.0, info)


def nearest_rank(values: Sequence[float], percentile: float) -> float:
    """Nearest-rank percentile. ``percentile`` is in ``(0, 1]``. No interpolation.

    An interpolated quantile invents a value the column never held. The planner's size quantiles
    make the same choice.
    """
    if not values:
        raise ValueError("nearest_rank needs at least one value")
    if not 0 < percentile <= 1:
        raise ValueError(f"percentile must be in (0, 1], got {percentile}")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _bucket(value: object) -> str:
    """NULL, zero, empty, whitespace, or a present value. Same rules as column quality."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "present"
    if isinstance(value, (int, float)) and value == 0:
        return "zero"
    if isinstance(value, str):
        if value == "":
            return "empty"
        if value.strip() == "":
            return "whitespace"
    return "present"


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return float(value)


def conditional_rates(
    rows: Sequence[tuple[object, object]],
) -> tuple[ConditionalRate, ...]:
    """NULL, zero, empty and whitespace share of Y inside each category of X, plus quantiles.

    Categories are the display form of X. NULL X is its own category (``""`` is not used; a
    missing X is the label ``<null>``) so it cannot collide with an empty string category.
    """
    grouped: dict[str, list[object]] = defaultdict(list)
    for x_value, y_value in rows:
        label = "<null>" if x_value is None else display_value(x_value)
        grouped[label].append(y_value)
    rates: list[ConditionalRate] = []
    for label in sorted(grouped):
        ys = grouped[label]
        counts = Counter(_bucket(value) for value in ys)
        total = len(ys)
        numbers = [number for value in ys if (number := _as_float(value)) is not None]
        rates.append(
            ConditionalRate(
                category=label,
                count=total,
                null_share=counts["null"] / total,
                zero_share=counts["zero"] / total,
                empty_share=counts["empty"] / total,
                whitespace_share=counts["whitespace"] / total,
                mean=(sum(numbers) / len(numbers)) if numbers else None,
                p50=nearest_rank(numbers, 0.50) if numbers else None,
                p90=nearest_rank(numbers, 0.90) if numbers else None,
            )
        )
    return tuple(rates)


def measure_association(
    rows: Sequence[tuple[object, object]],
    *,
    source: str,
    table: str,
    column_x: str,
    column_y: str,
    y_numeric: bool,
    collected_at: datetime,
    method: CollectionMethod,
) -> Association:
    """V and mutual information when Y is categorical, per-category quantiles when Y is numeric.

    Both categorical columns also get the conditional NULL/zero/empty rates: those are what a
    linter can say out loud ("status 'hold' is NULL in amount 80% of the time").
    """
    present = [(x, y) for x, y in rows if x is not None and y is not None]
    labelled = [(display_value(x), display_value(y)) for x, y in present]
    if y_numeric:
        kind = AssociationKind.NUMERIC
        score: float | None = None
    else:
        kind = AssociationKind.CRAMERS_V
        score = cramers_v(labelled)
    rates = conditional_rates(rows)
    return Association(
        source=source,
        table=table,
        column_x=column_x,
        column_y=column_y,
        kind=kind,
        score=score,
        by_category=rates,
        sample_rows=len(rows),
        method=method,
        collected_at=collected_at,
    )


def mutual_information_of(rows: Sequence[tuple[object, object]]) -> float:
    """Mutual information of the non-null pairs, in bits. Convenience for tests and reports."""
    labelled = [
        (display_value(x), display_value(y)) for x, y in rows if x is not None and y is not None
    ]
    return mutual_information(labelled)
