"""How many rows a spec is about to read, from statistics, before any source is called."""

from __future__ import annotations

from collections.abc import Sequence

from mdqm.stats.models import PartitionExtent

__all__ = ["estimate_result_rows", "narrow_window"]


def estimate_result_rows(
    *,
    table_rows: int,
    selectivity: float = 1.0,
    grouped: bool = False,
    group_cardinality: int | None = None,
) -> int:
    """``table_rows * selectivity``, capped by the group cardinality when the spec aggregates.

    ``selectivity`` is the fraction of rows a filter keeps. It is clamped to ``[0, 1]``. A
    grouped query cannot return more groups than the grouping column has distinct values.
    """
    if table_rows < 0:
        raise ValueError(f"table_rows must be >= 0, got {table_rows}")
    kept = min(1.0, max(0.0, selectivity))
    estimated = round(table_rows * kept)
    if grouped and group_cardinality is not None:
        return min(estimated, max(0, group_cardinality))
    return estimated


def narrow_window(extents: Sequence[PartitionExtent], max_rows: int) -> tuple[str, str, int] | None:
    """The newest partitions whose combined rows stay within ``max_rows``.

    Returns ``(earliest time_min, latest time_max, rows)`` or ``None`` when no extent carries a
    time. At least the single newest partition is included, even when it alone exceeds the cap:
    the agent still gets a concrete window, and the row count says it is the smallest one.
    """
    dated = [item for item in extents if item.time_max]
    if not dated or max_rows < 1:
        return None
    dated.sort(key=lambda item: (item.time_max or "", item.name), reverse=True)
    chosen: list[PartitionExtent] = []
    total = 0
    for item in dated:
        rows = item.rows or 0
        if chosen and total + rows > max_rows:
            break
        chosen.append(item)
        total += rows
    start = min(item.time_min or item.time_max or "" for item in chosen)
    end = max(item.time_max or "" for item in chosen)
    return start, end, total
