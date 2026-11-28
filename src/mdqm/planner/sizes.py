"""Size quantiles: the distribution of table and partition sizes, for the planner.

The planner needs to answer "how much will this read?" before it reads anything. A single
``reltuples`` per partition answers it badly twice: it is stale (it only moves when the source
analyzes) and it says nothing about *shape* — a window that touches two partitions of 40 000 rows is
cheap, the same two partitions when one holds 4 million is not, and the sum cannot tell you which.

So sizes are measured per table **and per partition**, kept across passes, and summarised as
quantiles: ``p50`` is the typical partition, ``p99`` is the fat one, and the ratio between them is
what makes a plan's estimate honest.

Pure functions only: no I/O, no clock, no source. Nearest-rank quantiles, deliberately: an
interpolated percentile invents a partition size nobody has, and the number here is used to decide
how much work to do.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from pydantic import Field

from mdqm.models.base import MDQMModel

__all__ = [
    "FAT_RATIO",
    "Quantiles",
    "SizeQuantiles",
    "fat_partitions",
    "quantiles",
    "summarise",
]

#: A partition more than this many times the table's median is called "fat" in a plan note. Chosen
#: so a normal monthly partition (median) never trips it while a backfill or a catch-all partition
#: does; it is a hint for the agent, never a refusal.
FAT_RATIO: Final[float] = 4.0

#: The points reported. ``p1`` and ``p99`` rather than ``p5``/``p95``: with ~34 monthly partitions,
#: p5 and p95 land on the second and thirty-third and hide the tail that matters.
POINTS: Final[tuple[int, ...]] = (1, 25, 50, 75, 99)


class Quantiles(MDQMModel):
    """The distribution of one measure across the samples it was computed from."""

    count: int = Field(ge=0)
    total: int = Field(ge=0)
    minimum: int = Field(ge=0)
    maximum: int = Field(ge=0)
    p1: int = Field(ge=0)
    p25: int = Field(ge=0)
    p50: int = Field(ge=0)
    p75: int = Field(ge=0)
    p99: int = Field(ge=0)

    @property
    def mean(self) -> float:
        """``total / count``, or ``0.0`` when nothing was measured."""
        return 0.0 if self.count == 0 else self.total / self.count

    @property
    def spread(self) -> float:
        """``maximum / p50`` — how far the biggest sample is from the typical one.

        ``0.0`` when the median is zero (nothing to compare against). A value above
        :data:`FAT_RATIO` is what the planner reports as an unbalanced table.
        """
        return 0.0 if self.p50 == 0 else self.maximum / self.p50


class SizeQuantiles(MDQMModel):
    """Rows and bytes of one table: the table itself, and its partitions when it has any."""

    rows: Quantiles
    bytes: Quantiles
    partitions: int = Field(default=0, ge=0)
    """How many partitions the partition quantiles were computed from (0: not partitioned)."""

    @property
    def unbalanced(self) -> bool:
        """True when the biggest partition dwarfs the median one."""
        return self.spread_ratio >= FAT_RATIO

    @property
    def spread_ratio(self) -> float:
        """The rows spread of the partitions, or of the table when it has none."""
        return self.rows.spread


def quantiles(values: Iterable[int | None]) -> Quantiles:
    """Nearest-rank quantiles of ``values``; ``None`` entries are absent, not zero.

    Args:
        values: one measurement per sample. ``None`` means "not reported by the source" and is
            dropped, so a source that reports rows but not bytes cannot drag the byte median to zero.

    Returns:
        The quantiles, with ``count=0`` and every field zero when nothing was reported.
    """
    known = sorted(value for value in values if value is not None)
    if not known:
        return Quantiles(count=0, total=0, minimum=0, maximum=0, p1=0, p25=0, p50=0, p75=0, p99=0)
    last = len(known) - 1
    picked = {f"p{point}": known[min(last, max(0, _rank(point, len(known))))] for point in POINTS}
    return Quantiles(
        count=len(known),
        total=sum(known),
        minimum=known[0],
        maximum=known[-1],
        **picked,
    )


def _rank(point: int, count: int) -> int:
    """The zero-based index of the nearest-rank ``point``-th percentile of ``count`` samples.

    Nearest rank: the smallest index whose cumulative share reaches the point. With one sample every
    point is that sample, which is what "min == max" should mean.
    """
    return (point * count + 99) // 100 - 1 if point < 100 else count - 1


def summarise(
    sizes: Mapping[str, tuple[int | None, int | None]], *, partitions: bool = False
) -> SizeQuantiles:
    """``SizeQuantiles`` from ``{name: (rows, bytes)}``.

    Args:
        sizes: one entry per table (or per partition when ``partitions`` is true).
        partitions: whether the samples are partitions, which only affects ``partitions``.
    """
    entries: Sequence[tuple[int | None, int | None]] = tuple(sizes.values())
    return SizeQuantiles(
        rows=quantiles(entry[0] for entry in entries),
        bytes=quantiles(entry[1] for entry in entries),
        partitions=len(entries) if partitions else 0,
    )


def fat_partitions(
    sizes: Mapping[str, tuple[int | None, int | None]], *, ratio: float = FAT_RATIO
) -> tuple[str, ...]:
    """Names of the partitions at least ``ratio`` times the median, largest first.

    Empty when the table is balanced or when nothing was measured: the caller reports a balanced
    table, never an empty hint it cannot explain.
    """
    known = {name: rows for name, (rows, _bytes) in sizes.items() if rows is not None}
    if not known:
        return ()
    median = quantiles(known.values()).p50
    if median <= 0:
        return ()
    return tuple(
        name
        for name, rows in sorted(known.items(), key=lambda item: (-item[1], item[0]))
        if rows >= median * ratio
    )
