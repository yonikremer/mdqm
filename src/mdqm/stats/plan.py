"""What to collect, and whether now is a legal time to collect it.

Incremental: a partition is read again only when it is new or its row or byte count changed.
Scheduling: the default quiet window is 21:00-07:00 Asia/Jerusalem, overridable by the
deployment's ``quiet_windows``. ``force`` is the explicit override a manual run passes; nothing
else skips the window.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, time

from mdqm.config.quiet import QuietWindow
from mdqm.stats.models import IncrementalPlan, PartitionExtent, PartitionStamp

__all__ = [
    "DEFAULT_STATS_WINDOW",
    "collection_open",
    "extents_to_stamps",
    "partitions_to_collect",
    "stats_windows",
    "unprofiled_partitions",
]

#: Heavy statistics collection. A deployment that sets ``quiet_windows`` replaces this.
DEFAULT_STATS_WINDOW = QuietWindow(tz="Asia/Jerusalem", start=time(21, 0), end=time(7, 0))


def stats_windows(
    configured: Sequence[QuietWindow], fallback: QuietWindow = DEFAULT_STATS_WINDOW
) -> tuple[QuietWindow, ...]:
    """The windows that apply: the deployment's, or the Jerusalem default when it set none."""
    return tuple(configured) if configured else (fallback,)


def collection_open(now: datetime, windows: Sequence[QuietWindow], *, force: bool = False) -> bool:
    """True when a collection may start. ``force`` is an operator override, not a default."""
    if force:
        return True
    return any(window.is_quiet(now) for window in windows)


def partitions_to_collect(
    current: Sequence[PartitionStamp], previous: Sequence[PartitionStamp]
) -> IncrementalPlan:
    """Partitions to read, partitions whose size is unchanged, and partitions that vanished.

    Names come out sorted. A stamp matches when the name, the row count and the byte count all
    match, so a partition that grew is collected and a partition that did not is not.
    """
    before = {item.name: (item.rows, item.bytes) for item in previous}
    now = {item.name: (item.rows, item.bytes) for item in current}
    collect = tuple(sorted(name for name, size in now.items() if before.get(name) != size))
    unchanged = tuple(sorted(name for name, size in now.items() if before.get(name) == size))
    dropped = tuple(sorted(name for name in before if name not in now))
    return IncrementalPlan(collect=collect, unchanged=unchanged, dropped=dropped)


def extents_to_stamps(extents: Sequence[PartitionExtent]) -> tuple[PartitionStamp, ...]:
    """The size half of a measured extent, which is all the incremental plan compares."""
    return tuple(
        PartitionStamp(name=item.name, rows=item.rows, bytes=item.bytes) for item in extents
    )


def unprofiled_partitions(
    extents: Sequence[PartitionExtent], profiled: Sequence[str]
) -> tuple[str, ...]:
    """Partition names we have a size for and no column profile for, sorted."""
    done = set(profiled)
    return tuple(sorted(item.name for item in extents if item.name not in done))
