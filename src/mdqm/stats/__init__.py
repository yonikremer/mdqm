"""Column, category and partition statistics.

D5 already profiles a table once (null rate, distinct, min/max, top values). This package is the
continuous record: the same facts, plus how often a column is ``0``, ``''`` or whitespace, how a
category is distributed inside each partition, and how one column moves with another. Collection
is read-only, incremental and quiet-window only. The linter and the Context Pack read what was
stored; they do not measure.
"""

from __future__ import annotations

from mdqm.stats.models import (
    Association,
    CategoryColumnStats,
    CollectionMethod,
    ColumnQuality,
    ObservedValue,
    PartitionExtent,
    PartitionStamp,
)

__all__ = [
    "Association",
    "CategoryColumnStats",
    "CollectionMethod",
    "ColumnQuality",
    "ObservedValue",
    "PartitionExtent",
    "PartitionStamp",
]
