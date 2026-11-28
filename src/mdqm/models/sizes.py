"""Measured sizes: what a table and each of its partitions actually hold.

These are **measurements**, not declarations, which is why they live in the metadata store beside the
ledger rather than in the semantic layer (a size changes nightly, and the layer's version hash covers
every fact in it). ``rows`` and ``bytes`` are nullable on purpose: a source may report one and not the
other, and ``None`` means "not reported", never zero.
"""

from __future__ import annotations

from mdqm.models.base import MDQMModel, UtcDatetime

__all__ = ["PartitionSize", "TableSize"]


class PartitionSize(MDQMModel):
    """One partition: the relation name the source uses, with its rows and bytes."""

    name: str
    rows: int | None = None
    bytes: int | None = None


class TableSize(MDQMModel):
    """One table at ``captured_at``: its own size, and one entry per partition when it has any.

    ``table`` is the **bare** name the source reports (``usage_events``), because that is what the
    partition bounds are named after. The store keys it together with the source, so two sources can
    both hold a ``customers``.
    """

    source: str
    table: str
    captured_at: UtcDatetime
    rows: int | None = None
    bytes: int | None = None
    partitions: tuple[PartitionSize, ...] = ()

    @property
    def partition_rows(self) -> dict[str, int | None]:
        """``{partition name: rows}``: what the planner needs to estimate a window."""
        return {partition.name: partition.rows for partition in self.partitions}
