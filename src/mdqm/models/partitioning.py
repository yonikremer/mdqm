"""Partition metadata and the strategy a source uses.

Scanning every partition is direct load on a primary database with no replica, so the metadata here
is what the planner needs *before* running anything.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from mdqm.models.base import MDQMModel, UtcDatetime


class PartitionStrategy(StrEnum):
    """How the source splits a table."""

    RANGE = "range"
    LIST = "list"
    HASH = "hash"
    EXPRESSION = "expression"
    NONE = "none"


class PartitionGranularity(StrEnum):
    """Width of a time range partition (also the indexing granularity of a time-based ES family)."""

    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class PartitionBound(MDQMModel):
    """One existing partition: half-open ``[lower, upper)`` for range, ``values`` for list.

    Range bounds are UTC datetimes because the planner aligns ranges and splits them into
    month-aligned cache segments; adapters parse the source's text form once, here.
    """

    name: str = Field(min_length=1)
    lower: UtcDatetime | None = None
    upper: UtcDatetime | None = None
    values: tuple[str, ...] = ()
    row_estimate: int | None = Field(default=None, ge=0)


class Partitioning(MDQMModel):
    """Partition metadata for one table, or ``strategy=none`` when the table is not partitioned."""

    strategy: PartitionStrategy
    key: tuple[str, ...] = ()
    expression: str | None = None
    granularity: PartitionGranularity | None = None
    bounds: tuple[PartitionBound, ...] = ()
    require_key_filter: bool = True

    @model_validator(mode="after")
    def _check_shape(self) -> Partitioning:
        if self.strategy is PartitionStrategy.NONE:
            if self.key or self.expression or self.bounds:
                raise ValueError(
                    "strategy 'none' means not partitioned: "
                    "key, expression and bounds must be empty."
                )
            return self
        if self.strategy is PartitionStrategy.EXPRESSION:
            if not self.expression:
                raise ValueError("strategy 'expression' requires the partition expression.")
            return self
        if not self.key:
            raise ValueError(f"strategy '{self.strategy.value}' requires at least one key column.")
        if self.expression:
            raise ValueError(
                f"strategy '{self.strategy.value}' keys on columns; "
                "use 'expression' for a function."
            )
        return self

    @property
    def partitions_total(self) -> int | None:
        """Number of known partitions, or ``None`` when the probe could not enumerate them."""
        return len(self.bounds) or None
