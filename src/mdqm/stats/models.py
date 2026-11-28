"""Measured column and partition statistics.

These are measurements, not declarations, so they live in the metadata store rather than in the
semantic layer (a nightly refresh must not change the layer's version hash). ``None`` means the
source did not report the number. A share is a fraction of the rows that were seen, in ``[0, 1]``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from mdqm.models.base import MDQMModel, UtcDatetime

__all__ = [
    "Association",
    "AssociationKind",
    "CategoryColumnStats",
    "CollectionMethod",
    "ColumnQuality",
    "ConditionalRate",
    "IncrementalPlan",
    "ObservedValue",
    "PartitionExtent",
    "PartitionStamp",
]


class CollectionMethod(StrEnum):
    """Where a number came from. Recorded on every row so a later reader can judge it.

    ``CATALOG`` is the source's own optimizer statistics. ``SAMPLE`` is a bounded
    ``TABLESAMPLE`` (or the dialect's equivalent). ``EXACT`` is a full aggregate of a table small
    enough to read. ``MIXED`` used the catalog for what it knows and a sample for the rest.
    """

    CATALOG = "catalog"
    SAMPLE = "sample"
    EXACT = "exact"
    MIXED = "mixed"


class ColumnQuality(MDQMModel):
    """NULL, zero, empty and whitespace shares, plus distinct and numeric/time bounds.

    ``partition`` is empty for a whole-table measurement. ``sample_rows`` is ``None`` when no
    sample was taken (a catalog-only reading). ``rows_seen`` is the denominator of the shares
    that came from a scan; a catalog fraction whose row estimate is unknown uses ``0``.
    """

    source: str = Field(min_length=1)
    table: str = Field(min_length=1)
    column: str = Field(min_length=1)
    partition: str = ""
    null_share: float | None = Field(default=None, ge=0.0, le=1.0)
    zero_share: float | None = Field(default=None, ge=0.0, le=1.0)
    empty_share: float | None = Field(default=None, ge=0.0, le=1.0)
    whitespace_share: float | None = Field(default=None, ge=0.0, le=1.0)
    distinct_count: int | None = Field(default=None, ge=0)
    minimum: str | None = None
    maximum: str | None = None
    rows_seen: int = Field(default=0, ge=0)
    sample_rows: int | None = Field(default=None, ge=0)
    method: CollectionMethod
    collected_at: UtcDatetime

    @property
    def qualified(self) -> str:
        """``source.table.column``, the key the linter and the Context Pack use."""
        return f"{self.source}.{self.table}.{self.column}"


class ObservedValue(MDQMModel):
    """One categorical value and how many sampled rows held it."""

    value: str
    count: int = Field(ge=0)


class CategoryColumnStats(MDQMModel):
    """Top values of one categorical column, plus the tail and whether the list is complete.

    ``distribution_complete`` is true only when every distinct value is listed (the tail is
    empty) and the reading covered the table, not a sample that happened to miss a rare value.
    ``unprofiled_partitions`` names partitions whose rows may be newer than ``collected_at``.
    A value missing from a complete, fully profiled column can be rejected immediately. A value
    missing from anything else must be checked with a bounded lookup first.
    """

    source: str = Field(min_length=1)
    table: str = Field(min_length=1)
    column: str = Field(min_length=1)
    #: Empty for the whole table. A partition name or a time bucket (``2026-11``) otherwise.
    bucket: str = ""
    values: tuple[ObservedValue, ...] = ()
    other_count: int = Field(default=0, ge=0)
    rows_seen: int = Field(default=0, ge=0)
    sample_rows: int | None = Field(default=None, ge=0)
    collected_at: UtcDatetime
    method: CollectionMethod
    distribution_complete: bool = False
    unprofiled_partitions: tuple[str, ...] = ()

    @property
    def qualified(self) -> str:
        """``source.table.column``."""
        return f"{self.source}.{self.table}.{self.column}"

    def share_of(self, count: int) -> float:
        """``count / rows_seen``, or ``0`` when nothing was seen."""
        if self.rows_seen <= 0:
            return 0.0
        return count / self.rows_seen


class AssociationKind(StrEnum):
    """What ``Association.score`` means.

    ``CRAMERS_V`` and ``MUTUAL_INFO`` are for two categorical columns. ``NUMERIC`` is a
    low-cardinality column against a number: the score is unused and ``by_category`` holds the
    per-category mean and quantiles.
    """

    CRAMERS_V = "cramers_v"
    MUTUAL_INFO = "mutual_info"
    NUMERIC = "numeric"


class ConditionalRate(MDQMModel):
    """How column Y behaves inside one category of X."""

    category: str
    count: int = Field(ge=0)
    null_share: float | None = Field(default=None, ge=0.0, le=1.0)
    zero_share: float | None = Field(default=None, ge=0.0, le=1.0)
    empty_share: float | None = Field(default=None, ge=0.0, le=1.0)
    whitespace_share: float | None = Field(default=None, ge=0.0, le=1.0)
    mean: float | None = None
    p50: float | None = None
    p90: float | None = None


class Association(MDQMModel):
    """One measured relationship between two columns of the same table.

    ``score`` is Cramér's V or mutual information in bits. It is ``None`` for a numeric
    relationship, which is described entirely by ``by_category``.
    """

    source: str = Field(min_length=1)
    table: str = Field(min_length=1)
    column_x: str = Field(min_length=1)
    column_y: str = Field(min_length=1)
    kind: AssociationKind
    score: float | None = Field(default=None, ge=0.0)
    by_category: tuple[ConditionalRate, ...] = ()
    sample_rows: int = Field(ge=0)
    method: CollectionMethod
    collected_at: UtcDatetime


class PartitionExtent(MDQMModel):
    """One partition: how many rows and bytes it holds, and the time span of its time column."""

    source: str = Field(min_length=1)
    table: str = Field(min_length=1)
    name: str = Field(min_length=1)
    rows: int | None = Field(default=None, ge=0)
    bytes: int | None = Field(default=None, ge=0)
    time_column: str | None = None
    time_min: str | None = None
    time_max: str | None = None
    method: CollectionMethod
    collected_at: UtcDatetime


class PartitionStamp(MDQMModel):
    """The identity of a partition for incremental collection: its name and its last size.

    A partition is collected again only when it is new or its row or byte count changed. Two
    stamps with the same name, rows and bytes are the same partition.
    """

    name: str = Field(min_length=1)
    rows: int | None = Field(default=None, ge=0)
    bytes: int | None = Field(default=None, ge=0)


class IncrementalPlan(MDQMModel):
    """Which partitions to read tonight, which to skip, and which disappeared."""

    collect: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
