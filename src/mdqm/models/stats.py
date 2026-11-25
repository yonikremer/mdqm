"""Table statistics: how much a table changes, which decides what may be cached.

Mutability is measured, never declared. When the counters are unreadable the fallback is a checksum
over a sample, and with neither the table is treated as mutable (conservative).
"""

from __future__ import annotations

from pydantic import Field

from mdqm.models.base import MDQMModel, UtcDatetime


class TableStats(MDQMModel):
    """Counters and freshness for one table, read at ``captured_at``.

    Two readings of the same table give the deltas; nothing here is self-updating.
    """

    source: str = Field(min_length=1)
    table: str = Field(min_length=1)
    captured_at: UtcDatetime
    row_estimate: int | None = Field(default=None, ge=0)
    rows_inserted: int | None = Field(default=None, ge=0)
    rows_updated: int | None = Field(default=None, ge=0)
    rows_deleted: int | None = Field(default=None, ge=0)
    hot_updates: int | None = Field(default=None, ge=0)
    rows_modified_since_analyze: int | None = Field(default=None, ge=0)
    last_analyze: UtcDatetime | None = None
    last_autovacuum: UtcDatetime | None = None
    checksum: str | None = None

    @property
    def changes(self) -> int | None:
        """Total insert+update+delete counter, or ``None`` when the counters are unavailable."""
        counters = (self.rows_inserted, self.rows_updated, self.rows_deleted)
        if any(value is None for value in counters):
            return None
        return sum(value for value in counters if value is not None)
