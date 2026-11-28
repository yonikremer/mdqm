"""Persist column, category, association and partition statistics in the metadata store.

Rows are appended. The newest ``collected_at`` of a column is the reading the linter uses; the
older rows are how drift is visible. Payloads are canonical JSON (sorted keys, UTF-8) so a Hebrew
value survives and two equal readings compare equal.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Final

from mdqm.sql.execution import run_local
from mdqm.stats.models import (
    Association,
    AssociationKind,
    CategoryColumnStats,
    CollectionMethod,
    ColumnQuality,
    ConditionalRate,
    ObservedValue,
    PartitionExtent,
)
from mdqm.store.db import Store

__all__ = ["StatsStore"]

_INSERT_QUALITY: Final = (
    "INSERT INTO data_column_quality (source, table_name, column_name, partition_name, "
    "null_share, zero_share, empty_share, whitespace_share, distinct_count, minimum, maximum, "
    "rows_seen, sample_rows, method, collected_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_INSERT_CATEGORY: Final = (
    "INSERT INTO data_category_bucket (source, table_name, column_name, bucket, payload, "
    "other_count, rows_seen, sample_rows, method, distribution_complete, unprofiled, collected_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_INSERT_ASSOCIATION: Final = (
    "INSERT INTO data_association (source, table_name, column_x, column_y, kind, score, payload, "
    "sample_rows, method, collected_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_INSERT_EXTENT: Final = (
    "INSERT INTO data_partition_extent (source, table_name, partition_name, rows, bytes, "
    "time_column, time_min, time_max, method, collected_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _load(payload: str) -> Any:
    return json.loads(payload)


class StatsStore:
    """Read and write the statistics tables over one :class:`~mdqm.store.db.Store`."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def record_quality(self, rows: Sequence[ColumnQuality]) -> None:
        """Append column-quality readings."""
        if not rows:
            return
        with self._store.writing() as conn:
            for item in rows:
                run_local(
                    conn,
                    _INSERT_QUALITY,
                    (
                        item.source,
                        item.table,
                        item.column,
                        item.partition,
                        item.null_share,
                        item.zero_share,
                        item.empty_share,
                        item.whitespace_share,
                        item.distinct_count,
                        item.minimum,
                        item.maximum,
                        item.rows_seen,
                        item.sample_rows,
                        item.method.value,
                        item.collected_at.isoformat(),
                    ),
                )

    def record_categories(self, rows: Sequence[CategoryColumnStats]) -> None:
        """Append categorical distributions, one row per column per bucket."""
        if not rows:
            return
        with self._store.writing() as conn:
            for item in rows:
                payload = [{"value": value.value, "count": value.count} for value in item.values]
                run_local(
                    conn,
                    _INSERT_CATEGORY,
                    (
                        item.source,
                        item.table,
                        item.column,
                        item.bucket,
                        _dump(payload),
                        item.other_count,
                        item.rows_seen,
                        item.sample_rows,
                        item.method.value,
                        1 if item.distribution_complete else 0,
                        _dump(list(item.unprofiled_partitions)),
                        item.collected_at.isoformat(),
                    ),
                )

    def record_associations(self, rows: Sequence[Association]) -> None:
        """Append association readings."""
        if not rows:
            return
        with self._store.writing() as conn:
            for item in rows:
                payload = [rate.model_dump(mode="json") for rate in item.by_category]
                run_local(
                    conn,
                    _INSERT_ASSOCIATION,
                    (
                        item.source,
                        item.table,
                        item.column_x,
                        item.column_y,
                        item.kind.value,
                        item.score,
                        _dump(payload),
                        item.sample_rows,
                        item.method.value,
                        item.collected_at.isoformat(),
                    ),
                )

    def record_extents(self, rows: Sequence[PartitionExtent]) -> None:
        """Append partition sizes."""
        if not rows:
            return
        with self._store.writing() as conn:
            for item in rows:
                run_local(
                    conn,
                    _INSERT_EXTENT,
                    (
                        item.source,
                        item.table,
                        item.name,
                        item.rows,
                        item.bytes,
                        item.time_column,
                        item.time_min,
                        item.time_max,
                        item.method.value,
                        item.collected_at.isoformat(),
                    ),
                )

    def latest_quality(self, source: str, table: str) -> tuple[ColumnQuality, ...]:
        """The newest whole-table quality row of each column, sorted by column name."""
        sql = (
            "SELECT q.source, q.table_name, q.column_name, q.partition_name, q.null_share, "
            "q.zero_share, q.empty_share, q.whitespace_share, q.distinct_count, q.minimum, "
            "q.maximum, q.rows_seen, q.sample_rows, q.method, q.collected_at "
            "FROM data_column_quality AS q "
            "JOIN (SELECT column_name, partition_name, MAX(collected_at) AS collected_at "
            "FROM data_column_quality WHERE source = ? AND table_name = ? "
            "GROUP BY column_name, partition_name) AS n "
            "ON n.column_name = q.column_name AND n.partition_name = q.partition_name "
            "AND n.collected_at = q.collected_at "
            "WHERE q.source = ? AND q.table_name = ? "
            "ORDER BY q.column_name, q.partition_name, q.id"
        )
        found = run_local(self._store.connect(), sql, (source, table, source, table)).fetchall()
        return tuple(_quality(row) for row in found)

    def latest_categories(self, source: str, table: str) -> tuple[CategoryColumnStats, ...]:
        """The newest distribution of each column and bucket."""
        sql = (
            "SELECT c.source, c.table_name, c.column_name, c.bucket, c.payload, c.other_count, "
            "c.rows_seen, c.sample_rows, c.method, c.distribution_complete, c.unprofiled, "
            "c.collected_at FROM data_category_bucket AS c "
            "JOIN (SELECT column_name, bucket, MAX(collected_at) AS collected_at "
            "FROM data_category_bucket WHERE source = ? AND table_name = ? "
            "GROUP BY column_name, bucket) AS n "
            "ON n.column_name = c.column_name AND n.bucket = c.bucket "
            "AND n.collected_at = c.collected_at "
            "WHERE c.source = ? AND c.table_name = ? "
            "ORDER BY c.column_name, c.bucket, c.id"
        )
        found = run_local(self._store.connect(), sql, (source, table, source, table)).fetchall()
        return tuple(_category(row) for row in found)

    def latest_associations(self, source: str, table: str) -> tuple[Association, ...]:
        """The newest association of each column pair."""
        sql = (
            "SELECT a.source, a.table_name, a.column_x, a.column_y, a.kind, a.score, a.payload, "
            "a.sample_rows, a.method, a.collected_at FROM data_association AS a "
            "JOIN (SELECT column_x, column_y, MAX(collected_at) AS collected_at "
            "FROM data_association WHERE source = ? AND table_name = ? "
            "GROUP BY column_x, column_y) AS n "
            "ON n.column_x = a.column_x AND n.column_y = a.column_y "
            "AND n.collected_at = a.collected_at "
            "WHERE a.source = ? AND a.table_name = ? "
            "ORDER BY a.column_x, a.column_y, a.id"
        )
        found = run_local(self._store.connect(), sql, (source, table, source, table)).fetchall()
        return tuple(_association(row) for row in found)

    def latest_extents(self, source: str, table: str) -> tuple[PartitionExtent, ...]:
        """The newest size of each partition, sorted by name."""
        sql = (
            "SELECT e.source, e.table_name, e.partition_name, e.rows, e.bytes, e.time_column, "
            "e.time_min, e.time_max, e.method, e.collected_at FROM data_partition_extent AS e "
            "JOIN (SELECT partition_name, MAX(collected_at) AS collected_at "
            "FROM data_partition_extent WHERE source = ? AND table_name = ? "
            "GROUP BY partition_name) AS n "
            "ON n.partition_name = e.partition_name AND n.collected_at = e.collected_at "
            "WHERE e.source = ? AND e.table_name = ? ORDER BY e.partition_name, e.id"
        )
        found = run_local(self._store.connect(), sql, (source, table, source, table)).fetchall()
        return tuple(_extent(row) for row in found)


def _when(value: object) -> datetime:
    return datetime.fromisoformat(str(value))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"a share must be a number, got {type(value).__name__}")
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"a count must be a number, got {type(value).__name__}")
    return int(value)


def _quality(row: Any) -> ColumnQuality:
    return ColumnQuality(
        source=str(row["source"]),
        table=str(row["table_name"]),
        column=str(row["column_name"]),
        partition=str(row["partition_name"]),
        null_share=_optional_float(row["null_share"]),
        zero_share=_optional_float(row["zero_share"]),
        empty_share=_optional_float(row["empty_share"]),
        whitespace_share=_optional_float(row["whitespace_share"]),
        distinct_count=_optional_int(row["distinct_count"]),
        minimum=None if row["minimum"] is None else str(row["minimum"]),
        maximum=None if row["maximum"] is None else str(row["maximum"]),
        rows_seen=int(row["rows_seen"]),
        sample_rows=_optional_int(row["sample_rows"]),
        method=CollectionMethod(str(row["method"])),
        collected_at=_when(row["collected_at"]),
    )


def _category(row: Any) -> CategoryColumnStats:
    payload = _load(str(row["payload"]))
    values = tuple(
        ObservedValue(value=str(item["value"]), count=int(item["count"])) for item in payload
    )
    unprofiled = _load(str(row["unprofiled"]))
    return CategoryColumnStats(
        source=str(row["source"]),
        table=str(row["table_name"]),
        column=str(row["column_name"]),
        bucket=str(row["bucket"]),
        values=values,
        other_count=int(row["other_count"]),
        rows_seen=int(row["rows_seen"]),
        sample_rows=_optional_int(row["sample_rows"]),
        method=CollectionMethod(str(row["method"])),
        distribution_complete=bool(row["distribution_complete"]),
        unprofiled_partitions=tuple(str(item) for item in unprofiled),
        collected_at=_when(row["collected_at"]),
    )


def _association(row: Any) -> Association:
    payload = _load(str(row["payload"]))
    return Association(
        source=str(row["source"]),
        table=str(row["table_name"]),
        column_x=str(row["column_x"]),
        column_y=str(row["column_y"]),
        kind=AssociationKind(str(row["kind"])),
        score=_optional_float(row["score"]),
        by_category=tuple(ConditionalRate.model_validate(item) for item in payload),
        sample_rows=int(row["sample_rows"]),
        method=CollectionMethod(str(row["method"])),
        collected_at=_when(row["collected_at"]),
    )


def _extent(row: Any) -> PartitionExtent:
    return PartitionExtent(
        source=str(row["source"]),
        table=str(row["table_name"]),
        name=str(row["partition_name"]),
        rows=_optional_int(row["rows"]),
        bytes=_optional_int(row["bytes"]),
        time_column=None if row["time_column"] is None else str(row["time_column"]),
        time_min=None if row["time_min"] is None else str(row["time_min"]),
        time_max=None if row["time_max"] is None else str(row["time_max"]),
        method=CollectionMethod(str(row["method"])),
        collected_at=_when(row["collected_at"]),
    )
