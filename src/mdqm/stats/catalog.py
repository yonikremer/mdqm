"""Turn a source's own statistics views into column quality and partition extents.

The SQL constants are what an adapter runs. The parsers accept the rows those statements
return, already decoded (Oracle's ``LOW_VALUE`` is RAW; the adapter decodes it before this
module sees it). PostgreSQL is executed by :mod:`mdqm.stats.postgres`. Oracle, SQL Server and
Elasticsearch parsers are the same shape, tested against fixture rows, so a later adapter does
not invent a second mapping.

None of these views know how often a column is ``0``, ``''`` or whitespace. Those shares stay
``None`` on a catalog reading and a sample fills them in. The method stays ``catalog`` until
that happens.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Final

from mdqm.stats.measure import top_values
from mdqm.stats.models import (
    CategoryColumnStats,
    CollectionMethod,
    ColumnQuality,
    PartitionExtent,
)

__all__ = [
    "ES_INDEX_STATS",
    "ORACLE_COLUMN_STATS_SQL",
    "PG_CLASS_SQL",
    "PG_STATS_SQL",
    "SQLSERVER_COLUMN_STATS_SQL",
    "parse_es_index_stats",
    "parse_oracle_column_stats",
    "parse_pg_class",
    "parse_pg_stats",
    "parse_sqlserver_column_stats",
]

#: ``pg_stats`` for one table. ``tablename`` is bound; the schema is whatever the search path sees.
PG_STATS_SQL: Final[str] = (
    "SELECT attname, null_frac, n_distinct, most_common_vals, most_common_freqs, histogram_bounds "
    "FROM pg_stats WHERE tablename = %s ORDER BY attname"
)

#: ``pg_class`` row estimate and bytes (``relpages`` times the 8 KiB page). Bound name.
PG_CLASS_SQL: Final[str] = (
    "SELECT reltuples, relpages FROM pg_class WHERE relname = %s "
    "AND relnamespace = ANY (SELECT oid FROM pg_namespace WHERE nspname = ANY (current_schemas(true)))"
)

#: ``ALL_TAB_COL_STATISTICS`` joined to ``ALL_TABLES`` for the denominator. Bind owner and table.
ORACLE_COLUMN_STATS_SQL: Final[str] = (
    "SELECT c.column_name, c.num_nulls, c.num_distinct, c.sample_size, t.num_rows "
    "FROM all_tab_col_statistics c "
    "JOIN all_tables t ON t.owner = c.owner AND t.table_name = c.table_name "
    "WHERE c.owner = :owner AND c.table_name = :table_name "
    "ORDER BY c.column_name"
)

#: ``sys.dm_db_stats_histogram`` does not carry nulls. Nulls come from ``sys.stats`` via
#: ``sys.dm_db_stats_properties`` when the caller has permission; otherwise ``null_count`` is
#: absent and the parser leaves ``null_share`` empty. The statement is the documented shape an
#: adapter normalises into :func:`parse_sqlserver_column_stats` rows, not a string we execute
#: blindly: the histogram step is verified when a SQL Server service is available.
SQLSERVER_COLUMN_STATS_SQL: Final[str] = (
    "SELECT c.name AS name, s.rows AS rows, p.rows_sampled AS rows_sampled "
    "FROM sys.tables t "
    "JOIN sys.columns c ON c.object_id = t.object_id "
    "JOIN sys.stats st ON st.object_id = t.object_id "
    "CROSS APPLY sys.dm_db_stats_properties(st.object_id, st.stats_id) p "
    "JOIN sys.partitions s ON s.object_id = t.object_id AND s.index_id IN (0, 1) "
    "WHERE t.name = ? "
    "ORDER BY c.column_id"
)

#: Elasticsearch ``GET index/_stats`` fields this parser reads. An alias is not a partition.
ES_INDEX_STATS: Final[str] = (
    "indices.*.primaries.docs.count, indices.*.primaries.store.size_in_bytes"
)


def _distinct(n_distinct: float, row_estimate: int) -> int:
    """``pg_stats.n_distinct``: a positive value is a count, a negative one is a fraction of rows."""
    if n_distinct >= 0:
        return round(n_distinct)
    return round(abs(n_distinct) * max(0, row_estimate))


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def parse_pg_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str,
    table: str,
    row_estimate: int,
    collected_at: datetime,
) -> tuple[tuple[ColumnQuality, ...], tuple[CategoryColumnStats, ...]]:
    """Column quality and, where the most-common values are the whole distribution, categories.

    A negative ``n_distinct`` is an estimate, so that column's distribution is never marked
    complete. A positive ``n_distinct`` equal to the number of most-common values, with no
    histogram, is the whole distribution (PostgreSQL stores no histogram for a low-cardinality
    column).
    """
    qualities: list[ColumnQuality] = []
    categories: list[CategoryColumnStats] = []
    for row in rows:
        name = row.get("attname")
        if not isinstance(name, str) or not name:
            continue
        null_frac = _as_float(row.get("null_frac"))
        n_distinct = _as_float(row.get("n_distinct"))
        distinct = None if n_distinct is None else _distinct(n_distinct, row_estimate)
        histogram = row.get("histogram_bounds")
        commons = row.get("most_common_vals")
        freqs = row.get("most_common_freqs")
        minimum, maximum = _bounds(histogram, commons)
        qualities.append(
            ColumnQuality(
                source=source,
                table=table,
                column=name,
                null_share=null_frac,
                distinct_count=distinct,
                minimum=minimum,
                maximum=maximum,
                rows_seen=max(0, row_estimate),
                method=CollectionMethod.CATALOG,
                collected_at=collected_at,
            )
        )
        if (
            isinstance(commons, Sequence)
            and not isinstance(commons, (str, bytes))
            and isinstance(freqs, Sequence)
            and not isinstance(freqs, (str, bytes))
            and distinct is not None
            and n_distinct is not None
            and n_distinct >= 0
            and len(commons) == distinct
            and not histogram
        ):
            counts = {
                str(value): round(float(freq) * max(row_estimate, 0))
                for value, freq in zip(commons, freqs, strict=False)
                if isinstance(freq, (int, float)) and not isinstance(freq, bool)
            }
            head, other = top_values(counts, len(counts))
            categories.append(
                CategoryColumnStats(
                    source=source,
                    table=table,
                    column=name,
                    values=head,
                    other_count=other,
                    rows_seen=max(0, row_estimate),
                    collected_at=collected_at,
                    method=CollectionMethod.CATALOG,
                    distribution_complete=other == 0 and row_estimate > 0,
                )
            )
    return tuple(qualities), tuple(categories)


def _bounds(histogram: object, commons: object) -> tuple[str | None, str | None]:
    for source in (histogram, commons):
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes)) and source:
            return str(source[0]), str(source[-1])
    return None, None


def parse_pg_class(row: Mapping[str, Any] | None) -> tuple[int | None, int | None]:
    """``(row estimate, bytes)`` from one ``pg_class`` row. Bytes are ``relpages * 8192``."""
    if row is None:
        return None, None
    tuples = _as_float(row.get("reltuples"))
    pages = _as_int(row.get("relpages"))
    rows = None if tuples is None or tuples < 0 else int(tuples)
    size = None if pages is None or pages < 0 else pages * 8192
    return rows, size


def parse_oracle_column_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str,
    table: str,
    collected_at: datetime,
) -> tuple[ColumnQuality, ...]:
    """Null share and distinct count from ``ALL_TAB_COL_STATISTICS``.

    ``num_nulls / num_rows``. A table with no ``num_rows`` leaves the share empty rather than
    dividing by zero. Distinct is ``num_distinct`` when it is present.
    """
    parsed: list[ColumnQuality] = []
    for row in rows:
        name = row.get("column_name")
        if not isinstance(name, str) or not name:
            continue
        num_rows = _as_int(row.get("num_rows"))
        num_nulls = _as_int(row.get("num_nulls"))
        distinct = _as_int(row.get("num_distinct"))
        null_share = None
        if num_rows is not None and num_rows > 0 and num_nulls is not None and num_nulls >= 0:
            null_share = min(1.0, num_nulls / num_rows)
        low = row.get("low_value")
        high = row.get("high_value")
        parsed.append(
            ColumnQuality(
                source=source,
                table=table,
                column=name,
                null_share=null_share,
                distinct_count=distinct if distinct is None or distinct >= 0 else None,
                minimum=None if low is None else str(low),
                maximum=None if high is None else str(high),
                rows_seen=num_rows or 0,
                method=CollectionMethod.CATALOG,
                collected_at=collected_at,
            )
        )
    return tuple(parsed)


def parse_sqlserver_column_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str,
    table: str,
    collected_at: datetime,
) -> tuple[ColumnQuality, ...]:
    """The normalised stats-DMV row: ``name``, ``rows``, optional ``null_count``, ``distinct``.

    The adapter is responsible for shaping ``sys.dm_db_stats_*`` into this. A missing
    ``null_count`` stays a missing share.
    """
    parsed: list[ColumnQuality] = []
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str) or not name:
            continue
        total = _as_int(row.get("rows"))
        nulls = _as_int(row.get("null_count"))
        distinct = _as_int(row.get("distinct"))
        null_share = None
        if total is not None and total > 0 and nulls is not None and nulls >= 0:
            null_share = min(1.0, nulls / total)
        minimum = row.get("minimum")
        maximum = row.get("maximum")
        parsed.append(
            ColumnQuality(
                source=source,
                table=table,
                column=name,
                null_share=null_share,
                distinct_count=distinct if distinct is None or distinct >= 0 else None,
                minimum=None if minimum is None else str(minimum),
                maximum=None if maximum is None else str(maximum),
                rows_seen=total or 0,
                method=CollectionMethod.CATALOG,
                collected_at=collected_at,
            )
        )
    return tuple(parsed)


def parse_es_index_stats(
    indices: Mapping[str, Mapping[str, Any]],
    *,
    source: str,
    table: str,
    collected_at: datetime,
    time_column: str | None = None,
) -> tuple[PartitionExtent, ...]:
    """One extent per index. ``docs.count`` is rows, ``store.size_in_bytes`` is bytes.

    Index names come out sorted. A missing side stays ``None``.
    """
    extents: list[PartitionExtent] = []
    for name in sorted(indices):
        body = indices[name]
        primaries = body.get("primaries", body)
        docs = primaries.get("docs") if isinstance(primaries, Mapping) else None
        store = primaries.get("store") if isinstance(primaries, Mapping) else None
        count = docs.get("count") if isinstance(docs, Mapping) else None
        size = store.get("size_in_bytes") if isinstance(store, Mapping) else None
        extents.append(
            PartitionExtent(
                source=source,
                table=table,
                name=name,
                rows=_as_int(count),
                bytes=_as_int(size),
                time_column=time_column,
                method=CollectionMethod.CATALOG,
                collected_at=collected_at,
            )
        )
    return tuple(extents)
