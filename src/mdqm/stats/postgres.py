"""PostgreSQL statistics: catalog views first, a bounded read when they are silent.

Read-only. ``pg_stats`` and ``pg_class`` answer null fraction, distinct count, min/max and the
most-common values. They do not know about zeros or blank strings, so a table the catalog has
never analyzed falls back to reading the rows up to ``sample_rows`` and measuring those.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import psycopg
import psycopg.rows

from mdqm.sql.execution import execute_one, prepare
from mdqm.stats.catalog import PG_CLASS_SQL, PG_STATS_SQL, parse_pg_class, parse_pg_stats
from mdqm.stats.collect import Collected, collect_rows
from mdqm.stats.measure import ColumnRequest
from mdqm.stats.models import CollectionMethod, ColumnQuality
from mdqm.stats.sqlgen import existence_sql, require_ident

__all__ = ["exists", "measure"]


def measure(
    connection: psycopg.Connection[tuple[Any, ...]],
    *,
    source: str,
    table: str,
    columns: Sequence[ColumnRequest],
    collected_at: datetime,
    sample_rows: int = 5_000,
) -> Collected:
    """Measure ``table`` on ``connection``. Catalog first; a capped read if the catalog is empty.

    ``table`` is a plain identifier (the relation name in the search path, including a temp
    table). Rows come back as dictionaries.
    """
    require_ident(table)
    with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        execute_one(cursor, prepare(PG_CLASS_SQL, dialect="postgres", params=(table,)))
        # Without the pg extra these fetches are Any. Narrowing with isinstance keeps mypy
        # quiet both when psycopg is installed and when CI type-checks without it.
        class_row = _dict_row(cursor.fetchone())
        row_estimate, _bytes = parse_pg_class(class_row)
        execute_one(cursor, prepare(PG_STATS_SQL, dialect="postgres", params=(table,)))
        stats_rows = _dict_rows(cursor.fetchall())
    if stats_rows and row_estimate:
        qualities, categories = parse_pg_stats(
            stats_rows,
            source=source,
            table=table,
            row_estimate=row_estimate,
            collected_at=collected_at,
        )
        # pg_stats has no share of 0, '' or whitespace. A capped read fills those three.
        # The catalog keeps null share, distinct and bounds; the method becomes mixed.
        sampled = collect_rows(
            _read_rows(connection, table, columns, sample_rows),
            columns,
            source=source,
            table=table,
            collected_at=collected_at,
            method=CollectionMethod.SAMPLE,
            covers_table=False,
            sample_rows=sample_rows,
        )
        by_sample = {item.column: item for item in sampled.quality}
        merged = tuple(_merge_quality(item, by_sample.get(item.column)) for item in qualities)
        return Collected(merged, categories, ())
    capped = _read_rows(connection, table, columns, sample_rows)
    method = CollectionMethod.EXACT if len(capped) < sample_rows else CollectionMethod.SAMPLE
    return collect_rows(
        capped,
        columns,
        source=source,
        table=table,
        collected_at=collected_at,
        method=method,
        covers_table=method is CollectionMethod.EXACT,
        sample_rows=None if method is CollectionMethod.EXACT else len(capped),
    )


def exists(
    connection: psycopg.Connection[tuple[Any, ...]],
    *,
    table: str,
    column: str,
    value: str,
) -> bool | None:
    """Whether ``value`` occurs, via ``SELECT 1 ... LIMIT 1``. ``None`` when the read fails.

    A failure is not evidence of absence: the caller must not reject a filter on it.
    """
    statement = existence_sql(table=table, column=column, value=value, dialect="postgres")
    try:
        with connection.cursor() as cursor:
            # Our own statement, built from validated identifiers, through the same chokepoint
            # every other dynamic statement uses: the guard sees it immediately before it runs.
            execute_one(cursor, prepare(statement, dialect="postgres"))
            return cursor.fetchone() is not None
    except psycopg.Error:
        return None


def _dict_row(value: object) -> dict[str, Any] | None:
    """One driver row as a dict, or None. Anything else is treated as missing."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return None


def _dict_rows(value: object) -> list[dict[str, Any]]:
    """Driver rows as dicts. A non-list (an untyped driver, or a failure) is an empty read."""
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        row = _dict_row(item)
        if row is not None:
            rows.append(row)
    return rows


def _merge_quality(catalog: ColumnQuality, sample: ColumnQuality | None) -> ColumnQuality:
    """Catalog numbers where the view had them, sample numbers for the shares it never stores."""
    if sample is None:
        return catalog
    return ColumnQuality(
        source=catalog.source,
        table=catalog.table,
        column=catalog.column,
        partition=catalog.partition,
        null_share=catalog.null_share if catalog.null_share is not None else sample.null_share,
        zero_share=sample.zero_share if catalog.zero_share is None else catalog.zero_share,
        empty_share=sample.empty_share if catalog.empty_share is None else catalog.empty_share,
        whitespace_share=(
            sample.whitespace_share
            if catalog.whitespace_share is None
            else catalog.whitespace_share
        ),
        distinct_count=(
            catalog.distinct_count if catalog.distinct_count is not None else sample.distinct_count
        ),
        minimum=catalog.minimum if catalog.minimum is not None else sample.minimum,
        maximum=catalog.maximum if catalog.maximum is not None else sample.maximum,
        rows_seen=catalog.rows_seen or sample.rows_seen,
        sample_rows=sample.rows_seen,
        method=CollectionMethod.MIXED,
        collected_at=catalog.collected_at,
    )


def _read_rows(
    connection: psycopg.Connection[tuple[Any, ...]],
    table: str,
    columns: Sequence[ColumnRequest],
    limit: int,
) -> list[dict[str, Any]]:
    """Up to ``limit`` rows. Identifiers were checked; the limit is an integer."""
    selected = ", ".join(require_ident(request.name).sql(dialect="postgres") for request in columns)
    relation = require_ident(table).sql(dialect="postgres")
    statement = f"SELECT {selected} FROM {relation} LIMIT {int(limit)}"  # noqa: S608
    with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        execute_one(cursor, prepare(statement, dialect="postgres"))
        return _dict_rows(cursor.fetchall())
