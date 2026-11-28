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

from mdqm.stats.catalog import PG_CLASS_SQL, PG_STATS_SQL, parse_pg_class, parse_pg_stats
from mdqm.stats.collect import Collected, collect_rows
from mdqm.stats.measure import ColumnRequest
from mdqm.stats.models import CollectionMethod
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
        cursor.execute(PG_CLASS_SQL, (table,))
        class_row = cursor.fetchone()
        row_estimate, _bytes = parse_pg_class(class_row)
        cursor.execute(PG_STATS_SQL, (table,))
        stats_rows = cursor.fetchall()
    if stats_rows and row_estimate:
        qualities, categories = parse_pg_stats(
            stats_rows,
            source=source,
            table=table,
            row_estimate=row_estimate,
            collected_at=collected_at,
        )
        return Collected(qualities, categories, ())
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
            cursor.execute(statement)
            return cursor.fetchone() is not None
    except psycopg.Error:
        return None


def _read_rows(
    connection: psycopg.Connection[tuple[Any, ...]],
    table: str,
    columns: Sequence[ColumnRequest],
    limit: int,
) -> list[dict[str, object]]:
    """Up to ``limit`` rows. Identifiers were checked; the limit is an integer."""
    selected = ", ".join(require_ident(request.name).sql(dialect="postgres") for request in columns)
    relation = require_ident(table).sql(dialect="postgres")
    statement = f"SELECT {selected} FROM {relation} LIMIT {int(limit)}"  # noqa: S608
    with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        cursor.execute(statement)
        return list(cursor.fetchall())
