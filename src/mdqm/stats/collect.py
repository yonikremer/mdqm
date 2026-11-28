"""Build statistics from rows in memory, or from a DuckDB table.

DuckDB is the unit-test source and the local engine. It never installs an extension: the sample
is a reservoir over a table this process already holds. PostgreSQL collection lives in
:mod:`mdqm.stats.postgres` so a machine without ``psycopg`` can still import this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

import duckdb
from sqlglot import exp

from mdqm.sql.execution import run_local
from mdqm.stats.association import choose_pairs, measure_association
from mdqm.stats.measure import (
    DEFAULT_TOP_K,
    ColumnRequest,
    is_categorical,
    measure_column,
    measure_distribution,
)
from mdqm.stats.models import (
    Association,
    CategoryColumnStats,
    CollectionMethod,
    ColumnQuality,
)
from mdqm.stats.sqlgen import require_ident, sample_sql

__all__ = [
    "Collected",
    "bucket_key",
    "collect_duckdb",
    "collect_rows",
    "distribution_drift",
]


class Collected:
    """One pass over one table: quality, distributions (whole table and per bucket), associations."""

    __slots__ = ("associations", "distributions", "quality")

    def __init__(
        self,
        quality: tuple[ColumnQuality, ...],
        distributions: tuple[CategoryColumnStats, ...],
        associations: tuple[Association, ...],
    ) -> None:
        self.quality = quality
        self.distributions = distributions
        self.associations = associations


def bucket_key(value: object) -> str:
    """A time value as ``YYYY-MM``, or the display text of anything else. NULL is ``<null>``."""
    if value is None:
        return "<null>"
    if isinstance(value, datetime):
        return f"{value.year:04d}-{value.month:02d}"
    text = str(value)
    if len(text) >= 7 and text[4:5] == "-" and text[:4].isdigit() and text[5:7].isdigit():
        return text[:7]
    return text


def _values(rows: Sequence[Mapping[str, object]], name: str) -> tuple[object, ...]:
    return tuple(row.get(name) for row in rows)


def distribution_drift(
    before: CategoryColumnStats, after: CategoryColumnStats
) -> tuple[tuple[str, float, float], ...]:
    """Values whose share changed between two readings of the same column.

    The result is ``(value, old share, new share)``, largest absolute change first, then the
    value. A value that exists on only one side has share ``0`` on the other. Shares use each
    reading's own ``rows_seen``.
    """
    old = {item.value: before.share_of(item.count) for item in before.values}
    new = {item.value: after.share_of(item.count) for item in after.values}
    keys = sorted(set(old) | set(new))
    moved = [
        (key, old.get(key, 0.0), new.get(key, 0.0))
        for key in keys
        if old.get(key, 0.0) != new.get(key, 0.0)
    ]
    moved.sort(key=lambda item: (-abs(item[2] - item[1]), item[0]))
    return tuple(moved)


def collect_rows(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[ColumnRequest],
    *,
    source: str,
    table: str,
    collected_at: datetime,
    method: CollectionMethod,
    bucket_column: str | None = None,
    co_used: Sequence[tuple[str, str]] = (),
    max_pairs: int = 12,
    top_k: int = DEFAULT_TOP_K,
    covers_table: bool = True,
    sample_rows: int | None = None,
) -> Collected:
    """Measure ``columns`` on ``rows``. Categorical columns also get a per-bucket distribution.

    Associations are computed only for the pairs :func:`~mdqm.stats.association.choose_pairs`
    returns, so a wide table does not become a wide cross product.
    """
    quality: list[ColumnQuality] = []
    distributions: list[CategoryColumnStats] = []
    sketches: list[tuple[str, int, bool]] = []
    for request in columns:
        values = _values(rows, request.name)
        measured = measure_column(
            values,
            request,
            source=source,
            table=table,
            collected_at=collected_at,
            method=method,
            sample_rows=sample_rows,
        )
        quality.append(measured)
        non_null = sum(1 for value in values if value is not None)
        distinct = measured.distinct_count or 0
        categorical = is_categorical(distinct=distinct, non_null=non_null)
        sketches.append((request.name, distinct, categorical))
        if not categorical:
            continue
        distributions.append(
            measure_distribution(
                values,
                source=source,
                table=table,
                column=request.name,
                collected_at=collected_at,
                method=method,
                top_k=top_k,
                sample_rows=sample_rows,
                covers_table=covers_table,
            )
        )
        if bucket_column is None:
            continue
        grouped: dict[str, list[object]] = {}
        for row in rows:
            grouped.setdefault(bucket_key(row.get(bucket_column)), []).append(row.get(request.name))
        for name in sorted(grouped):
            bucket_values = tuple(grouped[name])
            stats = measure_distribution(
                bucket_values,
                source=source,
                table=table,
                column=request.name,
                collected_at=collected_at,
                method=method,
                top_k=top_k,
                sample_rows=len(bucket_values),
                covers_table=covers_table,
            )
            distributions.append(stats.model_copy(update={"bucket": name}))
    pairs = choose_pairs(sketches, co_used, max_pairs=max_pairs)
    by_name = {request.name: request for request in columns}
    associations = tuple(
        measure_association(
            tuple((row.get(x_name), row.get(y_name)) for row in rows),
            source=source,
            table=table,
            column_x=x_name,
            column_y=y_name,
            y_numeric=by_name[y_name].numeric,
            collected_at=collected_at,
            method=method,
        )
        for x_name, y_name in pairs
    )
    return Collected(tuple(quality), tuple(distributions), associations)


def _full_read(relation: str, names: Sequence[str]) -> str:
    """``SELECT`` of validated columns. Names are quoted by sqlglot, not interpolated."""
    selected = [exp.column(require_ident(name)) for name in names]
    return exp.select(*selected).from_(require_ident(relation)).sql(dialect="duckdb")


def collect_duckdb(
    connection: duckdb.DuckDBPyConnection,
    relation: str,
    columns: Sequence[ColumnRequest],
    *,
    source: str,
    table: str,
    collected_at: datetime,
    bucket_column: str | None = None,
    co_used: Sequence[tuple[str, str]] = (),
    sample_rows: int | None = None,
    seed: int = 1,
    covers_table: bool | None = None,
) -> Collected:
    """Read ``relation`` (or a repeatable sample of it) and measure it.

    ``sample_rows`` ``None`` reads the whole relation and records method ``exact``. A positive
    ``sample_rows`` uses DuckDB's reservoir sample and records method ``sample`` and the number
    of rows the sample actually returned. ``covers_table`` defaults to true only for the full read.
    """
    names = [request.name for request in columns]
    if bucket_column is not None and bucket_column not in names:
        names.append(bucket_column)
    if sample_rows is None:
        query = _full_read(relation, names)
        method = CollectionMethod.EXACT
        complete = True if covers_table is None else covers_table
        sampled: int | None = None
    else:
        query = sample_sql(
            table=relation,
            columns=tuple(names),
            dialect="duckdb",
            rows=sample_rows,
            seed=seed,
        )
        method = CollectionMethod.SAMPLE
        complete = False if covers_table is None else covers_table
        sampled = sample_rows
    # Our own generated statement over a table this process already holds: the local-engine
    # chokepoint, not a driver call, so the one place that builds SQL stays the one place.
    cursor = run_local(connection, query)
    description = cursor.description
    headers = [str(item[0]) for item in description] if description else names
    fetched = cursor.fetchall()
    rows = [dict(zip(headers, row, strict=False)) for row in fetched]
    if method is CollectionMethod.SAMPLE:
        sampled = len(rows)
    return collect_rows(
        rows,
        columns,
        source=source,
        table=table,
        collected_at=collected_at,
        method=method,
        bucket_column=bucket_column,
        co_used=co_used,
        covers_table=complete,
        sample_rows=sampled,
    )


def whole_table(stats: Sequence[CategoryColumnStats], column: str) -> CategoryColumnStats | None:
    """The bucket-less distribution of ``column``, if one was collected."""
    for item in stats:
        if item.column == column and item.bucket == "":
            return item
    return None


def observed_pairs(stats: CategoryColumnStats) -> tuple[tuple[str, int], ...]:
    """``(value, count)`` in the stored order. Handy for a suggestion list."""
    return tuple((item.value, item.count) for item in stats.values)
