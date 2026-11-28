"""Reading a source's own size report: the row shapes each engine hands back, parsed once.

Every adapter's ``sizes()`` answers the same question -- *how many rows and how many bytes does this
relation (or this partition) hold?* -- and the answers differ only in the SQL that produced them.
Keeping the parse here, pure and apart from any driver, is what makes it testable without a server:
a fixture of captured rows proves the arithmetic, the "not reported" cases and the grouping, and the
adapter that runs the query has nothing left to decide.

Three shapes, one model:

* :func:`parse_sizes` -- a **seven-column row per relation**, ``(schema, table, relation,
  is_partition, is_partitioned_parent, rows, bytes)``. PostgreSQL's ``pg_class`` and Oracle's
  ``ALL_SEGMENTS`` both fit it, and a driver that returns booleans, ``'t'``, or ``1`` is normalised
  here rather than in each adapter.
* :func:`mdqm.adapters.sqlserver.parse_size_rows` -- SQL Server names a partition ``scheme.number``
  and has no relation of its own to read, so its row carries the pieces of that name instead.
* :func:`mdqm.adapters.elastic.parse_index_sizes` -- Elasticsearch answers with ``_cat`` JSON.

Two rules hold in all three, because they are what makes a size trustworthy:

* ``None`` means **not reported** (the source has no number), never zero. A source reporting rows but
  not bytes must not drag a byte median to zero.
* A negative number is unknown, not a count, and a boolean is not a size: both are dropped to
  ``None`` rather than read as ``0`` or ``1``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any, Final

from mdqm.models import TableSize

__all__ = ["SIZE_COLUMNS", "parse_sizes", "sql_flag", "sql_size"]

#: The seven columns :func:`parse_sizes` reads, in order. Named so an adapter's query can be checked
#: against this list instead of against a parser that silently indexed the wrong position.
SIZE_COLUMNS: Final[tuple[str, ...]] = (
    "schema_name",
    "table_name",
    "relation_name",
    "is_partition",
    "is_partitioned_parent",
    "rows",
    "bytes",
)


def parse_sizes(
    rows: Iterable[Sequence[Any]], *, source: str, captured_at: datetime
) -> tuple[TableSize, ...]:
    """One :class:`~mdqm.models.TableSize` per table, from :data:`SIZE_COLUMNS` rows.

    A row is either a table or one of its partitions (``is_partition``). Rows are grouped by
    ``(schema, table)``: the same bare name in two schemas is two tables, and the schema stays in the
    key so they cannot be merged by accident.

    A partitioned parent has no storage of its own -- ``pg_total_relation_size`` is 0 and Oracle
    keeps global statistics -- so its ``rows``/``bytes`` are reported as ``None`` and the partitions
    carry the numbers. That is the same reading PostgreSQL gets, and it is the one a planner needs:
    the sum of the partitions is the table, and a parent's own 0 would read as "free to scan".
    """
    grouped: dict[tuple[str, str], list[Sequence[Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row[0]), str(row[1])), []).append(row)

    sizes: list[TableSize] = []
    for (_schema, table), group in sorted(grouped.items()):
        own = next((row for row in group if not sql_flag(row[3])), None)
        partitions = [
            {
                "name": str(row[2]),
                "rows": sql_size(row[5]),
                "bytes": sql_size(row[6]),
            }
            for row in group
            if sql_flag(row[3])
        ]
        parent = own is not None and sql_flag(own[4])
        sizes.append(
            TableSize.model_validate(
                {
                    "source": source,
                    "table": table,
                    "rows": None if own is None or parent else sql_size(own[5]),
                    "bytes": None if own is None or parent else sql_size(own[6]),
                    "captured_at": captured_at,
                    "partitions": partitions,
                }
            )
        )
    return tuple(sizes)


def sql_flag(value: object) -> bool:
    """A SQL boolean from any driver: ``True``/``'t'``/``1`` all mean true, everything else false."""
    return value is True or value == "t" or value == 1


def sql_size(value: object) -> int | None:
    """A non-negative size, or ``None`` when the server did not report one.

    ``reltuples`` is a float (``-1`` means "never analyzed"), ``NUM_ROWS`` arrives as a ``Decimal``
    and a size can arrive as text from a driver; all four are normalised here so a caller never has
    to guard. A negative number reads as unknown, because it is not a count. A boolean is not a size,
    and reading it as 1 would turn a driver's ``True`` into "one byte".
    """
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str, Decimal)):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None
