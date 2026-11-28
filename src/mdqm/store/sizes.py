"""Measured table and partition sizes, in the metadata store.

The planner asks one question -- "how much will this read?" -- and until now the only answer was the
source's own ``reltuples`` folded into the catalog, which is stale (it moves when the source analyzes,
not when rows arrive) and has no shape (it cannot say that one partition holds 20 times the median).

This module keeps what was actually measured, one row per sample per pass, so:

* the newest capture of a table can answer the planner with rows **and bytes**;
* two captures show growth, which is what tells an operator to re-partition;
* :mod:`mdqm.planner.sizes` turns a table's partitions into quantiles (``p50``, ``p99``, spread).

Rows are appended, never updated: a size is a fact about one instant, and overwriting it would throw
away exactly the comparison that makes the number useful.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Final

from mdqm.models import TableSize
from mdqm.planner.sizes import SizeQuantiles, summarise
from mdqm.sql.execution import run_local
from mdqm.store.db import Store, StoreError

__all__ = ["SIZE_TABLE", "SizeStore"]

logger = logging.getLogger(__name__)

#: The table (see ``0009_sizes.sql``). Public because tests and tooling name it.
SIZE_TABLE: Final[str] = "size_sample"

Clock = Callable[[], datetime]

_INSERT: Final = (
    "INSERT INTO size_sample (source, table_name, kind, name, rows, bytes, captured_at)"
    " VALUES (?, ?, ?, ?, ?, ?, ?)"
)
_SELECT_LATEST_CAPTURE: Final = "SELECT captured_at FROM size_sample ORDER BY id DESC LIMIT 1"
_SELECT_TABLE: Final = (
    "SELECT source, table_name, kind, name, rows, bytes, captured_at FROM size_sample"
    " WHERE source = ? AND table_name = ? AND captured_at = ? ORDER BY id"
)
#: The newest capture **per table** (not per pass): each source records under its own timestamp, so
#: "the newest capture" has to mean "the newest rows of this table", or the last source measured
#: hides every other one.
_SELECT_ALL_LATEST: Final = """
WITH newest AS (
    SELECT source, table_name, MAX(captured_at) AS captured_at
      FROM size_sample
     GROUP BY source, table_name
)
SELECT s.source, s.table_name, s.kind, s.name, s.rows, s.bytes, s.captured_at
  FROM size_sample AS s
  JOIN newest AS n
    ON n.source = s.source AND n.table_name = s.table_name AND n.captured_at = s.captured_at
 ORDER BY s.source, s.table_name, s.id
"""
_SELECT_TABLE_LATEST: Final = (
    "SELECT captured_at FROM size_sample WHERE source = ? AND table_name = ?"
    " ORDER BY id DESC LIMIT 1"
)
_SELECT_CAPTURES: Final = (
    "SELECT captured_at, COUNT(*) AS samples FROM size_sample"
    " GROUP BY captured_at ORDER BY captured_at DESC LIMIT ?"
)


class SizeStore:
    """Reads and writes the size samples over one :class:`~mdqm.store.db.Store`.

    Not a singleton: pass the store and, in tests, the clock.
    """

    def __init__(self, store: Store, *, clock: Clock | None = None) -> None:
        self._store = store
        self._clock = clock if clock is not None else (lambda: datetime.now(UTC))

    # --- writing ----------------------------------------------------------------------

    def record(
        self, sizes: Sequence[TableSize], *, captured_at: datetime | None = None
    ) -> datetime:
        """Store one pass of samples under a single ``captured_at`` and return it.

        One timestamp for the whole pass, so "the newest capture" is a set of rows that were measured
        together and a partial failure cannot leave two halves of a pass looking like different
        nights.
        """
        instant = captured_at if captured_at is not None else self._clock()
        rows: list[tuple[object, ...]] = []
        for size in sizes:
            rows.append(
                (
                    size.source,
                    size.table,
                    "table",
                    size.table,
                    size.rows,
                    size.bytes,
                    instant.isoformat(),
                )
            )
            rows.extend(
                (
                    size.source,
                    size.table,
                    "partition",
                    partition.name,
                    partition.rows,
                    partition.bytes,
                    instant.isoformat(),
                )
                for partition in size.partitions
            )
        if not rows:
            return instant
        with self._store.writing() as conn:
            for row in rows:
                run_local(conn, _INSERT, row)
        logger.info(
            "sizes recorded",
            extra={
                "samples": len(rows),
                "tables": len(sizes),
                "captured_at": instant.isoformat(),
            },
        )
        return instant

    # --- reading ----------------------------------------------------------------------

    def latest_capture(self) -> datetime | None:
        """When the newest pass was measured, or ``None`` when nothing was ever measured."""
        row = run_local(self._store.connect(), _SELECT_LATEST_CAPTURE, ()).fetchone()
        return None if row is None else datetime.fromisoformat(str(row["captured_at"]))

    def latest(self, source: str, table: str) -> TableSize | None:
        """The newest measured size of one table, partitions included, or ``None``.

        The newest *capture of that table*, not the global newest pass: a source that failed last
        night still has a usable answer from the night before, and reporting nothing would be worse
        than reporting an older measurement.
        """
        newest = run_local(self._store.connect(), _SELECT_TABLE_LATEST, (source, table)).fetchone()
        if newest is None:
            return None
        captured = str(newest["captured_at"])
        rows = run_local(self._store.connect(), _SELECT_TABLE, (source, table, captured)).fetchall()
        return _table_size(rows)

    def latest_all(self) -> tuple[TableSize, ...]:
        """Every measured table at its own newest capture, sorted; ``()`` when nothing was measured.

        Per table, not per pass: each source records under its own instant, so a global newest
        capture would show only whichever source happened to be measured last.
        """
        rows = run_local(self._store.connect(), _SELECT_ALL_LATEST, ()).fetchall()
        grouped: dict[tuple[str, str], list[Any]] = {}
        for row in rows:
            grouped.setdefault((str(row["source"]), str(row["table_name"])), []).append(row)
        return tuple(_table_size(grouped[key]) for key in sorted(grouped))

    def profile(self, source: str, table: str) -> SizeQuantiles | None:
        """The quantiles of one table: its partitions when partitioned, else the table itself."""
        size = self.latest(source, table)
        if size is None:
            return None
        if size.partitions:
            return summarise(
                {item.name: (item.rows, item.bytes) for item in size.partitions},
                partitions=True,
            )
        return summarise({size.table: (size.rows, size.bytes)})

    def captures(self, limit: int = 10) -> tuple[tuple[datetime, int], ...]:
        """The newest passes and how many samples each held, newest first."""
        rows = run_local(self._store.connect(), _SELECT_CAPTURES, (limit,)).fetchall()
        return tuple(
            (datetime.fromisoformat(str(row["captured_at"])), int(row["samples"])) for row in rows
        )


def _table_size(rows: Sequence[Any]) -> TableSize:
    """Build one :class:`~mdqm.models.TableSize` from the rows of a single capture."""
    source = str(rows[0]["source"])
    table = str(rows[0]["table_name"])
    captured_at = datetime.fromisoformat(str(rows[0]["captured_at"]))
    own = next((row for row in rows if str(row["kind"]) == "table"), None)
    partitions = [
        {
            "name": str(row["name"]),
            "rows": None if row["rows"] is None else int(row["rows"]),
            "bytes": None if row["bytes"] is None else int(row["bytes"]),
        }
        for row in rows
        if str(row["kind"]) == "partition"
    ]
    return TableSize.model_validate(
        {
            "source": source,
            "table": table,
            "captured_at": captured_at,
            "rows": None if own is None or own["rows"] is None else int(own["rows"]),
            "bytes": None if own is None or own["bytes"] is None else int(own["bytes"]),
            "partitions": partitions,
        }
    )


# --- helpers used by the tooling ---------------------------------------------------------------


def describe(sizes: Sequence[TableSize]) -> str:
    """A one-line-per-table report, for the CLI and for a log line. Never a result row."""
    if not sizes:
        return "no sizes have been measured yet"
    lines: list[str] = []
    for size in sizes:
        quantiles = summarise(
            {item.name: (item.rows, item.bytes) for item in size.partitions}
            if size.partitions
            else {size.table: (size.rows, size.bytes)},
            partitions=bool(size.partitions),
        )
        rows = quantiles.rows
        lines.append(
            f"{size.source}.{size.table}  "
            f"partitions={quantiles.partitions}  "
            f"rows min={rows.minimum} p25={rows.p25} p50={rows.p50} p99={rows.p99} "
            f"max={rows.maximum} total={rows.total}  bytes total={quantiles.bytes.total}"
            + ("  UNBALANCED" if quantiles.unbalanced else "")
        )
    return "\n".join(lines)


def as_json(sizes: Sequence[TableSize]) -> str:
    """The same report as JSON, for a script that wants the numbers."""
    return json.dumps([size.model_dump(mode="json") for size in sizes], indent=2, sort_keys=True)


class SizeError(StoreError, ValueError):
    """A size could not be read or written with the arguments given."""
