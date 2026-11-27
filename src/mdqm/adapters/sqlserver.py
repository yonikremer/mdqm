"""The SQL Server source adapter (task A08): catalog, partition functions, streaming, cancel.

SQL Server is the third SQL engine and it splits partitioning in two objects, which is why this
module cannot reuse the PostgreSQL shape: a **partition function** owns the boundary values and a
**partition scheme** maps the resulting partitions onto filegroups, and the table merely says
``ON ps_name (column)``. So :meth:`SqlServerAdapter.extract_partitions` reads
``sys.partition_functions`` + ``sys.partition_range_values`` + ``sys.partition_schemes`` +
``sys.destination_data_spaces`` and reconstructs the half-open ranges the planner prunes with.

What A08 verified on SQL Server 2022 (16.0.4135.4) and what that decides here:

* **What an estimated plan says about partitions, and what it does not.** ``SET SHOWPLAN_XML ON``
  puts the pruned range in a **partition-id seek predicate** on the computed column ``PtnId1000``:
  ``<StartRange ScanType="GE">...<ScalarOperator ScalarString="(9)">`` and
  ``<EndRange ScanType="LE">..."(11)-(1)"`` -- partitions 9 through 10. So ``explain`` **can**
  prove pruning and count the partitions, as long as the key filter is a compile-time literal
  (which is what our compiler emits). What the XML never carries is
  ``RunTimePartitionSummary/PartitionsAccessed``: that element appears only in an **actual** plan
  (``SET STATISTICS XML ON``), i.e. only by executing the statement, and it reported
  ``PartitionCount="1"`` for the literal case and ``PartitionCount="18"`` for the ``varchar``
  one. A parameterised value shows ``RangePartitionNew(...)`` in the same place instead of a
  number, so the range is in the plan but its ends are unknown before execution: ``UNPROVEN``.
  ``docs/mssql-driver.md`` quotes all three shapes.
* **A ``varchar`` literal on a ``datetime2`` partition key eliminates nothing.** With
  ``start_time >= '2024-01-01'`` the actual plan accessed all 18 partitions and the estimated plan
  had **no** ``PtnId1000`` predicate (an implicit ``CONVERT`` on the key); with
  ``CONVERT(datetime2(6), '2024-01-01', 126)`` it accessed 1 of 18. That is a compiler/fragment
  concern, recorded here because it is the difference between a pruned scan and a full one, and it
  is what ``explain`` reports as ``UNPROVEN`` (a scan with no partition predicate).
* **Cancellation works through the driver.** ``cursor.cancel()`` interrupts a running statement
  (``Operation canceled``, SQLSTATE ``HYT00``-free) and the session stays usable; so does
  ``Connection.timeout`` (``Query timeout expired``). The deadline is enforced by our own
  watchdog calling ``cursor.cancel()``, with ``Connection.timeout`` as a backstop a few seconds
  beyond it -- the same shape as the Oracle adapter, and for the same reason: the deadline must
  be *ours*, not a driver knob that fires on its own schedule.
* **Read-only is the account, and the server enforces it.** ``mdqm_ro`` holds ``db_datareader``,
  so an ``INSERT``/``DELETE`` that reaches :meth:`execute` is refused by SQL Server
  (``The INSERT permission was denied on the object 'customers'``), audited by the integration
  test. ``explain`` needs ``GRANT SHOWPLAN`` for the login; the probe reports ``explain_plan`` as
  unsupported with that exact fix when the grant is missing.
* **Time is naive UTC.** SQL Server has no timestamp-with-time-zone type, so ``DATETIME2`` values
  are stored and read as naive UTC (same choice as D05) and are re-typed as UTC in Arrow.

Configuration lives in :class:`~mdqm.config.settings.SourceConfig`; the password is a *reference*
resolved through a :class:`~mdqm.config.secrets.SecretProvider` at connect time::

    sources:
      erp_mssql:
        adapter: sqlserver
        secret: erp_ro
        options:
          host: mssql.internal
          port: 1433
          database: erp
          user: mdqm_ro          # db_datareader (+ SHOWPLAN for explain)

Usage::

    adapter = SqlServerAdapter(source_config, env_secret_provider)
    stream = adapter.execute(fragment, Limits(timeout_s=5, max_rows=1000,
                                              max_bytes=4_000_000, max_partitions=8))
    for batch in stream:                    # pyarrow.RecordBatch, lazily
        ...
    stream.close()
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, date, datetime
from datetime import time as dt_time
from decimal import Decimal
from typing import Any, ClassVar, Final
from xml.etree import ElementTree

import pyarrow as pa

from mdqm.adapters.base import OPTIONAL_METHODS, ArrowStream, NotSupported
from mdqm.config.secrets import SecretProvider
from mdqm.config.settings import SourceConfig
from mdqm.models import (
    Capabilities,
    Capability,
    CatalogSnapshot,
    Column,
    ForeignKey,
    Fragment,
    Index,
    Limits,
    NormalizedType,
    PartitionBound,
    PartitionGranularity,
    Partitioning,
    PartitionStrategy,
    PlanSummary,
    PruningProof,
    RawQuery,
    Table,
    TableKind,
    TableStats,
)
from mdqm.sql.execution import prepare, run

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_POOL_SIZE",
    "DEFAULT_PORT",
    "SqlServerAdapter",
    "SqlServerArrowStream",
    "SqlServerCancelled",
    "SqlServerConfigError",
    "SqlServerTimeout",
    "arrow_type",
    "normalized_type",
    "parse_partition_bounds",
    "plan_summary",
]

_log = logging.getLogger(__name__)

#: Rows fetched per round trip. Small enough that ``max_rows``/``max_bytes`` can stop the scan
#: mid-flight instead of after one huge batch already crossed the wire.
DEFAULT_BATCH_SIZE: Final = 1024

#: Concurrent connections per source: the governor already limits concurrency, this is the hard cap.
DEFAULT_POOL_SIZE: Final = 4

#: SQL Server's well-known port. Host, database and user are the organisation's facts.
DEFAULT_PORT: Final = 1433

#: Seconds added to ``Limits.timeout_s`` for the driver's own ``Connection.timeout``. The deadline
#: is the watchdog's job; this only stops a call that never returns from pinning a pooled session.
CALL_TIMEOUT_BACKSTOP_S: Final = 5

#: Driver message fragments. mssql-python 1.19.0 reports both as ``OperationalError`` with the
#: useful text in ``DDBC Error``; the SQLSTATEs (``HYT00``) do not separate them.
_CANCELLED_MARKER: Final = "Operation canceled"
_TIMEOUT_MARKER: Final = "Query timeout expired"

#: The exact grant ``explain`` needs, quoted back to the operator when it is missing.
_SHOWPLAN_FIX: Final = (
    "GRANT SHOWPLAN TO [<read-only login>]: SET SHOWPLAN_XML ON itself is allowed, but compiling a "
    "statement into XML needs the SHOWPLAN permission"
)

_KNOWN_OPTIONS: Final = frozenset(
    {
        "host",
        "port",
        "database",
        "user",
        "dsn",
        "encrypt",
        "trust_server_certificate",
        "connect_timeout",
        "pool_size",
        "batch_size",
        "application_intent",
    }
)

#: ``sys.types.name`` -> the project's portable vocabulary (all of SQL Server's aliases included).
_NORMALIZED_BY_TYPE: Final[dict[str, NormalizedType]] = {
    "char": NormalizedType.STRING,
    "nchar": NormalizedType.STRING,
    "varchar": NormalizedType.STRING,
    "nvarchar": NormalizedType.STRING,
    "text": NormalizedType.STRING,
    "ntext": NormalizedType.STRING,
    "xml": NormalizedType.STRING,
    "sysname": NormalizedType.STRING,
    "uniqueidentifier": NormalizedType.UUID,
    "tinyint": NormalizedType.INTEGER,
    "smallint": NormalizedType.INTEGER,
    "int": NormalizedType.INTEGER,
    "bigint": NormalizedType.INTEGER,
    "bit": NormalizedType.BOOLEAN,
    "decimal": NormalizedType.DECIMAL,
    "numeric": NormalizedType.DECIMAL,
    "money": NormalizedType.DECIMAL,
    "smallmoney": NormalizedType.DECIMAL,
    "float": NormalizedType.FLOAT,
    "real": NormalizedType.FLOAT,
    "date": NormalizedType.DATE,
    "time": NormalizedType.TIME,
    "datetime": NormalizedType.TIMESTAMP,
    "datetime2": NormalizedType.TIMESTAMP,
    "smalldatetime": NormalizedType.TIMESTAMP,
    "datetimeoffset": NormalizedType.TIMESTAMP,
    "binary": NormalizedType.BINARY,
    "varbinary": NormalizedType.BINARY,
    "image": NormalizedType.BINARY,
    "rowversion": NormalizedType.BINARY,
    "timestamp": NormalizedType.BINARY,
    "sql_variant": NormalizedType.OTHER,
    "hierarchyid": NormalizedType.OTHER,
    "geometry": NormalizedType.OTHER,
    "geography": NormalizedType.OTHER,
}

#: Partition-function parameter types whose boundary values a half-open datetime bound can carry.
_TEMPORAL_TYPES: Final = frozenset(
    {"date", "datetime", "datetime2", "smalldatetime", "datetimeoffset"}
)

#: ``sys.types.name`` -> Arrow type, used when ``cursor.description`` only names a Python class.
_ARROW_BY_PY_TYPE: Final[dict[type, pa.DataType]] = {
    bool: pa.bool_(),
    int: pa.int64(),
    float: pa.float64(),
    str: pa.string(),
    bytes: pa.binary(),
    date: pa.date32(),
    dt_time: pa.time64("us"),
    # Naive datetime stays a naive UTC wall time, which is what DATETIME2 stores (DESIGN 5).
    datetime: pa.timestamp("us", tz="UTC"),
}


class SqlServerConfigError(ValueError):
    """The source cannot be connected with this configuration.

    The message names the source and the offending ``options`` key, never a credential, so it is
    safe to log and to show to an operator.
    """


class SqlServerTimeout(TimeoutError):
    """The statement exceeded ``Limits.timeout_s`` and the watchdog cancelled it."""


class SqlServerCancelled(RuntimeError):
    """The statement was aborted by :meth:`SqlServerAdapter.cancel` (``Operation canceled``)."""


# --- type mapping ---------------------------------------------------------------------------


def normalized_type(type_name: str) -> NormalizedType:
    """SQL Server's own type name -> the project's portable vocabulary."""
    return _NORMALIZED_BY_TYPE.get(type_name.lower(), NormalizedType.OTHER)


def arrow_type(py_type: object, precision: int | None, scale: int | None) -> pa.DataType:
    """Arrow type for one ``cursor.description`` entry.

    ``mssql-python`` describes a column with the *Python* class it will produce (``int``, ``str``,
    ``Decimal``, ``datetime``...), so the mapping is off that class; a ``Decimal`` keeps the
    precision and scale the description reports, and anything unrecognised travels as text.
    """
    if py_type is Decimal:
        if precision is None or scale is None:
            return pa.string()
        return pa.decimal128(min(max(precision, 1), 38), scale)
    if isinstance(py_type, type):
        arrow = _ARROW_BY_PY_TYPE.get(py_type)
        if arrow is not None:
            return arrow
    return pa.string()


# --- SQL ------------------------------------------------------------------------------------

#: Partitioned tables: the scheme is on the index's data space, and a scheme is what owns it.
_PARTITIONED_TABLES_SQL: Final = """
SELECT s.name AS schema_name,
       t.name AS table_name,
       ps.name AS scheme_name,
       pf.name AS function_name,
       pf.type_desc AS function_type,
       pf.boundary_value_on_right,
       pf.fanout,
       ty.name AS parameter_type
FROM sys.tables AS t
JOIN sys.schemas AS s ON s.schema_id = t.schema_id
JOIN sys.indexes AS i ON i.object_id = t.object_id AND i.index_id IN (0, 1)
JOIN sys.partition_schemes AS ps ON ps.data_space_id = i.data_space_id
JOIN sys.partition_functions AS pf ON pf.function_id = ps.function_id
JOIN sys.partition_parameters AS pp
  ON pp.function_id = pf.function_id AND pp.parameter_id = 1
JOIN sys.types AS ty ON ty.user_type_id = pp.user_type_id
ORDER BY s.name, t.name
"""

#: The partitioning column: ``partition_ordinal`` marks it in the base index's key.
_PARTITION_KEY_SQL: Final = """
SELECT s.name AS schema_name, t.name AS table_name, c.name AS column_name
FROM sys.tables AS t
JOIN sys.schemas AS s ON s.schema_id = t.schema_id
JOIN sys.indexes AS i ON i.object_id = t.object_id AND i.index_id IN (0, 1)
JOIN sys.index_columns AS ic
  ON ic.object_id = i.object_id AND ic.index_id = i.index_id AND ic.partition_ordinal > 0
JOIN sys.columns AS c ON c.object_id = ic.object_id AND c.column_id = ic.column_id
ORDER BY s.name, t.name, ic.partition_ordinal
"""

#: ``RANGE RIGHT`` tuples, as ISO text so the parse never depends on a server language setting.
_RANGE_VALUES_SQL: Final = """
SELECT pf.name AS function_name,
       prv.boundary_id,
       CONVERT(varchar(64), prv.value, 126) AS value_text
FROM sys.partition_range_values AS prv
JOIN sys.partition_functions AS pf ON pf.function_id = prv.function_id
ORDER BY prv.function_id, prv.boundary_id
"""

#: Destination filegroup per partition. ``ALL TO ([PRIMARY])`` reserves one *extra* destination for
#: the next split, so this list is one longer than ``fanout`` and only ``destination_id <= fanout``
#: names a real partition.
_DESTINATIONS_SQL: Final = """
SELECT ps.name AS scheme_name, dds.destination_id, ds.name AS filegroup_name
FROM sys.destination_data_spaces AS dds
JOIN sys.partition_schemes AS ps ON ps.data_space_id = dds.partition_scheme_id
JOIN sys.data_spaces AS ds ON ds.data_space_id = dds.data_space_id
ORDER BY dds.partition_scheme_id, dds.destination_id
"""

#: Rows per partition from the base index (heap ``0`` or clustered ``1``), never a second count.
_PARTITION_ROWS_SQL: Final = """
SELECT OBJECT_SCHEMA_NAME(p.object_id) AS schema_name,
       OBJECT_NAME(p.object_id) AS table_name,
       p.partition_number,
       SUM(p.rows) AS row_count
FROM sys.partitions AS p
JOIN sys.indexes AS i ON i.object_id = p.object_id AND i.index_id = p.index_id
WHERE i.index_id IN (0, 1)
GROUP BY p.object_id, p.partition_number
ORDER BY p.object_id, p.partition_number
"""

_CATALOG_TABLES_SQL: Final = """
SELECT s.name AS schema_name,
       t.name AS table_name,
       ps.name AS scheme_name
FROM sys.tables AS t
JOIN sys.schemas AS s ON s.schema_id = t.schema_id
OUTER APPLY (
    SELECT TOP 1 ps.name AS name
    FROM sys.indexes AS i
    JOIN sys.partition_schemes AS ps ON ps.data_space_id = i.data_space_id
    WHERE i.object_id = t.object_id AND i.index_id IN (0, 1)
) AS ps
ORDER BY s.name, t.name
"""

_CATALOG_VIEWS_SQL: Final = """
SELECT s.name AS schema_name, v.name AS view_name
FROM sys.views AS v
JOIN sys.schemas AS s ON s.schema_id = v.schema_id
ORDER BY s.name, v.name
"""

_CATALOG_COLUMNS_SQL: Final = """
SELECT s.name AS schema_name,
       o.name AS object_name,
       c.name AS column_name,
       ty.name AS type_name,
       c.max_length,
       c.precision,
       c.scale,
       c.is_nullable,
       c.column_id
FROM sys.columns AS c
JOIN sys.objects AS o ON o.object_id = c.object_id AND o.type IN ('U', 'V')
JOIN sys.schemas AS s ON s.schema_id = o.schema_id
JOIN sys.types AS ty ON ty.user_type_id = c.user_type_id
ORDER BY s.name, o.name, c.column_id
"""

_CATALOG_INDEXES_SQL: Final = """
SELECT s.name AS schema_name,
       t.name AS table_name,
       i.name AS index_name,
       i.is_primary_key,
       i.is_unique,
       i.type_desc,
       i.is_disabled,
       (SELECT STRING_AGG(c.name, ',') WITHIN GROUP (ORDER BY ic.key_ordinal)
        FROM sys.index_columns AS ic
        JOIN sys.columns AS c
          ON c.object_id = ic.object_id AND c.column_id = ic.column_id
        WHERE ic.object_id = i.object_id AND ic.index_id = i.index_id
          AND ic.is_included_column = 0) AS key_columns
FROM sys.indexes AS i
JOIN sys.tables AS t ON t.object_id = i.object_id
JOIN sys.schemas AS s ON s.schema_id = t.schema_id
WHERE i.type IN (0, 1, 2) AND i.name IS NOT NULL
ORDER BY s.name, t.name, i.index_id
"""

_CATALOG_FOREIGN_KEYS_SQL: Final = """
SELECT fk.name AS constraint_name,
       OBJECT_SCHEMA_NAME(fk.parent_object_id) AS schema_name,
       OBJECT_NAME(fk.parent_object_id) AS table_name,
       OBJECT_SCHEMA_NAME(fk.referenced_object_id) AS ref_schema,
       OBJECT_NAME(fk.referenced_object_id) AS ref_table,
       (SELECT STRING_AGG(c.name, ',') WITHIN GROUP (ORDER BY fkc.constraint_column_id)
        FROM sys.foreign_key_columns AS fkc
        JOIN sys.columns AS c
          ON c.object_id = fkc.parent_object_id AND c.column_id = fkc.parent_column_id
        WHERE fkc.constraint_object_id = fk.object_id) AS local_columns,
       (SELECT STRING_AGG(c.name, ',') WITHIN GROUP (ORDER BY fkc.constraint_column_id)
        FROM sys.foreign_key_columns AS fkc
        JOIN sys.columns AS c
          ON c.object_id = fkc.referenced_object_id
         AND c.column_id = fkc.referenced_column_id
        WHERE fkc.constraint_object_id = fk.object_id) AS ref_columns
FROM sys.foreign_keys AS fk
ORDER BY fk.name
"""

_CATALOG_ROWS_SQL: Final = """
SELECT OBJECT_SCHEMA_NAME(p.object_id) AS schema_name,
       OBJECT_NAME(p.object_id) AS table_name,
       SUM(p.rows) AS row_count
FROM sys.partitions AS p
JOIN sys.indexes AS i ON i.object_id = p.object_id AND i.index_id = p.index_id
WHERE i.index_id IN (0, 1)
GROUP BY p.object_id
"""

_VERSION_SQL: Final = (
    "SELECT CAST(SERVERPROPERTY('ProductVersion') AS nvarchar(64)) + ' ' + "
    "CAST(SERVERPROPERTY('Edition') AS nvarchar(128)) AS version"
)

_QUERY_STORE_STATE_SQL: Final = """
SELECT actual_state_desc, readonly_reason
FROM sys.database_query_store_options
"""

#: Query Store rows: cumulative counters per query (``AVG(avg_duration)`` is microseconds per
#: execution, so it multiplies back out to a total before being reported in milliseconds).
_QUERY_STORE_SQL: Final = """
SELECT qt.query_sql_text,
       SUM(rs.count_executions) AS executions,
       SUM(rs.avg_duration * rs.count_executions) AS total_duration_us,
       SUM(rs.avg_rowcount * rs.count_executions) AS total_rows,
       MAX(rs.last_execution_time) AS last_execution
FROM sys.query_store_query_text AS qt
JOIN sys.query_store_query AS q ON q.query_text_id = qt.query_text_id
JOIN sys.query_store_plan AS p ON p.query_id = q.query_id
JOIN sys.query_store_runtime_stats AS rs ON rs.plan_id = p.plan_id
WHERE q.is_internal_query = 0 AND rs.last_execution_time >= ?
GROUP BY qt.query_sql_text
ORDER BY MAX(rs.last_execution_time) DESC
"""

_MUTABILITY_ROWS_SQL: Final = """
SELECT OBJECT_SCHEMA_NAME(p.object_id) AS schema_name,
       OBJECT_NAME(p.object_id) AS table_name,
       SUM(p.row_count) AS row_count
FROM sys.dm_db_partition_stats AS p
JOIN sys.tables AS t ON t.object_id = p.object_id
WHERE p.index_id IN (0, 1)
GROUP BY p.object_id
"""

#: ``modification_counter`` is "rows changed since these statistics were last updated", which is
#: exactly the ``rows_modified_since_analyze`` the cache model wants.
_MUTABILITY_STATS_SQL: Final = """
SELECT s.name AS schema_name,
       t.name AS table_name,
       SUM(CONVERT(bigint, sp.modification_counter)) AS modifications,
       MAX(sp.last_updated) AS last_updated
FROM sys.tables AS t
JOIN sys.schemas AS s ON s.schema_id = t.schema_id
JOIN sys.stats AS st ON st.object_id = t.object_id
CROSS APPLY sys.dm_db_stats_properties(st.object_id, st.stats_id) AS sp
GROUP BY t.object_id, s.name, t.name
"""

_MUTABILITY_UNSUPPORTED: Final = (
    "SQL Server keeps no per-table row counters: sys.dm_db_index_usage_stats.user_updates counts "
    "index maintenance (once per index, not per row) and resets on restart, so reporting it as "
    "rows_updated would make every table look busy after a restart"
)


def _capability(verdict: tuple[bool, str]) -> Capability:
    """A probe verdict as a :class:`~mdqm.models.Capability`."""
    supported, detail = verdict
    return Capability(supported=supported, detail=detail)


def _is_no_permission(exc: BaseException) -> bool:
    """True when the server refused on privilege grounds (permission/denied/not have permission)."""
    text = str(exc).lower()
    return "permission" in text or "denied" in text or "not have permission" in text


# --- the deadline ---------------------------------------------------------------------------


class _Deadline:
    """One execution's deadline: a watchdog that calls the driver's ``cursor.cancel()``.

    ``cursor.cancel()`` is the interrupt ``mssql-python`` implements cleanly (the statement raises
    ``Operation canceled`` and the session survives, both measured on 1.19.0);
    ``Connection.timeout`` is a backstop several seconds further out, so it never decides a normal
    deadline. ``fired`` is what tells the error translation that the cancellation was the
    deadline and not a caller who asked for it.
    """

    def __init__(self, cursor: Any, timeout_s: float, interrupt: Callable[[], None]) -> None:
        self.fired = False
        self._interrupt = interrupt
        self._timer = threading.Timer(timeout_s, self._fire)
        self._timer.daemon = True

    def arm(self) -> None:
        """Start the watchdog; it fires once, ``timeout_s`` after this call."""
        self._timer.start()

    def _fire(self) -> None:
        self.fired = True
        with contextlib.suppress(Exception):
            self._interrupt()

    def cancel(self) -> None:
        """Stop the watchdog and break the statement if it is still running."""
        self._timer.cancel()
        with contextlib.suppress(Exception):
            self._interrupt()

    def disarm(self) -> None:
        """Stop the watchdog only: the statement is finished, so there is nothing to break."""
        self._timer.cancel()


# --- the stream -----------------------------------------------------------------------------


class SqlServerArrowStream:
    """The :class:`~mdqm.adapters.base.ArrowStream` a SQL Server ``execute`` returns.

    It owns one pooled connection until :meth:`close`. The statement is already running and the
    cursor already describes its columns, so batches are pulled with ``fetchmany`` and the caps
    stop the scan instead of trimming a buffer that already crossed the wire.
    """

    def __init__(
        self,
        *,
        cursor: Any,
        schema: pa.Schema,
        limits: Limits,
        deadline: float,
        clock: Callable[[], float],
        release: Callable[[], None],
        watchdog: _Deadline,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.schema: pa.Schema = schema
        self._cursor = cursor
        self._max_rows = limits.max_rows
        self._max_bytes = limits.max_bytes
        self._timeout_s = limits.timeout_s
        self._deadline = deadline
        self._clock = clock
        self._release = release
        self._watchdog = watchdog
        self._batch_size = batch_size
        self._closed = False
        self._cancelled = False
        self.rows_returned = 0
        self.bytes_returned = 0

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        """Yield batches until the source, a cap, a cancellation or a deadline stops the scan."""
        if self._closed:
            return
        while True:
            self._check_cancelled()
            self._check_deadline()
            rows = self._fetch()
            if not rows:
                return
            batch = self._trim(
                pa.RecordBatch.from_arrays(_arrays_from_rows(self.schema, rows), schema=self.schema)
            )
            if batch.num_rows == 0:
                return
            self.rows_returned += batch.num_rows
            self.bytes_returned += batch.nbytes
            yield batch
            if self.rows_returned >= self._max_rows or self.bytes_returned >= self._max_bytes:
                return

    def _fetch(self) -> Sequence[Sequence[Any]]:
        """Fetch the next batch, translating a cancellation into this adapter's own error."""
        try:
            rows = self._cursor.fetchmany(self._batch_size)
            return [tuple(row) for row in rows]
        except Exception as exc:
            raise _stream_error(
                exc,
                cancelled=self._cancelled,
                timed_out=self._watchdog.fired,
                timeout_s=self._timeout_s,
            ) from exc

    def _check_cancelled(self) -> None:
        """Stop the stream once :meth:`cancel` was called.

        The driver's ``SQLCancel`` only lands while a fetch is in flight, so a cancellation that
        arrives between two fast batches would otherwise be lost; this flag is what makes it
        authoritative. The server-side break has still been sent, so the abandoned statement stops
        producing rows instead of running to completion for nobody.
        """
        if self._cancelled:
            raise SqlServerCancelled("the SQL Server statement was cancelled by the caller")

    def _check_deadline(self) -> None:
        """Raise once the deadline passed, having stopped the server-side statement too."""
        if self._watchdog.fired or self._clock() > self._deadline:
            self._watchdog.cancel()
            raise SqlServerTimeout(
                f"the SQL Server statement exceeded the {self._timeout_s}s timeout while streaming"
            )

    def _trim(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        """Cut the batch at the row cap and the byte cap, so neither is batch-rounded."""
        remaining_rows = self._max_rows - self.rows_returned
        if batch.num_rows > remaining_rows:
            batch = batch.slice(0, remaining_rows)
        remaining_bytes = self._max_bytes - self.bytes_returned
        if batch.num_rows == 0 or batch.nbytes <= remaining_bytes:
            return batch
        per_row = batch.nbytes / batch.num_rows
        # pi-lens-ignore: unchecked-numeric-parse-python -- int/float division, never text
        keep = int(remaining_bytes / per_row) if per_row > 0 else batch.num_rows
        return batch.slice(0, min(batch.num_rows, max(keep, 0)))

    def cancel(self) -> None:
        """Ask SQL Server to abort the running statement. Idempotent.

        Only the break is sent: the reader (or :meth:`close`) still releases the cursor and the
        connection, so cancelling from another thread cannot close a handle mid-fetch.
        """
        if self._closed or self._cancelled:
            return
        self._cancelled = True
        with contextlib.suppress(Exception):
            self._cursor.cancel()

    def close(self) -> None:
        """Release the cursor and the connection back to the pool. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._watchdog.disarm()
        with contextlib.suppress(Exception):
            self._cursor.close()
        self._release()


# --- the pool -------------------------------------------------------------------------------


class _Pool:
    """A bounded pool of DB-API connections: at most ``size`` at once, idle ones reused.

    A tiny hand-rolled pool on purpose: ``mssql-python`` ships none, and the governor already owns
    concurrency policy, so all this needs to do is cap connections and hand back a warm one.
    """

    def __init__(self, open_connection: Callable[[], Any], size: int) -> None:
        self._open = open_connection
        self._slots = threading.BoundedSemaphore(size)
        self._idle: list[Any] = []
        self._lock = threading.Lock()
        self._closed = False

    def acquire(self) -> Any:
        """Take a connection, opening one when none is idle."""
        self._slots.acquire()
        try:
            with self._lock:
                connection = self._idle.pop() if self._idle else None
            return connection if connection is not None else self._open()
        except BaseException:
            self._slots.release()
            raise

    def release(self, connection: Any) -> None:
        """Return a connection; a closed one is dropped instead of handed out again."""
        with self._lock:
            if self._closed or getattr(connection, "closed", False):
                keep = False
            else:
                self._idle.append(connection)
                keep = True
        if not keep:
            with contextlib.suppress(Exception):
                connection.close()
        self._slots.release()

    def close(self) -> None:
        """Close every idle connection. Idle ones in use are closed by their owner. Idempotent."""
        with self._lock:
            self._closed = True
            idle, self._idle = self._idle, []
        for connection in idle:
            with contextlib.suppress(Exception):
                connection.close()


# --- the adapter ----------------------------------------------------------------------------


class SqlServerAdapter:
    """A pooled, streaming, read-only ``SourceAdapter`` for one SQL Server database.

    The pool is built lazily, so an adapter for an unreachable source can be constructed (and
    ``mdqm probe`` can report the failure) without a connection attempt at import or wiring time.
    """

    kind: ClassVar[str] = "sqlserver"

    def __init__(
        self,
        config: SourceConfig,
        secrets: SecretProvider | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        """Build the adapter from its source config; connect on first use."""
        if config.adapter != self.kind:
            raise SqlServerConfigError(
                f"config names adapter {config.adapter!r} but this is the {self.kind!r} adapter"
            )
        options = dict(config.options or {})
        for unknown in sorted(set(options) - _KNOWN_OPTIONS):
            _log.warning("source %s: ignoring unknown sqlserver option %r", config.name, unknown)
        self._source = config.name
        self._conninfo = _conninfo(config, secrets)
        self._pool_size = _positive_int(options.get("pool_size"), DEFAULT_POOL_SIZE, "pool_size")
        self._batch_size = _positive_int(
            options.get("batch_size"), DEFAULT_BATCH_SIZE, "batch_size"
        )
        self._connect_timeout_s = _positive_int(
            options.get("connect_timeout"), 10, "connect_timeout"
        )
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic
        self._pool: _Pool | None = None

    def __repr__(self) -> str:
        """Name the source and the host/database, never the password."""
        server = self._conninfo.split(";")[0]
        return f"SqlServerAdapter(source={self._source!r}, {server!r})"

    # --- SourceAdapter: identity ---------------------------------------------------------

    def dialect(self) -> str:
        """``tsql``: the sqlglot dialect the compiler emits for this source."""
        return "tsql"

    @property
    def source(self) -> str:
        """The configured source name; the catalog and the probe both report it."""
        return self._source

    # --- lifecycle ------------------------------------------------------------------------

    @property
    def pool(self) -> _Pool:
        """The connection pool, created on first access."""
        if self._pool is None:
            self._pool = _Pool(self._open_connection, self._pool_size)
        return self._pool

    def close(self) -> None:
        """Close every idle connection. Idempotent."""
        if self._pool is not None:
            self._pool.close()

    def _open_connection(self) -> Any:
        """Open one connection with the driver, imported lazily so unit tests need no driver."""
        try:
            import mssql_python
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on the extra
            raise SqlServerConfigError(
                "the SQL Server driver is not installed: `pip install -e .[mssql]` (mssql-python)"
            ) from exc
        return mssql_python.connect(
            self._conninfo, autocommit=True, timeout=self._connect_timeout_s
        )

    @contextlib.contextmanager
    def _connection(self) -> Iterator[Any]:
        """One pooled connection for a probe/catalog read, returned on exit."""
        connection = self.pool.acquire()
        try:
            yield connection
        finally:
            self.pool.release(connection)

    # --- SourceAdapter: probe -------------------------------------------------------------

    def probe(self) -> Capabilities:
        """Detect what this database offers, reporting only what this adapter can serve."""
        version: str | None = None
        with self._connection() as connection:
            cursor = connection.cursor()
            try:
                run(cursor, prepare(_VERSION_SQL, dialect="tsql"))
                row = cursor.fetchone()
                version = None if row is None or row[0] is None else str(row[0])
            finally:
                cursor.close()
        capabilities = {
            name: _capability(verdict)
            for name, verdict in (
                (OPTIONAL_METHODS["extract_partitions"], self._probe_partitions()),
                (OPTIONAL_METHODS["explain"], self._probe_explain()),
                (OPTIONAL_METHODS["mine_queries"], self._probe_query_store()),
                (OPTIONAL_METHODS["mutability_stats"], self._probe_mutability()),
            )
        }
        return Capabilities(
            source=self._source,
            adapter=self.kind,
            version=version,
            retrieved_at=self._clock(),
            capabilities=capabilities,
        )

    def _probe_partitions(self) -> tuple[bool, str]:
        """Claim ``partition_metadata`` only if extraction really returns metadata."""
        try:
            found = self.extract_partitions()
        except NotSupported as exc:
            return False, exc.detail or exc.capability
        except Exception as exc:
            return False, f"the partition catalog could not be read: {exc}"
        return True, (
            "sys.partition_functions + sys.partition_range_values + sys.partition_schemes + "
            f"sys.destination_data_spaces: {len(found)} partitioned table(s) read, RANGE RIGHT "
            "tuples turned into real half-open bounds"
        )

    def _probe_explain(self) -> tuple[bool, str]:
        """Claim ``explain_plan`` only if ``SET SHOWPLAN_XML ON`` really returns a plan."""
        try:
            xml = self._showplan_xml("SELECT 1 AS one")
        except NotSupported as exc:
            return False, exc.detail or exc.capability
        except Exception as exc:
            return False, f"SET SHOWPLAN_XML ON failed: {exc}"
        if "ShowPlanXML" not in xml:
            return False, "SET SHOWPLAN_XML ON returned no ShowPlanXML document"
        return True, (
            "SET SHOWPLAN_XML ON: cost, estimates, indexes and the partition-id seek predicate "
            "(PtnId1000) are in the XML; RunTimePartitionSummary/PartitionsAccessed is not, so a "
            "parameterised range reads as unproven"
        )

    def _probe_query_store(self) -> tuple[bool, str]:
        """Claim ``query_store`` only when the store is on *and* this login may read it."""
        with self._connection() as connection:
            cursor = connection.cursor()
            try:
                run(cursor, prepare(_QUERY_STORE_STATE_SQL, dialect="tsql"))
                row = cursor.fetchone()
            except Exception as exc:
                return False, f"sys.database_query_store_options is unreadable: {exc}"
            finally:
                cursor.close()
        state = "" if row is None else str(row[0])
        if state not in {"READ_WRITE", "READ_ONLY"}:
            return False, (
                f"Query Store is {state or 'absent'} (ALTER DATABASE ... SET QUERY_STORE = ON)"
            )
        try:
            with self._connection() as connection:
                cursor = connection.cursor()
                try:
                    window = prepare(
                        _QUERY_STORE_SQL, dialect="tsql", params=(datetime(1970, 1, 1),)
                    )
                    run(cursor, window)
                    cursor.fetchall()
                finally:
                    cursor.close()
        except Exception as exc:
            detail = f"the sys.query_store_* views are unreadable: {exc}"
            if _is_no_permission(exc):
                detail += " (needs VIEW DATABASE STATE)"
            return False, detail
        return True, (
            f"Query Store {state}: sys.query_store_query_text + _query + _plan + _runtime_stats"
        )

    def _probe_mutability(self) -> tuple[bool, str]:
        """Claim ``mutability_counters`` only if the stats counters are readable."""
        try:
            self._mutability_rows()
        except Exception as exc:
            return False, f"sys.dm_db_stats_properties is unreadable: {exc}"
        return True, (
            "sys.dm_db_stats_properties.modification_counter (rows changed since the statistics "
            "were last updated) plus sys.dm_db_partition_stats row counts; per-table row insert/"
            "update/delete counters do not exist in SQL Server"
        )

    # --- SourceAdapter: catalog -----------------------------------------------------------

    def extract_catalog(self) -> CatalogSnapshot:
        """Tables, views, columns, types, indexes, keys and row counts from the ``sys.*`` views.

        Partition children are not listed as tables: SQL Server exposes a partition as
        ``sys.partitions.partition_number``, never as a name an agent could query, so
        :meth:`extract_partitions` reports them where the planner needs them.
        """
        columns: dict[tuple[str, str], list[Column]] = {}
        indexes: dict[tuple[str, str], list[Index]] = {}
        foreign_keys: dict[tuple[str, str], list[ForeignKey]] = {}
        rows_by_table: dict[tuple[str, str], int] = {}
        with self._connection() as connection:
            cursor = connection.cursor()
            try:
                table_rows = self._fetch_all(cursor, _CATALOG_TABLES_SQL)
                view_rows = self._fetch_all(cursor, _CATALOG_VIEWS_SQL)
                column_rows = self._fetch_all(cursor, _CATALOG_COLUMNS_SQL)
                index_rows = self._fetch_all(cursor, _CATALOG_INDEXES_SQL)
                key_rows = self._fetch_all(cursor, _CATALOG_FOREIGN_KEYS_SQL)
                count_rows = self._fetch_all(cursor, _CATALOG_ROWS_SQL)
            finally:
                cursor.close()

        for row in column_rows:
            key = (str(row[0]), str(row[1]))
            type_name = str(row[3])
            columns.setdefault(key, []).append(
                Column(
                    name=str(row[2]),
                    native_type=_native_type(
                        type_name, _as_int(row[4]), _as_int(row[5]), _as_int(row[6])
                    ),
                    normalized_type=normalized_type(type_name),
                    nullable=bool(row[7]),
                    ordinal=_as_int(row[8]),
                )
            )
        for row in index_rows:
            key_columns = tuple(part for part in str(row[7] or "").split(",") if part)
            if not key_columns:
                continue
            indexes.setdefault((str(row[0]), str(row[1])), []).append(
                Index(
                    name=str(row[2]),
                    columns=key_columns,
                    unique=bool(row[4]),
                    primary=bool(row[3]),
                    index_type=str(row[5]),
                )
            )
        for row in key_rows:
            local = tuple(part for part in str(row[5] or "").split(",") if part)
            referenced = tuple(part for part in str(row[6] or "").split(",") if part)
            if not local or len(local) != len(referenced):
                continue
            foreign_keys.setdefault((str(row[1]), str(row[2])), []).append(
                ForeignKey(
                    columns=local,
                    ref_table=f"{row[3]}.{row[4]}",
                    ref_columns=referenced,
                    name=str(row[0]),
                )
            )
        for row in count_rows:
            row_count = _as_int(row[2])
            if row_count is not None:
                rows_by_table[(str(row[0]), str(row[1]))] = row_count

        tables: list[Table] = []
        for row in table_rows:
            key = (str(row[0]), str(row[1]))
            tables.append(
                _table(
                    str(row[0]),
                    str(row[1]),
                    TableKind.PARTITIONED_TABLE if row[2] is not None else TableKind.TABLE,
                    columns,
                    indexes,
                    foreign_keys,
                    rows_by_table.get(key),
                )
            )
        for row in view_rows:
            tables.append(
                _table(
                    str(row[0]),
                    str(row[1]),
                    TableKind.VIEW,
                    columns,
                    indexes,
                    foreign_keys,
                    None,
                )
            )
        return CatalogSnapshot(source=self._source, captured_at=self._clock(), tables=tuple(tables))

    # --- SourceAdapter: partitions --------------------------------------------------------

    def extract_partitions(self) -> list[Partitioning]:
        """Partition metadata per partitioned table, with the function's real boundary values.

        SQL Server splits the mechanism in two: ``sys.partition_functions`` owns the tuples and
        ``sys.partition_schemes`` + ``sys.destination_data_spaces`` map the resulting partitions
        onto filegroups. With ``RANGE RIGHT`` a tuple is the **lower** bound of the partition to
        its right, so N tuples yield N+1 half-open ranges -- exactly what
        :class:`~mdqm.models.Partitioning` wants, and the reason the count is ``fanout`` and not
        the (one longer) destination list: ``ALL TO ([PRIMARY])`` reserves a destination for the
        next split, which is not a partition anyone can read.

        ``RANGE LEFT`` is refused: a tuple there *closes* the partition on its left, so a
        half-open bound would silently drop the rows exactly on the tuple.
        """
        with self._connection() as connection:
            cursor = connection.cursor()
            try:
                table_rows = self._fetch_all(cursor, _PARTITIONED_TABLES_SQL)
                key_rows = self._fetch_all(cursor, _PARTITION_KEY_SQL)
                value_rows = self._fetch_all(cursor, _RANGE_VALUES_SQL)
                destination_rows = self._fetch_all(cursor, _DESTINATIONS_SQL)
                row_rows = self._fetch_all(cursor, _PARTITION_ROWS_SQL)
            finally:
                cursor.close()

        keys: dict[tuple[str, str], list[str]] = {}
        for row in key_rows:
            keys.setdefault((str(row[0]), str(row[1])), []).append(str(row[2]))
        tuples: dict[str, list[datetime]] = {}
        for row in value_rows:
            parsed = _parse_boundary(str(row[2]))
            if parsed is not None:
                tuples.setdefault(str(row[0]), []).append(parsed)
        filegroups: dict[str, dict[int, str]] = {}
        for row in destination_rows:
            destination_id = _as_int(row[1])
            if destination_id is not None:
                filegroups.setdefault(str(row[0]), {})[destination_id] = str(row[2])
        counts: dict[tuple[str, str, int], int] = {}
        for row in row_rows:
            boundary_id, row_count = _as_int(row[2]), _as_int(row[3])
            if boundary_id is not None and row_count is not None:
                counts[(str(row[0]), str(row[1]), boundary_id)] = row_count

        out: list[Partitioning] = []
        for row in table_rows:
            out.append(
                _partitioning(
                    table=f"{row[0]}.{row[1]}",
                    scheme=str(row[2]),
                    function=str(row[3]),
                    function_type=str(row[4]),
                    boundary_on_right=bool(row[5]),
                    fanout=_as_int(row[6]) or 0,
                    parameter_type=str(row[7]),
                    key=tuple(keys.get((str(row[0]), str(row[1])), ())),
                    boundaries=tuple(tuples.get(str(row[3]), ())),
                    filegroups=filegroups.get(str(row[2]), {}),
                    row_counts=counts,
                )
            )
        return out

    # --- SourceAdapter: plan --------------------------------------------------------------

    def explain(self, fragment: Fragment) -> PlanSummary:
        """The optimizer's own plan for ``fragment``, from ``SET SHOWPLAN_XML ON``.

        Nothing is executed: ``SHOWPLAN`` only asks the optimizer, so asking the cost of a scan
        never causes it. What the XML carries on SQL Server 2022 (measured, see
        ``docs/mssql-driver.md``): the cost, the estimate, the indexes, and -- for a partitioned
        table -- a partition-id seek predicate that names the pruned partition range whenever the
        key filter is a literal (``PROVEN``, with a counted ``partitions_scanned``). A
        parameterised filter shows ``RangePartitionNew(...)`` there instead, and a scan with no
        partition predicate reaches every partition: both are ``UNPROVEN``, because neither proves
        a narrower scan. ``UNKNOWN`` is kept for a plan with no partitioned table at all, where
        there is nothing to prove.
        """
        text = fragment.sql
        if text is None:
            raise NotSupported(OPTIONAL_METHODS["explain"], "this fragment carries no SQL")
        xml = self._showplan_xml(text)
        return plan_summary(xml, partitions_total=self._partitions_total(fragment.tables))

    def _showplan_xml(self, statement: str) -> str:
        """The ``ShowPlanXML`` document for ``statement``, or :class:`NotSupported`."""
        connection = self.pool.acquire()
        try:
            cursor = connection.cursor()
            try:
                self._set_showplan(cursor, True)
                try:
                    # With SHOWPLAN on nothing is executed, so the guard runs on the fragment --
                    # the statement whose plan was asked for -- before it is sent.
                    run(cursor, prepare(statement, dialect="tsql"))
                    row = cursor.fetchone()
                finally:
                    with contextlib.suppress(Exception):
                        self._set_showplan(cursor, False)
            finally:
                cursor.close()
        finally:
            self.pool.release(connection)
        text = "" if row is None or row[0] is None else str(row[0])
        if not text:
            raise NotSupported(OPTIONAL_METHODS["explain"], "SET SHOWPLAN_XML ON returned no plan")
        return text

    def _set_showplan(self, cursor: Any, enabled: bool) -> None:
        """Toggle ``SHOWPLAN_XML``, translating a refusal into the grant an operator must issue."""
        try:
            # `SET SHOWPLAN_XML` is a session command the guard refuses, and it is a fixed literal
            # with no value in it; there is nothing for the guard to inspect in it either.
            if enabled:
                cursor.execute("SET SHOWPLAN_XML ON")
            else:
                cursor.execute("SET SHOWPLAN_XML OFF")
        except Exception as exc:
            raise NotSupported(
                OPTIONAL_METHODS["explain"],
                f"this account may not capture a plan ({_SHOWPLAN_FIX}): {exc}",
            ) from exc

    def _partitions_total(self, tables: Sequence[str]) -> int | None:
        """How many partitions the fragment's (single) table has, straight from the catalog."""
        if len(tables) != 1:
            return None
        bare = tables[0].split(".")[-1]
        with self._connection() as connection:
            cursor = connection.cursor()
            try:
                cursor.execute(
                    """
                    SELECT pf.fanout
                    FROM sys.partition_functions AS pf
                    JOIN sys.partition_schemes AS ps ON ps.function_id = pf.function_id
                    JOIN sys.indexes AS i ON i.data_space_id = ps.data_space_id
                    JOIN sys.tables AS t
                      ON t.object_id = i.object_id AND i.index_id IN (0, 1)
                    WHERE t.name = ?
                    """,
                    (bare,),
                )
                row = cursor.fetchone()
            except Exception:  # a missing grant must not fail an explain
                return None
            finally:
                cursor.close()
        return None if row is None else _as_int(row[0])

    # --- SourceAdapter: execute / cancel -------------------------------------------------

    def execute(self, fragment: Fragment, limits: Limits) -> ArrowStream:
        """Run one compiled fragment and stream it as Arrow batches under ``limits``.

        ``Limits.timeout_s`` becomes the watchdog's deadline (see :class:`_Deadline`) plus a
        driver ``Connection.timeout`` several seconds beyond it. T-SQL placeholders are positional
        (``?``) while a fragment's ``params`` is a JSON object, and the compiler inlines values
        into the SQL it emits, so a fragment that carries params is refused rather than guessed at.
        """
        if fragment.sql is None:
            raise ValueError(
                f"the sqlserver adapter runs SQL; fragment kind {fragment.kind!r} is not supported"
            )
        if fragment.params:
            raise ValueError(
                "the sqlserver adapter cannot bind a fragment's params: T-SQL placeholders are "
                "positional (?), a fragment's params is an unordered object, and the compiler "
                "inlines values into the SQL it emits"
            )
        # Guarded here, immediately before it runs, and before a pooled connection is taken: a
        # refused statement never reaches the server and never occupies the pool.
        prepared = prepare(fragment.sql, dialect="tsql")
        connection = self.pool.acquire()
        try:
            connection.timeout = int(limits.timeout_s) + CALL_TIMEOUT_BACKSTOP_S
            cursor = connection.cursor()
            watchdog = _Deadline(cursor, limits.timeout_s, cursor.cancel)
            watchdog.arm()
            try:
                run(cursor, prepared)
            except Exception as exc:
                watchdog.disarm()
                with contextlib.suppress(Exception):
                    cursor.close()
                raise _stream_error(
                    exc, cancelled=False, timed_out=watchdog.fired, timeout_s=limits.timeout_s
                ) from exc
            return SqlServerArrowStream(
                cursor=cursor,
                schema=_arrow_schema(cursor.description),
                limits=limits,
                deadline=self._monotonic() + limits.timeout_s,
                clock=self._monotonic,
                release=lambda: self._release(cursor, connection),
                watchdog=watchdog,
                batch_size=self._batch_size,
            )
        except BaseException:
            self.pool.release(connection)
            raise

    def cancel(self, handle: ArrowStream) -> None:
        """Abort the statement behind ``handle``. Idempotent; safe to call twice."""
        cancel = getattr(handle, "cancel", None)
        if callable(cancel):
            cancel()
        else:
            handle.close()

    def _release(self, cursor: Any, connection: Any) -> None:
        """Close the cursor and hand the connection back, ignoring one already gone."""
        with contextlib.suppress(Exception):
            cursor.close()
        self.pool.release(connection)

    # --- SourceAdapter: query store -------------------------------------------------------

    def mine_queries(self, since: datetime) -> Iterator[RawQuery]:
        """Queries this database already ran, from Query Store.

        Query Store keeps *cumulative* counters per query, so a statement executed twice appears
        once with ``calls=2``; durations are stored in microseconds and reported in milliseconds.
        The store carries no executor, so ``user`` stays ``None`` rather than naming the login that
        compiled the plan. Raises :class:`NotSupported` when the store is off or unreadable -- never
        an empty list that looks like "this database runs nothing".
        """
        window = since.astimezone(UTC).replace(tzinfo=None)
        with self._connection() as connection:
            cursor = connection.cursor()
            try:
                run(
                    cursor,
                    prepare(_QUERY_STORE_SQL, dialect="tsql", params=(window,)),
                )
                rows = [tuple(row) for row in cursor.fetchall()]
            except Exception as exc:
                if _is_no_permission(exc):
                    raise NotSupported(
                        OPTIONAL_METHODS["mine_queries"],
                        f"this account may not read the Query Store views (needs VIEW DATABASE "
                        f"STATE): {exc}",
                    ) from exc
                raise
            finally:
                cursor.close()
        return iter(_raw_queries(rows))

    # --- SourceAdapter: mutability --------------------------------------------------------

    def mutability_stats(self) -> list[TableStats]:
        """Row counts now and rows changed since each table's statistics were last updated.

        Both numbers are SQL Server's own: ``sys.dm_db_partition_stats`` for the count and
        ``sys.dm_db_stats_properties.modification_counter`` for the changes. ``rows_inserted``,
        ``rows_updated`` and ``rows_deleted`` stay ``None``: SQL Server keeps no per-table row
        counters, and ``sys.dm_db_index_usage_stats.user_updates`` counts index maintenance (once
        per index, reset by a restart), so reporting it as rows would make every table look busy
        after a restart.
        """
        rows = self._mutability_rows()
        changed = self._mutability_counters()
        out: list[TableStats] = []
        for schema, table, row_count in rows:
            name = f"{schema}.{table}"
            modifications, last_updated = changed.get(name, (None, None))
            out.append(
                TableStats(
                    source=self._source,
                    table=name,
                    captured_at=self._clock(),
                    row_estimate=row_count,
                    rows_modified_since_analyze=modifications,
                    last_analyze=last_updated,
                )
            )
        return out

    def _mutability_rows(self) -> list[tuple[str, str, int]]:
        """``schema.table`` row counts from the partition statistics view."""
        with self._connection() as connection:
            cursor = connection.cursor()
            try:
                rows = self._fetch_all(cursor, _MUTABILITY_ROWS_SQL)
            finally:
                cursor.close()
        out: list[tuple[str, str, int]] = []
        for row in rows:
            row_count = _as_int(row[2])
            if row_count is not None:
                out.append((str(row[0]), str(row[1]), row_count))
        return out

    def _mutability_counters(self) -> dict[str, tuple[int | None, datetime | None]]:
        """``schema.table`` -> (rows changed since the statistics were updated, that time)."""
        with self._connection() as connection:
            cursor = connection.cursor()
            try:
                rows = self._fetch_all(cursor, _MUTABILITY_STATS_SQL)
            finally:
                cursor.close()
        out: dict[str, tuple[int | None, datetime | None]] = {}
        for schema, table, modifications, last_updated in rows:
            out[f"{schema}.{table}"] = (
                _as_int(modifications),
                _as_utc(last_updated),
            )
        return out

    # --- internals ------------------------------------------------------------------------

    @staticmethod
    def _fetch_all(cursor: Any, statement: str) -> list[tuple[Any, ...]]:
        """Run a module-constant statement through the chokepoint and return its rows.

        Every caller passes a constant of this file (the catalogs take no parameter), so the guard
        sees the statement immediately before it runs and this is the only door it uses.
        """
        run(cursor, prepare(statement, dialect="tsql"))
        return [tuple(row) for row in cursor.fetchall()]


# --- pure helpers ---------------------------------------------------------------------------


def _table(
    schema: str,
    name: str,
    kind: TableKind,
    columns: Mapping[tuple[str, str], list[Column]],
    indexes: Mapping[tuple[str, str], list[Index]],
    foreign_keys: Mapping[tuple[str, str], list[ForeignKey]],
    row_estimate: int | None,
) -> Table:
    """One :class:`~mdqm.models.Table` from the pieces gathered per ``(schema, name)``."""
    key = (schema, name)
    items = indexes.get(key, [])
    return Table(
        name=name,
        schema_name=schema,
        kind=kind,
        columns=tuple(columns.get(key, ())),
        indexes=tuple(items),
        foreign_keys=tuple(foreign_keys.get(key, ())),
        primary_key=next((item.columns for item in items if item.primary), ()),
        row_estimate=row_estimate,
    )


def _partitioning(
    *,
    table: str,
    scheme: str,
    function: str,
    function_type: str,
    boundary_on_right: bool,
    fanout: int,
    parameter_type: str,
    key: tuple[str, ...],
    boundaries: tuple[datetime, ...],
    filegroups: Mapping[int, str],
    row_counts: Mapping[tuple[str, str, int], int],
) -> Partitioning:
    """One :class:`~mdqm.models.Partitioning` from the ``sys.*`` partition rows."""
    kind = function_type.upper()
    if kind != "RANGE":
        raise NotSupported(
            OPTIONAL_METHODS["extract_partitions"],
            f"{table} uses partition function {function} of type {function_type}, which has no "
            "planner equivalent (SQL Server partitions by RANGE only)",
        )
    if not boundary_on_right:
        raise NotSupported(
            OPTIONAL_METHODS["extract_partitions"],
            f"{table} uses RANGE LEFT function {function}: a tuple closes the partition on its "
            "left, so a half-open bound would drop the rows exactly on the tuple",
        )
    if parameter_type.lower() not in _TEMPORAL_TYPES:
        raise NotSupported(
            OPTIONAL_METHODS["extract_partitions"],
            f"{table} is partitioned on a {parameter_type} column, and a partition bound in this "
            "system is a UTC datetime; the planner cannot align a non-temporal range",
        )
    return Partitioning(
        strategy=PartitionStrategy.RANGE,
        key=key,
        granularity=granularity(boundaries),
        bounds=parse_partition_bounds(
            scheme=scheme,
            fanout=fanout,
            boundaries=boundaries,
            filegroups=filegroups,
            table=table,
            row_counts=row_counts,
        ),
    )


def parse_partition_bounds(
    *,
    scheme: str,
    fanout: int,
    boundaries: Sequence[datetime],
    filegroups: Mapping[int, str],
    table: str = "",
    row_counts: Mapping[tuple[str, str, int], int] | None = None,
) -> tuple[PartitionBound, ...]:
    """``RANGE RIGHT`` tuples -> the half-open partitions they define.

    N tuples define N+1 partitions: ``(-inf, t1)``, ``[t1, t2)`` ... ``[tN, +inf)``. The ``fanout``
    SQL Server reports is the authority on how many there are, and a mismatch with the tuple count
    is refused instead of patched: a planner pruning on a wrong count is worse than no metadata.
    """
    if len(boundaries) + 1 != fanout:
        raise NotSupported(
            OPTIONAL_METHODS["extract_partitions"],
            f"{table or scheme}: partition function reports fanout {fanout} but "
            f"{len(boundaries)} boundary value(s) were read; {len(boundaries) + 1} partitions "
            "were expected, so the bound list is incomplete",
        )
    counts = row_counts or {}
    out: list[PartitionBound] = []
    previous: datetime | None = None
    for number in range(1, fanout + 1):
        upper = boundaries[number - 1] if number <= len(boundaries) else None
        name = filegroups.get(number)
        out.append(
            PartitionBound(
                name=f"{scheme}.{number}" if name is None else f"{scheme}.{number} ({name})",
                lower=previous,
                upper=upper,
                row_estimate=counts.get((table.split(".")[0], table.split(".")[-1], number))
                if table
                else None,
            )
        )
        previous = upper if upper is not None else previous
    return tuple(out)


def granularity(boundaries: Sequence[datetime]) -> PartitionGranularity | None:
    """The width of the time ranges when every step is the same, else ``None``.

    A D06-style monthly function steps by one calendar month, which is 28..31 days, so the month
    step is checked on the month component and the day step on days.
    """
    if len(boundaries) < 2:
        return None
    steps = list(itertools.pairwise(boundaries))
    months = {(b.year - a.year) * 12 + (b.month - a.month) for a, b in steps}
    if months == {1} and all(a.day == 1 and b.day == 1 for a, b in steps):
        return PartitionGranularity.MONTH
    days = {(b - a).days for a, b in steps}
    if len(days) != 1:
        return None
    step = days.pop()
    if step >= 365 and all(a.month == b.month and b.year - a.year == step // 365 for a, b in steps):
        return PartitionGranularity.YEAR
    return {
        1: PartitionGranularity.DAY,
        7: PartitionGranularity.WEEK,
    }.get(step)


def plan_summary(xml: str, *, partitions_total: int | None = None) -> PlanSummary:
    """Turn a ``ShowPlanXML`` document into a :class:`~mdqm.models.PlanSummary`.

    The pruning verdict is read from the partition-id seek predicate, which is where an *estimated*
    SQL Server plan puts it. Measured on 2022 (16.0.4135.4) with a literal key filter::

        <IndexScan ...><Object ... Table="[usage_events]" Index="[usage_events_pk]" .../>
          <SeekPredicates><SeekPredicateNew><SeekKeys>
            <StartRange ScanType="GE"><RangeColumns>
              <ColumnReference Column="PtnId1000"/>...</RangeColumns>
              <RangeExpressions><ScalarOperator ScalarString="(9)">           <!-- partition 9 -->
            <EndRange ScanType="LE">...<ScalarOperator ScalarString="(11)-(1)"> <!-- last = 10 -->

    So the plan states the partition ids when the filter value is a compile-time literal -- which
    is what the compiler emits (it inlines literals). A parameterised value shows
    ``RangePartitionNew(...)`` there instead, which is a runtime expression: the range is in the
    plan but its ends are not known before execution, so the verdict is ``UNPROVEN`` (Oracle's
    ``PSTART = KEY``). A partitioned scan with no partition-id predicate at all reaches every
    partition, which proves nothing either. ``UNKNOWN`` is reserved for a plan with no partitioned
    table, where there is nothing to prove. The element that would report the count directly
    (``RunTimePartitionSummary/PartitionsAccessed``) only exists in an *actual* plan.
    """
    root = ElementTree.fromstring(xml)  # noqa: S314 - the document comes from the server
    statement = root.find(".//{*}StmtSimple")
    indexes: list[str] = []
    for relop in root.findall(".//{*}RelOp"):
        if "Index" not in str(relop.get("PhysicalOp", "")):
            continue
        for obj in relop.findall(".//{*}Object"):
            name = obj.get("Index")
            if name and name not in indexes:
                indexes.append(name)
    proof, scanned = _pruning_verdict(root)
    raw = xml if len(xml) <= _RAW_PLAN_LIMIT else xml[:_RAW_PLAN_LIMIT] + "<!-- truncated -->"
    return PlanSummary(
        partitions_scanned=scanned,
        partitions_total=partitions_total if proof is not PruningProof.UNKNOWN else None,
        uses_index=bool(indexes) if indexes else None,
        indexes_used=tuple(indexes),
        est_rows=_as_float(None if statement is None else statement.get("StatementEstRows")),
        est_cost=_as_float(None if statement is None else statement.get("StatementSubTreeCost")),
        pruning_proof=proof,
        raw=raw or None,
    )


def _pruning_verdict(root: ElementTree.Element) -> tuple[PruningProof, int | None]:
    """``(proof, partitions_scanned)`` from the partition-id seek predicates in the plan."""
    partitioned = [
        relop for relop in root.findall(".//{*}RelOp") if relop.get("Partitioned") == "1"
    ]
    if not partitioned:
        return PruningProof.UNKNOWN, None
    if len(partitioned) == 1:
        bounds = _partition_id_range(partitioned[0])
        if bounds is not None:
            first, last = bounds
            if first is not None and last is not None and last >= first:
                return PruningProof.PROVEN, last - first + 1
    # Either the range is a runtime expression (a parameterised value) or there is no partition-id
    # predicate at all, which is a scan of every partition: neither proves anything was pruned.
    return PruningProof.UNPROVEN, None


def _partition_id_range(relop: ElementTree.Element) -> tuple[int | None, int | None] | None:
    """The literal partition-id range a ``PtnId`` seek predicate names, or ``None`` if absent."""
    for keys in relop.findall(".//{*}SeekKeys"):
        columns = [ref.get("Column", "") for ref in keys.findall(".//{*}ColumnReference")]
        if not any(column.startswith("PtnId") for column in columns):
            continue
        bounds: list[int | None] = []
        for side in ("StartRange", "EndRange"):
            expression = keys.find(f"{{*}}{side}/{{*}}RangeExpressions/{{*}}ScalarOperator")
            bounds.append(
                None if expression is None else _partition_id(expression.get("ScalarString"))
            )
        return bounds[0], bounds[1]
    return None


def _partition_id(text: str | None) -> int | None:
    """``"(9)"`` -> 9 and ``"(11)-(1)"`` -> 10 (``end - 1`` as SQL Server writes it); else ``None``.

    Only those two literal shapes are accepted: anything else (``RangePartitionNew(...)``) is a
    runtime expression, and guessing a number from it is exactly the invention this method must not
    do.
    """
    if not text:
        return None
    match = _PARTITION_ID_RE.match(text.strip())
    if match is None:
        return None
    try:
        first = int(match.group(1))
        subtract = match.group(2)
        return first if subtract is None else first - int(subtract)
    except ValueError:  # the regex captures digits only; a parse failure is still "no number"
        return None


#: ``(9)`` or ``(11)-(1)``: the two literal shapes a partition-id range expression takes.
_PARTITION_ID_RE: Final = re.compile(r"^\((-?\d+)\)(?:-\((\d+)\))?$")


#: How much plan XML is kept in ``PlanSummary.raw``. A plan for a wide query can run to megabytes,
#: and the summary is what a model reads; the head carries the statement and the scan nodes.
_RAW_PLAN_LIMIT: Final = 20_000


def _raw_queries(rows: Sequence[Sequence[Any]]) -> list[RawQuery]:
    """Query Store rows -> :class:`RawQuery`; counters are cumulative, so calls are reported."""
    out: list[RawQuery] = []
    for text, executions, total_us, rows_processed, last_execution in rows:
        body = None if text is None else str(text)
        if not body:
            continue
        calls = _as_int(executions) or 0
        total_ms = _as_float(total_us)
        total_ms = None if total_ms is None else total_ms / 1000.0
        out.append(
            RawQuery(
                text=body,
                dialect="tsql",
                calls=calls,
                total_time_ms=total_ms,
                mean_time_ms=None if total_ms is None or calls == 0 else total_ms / calls,
                rows=_as_int(rows_processed),
                last_seen=_as_utc(last_execution),
            )
        )
    return out


def _stream_error(
    exc: Exception, *, cancelled: bool, timed_out: bool, timeout_s: float
) -> Exception:
    """Translate a driver error: a cancellation or a missed deadline, otherwise the raw error.

    ``mssql-python`` 1.19.0 reports both interrupts as ``OperationalError`` with the useful text in
    the message ("Operation canceled" vs "Query timeout expired"), so the message is what separates
    "the caller cancelled" from "the deadline passed".
    """
    message = str(exc)
    if _TIMEOUT_MARKER in message or _CANCELLED_MARKER in message:
        if cancelled:
            return SqlServerCancelled("the SQL Server statement was cancelled by the caller")
        return SqlServerTimeout(
            f"the SQL Server statement exceeded the {timeout_s}s timeout and was cancelled: "
            f"{message}"
        )
    if timed_out:
        # The watchdog fired but the driver reported something else; the deadline is still the
        # reason, and saying so beats a driver error the caller cannot act on.
        return SqlServerTimeout(
            f"the SQL Server statement exceeded the {timeout_s}s timeout: {message}"
        )
    return exc


def _parse_boundary(text: str) -> datetime | None:
    """A ``CONVERT(varchar(64), value, 126)`` boundary as a UTC datetime, or ``None``.

    The value is naive UTC in the database (DESIGN 5), so UTC is attached here, once, at the edge.
    """
    candidate = text.strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = candidate[:-1]
    if _BOUNDARY_RE.match(candidate) is None:
        return None
    try:
        return datetime.fromisoformat(candidate).replace(tzinfo=UTC)
    except ValueError:
        return None


#: The shapes ``CONVERT(varchar, value, 126)`` produces: a date, optionally with a time and up to
#: nine fractional digits (``datetime2(7)`` renders seven). Anything else is refused, not guessed.
_BOUNDARY_RE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d{1,9})?)?$")


def _native_type(
    type_name: str, length: int | None, precision: int | None, scale: int | None
) -> str:
    """The type as SQL Server spells it, with the size/precision the catalog keeps separately."""
    lowered = type_name.lower()
    if lowered in {"varchar", "char", "varbinary", "binary"} and length is not None:
        return f"{type_name}({'max' if length == -1 else length})"
    if lowered in {"nvarchar", "nchar"} and length is not None:
        return f"{type_name}({'max' if length == -1 else length // 2})"
    if lowered in {"decimal", "numeric"} and precision is not None:
        return f"{type_name}({precision},{scale if scale is not None else 0})"
    if lowered in {"datetime2", "time", "datetimeoffset"} and scale is not None:
        return f"{type_name}({scale})"
    return type_name


def _arrow_schema(description: Sequence[Any] | None) -> pa.Schema:
    """The stream's schema, known before the first row arrives (a cursor describes itself)."""
    fields = []
    for entry in description or ():
        field_type = arrow_type(entry[1], _as_int(entry[4]), _as_int(entry[5]))
        fields.append(pa.field(str(entry[0]), field_type, nullable=entry[6] is not False))
    return pa.schema(fields)


def _arrays_from_rows(schema: pa.Schema, rows: Sequence[Sequence[Any]]) -> list[pa.Array]:
    """One Arrow array per field, converting each cell to the schema's declared type."""
    arrays: list[pa.Array] = []
    for position, field in enumerate(schema):
        column = [row[position] for row in rows]
        if pa.types.is_string(field.type):
            arrays.append(
                pa.array(
                    [None if value is None else str(value) for value in column], type=field.type
                )
            )
        elif pa.types.is_binary(field.type):
            arrays.append(
                pa.array(
                    [None if value is None else bytes(value) for value in column], type=field.type
                )
            )
        else:
            arrays.append(pa.array(column, type=field.type))
    return arrays


def _as_int(value: object) -> int | None:
    """An int from a driver value, or ``None`` (never a silent 0)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    """A float from a driver value or from an XML attribute, or ``None``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float | Decimal):
        # pi-lens-ignore: unchecked-numeric-parse-python -- a number by the branch above
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _as_utc(value: object) -> datetime | None:
    """A naive value is UTC (SQL Server has no zoned type); an aware one is converted."""
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _positive_int(value: object, default: int, key: str) -> int:
    """A positive integer option, with the key named in the error."""
    if value is None:
        return default
    number = _as_int(value)
    if number is None or number <= 0:
        raise SqlServerConfigError(f"options.{key} must be a positive integer, got {value!r}")
    return number


def _conninfo(config: SourceConfig, secrets: SecretProvider | None) -> str:
    """The ODBC connection string, with the password resolved from its reference.

    The password only lives in the string this module hands the driver; nothing logs it, and
    ``__repr__`` prints the server half only.
    """
    options = dict(config.options or {})
    user = str(options.get("user") or "").strip()
    if not user:
        raise SqlServerConfigError(
            f"source {config.name!r}: no database user. Set options.user; the server needs a "
            "per-person identity, not a shared login."
        )
    dsn = str(options.get("dsn") or "").strip()
    if not dsn:
        host = str(options.get("host") or "").strip()
        database = str(options.get("database") or "").strip()
        if not host or not database:
            raise SqlServerConfigError(
                f"source {config.name!r}: set options.host and options.database "
                "(or a whole options.dsn)"
            )
        port = _positive_int(options.get("port"), DEFAULT_PORT, "port")
        dsn = f"Server={host},{port};Database={database}"
    if config.secret is None:
        raise SqlServerConfigError(
            f"source {config.name!r}: no secret reference. Set `secret: <name>`; a password in "
            "config would be a password in git."
        )
    if secrets is None:
        raise SqlServerConfigError(
            f"source {config.name!r} references secret {config.secret!r} but no secret provider is "
            "configured; without one the password cannot be resolved"
        )
    password = secrets.get(config.secret)
    encrypt = str(options.get("encrypt", "yes"))
    trust = str(options.get("trust_server_certificate", "no"))
    intent = str(options.get("application_intent", "")).strip()
    parts = [
        dsn,
        f"UID={user}",
        f"PWD={password}",
        f"Encrypt={encrypt}",
        f"TrustServerCertificate={trust}",
        "APP=mdqm",
    ]
    if intent:
        parts.append(f"ApplicationIntent={intent}")
    return ";".join(parts)
