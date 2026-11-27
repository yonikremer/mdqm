"""The Oracle source adapter (task A07): thin connect, catalog, partitions, streaming, cancel.

Oracle is the second SQL engine in the MVP and the one that punishes assumptions, so this module
follows :mod:`mdqm.adapters.postgresql` and says where the engine differs:

* **Thin mode, no Oracle Client.** ``oracledb`` talks the wire protocol itself, so the deployment
  needs no Instant Client and no ``ORACLE_HOME``. :meth:`OracleAdapter.probe` reports
  ``is_thin_mode()`` rather than assuming it.
* **Read-only in three layers, as everywhere else.** The *account* is read-only (the per-person
  user in ``options.user``), the *session* is read-only (``SET TRANSACTION READ ONLY`` before every
  execution, which ends when the transaction ends) and the AST guard is a separate module. A write
  sent straight to :meth:`OracleAdapter.execute` therefore dies twice over, and the integration test
  prints Oracle's own error for each layer: ``ORA-41900`` (missing privilege) for a read-only
  account, ``ORA-01456`` (write inside a READ ONLY transaction) for a user that *could* write.
* **Streaming with two real interrupts.** A cursor is fetched in ``arraysize`` batches, so
  ``Limits.max_rows``/``max_bytes`` stop the scan mid-flight, and the deadline is enforced from both
  sides: a watchdog calls ``Connection.cancel()`` at ``Limits.timeout_s`` (Oracle answers
  ``ORA-01013`` and the session survives) and the client clock is checked between batches.
  ``Connection.call_timeout`` is only a generous backstop, never the deadline itself: measured on
  oracledb 26.2.0, when ``call_timeout`` fires the driver has to drain the response, and a slow
  server makes *that* drain time out, which ends in ``DPY-4011`` ("the database or network closed
  the connection") and a dead session -- the opposite of a clean deadline. ``cancel()`` leaves the
  connection reusable, which the integration test proves by querying again on the same pool.
* **Deterministic time.** ``exec_ts`` arrives inside the fragment and is never substituted here.
  ``ALTER SESSION SET TIME_ZONE = 'UTC'`` makes a naive literal unambiguous. Oracle's ``TIMESTAMP``
  has no zone; the stored values are naive UTC (D05 documents why ``WITH TIME ZONE`` is not an
  option for interval partitioning) and are read back as naive UTC.
* **Statistics are read, never refreshed.** ``EXPLAIN PLAN`` writes into ``SYS.PLAN_TABLE$``, which
  on 12c+ is a *session* global temporary table with ``PUBLIC`` DML grants: a read-only account can
  use it without owning or creating anything. What a locked-down account needs is that grant plus
  ``SELECT``/``DELETE`` on the plan table it names (``explain`` reports ``NotSupported`` otherwise).

Configuration lives in :class:`~mdqm.config.settings.SourceConfig`; the password is a *reference*
resolved through a :class:`~mdqm.config.secrets.SecretProvider` at connect time::

    sources:
      billing_oracle:
        adapter: oracle
        secret: billing_ro                 # a reference, never the value
        options:
          host: oracle.internal
          port: 1521
          service_name: MDQM               # or: dsn: "oracle.internal:1521/MDQM"
          user: MDQM_RO                    # the per-person identity, not a shared login

Usage::

    adapter = OracleAdapter(source_config, env_secret_provider)
    stream = adapter.execute(fragment, Limits(timeout_s=5, max_rows=1000, max_bytes=4_000_000,
                                             max_partitions=8))
    for batch in stream:                    # pyarrow.RecordBatch, lazily
        ...
    stream.close()
"""

from __future__ import annotations

import contextlib
import logging
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, Final

import oracledb
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

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CALL_TIMEOUT_MS",
    "DEFAULT_POOL_SIZE",
    "OracleAdapter",
    "OracleArrowStream",
    "OracleCancelled",
    "OracleConfigError",
    "OracleTimeout",
    "arrow_type",
    "normalized_type",
    "parse_high_value",
    "plan_summary",
]

_log = logging.getLogger(__name__)

#: Rows fetched per round trip. Small on purpose: the row/byte cap must be able to stop mid-scan
#: instead of after one huge batch has already crossed the wire.
DEFAULT_BATCH_SIZE: Final = 1024

#: Connections the pool keeps at hand. The governor already limits concurrency per source.
DEFAULT_POOL_SIZE: Final = 4

#: Safety net for a statement that reaches Oracle without :class:`~mdqm.models.Limits` (a probe, a
#: catalog read).
DEFAULT_CALL_TIMEOUT_MS: Final = 60_000

#: How far beyond ``Limits.timeout_s`` the driver's own ``call_timeout`` sits. The deadline is
#: enforced by cancelling the statement (see :class:`_Deadline`); this is only here so a call that
#: never returns cannot pin a pooled session for ever.
CALL_TIMEOUT_BACKSTOP_MS: Final = 5_000

#: Oracle's well-known listener port. The host and the service are the organisation's facts and
#: have no default; a wrong guess there would connect to the wrong database.
DEFAULT_PORT: Final = 1521

#: Error numbers that mean "this account may not read that view": the object is invisible
#: (``ORA-00942``), the privilege is missing (``ORA-01031``) or the view's underlying table is
#: (``ORA-01042``). They are how a probe reports a capability as unsupported instead of crashing.
_NO_PRIVILEGE: Final = frozenset({942, 1031, 1042})

#: ``ORA-01013``: the statement was cancelled, by us or by a client-side timeout.
_CANCELLED: Final = 1013

#: ``DPY-4024``: oracledb's ``Connection.call_timeout`` fired. Thin mode reports it with code 0,
#: so the message is the only identification (verified on oracledb 26.2.0).
_CALL_TIMEOUT_MARKER: Final = "DPY-4024"

#: ``DPY-4011``: the driver lost the connection. It happens when ``call_timeout`` fires and draining
#: the response times out in turn, so it only counts as a deadline once the deadline passed.
_CONNECTION_LOST_MARKER: Final = "DPY-4011"

_KNOWN_OPTIONS: Final = frozenset(
    {
        "dsn",
        "host",
        "port",
        "service_name",
        "user",
        "pool_size",
        "connect_timeout",
        "call_timeout_ms",
        "batch_size",
    }
)


class OracleConfigError(ValueError):
    """The source cannot be connected with this configuration.

    The message names the source and the offending ``options`` key, never a credential, so it is
    safe to log and to show to an operator.
    """


class OracleTimeout(TimeoutError):
    """The statement exceeded ``Limits.timeout_s`` (client deadline or ``call_timeout``).

    A ``TimeoutError``, never a :class:`~mdqm.adapters.base.NotSupported`: a deadline is an answer
    about this execution, not about what the source can do.
    """


class OracleCancelled(RuntimeError):
    """The statement was aborted by :meth:`OracleAdapter.cancel` (``ORA-01013``)."""


# --- type mapping ---------------------------------------------------------------------------

#: ``ALL_TAB_COLS.data_type`` (without any ``(p,s)`` suffix) -> the project's portable vocabulary.
_NORMALIZED_BY_TYPE: Final[dict[str, NormalizedType]] = {
    "VARCHAR2": NormalizedType.STRING,
    "NVARCHAR2": NormalizedType.STRING,
    "VARCHAR": NormalizedType.STRING,
    "CHAR": NormalizedType.STRING,
    "NCHAR": NormalizedType.STRING,
    "CLOB": NormalizedType.STRING,
    "NCLOB": NormalizedType.STRING,
    "LONG": NormalizedType.STRING,
    "ROWID": NormalizedType.STRING,
    "UROWID": NormalizedType.STRING,
    "XMLTYPE": NormalizedType.STRING,
    "JSON": NormalizedType.JSON,
    "NUMBER": NormalizedType.DECIMAL,
    "INTEGER": NormalizedType.INTEGER,
    "FLOAT": NormalizedType.FLOAT,
    "BINARY_FLOAT": NormalizedType.FLOAT,
    "BINARY_DOUBLE": NormalizedType.FLOAT,
    "DATE": NormalizedType.DATE,
    "TIMESTAMP": NormalizedType.TIMESTAMP,
    "TIMESTAMP WITH TIME ZONE": NormalizedType.TIMESTAMP,
    "TIMESTAMP WITH LOCAL TIME ZONE": NormalizedType.TIMESTAMP,
    "RAW": NormalizedType.BINARY,
    "LONG RAW": NormalizedType.BINARY,
    "BLOB": NormalizedType.BINARY,
    "BFILE": NormalizedType.BINARY,
}

#: ``ALL_TAB_COLS.data_type`` -> Arrow type, used when the description carries no precision.
_ARROW_BY_DB_TYPE: Final[dict[str, pa.DataType]] = {
    "VARCHAR2": pa.string(),
    "NVARCHAR2": pa.string(),
    "CHAR": pa.string(),
    "NCHAR": pa.string(),
    "CLOB": pa.string(),
    "NCLOB": pa.string(),
    "LONG": pa.string(),
    "ROWID": pa.string(),
    "UROWID": pa.string(),
    "XMLTYPE": pa.string(),
    "JSON": pa.string(),
    "BINARY_FLOAT": pa.float32(),
    "BINARY_DOUBLE": pa.float64(),
    "DATE": pa.timestamp("us"),
    "TIMESTAMP": pa.timestamp("us"),
    "TIMESTAMP WITH TIME ZONE": pa.timestamp("us", tz="UTC"),
    "TIMESTAMP WITH LOCAL TIME ZONE": pa.timestamp("us", tz="UTC"),
    "RAW": pa.binary(),
    "LONG RAW": pa.binary(),
    "BLOB": pa.binary(),
}

#: Oracle precision above which a scale-0 ``NUMBER`` no longer fits ``int64`` (19 digits can exceed
#: ``2**63 - 1``); those stay decimals so a big id is never silently rounded.
_INT64_MAX_PRECISION: Final = 18

#: Widest decimal Arrow 128-bit can hold. A ``NUMBER`` is at most ``NUMBER(38, s)``.
_DECIMAL_PRECISION: Final = 38

_SIZE_SUFFIX_RE: Final = re.compile(r"\s*\([^)]*\)")
_INTERVAL_RE: Final = re.compile(
    r"NUMTOY?[MD]S?INTERVAL\(\s*(?P<count>\d+)\s*,\s*'(?P<unit>YEAR|MONTH|DAY|HOUR|MINUTE|SECOND)'\s*\)",
    re.IGNORECASE,
)
_UNIT_MONTHS: Final[dict[str, int]] = {
    "YEAR": 12,
    "MONTH": 1,
    "DAY": 0,
    "HOUR": 0,
    "MINUTE": 0,
    "SECOND": 0,
}

#: Interval -> the granularity the planner reasons about. Only whole-month and whole-day intervals
#: have a granularity here; a finer one keeps ``None`` and the bounds tell the truth anyway.
_GRANULARITY_BY_MONTHS: Final[dict[int, PartitionGranularity]] = {
    1: PartitionGranularity.MONTH,
    3: PartitionGranularity.QUARTER,
    12: PartitionGranularity.YEAR,
}
_GRANULARITY_DAYS: Final[dict[int, PartitionGranularity]] = {
    1: PartitionGranularity.DAY,
    7: PartitionGranularity.WEEK,
}


def normalized_type(data_type: str, precision: int | None, scale: int | None) -> NormalizedType:
    """Map ``ALL_TAB_COLS`` type text onto the project's portable vocabulary.

    ``TIMESTAMP(6)`` and ``VARCHAR2`` arrive with their size suffix stripped by
    :func:`base_type`, so the lookup is on the bare type. A ``NUMBER`` with scale 0 *and* a
    declared precision is a whole number (``INTEGER``); with a fractional scale it is ``DECIMAL``
    (money), which is the distinction the conversion lane needs.
    """
    base = base_type(data_type)
    if base == "NUMBER":
        if scale == 0 and precision is not None:
            return NormalizedType.INTEGER
        return NormalizedType.DECIMAL
    known = _NORMALIZED_BY_TYPE.get(base)
    if known is not None:
        return known
    if base.startswith("INTERVAL"):
        return NormalizedType.TIME
    return NormalizedType.OTHER


def base_type(data_type: str) -> str:
    """``'TIMESTAMP(6)'`` -> ``'TIMESTAMP'``: the type without its precision suffixes.

    Both positions occur in the dictionary: ``VARCHAR2``-style types carry the size at the end
    (``TIMESTAMP(6)``) while an interval carries it in the middle
    (``INTERVAL DAY(2) TO SECOND(6)``), so every parenthesised group is removed, not just the last.
    """
    return _SIZE_SUFFIX_RE.sub("", data_type.strip()).strip().upper()


def arrow_type(type_code: object, precision: int | None, scale: int | None) -> pa.DataType:
    """Arrow type for one result column, from the cursor description.

    oracledb reports a ``NUMBER`` with its declared precision/scale, so money keeps its exact scale
    (``NUMBER(12,2)`` -> ``decimal128(38, 2)``) instead of drifting through ``float``. An
    unconstrained ``NUMBER`` (no precision) can hold a value no Arrow decimal can, and reading it as
    ``float64`` is the only honest carrier.
    """
    if type_code == oracledb.DB_TYPE_NUMBER:
        if scale is not None and scale > 0:
            return pa.decimal128(_DECIMAL_PRECISION, min(scale, _DECIMAL_PRECISION))
        if precision is not None and precision <= _INT64_MAX_PRECISION:
            return pa.int64()
        if precision is not None:
            return pa.decimal128(_DECIMAL_PRECISION, 0)
        return pa.float64()
    by_code: dict[object, pa.DataType] = {
        oracledb.DB_TYPE_VARCHAR: pa.string(),
        oracledb.DB_TYPE_NVARCHAR: pa.string(),
        oracledb.DB_TYPE_CHAR: pa.string(),
        oracledb.DB_TYPE_NCHAR: pa.string(),
        oracledb.DB_TYPE_LONG: pa.string(),
        oracledb.DB_TYPE_CLOB: pa.string(),
        oracledb.DB_TYPE_NCLOB: pa.string(),
        oracledb.DB_TYPE_JSON: pa.string(),
        oracledb.DB_TYPE_DATE: pa.timestamp("us"),
        oracledb.DB_TYPE_TIMESTAMP: pa.timestamp("us"),
        oracledb.DB_TYPE_TIMESTAMP_TZ: pa.timestamp("us", tz="UTC"),
        oracledb.DB_TYPE_TIMESTAMP_LTZ: pa.timestamp("us", tz="UTC"),
        oracledb.DB_TYPE_BINARY_FLOAT: pa.float32(),
        oracledb.DB_TYPE_BINARY_DOUBLE: pa.float64(),
        oracledb.DB_TYPE_RAW: pa.binary(),
        oracledb.DB_TYPE_LONG_RAW: pa.binary(),
        oracledb.DB_TYPE_BLOB: pa.binary(),
    }
    return by_code.get(type_code, pa.string())


def parse_high_value(text: object) -> datetime | None:
    """Parse a partition's ``HIGH_VALUE`` text into a UTC datetime, or ``None``.

    ``ALL_TAB_PARTITIONS.HIGH_VALUE`` is a ``LONG`` holding the value as DDL text:
    ``TIMESTAMP' 2026-01-01 00:00:00'`` or ``DATE ' 2026-01-01 00:00:00'`` (note Oracle's space
    after the quote), and ``MAXVALUE`` for the last range partition. Anything unparseable returns
    ``None`` instead of a guessed bound: a wrong bound is worse than a missing one, because the
    planner prunes on it.
    """
    if text is None:
        return None
    body = str(text).strip()
    if not body or body.upper() == "MAXVALUE":
        return None
    match = re.match(r"^(?P<type>[A-Z ]+?)\s*'(?P<value>.*)'\s*$", body, re.IGNORECASE)
    if match is not None:
        body = match.group("value").strip()
    try:
        return datetime.fromisoformat(body).replace(tzinfo=UTC)
    except ValueError:
        return None


def interval_granularity(interval: object) -> PartitionGranularity | None:
    """``NUMTOYMINTERVAL(1, 'MONTH')`` -> ``MONTH``; an unparsable interval is ``None``."""
    if interval is None:
        return None
    match = _INTERVAL_RE.search(str(interval))
    if match is None:
        return None
    count = int(match.group("count"))
    unit = match.group("unit").upper()
    months = _UNIT_MONTHS[unit] * count
    if months:
        return _GRANULARITY_BY_MONTHS.get(months)
    return _GRANULARITY_DAYS.get(count)


# --- SQL ------------------------------------------------------------------------------------

# Every catalog query below excludes the accounts Oracle itself maintains (``SYS``, ``SYSTEM``,
# ``XDB``, ...): they own the dictionary, not the organisation's data, and the exclusion is
# data-driven instead of a hard-coded list of schema names that grows with every release.

_CATALOG_TABLES_SQL: Final = """
SELECT t.owner, t.table_name, t.num_rows
  FROM all_tables t
 WHERE t.owner NOT IN (SELECT username FROM all_users WHERE oracle_maintained = 'Y')
   AND t.temporary = 'N'
   AND t.nested = 'NO'
   AND t.secondary = 'N'
   AND t.iot_name IS NULL
   AND t.table_name NOT LIKE 'BIN$%'
 ORDER BY t.owner, t.table_name
"""

_CATALOG_VIEWS_SQL: Final = """
SELECT v.owner, v.view_name
  FROM all_views v
 WHERE v.owner NOT IN (SELECT username FROM all_users WHERE oracle_maintained = 'Y')
 ORDER BY v.owner, v.view_name
"""

_CATALOG_COLUMNS_SQL: Final = """
SELECT c.owner, c.table_name, c.column_name, c.data_type, c.data_length,
       c.data_precision, c.data_scale, c.nullable, c.column_id
  FROM all_tab_cols c
 WHERE c.owner NOT IN (SELECT username FROM all_users WHERE oracle_maintained = 'Y')
   AND c.hidden_column = 'NO'
 ORDER BY c.owner, c.table_name, c.column_id
"""

_CATALOG_INDEXES_SQL: Final = """
SELECT i.table_owner, i.table_name, i.index_name, i.uniqueness, i.index_type, i.partitioned,
       ic.column_name, ic.column_position
  FROM all_indexes i
  LEFT JOIN all_ind_columns ic
         ON ic.index_owner = i.owner AND ic.index_name = i.index_name
 WHERE i.table_owner NOT IN (SELECT username FROM all_users WHERE oracle_maintained = 'Y')
   AND i.index_type <> 'LOB'
   AND i.index_type <> 'DOMAIN'
 ORDER BY i.table_owner, i.table_name, i.index_name, ic.column_position
"""

_CATALOG_CONSTRAINTS_SQL: Final = """
SELECT c.owner, c.table_name, c.constraint_name, c.constraint_type,
       c.r_owner, c.r_constraint_name
  FROM all_constraints c
 WHERE c.owner NOT IN (SELECT username FROM all_users WHERE oracle_maintained = 'Y')
   AND c.constraint_type IN ('P', 'R')
 ORDER BY c.owner, c.table_name, c.constraint_name
"""

_CATALOG_CONSTRAINT_COLUMNS_SQL: Final = """
SELECT cc.owner, cc.constraint_name, cc.column_name, cc.position
  FROM all_cons_columns cc
 WHERE cc.owner NOT IN (SELECT username FROM all_users WHERE oracle_maintained = 'Y')
 ORDER BY cc.owner, cc.constraint_name, cc.position
"""

_PARTITIONED_TABLES_SQL: Final = """
SELECT t.owner, t.table_name, t.partitioning_type, t.subpartitioning_type, t.interval
  FROM all_part_tables t
 WHERE t.owner NOT IN (SELECT username FROM all_users WHERE oracle_maintained = 'Y')
 ORDER BY t.owner, t.table_name
"""

_PARTITION_KEYS_SQL: Final = """
SELECT k.owner, k.name, k.column_name, k.column_position
  FROM all_part_key_columns k
 WHERE k.owner NOT IN (SELECT username FROM all_users WHERE oracle_maintained = 'Y')
   AND k.object_type = 'TABLE'
 ORDER BY k.owner, k.name, k.column_position
"""

_PARTITIONS_SQL: Final = """
SELECT p.table_owner, p.table_name, p.partition_name, p.partition_position,
       p.high_value, p.num_rows
  FROM all_tab_partitions p
 WHERE p.table_owner NOT IN (SELECT username FROM all_users WHERE oracle_maintained = 'Y')
 ORDER BY p.table_owner, p.table_name, p.partition_position
"""

_XPLAN_SQL: Final = """
SELECT plan_table_output
  FROM table(dbms_xplan.display('PLAN_TABLE', :1, 'BASIC +PARTITION'))
"""

_PLAN_SQL: Final = """
SELECT id, parent_id, operation, options, object_name, cardinality, cost,
       partition_start, partition_stop
  FROM plan_table
 WHERE statement_id = :1
 ORDER BY id
"""

_VERSION_SQL: Final = "SELECT banner_full FROM v$version WHERE rownum = 1"

_QUERY_STORE_SQL: Final = """
SELECT sql_fulltext, executions, elapsed_time, rows_processed, last_active_time
  FROM v$sql
 WHERE last_active_time >= :1
   AND executions > 0
 ORDER BY elapsed_time DESC
"""

_MUTABILITY_UNSUPPORTED: Final = (
    "ALL_TAB_MODIFICATIONS is only refreshed by DBMS_STATS (FLUSH_DATABASE_MONITORING_INFO or a "
    "stats gather), so a hot table reads as unchanged between gathers; a stale counter would make "
    "a mutable_critical table look cacheable, so the checksum fallback is the safer answer"
)


class _Deadline:
    """The deadline of one execution: a watchdog that cancels the statement, plus its flag.

    ``Connection.cancel()`` is the interrupt this driver implements cleanly (``ORA-01013``, session
    reusable); ``call_timeout`` is left as a backstop several seconds beyond the deadline. The flag
    is what tells the error translation that a cancellation came from the deadline rather than from
    a caller who asked for it.
    """

    def __init__(
        self, connection: oracledb.Connection, timeout_s: float, interrupt: Callable[[], None]
    ) -> None:
        self.fired = False
        self._interrupt = interrupt
        self._timer = threading.Timer(timeout_s, self._fire)
        self._timer.daemon = True

    def arm(self) -> None:
        """Start the watchdog; it fires once, ``timeout_s`` after this call."""
        self._timer.start()

    def _fire(self) -> None:
        self.fired = True
        with contextlib.suppress(oracledb.Error):
            self._interrupt()

    def cancel(self) -> None:
        """Stop the watchdog, and break the statement if it is still running."""
        self._timer.cancel()
        with contextlib.suppress(oracledb.Error):
            self._interrupt()

    def disarm(self) -> None:
        """Stop the watchdog only: the statement is finished, so there is nothing to break."""
        self._timer.cancel()


class OracleArrowStream:
    """The :class:`~mdqm.adapters.base.ArrowStream` an Oracle ``execute`` returns.

    It owns one pooled connection until :meth:`close`. The statement is already running (the
    cursor was opened by ``execute``, which is where the schema comes from), so batches are pulled
    with ``fetchmany`` and the caps stop the scan instead of trimming a buffer that already crossed
    the wire.
    """

    def __init__(
        self,
        *,
        connection: oracledb.Connection,
        cursor: oracledb.Cursor,
        schema: pa.Schema,
        limits: Limits,
        deadline: float,
        clock: Callable[[], float],
        release: Callable[[oracledb.Connection], None],
        watchdog: _Deadline | None = None,
        default_timeout_ms: int = DEFAULT_CALL_TIMEOUT_MS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.schema: pa.Schema = schema
        self._connection = connection
        self._cursor = cursor
        self._max_rows = limits.max_rows
        self._max_bytes = limits.max_bytes
        self._timeout_s = limits.timeout_s
        self._deadline = deadline
        self._clock = clock
        self._release = release
        self._watchdog = watchdog
        self._default_timeout_ms = default_timeout_ms
        self._batch_size = batch_size
        self._closed = False
        self._cancelled = False
        self.rows_returned = 0
        self.bytes_returned = 0

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        """Yield batches until the source, a cap or a deadline stops the scan."""
        if self._closed or self._cancelled:
            return
        while True:
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
        except oracledb.DatabaseError as exc:
            raise _stream_error(
                exc,
                cancelled=self._cancelled,
                timed_out=self._fired(),
                timeout_s=self._timeout_s,
            ) from exc

    def _fired(self) -> bool:
        """True when the watchdog already reached the deadline for this execution."""
        return self._watchdog is not None and self._watchdog.fired

    def _check_deadline(self) -> None:
        if self._fired() or self._clock() > self._deadline:
            # Stop the server too: we are not going to read the rest, and a primary database must
            # not keep producing rows for a client that walked away.
            if self._watchdog is not None:
                self._watchdog.cancel()
            raise OracleTimeout(
                f"the Oracle statement exceeded the {self._timeout_s}s timeout while streaming"
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
        keep = int(remaining_bytes / per_row) if per_row > 0 else batch.num_rows
        return batch.slice(0, min(batch.num_rows, max(keep, 0)))

    def cancel(self) -> None:
        """Ask Oracle to abort the running statement (``ORA-01013``). Idempotent.

        Only the break is sent: the reader (or :meth:`close`) still releases the cursor and the
        connection, so cancelling from another thread cannot close the socket underneath a fetch
        that is in flight.
        """
        if self._closed or self._cancelled:
            return
        self._cancelled = True
        with contextlib.suppress(oracledb.Error):
            self._connection.cancel()

    def close(self) -> None:
        """Release cursor, read-only transaction and connection. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._watchdog is not None:
            self._watchdog.disarm()
        with contextlib.suppress(oracledb.Error):
            self._cursor.close()
        with contextlib.suppress(oracledb.Error):
            # Ending the transaction is what makes the next execution start one again; without it
            # the connection would keep a read-only transaction (or an aborted one) alive.
            self._connection.rollback()
        self._release(self._connection)


# --- the adapter ----------------------------------------------------------------------------


class OracleAdapter:
    """A pooled, streaming, read-only ``SourceAdapter`` for one Oracle database or PDB.

    The pool is built lazily, so an adapter for an unreachable source can be constructed (and
    ``mdqm probe`` can report the failure) without a connection attempt at import or wiring time.
    """

    kind: ClassVar[str] = "oracle"

    def __init__(
        self,
        config: SourceConfig,
        secrets: SecretProvider | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        """Build the adapter from its source config; connect on first use.

        ``secrets`` resolves ``config.secret`` to the password. It is optional so an external
        wallet works, and the password is never stored on the instance.
        """
        if config.adapter != self.kind:
            raise OracleConfigError(
                f"config names adapter {config.adapter!r} but this is the {self.kind!r} adapter"
            )
        options = dict(config.options or {})
        for unknown in sorted(set(options) - _KNOWN_OPTIONS):
            _log.warning("source %s: ignoring unknown oracle option %r", config.name, unknown)
        self._source = config.name
        self._dsn, self._user, password = _connection_parts(config, secrets)
        self._password = password
        self._pool_size = _positive_int(options.get("pool_size"), DEFAULT_POOL_SIZE, "pool_size")
        self._call_timeout_ms = _positive_int(
            options.get("call_timeout_ms"), DEFAULT_CALL_TIMEOUT_MS, "call_timeout_ms"
        )
        self._batch_size = _positive_int(
            options.get("batch_size"), DEFAULT_BATCH_SIZE, "batch_size"
        )
        self._connect_timeout_s = _positive_int(
            options.get("connect_timeout"), 10, "connect_timeout"
        )
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic
        self._pool: oracledb.ConnectionPool | None = None
        self._statement_seq = 0

    def __repr__(self) -> str:
        """Name the source and the user, never the password or the DSN."""
        return f"OracleAdapter(source={self._source!r}, user={self._user!r})"

    # --- SourceAdapter: identity ---------------------------------------------------------

    def dialect(self) -> str:
        """``oracle``: the sqlglot dialect the compiler emits for this source."""
        return "oracle"

    @property
    def source(self) -> str:
        """The configured source name; the catalog and the probe both report it."""
        return self._source

    # --- lifecycle ------------------------------------------------------------------------

    @property
    def pool(self) -> oracledb.ConnectionPool:
        """The connection pool, created on first access."""
        if self._pool is None:
            self._pool = oracledb.create_pool(
                user=self._user,
                password=self._password,
                dsn=self._dsn,
                min=0,
                max=self._pool_size,
                increment=1,
                timeout=self._connect_timeout_s,
                getmode=oracledb.POOL_GETMODE_WAIT,
                wait_timeout=self._connect_timeout_s * 1000,
                ping_interval=60,
            )
        return self._pool

    def close(self) -> None:
        """Close the pool. Idempotent."""
        pool, self._pool = self._pool, None
        if pool is not None:
            with contextlib.suppress(oracledb.Error):
                pool.close()

    @contextlib.contextmanager
    def _connection(self) -> Iterator[oracledb.Connection]:
        """A pooled connection for a probe/catalog read; rolled back and returned on exit."""
        pool = self.pool
        connection = pool.acquire()
        try:
            with contextlib.suppress(oracledb.Error):
                connection.call_timeout = self._call_timeout_ms
            yield connection
        finally:
            with contextlib.suppress(oracledb.Error):
                connection.rollback()
            with contextlib.suppress(oracledb.Error):
                pool.release(connection)

    def _prepare(self, connection: oracledb.Connection, timeout_s: float) -> None:
        """Apply the per-execution session contract: UTC, read-only transaction, call timeout.

        ``ALTER SESSION`` first (it is a session attribute and does not end the transaction), then
        the rollback that ends whatever transaction the previous statement left open, then
        ``SET TRANSACTION READ ONLY`` for this one. Oracle has no session-wide read-only mode: the
        setting dies with the transaction, so it is re-issued here rather than assumed.

        ``call_timeout`` is deliberately set *beyond* the deadline
        (:data:`CALL_TIMEOUT_BACKSTOP_MS`): it is the last-resort breaker for a call that never
        returns, not the deadline. :class:`_Deadline` says why it cannot be the deadline.
        """
        connection.call_timeout = max(
            self._call_timeout_ms, round(timeout_s * 1000) + CALL_TIMEOUT_BACKSTOP_MS
        )
        with connection.cursor() as cur:
            cur.execute("ALTER SESSION SET TIME_ZONE = 'UTC'")
            cur.execute("SET TRANSACTION READ ONLY")

    # --- SourceAdapter: probe ------------------------------------------------------------

    def probe(self) -> Capabilities:
        """Detect what this database offers, reporting only what this adapter can serve."""
        with self._connection() as connection, connection.cursor() as cur:
            version = self._server_version(cur)
            query_store, query_store_detail = _query_store_state(cur)
        partitions_supported, partitions_detail = self._probe_partitions()
        explain_supported, explain_detail = self._probe_explain()
        capabilities = {
            OPTIONAL_METHODS["extract_partitions"]: Capability(
                supported=partitions_supported, detail=partitions_detail
            ),
            OPTIONAL_METHODS["explain"]: Capability(
                supported=explain_supported, detail=explain_detail
            ),
            OPTIONAL_METHODS["mine_queries"]: Capability(
                supported=query_store, detail=query_store_detail
            ),
            OPTIONAL_METHODS["mutability_stats"]: Capability(
                supported=False, detail=_MUTABILITY_UNSUPPORTED
            ),
        }
        thin = oracledb.is_thin_mode()
        return Capabilities(
            source=self._source,
            adapter=self.kind,
            version=version if version is not None else f"thin mode: {thin}",
            retrieved_at=self._clock(),
            capabilities=capabilities,
        )

    def _server_version(self, cur: oracledb.Cursor) -> str | None:
        """The server's own banner, shortened. A hidden version is ``None``, never an error."""
        for statement in (_VERSION_SQL, "SELECT version_full FROM product_component_version"):
            try:
                # `statement` is one of the two module constants above, never a value.
                cur.execute(statement)  # nosemgrep: python-sql-injection
                row = cur.fetchone()
            except oracledb.DatabaseError:
                continue
            if row is not None and row[0] is not None:
                banner = str(row[0]).split("\n")[0]
                return banner.split(" - ")[0].strip() or None
        return None

    def _probe_partitions(self) -> tuple[bool, str]:
        """Claim ``partition_metadata`` only if extraction really returns metadata."""
        try:
            found = self.extract_partitions()
        except NotSupported as exc:
            return False, exc.detail or exc.capability
        except oracledb.DatabaseError as exc:
            return False, f"the partition catalog could not be read: {_message(exc)}"
        return True, (
            f"all_part_tables + all_part_key_columns + all_tab_partitions: "
            f"{len(found)} partitioned table(s) read"
        )

    def _probe_explain(self) -> tuple[bool, str]:
        """Claim ``explain_plan`` only if an ``EXPLAIN PLAN`` really lands in the plan table."""
        probe_sql = "SELECT 1 AS one FROM dual"
        try:
            rows, _ = self._explain(probe_sql, {})
        except NotSupported as exc:
            return False, exc.detail or exc.capability
        except oracledb.DatabaseError as exc:
            return False, f"EXPLAIN PLAN failed: {_message(exc)}"
        if not rows:
            return False, "EXPLAIN PLAN ran but PLAN_TABLE stayed empty"
        return True, (
            "EXPLAIN PLAN into the session plan table (SYS.PLAN_TABLE$ on 12c+, a PUBLIC-granted "
            "GTT); PSTART/PSTOP numeric means proven pruning, KEY means unproven"
        )

    # --- SourceAdapter: catalog ----------------------------------------------------------

    def extract_catalog(self) -> CatalogSnapshot:
        """Tables, views, columns, types, keys and indexes from the ``ALL_*`` dictionary views.

        Partition *children* are not listed (Oracle does not expose a partition as a table at all):
        :meth:`extract_partitions` reports them where the planner needs them, and the parent stays
        the only name an agent can choose.
        """
        # Every statement below is a module-level constant: the catalogs take no parameter.
        with self._connection() as connection, connection.cursor() as cur:
            cur.execute(_CATALOG_TABLES_SQL)  # nosemgrep: python-sql-injection
            table_rows = cur.fetchall()
            cur.execute(_CATALOG_VIEWS_SQL)  # nosemgrep: python-sql-injection
            view_rows = cur.fetchall()
            cur.execute(_CATALOG_COLUMNS_SQL)  # nosemgrep: python-sql-injection
            column_rows = cur.fetchall()
            cur.execute(_CATALOG_INDEXES_SQL)  # nosemgrep: python-sql-injection
            index_rows = cur.fetchall()
            cur.execute(_CATALOG_CONSTRAINTS_SQL)  # nosemgrep: python-sql-injection
            constraint_rows = cur.fetchall()
            cur.execute(_CATALOG_CONSTRAINT_COLUMNS_SQL)  # nosemgrep: python-sql-injection
            constraint_column_rows = cur.fetchall()
            cur.execute(_PARTITIONED_TABLES_SQL)  # nosemgrep: python-sql-injection
            partitioned = {(str(row[0]), str(row[1])) for row in cur.fetchall()}

        columns: dict[tuple[str, str], list[Column]] = {}
        for row in column_rows:
            key = (str(row[0]), str(row[1]))
            data_type = str(row[3])
            precision, scale = _as_int(row[5]), _as_int(row[6])
            columns.setdefault(key, []).append(
                Column(
                    name=str(row[2]),
                    native_type=_native_type(data_type, precision, scale),
                    normalized_type=normalized_type(data_type, precision, scale),
                    nullable=str(row[7]).upper() == "Y",
                    ordinal=_as_int(row[8]),
                )
            )

        indexes: dict[tuple[str, str], list[Index]] = {}
        for row in index_rows:
            key = (str(row[0]), str(row[1]))
            column = row[6]
            if column is None:
                continue  # a function-based index has no plain column list to report
            existing = next(
                (index for index in indexes.get(key, ()) if index.name == str(row[2])), None
            )
            if existing is None:
                indexes.setdefault(key, []).append(
                    Index(
                        name=str(row[2]),
                        columns=(str(column),),
                        unique=str(row[3]).upper() == "UNIQUE",
                        index_type=str(row[4]),
                        predicate=None,
                    )
                )
            else:
                indexes[key] = [
                    item.model_copy(update={"columns": (*item.columns, str(column))})
                    if item.name == existing.name
                    else item
                    for item in indexes[key]
                ]

        constraint_columns: dict[tuple[str, str], list[str]] = {}
        for row in constraint_column_rows:
            constraint_columns.setdefault((str(row[0]), str(row[1])), []).append(str(row[2]))

        primary_keys: dict[tuple[str, str], tuple[str, ...]] = {}
        foreign_keys: dict[tuple[str, str], list[ForeignKey]] = {}
        for row in constraint_rows:
            owner, table_name, constraint_name, kind = (
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]),
            )
            local = tuple(constraint_columns.get((owner, constraint_name), ()))
            if kind == "P":
                primary_keys[(owner, table_name)] = local
                continue
            referenced = tuple(constraint_columns.get((str(row[4]), str(row[5])), ()))
            if not local or len(local) != len(referenced):
                continue
            foreign_keys.setdefault((owner, table_name), []).append(
                ForeignKey(
                    columns=local,
                    ref_table=f"{row[4]}.{row[5]}",
                    ref_columns=referenced,
                    name=constraint_name,
                )
            )

        # An index that backs a primary key or a unique constraint is how the constraint is
        # enforced; reporting it as primary is what lets the linter trust the column.
        for key, table_indexes in indexes.items():
            pk = primary_keys.get(key, ())
            if pk:
                indexes[key] = [
                    item.model_copy(update={"primary": item.columns == pk})
                    for item in table_indexes
                ]

        tables: list[Table] = []
        for row in table_rows:
            tables.append(
                self._table(
                    str(row[0]),
                    str(row[1]),
                    TableKind.PARTITIONED_TABLE
                    if (str(row[0]), str(row[1])) in partitioned
                    else TableKind.TABLE,
                    columns,
                    indexes,
                    foreign_keys,
                    primary_keys,
                    _as_int(row[2]),
                )
            )
        for row in view_rows:
            tables.append(
                self._table(
                    str(row[0]),
                    str(row[1]),
                    TableKind.VIEW,
                    columns,
                    indexes,
                    foreign_keys,
                    primary_keys,
                    None,
                )
            )
        return CatalogSnapshot(source=self._source, captured_at=self._clock(), tables=tuple(tables))

    def _table(
        self,
        schema: str,
        name: str,
        kind: TableKind,
        columns: Mapping[tuple[str, str], list[Column]],
        indexes: Mapping[tuple[str, str], list[Index]],
        foreign_keys: Mapping[tuple[str, str], list[ForeignKey]],
        primary_keys: Mapping[tuple[str, str], tuple[str, ...]],
        row_estimate: int | None,
    ) -> Table:
        key = (schema, name)
        return Table(
            name=name,
            schema_name=schema,
            kind=kind,
            columns=tuple(columns.get(key, ())),
            indexes=tuple(indexes.get(key, ())),
            foreign_keys=tuple(foreign_keys.get(key, ())),
            primary_key=primary_keys.get(key, ()),
            # ALL_TABLES.num_rows is "unknown" (NULL) until a stats gather; 0 would be a claim.
            row_estimate=row_estimate,
        )

    # --- SourceAdapter: partitions --------------------------------------------------------

    def extract_partitions(self) -> list[Partitioning]:
        """Partition metadata per partitioned table: strategy, key, interval and real bounds.

        Oracle is where partitioning is *not* a property of the DDL alone: an interval-partitioned
        table (D05's ``usage_events``) declares one initial partition and the engine derives one
        per month from the data. ``ALL_TAB_PARTITIONS`` is therefore the only honest source for the
        bounds, and the lower bound is the previous partition's ``HIGH_VALUE`` -- which is exactly
        what a range partition means.
        """
        with self._connection() as connection, connection.cursor() as cur:
            cur.execute(_PARTITIONED_TABLES_SQL)
            table_rows = cur.fetchall()
            cur.execute(_PARTITION_KEYS_SQL)
            key_rows = cur.fetchall()
            cur.execute(_PARTITIONS_SQL)
            partition_rows = cur.fetchall()

        keys: dict[tuple[str, str], list[str]] = {}
        for row in key_rows:
            keys.setdefault((str(row[0]), str(row[1])), []).append(str(row[2]))
        bounds: dict[tuple[str, str], list[tuple[str, object, object]]] = {}
        for row in partition_rows:
            bounds.setdefault((str(row[0]), str(row[1])), []).append((str(row[2]), row[4], row[5]))

        out: list[Partitioning] = []
        for row in table_rows:
            owner, table_name = str(row[0]), str(row[1])
            out.append(
                _partitioning(
                    strategy=str(row[2]),
                    subpartitioning_type=str(row[3]),
                    interval=row[4],
                    key=tuple(keys.get((owner, table_name), ())),
                    partitions=bounds.get((owner, table_name), []),
                    table=f"{owner}.{table_name}",
                )
            )
        return out

    # --- SourceAdapter: plan --------------------------------------------------------------

    def explain(self, fragment: Fragment) -> PlanSummary:
        """The engine's own plan for ``fragment`` from ``EXPLAIN PLAN``.

        No statement is executed: ``EXPLAIN PLAN`` only asks the optimizer, so asking the cost of a
        scan never causes it. Two things are read out of ``PLAN_TABLE`` (and kept verbatim in
        ``raw`` via ``DBMS_XPLAN``):

        * **pruning**: a ``PARTITION RANGE``/``LIST``/``HASH`` row whose ``PSTART``/``PSTOP`` are
          partition *numbers* was pruned at parse time -- a literal filter, so ``PROVEN``. A
          ``PSTART`` of ``KEY`` means the value is only known at run time (a bind variable), so
          ``UNPROVEN``. No partition row at all means there is nothing to prove: ``UNKNOWN``.
        * **counts**: Oracle reports internal partition numbers, not a number of partitions, so
          ``partitions_scanned`` stays ``None`` (a count derived from them would be fiction) and
          ``partitions_total`` is the catalog's count for the fragment's table.

        ``pruning_proof`` is only set for a partitioned table: a table with no partitions has
        nothing to prove, and claiming otherwise is the one answer this method must never give.
        """
        text = fragment.sql
        if text is None:
            raise NotSupported(OPTIONAL_METHODS["explain"], "this fragment carries no SQL")
        rows, raw = self._explain(text, dict(fragment.params))
        total = self._partitions_total(fragment.tables)
        return plan_summary(rows, partitions_total=total, raw=raw)

    def _explain(
        self, statement: str, params: Mapping[str, Any]
    ) -> tuple[list[tuple[Any, ...]], str | None]:
        """Run ``EXPLAIN PLAN`` for ``statement`` and return its ``PLAN_TABLE`` rows.

        ``statement`` is a fragment the compiler built through the AST and the guard inspected; it
        is SQL, not a value, so it cannot be a bind parameter. The statement id is unique per call
        and the rows are deleted afterwards, so a session that never commits does not accumulate
        plans in the session GTT.
        """
        self._statement_seq += 1
        statement_id = f"mdqm_plan_{self._statement_seq}"
        raw: str | None = None
        with self._connection() as connection, connection.cursor() as cur:
            try:
                # `statement` is compiled-and-guarded SQL text, not a value; binds stay binds.
                cur.execute(  # noqa: S608  # nosemgrep: python-sql-injection
                    f"EXPLAIN PLAN SET STATEMENT_ID = '{statement_id}' FOR {statement}",
                    dict(params) if params else None,  # pyright: ignore[reportArgumentType]
                )
            except oracledb.DatabaseError as exc:
                if _is_no_privilege(exc):
                    raise NotSupported(
                        OPTIONAL_METHODS["explain"],
                        "this account may not write the plan table (EXPLAIN PLAN needs INSERT on "
                        f"PLAN_TABLE / the session GTT): {_message(exc)}",
                    ) from exc
                raise
            try:
                cur.execute(_PLAN_SQL, [statement_id])
                rows = [tuple(row) for row in cur.fetchall()]
                with contextlib.suppress(oracledb.DatabaseError):
                    cur.execute(_XPLAN_SQL, [statement_id])
                    raw = "\n".join(str(row[0]) for row in cur.fetchall())
                cur.execute("DELETE FROM plan_table WHERE statement_id = :1", [statement_id])
            finally:
                with contextlib.suppress(oracledb.DatabaseError):
                    connection.rollback()
        return rows, raw

    def _partitions_total(self, tables: Sequence[str]) -> int | None:
        """How many partitions the fragment's (single) table has, straight from the catalog."""
        if len(tables) != 1:
            return None
        bare = tables[0].split(".")[-1].upper()
        try:
            with self._connection() as connection, connection.cursor() as cur:
                cur.execute("SELECT count(*) FROM all_tab_partitions WHERE table_name = :1", [bare])
                row = cur.fetchone()
        except oracledb.DatabaseError:
            return None
        count = _as_int(None if row is None else row[0])
        # A table without a single partition row is not partitioned: nothing to report as a total.
        return count or None

    # --- SourceAdapter: execute / cancel -------------------------------------------------

    def execute(self, fragment: Fragment, limits: Limits) -> ArrowStream:
        """Run one compiled fragment and stream it as Arrow batches under ``limits``.

        ``Limits.timeout_s`` becomes ``Connection.call_timeout``, so the deadline is enforced by
        the server aborting the statement (``DPY-4024``) rather than by us noticing afterwards.
        ``fragment.params``, when present, must use Oracle's named placeholders (``:name``); the
        compiler inlines literals, so this is the exception, not the rule.
        """
        if fragment.sql is None:
            raise ValueError(
                f"the oracle adapter runs SQL; fragment kind {fragment.kind!r} is not supported"
            )
        pool = self.pool
        connection = pool.acquire()
        try:
            self._prepare(connection, limits.timeout_s)
            cursor = connection.cursor()
            cursor.arraysize = self._batch_size
            watchdog = _Deadline(connection, limits.timeout_s, connection.cancel)
            watchdog.arm()
            try:
                cursor.execute(fragment.sql, dict(fragment.params) if fragment.params else None)
            except oracledb.DatabaseError as exc:
                watchdog.disarm()
                raise _stream_error(
                    exc, cancelled=False, timed_out=watchdog.fired, timeout_s=limits.timeout_s
                ) from exc
            return OracleArrowStream(
                connection=connection,
                cursor=cursor,
                schema=_arrow_schema(cursor.description),
                limits=limits,
                deadline=self._monotonic() + limits.timeout_s,
                clock=self._monotonic,
                release=self._release,
                watchdog=watchdog,
                default_timeout_ms=self._call_timeout_ms,
                batch_size=self._batch_size,
            )
        except BaseException:
            with contextlib.suppress(oracledb.Error):
                connection.rollback()
            with contextlib.suppress(oracledb.Error):
                pool.release(connection)
            raise

    def cancel(self, handle: ArrowStream) -> None:
        """Abort the statement behind ``handle``. Idempotent; safe to call twice."""
        cancel = getattr(handle, "cancel", None)
        if callable(cancel):
            cancel()
        else:
            handle.close()

    def _release(self, connection: oracledb.Connection) -> None:
        """Return a connection to the pool, ignoring one the pool already closed."""
        pool = self._pool
        if pool is not None:
            with contextlib.suppress(oracledb.Error):
                pool.release(connection)

    # --- SourceAdapter: query store -------------------------------------------------------

    def mine_queries(self, since: datetime) -> Iterator[RawQuery]:
        """Statements this database has already run, from ``V$SQL``.

        ``V$SQL`` is one row per cached child cursor with *cumulative* counters, so each call
        reports what the cache holds for the window ``since`` -- a statement executed twice appears
        once with ``calls=2``. The view carries no executor: ``PARSING_SCHEMA_NAME`` is who parsed
        the statement, not who ran it, so ``user`` stays ``None`` rather than naming the wrong
        person.

        Raises :class:`NotSupported` with the ``query_store`` capability when this account may not
        read ``V$SQL`` -- never a crash, and never an empty list that looks like "this database
        runs nothing".
        """
        window = since.astimezone(UTC).replace(tzinfo=None)
        with self._connection() as connection, connection.cursor() as cur:
            try:
                cur.execute(_QUERY_STORE_SQL, [window])
                rows = cur.fetchall()
            except oracledb.DatabaseError as exc:
                if _is_no_privilege(exc):
                    raise NotSupported(
                        OPTIONAL_METHODS["mine_queries"],
                        f"this account may not read V$SQL: {_message(exc)}",
                    ) from exc
                raise
        return iter(_raw_queries(rows))

    def mutability_stats(self) -> list[TableStats]:
        """Not served: Oracle's change counters are statistics, not triggers.

        ``ALL_TAB_MODIFICATIONS`` is refreshed by ``DBMS_STATS``, and this adapter must not flush
        or gather statistics on somebody's primary database -- so between gathers a table that
        changed all night reads as unchanged.
        """
        raise NotSupported(OPTIONAL_METHODS["mutability_stats"], _MUTABILITY_UNSUPPORTED)


# --- pure helpers ---------------------------------------------------------------------------


def _partitioning(
    *,
    strategy: str,
    subpartitioning_type: str,
    interval: object,
    key: tuple[str, ...],
    partitions: Sequence[tuple[str, object, object]],
    table: str,
) -> Partitioning:
    """One :class:`Partitioning`, or :class:`NotSupported` for a shape we cannot describe."""
    if subpartitioning_type.upper() != "NONE":
        raise NotSupported(
            OPTIONAL_METHODS["extract_partitions"],
            f"{table} is composite-partitioned ({subpartitioning_type}); this adapter reports "
            "single-level partitions only, and inventing a scheme would let the planner prune on "
            "fiction",
        )
    kind = strategy.upper()
    if kind == "SYSTEM":
        return Partitioning(
            strategy=PartitionStrategy.EXPRESSION,
            expression="SYSTEM",
            granularity=interval_granularity(interval),
            bounds=_bounds(partitions),
        )
    mapped = {
        "RANGE": PartitionStrategy.RANGE,
        "LIST": PartitionStrategy.LIST,
        "HASH": PartitionStrategy.HASH,
    }.get(kind)
    if mapped is None:
        raise NotSupported(
            OPTIONAL_METHODS["extract_partitions"],
            f"{table} is partitioned with strategy {strategy!r}, which has no planner equivalent",
        )
    if mapped is PartitionStrategy.LIST:
        return Partitioning(
            strategy=mapped,
            key=key,
            bounds=_bounds(partitions, list_values=True),
        )
    return Partitioning(
        strategy=mapped,
        key=key,
        granularity=interval_granularity(interval),
        bounds=_bounds(partitions),
    )


def _bounds(
    partitions: Sequence[tuple[str, object, object]], *, list_values: bool = False
) -> tuple[PartitionBound, ...]:
    """``(name, high_value, num_rows)`` per partition -> half-open bounds in position order.

    A range partition's lower bound is the previous partition's upper bound; the first one is
    unbounded below (``None``). ``MAXVALUE`` is unbounded above, not a date to be guessed. For a
    LIST partition the ``HIGH_VALUE`` text is the value list, which is carried as ``values``.
    """
    out: list[PartitionBound] = []
    previous: datetime | None = None
    for name, high_value, num_rows in partitions:
        upper = parse_high_value(high_value)
        if list_values:
            values = _list_values(high_value)
            out.append(PartitionBound(name=name, values=values, row_estimate=_as_int(num_rows)))
            continue
        out.append(
            PartitionBound(
                name=name,
                lower=previous,
                upper=upper,
                row_estimate=_as_int(num_rows),
            )
        )
        previous = upper if upper is not None else previous
    return tuple(out)


def _list_values(high_value: object) -> tuple[str, ...]:
    """``'A', 'B'`` (a LIST partition's ``HIGH_VALUE``) -> ``('A', 'B')``."""
    if high_value is None:
        return ()
    text = str(high_value).strip()
    if text.upper() in {"DEFAULT", "MAXVALUE"}:
        return (text.upper(),)
    return tuple(chunk.strip().strip("'") for chunk in text.split(",") if chunk.strip())


def plan_summary(
    rows: Sequence[Sequence[Any]], *, partitions_total: int | None, raw: str | None = None
) -> PlanSummary:
    """Turn ``PLAN_TABLE`` rows into a :class:`~mdqm.models.PlanSummary`, with the pruning proof.

    ``rows`` are ``(id, parent_id, operation, options, object_name, cardinality, cost, pstart,
    pstop)`` in plan order. The proof is read from the partition operations only:

    * at least one partition operation, every ``PSTART``/``PSTOP`` numeric -> ``PROVEN``
    * at least one partition operation with ``PSTART = KEY`` -> ``UNPROVEN``
    * no partition operation -> ``UNKNOWN`` (nothing to prove; or not a partitioned table)

    ``partitions_scanned`` stays ``None``: Oracle's ``PSTART``/``PSTOP`` are internal partition
    numbers (a nine-partition interval table reported ``312``), so a count derived from them would
    be invented. The counts that *are* known are kept: the catalog's ``partitions_total``.
    """
    partition_rows = [row for row in rows if _is_partition_operation(row[2])]
    proof = PruningProof.UNKNOWN
    if partition_rows:
        markers = [
            (str(row[7]).strip().upper(), str(row[8]).strip().upper()) for row in partition_rows
        ]
        proof = (
            PruningProof.PROVEN
            if all(start.isdigit() and stop.isdigit() for start, stop in markers)
            else PruningProof.UNPROVEN
        )
    index_names = tuple(
        dict.fromkeys(
            str(row[4])
            for row in rows
            if row[4] is not None and str(row[2]).upper().startswith(("INDEX", "BITMAP INDEX"))
        )
    )
    root = next((row for row in rows if str(row[0]) == "0"), rows[0] if rows else None)
    return PlanSummary(
        partitions_scanned=None,
        partitions_total=partitions_total if partition_rows else None,
        uses_index=bool(index_names) or None,
        indexes_used=index_names,
        est_rows=_as_float(None if root is None else root[5]),
        est_cost=_as_float(None if root is None else root[6]),
        pruning_proof=proof,
        raw=raw or None,
    )


def _is_partition_operation(operation: object) -> bool:
    return str(operation).upper().startswith("PARTITION")


def _raw_queries(rows: Sequence[Sequence[Any]]) -> list[RawQuery]:
    """``V$SQL`` rows -> :class:`RawQuery`; counters are cumulative, so calls are reported."""
    out: list[RawQuery] = []
    for text, executions, elapsed, rows_processed, last_active in rows:
        body = _as_text(text)
        if not body:
            continue
        calls = _as_int(executions) or 0
        total_ms = None if elapsed is None else float(elapsed) / 1000.0
        out.append(
            RawQuery(
                text=body,
                dialect="oracle",
                calls=calls,
                total_time_ms=total_ms,
                mean_time_ms=None if total_ms is None or calls == 0 else total_ms / calls,
                rows=_as_int(rows_processed),
                last_seen=None if last_active is None else _as_utc(last_active),
            )
        )
    return out


def _as_utc(value: object) -> datetime | None:
    """A naive ``LAST_ACTIVE_TIME`` is UTC (the session runs in UTC), an aware one is converted."""
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _native_type(data_type: str, precision: int | None, scale: int | None) -> str:
    """Oracle's own type text, with the number precision the catalog keeps separately."""
    if data_type.upper().startswith("NUMBER") and precision is not None:
        return f"NUMBER({precision},{scale if scale is not None else 0})"
    return data_type


def _arrow_schema(description: Sequence[Any] | None) -> pa.Schema:
    """The stream's schema, known before the first row arrives (a cursor describes itself)."""
    fields = [
        pa.field(
            str(entry[0]),
            arrow_type(entry[1], _as_int(entry[4]), _as_int(entry[5])),
            nullable=entry[6] is not False,
        )
        for entry in description or ()
    ]
    return pa.schema(fields)


def _arrays_from_rows(schema: pa.Schema, rows: Sequence[Sequence[Any]]) -> list[pa.Array]:
    """One Arrow array per field, converting each cell to the schema's declared type."""
    arrays: list[pa.Array] = []
    for position, field in enumerate(schema):
        column = [row[position] for row in rows]
        if pa.types.is_string(field.type):
            arrays.append(pa.array([_as_text(value) for value in column], type=field.type))
        elif pa.types.is_binary(field.type):
            arrays.append(pa.array([_as_bytes(value) for value in column], type=field.type))
        else:
            arrays.append(pa.array(column, type=field.type))
    return arrays


def _as_text(value: object) -> str | None:
    """Render a value for a column carried as text (CLOB, JSON, XMLTYPE, ROWID)."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, oracledb.LOB):
        return str(value.read())
    if isinstance(value, bytes | bytearray | memoryview):
        return bytes(value).decode("utf-8", "replace")
    return str(value)


def _as_bytes(value: object) -> bytes | None:
    """Render a value for a binary column (RAW/BLOB); a LOB is read in full."""
    if value is None or isinstance(value, bytes):
        return value
    if isinstance(value, oracledb.LOB):
        data = value.read()
        return data if isinstance(data, bytes) else str(data).encode("utf-8")
    if isinstance(value, bytearray | memoryview):
        return bytes(value)
    return str(value).encode("utf-8")


def _stream_error(
    exc: oracledb.DatabaseError, *, cancelled: bool, timed_out: bool, timeout_s: float
) -> Exception:
    """Translate a driver error: a cancellation or a missed deadline, otherwise the raw error.

    ``cancelled`` is a caller who asked for the cancellation (``OracleCancelled``); ``timed_out`` is
    this execution's watchdog, which also cancels but means the deadline (``OracleTimeout``).
    """
    code = _code(exc)
    message = _message(exc)
    if code == _CANCELLED:
        if cancelled:
            return OracleCancelled("the Oracle statement was cancelled by the caller")
        return OracleTimeout(
            f"the Oracle statement exceeded the {timeout_s}s timeout and was cancelled by the "
            f"adapter: {message}"
        )
    if _CALL_TIMEOUT_MARKER in message:
        return OracleTimeout(f"the Oracle statement exceeded the {timeout_s}s timeout: {message}")
    if _CONNECTION_LOST_MARKER in message and timed_out:
        # The backstop call_timeout fired and the driver could not drain the response, so the
        # session is gone. It is still a deadline, and saying so beats a raw driver error.
        return OracleTimeout(
            f"the Oracle statement exceeded the {timeout_s}s timeout and the connection was lost "
            f"while recovering: {message}"
        )
    return exc


def _code(exc: BaseException) -> int:
    """Oracle's numeric error code, or 0 (a thin-mode client error carries none)."""
    if isinstance(exc, oracledb.DatabaseError) and exc.args:
        code = getattr(exc.args[0], "code", None)
        return int(code) if isinstance(code, int) else 0
    return 0


def _message(exc: BaseException) -> str:
    """One line: the driver's own message without the documentation link Oracle appends."""
    if isinstance(exc, oracledb.DatabaseError) and exc.args:
        text = str(getattr(exc.args[0], "message", exc))
    else:
        text = str(exc)
    return text.split("\n")[0].strip()


def _is_no_privilege(exc: object) -> bool:
    """True when the driver error says this account may not read the object."""
    return isinstance(exc, oracledb.DatabaseError) and _code(exc) in _NO_PRIVILEGE


def _query_store_state(cur: oracledb.Cursor) -> tuple[bool, str]:
    """Ask ``V$SQL`` whether it is readable, without letting a refusal escape."""
    try:
        cur.execute("SELECT count(*) FROM v$sql")
        row = cur.fetchone()
    except oracledb.DatabaseError as exc:
        if _is_no_privilege(exc):
            return False, f"this account may not read V$SQL: {_message(exc)}"
        return False, f"V$SQL could not be read: {_message(exc)}"
    count = 0 if row is None else int(row[0])
    return True, f"V$SQL is readable ({count} cached statements)"


def _as_int(value: object) -> int | None:
    """``int`` or ``None``: a catalog counter is NULL for a table no stats gather has touched."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    """``float`` or ``None``: a plan may omit a cost, and a missing number is not zero."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _positive_int(value: object, default: int, key: str) -> int:
    """Read a positive integer option, naming the key instead of raising a bare ``ValueError``."""
    if value is None:
        return default
    try:
        parsed = int(str(value))
    except ValueError:
        raise OracleConfigError(
            f"options.{key} must be a positive integer, got {value!r}"
        ) from None
    if parsed <= 0:
        raise OracleConfigError(f"options.{key} must be a positive integer, got {value!r}")
    return parsed


def _connection_parts(config: SourceConfig, secrets: SecretProvider | None) -> tuple[str, str, str]:
    """Build ``(dsn, user, password)``, resolving the secret reference.

    The password lives only in the pool's construction, which this module never logs; the config
    carries a *reference* (``SourceConfig.secret``), never the value.
    """
    options = dict(config.options or {})
    user = str(options.get("user") or "").strip()
    if not user:
        raise OracleConfigError(
            f"source {config.name!r}: no database user. Set options.user; the server needs a "
            "per-person identity, not a shared login."
        )
    dsn = str(options.get("dsn") or "").strip()
    if not dsn:
        host = str(options.get("host") or "").strip()
        service = str(options.get("service_name") or "").strip()
        if not host or not service:
            raise OracleConfigError(
                f"source {config.name!r}: set options.host and options.service_name "
                "(or a whole options.dsn such as host:1521/service)"
            )
        port = _positive_int(options.get("port"), DEFAULT_PORT, "port")
        dsn = f"{host}:{port}/{service}"
    if config.secret is not None:
        if secrets is None:
            raise OracleConfigError(
                f"source {config.name!r} references secret {config.secret!r} but no secret "
                "provider is configured; without one the password cannot be resolved"
            )
        password = secrets.get(config.secret)
    else:
        raise OracleConfigError(
            f"source {config.name!r}: no secret reference. Set `secret: <name>`; a password in "
            "config would be a password in git."
        )
    return dsn, user, password
