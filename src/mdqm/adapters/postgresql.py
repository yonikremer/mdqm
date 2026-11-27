"""The PostgreSQL source adapter (task A02): connect, catalog, execute, cancel.

PostgreSQL is the MVP's primary target and the first source a weak agent ever gets wrong, so this
module is written as the reference for every other vendor adapter:

* **Read-only by construction.** Every pooled connection is configured with ``SET SESSION
  CHARACTERISTICS AS TRANSACTION READ ONLY``; nothing here has a code path that can modify a
  source. The AST guard is a *different* layer (``mdqm.sql.guard``) and the adapter never relies on
  it: a write sent straight to :meth:`PostgresAdapter.execute` is still refused by the server, and
  the integration test proves that with the guard deliberately bypassed.
* **Streaming, not buffering.** :meth:`PostgresAdapter.execute` opens a *server-side* cursor and
  reads ``fetchmany`` batches into Arrow, so ``Limits.max_rows``/``max_bytes`` stop the scan while
  it is still running and ``cancel`` has a live query to interrupt. A primary database with no
  replica never gets asked for a result set nobody will read.
* **Deterministic time.** ``exec_ts`` arrives inside the fragment and is never substituted here;
  the session runs in UTC (``SET TimeZone = 'UTC'``) so a naive timestamp literal added by another
  lane cannot be read in the host's zone.
* **Nothing silent.** An unknown ``options`` key is logged by name; a numeric column whose scale
  does not fit the declared Arrow decimal raises instead of rounding money away; a capability that
  a later task owns is reported as unprobed with the reason, never as "supported".

Configuration lives in :class:`~mdqm.config.settings.SourceConfig`: the password is a *secret
reference* (``SourceConfig.secret``) resolved through a
:class:`~mdqm.config.secrets.SecretProvider` at connect time, and the location comes from
``options``::

    sources:
      billing:
        adapter: postgresql
        secret: billing_ro                       # a reference, never the value
        options: { host: pg.internal, port: 5432, dbname: billing, user: mdqm_ro }
        # or a whole connection string without a password:
        # options: { dsn: "postgresql://mdqm_ro@pg.internal:5432/billing?sslmode=require" }

Usage::

    adapter = PostgresAdapter(source_config, env_secret_provider)
    stream = adapter.execute(fragment, Limits(timeout_s=5, max_rows=1000, max_bytes=4_000_000,
                                              max_partitions=8))
    for batch in stream:                     # pyarrow.RecordBatch, lazily
        ...
    stream.close()

"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, Final

import psycopg
import psycopg_pool
import pyarrow as pa
from psycopg import conninfo

from mdqm.adapters.base import OPTIONAL_METHODS, ArrowStream, NotSupported
from mdqm.adapters.postgresql_plan import (
    PARTITIONS_SQL,
    parse_partitions,
    parse_plan,
    row_estimate,
    split_names,
)
from mdqm.adapters.postgresql_stats import (
    MUTABILITY_CAPABILITY,
    MUTABILITY_SQL,
    QueryStoreReader,
    mutability_from_rows,
    query_store_state,
)
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
    Partitioning,
    PlanSummary,
    RawQuery,
    Table,
    TableKind,
    TableStats,
)
from mdqm.sql.execution import PreparedStatement, execute_one, prepare

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_POOL_SIZE",
    "DEFAULT_STATEMENT_TIMEOUT_MS",
    "NUMERIC_SCALE",
    "PgArrowStream",
    "PostgresAdapter",
    "PostgresCancelled",
    "PostgresConfigError",
    "PostgresTimeout",
]

_log = logging.getLogger(__name__)

#: Rows asked of the server cursor per ``FETCH``. Small on purpose: the row/byte cap must be able
#: to stop mid-scan instead of after one huge batch has already crossed the wire.
DEFAULT_BATCH_SIZE: Final = 1024

#: Connections the pool keeps at hand. The governor already limits concurrency per source, so this
#: is a ceiling for parallel reads, not a target.
DEFAULT_POOL_SIZE: Final = 4

#: Session-level deadline, in milliseconds. It is only a safety net for a statement that reaches
#: PostgreSQL without :class:`~mdqm.models.Limits` (a probe, a catalog read); every ``execute``
#: narrows it to ``Limits.timeout_s`` with ``SET LOCAL``.
DEFAULT_STATEMENT_TIMEOUT_MS: Final = 60_000

#: Fractional digits declared for a ``numeric`` column in Arrow. Arrow decimals are fixed-scale and
#: PostgreSQL's ``numeric`` is not, so a value with more digits than this raises ``ArrowInvalid``
#: rather than being rounded: losing a digit of money silently is worse than a loud failure.
#:
#: 18 is not arbitrary. ``AVG(numeric(14,2))`` is the canonical query of this project, and
#: PostgreSQL returns it with scale 16+ (e.g. ``29731.68654173702905``); at scale 9 every average
#: failed to convert. A value that still exceeds 18 digits keeps failing loudly, which is the point.
NUMERIC_SCALE: Final = 18

_CAPABILITY_PARTITIONS: Final = "partition_metadata"

#: Why the optional features this adapter does not serve are reported *unprobed* (``None``) rather
#: than unsupported. PostgreSQL can do all of them; the code that reads them belongs to another
#: task, and saying "this source cannot" would be a lie the capability report repeats.
_NOT_YET_QUERIES: Final = "query-store mining is task A04 (adapters/postgresql_stats.py)"
_NOT_YET_STATS: Final = "mutability counters are task A04 (adapters/postgresql_stats.py)"

#: Why ``explain_plan`` stays unprobed. The plan and its partition counts are served (A03:
#: :mod:`mdqm.adapters.postgresql_plan`), but the pruning *verdict* needs the window a spec intends,
#: which an adapter cannot see; the adapter never decides whether a scan is acceptable.
_UNPROBED_EXPLAIN: Final = (
    "EXPLAIN counts are parsed (adapters/postgresql_plan.py); the pruning verdict needs the "
    "spec's window, which the planner owns"
)


def _partition_children(cursor: psycopg.Cursor[Any]) -> dict[str, str]:
    """Partition relation name -> its parent's name, for this whole database.

    ``EXPLAIN`` names the partitions it scans, so the plan is the source of truth for *scanned*;
    this map supplies the other half, how many partitions the scanned table has in total, and which
    other partitioned tables the plan happened to touch (a join reads two of them).
    """
    rows = cursor.execute(
        """
        SELECT c.relname, p.relname
        FROM pg_inherits i
        JOIN pg_class c ON c.oid = i.inhrelid
        JOIN pg_class p ON p.oid = i.inhparent
        ORDER BY c.relname
        """
    ).fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def _as_int(value: object) -> int | None:
    """``int`` or ``None``: counters are NULL for never-used tables, and a value may be text."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # pi-lens-ignore: unchecked-numeric-parse-python -- a float by the branch above
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _as_float(value: object) -> float | None:
    """``float`` or ``None``: a plan may omit a cost, and a missing number is not zero."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        # pi-lens-ignore: unchecked-numeric-parse-python -- a number by the branch above
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


_KNOWN_OPTIONS: Final = frozenset(
    {
        "dsn",
        "host",
        "port",
        "dbname",
        "database",
        "user",
        "sslmode",
        "connect_timeout",
        "statement_timeout_ms",
        "pool_size",
    }
)

#: ``options`` key -> libpq connection parameter.
_CONNINFO_KEYS: Final = {
    "host": "host",
    "port": "port",
    "dbname": "dbname",
    "database": "dbname",
    "user": "user",
    "sslmode": "sslmode",
    "connect_timeout": "connect_timeout",
}


class PostgresConfigError(ValueError):
    """The source cannot be connected with this configuration.

    The message names the source and the offending ``options`` key, never a credential and never
    the resolved connection string, so it is safe to log and to show to an operator.
    """


class PostgresTimeout(TimeoutError):
    """The statement exceeded ``Limits.timeout_s`` (client deadline or server statement timeout).

    A ``TimeoutError``, never a :class:`~mdqm.adapters.base.NotSupported`: a deadline is an answer
    about this execution, not about what the source can do.
    """


class PostgresCancelled(RuntimeError):
    """The statement was aborted by :meth:`PostgresAdapter.cancel` after the server acknowledged."""


# --- catalog/normalisation ------------------------------------------------------------------


def _normalized_type(name: str, type_type: str, category: str, dimensions: int) -> NormalizedType:
    """Map a ``pg_attribute``/``pg_type`` pair onto the project's portable vocabulary.

    ``name`` is ``pg_type.typname`` (not the formatted type): an array column reports ``_int4``,
    an enum reports its own type name. ``type_type`` is ``b`` (base), ``e`` (enum), ``d``
    (domain), ``c`` (composite). An enum is a string to every consumer that matters, so it is
    normalised as one instead of falling into ``OTHER`` and breaking every value map.
    """
    if dimensions > 0 or name.startswith("_") or category == "A":
        return NormalizedType.ARRAY
    known = _NORMALIZED_BY_NAME.get(name)
    if known is not None:
        return known
    if type_type == "e":
        return NormalizedType.STRING
    return _NORMALIZED_BY_CATEGORY.get(category, NormalizedType.OTHER)


_NORMALIZED_BY_NAME: Final[dict[str, NormalizedType]] = {
    "bool": NormalizedType.BOOLEAN,
    "int2": NormalizedType.INTEGER,
    "int4": NormalizedType.INTEGER,
    "int8": NormalizedType.INTEGER,
    "oid": NormalizedType.INTEGER,
    "numeric": NormalizedType.DECIMAL,
    "money": NormalizedType.DECIMAL,
    "float4": NormalizedType.FLOAT,
    "float8": NormalizedType.FLOAT,
    "date": NormalizedType.DATE,
    "timestamp": NormalizedType.TIMESTAMP,
    "timestamptz": NormalizedType.TIMESTAMP,
    "time": NormalizedType.TIME,
    "timetz": NormalizedType.TIME,
    "interval": NormalizedType.TIME,
    "bytea": NormalizedType.BINARY,
    "json": NormalizedType.JSON,
    "jsonb": NormalizedType.JSON,
    "uuid": NormalizedType.UUID,
    "char": NormalizedType.STRING,
    "bpchar": NormalizedType.STRING,
    "varchar": NormalizedType.STRING,
    "text": NormalizedType.STRING,
    "name": NormalizedType.STRING,
    "citext": NormalizedType.STRING,
    "xml": NormalizedType.STRING,
    "bit": NormalizedType.STRING,
    "varbit": NormalizedType.STRING,
}

_NORMALIZED_BY_CATEGORY: Final[dict[str, NormalizedType]] = {
    "S": NormalizedType.STRING,
    "N": NormalizedType.DECIMAL,
    "D": NormalizedType.DATE,
    "B": NormalizedType.BOOLEAN,
    "V": NormalizedType.STRING,
}

#: PostgreSQL type OID -> Arrow type. OIDs are stable across releases; anything else is text.
_ARROW_TYPES: Final[dict[int, pa.DataType]] = {
    16: pa.bool_(),
    17: pa.binary(),
    20: pa.int64(),
    21: pa.int16(),
    23: pa.int32(),
    26: pa.uint32(),
    700: pa.float32(),
    701: pa.float64(),
    1082: pa.date32(),
    1083: pa.time64("us"),
    1114: pa.timestamp("us"),
    1184: pa.timestamp("us", tz="UTC"),
    1700: pa.decimal128(38, NUMERIC_SCALE),
}

_TABLE_KINDS: Final[dict[str, TableKind]] = {
    "r": TableKind.TABLE,
    "p": TableKind.PARTITIONED_TABLE,
    "v": TableKind.VIEW,
    "m": TableKind.MATERIALIZED_VIEW,
    "f": TableKind.OTHER,
}

# --- SQL ------------------------------------------------------------------------------------

# Partition children are excluded: they are how the parent is stored, not tables an agent should
# choose between. `extract_partitions` reports them, which is where the planner wants them.
# A partition's indexes and keys are inherited from the parent, so listing the child would repeat
# the parent's own PK once per month.
_CATALOG_TABLES_SQL: Final = """
SELECT n.nspname  AS schema_name,
       c.relname  AS table_name,
       c.relkind  AS relkind,
       c.reltuples::bigint AS row_estimate,
       obj_description(c.oid, 'pg_class') AS comment
  FROM pg_class c
  JOIN pg_namespace n ON n.oid = c.relnamespace
 WHERE c.relkind IN ('r', 'p', 'v', 'm', 'f')
   AND n.nspname NOT IN ('pg_catalog', 'information_schema')
   AND n.nspname NOT LIKE 'pg\\_toast%'
   AND n.nspname NOT LIKE 'pg\\_temp%'
   AND NOT EXISTS (SELECT 1 FROM pg_inherits i WHERE i.inhrelid = c.oid)
 ORDER BY n.nspname, c.relname
"""

_CATALOG_COLUMNS_SQL: Final = """
SELECT n.nspname AS schema_name,
       c.relname AS table_name,
       a.attname AS column_name,
       format_type(a.atttypid, a.atttypmod) AS native_type,
       coalesce(ty.typname, '') AS type_name,
       coalesce(ty.typtype, '') AS type_type,
       coalesce(ty.typcategory, '') AS type_category,
       a.attndims AS dimensions,
       a.attnotnull AS not_null,
       a.attnum   AS ordinal,
       col_description(c.oid, a.attnum) AS comment
  FROM pg_class c
  JOIN pg_namespace n ON n.oid = c.relnamespace
  JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
  LEFT JOIN pg_type ty ON ty.oid = a.atttypid
 WHERE c.relkind IN ('r', 'p', 'v', 'm', 'f')
   AND n.nspname NOT IN ('pg_catalog', 'information_schema')
   AND n.nspname NOT LIKE 'pg\\_toast%'
   AND n.nspname NOT LIKE 'pg\\_temp%'
   AND NOT EXISTS (SELECT 1 FROM pg_inherits i WHERE i.inhrelid = c.oid)
 ORDER BY n.nspname, c.relname, a.attnum
"""

# `indkey` holds the *key* columns first and then the INCLUDE columns; `indnkeyatts` separates them,
# so a covering index does not pretend its payload columns are indexed.
_CATALOG_INDEXES_SQL: Final = """
SELECT n.nspname AS schema_name,
       c.relname AS table_name,
       ic.relname AS index_name,
       i.indisunique AS is_unique,
       i.indisprimary AS is_primary,
       am.amname AS index_type,
       pg_get_expr(i.indpred, i.indrelid) AS predicate,
       (SELECT string_agg(a.attname, ',' ORDER BY k.ord)
          FROM unnest(i.indkey) WITH ORDINALITY AS k(attnum, ord)
          JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = k.attnum
         WHERE k.ord <= i.indnkeyatts) AS columns
  FROM pg_index i
  JOIN pg_class c  ON c.oid = i.indrelid
  JOIN pg_class ic ON ic.oid = i.indexrelid
  JOIN pg_namespace n ON n.oid = c.relnamespace
  JOIN pg_am am ON am.oid = ic.relam
 WHERE c.relkind IN ('r', 'p', 'm')
   AND NOT c.relispartition
   AND n.nspname NOT IN ('pg_catalog', 'information_schema')
 ORDER BY n.nspname, c.relname, ic.relname
"""

_CATALOG_FOREIGN_KEYS_SQL: Final = """
SELECT n.nspname AS schema_name,
       c.relname AS table_name,
       con.conname AS constraint_name,
       rn.nspname AS ref_schema,
       rc.relname AS ref_table,
       (SELECT string_agg(a.attname, ',' ORDER BY k.ord)
          FROM unnest(con.conkey) WITH ORDINALITY AS k(attnum, ord)
          JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = k.attnum) AS columns,
       (SELECT string_agg(a.attname, ',' ORDER BY k.ord)
          FROM unnest(con.confkey) WITH ORDINALITY AS k(attnum, ord)
          JOIN pg_attribute a ON a.attrelid = rc.oid AND a.attnum = k.attnum) AS ref_columns
  FROM pg_constraint con
  JOIN pg_class c ON c.oid = con.conrelid
  JOIN pg_namespace n ON n.oid = c.relnamespace
  JOIN pg_class rc ON rc.oid = con.confrelid
  JOIN pg_namespace rn ON rn.oid = rc.relnamespace
 WHERE con.contype = 'f'
   AND NOT c.relispartition
   AND n.nspname NOT IN ('pg_catalog', 'information_schema')
 ORDER BY n.nspname, c.relname, con.conname
"""


# --- Arrow conversion -----------------------------------------------------------------------


def _arrow_type(type_code: int) -> pa.DataType:
    """Arrow type for a PostgreSQL result-column OID; anything unknown is carried as text."""
    return _ARROW_TYPES.get(type_code, pa.string())


def _arrow_schema(description: Sequence[psycopg.Column] | None) -> pa.Schema:
    """The stream's schema, known before the first row arrives (a cursor describes itself)."""
    fields = [pa.field(column.name, _arrow_type(column.type_code)) for column in description or ()]
    return pa.schema(fields)


def _as_text(value: object) -> str | None:
    """Render a value for a column carried as text (jsonb, uuid, arrays, ranges, intervals).

    psycopg hands back Python objects for the types it understands, while Arrow needs one text
    form. JSON is dumped in a stable key order so the same document always renders identically.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(dict(value), ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False, default=str)
    return str(value)


def _arrays_from_rows(schema: pa.Schema, rows: Sequence[Sequence[Any]]) -> list[pa.Array]:
    """One Arrow array per field, converting each cell to the schema's declared type."""
    arrays: list[pa.Array] = []
    for position, field in enumerate(schema):
        column = [row[position] for row in rows]
        if pa.types.is_string(field.type):
            arrays.append(pa.array([_as_text(value) for value in column], type=field.type))
        else:
            arrays.append(pa.array(column, type=field.type))
    return arrays


# --- the stream -----------------------------------------------------------------------------


class PgArrowStream:
    """The :class:`~mdqm.adapters.base.ArrowStream` a PostgreSQL ``execute`` returns.

    It owns one pooled connection until :meth:`close`. Batches are pulled from a *server-side*
    cursor, so the row and byte caps stop the scan instead of trimming a buffer that already
    crossed the wire, and the client deadline is checked between batches in addition to the
    server's own ``statement_timeout``.
    """

    def __init__(
        self,
        *,
        connection: psycopg.Connection[Any],
        cursor: psycopg.ServerCursor[Any],
        schema: pa.Schema,
        limits: Limits,
        deadline: float,
        clock: Callable[[], float],
        release: Callable[[psycopg.Connection[Any]], None],
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
        """Ask the server for the next batch, translating a cancellation into our own error."""
        try:
            return self._cursor.fetchmany(self._batch_size)
        except psycopg.errors.QueryCanceled as exc:
            if self._cancelled:
                raise PostgresCancelled(
                    f"the PostgreSQL statement was cancelled (timeout_s={self._timeout_s})"
                ) from exc
            raise PostgresTimeout(
                f"the PostgreSQL statement exceeded the {self._timeout_s}s timeout: {exc}"
            ) from exc

    def _check_deadline(self) -> None:
        if self._clock() > self._deadline:
            raise PostgresTimeout(
                f"the PostgreSQL statement exceeded the {self._timeout_s}s timeout while streaming"
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
        """Ask PostgreSQL to abort the running statement. Idempotent and thread-safe.

        Only the cancel request is sent: the reader (or :meth:`close`) still releases the cursor and
        the connection, so cancelling from another thread cannot close the socket underneath a
        fetch that is in flight.
        """
        if self._closed or self._cancelled:
            return
        self._cancelled = True
        with contextlib.suppress(psycopg.Error):
            # A cancel request needs a second, throwaway connection to the same backend; psycopg
            # does that for us. An already-finished statement makes the request a no-op.
            self._connection.cancel()

    def close(self) -> None:
        """Release cursor, transaction and connection. Idempotent."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(psycopg.Error):
            self._cursor.close()
        with contextlib.suppress(psycopg.Error):
            # The transaction is aborted after a timeout/cancel; without the rollback the pool would
            # hand the next caller a connection it cannot use.
            self._connection.rollback()
        with contextlib.suppress(psycopg.Error, psycopg_pool.PoolClosed):
            self._release(self._connection)


# --- the adapter ----------------------------------------------------------------------------


class PostgresAdapter:
    """A pooled, streaming, read-only ``SourceAdapter`` for one PostgreSQL database.

    The pool is built lazily, so an adapter for an unreachable source can be constructed (and
    ``mdqm probe`` can report the failure) without a connection attempt at import or wiring time.
    Every connection it hands out is configured read-only with a session ``statement_timeout`` and
    ``application_name = mdqm:<user>``, which is how a DBA sees whose query is on the primary.
    """

    kind: ClassVar[str] = "postgresql"

    def __init__(
        self,
        config: SourceConfig,
        secrets: SecretProvider | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        """Build the adapter from its source config; connect on first use.

        ``secrets`` resolves ``config.secret`` to the password. It is optional so a peer-auth or
        PGPASS-based deployment works, and the password is never stored on the instance.
        """
        if config.adapter != self.kind:
            raise PostgresConfigError(
                f"config names adapter {config.adapter!r} but this is the {self.kind!r} adapter"
            )
        options = dict(config.options or {})
        for unknown in sorted(set(options) - _KNOWN_OPTIONS):
            _log.warning("source %s: ignoring unknown postgresql option %r", config.name, unknown)
        self._source = config.name
        self._conninfo, self._user = _conninfo(config, secrets)
        self._pool_size = _positive_int(options.get("pool_size"), DEFAULT_POOL_SIZE, "pool_size")
        self._statement_timeout_ms = _positive_int(
            options.get("statement_timeout_ms"),
            DEFAULT_STATEMENT_TIMEOUT_MS,
            "statement_timeout_ms",
        )
        # pi-lens-ignore: unchecked-numeric-parse-python -- _positive_int returned an int
        self._connect_wait_s = float(self._statement_timeout_ms) / 1000.0
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic
        self._pool: psycopg_pool.ConnectionPool[psycopg.Connection[Any]] | None = None
        self._cursor_seq = 0
        #: pg_stat_statements keeps cumulative counters; this holds the previous reading.
        self._query_store = QueryStoreReader()

    def __repr__(self) -> str:
        """Name the source and the user, never the connection string (it may hold a password)."""
        return f"PostgresAdapter(source={self._source!r}, user={self._user!r})"

    # --- SourceAdapter: identity ---------------------------------------------------------

    def dialect(self) -> str:
        """``postgres``: the sqlglot dialect the compiler emits for this source."""
        return "postgres"

    @property
    def source(self) -> str:
        """The configured source name; the catalog and the probe both report it."""
        return self._source

    # --- lifecycle ------------------------------------------------------------------------

    @property
    def pool(self) -> psycopg_pool.ConnectionPool:
        """The connection pool, opened on first access."""
        if self._pool is None:
            pool = psycopg_pool.ConnectionPool[psycopg.Connection[Any]](
                self._conninfo,
                min_size=0,
                max_size=self._pool_size,
                configure=self._configure,
                kwargs={"application_name": f"mdqm:{self._user}"},
                name=f"mdqm-{self._source}",
                open=False,
            )
            pool.open(wait=False)
            self._pool = pool
        return self._pool

    def _configure(self, connection: psycopg.Connection[Any]) -> None:
        """Apply the session contract to every new connection, before anyone can use it."""
        with connection.cursor() as cur:
            cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
            # `SET` takes no bind parameter, but `set_config` does: the value travels as a real
            # parameter, so this statement has no interpolation at all.
            cur.execute(
                "SELECT set_config('statement_timeout', %s, false)",
                (str(self._statement_timeout_ms),),
            )
            cur.execute(
                "SELECT set_config('idle_in_transaction_session_timeout', %s, false)",
                (str(self._statement_timeout_ms * 2),),
            )
            # A naive timestamp literal must never be read in the host's zone: exec_ts is UTC.
            cur.execute("SET TimeZone = 'UTC'")
        connection.commit()

    def close(self) -> None:
        """Close the pool. Idempotent."""
        pool = self._pool
        self._pool = None
        if pool is not None:
            pool.close()

    @contextlib.contextmanager
    def _connection(self) -> Iterator[psycopg.Connection[Any]]:
        """A pooled connection for a probe/catalog read; committed and returned on exit."""
        with self.pool.connection() as connection:
            yield connection

    # --- SourceAdapter: probe ------------------------------------------------------------

    def probe(self) -> Capabilities:
        """Detect what this server supports, reporting only what this adapter can actually serve."""
        with self._connection() as connection, connection.cursor() as cur:
            cur.execute("SELECT version()")
            row = cur.fetchone()
            version = _short_version("" if row is None else str(row[0]))
            cur.execute("SELECT to_regclass('pg_catalog.pg_partitioned_table') IS NOT NULL")
            partitioned = cur.fetchone()
            can_read_partitions = bool(partitioned and partitioned[0])
            queries = query_store_state(cur)
        partitions_supported, partitions_detail = self._probe_partitions(can_read_partitions)
        capabilities = {
            _CAPABILITY_PARTITIONS: Capability(
                supported=partitions_supported, detail=partitions_detail
            ),
            "explain_plan": Capability(detail=_UNPROBED_EXPLAIN),
            # Claimed only when the extension is installed *and* this role may select from it:
            # the conformance kit requires a claimed capability's method not to raise.
            OPTIONAL_METHODS["mine_queries"]: Capability(
                supported=queries.supported, detail=queries.detail
            ),
            MUTABILITY_CAPABILITY: Capability(
                supported=True, detail="pg_stat_user_tables (n_tup_ins/upd/del/hot_upd)"
            ),
        }
        return Capabilities(
            source=self._source,
            adapter=self.kind,
            version=version,
            retrieved_at=self._clock(),
            capabilities=capabilities,
        )

    def _probe_partitions(self, has_catalog: bool) -> tuple[bool, str]:
        """Claim ``partition_metadata`` only if extraction really returns metadata.

        A catalog table that exists says nothing about whether *our* reading of it works: an
        unsupported partitioning shape raises :class:`NotSupported`, and a permission problem
        raises a driver error. Both mean the planner must take its fallback, so both are reported
        as unsupported with the reason, never as a claim the rest of the system then relies on.
        """
        if not has_catalog:
            return False, "this server has no declarative partitioning catalog (pre-10)"
        try:
            found = self.extract_partitions()
        except NotSupported as exc:
            return False, exc.detail or exc.capability
        except psycopg.Error as exc:
            return False, f"the partition catalog could not be read: {exc.diag.message_primary}"
        return True, f"pg_partitioned_table + pg_inherits: {len(found)} partitioned table(s) read"

    # --- SourceAdapter: catalog ----------------------------------------------------------

    def extract_catalog(self) -> CatalogSnapshot:
        """Tables, columns, types, keys and indexes from ``pg_catalog``.

        Partition children are deliberately not listed: they are how their parent is stored, they
        inherit the parent's primary key and indexes, and asking a weak agent to choose between
        ``usage_events`` and ``usage_events_p2025_12`` is a way to scan one month and call it a
        year. :meth:`extract_partitions` reports the children where the planner needs them.
        """
        with self._connection() as connection, connection.cursor() as cur:
            cur.execute(_CATALOG_TABLES_SQL)  # pi-lens-ignore: module-constant SQL
            table_rows = cur.fetchall()
            cur.execute(_CATALOG_COLUMNS_SQL)
            column_rows = cur.fetchall()
            cur.execute(_CATALOG_INDEXES_SQL)
            index_rows = cur.fetchall()
            cur.execute(_CATALOG_FOREIGN_KEYS_SQL)
            foreign_key_rows = cur.fetchall()

        columns: dict[tuple[str, str], list[Column]] = {}
        for row in column_rows:
            key = (str(row[0]), str(row[1]))
            columns.setdefault(key, []).append(
                Column(
                    name=str(row[2]),
                    native_type=str(row[3]),
                    normalized_type=_normalized_type(
                        str(row[4]),
                        str(row[5]),
                        str(row[6]),
                        # attndims is NOT NULL; 0 is also what "not an array" means.
                        _as_int(row[7]) or 0,
                    ),
                    nullable=not bool(row[8]),
                    ordinal=_as_int(row[9]),
                    comment=_text(row[10]),
                )
            )

        indexes: dict[tuple[str, str], list[Index]] = {}
        for row in index_rows:
            key = (str(row[0]), str(row[1]))
            names = split_names(row[7])
            if not names:
                continue  # an expression-only index has no column list to report
            indexes.setdefault(key, []).append(
                Index(
                    name=str(row[2]),
                    columns=names,
                    unique=bool(row[3]),
                    primary=bool(row[4]),
                    index_type=_text(row[5]),
                    predicate=_text(row[6]),
                )
            )

        foreign_keys: dict[tuple[str, str], list[ForeignKey]] = {}
        for row in foreign_key_rows:
            local = split_names(row[5])
            referenced = split_names(row[6])
            if not local or len(local) != len(referenced):
                continue
            key = (str(row[0]), str(row[1]))
            foreign_keys.setdefault(key, []).append(
                ForeignKey(
                    columns=local,
                    ref_table=f"{row[3]}.{row[4]}",
                    ref_columns=referenced,
                    name=str(row[2]),
                )
            )

        tables: list[Table] = []
        for row in table_rows:
            schema_name, table_name = str(row[0]), str(row[1])
            key = (schema_name, table_name)
            table_indexes = tuple(indexes.get(key, ()))
            tables.append(
                Table(
                    name=table_name,
                    schema_name=schema_name,
                    kind=_TABLE_KINDS.get(str(row[2]), TableKind.OTHER),
                    columns=tuple(columns.get(key, ())),
                    indexes=table_indexes,
                    foreign_keys=tuple(foreign_keys.get(key, ())),
                    primary_key=next(
                        (index.columns for index in table_indexes if index.primary), ()
                    ),
                    # reltuples is -1 for a relation no ANALYZE has ever touched (and for views):
                    # that is "unknown", and 0 would be a claim about an empty table.
                    row_estimate=row_estimate(row[3]),
                    comment=_text(row[4]),
                )
            )
        return CatalogSnapshot(source=self._source, captured_at=self._clock(), tables=tuple(tables))

    # --- SourceAdapter: execute / cancel -------------------------------------------------

    def explain(self, fragment: Fragment) -> PlanSummary:
        """The engine's own plan for ``fragment``, and how many partitions it would scan.

        No statement is executed: ``EXPLAIN`` without ``ANALYZE`` only asks the planner. The plan
        is parsed (:func:`~mdqm.adapters.postgresql_plan.parse_plan`) rather than searched as text,
        and the partition counts are the *fragment's* tables: the partitions the plan named, over
        the partitions those tables have.

        ``pruning_proof`` is deliberately left ``UNKNOWN``. Whether a scan is acceptable depends on
        the window the spec intends, which the adapter cannot see; the verdict belongs to
        :func:`~mdqm.adapters.postgresql_plan.pruning_verdict`, and overstating it here would be
        the adapter claiming a promise that belongs to someone else.
        """
        text = fragment.sql
        if text is None:
            raise NotSupported(OPTIONAL_METHODS["explain"], "this fragment carries no SQL")
        with self._connection() as connection, connection.cursor() as cur:
            # `text` is a fragment the compiler built through the AST and the guard inspected, so it
            # is code, not data: `as_query` marks that (see mdqm.sql.text). Its bound values travel
            # with it -- explaining a parameterised statement without them fails outright.
            # The inner statement is what can carry an injection; it is guarded before it is wrapped.
            inner = prepare(text, dialect=self.dialect(), params=fragment.params or ())
            execute_one(cur, PreparedStatement(
                text="EXPLAIN (FORMAT JSON) " + inner.text,
                params=inner.params,
                dialect=inner.dialect,
            ))
            row = cur.fetchone()
            document = row[0] if row else None
            children = _partition_children(cur)

        raw = json.dumps(document, ensure_ascii=False) if document is not None else None
        plan = parse_plan(document)
        scanned = () if plan is None else tuple(name for name in plan.relations if name in children)
        parents = {children[name] for name in scanned}
        sizes = Counter(children.values())
        total = sum(sizes[parent] for parent in parents) if parents else None
        indexes = () if plan is None else plan.indexes
        return PlanSummary(
            partitions_scanned=len(scanned) if total else None,
            partitions_total=total,
            uses_index=bool(indexes),
            indexes_used=indexes,
            est_rows=None if plan is None else plan.plan_rows,
            est_cost=None if plan is None else plan.total_cost,
            raw=raw,
        )

    def mutability_stats(self) -> list[TableStats]:
        """Change counters per table from ``pg_stat_user_tables``, which Postgres keeps cheaply.

        These are the numbers the bootstrap pipeline needs to classify a table as append-only or
        ``mutable_critical`` (S05). They are counters, not deltas: two readings give the change.
        """
        with self._connection() as connection, connection.cursor() as cur:
            cur.execute(MUTABILITY_SQL)
            rows = cur.fetchall()
        return mutability_from_rows(self._source, rows, self._clock())

    def mine_queries(self, since: datetime) -> Iterator[RawQuery]:
        """Statements this database has already run, from ``pg_stat_statements``.

        The view keeps *cumulative* counters, so each call reports the delta against the previous
        call on this adapter (the first call is the baseline): stage D9 sums what it is given, and
        the same totals twice would count one execution twice. ``since`` cannot narrow a statement:
        the view carries no per-statement timestamps (see :mod:`mdqm.adapters.postgresql_stats`).

        Raises :class:`NotSupported` with the ``query_store`` capability when the extension is not
        installed or this role may not read it -- never a crash, and never an empty list that looks
        like "this database runs nothing".
        """
        with self._connection() as connection, connection.cursor() as cur:
            mined = self._query_store.read(
                cur, source=self._source, dialect=self.dialect(), since=since
            )
        return iter(mined)

    def execute(self, fragment: Fragment, limits: Limits) -> ArrowStream:
        """Run one compiled fragment on a server-side cursor and stream it as Arrow batches.

        The transaction is narrowed to ``limits.timeout_s`` with ``SET LOCAL statement_timeout``
        (the server aborts, so a runaway never keeps the primary busy after we stopped listening)
        and the client deadline is checked between batches. ``fragment.params``, when present, must
        use psycopg's named placeholders (``%(name)s``); the compiler inlines literals, so this is
        the exception, not the rule.
        """
        if fragment.sql is None:
            raise ValueError(
                f"the postgresql adapter runs SQL; fragment kind {fragment.kind!r} is not supported"
            )
        connection = self.pool.getconn(timeout=self._connect_wait_s)
        try:
            with connection.cursor() as setup:
                # SET LOCAL is transaction-scoped, so a returned connection never keeps this
                # deadline. It must run on a plain cursor: a named cursor would DECLARE it.
                # `is_local=true` is `SET LOCAL`: the deadline dies with the transaction. The value
                # is bound, so no SQL is composed here.
                setup.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (str(max(1, round(limits.timeout_s * 1000))),),
                )
            self._cursor_seq += 1
            cursor = connection.cursor(name=f"mdqm_{self._cursor_seq}")
            # Defence in depth: the fragment was guarded when it was compiled, and it is guarded
            # again here, at the moment it reaches the driver. A fragment that somehow skipped the
            # compiler still cannot execute, and the values stay bound.
            statement = prepare(
                fragment.sql, dialect=self.dialect(), params=fragment.params or ()
            )
            execute_one(cursor, statement)
            return PgArrowStream(
                connection=connection,
                cursor=cursor,
                schema=_arrow_schema(cursor.description),
                limits=limits,
                deadline=self._monotonic() + limits.timeout_s,
                clock=self._monotonic,
                release=self._release,
            )
        except BaseException:
            with contextlib.suppress(psycopg.Error):
                connection.rollback()
            with contextlib.suppress(psycopg.Error, psycopg_pool.PoolClosed):
                self.pool.putconn(connection)
            raise

    def cancel(self, handle: ArrowStream) -> None:
        """Abort the statement behind ``handle``. Idempotent; safe to call twice."""
        cancel = getattr(handle, "cancel", None)
        if callable(cancel):
            cancel()
        else:
            handle.close()

    def _release(self, connection: psycopg.Connection[Any]) -> None:
        """Return a connection to the pool, discarding it when the pool is already closed."""
        pool = self._pool
        if pool is not None:
            pool.putconn(connection)

    # --- optional capabilities ------------------------------------------------------------

    def extract_partitions(self) -> list[Partitioning]:
        """Partition metadata per partitioned table: strategy, key, bounds, rows per partition.

        The catalog columns and their parsing live in
        :func:`~mdqm.adapters.postgresql_plan.parse_partitions` because the parser is the part
        worth testing without a server; this method only runs the query.
        """
        with self._connection() as connection, connection.cursor() as cur:
            cur.execute(PARTITIONS_SQL)
            rows = cur.fetchall()
        return list(parse_partitions(rows))


# --- helpers --------------------------------------------------------------------------------


def _conninfo(config: SourceConfig, secrets: SecretProvider | None) -> tuple[str, str]:
    """Build the libpq connection string and the effective user, resolving the secret reference.

    Returns ``(conninfo, user)``. The password lives only inside the returned string, which the
    adapter never logs; ``user`` is used for ``application_name`` so a DBA can attribute a query.
    """
    options = dict(config.options or {})
    raw_dsn = options.get("dsn")
    params: dict[str, str] = {}
    if raw_dsn is not None:
        try:
            parsed = conninfo.conninfo_to_dict(str(raw_dsn))
        except psycopg.ProgrammingError as exc:
            raise PostgresConfigError(
                f"source {config.name!r}: options.dsn is not a valid connection string ({exc}). "
                "Use e.g. postgresql://user@host:5432/dbname?sslmode=require"
            ) from None
        if parsed.get("password"):
            raise PostgresConfigError(
                f"source {config.name!r}: options.dsn carries a literal password. "
                "Remove it and reference a secret with `secret: <name>` instead."
            )
        params = {key: str(value) for key, value in parsed.items() if value is not None}
    else:
        for option_key, conninfo_key in _CONNINFO_KEYS.items():
            if option_key in options:
                params[conninfo_key] = str(options[option_key])

    if config.secret is not None:
        if secrets is None:
            raise PostgresConfigError(
                f"source {config.name!r} references secret {config.secret!r} but no secret "
                "provider is configured; without one the password cannot be resolved"
            )
        params["password"] = secrets.get(config.secret)

    params.setdefault("connect_timeout", "10")
    user = params.get("user", "")
    if not user:
        raise PostgresConfigError(
            f"source {config.name!r}: no database user. Set options.user (or options.dsn with a "
            "user); the server needs a per-person identity, not a shared one."
        )
    return conninfo.make_conninfo(**params), user


def _positive_int(value: object, default: int, key: str) -> int:
    """Read a positive integer option, naming the key instead of raising a bare ``ValueError``."""
    if value is None:
        return default
    try:
        parsed = int(str(value))
    except ValueError:
        raise PostgresConfigError(
            f"options.{key} must be a positive integer, got {value!r}"
        ) from None
    if parsed <= 0:
        raise PostgresConfigError(f"options.{key} must be a positive integer, got {value!r}")
    return parsed


def _short_version(version_text: str) -> str | None:
    """``PostgreSQL 16.16 (Debian ...) on x86_64`` -> ``PostgreSQL 16.16``."""
    parts = version_text.split(" on ")[0].split()
    if len(parts) >= 2 and parts[0].lower() == "postgresql":
        return f"{parts[0]} {parts[1]}"
    return version_text.strip() or None


def _text(value: object) -> str | None:
    """A nullable text column as ``str | None`` (``None`` stays ``None``)."""
    return None if value is None else str(value)
