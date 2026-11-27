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
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import psycopg
import psycopg_pool
import pyarrow as pa
from psycopg import conninfo, sql

from mdqm.adapters.base import ArrowStream, NotSupported
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
    Table,
    TableKind,
)

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
NUMERIC_SCALE: Final = 9

_CAPABILITY_PARTITIONS: Final = "partition_metadata"

#: Why the optional features this adapter does not serve are reported *unprobed* (``None``) rather
#: than unsupported. PostgreSQL can do all of them; the code that reads them belongs to another
#: task, and saying "this source cannot" would be a lie the capability report repeats.
_NOT_YET_EXPLAIN: Final = "EXPLAIN plan parsing is task A03 (adapters/postgresql_plan.py)"
_NOT_YET_QUERIES: Final = "query-store mining is task A04 (adapters/postgresql_stats.py)"
_NOT_YET_STATS: Final = "mutability counters are task A04 (adapters/postgresql_stats.py)"

#: ``options`` keys this adapter understands. Anything else is logged by name, never guessed at.
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

_PARTITION_STRATEGIES: Final[dict[str, PartitionStrategy]] = {
    "r": PartitionStrategy.RANGE,
    "l": PartitionStrategy.LIST,
    "h": PartitionStrategy.HASH,
}

_MONTH_STEPS: Final[dict[timedelta, PartitionGranularity]] = {
    timedelta(hours=1): PartitionGranularity.HOUR,
    timedelta(days=1): PartitionGranularity.DAY,
    timedelta(days=7): PartitionGranularity.WEEK,
    timedelta(days=90): PartitionGranularity.QUARTER,
    timedelta(days=365): PartitionGranularity.YEAR,
}

_RANGE_BOUND_RE: Final = re.compile(
    r"\AFOR VALUES FROM \((?P<lower>.*)\) TO \((?P<upper>.*)\)\Z", re.DOTALL
)
_LIST_BOUND_RE: Final = re.compile(r"\AFOR VALUES IN \((?P<values>.*)\)\Z", re.DOTALL)
_HASH_BOUND_RE: Final = re.compile(
    r"\AFOR VALUES WITH \(MODULUS (?P<modulus>\d+), REMAINDER (?P<remainder>\d+)\)\Z",
    re.IGNORECASE,
)
_QUOTED_RE: Final = re.compile(r"\A'(?P<text>.*)'\Z", re.DOTALL)
_PARTKEY_RE: Final = re.compile(r"\A[A-Z]+ \((?P<key>.*)\)\Z")


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

# Index partitions inherit the parent's index definition (`accounts_hash_p0_pkey`), so the child
# relation is restricted to a real table: otherwise every partition of every index shows up.
_PARTITIONS_SQL: Final = """
SELECT pn.nspname AS parent_schema,
       pc.relname AS parent_table,
       pt.partstrat AS strategy,
       pg_get_partkeydef(pc.oid) AS partition_key,
       cn.nspname AS child_schema,
       cc.relname AS child_table,
       pg_get_expr(cc.relpartbound, cc.oid) AS bound,
       cc.reltuples::bigint AS row_estimate
  FROM pg_inherits i
  JOIN pg_class pc ON pc.oid = i.inhparent
  JOIN pg_namespace pn ON pn.oid = pc.relnamespace
  JOIN pg_class cc ON cc.oid = i.inhrelid
  JOIN pg_namespace cn ON cn.oid = cc.relnamespace
  LEFT JOIN pg_partitioned_table pt ON pt.partrelid = pc.oid
 WHERE pn.nspname NOT IN ('pg_catalog', 'information_schema')
   AND pc.relkind = 'p'
   AND cc.relkind = 'r'
 ORDER BY pn.nspname, pc.relname, cc.relname
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

    kind: Final = "postgresql"

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
        self._connect_wait_s = float(self._statement_timeout_ms) / 1000.0
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic
        self._pool: psycopg_pool.ConnectionPool | None = None
        self._cursor_seq = 0

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
            pool = psycopg_pool.ConnectionPool(
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
            # SET takes no bind parameter, so the number is composed as a literal through
            # psycopg.sql (never concatenated) and is already an int by construction.
            cur.execute(
                sql.SQL("SET statement_timeout = {}").format(
                    sql.Literal(self._statement_timeout_ms)
                )
            )
            cur.execute(
                sql.SQL("SET idle_in_transaction_session_timeout = {}").format(
                    sql.Literal(self._statement_timeout_ms * 2)
                )
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
        capabilities = {
            _CAPABILITY_PARTITIONS: Capability(
                supported=can_read_partitions,
                detail="pg_partitioned_table + pg_inherits"
                if can_read_partitions
                else "this server has no declarative partitioning catalog (pre-10)",
            ),
            "explain_plan": Capability(detail=_NOT_YET_EXPLAIN),
            "query_store": Capability(detail=_NOT_YET_QUERIES),
            "mutability_counters": Capability(detail=_NOT_YET_STATS),
        }
        return Capabilities(
            source=self._source,
            adapter=self.kind,
            version=version,
            retrieved_at=self._clock(),
            capabilities=capabilities,
        )

    # --- SourceAdapter: catalog ----------------------------------------------------------

    def extract_catalog(self) -> CatalogSnapshot:
        """Tables, columns, types, keys and indexes from ``pg_catalog``.

        Partition children are deliberately not listed: they are how their parent is stored, they
        inherit the parent's primary key and indexes, and asking a weak agent to choose between
        ``usage_events`` and ``usage_events_p2025_12`` is a way to scan one month and call it a
        year. :meth:`extract_partitions` reports the children where the planner needs them.
        """
        with self._connection() as connection, connection.cursor() as cur:
            cur.execute(_CATALOG_TABLES_SQL)
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
                        str(row[4]), str(row[5]), str(row[6]), int(row[7])
                    ),
                    nullable=not bool(row[8]),
                    ordinal=int(row[9]),
                    comment=_text(row[10]),
                )
            )

        indexes: dict[tuple[str, str], list[Index]] = {}
        for row in index_rows:
            key = (str(row[0]), str(row[1]))
            names = _split_names(row[7])
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
            local = _split_names(row[5])
            referenced = _split_names(row[6])
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
                    row_estimate=_row_estimate(row[3]),
                    comment=_text(row[4]),
                )
            )
        return CatalogSnapshot(source=self._source, captured_at=self._clock(), tables=tuple(tables))

    # --- SourceAdapter: execute / cancel -------------------------------------------------

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
                setup.execute(
                    sql.SQL("SET LOCAL statement_timeout = {}").format(
                        sql.Literal(max(1, round(limits.timeout_s * 1000)))
                    )
                )
            self._cursor_seq += 1
            cursor = connection.cursor(name=f"mdqm_{self._cursor_seq}")
            cursor.execute(fragment.sql, fragment.params or None)
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
        """Partition metadata per partitioned table, read from the catalog.

        Task A03 owns the definitive version (:mod:`mdqm.adapters.postgresql_plan`): the range
        bounds here come from ``pg_get_expr``'s text form, which is enough to count and align the
        monthly set a query must prune to. The strategy, key and bound list are the same either
        way, so a caller does not have to change shape when A03 lands.
        """
        with self._connection() as connection, connection.cursor() as cur:
            cur.execute(_PARTITIONS_SQL)
            rows = cur.fetchall()
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            parent = (str(row[0]), str(row[1]))
            entry = grouped.setdefault(
                parent,
                {"strategy": row[2], "key": row[3], "bounds": []},
            )
            bounds: list[PartitionBound] = entry["bounds"]
            bounds.append(_bound(str(row[5]), str(row[6]), _row_estimate(row[7])))
        return [self._partitioning(parent, entry) for parent, entry in sorted(grouped.items())]

    def _partitioning(self, parent: tuple[str, str], entry: Mapping[str, Any]) -> Partitioning:
        """Turn one grouped catalog row set into a validated :class:`Partitioning`."""
        strategy = _PARTITION_STRATEGIES.get(str(entry["strategy"]))
        bounds = tuple(entry["bounds"])
        if strategy is None:
            # An inheritance tree that is not declarative partitioning still exists and is still
            # scanned in full; refusing to claim a strategy is the honest answer.
            raise NotSupported(
                _CAPABILITY_PARTITIONS,
                f"{parent[0]}.{parent[1]} is partitioned by a mechanism this adapter does not read",
            )
        key = _split_names(_partition_key_text(str(entry["key"])))
        if strategy is not PartitionStrategy.EXPRESSION and not key:
            raise NotSupported(
                _CAPABILITY_PARTITIONS,
                f"{parent[0]}.{parent[1]}: the partition key expression is not a column list",
            )
        return Partitioning(
            strategy=strategy,
            key=key,
            granularity=_infer_granularity(bounds) if strategy is PartitionStrategy.RANGE else None,
            bounds=bounds,
        )


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


def _row_estimate(value: object) -> int | None:
    """``pg_class.reltuples`` as a non-negative estimate, or ``None`` when it is unknown (-1)."""
    try:
        estimate = int(str(value))
    except (TypeError, ValueError):
        return None
    return estimate if estimate >= 0 else None


def _split_names(value: object) -> tuple[str, ...]:
    """Split a comma-joined column list into a tuple, dropping blanks."""
    if value is None:
        return ()
    return tuple(part for part in (chunk.strip() for chunk in str(value).split(",")) if part)


def _partition_key_text(definition: str) -> str:
    """``RANGE (start_time)`` -> ``start_time``; an expression key keeps its parentheses."""
    match = _PARTKEY_RE.match(definition.strip())
    return match.group("key") if match else ""


def _bound(name: str, definition: str, row_estimate: int | None) -> PartitionBound:
    """Parse one partition's ``pg_get_expr(relpartbound)`` text into a bound.

    Range bounds become UTC datetimes because the planner aligns ranges and cuts cache segments on
    month boundaries; list and hash partitions carry their own text, since their values are not
    datetimes and inventing one would be worse than reporting none.
    """
    text = definition.strip()
    range_match = _RANGE_BOUND_RE.match(text)
    if range_match is not None:
        return PartitionBound(
            name=name,
            lower=_bound_datetime(range_match.group("lower")),
            upper=_bound_datetime(range_match.group("upper")),
            row_estimate=row_estimate,
        )
    if _LIST_BOUND_RE.match(text) is not None or _HASH_BOUND_RE.match(text) is not None:
        return PartitionBound(name=name, values=(text,), row_estimate=row_estimate)
    # DEFAULT (and anything a future release writes): the partition exists, its bound is unknown.
    return PartitionBound(name=name, row_estimate=row_estimate)


def _bound_datetime(token: str) -> datetime | None:
    """Parse a range bound literal (``'2024-11-01 00:00:00+00'``), or ``None`` for MINVALUE."""
    text = token.strip()
    if text.upper() in {"MINVALUE", "MAXVALUE"}:
        return None
    quoted = _QUOTED_RE.match(text)
    if quoted is not None:
        text = quoted.group("text")
    candidate = text.replace(" ", "T", 1)
    with contextlib.suppress(ValueError):
        return datetime.fromisoformat(candidate)
    # PostgreSQL prints a zero offset as "+00"; fromisoformat before 3.11 wants "+00:00".
    with contextlib.suppress(ValueError):
        return datetime.fromisoformat(f"{candidate}:00")
    return None


def _infer_granularity(bounds: Sequence[PartitionBound]) -> PartitionGranularity | None:
    """Infer the partition width when every step agrees, else ``None`` (never a guess)."""
    steps = [(bound.lower, bound.upper) for bound in bounds if bound.lower and bound.upper]
    if not steps:
        return None
    if all(_is_month_step(lower, upper) for lower, upper in steps if lower and upper):
        return PartitionGranularity.MONTH
    deltas = {(upper - lower) for lower, upper in steps if lower and upper}
    if len(deltas) != 1:
        return None
    return _MONTH_STEPS.get(deltas.pop())


def _is_month_step(lower: datetime, upper: datetime) -> bool:
    """True when ``upper`` is exactly the first instant of the month after ``lower``."""
    start = lower.astimezone(UTC)
    end = upper.astimezone(UTC)
    if (start.day, start.hour, start.minute, start.second, start.microsecond) != (1, 0, 0, 0, 0):
        return False
    next_month = datetime(start.year + start.month // 12, start.month % 12 + 1, 1, tzinfo=UTC)
    return end == next_month
