"""The Elasticsearch adapter: index families as read-only tables (task A09).

Elasticsearch stores documents, not rows, and its "partitioning" is a naming convention: one index
per month behind an alias. A weak agent should never have to learn either fact, so this adapter
presents an **alias as a table** and the monthly indices as that table's partitions
(:meth:`ElasticAdapter.extract_partitions`). The alias is the user-facing table because that is the
name that survives rotation: `mdqm-usage-events` keeps working when a job adds
`mdqm-usage-events-2026-02`, while a query against one month's index silently loses next month's
rows. An index that belongs to no alias is its own table, so nothing becomes invisible.

Read-only, three ways (``AGENTS.md`` principle 5). The adapter builds only ``GET`` requests —
and the one exception the server forces: ES|QL answers ``POST /_query`` and refuses ``GET`` (a real
405 from 8.15: ``Incorrect HTTP method for uri [/_query] and method [GET], allowed: [POST]``). The
transport (:class:`SearchOnlyTransport`) therefore allows ``GET`` and ``POST /_query`` and nothing
else, so a future code path cannot write to a cluster even by accident; a refused method becomes
``WRITE_BLOCKED`` with the method and path. ``execute`` only ever touches ``/_search`` and
``/_query``.

Economy (``AGENTS.md`` principle 6). A family is never searched as one blob: a range filter on the
family's *time field* is turned into the exact list of monthly indices to touch (``_shards.total``
in the response proves it), because Elasticsearch would otherwise read all twelve. A query with no
time bound over a family that holds more indices than ``Limits.max_partitions`` is refused with
``PARTITION_SCAN_ALL`` and a fix that names the field, and one whose bound covers too many months is
refused with ``COST_TOO_HIGH`` — both before any search is sent. Row and byte caps are enforced
here, and whatever was dropped is reported on the stream (``dropped_rows``, ``dropped_values``,
``total_hits``), never silently.

ES|QL is used when the server has it and the fragment carries an ES|QL statement; otherwise the
same fragment's Query DSL is sent. The fragment is ``Fragment.dsl``::

    {"table": "usage_events", "query": {"bool": {"filter": [{"range": {"start_time": ...}}]}},
     "esql": "FROM usage_events | STATS ...", "sort": [...], "fields": [...]}

Only ``table`` and ``query`` are required; ``esql`` is optional (its presence is what selects the
ES|QL path), and aggregation bodies are refused with a fix rather than half-implemented — the
federated engine aggregates after the fetch.

What it cannot do, reported as such and never pretended: Elasticsearch has no ``EXPLAIN`` (so
:meth:`explain` reports the index set and ``pruning_proof`` stays ``unknown``), no query store
(the slow log is opt-in, off by default, and not mined) and no change counters.

Usage::

    adapter = ElasticAdapter(config, secrets)  # options: base_url, or host/port/scheme
    run_conformance(lambda: adapter, fragment=adapter.fragment("usage_events", ...))
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar, Final

import httpx
import pyarrow as pa

from mdqm.adapters.base import OPTIONAL_METHODS, ArrowStream, NotSupported
from mdqm.config.secrets import SecretNotFound, SecretProvider
from mdqm.config.settings import SourceConfig
from mdqm.errors import (
    cost_too_high,
    invalid_spec,
    partition_scan_all,
    source_unavailable,
    timeout,
    unknown_table,
    write_blocked,
)
from mdqm.models import (
    Capabilities,
    Capability,
    CatalogSnapshot,
    Column,
    ErrorResponse,
    Fragment,
    Limits,
    NormalizedType,
    PartitionBound,
    PartitionGranularity,
    Partitioning,
    PartitionSize,
    PartitionStrategy,
    PlanSummary,
    PruningProof,
    RawQuery,
    Table,
    TableKind,
    TableSize,
    TableStats,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "ESQL_PATH",
    "ElasticAdapter",
    "ElasticError",
    "ElasticUnavailable",
    "EsFeatures",
    "SearchOnlyTransport",
    "parse_index_sizes",
    "table_sizes",
]

_log = logging.getLogger(__name__)

DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 9200
"""Elasticsearch's own defaults; a deployment overrides them in ``SourceConfig.options``."""

ESQL_PATH: Final = "/_query"
"""The only path this adapter is allowed to ``POST``: ES|QL refuses ``GET``."""

MAX_HITS_PER_REQUEST: Final = 10_000
"""``index.max_result_window``'s default: a larger ``size`` is refused by the server, so it is not
asked for, and the shortfall is reported on the stream as ``dropped_rows``."""

_MONTHLY_INDEX: Final = re.compile(r"^(?P<family>.+)-(?P<year>\d{4})-(?P<month>\d{2})$")
"""``family-YYYY-MM``: the naming convention that makes a family time-partitioned."""

_DSL_KEYS: Final = ("esql", "fields", "query", "sort", "table")
"""Every key a fragment's ``dsl`` payload may carry. Anything else is refused with a fix."""

_UNABLE: Final = object()
"""Marker for a value the mapping's type could not represent; counted, never silently dropped."""


class ElasticError(Exception):
    """An agent-visible failure, carrying the :class:`~mdqm.models.ErrorResponse` to return.

    Adapters neither print nor leak a driver message: the caller (governor or tool layer) hands
    ``response`` straight to the agent. ``code`` is repeated so a caller can branch on it.
    """

    def __init__(self, response: ErrorResponse) -> None:
        self.response = response
        super().__init__(response.code)

    @property
    def code(self) -> str:
        """The stable error code, e.g. ``SOURCE_UNAVAILABLE``."""
        return self.response.code


class ElasticUnavailable(ElasticError, OSError):
    """The cluster did not answer (or refused the credentials).

    Also an :class:`OSError`, so :func:`mdqm.errors.from_exception` still classifies it as a source
    failure for a caller that only knows the generic mapping.
    """


class SearchOnlyTransport(httpx.BaseTransport):
    """A transport that refuses every request except a read (``AGENTS.md`` principle 5).

    Allowed: ``GET`` on the configured host and ``POST /_query``. Everything else — ``PUT``,
    ``DELETE``, ``POST /_bulk``, ``POST /{index}/_delete_by_query`` — raises ``WRITE_BLOCKED``
    before a byte is sent. The adapter has no code path that would build such a request; this layer
    is what proves it, so a later change cannot quietly start writing to a cluster.

    It also pins the host: ``httpx`` drops ``base_url`` for an absolute URL, so a fragment or a
    redirect cannot make the adapter talk to another machine.
    """

    def __init__(
        self,
        inner: httpx.BaseTransport,
        *,
        allowed_host: str | None = None,
        allowed_posts: frozenset[str] = frozenset({ESQL_PATH}),
    ) -> None:
        self._inner = inner
        self._allowed_host = allowed_host
        self._allowed_posts = allowed_posts

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        readable = request.method == "GET" or (
            request.method == "POST" and request.url.path in self._allowed_posts
        )
        if not readable:
            raise ElasticError(write_blocked(operation=f"{request.method} {request.url.path}"))
        if self._allowed_host is not None and request.url.host != self._allowed_host:
            raise ElasticError(
                invalid_spec(
                    field="options.base_url",
                    reason=f"a request would leave the configured host ({self._allowed_host})",
                )
            )
        return self._inner.handle_request(request)

    def close(self) -> None:
        """Close the wrapped transport (idempotent, as ``httpx`` promises)."""
        self._inner.close()


# --- what the server can do ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EsFeatures:
    """What one probe found: the version, ES|QL availability and whether aggregations run."""

    version: str | None = None
    esql: bool | None = None
    aggregations: bool | None = None
    esql_detail: str = ""
    aggregations_detail: str = ""
    probed_at: datetime | None = None


# --- the catalog, in this adapter's vocabulary ---------------------------------------------

#: Elasticsearch field type -> (normalised type, arrow type). Anything unknown reads as ``other``
#: and as text, which is honest: the mapping declared something this adapter has no reading for.
_FIELD_TYPES: Final[dict[str, tuple[NormalizedType, pa.DataType]]] = {
    "keyword": (NormalizedType.STRING, pa.string()),
    "constant_keyword": (NormalizedType.STRING, pa.string()),
    "wildcard": (NormalizedType.STRING, pa.string()),
    "text": (NormalizedType.STRING, pa.string()),
    "match_only_text": (NormalizedType.STRING, pa.string()),
    "ip": (NormalizedType.STRING, pa.string()),
    "version": (NormalizedType.STRING, pa.string()),
    "long": (NormalizedType.INTEGER, pa.int64()),
    "integer": (NormalizedType.INTEGER, pa.int32()),
    "short": (NormalizedType.INTEGER, pa.int32()),
    "byte": (NormalizedType.INTEGER, pa.int32()),
    "unsigned_long": (NormalizedType.INTEGER, pa.int64()),
    "double": (NormalizedType.FLOAT, pa.float64()),
    "float": (NormalizedType.FLOAT, pa.float64()),
    "half_float": (NormalizedType.FLOAT, pa.float64()),
    "scaled_float": (NormalizedType.DECIMAL, pa.float64()),
    "boolean": (NormalizedType.BOOLEAN, pa.bool_()),
    "date": (NormalizedType.TIMESTAMP, pa.timestamp("us", tz="UTC")),
    "date_nanos": (NormalizedType.TIMESTAMP, pa.timestamp("us", tz="UTC")),
    "binary": (NormalizedType.BINARY, pa.binary()),
    "object": (NormalizedType.JSON, pa.string()),
    "nested": (NormalizedType.JSON, pa.string()),
    "flattened": (NormalizedType.JSON, pa.string()),
}

_ESQL_TYPES: Final[dict[str, tuple[NormalizedType, pa.DataType]]] = {
    "long": (NormalizedType.INTEGER, pa.int64()),
    "integer": (NormalizedType.INTEGER, pa.int32()),
    "short": (NormalizedType.INTEGER, pa.int32()),
    "byte": (NormalizedType.INTEGER, pa.int32()),
    "unsigned_long": (NormalizedType.INTEGER, pa.int64()),
    "double": (NormalizedType.FLOAT, pa.float64()),
    "float": (NormalizedType.FLOAT, pa.float64()),
    "boolean": (NormalizedType.BOOLEAN, pa.bool_()),
    "keyword": (NormalizedType.STRING, pa.string()),
    "text": (NormalizedType.STRING, pa.string()),
    "ip": (NormalizedType.STRING, pa.string()),
    "version": (NormalizedType.STRING, pa.string()),
    "date": (NormalizedType.TIMESTAMP, pa.timestamp("us", tz="UTC")),
    "datetime": (NormalizedType.TIMESTAMP, pa.timestamp("us", tz="UTC")),
    "date_nanos": (NormalizedType.TIMESTAMP, pa.timestamp("us", tz="UTC")),
}


@dataclass(frozen=True, slots=True)
class EsField:
    """One mapped field of a table: what the cluster says it is, and how it becomes a column."""

    name: str
    es_type: str
    normalized_type: NormalizedType
    arrow_type: pa.DataType
    aggregatable: bool


@dataclass(frozen=True, slots=True)
class EsTable:
    """One user-facing table: an alias over its member indices, or a lone index."""

    name: str
    members: tuple[str, ...]
    fields: tuple[EsField, ...]
    docs: int
    aliased: bool
    time_field: str | None
    granularity: PartitionGranularity | None

    @property
    def target(self) -> str:
        """The index expression a search path uses: the table name itself."""
        return self.name

    @property
    def column_names(self) -> tuple[str, ...]:
        """Field names in catalog order."""
        return tuple(field.name for field in self.fields)

    def field(self, name: str) -> EsField | None:
        """Look a field up by name; ``None`` when the mapping has no such field."""
        return next((field for field in self.fields if field.name == name), None)


@dataclass(frozen=True, slots=True)
class EsRequest:
    """What one fragment asks for, after validation: a table, a filter, optionally ES|QL."""

    table: str
    query: Mapping[str, Any] | None
    esql: str | None
    sort: tuple[Mapping[str, Any], ...] = ()
    fields: tuple[str, ...] = ()


# --- value conversion: never raise on remote data -----------------------------------------


def _to_datetime(value: Any) -> datetime | None:
    """``value`` as an aware UTC datetime; ``None`` when it is not a date at all."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        # pi-lens-ignore: unchecked-numeric-parse-python -- a number by the branch above
        return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
    if isinstance(value, str):
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None


def _coerce(value: Any, normalized: NormalizedType) -> Any:
    """Convert one document value to what its normalised type promises; ``_UNABLE`` if it cannot.

    Remote data is not trusted to match its own mapping (that is what a mapping conflict *means*),
    so nothing here raises: an impossible value becomes a NULL and is counted by the caller.
    """
    if value is None:
        return None
    if normalized is NormalizedType.INTEGER:
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            # pi-lens-ignore: unchecked-numeric-parse-python -- a float by the branch above
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return _UNABLE
        return _UNABLE
    if normalized is NormalizedType.FLOAT or normalized is NormalizedType.DECIMAL:
        try:
            return float(value)
        except (TypeError, ValueError):
            return _UNABLE
    if normalized is NormalizedType.BOOLEAN:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        return _UNABLE
    if normalized is NormalizedType.TIMESTAMP:
        return _to_datetime(value) or _UNABLE
    if normalized is NormalizedType.JSON:
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return _UNABLE
    if normalized is NormalizedType.BINARY:
        return value if isinstance(value, bytes) else _UNABLE
    return value if isinstance(value, str) else str(value)


def _source_value(source: Mapping[str, Any], name: str) -> Any:
    """The value of ``name`` in one ``_source``, following dotted paths for nested objects."""
    if name in source:
        return source[name]
    node: Any = source
    for part in name.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _rows_to_table(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[EsField]
) -> tuple[pa.Table, int]:
    """Build the Arrow table for ``rows`` from the mapping's fields; return it and the bad count."""
    arrays: dict[str, pa.Array] = {}
    bad = 0
    for field in fields:
        values: list[Any] = []
        for row in rows:
            raw = _source_value(row, field.name)
            converted = _coerce(raw, field.normalized_type)
            if converted is _UNABLE:
                bad += 1
                values.append(None)
            else:
                values.append(converted)
        arrays[field.name] = pa.array(values, type=field.arrow_type)
    schema = pa.schema([pa.field(f.name, f.arrow_type) for f in fields])
    return pa.Table.from_pydict(arrays, schema=schema), bad


def _esql_table(payload: Mapping[str, Any], max_rows: int) -> tuple[pa.Table, int, int]:
    """ES|QL ``columns``/``values`` into Arrow; returns (table, dropped rows, bad values)."""
    columns = payload.get("columns")
    values = payload.get("values")
    if not isinstance(columns, list) or not isinstance(values, list):
        return pa.table({}), 0, 0
    kept = values[:max_rows]
    dropped = len(values) - len(kept)
    arrays: dict[str, pa.Array] = {}
    bad = 0
    for index, column in enumerate(columns):
        if not isinstance(column, Mapping):
            continue
        name = str(column.get("name", f"col{index}"))
        normalized, arrow_type = _ESQL_TYPES.get(
            str(column.get("type", "keyword")), (NormalizedType.OTHER, pa.string())
        )
        converted: list[Any] = []
        for row in kept:
            raw = row[index] if isinstance(row, list) and index < len(row) else None
            value = _coerce(raw, normalized)
            if value is _UNABLE:
                bad += 1
                converted.append(None)
            else:
                converted.append(value)
        arrays[name] = pa.array(converted, type=arrow_type)
    if not arrays:
        return pa.table({}), dropped, bad
    return pa.Table.from_pydict(arrays), dropped, bad


def _trim_bytes(table: pa.Table, max_bytes: int) -> tuple[pa.Table, int]:
    """Keep whole batches until ``max_bytes`` is reached; return (kept table, dropped rows)."""
    if table.nbytes <= max_bytes or table.num_rows == 0:
        return table, 0
    kept: list[pa.RecordBatch] = []
    used = 0
    rows = 0
    for batch in table.to_batches(max_chunksize=256):
        if used + batch.nbytes > max_bytes:
            break
        kept.append(batch)
        used += batch.nbytes
        rows += batch.num_rows
    if not kept:
        return table.slice(0, 0), table.num_rows
    return pa.Table.from_batches(kept, schema=table.schema), table.num_rows - rows


# --- the stream --------------------------------------------------------------------------


class _RowsStream:
    """The ``ArrowStream`` ``execute`` returns: already-fetched, capped batches plus provenance.

    The fetch happens inside :meth:`ElasticAdapter.execute` — under the row and byte caps and the
    deadline — so a cancelled stream simply stops yielding. ``touched_indices`` and ``took_ms`` are
    what the cluster reported about the request, so a caller can prove the pruning instead of
    trusting it.
    """

    def __init__(
        self,
        table: pa.Table,
        *,
        schema: pa.Schema,
        fetched_at: datetime,
        index_target: str,
        touched: tuple[str, ...],
        took_ms: int,
        total_hits: int,
        esql: bool,
        dropped_rows: int,
        dropped_values: int,
    ) -> None:
        self.schema: pa.Schema = schema
        self.fetched_at = fetched_at
        self.index_target = index_target
        self.touched_indices = touched
        self.took_ms = took_ms
        self.total_hits = total_hits
        self.esql = esql
        self.dropped_rows = dropped_rows
        self.dropped_values = dropped_values
        self.cancelled = False
        self.rows_returned = 0
        self.bytes_returned = 0
        self._batches = table.to_batches()

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        for batch in self._batches:
            if self.cancelled:
                return
            self.rows_returned += batch.num_rows
            self.bytes_returned += batch.nbytes
            yield batch

    def cancel(self) -> None:
        """Stop yielding. Idempotent."""
        self.cancelled = True

    def close(self) -> None:
        """Release the batches. Idempotent."""
        self.cancelled = True
        self._batches = []


# --- the adapter -------------------------------------------------------------------------


class ElasticAdapter:
    """A read-only ``SourceAdapter`` over one Elasticsearch cluster (``kind="elastic"``).

    Args:
        config: the source's configuration. ``options["base_url"]`` wins; otherwise
            ``options["host"]``/``["port"]``/``["scheme"]`` build it. ``options["time_field"]``
            names the field that decides the index (``start_time`` for the usage-event family);
            without it a family whose mapping has exactly one date field infers that one, and a
            family with several date fields is never guessed at.
        secrets: resolves ``config.auth.secret`` into the header value; ``None`` means the auth
            header is not configured. The value is never logged.
        transport: the HTTP transport to wrap; always wrapped in :class:`SearchOnlyTransport`, so
            an injected transport cannot widen what may be sent. Tests inject one over a fake
            connection.
        clock: UTC clock for ``retrieved_at``/``captured_at``/``fetched_at``.
        esql_support: forces the ES|QL verdict instead of probing it (tests, or an operator whose
            proxy blocks ``/_query``).
    """

    kind: ClassVar[str] = "elastic"

    def __init__(
        self,
        config: SourceConfig,
        secrets: SecretProvider | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], datetime] | None = None,
        esql_support: bool | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._name = config.name
        self._config = config
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._time_field = _optional_str(config.options.get("time_field"))
        self._forced_esql = esql_support
        self._features: EsFeatures | None = None
        self._tables: tuple[EsTable, ...] | None = None
        self._base_url = _base_url(config)
        host = httpx.URL(self._base_url).host
        inner = transport if transport is not None else httpx.HTTPTransport()
        self._client: httpx.Client | None = httpx.Client(
            base_url=self._base_url,
            # Elasticsearch reads a body only when it is declared JSON, and says so loudly:
            # "Content-Type header [] is not supported".
            headers={"content-type": "application/json", **_auth_headers(config, secrets)},
            transport=SearchOnlyTransport(inner, allowed_host=host),
            timeout=timeout_s,
        )

    # --- SourceAdapter: identity ---------------------------------------------------------

    def dialect(self) -> str:
        """``dsl``: a fragment carries a Query DSL (or ES|QL) payload, not SQL."""
        return "dsl"

    @property
    def source(self) -> str:
        """The configured source name; the catalog and the probe both report it."""
        return self._name

    @property
    def base_url(self) -> str:
        """The origin every request goes to (no trailing slash)."""
        return self._base_url

    def fragment(
        self,
        table: str,
        *,
        query: Mapping[str, Any] | None = None,
        esql: str | None = None,
        sort: Sequence[Mapping[str, Any]] = (),
        fields: Sequence[str] = (),
    ) -> Fragment:
        """Build the fragment for ``table``: the shape this adapter's ``execute`` reads.

        The DSL compiler is another lane; until it exists this is the documented shape, and it is
        what the conformance kit runs.
        """
        dsl: dict[str, Any] = {"table": table}
        if query is not None:
            dsl["query"] = dict(query)
        if esql is not None:
            dsl["esql"] = esql
        if sort:
            dsl["sort"] = [dict(item) for item in sort]
        if fields:
            dsl["fields"] = list(fields)
        return Fragment(
            source=self._name,
            dialect=self.dialect(),
            dsl=dsl,
            tables=(table,),
            exec_ts=self._clock(),
        )

    def close(self) -> None:
        """Close the HTTP client. Idempotent."""
        client, self._client = self._client, None
        if client is not None:
            client.close()

    # --- SourceAdapter: probe -------------------------------------------------------------

    def features(self) -> EsFeatures:
        """The cached probe result, probing once on first use.

        The catalog is read first because the aggregation probe needs a table to aggregate on; a
        cluster with no index reported ``aggregations`` as unknown, not as unsupported.
        """
        if self._features is None:
            tables = self._catalog_tables()
            self._features = self._detect_features(tables)
        return self._features

    def probe(self) -> Capabilities:
        """Report what this cluster offers; claim an optional capability only if it works here."""
        features = self.features()
        tables = self._catalog_tables()
        partitioned = any(table.granularity is not None for table in tables)
        sizes_supported, sizes_detail = self._probe_sizes()
        capabilities = {
            OPTIONAL_METHODS["extract_partitions"]: Capability(
                supported=partitioned,
                detail="monthly indices behind the alias; their names give the bounds"
                if partitioned
                else "no index family is named family-YYYY-MM, so nothing is time-partitioned",
            ),
            OPTIONAL_METHODS["explain"]: Capability(
                supported=True,
                detail="Elasticsearch has no EXPLAIN: explain() reports the index set the request "
                "targets and pruning_proof stays unknown",
            ),
            OPTIONAL_METHODS["mine_queries"]: Capability(
                supported=False,
                detail="no query store; the slow log is opt-in and this adapter does not mine it",
            ),
            OPTIONAL_METHODS["mutability_stats"]: Capability(
                supported=False, detail="Elasticsearch keeps no per-index change counters"
            ),
            OPTIONAL_METHODS["sizes"]: Capability(supported=sizes_supported, detail=sizes_detail),
        }
        return Capabilities(
            source=self._name,
            adapter=self.kind,
            version=features.version,
            retrieved_at=features.probed_at or self._clock(),
            capabilities=capabilities,
        )

    def _detect_features(self, tables: tuple[EsTable, ...]) -> EsFeatures:
        """Read the version, then ask the cluster whether ES|QL and aggregations actually answer."""
        info = self._json("GET", "/")
        version_node = info.get("version")
        version = (
            str(version_node.get("number"))
            if isinstance(version_node, Mapping) and version_node.get("number")
            else None
        )
        if self._forced_esql is not None:
            esql, esql_detail = self._forced_esql, "forced by configuration"
        else:
            esql, esql_detail = self._probe_esql()
        aggregations, aggs_detail = self._probe_aggregations(tables)
        return EsFeatures(
            version=version,
            esql=esql,
            aggregations=aggregations,
            esql_detail=esql_detail,
            aggregations_detail=aggs_detail,
            probed_at=self._clock(),
        )

    def _probe_esql(self) -> tuple[bool, str]:
        """``POST /_query`` with a one-row query: present in 8.11+, absent (404/400) before."""
        try:
            payload = self._json("POST", ESQL_PATH, body={"query": "ROW mdqm_probe = 1"})
        except ElasticError as error:
            return False, f"/_query refused the probe ({_describe(error)})"
        if payload.get("values"):
            return True, "/_query answered a one-row probe"
        return False, "/_query answered without rows"

    def _probe_aggregations(self, tables: tuple[EsTable, ...]) -> tuple[bool | None, str]:
        """Run a zero-hit filter aggregation on the first table: does ``aggs`` work here?"""
        if not tables:
            return None, "no index to probe aggregations on"
        target = tables[0].name
        body = {
            "size": 0,
            "track_total_hits": False,
            "aggs": {"mdqm_probe": {"filter": {"match_all": {}}}},
        }
        try:
            payload = self._json("GET", f"/{_path(target)}/_search", body=body)
        except ElasticError as error:
            return False, f"aggregations refused on {target} ({_describe(error)})"
        if isinstance(payload.get("aggregations"), Mapping):
            return True, f"filter aggregation on {target} answered"
        return False, f"aggregation response from {target} carried no aggregations"

    # --- SourceAdapter: catalog -----------------------------------------------------------

    def extract_catalog(self) -> CatalogSnapshot:
        """Aliases and lone indices as tables, their mappings as columns.

        The alias is the table on purpose: it is the name that survives monthly rotation, so an
        agent that queries it keeps reading new data, while the monthly indices are reported as
        that table's partitions. An index in no alias becomes a table of its own, so a family
        without an alias is visible rather than hidden.
        """
        tables = self._catalog_tables()
        return CatalogSnapshot(
            source=self._name,
            captured_at=self._clock(),
            tables=tuple(_table_model(table) for table in tables),
        )

    def _catalog_tables(self) -> tuple[EsTable, ...]:
        """Read aliases, indices and mappings once and cache the reading."""
        if self._tables is not None:
            return self._tables
        aliases = self._alias_members()
        docs, indices = self._index_docs()
        mappings = self._mappings(tuple(indices) or tuple(sorted(aliases)))
        self._tables = _build_tables(aliases, indices, docs, mappings, time_field=self._time_field)
        return self._tables

    def _alias_members(self) -> dict[str, tuple[str, ...]]:
        """alias name -> member indices, sorted; empty when the cluster has no alias."""
        payload = self._json("GET", "/_alias")
        members: dict[str, list[str]] = {}
        for index, meta in sorted(payload.items()):
            aliases = meta.get("aliases") if isinstance(meta, Mapping) else None
            if not isinstance(aliases, Mapping):
                continue
            for alias in aliases:
                members.setdefault(str(alias), []).append(str(index))
        return {alias: tuple(sorted(names)) for alias, names in sorted(members.items())}

    def _index_docs(self) -> tuple[dict[str, int], tuple[str, ...]]:
        """Document counts per open, non-system index, from ``_cat/indices``."""
        rows = self._json_list(
            "GET",
            "/_cat/indices",
            params={"format": "json", "h": "index,docs.count,status"},
        )
        docs: dict[str, int] = {}
        names: list[str] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("index", ""))
            if not name or name.startswith(".") or row.get("status", "open") != "open":
                continue
            names.append(name)
            try:
                docs[name] = int(row.get("docs.count") or 0)
            except (TypeError, ValueError):
                docs[name] = 0
        return docs, tuple(sorted(names))

    def _mappings(self, indices: Sequence[str]) -> dict[str, Mapping[str, Any]]:
        """``GET /{indices}/_mapping``: one mapping document per index."""
        if not indices:
            return {}
        payload = self._json("GET", f"/{_path(','.join(indices))}/_mapping")
        return {
            str(index): meta["mappings"]
            for index, meta in payload.items()
            if isinstance(meta, Mapping) and isinstance(meta.get("mappings"), Mapping)
        }

    def _table(self, name: str) -> EsTable:
        """The cached table ``name``, or ``UNKNOWN_TABLE`` naming the ones that exist."""
        tables = self._catalog_tables()
        found = next((table for table in tables if table.name == name), None)
        if found is None:
            raise ElasticError(unknown_table(name=name, known=[t.name for t in tables]))
        return found

    # --- SourceAdapter: partitions and plan -----------------------------------------------

    def extract_partitions(self) -> list[Partitioning]:
        """One entry per table: monthly range partitions when the family is named that way."""
        partitions: list[Partitioning] = []
        for table in self._catalog_tables():
            if table.granularity is None or table.time_field is None:
                partitions.append(Partitioning(strategy=PartitionStrategy.NONE))
                continue
            partitions.append(
                Partitioning(
                    strategy=PartitionStrategy.RANGE,
                    key=(table.time_field,),
                    granularity=table.granularity,
                    bounds=tuple(_bound(name, table.members) for name in table.members),
                )
            )
        return partitions

    def explain(self, fragment: Fragment) -> PlanSummary:
        """What the request will touch: the index set, not a plan (Elasticsearch has no EXPLAIN).

        ``pruning_proof`` stays ``UNKNOWN`` on purpose — an index list is a fact about this
        adapter's own narrowing, not a plan the engine produced — so the planner caps the scan by
        statistics instead of trusting a proof nobody gave.
        """
        request = self._parse_request(fragment)
        table = self._table(request.table)
        selected = self._selected_indices(table, request.query, None)
        return PlanSummary(
            partitions_scanned=len(selected),
            partitions_total=len(table.members),
            uses_index=len(selected) < len(table.members),
            indexes_used=selected,
            pruning_proof=PruningProof.UNKNOWN,
        )

    def mine_queries(self, since: datetime) -> Iterator[RawQuery]:
        """No query store: the slow log is off by default and is not a statement log."""
        raise NotSupported(
            OPTIONAL_METHODS["mine_queries"],
            "Elasticsearch has no query store; the slow log is opt-in and off by default",
        )

    def mutability_stats(self) -> list[TableStats]:
        """No change counters: Elasticsearch counts nothing per index."""
        raise NotSupported(
            OPTIONAL_METHODS["mutability_stats"],
            "Elasticsearch keeps no per-index insert/update/delete counters",
        )

    def sizes(self) -> list[TableSize]:
        """Document counts and store sizes per table, from one ``_cat/indices`` request.

        The table is the alias (the name that survives monthly rotation) and its partitions are the
        member indices, which is exactly how the catalog and the partition probe name them -- so a
        measured size can be matched to the bound it measures. A lone index is one unpartitioned
        table.

        ``store.size`` is the byte figure Elasticsearch itself reports; ``None`` means the cluster
        did not report one (a closed index, or a version that omits the field), never zero.
        """
        rows = self._json_list(
            "GET",
            "/_cat/indices",
            params={"format": "json", "bytes": "b", "h": "index,docs.count,store.size,status"},
        )
        return table_sizes(
            self._catalog_tables(),
            parse_index_sizes(rows),
            source=self._name,
            captured_at=self._clock(),
        )

    def _probe_sizes(self) -> tuple[bool, str]:
        """Claim ``size_stats`` only when ``_cat/indices`` really answers with numbers."""
        if not self._catalog_tables():
            return False, "no open index or alias is visible to this adapter"
        try:
            found = self.sizes()
        except (ElasticUnavailable, ElasticError, NotSupported) as exc:
            return False, f"index sizes could not be read: {exc}"
        if not found:
            return False, "no open index was reported, so there is nothing to size"
        measured = sum(1 for size in found if size.bytes is not None)
        return True, (
            "_cat/indices store.size (bytes) and docs.count per index: "
            f"{len(found)} table(s), {measured} with a byte figure, "
            f"{sum(len(size.partitions) for size in found)} partition(s)"
        )

    # --- SourceAdapter: execute ----------------------------------------------------------

    def execute(self, fragment: Fragment, limits: Limits) -> ArrowStream:
        """Run ``fragment`` as one ES|QL or Query DSL search, under ``limits``.

        The index set is chosen before any request: a range on the table's time field narrows the
        alias to the monthly indices it can touch, and a request that would read too many indices
        is refused here, so the cluster never runs it.
        """
        request = self._parse_request(fragment)
        table = self._table(request.table)
        selected = self._selected_indices(table, request.query, limits)
        deadline = time.monotonic() + limits.timeout_s
        if not selected:
            return self._empty_stream(table, request, limits, target=f"{table.name}:no-index")
        features = self.features()
        if request.esql is not None and features.esql:
            return self._execute_esql(request, table, selected, limits, deadline)
        return self._execute_dsl(request, table, selected, limits, deadline)

    def _execute_dsl(
        self,
        request: EsRequest,
        table: EsTable,
        selected: tuple[str, ...],
        limits: Limits,
        deadline: float,
    ) -> ArrowStream:
        """The Query DSL path: ``GET /{indices}/_search`` with ``size`` from the limits."""
        size = min(limits.max_rows, MAX_HITS_PER_REQUEST)
        body: dict[str, Any] = {
            "query": dict(request.query) if request.query is not None else {"match_all": {}},
            "size": size,
            "from": 0,
            # Counting the matches is what ``_count`` would do anyway; reporting a truncation
            # without the real total would make a capped result look complete.
            "track_total_hits": True,
        }
        if request.sort:
            body["sort"] = [dict(item) for item in request.sort]
        if request.fields:
            body["_source"] = list(request.fields)
        target = ",".join(selected)
        payload = self._json("GET", f"/{_path(target)}/_search", body=body, deadline=deadline)
        hits = _hits(payload)
        rows = [hit["_source"] for hit in hits if isinstance(hit.get("_source"), Mapping)]
        result, bad = _rows_to_table(rows, table.fields)
        capped, trimmed = _trim_bytes(result, limits.max_bytes)
        total = _total_hits(payload)
        took = _int(payload.get("took"))
        return _RowsStream(
            capped,
            schema=result.schema,
            fetched_at=self._clock(),
            index_target=target,
            touched=selected,
            took_ms=took,
            total_hits=total,
            esql=False,
            dropped_rows=trimmed,
            dropped_values=bad,
        )

    def _execute_esql(
        self,
        request: EsRequest,
        table: EsTable,
        selected: tuple[str, ...],
        limits: Limits,
        deadline: float,
    ) -> ArrowStream:
        """The ES|QL path: ``POST /_query`` with the statement the fragment carries.

        ES|QL is not SQL and sqlglot cannot parse it, so the execution chokepoint
        (:mod:`mdqm.sql.execution`) does not apply and no statement text is guarded here. What keeps
        this read-only is structural: the transport allows ``GET`` and ``POST /_query`` only, and the
        statement travels as a JSON *value* -- no value is ever interpolated into a request target.
        """
        payload = self._json(
            "POST",
            ESQL_PATH,
            # ES|QL takes no top-level ``time_zone`` (8.15: "unknown field [time_zone]"); its
            # datetime arithmetic is UTC unless the statement itself says otherwise.
            body={"query": request.esql or ""},
            deadline=deadline,
        )
        result, dropped, bad = _esql_table(payload, limits.max_rows)
        capped, trimmed = _trim_bytes(result, limits.max_bytes)
        if not capped.schema.names:
            capped = pa.Table.from_batches([], schema=_arrow_schema(table))
        return _RowsStream(
            capped,
            schema=capped.schema,
            fetched_at=self._clock(),
            index_target=request.table,
            touched=selected,
            took_ms=0,
            total_hits=capped.num_rows,
            esql=True,
            dropped_rows=dropped + trimmed,
            dropped_values=bad,
        )

    def _empty_stream(
        self, table: EsTable, request: EsRequest, limits: Limits, *, target: str
    ) -> ArrowStream:
        """Answer for a window outside every index: zero rows, zero requests."""
        schema = _arrow_schema(table)
        return _RowsStream(
            pa.Table.from_batches([], schema=schema),
            schema=schema,
            fetched_at=self._clock(),
            index_target=target,
            touched=(),
            took_ms=0,
            total_hits=0,
            esql=request.esql is not None and bool(self.features().esql),
            dropped_rows=0,
            dropped_values=0,
        )

    def cancel(self, handle: ArrowStream) -> None:
        """Stop the stream ``execute`` returned. Idempotent."""
        cancel = getattr(handle, "cancel", None)
        stop = cancel if callable(cancel) else getattr(handle, "close", None)
        if callable(stop):
            stop()

    # --- request validation ---------------------------------------------------------------

    def _parse_request(self, fragment: Fragment) -> EsRequest:
        """Validate the fragment's DSL payload; every refusal names the field and the fix."""
        dsl = fragment.dsl
        if not isinstance(dsl, Mapping):
            raise ElasticError(
                invalid_spec(
                    field="fragment.dsl",
                    reason="this source takes Query DSL (or ES|QL), not SQL",
                    expected=sorted(_DSL_KEYS),
                )
            )
        unknown = sorted(str(key) for key in dsl if key not in _DSL_KEYS)
        if unknown:
            raise ElasticError(
                invalid_spec(
                    field=f"dsl.{unknown[0]}",
                    reason="this adapter compiles only filters, sort and paging into a search",
                    expected=list(_DSL_KEYS),
                )
            )
        table = dsl.get("table")
        if not isinstance(table, str) or not table.strip():
            fallback = fragment.tables[0] if len(fragment.tables) == 1 else None
            if fallback is None:
                raise ElasticError(
                    invalid_spec(
                        field="dsl.table", reason="it is required", expected=["<table name>"]
                    )
                )
            table = fallback
        query = dsl.get("query")
        if query is not None and not isinstance(query, Mapping):
            raise ElasticError(
                invalid_spec(field="dsl.query", reason="it must be a Query DSL object")
            )
        esql = dsl.get("esql")
        if esql is not None and not isinstance(esql, str):
            raise ElasticError(invalid_spec(field="dsl.esql", reason="it must be a string"))
        sort = dsl.get("sort") or ()
        if not isinstance(sort, Sequence) or isinstance(sort, str | bytes):
            raise ElasticError(invalid_spec(field="dsl.sort", reason="it must be a list"))
        sort_items = tuple(item for item in sort if isinstance(item, Mapping))
        if len(sort_items) != len(sort):
            raise ElasticError(
                invalid_spec(field="dsl.sort", reason="every entry must be a sort object")
            )
        fields = dsl.get("fields") or ()
        if not isinstance(fields, Sequence) or isinstance(fields, str | bytes):
            raise ElasticError(invalid_spec(field="dsl.fields", reason="it must be a list"))
        return EsRequest(
            table=table,
            query=query,
            esql=esql,
            sort=sort_items,
            fields=tuple(str(name) for name in fields),
        )

    # --- index selection ------------------------------------------------------------------

    def _selected_indices(
        self, table: EsTable, query: Mapping[str, Any] | None, limits: Limits | None
    ) -> tuple[str, ...]:
        """The monthly indices this request can touch, narrowed by the time bound it carries.

        No bound is not "no answer": a family small enough to read whole is read whole, and one
        above the cap is refused with ``PARTITION_SCAN_ALL`` — before any request is sent.
        """
        if table.granularity is None or table.time_field is None:
            return table.members
        bounds = _time_bounds(query, table.time_field)
        if bounds is None:
            if limits is not None and len(table.members) > limits.max_partitions:
                raise ElasticError(
                    partition_scan_all(
                        table=table.name,
                        key=table.time_field,
                        granularity=table.granularity.value,
                        partitions_if_fixed=str(len(table.members)),
                    )
                )
            return table.members
        lower, upper = bounds
        selected = tuple(
            name for name in table.members if _overlaps(name, lower, upper, table.granularity)
        )
        if limits is not None and len(selected) > limits.max_partitions:
            raise ElasticError(
                cost_too_high(
                    table=table.name,
                    estimated=f"{len(selected)} monthly indices",
                    limit=f"{limits.max_partitions} partitions",
                )
            )
        return selected

    # --- HTTP -----------------------------------------------------------------------------

    def _json(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        deadline: float | None = None,
    ) -> Mapping[str, Any]:
        """One request whose answer must be a JSON object."""
        payload = self._payload(method, path, body=body, params=params, deadline=deadline)
        if not isinstance(payload, Mapping):
            raise ElasticUnavailable(source_unavailable(source=self._name))
        return payload

    def _json_list(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        deadline: float | None = None,
    ) -> list[Any]:
        """One request whose answer must be a JSON array (``_cat`` endpoints are arrays)."""
        payload = self._payload(method, path, params=params, deadline=deadline)
        if not isinstance(payload, list):
            raise ElasticUnavailable(source_unavailable(source=self._name))
        return payload

    def _payload(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        deadline: float | None = None,
    ) -> Any:
        """One request under the deadline; a failure is an ``ErrorResponse``, not a traceback."""
        client = self._client
        if client is None:
            raise ElasticUnavailable(source_unavailable(source=self._name))
        request_timeout = None
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ElasticError(timeout(seconds=None, target=f"{self._name} (elasticsearch)"))
            request_timeout = max(0.001, remaining)
        content = json.dumps(body, sort_keys=True) if body is not None else None
        try:
            response = client.request(
                method, path, params=params, content=content, timeout=request_timeout
            )
        except httpx.TimeoutException:
            raise ElasticError(
                timeout(seconds=request_timeout, target=f"{self._name} (elasticsearch)")
            ) from None
        except httpx.HTTPError:
            raise ElasticUnavailable(source_unavailable(source=self._name)) from None
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise ElasticError(_error_for(response, self._name, method, path))
        return response.json() if response.content else {}


# --- helpers ------------------------------------------------------------------------------


def _optional_str(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value.strip() else None


def _base_url(config: SourceConfig) -> str:
    """The origin from ``options``: ``base_url``, else host/port/scheme, else ES's own default."""
    base = _optional_str(config.options.get("base_url"))
    if base is not None:
        return base.rstrip("/")
    scheme = _optional_str(config.options.get("scheme")) or "http"
    host = _optional_str(config.options.get("host")) or DEFAULT_HOST
    raw_port = config.options.get("port", DEFAULT_PORT)
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        raise ElasticError(
            invalid_spec(field="options.port", reason="it must be an integer port number")
        ) from None
    return f"{scheme}://{host}:{port}"


def _auth_headers(config: SourceConfig, secrets: SecretProvider | None) -> dict[str, str]:
    """The configured auth header; the secret's *name* is reported, never its value."""
    auth = config.auth
    if auth is None:
        return {}
    if secrets is None:
        raise ElasticError(source_unavailable(source=config.name, secret_name=auth.secret))
    try:
        value = secrets.get(auth.secret)
    except SecretNotFound:
        raise ElasticError(
            source_unavailable(source=config.name, secret_name=auth.secret)
        ) from None
    return {auth.header: value}


def _path(target: str) -> str:
    """URL-encode an index expression for the path, keeping the separators that mean something."""
    return httpx.URL(f"/{target}").raw_path.decode("ascii").lstrip("/")


def _describe(error: ElasticError) -> str:
    """A short, safe description of a failed probe (the error's own problem line)."""
    return error.response.problem if hasattr(error.response, "problem") else error.code


def _error_for(response: httpx.Response, name: str, method: str, path: str) -> ErrorResponse:
    """Map one Elasticsearch error status to an agent-facing response, without leaking anything.

    A 4xx is the request's fault and the message carries the cluster's own reason, so the agent can
    fix it; 5xx/401/403 are the source's, and the detail goes to the log rather than to the agent
    (an auth failure message must never hint at a credential).
    """
    reason = _excerpt(response)
    if response.status_code < httpx.codes.INTERNAL_SERVER_ERROR and response.status_code not in (
        httpx.codes.UNAUTHORIZED,
        httpx.codes.FORBIDDEN,
    ):
        return invalid_spec(
            field="esql" if path == ESQL_PATH else "dsl",
            reason=f"{method} {path} failed: {reason}",
        )
    _log.warning(
        "elasticsearch refused %s %s: HTTP %s",
        method,
        path,
        response.status_code,
        extra={"source": name, "detail": reason},
    )
    return source_unavailable(source=name)


def _excerpt(response: httpx.Response, limit: int = 300) -> str:
    """A bounded, one-line slice of a response body for an error message that names the problem."""
    try:
        payload: Any = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping) and error.get("reason"):
            text = f"{error.get('type', 'error')}: {error['reason']}"
        else:
            text = json.dumps(error if error is not None else payload, ensure_ascii=False)
    elif payload is not None:
        text = json.dumps(payload, ensure_ascii=False)
    else:
        text = response.text
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _hits(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The hit list of a search response, defensively read."""
    node = payload.get("hits")
    if not isinstance(node, Mapping):
        return []
    hits = node.get("hits")
    if not isinstance(hits, list):
        return []
    return [hit for hit in hits if isinstance(hit, Mapping)]


def _total_hits(payload: Mapping[str, Any]) -> int:
    """``hits.total.value`` (8.x object form) as an int, 0 when it was not tracked."""
    node = payload.get("hits")
    if not isinstance(node, Mapping):
        return 0
    total = node.get("total")
    if isinstance(total, Mapping):
        return _int(total.get("value"))
    return _int(total)


def _flatten(
    properties: Mapping[str, Any], prefix: str = ""
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Mapped fields as flat ``a.b`` names; a nested object with ``properties`` is walked into."""
    for name, spec in properties.items():
        if not isinstance(spec, Mapping):
            continue
        full = f"{prefix}{name}"
        nested = spec.get("properties")
        if isinstance(nested, Mapping) and nested:
            # pi-lens-ignore: no-yield-from-non-iterable -- _flatten is a generator
            yield from _flatten(nested, f"{full}.")
        else:
            yield full, spec


def _fields_from_mappings(mappings: Sequence[Mapping[str, Any]]) -> tuple[EsField, ...]:
    """Union the member indices' mappings into one field list, sorted by name.

    A field mapped differently in two months (a mapping conflict) keeps the type of the
    alphabetically first index, so the same cluster always produces the same catalog.
    """
    merged: dict[str, Mapping[str, Any]] = {}
    for mapping in mappings:
        properties = mapping.get("properties")
        if not isinstance(properties, Mapping):
            continue
        for name, spec in _flatten(properties):
            merged.setdefault(name, spec)
    fields: list[EsField] = []
    for name in sorted(merged):
        spec = merged[name]
        es_type = str(spec.get("type", "object"))
        normalized, arrow_type = _FIELD_TYPES.get(es_type, (NormalizedType.OTHER, pa.string()))
        fields.append(
            EsField(
                name=name,
                es_type=es_type,
                normalized_type=normalized,
                arrow_type=arrow_type,
                aggregatable=es_type != "text",
            )
        )
    return tuple(fields)


def month_of(index: str) -> datetime | None:
    """The first instant of the month an index name promises, or ``None`` when it promises none."""
    match = _MONTHLY_INDEX.match(index)
    if match is None:
        return None
    try:
        return datetime(int(match.group("year")), int(match.group("month")), 1, tzinfo=UTC)
    except ValueError:
        return None


def _next_month(moment: datetime) -> datetime:
    return (
        moment.replace(year=moment.year + 1, month=1)
        if moment.month == 12
        else moment.replace(month=moment.month + 1)
    )


def parse_index_sizes(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[int | None, int | None]]:
    """``{index: (docs, bytes)}`` from ``_cat/indices`` rows asked for with ``bytes=b``.

    ``_cat`` renders every number as text, so both cells are parsed here and a value the cluster did
    not report (``"-"``, an empty cell, a missing key) becomes ``None`` -- "not reported", never zero.
    A system index (``.``-prefixed) and a closed one are skipped for the same reason the catalog
    skips them: they are not data an agent may query.
    """
    sizes: dict[str, tuple[int | None, int | None]] = {}
    for row in rows:
        name = _optional_str(row.get("index"))
        if name is None or name.startswith("."):
            continue
        if row.get("status", "open") != "open":
            continue
        sizes[name] = (_cat_number(row.get("docs.count")), _cat_number(row.get("store.size")))
    return sizes


def table_sizes(
    tables: Sequence[EsTable],
    sizes: Mapping[str, tuple[int | None, int | None]],
    *,
    source: str,
    captured_at: datetime,
) -> list[TableSize]:
    """One :class:`~mdqm.models.TableSize` per visible table, its member indices as partitions.

    An alias has no storage of its own -- its member indices do -- so the alias reports ``None`` for
    its own rows and bytes and the partitions carry the numbers, exactly the reading a partitioned
    PostgreSQL parent gets. A lone index is one unpartitioned table with its own figures.
    """
    out: list[TableSize] = []
    for table in tables:
        members = tuple(
            PartitionSize(name=member, rows=sizes[member][0], bytes=sizes[member][1])
            for member in table.members
            if member in sizes
        )
        own = sizes.get(table.members[0]) if not table.aliased and table.members else None
        out.append(
            TableSize(
                source=source,
                table=table.name,
                captured_at=captured_at,
                rows=None if own is None else own[0],
                bytes=None if own is None else own[1],
                partitions=members if table.aliased else (),
            )
        )
    return out


def _cat_number(value: Any) -> int | None:
    """One ``_cat`` cell as a non-negative integer, or ``None`` when it holds no number.

    ``_cat`` answers ``"-"`` or an empty string for "this cluster has no such figure", and reading
    that as ``0`` would make an unknown size indistinguishable from an empty index.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    text = str(value).strip()
    if not text or text == "-":
        return None
    try:
        number = int(text)
    except ValueError:
        return None
    return number if number >= 0 else None


def _build_tables(
    aliases: Mapping[str, tuple[str, ...]],
    indices: Sequence[str],
    docs: Mapping[str, int],
    mappings: Mapping[str, Mapping[str, Any]],
    *,
    time_field: str | None,
) -> tuple[EsTable, ...]:
    """Aliases as tables, every un-aliased index as its own table, in name order."""
    aliased = {name for members in aliases.values() for name in members}
    families: list[tuple[str, tuple[str, ...], bool]] = [
        (alias, members, True) for alias, members in aliases.items()
    ]
    families.extend((index, (index,), False) for index in indices if index not in aliased)
    tables: list[EsTable] = []
    for name, members, is_alias in sorted(families, key=lambda item: item[0]):
        fields = _fields_from_mappings(
            [mappings[member] for member in members if member in mappings]
        )
        granularity = (
            PartitionGranularity.MONTH
            if len(members) > 0 and all(month_of(member) is not None for member in members)
            else None
        )
        date_fields = [
            field.name for field in fields if field.normalized_type is NormalizedType.TIMESTAMP
        ]
        resolved_time = time_field or (date_fields[0] if len(date_fields) == 1 else None)
        if resolved_time is not None and resolved_time not in {f.name for f in fields}:
            resolved_time = None
        tables.append(
            EsTable(
                name=name,
                members=members,
                fields=fields,
                docs=sum(docs.get(member, 0) for member in members),
                aliased=is_alias,
                time_field=resolved_time,
                granularity=granularity,
            )
        )
    return tuple(tables)


def _table_model(table: EsTable) -> Table:
    """The domain ``Table`` for one index family: columns from the mapping, kind from its shape."""
    comment = (
        f"alias over {len(table.members)} index(es): {', '.join(table.members)}"
        if table.aliased
        else f"index {table.name}"
    )
    return Table(
        name=table.name,
        kind=TableKind.INDEX,
        columns=tuple(
            Column(
                name=field.name,
                native_type=field.es_type,
                normalized_type=field.normalized_type,
                nullable=True,
                ordinal=position,
            )
            for position, field in enumerate(table.fields, start=1)
        ),
        row_estimate=table.docs,
        comment=comment,
    )


def _arrow_schema(table: EsTable) -> pa.Schema:
    """The Arrow schema of a table: what a stream reports before its first batch."""
    return pa.schema([pa.field(field.name, field.arrow_type) for field in table.fields])


def _bound(name: str, members: Sequence[str]) -> PartitionBound:
    """One monthly index as a half-open ``[lower, upper)`` partition."""
    lower = month_of(name)
    upper = _next_month(lower) if lower is not None else None
    return PartitionBound(name=name, lower=lower, upper=upper)


def _overlaps(
    index: str, lower: datetime, upper: datetime, granularity: PartitionGranularity
) -> bool:
    """True when the month ``index`` names intersects ``[lower, upper)``."""
    if granularity is not PartitionGranularity.MONTH:
        return True
    start = month_of(index)
    if start is None:
        return True
    return start < upper and _next_month(start) > lower


def _time_bounds(query: Mapping[str, Any] | None, field: str) -> tuple[datetime, datetime] | None:
    """The ``[lower, upper)`` window a query bounds ``field`` to, or ``None`` when it bounds none.

    Reads ``range`` clauses anywhere in the query (``bool.filter``/``must``/``should``, nested),
    taking the tightest lower bound and the tightest upper bound. Absent an upper bound the window
    is open, which cannot narrow anything and is reported as no bound: guessing a width would
    silently drop rows.
    """
    ranges: list[Mapping[str, Any]] = []
    _collect_ranges(query, field, ranges)
    lowers: list[datetime] = []
    uppers: list[datetime] = []
    for clause in ranges:
        for key in ("gte", "gt", "from"):
            moment = _to_datetime(clause.get(key))
            if moment is not None:
                lowers.append(moment)
        for key in ("lte", "lt", "to"):
            moment = _to_datetime(clause.get(key))
            if moment is not None:
                uppers.append(moment)
    if not lowers or not uppers:
        return None
    return max(lowers), min(uppers)


def _collect_ranges(node: Any, field: str, out: list[Mapping[str, Any]]) -> None:
    """Collect every ``range`` clause on ``field`` in a query tree, in a stable order."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key == "range" and isinstance(value, Mapping):
                bounded = value.get(field)
                if isinstance(bounded, Mapping):
                    out.append(bounded)
                for name, spec in value.items():
                    if name != field and isinstance(spec, list):
                        out.extend(item for item in spec if isinstance(item, Mapping))
            elif isinstance(value, Mapping):
                _collect_ranges(value, field, out)
            elif isinstance(value, list):
                for item in value:
                    _collect_ranges(item, field, out)
    elif isinstance(node, list):
        for item in node:
            _collect_ranges(item, field, out)
