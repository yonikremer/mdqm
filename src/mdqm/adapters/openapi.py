"""The OpenAPI adapter: REST endpoints as read-only virtual tables (task A06).

An internal API described by an OpenAPI 3.x document is a data source like any other, and a weak
agent should not have to learn it: it is a set of tables with columns, filterable "indexes" and
(because a REST collection answers in pages) a pagination style. This module turns exactly that
reading of the spec into a real :class:`~mdqm.adapters.base.SourceAdapter`.

What is a "table" here. A ``GET`` operation becomes a virtual table when it is a *collection*: no
path parameters (nothing the caller must substitute) and a success response that is a JSON array,
or an object holding one (``{"items": [...], "next_cursor": ...}``). Columns come from the array's
item schema, the query parameters become :class:`~mdqm.models.Index` entries — that is the
"filterable column" the planner reasons about — and the response envelope's own keys (``total``,
``next_cursor``) are read as paging metadata, never as columns. A single-object endpoint
(``/customers/{id}/score``) is not a table: it returns one record, not a set, so it is reported as
ignored rather than pretended into a one-row table.

Read-only, three ways (``AGENTS.md`` principle 5). The adapter only ever *builds* ``GET`` requests
— there is no method parameter to get wrong — and the transport it wraps refuses any other method
(:class:`GetOnlyTransport`), so a bug in future code cannot reach a mutating endpoint either. A
mutating operation in the spec (``post``, ``put``, ``patch``, ``delete``) is *never* registered as a
table and every one of them is listed in :attr:`OpenApiAdapter.ignored_operations` so the omission
is visible ("everything reported, nothing silent", principle 10).

Economy (``AGENTS.md`` principle 6, ``docs/DESIGN.md`` section 8). Identical calls inside a TTL are
answered from an in-memory cache with zero HTTP requests, and concurrent identical calls coalesce
behind one fetch — the same promise :mod:`mdqm.cache.singleflight` makes for the async server, in
the blocking shape this adapter's callers use. The optional rate limiter (``A05``) is consulted
before every request and told about a ``429`` so the whole pool backs off.

Errors are :class:`~mdqm.models.ErrorResponse` values the agent can act on: a missing or refused key
names the *secret reference* (never the value) with code ``SOURCE_UNAVAILABLE``; a spent budget is
``QUOTA_EXCEEDED``; a write attempt is ``WRITE_BLOCKED``. They travel on :class:`OpenApiError`,
which is also an ``OSError`` so the governor's existing mapping classifies it as a source failure
even if a caller does not read the carried response.

The spec is read in house — no ``openapi-core``/``prance`` dependency — because the parts needed
(operations, parameters, one schema level per column, ``$ref`` into ``components``) are small, and
an extra parser is a wheel that has to exist on the air-gapped mirror for every Python version.

Usage::

    adapter = OpenApiAdapter(config, secrets, spec=api.spec(), transport=transport)
    run_conformance(lambda: adapter, fragment=adapter.fragment("rates", currency="USD"))
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Final

import httpx
import pyarrow as pa

from mdqm.adapters.base import ArrowStream
from mdqm.config.secrets import SecretNotFound, SecretProvider
from mdqm.config.settings import SourceConfig
from mdqm.errors import (
    invalid_spec,
    quota_exceeded,
    source_unavailable,
    unknown_table,
    write_blocked,
)
from mdqm.models import (
    Capabilities,
    CatalogSnapshot,
    Column,
    ErrorResponse,
    Fragment,
    Index,
    Limits,
    NormalizedType,
    Table,
    TableKind,
)
from mdqm.ratelimit import RateLimiter, denial_error, parse_retry_after

__all__ = [
    "DEFAULT_CACHE_TTL_S",
    "DEFAULT_PAGE_SIZE",
    "MAX_PAGES",
    "GetOnlyTransport",
    "OpenApiAdapter",
    "OpenApiError",
    "OpenApiUnavailable",
    "SkippedOperation",
    "SpecColumn",
    "SpecEndpoint",
    "SpecPaging",
    "SpecParam",
    "read_endpoints",
    "read_spec",
]

_log = logging.getLogger(__name__)

#: How long an identical call is served from memory. Short on purpose: an API is a live source and
#: the agent can ask for a fresh answer; this only removes the repeat that costs the same bytes.
DEFAULT_CACHE_TTL_S: Final = 60.0

#: Rows asked for per request when the spec does not declare its own page-size parameter.
DEFAULT_PAGE_SIZE: Final = 100

#: Hard stop for a source that pages forever (a broker that always answers "there is more").
MAX_PAGES: Final = 1000

#: The methods a spec may declare. Only ``GET`` ever becomes executable; the rest are reported.
_READ_METHOD: Final = "get"
_WRITE_METHODS: Final[frozenset[str]] = frozenset(
    {"post", "put", "patch", "delete", "head", "options", "trace"}
)

#: Query parameters that mean "give me the next page", by paging style.
_CURSOR_PARAMS: Final[tuple[str, ...]] = ("cursor", "page_token", "next_token", "after")
_OFFSET_PARAMS: Final[tuple[str, ...]] = ("offset", "skip", "start")
_PAGE_PARAMS: Final[tuple[str, ...]] = ("page", "page_number", "pageno")
_LIMIT_PARAMS: Final[tuple[str, ...]] = ("limit", "page_size", "per_page", "size", "count")

#: Response envelope key that carries the next page's token, by name fragment.
_NEXT_KEY_HINTS: Final[tuple[str, ...]] = ("next_cursor", "next_token", "next_page", "next")

#: JSON Schema type -> the portable vocabulary and the Arrow type the batches carry.
_SCALARS: Final[Mapping[str, tuple[NormalizedType, pa.DataType]]] = {
    "string": (NormalizedType.STRING, pa.string()),
    "integer": (NormalizedType.INTEGER, pa.int64()),
    "number": (NormalizedType.FLOAT, pa.float64()),
    "boolean": (NormalizedType.BOOLEAN, pa.bool_()),
}


def _arrow_for(normalized: NormalizedType) -> pa.DataType:
    """The Arrow type a column of ``normalized`` is built as."""
    for candidate in (*_SCALARS.values(), (NormalizedType.DATE, pa.date32())):
        if candidate[0] is normalized:
            return candidate[1]
    if normalized is NormalizedType.TIMESTAMP:
        return pa.timestamp("us", tz="UTC")
    if normalized is NormalizedType.JSON:
        return pa.string()
    return pa.string()


class OpenApiError(Exception):
    """An agent-visible failure, carrying the :class:`~mdqm.models.ErrorResponse` to return.

    Adapters do not print and do not leak a driver message; the caller (the governor or the tool
    layer) hands ``response`` straight to the agent. ``code`` is repeated so a caller can branch
    without unpacking the model.
    """

    def __init__(self, response: ErrorResponse) -> None:
        self.response = response
        super().__init__(response.code)

    @property
    def code(self) -> str:
        """The stable error code, e.g. ``SOURCE_UNAVAILABLE``."""
        return self.response.code


class OpenApiUnavailable(OpenApiError, OSError):
    """The source did not answer (or refused the key).

    Also an :class:`OSError` so :func:`mdqm.errors.from_exception` still classifies it as a source
    failure for a caller that only knows the generic mapping.
    """


# --- the spec, read in house ---------------------------------------------------------------------


class _SpecError(ValueError):
    """The document is not a readable OpenAPI 3.x description (a config problem, not a query)."""

    def __init__(self, message: str, *, error: ErrorResponse | None = None) -> None:
        super().__init__(message)
        self.error = error
        """The agent-facing refusal, when the problem has one; ``None`` for a plain config error."""


@dataclass(frozen=True, slots=True)
class SpecColumn:
    """One response field, as the catalog and the Arrow schema need it."""

    name: str
    native_type: str
    normalized: NormalizedType


@dataclass(frozen=True, slots=True)
class SpecParam:
    """A declared query parameter: a filterable column in the planner's vocabulary."""

    name: str
    location: str
    required: bool


@dataclass(frozen=True, slots=True)
class SpecPaging:
    """How a collection answers in pages, detected from the parameters and the envelope."""

    style: str
    param: str | None = None
    next_key: str | None = None
    limit_param: str | None = None


@dataclass(frozen=True, slots=True)
class SpecEndpoint:
    """One GET collection: a virtual table."""

    table: str
    path: str
    columns: tuple[SpecColumn, ...]
    params: tuple[SpecParam, ...]
    paging: SpecPaging
    items_key: str | None


@dataclass(frozen=True, slots=True)
class SkippedOperation:
    """An operation in the document that is deliberately not exposed, and why (principle 10).

    The reason is a sentence a human or an agent can act on: "a path parameter means a lookup, not
    a set" is the fact that explains the missing table, while the label alone explains nothing.
    """

    label: str
    reason: str


def _resolve(spec: Mapping[str, Any], node: Any) -> Any:
    """Follow one ``$ref`` into the document (the only indirection an OpenAPI schema uses here)."""
    seen = 0
    while isinstance(node, Mapping) and "$ref" in node:
        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise _SpecError(f"only local $ref is supported, got {ref!r}")
        target: Any = spec
        for part in ref[2:].split("/"):
            if not isinstance(target, Mapping) or part not in target:
                raise _SpecError(f"$ref {ref!r} does not resolve in this document")
            target = target[part]
        node = target
        seen += 1
        if seen > 32:
            raise _SpecError(f"$ref {ref!r} nests too deeply (a cycle?)")
    return node


def _schema_kind(schema: Any) -> str | None:
    """The JSON type of a schema, reading through an ``anyOf``/``oneOf`` nullable union."""
    if not isinstance(schema, Mapping):
        return None
    kind = schema.get("type")
    if isinstance(kind, str):
        return kind
    for key in ("anyOf", "oneOf", "allOf"):
        options = schema.get(key)
        if isinstance(options, list):
            for option in options:
                found = _schema_kind(option)
                if found is not None and found != "null":
                    return found
    return None


def _column_from(name: str, schema: Any) -> SpecColumn:
    """Map one JSON Schema property onto a column, keeping the source's own type text."""
    resolved = schema if isinstance(schema, Mapping) else {}
    native = str(resolved.get("format") or _schema_kind(resolved) or "unknown")
    kind = _schema_kind(resolved)
    fmt = resolved.get("format")
    if kind == "string" and fmt == "date":
        return SpecColumn(name=name, native_type=native, normalized=NormalizedType.DATE)
    if kind == "string" and fmt in {"date-time", "datetime"}:
        return SpecColumn(name=name, native_type=native, normalized=NormalizedType.TIMESTAMP)
    if kind == "array":
        return SpecColumn(name=name, native_type=native, normalized=NormalizedType.ARRAY)
    if kind == "object" or kind is None:
        return SpecColumn(name=name, native_type=native, normalized=NormalizedType.JSON)
    normalized = _SCALARS.get(kind, (NormalizedType.OTHER, pa.string()))[0]
    return SpecColumn(name=name, native_type=native, normalized=normalized)


def _properties(spec: Mapping[str, Any], schema: Any) -> Mapping[str, Any]:
    """The object's declared properties, or an empty map when it declares none."""
    resolved = _resolve(spec, schema)
    if not isinstance(resolved, Mapping):
        return {}
    properties = resolved.get("properties")
    return properties if isinstance(properties, Mapping) else {}


def _array_key(spec: Mapping[str, Any], schema: Any) -> tuple[str | None, Any]:
    """Find the rows in a success schema: ``(envelope key or None, array schema or None)``."""
    resolved = _resolve(spec, schema)
    if _schema_kind(resolved) == "array":
        return None, resolved.get("items")
    for name, child in _properties(spec, resolved).items():
        if _schema_kind(_resolve(spec, child)) == "array":
            return name, _resolve(spec, child).get("items")
    return None, None


def _next_key(spec: Mapping[str, Any], schema: Any) -> str | None:
    """The envelope property that carries the next page's token, when the spec declares one."""
    for name in _properties(spec, schema):
        lowered = name.lower()
        if any(hint in lowered for hint in _NEXT_KEY_HINTS):
            return name
    return None


def _paging_for(params: Sequence[SpecParam], next_key: str | None) -> SpecPaging:
    """Detect the paging style from the declared parameters (and the envelope's next token).

    Deterministic: the first declared parameter naming a style wins, in the order cursor, offset,
    page. A collection declaring none of them is fetched in a single request.
    """
    names = {param.name.lower(): param.name for param in params}
    limit = next((names[n] for n in _LIMIT_PARAMS if n in names), None)
    for style, candidates, key in (
        ("cursor", _CURSOR_PARAMS, next_key),
        ("offset", _OFFSET_PARAMS, None),
        ("page", _PAGE_PARAMS, None),
    ):
        for candidate in candidates:
            if candidate in names:
                return SpecPaging(
                    style=style, param=names[candidate], next_key=key, limit_param=limit
                )
    return SpecPaging(style="none", limit_param=limit)


def _require_relative_path(path: str) -> None:
    """Refuse a ``paths`` key that is not a single-slash-relative path.

    ``httpx`` treats an absolute URL as the whole URL and *discards* ``base_url``, so a document
    whose key was ``https://attacker.example/x`` would send the source's auth header to that host
    with no redirect and no write. A path key must start with exactly one ``/`` and name no host.
    """
    if path.startswith("/") and not path.startswith("//") and "://" not in path:
        return
    raise _SpecError(
        f"the path key {path!r} is not a relative path: an OpenAPI path key starts with exactly "
        "one '/' and names no host (for example '/v1/customers'). fix: rewrite the key as a path "
        "on the document's servers[0].url, or put the host in options.base_url instead",
        error=invalid_spec(
            field="paths",
            reason=f"the path key {path!r} is not a relative path, so a request built from it "
            "would leave the configured base URL",
            expected=("/v1/collection", "/collection"),
        ),
    )


def _operations(spec: Mapping[str, Any]) -> Iterator[tuple[str, str, Mapping[str, Any]]]:
    """Every operation in the document as ``(path, method, operation)``, in document order."""
    paths = spec.get("paths")
    if not isinstance(paths, Mapping):
        raise _SpecError("the document has no 'paths' object")
    for path, item in paths.items():
        _require_relative_path(str(path))
        if not isinstance(item, Mapping):
            continue
        for method, operation in item.items():
            if not isinstance(operation, Mapping):
                continue
            yield str(path), str(method).lower(), operation


def _table_name(path: str) -> str:
    """A virtual table's name: the path without its leading slash, ``/`` folded to ``_``."""
    trimmed = path.strip("/")
    name = trimmed.replace("/", "_") if trimmed else path
    return name or "root"


def _success_schema(spec: Mapping[str, Any], operation: Mapping[str, Any]) -> Any:
    """The schema of the operation's 2xx JSON response (the first one declared)."""
    responses = operation.get("responses")
    if not isinstance(responses, Mapping):
        return None
    for status, response in sorted(responses.items(), key=lambda kv: str(kv[0])):
        if not str(status).startswith("2") or not isinstance(response, Mapping):
            continue
        content = response.get("content")
        if not isinstance(content, Mapping):
            continue
        body = content.get("application/json") or content.get("application/*+json")
        if isinstance(body, Mapping):
            return body.get("schema")
    return None


def _params_of(
    spec: Mapping[str, Any], path_item: Mapping[str, Any], operation: Mapping[str, Any]
) -> tuple[SpecParam, ...]:
    """The operation's declared parameters, path-item level ones included and de-duplicated."""
    found: dict[tuple[str, str], SpecParam] = {}
    for source in (path_item.get("parameters"), operation.get("parameters")):
        if not isinstance(source, list):
            continue
        for raw in source:
            declared = _resolve(spec, raw)
            if not isinstance(declared, Mapping):
                continue
            name = declared.get("name")
            location = declared.get("in")
            if not isinstance(name, str) or not isinstance(location, str):
                continue
            found[(name, location)] = SpecParam(
                name=name, location=location, required=bool(declared.get("required"))
            )
    return tuple(found[key] for key in sorted(found))


def read_endpoints(
    spec: Mapping[str, Any],
) -> tuple[tuple[SpecEndpoint, ...], tuple[SkippedOperation, ...]]:
    """Read the document into virtual tables plus the operations deliberately not exposed.

    Only ``GET`` collections become tables: a path parameter means the caller must substitute a
    value (a lookup, not a set) and a non-array response is one object, not rows. Everything skipped
    is named in the second return value *with a reason*, so the omission is reported rather than
    silent, and a reader (stage D4) can forward that reason instead of explaining it again.
    """
    paths = spec.get("paths")
    if not isinstance(paths, Mapping):
        raise _SpecError("the document has no 'paths' object")
    tables: list[SpecEndpoint] = []
    ignored: list[SkippedOperation] = []
    for path, item in paths.items():
        text = str(path)
        _require_relative_path(text)
        if not isinstance(item, Mapping):
            continue
        for method, operation in item.items():
            verb = str(method).lower()
            if not isinstance(operation, Mapping):
                continue
            label = f"{verb.upper()} {text}"
            if verb in _WRITE_METHODS:
                ignored.append(
                    SkippedOperation(
                        label=label,
                        reason=f"a {verb.upper()} operation: only GET is ever exposed",
                    )
                )
                continue
            if verb != _READ_METHOD:
                continue
            params = _params_of(spec, item, operation)
            if any(param.location == "path" for param in params):
                ignored.append(
                    SkippedOperation(
                        label=label,
                        reason="it takes a path parameter: one resource, not a collection",
                    )
                )
                continue
            schema = _success_schema(spec, operation)
            items_key, rows = _array_key(spec, schema)
            if rows is None:
                ignored.append(
                    SkippedOperation(
                        label=label,
                        reason="its success response is not a JSON array: one object, not rows",
                    )
                )
                continue
            columns = tuple(
                _column_from(name, child) for name, child in _properties(spec, rows).items()
            )
            if not columns:
                ignored.append(
                    SkippedOperation(
                        label=label,
                        reason="its array items declare no properties: no columns to filter on",
                    )
                )
                continue
            tables.append(
                SpecEndpoint(
                    table=_table_name(text),
                    path=text,
                    columns=columns,
                    params=params,
                    paging=_paging_for(params, _next_key(spec, schema)),
                    items_key=items_key,
                )
            )
    return tuple(tables), tuple(sorted(ignored, key=lambda item: (item.label, item.reason)))


# --- the transport guard -------------------------------------------------------------------------


class GetOnlyTransport(httpx.BaseTransport):
    """A transport that refuses every method but ``GET`` (``AGENTS.md`` principle 5).

    The adapter has no code path that would build another method; this wrapper is the layer that
    proves it, so a future change cannot quietly start writing to a source. ``close`` forwards, so
    the wrapped transport's own cleanup still runs.

    It also refuses a request whose host is not ``allowed_host`` (the configured base URL's host),
    because ``httpx`` drops ``base_url`` for an absolute URL. That check lives here, not in
    ``_get``, so a subclass overriding the request path cannot skip it.
    """

    def __init__(self, inner: httpx.BaseTransport, *, allowed_host: str | None = None) -> None:
        self._inner = inner
        self._allowed_host = allowed_host

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != "GET":
            raise OpenApiError(write_blocked(operation=request.method))
        if self._allowed_host is not None and request.url.host != self._allowed_host:
            raise OpenApiError(
                invalid_spec(
                    field="options.base_url",
                    reason="a request would leave the configured base URL's host",
                )
            )
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


# --- the stream ----------------------------------------------------------------------------------


class _RowsStream:
    """The ``ArrowStream`` an ``execute`` returns: the already-fetched, capped batches.

    The fetch happens in :meth:`OpenApiAdapter.execute` — under the row cap and the deadline — so a
    cancelled stream simply stops yielding. Batches are yielded one at a time so the governor can
    stop early and ``cancel`` has something to interrupt.
    """

    def __init__(self, table: pa.Table, *, fetched_at: datetime, cached: bool) -> None:
        self.schema: pa.Schema = table.schema
        self._batches = table.to_batches()
        self.fetched_at = fetched_at
        """When the bytes behind this stream left the source (a cache hit keeps the first fetch)."""
        self.cached = cached
        """True when no HTTP request was made for this stream."""
        self.cancelled = False

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        for batch in self._batches:
            if self.cancelled:
                return
            yield batch

    def cancel(self) -> None:
        """Stop yielding. Idempotent."""
        self.cancelled = True

    def close(self) -> None:
        """Release the batches. Idempotent."""
        self.cancelled = True
        self._batches = []


# --- the adapter ---------------------------------------------------------------------------------


@dataclass(slots=True)
class _CacheEntry:
    table: pa.Table
    fetched_at: datetime


class OpenApiAdapter:
    """A read-only ``SourceAdapter`` over an OpenAPI-described HTTP API (``kind="openapi"``).

    Args:
        config: the source's configuration; ``name`` labels the source, ``auth`` names the header
            and the *secret reference* the key comes from, ``options["base_url"]`` overrides the
            spec's ``servers[0].url``.
        secrets: resolves the auth secret reference; ``None`` means no header is sent.
        spec: the parsed OpenAPI document. When omitted it is read from ``config.spec`` as YAML or
            JSON (a path), which is how a deployed source is configured.
        transport: the HTTP transport to use; wrapped in :class:`GetOnlyTransport`. Tests inject
            one over the in-process ASGI app, so no port and no socket is needed.
        limiter: the shared rate limiter (``A05``); ``None`` leaves the source unmetered.
        cache_ttl_s: how long an identical call is served from memory.
        clock: UTC clock for ``retrieved_at``/``fetched_at``; ``monotonic`` for the deadline.
        page_size: rows requested per page when the spec declares no page-size parameter.
    """

    kind: Final[str] = "openapi"

    def __init__(
        self,
        config: SourceConfig | None = None,
        secrets: SecretProvider | None = None,
        *,
        spec: Mapping[str, Any] | None = None,
        transport: httpx.BaseTransport | None = None,
        limiter: RateLimiter | None = None,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
        page_size: int = DEFAULT_PAGE_SIZE,
        timeout_s: float = 30.0,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if cache_ttl_s < 0:
            raise ValueError(f"cache_ttl_s must be >= 0, got {cache_ttl_s}")
        if page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {page_size}")
        self._name = config.name if config is not None else self.kind
        if config is not None and config.adapter != self.kind:
            raise ValueError(
                f"config names adapter {config.adapter!r} but this is the {self.kind!r} adapter"
            )
        self._secrets = secrets
        self._auth = config.auth if config is not None else None
        self._secret_ref = (
            self._auth.secret if self._auth is not None else (config.secret if config else None)
        )
        self._cache_ttl_s = cache_ttl_s
        self._page_size = page_size
        self._timeout_s = timeout_s
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic
        self._limiter = limiter

        document = dict(spec) if spec is not None else read_spec(config)
        self._spec = document
        self._spec_tables, self._skipped = read_endpoints(document)
        self._by_table: dict[str, SpecEndpoint] = {e.table: e for e in self._spec_tables}
        self._base_url = _base_url(config, document)

        self.ignored_operations: tuple[str, ...] = tuple(item.label for item in self._skipped)
        """Operations in the spec that are deliberately not reachable (writes and lookups)."""

        self.skipped_operations: tuple[SkippedOperation, ...] = self._skipped
        """The same operations with the reason each one is not a table."""

        self._client = httpx.Client(
            base_url=self._base_url,
            transport=GetOnlyTransport(
                transport or httpx.HTTPTransport(), allowed_host=httpx.URL(self._base_url).host
            ),
            timeout=timeout_s,
            follow_redirects=False,
        )
        self._cache: dict[tuple[Any, ...], _CacheEntry] = {}
        self._cache_lock = threading.Lock()
        self._key_locks: dict[tuple[Any, ...], threading.Lock] = {}
        self._key_locks_guard = threading.Lock()

        #: Every HTTP request issued, in order, as ``"GET <path>"``. What the cache tests assert on.
        self.requests: list[str] = []

    # --- construction helpers ---------------------------------------------------------------

    def fragment(self, table: str, **params: Any) -> Fragment:
        """Build the request fragment for ``table``: the shape this adapter's ``execute`` reads.

        The compiler for API sources is another lane; until it exists this is the documented shape,
        and it is what the conformance kit runs.
        """
        return self._fragment(table, params, exec_ts=self._clock())

    def _fragment(self, table: str, params: Mapping[str, Any], *, exec_ts: datetime) -> Fragment:
        return Fragment(
            source=self._name,
            dialect=self.dialect(),
            request={"table": table, "params": dict(params)},
            tables=(table,),
            exec_ts=exec_ts,
        )

    def close(self) -> None:
        """Close the HTTP client. Idempotent."""
        client, self._client = self._client, None  # type: ignore[assignment]
        if client is not None:
            client.close()

    # --- SourceAdapter: identity ------------------------------------------------------------

    def dialect(self) -> str:
        """``dsl``: a fragment carries an HTTP request, not SQL (``docs/DESIGN.md``)."""
        return "dsl"

    @property
    def source(self) -> str:
        """The configured source name; the catalog and the probe both report it."""
        return self._name

    # --- SourceAdapter: probe ---------------------------------------------------------------

    def probe(self) -> Capabilities:
        """Report what this source can do: HTTP, so no plan, no partition metadata, no query store.

        Nothing is claimed that the adapter cannot serve: it implements no optional method, so no
        optional capability is reported as supported (``docs/DESIGN.md`` "Capabilities and
        fallbacks" then runs each feature's declared fallback).
        """
        info = self._spec.get("info")
        version = info.get("version") if isinstance(info, Mapping) else None
        return Capabilities(
            source=self._name,
            adapter=self.kind,
            version=str(version) if version is not None else None,
            retrieved_at=self._clock(),
            capabilities={},
        )

    # --- SourceAdapter: catalog -------------------------------------------------------------

    def extract_catalog(self) -> CatalogSnapshot:
        """The virtual tables the spec describes, with query parameters as their indexes."""
        tables = tuple(
            Table(
                name=endpoint.table,
                kind=TableKind.ENDPOINT,
                columns=tuple(
                    Column(
                        name=column.name,
                        native_type=column.native_type,
                        normalized_type=column.normalized,
                        ordinal=position,
                    )
                    for position, column in enumerate(endpoint.columns, start=1)
                ),
                indexes=tuple(
                    Index(
                        name=param.name,
                        columns=(param.name,),
                        index_type=f"{param.location}_param",
                    )
                    for param in endpoint.params
                    if param.location == "query"
                ),
                comment=f"{_READ_METHOD.upper()} {endpoint.path}",
            )
            for endpoint in self._spec_tables
        )
        return CatalogSnapshot(source=self._name, captured_at=self._clock(), tables=tables)

    # --- SourceAdapter: execute / cancel ----------------------------------------------------

    def execute(self, fragment: Fragment, limits: Limits) -> ArrowStream:
        """Fetch ``fragment``'s table, following the spec's paging, under ``limits``.

        The row cap is enforced here rather than by the caller, and the deadline is checked between
        pages, so a slow source stops at the limit instead of after it.
        """
        table_name, params = self._parse_request(fragment)
        endpoint = self._by_table[table_name]
        self._validate_params(endpoint, params)
        key = (table_name, tuple(sorted(params.items())), limits.max_rows)
        cached = self._cached(key)
        if cached is not None:
            return self._stream(cached.table, fetched_at=cached.fetched_at, cached=True)
        with self._key_lock(key):
            # A concurrent identical call may have filled the cache while this one waited.
            cached = self._cached(key)
            if cached is not None:
                return self._stream(cached.table, fetched_at=cached.fetched_at, cached=True)
            fetched_at = self._clock()
            table = self._fetch(endpoint, params, limits)
            self._remember(key, _CacheEntry(table=table, fetched_at=fetched_at))
            return self._stream(table, fetched_at=fetched_at, cached=False)

    def cancel(self, handle: ArrowStream) -> None:
        """Stop the stream ``execute`` returned. Idempotent."""
        close = getattr(handle, "cancel", None)
        stop = close if callable(close) else getattr(handle, "close", None)
        if callable(stop):
            stop()

    # --- request parsing --------------------------------------------------------------------

    def _parse_request(self, fragment: Fragment) -> tuple[str, dict[str, Any]]:
        """Read ``{"table", "params"}`` out of the fragment, refusing anything else."""
        request = fragment.request or {}
        table_name = request.get("table")
        if not isinstance(table_name, str) or not table_name:
            table_name = fragment.tables[0] if fragment.tables else None
        if not isinstance(table_name, str) or not table_name:
            raise OpenApiError(invalid_spec(field="request.table", reason="it is required"))
        if table_name not in self._by_table:
            raise OpenApiError(unknown_table(name=table_name, known=tuple(sorted(self._by_table))))
        raw = request.get("params", {})
        if not isinstance(raw, Mapping):
            raise OpenApiError(invalid_spec(field="request.params", reason="it must be an object"))
        return table_name, {str(name): value for name, value in raw.items()}

    def _validate_params(self, endpoint: SpecEndpoint, params: Mapping[str, Any]) -> None:
        """Refuse a parameter the spec does not declare: only declared filters reach the source."""
        declared = {param.name for param in endpoint.params if param.location == "query"}
        unknown = sorted(name for name in params if name not in declared)
        if unknown:
            raise OpenApiError(
                invalid_spec(
                    field=f"request.params.{unknown[0]}",
                    reason="this endpoint declares no such query parameter",
                    expected=tuple(sorted(declared)),
                )
            )

    # --- the fetch ----------------------------------------------------------------------------

    def _fetch(self, endpoint: SpecEndpoint, params: Mapping[str, Any], limits: Limits) -> pa.Table:
        """Page through the endpoint until the rows end, the cap is hit or the deadline passes."""
        deadline = self._monotonic() + limits.timeout_s
        query: dict[str, Any] = dict(params)
        paging = endpoint.paging
        if paging.style == "none" and paging.limit_param is not None:
            query[paging.limit_param] = min(self._page_size, limits.max_rows)
        elif paging.limit_param is not None:
            query.setdefault(paging.limit_param, min(self._page_size, limits.max_rows))
        rows: list[Mapping[str, Any]] = []
        for _ in range(MAX_PAGES):
            body = self._get(endpoint.path, query, deadline=deadline, timeout_s=limits.timeout_s)
            page = body.get(endpoint.items_key) if endpoint.items_key is not None else body
            if not isinstance(page, list):
                raise OpenApiError(
                    invalid_spec(
                        field="request.table",
                        reason="the source answered without the array this table is made of",
                    )
                )
            rows.extend(row for row in page if isinstance(row, Mapping))
            if len(rows) >= limits.max_rows:
                break
            following = self._next_page(endpoint, page, body, query)
            if following is None:
                break
            query = following
        return self._table(endpoint, rows[: limits.max_rows])

    def _next_page(
        self,
        endpoint: SpecEndpoint,
        page: Sequence[Any],
        body: Mapping[str, Any],
        query: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """The parameters for the next request, or ``None`` when this was the last page."""
        paging = endpoint.paging
        if not page:
            return None
        if paging.style == "cursor":
            if paging.next_key is None or paging.param is None:
                return None
            token = body.get(paging.next_key)
            if token is None or token == "":
                return None
            return {**query, paging.param: token}
        if paging.style == "offset" and paging.param is not None:
            step = len(page)
            current = query.get(paging.param, 0)
            return {**query, paging.param: (current if isinstance(current, int) else 0) + step}
        if paging.style == "page" and paging.param is not None:
            current = query.get(paging.param, 1)
            return {**query, paging.param: (current if isinstance(current, int) else 1) + 1}
        return None

    def _get(
        self,
        path: str,
        query: Mapping[str, Any],
        *,
        deadline: float,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        """Issue one ``GET``, metered and mapped onto the agent-facing error taxonomy."""
        if self._monotonic() > deadline:
            raise TimeoutError(f"{path} did not answer within {timeout_s}s")
        self._meter(path)
        client = self._client
        if client is None:
            raise OpenApiError(source_unavailable(source=self._name, secret_name=self._secret_ref))
        try:
            response = client.get(path, params=dict(query), headers=self._headers())
        except OpenApiError:
            raise
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"{path} did not answer within {timeout_s}s") from exc
        except httpx.HTTPError as exc:
            _log.warning(
                "the API source did not answer",
                extra={"source": self._name, "exception_type": type(exc).__name__},
            )
            raise OpenApiUnavailable(
                source_unavailable(source=self._name, secret_name=self._secret_ref)
            ) from exc
        self.requests.append(f"GET {path}")
        self._check_status(response)
        body = response.json()
        if not isinstance(body, Mapping):
            raise OpenApiError(
                invalid_spec(field="request.table", reason="the source answered with a non-object")
            )
        return body

    def _check_status(self, response: httpx.Response) -> None:
        """Map a non-2xx answer onto an ``ErrorResponse`` (never onto a driver message)."""
        status = response.status_code
        if 200 <= status < 300:
            return
        if status in (401, 403):
            raise OpenApiUnavailable(
                source_unavailable(source=self._name, secret_name=self._secret_ref)
            )
        if status == 429:
            retry_after = parse_retry_after(response.headers.get("Retry-After"), now=self._clock())
            if self._limiter is not None:
                self._limiter.report_429(self._name, retry_after)
            wait = f"in {retry_after:.1f}s" if retry_after is not None else "when its budget resets"
            raise OpenApiError(
                quota_exceeded(
                    quota=f'the request budget of "{self._name}"',
                    limit="the source's published limit",
                    resets_at=wait,
                )
            )
        raise OpenApiUnavailable(
            source_unavailable(source=self._name, secret_name=self._secret_ref)
        )

    def _headers(self) -> dict[str, str]:
        """The auth header, resolved from the secret provider; the value is never logged."""
        if self._auth is None:
            return {}
        if self._secrets is None:
            raise OpenApiUnavailable(
                source_unavailable(source=self._name, secret_name=self._auth.secret)
            )
        try:
            value = self._secrets.get(self._auth.secret)
        except SecretNotFound:
            # The reference is reported; the value never exists here to leak.
            raise OpenApiUnavailable(
                source_unavailable(source=self._name, secret_name=self._auth.secret)
            ) from None
        return {self._auth.header: value}

    def _meter(self, path: str) -> None:
        """Spend one request from the shared budget, refusing when the wait is not affordable."""
        if self._limiter is None:
            return
        decision = self._limiter.acquire(self._name)
        if not decision.granted:
            raise OpenApiError(denial_error(decision))

    # --- rows -> Arrow ------------------------------------------------------------------------

    def _table(self, endpoint: SpecEndpoint, rows: Sequence[Mapping[str, Any]]) -> pa.Table:
        """Build the batch's Arrow table from JSON objects, using the spec's declared types."""
        arrays: dict[str, pa.Array] = {}
        for column in endpoint.columns:
            values = [_coerce(row.get(column.name), column.normalized) for row in rows]
            arrays[column.name] = pa.array(values, type=_arrow_for(column.normalized))
        return pa.table(arrays)

    # --- the cache and the coalescer ----------------------------------------------------------

    def _cached(self, key: tuple[Any, ...]) -> _CacheEntry | None:
        """The cached fetch for ``key`` while it is inside the TTL, else ``None``."""
        if self._cache_ttl_s <= 0:
            return None
        with self._cache_lock:
            entry = self._cache.get(key)
        if entry is None:
            return None
        age_s = (self._clock() - entry.fetched_at).total_seconds()
        if age_s > self._cache_ttl_s:
            with self._cache_lock:
                self._cache.pop(key, None)
            return None
        return entry

    def _remember(self, key: tuple[Any, ...], entry: _CacheEntry) -> None:
        with self._cache_lock:
            self._cache[key] = entry

    def _key_lock(self, key: tuple[Any, ...]) -> threading.Lock:
        """One lock per call key: identical concurrent calls coalesce into one fetch (``C13``'s
        promise, in the blocking shape this adapter's callers use)."""
        with self._key_locks_guard:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def _stream(self, table: pa.Table, *, fetched_at: datetime, cached: bool) -> _RowsStream:
        return _RowsStream(table, fetched_at=fetched_at, cached=cached)


# --- module helpers ------------------------------------------------------------------------------


def _coerce(value: Any, normalized: NormalizedType) -> Any:
    """Convert one JSON value to the Python type its column was declared as.

    Only the two wire forms that differ from their Arrow type are converted: a JSON date/datetime
    is text, and Arrow wants ``date``/``datetime``. Anything else is passed through, so an
    unexpected value surfaces as an Arrow conversion error rather than a silent truncation.
    """
    if value is None:
        return None
    if normalized is NormalizedType.DATE:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value[:10])
        return None
    if normalized is NormalizedType.TIMESTAMP:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        return None
    return value


def read_spec(config: SourceConfig | None) -> dict[str, Any]:
    """Read the spec named by an OpenAPI source's config (YAML or JSON, one parser).

    YAML 1.2 is a superset of JSON, so :mod:`ruamel.yaml` reads both; no second format-specific
    branch is needed.
    """
    from pathlib import Path

    from ruamel.yaml import YAML

    path = Path(config.spec) if config is not None and config.spec else None
    if path is None:
        raise _SpecError(
            "an openapi source needs a spec: set 'spec' in the source config, or pass spec= "
            "the parsed document"
        )
    if not path.is_file():
        raise _SpecError(f"the OpenAPI spec {path} does not exist")
    try:
        loaded = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # ruamel raises several unrelated types for a broken document
        raise _SpecError(f"the OpenAPI spec {path} is not readable: {type(exc).__name__}") from None
    if not isinstance(loaded, Mapping):
        raise _SpecError(f"the OpenAPI spec {path} is not a mapping")
    return {str(key): value for key, value in loaded.items()}


#: Hosts a plaintext ``http://`` base URL is allowed for: a sidecar on the same machine cannot be
#: eavesdropped on the way.
_LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset({"127.0.0.1", "::1", "localhost"})


def _checked_base_url(url: str, *, origin: str) -> str:
    """Refuse a plaintext base URL: the source's auth header would travel unencrypted.

    A loopback ``http://`` URL is allowed because the request never leaves the host.
    """
    try:
        parsed = httpx.URL(url)
    except Exception:  # httpx raises several types for an unparseable URL
        raise _SpecError(
            f"the base URL {url!r} from {origin} is not a URL. fix: use an https:// URL such as "
            "https://api.internal"
        ) from None
    if parsed.scheme == "https" or (
        parsed.scheme == "http" and (parsed.host or "") in _LOOPBACK_HOSTS
    ):
        return url
    raise _SpecError(
        f"the base URL {url!r} from {origin} is not https. fix: use an https:// URL, or a "
        "loopback http:// URL (127.0.0.1, ::1, localhost) for a sidecar on the same host",
        error=invalid_spec(
            field="options.base_url",
            reason="it is not an https:// URL (only a loopback host may use http://)",
            expected=("https://api.internal", "http://127.0.0.1:8000"),
        ),
    )


def _base_url(config: SourceConfig | None, spec: Mapping[str, Any]) -> str:
    """The base URL: the source's configured override, else the spec's first server."""
    override = config.options.get("base_url") if config is not None else None
    if isinstance(override, str) and override:
        return _checked_base_url(override, origin="options.base_url")
    servers = spec.get("servers")
    if isinstance(servers, list):
        for server in servers:
            if isinstance(server, Mapping) and isinstance(server.get("url"), str):
                url = str(server["url"])
                if url.startswith(("http://", "https://")):
                    return _checked_base_url(url, origin="servers[0].url")
    # A spec with a relative server URL needs an explicit base_url; refuse it loudly, not at request
    # time (a request without a base would raise a message the agent cannot act on).
    raise _SpecError(
        "cannot determine the API's base URL: set options.base_url on the source, or declare an "
        "absolute servers[0].url in the spec"
    )
