"""The conformance kit: the single thing a new adapter must pass (task F04).

An adapter is a plugin, so nothing in core can know how it behaves. This module is the shared
answer to "is this a real adapter?": one function, callable from a plain pytest test in any lane,
that exercises every promise of ``docs/DESIGN.md`` "Adapter interface" and fails with a message
that says which promise broke.

    def test_conformance() -> None:
        run_conformance(lambda: PostgresAdapter(config_for("crm"), secrets))

It only asserts; it imports no test framework, so ``src`` never depends on ``pytest``. Pass
``fragment=`` when a ``SELECT 1`` is not a valid query for the source (Elasticsearch DSL, REST
requests) and ``slow_fragment=`` when the adapter can be made to run longer than a timeout.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import NoReturn

from mdqm.adapters.base import (
    OPTIONAL_METHODS,
    NotSupported,
    SourceAdapter,
    supports_method,
)
from mdqm.models import (
    Capabilities,
    CatalogSnapshot,
    Fragment,
    Limits,
    Partitioning,
    PlanSummary,
    RawQuery,
    TableStats,
)

__all__ = [
    "DEFAULT_GRACE_S",
    "DEFAULT_LIMITS",
    "DEFAULT_SQL",
    "run_conformance",
]

_log = logging.getLogger(__name__)

#: Limits the kit uses when the caller does not care: generous, so only real cap violations fail.
DEFAULT_LIMITS = Limits(timeout_s=10.0, max_rows=100, max_bytes=1_000_000, max_partitions=8)

#: A query every SQL dialect can run, used when the caller passes no ``fragment``.
DEFAULT_SQL = "SELECT 1 AS one"

#: Extra wall-clock allowance on top of ``timeout_s`` (scheduling jitter, slow CI machines).
DEFAULT_GRACE_S = 10.0


def run_conformance(
    adapter_factory: Callable[[], SourceAdapter],
    *,
    fragment: Fragment | None = None,
    limits: Limits | None = None,
    slow_fragment: Fragment | None = None,
    grace_s: float = DEFAULT_GRACE_S,
) -> None:
    """Check one adapter end to end; raises ``AssertionError`` naming the broken promise.

    ``adapter_factory`` is a zero-argument callable returning a *connected, ready* adapter.
    ``fragment`` is a query the adapter can run; when omitted a ``SELECT 1`` fragment is built
    from ``adapter.dialect()`` (so non-SQL sources must pass their own). ``slow_fragment`` is a
    query that takes longer than a timeout; when given, the kit also proves the adapter honours
    ``Limits.timeout_s``.
    """
    adapter = adapter_factory()
    _check_identity(adapter)
    capabilities = _check_probe(adapter)
    _check_catalog(adapter, capabilities)
    runnable = fragment if fragment is not None else _sql_fragment(adapter)
    _check_optional_capabilities(adapter, capabilities, runnable)
    active_limits = limits or DEFAULT_LIMITS
    _check_execute(adapter, runnable, active_limits, grace_s)
    _check_cancel(adapter, runnable, active_limits)
    if slow_fragment is not None:
        _check_timeout(adapter, slow_fragment, active_limits, grace_s)


def _fail(message: str) -> NoReturn:
    """Fail with the kit's prefix, so the culprit is obvious in a shared test run."""
    raise AssertionError(f"adapter conformance: {message}")


def _check_identity(adapter: SourceAdapter) -> None:
    kind = getattr(adapter, "kind", None)
    if not isinstance(kind, str) or not kind.strip():
        _fail(f"kind must be a non-empty string, got {kind!r}")
    dialect = adapter.dialect()
    if not isinstance(dialect, str) or not dialect.strip():
        _fail(f"dialect() must return a sqlglot dialect name or 'dsl', got {dialect!r}")


def _check_probe(adapter: SourceAdapter) -> Capabilities:
    capabilities = adapter.probe()
    if not isinstance(capabilities, Capabilities):
        _fail(f"probe() must return Capabilities, got {type(capabilities).__name__}")
    if capabilities.adapter != adapter.kind:
        _fail(
            f"probe() reports adapter {capabilities.adapter!r} but the adapter's kind is "
            f"{adapter.kind!r}"
        )
    return capabilities


def _check_catalog(adapter: SourceAdapter, capabilities: Capabilities) -> None:
    snapshot = adapter.extract_catalog()
    if not isinstance(snapshot, CatalogSnapshot):
        _fail(f"extract_catalog() must return CatalogSnapshot, got {type(snapshot).__name__}")
    if snapshot.source != capabilities.source:
        _fail(
            f"extract_catalog() describes source {snapshot.source!r} but probe() reported "
            f"{capabilities.source!r}; both must name the same configured source"
        )
    if not snapshot.tables:
        _fail(
            "extract_catalog() returned no tables. An adapter must report a non-empty catalog "
            "(an empty source is a configuration/privilege problem, not a valid answer)"
        )
    for table in snapshot.tables:
        if not table.columns:
            _fail(f"table {table.qualified_name!r} has no columns")


def _sql_fragment(adapter: SourceAdapter) -> Fragment:
    dialect = adapter.dialect()
    if dialect == "dsl":
        _fail(
            "this adapter is not SQL (dialect 'dsl'), so the kit cannot build a query for it: "
            "pass fragment=<Fragment ...> to run_conformance()"
        )
    source = str(getattr(adapter, "source", adapter.kind))
    return Fragment(source=source, dialect=dialect, sql=DEFAULT_SQL, exec_ts=datetime.now(UTC))


def _check_optional_capabilities(
    adapter: SourceAdapter, capabilities: Capabilities, fragment: Fragment
) -> None:
    """Each optional method must work or say :class:`NotSupported` — never return garbage."""
    for method, capability in OPTIONAL_METHODS.items():
        if not supports_method(adapter, method):
            if capabilities.supports(capability):
                _fail(
                    f"probe() reports {capability!r} as supported, but the adapter has no "
                    f"{method}() method"
                )
            continue
        call = getattr(adapter, method)
        try:
            if method == "explain":
                result: object = call(fragment)
            elif method == "mine_queries":
                result = list(call(datetime.now(UTC)))
            else:
                result = list(call())
        except NotSupported as exc:
            if exc.capability != capability:
                _fail(
                    f"{method}() raised NotSupported({exc.capability!r}) but that method "
                    f"provides {capability!r}"
                )
            if capabilities.supports(capability):
                _fail(
                    f"{method}() raised NotSupported, yet probe() reports {capability!r} as "
                    "supported; the probe and the method must agree"
                )
            continue
        _check_optional_result(method, capability, result)


def _check_optional_result(method: str, capability: str, result: object) -> None:
    if method == "explain":
        if not isinstance(result, PlanSummary):
            _fail(f"explain() must return PlanSummary, got {type(result).__name__}")
        return
    expected: type[object] = {
        "extract_partitions": Partitioning,
        "mine_queries": RawQuery,
        "mutability_stats": TableStats,
    }[method]
    if not isinstance(result, list):
        _fail(f"{method}() must return a sequence, got {type(result).__name__}")
    for item in result:
        if not isinstance(item, expected):
            _fail(
                f"{method}() returned {type(item).__name__} where {expected.__name__} was "
                f"promised ({capability!r})"
            )


def _check_execute(
    adapter: SourceAdapter, fragment: Fragment, limits: Limits, grace_s: float
) -> None:
    import pyarrow as pa  # imported here: the kit is test-time code and pyarrow is heavy

    started = time.monotonic()
    stream = adapter.execute(fragment, limits)
    if not supports_method(stream, "__iter__") or not supports_method(stream, "close"):
        _fail(
            "execute() must return an ArrowStream: iterable of pyarrow.RecordBatch with "
            f"schema and close(); got {type(stream).__name__}"
        )
    if not hasattr(stream, "schema"):
        _fail(f"the stream from execute() exposes no schema ({type(stream).__name__})")
    rows = 0
    try:
        for batch in stream:
            if not isinstance(batch, pa.RecordBatch):
                _fail(
                    "the stream from execute() yielded "
                    f"{type(batch).__name__}, not pyarrow.RecordBatch"
                )
            rows += batch.num_rows
    finally:
        stream.close()
    elapsed = time.monotonic() - started
    if rows > limits.max_rows:
        _fail(
            f"execute() returned {rows} rows with max_rows={limits.max_rows}; the row cap must "
            "be enforced by the adapter, not by the caller"
        )
    if elapsed > limits.timeout_s + grace_s:
        _fail(
            f"execute() took {elapsed:.1f}s with timeout_s={limits.timeout_s}; a timeout must "
            "raise once the deadline passes"
        )


def _check_cancel(adapter: SourceAdapter, fragment: Fragment, limits: Limits) -> None:
    import pyarrow as pa

    stream = adapter.execute(fragment, limits)
    try:
        adapter.cancel(stream)
        adapter.cancel(stream)  # cancelling twice must be a no-op, not an error
    except Exception as exc:
        _fail(f"cancel() raised {type(exc).__name__}: {exc}")
    rows = 0
    try:
        for batch in stream:
            if not isinstance(batch, pa.RecordBatch):
                _fail(
                    "after cancel() the stream yielded "
                    f"{type(batch).__name__}, not pyarrow.RecordBatch"
                )
            rows += batch.num_rows
    except AssertionError:
        raise
    except Exception as exc:
        # Reporting a cancellation by raising is allowed; extra rows are still counted above.
        _log.debug("stream raised after cancel(): %s", type(exc).__name__)
    finally:
        stream.close()
    if rows > limits.max_rows:
        _fail(
            f"after cancel() the stream still produced {rows} rows with max_rows={limits.max_rows}"
        )


def _check_timeout(
    adapter: SourceAdapter, slow_fragment: Fragment, limits: Limits, grace_s: float
) -> None:
    tight = limits.model_copy(update={"timeout_s": min(limits.timeout_s, 0.05)})
    started = time.monotonic()
    raised: Exception | None = None
    try:
        stream = adapter.execute(slow_fragment, tight)
        try:
            for _ in stream:
                pass
        finally:
            # A cursor closed after a timeout is not the failure under test.
            with contextlib.suppress(Exception):
                stream.close()
    except Exception as exc:
        raised = exc
    elapsed = time.monotonic() - started
    if raised is None:
        _fail(
            f"execute() ignored timeout_s={tight.timeout_s} on the slow fragment: it completed "
            "instead of raising a timeout"
        )
    if isinstance(raised, NotSupported):
        _fail(
            "execute() raised NotSupported on the slow fragment; only optional methods may do "
            "that, a timeout must be reported as a timeout"
        )
    if elapsed > tight.timeout_s + grace_s:
        _fail(
            f"execute() raised after {elapsed:.1f}s with timeout_s={tight.timeout_s}: the "
            "deadline must be honoured, not checked afterwards"
        )
