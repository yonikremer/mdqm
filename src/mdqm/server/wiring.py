"""Build the query pipeline from deployment settings.

The tools are useless without this: ``query`` refuses with "no query pipeline is configured" until
something hands it a `QueryServices`. That "something" is production code, not a test, so it lives
here -- one function, called by ``serve()``, that turns settings into the object the tool needs.

Two rules it keeps:

* **every source goes through the governor.** The adapters are built once and handed to the
  governor; nothing else gets a direct path to a source.
* **a broken source fails loudly at wiring time**, naming the source and the reason. A server that
  starts with half its sources silently missing would answer "unknown table" for the rest of the run
  and blame the agent's spec.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from mdqm.adapters.base import SourceAdapter
from mdqm.adapters.registry import AdapterRegistry
from mdqm.config.settings import Settings
from mdqm.executor.cache import FragmentCache
from mdqm.executor.federate import FederatedExecutor
from mdqm.governor.clock import Clock
from mdqm.governor.core import Governor
from mdqm.governor.handles import ResultStore
from mdqm.models import Limits
from mdqm.semantic.models import SemanticLayer
from mdqm.server.jobs import DEFAULT_MAX_JOBS, JobRegistry
from mdqm.server.tools.misc import ToolServices
from mdqm.server.tools.query import QueryServices
from mdqm.store.audit import AuditLog
from mdqm.store.db import Store
from mdqm.store.feedback import FeedbackStore
from mdqm.store.history import HistoryStore
from mdqm.store.jobs import QueryJobStore
from mdqm.store.sizes import SizeStore

_log = logging.getLogger(__name__)

DEFAULT_QUERY_TIMEOUT_S = 60.0
DEFAULT_MAX_ROWS = 10_000
CACHE_FILE = "fragments.duckdb"


class WiringError(RuntimeError):
    """A source could not be built, or the pipeline cannot be assembled from these settings."""


def build_adapters(settings: Settings) -> dict[str, SourceAdapter]:
    """One adapter per configured source, or a `WiringError` naming the one that failed."""
    registry = AdapterRegistry(secrets=settings.secret_provider())
    # Discovery is explicit (F04): the entry-point group is what a real deployment reads, so ask for
    # it here and say what was found -- an empty group is why `query` would report "no adapter".
    found = registry.discover()
    _log.info("adapter kinds discovered: %s", ", ".join(found) or "none")
    if not found:
        raise WiringError(
            "no adapters are registered in this build (entry-point group mdqm.adapters); "
            "install the package that provides them"
        )
    adapters: dict[str, SourceAdapter] = {}
    for config in settings.sources:
        try:
            adapters[config.name] = registry.create(config)
        except Exception as exc:
            raise WiringError(f"source {config.name!r}: {type(exc).__name__}: {exc}") from exc
    return adapters


def catalog_columns(adapters: Mapping[str, SourceAdapter]) -> dict[str, tuple[str, ...]]:
    """``{source.table: (column, ...)}`` from each source's own catalog.

    A source whose catalog cannot be read contributes no columns and is reported: the compiler then
    refuses a spec that names its tables, which is honest -- better than compiling against a table
    name nobody verified.
    """
    columns: dict[str, tuple[str, ...]] = {}
    for name, adapter in adapters.items():
        try:
            snapshot = adapter.extract_catalog()
        except Exception as exc:
            _log.warning(
                "source %s: no catalog, so its tables cannot be compiled (%s: %s)",
                name,
                type(exc).__name__,
                exc,
            )
            continue
        for table in snapshot.tables:
            columns[f"{name}.{table.name}"] = tuple(column.name for column in table.columns)
    return columns


def query_limits(settings: Settings) -> Limits:
    """The caps one fragment runs under, from settings where they are declared."""
    limits = settings.limits
    return Limits(
        timeout_s=float(getattr(limits, "query_timeout_s", DEFAULT_QUERY_TIMEOUT_S)),
        max_rows=int(getattr(limits, "max_rows", DEFAULT_MAX_ROWS)),
        max_bytes=int(getattr(limits, "max_bytes", 64 << 20)),
        max_partitions=int(getattr(limits, "max_partitions_scanned", 8)),
    )


def open_cache(cache_dir: Path, clock: Clock | None = None) -> FragmentCache:
    """A DuckDB fragment cache inside the writable cache directory, with no extension loading."""
    import duckdb

    cache_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(cache_dir / CACHE_FILE))
    connection.execute("SET enable_external_access=false")
    return FragmentCache(connection, clock=clock)


def tool_services(
    settings: Settings,
    *,
    clock: Clock | None = None,
    handles: ResultStore | None = None,
    jobs: JobRegistry | None = None,
) -> ToolServices:
    """The stores `history`, `feedback` and `get_more` read.

    They share one metadata database with the query path: a run recorded by `query` must be visible to
    `history`, and a handle it returns must be pageable by `get_more`. Wiring them here keeps that true
    in one place instead of relying on two constructors agreeing.

    ``jobs`` must be the same registry the query tool was given, or a ``query_id`` it issued would
    be unreadable here: pass the object :func:`job_registry` built.
    """
    store = Store(settings.paths.metadata_db)
    store.initialize()
    # The stores take a plain `now` callable while the governor's clock also carries `monotonic`;
    # passing the method keeps every component on one notion of time.
    now = None if clock is None else clock.now
    return ToolServices(
        handles=handles if handles is not None else ResultStore(clock=clock),
        history=HistoryStore(store, clock=now),
        feedback=FeedbackStore(store, clock=now),
        audit=AuditLog(store, clock=now),
        jobs=jobs,
    )


def job_registry(settings: Settings, *, clock: Clock | None = None) -> JobRegistry:
    """The asynchronous job registry over the metadata database, migrated and ready to start work.

    One registry per server process, shared by the `query` tool (which starts jobs) and `get_more`
    (which reports on them): a second one would keep a second set of running threads and neither
    could see the other's in-memory progress.
    """
    store = Store(settings.paths.metadata_db)
    store.initialize()
    return JobRegistry(QueryJobStore(store), clock=clock, max_jobs=DEFAULT_MAX_JOBS)


def query_services(
    settings: Settings,
    layer: SemanticLayer,
    *,
    clock: Clock | None = None,
    tables: Sequence[str] | None = None,
    handles: ResultStore | None = None,
    jobs: JobRegistry | None = None,
) -> QueryServices:
    """Assemble everything the ``query`` tool needs from one `Settings` object."""
    adapters = build_adapters(settings)
    if not adapters:
        raise WiringError(
            "no sources are configured, so `query` has nothing to run against; "
            "set `sources:` in mdqm.yaml or MDQM_SOURCES"
        )
    store = Store(settings.paths.metadata_db)
    store.initialize()
    columns = catalog_columns(adapters)
    if tables is not None:
        allowed = set(tables)
        columns = {key: value for key, value in columns.items() if key in allowed}
    # One governor for the single-source path and the federated executor, so a handle `query`
    # returns is the handle `get_more` pages, and one executor survives between calls: that is
    # what makes a repeated cross-source question a cache hit instead of a cold per-call database.
    governor = Governor(
        adapters,
        clock=clock,
        store=handles if handles is not None else ResultStore(clock=clock),
    )
    spill = settings.paths.spill_dir
    return QueryServices(
        layer=layer,
        governor=governor,
        # history and audit take a plain `now` callable while the governor's clock also carries
        # `monotonic`; passing the method keeps every component on the same notion of time.
        history=HistoryStore(store, clock=None if clock is None else clock.now),
        audit=AuditLog(store, clock=None if clock is None else clock.now),
        adapters=adapters,
        columns=columns,
        cache=open_cache(settings.paths.cache_dir, clock),
        clock=clock,
        limits=query_limits(settings),
        executor=FederatedExecutor(governor, layer=layer, clock=clock, spill_dir=spill),
        spill_dir=spill,
        # The measured sizes (`mdqm sizes measure`), so a plan counts what was measured instead of
        # the catalog's own reltuples. An install that never measured plans exactly as before.
        sizes=SizeStore(store, clock=None if clock is None else clock.now),
        # The asynchronous registry: `mode="async"` starts a job here and `get_more` polls the same
        # object. `None` keeps the tool blocking-only and says so when asked for a job.
        jobs=jobs,
    )
