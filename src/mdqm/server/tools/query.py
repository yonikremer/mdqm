"""The ``query`` tool body: lint, compile, plan, cache, execute, summarise, record (M03).

One request walks the whole pipeline in this order, and each stage's refusal is the agent's answer:

1. **lint** (:mod:`mdqm.linter`) repairs what it can and refuses what it must, before any work is
   spent. Every repair is reported in ``applied_fixes`` (principle 10: nothing silent).
2. **compile** (:mod:`mdqm.compiler`) turns the repaired spec into exactly one fragment for exactly
   one source, through the sqlglot AST and the read-only guard.
3. **plan**: the static partition estimate (``planner/partitions.py``), then -- only when the source
   is really about to run -- the source's own ``EXPLAIN`` and A03's pruning verdict, gated with
   :func:`~mdqm.planner.partitions.gate_plan`. A plan that reads every partition is refused there,
   so the refusal costs the source nothing.
4. **cache** (:class:`~mdqm.executor.cache.FragmentCache`): an identical fragment inside its TTL is
   answered with **zero** source calls and its own ``fetched_at``. ``fresh=true`` skips the lookup.
5. **execute** through :meth:`~mdqm.governor.core.Governor.execute` and never through an adapter:
   the guard, the breaker, the slots, the quota, the deadline and the result handle live there.
6. **summarise** with :func:`~mdqm.results.summary.summarise`, so the agent gets a bounded sample,
   statistics and a handle instead of rows.
7. **history** (:mod:`mdqm.store.history`) and **audit** (:mod:`mdqm.store.audit`), on the refusal
   path too: a refused ``DELETE`` is a row, not a rumour.

Two decisions worth knowing. The **cache unit is the fragment**, not a partition window: C17's
``FragmentCache`` applies C12's freshness rule to one whole fragment, which is the smallest unit in
which "a second identical call touches no source" is true;
:class:`~mdqm.cache.segments.SegmentCache` needs a partition-aligned window per table, and a
restriction-free table has none.
And the **raw-SQL path** is an emergency exit only: it runs the same linter (whose first rule is the
guard) and the same governor, but it names its source explicitly and skips the ``EXPLAIN`` gate,
because nothing guarantees which tables the statement reads.

The services are injected once, process-wide, exactly like the lookup tools: the tool signature is
the MCP wire schema, so a dependency cannot travel as an argument.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, cast

import pyarrow as pa
from pydantic import Field

from mdqm.adapters.base import NotSupported, SourceAdapter
from mdqm.cache.freshness import Mutability
from mdqm.compiler import CompileContext, compile_spec
from mdqm.compiler.resolve import TableRef, known_tables, resolve_table
from mdqm.errors import from_exception, invalid_spec
from mdqm.executor.cache import FragmentCache, fragment_key
from mdqm.executor.federate import FederatedExecutor
from mdqm.executor.models import QualityReport
from mdqm.governor import Clock, Governor, GovernorRefusal, SystemClock
from mdqm.linter import LintContext, lint_spec, lint_sql
from mdqm.models import ErrorResponse, Fragment, JsonObject, Limits, PlanSummary, PruningProof
from mdqm.models.base import MDQMModel, UtcDatetime
from mdqm.planner.partitions import PartitionPlan, gate_plan
from mdqm.results.summary import ResultSummary, summarise
from mdqm.semantic.models import SemanticLayer
from mdqm.server.context import current_context
from mdqm.spec.errors import SpecError
from mdqm.spec.models import QuerySpec
from mdqm.spec.validate import parse_spec
from mdqm.sql.exec_ts import ExecTsMode, rewrite_sql
from mdqm.stats.guard import ValueProbe, gate_spec
from mdqm.stats.models import CategoryColumnStats, ColumnQuality, PartitionExtent
from mdqm.store.audit import AuditEntry, AuditLog, AuditOutcome
from mdqm.store.history import CacheInfo, HistoryEntry, HistoryStore, RecordedFragment
from mdqm.store.sizes import SizeStore

__all__ = [
    "PlanInfo",
    "QueryAnswer",
    "QueryServices",
    "QueryServicesUnavailable",
    "SegmentFreshness",
    "configure_query_services",
    "current_query_services",
    "query",
    "reset_query_services",
    "use_query_services",
]

logger = logging.getLogger(__name__)

#: One clock for a server built without one: the real walls, and it holds no state.
_SYSTEM_CLOCK: Clock = SystemClock()

#: Conservative caps for a server that configures none (DESIGN section 6: protect the primary).
_DEFAULT_LIMITS: Limits = Limits(
    timeout_s=30.0, max_rows=50_000, max_bytes=64 << 20, max_partitions=64
)


# --- what the tool is given --------------------------------------------------------------


@dataclass(frozen=True)
class QueryServices:
    """Everything one query needs, injected by the server.

    Args:
        layer: the semantic layer the spec is linted and compiled against.
        governor: the only path to a source: guard, slots, quota, deadline, result handle.
        history: where the run is recorded, so it can be asked again exactly.
        audit: where the run (and every refusal) is appended, as one hash-chained row.
        adapters: the same mapping the governor holds. Never used to execute: only to read a
            source's dialect and to ask it for its plan.
        columns: the catalog's columns per ``source.table`` key, the only source of identifiers.
        cache: the fragment cache, or ``None`` when this server has no local DuckDB to hold it.
        clock: the injected clock, so ``exec_ts`` and every TTL are testable.
        limits: the caps one fragment runs under.
        executor: the federated executor (M04's cross-source path), or ``None``. It holds its own
            DuckDB and its own fragment cache, so a shared one is what makes a repeated
            cross-source query free; without it each such query builds one, and its cache is cold.
        spill_dir: where that self-built executor spills. ``None`` means the configured default.
    """

    layer: SemanticLayer
    governor: Governor
    history: HistoryStore
    audit: AuditLog
    adapters: Mapping[str, SourceAdapter] = field(default_factory=dict)
    columns: Mapping[str, Sequence[str]] = field(default_factory=dict)
    cache: FragmentCache | None = None
    clock: Clock | None = None
    limits: Limits = _DEFAULT_LIMITS
    executor: FederatedExecutor | None = None
    spill_dir: Path | None = None
    sizes: SizeStore | None = None
    """Where the measured table/partition sizes live (``mdqm sizes measure``), or ``None``.

    ``None`` is the honest default: a deployment that never measured sizes plans exactly as it did
    before, from the catalog's own estimates."""
    column_stats: Mapping[str, CategoryColumnStats] = field(default_factory=dict)
    """Categorical distributions keyed ``source.table.column``. Empty skips the value gate."""
    column_quality: Mapping[str, ColumnQuality] = field(default_factory=dict)
    sentinels: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    partition_extents: Mapping[str, tuple[PartitionExtent, ...]] = field(default_factory=dict)
    table_rows: Mapping[str, int] = field(default_factory=dict)
    value_probe: ValueProbe | None = None
    """Bounded existence lookup used when category stats may be stale. ``None`` does not reject."""

    def now(self) -> datetime:
        """The one instant of this execution (C02), normalised to UTC."""
        clock = self.clock if self.clock is not None else _SYSTEM_CLOCK
        return clock.now().astimezone(UTC)


class QueryServicesUnavailable(RuntimeError):
    """No pipeline was configured: the tool cannot answer, and says so instead of guessing."""


_services: QueryServices | None = None


def configure_query_services(services: QueryServices | None) -> None:
    """Hand the ``query`` tool its pipeline. Called once when the server is built."""
    global _services
    _services = services


def reset_query_services() -> None:
    """Forget the configured pipeline. Tests use it; production configures once at startup."""
    configure_query_services(None)


def current_query_services() -> QueryServices:
    """The configured pipeline.

    Raises:
        QueryServicesUnavailable: the server was built without it. The tool reports that as
            ``not_implemented`` rather than raising, because a wiring bug is not the agent's.
    """
    if _services is None:
        raise QueryServicesUnavailable(
            "the query tool is registered but no pipeline was configured: call "
            "configure_query_services() when building the server"
        )
    return _services


@contextmanager
def use_query_services(services: QueryServices) -> Iterator[QueryServices]:
    """Configure the pipeline for one block (tests, and a future per-request scope)."""
    previous = _services
    configure_query_services(services)
    try:
        yield services
    finally:
        configure_query_services(previous)


# --- what the agent sees ----------------------------------------------------------------


class SegmentFreshness(MDQMModel):
    """Where one source's rows came from: a fresh cache hit, a refetch, or no cache at all.

    ``fetched_at`` is when the server learned these rows, never "now": a cache hit that reported
    "now" would be indistinguishable from a fresh read, which is the confusion this field prevents.
    """

    source: str
    table: str
    status: Literal["hit", "miss", "stale", "uncached"] = "uncached"
    fetched_at: UtcDatetime | None = None
    age_s: float | None = Field(default=None, ge=0)
    reason: str = ""


class PlanInfo(MDQMModel):
    """What the plan says the query reads: the static estimate, plus the source's own plan.

    ``pruning_proof`` is A03's verdict, not a promise: ``proven`` means the source's plan showed the
    pruning, ``unproven`` means it did not, and ``unknown`` means no plan was obtained (no
    ``EXPLAIN`` capability, a raw statement, or a cache hit).
    """

    partitions_scanned: int | None = Field(default=None, ge=0)
    partitions_total: int | None = Field(default=None, ge=0)
    partitions_if_fixed: str | None = Field(
        default=None, description="What a bounded window would read, e.g. '2 of 14'."
    )
    uses_index: bool | None = None
    est_rows: float | None = Field(default=None, ge=0)
    est_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Bytes the window would read, when a measured size profile answered.",
    )
    estimate_source: str | None = Field(
        default=None,
        description=(
            "'profile' when the measured sizes answered (mdqm sizes measure), 'catalog' when the "
            "source's own per-partition estimates did. It describes est_bytes and the static "
            "partition estimate, NOT est_rows: when the source returned its own plan, est_rows is "
            "that plan's figure and this field is about the other numbers."
        ),
    )
    estimate_incomplete: bool = Field(
        default=False,
        description="A partition in the window reported no size, so est_rows is a lower bound.",
    )
    pruning_proof: PruningProof = PruningProof.UNKNOWN
    notes: tuple[str, ...] = ()


class QueryAnswer(MDQMModel):
    """One answer from ``query``: a summary, a small sample and a handle, or the refusal.

    ``status`` is ``ok`` with an answer, ``dry_run`` with the plan only, ``error`` with the
    structured :class:`~mdqm.models.ErrorResponse` the agent must act on, or ``not_implemented``
    when the server was built without a pipeline.
    """

    status: Literal["ok", "dry_run", "error", "not_implemented"] = "ok"
    surface: str = "query"
    user_id: str = ""
    error: ErrorResponse | None = Field(default=None, description="Set when status is 'error'.")
    summary: ResultSummary | None = Field(default=None, description="The bounded answer.")
    sample: tuple[JsonObject, ...] = Field(
        default=(), description="The first rows, small enough to read whole."
    )
    result_handle: str | None = Field(default=None, description="Page it with `get_more`.")
    rows_total: int = Field(default=0, ge=0)
    cache_hit: bool = Field(default=False, description="True when no source call was made.")
    freshness: tuple[SegmentFreshness, ...] = ()
    quality: QualityReport | None = Field(
        default=None,
        description=(
            "A cross-source answer's receipt: rows per source with their own `fetched_at`, and "
            "what each join matched, dropped and multiplied. `complete: false` means no answer."
        ),
    )
    plan: PlanInfo | None = None
    applied_fixes: tuple[str, ...] = Field(
        default=(), description="Every change the server made to your request, in order."
    )
    problem: str | None = Field(default=None, description="What is missing (not_implemented).")
    fix: str | None = Field(default=None, description="What to do instead (not_implemented).")


# --- the tool ----------------------------------------------------------------------------


def query(
    spec: Annotated[
        QuerySpec | None,
        Field(description="The query spec. Send this, not SQL: it is linted and compiled for you."),
    ] = None,
    sql: Annotated[
        str | None,
        Field(
            description="Emergency only: one read-only statement. Needs `source`. Writes are "
            "refused and audited."
        ),
    ] = None,
    source: Annotated[
        str | None,
        Field(description="The source `sql` runs on. Required with `sql`, ignored with `spec`."),
    ] = None,
) -> QueryAnswer:
    """Run one query spec and get a summary, a small sample and a handle. This is the main tool.

    Send `spec`, never SQL. The server lints it, prunes the partitions and the time window, picks
    the source, and answers with `summary`, `sample` and `result_handle` (page it with `get_more`).
    A `dry_run` in the spec stops after the plan; `fresh` skips the cache. Errors carry `code`,
    `problem` and an exact `fix`.
    Example: query({"spec": {"from": "billing.accounts", "select": ["customer_status"],
    "limit": 5}})
    """
    context = current_context()
    user = context.user_id
    services = _services
    if services is None:
        return QueryAnswer(
            status="not_implemented",
            user_id=user,
            problem="no query pipeline is configured on this server, so nothing can run",
            fix="this build serves the protocol only; start the server with its pipeline wired "
            "(`mdqm serve`), or call `find_data` to explore what exists",
        )
    if (spec is None) == (sql is None):
        return _refused(
            services,
            user,
            error=invalid_spec(
                field="spec",
                reason="send exactly one of `spec` (the normal path) or `sql` (the emergency exit)",
                expected=("spec",),
            ),
        )
    if spec is not None:
        return _run_spec(services, user, spec)
    return _run_sql(services, user, sql or "", source)


# --- the two paths -----------------------------------------------------------------------


def _run_spec(services: QueryServices, user: str, spec: QuerySpec) -> QueryAnswer:
    """The normal path: spec in, linted and compiled, planned, cached, executed, recorded."""
    exec_ts = services.now()
    # The wire model checks the shape; parse_spec also runs the cross-field rules, so a spec that
    # arrived as JSON gets the same INVALID_SPEC answer as one that arrived as an object.
    try:
        parsed = parse_spec(spec.model_dump(mode="json", by_alias=True, exclude_none=True))
    except SpecError as spec_error:
        return _refused(
            services, user, error=spec_error.to_error_response(), spec=spec, exec_ts=exec_ts
        )
    target = _target(services, parsed)
    if isinstance(target, ErrorResponse):
        return _refused(services, user, error=target, spec=parsed, exec_ts=exec_ts)
    lint = lint_spec(parsed, _lint_context(services, exec_ts, target.dialect))
    if lint.error is not None:
        return _refused(services, user, error=lint.error, spec=parsed, exec_ts=exec_ts)
    repaired = lint.spec if lint.spec is not None else parsed
    # Stale category stats are not rejected by the linter (it has no source). A bounded lookup
    # runs here, before compile, and only when the value might be newer than the last collection.
    gated = gate_spec(
        repaired,
        layer=services.layer,
        columns=services.columns,
        stats=services.column_stats,
        probe=services.value_probe,
    )
    if gated.error is not None:
        return _refused(
            services,
            user,
            error=gated.error,
            spec=repaired,
            exec_ts=exec_ts,
            applied_fixes=lint.applied_fixes,
        )
    reported = _fixes(lint.applied_fixes, gated.notes)
    # A spec whose approved joins reach another source is not one statement: it becomes a
    # federated plan (M04). A single-source spec gets ``None`` here and compiles below, unchanged.
    from mdqm.server.tools.query_federated import federated_plan, run_federated

    federated = federated_plan(services, repaired, exec_ts)
    if isinstance(federated, ErrorResponse):
        return _refused(
            services,
            user,
            error=federated,
            spec=repaired,
            exec_ts=exec_ts,
            sources=(target.source,),
            applied_fixes=reported,
        )
    if federated is not None:
        return run_federated(
            services,
            user,
            federated=federated,
            spec=repaired,
            exec_ts=exec_ts,
            applied_fixes=reported,
        )
    compiled = compile_spec(
        repaired,
        CompileContext(
            layer=services.layer,
            source=target.source,
            dialect=target.dialect,
            exec_ts=exec_ts,
            columns=services.columns,
            max_rows=services.limits.max_rows,
            max_partitions=services.limits.max_partitions,
            sizes=_sizes_for(services, target.source, repaired.from_table),
            sentinels=services.sentinels,
        ),
    )
    fixes = _fixes(reported, compiled.applied_fixes)
    plan = _plan(None, compiled.partition_plan)
    if compiled.error is not None or compiled.fragment is None:
        return _refused(
            services,
            user,
            error=compiled.error
            or invalid_spec(field="spec", reason="the compiler produced no statement"),
            spec=repaired,
            exec_ts=exec_ts,
            sources=(target.source,),
            applied_fixes=fixes,
            plan=plan,
        )
    return _run(
        services,
        user,
        source=target.source,
        dialect=target.dialect,
        fragment=compiled.fragment,
        exec_ts=exec_ts,
        spec=repaired,
        original_sql=None,
        fixes=fixes,
        partition_plan=compiled.partition_plan,
        anchor=target.anchor,
    )


def _run_sql(services: QueryServices, user: str, sql: str, source: str | None) -> QueryAnswer:
    """The emergency exit: one read-only statement, through the same linter and the governor."""
    exec_ts = services.now()
    adapter = services.adapters.get(source) if source is not None else None
    if source is None or adapter is None:
        reason = (
            "a raw statement names no source, so the server cannot choose a dialect"
            if source is None
            else f"{source!r} is not a configured source"
        )
        return _refused(
            services,
            user,
            error=invalid_spec(field="source", reason=reason, expected=sorted(services.adapters)),
            original_sql=sql,
            exec_ts=exec_ts,
        )
    dialect = adapter.dialect()
    lint = lint_sql(sql, _lint_context(services, exec_ts, dialect))
    if lint.error is not None:
        # A refused statement is still a run: it is audited, and its history row says why.
        return _refused(
            services,
            user,
            error=lint.error,
            original_sql=sql,
            dialect=dialect,
            exec_ts=exec_ts,
            sources=(source,),
            applied_fixes=lint.applied_fixes,
        )
    repaired = lint.sql or sql
    fragment = Fragment(
        source=source,
        dialect=dialect,
        sql=rewrite_sql(repaired, dialect=dialect, exec_ts=exec_ts).sql,
        exec_ts=exec_ts,
    )
    return _run(
        services,
        user,
        source=source,
        dialect=dialect,
        fragment=fragment,
        exec_ts=exec_ts,
        spec=None,
        original_sql=(repaired, dialect),
        fixes=lint.applied_fixes,
        partition_plan=None,
        anchor=None,
    )


# --- one compiled fragment ----------------------------------------------------------------


def _run(
    services: QueryServices,
    user: str,
    *,
    source: str,
    dialect: str,
    fragment: Fragment,
    exec_ts: datetime,
    spec: QuerySpec | None,
    original_sql: tuple[str, str] | None,
    fixes: tuple[str, ...],
    partition_plan: PartitionPlan | None,
    anchor: TableRef | None,
) -> QueryAnswer:
    """Plan, cache, execute, summarise and record one compiled fragment.

    ``original_sql`` is ``(the statement before exec_ts substitution, its dialect)`` on the raw-SQL
    path, and ``None`` on the spec path, where the compiler built the statement already frozen.
    """
    fresh = spec is not None and spec.fresh
    dry_run = spec is not None and spec.dry_run
    plan = _plan(None, partition_plan)
    if dry_run:
        return QueryAnswer(
            user_id=user,
            status="dry_run",
            plan=plan,
            applied_fixes=fixes,
            freshness=(SegmentFreshness(source=source, table=_table_of(anchor, source)),),
        )
    cache = services.cache
    key: str | None = None
    mutability: Mutability | None = None
    if cache is not None:
        key, mutability = _cache_key(services, source, anchor, fragment)
    if cache is not None and key is not None and mutability is not None and not fresh:
        lookup = cache.lookup(key, mutability=mutability, exec_ts=exec_ts)
        if lookup.hit and lookup.table is not None and lookup.fetched_at is not None:
            summary = _summarise(services, user, source, lookup.table)
            fetched_at = lookup.fetched_at
            _record(
                services,
                user,
                source=source,
                dialect=dialect,
                fragment=fragment,
                exec_ts=exec_ts,
                spec=spec,
                original_sql=original_sql,
                fixes=fixes,
                cache=CacheInfo(hit=True, fetched_at=fetched_at, fresh=True),
            )
            return QueryAnswer(
                user_id=user,
                summary=summary,
                sample=summary.sample,
                result_handle=summary.handle,
                rows_total=summary.rows_total,
                cache_hit=True,
                freshness=(
                    SegmentFreshness(
                        source=source,
                        table=_table_of(anchor, source),
                        status="hit",
                        fetched_at=fetched_at,
                        age_s=lookup.age_s,
                        reason=lookup.reason,
                    ),
                ),
                plan=plan,
                applied_fixes=fixes,
            )

    # The source is about to be asked, so spend the one plan on it and let A03's verdict refuse a
    # scan that does not prune. A cache hit never gets here, which is why it costs no source call.
    explained, refusal = _explain(services, source, fragment, anchor)
    if refusal is not None:
        return _refused(
            services,
            user,
            error=refusal,
            spec=spec,
            original_sql=None if original_sql is None else original_sql[0],
            dialect=None if original_sql is None else original_sql[1],
            exec_ts=exec_ts,
            sources=(source,),
            applied_fixes=fixes,
            plan=_plan(explained, partition_plan),
        )
    plan = _plan(explained, partition_plan)
    try:
        outcome = services.governor.execute(
            source=source, user=user, fragment=fragment, limits=services.limits
        )
    except GovernorRefusal as governor_refusal:
        return _refused(
            services,
            user,
            error=governor_refusal.response,
            spec=spec,
            original_sql=None if original_sql is None else original_sql[0],
            dialect=None if original_sql is None else original_sql[1],
            exec_ts=exec_ts,
            sources=(source,),
            applied_fixes=fixes,
            plan=plan,
        )
    except Exception as exc:
        logger.exception("query execution failed", extra={"source": source})
        return _refused(
            services,
            user,
            error=from_exception(exc, source=source, timeout_s=services.limits.timeout_s),
            spec=spec,
            original_sql=None if original_sql is None else original_sql[0],
            dialect=None if original_sql is None else original_sql[1],
            exec_ts=exec_ts,
            sources=(source,),
            applied_fixes=fixes,
            plan=plan,
        )

    table = services.governor.store.table(outcome.handle.handle, owner=user)
    fetched_at = services.now()
    status: Literal["miss", "uncached"] = "uncached"
    if cache is not None and key is not None and mutability is not None:
        fetched_at = cache.store(
            key,
            table,
            source=source,
            table_name=_table_of(anchor, source),
            version_hash=services.layer.version_hash,
        )
        status = "miss"
    summary = outcome.summary
    _record(
        services,
        user,
        source=source,
        dialect=dialect,
        fragment=fragment,
        exec_ts=exec_ts,
        spec=spec,
        original_sql=original_sql,
        fixes=fixes,
        cache=CacheInfo(hit=False, fetched_at=fetched_at, fresh=not fresh),
        rows=summary.rows_total,
    )
    return QueryAnswer(
        user_id=user,
        summary=summary,
        sample=summary.sample,
        result_handle=outcome.handle.handle,
        rows_total=summary.rows_total,
        freshness=(
            SegmentFreshness(
                source=source,
                table=_table_of(anchor, source),
                status=status,
                fetched_at=fetched_at,
            ),
        ),
        plan=plan,
        applied_fixes=fixes,
    )


# --- the stages --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Target:
    """Which source the spec reads, and in which dialect: resolved before anything is linted."""

    source: str
    dialect: str
    anchor: TableRef


def _target(services: QueryServices, spec: QuerySpec) -> _Target | ErrorResponse:
    """Resolve the spec's table to its source and dialect, or the refusal that says why not."""
    anchor = resolve_table(spec.from_table, tables=known_tables(services.layer, services.columns))
    if isinstance(anchor, ErrorResponse):
        return anchor
    source = anchor.key.partition(".")[0] if "." in anchor.key else ""
    adapter = services.adapters.get(source)
    if not source or adapter is None:
        return invalid_spec(
            field="from",
            reason=(
                f'"{anchor.key}" names no configured source; a table is written as "source.table"'
            ),
            expected=sorted(services.adapters) or sorted(services.columns),
        )
    return _Target(source=source, dialect=adapter.dialect(), anchor=anchor)


def _lint_context(services: QueryServices, exec_ts: datetime, dialect: str) -> LintContext:
    """The execution's facts the linter needs: one clock reading, the catalog, the caps."""
    return LintContext(
        layer=services.layer,
        columns=services.columns,
        exec_ts=exec_ts,
        dialect=dialect,
        max_rows=services.limits.max_rows,
        max_partitions=services.limits.max_partitions,
        category_stats=services.column_stats,
        column_quality=services.column_quality,
        sentinels=services.sentinels,
        partition_extents=services.partition_extents,
        table_rows=services.table_rows,
    )


def _plan(summary: PlanSummary | None, partition_plan: PartitionPlan | None) -> PlanInfo:
    """The agent-facing plan: the static estimate, and the source's own plan when there is one."""
    estimate = None if partition_plan is None else partition_plan.estimate
    notes = () if partition_plan is None else tuple(note.value for note in partition_plan.notes)
    static_total = None if estimate is None else estimate.total
    if summary is None:
        scanned = None if estimate is None else estimate.partitions
        est_rows = None if estimate is None or estimate.rows is None else float(estimate.rows)
        return PlanInfo(
            partitions_scanned=scanned,
            partitions_total=static_total,
            partitions_if_fixed=None if estimate is None else estimate.label,
            est_rows=est_rows,
            est_bytes=None if estimate is None else estimate.bytes,
            estimate_source=None if estimate is None else estimate.estimated_from,
            estimate_incomplete=False if estimate is None else estimate.incomplete,
            notes=notes,
        )
    return PlanInfo(
        partitions_scanned=summary.partitions_scanned,
        partitions_total=(
            summary.partitions_total if summary.partitions_total is not None else static_total
        ),
        partitions_if_fixed=None if estimate is None else estimate.label,
        uses_index=summary.uses_index,
        est_rows=summary.est_rows,
        est_bytes=None if estimate is None else estimate.bytes,
        estimate_source=None if estimate is None else estimate.estimated_from,
        estimate_incomplete=False if estimate is None else estimate.incomplete,
        pruning_proof=summary.pruning_proof,
        notes=notes,
    )


def _explain(
    services: QueryServices, source: str, fragment: Fragment, anchor: TableRef | None
) -> tuple[PlanSummary | None, ErrorResponse | None]:
    """Ask the source for its plan and gate it, or say why no plan was obtained.

    A source without the ``explain_plan`` capability is not a failure: the declared fallback is the
    static estimate from the partition planner, which already refused an unpruned scan. A source
    that fails to explain IS a failure, and the agent gets the classified error rather than a guess
    about whether the scan prunes.
    """
    adapter = services.adapters.get(source)
    if adapter is None or anchor is None:
        return None, None
    explain = getattr(adapter, "explain", None)
    if not callable(explain):
        return None, None
    ask_plan = cast(Callable[[Fragment], PlanSummary], explain)
    try:
        summary = ask_plan(fragment)
    except NotSupported:
        return None, None
    except Exception as exc:
        logger.warning("EXPLAIN failed", extra={"source": source}, exc_info=exc)
        return None, from_exception(exc, source=source, timeout_s=services.limits.timeout_s)
    return summary, gate_plan(
        summary,
        table=anchor.key,
        partitioning=anchor.semantics.partitioning,
        max_partitions_scanned=services.limits.max_partitions,
    )


def _cache_key(
    services: QueryServices, source: str, anchor: TableRef | None, fragment: Fragment
) -> tuple[str | None, Mutability | None]:
    """The fragment's cache identity, or ``(None, None)`` when this table must not be cached.

    The mutability class is the semantic layer's measured and approved one: an unclassified table is
    ``unknown``, and nothing unknown is ever served from cache.
    """
    if anchor is None or fragment.sql is None:
        return None, None
    semantics = services.layer.tables.get(anchor.key)
    if semantics is None or semantics.mutability is None or not semantics.may_cache:
        return None, None
    return (
        fragment_key(
            source=source,
            table=anchor.key,
            sql=fragment.sql,
            version_hash=services.layer.version_hash,
        ),
        semantics.mutability.class_,
    )


def _summarise(services: QueryServices, user: str, source: str, table: pa.Table) -> ResultSummary:
    """Store the rows and summarise them: the same two steps the governor takes on a live answer."""
    handle = services.governor.store.put(owner=user, source=source, table=table)
    return summarise(table, handle=handle.handle)


def _table_of(anchor: TableRef | None, source: str) -> str:
    """The table the answer is about, for the freshness report."""
    return source if anchor is None else anchor.key


def _sizes_for(
    services: QueryServices, source: str, table: str
) -> Mapping[str, tuple[int | None, int | None]]:
    """The measured profile of ``table``, as ``{partition name: (rows, bytes)}``.

    Empty when this deployment never measured sizes (``mdqm sizes measure``) or when the measurement
    failed: the estimate then falls back to the catalog, which is what it always did. A measured
    profile that exists is preferred, because it is ours, it is dated, and it knows bytes.
    """
    if services.sizes is None:
        return {}
    bare = table.split(".")[-1]
    measured = services.sizes.latest(source, bare)
    if measured is None:
        return {}
    return {partition.name: (partition.rows, partition.bytes) for partition in measured.partitions}


def _fixes(*groups: Sequence[str]) -> tuple[str, ...]:
    """Every reported repair, in the order the stages applied them, without repeats."""
    seen: dict[str, None] = {}
    for group in groups:
        for fix in group:
            seen.setdefault(fix, None)
    return tuple(seen)


# --- recording ---------------------------------------------------------------------------


def _refused(
    services: QueryServices,
    user: str,
    *,
    error: ErrorResponse,
    spec: QuerySpec | None = None,
    original_sql: str | None = None,
    dialect: str | None = None,
    exec_ts: datetime | None = None,
    sources: Sequence[str] = (),
    applied_fixes: tuple[str, ...] = (),
    plan: PlanInfo | None = None,
) -> QueryAnswer:
    """Audit the refusal, record what was asked, and hand the agent the same structured error."""
    try:
        services.audit.record_refusal(
            user_id=user,
            tool="query",
            error=error,
            version_hash=services.layer.version_hash,
            sources=tuple(sources),
        )
    except Exception:
        logger.exception("could not audit a query refusal", extra={"error_code": error.code})
    frozen: str | None = None
    if original_sql is not None and dialect is not None:
        try:
            moment = exec_ts if exec_ts is not None else services.now()
            frozen = rewrite_sql(original_sql, dialect=dialect, exec_ts=moment).sql
        except Exception:
            frozen = None
    _record_history(
        services,
        lambda: HistoryEntry(
            user_id=user,
            exec_ts=exec_ts if exec_ts is not None else services.now(),
            mode=ExecTsMode.RELATIVE,
            spec=spec,
            original_sql=original_sql if frozen is not None else None,
            frozen_sql=frozen,
            dialect=dialect,
            applied_fixes=applied_fixes,
            error=error,
        ),
        replayable=spec is not None or frozen is not None,
    )
    return QueryAnswer(
        user_id=user, status="error", error=error, plan=plan, applied_fixes=applied_fixes
    )


def _record(
    services: QueryServices,
    user: str,
    *,
    source: str,
    dialect: str,
    fragment: Fragment,
    exec_ts: datetime,
    spec: QuerySpec | None,
    original_sql: tuple[str, str] | None,
    fixes: tuple[str, ...],
    cache: CacheInfo,
    rows: int = 0,
) -> None:
    """Write the run to history and to the audit chain: the ok path, reported like the refusal."""
    original, sql_dialect = (None, dialect) if original_sql is None else original_sql
    _record_history(
        services,
        lambda: HistoryEntry(
            user_id=user,
            exec_ts=exec_ts,
            mode=ExecTsMode.RELATIVE,
            spec=spec,
            original_sql=original,
            frozen_sql=None if original is None else fragment.sql,
            dialect=None if original is None else sql_dialect,
            fragments=(
                RecordedFragment(
                    source=source,
                    dialect=dialect,
                    sql=fragment.sql,
                    tables=fragment.tables,
                    original_sql=original,
                ),
            ),
            applied_fixes=fixes,
            cache=cache,
        ),
        replayable=True,
    )
    try:
        services.audit.record(
            AuditEntry(
                user_id=user,
                tool="query",
                outcome=AuditOutcome.OK,
                version_hash=services.layer.version_hash,
                sources=(source,),
                row_count=max(0, rows),
            )
        )
    except Exception:
        logger.exception("could not audit a query", extra={"source": source})


def _record_history(
    services: QueryServices, entry: Callable[[], HistoryEntry], *, replayable: bool
) -> None:
    """Store one run, or log why it could not be stored: a refusal is never lost silently.

    ``replayable`` is False when the request carried nothing a re-run could repeat (a call with
    neither a spec nor a statement): there is no history row to write, and the audit row above is
    the whole record.
    """
    if not replayable:
        return
    try:
        services.history.record(entry())
    except Exception:
        logger.warning("could not record a run in history", exc_info=True)
