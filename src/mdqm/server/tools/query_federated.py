"""The ``query`` tool's federated branch: a spec that spans sources becomes one plan (M04).

M03 compiles a spec into exactly one fragment for exactly one source. A spec that names an approved
join whose other side lives in another source cannot be one statement, so it becomes a
:class:`~mdqm.executor.models.FederatedPlan`:

1. **the plan is derived from the spec, never guessed.** The tables are the ``from`` table plus the
   partner of every approved join the spec names; the output is the spec's own ``group_by`` and
   measures; nothing is read out of a fragment's SQL. Each physical column is taken from the
   semantic layer's measure/join declarations, so no agent text becomes an identifier.
2. **one fragment per source, each compiled by the ordinary compiler** (:mod:`mdqm.compiler`), with
   the filters and the time window that belong to *that* table pushed into it. A fragment is
   therefore linted, planned and guarded exactly like a single-source one -- the partition refusal
   and ``COST_TOO_HIGH`` happen before any source is asked.
3. **each fragment goes through the governor**, never an adapter
   (:class:`~mdqm.executor.federate.FederatedExecutor` does that), so the per-source slots, breaker,
   quota and deadline apply to a cross-source question too.
4. **the quality report is the answer's receipt**: rows per source with their own ``fetched_at``,
   matched/dropped/multiplied per join with the unmatched key sample, and ``complete=False`` with
   **no rows at all** when a source failed. Half a join presented as a whole is the failure this
   module exists to prevent.

Which join shape a step gets is decided here, once, and reported:

* ``inner`` when the partner's columns reach the answer (a grouping key, a measure, a filter): a row
  with no partner cannot be reported correctly, so it is dropped *and counted*;
* ``semi`` when the partner is the *many* side and its columns do not reach the answer: membership
  without fan-out, so a partner's duplicate keys cannot inflate an average;
* ``left`` otherwise: the anchor's rows are kept and an unmatched one is still counted as dropped.

The mode is chosen by module-level constants and the approved join's declared cardinality, never by
looking at the data.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import pyarrow as pa

from mdqm.compiler import CompileContext, compile_spec
from mdqm.compiler.resolve import (
    TableRef,
    find_join,
    join_tables,
    known_tables,
    resolve_column,
    resolve_table,
)
from mdqm.errors import invalid_spec, join_not_approved
from mdqm.executor.federate import FederatedExecutor, FederatedResult
from mdqm.executor.models import (
    FederatedOutput,
    FederatedPlan,
    FragmentStep,
    GroupColumn,
    JoinHow,
    JoinStep,
    LocalAggregate,
    QualityReport,
    StepCache,
)
from mdqm.models import ErrorResponse
from mdqm.results.summary import ResultSummary, summarise
from mdqm.semantic.models import JoinCardinality, Measure, SemanticLayer
from mdqm.server.tools.query import (
    PlanInfo,
    QueryAnswer,
    QueryServices,
    SegmentFreshness,
    _plan,
    _record_history,
    _refused,
)
from mdqm.spec.errors import SpecError
from mdqm.spec.models import Agg, Filter, QuerySpec, TimeSpec
from mdqm.spec.validate import parse_spec
from mdqm.store.audit import AuditEntry, AuditOutcome
from mdqm.store.history import CacheInfo, HistoryEntry, RecordedFragment

__all__ = [
    "FederatedQuery",
    "federated_plan",
    "run_federated",
    "step_name",
]

logger = logging.getLogger(__name__)

#: A step name becomes a DuckDB relation name, so a table key is made safe rather than trusted.
_UNSAFE: Final = re.compile(r"[^A-Za-z0-9_]")


@dataclass(frozen=True)
class FederatedQuery:
    """A derived plan and the repairs the derivation applied, reported as ``applied_fixes``."""

    plan: FederatedPlan
    fixes: tuple[str, ...] = ()


def step_name(table: TableRef) -> str:
    """The plan step (and local relation alias) for one table: ``billing.accounts`` ->
    ``billing_accounts``.

    A step name is reused as a DuckDB relation name and as a SQL alias, so it is built from the
    semantic key (which the layer validated) and sanitised anyway.
    """
    return _UNSAFE.sub("_", table.key).strip("_") or "step"


# --- deriving the plan ---------------------------------------------------------------------


def federated_plan(
    services: QueryServices, spec: QuerySpec, exec_ts: datetime
) -> FederatedQuery | ErrorResponse | None:
    """The federated plan this spec needs, or ``None`` when it reads one source.

    ``None`` is the important answer: a single-source spec keeps taking M03's path unchanged, so
    this function can be called unconditionally by the tool.

    Returns:
        A :class:`FederatedQuery` when the spec spans more than one source, ``None`` when it does
        not (M03's path is the right one), or the :class:`~mdqm.models.ErrorResponse` that refuses
        the spec before any source is asked.
    """
    layer = services.layer
    tables = known_tables(layer, services.columns)
    anchor = resolve_table(spec.from_table, tables=tables)
    if isinstance(anchor, ErrorResponse):
        return None  # the single-source path already answers this, with the same error

    scope: list[TableRef] = [anchor]
    steps_of: dict[str, str] = {anchor.key: step_name(anchor)}
    joins: list[tuple[str, TableRef, JoinCardinality]] = []
    for index, name in enumerate(spec.joins):
        approved = find_join(layer, name)
        if approved is None:
            return join_not_approved(left=anchor.key, right=name)
        left_key, right_key = join_tables(approved)
        if anchor.key not in (left_key, right_key):
            return invalid_spec(
                field=f"joins.{index}",
                reason=(
                    f'the approved join "{name}" connects {left_key} and {right_key}, and this '
                    f'query starts from {anchor.key}'
                ),
                expected=[
                    join.name
                    for join in layer.approved_joins
                    if anchor.key in join_tables(join)
                ],
            )
        partner_key = right_key if left_key == anchor.key else left_key
        partner = resolve_table(partner_key, tables=tables, field=f"joins.{index}")
        if isinstance(partner, ErrorResponse):
            return partner
        if partner.key not in steps_of:
            steps_of[partner.key] = step_name(partner)
            scope.append(partner)
        joins.append((name, partner, approved.cardinality))

    sources = {table.source for table in scope}
    if len(sources) < 2:
        return None
    if "" in sources:  # an unqualified table has no source, so no fragment can be built for it
        return None

    selects: dict[str, list[str]] = {table.key: [] for table in scope}
    filters: dict[str, list[Filter]] = {table.key: [] for table in scope}
    groups: list[GroupColumn] = []
    aggregates: list[LocalAggregate] = []
    fixes: list[str] = []

    def word(name: str, path: str) -> TableRef | ErrorResponse:
        """The table one spec word (a column, a map field, a time bucket) belongs to."""
        reference = resolve_column(name, scope=tuple(scope), layer=layer, field=path)
        if isinstance(reference, ErrorResponse):
            return reference
        if name not in selects[reference.table.key]:
            selects[reference.table.key].append(name.strip())
        return reference.table

    for index, item in enumerate(spec.select):
        if isinstance(item, str):
            owner = word(item, f"select.{index}")
            if isinstance(owner, ErrorResponse):
                return owner
            continue
        resolved = _measure_owner(layer, scope, item.measure, path=f"select.{index}.measure")
        if isinstance(resolved, ErrorResponse):
            return resolved
        table, column = resolved
        if column not in selects[table.key]:
            selects[table.key].append(column)
        aggregate = item.agg if item.agg is not None else Agg.SUM
        aggregates.append(
            LocalAggregate(
                step=steps_of[table.key],
                column=column,
                agg=aggregate,
                alias=item.alias
                or (item.measure if item.agg is None else f"{item.measure}_{aggregate.value}"),
                measure=item.measure,
            )
        )
    for index, key in enumerate(spec.group_by):
        owner = word(key, f"group_by.{index}")
        if isinstance(owner, ErrorResponse):
            return owner
        groups.append(GroupColumn(step=steps_of[owner.key], column=key.strip()))
    for index, condition in enumerate(spec.filters):
        owner = word(condition.column, f"filters.{index}.column")
        if isinstance(owner, ErrorResponse):
            return owner
        filters[owner.key].append(condition)
    sort_keys = tuple(order.by for order in spec.order_by)
    named = set(spec.selected_names()) | set(spec.group_by)
    for index, order in enumerate(spec.order_by):
        if order.by in named:
            continue
        return invalid_spec(
            field=f"order_by.{index}.by",
            reason=(
                "a cross-source answer is sorted by its output columns only, and this spec sorts "
                "by something else"
            ),
            expected=sorted(named),
        )

    # The approved joins' own key columns must survive into the fragments: the ON clause names them.
    for name, partner, _ in joins:
        approved = find_join(layer, name)
        assert approved is not None  # resolved above
        for side in (approved.left, approved.right):
            table_key, _, column = side.rpartition(".")
            if table_key in selects and column not in selects[table_key]:
                selects[table_key].append(column)
        fixes.append(f'joined "{partner.key}" through the approved join "{name}"')

    chain: list[JoinStep] = []
    used = {group.step for group in groups} | {aggregate.step for aggregate in aggregates}
    used |= {steps_of[key] for key, conditions in filters.items() if conditions}
    for name, partner, cardinality in joins:
        partner_step = steps_of[partner.key]
        how = _how(
            partner_step=partner_step,
            used=used,
            partner_is_many=_partner_is_many(
                cardinality, name=name, layer=layer, partner=partner
            ),
        )
        chain.append(
            JoinStep(join=name, left=steps_of[anchor.key], right=partner_step, how=how)
        )

    steps: list[FragmentStep] = []
    for table in scope:
        built = _fragment_step(
            services,
            table=table,
            name=steps_of[table.key],
            columns=selects[table.key],
            filters=filters[table.key],
            time_spec=spec.time if table.key == anchor.key else None,
            exec_ts=exec_ts,
        )
        if isinstance(built, ErrorResponse):
            return built
        steps.append(built)

    output = FederatedOutput(
        group_by=tuple(groups),
        aggregates=tuple(aggregates),
        order_by=sort_keys,
        limit=spec.limit if spec.limit is not None else services.limits.max_rows,
    )
    try:
        plan = FederatedPlan(exec_ts=exec_ts, steps=tuple(steps), joins=tuple(chain), output=output)
    except ValueError as exc:  # a plan the executor refuses to represent: our derivation is wrong
        logger.warning("derived an inconsistent federated plan", exc_info=exc)
        return invalid_spec(
            field="spec", reason=f"this spec cannot be answered across sources: {exc}"
        )
    return FederatedQuery(plan=plan, fixes=tuple(fixes))


def _how(*, partner_step: str, used: set[str], partner_is_many: bool) -> JoinHow:
    """Which join shape a step gets; see the module docstring for the rule and its reasons."""
    if partner_step in used:
        return "inner"
    return "semi" if partner_is_many else "left"


def _partner_is_many(
    cardinality: JoinCardinality, *, name: str, layer: SemanticLayer, partner: TableRef
) -> bool:
    """True when the *partner* side of an approved join can hold more than one row per key.

    ``many_to_one`` means the join's left side is the many one, so the partner is many exactly when
    it is the join's left table. Anything declared many-to-many is treated as many: a fan-out that
    was declared is a fan-out to plan for.
    """
    approved = find_join(layer, name)
    assert approved is not None  # resolved by the caller
    left_key, _ = join_tables(approved)
    partner_on_left = left_key == partner.key
    if cardinality is JoinCardinality.MANY_TO_MANY:
        return True
    if cardinality is JoinCardinality.MANY_TO_ONE:
        return partner_on_left
    if cardinality is JoinCardinality.ONE_TO_MANY:
        return not partner_on_left
    return False


def _measure_owner(
    layer: SemanticLayer, scope: Sequence[TableRef], name: str, *, path: str
) -> tuple[TableRef, str] | ErrorResponse:
    """The table that stores one measure, and its physical column there.

    Mirrors the compiler's rule (``_measure_column``): scope order first, then the measure's own key
    order, so the same spec picks the same source on every run.
    """
    measure: Measure | None = layer.measures.get(name)
    if measure is None:
        return invalid_spec(
            field=path,
            reason=f'the semantic layer has no measure named "{name}"',
            expected=sorted(layer.measures),
        )
    for table in scope:
        prefix = f"{table.key}."
        for key in sorted(measure.sources):
            if not key.startswith(prefix):
                continue
            column = key[len(prefix) :]
            if column in table.columns:
                return table, column
    return invalid_spec(
        field=path,
        reason=(
            f'the measure "{name}" is not stored on any table this query reads '
            f"({', '.join(table.key for table in scope)})"
        ),
        expected=sorted(measure.sources),
    )


def _fragment_step(
    services: QueryServices,
    *,
    table: TableRef,
    name: str,
    columns: Sequence[str],
    filters: Sequence[Filter],
    time_spec: TimeSpec | None,
    exec_ts: datetime,
) -> FragmentStep | ErrorResponse:
    """Compile one step's fragment: a sub-spec carrying only what that table answers."""
    if not columns:
        return invalid_spec(
            field="select",
            reason=f"nothing this query asks for reads {table.key}, so it cannot be joined in",
            expected=[table.key],
        )
    adapter = services.adapters.get(table.source)
    dialect = "" if adapter is None else adapter.dialect()
    if not dialect:
        return invalid_spec(
            field="from",
            reason=f'"{table.key}" names no configured source',
            expected=sorted(services.adapters),
        )
    sub: dict[str, object] = {"from": table.key, "select": list(columns)}
    if filters:
        sub["filters"] = [
            condition.model_dump(mode="json", by_alias=True, exclude_none=True)
            for condition in filters
        ]
    if isinstance(time_spec, TimeSpec):
        sub["time"] = time_spec.model_dump(mode="json", by_alias=True, exclude_none=True)
    try:
        parsed = parse_spec(sub)
    except SpecError as spec_error:  # a derivation that cannot be re-parsed is refused, not run
        return spec_error.to_error_response()
    # One row past the cap: the executor's probe is how a row cap is noticed rather than truncated.
    compiled = compile_spec(
        parsed,
        CompileContext(
            layer=services.layer,
            source=table.source,
            dialect=dialect,
            exec_ts=exec_ts,
            columns=services.columns,
            max_rows=services.limits.max_rows + 1,
            max_partitions=services.limits.max_partitions,
        ),
    )
    if compiled.error is not None or compiled.fragment is None:
        return compiled.error or invalid_spec(
            field="spec", reason=f"the compiler produced no statement for {table.key}"
        )
    semantics = services.layer.tables.get(table.key)
    cache = (
        StepCache(
            version_hash=services.layer.version_hash,
            mutability="unknown"
            if semantics is None or semantics.mutability is None
            else semantics.mutability.class_,
        )
        if semantics is not None and semantics.may_cache
        else None
    )
    return FragmentStep(
        name=name,
        source=table.source,
        table=table.key,
        fragment=compiled.fragment,
        limits=services.limits,
        cache=cache,
    )


# --- running the plan ----------------------------------------------------------------------


def run_federated(
    services: QueryServices,
    user: str,
    *,
    federated: FederatedQuery,
    spec: QuerySpec,
    exec_ts: datetime,
    applied_fixes: Sequence[str] = (),
) -> QueryAnswer:
    """Run one derived plan and answer with the rows plus the report that explains them."""
    plan = federated.plan
    fixes = _merge(applied_fixes, federated.fixes)
    sources = tuple(dict.fromkeys(step.source for step in plan.steps))
    if spec.dry_run:
        return QueryAnswer(
            user_id=user,
            status="dry_run",
            applied_fixes=fixes,
            plan=PlanInfo(
                notes=(
                    f"{len(plan.steps)} fragments over {len(sources)} sources "
                    f"({' + '.join(sources)}); the join runs locally, so nothing was fetched",
                )
            ),
            freshness=tuple(
                SegmentFreshness(source=step.source, table=step.table) for step in plan.steps
            ),
        )

    executor = services.executor
    own: FederatedExecutor | None = None
    if executor is None:
        # No shared executor was configured, so this call builds its own and its cache is cold: an
        # honest state to report (``freshness`` says "uncached") rather than a fake hit.
        own = FederatedExecutor(
            services.governor,
            layer=services.layer,
            clock=services.clock,
            spill_dir=services.spill_dir,
        )
        executor = own
    try:
        result = executor.execute(plan, user=user)
    finally:
        if own is not None:
            own.close()

    freshness = _freshness(result.quality)
    if not result.ok:
        answer = _refused(
            services,
            user,
            error=result.error
            or invalid_spec(field="spec", reason="the federated execution produced no answer"),
            spec=spec,
            exec_ts=exec_ts,
            sources=sources,
            applied_fixes=fixes,
        )
        # No rows, no handle and the report that says why: a partial join is never an answer.
        return answer.model_copy(update={"quality": result.quality, "freshness": freshness})

    summary = _summarise_joined(services, user, sources, result)
    cache_hit = not any(step.fetched for step in result.quality.steps)
    _record_federated(
        services,
        user,
        plan=plan,
        spec=spec,
        exec_ts=exec_ts,
        fixes=fixes,
        quality=result.quality,
        rows=summary.rows_total,
        cache_hit=cache_hit,
        fresh=spec.fresh,
    )
    return QueryAnswer(
        user_id=user,
        summary=summary,
        sample=summary.sample,
        result_handle=summary.handle,
        rows_total=summary.rows_total,
        cache_hit=cache_hit,
        freshness=freshness,
        plan=_plan(None, None).model_copy(
            update={
                "notes": (
                    f"{len(plan.steps)} fragments over {len(sources)} sources "
                    f"({' + '.join(sources)}), joined locally",
                    *result.quality.notes,
                )
            }
        ),
        applied_fixes=fixes,
        quality=result.quality,
    )


def _freshness(quality: QualityReport) -> tuple[SegmentFreshness, ...]:
    """One freshness row per source, with the ``fetched_at`` of the rows that were really read."""
    rows: list[SegmentFreshness] = []
    for step in quality.steps:
        rows.append(
            SegmentFreshness(
                source=step.source,
                table=step.table,
                status=step.cache_status,
                fetched_at=step.fetched_at,
                age_s=step.age_s,
                reason="" if step.fetched else "served from the fragment cache",
            )
        )
    return tuple(rows)


def _summarise_joined(
    services: QueryServices, user: str, sources: Sequence[str], result: FederatedResult
) -> ResultSummary:
    """Store the joined rows and summarise them, exactly as the single-source path does."""
    assert result.table is not None  # ``result.ok`` guarantees it
    table: pa.Table = result.table
    handle = services.governor.store.put(owner=user, source="+".join(sources), table=table)
    return summarise(table, handle=handle.handle)


def _record_federated(
    services: QueryServices,
    user: str,
    *,
    plan: FederatedPlan,
    spec: QuerySpec,
    exec_ts: datetime,
    fixes: tuple[str, ...],
    quality: QualityReport,
    rows: int,
    cache_hit: bool,
    fresh: bool,
) -> None:
    """Write the federated run to history and to the audit chain: every fragment, every source."""
    fetched = tuple(
        step.fetched_at
        for step in quality.steps
        if not step.fetched and step.fetched_at is not None
    )
    cache = CacheInfo(
        hit=cache_hit,
        fetched_at=fetched[0] if fetched else None,
        fresh=not fresh,
    )
    _record_history(
        services,
        lambda: HistoryEntry(
            user_id=user,
            exec_ts=exec_ts,
            spec=spec,
            fragments=tuple(
                RecordedFragment(
                    source=step.source,
                    dialect=step.fragment.dialect,
                    sql=step.fragment.sql,
                    tables=step.fragment.tables,
                )
                for step in plan.steps
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
                sources=tuple(dict.fromkeys(step.source for step in plan.steps)),
                row_count=max(0, rows),
            )
        )
    except Exception:  # noqa: BLE001 - the answer is the caller's; a lost audit row is ours
        logger.exception("could not audit a federated query", extra={"sources": len(plan.steps)})


def _merge(*groups: Sequence[str]) -> tuple[str, ...]:
    """Every reported repair, in order, without repeats."""
    seen: dict[str, None] = {}
    for group in groups:
        for fix in group:
            seen.setdefault(fix, None)
    return tuple(seen)
