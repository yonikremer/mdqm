"""Linter rules that read collected statistics.

The rules are pure: a missing statistic is silence, never a guess. The unknown-category rule
rejects only when the stored distribution is complete. A stale distribution is left to
:func:`mdqm.stats.guard.gate_spec`, which can run a bounded lookup; this rule calls that gate
without a probe, so a stale column is not rejected here.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mdqm.compiler.resolve import find_join
from mdqm.errors import ErrorCode, cost_too_high
from mdqm.linter.rules import (
    LintContext,
    LintState,
    Rule,
    RuleCase,
    RuleOutcome,
    Severity,
    Target,
)
from mdqm.models import ErrorResponse
from mdqm.spec.models import MeasureSelect
from mdqm.stats.estimate import estimate_result_rows, narrow_window
from mdqm.stats.guard import gate_spec
from mdqm.stats.models import (
    CategoryColumnStats,
    CollectionMethod,
    ColumnQuality,
    ObservedValue,
    PartitionExtent,
)
from mdqm.stats.sqlgen import sentinel_sentence

__all__ = ["STATS_RULES"]

_WHEN = datetime(2026, 11, 1, tzinfo=UTC)

_STATUS = CategoryColumnStats(
    source="billing",
    table="accounts",
    column="state",
    values=(
        ObservedValue(value="Active", count=64),
        ObservedValue(value="Closed", count=30),
        ObservedValue(value="Hold", count=6),
    ),
    other_count=0,
    rows_seen=100,
    collected_at=_WHEN,
    method=CollectionMethod.EXACT,
    distribution_complete=True,
)
_STATUS_KEY = "billing.accounts.state"

_JOIN_SPEC = {
    "from": "billing.usage_events",
    "select": ["event_id"],
    "joins": ["usage_to_customers"],
    "limit": 5,
}
_NULL_KEY = "billing.usage_events.client_no"

_HUGE = {
    "from": "crm.customers",
    "select": ["cust_id"],
    "limit": 5,
}
_GROUPED = {
    "from": "crm.customers",
    "select": ["status", {"measure": "credit_limit_r5000", "agg": "avg"}],
    "group_by": ["status"],
    "limit": 5,
}


def _quality(share: float) -> ColumnQuality:
    return ColumnQuality(
        source="billing",
        table="usage_events",
        column="client_no",
        null_share=share,
        distinct_count=10,
        rows_seen=100,
        method=CollectionMethod.EXACT,
        collected_at=_WHEN,
    )


def _extent(rows: int) -> PartitionExtent:
    return PartitionExtent(
        source="crm",
        table="customers",
        name="p2026",
        rows=rows,
        time_column="region",
        time_min="2026-01-01",
        time_max="2026-06-01",
        method=CollectionMethod.CATALOG,
        collected_at=_WHEN,
    )


class UnknownCategoryRule(Rule):
    """A filter value that the column never holds is a hard error, with real alternatives.

    Suggestions are the closest stored values (case, whitespace, Hebrew, transliteration and the
    value map's synonyms all count as the same value) and each one carries its frequency.
    Stale stats do not fire this rule: rejecting them without a lookup would drop a value that
    arrived after the last collection.
    """

    id = "spec.unknown_category"
    severity = Severity.ERROR
    autofix = False
    target = Target.SPEC
    title = "categorical filter value that never occurs"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        spec = state.spec
        if spec is None or not ctx.category_stats:
            return None
        result = gate_spec(
            spec,
            layer=ctx.layer,
            columns=ctx.columns,
            stats=ctx.category_stats,
            probe=None,
        )
        if result.error is None:
            return None
        return RuleOutcome(error=result.error)

    cases = (
        RuleCase(
            "a near miss is rejected with the real value and its frequency",
            fires=True,
            spec={
                "from": "billing.accounts",
                "select": ["client_no"],
                "filters": [{"column": "state", "op": "eq", "value": "activ"}],
                "limit": 5,
            },
            expect_code=ErrorCode.VALUE_NOT_IN_DATA.value,
            expect_fix="64 of 100",
            expect_suggestions=("active (64 of 100, 64%)",),
            context={"category_stats": {_STATUS_KEY: _STATUS}},
        ),
        RuleCase(
            "a value the column holds is left alone",
            spec={
                "from": "billing.accounts",
                "select": ["client_no"],
                "filters": [{"column": "state", "op": "eq", "value": "active"}],
                "limit": 5,
            },
            context={"category_stats": {_STATUS_KEY: _STATUS}},
        ),
    )


class SentinelNullRule(Rule):
    """A column the dictionary marks is read through NULLIF, and the spec says so.

    The spec itself does not change: NULLIF belongs in the SQL, which the compiler emits from
    the same sentence. Reporting it here means a lint-only caller still sees the fix.
    """

    id = "spec.sentinel_null"
    severity = Severity.NOTICE
    autofix = True
    target = Target.SPEC
    title = "sentinel empty string or zero read as NULL"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        spec = state.spec
        if spec is None or not ctx.sentinels:
            return None
        scope = ctx.scope(spec)
        if isinstance(scope, ErrorResponse):
            return None
        notes: list[str] = []
        for table in scope:
            for column in table.columns:
                if not _mentions(spec, column):
                    continue
                key = f"{table.key}.{column}"
                sentinels = ctx.sentinels.get(key)
                if not sentinels:
                    continue
                sentence = sentinel_sentence(column, sentinels)
                if sentence not in notes:
                    notes.append(sentence)
        if not notes:
            return None
        return RuleOutcome(spec=spec, fixes=tuple(notes))

    cases = (
        RuleCase(
            "a marked column is reported",
            fires=True,
            spec={"from": "crm.customers", "select": ["credit_limit"], "limit": 5},
            expect_spec={"from": "crm.customers", "select": ["credit_limit"], "limit": 5},
            expect_fix="NULLIF",
            context={"sentinels": {"crm.customers.credit_limit": ("0",)}},
        ),
        RuleCase(
            "an unmarked column is quiet",
            spec={"from": "crm.customers", "select": ["cust_id"], "limit": 5},
            context={"sentinels": {"crm.customers.credit_limit": ("0",)}},
        ),
    )


class JoinNullRateRule(Rule):
    """Warn when a join key is NULL often enough that the join will drop a large share of rows."""

    id = "spec.join_null_rate"
    severity = Severity.NOTICE
    autofix = True
    target = Target.SPEC
    title = "join key with a high NULL rate"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        spec = state.spec
        if spec is None or not spec.joins or not ctx.column_quality:
            return None
        notes: list[str] = []
        for name in spec.joins:
            join = find_join(ctx.layer, name)
            if join is None:
                continue
            for qualified in (join.left, join.right):
                quality = ctx.column_quality.get(qualified)
                if quality is None or quality.null_share is None:
                    continue
                if quality.null_share < ctx.null_warn_share:
                    continue
                percent = round(quality.null_share * 100)
                notes.append(
                    f"join {name} key {qualified} is NULL in {percent}% of rows; "
                    "rows with a NULL key are dropped by the join"
                )
        if not notes:
            return None
        return RuleOutcome(spec=spec, fixes=tuple(notes))

    cases = (
        RuleCase(
            "a mostly-null key is reported",
            fires=True,
            spec=_JOIN_SPEC,
            expect_spec=_JOIN_SPEC,
            expect_fix="NULL in 50% of rows",
            context={"column_quality": {_NULL_KEY: _quality(0.5)}},
        ),
        RuleCase(
            "a rare NULL is quiet",
            spec=_JOIN_SPEC,
            context={"column_quality": {_NULL_KEY: _quality(0.01)}},
        ),
    )


class ResultSizeRule(Rule):
    """Refuse an unaggregated read whose statistics say it will be huge, and warn below that.

    The estimate is ``table rows x filter selectivity``. It is not a run. A grouped spec is not
    refused: the agent already aggregated, which is the fix the error asks for.
    """

    id = "spec.result_size"
    severity = Severity.ERROR
    autofix = True
    target = Target.SPEC
    title = "estimated result is too large"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        spec = state.spec
        if spec is None:
            return None
        anchor = ctx.anchor(spec)
        if isinstance(anchor, ErrorResponse):
            return None
        table_rows = ctx.table_rows.get(anchor.key)
        if table_rows is None:
            return None
        grouped = bool(spec.group_by) or any(
            isinstance(item, MeasureSelect) for item in spec.select
        )
        cardinality = _group_cardinality(spec, anchor.key, ctx)
        estimated = estimate_result_rows(
            table_rows=table_rows,
            selectivity=_selectivity(spec, anchor.key, ctx),
            grouped=grouped,
            group_cardinality=cardinality,
        )
        if estimated > ctx.result_block_rows and not grouped:
            return RuleOutcome(
                error=cost_too_high(
                    table=anchor.key,
                    estimated=f"{estimated} rows",
                    limit=f"{ctx.result_block_rows} rows",
                )
            )
        if estimated > ctx.result_warn_rows:
            return RuleOutcome(
                spec=spec,
                fixes=(
                    f"estimated {estimated} rows from {anchor.key}; "
                    "aggregate with group_by or add a filter before running",
                ),
            )
        return None

    cases = (
        RuleCase(
            "an unaggregated read over the block is refused",
            fires=True,
            spec=_HUGE,
            expect_code=ErrorCode.COST_TOO_HIGH.value,
            expect_fix="group_by",
            context={"table_rows": {"crm.customers": 250_000}, "result_block_rows": 100_000},
        ),
        RuleCase(
            "a small table is quiet",
            spec=_HUGE,
            context={"table_rows": {"crm.customers": 20}, "result_block_rows": 100_000},
        ),
        RuleCase(
            "a grouped read over the warn threshold is reported, not refused",
            fires=True,
            spec=_GROUPED,
            expect_spec=_GROUPED,
            expect_fix="estimated",
            context={
                "table_rows": {"crm.customers": 250_000},
                "result_block_rows": 100_000,
                "result_warn_rows": 1_000,
            },
        ),
    )


class PartitionWeightRule(Rule):
    """Require a time window when the measured partitions are larger than the block.

    The existing partition rule refuses a missing key filter from the semantic layer. This one
    uses the measured sizes, including a table the layer has not marked as partitioned, and names
    a window that stays under the block.
    """

    id = "spec.partition_weight"
    severity = Severity.ERROR
    autofix = False
    target = Target.SPEC
    title = "partition sizes need a narrower filter"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        spec = state.spec
        if spec is None or spec.time is not None:
            return None
        anchor = ctx.anchor(spec)
        if isinstance(anchor, ErrorResponse):
            return None
        extents = ctx.partition_extents.get(anchor.key, ())
        if not extents:
            return None
        if _filters_time(spec, extents):
            return None
        total = sum(item.rows or 0 for item in extents)
        if total <= ctx.result_block_rows:
            return None
        window = narrow_window(extents, ctx.result_block_rows)
        narrower = None
        if window is not None:
            start, end, rows = window
            narrower = (
                f'{{"mode": "events_in_range", "from": "{start}", "to": "{end}"}} ({rows} rows)'
            )
        return RuleOutcome(
            error=cost_too_high(
                table=anchor.key,
                estimated=f"{total} rows across {len(extents)} partitions",
                limit=f"{ctx.result_block_rows} rows",
                narrower=narrower,
            )
        )

    cases = (
        RuleCase(
            "a fat partition without a time filter is refused",
            fires=True,
            spec=_HUGE,
            expect_code=ErrorCode.COST_TOO_HIGH.value,
            expect_fix="2026-01-01",
            context={
                "partition_extents": {"crm.customers": (_extent(250_000),)},
                "result_block_rows": 100_000,
            },
        ),
        RuleCase(
            "partitions under the block are quiet",
            spec=_HUGE,
            context={
                "partition_extents": {"crm.customers": (_extent(10),)},
                "result_block_rows": 100_000,
            },
        ),
    )


def _mentions(spec: object, column: str) -> bool:
    from mdqm.spec.models import QuerySpec

    if not isinstance(spec, QuerySpec):
        return False
    names = {item for item in spec.select if isinstance(item, str)}
    names.update(condition.column for condition in spec.filters)
    plain = {name.rpartition(".")[2] or name for name in names}
    return column in names or column in plain


def _selectivity(spec: object, table: str, ctx: LintContext) -> float:
    from mdqm.spec.models import FilterOp, QuerySpec

    if not isinstance(spec, QuerySpec):
        return 1.0
    factor = 1.0
    for condition in spec.filters:
        if condition.op is not FilterOp.EQ:
            continue
        column = condition.column.rpartition(".")[2] or condition.column
        stats = ctx.category_stats.get(f"{table}.{column}")
        if stats is None or stats.rows_seen <= 0 or not isinstance(condition.value, str):
            continue
        count = next((item.count for item in stats.values if item.value == condition.value), None)
        if count is None:
            continue
        factor *= count / stats.rows_seen
    return factor


def _group_cardinality(spec: object, table: str, ctx: LintContext) -> int | None:
    from mdqm.spec.models import QuerySpec

    if not isinstance(spec, QuerySpec) or not spec.group_by:
        return None
    cardinalities: list[int] = []
    for key in spec.group_by:
        column = key.rpartition(".")[2] or key
        stats = ctx.category_stats.get(f"{table}.{column}")
        if stats is not None and stats.distribution_complete:
            cardinalities.append(len(stats.values))
    if not cardinalities:
        return None
    product = 1
    for value in cardinalities:
        product *= max(1, value)
    return product


def _filters_time(spec: object, extents: tuple[PartitionExtent, ...]) -> bool:
    from mdqm.spec.models import QuerySpec

    if not isinstance(spec, QuerySpec):
        return False
    columns = {item.time_column for item in extents if item.time_column}
    filtered = {
        condition.column.rpartition(".")[2] or condition.column for condition in spec.filters
    }
    return bool(columns & filtered)


STATS_RULES: tuple[Rule, ...] = (
    UnknownCategoryRule(),
    SentinelNullRule(),
    JoinNullRateRule(),
    PartitionWeightRule(),
    ResultSizeRule(),
)
