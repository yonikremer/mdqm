"""The spec rules: the primary path, where the agent writes a structured spec (DESIGN section 5).

Each rule here reads a :class:`~mdqm.spec.models.QuerySpec` and either repairs it (and reports the
repair) or refuses it with the ``ErrorResponse`` a weak agent can act on. Nothing in this module
touches a source, a file or a clock: names come from the semantic layer and the catalog, time comes
from the injected ``exec_ts``.

The split of responsibility with the compiler (C09) and the planners (C07, C08) is deliberate:

* the linter repairs what is *expressible in the spec* -- a label, a missing cap, a derived bound;
* a fix the planners own (the default partition window, the event-time predicate of a time intent)
  is left to them, because they already report it in ``applied_fixes``. The linter only refuses the
  case they would refuse, so the agent hears about it before a compile fails.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from mdqm.compiler.resolve import (
    TIME_BUCKETS,
    ColumnRef,
    TableRef,
    covering_join,
    on_text,
    resolve_column,
    resolve_table,
)
from mdqm.errors import (
    ErrorCode,
    insertion_time_no_bound,
    invalid_spec,
    join_not_approved,
    partition_scan_all,
)
from mdqm.linter.rules import (
    LintContext,
    LintState,
    Rule,
    RuleCase,
    RuleOutcome,
    Severity,
    Target,
    duration_text,
)
from mdqm.models import ErrorResponse
from mdqm.models.partitioning import Partitioning, PartitionStrategy
from mdqm.planner.partitions import Range, estimate_partition_scan
from mdqm.planner.time import event_column, ingestion_column
from mdqm.semantic.models import Measure, TimeRole
from mdqm.semantic.values import resolve_value
from mdqm.spec.models import Filter, FilterOp, MeasureSelect, QuerySpec

__all__ = ["SPEC_RULES"]

#: Stands in for a column while a normalisation template is rendered, to see what it does to it.
_SENTINEL = "mdqm_col"

#: Ops that order or search a value: they need a number or an instant, never a categorical label.
_ORDERING = frozenset(
    {FilterOp.GT, FilterOp.GTE, FilterOp.LT, FilterOp.LTE, FilterOp.RANGE, FilterOp.CONTAINS}
)

#: Ops a categorical field can be compared with (label or source code).
_CATEGORICAL = frozenset({FilterOp.EQ, FilterOp.NE, FilterOp.IN})


# --- shared helpers ----------------------------------------------------------------------


def _plain(name: str) -> str:
    """The column part of a possibly qualified spec name (``customers.status`` -> ``status``)."""
    return name.rpartition(".")[2] or name


def _scope(ctx: LintContext, spec: QuerySpec) -> tuple[TableRef, ...] | ErrorResponse:
    return ctx.scope(spec)


def _resolve(
    ctx: LintContext, scope: tuple[TableRef, ...], name: str, path: str
) -> ColumnRef | ErrorResponse:
    return resolve_column(name, scope=scope, layer=ctx.layer, field=path)


def _column_refs(spec: QuerySpec, scope: tuple[TableRef, ...]) -> tuple[tuple[str, str], ...]:
    """Every ``(path, name)`` in the spec that must name a column, in document order.

    ``select`` items that are measures are handled by the measure rule; a ``group_by`` key or a
    ``select`` entry that is a time bucket (``month``) is skipped when no table has that column, the
    same way the compiler reads it.
    """
    taken = {*spec.selected_names(), *spec.group_by}
    refs: list[tuple[str, str]] = []
    for index, item in enumerate(spec.select):
        if isinstance(item, str):
            refs.append((f"select.{index}", item))
    for index, condition in enumerate(spec.filters):
        refs.append((f"filters.{index}.column", condition.column))
    for index, key in enumerate(spec.group_by):
        refs.append((f"group_by.{index}", key))
    for index, order in enumerate(spec.order_by):
        if order.by not in taken:
            refs.append((f"order_by.{index}.by", order.by))
    buckets = tuple(
        (path, name)
        for path, name in refs
        if name.strip() in TIME_BUCKETS and not any(name.strip() in t.columns for t in scope)
    )
    return tuple(ref for ref in refs if ref not in buckets)


def _join_hint(ctx: LintContext, scope: tuple[TableRef, ...], name: str) -> ErrorResponse | None:
    """The ``JOIN_NOT_APPROVED`` refusal when one approved join would make ``name`` reachable.

    This is the difference between "that column does not exist" and "you left out the join that
    reaches it", and the ``fix`` carries the whole approved ON clause so the agent can copy it.
    """
    anchor = scope[0]
    in_scope = {table.key for table in scope}
    for candidate in ctx.tables():
        if candidate.key in in_scope:
            continue
        probe = resolve_column(name, scope=(candidate,), layer=ctx.layer, field="")
        if not isinstance(probe, ColumnRef):
            continue
        join = covering_join(ctx.layer, anchor.key, candidate.key)
        if join is None:
            continue
        return join_not_approved(
            left=anchor.key,
            right=candidate.key,
            approved_on=on_text(join, anchor=anchor, partner=candidate),
        )
    return None


def _instant(value: object) -> datetime | None:
    """The value as an aware UTC datetime, or ``None`` when it is not an instant at all."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else None
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment


def _is_identity(measure: Measure, qualified: str) -> bool:
    """True when this source stores the measure already in canonical units (``{col}``)."""
    rendered = measure.normalize_for(qualified, _SENTINEL)
    if rendered is None:
        return False
    try:
        return bool(sqlglot.parse_one(rendered) == sqlglot.parse_one(_SENTINEL))
    except SqlglotError:
        return False


def _key_columns(partitioning: Partitioning) -> tuple[str, ...]:
    """The columns a partition key ranges over: the declared key, or the expression's columns."""
    if partitioning.strategy is not PartitionStrategy.EXPRESSION:
        return partitioning.key
    try:
        parsed = sqlglot.parse_one(partitioning.expression or "", read="duckdb")
    except SqlglotError:
        return ()
    return tuple(sorted({column.name for column in parsed.find_all(exp.Column)}))


# --- rules --------------------------------------------------------------------------------


class TableRule(Rule):
    """``from`` must name a table the semantic layer or the catalog knows.

    An unknown table has no semantics, no columns and no time roles, so every later rule would be
    reasoning about nothing; this runs first and answers with "did you mean".
    """

    id = "spec.table"
    severity = Severity.ERROR
    autofix = False
    target = Target.SPEC
    title = "unknown table in 'from'"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        spec = state.spec
        if spec is None:
            return None
        resolved = resolve_table(spec.from_table, tables=ctx.tables())
        if not isinstance(resolved, ErrorResponse):
            return None
        return RuleOutcome(error=resolved)

    cases = (
        RuleCase(
            "misspelled table",
            fires=True,
            spec={"from": "customerz", "select": ["cust_id"], "limit": 5},
            expect_code=ErrorCode.UNKNOWN_TABLE.value,
            expect_fix="known tables",
            expect_suggestions=("crm.customers",),
        ),
        RuleCase("known table", spec={"from": "crm.customers", "select": ["cust_id"], "limit": 5}),
    )


class MeasureRule(Rule):
    """A ``{"measure": ...}`` select item must name a measure the semantic layer defines.

    The measure is the only thing that knows how each source stores the number, so an unknown one
    cannot be compiled into anything: the fix lists the names that exist.
    """

    id = "spec.measure"
    severity = Severity.ERROR
    autofix = False
    target = Target.SPEC
    title = "unknown measure"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        spec = state.spec
        if spec is None:
            return None
        for index, item in enumerate(spec.select):
            if not isinstance(item, MeasureSelect) or item.measure in ctx.layer.measures:
                continue
            return RuleOutcome(
                error=invalid_spec(
                    field=f"select.{index}.measure",
                    reason=f'the semantic layer has no measure named "{item.measure}"',
                    expected=sorted(ctx.layer.measures),
                )
            )
        return None

    cases = (
        RuleCase(
            "unknown measure",
            fires=True,
            spec={
                "from": "crm.customers",
                "select": ["cust_id", {"measure": "credit_limit"}],
                "group_by": ["cust_id"],
                "limit": 5,
            },
            expect_code=ErrorCode.INVALID_SPEC.value,
            expect_fix="credit_limit_r5000",
        ),
        RuleCase(
            "known measure",
            spec={
                "from": "crm.customers",
                "select": ["cust_id", {"measure": "credit_limit_r5000", "agg": "avg"}],
                "group_by": ["cust_id"],
                "limit": 5,
            },
        ),
    )


class JoinRule(Rule):
    """``joins`` must name approved joins, and they must touch the table the spec reads.

    A join is a fact a human approved, with its own ON clause and format conversion. A name that
    does not exist, or one that connects two other tables, is refused here rather than becoming a
    cross product.
    """

    id = "spec.join"
    severity = Severity.ERROR
    autofix = False
    target = Target.SPEC
    title = "join is not approved for this table"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        spec = state.spec
        if spec is None:
            return None
        scope = ctx.scope(spec)
        if isinstance(scope, ErrorResponse):
            return RuleOutcome(error=scope)
        return None

    cases = (
        RuleCase(
            "unknown join name",
            fires=True,
            spec={
                "from": "crm.customers",
                "select": ["cust_id"],
                "joins": ["customers_to_usage"],
                "limit": 5,
            },
            expect_code=ErrorCode.JOIN_NOT_APPROVED.value,
            expect_fix="ask an admin to approve",
        ),
        RuleCase(
            "join that does not touch the table",
            fires=True,
            spec={
                "from": "erp.customers",
                "select": ["cust_no"],
                "joins": ["usage_to_customers"],
                "limit": 5,
            },
            expect_code=ErrorCode.INVALID_SPEC.value,
            expect_fix="joins.0",
        ),
        RuleCase(
            "approved join",
            spec={
                "from": "crm.customers",
                "select": ["cust_id"],
                "joins": ["usage_to_customers"],
                "limit": 5,
            },
        ),
    )


class ColumnRule(Rule):
    """Every column the spec names must exist on a table the query reads.

    An unknown column is answered with "did you mean" from the columns that really exist; a column
    that only another table has is answered with the approved join and its ON clause, because that
    is the change the agent actually has to make.

    A table with no catalog columns is not judged: an empty column list means the caller did not
    pass the catalog, not that the table is empty.
    """

    id = "spec.column"
    severity = Severity.ERROR
    autofix = False
    target = Target.SPEC
    title = "unknown column"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        spec = state.spec
        if spec is None:
            return None
        scope = _scope(ctx, spec)
        if isinstance(scope, ErrorResponse):
            return RuleOutcome(error=scope)
        if not any(table.columns for table in scope):
            return None
        for path, name in _column_refs(spec, scope):
            resolved = _resolve(ctx, scope, name, path)
            if isinstance(resolved, ColumnRef):
                continue
            hint = _join_hint(ctx, scope, name)
            return RuleOutcome(error=hint or resolved)
        return None

    cases = (
        RuleCase(
            "unknown column",
            fires=True,
            spec={"from": "crm.customers", "select": ["custid"], "limit": 5},
            expect_code=ErrorCode.UNKNOWN_COLUMN.value,
            expect_fix="crm.customers's columns",
            expect_suggestions=("cust_id",),
        ),
        RuleCase(
            "column only another table has, with an approved join",
            fires=True,
            spec={"from": "crm.customers", "select": ["units"], "limit": 5},
            expect_code=ErrorCode.JOIN_NOT_APPROVED.value,
            expect_fix="LPAD",
        ),
        RuleCase(
            "known columns",
            spec={
                "from": "crm.customers",
                "select": ["cust_id", "status"],
                "filters": [{"column": "status", "op": "eq", "value": "A"}],
                "group_by": ["cust_id", "status"],
                "limit": 5,
            },
        ),
    )


class CategoricalOrderingRule(Rule):
    """A categorical field cannot be ordered or searched with ``contains``.

    The source stores codes (``A``), the spec carries canonical values (``active``), and an
    ordering would order the codes -- which is not the business order and is not even stable across
    the three sources. Refused, with the canonical values in the fix.
    """

    id = "spec.categorical_ordering"
    severity = Severity.ERROR
    autofix = False
    target = Target.SPEC
    title = "ordering or 'contains' on a categorical column"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        spec = state.spec
        if spec is None:
            return None
        scope = _scope(ctx, spec)
        if isinstance(scope, ErrorResponse):
            return None
        for index, condition in enumerate(spec.filters):
            if condition.op not in _ORDERING:
                continue
            resolved = _resolve(ctx, scope, condition.column, f"filters.{index}.column")
            if not isinstance(resolved, ColumnRef) or resolved.value_map is None:
                continue
            return RuleOutcome(
                error=invalid_spec(
                    field=f"filters.{index}",
                    reason=(
                        f'"{condition.column}" is categorical (translated by a value map); '
                        f'op "{condition.op.value}" orders or searches its source codes, '
                        "which is not the business value"
                    ),
                    expected=sorted(resolved.value_map.canonical),
                )
            )
        return None

    cases = (
        RuleCase(
            "greater-than on a status",
            fires=True,
            spec={
                "from": "crm.customers",
                "select": ["cust_id"],
                "filters": [{"column": "status", "op": "gt", "value": 2}],
                "limit": 5,
            },
            expect_code=ErrorCode.INVALID_SPEC.value,
            expect_fix="churned",
        ),
        RuleCase(
            "equality on a status",
            spec={
                "from": "crm.customers",
                "select": ["cust_id"],
                "filters": [{"column": "status", "op": "eq", "value": "A"}],
                "limit": 5,
            },
        ),
    )


class ValueRule(Rule):
    """A categorical filter value becomes its canonical value, or is refused with the valid set.

    The agent (or the user it transcribes) writes ``פעיל``, ``active``, ``A`` or ``1``; the
    semantic layer names the value once and each source spells it its own way. This rule repairs
    the *spec* to the canonical spelling and reports every replacement, so the compiler then maps
    one canonical value to each source's code.

    A label that matches more than one canonical value is **never** repaired: picking one would
    change the answer, so it is refused with the candidates. The same holds for an unknown label.
    """

    id = "spec.value"
    severity = Severity.ERROR
    autofix = True
    target = Target.SPEC
    title = "categorical label -> canonical value"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        spec = state.spec
        if spec is None:
            return None
        scope = _scope(ctx, spec)
        if isinstance(scope, ErrorResponse):
            return None
        filters = list(spec.filters)
        fixes: list[str] = []
        for index, condition in enumerate(spec.filters):
            if condition.op not in _CATEGORICAL:
                continue
            resolved = _resolve(ctx, scope, condition.column, f"filters.{index}.column")
            if not isinstance(resolved, ColumnRef) or resolved.value_map is None:
                continue
            path = f"filters.{index}.value"
            single = condition.op is not FilterOp.IN
            supplied = (condition.value,) if single else tuple(condition.value)
            canonical: list[str] = []
            for item in supplied:
                if not isinstance(item, (str, int)) or isinstance(item, bool):
                    return RuleOutcome(
                        error=invalid_spec(
                            field=path,
                            reason="a categorical column compares against a label or a source code",
                        )
                    )
                resolution = resolve_value(
                    resolved.value_map, item, field=path, column=resolved.map_key
                )
                if resolution.error is not None or resolution.canonical is None:
                    return RuleOutcome(error=resolution.error)
                canonical.append(resolution.canonical)
                if str(item) != resolution.canonical:
                    fixes.append(
                        f"{path}: replaced {str(item)!r} with the canonical value "
                        f"{resolution.canonical!r} of value map "
                        f'"{_map_name(ctx, resolved.map_key)}"'
                    )
            if canonical != [str(item) for item in supplied]:
                value: object = canonical[0] if single else tuple(canonical)
                filters[index] = condition.model_copy(update={"value": value})
        if not fixes:
            return None
        return RuleOutcome(
            spec=spec.model_copy(update={"filters": tuple(filters)}), fixes=tuple(fixes)
        )

    cases = (
        RuleCase(
            "Hebrew label becomes the canonical value",
            fires=True,
            spec={
                "from": "billing.accounts",
                "select": ["client_no"],
                "filters": [{"column": "state", "op": "eq", "value": "פעיל"}],
                "limit": 5,
            },
            expect_spec={
                "from": "billing.accounts",
                "select": ["client_no"],
                "filters": [{"column": "state", "op": "eq", "value": "active"}],
                "limit": 5,
            },
            expect_fix="canonical value 'active'",
        ),
        RuleCase(
            "source code becomes the canonical value",
            fires=True,
            spec={
                "from": "erp.customers",
                "select": ["cust_no"],
                "filters": [{"column": "status_code", "op": "in", "value": ["1", "3"]}],
                "limit": 5,
            },
            expect_spec={
                "from": "erp.customers",
                "select": ["cust_no"],
                "filters": [
                    {"column": "status_code", "op": "in", "value": ["active", "suspended"]}
                ],
                "limit": 5,
            },
            expect_fix="canonical value",
        ),
        RuleCase(
            "unknown label is refused, not guessed",
            fires=True,
            spec={
                "from": "billing.accounts",
                "select": ["client_no"],
                "filters": [{"column": "state", "op": "eq", "value": "activ"}],
                "limit": 5,
            },
            expect_code=ErrorCode.UNKNOWN_VALUE.value,
            expect_fix="active",
            expect_suggestions=("active",),
        ),
        RuleCase(
            "already canonical",
            spec={
                "from": "billing.accounts",
                "select": ["client_no"],
                "filters": [{"column": "state", "op": "eq", "value": "active"}],
                "limit": 5,
            },
        ),
    )


class RawColumnMeasureRule(Rule):
    """A raw source column that a measure covers is compared in stored units, so it is refused.

    ``erp.customers.limit_k`` holds the credit limit divided by ten, and
    ``billing.accounts.credit_limit`` holds it before rounding. Selecting or filtering the raw
    column answers in the source's own units, and the spec has no way to say "this column,
    normalised" -- only a measure is normalised.
    Repairing that here would mean inventing an aggregation the agent never asked for, which
    changes the answer, so the rule refuses and names the measure to select instead.

    When the source stores the measure already in canonical units (a ``{col}`` template) the raw
    column *is* the business value and the rule stays silent.
    """

    id = "spec.raw_column_measure"
    severity = Severity.ERROR
    autofix = False
    target = Target.SPEC
    title = "raw column that a measure covers"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        spec = state.spec
        if spec is None:
            return None
        scope = _scope(ctx, spec)
        if isinstance(scope, ErrorResponse):
            return None
        for path, name in _column_refs(spec, scope):
            resolved = _resolve(ctx, scope, name, path)
            if not isinstance(resolved, ColumnRef) or resolved.value_map is not None:
                continue
            qualified = f"{resolved.table.key}.{resolved.column}"
            for measure_name in sorted(ctx.layer.measures):
                measure = ctx.layer.measures[measure_name]
                if qualified not in measure.sources or _is_identity(measure, qualified):
                    continue
                return RuleOutcome(
                    error=invalid_spec(
                        field=path,
                        reason=(
                            f'"{name}" stores the measure "{measure_name}" in this source\'s own '
                            "units, so comparing or returning it raw answers in those units, not "
                            "in the business value"
                        ),
                        expected=[measure_name],
                    )
                )
        return None

    cases = (
        RuleCase(
            "raw column in source units",
            fires=True,
            spec={"from": "erp.customers", "select": ["limit_k"], "limit": 5},
            expect_code=ErrorCode.INVALID_SPEC.value,
            expect_fix="credit_limit_r5000",
        ),
        RuleCase(
            "raw column in canonical units",
            spec={"from": "crm.customers", "select": ["credit_limit"], "limit": 5},
        ),
    )


class InsertionBoundRule(Rule):
    """A filter on the ingestion time gets the bound on the indexed event column it implies.

    ``insertion_time`` is not indexed, so filtering it alone reads the whole table. The approved
    lag makes the bound provable: a row with ``insertion_time >= X`` has
    ``start_time = insertion_time - lag >= X - max_lag``, so the derived predicate can only narrow
    the scan and never loses a row. The original filter stays.

    Without a measured lag (or after a failed nightly verification) no bound can be derived, so the
    rule refuses instead of guessing.
    """

    id = "spec.insertion_bound"
    severity = Severity.ERROR
    autofix = True
    target = Target.SPEC
    title = "ingestion-time filter -> derived event-time bound"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        spec = state.spec
        if spec is None:
            return None
        anchor = ctx.anchor(spec)
        if isinstance(anchor, ErrorResponse):
            return None
        semantics = anchor.semantics
        chosen = ingestion_column(semantics)
        if chosen is None or chosen[1].role is not TimeRole.INGESTION_TIME:
            return None
        name, column = chosen
        lower = _lower_bound(spec, name)
        if lower is None:
            return None
        event = event_column(semantics)
        target = column.derives_bound_on
        if spec.time is not None:
            return None  # the time planner owns the bound when the spec states a time intent
        if any(_plain(condition.column) == target for condition in spec.filters):
            return None  # already bounded, on the event column itself
        if target is None or column.lag is None or name in ctx.lag_unverified:
            return RuleOutcome(
                error=insertion_time_no_bound(
                    table=anchor.key,
                    event_column=target or (event[0] if event is not None else name),
                )
            )
        bound = lower - column.lag.maximum
        derived = Filter(column=target, op=FilterOp.GTE, value=bound.isoformat())
        return RuleOutcome(
            spec=spec.model_copy(update={"filters": (*spec.filters, derived)}),
            fixes=(
                f'added a bound on "{target}" (>= {bound.isoformat()}) derived from the "{name}" '
                f"filter and its approved max lag {column.lag.maximum}; the original predicate "
                "stays, so no row is lost",
            ),
        )

    cases = (
        RuleCase(
            "ingestion filter derives the event bound",
            fires=True,
            spec={
                "from": "billing.usage_events",
                "select": ["event_id"],
                "filters": [
                    {"column": "insertion_time", "op": "gte", "value": "2026-11-25T15:00:00"}
                ],
                "limit": 5,
            },
            expect_spec={
                "from": "billing.usage_events",
                "select": ["event_id"],
                "filters": [
                    {"column": "insertion_time", "op": "gte", "value": "2026-11-25T15:00:00"},
                    {"column": "start_time", "op": "gte", "value": "2026-11-25T09:00:00+00:00"},
                ],
                "limit": 5,
            },
            expect_fix="derived from the",
        ),
        RuleCase(
            "already bounded on the event column",
            spec={
                "from": "billing.usage_events",
                "select": ["event_id"],
                "filters": [
                    {"column": "insertion_time", "op": "gte", "value": "2026-11-25T15:00:00"},
                    {"column": "start_time", "op": "gte", "value": "2026-11-25T09:00:00"},
                ],
                "limit": 5,
            },
        ),
        RuleCase(
            "no approved lag, so no bound can be derived",
            fires=True,
            spec={
                "from": "billing.usage_events",
                "select": ["event_id"],
                "filters": [
                    {"column": "insertion_time", "op": "gte", "value": "2026-11-25T15:00:00"}
                ],
                "limit": 5,
            },
            layer={
                "version": 1,
                "tables": {
                    "billing.usage_events": {
                        "key": ["event_id"],
                        "time_columns": {
                            "start_time": {"role": "event_time", "indexed": True},
                            "insertion_time": {"role": "ingestion_time"},
                        },
                    }
                },
            },
            columns={"billing.usage_events": ("event_id", "units", "start_time", "insertion_time")},
            expect_code=ErrorCode.INSERTION_TIME_NO_BOUND.value,
            expect_fix="no bound can be derived",
        ),
    )


class PartitionRule(Rule):
    """A table that needs a partition-key filter must get one, or the query is refused.

    Where the table allows a query without a key filter, the partition planner (C08) injects the
    default window and reports it itself; this rule stays silent then. What it adds is the refusal
    *before* a compile, with the same code, the same window suggestion and the same scan estimate an
    agent sees from the planner.
    """

    id = "spec.partition"
    severity = Severity.ERROR
    autofix = False
    target = Target.SPEC
    title = "no filter on the partition key"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        spec = state.spec
        if spec is None or spec.time is not None:
            return None
        anchor = ctx.anchor(spec)
        if isinstance(anchor, ErrorResponse):
            return None
        partitioning = anchor.semantics.partitioning
        if partitioning is None or partitioning.strategy is PartitionStrategy.NONE:
            return None
        keys = _key_columns(partitioning)
        if not keys or {_plain(condition.column) for condition in spec.filters} & set(keys):
            return None
        window = anchor.semantics.default_window
        injectable = (
            window is not None
            and not partitioning.require_key_filter
            and partitioning.strategy is PartitionStrategy.RANGE
            and partitioning.granularity is not None
            and len(keys) == 1
        )
        if injectable:
            return None
        estimate = estimate_partition_scan(
            partitioning, Range(ctx.now - (window or _THIRTY_DAYS), ctx.now)
        )
        return RuleOutcome(
            error=partition_scan_all(
                table=anchor.key,
                key=" ".join(partitioning.key) or (partitioning.expression or ""),
                granularity=(
                    partitioning.granularity.value
                    if partitioning.granularity is not None
                    else partitioning.strategy.value
                ),
                default_window=duration_text(window) if window is not None else None,
                partitions_if_fixed=estimate.label,
            )
        )

    cases = (
        RuleCase(
            "nothing filters the monthly key",
            fires=True,
            spec={"from": "billing.usage_events", "select": ["event_id"], "limit": 5},
            expect_code=ErrorCode.PARTITION_SCAN_ALL.value,
            expect_fix="events_in_range",
        ),
        RuleCase(
            "a time intent bounds the key",
            spec={
                "from": "billing.usage_events",
                "select": ["event_id"],
                "time": {"mode": "events_in_range", "last": "30d"},
                "limit": 5,
            },
        ),
    )


class LimitRule(Rule):
    """A spec with no ``limit`` gets the server row cap.

    The cap is what the server applies anyway, so writing it into the spec changes nothing an agent
    asked for and keeps an unbounded read off the primary database (principle 6).
    """

    id = "spec.limit"
    severity = Severity.NOTICE
    autofix = True
    target = Target.SPEC
    title = "missing limit"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        spec = state.spec
        if spec is None or spec.limit is not None:
            return None
        return RuleOutcome(
            spec=spec.model_copy(update={"limit": ctx.max_rows}),
            fixes=(
                f"applied the server row cap of {ctx.max_rows} because the spec names no limit",
            ),
        )

    cases = (
        RuleCase(
            "no limit",
            fires=True,
            spec={"from": "crm.customers", "select": ["cust_id"]},
            expect_spec={"from": "crm.customers", "select": ["cust_id"], "limit": 1000},
            expect_fix="row cap of 1000",
        ),
        RuleCase("limit given", spec={"from": "crm.customers", "select": ["cust_id"], "limit": 5}),
    )


#: The order the rules run in: names first (a rule cannot reason about an unresolvable column),
#: then values, then the shapes that protect the source, then the cap.
SPEC_RULES: tuple[Rule, ...] = (
    TableRule(),
    JoinRule(),
    MeasureRule(),
    ColumnRule(),
    CategoricalOrderingRule(),
    ValueRule(),
    RawColumnMeasureRule(),
    InsertionBoundRule(),
    PartitionRule(),
    LimitRule(),
)


def _map_name(ctx: LintContext, map_key: str) -> str:
    """The value map's own name, for a sentence an agent can look up."""
    for name, value_map in sorted(ctx.layer.value_maps.items()):
        if map_key in value_map.sources:
            return name
    return map_key


def _lower_bound(spec: QuerySpec, column: str) -> datetime | None:
    """The strictest lower bound a filter puts on ``column``, as an instant."""
    found: list[datetime | None] = []
    for condition in spec.filters:
        if _plain(condition.column) != column:
            continue
        if condition.op in (FilterOp.GTE, FilterOp.GT):
            found.append(_instant(condition.value))
        elif condition.op is FilterOp.RANGE and isinstance(condition.value, tuple):
            found.append(_instant(condition.value[0]))
    return max((value for value in found if value is not None), default=None)


#: The window the partition planner suggests when a table declares none (DESIGN section 5).
_THIRTY_DAYS = timedelta(days=30)
