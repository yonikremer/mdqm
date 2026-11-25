"""The single-source compiler: one validated spec in, one :class:`Fragment` out (task C09).

The spec is the contract an agent writes; this module is where it becomes the exact SQL one source
receives. Five things happen, in this order, and each one is a module that already owns its rule:

1. **Names resolve** against the catalog and the semantic layer (:mod:`mdqm.compiler.resolve`), so
   no value from the spec can become an identifier and no join can be invented.
2. **Measures normalise** through the semantic layer's per-source template
   (:func:`~mdqm.semantic.measures.render_normalize`), and every conversion is reported as an
   :class:`~mdqm.semantic.measures.AppliedNormalization`.
3. **Time** becomes predicates on the right column through the time planner
   (:func:`~mdqm.planner.time.plan_time`): event time for a period, ingestion time plus its derived
   bound for "what is new".
4. **Partitions** are planned from those predicates
   (:func:`~mdqm.planner.partitions.plan_partitions`), which either adds a pruning-safe predicate or
   refuses the query outright.
5. **The statement is finished and checked**: ``exec_ts`` is substituted
   (:func:`~mdqm.sql.exec_ts.rewrite_sql`), then the read-only guard
   (:func:`~mdqm.sql.guard.guard`) inspects the SQL this module compiled, not just the SQL an agent
   wrote.

The SQL is built as a sqlglot AST and rendered once; **no string concatenation of a spec value
happens anywhere** (AGENTS.md section 5). Everything except the semantic normalisation templates —
which are allowlisted at load time — is assembled from :mod:`sqlglot` expression nodes.

Errors are values, never exceptions to the agent: :attr:`CompiledQuery.error` holds the
``ErrorResponse`` with a code, a problem and an exact ``fix``, and `fragment` is then ``None``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Final, NoReturn, cast

import sqlglot
from sqlglot import exp

from mdqm.compiler.resolve import (
    TIME_BUCKETS,
    ColumnRef,
    TableRef,
    covering_join,
    find_join,
    join_tables,
    known_tables,
    map_entry,
    on_expression,
    on_text,
    resolve_column,
    resolve_table,
)
from mdqm.errors import invalid_spec, join_not_approved, unknown_value
from mdqm.models import ErrorResponse, Fragment
from mdqm.planner.partitions import PartitionPlan, plan_partitions
from mdqm.planner.time import TimePlan, event_column, plan_time
from mdqm.semantic.measures import (
    AppliedNormalization,
    normalizations_for,
    render_normalize,
)
from mdqm.semantic.models import Measure, SemanticLayer
from mdqm.semantic.values import reverse_value
from mdqm.spec.durations import Duration as SpecDuration
from mdqm.spec.models import Agg, Filter, FilterOp, MeasureSelect, QuerySpec, SelectItem
from mdqm.sql.exec_ts import SUPPORTED_DIALECTS, rewrite_sql
from mdqm.sql.guard import guard

__all__ = [
    "DEFAULT_MAX_ROWS",
    "CompileContext",
    "CompiledQuery",
    "compile_spec",
]

#: The row cap the server injects when the spec names none: protect the primary database first.
DEFAULT_MAX_ROWS: Final[int] = 1_000

#: Comparison builders, so a filter operator maps to one node type and not to a string.
_COMPARISONS: Final[
    Mapping[FilterOp, Callable[[exp.Expression, exp.Expression], exp.Expression]]
] = {
    FilterOp.EQ: lambda left, right: exp.EQ(this=left, expression=right),
    FilterOp.NE: lambda left, right: exp.NEQ(this=left, expression=right),
    FilterOp.GT: lambda left, right: exp.GT(this=left, expression=right),
    FilterOp.GTE: lambda left, right: exp.GTE(this=left, expression=right),
    FilterOp.LT: lambda left, right: exp.LT(this=left, expression=right),
    FilterOp.LTE: lambda left, right: exp.LTE(this=left, expression=right),
}

#: Aggregates over one expression. ``count_distinct`` is built separately (``COUNT(DISTINCT x)``).
_AGGREGATES: Final[Mapping[Agg, Callable[[exp.Expression], exp.Expression]]] = {
    Agg.AVG: lambda node: exp.Avg(this=node),
    Agg.SUM: lambda node: exp.Sum(this=node),
    Agg.MIN: lambda node: exp.Min(this=node),
    Agg.MAX: lambda node: exp.Max(this=node),
    Agg.COUNT: lambda node: exp.Count(this=node),
}

#: Ops that order a value. They need a number or an instant, never a categorical label.
_ORDERING: Final[frozenset[FilterOp]] = frozenset(
    {FilterOp.GT, FilterOp.GTE, FilterOp.LT, FilterOp.LTE, FilterOp.RANGE}
)


@dataclass(frozen=True, slots=True)
class CompileContext:
    """Everything the compiler needs about *this* execution and *this* source.

    ``columns`` is the catalog's column list per table (the adapter's ``extract_catalog``), which is
    what makes an identifier emitted here a fact rather than a guess. ``exec_ts`` is one value per
    execution (C02): every literal the SQL contains is derived from it, so a re-run sees the same
    window.
    """

    layer: SemanticLayer
    source: str
    dialect: str
    exec_ts: datetime
    columns: Mapping[str, Sequence[str]]
    max_rows: int = DEFAULT_MAX_ROWS
    max_partitions: int | None = None
    lag_unverified: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class CompiledQuery:
    """The result of compiling one spec: a fragment, or the refusal that stopped it.

    ``applied_fixes`` and ``notes`` are reported, never silent (principle 10): a value-map
    translation, a normalisation, an injected default window and a partition rewrite all appear in
    them. ``time_plan`` and ``partition_plan`` are kept so a caller can reuse their estimates.
    """

    request: QuerySpec
    exec_ts: datetime
    fragment: Fragment | None = None
    error: ErrorResponse | None = None
    applied_fixes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    normalizations: tuple[AppliedNormalization, ...] = ()
    time_plan: TimePlan | None = None
    partition_plan: PartitionPlan | None = None
    output_columns: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when a runnable fragment was produced."""
        return self.fragment is not None and self.error is None


class _Stop(Exception):
    """Internal: carries the agent-facing refusal out of a deep helper. Never escapes."""

    def __init__(self, error: ErrorResponse) -> None:
        super().__init__(error.problem)
        self.error = error


def compile_spec(spec: QuerySpec, context: CompileContext) -> CompiledQuery:
    """Compile one validated spec for one source into a :class:`Fragment`.

    Args:
        spec: a spec that already passed :func:`mdqm.spec.parse_spec`.
        context: the semantic layer, the target source and dialect, the catalog columns and the
            single ``exec_ts`` of this execution.

    Returns:
        A :class:`CompiledQuery`. When it carries an ``error`` no SQL was produced and the source
        must not be called; the error's ``fix`` says what to change in the spec.

    Raises:
        ValueError: a programming error in the caller — an unsupported dialect, a row cap below 1,
            or a naive ``exec_ts`` (never a spec problem; those come back as ``error``).
    """
    return _Compilation(spec, context).run()


# --- literals ----------------------------------------------------------------------------


def _spec_duration(value: timedelta | None) -> SpecDuration | None:
    """A semantic-layer duration as the spec's own :class:`Duration`.

    The two lanes model a span differently (C04 validates a ``timedelta``, C08 expects the spec's
    month-aware model), so the compiler is where they meet. Sub-second precision is not expressible
    in the spec model and no default window needs it.
    """
    if value is None:
        return None
    total = int(value.total_seconds())
    return SpecDuration(days=total // 86_400, seconds=total % 86_400)


def _value_literal(value: str | int | float | bool) -> exp.Expression:
    """A JSON scalar as a SQL literal. Booleans first: ``bool`` is an ``int`` in Python."""
    if isinstance(value, bool):
        return exp.true() if value else exp.false()
    if isinstance(value, (int, float)):
        return exp.Literal.number(value)
    return exp.Literal.string(value)


def _instant_literal(text: str) -> exp.Expression:
    """An ISO-8601 instant as a timestamp literal, normalised to UTC.

    A cast is explicit because a bare string compares as text on Oracle and SQL Server, where
    ``'2025-11-01' < start_time`` is a type error or a silent string comparison.
    """
    moment = datetime.fromisoformat(text.strip())
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC).replace(tzinfo=None)
    node = exp.cast(exp.Literal.string(moment.isoformat(sep=" ")), exp.DataType.build("TIMESTAMP"))
    return cast(exp.Expression, node)


def _code_literal(code: str) -> exp.Expression:
    """One source code as a literal, typed the way the source stores it (``1`` is a number).

    Mirrors :func:`mdqm.semantic.values._literal`: a code that is canonically an integer is compared
    as an integer, because ``status_code = '1'`` is a type error on a strict engine.
    """
    try:
        return exp.Literal.number(int(code))
    except ValueError:
        return exp.Literal.string(code)


def _aliased(expression: exp.Expression, name: str) -> exp.Expression:
    """Name an output column. A plain column keeps its own name, so the SQL stays readable."""
    if isinstance(expression, exp.Column) and not expression.table and expression.name == name:
        return expression
    return cast(exp.Expression, exp.alias_(expression, exp.to_identifier(name, quoted=True)))


# --- the compilation ---------------------------------------------------------------------


@dataclass
class _Compilation:
    """One run of the compiler over one spec. Mutable, private, and discarded afterwards."""

    spec: QuerySpec
    context: CompileContext
    anchor: TableRef | None = None
    scope: list[TableRef] = field(default_factory=list)
    joins: list[tuple[TableRef, exp.Expression]] = field(default_factory=list)
    outputs: dict[str, exp.Expression] = field(default_factory=dict)
    fixes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    normalizations: list[AppliedNormalization] = field(default_factory=list)
    time_plan: TimePlan | None = None
    partition_plan: PartitionPlan | None = None
    tables: tuple[TableRef, ...] = ()

    # -- entry point ---------------------------------------------------------------------

    def run(self) -> CompiledQuery:
        """Compile, or return the refusal.

        Validates the caller's context first: a bad dialect or row cap is a bug on our side, not a
        spec problem, so it raises instead of becoming an agent-facing error.
        """
        now = self._now()
        self._check_context()
        try:
            self.tables = known_tables(self.context.layer, self.context.columns)
            self._pick_anchor()
            self._pick_joins()
            select_items, group_items = self._project()
            predicates = self._predicates()
            sql = self._render(select_items, group_items, predicates)
            fragment = self._fragment(sql, now)
        except _Stop as stop:
            return CompiledQuery(
                request=self.spec,
                exec_ts=now,
                error=stop.error,
                applied_fixes=tuple(self.fixes),
                notes=tuple(self.notes),
                normalizations=tuple(self.normalizations),
                time_plan=self.time_plan,
                partition_plan=self.partition_plan,
            )
        return CompiledQuery(
            request=self.spec,
            exec_ts=now,
            fragment=fragment,
            applied_fixes=tuple(self.fixes),
            notes=tuple(self.notes),
            normalizations=tuple(self.normalizations),
            time_plan=self.time_plan,
            partition_plan=self.partition_plan,
            output_columns=tuple(self.outputs),
        )

    def _check_context(self) -> None:
        if self.context.dialect not in SUPPORTED_DIALECTS:
            raise ValueError(
                f"unsupported target dialect {self.context.dialect!r}: "
                f"the compiler knows {sorted(SUPPORTED_DIALECTS)}"
            )
        if self.context.max_rows < 1:
            raise ValueError(f"max_rows must be >= 1, got {self.context.max_rows}")
        if not self.context.source:
            raise ValueError("a fragment targets one named source")

    def _now(self) -> datetime:
        moment = self.context.exec_ts
        if moment.tzinfo is None:
            raise ValueError("exec_ts must be timezone-aware UTC; a naive timestamp is a bug")
        return moment.astimezone(UTC)

    def _fail(self, error: ErrorResponse) -> NoReturn:
        raise _Stop(error)

    # -- steps ---------------------------------------------------------------------------

    def _pick_anchor(self) -> None:
        """The ``from`` table: resolved, and required to belong to the fragment's source."""
        resolved = resolve_table(self.spec.from_table, tables=self.tables)
        if isinstance(resolved, ErrorResponse):
            self._fail(resolved)
        if resolved.source and resolved.source != self.context.source:
            self._fail(
                invalid_spec(
                    field="from",
                    reason=(
                        f'the table belongs to source "{resolved.source}", but this fragment '
                        f'targets "{self.context.source}"'
                    ),
                    expected=[
                        table.key
                        for table in self.tables
                        if table.source in ("", self.context.source)
                    ],
                )
            )
        self.anchor = resolved
        self.scope.append(resolved)

    def _pick_joins(self) -> None:
        """Resolve the named joins, refusing anything the semantic layer did not approve."""
        anchor = self._require_anchor()
        for index, name in enumerate(self.spec.joins):
            join = find_join(self.context.layer, name)
            if join is None:
                self._fail(join_not_approved(left=anchor.key, right=name))
            left, right = join_tables(join)
            if anchor.key not in (left, right):
                self._fail(join_not_approved(left=anchor.key, right=left))
            partner_key = right if left == anchor.key else left
            partner = resolve_table(partner_key, tables=self.tables, field=f"joins.{index}")
            if isinstance(partner, ErrorResponse):
                self._fail(partner)
            if any(table.key == partner.key for table in self.scope):
                continue
            on = on_expression(join, anchor=anchor, partner=partner)
            if on is None:
                self._fail(join_not_approved(left=anchor.key, right=partner.key))
            self.scope.append(partner)
            self.joins.append((partner, on))
            self.fixes.append(f'joined "{partner.key}" through the approved join "{name}"')

    def _project(self) -> tuple[tuple[exp.Expression, ...], tuple[exp.Expression, ...]]:
        """The SELECT list and the GROUP BY list, in a deterministic order.

        ``group_by`` keys come first, because a key that is not also selected must still be an
        output column (otherwise ``ORDER BY`` on it and the group itself would have nothing to
        name). A key that is selected too is emitted once.
        """
        groups: list[exp.Expression] = []
        ordered: list[exp.Expression] = []
        for index, key in enumerate(self.spec.group_by):
            name, expression = self._logical(key, f"group_by.{index}")
            if name in self.outputs:
                continue
            self.outputs[name] = expression
            groups.append(expression.copy())
            ordered.append(_aliased(expression.copy(), name))
        for index, item in enumerate(self.spec.select):
            name, expression = self._select_item(item, f"select.{index}")
            if name in self.outputs:
                if isinstance(item, str):
                    continue  # the group_by pass already emitted this column
                self._fail(
                    invalid_spec(
                        field=f"select.{index}",
                        reason=f"the output column '{name}' is produced twice",
                        expected=sorted(self.outputs),
                    )
                )
            self.outputs[name] = expression
            ordered.append(_aliased(expression.copy(), name))
        if not ordered:  # unreachable: the spec requires at least one select item
            self._fail(invalid_spec(field="select", reason="it is empty"))
        return tuple(ordered), tuple(groups)

    def _select_item(self, item: SelectItem, path: str) -> tuple[str, exp.Expression]:
        if isinstance(item, str):
            return self._logical(item, path)
        return self._measure(item, path)

    def _logical(self, name: str, path: str) -> tuple[str, exp.Expression]:
        """A group key or a plain selected column: a column, a canonical value or a time bucket."""
        key = name.strip()
        if key in TIME_BUCKETS and not any(key in table.columns for table in self.scope):
            chosen = event_column(self._require_anchor().semantics)
            if chosen is None:
                self._fail(
                    invalid_spec(
                        field=path,
                        reason=(
                            f"'{key}' is a time bucket, and "
                            f'"{self._require_anchor().key}" has no event-time column to bucket'
                        ),
                    )
                )
            column, _ = chosen
            table = self._require_anchor().physical if self.joins else None
            bucketed = exp.TimestampTrunc(
                unit=exp.var(key.upper()), this=exp.column(column, table=table)
            )
            return key, bucketed
        reference = self._column(key, path)
        return key, self._canonical(reference)

    def _column_expr(self, reference: ColumnRef) -> exp.Expression:
        """A column reference, qualified with its table as soon as the fragment joins another one.

        With one table a bare name is unambiguous and reads better; with a join, qualifying it means
        a column that later appears on the other table cannot turn the statement ambiguous at run
        time.
        """
        table = reference.table.physical if self.joins else None
        return exp.column(reference.column, table=table)

    def _canonical(self, reference: ColumnRef) -> exp.Expression:
        """The column itself, or a CASE that turns each source code into its canonical value.

        A grouped or selected categorical field must answer in the canonical vocabulary, not in
        ``A``/``Active``/``1``: the three storage forms are one business value, so the SQL returns
        that value. An undeclared code falls through to NULL and shows up as a NULL group rather
        than being silently dropped into the wrong bucket.
        """
        if reference.value_map is None:
            return self._column_expr(reference)
        codes = reference.value_map.sources[reference.map_key]
        cases = [
            exp.If(this=_code_literal(code), true=exp.Literal.string(canonical))
            for code, canonical in sorted(codes.items())
        ]
        self.fixes.append(
            f'grouped "{reference.map_key}" as canonical values through its value map'
        )
        return exp.Case(this=self._column_expr(reference), ifs=cases)

    def _measure(self, item: MeasureSelect, path: str) -> tuple[str, exp.Expression]:
        """One aggregate over a normalised measure column."""
        measure = self.context.layer.measures.get(item.measure)
        if measure is None:
            self._fail(
                invalid_spec(
                    field=f"{path}.measure",
                    reason=f'the semantic layer has no measure named "{item.measure}"',
                    expected=sorted(self.context.layer.measures),
                )
            )
        reference = self._measure_column(measure, item, path)
        qualified = f"{reference.table.key}.{reference.column}"
        identifier = (
            f"{reference.table.physical}.{reference.column}"
            if self.joins
            else reference.column
        )
        rendered = render_normalize(
            measure, qualified, identifier, dialect=self.context.dialect
        )
        if rendered is None:  # unreachable: the column came from measure.sources
            self._fail(
                invalid_spec(field=f"{path}.measure", reason="the measure lists no such column")
            )
        self.normalizations.extend(
            normalizations_for(measure, {qualified: identifier}, dialect=self.context.dialect)
        )
        normalised = cast(exp.Expression, sqlglot.parse_one(rendered, read=self.context.dialect))
        if item.agg is None:
            aggregate = Agg.SUM
            self.fixes.append(
                f'"{item.measure}" has no "agg"; used sum (send "agg" to choose another)'
            )
        else:
            aggregate = item.agg
        name = item.alias or (
            item.measure if item.agg is None else f"{item.measure}_{item.agg.value}"
        )
        if aggregate is Agg.COUNT_DISTINCT:
            return name, exp.Count(this=exp.Distinct(expressions=[normalised]))
        return name, _AGGREGATES[aggregate](normalised)

    def _measure_column(self, measure: Measure, item: MeasureSelect, path: str) -> ColumnRef:
        """The scope table that stores this measure, in scope order then measure-key order."""
        for table in self.scope:
            prefix = f"{table.key}."
            for key in sorted(measure.sources):
                if not key.startswith(prefix):
                    continue
                column = key[len(prefix) :]
                if column in table.columns:
                    entry = map_entry(self.context.layer, table.key, column)
                    return ColumnRef(
                        table,
                        column,
                        None if entry is None else entry[0],
                        "" if entry is None else entry[1],
                    )
        self._fail(
            invalid_spec(
                field=f"{path}.measure",
                reason=(
                    f'the measure "{item.measure}" is not stored on any table this query reads '
                    f"({', '.join(table.key for table in self.scope)})"
                ),
                expected=sorted(measure.sources),
            )
        )

    def _column(self, name: str, path: str) -> ColumnRef:
        """Resolve a column, or explain which approved join would make it reachable."""
        resolved = resolve_column(name, scope=self.scope, layer=self.context.layer, field=path)
        if isinstance(resolved, ColumnRef):
            return resolved
        hint = self._join_hint(name)
        self._fail(hint if hint is not None else resolved)

    def _join_hint(self, name: str) -> ErrorResponse | None:
        """The ``JOIN_NOT_APPROVED`` rejection when one approved join would resolve ``name``.

        This is the difference between "that column does not exist" and "you left out the join that
        reaches it", and the ``fix`` carries the approved ON clause so the agent can copy it.
        """
        anchor = self._require_anchor()
        in_scope = {table.key for table in self.scope}
        for candidate in self.tables:
            if candidate.key in in_scope:
                continue
            probe = resolve_column(name, scope=(candidate,), layer=self.context.layer, field="")
            if not isinstance(probe, ColumnRef):
                continue
            join = covering_join(self.context.layer, anchor.key, candidate.key)
            if join is None:
                continue
            return join_not_approved(
                left=anchor.key,
                right=candidate.key,
                approved_on=on_text(join, anchor=anchor, partner=candidate),
            )
        return None

    # -- predicates ----------------------------------------------------------------------

    def _predicates(self) -> tuple[exp.Expression, ...]:
        """Time predicates, then filter predicates, then the partition planner's additions."""
        anchor = self._require_anchor()
        predicates: list[exp.Expression] = []
        plan = plan_time(
            self.spec.time,
            anchor.semantics,
            exec_ts=self._now(),
            table_name=anchor.key,
            lag_unverified=self.context.lag_unverified,
        )
        self.time_plan = plan
        if not plan.ok:
            assert plan.error is not None
            self._fail(plan.error)
        predicates.extend(plan.predicates)
        self.notes.extend(note.value for note in plan.notes)
        for index, condition in enumerate(self.spec.filters):
            predicates.append(self._filter(condition, f"filters.{index}"))
        partition_plan = plan_partitions(
            anchor.semantics.partitioning,
            predicates,
            exec_ts=self._now(),
            table=anchor.key,
            default_window=_spec_duration(anchor.semantics.default_window),
            max_partitions=self.context.max_partitions,
        )
        self.partition_plan = partition_plan
        if not partition_plan.ok:
            assert partition_plan.error is not None
            self._fail(partition_plan.error)
        predicates.extend(partition_plan.predicates)
        self.fixes.extend(partition_plan.applied_fixes)
        self.notes.extend(note.value for note in partition_plan.notes)
        return tuple(predicates)

    def _filter(self, condition: Filter, path: str) -> exp.Expression:
        """One predicate: value-map translated, catalog column, sqlglot node."""
        reference = self._column(condition.column, f"{path}.column")
        column = self._column_expr(reference)
        if reference.value_map is not None and (
            condition.op in _ORDERING or condition.op is FilterOp.CONTAINS
        ):
            self._fail(
                invalid_spec(
                    field=path,
                    reason=(
                        f'"{condition.column}" is categorical (translated by a value map); '
                        f'op "{condition.op.value}" orders or searches its source codes, '
                        "which is not the business value"
                    ),
                    expected=sorted(reference.value_map.canonical),
                )
            )
        if condition.op is FilterOp.CONTAINS:
            value = condition.value
            if not isinstance(value, str):  # unreachable: the spec requires text here
                self._fail(invalid_spec(field=f"{path}.value", reason="'contains' needs text"))
            found = exp.StrPosition(this=column, substr=exp.Literal.string(value))
            return exp.GT(this=found, expression=exp.Literal.number(0))
        if condition.op in (FilterOp.IN, FilterOp.RANGE):
            items = tuple(condition.value)
            literals = tuple(
                self._filter_literal(reference, item, path, ordering=condition.op is FilterOp.RANGE)
                for item in items
            )
            if condition.op is FilterOp.IN:
                return exp.In(this=column, expressions=list(literals))
            return exp.Between(this=column, low=literals[0], high=literals[1])
        literal = self._filter_literal(
            reference, condition.value, path, ordering=condition.op in _ORDERING
        )
        return _COMPARISONS[condition.op](column, literal)

    def _filter_literal(
        self,
        reference: ColumnRef,
        value: object,
        path: str,
        *,
        ordering: bool,
    ) -> exp.Expression:
        """One comparison value: through the value map, as an instant, or as a JSON scalar."""
        if reference.value_map is not None:
            if not isinstance(value, (str, int)) or isinstance(value, bool):
                self._fail(
                    invalid_spec(
                        field=f"{path}.value",
                        reason="a categorical column compares against a label or a source code",
                    )
                )
            result = reverse_value(
                reference.value_map,
                value,
                field=f"{path}.value",
                column=reference.map_key,
            )
            if result.error is not None:
                self._fail(result.error)
            if not result.literals:
                self._fail(
                    unknown_value(
                        field=f"{path}.value",
                        value=str(value),
                        valid=sorted(reference.value_map.sources[reference.map_key].values()),
                    )
                )
            literal = result.literals[0].literal
            self.fixes.append(
                f'resolved "{path}.value" to {reference.map_key} = {literal!r} '
                "through the value map"
            )
            return _value_literal(literal)
        if isinstance(value, str) and (
            ordering or reference.column in reference.table.semantics.time_columns
        ):
            return _instant_literal(value)
        if isinstance(value, (str, int, float, bool)):
            return _value_literal(value)
        self._fail(invalid_spec(field=f"{path}.value", reason="the value is not a JSON scalar"))

    # -- statement -----------------------------------------------------------------------

    def _render(
        self,
        select_items: Sequence[exp.Expression],
        group_items: Sequence[exp.Expression],
        predicates: Sequence[exp.Expression],
    ) -> str:
        """Assemble the AST and render it once in the target dialect."""
        anchor = self._require_anchor()
        query = exp.select(*select_items).from_(exp.to_table(anchor.physical))
        for partner, on in self.joins:
            query = query.join(exp.to_table(partner.physical), on=on, join_type="INNER")
        if predicates:
            query = query.where(*predicates)
        if group_items:
            query = query.group_by(*group_items, copy=True)
        ordered = self._order_by()
        if ordered:
            query = query.order_by(*ordered, copy=False)
        limit = self.spec.limit
        if limit is None:
            limit = self.context.max_rows
            self.fixes.append(
                f"applied the server row cap of {limit} because the spec names no limit"
            )
        query = query.limit(limit)
        return query.sql(dialect=self.context.dialect)

    def _order_by(self) -> tuple[exp.Ordered, ...]:
        """``ORDER BY`` over the output expressions themselves, never over an alias.

        Repeating the expression is what every dialect reads the same way: PostgreSQL matches
        ``GROUP BY`` aliases against input columns first, so an alias there is a correctness hazard.
        """
        ordered: list[exp.Ordered] = []
        for index, item in enumerate(self.spec.order_by):
            expression = self.outputs.get(item.by)
            if expression is None:  # unreachable: the spec validates order_by against the outputs
                self._fail(
                    invalid_spec(
                        field=f"order_by.{index}.by",
                        reason=f"'{item.by}' is not an output column of this query",
                        expected=sorted(self.outputs),
                    )
                )
            ordered.append(exp.Ordered(this=expression.copy(), desc=item.direction.value == "desc"))
        return tuple(ordered)

    def _fragment(self, sql: str, now: datetime) -> Fragment:
        """Freeze the timestamps, guard the SQL, then build the fragment."""
        frozen = rewrite_sql(sql, dialect=self.context.dialect, exec_ts=now)
        checked = guard(frozen.sql, self.context.dialect)
        if not checked.ok:
            assert checked.error is not None
            self._fail(checked.error)
        return Fragment(
            source=self.context.source,
            dialect=self.context.dialect,
            sql=frozen.sql,
            tables=tuple(table.key for table in self.scope),
            exec_ts=now,
        )

    def _require_anchor(self) -> TableRef:
        assert self.anchor is not None
        return self.anchor
