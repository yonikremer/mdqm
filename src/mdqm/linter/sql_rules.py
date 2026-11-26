"""The raw-SQL rules: the emergency exit, held to the same standard as the spec path.

Free SQL exists only as an emergency exit (DESIGN section 4), and it goes through the same linter
and the same guard as SQL the compiler wrote. The rules here work on the parsed statement, never on
the text, so a comment, a string literal or a quoted identifier cannot fool them:

* the read-only guard refuses a write, a DDL, a command or a second statement (the taxonomy's own
  codes and texts, via :func:`mdqm.errors.from_guard`);
* an unknown relation is refused with "did you mean";
* ``now()``/``CURRENT_TIMESTAMP``/``SYSDATE``/``GETDATE()`` become one absolute ``exec_ts`` literal,
  so a re-run sees exactly the same window (C02);
* a statement without ``LIMIT`` gets the server row cap;
* a partition key that nothing filters, or that a function hides, gets the predicate that lets the
  source prune (the partition planner builds it, so the rewrite is the one C08 proved);
* a categorical label compared against a stored code is replaced with the code that column holds.

Two deliberate omissions, both because a wrong refusal blocks a valid query:

* **unknown columns** are left to the source on this path. ``ORDER BY`` on an output alias, a
  dialect-specific expression and a correlated subquery all look like an unknown column to a static
  check, and the primary path (the spec) refuses an unknown column with "did you mean" anyway.
* **an unapproved join** cannot be read from free SQL: the spec's ``joins`` is where approval lives,
  so free SQL that joins two tables is allowed by design and the join limitation is documented.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from mdqm.compiler.resolve import TableRef, map_entry
from mdqm.errors import ErrorCode, from_guard, unknown_table, unknown_value
from mdqm.linter.rules import (
    LintContext,
    LintState,
    Rule,
    RuleCase,
    RuleOutcome,
    Severity,
    Target,
    spec_duration,
)
from mdqm.models import ErrorResponse
from mdqm.models.partitioning import PartitionStrategy
from mdqm.planner.partitions import PartitionNote, plan_partitions
from mdqm.semantic.models import ValueMap
from mdqm.semantic.values import resolve_value
from mdqm.sql.exec_ts import rewrite_sql
from mdqm.sql.guard import guard


def _parse(sql: str, dialect: str) -> exp.Expression | None:
    """The statement's AST, or ``None`` when it does not parse (the guard rule reports that)."""
    try:
        parsed = sqlglot.parse(sql, read=dialect, error_level=sqlglot.ErrorLevel.RAISE)
    except SqlglotError:
        return None
    statements = [statement for statement in parsed if statement is not None]
    return cast(exp.Expression, statements[0]) if len(statements) == 1 else None


def _conjuncts(node: exp.Expression) -> Iterator[exp.Expression]:
    """The top-level AND parts of a predicate."""
    if isinstance(node, exp.And):
        yield from _conjuncts(node.this)
        yield from _conjuncts(node.expression)
        return
    if isinstance(node, exp.Paren):
        yield from _conjuncts(node.this)
        return
    yield node


def _where(statement: exp.Expression) -> tuple[exp.Expression, ...]:
    """Every top-level conjunct of the statement's ``WHERE``, in written order."""
    where = statement.args.get("where")
    if not isinstance(where, exp.Where):
        return ()
    return tuple(_conjuncts(where.this))


def _matches(name: str, candidate: TableRef) -> bool:
    """True when ``name`` is this table, ignoring case.

    Case matters in PostgreSQL and does not in Oracle, which folds an unquoted identifier to upper
    case, so ``"CRM"."CUSTOMERS"`` and ``crm.customers`` are one table there. Refusing a valid
    statement is worse than accepting a case variant, so the comparison folds case.
    """
    folded = name.casefold()
    return folded in (candidate.key.casefold(), candidate.physical.casefold())


def _base_table(statement: exp.Expression, ctx: LintContext) -> TableRef | None:
    """The first table the statement reads, matched against the catalog and the semantic layer."""
    for table in statement.find_all(exp.Table):
        name = table.name
        if not name:
            continue
        for candidate in ctx.tables():
            if _matches(name, candidate):
                return candidate
    return None


def _known_relations(ctx: LintContext) -> tuple[str, ...]:
    """Every name a statement may say out loud: a key, a physical name, a declared extra one."""
    names: set[str] = set(ctx.extra_relations)
    for table in ctx.tables():
        names.update((table.key, table.physical))
    return tuple(sorted(names))


class GuardRule(Rule):
    """One read-only statement, or the guard's refusal.

    The guard is the same one the compiler runs on its own SQL: a write, a DDL, a lock, a server
    command or a second statement is refused there, and this rule only relays that refusal through
    the taxonomy (so the code is validated and the text re-sanitised).
    """

    id = "sql.guard"
    severity = Severity.ERROR
    autofix = False
    target = Target.SQL
    title = "read-only guard"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        sql = state.sql
        if sql is None:
            return None
        result = guard(sql, ctx.dialect)
        error = from_guard(result)
        return None if error is None else RuleOutcome(error=error)

    cases = (
        RuleCase(
            "delete is refused",
            fires=True,
            sql="DELETE FROM customers WHERE cust_id = 1",
            expect_code=ErrorCode.WRITE_BLOCKED.value,
            expect_fix="SELECT",
        ),
        RuleCase(
            "two statements are refused",
            fires=True,
            sql="SELECT 1; SELECT 2",
            expect_code=ErrorCode.MULTI_STATEMENT.value,
            expect_fix="one statement per call",
        ),
        RuleCase("one read-only select", sql="SELECT cust_id FROM customers"),
    )


class RelationRule(Rule):
    """Every relation the statement reads must exist in the catalog or the semantic layer.

    A common table expression is not a relation of its own, so it is skipped; a table function has
    no name and is skipped too; a name the deployment declared in ``extra_relations`` (``dual``) is
    known even though no catalog lists it; and a name that differs only in case is the same table.
    What is left is a name the source will really look up, and an unknown one is answered with the
    names that do exist.
    """

    id = "sql.table"
    severity = Severity.ERROR
    autofix = False
    target = Target.SQL
    title = "unknown table"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        sql = state.sql
        if sql is None:
            return None
        statement = _parse(sql, ctx.dialect)
        if statement is None:
            return None
        known = _known_relations(ctx)
        folded = {name.casefold() for name in known}
        cte_names = {cte.alias_or_name.casefold() for cte in statement.find_all(exp.CTE)}
        for table in statement.find_all(exp.Table):
            name = table.name
            if not name or name.casefold() in cte_names or name.casefold() in folded:
                continue
            return RuleOutcome(error=unknown_table(name=name, known=known))
        return None

    cases = (
        RuleCase(
            "misspelled relation",
            fires=True,
            sql="SELECT cust_id FROM customerz",
            expect_code=ErrorCode.UNKNOWN_TABLE.value,
            expect_fix="known tables",
            expect_suggestions=("customers",),
        ),
        RuleCase(
            "a CTE is not an unknown relation",
            sql=("WITH recent AS (SELECT cust_id FROM customers) SELECT cust_id FROM recent"),
        ),
    )


class NowRule(Rule):
    """Every now-like value becomes one absolute ``exec_ts`` literal.

    A relative clock in a query means two runs of the same statement answer differently, the cache
    key moves on every call and an audit trail cannot be replayed. The rewrite is C02's, so the
    literal is rendered in the source's dialect exactly as the source would have read its clock.
    """

    id = "sql.now"
    severity = Severity.NOTICE
    autofix = True
    target = Target.SQL
    title = "now() -> exec_ts"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        sql = state.sql
        if sql is None:
            return None
        try:
            rewritten = rewrite_sql(sql, dialect=ctx.dialect, exec_ts=ctx.now)
        except (SqlglotError, ValueError):
            return None
        if not rewritten.replacements:
            return None
        fixes = tuple(
            f"replaced {replacement.original} with {replacement.replaced_with}: one exec_ts per "
            "execution, so a re-run sees the same window"
            for replacement in rewritten.replacements
        )
        return RuleOutcome(sql=rewritten.sql, fixes=fixes)

    cases = (
        RuleCase(
            "now() becomes one exec_ts",
            fires=True,
            sql="SELECT event_id FROM events WHERE start_time > now()",
            expect_sql=("2026-11-25",),
            expect_absent=("now()", "CURRENT_TIMESTAMP"),
            expect_fix="one exec_ts per execution",
        ),
        RuleCase("no relative clock", sql="SELECT event_id FROM events WHERE units > 0"),
    )


class SqlLimitRule(Rule):
    """A statement with no ``LIMIT`` gets the server row cap.

    The cap is what the server applies to the fetched result anyway; writing it into the statement
    means the source stops early instead of streaming a whole table into memory first.
    """

    id = "sql.limit"
    severity = Severity.NOTICE
    autofix = True
    target = Target.SQL
    title = "missing LIMIT"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        sql = state.sql
        if sql is None:
            return None
        statement = _parse(sql, ctx.dialect)
        if statement is None or not isinstance(statement, (exp.Select, exp.SetOperation)):
            return None
        if statement.args.get("limit") is not None:
            return None
        capped = statement.limit(ctx.max_rows)
        return RuleOutcome(
            sql=capped.sql(dialect=ctx.dialect),
            fixes=(
                f"applied the server row cap of {ctx.max_rows} because the statement has no LIMIT",
            ),
        )

    cases = (
        RuleCase(
            "no limit",
            fires=True,
            sql="SELECT cust_id FROM customers",
            expect_sql=("LIMIT 1000",),
            expect_fix="row cap of 1000",
        ),
        RuleCase("limit given", sql="SELECT cust_id FROM customers LIMIT 5"),
    )


class SqlPartitionRule(Rule):
    """The statement must let the source prune its partitions.

    Two shapes are repaired with the predicates the partition planner (C08) builds, so the fix is
    the one already proved: no key filter at all (the table's default window is injected) and a
    function or cast wrapped around the key (a range on the raw column is added next to it). What
    the planner refuses, this rule refuses with the same code and the same estimate.

    A set operation (``UNION``) is left alone: each side needs its own check, and rewriting half of
    one would change which rows the statement returns.
    """

    id = "sql.partition"
    severity = Severity.ERROR
    autofix = True
    target = Target.SQL
    title = "partition key access"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        sql = state.sql
        if sql is None:
            return None
        statement = _parse(sql, ctx.dialect)
        if not isinstance(statement, exp.Select):
            return None  # a set operation is checked per side by the source, not rewritten here
        table = _base_table(statement, ctx)
        if table is None:
            return None
        partitioning = table.semantics.partitioning
        if partitioning is None or partitioning.strategy is PartitionStrategy.NONE:
            return None
        conjuncts = _where(statement)
        plan = plan_partitions(
            partitioning,
            conjuncts,
            exec_ts=ctx.now,
            table=table.key,
            default_window=spec_duration(table.semantics.default_window),
            max_partitions=ctx.max_partitions,
        )
        if plan.error is not None:
            return RuleOutcome(error=plan.error)
        repair = {PartitionNote.DEFAULT_WINDOW_APPLIED, PartitionNote.KEY_FUNCTION_REWRITTEN}
        if not repair & set(plan.notes):
            return None
        added = tuple(
            predicate
            for predicate in plan.predicates
            if not any(predicate == conjunct for conjunct in conjuncts)
        )
        if not added:
            return None
        fixed = statement.where(*added)
        fixes = plan.applied_fixes or (
            f'added a predicate on "{partitioning.key[0]}" so the source can prune partitions',
        )
        return RuleOutcome(sql=fixed.sql(dialect=ctx.dialect), fixes=fixes)

    cases = (
        RuleCase(
            "the table allows a default window",
            fires=True,
            sql="SELECT event_id FROM usage_events WHERE units > 0",
            layer={
                "version": 1,
                "tables": {
                    "billing.usage_events": {
                        "key": ["event_id"],
                        "default_window": "7d",
                        "partitioning": {
                            "key": "start_time",
                            "granularity": "month",
                            "require_key_filter": False,
                        },
                    }
                },
            },
            columns={
                "billing.usage_events": (
                    "event_id",
                    "client_no",
                    "units",
                    "start_time",
                    "insertion_time",
                )
            },
            expect_sql=("start_time >=", "start_time <"),
            expect_fix="default window",
        ),
        RuleCase(
            "a function on the key becomes a range",
            fires=True,
            sql="SELECT event_id FROM usage_events WHERE CAST(start_time AS DATE) = '2026-11-05'",
            expect_sql=("start_time <", "CAST(start_time AS DATE)"),
            expect_fix="range on the raw column",
        ),
        RuleCase(
            "nothing filters a required key",
            fires=True,
            sql="SELECT event_id FROM usage_events",
            expect_code=ErrorCode.PARTITION_SCAN_ALL.value,
            expect_fix="events_in_range",
        ),
        RuleCase(
            "the key is already ranged",
            sql="SELECT event_id FROM usage_events WHERE start_time >= '2026-11-01'",
        ),
    )


class SqlValueRule(Rule):
    """A categorical label compared against a column becomes the code that column stores.

    The where clause of free SQL compares against the stored value, so a canonical label (or a
    Hebrew synonym) matches nothing and the query silently answers zero rows. The value map knows
    the code this column holds for that value: ``= 'label'`` becomes ``= 'code'``, an ``IN`` list
    has each label replaced, and a value the column stores under no code is refused with the
    canonical values that exist. A canonical value the column spells several ways becomes an ``IN``
    so no row is lost.
    """

    id = "sql.value"
    severity = Severity.ERROR
    autofix = True
    target = Target.SQL
    title = "categorical label -> source code"

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        sql = state.sql
        if sql is None:
            return None
        statement = _parse(sql, ctx.dialect)
        if statement is None:
            return None
        table = _base_table(statement, ctx)
        if table is None:
            return None
        fixes: list[str] = []
        error = self._repair(statement, table, ctx, fixes)
        if error is not None:
            return RuleOutcome(error=error)
        if not fixes:
            return None
        return RuleOutcome(sql=statement.sql(dialect=ctx.dialect), fixes=tuple(fixes))

    def _repair(
        self,
        statement: exp.Expression,
        table: TableRef,
        ctx: LintContext,
        fixes: list[str],
    ) -> ErrorResponse | None:
        """Rewrite every label comparison in place; returns the refusal, or ``None``."""
        for comparison in list(statement.find_all(exp.EQ)):
            pair = _label_comparison(comparison)
            if pair is None:
                continue
            column, literal = pair
            entry = map_entry(ctx.layer, table.key, column)
            if entry is None:
                continue
            value_map, map_key = entry
            resolved = _codes(value_map, map_key, literal)
            if isinstance(resolved, ErrorResponse):
                return resolved
            canonical, codes = resolved
            if literal in codes:
                continue
            column_side = _column_side(comparison)
            assert column_side is not None
            comparison.replace(_code_comparison(column_side, codes))
            fixes.append(
                f"replaced the label {literal!r} with {', '.join(repr(code) for code in codes)}, "
                f'the value "{map_key}" stores for the canonical value {canonical!r}'
            )
        for condition in list(statement.find_all(exp.In)):
            column = condition.this
            if not isinstance(column, exp.Column):
                continue
            literals = tuple(
                str(item.this)
                for item in condition.expressions
                if isinstance(item, exp.Literal) and item.is_string
            )
            if len(literals) != len(condition.expressions):
                continue  # a parameter, a function or a subquery: not a label list
            entry = map_entry(ctx.layer, table.key, column.name)
            if entry is None:
                continue
            value_map, map_key = entry
            wanted: list[str] = []
            changed = False
            for literal in literals:
                resolved = _codes(value_map, map_key, literal)
                if isinstance(resolved, ErrorResponse):
                    return resolved
                canonical, codes = resolved
                changed = changed or literal not in codes
                wanted.extend(code for code in codes if code not in wanted)
            if not changed:
                continue
            condition.set("expressions", [exp.Literal.string(code) for code in wanted])
            fixes.append(
                f"replaced the labels ({', '.join(repr(item) for item in literals)}) in the "
                f'filter on "{map_key}" with the codes that column stores: '
                f"{', '.join(repr(code) for code in wanted)}"
            )
        return None

    cases = (
        RuleCase(
            "Hebrew label",
            fires=True,
            sql="SELECT client_no FROM accounts WHERE state = 'פעיל'",
            expect_sql=("state = 'Active'",),
            expect_absent=("פעיל",),
            expect_fix="canonical value 'active'",
        ),
        RuleCase(
            "a label inside an IN list",
            fires=True,
            sql="SELECT client_no FROM accounts WHERE state IN ('פעיל', 'Hold')",
            expect_sql=("state IN ('Active', 'Hold')",),
            expect_absent=("פעיל",),
            expect_fix="replaced the labels",
        ),
        RuleCase(
            "unknown label is refused",
            fires=True,
            sql="SELECT client_no FROM accounts WHERE state = 'activ'",
            expect_code=ErrorCode.UNKNOWN_VALUE.value,
            expect_fix="active",
        ),
        RuleCase(
            "the stored code already", sql="SELECT client_no FROM accounts WHERE state = 'Active'"
        ),
    )


#: The order the rules run in: the guard first (a statement that must not run is not repaired),
#: then names, then the fixes, then the cost gates.
SQL_RULES: tuple[Rule, ...] = (
    GuardRule(),
    RelationRule(),
    NowRule(),
    SqlLimitRule(),
    SqlValueRule(),
    SqlPartitionRule(),
)


def _codes(
    value_map: ValueMap, map_key: str, literal: str
) -> tuple[str, tuple[str, ...]] | ErrorResponse:
    """The canonical value a label means, and the codes this column stores it under."""
    resolution = resolve_value(value_map, literal, field="sql", column=map_key)
    canonical = resolution.canonical
    if resolution.error is not None or canonical is None:
        return resolution.error or unknown_value(
            field="sql", value=literal, valid=sorted(value_map.canonical)
        )
    codes = tuple(
        code for code, target in value_map.sources[map_key].items() if target == canonical
    )
    if not codes:
        return unknown_value(
            field="sql", value=literal, valid=sorted(value_map.sources[map_key].values())
        )
    return canonical, codes


def _column_side(comparison: exp.EQ) -> exp.Column | None:
    """The column of a ``column = literal`` comparison, whichever side it is written on."""
    for side in (comparison.this, comparison.expression):
        if isinstance(side, exp.Column):
            return side
    return None


def _label_comparison(comparison: exp.EQ) -> tuple[str, str] | None:
    """``(column, label)`` when one side is a column and the other a text literal."""
    column = _column_side(comparison)
    if column is None:
        return None
    other = comparison.expression if comparison.this is column else comparison.this
    if not isinstance(other, exp.Literal) or not other.is_string:
        return None
    return column.name, str(other.this)


def _code_comparison(column: exp.Column, codes: tuple[str, ...]) -> exp.Expression:
    """``col = 'A'`` for one code, ``col IN ('A', 'Active')`` for several.

    Both forms prune and filter identically; ``IN`` is what keeps a canonical value that the source
    spells two ways reachable.
    """
    if len(codes) == 1:
        return exp.EQ(this=column.copy(), expression=exp.Literal.string(codes[0]))
    return exp.In(this=column.copy(), expressions=[exp.Literal.string(code) for code in codes])
