"""Deterministic facts and fingerprints for past SQL (stage D9, no model involved).

Mining past queries is *code*: same input, same output, byte for byte. This module is the pure
part of that: it parses one statement in its own sqlglot dialect and answers, without any I/O,
clock or randomness:

* **What does the statement touch?** base tables, equality join edges (with the conversion
  expression applied to one side), filter columns, the time columns used as a range bound,
  ``GROUP BY`` terms, aggregate functions and rounding/scale expressions.
* **Are two statements the same query?** After *normalisation* -- literals replaced by a marker,
  now-like time functions replaced by a marker, keywords upper-cased by the generator and table
  aliases renamed to positional ``t1``/``t2``/... -- the rendered SQL is hashed into a
  fingerprint. Two statements that differ only in literals, whitespace or alias names therefore
  share a fingerprint; a different join does not.
* **What conversion does the history suggest?** Every conversion-carrying join edge and every
  rounding/scale expression becomes a typed :class:`ConversionHint` with the evidence behind it.
  A hint is a *hypothesis for the validators of stages D7/D8 to check against data*, never a fact:
  "common" is not "correct" (DESIGN.md section 11).

Nothing here decides anything and nothing is stored: the counts, watermarks and ranking live in
:mod:`mdqm.pipeline.stages.mining`. Parse failures are reported, never raised, because a source's
query statistics legitimately hold statements we cannot parse (proprietary hints, vendor syntax).

What is deliberately *not* inferred from the name: a column is treated as time-typed only when its
name looks like a time column **and** it is bounded by a range predicate; mining has no catalog to
ask (D5/D6 have one -- a later stage may refine this).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping, Sequence
from enum import StrEnum
from typing import Final, NamedTuple, cast

import sqlglot
from sqlglot import exp
from sqlglot.dialects import Dialect
from sqlglot.errors import SqlglotError

from mdqm.models.base import MDQMModel

__all__ = [
    "AGGREGATE_FUNCTIONS",
    "CONVERSION_FUNCTIONS",
    "INTERVAL_MARKER",
    "LITERAL_MARKER",
    "NOW_FUNCTION_NAMES",
    "NOW_MARKER",
    "TIME_COLUMN_SUFFIXES",
    "ConversionHint",
    "HintEvidence",
    "HintKind",
    "JoinEdge",
    "QueryFacts",
    "analyze",
    "fingerprint",
    "normalize",
    "now_functions",
]

#: What a stripped literal becomes in normalised SQL. Braces keep it distinguishable from a
#: user column, and it renders unquoted in every dialect.
LITERAL_MARKER: Final[str] = "{lit}"

#: What a now-like time function becomes: every one of them is "the moment the query ran", which is
#: exactly what must not affect the fingerprint.
NOW_MARKER: Final[str] = "{now}"

#: What a literal duration becomes: ``INTERVAL '30 days'`` and ``INTERVAL '6' HOUR`` are the same
#: shape ("some interval"), and the marker keeps the normalised SQL readable.
INTERVAL_MARKER: Final[str] = "{interval}"

#: Now-like function names per dialect, lowercase. Kept in step with :mod:`mdqm.sql.exec_ts`
#: (which *rewrites* these); mining must *recognise* them on the statement text, because sqlglot
#: normalises ``now()``, ``GETDATE()`` and ``SYSDATE`` into the same AST node.
NOW_FUNCTION_NAMES: Final[Mapping[str, frozenset[str]]] = {
    "duckdb": frozenset(
        {"now", "current_date", "current_timestamp", "localtime", "localtimestamp"}
    ),
    "mysql": frozenset(
        {
            "curdate",
            "current_date",
            "current_timestamp",
            "localtime",
            "localtimestamp",
            "now",
            "utc_date",
            "utc_timestamp",
        }
    ),
    "oracle": frozenset(
        {
            "current_date",
            "current_timestamp",
            "localtime",
            "localtimestamp",
            "sysdate",
            "systimestamp",
        }
    ),
    "postgres": frozenset(
        {"current_date", "current_timestamp", "localtime", "localtimestamp", "now"}
    ),
    "tsql": frozenset(
        {
            "current_timestamp",
            "getdate",
            "getutcdate",
            "sysdatetime",
            "sysdatetimeoffset",
            "sysutcdatetime",
        }
    ),
}

#: Functions that convert a value's *format or scale*: an equality that needs one of them on one
#: side is the join the analyst could not write without help (DESIGN.md section 11).
CONVERSION_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {"cast", "convert", "lpad", "to_char", "to_number", "try_cast"}
)

#: Aggregate functions reported. A closed set, because "any function in the select list" would
#: report window functions and arithmetic as aggregates.
AGGREGATE_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {"avg", "count", "max", "min", "stddev", "sum", "variance"}
)

#: Token kinds that can never *name* a function: a string literal (``'now'``), a number, a comment.
#: Compared by name rather than by the enum member, which sqlglot does not export for typing.
_NOT_A_NAME: Final[frozenset[str]] = frozenset({"COMMENT", "NATIONAL_STRING", "NUMBER", "STRING"})

#: Pseudo-columns a dialect exposes as if they were columns. They describe the row's position, not
#: the data, so they are not filter columns and never appear in a join.
PSEUDO_COLUMNS: Final[frozenset[str]] = frozenset({"level", "rowid", "rownum"})

#: A filter column whose name ends like this is *possibly* a time column; it only becomes one when a
#: range predicate bounds it (``deleted_at IS NULL`` is a flag test, not a time window).
TIME_COLUMN_SUFFIXES: Final[tuple[str, ...]] = ("_at", "_date", "_time", "_ts")

_MAX_ERROR_CHARS: Final[int] = 200
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class JoinEdge(MDQMModel):
    """One equality edge: the two table labels and the two columns, in table-appearance order.

    ``conversion``/``conversion_expr`` are set when one side of the equality is wrapped in a
    format/scale conversion (``LPAD(cust_id, 10, '0') = client_no``); the pair goes together, and
    the expression is rendered in the statement's dialect.
    """

    tables: tuple[str, str]
    columns: tuple[str, str]
    conversion: str | None = None
    conversion_expr: str | None = None


class HintKind(StrEnum):
    """What kind of hypothesis a conversion hint is."""

    JOIN_KEY = "join_key"
    """A join equality needs a conversion on one side (stages D7/D8 validate it against data)."""
    ROUNDING = "rounding"
    """A ``ROUND``/``FLOOR``/``CEIL`` expression, optionally to a granularity."""
    SCALE = "scale"
    """A pure multiply/divide by a constant (``limit_k * 10``, ``x / 1000``)."""


class HintEvidence(MDQMModel):
    """Where a hint came from: the source, the statement fingerprint, and how often it ran."""

    source: str
    fingerprint: str
    calls: int = 0


class ConversionHint(MDQMModel):
    """A typed conversion hypothesis mined from history, with the evidence that produced it.

    Stages D7 (join candidates) and D8 (conversion detection) consume these as *extra hypotheses*:
    a hint raises the prior of a formula, and the validators still have to confirm it against real
    values. ``scale`` is the multiplier that takes the source form to the canonical one; a divide by
    1000 therefore reports ``0.001``.
    """

    kind: HintKind
    function: str | None = None
    expression: str
    dialect: str
    tables: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    scale: float | None = None
    granularity: float | None = None
    width: int | None = None
    fill: str | None = None
    pattern: str | None = None
    target_type: str | None = None
    evidence: HintEvidence | None = None


class QueryFacts(MDQMModel):
    """Everything mining extracts from one statement; ``error`` set means it did not parse."""

    dialect: str
    fingerprint: str | None = None
    normalized: str | None = None
    tables: tuple[str, ...] = ()
    joins: tuple[JoinEdge, ...] = ()
    filters: tuple[str, ...] = ()
    time_columns: tuple[str, ...] = ()
    group_by: tuple[str, ...] = ()
    aggregations: tuple[str, ...] = ()
    rounding: tuple[str, ...] = ()
    now_functions: tuple[str, ...] = ()
    hints: tuple[ConversionHint, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the statement parsed and the facts above mean something."""
        return self.error is None


class _Slot(NamedTuple):
    """One table source in the statement: what joins call it, and how columns reach it."""

    label: str
    qualifiers: tuple[str, ...]
    is_base: bool = True


class _FoundEdge(NamedTuple):
    """A join edge plus the conversion node it came from, when it had one."""

    edge: JoinEdge
    node: exp.Expression | None


def _collapse(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _error_text(exc: BaseException) -> str:
    """One bounded line: a parse failure is reported, and never becomes a wall of SQL."""
    first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return _collapse(f"{type(exc).__name__}: {first}")[:_MAX_ERROR_CHARS]


def _parse(sql: str, dialect: str) -> tuple[exp.Expression | None, str | None]:
    """Parse exactly one statement, returning ``(tree, error)`` instead of raising."""
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except (SqlglotError, ValueError) as exc:  # unknown dialect is a ValueError
        return None, _error_text(exc)
    parsed = [statement for statement in statements if statement is not None]
    if not parsed:
        return None, "empty statement"
    if len(parsed) > 1:
        return None, f"{len(parsed)} statements in one query"
    return cast(exp.Expression, parsed[0]), None


def _now_nodes() -> tuple[type[exp.Expression], ...]:
    """The AST node types sqlglot folds the now-like functions into."""
    node_types: list[type[exp.Expression]] = [
        exp.CurrentTimestamp,
        exp.CurrentDate,
        exp.CurrentTime,
    ]
    current_datetime = getattr(exp, "CurrentDatetime", None)  # not on every sqlglot version
    if isinstance(current_datetime, type) and issubclass(current_datetime, exp.Expression):
        node_types.append(current_datetime)
    return tuple(node_types)


def now_functions(sql: str, dialect: str) -> tuple[str, ...]:
    """Names of the non-deterministic time functions in ``sql``, lowercase and sorted.

    Read from the statement's own tokens, because the AST cannot tell ``now()`` from ``GETDATE()``
    once sqlglot has folded both into one node. Function arguments are text: the marker is "the
    clock was read", which is what the fingerprint has to ignore. An unknown dialect reports none.
    """
    names = NOW_FUNCTION_NAMES.get(dialect)
    if names is None:
        return ()
    try:
        tokens = Dialect.get_or_raise(dialect).tokenizer_class().tokenize(sql)
    except (SqlglotError, ValueError):
        return ()
    found: set[str] = set()
    for token in tokens:
        if token.token_type.name in _NOT_A_NAME:
            continue
        text = token.text.lower()
        if text in names:
            found.add(text)
    return tuple(sorted(found))


def _slots(tree: exp.Expression) -> tuple[list[_Slot], dict[str, _Slot]]:
    """Table sources in appearance order, plus the qualifier -> slot map used to resolve columns.

    A base table is labelled by its bare name (schema and alias stripped, per the corpus contract);
    a CTE reference keeps the CTE's name (the analyst reads ``recent``, not the alias ``r``); any
    other derived source is labelled by its alias, which is the only name it has.
    """
    cte_names = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    slots: list[_Slot] = []
    by_qualifier: dict[str, _Slot] = {}
    for node in tree.find_all(exp.Table, exp.Subquery):
        label: str
        qualifiers: tuple[str, ...]
        is_base = True
        if isinstance(node, exp.Subquery):
            if not node.alias:
                continue  # a scalar subquery in a select list is not a table source
            label = node.alias
            qualifiers = (node.alias,)
            is_base = False
        elif not isinstance(node, exp.Table):
            continue
        elif node.name in cte_names and not node.db:
            label = node.name
            qualifiers = tuple(name for name in (node.alias, node.name) if name)
            is_base = False
        else:
            label = node.name
            qualifiers = tuple(name for name in (node.alias, node.name) if name)
        if not label:
            continue
        slot = _Slot(label=label.lower(), qualifiers=qualifiers, is_base=is_base)
        slots.append(slot)
        for qualifier in qualifiers:
            by_qualifier.setdefault(qualifier, slot)
    return slots, by_qualifier


def _normalize_aliases(tree: exp.Expression, slots: Sequence[_Slot]) -> exp.Expression:
    """Rename every table alias to positional ``t1``/``t2``/... and rewrite column qualifiers."""
    canonical: dict[str, str] = {}
    for index, slot in enumerate(slots, start=1):
        for qualifier in slot.qualifiers:
            canonical.setdefault(qualifier, f"t{index}")

    def transform(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Table):
            qualifiers = tuple(name for name in (node.alias, node.name) if name)
            target = next((canonical[name] for name in qualifiers if name in canonical), None)
            if target is not None and node.name not in {target}:
                node.set("alias", exp.TableAlias(this=exp.to_identifier(target)))
            return node
        if isinstance(node, exp.Subquery) and node.alias:
            node.set(
                "alias",
                exp.TableAlias(this=exp.to_identifier(canonical.get(node.alias, node.alias))),
            )
            return node
        if isinstance(node, exp.Column) and node.table:
            if node.table in canonical:
                node.set("table", exp.to_identifier(canonical[node.table]))
            return node
        return node

    return tree.transform(transform)


def _strip_intervals(tree: exp.Expression) -> exp.Expression:
    """Replace a literal duration with one marker before its literal is stripped.

    Two passes, because the interval *is* the duration: ``INTERVAL '30 days'`` and
    ``INTERVAL '6' HOUR`` both mean "some interval", and leaving the bare marker inside an
    ``INTERVAL`` also makes the SQL generator complain about an unexpected unit.
    """

    def transform(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Interval):
            return exp.Var(this=INTERVAL_MARKER)
        return node

    stripped: exp.Expression = tree.transform(transform)
    return stripped


def _strip_literals_and_now(tree: exp.Expression) -> exp.Expression:
    """Replace literals and now-like reads with their markers, so only the *shape* is hashed."""
    now_types = _now_nodes()

    def transform(node: exp.Expression) -> exp.Expression:
        if isinstance(node, now_types):
            return exp.Var(this=NOW_MARKER)
        if isinstance(node, (exp.Literal, exp.Boolean, exp.HexString, exp.National, exp.RawString)):
            parent: exp.Expression | None = cast("exp.Expression | None", node.parent)
            if parent is not None and parent.args.get("unit") is node:
                return cast(exp.Expression, node)  # a unit names a granularity, not a value
            return exp.Var(this=LITERAL_MARKER)
        return node

    marked: exp.Expression = _strip_intervals(tree).transform(transform)
    return marked


def _normalized_tree(sql: str, dialect: str) -> exp.Expression:
    """The canonical tree for ``sql``: aliases renamed, literals and clock reads marked."""
    tree = _parse_or_raise(sql, dialect)
    slots, _ = _slots(tree)
    tree = _normalize_aliases(tree, slots)
    return _strip_literals_and_now(tree)


def _parse_or_raise(sql: str, dialect: str) -> exp.Expression:
    tree, error = _parse(sql, dialect)
    if tree is None:
        raise ValueError(error or "unparseable statement")
    return tree


def normalize(sql: str, dialect: str) -> str:
    """Render ``sql`` as its literal-free, alias-canonical form: the fingerprint's input.

    Two statements that differ only in literals, whitespace or alias names normalise to the same
    text. The dialect is part of the fingerprint (not of this string), because the same words mean
    different things in two engines.
    """
    return _collapse(_normalized_tree(sql, dialect).sql(dialect=dialect))


def fingerprint(sql: str, dialect: str) -> str:
    """The sha256 hex fingerprint of one statement: ``sha256(dialect + normalized SQL)``.

    Deterministic across processes, platforms and Python versions: it hashes text, never a
    ``hash()`` value.
    """
    material = f"{dialect}\n{normalize(sql, dialect)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _sole_column(node: exp.Expression) -> exp.Column | None:
    """The one column ``node`` refers to, or ``None`` when it refers to none or several."""
    columns = list(node.find_all(exp.Column))
    return columns[0] if len(columns) == 1 else None


def _column_name(node: exp.Expression) -> str:
    """A column's bare name, lowercased and unquoted (identifiers are case-insensitive in SQL)."""
    return node.name.lower() if isinstance(node, exp.Column) else _collapse(node.sql())


def _slot_label(node: exp.Expression, by_qualifier: Mapping[str, _Slot]) -> _Slot | None:
    if not isinstance(node, exp.Column) or not node.table:
        return None
    return by_qualifier.get(node.table)


def _equalities(tree: exp.Expression) -> Iterator[exp.EQ]:
    """Every ``=`` predicate the statement compares with, in a stable order.

    Per scope: the join ``ON`` clauses first (that is the order the corpus contract uses), then
    ``WHERE``, then ``HAVING``. An equality inside a nested subquery shows up again when that
    subquery's own scope is visited; callers deduplicate, which keeps the first position.
    """
    for node in tree.find_all(exp.Select):
        for join in node.args.get("joins") or ():
            on = join.args.get("on")
            if on is not None:
                yield from on.find_all(exp.EQ)
        for key in ("where", "having"):
            clause = node.args.get(key)
            if clause is not None:
                yield from clause.find_all(exp.EQ)


def _conversion(node: exp.Expression) -> str | None:
    """The lowercase name of the format/scale conversion at the top of ``node``, if any."""
    if isinstance(node, exp.Pad):
        return "lpad"
    if isinstance(node, exp.TryCast):
        return "try_cast"
    if isinstance(node, exp.Cast):
        return "cast"
    if isinstance(node, (exp.ToChar, exp.TimeToStr)):
        return "to_char"
    if isinstance(node, exp.ToNumber):
        return "to_number"
    if isinstance(node, exp.Convert):
        return "convert"
    if isinstance(node, exp.Func):
        name = node.key.lower()
        if name in CONVERSION_FUNCTIONS:
            return name
        sql_name = type(node).sql_name().lower()
        return sql_name if sql_name in CONVERSION_FUNCTIONS else None
    if isinstance(node, exp.Anonymous):
        name = node.name.lower()
        return name if name in CONVERSION_FUNCTIONS else None
    return None


def _join_edges(
    tree: exp.Expression, slots: Sequence[_Slot], by_qualifier: Mapping[str, _Slot], dialect: str
) -> list[_FoundEdge]:
    """Equality predicates that join two different table sources, in appearance order."""
    order = {id(slot): index for index, slot in enumerate(slots)}
    by_label: dict[str, int] = {}
    for index, slot in enumerate(slots):
        by_label.setdefault(slot.label, index)
    edges: list[_FoundEdge] = []
    for equality in _equalities(tree):
        left = _sole_column(equality.this)
        right = _sole_column(equality.expression)
        if left is None or right is None:
            continue
        left_slot = _slot_label(left, by_qualifier)
        right_slot = _slot_label(right, by_qualifier)
        if left_slot is None or right_slot is None or left_slot.label == right_slot.label:
            continue
        sides = [
            (
                by_label.get(left_slot.label, order[id(left_slot)]),
                left_slot.label,
                left,
                equality.this,
            ),
            (
                by_label.get(right_slot.label, order[id(right_slot)]),
                right_slot.label,
                right,
                equality.expression,
            ),
        ]
        sides.sort(key=lambda side: side[0])
        (
            (_, first_label, first_column, first_expr),
            (
                _,
                second_label,
                second_column,
                second_expr,
            ),
        ) = sides
        converted = first_expr if _conversion(first_expr) else second_expr
        conversion = _conversion(converted)
        edges.append(
            _FoundEdge(
                edge=JoinEdge(
                    tables=(first_label, second_label),
                    columns=(_column_name(first_column), _column_name(second_column)),
                    conversion=conversion,
                    conversion_expr=(
                        None if conversion is None else _collapse(converted.sql(dialect=dialect))
                    ),
                ),
                node=converted if conversion is not None else None,
            )
        )
    unique: list[_FoundEdge] = []
    for found in edges:
        if all(found.edge != seen.edge for seen in unique):
            unique.append(found)
    return unique


def _predicate_columns(clause: exp.Expression) -> set[str]:
    """Column names a predicate compares, ignoring columns that sit inside an aggregate.

    ``HAVING SUM(credit_limit) > 100000`` filters on the *sum*, not on the column, so
    ``credit_limit`` is not a filter here; a plain ``credit_limit > 0`` is.
    """
    found: set[str] = set()
    for column in clause.find_all(exp.Column):
        current = cast(exp.Expression | None, column.parent)
        while current is not None and current is not clause:
            if isinstance(current, exp.AggFunc):
                break
            current = cast(exp.Expression | None, current.parent)
        else:
            found.add(_column_name(column))
    return found


def _all_predicate_columns(tree: exp.Expression) -> set[str]:
    """Columns named by WHERE/HAVING/ON predicates, aggregate FILTERs and CASE conditions."""
    found: set[str] = set()
    for select in tree.find_all(exp.Select):
        for key in ("where", "having"):
            clause = select.args.get(key)
            if clause is not None:
                found.update(_predicate_columns(clause))
        for join in select.args.get("joins") or ():
            on = join.args.get("on")
            if on is not None:
                found.update(_predicate_columns(on))
    for holder in tree.find_all(exp.Filter, exp.If):
        condition = holder.args.get("expression") if isinstance(holder, exp.Filter) else holder.this
        if condition is not None:
            found.update(_predicate_columns(condition))
    return found - PSEUDO_COLUMNS


_RANGE_NODES: Final[tuple[type[exp.Expression], ...]] = (
    exp.GT,
    exp.GTE,
    exp.LT,
    exp.LTE,
    exp.Between,
)


def _range_columns(tree: exp.Expression) -> set[str]:
    """Columns bounded by a range predicate: the only ones that can be a time *window*."""
    found: set[str] = set()
    for node in tree.find_all(*_RANGE_NODES):
        found.update(_column_name(column) for column in node.find_all(exp.Column))
    return found


def _is_time_column(name: str) -> bool:
    return name.endswith(TIME_COLUMN_SUFFIXES)


def _aggregations(tree: exp.Expression) -> tuple[str, ...]:
    names: set[str] = set()
    for node in tree.walk():
        if isinstance(node, exp.AggFunc):
            names.add(node.key.lower())
        elif isinstance(node, exp.Anonymous):
            names.add(node.name.lower())
    return tuple(sorted(names & AGGREGATE_FUNCTIONS))


def _group_by(tree: exp.Expression, dialect: str) -> tuple[str, ...]:
    terms: list[str] = []
    for node in tree.find_all(exp.Select):
        group = node.args.get("group")
        if group is None:
            continue
        for term in group.expressions:
            text = _collapse(term.sql(dialect=dialect))
            if text and text not in terms:
                terms.append(text)
    return tuple(terms)


def _contains_column(node: exp.Expression) -> bool:
    return any(isinstance(child, exp.Column) for child in node.walk())


def _literal_number(node: exp.Expression | None) -> float | None:
    """The value of a numeric literal (optionally negated or parenthesised), else ``None``."""
    if node is None:
        return None
    if isinstance(node, exp.Paren):
        return _literal_number(node.this)
    if isinstance(node, exp.Neg):
        inner = _literal_number(node.this)
        return None if inner is None else -inner
    if isinstance(node, exp.Literal) and not node.is_string:
        try:
            return float(node.name)
        except ValueError:
            return None
    return None


_ROUNDING_TYPES: Final[tuple[type[exp.Expression], ...]] = (exp.Round, exp.Floor, exp.Ceil)


def _outermost_scale(node: exp.Expression) -> exp.Expression:
    """Climb to the arithmetic that makes a rounding expression a *scale* expression.

    ``ROUND(x / 5000) * 5000`` is one conversion, not a rounding and a multiplication: the climb
    only passes through parentheses and through ``*``/``/``/``+``/``-`` whose other side is a
    numeric literal.
    """
    current = node
    while True:
        parent = current.parent
        if isinstance(parent, exp.Paren):
            current = parent
            continue
        if isinstance(parent, (exp.Mul, exp.Div, exp.Add, exp.Sub)) and current in (
            parent.this,
            parent.expression,
        ):
            other = parent.expression if current is parent.this else parent.this
            if _literal_number(other) is not None:
                current = parent
                continue
        return current


def _inside_rounding(node: exp.Expression) -> bool:
    """True when ``node`` sits inside a ``ROUND``/``FLOOR``/``CEIL`` call, i.e. part of it."""
    current = node.parent
    while current is not None:
        if isinstance(current, _ROUNDING_TYPES):
            return True
        current = current.parent
    return False


def _rounding_expressions(tree: exp.Expression, dialect: str) -> list[tuple[str, exp.Expression]]:
    """Rounding and scale expressions, outermost first, deduplicated by rendered text.

    A rounding call takes its whole arithmetic chain with it (``ROUND(x / 5000) * 5000``), and a
    multiply or divide by a constant is a scale expression on its own (``limit_k * 10``,
    ``x / 1000``) unless it is already part of a rounding expression. Dividing by something that is
    not a constant (``SUM(x) / COUNT(*)``) is arithmetic, not a conversion, and is not reported.
    """
    chosen: list[exp.Expression] = []
    for node in tree.walk():
        if not isinstance(node, _ROUNDING_TYPES):
            continue
        candidate = _outermost_scale(node)
        if any(_is_inside(candidate, picked) for picked in chosen):
            continue
        chosen = [picked for picked in chosen if not _is_inside(picked, candidate)]
        chosen.append(candidate)
    for node in tree.walk():
        if not isinstance(node, (exp.Mul, exp.Div)):
            continue
        if any(_is_inside(node, picked) for picked in chosen) or _inside_rounding(node):
            continue
        sides = (node.this, node.expression)
        constants = sum(1 for side in sides if _literal_number(side) is not None)
        if constants != 1 or not any(_contains_column(side) for side in sides):
            continue
        chosen.append(node)
    chosen.sort(key=lambda node: (node.depth, _collapse(node.sql(dialect=dialect))))
    rendered: list[tuple[str, exp.Expression]] = []
    for node in chosen:
        text = _collapse(node.sql(dialect=dialect))
        if all(text != existing for existing, _ in rendered):
            rendered.append((text, node))
    return rendered


def _is_inside(node: exp.Expression, container: exp.Expression) -> bool:
    """True when ``node`` is ``container`` or sits under it."""
    current: exp.Expression | None = node
    while current is not None:
        if current is container:
            return True
        current = cast(exp.Expression | None, current.parent)
    return False


def _hint_params(node: exp.Expression, dialect: str) -> dict[str, object]:
    """The structured parameters of a conversion node, for a consumer that must re-render it."""
    params: dict[str, object] = {}
    if isinstance(node, exp.Pad):
        width = _literal_number(node.expression)
        if width is not None:
            params["width"] = int(width)
        fill_node = node.args.get("fill_pattern")
        if isinstance(fill_node, exp.Literal):
            params["fill"] = fill_node.name
    elif isinstance(node, (exp.ToChar, exp.TimeToStr)):
        literals = [child for child in node.find_all(exp.Literal) if child.is_string]
        if literals:
            params["pattern"] = literals[-1].name
    elif isinstance(node, (exp.Cast, exp.TryCast)):
        target = node.args.get("to")
        if isinstance(target, exp.DataType):
            params["target_type"] = _collapse(target.sql(dialect=dialect)).upper()
    elif isinstance(node, exp.Convert):
        target = next((child for child in node.find_all(exp.DataType)), None)
        if target is not None:
            params["target_type"] = _collapse(target.sql(dialect=dialect)).upper()
    return params


def _rounding_shape(node: exp.Expression) -> tuple[float | None, float | None]:
    """``(granularity, scale)`` for a rounding/scale expression, when the shape makes them certain.

    ``ROUND(x / 5000) * 5000`` is "round to a multiple of 5000", so it reports a granularity and no
    scale: the division and the multiplication cancel out. ``ROUND(x, -3)`` is the same idea written
    as a negative digit count. A bare ``x / 1000`` reports the multiplier ``0.001``.
    """
    if isinstance(node, _ROUNDING_TYPES):
        inner = node.this
        if isinstance(inner, exp.Mul):
            for side, other in ((inner.this, inner.expression), (inner.expression, inner.this)):
                granularity = _literal_number(other)
                if granularity is not None and _contains_column(side):
                    return granularity, None
        if isinstance(inner, exp.Div):
            divisor = _literal_number(inner.expression)
            if divisor:
                return divisor, None
        digits = _literal_number(node.args.get("decimals"))
        if digits is not None and digits < 0:
            return 10.0 ** (-digits), None
        return None, None
    if isinstance(node, (exp.Mul, exp.Div)):
        sides = (node.this, node.expression)
        for side in sides:
            inner_rounding = side.this if isinstance(side, exp.Paren) else side
            if isinstance(inner_rounding, _ROUNDING_TYPES):
                return _rounding_shape(inner_rounding)
        if not isinstance(node, exp.Mul):
            divisor = _literal_number(node.expression)
            if divisor:
                return None, 1.0 / divisor
            return None, None
        for side, other in ((node.this, node.expression), (node.expression, node.this)):
            factor = _literal_number(other)
            if factor is not None and _contains_column(side):
                return None, factor
        return None, None
    return None, None


def _rounding_function(node: exp.Expression) -> str | None:
    """The conversion name a rounding/scale node represents (``round``, ``floor``, ``multiply``...).

    A multiply or divide only counts as a *scale* expression when there is no rounding call inside
    it: ``ROUND(x / 5000) * 5000`` is a rounding to a granularity, not a scale by 0.002.
    """
    if isinstance(node, exp.Round):
        return "round"
    if isinstance(node, exp.Floor):
        return "floor"
    if isinstance(node, exp.Ceil):
        return "ceil"
    if isinstance(node, (exp.Mul, exp.Div)):
        for side in (node.this, node.expression):
            inner = side.this if isinstance(side, exp.Paren) else side
            rounded = _rounding_function(inner)
            if rounded in {"round", "floor", "ceil"}:
                return rounded
        return "multiply" if isinstance(node, exp.Mul) else "divide"
    return None


def _conversion_hints(
    joins: Sequence[_FoundEdge],
    rounding: Sequence[tuple[str, exp.Expression]],
    dialect: str,
) -> list[ConversionHint]:
    """Typed hypotheses from conversion joins and rounding/scale expressions, deduplicated."""
    hints: list[ConversionHint] = []
    for found in joins:
        edge = found.edge
        if edge.conversion is None or edge.conversion_expr is None or found.node is None:
            continue
        params = _hint_params(found.node, dialect)
        hints.append(
            ConversionHint(
                kind=HintKind.JOIN_KEY,
                function=edge.conversion,
                expression=edge.conversion_expr,
                dialect=dialect,
                tables=edge.tables,
                columns=edge.columns,
                width=cast(int | None, params.get("width")),
                fill=cast(str | None, params.get("fill")),
                pattern=cast(str | None, params.get("pattern")),
                target_type=cast(str | None, params.get("target_type")),
            )
        )
    for text, node in rounding:
        function = _rounding_function(node)
        granularity, scale = _rounding_shape(node)
        params = _hint_params(node, dialect)
        hints.append(
            ConversionHint(
                kind=HintKind.ROUNDING
                if function in {"round", "floor", "ceil"}
                else HintKind.SCALE,
                function=function,
                expression=text,
                dialect=dialect,
                scale=scale,
                granularity=granularity,
                width=cast(int | None, params.get("width")),
                fill=cast(str | None, params.get("fill")),
                pattern=cast(str | None, params.get("pattern")),
                target_type=cast(str | None, params.get("target_type")),
            )
        )
    unique: list[ConversionHint] = []
    for hint in hints:
        if hint not in unique:
            unique.append(hint)
    return unique


def analyze(sql: str, dialect: str) -> QueryFacts:
    """Extract every fact mining needs from one statement, or report why it did not parse.

    Never raises for a statement that does not parse or for an unknown dialect: the caller counts
    failures (DESIGN.md section 11, "failures counted"), and a single unparseable statement must not
    stop a source's whole history from being mined.
    """
    names = now_functions(sql, dialect)
    tree, error = _parse(sql, dialect)
    if tree is None:
        return QueryFacts(dialect=dialect, now_functions=names, error=error)
    try:
        slots, by_qualifier = _slots(tree)
        joins = _join_edges(tree, slots, by_qualifier, dialect)
        join_columns = {column for found in joins for column in found.edge.columns}
        predicates = _all_predicate_columns(tree) - join_columns
        ranges = _range_columns(tree)
        rounding = _rounding_expressions(tree, dialect)
        rounding_texts = tuple(text for text, _ in rounding)
        return QueryFacts(
            dialect=dialect,
            fingerprint=fingerprint(sql, dialect),
            normalized=normalize(sql, dialect),
            tables=tuple(dict.fromkeys(slot.label for slot in slots if slot.is_base)),
            joins=tuple(found.edge for found in joins),
            filters=tuple(sorted(predicates)),
            time_columns=tuple(
                sorted(name for name in predicates if _is_time_column(name) and name in ranges)
            ),
            group_by=_group_by(tree, dialect),
            aggregations=_aggregations(tree),
            rounding=rounding_texts,
            now_functions=names,
            hints=tuple(_conversion_hints(joins, rounding, dialect)),
        )
    except (SqlglotError, ValueError, RecursionError) as exc:  # a shape we cannot read is a failure
        return QueryFacts(dialect=dialect, now_functions=names, error=_error_text(exc))
