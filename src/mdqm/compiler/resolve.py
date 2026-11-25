"""Name resolution for the compiler: the spec's words -> a table, a column and an approved join.

A spec never carries SQL or a physical identifier: it names a *table* (``accounts``,
``billing.usage_events``), a *column* (``customer_status``), a *measure* and a *join*. This module
turns those names into the three things the compiler may emit — a validated table reference, a
validated column inside it, and an approved join's ON clause — or into the agent-facing error that
says exactly what to write instead.

Two rules decide the shape (``docs/DESIGN.md`` sections 5 and 10):

* **Identifiers come from the catalog and the semantic layer, never from the spec.** A spec name is
  only ever *matched* against one of those; the identifier that reaches the SQL is the one the
  catalog spells. So no agent text can become an identifier, and no string concatenation is needed.
* **A join is a fact a human approved.** The compiler reads joins by name and refuses anything else,
  naming the approved ON clause in the ``fix`` so a weak model can copy it.

Pure: no I/O, no clock. Resolution is deterministic — candidates are scanned in sorted order, so the
same inputs pick the same table on every run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

import sqlglot
from sqlglot import exp

from mdqm.errors import (
    invalid_spec,
    join_not_approved,
    unknown_column,
    unknown_table,
)
from mdqm.models import ErrorResponse
from mdqm.semantic.models import (
    ApprovedJoin,
    SemanticLayer,
    TableSemantics,
    ValueMap,
    render_template,
)

__all__ = [
    "TIME_BUCKETS",
    "ColumnRef",
    "TableRef",
    "covering_join",
    "find_join",
    "join_tables",
    "known_tables",
    "map_entry",
    "on_expression",
    "on_text",
    "resolve_column",
    "resolve_table",
]

#: A ``group_by`` key that buckets the event time instead of naming a column. Checked *after* the
#: catalog, so a real column called ``month`` still wins over the bucket.
TIME_BUCKETS: Final[tuple[str, ...]] = ("day", "week", "month", "quarter", "year")


@dataclass(frozen=True, slots=True)
class TableRef:
    """One table the compiler may read: its semantic key, its semantics and its catalog columns.

    ``key`` is ``source.table`` when the semantic layer qualifies it and a bare name otherwise.
    ``columns`` is what the source's catalog reports, which is the only source of identifiers.
    """

    key: str
    semantics: TableSemantics
    columns: tuple[str, ...]

    @property
    def physical(self) -> str:
        """The name the source's own SQL uses: the last segment of ``source.table``.

        A fragment targets one source, so the database is already chosen by the connection and the
        table is the last segment. ``crm.customers`` is ``customers`` inside the ``crm`` database.
        """
        return self.key.rpartition(".")[2] or self.key

    @property
    def source(self) -> str:
        """The source the key belongs to (``""`` for an unqualified key)."""
        head, _, _ = self.key.rpartition(".")
        return head


@dataclass(frozen=True, slots=True)
class ColumnRef:
    """One resolved column: its table, the catalog's spelling, and the value map that owns it.

    ``value_map`` is set when the column is a categorical field the semantic layer translates, and
    ``map_key`` is the qualified key inside that map (``crm.customers.status``), which is what the
    reverse lookup needs to pick this source's code.
    """

    table: TableRef
    column: str
    value_map: ValueMap | None = None
    map_key: str = ""


def known_tables(
    layer: SemanticLayer, columns: Mapping[str, Sequence[str]]
) -> tuple[TableRef, ...]:
    """Every table the query could touch, ordered by key.

    The union of the semantic layer's tables and the catalog's, so a table that has no semantics yet
    (a fresh database, ``tests/data/semantic``) still resolves — with default semantics, which means
    no partition refusal and no time role.
    """
    keys = sorted(set(layer.tables) | set(columns))
    return tuple(
        TableRef(
            key=key,
            semantics=layer.tables.get(key) or TableSemantics(),
            columns=tuple(columns.get(key, ())),
        )
        for key in keys
    )


def resolve_table(
    name: str, *, tables: Sequence[TableRef], field: str = "from"
) -> TableRef | ErrorResponse:
    """The table a spec name means: an exact semantic key first, then its last segment.

    A bare ``customers`` that two sources both have is ambiguous, not a guess: the two are different
    tables with different storage forms, so the agent must write ``crm.customers``.
    """
    for table in tables:
        if table.key == name:
            return table
    hits = [table for table in tables if table.physical == name]
    if len(hits) == 1:
        return hits[0]
    if hits:
        return invalid_spec(
            field=field,
            reason=(
                f"it matches {len(hits)} tables ({', '.join(table.key for table in hits)}); "
                'write the full "source.table" name'
            ),
            expected=[table.key for table in hits],
        )
    return unknown_table(name=name, known=[table.key for table in tables])


def map_entry(layer: SemanticLayer, table_key: str, column: str) -> tuple[ValueMap, str] | None:
    """The value map that translates ``table_key.column``, and its qualified key, if any."""
    qualified = f"{table_key}.{column}"
    for name in sorted(layer.value_maps):
        value_map = layer.value_maps[name]
        if qualified in value_map.sources:
            return value_map, qualified
    return None


def _map_columns(value_map: ValueMap, table: TableRef) -> tuple[tuple[str, str], ...]:
    """``(column, qualified key)`` pairs this map declares for ``table``, in key order."""
    prefix = f"{table.key}."
    return tuple(
        (key[len(prefix) :], key)
        for key in sorted(value_map.sources)
        if key.startswith(prefix) and key[len(prefix) :] in table.columns
    )


def _column_name(name: str) -> tuple[str, str] | None:
    """Split an already-qualified spec name into ``(table prefix, column)``, or ``None``."""
    prefix, _, column = name.rpartition(".")
    return (prefix, column) if prefix and column else None


def resolve_column(
    name: str,
    *,
    scope: Sequence[TableRef],
    layer: SemanticLayer,
    field: str,
) -> ColumnRef | ErrorResponse:
    """The column a spec name means, inside the tables this query already reads.

    Resolution order (deterministic, documented so a difference is a bug):

    1. a fully qualified name (``crm.customers.status`` or ``customers.status``);
    2. a value-map *field* name (``customer_status``), which only makes sense with its map;
    3. a catalog column name;
    4. anything else is ``UNKNOWN_COLUMN`` listing what this query's tables really have.
    """
    qualified = _column_name(name)
    if qualified is not None:
        prefix, column = qualified
        for table in scope:
            if prefix in (table.key, table.physical) and column in table.columns:
                entry = map_entry(layer, table.key, column)
                return ColumnRef(
                    table,
                    column,
                    None if entry is None else entry[0],
                    "" if entry is None else entry[1],
                )
        return unknown_column(
            table=prefix,
            name=column,
            known=[column for table in scope for column in table.columns],
        )

    value_map = layer.value_maps.get(name)
    if value_map is not None:
        hits = tuple(
            (table, column, key)
            for table in scope
            for column, key in _map_columns(value_map, table)
        )
        if len(hits) == 1:
            table, column, key = hits[0]
            return ColumnRef(table, column, value_map, key)
        if hits:
            return invalid_spec(
                field=field,
                reason=(
                    f"the value map '{name}' is stored on more than one table in this query "
                    f"({', '.join(table.key for table, _, _ in hits)}); filter on one of them "
                    "directly or select the other"
                ),
            )

    found = tuple(table for table in scope if name in table.columns)
    if len(found) == 1:
        table = found[0]
        entry = map_entry(layer, table.key, name)
        return ColumnRef(
            table, name, None if entry is None else entry[0], "" if entry is None else entry[1]
        )
    if found:
        return invalid_spec(
            field=field,
            reason=(
                f"'{name}' exists on more than one table in this query "
                f"({', '.join(table.key for table in found)}); write 'table.column' instead"
            ),
        )
    return unknown_column(
        table=scope[0].key if scope else "",
        name=name,
        known=sorted({column for table in scope for column in table.columns}),
    )


# --- approved joins ----------------------------------------------------------------------


def join_tables(join: ApprovedJoin) -> tuple[str, str]:
    """The two table keys an approved join connects."""
    return join.left.rpartition(".")[0], join.right.rpartition(".")[0]


def find_join(layer: SemanticLayer, name: str) -> ApprovedJoin | None:
    """The approved join called ``name``, or ``None``. Names are unique (the layer checks it)."""
    for join in layer.approved_joins:
        if join.name == name:
            return join
    return None


def covering_join(layer: SemanticLayer, left_key: str, right_key: str) -> ApprovedJoin | None:
    """The approved join between exactly these two tables, in layer order, or ``None``.

    Sorted by name so the answer never depends on the order a YAML file happens to list them.
    """
    for join in sorted(layer.approved_joins, key=lambda item: item.name):
        if set(join_tables(join)) == {left_key, right_key}:
            return join
    return None


def _sides(
    join: ApprovedJoin, anchor: TableRef, partner: TableRef
) -> tuple[exp.Expression, exp.Expression] | None:
    """``(left, right)`` column references with the real tables' names, or ``None`` if unrelated."""
    left_table, left_column = join.left.rpartition(".")[0], join.left.rpartition(".")[2]
    right_table, right_column = join.right.rpartition(".")[0], join.right.rpartition(".")[2]
    if left_table == anchor.key and right_table == partner.key:
        return (
            exp.column(left_column, table=anchor.physical),
            exp.column(right_column, table=partner.physical),
        )
    if left_table == partner.key and right_table == anchor.key:
        return (
            exp.column(left_column, table=partner.physical),
            exp.column(right_column, table=anchor.physical),
        )
    return None


def on_text(join: ApprovedJoin, *, anchor: TableRef, partner: TableRef) -> str | None:
    """The join's full ON clause over the two real tables, or ``None`` when it connects others.

    This is the ``fix`` text an agent copies, so it is the whole condition including the key
    conversion the approved join owns (``LPAD(CAST(customers.cust_id AS VARCHAR), 10, '0') = ...``).
    """
    sides = _sides(join, anchor, partner)
    if sides is None:
        return None
    return render_template(join.on, {"left": sides[0].sql(), "right": sides[1].sql()})


def on_expression(
    join: ApprovedJoin, *, anchor: TableRef, partner: TableRef
) -> exp.Expression | None:
    """The ON clause as an AST node, parsed from the approved template.

    The template was validated and allowlisted at load time (``ApprovedJoin._shape``) and the two
    identifiers come from the catalog and the semantic key, so this parses our own text — it never
    concatenates an agent-supplied value.
    """
    text = on_text(join, anchor=anchor, partner=partner)
    if text is None:
        return None
    return cast(exp.Expression, sqlglot.parse_one(text))


def unapproved(*, left: str, right: str, approved_on: str | None = None) -> ErrorResponse:
    """The ``JOIN_NOT_APPROVED`` rejection, so every caller builds it the same way."""
    return join_not_approved(left=left, right=right, approved_on=approved_on)
