"""Reject a categorical filter whose value never occurs, unless the stats might be stale.

The decision is pure. A lookup runs only when the value is absent AND the stats are not a
complete reading of every partition (a sample, a tail, or a partition collected later than the
stats). The lookup is a caller's :class:`ValueProbe`: this module does not open a connection.

A fresh, complete column rejects immediately, with no probe call. That is the hard error. A
stale column is not rejected until the probe says the value is absent. A probe that cannot run
does not reject: a false rejection from old stats is worse than letting the query through.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Protocol

from mdqm.compiler.resolve import (
    TableRef,
    find_join,
    join_tables,
    known_tables,
    resolve_column,
    resolve_table,
)
from mdqm.errors import value_not_in_data
from mdqm.models import ErrorResponse
from mdqm.semantic.models import SemanticLayer, ValueMap
from mdqm.semantic.text import normalize_label
from mdqm.spec.models import FilterOp, QuerySpec
from mdqm.stats.models import CategoryColumnStats, ObservedValue
from mdqm.stats.similarity import Suggestion, equivalent, suggest

__all__ = [
    "Decision",
    "GateResult",
    "ValueProbe",
    "decide",
    "gate_spec",
    "match_value",
    "must_verify",
]

_CHECKED_OPS = frozenset({FilterOp.EQ, FilterOp.NE, FilterOp.IN})
_LOOKUP_CAP = 8


class Decision(StrEnum):
    """What to do with one filter value that was compared to stored statistics."""

    ALLOW = "allow"
    REJECT = "reject"
    VERIFY = "verify"


class ValueProbe(Protocol):
    """A cheap existence check. ``None`` means the probe could not run, not that the value is absent."""

    def exists(self, *, source: str, table: str, column: str, value: str) -> bool | None:
        """True when a bounded read found ``value`` in ``column``."""
        ...


class GateResult:
    """The gate's answer: an error that stops the query, notes that must be reported, or neither."""

    __slots__ = ("error", "notes")

    def __init__(self, error: ErrorResponse | None = None, notes: tuple[str, ...] = ()) -> None:
        self.error = error
        self.notes = notes

    @property
    def ok(self) -> bool:
        """True when the query may run."""
        return self.error is None


class _Allow(GateResult):
    """The shared 'nothing to say' result. Notes are empty, so it is safe to reuse."""


_OK = _Allow()


def must_verify(stats: CategoryColumnStats) -> bool:
    """True when a missing value might still exist outside what was collected.

    A complete distribution over every partition is the only reading that can reject on its own.
    """
    return bool(stats.unprofiled_partitions) or not stats.distribution_complete


def _labels(value: str, value_map: ValueMap | None, column: str | None) -> tuple[str, ...]:
    """The query plus the canonical value, synonyms and source codes that mean the same thing."""
    found: list[str] = [value]
    if value_map is None:
        return (value,)
    key = normalize_label(value)
    canonicals: list[str] = []
    resolved = value_map.resolve(value)
    if resolved is not None:
        canonicals.append(resolved)
    for name, codes in sorted(value_map.sources.items()):
        if column is not None and name != column:
            continue
        canonicals.extend(
            canonical for code, canonical in codes.items() if normalize_label(code) == key
        )
    for canonical in dict.fromkeys(canonicals):
        found.append(canonical)
        found.extend(value_map.synonyms.get(canonical, ()))
        for name, codes in sorted(value_map.sources.items()):
            if column is not None and name != column:
                continue
            found.extend(code for code, target in codes.items() if target == canonical)
    return tuple(dict.fromkeys(found))


def match_value(
    value: str,
    stats: CategoryColumnStats,
    value_map: ValueMap | None = None,
    column: str | None = None,
) -> ObservedValue | None:
    """The stored value ``value`` refers to, or ``None`` when none of its forms were observed.

    Matching folds case and whitespace, strips Hebrew points, accepts a consonantal
    transliteration across Hebrew and Latin, and follows the value map's synonyms and source
    codes. Two Latin typos are not folded into each other.
    """
    labels = _labels(value, value_map, column)
    found: ObservedValue | None = None
    for item in stats.values:
        if not any(equivalent(label, item.value) for label in labels):
            continue
        if (
            found is None
            or item.count > found.count
            or (item.count == found.count and item.value < found.value)
        ):
            found = item
    return found


def decide(
    value: str,
    stats: CategoryColumnStats,
    value_map: ValueMap | None = None,
    column: str | None = None,
) -> Decision:
    """ALLOW when the value occurs, REJECT when fresh stats say it does not, else VERIFY."""
    if match_value(value, stats, value_map, column) is not None:
        return Decision.ALLOW
    if must_verify(stats):
        return Decision.VERIFY
    return Decision.REJECT


def _agent_values(
    stats: CategoryColumnStats, value_map: ValueMap | None, column: str | None
) -> tuple[tuple[str, int], ...]:
    """Observed values in the spelling an agent should write back, with counts summed."""
    totals: dict[str, int] = {}
    for item in stats.values:
        label = item.value
        if value_map is not None and column is not None:
            canonical = value_map.canonical_for(column, item.value)
            if canonical is None:
                canonical = value_map.resolve(item.value)
            if canonical is not None:
                label = canonical
        totals[label] = totals.get(label, 0) + item.count
    return tuple(sorted(totals.items(), key=lambda item: (-item[1], item[0])))


def _checked(stats: CategoryColumnStats, *, verified_absent: bool) -> str:
    when = stats.collected_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    if not verified_absent:
        return f"Stats collected {when} by {stats.method.value} list every value of this column."
    if stats.unprofiled_partitions:
        missed = ", ".join(stats.unprofiled_partitions)
        return (
            f"A bounded lookup found no row. Stats collected {when} by {stats.method.value} "
            f"did not cover partition {missed}."
        )
    return (
        f"A bounded lookup found no row. Stats collected {when} by {stats.method.value} "
        "are a sample, not the whole column."
    )


def _rejection(
    value: str,
    stats: CategoryColumnStats,
    value_map: ValueMap | None,
    column: str | None,
    *,
    verified_absent: bool,
) -> ErrorResponse:
    labels = _labels(value, value_map, column)
    observed = _agent_values(stats, value_map, column)
    suggestions: tuple[Suggestion, ...] = suggest(
        value, observed, rows=stats.rows_seen, also=labels
    )
    # An exact folded match would have been ALLOW. Drop distance 0 so the list is alternatives.
    alternatives = tuple(item for item in suggestions if item.distance > 0) or suggestions
    return value_not_in_data(
        column=stats.qualified,
        value=value,
        suggestions=tuple(item.value for item in alternatives),
        frequencies=tuple(item.frequency() for item in alternatives),
        checked=_checked(stats, verified_absent=verified_absent),
    )


def gate_value(
    value: str,
    stats: CategoryColumnStats,
    *,
    value_map: ValueMap | None = None,
    column: str | None = None,
    probe: ValueProbe | None = None,
    source: str = "",
    table: str = "",
    physical_column: str = "",
) -> GateResult:
    """One filter value. See the module docstring for when a probe runs."""
    decision = decide(value, stats, value_map, column)
    if decision is Decision.ALLOW:
        return _OK
    verified_absent = False
    if decision is Decision.VERIFY:
        if probe is None or not source or not table or not physical_column:
            return GateResult(
                notes=(
                    f"{stats.qualified}: {value!r} is absent from stats collected "
                    f"{stats.collected_at.strftime('%Y-%m-%dT%H:%M:%SZ')} "
                    f"({stats.method.value}), which do not cover the whole column, and no "
                    "bounded lookup was available, so the filter was not rejected",
                )
            )
        literals = _labels(value, value_map, column)[:_LOOKUP_CAP]
        saw_unknown = False
        for literal in literals:
            found = probe.exists(source=source, table=table, column=physical_column, value=literal)
            if found is True:
                return GateResult(
                    notes=(
                        f"{stats.qualified}: {value!r} is absent from stats collected "
                        f"{stats.collected_at.strftime('%Y-%m-%dT%H:%M:%SZ')} but a bounded "
                        f"lookup found {literal!r}",
                    )
                )
            if found is None:
                saw_unknown = True
        if saw_unknown:
            return GateResult(
                notes=(
                    f"{stats.qualified}: stats collected "
                    f"{stats.collected_at.strftime('%Y-%m-%dT%H:%M:%SZ')} may be stale and "
                    "the bounded lookup could not run, so the filter was not rejected",
                )
            )
        verified_absent = True
    return GateResult(
        error=_rejection(value, stats, value_map, column, verified_absent=verified_absent)
    )


def _scope(
    spec: QuerySpec, layer: SemanticLayer, columns: Mapping[str, Sequence[str]]
) -> tuple[TableRef, ...] | None:
    """The tables the spec reads, or ``None`` when the spec is not resolvable yet.

    An unresolvable spec is someone else's error (unknown table, unapproved join). This gate
    does not invent a second refusal for it.
    """
    tables = known_tables(layer, columns)
    anchor = resolve_table(spec.from_table, tables=tables)
    if isinstance(anchor, ErrorResponse):
        return None
    scope: list[TableRef] = [anchor]
    for name in spec.joins:
        join = find_join(layer, name)
        if join is None:
            return None
        left, right = join_tables(join)
        if anchor.key not in (left, right):
            return None
        partner_key = right if left == anchor.key else left
        partner = resolve_table(partner_key, tables=tables, field="joins")
        if isinstance(partner, ErrorResponse):
            return None
        if not any(table.key == partner.key for table in scope):
            scope.append(partner)
    return tuple(scope)


def _items(op: FilterOp, value: object) -> tuple[object, ...] | None:
    if op is FilterOp.IN:
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return None
    return (value,)


def gate_spec(
    spec: QuerySpec,
    *,
    layer: SemanticLayer,
    columns: Mapping[str, Sequence[str]],
    stats: Mapping[str, CategoryColumnStats],
    probe: ValueProbe | None = None,
) -> GateResult:
    """Every categorical equality in ``spec``. The first real refusal wins; notes accumulate.

    Columns with no stored distribution are skipped. No distribution means "we have not
    measured", which is not evidence that a value is absent.
    """
    if not stats:
        return _OK
    scope = _scope(spec, layer, columns)
    if scope is None:
        return _OK
    notes: list[str] = []
    for index, condition in enumerate(spec.filters):
        if condition.op not in _CHECKED_OPS:
            continue
        resolved = resolve_column(
            condition.column, scope=scope, layer=layer, field=f"filters.{index}.column"
        )
        if isinstance(resolved, ErrorResponse):
            continue
        key = f"{resolved.table.key}.{resolved.column}"
        column_stats = stats.get(key)
        if column_stats is None:
            continue
        items = _items(condition.op, condition.value)
        if items is None:
            continue
        for item in items:
            if isinstance(item, bool) or not isinstance(item, (str, int)):
                continue
            result = gate_value(
                str(item),
                column_stats,
                value_map=resolved.value_map,
                column=resolved.map_key or key,
                probe=probe,
                source=resolved.table.source,
                table=resolved.table.physical,
                physical_column=resolved.column,
            )
            if result.error is not None:
                return GateResult(error=result.error, notes=tuple(notes))
            notes.extend(result.notes)
    return GateResult(notes=tuple(notes))
