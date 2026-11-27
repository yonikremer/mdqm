"""PostgreSQL partition metadata and the EXPLAIN pruning proof (task A03).

Two jobs, one question: *does this query read only the partitions it means to?* A primary database
with no replica answers that with its own planner, and this module is where that answer is read.

**Partition metadata** (:func:`parse_partitions`) is the definitive reading of PostgreSQL's
declarative partitioning catalogs (``pg_partitioned_table``, ``pg_get_partkeydef``,
``pg_inherits``): strategy, key, granularity and one :class:`~mdqm.models.PartitionBound` per
partition with its range bounds as UTC datetimes, never as the text form, because the planner
aligns windows and cuts cache segments on those bounds. The row estimate per partition is
``pg_class.reltuples``, which the server keeps; it is an estimate, not a count, and stays ``None``
when the server has never looked.

**The plan** (:func:`parse_plan`) is a pure parse of ``EXPLAIN (FORMAT JSON)`` *without* ANALYZE:
no statement is executed to obtain it, so asking about the cost of a scan never causes the scan.
The parser reports the relations and indexes the plan named, how many partition subplans the
planner removed before execution, and any parameter markers (``$0``) that pushed pruning to run
time.

**The verdict** (:func:`pruning_verdict`) is the only place a ``PlanSummary`` becomes a promise,
and it is falsifiable on purpose: it carries the partitions the declared window *expects* and the
partitions the plan *named*, and it says ``PROVEN`` only when those two sets are equal. A plan that
scanned something else is a visible disagreement — two named lists — rather than a count nobody can
check. ``PROVEN`` is never produced without a window, without declared bounds, or for a table that
is not partitioned: a table with no partitions has nothing to prove, and claiming otherwise is the
one answer this module must never give.

Pure module: no connection, no clock. The adapter supplies catalog rows and a plan document; the
planner supplies the window it intends to read.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from mdqm.adapters.base import OPTIONAL_METHODS, NotSupported
from mdqm.models import (
    PartitionBound,
    PartitionGranularity,
    Partitioning,
    PartitionStrategy,
    PlanSummary,
    PruningProof,
)
from mdqm.planner.partitions import Range

__all__ = [
    "PARTITIONS_SQL",
    "PgPlan",
    "PruningVerdict",
    "parse_partitions",
    "parse_plan",
    "partitions_in_window",
    "pruning_verdict",
    "row_estimate",
    "split_names",
]

#: One row of :data:`PARTITIONS_SQL`: parent schema/table, strategy, partkey text, child
#: schema/table, ``pg_get_expr(relpartbound)`` and the child's ``reltuples``.
CatalogRow = Sequence[Any]

#: Every declarative partition and its bound, one row per partition.
#:
#: Index partitions inherit the parent's index definition (``accounts_hash_p0_pkey``), so the child
#: relation is restricted to a real table: otherwise every partition of every index shows up as a
#: table with no columns.
PARTITIONS_SQL: Final = """
SELECT pn.nspname AS parent_schema,
       pc.relname AS parent_table,
       pt.partstrat AS strategy,
       pg_get_partkeydef(pc.oid) AS partition_key,
       cn.nspname AS child_schema,
       cc.relname AS child_table,
       pg_get_expr(cc.relpartbound, cc.oid) AS bound,
       cc.reltuples::bigint AS row_estimate
  FROM pg_inherits i
  JOIN pg_class pc ON pc.oid = i.inhparent
  JOIN pg_namespace pn ON pn.oid = pc.relnamespace
  JOIN pg_class cc ON cc.oid = i.inhrelid
  JOIN pg_namespace cn ON cn.oid = cc.relnamespace
  LEFT JOIN pg_partitioned_table pt ON pt.partrelid = pc.oid
 WHERE pn.nspname NOT IN ('pg_catalog', 'information_schema')
   AND pc.relkind = 'p'
   AND cc.relkind = 'r'
 ORDER BY pn.nspname, pc.relname, cc.relname
"""

#: ``pg_partitioned_table.partstrat`` -> the strategy the planner reasons about.
PARTITION_STRATEGIES: Final[dict[str, PartitionStrategy]] = {
    "r": PartitionStrategy.RANGE,
    "l": PartitionStrategy.LIST,
    "h": PartitionStrategy.HASH,
}

_MONTH_STEPS: Final[dict[timedelta, PartitionGranularity]] = {
    timedelta(hours=1): PartitionGranularity.HOUR,
    timedelta(days=1): PartitionGranularity.DAY,
    timedelta(days=7): PartitionGranularity.WEEK,
    timedelta(days=90): PartitionGranularity.QUARTER,
    timedelta(days=365): PartitionGranularity.YEAR,
}

_RANGE_BOUND_RE: Final = re.compile(
    r"\AFOR VALUES FROM \((?P<lower>.*)\) TO \((?P<upper>.*)\)\Z", re.DOTALL
)
_LIST_BOUND_RE: Final = re.compile(r"\AFOR VALUES IN \((?P<values>.*)\)\Z", re.DOTALL)
_HASH_BOUND_RE: Final = re.compile(
    r"\AFOR VALUES WITH \(MODULUS (?P<modulus>\d+), REMAINDER (?P<remainder>\d+)\)\Z",
    re.IGNORECASE,
)
_QUOTED_RE: Final = re.compile(r"\A'(?P<text>.*)'\Z", re.DOTALL)
_PARTKEY_RE: Final = re.compile(r"\A[A-Z]+ \((?P<key>.*)\)\Z")

#: A parameter marker in a filter or index condition (``((start_time >= $0) AND ...)``).
_PARAM_RE: Final = re.compile(r"\$(?P<number>\d+)")

#: Plan-node keys whose text can carry a parameter marker. Reading only these keeps a parameter
#: that happens to appear in a comment or an alias from being mistaken for pruning pressure.
_CONDITION_KEYS: Final = (
    "Filter",
    "Hash Cond",
    "Index Cond",
    "Join Filter",
    "One-Time Filter",
    "Recheck Cond",
    "TID Cond",
)


# --- catalog text -----------------------------------------------------------------------


def split_names(value: object) -> tuple[str, ...]:
    """Split a comma-joined column list (``'a, b'``) into a tuple, dropping blanks."""
    if value is None:
        return ()
    return tuple(part for part in (chunk.strip() for chunk in str(value).split(",")) if part)


def row_estimate(value: object) -> int | None:
    """``pg_class.reltuples`` as a non-negative estimate, or ``None`` when it is unknown (-1)."""
    try:
        estimate = int(str(value))
    except (TypeError, ValueError):
        return None
    return estimate if estimate >= 0 else None


def _bound_datetime(token: str) -> datetime | None:
    """Parse a range bound literal (``'2024-11-01 00:00:00+00'``), or ``None`` for MINVALUE."""
    text = token.strip()
    if text.upper() in {"MINVALUE", "MAXVALUE"}:
        return None
    quoted = _QUOTED_RE.match(text)
    if quoted is not None:
        text = quoted.group("text")
    candidate = text.replace(" ", "T", 1)
    try:
        moment = datetime.fromisoformat(candidate)
    except ValueError:
        # PostgreSQL prints a zero offset as "+00"; fromisoformat before 3.11 wants "+00:00".
        try:
            moment = datetime.fromisoformat(f"{candidate}:00")
        except ValueError:
            return None
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _bound(name: str, definition: str, estimate: int | None) -> PartitionBound:
    """Parse one partition's ``pg_get_expr(relpartbound)`` text into a bound.

    Range bounds become UTC datetimes because the planner aligns ranges and cuts cache segments on
    month boundaries; list and hash partitions carry their own text, since their values are not
    datetimes and inventing one would be worse than reporting none.
    """
    text = definition.strip()
    range_match = _RANGE_BOUND_RE.match(text)
    if range_match is not None:
        return PartitionBound(
            name=name,
            lower=_bound_datetime(range_match.group("lower")),
            upper=_bound_datetime(range_match.group("upper")),
            row_estimate=estimate,
        )
    if _LIST_BOUND_RE.match(text) is not None or _HASH_BOUND_RE.match(text) is not None:
        return PartitionBound(name=name, values=(text,), row_estimate=estimate)
    # DEFAULT (and anything a future release writes): the partition exists, its bound is unknown.
    return PartitionBound(name=name, row_estimate=estimate)


def _is_month_step(lower: datetime, upper: datetime) -> bool:
    """True when ``upper`` is exactly the first instant of the month after ``lower``."""
    start = lower.astimezone(UTC)
    end = upper.astimezone(UTC)
    if (start.day, start.hour, start.minute, start.second, start.microsecond) != (1, 0, 0, 0, 0):
        return False
    next_month = datetime(start.year + start.month // 12, start.month % 12 + 1, 1, tzinfo=UTC)
    return end == next_month


def _infer_granularity(bounds: Sequence[PartitionBound]) -> PartitionGranularity | None:
    """Infer the partition width when every step agrees, else ``None`` (never a guess)."""
    steps = [(bound.lower, bound.upper) for bound in bounds if bound.lower and bound.upper]
    if not steps:
        return None
    if all(_is_month_step(lower, upper) for lower, upper in steps if lower and upper):
        return PartitionGranularity.MONTH
    deltas = {(upper - lower) for lower, upper in steps if lower and upper}
    if len(deltas) != 1:
        return None
    return _MONTH_STEPS.get(deltas.pop())


def _partition_key_text(definition: str) -> str:
    """``RANGE (start_time)`` -> ``start_time``; an expression key keeps its parentheses."""
    match = _PARTKEY_RE.match(definition.strip())
    return match.group("key") if match else ""


def _partitioning(parent: tuple[str, str], rows: Sequence[CatalogRow]) -> Partitioning:
    """Turn one parent's catalog rows into validated :class:`Partitioning` metadata."""
    qualified = f"{parent[0]}.{parent[1]}"
    strategy = PARTITION_STRATEGIES.get(str(rows[0][2]))
    if strategy is None:
        # An inheritance tree that is not declarative partitioning still exists and is still
        # scanned in full; refusing to claim a strategy is the honest answer.
        raise NotSupported(
            OPTIONAL_METHODS["extract_partitions"],
            f"{qualified} is partitioned by a mechanism this adapter does not read",
        )
    bounds = tuple(_bound(str(row[5]), str(row[6]), row_estimate(row[7])) for row in rows)
    key = split_names(_partition_key_text(str(rows[0][3])))
    if strategy is not PartitionStrategy.EXPRESSION and not key:
        raise NotSupported(
            OPTIONAL_METHODS["extract_partitions"],
            f"{qualified}: the partition key expression is not a column list",
        )
    return Partitioning(
        strategy=strategy,
        key=key,
        granularity=_infer_granularity(bounds) if strategy is PartitionStrategy.RANGE else None,
        bounds=bounds,
    )


def parse_partitions(rows: Iterable[CatalogRow]) -> tuple[Partitioning, ...]:
    """One :class:`Partitioning` per partitioned table, from :data:`PARTITIONS_SQL` rows.

    Rows are grouped by ``(schema, table)`` so a table with fourteen partitions is one entry, and
    the partitions keep the ``ORDER BY`` the query asked for (by name, which for the monthly naming
    convention the loader uses is also chronological).

    Raises:
        NotSupported: a partitioned table is partitioned in a shape this adapter cannot describe.
            Reporting an empty or invented scheme would let the planner prune on fiction.
    """
    grouped: dict[tuple[str, str], list[CatalogRow]] = {}
    for row in rows:
        grouped.setdefault((str(row[0]), str(row[1])), []).append(row)
    return tuple(_partitioning(parent, group) for parent, group in sorted(grouped.items()))


# --- the plan ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PgPlan:
    """What one ``EXPLAIN (FORMAT JSON)`` document says, in the terms the gate reasons about.

    ``relations`` is ordered by first appearance and holds every relation a plan node named, so a
    join lists both sides; the caller decides which of them are partitions of the table it asked
    about. ``subplans_removed`` is PostgreSQL's own statement that it dropped partitions before
    running: a plan-time prune, which is the strong form of the proof.
    """

    relations: tuple[str, ...] = ()
    indexes: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    subplans_removed: int = 0
    plan_rows: float | None = None
    total_cost: float | None = None

    @property
    def pruned_at_plan_time(self) -> bool:
        """True when the planner removed partition subplans before execution."""
        return self.subplans_removed > 0


def _as_float(value: object) -> float | None:
    """``float`` or ``None``: a plan may omit a cost, and a missing number is not zero."""
    if value is None or isinstance(value, bool):
        return None
    return float(value) if isinstance(value, int | float | str) else None


def _as_count(value: object) -> int:
    """A small non-negative counter from the plan (``Subplans Removed``), ``0`` when absent."""
    if isinstance(value, bool) or value is None:
        return 0
    return int(value) if isinstance(value, int | float | str) and int(value) > 0 else 0


def _root(document: object) -> Mapping[str, Any] | None:
    """The plan node at the top of an ``EXPLAIN (FORMAT JSON)`` document, or ``None``."""
    node: object = document
    if isinstance(node, list):
        node = node[0] if node else None
    if not isinstance(node, Mapping):
        return None
    inner = node.get("Plan")
    if isinstance(inner, Mapping):
        return inner if "Node Type" in inner else None
    return node if "Node Type" in node else None


def _walk(node: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Every plan node under ``node``, parents first (``Plans`` is the child list)."""
    yield node
    children = node.get("Plans")
    if isinstance(children, list):
        for child in children:
            if isinstance(child, Mapping):
                yield from _walk(child)


def parse_plan(document: object) -> PgPlan | None:
    """Read one decoded ``EXPLAIN (FORMAT JSON)`` document, or ``None`` when it is not a plan.

    ``None`` is a real answer, not a failure: a plan from a source that does not produce JSON must
    not be read as "zero partitions scanned", because that reads as a perfect prune.
    """
    root = _root(document)
    if root is None:
        return None
    relations: dict[str, None] = {}
    indexes: dict[str, None] = {}
    parameters: dict[str, None] = {}
    removed = 0
    for node in _walk(root):
        relation = node.get("Relation Name")
        if isinstance(relation, str):
            relations[relation] = None
        index = node.get("Index Name")
        if isinstance(index, str):
            indexes[index] = None
        removed += _as_count(node.get("Subplans Removed"))
        for key in _CONDITION_KEYS:
            condition = node.get(key)
            if isinstance(condition, str):
                for match in _PARAM_RE.finditer(condition):
                    parameters[f"${match.group('number')}"] = None
    return PgPlan(
        relations=tuple(relations),
        indexes=tuple(indexes),
        parameters=tuple(parameters),
        subplans_removed=removed,
        plan_rows=_as_float(root.get("Plan Rows")),
        total_cost=_as_float(root.get("Total Cost")),
    )


def parse_plan_text(text: str | None) -> PgPlan | None:
    """Parse the plan text a :class:`PlanSummary` carries back into a :class:`PgPlan`."""
    if text is None:
        return None
    try:
        return parse_plan(json.loads(text))
    except json.JSONDecodeError:
        return None


# --- the verdict ------------------------------------------------------------------------


def partitions_in_window(partitioning: Partitioning, window: Range) -> tuple[str, ...]:
    """The declared partitions a half-open ``window`` can touch, in declared order.

    The rule is the planner's (:func:`mdqm.planner.partitions.estimate_partition_scan`): a partition
    is touched when it starts before the window ends and ends after the window starts. A partition
    whose bound did not parse (``MINVALUE``/``MAXVALUE``, or a ``DEFAULT`` partition that can hold
    rows of any date) is always counted, because it can hold rows of any window.
    """
    if window.is_empty:
        return ()
    touched: list[str] = []
    for bound in partitioning.bounds:
        if bound.upper is not None and bound.upper <= window.lower:
            continue
        if bound.lower is not None and bound.lower >= window.upper:
            continue
        touched.append(bound.name)
    return tuple(touched)


@dataclass(frozen=True, slots=True)
class PruningVerdict:
    """A justified answer about pruning: what was expected, what the plan named, and why.

    ``PROVEN`` means the two lists are equal, which is checkable by anyone who runs the same
    ``EXPLAIN``. ``UNPROVEN`` means the plan named something else and ``unexpected``/``missing``
    says what. ``UNKNOWN`` means this source's metadata cannot support a verdict at all.
    """

    proof: PruningProof = PruningProof.UNKNOWN
    expected: tuple[str, ...] = ()
    """The declared partitions the intended window spans."""
    named: tuple[str, ...] = ()
    """The declared partitions the plan named as scanned."""
    other_relations: tuple[str, ...] = ()
    """Relations the plan named that are not partitions of this table (join inputs, indexes)."""
    reason: str = ""

    @property
    def unexpected(self) -> tuple[str, ...]:
        """Declared partitions the plan scanned although the window does not span them."""
        return tuple(name for name in self.named if name not in self.expected)

    @property
    def missing(self) -> tuple[str, ...]:
        """Declared partitions the window spans that the plan did not name."""
        return tuple(name for name in self.expected if name not in self.named)

    @property
    def ok(self) -> bool:
        """True only when the plan proved it scans no more than the window spans."""
        return self.proof is PruningProof.PROVEN

    def stamp(self, summary: PlanSummary) -> PlanSummary:
        """``summary`` with this verdict's ``pruning_proof`` filled in (what history records)."""
        return summary.model_copy(update={"pruning_proof": self.proof})


def _names(names: Sequence[str]) -> str:
    """Names for a one-line reason: comma-separated, in the order the caller built them."""
    return ", ".join(names) if names else "none"


def pruning_verdict(
    summary: PlanSummary,
    partitioning: Partitioning | None,
    scope: Range | None,
) -> PruningVerdict:
    """Decide whether ``summary`` proves the scan stayed inside ``scope``.

    Args:
        summary: what the source's ``EXPLAIN`` returned, ``raw`` included.
        partitioning: the table's declared partition metadata (``None`` when it has none).
        scope: the window the spec intends to read, or ``None`` when the fragment carries no window
            on the partition key.

    Returns:
        A :class:`PruningVerdict`. The order of the early returns is the order of the excuses: a
        non-partitioned table, then a missing window, then missing bounds, then a partitioning
        scheme a time window cannot address, then an unreadable plan. Every one of them is
        ``UNKNOWN``; only an exact set match is ``PROVEN``.
    """
    if partitioning is None or partitioning.strategy is PartitionStrategy.NONE:
        return PruningVerdict(
            reason="the table is not partitioned, so there is no partition list to prove against",
        )
    if scope is None:
        return PruningVerdict(
            reason="the fragment carries no window on the partition key, so no partition list "
            "can be expected",
        )
    if not partitioning.bounds:
        return PruningVerdict(
            reason="the source reported no partition bounds, so nothing can be expected",
        )
    if partitioning.strategy is not PartitionStrategy.RANGE:
        return PruningVerdict(
            reason=f"the table is partitioned by {partitioning.strategy.value}, and a window does "
            "not determine which of its partitions must be read",
        )
    plan = parse_plan_text(summary.raw)
    if plan is None:
        return PruningVerdict(
            reason="the plan summary carries no EXPLAIN document that could be parsed",
        )
    expected = partitions_in_window(partitioning, scope)
    declared = frozenset(bound.name for bound in partitioning.bounds)
    named = tuple(name for name in plan.relations if name in declared)
    other = tuple(name for name in plan.relations if name not in declared)
    if set(named) == set(expected):
        removed = (
            f"; PostgreSQL removed {plan.subplans_removed} partition subplans at plan time"
            if plan.pruned_at_plan_time
            else ""
        )
        return PruningVerdict(
            proof=PruningProof.PROVEN,
            expected=expected,
            named=named,
            other_relations=other,
            reason=f"the plan named exactly the {len(expected)} declared partition(s) the window "
            f"spans ({_names(named)}){removed}",
        )
    unexpected = tuple(name for name in named if name not in expected)
    missing = tuple(name for name in expected if name not in named)
    parts: list[str] = []
    if unexpected:
        parts.append(
            f"scanned {len(unexpected)} partition(s) outside the window ({_names(unexpected)})"
        )
    if missing:
        parts.append(
            f"did not name {len(missing)} partition(s) the window spans ({_names(missing)})"
        )
    if plan.parameters:
        parts.append(
            f"pruning is deferred to run time ({', '.join(plan.parameters)}), so the plan cannot "
            "prove what it will read"
        )
    return PruningVerdict(
        proof=PruningProof.UNPROVEN,
        expected=expected,
        named=named,
        other_relations=other,
        reason="; ".join(parts) or "the plan and the declared bounds disagree",
    )
