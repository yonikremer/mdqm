"""The federated executor: one fragment per source, one local join, one honest report (task C17).

This is the engine of the MVP slice. A cross-source question never becomes one statement: it is a
:class:`~mdqm.executor.models.FederatedPlan` of already-compiled fragments (C09 — one per source,
with every filter the source can apply pushed into it), and this module does the four things the
sources cannot do for us:

1. **Fetch through the governor, never around it.** Every fragment goes through
   :meth:`mdqm.governor.core.Governor.execute`, so the breaker, the per-source slots, the quota, the
   deadline and the result store all apply. An adapter is never called from here.
2. **Cap before the join.** A fragment that returns more rows (or bytes) than its
   :class:`~mdqm.models.Limits` allows is refused with ``COST_TOO_HIGH`` and a ``fix`` that narrows
   the filter — before any local join runs, so an over-broad read cannot quietly become a slow one.
3. **Join and aggregate locally in DuckDB.** The Arrow results are registered as relations in this
   executor's *own* connection (see "Two DuckDB instances" below) and the answer is one statement
   over them, built with sqlglot: the approved join's own ON clause (with its key conversion),
   the grouping and the aggregates.
4. **Report what happened.** Rows per source with their own ``fetched_at``, rows each join matched,
   dropped and multiplied, NULL groups a value map does not explain, normalisation values that miss
   the measure's rounding step. A silent cross-source row loss is the failure this task exists to
   prevent, so it is a field in :class:`~mdqm.executor.models.QualityReport`, not a log line.

Two DuckDB instances, kept apart on purpose
-------------------------------------------
The fake adapter (F05) *is* a DuckDB database: it runs the fragment SQL in its own connection. The
executor opens its own connection for the cache and the join. They are separate databases and the
executor never reaches into an adapter's connection — the Arrow table the governor drained is the
only thing that crosses, so a local join cannot accidentally read a source's table and call it a
fetch. The flip side is stated where it matters: nothing here is tested against the fake's internal
tables; every assertion goes through the arrow result and the report.

Failure means no answer
-----------------------
When a source fails (a refusal, a timeout, an open breaker), :meth:`FederatedExecutor.execute`
returns *no rows* and an ``ErrorResponse``, plus the partial report of the steps that did run. Half
a join returned as if it were whole is exactly the wrong answer, so it is not representable here.

Not in this module: splitting one multi-source *spec* into fragments. The spec model (C04) is
single-source, so the split belongs to whoever owns the multi-source spec (M04 builds it on top of
this executor); this module takes the fragments and the plan, which is where the pushdown decisions
have already been made per source.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, NamedTuple, cast

import duckdb
import pyarrow as pa
import sqlglot
from sqlglot import exp

from mdqm.cache.freshness import (
    DEFAULT_MAX_TTL,
    DEFAULT_MIN_TTL,
    Clock,
    SystemClock,
)
from mdqm.compiler.resolve import find_join, join_tables
from mdqm.errors import cost_too_high, from_exception
from mdqm.executor.cache import FragmentCache, fragment_key
from mdqm.executor.models import (
    SPEC_SAMPLE_KEYS,
    FederatedPlan,
    FragmentStep,
    JoinReport,
    JoinStep,
    LocalAggregate,
    NullGroup,
    QualityReport,
    SanityViolation,
    StepReport,
)
from mdqm.governor.core import Governor
from mdqm.governor.refusal import GovernorRefusal
from mdqm.models import ErrorResponse
from mdqm.semantic.measures import render_normalize
from mdqm.semantic.models import ApprovedJoin, SemanticLayer, render_template
from mdqm.spec.models import Agg

__all__ = [
    "DEFAULT_MEMORY_LIMIT",
    "FederatedExecutor",
    "FederatedResult",
]

#: Where DuckDB spills a join that outgrows memory. Its own spilling: no extension, no network.
DEFAULT_MEMORY_LIMIT: Final = "512MB"

#: A DuckDB size setting comes from config, so it is checked before it is inlined into a ``SET``.
_SIZE_PATTERN: Final = re.compile(r"[0-9]+(\.[0-9]+)?[KMGT]?B")

#: Relation-name prefix for a registered step result. The prefix keeps a step named like a cache
#: table from colliding with it; the alias the SQL uses is the plain step name.
_STEP_PREFIX: Final = "_mdqm_step_"

#: Aggregates over one local column. ``count_distinct`` is built separately (``COUNT(DISTINCT x)``).
_AGGREGATES: Final[Mapping[Agg, Callable[[exp.Expression], exp.Expression]]] = {
    Agg.AVG: lambda node: exp.Avg(this=node),
    Agg.SUM: lambda node: exp.Sum(this=node),
    Agg.MIN: lambda node: exp.Min(this=node),
    Agg.MAX: lambda node: exp.Max(this=node),
    Agg.COUNT: lambda node: exp.Count(this=node),
}

#: Tolerance for the normalisation sanity check, as a fraction of the rounding step. A DECIMAL
#: template lands exactly on the step; this only forgives a float representation of the same value.
_SANITY_TOLERANCE: Final = 1e-6


@dataclass(frozen=True, slots=True)
class FederatedResult:
    """A federated answer: rows, the report that explains them, and the statement that made them.

    ``table`` is ``None`` exactly when ``error`` is set. Arrow is not JSON, so this is a plain
    frozen dataclass rather than a :class:`~mdqm.models.base.MDQMModel`: it never crosses the wire.
    """

    quality: QualityReport
    table: pa.Table | None = None
    sql: str | None = None
    error: ErrorResponse | None = None

    @property
    def ok(self) -> bool:
        """True when there is an answer to return."""
        return self.table is not None and self.error is None

    @property
    def rows(self) -> int:
        """Rows in the answer, or 0 when there is none."""
        return 0 if self.table is None else int(self.table.num_rows)


class _On(NamedTuple):
    """One resolved join condition, and the plain columns on each side it compares."""

    expression: exp.Expression
    text: str
    left_column: str
    right_column: str


class _Materialised(NamedTuple):
    """One step's outcome: its report, its rows, and the refusal when it produced none."""

    report: StepReport
    table: pa.Table | None = None
    error: ErrorResponse | None = None


class FederatedExecutor:
    """Runs a :class:`~mdqm.executor.models.FederatedPlan`; see the module docstring.

    Args:
        governor: the only path to a source. Injected so a test can pin every protection.
        layer: the semantic layer, because the ON clause of an approved join (key conversion
            included) lives there and must not be re-invented here.
        clock: injected time source for cache freshness; tests drive staleness instead of sleeping.
        path: the DuckDB file for the cache and the join, or ``None`` for in-memory.
        memory_limit: DuckDB's memory ceiling, so a big join spills instead of dying.
        spill_dir: DuckDB's temp directory for that spill.
        min_ttl: cache floor TTL, also the TTL of a whole-fragment read.
        max_ttl: cache ceiling TTL.

    Raises:
        ValueError: ``memory_limit`` is not a DuckDB size such as ``"512MB"``.
    """

    def __init__(
        self,
        governor: Governor,
        *,
        layer: SemanticLayer,
        clock: Clock | None = None,
        path: Path | None = None,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        spill_dir: Path | None = None,
        min_ttl: timedelta = DEFAULT_MIN_TTL,
        max_ttl: timedelta = DEFAULT_MAX_TTL,
    ) -> None:
        if not _SIZE_PATTERN.fullmatch(memory_limit):
            raise ValueError(f"memory_limit must look like '512MB', got {memory_limit!r}")
        self._governor = governor
        self._layer = layer
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._closed = False
        if path is None:
            self._con: duckdb.DuckDBPyConnection = duckdb.connect(":memory:")
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._con = duckdb.connect(str(path))
        # UTC everywhere: a fetched_at must round-trip to the same instant it was recorded at.
        self._con.execute("SET TimeZone='UTC'").fetchall()
        # A SET value cannot be bound as a parameter; the regex above restricts it to <size><unit>.
        self._con.execute(f"SET memory_limit='{memory_limit}'").fetchall()  # nosemgrep
        if spill_dir is not None:
            spill = Path(spill_dir)
            spill.mkdir(parents=True, exist_ok=True)
            self._con.execute(f"SET temp_directory='{spill.as_posix()}'").fetchall()  # nosemgrep
        self._cache = FragmentCache(self._con, clock=self._clock, min_ttl=min_ttl, max_ttl=max_ttl)

    # --- lifecycle ------------------------------------------------------------------------

    def close(self) -> None:
        """Close the local database. Idempotent."""
        if not self._closed:
            self._closed = True
            self._con.close()

    def __enter__(self) -> FederatedExecutor:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @property
    def cache(self) -> FragmentCache:
        """The fragment cache, so a caller can report or clear what it holds."""
        return self._cache

    # --- the execution --------------------------------------------------------------------

    def execute(self, plan: FederatedPlan, *, user: str) -> FederatedResult:
        """Fetch every fragment, join them locally, and report what the answer cost and lost.

        Args:
            plan: the fragments, their caps and joins, and the requested output.
            user: the user the governor's quota and result handles belong to.

        Returns:
            A :class:`FederatedResult`. When ``error`` is set there is no table: a partial join is
            never returned as an answer.

        Raises:
            ValueError: the plan names an approved join the layer does not have, or the join does
                not connect the two steps it is attached to. Both are plan bugs, not agent errors.
        """
        reports: list[StepReport] = []
        tables: dict[str, pa.Table] = {}
        for step in plan.steps:
            materialised = self._materialise(step, plan.exec_ts, user)
            reports.append(materialised.report)
            if materialised.error is not None or materialised.table is None:
                quality = QualityReport(
                    exec_ts=plan.exec_ts,
                    complete=False,
                    steps=tuple(reports),
                    notes=(
                        f'step "{step.name}" on source "{step.source}" did not return rows; '
                        "no answer is produced rather than a partial join",
                    ),
                )
                return FederatedResult(
                    quality=quality,
                    error=materialised.error,
                )
            tables[step.name] = materialised.table

        registered: list[str] = []
        try:
            for step in plan.steps:
                relation = _STEP_PREFIX + step.name
                self._con.register(relation, tables[step.name])
                registered.append(relation)
            reports = [
                report.model_copy(update={"deduplicated": self._deduplicated(step)})
                if step.dedupe_on
                else report
                for report, step in zip(reports, plan.steps, strict=True)
            ]
            sanity = tuple(
                violation
                for step in plan.steps
                for violation in self._sanity(step, tables[step.name])
            )
            joins = tuple(self._join_report(plan, join, tables) for join in plan.joins)
            null_groups = self._null_groups(plan)
            sql, table = self._answer(plan)
        except duckdb.Error as exc:
            # The plan and the fragments disagree (an unknown column, a type clash): our bug, so it
            # is mapped with a correlation id instead of being shown as an agent-facing error.
            return FederatedResult(
                quality=QualityReport(
                    exec_ts=plan.exec_ts,
                    complete=False,
                    steps=tuple(reports),
                    notes=("the local join failed; the plan and the fragments disagree",),
                ),
                error=from_exception(exc),
            )
        finally:
            # The connection may still be unwinding a failed statement; the caller already has that
            # failure, so a cleanup error must not replace it.
            for relation in registered:
                with suppress(duckdb.Error):
                    self._con.unregister(relation)

        notes = tuple(
            f'join "{join.join}" dropped {join.dropped} of {join.rows_left} left rows'
            for join in joins
            if join.dropped
        ) + tuple(
            f'join "{join.join}" multiplied {join.multiplied} rows on duplicate partner keys'
            for join in joins
            if join.multiplied
        )
        quality = QualityReport(
            exec_ts=plan.exec_ts,
            complete=True,
            steps=tuple(reports),
            joins=joins,
            null_groups=null_groups,
            sanity=sanity,
            notes=notes,
        )
        return FederatedResult(quality=quality, table=table, sql=sql)

    # --- one fragment ---------------------------------------------------------------------

    def _materialise(self, step: FragmentStep, exec_ts: datetime, user: str) -> _Materialised:
        """Serve one step from cache or from its source, and enforce its caps."""
        key: str | None = None
        if step.cache is not None:
            key = fragment_key(
                source=step.source,
                table=step.table,
                sql=step.fragment.sql,
                version_hash=step.cache.version_hash,
            )
            lookup = self._cache.lookup(key, mutability=step.cache.mutability, exec_ts=exec_ts)
            if lookup.hit and lookup.table is not None and lookup.fetched_at is not None:
                # The caps are re-checked on a hit: a limit that shrank since the rows were cached
                # must not be bypassed by the cache.
                over = self._over_cap(step, int(lookup.table.num_rows), int(lookup.table.nbytes))
                if over is not None:
                    return _Materialised(
                        report=StepReport(
                            name=step.name,
                            source=step.source,
                            table=step.table,
                            sql=step.fragment.sql,
                            rows=int(lookup.table.num_rows),
                            nbytes=int(lookup.table.nbytes),
                            fetched=False,
                            cache_status="hit",
                            age_s=lookup.age_s,
                            fetched_at=lookup.fetched_at,
                        ),
                        error=over,
                    )
                return _Materialised(
                    report=StepReport(
                        name=step.name,
                        source=step.source,
                        table=step.table,
                        sql=step.fragment.sql,
                        rows=int(lookup.table.num_rows),
                        nbytes=int(lookup.table.nbytes),
                        fetched=False,
                        cache_status="hit",
                        age_s=lookup.age_s,
                        fetched_at=lookup.fetched_at,
                    ),
                    table=lookup.table,
                )
            status = lookup.status
        else:
            status = "uncached"
        fetched_at = self._clock.now()

        # One row past the cap, so an exactly-at-the-cap result is not mistaken for a truncated one.
        probe = step.limits.model_copy(update={"max_rows": step.limits.max_rows + 1})
        try:
            outcome = self._governor.execute(
                source=step.source,
                user=user,
                fragment=step.fragment,
                limits=probe,
            )
        except GovernorRefusal as refusal:
            return _Materialised(
                report=StepReport(
                    name=step.name,
                    source=step.source,
                    table=step.table,
                    sql=step.fragment.sql,
                    cache_status=status,
                    fetched_at=fetched_at,
                ),
                error=refusal.response,
            )
        table = self._governor.store.table(outcome.handle.handle, owner=user)
        rows = int(table.num_rows)
        nbytes = int(table.nbytes)
        row_cap_hit = rows > step.limits.max_rows
        byte_cap_hit = outcome.report.byte_cap_hit or nbytes > step.limits.max_bytes
        report = StepReport(
            name=step.name,
            source=step.source,
            table=step.table,
            sql=step.fragment.sql,
            rows=rows,
            nbytes=nbytes,
            fetched=True,
            cache_status=status,
            fetched_at=fetched_at,
            row_cap_hit=row_cap_hit,
            byte_cap_hit=byte_cap_hit,
            waited_s=outcome.report.waited_s,
        )
        over = self._over_cap(step, rows, nbytes, flagged=row_cap_hit or byte_cap_hit)
        if over is not None:
            return _Materialised(report=report, error=over)
        if key is not None and step.cache is not None:
            fetched_at = self._cache.store(
                key,
                table,
                source=step.source,
                table_name=step.table,
                version_hash=step.cache.version_hash,
            )
            report = report.model_copy(update={"fetched_at": fetched_at})
        return _Materialised(report=report, table=table)

    @staticmethod
    def _over_cap(
        step: FragmentStep, rows: int, nbytes: int, *, flagged: bool = False
    ) -> ErrorResponse | None:
        """The ``COST_TOO_HIGH`` refusal when a step's rows (or bytes) exceed its cap.

        ``flagged`` is the governor's own report that a cap stopped the stream, which can be true
        even when the drained total is exactly at the cap.
        """
        if not flagged and rows <= step.limits.max_rows and nbytes <= step.limits.max_bytes:
            return None
        return cost_too_high(
            table=step.table,
            estimated=f"{rows} rows / {nbytes} bytes",
            limit=f"{step.limits.max_rows} rows / {step.limits.max_bytes} bytes",
        )

    # --- the local answer -----------------------------------------------------------------

    def _relation(self, step: FragmentStep) -> exp.Expression:
        """The step's registered rows under the alias every expression refers to.

        A step with ``dedupe_on`` is presented as ``SELECT DISTINCT ON (key) *``: the key the
        semantic layer declares must be unique before a join, so duplicate rows are dropped here
        and counted in the report instead of multiplying the join's rows.
        """
        registered = _STEP_PREFIX + step.name
        if not step.dedupe_on:
            return cast(exp.Expression, exp.to_table(registered).as_(step.name))
        query = exp.select(exp.Star()).from_(exp.to_table(registered))
        query.set(
            "distinct",
            exp.Distinct(on=exp.Tuple(expressions=[exp.column(name) for name in step.dedupe_on])),
        )
        return cast(exp.Expression, query.subquery(step.name))

    def _deduplicated(self, step: FragmentStep) -> int:
        """How many duplicate rows the step's ``dedupe_on`` key removed from the join's input."""
        if not step.dedupe_on:
            return 0
        rows = self._count(
            exp.select(exp.Count(this=exp.Star())).from_(exp.to_table(_STEP_PREFIX + step.name))
        )
        distinct = self._count(
            exp.select(exp.Count(this=exp.Star())).from_(
                exp.select(*[exp.column(name) for name in step.dedupe_on])
                .distinct()
                .from_(exp.to_table(_STEP_PREFIX + step.name))
                .subquery()
            )
        )
        return max(0, rows - distinct)

    def _answer(self, plan: FederatedPlan) -> tuple[str, pa.Table]:
        """Build and run the local statement; return its SQL and its rows."""
        groups: list[exp.Expression] = []
        selects: list[exp.Expression] = []
        for group in plan.output.group_by:
            column = exp.column(group.column, table=group.step)
            groups.append(column.copy())
            selects.append(
                cast(
                    exp.Expression,
                    exp.alias_(column, exp.to_identifier(group.output_name, quoted=True)),
                )
            )
        for aggregate in plan.output.aggregates:
            node = _aggregate(aggregate.agg, self._measure_column(plan, aggregate))
            selects.append(
                cast(
                    exp.Expression,
                    exp.alias_(node, exp.to_identifier(aggregate.alias, quoted=True)),
                )
            )
        base = plan.steps[0]
        query = exp.select(*selects).from_(self._relation(base))
        filters: list[exp.Expression] = []
        for join in plan.joins:
            on = self._on(plan, join)
            if join.how == "semi":
                filters.append(_exists(plan.step(join.right), on.expression))
            else:
                query = query.join(
                    self._relation(plan.step(join.right)),
                    on=on.expression.copy(),
                    join_type="INNER" if join.how == "inner" else "LEFT",
                )
        for condition in filters:
            query = query.where(condition)
        if groups:
            query = query.group_by(*groups)
        for name in plan.output.sort_keys:
            query = query.order_by(name)
        query = query.limit(plan.output.limit)
        sql = query.sql(dialect="duckdb")
        table = cast(pa.Table, self._con.sql(sql).to_arrow_table())
        return sql, table

    def _null_groups(self, plan: FederatedPlan) -> tuple[NullGroup, ...]:
        """Count, per grouping column, the rows whose value is NULL before the join.

        Counted on the step's own result rather than on the answer: a grouped answer collapses every
        unmapped code into one NULL row, so the answer cannot say how much data was affected.
        """
        groups: list[NullGroup] = []
        for group in plan.output.group_by:
            column = exp.column(group.column)
            rows = self._count(
                exp.select(exp.Count(this=exp.Star()))
                .from_(exp.to_table(_STEP_PREFIX + group.step))
                .where(exp.Is(this=column, expression=exp.Null()))
            )
            if rows:
                groups.append(NullGroup(column=group.output_name, step=group.step, rows=rows))
        return tuple(groups)

    def _measure_column(self, plan: FederatedPlan, aggregate: LocalAggregate) -> exp.Expression:
        """The column to aggregate, in canonical units when it is a source form of a measure."""
        column = exp.column(aggregate.column, table=aggregate.step)
        if aggregate.measure is None:
            return cast(exp.Expression, column)
        measure = self._layer.measures.get(aggregate.measure)
        if measure is None:
            raise ValueError(
                f'aggregate "{aggregate.alias}" names measure "{aggregate.measure}", which this '
                f"semantic layer does not define "
                f"({', '.join(sorted(self._layer.measures))})"
            )
        step = plan.step(aggregate.step)
        qualified = f"{step.table}.{aggregate.column}"
        rendered = render_normalize(
            measure, qualified, column.sql(dialect="duckdb"), dialect="duckdb"
        )
        if rendered is None:
            raise ValueError(
                f'measure "{aggregate.measure}" declares no normalisation for {qualified}, so its '
                "unit is unknown; add a source template to the semantic layer"
            )
        return cast(exp.Expression, sqlglot.parse_one(rendered, read="duckdb"))

    def _on(self, plan: FederatedPlan, join: JoinStep) -> _On:
        """Resolve one approved join onto the two steps it connects.

        Raises:
            ValueError: the join is not in the layer, or connects other tables than the steps say.
        """
        approved = find_join(self._layer, join.join)
        if approved is None:
            raise ValueError(
                f'join "{join.join}" is not an approved join in this semantic layer '
                f"({', '.join(sorted(item.name for item in self._layer.approved_joins))})"
            )
        left_step, right_step = plan.step(join.left), plan.step(join.right)
        return _resolve_on(approved, left_step=left_step, right_step=right_step)

    # --- the joins' own report ------------------------------------------------------------

    def _join_report(
        self, plan: FederatedPlan, join: JoinStep, tables: Mapping[str, pa.Table]
    ) -> JoinReport:
        """Count what one join matched, dropped and multiplied; sample the unmatched keys."""
        on = self._on(plan, join)
        left, right = plan.step(join.left), plan.step(join.right)
        rows_left = int(tables[left.name].num_rows)
        matched = self._count(
            exp.select(exp.Count(this=exp.Star()))
            .from_(self._relation(left))
            .where(_exists(right, on.expression))
        )
        pairs = 0
        if join.how != "semi":
            pairs = self._count(
                exp.select(exp.Count(this=exp.Star()))
                .from_(self._relation(left))
                .join(
                    self._relation(right),
                    on=on.expression.copy(),
                    join_type="INNER",
                )
            )
        dropped = max(0, rows_left - matched)
        keys: tuple[str, ...] = ()
        if dropped:
            keys = self._dropped_keys(left, right, on)
        return JoinReport(
            join=join.join,
            left=join.left,
            right=join.right,
            how=join.how,
            on=on.text,
            rows_left=rows_left,
            matched=matched,
            dropped=dropped,
            multiplied=max(0, pairs - matched),
            dropped_keys=keys,
            reason=(
                f"left rows whose {left.table}.{on.left_column} has no {right.table} row"
                if dropped
                else "every left row matched"
            ),
        )

    def _dropped_keys(self, left: FragmentStep, right: FragmentStep, on: _On) -> tuple[str, ...]:
        """A bounded sample of the key values a join could not match."""
        query = (
            exp.select(exp.column(on.left_column, table=left.name))
            .from_(self._relation(left))
            .where(exp.Not(this=_exists(right, on.expression.copy())))
            .distinct()
            .limit(SPEC_SAMPLE_KEYS)
        )
        result = self._con.sql(query.sql(dialect="duckdb")).fetchall()
        return tuple(str(row[0]) for row in cast(list[tuple[object]], result))

    def _count(self, query: exp.Expression) -> int:
        """Run a ``SELECT COUNT(*)`` and return the number."""
        rows = cast(
            list[tuple[object]], self._con.sql(query.sql(dialect="duckdb")).limit(1).fetchall()
        )
        return 0 if not rows else cast(int, rows[0][0])

    # --- per-step checks ------------------------------------------------------------------

    def _sanity(self, step: FragmentStep, table: pa.Table) -> tuple[SanityViolation, ...]:
        """Count rows whose normalised measure is not a multiple of the rounding step."""
        violations: list[SanityViolation] = []
        for check in step.checks:
            if check.column not in table.column_names:
                raise ValueError(
                    f'step "{step.name}" checks column {check.column!r}, which its fragment does '
                    f"not return ({', '.join(table.column_names)})"
                )
            tolerance = _SANITY_TOLERANCE * check.multiple_of
            offending: list[str] = []
            count = 0
            for value in table.column(check.column).to_pylist():
                if value is None:
                    continue
                number = float(value) * check.scale
                if abs(number - round(number / check.multiple_of) * check.multiple_of) > tolerance:
                    count += 1
                    if len(offending) < SPEC_SAMPLE_KEYS:
                        offending.append(str(value))
            if count:
                violations.append(
                    SanityViolation(
                        step=step.name,
                        column=check.column,
                        multiple_of=check.multiple_of,
                        offending=count,
                        examples=tuple(offending),
                    )
                )
        return tuple(violations)


# --- helpers -----------------------------------------------------------------------------


def _aggregate(agg: Agg, node: exp.Expression) -> exp.Expression:
    """One aggregate over one local column, ``count_distinct`` included."""
    if agg is Agg.COUNT_DISTINCT:
        return cast(exp.Expression, exp.Count(this=exp.Distinct(expressions=[node])))
    return _AGGREGATES[agg](node)


def _exists(right: FragmentStep, condition: exp.Expression) -> exp.Expression:
    """``EXISTS (SELECT 1 FROM <right> WHERE <condition>)``: membership without fan-out."""
    return cast(
        exp.Expression,
        exp.Exists(
            this=exp.select(exp.Literal.number(1))
            .from_(exp.to_table(_STEP_PREFIX + right.name).as_(right.name))
            .where(condition.copy()),
        ),
    )


def _resolve_on(
    approved: ApprovedJoin, *, left_step: FragmentStep, right_step: FragmentStep
) -> _On:
    """Render an approved join's ON clause onto two plan steps.

    The template belongs to the semantic layer and carries the key conversion the join owns
    (``LPAD(CAST(cust_id AS VARCHAR), 10, '0') = client_no``); this function only decides which
    side is which and which relation alias each identifier names. Nothing is concatenated: the two
    identifiers come from the approved join and the alias is a validated plan step name.

    Raises:
        ValueError: the join connects other tables than the two steps claim.
    """
    left_table, right_table = join_tables(approved)
    if (left_step.table, right_step.table) == (left_table, right_table):
        sides = ((approved.left, "left", left_step), (approved.right, "right", right_step))
    elif (left_step.table, right_step.table) == (right_table, left_table):
        sides = ((approved.right, "right", left_step), (approved.left, "left", right_step))
    else:
        raise ValueError(
            f'approved join "{approved.name}" connects {left_table} and {right_table}, not '
            f"{left_step.table} and {right_step.table}"
        )
    # The template's {left}/{right} are the approved join's own sides, so each placeholder takes the
    # alias of the plan step that holds that side -- whichever way round the plan wrote the join.
    values: dict[str, str] = {}
    for side, placeholder, step in sides:
        identifier = exp.column(side.rpartition(".")[2], table=step.name)
        values[placeholder] = identifier.sql(dialect="duckdb")
    text = render_template(approved.on, values)
    return _On(
        expression=cast(exp.Expression, sqlglot.parse_one(text, dialect="duckdb")),
        text=text,
        left_column=sides[0][0].rpartition(".")[2],
        right_column=sides[1][0].rpartition(".")[2],
    )
