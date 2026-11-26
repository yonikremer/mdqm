"""The plan a federated execution follows, and the quality report it must produce (task C17).

The compiler (C09) is single-source: it turns one spec into one :class:`~mdqm.models.Fragment` for
one source. A cross-source question is therefore a *plan*: one already-compiled fragment per source,
the approved joins that connect them, and the output the analyst asked for. This module is the
shape of that plan and of the report that comes back with the answer.

Two rules decide the shapes here:

* **The plan is explicit, never guessed.** Which fragment belongs to which source, which key column
  each step publishes, which approved join connects two steps and how (``inner`` / ``left`` /
  ``semi``) is written down by whoever built the plan. Nothing is inferred from a fragment's SQL, so
  a plan that disagrees with the fragments fails loudly instead of answering something else.
* **The report carries the bad news.** Rows a join dropped, rows a join *multiplied* (a partner
  with duplicate keys inflates an average), groups whose categorical value no value map covers,
  normalisation values that are not a multiple of the measure's rounding step, and the
  ``fetched_at`` of every source: all of it is a field, so a silent loss is impossible by
  construction (``docs/DESIGN.md`` principle 10 / AGENTS.md principle 10).

Frozen and ``extra="forbid"``, like every other shared model.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, model_validator

from mdqm.cache.freshness import Mutability
from mdqm.models import Fragment, Limits
from mdqm.models.base import MDQMModel, UtcDatetime
from mdqm.spec.models import Agg

__all__ = [
    "SPEC_SAMPLE_KEYS",
    "ColumnCheck",
    "FederatedOutput",
    "FederatedPlan",
    "FragmentStep",
    "GroupColumn",
    "JoinReport",
    "JoinStep",
    "LocalAggregate",
    "NullGroup",
    "QualityReport",
    "SanityViolation",
    "StepCache",
    "StepReport",
]

#: How many unmatched key values a join report quotes. A bounded sample, never the whole set: the
#: count is the fact, the sample is for a human reading the report.
SPEC_SAMPLE_KEYS = 5

#: A step name becomes a DuckDB relation name, so it is checked rather than trusted.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

#: Which join shapes the executor can run. ``semi`` keeps the left rows but never multiplies them:
#: the partner only decides membership (a partner with duplicate keys must not inflate an average).
JoinHow = Literal["inner", "left", "semi"]


def _identifier(value: str, what: str) -> str:
    """Reject a name that could not safely be a relation name in the local DuckDB."""
    if not _IDENTIFIER.match(value):
        raise ValueError(f"{what} must match {_IDENTIFIER.pattern}, got {value!r}")
    return value


def _qualified(value: str, what: str) -> str:
    """Reject a table key that is not ``source.table``: the join sides are matched on it."""
    head, _, tail = value.rpartition(".")
    if not head or not tail:
        raise ValueError(f"{what} must be qualified as 'source.table', got {value!r}")
    return value


class StepCache(MDQMModel):
    """Whether one step's rows may be reused, and under which rule.

    ``mutability`` is the table's measured class (C12); ``version_hash`` is the semantic-layer
    version the fragment was compiled under, so a semantic change misses instead of serving rows
    computed by the old rules. A step with no :class:`StepCache` is always fetched.
    """

    version_hash: str = Field(
        min_length=1, description="Semantic-layer version the fragment was compiled under."
    )
    mutability: Mutability = Field(
        default="unknown",
        description="Measured mutability class; 'unknown' is never served from cache.",
    )


class ColumnCheck(MDQMModel):
    """A normalisation sanity check on a step's own rows.

    The semantic layer declares that a measure is rounded to a step (5000 in the motivating case).
    A source whose normalisation template does not deliver a multiple of that step is a conversion
    bug, and it is invisible in an average, so the executor counts the offending rows instead.

    ``scale`` is the source's own unit: erp stores ``limit_k`` as the canonical limit divided by 10,
    so ``scale=10`` checks ``col * 10`` against ``multiple_of``. Getting the scale wrong is exactly
    the bug this catches, and it is declared here rather than inferred, because inferring the scale
    is D8's job, not a report's.
    """

    column: str = Field(min_length=1, description="Column of this step's result.")
    multiple_of: int = Field(gt=1, description="The measure's rounding step.")
    scale: int = Field(
        default=1, ge=1, description="The source's unit: canonical = column * scale."
    )


class FragmentStep(MDQMModel):
    """One compiled fragment, its source, and the rules that apply to running it.

    ``table`` is the semantic table key the fragment anchors on (``crm.customers``); it is what the
    approved joins are matched against, so a join can never be attached to the wrong side.
    """

    name: str = Field(min_length=1, description="Step id and local relation name; [A-Za-z0-9_].")
    source: str = Field(min_length=1)
    table: str = Field(min_length=1, description="Semantic key of the fragment's anchor table.")
    fragment: Fragment
    limits: Limits = Field(description="Per-fragment caps; exceeding one is COST_TOO_HIGH.")
    cache: StepCache | None = Field(
        default=None, description="Cache policy, or null to always fetch."
    )
    dedupe_on: tuple[str, ...] = Field(
        default=(),
        description=(
            "Columns that must be unique before a join (a table's declared key): duplicate rows "
            "are dropped and counted instead of multiplying the join's rows."
        ),
    )
    checks: tuple[ColumnCheck, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> FragmentStep:
        _identifier(self.name, "a step name")
        _qualified(self.table, "a step's table")
        if self.fragment.source != self.source:
            raise ValueError(
                f"step {self.name!r} targets source {self.source!r} but its fragment is for "
                f"{self.fragment.source!r}"
            )
        # shortcut: a substring test on the SQL text, not a parse — enough to catch a check aimed
        # at a column the fragment never returns; upgrade to the compiler's output columns once
        # a plan is built from a CompiledQuery instead of a bare Fragment.
        sql = self.fragment.sql or ""
        for check in self.checks:
            if check.column not in sql:
                raise ValueError(
                    f"step {self.name!r} checks column {check.column!r}, which the fragment "
                    "does not select"
                )
        return self


class JoinStep(MDQMModel):
    """One approved join between two steps of the plan.

    ``how`` says what happens to a left row with no partner: ``left`` keeps it (the drop is
    reported), ``inner`` drops it (the drop is reported *and* the answer changes), ``semi`` keeps
    the left row exactly once when a partner exists and drops it otherwise.
    """

    join: str = Field(min_length=1, description="Approved join name from the semantic layer.")
    left: str = Field(min_length=1, description="Step that is already part of the join chain.")
    right: str = Field(min_length=1, description="Step this join attaches.")
    how: JoinHow = "left"


class GroupColumn(MDQMModel):
    """One grouping column of the answer, taken from one step's result."""

    step: str = Field(min_length=1)
    column: str = Field(min_length=1)
    alias: str | None = Field(default=None, min_length=1)

    @property
    def output_name(self) -> str:
        """The answer's column name: the alias, else the column's own name."""
        return self.alias or self.column


class LocalAggregate(MDQMModel):
    """One aggregate over a step's column, computed by the local engine after the join.

    ``measure`` says the column is a source form of a semantic measure, so the executor applies
    that measure's normalisation for this table before aggregating (``docs/DESIGN.md`` section 12:
    identity, ``ROUND(x/5000)*5000`` or ``x*10``). The normalisation happens locally because the
    compiler only normalises *aggregates* it emits itself: a fragment hands back rows, so the unit
    conversion of a row value belongs here, on the way into the local aggregate.
    """

    step: str = Field(min_length=1)
    column: str = Field(min_length=1)
    agg: Agg
    alias: str = Field(min_length=1, description="The answer's column name.")
    measure: str | None = Field(
        default=None,
        description="Semantic measure whose per-source normalisation applies to this column.",
    )


class FederatedOutput(MDQMModel):
    """What the analyst asked for, computed locally over the joined rows.

    ``order_by`` names output columns; when it is empty the answer is sorted by its grouping
    columns, because a deterministic order is a test's and a human's only defence against a
    silently reordered result.
    """

    group_by: tuple[GroupColumn, ...] = ()
    aggregates: tuple[LocalAggregate, ...] = Field(min_length=1)
    order_by: tuple[str, ...] = ()
    limit: int = Field(default=1_000, gt=0)

    @property
    def columns(self) -> tuple[str, ...]:
        """Output column names, in select order."""
        return (
            *(group.output_name for group in self.group_by),
            *(aggregate.alias for aggregate in self.aggregates),
        )

    @property
    def sort_keys(self) -> tuple[str, ...]:
        """Output columns the answer is ordered by: the declared ones, else the grouping keys."""
        return self.order_by or tuple(group.output_name for group in self.group_by)

    @model_validator(mode="after")
    def _names(self) -> FederatedOutput:
        names = self.columns
        if len(set(names)) != len(names):
            raise ValueError(f"output column names must be unique, got {list(names)}")
        for name in self.sort_keys:
            if name not in names:
                raise ValueError(
                    f"order_by names {name!r}, which is not an output column ({', '.join(names)})"
                )
        return self


class FederatedPlan(MDQMModel):
    """One federated execution: fragments, the joins between them, and the requested output.

    The first step is the ``FROM`` relation; every join attaches a step that is not in the chain
    yet to one that is. That restriction keeps the local statement left-deep, which is what makes
    the drop count of each join a single ``EXISTS`` query rather than a plan-diffing exercise.
    """

    exec_ts: UtcDatetime = Field(description="One timestamp per execution (C02).")
    steps: tuple[FragmentStep, ...] = Field(min_length=1)
    joins: tuple[JoinStep, ...] = ()
    output: FederatedOutput

    @property
    def names(self) -> tuple[str, ...]:
        """Step names, in plan order."""
        return tuple(step.name for step in self.steps)

    def step(self, name: str) -> FragmentStep:
        """The step called ``name``.

        Raises:
            KeyError: no such step; the validator on this model makes that a caller bug.
        """
        for step in self.steps:
            if step.name == name:
                return step
        raise KeyError(f"no step named {name!r} in this plan")

    @model_validator(mode="after")
    def _chain(self) -> FederatedPlan:
        names = self.names
        if len(set(names)) != len(names):
            raise ValueError(f"step names must be unique, got {list(names)}")
        attached = {names[0]}
        for join in self.joins:
            if join.left not in names or join.right not in names:
                raise ValueError(
                    f"join {join.join!r} names steps {join.left!r}/{join.right!r}; the plan has "
                    f"{', '.join(names)}"
                )
            if join.left not in attached:
                raise ValueError(
                    f"join {join.join!r} starts from {join.left!r}, which is not in the join chain "
                    f"yet ({', '.join(sorted(attached))})"
                )
            if join.right in attached:
                raise ValueError(f"join {join.join!r} attaches {join.right!r}, which is already in")
            if join.left == join.right:
                raise ValueError(f"join {join.join!r} joins step {join.left!r} to itself")
            attached.add(join.right)
        for group in self.output.group_by:
            if group.step not in names:
                raise ValueError(f"group_by names step {group.step!r}, not one of {list(names)}")
        for aggregate in self.output.aggregates:
            if aggregate.step not in names:
                raise ValueError(
                    f"aggregate {aggregate.alias!r} names step {aggregate.step!r}, not one of "
                    f"{list(names)}"
                )
        return self


# --- the report --------------------------------------------------------------------------


class StepReport(MDQMModel):
    """What one fragment cost and where its rows came from.

    ``sql`` is the exact statement the source ran, so the audit trail (DESIGN section 13) and a
    human reading a wrong answer both have the text. A cache hit reports ``fetched=False`` and the
    ``fetched_at`` of the rows it reused, never the time of this call.
    """

    name: str
    source: str
    table: str
    sql: str | None = None
    rows: int = Field(default=0, ge=0)
    nbytes: int = Field(default=0, ge=0)
    fetched: bool = Field(default=True, description="False when the rows came from the cache.")
    cache_status: Literal["hit", "stale", "miss", "uncached"] = "uncached"
    age_s: float | None = Field(default=None, ge=0, description="Age of the reused rows.")
    fetched_at: UtcDatetime
    row_cap_hit: bool = False
    byte_cap_hit: bool = False
    deduplicated: int = Field(
        default=0, ge=0, description="Duplicate rows dropped by the step's ``dedupe_on`` key."
    )
    waited_s: float = Field(default=0.0, ge=0, description="Time spent queueing for a slot.")


class JoinReport(MDQMModel):
    """What one join did to the row count, in counts rather than in adjectives.

    ``dropped`` is the number of left rows a partner did not match; those rows are gone from an
    ``inner`` or ``semi`` answer and kept by a ``left`` one. ``multiplied`` is the number of extra
    rows a many-side partner added, so a duplicated key that would inflate a sum or an average
    cannot pass unnoticed either.
    """

    join: str
    left: str
    right: str
    how: JoinHow
    on: str = Field(description="The rendered ON clause, key conversion included.")
    rows_left: int = Field(ge=0)
    matched: int = Field(ge=0, description="Left rows that found at least one partner.")
    dropped: int = Field(ge=0, description="Left rows that found none.")
    multiplied: int = Field(ge=0, description="Extra rows this join emitted for matched keys.")
    dropped_keys: tuple[str, ...] = ()
    reason: str = ""


class NullGroup(MDQMModel):
    """A grouping column whose value is NULL for some rows of a step's source result.

    After a value map, a NULL means the source held a code the map does not cover (the compiler's
    CASE falls through to NULL on purpose, so the row is visible rather than misfiled); a genuine
    NULL in a non-categorical column lands here too, which is why the reason says both. ``rows``
    counts source rows, so the number says how much of the data is affected rather than how many
    groups the answer happened to produce.
    """

    column: str
    step: str
    rows: int = Field(ge=0)
    reason: str = (
        "the group value is NULL: either a source value the value map does not cover, or a "
        "genuinely missing value"
    )


class SanityViolation(MDQMModel):
    """Rows of one step whose normalised measure is not a multiple of its rounding step."""

    step: str
    column: str
    multiple_of: int = Field(gt=1)
    offending: int = Field(ge=0)
    examples: tuple[str, ...] = ()
    reason: str = "a normalisation that does not land on the measure's rounding step is a bug"


class QualityReport(MDQMModel):
    """The receipt of a federated answer: what it cost, what it lost, and what it could not know.

    ``complete`` is the one field to read first: when a source failed, the report says so and there
    is no answer at all, because half a join returned as a whole one is the failure this executor
    exists to prevent.
    """

    exec_ts: UtcDatetime
    complete: bool
    steps: tuple[StepReport, ...] = ()
    joins: tuple[JoinReport, ...] = ()
    null_groups: tuple[NullGroup, ...] = ()
    sanity: tuple[SanityViolation, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def rows_fetched(self) -> int:
        """Rows every source returned, cache hits included (they were fetched once)."""
        return sum(step.rows for step in self.steps)

    @property
    def rows_dropped(self) -> int:
        """Rows the joins dropped, in total. The number an analyst must be told."""
        return sum(join.dropped for join in self.joins)
