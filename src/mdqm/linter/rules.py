"""The rule contract: one small class per lint rule, plus the cases that prove it works.

A rule is a pure function over one input (:class:`LintState`) and this execution's facts
(:class:`LintContext`). It does exactly one of two things:

* **fixes** the input and says so in a sentence, which the caller reports in ``applied_fixes``
  (principle 10: an auto-fix is never silent), or
* **refuses** with the ``ErrorResponse`` an agent can act on, whose ``fix`` names the exact change
  to make instead.

It never does both, never touches a source, a file or a clock, and never guesses: when a fix could
change the answer, the rule refuses.

Every rule declares :attr:`Rule.cases`: the before/after examples that
``tests/unit/test_linter.py`` replays. :class:`~mdqm.linter.registry.Registry` refuses a rule
without one firing and one silent case, refuses an auto-fix rule without a case showing its fixed
output, and refuses an ``error``-severity rule without a case asserting the refusal code. "Add a
rule, forget the test" therefore cannot reach ``main``: the package does not import.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, ClassVar, Final

from pydantic import TypeAdapter

from mdqm.compiler import DEFAULT_MAX_ROWS
from mdqm.compiler.resolve import (
    TableRef,
    find_join,
    join_tables,
    known_tables,
    resolve_table,
)
from mdqm.errors import invalid_spec, join_not_approved
from mdqm.models import ErrorResponse
from mdqm.semantic.models import SemanticLayer
from mdqm.spec.durations import Duration, DurationField
from mdqm.spec.models import QuerySpec
from mdqm.sql.exec_ts import SUPPORTED_DIALECTS

__all__ = [
    "LintContext",
    "LintState",
    "Rule",
    "RuleCase",
    "RuleOutcome",
    "Severity",
    "Target",
    "duration_text",
    "spec_dump",
    "spec_duration",
]

#: One adapter for the semantic layer's ``timedelta`` window -> the spec's calendar-aware duration.
_DURATION: Final[TypeAdapter[Duration]] = TypeAdapter(DurationField)


def spec_duration(window: object) -> Duration:
    """A semantic-layer window (a ``timedelta``) as the spec's calendar-aware duration.

    The two lanes model a span differently (the semantic layer validates a ``timedelta``, the spec
    keeps calendar months), so the days/seconds split happens here; a ``timedelta`` has no months
    to lose.
    """
    if window is None:
        # Every Duration field defaults to 0; pyright reads a pydantic field with a default as
        # required, so a zero-arg construction it accepts at runtime is reported as incomplete.
        return Duration()  # pyright: ignore[reportCallIssue]
    if isinstance(window, timedelta):
        total = int(window.total_seconds())
        # Same false positive: ``months`` keeps its default of 0 here, which pyright still counts.
        return Duration(  # pyright: ignore[reportCallIssue]
            days=total // 86_400, seconds=total % 86_400
        )
    return _DURATION.validate_python(window)


def duration_text(window: object) -> str:
    """Render a semantic-layer window (a ``timedelta``) the way a spec writes it: ``"7d"``."""
    return spec_duration(window).to_text()


def spec_dump(spec: QuerySpec) -> dict[str, Any]:
    """A spec as JSON, with defaults and unset fields omitted: the shape a case expectation uses.

    Only the fields the caller actually wrote survive, so an ``expect_spec`` in a
    :class:`RuleCase` reads exactly like the spec an agent would send.
    """
    return spec.model_dump(mode="json", by_alias=True, exclude_defaults=True, exclude_none=True)


class Severity(StrEnum):
    """How a rule reacts when it applies.

    ``error``: it can refuse, and the query must not run until the agent changes the spec.
    ``notice``: it only ever repairs the input, so the agent has nothing to do.
    """

    ERROR = "error"
    NOTICE = "notice"


class Target(StrEnum):
    """Which input a rule reads: the structured spec (primary) or raw SQL (emergency exit)."""

    SPEC = "spec"
    SQL = "sql"


@dataclass(frozen=True, slots=True)
class LintContext:
    """Everything a rule needs about this execution and this source, and nothing more.

    ``columns`` is the catalog's column list per table (``source.table -> columns``). It must cover
    every table the query can reach, because an identifier is only known to be real when the
    catalog spells it; a table with no columns listed is *not* judged (an empty list means "the
    catalog was not passed", not "the table has no columns").

    ``exec_ts`` is the single value of this execution (C02), so an auto-fix that needs "now"
    (a derived bound, an injected window) is reproducible.

    ``extra_relations`` names the relations a dialect provides that no catalog reports, such as
    Oracle's ``dual``: a deployment declares them rather than the linter guessing them, and a
    raw-SQL rule never refuses one of them as unknown.
    """

    layer: SemanticLayer
    columns: Mapping[str, Sequence[str]]
    exec_ts: datetime
    dialect: str = "postgres"
    max_rows: int = DEFAULT_MAX_ROWS
    max_partitions: int | None = None
    lag_unverified: frozenset[str] = frozenset()
    #: Relations a dialect provides that no catalog lists (Oracle's ``dual``, a session table).
    extra_relations: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.exec_ts.tzinfo is None:
            raise ValueError("exec_ts must be timezone-aware UTC; a naive timestamp is a bug")
        if self.dialect not in SUPPORTED_DIALECTS:
            raise ValueError(
                f"unsupported dialect {self.dialect!r}: the linter knows "
                f"{sorted(SUPPORTED_DIALECTS)}"
            )
        if self.max_rows < 1:
            raise ValueError(f"max_rows must be >= 1, got {self.max_rows}")

    @property
    def now(self) -> datetime:
        """The execution timestamp, normalised to UTC."""
        return self.exec_ts.astimezone(UTC)

    def tables(self) -> tuple[TableRef, ...]:
        """Every table the semantic layer or the catalog knows, ordered by key."""
        return known_tables(self.layer, self.columns)

    def anchor(self, spec: QuerySpec) -> TableRef | ErrorResponse:
        """The ``from`` table, or the ``UNKNOWN_TABLE``/``INVALID_SPEC`` refusal."""
        return resolve_table(spec.from_table, tables=self.tables())

    def scope(self, spec: QuerySpec) -> tuple[TableRef, ...] | ErrorResponse:
        """The tables the spec reads: the anchor, then each named approved join's partner.

        Mirrors the compiler's join resolution, so a rule that reasons about a column sees exactly
        the tables the query will read -- or the refusal the compiler would produce.
        """
        anchor = self.anchor(spec)
        if isinstance(anchor, ErrorResponse):
            return anchor
        scope = [anchor]
        tables = self.tables()
        for index, name in enumerate(spec.joins):
            join = find_join(self.layer, name)
            if join is None:
                return join_not_approved(left=anchor.key, right=name)
            left, right = join_tables(join)
            if anchor.key not in (left, right):
                return _wrong_anchor(self, spec, index, anchor.key, (left, right))
            partner_key = right if left == anchor.key else left
            partner = resolve_table(partner_key, tables=tables, field=f"joins.{index}")
            if isinstance(partner, ErrorResponse):
                return partner
            if not any(table.key == partner.key for table in scope):
                scope.append(partner)
        return tuple(scope)


def _wrong_anchor(
    ctx: LintContext, spec: QuerySpec, index: int, anchor: str, pair: tuple[str, str]
) -> ErrorResponse:
    """The refusal for a join that exists but does not touch the spec's own table.

    The ``reason`` carries the diagnosis (which two tables the named join connects); the ``fix``
    lists join names the agent may use -- the ones that do touch its table, or, when no approved
    join does, every approved name, because a weak agent must never be left with "a valid value"
    and no candidate list.
    """
    touching = [join.name for join in ctx.layer.approved_joins if anchor in join_tables(join)]
    touching = touching or [join.name for join in ctx.layer.approved_joins]
    return invalid_spec(
        field=f"joins.{index}",
        reason=(
            f'the join "{spec.joins[index]}" connects {pair[0]} and {pair[1]}, '
            f'and the spec reads "{anchor}"'
        ),
        expected=sorted(touching),
    )


@dataclass(frozen=True, slots=True)
class LintState:
    """The document a rule works on: one spec, or one statement, never both."""

    spec: QuerySpec | None = None
    sql: str | None = None

    def __post_init__(self) -> None:
        if (self.spec is None) == (self.sql is None):
            raise ValueError("a lint state holds exactly one of a spec or a statement")


@dataclass(frozen=True, slots=True)
class RuleOutcome:
    """What one rule did: the repaired input plus a sentence per repair, or the refusal.

    The invariants are enforced here rather than trusted: a fix without a sentence is exactly the
    silent auto-fix principle 10 forbids, and an outcome that both fixes and refuses is a rule that
    could not decide.
    """

    fixes: tuple[str, ...] = ()
    spec: QuerySpec | None = None
    sql: str | None = None
    error: ErrorResponse | None = None

    def __post_init__(self) -> None:
        if self.error is not None:
            if self.fixes or self.spec is not None or self.sql is not None:
                raise ValueError("a refusal carries no fix and no repaired document")
            return
        if not self.fixes:
            raise ValueError("an applied fix must be reported: send at least one fix sentence")
        if self.spec is None and self.sql is None:
            raise ValueError("an outcome without an error must carry the repaired document")


@dataclass(frozen=True, slots=True)
class RuleCase:
    """One before/after example a rule declares and the test suite replays.

    A case supplies exactly one input (``spec`` or ``sql``) and says what must happen:

    * ``fires=False``: the rule leaves the input alone (no expectation may be set);
    * ``fires=True`` with ``expect_code``: the rule refuses, and the ``fix`` contains
      ``expect_fix``;
    * ``fires=True`` with ``expect_spec``/``expect_sql``: the rule repairs, and the result is the
      expected dump (see :func:`spec_dump`) or contains every ``expect_sql`` fragment.
    """

    name: str
    fires: bool = False
    spec: Mapping[str, Any] | None = None
    sql: str | None = None
    expect_spec: Mapping[str, Any] | None = None
    expect_sql: tuple[str, ...] = ()
    expect_absent: tuple[str, ...] = ()
    expect_code: str | None = None
    expect_fix: str | None = None
    expect_suggestions: tuple[str, ...] = ()
    #: An inline semantic layer for the rare rule whose two branches need different metadata.
    layer: Mapping[str, Any] | None = None
    #: The catalog for an inline layer (ignored when ``layer`` is unset).
    columns: Mapping[str, tuple[str, ...]] | None = None


class Rule(ABC):
    """One lint rule: an id, how it reacts, and the cases that prove both directions."""

    #: Stable, dotted and lower case (``spec.value``): an ``applied_fixes`` entry names it.
    id: ClassVar[str]
    severity: ClassVar[Severity]
    #: True when the rule can repair the input itself (it may still refuse
    #: where a repair would change the answer).
    autofix: ClassVar[bool]
    target: ClassVar[Target]
    #: One line for the docs and the review digest.
    title: ClassVar[str]
    #: The examples the test suite replays; the registry refuses an empty or one-sided set.
    cases: ClassVar[tuple[RuleCase, ...]]

    @abstractmethod
    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        """Apply the rule: an :class:`RuleOutcome`, or ``None`` when it does not apply."""
        raise NotImplementedError
