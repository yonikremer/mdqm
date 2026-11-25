"""Measure normalisation: one canonical number out of three different storage forms.

The motivating case (``docs/DESIGN.md`` section 12): the credit limit means
``round_half_up(x / 5000) * 5000``. One source stores it already rounded, one stores it raw and one
stores it rounded and then divided by ten. Joining the three without converting them compares
thousands with hundreds and looks like success, so every source column is normalised before the
join, and the normalisation is *reported* (``applied_fixes``), never silent.

Two decisions live here, because both are meaning, not plumbing:

**Rounding is explicit.** A tie at ``x.5`` is the whole difference between the two plausible rules,
and the same ``ROUND`` means different things per engine: PostgreSQL rounds a ``numeric`` half-up
but a ``double precision`` half-even, while Oracle and SQL Server round half away from zero. A
measure therefore *declares* its mode (``half_up`` | ``half_even``) and this module renders the rule
as arithmetic (``FLOOR``/``CEIL``), so the emitted SQL cannot be dialect-lucky. The Python side
implements the same rule with exact integer fractions, never ``round()`` (which is half-even) and
never a binary float.

**Templates are SQL, so they are parsed and allowlisted.** A normalize template is written by the
pipeline or by a human, and it becomes part of a query. It may use arithmetic, ``ROUND``,
``FLOOR``, ``CEIL``, ``CAST`` and ``{col}`` -- nothing else: a subquery or an unknown function is a
load-time error naming the template, not a surprise at execution time.

Integer columns also need the ``CAST`` to an exact decimal type before a division. In PostgreSQL
and SQL Server ``2500 / 5000`` is integer division and gives ``0``, which would round to ``0`` and
lose the boundary entirely; casting to ``DECIMAL`` keeps the division exact and keeps money out of
binary floating point.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Final, cast

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from mdqm.semantic.models import Measure, RoundingMode, render_template

__all__ = [
    "ALLOWED_ROUNDING",
    "AppliedNormalization",
    "MultiplicityReport",
    "UndeclaredRounding",
    "check_multiples",
    "check_normalize_template",
    "normalizations_for",
    "render_normalize",
    "round_to_multiple",
]

#: The only functions a normalize template may call. ``CAST`` is included because an integer
#: column must become an exact decimal before a division (see the module docstring).
ALLOWED_ROUNDING: Final[frozenset[str]] = frozenset({"ROUND", "FLOOR", "CEIL"})

#: Node types a normalize template may contain: arithmetic over ``{col}`` and numeric literals.
_ALLOWED_NODES: Final[tuple[type[exp.Expression], ...]] = (
    exp.Add,
    exp.Sub,
    exp.Mul,
    exp.Div,
    exp.Mod,
    exp.Neg,
    exp.Paren,
    exp.Round,
    exp.Floor,
    exp.Ceil,
    exp.Cast,
    exp.Column,
    exp.Identifier,
    exp.Literal,
    exp.DataType,
    exp.DataTypeParam,
)

#: Stands in for ``{col}`` while a template is parsed, then replaced by the real identifier.
_PLACEHOLDER: Final = "__mdqm_col__"

#: Exact decimal type for a division. ``DECIMAL`` renders as ``NUMBER`` on Oracle and ``NUMERIC``
#: on SQL Server through sqlglot, so no dialect needs its own spelling.
_EXACT_TYPE: Final = "DECIMAL(38,6)"


class UndeclaredRounding(ValueError):
    """A template rounds but the measure never declared which rounding it means.

    Guessing half-up here would answer with the right number for one organisation and the wrong
    one for the next, so this is a declared fact or an error.
    """


# --- template validation -----------------------------------------------------------------


def _parse_template(template: str) -> exp.Expression:
    """Parse a template once ``{col}`` is replaced by a placeholder identifier."""
    text = render_template(template, {"col": _PLACEHOLDER})
    try:
        # sqlglot annotates this as ``Expr`` (a union including non-Expression leaves); a parsed
        # statement is always an Expression, so the cast records that rather than widening.
        return cast(exp.Expression, sqlglot.parse_one(text))
    except ParseError as exc:
        first = str(exc).splitlines()[0]
        raise ValueError(f"is not valid SQL: {first}") from None


def check_normalize_template(template: str, *, where: str = "a normalize template") -> None:
    """Reject a template that leaves the arithmetic/``ROUND``/``FLOOR``/``CEIL``/``CAST`` subset.

    Raises ``ValueError`` naming ``where`` and the offending construct, so the loader can point at
    the file and line. Called when a :class:`~mdqm.semantic.models.MeasureSource` is validated, so
    a hand-edited YAML file fails at load rather than in the middle of a query.
    """
    expression = _parse_template(template)
    for node in expression.walk():
        if isinstance(node, exp.Round):
            decimals = node.args.get("decimals")
            if decimals is not None and str(decimals) not in {"0", "None"}:
                raise ValueError(
                    f"{where} must round to a whole number, got ROUND(x, {decimals}): "
                    "round after converting to the canonical unit instead"
                )
        if isinstance(node, _ALLOWED_NODES):
            continue
        if isinstance(node, (exp.Subquery, exp.Query)):
            raise ValueError(
                f"{where} must be one arithmetic expression: a subquery is not allowed"
            )
        if isinstance(node, exp.Anonymous):
            name = str(node.name or "").upper()
            raise ValueError(
                f"{where} calls the function {name!r}, which is not allowed: "
                f"use arithmetic, {', '.join(sorted(ALLOWED_ROUNDING))} and CAST only"
            )
        if isinstance(node, exp.Func):
            name = str(node.key or "").upper()
            raise ValueError(
                f"{where} calls the function {name!r}, which is not allowed: "
                f"use arithmetic, {', '.join(sorted(ALLOWED_ROUNDING))} and CAST only"
            )
        raise ValueError(
            f"{where} uses {type(node).__name__}, which is not allowed: "
            "keep to arithmetic, ROUND/FLOOR/CEIL, CAST and {col}"
        )


# --- explicit rounding -------------------------------------------------------------------
#
# Rounding ``q`` to an integer, expressed as arithmetic so every engine computes the same value.
# ``half_up`` is away from zero on a tie and ``half_even`` goes to the even neighbour: they differ
# exactly at 0.5, 1.5, 2.5 -- the boundary values the synthetic world plants on purpose.


def _num(text: str) -> exp.Expression:
    """A numeric literal; ``0.5`` is exact, so it never pulls a binary float into money."""
    return exp.Literal.number(text)


def _floor(node: exp.Expression) -> exp.Expression:
    return exp.Floor(this=node.copy())


def _half_up(target: exp.Expression) -> exp.Expression:
    """``FLOOR(ABS(q) + 0.5) * SIGN(q)``: nearest integer, a tie going away from zero.

    ``SIGN`` also handles the negative half exactly, because ``ABS`` makes the distance positive
    and ``SIGN`` puts the direction back, so no sign ``CASE`` (or three copies of the operand) is
    needed. ``SIGN(0)`` is 0, which keeps ``q = 0`` at 0.
    """
    return exp.Mul(
        this=_floor(exp.Add(this=exp.Abs(this=target.copy()), expression=_num("0.5"))),
        expression=exp.Sign(this=target.copy()),
    )


def _half_down(target: exp.Expression) -> exp.Expression:
    """``CEIL(ABS(q) - 0.5) * SIGN(q)``: nearest integer, a tie going toward zero."""
    return exp.Mul(
        this=exp.Ceil(this=exp.Sub(this=exp.Abs(this=target.copy()), expression=_num("0.5"))),
        expression=exp.Sign(this=target.copy()),
    )


def _half_even(target: exp.Expression) -> exp.Expression:
    """Rounds to the nearest integer, a tie going to the even neighbour.

    ``frac`` decides the direction; on the exact tie ``MOD(FLOOR(q), 2) = 0`` keeps the floor when
    it is already even and steps up otherwise. ``FLOOR`` also handles negatives, so no CASE on the
    sign is needed here.
    """
    floor_q = _floor(target)
    frac = exp.Sub(this=target.copy(), expression=_floor(target))
    even_floor = exp.EQ(
        this=exp.Mod(this=floor_q.copy(), expression=_num("2")), expression=_num("0")
    )
    return exp.Case(
        ifs=[
            exp.If(this=exp.GT(this=frac.copy(), expression=_num("0.5")), true=floor_q.copy() + 1),
            exp.If(this=exp.LT(this=frac.copy(), expression=_num("0.5")), true=floor_q.copy()),
        ],
        default=exp.Case(
            ifs=[exp.If(this=even_floor, true=floor_q.copy())],
            default=floor_q.copy() + 1,
        ),
    )


_ROUNDERS: Final[dict[RoundingMode, Callable[[exp.Expression], exp.Expression]]] = {
    RoundingMode.HALF_UP: _half_up,
    RoundingMode.HALF_EVEN: _half_even,
    RoundingMode.HALF_DOWN: _half_down,
}


def _rounding_mode(measure: Measure) -> RoundingMode | None:
    rule = measure.canonical.rounding
    return None if rule is None else rule.mode


def _rewrite_round(node: exp.Expression, mode: RoundingMode | None) -> exp.Expression:
    """Replace ``ROUND(x)`` with explicit arithmetic, so no engine picks the tie rule."""
    if not isinstance(node, exp.Round):
        return node
    if mode is None:
        raise UndeclaredRounding(
            "a normalize template calls ROUND but the measure declares no canonical rounding: "
            "add measures.<name>.canonical.rounding = {to: ..., mode: half_up}"
        )
    target = node.this
    if mode is RoundingMode.FLOOR:
        return exp.Floor(this=target.copy())
    if mode is RoundingMode.CEIL:
        return exp.Ceil(this=target.copy())
    return _ROUNDERS[mode](target)


def _cast_divisions(node: exp.Expression) -> exp.Expression:
    """Cast both sides of a division to an exact decimal type.

    PostgreSQL and SQL Server divide integers with truncation, so ``2500 / 5000`` is ``0`` and the
    boundary value would round to ``0``. An exact ``DECIMAL`` keeps the quotient (and the money)
    off binary floating point as well.
    """
    if not isinstance(node, exp.Div):
        return node
    return exp.Div(
        this=exp.Cast(this=node.this.copy(), to=exp.DataType.build(_EXACT_TYPE)),
        expression=exp.Cast(this=node.expression.copy(), to=exp.DataType.build(_EXACT_TYPE)),
    )


def _substitute_placeholder(node: exp.Expression, column: exp.Expression) -> exp.Expression:
    if isinstance(node, exp.Column) and node.name == _PLACEHOLDER:
        return column.copy()
    return node


def _parse_identifier(identifier: str) -> exp.Expression:
    """Parse an already-validated column expression; never build it by string concatenation."""
    parsed = sqlglot.parse_one(identifier)
    if not isinstance(parsed, (exp.Column, exp.Identifier)):
        raise ValueError(f"{identifier!r} is not a column reference")
    return parsed


# --- public rendering --------------------------------------------------------------------


def render_normalize(measure: Measure, column: str, identifier: str, *, dialect: str) -> str | None:
    """The normalisation SQL for one source column, or None when the measure does not list it.

    ``identifier`` is the column reference to normalise (already validated against the catalog);
    ``dialect`` is any sqlglot dialect name. The result is one expression, with rounding rendered
    as arithmetic and integer divisions cast to an exact decimal type.
    """
    source = measure.sources.get(column)
    if source is None:
        return None
    expression = _parse_template(source.normalize)
    real = _parse_identifier(identifier)
    expression = expression.transform(lambda node: _substitute_placeholder(node, real))
    expression = expression.transform(_cast_divisions)
    mode = _rounding_mode(measure)
    expression = expression.transform(lambda node: _rewrite_round(node, mode))
    return expression.sql(dialect=dialect)


@dataclass(frozen=True, slots=True)
class AppliedNormalization:
    """One conversion this query really applied, for the ``applied_fixes`` list in history.

    A query that silently converted a column looks identical to one that did not, so every
    non-identity normalisation is recorded with the template and the rounding it used.
    """

    column: str
    template: str
    mode: RoundingMode | None
    sql: str
    identity: bool = False


def normalizations_for(
    measure: Measure, identifiers: dict[str, str], *, dialect: str
) -> tuple[AppliedNormalization, ...]:
    """Every normalisation this measure applies to ``identifiers`` (column -> column reference).

    Identity templates are included with ``identity=True`` so a caller can report "nothing was
    converted here" explicitly instead of leaving it ambiguous.
    """
    mode = _rounding_mode(measure)
    applied: list[AppliedNormalization] = []
    for column, identifier in identifiers.items():
        source = measure.sources.get(column)
        if source is None:
            continue
        sql = render_normalize(measure, column, identifier, dialect=dialect)
        if sql is None:  # unreachable: the source was looked up just above
            continue
        applied.append(
            AppliedNormalization(
                column=column,
                template=source.normalize,
                mode=mode,
                sql=sql,
                identity=source.normalize.strip() == "{col}",
            )
        )
    return tuple(applied)


# --- the same rules in Python ------------------------------------------------------------


def _round_fraction(value: Fraction, mode: RoundingMode) -> int:
    """Round an exact fraction to an integer under ``mode``, using integer arithmetic only."""
    numerator, denominator = value.numerator, value.denominator
    quotient, remainder = divmod(numerator, denominator)  # floor division, remainder >= 0
    twice = 2 * remainder
    if mode is RoundingMode.FLOOR:
        return quotient
    if mode is RoundingMode.CEIL:
        return quotient + 1 if remainder else quotient
    if twice > denominator:
        return quotient + 1
    if twice < denominator:
        return quotient
    # Exactly half. Away from zero on a negative number means the floor, hence the two branches.
    if mode is RoundingMode.HALF_UP:
        return quotient + 1 if numerator > 0 else quotient
    if mode is RoundingMode.HALF_DOWN:
        return quotient + 1 if numerator < 0 else quotient
    return quotient if quotient % 2 == 0 else quotient + 1  # half even


def round_to_multiple(
    value: int | Decimal, step: int, mode: RoundingMode = RoundingMode.HALF_UP
) -> Decimal:
    """Round ``value`` to the nearest multiple of ``step`` under ``mode``, exactly.

    The reference implementation for the rendered SQL and for callers that compute locally.
    ``step`` is the canonical granularity (5000 for the credit limit). Exact integer fractions mean
    no binary float and no dependence on Python's ``round()``, which is half-even.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    fraction = Fraction(Decimal(value)) / step
    return Decimal(_round_fraction(fraction, mode) * step)


# --- result sanity check -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MultiplicityReport:
    """What a sanity check on normalised values found. Reports; it never raises.

    A value that is not a multiple of the canonical step means either the wrong template or an
    exception the data team planted deliberately, and both are facts a human must see rather than
    a reason to fail a query.
    """

    step: int
    checked: int
    violations: tuple[Decimal, ...]

    @property
    def violation_count(self) -> int:
        """How many checked values are not multiples of the canonical step."""
        return len(self.violations)

    @property
    def ok(self) -> bool:
        """True when every checked value already sat on the canonical grid."""
        return not self.violations

    @property
    def violation_rate(self) -> float:
        """Share of checked values that broke the grid; 0.0 when nothing was checked."""
        return len(self.violations) / self.checked if self.checked else 0.0


def check_multiples(
    values: Iterable[int | Decimal | None], step: int, *, limit: int = 10
) -> MultiplicityReport:
    """Count normalised ``values`` that are not multiples of ``step``, keeping a sample.

    NULLs are skipped (a missing value is not a wrong one) and at most ``limit`` offending values
    are kept, so a badly broken template cannot flood a report.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    checked = 0
    violations: list[Decimal] = []
    for value in values:
        if value is None:
            continue
        checked += 1
        exact = Decimal(value)
        if exact % step != 0 and len(violations) < limit:
            violations.append(exact)
    return MultiplicityReport(step=step, checked=checked, violations=tuple(violations))
