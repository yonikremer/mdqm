"""Stage D7: join candidates measured from metadata *and* data.

DESIGN.md section 10 lists D7 as "name similarity, types, containment, uniqueness; format
conversions". This module is that, and its one rule is the inverse of a guess: **a wrong join is
worse than no join**. So a pair is proposed only when its value sets really line up, and the
proposal carries the containment ratio and the count of keys that did not match -- never a bare
"these look similar".

Four measurements, all fed by the same bounded read:

* **name similarity** (:func:`name_similarity`) -- ``cust_id``/``cust_no``/``client_no`` share a
  stem once the generic key tokens (``id``, ``no``, ``code``) are dropped. It orders proposals and
  raises confidence; it never admits one by itself, because ``account_id`` and ``customer_id`` are
  similar names on two unrelated keys.
* **type compatibility** -- asked of the *values* as well as the declared type: a column whose
  values are all digits can key an integer column, and one holding letters cannot.
* **containment** (:class:`Containment`) -- of the distinct values on each side, how many exist on
  the other side; reported in both directions, as an overlap (Jaccard), and as unmatched *distinct
  keys* plus unmatched *rows*, which is the orphan count a human has to see.
* **format conversion** (:func:`conversions`) -- ``cust_id`` (integer) against ``client_no``
  (``VARCHAR(10)`` holding ten zero-padded digits) is the case AGENTS.md section 6 plants. The
  conversion is measured, not assumed: every applicable candidate is tried, the one with the best
  overlap wins, and the winner is emitted as the ``on`` template of an
  :class:`~mdqm.semantic.models.ApprovedJoin`, so approving a proposal is a copy, not a rewrite.

Where the two sides cannot be compared honestly, nothing is proposed: a value set that hit the
distinct cap is truncated, so its containment is unknowable and the pair is left out with the
reason in ``skipped`` (principle 10: nothing silent). A declared foreign key whose containment is
total is marked ``auto_approvable`` (DESIGN.md section 10's "auto" gate); every other candidate is
a proposal for a human, submitted through :class:`~mdqm.review.queue.ReviewQueue`. This stage never
writes ``curated/`` and never writes a ``generated/`` section: a join is a human gate, and the
queue is where that gate is.

**How this relates to D9** (:mod:`mdqm.pipeline.stages.mining`). D9 learns join *edges from past
queries*: what people already wrote, ``LPAD`` conversions included. This stage learns joins *from
the data*: what the values can support, queried or never queried. The two must not share logic and
do not: D9 parses SQL and counts fingerprints, D7 reads values and measures sets. When both reach
the same edge, the queue's content-addressed id makes it one proposal, and a conversion D9 saw in
history raises the prior of the matching D7 hypothesis -- ranking, never approval.

**Cost.** A join key's values cannot be read without reading values, so this is a heavy stage: one
bounded ``GROUP BY`` per *candidate key column* (never per table, never a full scan into the
agent), run inside a quiet window and behind the per-source gate. Candidate columns are the ones
whose name looks like a key or that carry a declared primary/unique/foreign-key role; everything
else is skipped and reported.

Deterministic: values are read in sorted order, the caps and thresholds are declared inputs, no
clock is read (``exec_ts`` is injected), and candidates come out in a total order.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Final

import pyarrow as pa
from pydantic import Field
from sqlglot import exp

from mdqm.adapters.base import NotSupported, SourceAdapter
from mdqm.models.base import MDQMModel
from mdqm.models.catalog import Column, NormalizedType, Table
from mdqm.models.fragment import Fragment
from mdqm.models.limits import Limits
from mdqm.pipeline.core import (
    PipelineContext,
    StageReport,
    StageStatus,
    hash_inputs,
    next_quiet_start,
    quiet_allows,
)
from mdqm.pipeline.stages.profiling import QuietWindowClosed
from mdqm.review.queue import Proposal, ProposalKind, ReviewQueue, SubmitAction, proposal_id
from mdqm.semantic.models import ApprovedJoin, JoinCardinality, Provenance

__all__ = [
    "DEFAULT_MAX_PAIRS",
    "DEFAULT_MAX_SIZE_RATIO",
    "DEFAULT_MIN_CONTAINMENT",
    "DEFAULT_MIN_JACCARD",
    "DEFAULT_VALUE_CAP",
    "STAGE_VERSION",
    "Containment",
    "ConversionKind",
    "FormatConversion",
    "JoinCandidate",
    "JoinMeasurement",
    "JoinStage",
    "ValueFamily",
    "ValueSet",
    "conversions",
    "distinct_query",
    "measure",
    "name_similarity",
    "qualified",
]

logger = logging.getLogger(__name__)

#: Bumped when the measurement itself changes, so a rerun refreshes the candidates.
STAGE_VERSION: Final[int] = 1

#: Distinct values read per column. A join key is small next to its table; a column with more
#: distinct values than this cannot be compared set-wise, so it is reported and left alone.
DEFAULT_VALUE_CAP: Final[int] = 20_000

#: The best overlap accepted as a proposal. One side may be a subset (an orphan-bearing child
#: table), so the *larger* of the two containments has to reach this.
DEFAULT_MIN_CONTAINMENT: Final[float] = 0.95

#: Symmetric overlap floor. This is what rejects two unrelated integer columns whose ranges merely
#: intersect: half of each set matching on the other side scores about 0.33, not 0.95.
DEFAULT_MIN_JACCARD: Final[float] = 0.5

#: How far apart the two distinct counts may be. A three-value code column is fully contained in
#: any id column, and that is not a join: it is a coincidence the ratio catches.
DEFAULT_MAX_SIZE_RATIO: Final[float] = 10.0

#: Candidate pairs measured in one pass; above it the rest is reported as unmeasured, never dropped
#: silently.
DEFAULT_MAX_PAIRS: Final[int] = 20_000

#: A catalog identifier we are willing to put in SQL. The catalog is the source of the name, but a
#: source that reports a wild name must not reach the query text.
_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")

#: Tokens that mark a column as a key but say nothing about *which* entity it identifies. Dropping
#: them is what lets ``cust_id`` and ``client_no`` be compared on ``cust`` against ``client``.
_GENERIC_KEY_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "id",
        "ident",
        "key",
        "no",
        "nr",
        "nbr",
        "num",
        "number",
        "code",
        "ref",
        "uid",
        "uuid",
        "pk",
        "fk",
    }
)

#: A column name carrying one of these tokens is a candidate key.
_KEY_TOKEN: Final[re.Pattern[str]] = re.compile(
    r"(^|_)(id|ident|key|no|nr|nbr|num|number|code|ref|uid|uuid|pk|fk)($|_)"
)

#: The normalised types a join key can have. Dates and timestamps join on ranges, not on keys.
_KEY_TYPES: Final[frozenset[NormalizedType]] = frozenset(
    {NormalizedType.INTEGER, NormalizedType.DECIMAL, NormalizedType.STRING, NormalizedType.UUID}
)

#: ASCII digits: the shape a zero pad or a numeric cast can be read from.
_DIGITS: Final[re.Pattern[str]] = re.compile(r"^[0-9]+$")

#: A constant non-digit prefix followed by digits, as in ``CUST-000123``.
_PREFIXED: Final[re.Pattern[str]] = re.compile(r"^([^0-9]+)([0-9]+)$")


class ConversionKind(StrEnum):
    """The format conversion a pair of key columns needs, or :attr:`IDENTITY` for none.

    ``ZERO_PAD`` and ``NUMERIC`` compare the same values -- ``int("0000000001")`` is ``1`` -- but
    say different things to a reader: the first that one side is a fixed-width padded text key, the
    second that the widths simply differ. The emitted ``on`` template follows the kind.
    """

    IDENTITY = "identity"
    ZERO_PAD = "zero_pad"
    NUMERIC = "numeric"
    CASE_FOLD = "case_fold"
    PREFIX_STRIP = "prefix_strip"


class ValueFamily(StrEnum):
    """What two columns can be compared as. Measured from the values, not from the declared type."""

    DIGITS = "digits"
    TEXT = "text"


def qualified(source: str, table: str, column: str) -> str:
    """``crm.customers_crm.cust_id``: the one spelling a join side has in this module."""
    return f"{source}.{table}.{column}"


def _all_digits(values: Sequence[str]) -> bool:
    """True when every value is ASCII digits (an empty sequence is not a digit set)."""
    return bool(values) and all(_DIGITS.match(value) is not None for value in values)


def _numeric_text(value: str) -> str:
    """The canonical text of a digits value, so ``0000000001`` and ``1`` compare equal.

    A value that is not digits is returned unchanged: conversions are only offered for digit sets,
    and a caller that gets here anyway sees "not matched" rather than an exception.
    """
    if _DIGITS.match(value) is None:
        return value
    try:
        return str(int(value))
    except ValueError:  # more digits than int() will parse: compare it as the text it is
        return value


# --- one measured column ----------------------------------------------------------------------


class ValueSet(MDQMModel):
    """The distinct values of one candidate key column, with the row count behind each.

    ``truncated`` is the honest flag: a set that hit the cap is a prefix of the column's values, so
    containment computed from it is unknown rather than merely approximate, and no pair is proposed
    from it.
    """

    source: str = Field(min_length=1)
    table: str = Field(min_length=1)
    column: str = Field(min_length=1)
    normalized_type: NormalizedType
    values: tuple[str, ...] = ()
    counts: tuple[int, ...] = ()
    truncated: bool = False
    declared_unique: bool = False

    @property
    def label(self) -> str:
        """``crm.customers_crm.cust_id``: how a human names this side of a join."""
        return qualified(self.source, self.table, self.column)

    @property
    def distinct(self) -> int:
        """How many distinct values were read."""
        return len(self.values)

    @property
    def rows(self) -> int:
        """How many source rows those values cover."""
        return sum(self.counts)

    @property
    def family(self) -> ValueFamily:
        """``digits`` when every value is ASCII digits, else ``text``."""
        return ValueFamily.DIGITS if _all_digits(self.values) else ValueFamily.TEXT

    @property
    def widths(self) -> tuple[int, ...]:
        """The distinct value lengths when every value is digits, else empty."""
        if self.family is not ValueFamily.DIGITS:
            return ()
        return tuple(sorted({len(value) for value in self.values}))


class FormatConversion(MDQMModel):
    """One way to make two key columns comparable, and the ``on`` template it becomes.

    ``on`` is an :class:`~mdqm.semantic.models.ApprovedJoin` template: ``{left}`` and ``{right}``
    are the two sides' column references, and the template is written in the dialect of the
    converted side, so its ``CAST`` type really exists there.
    """

    kind: ConversionKind
    on: str = Field(min_length=1)
    converted: str | None = Field(
        default=None, description="Which side carries the conversion: 'left', 'right' or 'both'."
    )
    width: int | None = Field(default=None, ge=1)
    prefixed_side: str | None = Field(
        default=None, description="For a prefix strip: the side whose values carry the prefix."
    )
    prefix_length: int | None = Field(default=None, ge=1)
    description: str = ""

    def convert(self, value: str, side: str) -> str:
        """The comparable key for one raw value of ``side``, as text.

        Both sides land in one space, which is the point: a padded key and a plain integer become
        the same text, and only the side that really carries a prefix is shortened -- stripping it
        from both would turn the short side into an empty string and match nothing.
        """
        if self.prefix_length is not None and self.prefixed_side is not None:
            if side == self.prefixed_side:
                return _numeric_text(value[self.prefix_length :])
            return _numeric_text(value)
        if self.kind is ConversionKind.CASE_FOLD:
            return value.upper()
        if self.kind in (ConversionKind.ZERO_PAD, ConversionKind.NUMERIC):
            return _numeric_text(value)
        return value


class Containment(MDQMModel):
    """What one candidate conversion actually achieved, in both directions and in rows.

    ``unmatched_left_rows`` and ``unmatched_right_rows`` are the orphan counts: the source rows
    whose key does not exist on the other side. Those are the numbers a human needs before
    approving a join, and the numbers that make a "this join dropped rows" surprise impossible.
    """

    conversion: FormatConversion
    matched_distinct: int = Field(ge=0)
    left_distinct: int = Field(ge=0)
    right_distinct: int = Field(ge=0)
    unmatched_left_distinct: int = Field(ge=0)
    unmatched_right_distinct: int = Field(ge=0)
    unmatched_left_rows: int = Field(ge=0)
    unmatched_right_rows: int = Field(ge=0)
    unmatched_left_examples: tuple[str, ...] = Field(
        default=(), description="A few of the unmatched keys themselves: the orphans, named."
    )
    unmatched_right_examples: tuple[str, ...] = Field(
        default=(), description="A few of the unmatched keys themselves: the orphans, named."
    )
    containment_left: float = Field(ge=0.0, le=1.0)
    containment_right: float = Field(ge=0.0, le=1.0)
    jaccard: float = Field(ge=0.0, le=1.0)

    @property
    def best(self) -> float:
        """The larger of the two containments: how much of the bigger side is covered."""
        return max(self.containment_left, self.containment_right)

    @property
    def total(self) -> bool:
        """True when every key on both sides matched."""
        return self.unmatched_left_distinct == 0 and self.unmatched_right_distinct == 0


def _ratio(part: int, whole: int) -> float:
    """``part / whole``, rounded so two runs produce the same text, and 0 for an empty whole."""
    return round(part / whole, 6) if whole else 0.0


def _examples(keys: Sequence[str], limit: int = 3) -> tuple[str, ...]:
    """The first few unmatched keys, sorted: enough for a reader to recognise the orphans."""
    return tuple(sorted(keys)[:limit])


def measure(left: ValueSet, right: ValueSet, conversion: FormatConversion) -> Containment:
    """Compare both value sets under ``conversion`` and report the overlap in both directions."""
    left_keys = {
        conversion.convert(value, "left"): count
        for value, count in zip(left.values, left.counts, strict=True)
    }
    right_keys = {
        conversion.convert(value, "right"): count
        for value, count in zip(right.values, right.counts, strict=True)
    }
    matched = left_keys.keys() & right_keys.keys()
    union = left_keys.keys() | right_keys.keys()
    unmatched_left = [key for key in left_keys if key not in matched]
    unmatched_right = [key for key in right_keys if key not in matched]
    return Containment(
        conversion=conversion,
        matched_distinct=len(matched),
        left_distinct=len(left_keys),
        right_distinct=len(right_keys),
        unmatched_left_distinct=len(unmatched_left),
        unmatched_right_distinct=len(unmatched_right),
        unmatched_left_rows=sum(left_keys[key] for key in unmatched_left),
        unmatched_right_rows=sum(right_keys[key] for key in unmatched_right),
        unmatched_left_examples=_examples(unmatched_left),
        unmatched_right_examples=_examples(unmatched_right),
        containment_left=_ratio(len(matched), len(left_keys)),
        containment_right=_ratio(len(matched), len(right_keys)),
        jaccard=_ratio(len(matched), len(union)),
    )


# --- conversions ------------------------------------------------------------------------------


def _cast_type(data_type: str, dialect: str) -> str:
    """The dialect's own spelling of a type, so an ``on`` template is valid where it will run.

    sqlglot knows the mappings (``VARCHAR`` is ``VARCHAR2`` on Oracle, ``TEXT`` on DuckDB), so no
    dialect table lives here. An unknown dialect falls back to the generic spelling, which parses.
    """
    try:
        return exp.DataType.build(data_type).sql(dialect=dialect)
    except (ValueError, KeyError):
        logger.debug("no dialect %r: the cast keeps its generic spelling", dialect)
        return data_type


def _constant_prefix(values: Sequence[str]) -> str | None:
    """The one non-digit prefix every value shares, or ``None`` when there is not exactly one."""
    prefixes = set()
    for value in values:
        match = _PREFIXED.match(value)
        if match is None:
            return None
        prefixes.add(match.group(1))
        if len(prefixes) > 1:
            return None
    return next(iter(prefixes)) if prefixes else None


def conversions(
    left: ValueSet, right: ValueSet, *, dialect: str = ""
) -> tuple[FormatConversion, ...]:
    """Every conversion that could make these two columns comparable, cheapest first.

    Applicability is decided on the *values*: a zero pad needs one side at a single uniform width
    and the other narrow enough to fit it, a numeric cast needs digits on both sides, a case fold
    needs text, and a prefix strip needs one constant non-digit prefix. Identity is always offered,
    so a pair that needs nothing is never saddled with a conversion.
    """
    offers = [
        FormatConversion(
            kind=ConversionKind.IDENTITY,
            on="{left} = {right}",
            converted=None,
            description="the values already compare equal with no conversion",
        )
    ]
    left_widths = left.widths
    right_widths = right.widths
    if left_widths and right_widths:
        # The padded target is the side with a single uniform width: that is the side a fixed-width
        # key really has. The other side is padded up to it, if it fits.
        for narrow, wide, side in ((left, right, "left"), (right, left, "right")):
            if len(wide.widths) != 1 or not narrow.widths:
                continue
            width = wide.widths[0]
            if max(narrow.widths) >= width:
                continue
            target = "{left}" if side == "left" else "{right}"
            other = "{right}" if side == "left" else "{left}"
            offers.append(
                FormatConversion(
                    kind=ConversionKind.ZERO_PAD,
                    on=(
                        f"{target} = "
                        f"LPAD(CAST({other} AS {_cast_type('VARCHAR', dialect)}), "
                        f"{width}, '0')"
                    ),
                    converted="right" if side == "left" else "left",
                    width=width,
                    description=(
                        f"{'right' if side == 'left' else 'left'} is zero padded to {width} "
                        f"characters to match {side}"
                    ),
                )
            )
        if left_widths != right_widths:
            number = _cast_type("DECIMAL(38,0)", dialect)
            offers.append(
                FormatConversion(
                    kind=ConversionKind.NUMERIC,
                    on=f"CAST({{left}} AS {number}) = CAST({{right}} AS {number})",
                    converted="both",
                    description=(
                        "the digit strings have differing widths, so both sides are compared as "
                        "numbers"
                    ),
                )
            )
    if left.family is ValueFamily.TEXT and right.family is ValueFamily.TEXT:
        offers.append(
            FormatConversion(
                kind=ConversionKind.CASE_FOLD,
                on="UPPER({left}) = UPPER({right})",
                converted="both",
                description="the two keys differ only by letter case",
            )
        )
    for side, values, source_label in (
        ("left", left.values, left.label),
        ("right", right.values, right.label),
    ):
        prefix = _constant_prefix(values)
        if prefix is None:
            continue
        # DECIMAL(38,0), not BIGINT: a padded text key can hold more digits than a 32-bit int, and
        # sqlglot maps BIGINT to a 32-bit INT on Oracle.
        number = _cast_type("DECIMAL(38,0)", dialect)
        target = "{left}" if side == "left" else "{right}"
        other = "{right}" if side == "left" else "{left}"
        offers.append(
            FormatConversion(
                kind=ConversionKind.PREFIX_STRIP,
                on=(
                    f"CAST({other} AS {number}) = "
                    f"CAST(SUBSTR({target}, {len(prefix) + 1}) AS {number})"
                ),
                converted=side,
                prefixed_side=side,
                prefix_length=len(prefix),
                description=f"{source_label} carries the constant prefix {prefix!r}",
            )
        )
    return tuple(offers)


def best_conversion(
    left: ValueSet, right: ValueSet, *, dialect: str = ""
) -> tuple[Containment, tuple[Containment, ...]]:
    """Measure every applicable conversion and return the best one plus all measurements.

    Ties go to the earlier kind in :data:`_KIND_ORDER`, so an un-converted join wins against an
    equally good converted one: fewer moving parts in the query and in the approval.
    """
    measured = [measure(left, right, offer) for offer in conversions(left, right, dialect=dialect)]
    order = {kind: index for index, kind in enumerate(ConversionKind)}
    best = max(measured, key=lambda item: (item.jaccard, item.best, -order[item.conversion.kind]))
    return best, tuple(measured)

    best = max(measured, key=_conversion_rank)


def _stems(name: str) -> tuple[str, ...]:
    """The name's tokens with the generic key tokens removed: ``client_no`` -> ``('client',)``."""
    tokens = [token for token in re.split(r"[^A-Za-z0-9]+|_", name.lower()) if token]
    stems = tuple(token for token in tokens if token not in _GENERIC_KEY_TOKENS)
    return stems or tuple(tokens)


def name_similarity(left: str, right: str) -> float:
    """How much two column names suggest the same entity, in ``[0, 1]``.

    Token stems first: ``cust_id`` and ``cust_no`` both reduce to ``cust`` and score 1.0, while
    ``cust_id`` and ``client_no`` score the letter overlap of ``cust`` against ``client``. Only the
    ordering and the confidence use this -- ``account_id``/``customer_id`` are similar names on
    different keys, so a name never admits a join.
    """
    left_stems = _stems(left)
    right_stems = _stems(right)
    if left_stems == right_stems:
        return 1.0
    left_text = "".join(left_stems)
    right_text = "".join(right_stems)
    common = sum(1 for token in left_stems if token in right_stems)
    token_score = common / max(len(left_stems), len(right_stems))
    letters = sum(1 for token in left_stems for other in right_stems if token[:3] == other[:3])
    stem_score = min(1.0, letters / max(len(left_stems), len(right_stems)))
    # The character ratio is the weakest signal, so it is capped below what a shared stem scores.
    char_score = difflib.SequenceMatcher(None, left_text, right_text).ratio()
    return round(max(token_score, stem_score, 0.8 * char_score), 6)


# --- one proposal -----------------------------------------------------------------------------


class JoinCandidate(MDQMModel):
    """A join between two columns that the data supports, with the evidence that supports it.

    ``evidence`` is what a human reads to decide, so it holds the numbers -- containment both ways,
    unmatched keys and rows, the conversion and the ``on`` clause -- and never a bare adjective.
    """

    name: str = Field(min_length=1)
    left: str = Field(min_length=1)
    right: str = Field(min_length=1)
    left_table: str = Field(min_length=1)
    right_table: str = Field(min_length=1)
    containment: Containment
    name_similarity: float = Field(ge=0.0, le=1.0)
    uniqueness: tuple[bool, bool] = Field(
        default=(False, False), description="(left, right): is each side a declared unique key?"
    )
    cardinality: JoinCardinality = JoinCardinality.MANY_TO_MANY
    declared_fk: bool = False
    auto_approvable: bool = Field(
        default=False,
        description="A declared foreign key with total containment: DESIGN section 10's auto gate.",
    )
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: tuple[str, ...] = ()
    tried: tuple[Containment, ...] = Field(
        default=(), description="Every conversion measured, best first: why the winner won."
    )

    def as_approved_join(self) -> ApprovedJoin:
        """The curated-layer entry this proposal becomes when a human approves it.

        Building the real model is the proof that a proposal is expressible as an
        ``approved_joins`` entry: a bad ``on`` template fails validation here, not at query time.
        """
        return ApprovedJoin(
            name=self.name,
            left=self.left,
            right=self.right,
            on=self.containment.conversion.on,
            cardinality=self.cardinality,
            provenance=Provenance(
                detected_by="D7",
                evidence=self.evidence,
                match_rate=self.containment.jaccard,
            ),
        )


def _cardinality(left_unique: bool, right_unique: bool) -> JoinCardinality:
    """The join shape the two uniqueness facts prove; many_to_many when neither is proven."""
    if left_unique and right_unique:
        return JoinCardinality.ONE_TO_ONE
    if right_unique:
        return JoinCardinality.MANY_TO_ONE
    if left_unique:
        return JoinCardinality.ONE_TO_MANY
    return JoinCardinality.MANY_TO_MANY


def _join_name(left: ValueSet, right: ValueSet) -> str:
    """A stable, readable name for the proposed join: the two sides, source and table included."""
    return f"{left.label}__{right.label}".replace(".", "_")


# --- the measurement --------------------------------------------------------------------------


class JoinMeasurement(MDQMModel):
    """Everything one D7 pass measured: the candidates, the columns read, and what it left out."""

    candidates: tuple[JoinCandidate, ...] = ()
    columns: tuple[ValueSet, ...] = ()
    counts: dict[str, int] = Field(default_factory=dict)
    skipped: tuple[str, ...] = ()

    def candidate(self, left: str, right: str) -> JoinCandidate | None:
        """The candidate for an ordered pair of labels, or ``None`` when it was not proposed."""
        return next(
            (item for item in self.candidates if item.left == left and item.right == right), None
        )


def distinct_query(table: Table, column: Column, *, dialect: str, cap: int) -> str:
    """The one bounded statement that reads a candidate key's distinct values, rendered by sqlglot.

    ``GROUP BY`` over the value, ordered so a capped read is still deterministic, and limited to one
    row past the cap so truncation is detectable rather than assumed. Identifiers come from the
    catalog and are quoted by sqlglot, and both must be plain identifiers before that.
    """
    if cap < 1:
        raise ValueError(f"cap must be at least 1, got {cap}")
    if not _IDENTIFIER.match(table.name) or (
        table.schema_name is not None and not _IDENTIFIER.match(table.schema_name)
    ):
        raise ValueError(
            f"refusing to read {table.name!r}: the catalog's table name is not a plain identifier"
        )
    if not _IDENTIFIER.match(column.name):
        raise ValueError(
            f"refusing to read {table.name}.{column.name!r}: the column name is not an identifier"
        )
    source = exp.Table(
        this=exp.to_identifier(table.name, quoted=True),
        db=(
            exp.to_identifier(table.schema_name, quoted=True)
            if table.schema_name is not None
            else None
        ),
    )
    value = exp.Cast(this=exp.column(column.name, quoted=True), to=exp.DataType.build("VARCHAR"))
    statement = (
        exp.select(
            exp.alias_(value.copy(), "value"),
            exp.alias_(exp.Count(this=exp.Star()), "rows"),
        )
        .from_(source)
        .group_by(value.copy())
        .order_by(value.copy())
        .limit(cap + 1)
    )
    try:
        return statement.sql(dialect=dialect)
    except ValueError:
        # sqlglot raises ValueError for a name it has no dialect for (an API or ES source).
        raise NotSupported("sql", f"no SQL dialect {dialect!r} for a distinct read") from None


def _rows_of(batch: pa.RecordBatch) -> list[Mapping[str, Any]]:
    """One batch as row mappings; the same shape the profiling stage reads."""
    data = batch.to_pydict()
    names = list(data)
    return [dict(zip(names, values, strict=True)) for values in zip(*data.values(), strict=True)]


def _declared_unique(table: Table, column: str) -> bool:
    """True when the catalog proves this column is unique: the primary key or a unique index."""
    if table.primary_key == (column,):
        return True
    return any(index.unique and index.columns == (column,) for index in table.indexes)


def _declares_fk(table: Table, column: str, ref_table: str, ref_column: str) -> bool:
    """True when the catalog declares that ``table.column`` references ``ref_table.ref_column``."""
    return any(
        fk.columns == (column,) and fk.ref_columns == (ref_column,) and fk.ref_table == ref_table
        for fk in table.foreign_keys
    )


def _comparable(left: ValueSet, right: ValueSet) -> bool:
    """True when some conversion could compare these two value shapes at all.

    Digits with digits, text with text, and -- because a constant prefix is a conversion --
    digits with a text key that is a prefix plus digits. Everything else is not a key pair.
    """
    if left.family is right.family:
        return True
    text_side = left if left.family is ValueFamily.TEXT else right
    return _constant_prefix(text_side.values) is not None


def _is_candidate_column(table: Table, column: Column) -> bool:
    """True for a column a join could use: a key-shaped name, a type that can key, a key role."""
    if column.normalized_type not in _KEY_TYPES:
        return False
    if _KEY_TOKEN.search(column.name.lower()):
        return True
    if _declared_unique(table, column.name):
        return True
    return any(fk.columns == (column.name,) for fk in table.foreign_keys)


# --- sort keys --------------------------------------------------------------------------------
#
# Module-level functions rather than inline keys: mypy --strict wants a return type, and a named
# key is also the place where "what makes this order total" is written down.


def _by_name(item: Table | Column) -> str:
    """Order tables and columns by name, so a read is issued in the same order on every run."""
    return item.name


def _by_label(item: ValueSet) -> str:
    """Order value sets by their fully qualified label."""
    return item.label


def _conversion_rank(item: Containment) -> tuple[float, float, int]:
    """Order measured conversions: best overlap first, then the cheapest kind (identity wins)."""
    kinds = tuple(ConversionKind)
    return (item.jaccard, item.best, -kinds.index(item.conversion.kind))


def _overlap_rank(item: Containment) -> tuple[float, str]:
    """Order the measurements kept on a proposal: strongest overlap first."""
    return (-item.jaccard, item.conversion.kind.value)


def _candidate_rank(item: JoinCandidate) -> tuple[float, str, str]:
    """Order proposals: highest score first, then by label so a tie is still a total order."""
    return (-item.score, item.left, item.right)


class JoinStage:
    """Stage D7. Construct with the adapters to read; ``measure`` does the work.

    ``queue`` is where proposals go when it is given. Without one the stage still measures and
    reports -- a caller that only wants the numbers (a test, a dry run) does not need a store.
    ``tables`` narrows every source to a set of table names, the same escape hatch D5 has.
    """

    name: str = "D7"
    heavy: bool = True

    def __init__(
        self,
        adapters: Mapping[str, SourceAdapter],
        *,
        queue: ReviewQueue | None = None,
        value_cap: int = DEFAULT_VALUE_CAP,
        min_containment: float = DEFAULT_MIN_CONTAINMENT,
        min_jaccard: float = DEFAULT_MIN_JACCARD,
        max_size_ratio: float = DEFAULT_MAX_SIZE_RATIO,
        max_pairs: int = DEFAULT_MAX_PAIRS,
        timeout_s: float = 30.0,
        max_bytes: int = 8 * 1024 * 1024,
        tables: Sequence[str] | None = None,
    ) -> None:
        if value_cap < 1:
            raise ValueError(f"value_cap must be at least 1, got {value_cap}")
        if not 0.0 < min_containment <= 1.0:
            raise ValueError(f"min_containment must be in (0, 1], got {min_containment}")
        if not 0.0 < min_jaccard <= 1.0:
            raise ValueError(f"min_jaccard must be in (0, 1], got {min_jaccard}")
        if max_size_ratio < 1.0:
            raise ValueError(f"max_size_ratio must be at least 1, got {max_size_ratio}")
        if max_pairs < 1:
            raise ValueError(f"max_pairs must be at least 1, got {max_pairs}")
        self._adapters = dict(adapters)
        self._queue = queue
        self._value_cap = value_cap
        self._min_containment = min_containment
        self._min_jaccard = min_jaccard
        self._max_size_ratio = max_size_ratio
        self._max_pairs = max_pairs
        self._timeout_s = timeout_s
        self._max_bytes = max_bytes
        self._tables = tuple(tables) if tables is not None else None

    @property
    def sources(self) -> tuple[str, ...]:
        """The sources this instance reads, sorted (what the runner gates and filters on)."""
        return tuple(sorted(self._adapters))

    def inputs_hash(self, ctx: PipelineContext) -> str:
        """sha256 over the declared inputs: sources, tables, thresholds, caps and the day.

        The **date** of ``exec_ts`` is deliberate, as in D5: values change every day, so a nightly
        pass must re-measure, while two passes on one day measure the same day and the second one
        skips. Nothing reads the data here, so the hash stays a pure function of declared inputs.
        """
        return hash_inputs(
            self.name,
            STAGE_VERSION,
            sorted(self._adapters),
            self._tables,
            ctx.source,
            ctx.exec_ts.date().isoformat(),
            self._value_cap,
            self._min_containment,
            self._min_jaccard,
            self._max_size_ratio,
            self._max_pairs,
        )

    def run(self, ctx: PipelineContext) -> StageReport:
        """Measure every selected source and propose each candidate through the review queue."""
        if not ctx.force and not quiet_allows(ctx.quiet_windows, ctx.exec_ts):
            raise QuietWindowClosed(self._refusal(ctx))
        measurement = self.measure(ctx)
        counts = dict(measurement.counts)
        skipped = list(measurement.skipped)
        fixes: list[str] = []
        submitted = {
            action.value: 0 for action in SubmitAction
        }  # every outcome is counted, none dropped
        if self._queue is None:
            skipped.append(
                f"{len(measurement.candidates)} join candidate(s) measured but not proposed: "
                "no review queue was given"
            )
        else:
            for candidate in measurement.candidates:
                result = self._queue.submit(self._proposal(candidate, ctx))
                submitted[result.action.value] = submitted.get(result.action.value, 0) + 1
        for action, count in submitted.items():
            counts[f"proposals_{action}"] = count
        counts["auto_approvable"] = sum(
            1 for item in measurement.candidates if item.auto_approvable
        )
        return StageReport(
            stage=self.name,
            status=StageStatus.RAN,
            counts=counts,
            skipped=tuple(skipped),
            applied_fixes=tuple(fixes),
        )

    # -- measurement -----------------------------------------------------------------------

    def measure(self, ctx: PipelineContext) -> JoinMeasurement:
        """Read the candidate key columns of every selected source and measure the pairs."""
        selected = (ctx.source,) if ctx.source is not None else self.sources
        missing = [source for source in selected if source not in self._adapters]
        if missing:
            raise ValueError(f"no adapter for source(s) {', '.join(missing)}")
        counts: dict[str, int] = {
            "sources": 0,
            "tables": 0,
            "columns_read": 0,
            "columns_skipped": 0,
            "pairs_considered": 0,
            "pairs_unmeasurable": 0,
            "pairs_rejected": 0,
            "candidates": 0,
            "conversions": 0,
        }
        skipped: list[str] = []
        columns: list[ValueSet] = []
        dialects: dict[str, str] = {}
        catalogs: dict[tuple[str, str], Table] = {}
        for source in selected:
            adapter = self._adapters[source]
            try:
                dialect = adapter.dialect()
            except NotSupported:
                skipped.append(f"{source}: no SQL dialect, so no value set can be read")
                continue
            dialects[source] = dialect
            counts["sources"] += 1
            catalog = adapter.extract_catalog()
            wanted = None if self._tables is None else set(self._tables)
            for table in sorted(catalog.tables, key=_by_name):
                if wanted is not None and table.name not in wanted:
                    continue
                counts["tables"] += 1
                catalogs[(source, table.name)] = table
                for column in sorted(table.columns, key=_by_name):
                    if not _is_candidate_column(table, column):
                        counts["columns_skipped"] += 1
                        continue
                    values, reason = self._read_values(source, adapter, table, column, ctx, dialect)
                    if values is None:
                        counts["columns_skipped"] += 1
                        skipped.append(reason)
                        continue
                    columns.append(values)
                    counts["columns_read"] += 1
        candidates, rejected, unmeasurable, capped = self._pairs(columns, dialects, catalogs)
        counts["pairs_considered"] = rejected + unmeasurable + len(candidates)
        counts["pairs_rejected"] = rejected
        counts["pairs_unmeasurable"] = unmeasurable
        counts["candidates"] = len(candidates)
        counts["conversions"] = sum(
            1
            for item in candidates
            if item.containment.conversion.kind is not ConversionKind.IDENTITY
        )
        skipped.extend(self._unmeasurable_reasons(columns))
        if capped:
            skipped.append(
                f"stopped after {self._max_pairs} candidate pairs: raise max_pairs to measure the "
                "rest"
            )
        return JoinMeasurement(
            candidates=candidates,
            columns=tuple(columns),
            counts=counts,
            skipped=tuple(sorted(set(skipped))),
        )

    def _read_values(
        self,
        source: str,
        adapter: SourceAdapter,
        table: Table,
        column: Column,
        ctx: PipelineContext,
        dialect: str,
    ) -> tuple[ValueSet | None, str]:
        """The distinct values of one column, or the reason it could not be read."""
        label = f"{source}.{table.name}.{column.name}"
        try:
            sql = distinct_query(table, column, dialect=dialect, cap=self._value_cap)
        except (NotSupported, ValueError) as exc:
            return None, f"{label}: {exc}"
        fragment = Fragment(
            source=source, dialect=dialect, sql=sql, tables=(table.name,), exec_ts=ctx.exec_ts
        )
        limits = Limits(
            timeout_s=self._timeout_s,
            max_rows=self._value_cap + 1,
            max_bytes=self._max_bytes,
            max_partitions=1,
        )
        rows: list[Mapping[str, Any]] = []
        try:
            with ctx.access(source):
                stream = adapter.execute(fragment, limits)
                try:
                    rows = [row for batch in stream for row in _rows_of(batch)]
                finally:
                    stream.close()
        except (NotSupported, TimeoutError) as exc:
            return None, f"{label}: the value read stopped early ({exc})"
        truncated = len(rows) > self._value_cap
        rows = rows[: self._value_cap]
        values = tuple(str(row["value"]) for row in rows)
        try:
            counts = tuple(int(row["rows"]) for row in rows)
        except (TypeError, ValueError) as exc:
            return None, f"{label}: a row count is not a number ({exc})"
        return (
            ValueSet(
                source=source,
                table=table.name,
                column=column.name,
                normalized_type=column.normalized_type,
                values=values,
                counts=counts,
                truncated=truncated,
                declared_unique=_declared_unique(table, column.name),
            ),
            "",
        )

    def _pairs(
        self,
        columns: Sequence[ValueSet],
        dialects: Mapping[str, str],
        catalogs: Mapping[tuple[str, str], Table],
    ) -> tuple[tuple[JoinCandidate, ...], int, int, bool]:
        """Measure every cross-table pair, and count what was rejected and what was unmeasurable."""
        ordered = sorted(columns, key=_by_label)
        candidates: list[JoinCandidate] = []
        rejected = 0
        unmeasurable = 0
        pairs = 0
        for index, left in enumerate(ordered):
            if pairs >= self._max_pairs:
                break
            for right in ordered[index + 1 :]:
                if pairs >= self._max_pairs:
                    break
                if (left.source, left.table) == (right.source, right.table):
                    continue
                if not _comparable(left, right):
                    rejected += 1
                    continue
                if left.truncated or right.truncated:
                    unmeasurable += 1
                    continue
                if min(left.distinct, right.distinct) == 0:
                    continue
                ratio = max(left.distinct, right.distinct) / min(left.distinct, right.distinct)
                if ratio > self._max_size_ratio:
                    rejected += 1
                    continue
                pairs += 1
                dialect = dialects.get(right.source, "")
                best, tried = best_conversion(left, right, dialect=dialect)
                if not self._admits(best):
                    rejected += 1
                    continue
                tables = (
                    catalogs.get((left.source, left.table)),
                    catalogs.get((right.source, right.table)),
                )
                candidates.append(self._candidate(left, right, best, tried, tables))
        candidates.sort(key=_candidate_rank)
        return tuple(candidates), rejected, unmeasurable, pairs >= self._max_pairs

    def _admits(self, containment: Containment) -> bool:
        """True when an overlap is strong enough to put in front of a human."""
        return (
            containment.jaccard >= self._min_jaccard and containment.best >= self._min_containment
        )

    def _candidate(
        self,
        left: ValueSet,
        right: ValueSet,
        best: Containment,
        tried: Sequence[Containment],
        tables: tuple[Table | None, Table | None],
    ) -> JoinCandidate:
        """Assemble one proposal: the numbers, the conversion, and the words that explain them."""
        similarity = name_similarity(left.column, right.column)
        left_table, right_table = tables
        # A foreign key may be declared on either side: the child is the one that names one row of
        # the parent, and which side sorts first is not a fact about the data.
        fk = (
            left_table is not None
            and _declares_fk(left_table, left.column, right.table, right.column)
        ) or (
            right_table is not None
            and _declares_fk(right_table, right.column, left.table, left.column)
        )
        uniqueness = (left.declared_unique, right.declared_unique)
        conversion = best.conversion
        evidence = [
            (
                f"{left.label} vs {right.label}: overlap {best.jaccard:.4f} "
                f"({best.matched_distinct} shared of {best.left_distinct} and "
                f"{best.right_distinct})"
            ),
            (
                f"containment {best.containment_left:.4f} left / {best.containment_right:.4f} "
                "right under " + conversion.kind.value
            ),
            (
                f"{left.label}: {best.unmatched_left_distinct} unmatched keys "
                f"covering {best.unmatched_left_rows} rows"
            ),
            (
                f"{right.label}: {best.unmatched_right_distinct} unmatched keys "
                f"covering {best.unmatched_right_rows} rows"
            ),
            f"conversion: {conversion.description} -> on {conversion.on}",
            f"name similarity {similarity:.3f}; "
            f"widths {left.widths or 'n/a'} vs {right.widths or 'n/a'}",
            (
                "uniqueness: left "
                + ("declared unique" if uniqueness[0] else "not declared unique")
                + ", right "
                + ("declared unique" if uniqueness[1] else "not declared unique")
            ),
            (
                f"values read from the whole column ({left.rows} rows)"
                if not left.truncated
                else f"values capped at {self._value_cap}"
            ),
        ]
        if fk:
            evidence.append(f"declared foreign key: {left.label} references {right.table}")
        score = round(
            0.6 * best.jaccard + 0.2 * best.best + 0.2 * similarity,
            6,
        )
        confidence = round(min(1.0, 0.75 * best.jaccard + 0.25 * best.best), 6)
        return JoinCandidate(
            name=_join_name(left, right),
            left=left.label,
            right=right.label,
            left_table=left.table,
            right_table=right.table,
            containment=best,
            name_similarity=similarity,
            uniqueness=uniqueness,
            cardinality=_cardinality(*uniqueness),
            declared_fk=fk,
            auto_approvable=fk and best.total,
            score=min(1.0, score),
            confidence=confidence,
            evidence=tuple(evidence),
            tried=tuple(sorted(tried, key=_overlap_rank)),
        )

    def _unmeasurable_reasons(self, columns: Sequence[ValueSet]) -> list[str]:
        """One line per truncated column: why its pairs were not measured at all."""
        return [
            (
                f"{item.label}: more than {self._value_cap} distinct values, so containment is "
                "unknown and no pair from it was proposed"
            )
            for item in columns
            if item.truncated
        ]

    def _proposal(self, candidate: JoinCandidate, ctx: PipelineContext) -> Proposal:
        """The review-queue proposal that carries one candidate and its evidence."""
        patch = candidate.as_approved_join().model_dump(mode="json", exclude_none=True)
        return Proposal(
            id=proposal_id(ProposalKind.JOIN, "approved_joins", candidate.name, "", patch),
            kind=ProposalKind.JOIN,
            section="approved_joins",
            key=candidate.name,
            collection=True,
            patch=patch,
            evidence=candidate.evidence,
            confidence=candidate.confidence,
            proposed_by=self.name,
            proposed_at=ctx.exec_ts,
        )

    def _refusal(self, ctx: PipelineContext) -> str:
        """The message a direct caller sees: the same rule and hint the runner gives."""
        upcoming = next_quiet_start(ctx.quiet_windows, ctx.exec_ts)
        detail = (
            f"the next window opens at {upcoming[0].isoformat()}"
            if upcoming is not None
            else "no quiet window is configured so none ever opens"
        )
        return (
            f"stage {self.name} join measurement is heavy and its quiet window is closed "
            f"({detail}); run it inside the window or pass --force to measure now"
        )
