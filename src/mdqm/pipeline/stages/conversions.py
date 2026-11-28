"""Stage D8: rediscover a measure's conversion from the data, and propose it for a human.

DESIGN.md section 12 states the motivating case: one credit limit means ``round_half_up(x / 5000)
* 5000``; ``crm`` stores it rounded, ``billing`` stores it raw and ``erp`` stores it rounded and
then divided by ten. Joining the three without converting them compares 5000 with 2500 and 500, so
every source needs a normalisation -- and **nobody may be told which one**. This stage finds it:

* the hypothesis space is **finite and declared**, never invented: a scale from
  :data:`SCALES`, an optional rounding step from D5's :data:`~mdqm.pipeline.stages.profiling.
  GRANULARITY_LADDER`, a rounding mode (``half_up`` | ``half_even``), and the **order of
  operations** -- ``scale_then_round`` (``ROUND(x * k / g) * g``) or ``round_then_scale``
  (``ROUND(x / g) * g * k``). Every candidate renders to a template the measure loader already
  accepts (:func:`~mdqm.semantic.measures.check_normalize_template`), so what D8 proposes C06 can
  execute; there is no second vocabulary.
* **the score is the match rate on aligned pairs**, and the pairs that do not fit are reported as
  :class:`CounterExample` rows. A hypothesis below :data:`DEFAULT_MIN_MATCH` is not a weak match to
  be apologised for: the answer is :attr:`~ConversionOutcome.NO_HYPOTHESIS`.
* **nothing is approved here.** A detection becomes a :class:`MeasureProposal` -- evidence,
  confidence, counter-examples -- and :meth:`MeasureProposal.to_proposal` hands it to the review
  queue (S12), the only writer of ``curated/``. DESIGN.md section 10 puts every conversion behind a
  human, so there is no auto-approve path in this module at all, and no write to ``curated/``.
* **direction is declared, not guessed.** A pair names one *reference* column (the column whose
  units the measure is defined in) and one *other* column to convert into it. Which columns mean the
  same business field is D7's question, and which column is canonical is the caller's declaration;
  guessing it from the data would make the tie between ``erp * 10`` and ``crm / 10`` an invention.

Determinism: candidates are enumerated and scored in a total order (see :func:`candidate_key`), the
pairs are sorted before any cap, counter-examples are sorted, and money never passes through a
binary float -- :func:`round_to_multiple` from :mod:`mdqm.semantic.measures` does the rounding in
exact integer fractions, so the detector and the emitted SQL agree at the ``.5`` boundary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Final

import pyarrow as pa
from pydantic import Field

from mdqm.adapters.base import NotSupported, SourceAdapter
from mdqm.models.base import MDQMModel
from mdqm.models.catalog import Table
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
from mdqm.pipeline.stages.profiling import GRANULARITY_LADDER, sample_query
from mdqm.review.queue import Proposal, ProposalKind, ReviewQueue, proposal_id
from mdqm.semantic.measures import check_normalize_template, round_to_multiple
from mdqm.semantic.models import RoundingMode

__all__ = [
    "DEFAULT_COUNTER_EXAMPLES",
    "DEFAULT_MAX_PAIRS",
    "DEFAULT_MIN_MATCH",
    "MIN_PAIRS",
    "SCALES",
    "STAGE_VERSION",
    "ColumnPair",
    "ColumnRef",
    "ConversionOutcome",
    "ConversionStage",
    "CounterExample",
    "Hypothesis",
    "MeasureProposal",
    "Order",
    "PairDetection",
    "QuietWindowClosed",
    "candidate_key",
    "detect_conversion",
    "detect_pairs",
    "hypotheses",
]

#: Bumped when a change to the search itself changes what is proposed, so a rerun re-detects.
STAGE_VERSION: Final[int] = 1

#: The scale factors worth trying, from ``0.001`` to ``1000``. ``1`` is identity.
SCALES: Final[tuple[Decimal, ...]] = (
    Decimal(1),
    Decimal("0.001"),
    Decimal("0.01"),
    Decimal("0.1"),
    Decimal(10),
    Decimal(100),
    Decimal(1000),
)

#: The rounding steps: D5's granularity ladder, so the two stages agree on what a grid is.
ROUNDING_STEPS: Final[tuple[int, ...]] = GRANULARITY_LADDER

#: The modes the search tries. A tie at ``x.5`` is the whole difference, so both are scored and the
#: boundary rows decide (DESIGN.md section 12: the mode is a stated, human-approved property).
MODES: Final[tuple[RoundingMode, ...]] = (RoundingMode.HALF_UP, RoundingMode.HALF_EVEN)

#: A hypothesis must explain at least this share of the pairs to be proposed at all. The world
#: plants real exceptions (rows that genuinely disagree), and 1% is the tolerance for them: below
#: this the honest answer is "no conversion found", never a weak match.
DEFAULT_MIN_MATCH: Final[float] = 0.99

#: Fewer aligned pairs than this is not evidence; the outcome is "insufficient data".
MIN_PAIRS: Final[int] = 3

#: How many pairs one detection scores. A rate does not need the whole table, and this bounds the
#: work of the 215-candidate search.
DEFAULT_MAX_PAIRS: Final[int] = 2_000

#: How many rows that do not fit are kept per hypothesis: enough to see a pattern, never a flood.
DEFAULT_COUNTER_EXAMPLES: Final[int] = 10

#: Rows one bounded fetch may return, matching D5's sample size so the two stages read alike.
DEFAULT_SAMPLE_ROWS: Final[int] = 5_000

#: ``half_up`` first: it is the project's stated default (``RoundingRule.mode``), so an exact tie
#: between the two modes resolves to it rather than to whichever name sorts first.
_MODE_PREFERENCE: Final[dict[RoundingMode, int]] = {
    RoundingMode.HALF_UP: 0,
    RoundingMode.HALF_EVEN: 1,
    RoundingMode.HALF_DOWN: 2,
    RoundingMode.FLOOR: 3,
    RoundingMode.CEIL: 4,
}

#: The byte cap for one bounded fetch; with the row cap it keeps a pair read from becoming a scan.
_MAX_BYTES: Final[int] = 8 * 1024 * 1024


class QuietWindowClosed(RuntimeError):
    """A heavy stage was asked to run outside its quiet window."""


class Order(StrEnum):
    """Where the scaling sits relative to the rounding. Both are in the hypothesis space."""

    #: ``ROUND(x * k / g) * g`` -- the source is converted into canonical units, then snapped.
    SCALE_THEN_ROUND = "scale_then_round"
    #: ``ROUND(x / g) * g * k`` -- the source is snapped on its own grid, then scaled up.
    ROUND_THEN_SCALE = "round_then_scale"


class ConversionOutcome(StrEnum):
    """What the search concluded about one pair of columns."""

    IDENTITY = "identity"
    """The column is already in canonical units; nothing to propose."""

    CONVERTED = "converted"
    """A hypothesis explains the pairs well enough to propose; a human still decides."""

    NO_HYPOTHESIS = "no_hypothesis"
    """Nothing in the declared space fits. Reported as a fact, never patched over."""

    INSUFFICIENT_DATA = "insufficient_data"
    """Too few usable pairs to say anything."""


# --- the hypothesis space ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Spec:
    """One point of the search space, before it is scored against the pairs."""

    scale: Decimal
    step: int | None
    mode: RoundingMode | None
    order: Order | None

    @property
    def rank(self) -> int:
        """Simplicity: the fewer operations, the better, when two candidates score the same.

        A rounding step that changes nothing is structure the data does not support, so
        ``x * 10`` beats ``ROUND(x * 10 / 5000) * 5000`` on an erp column that is already rounded.
        """
        if self.step is None:
            return 0 if self.scale == 1 else 1
        return 2 if self.scale == 1 else 3


def _literal(value: Decimal) -> str:
    """A decimal as plain digits (``0.001``, never ``1E-3``), so sqlglot reads it as a number."""
    return format(value.normalize(), "f")


def _template(spec: _Spec) -> str:
    """The normalize template this candidate means, in the vocabulary C06 validates.

    Rendered here rather than assembled at proposal time so the string a human approves is the
    string that was scored -- and so :func:`check_normalize_template` can refuse it immediately.
    """
    if spec.step is None:
        return "{col}" if spec.scale == 1 else f"{{col}} * {_literal(spec.scale)}"
    step = spec.step
    if spec.scale == 1:
        # Both orders are the same expression when the scale is 1, so there is one candidate.
        return f"ROUND({{col}} / {step}) * {step}"
    if spec.order is Order.ROUND_THEN_SCALE:
        return f"ROUND({{col}} / {step}) * {step} * {_literal(spec.scale)}"
    return f"ROUND({{col}} * {_literal(spec.scale)} / {step}) * {step}"


def _candidate_specs(
    scales: Sequence[Decimal], steps: Sequence[int], modes: Sequence[RoundingMode]
) -> tuple[_Spec, ...]:
    """Every point of the declared space, identity first and no duplicates by rendering.

    ``step=None`` needs no mode, and ``scale=1`` makes the two orders the same expression, so those
    two collapses are applied here -- otherwise the runner-up list would be full of the same formula
    written twice.
    """
    specs: list[_Spec] = [_Spec(scale=Decimal(1), step=None, mode=None, order=None)]
    for scale in scales:
        if scale != 1:
            specs.append(_Spec(scale=scale, step=None, mode=None, order=None))
    for step in steps:
        for mode in modes:
            specs.append(_Spec(scale=Decimal(1), step=step, mode=mode, order=None))
            for scale in scales:
                if scale == 1:
                    continue
                for order in (Order.SCALE_THEN_ROUND, Order.ROUND_THEN_SCALE):
                    specs.append(_Spec(scale=scale, step=step, mode=mode, order=order))
    return tuple(specs)


def _predicted(spec: _Spec, value: Decimal) -> Decimal:
    """Apply the candidate exactly: no binary float, and the same rule the rendered SQL uses."""
    if spec.step is None:
        return value * spec.scale
    if spec.order is Order.ROUND_THEN_SCALE:
        return round_to_multiple(value, spec.step, spec.mode or RoundingMode.HALF_UP) * spec.scale
    return round_to_multiple(value * spec.scale, spec.step, spec.mode or RoundingMode.HALF_UP)


def candidate_key(hypothesis: Hypothesis) -> tuple[Any, ...]:
    """The total order candidates are ranked by: best fit first, then the simplest claim.

    Ties are decided by declared facts, never by dict order: match rate, then rank (identity <
    scale-only < round-only < round-and-scale), then the **coarsest** step (a grid of 5000 that fits
    is 5000, not 1), then the smaller scale, then ``half_up`` before ``half_even``.
    """
    mode = hypothesis.mode
    return (
        -hypothesis.match_rate,
        hypothesis.rank,
        -(hypothesis.step or 0),
        hypothesis.scale,
        _MODE_PREFERENCE.get(mode, 99) if mode is not None else -1,
        hypothesis.order.value if hypothesis.order is not None else "",
    )


class Hypothesis(MDQMModel):
    """One candidate conversion and what it scored. ``apply`` is what the rendered SQL computes."""

    scale: Decimal = Field(gt=0, description="Multiply the source value by this before/after.")
    step: int | None = Field(default=None, gt=0, description="Snap to a multiple of this.")
    mode: RoundingMode | None = Field(default=None, description="The tie rule, when it rounds.")
    order: Order | None = Field(
        default=None, description="Which side of the scale the round is on."
    )
    template: str = Field(min_length=1, description="The normalize template in the measure YAML.")
    matched: int = Field(ge=0)
    compared: int = Field(ge=0)

    @property
    def match_rate(self) -> float:
        """The share of compared pairs this hypothesis explains exactly; 0.0 when none were."""
        return self.matched / self.compared if self.compared else 0.0

    @property
    def rank(self) -> int:
        """How many operations the claim uses; the tie-break when two fit equally well."""
        if self.step is None:
            return 0 if self.scale == 1 else 1
        return 2 if self.scale == 1 else 3

    @property
    def identity(self) -> bool:
        """True for the do-nothing hypothesis: it changes nothing, so nobody approves it."""
        return self.scale == 1 and self.step is None

    def apply(self, value: Decimal) -> Decimal:
        """The predicted canonical value, computed exactly (decimal/integer maths only)."""
        return _predicted(
            _Spec(scale=self.scale, step=self.step, mode=self.mode, order=self.order), value
        )

    def describe(self) -> str:
        """One readable line: the formula, the rule, and the rate a human is being asked about."""
        rule = f" ({self.mode.value})" if self.mode is not None else ""
        return (
            f"{self.template}{rule} matches {self.matched}/{self.compared} "
            f"pair(s) ({self.match_rate:.1%})"
        )


class CounterExample(MDQMModel):
    """One pair the hypothesis does not explain: the strongest thing a reviewer can be shown."""

    other: str = Field(description="The source value, as text.")
    reference: str = Field(description="The value actually stored on the reference side.")
    predicted: str = Field(description="What the hypothesis expected instead.")


class PairDetection(MDQMModel):
    """Everything the search found for one pair of columns: the answer, the rivals, the misfits."""

    measure: str = Field(min_length=1)
    reference: str = Field(min_length=1, description="The column whose units are canonical.")
    other: str = Field(min_length=1, description="The column being converted into those units.")
    outcome: ConversionOutcome
    pairs_compared: int = Field(ge=0)
    pairs_dropped: int = Field(
        ge=0, description="Pairs skipped: a NULL, or a value that is not a number."
    )
    best: Hypothesis | None = None
    runner_ups: tuple[Hypothesis, ...] = ()
    counter_examples: tuple[CounterExample, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def proposes_anything(self) -> bool:
        """True only when a conversion was found: identity and ``no_hypothesis`` ask nobody."""
        return self.outcome is ConversionOutcome.CONVERTED

    @property
    def confidence(self) -> float:
        """The validator's number: the observed match rate, never a model's opinion."""
        return self.best.match_rate if self.best is not None else 0.0

    def evidence(self) -> tuple[str, ...]:
        """The lines a reviewer reads in the review queue before approving or rejecting."""
        lines = [
            f"{self.measure}: converting {self.other} into {self.reference} over "
            f"{self.pairs_compared} aligned pair(s)"
            + (
                f"; {self.pairs_dropped} pair(s) unusable (NULL or not a number)"
                if self.pairs_dropped
                else ""
            )
        ]
        if self.best is not None:
            lines.append(f"best: {self.best.describe()}")
        for rival in self.runner_ups:
            lines.append(f"runner-up: {rival.describe()}")
        if self.counter_examples:
            first = self.counter_examples[0]
            lines.append(
                f"counter-examples: {len(self.counter_examples)} pair(s) don't fit, e.g. "
                f"{self.other}={first.other} stored {self.reference}={first.reference} but the "
                f"hypothesis predicts {first.predicted}"
            )
        lines.extend(self.notes)
        return tuple(lines)


# --- values -----------------------------------------------------------------------------------


def _decimal(value: object) -> Decimal | None:
    """A number as an exact ``Decimal``, or ``None`` when the value is missing or not a number.

    A ``bool`` is refused (``True`` is not a measure) and a float goes through ``repr`` because the
    column has already lost exactness at its source; the detector then compares exactly, so a
    noisy float row shows up as a counter-example instead of being hidden behind a tolerance.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if not math.isfinite(value):  # NaN / +-Inf carry no exact decimal
            return None
        return Decimal(repr(value))
    if isinstance(value, str):
        try:
            return Decimal(value.strip())
        except (InvalidOperation, ValueError):
            return None
    return None


def _aligned(
    reference: Sequence[object], other: Sequence[object], *, limit: int
) -> tuple[list[tuple[Decimal, Decimal]], int]:
    """The usable pairs, sorted and capped, plus how many pairs were dropped.

    Sorted before the cap so a bounded detection is reproducible: the same input always scores the
    same rows, whatever order the source returned them in.
    """
    if len(reference) != len(other):
        raise ValueError(
            f"the two sides must be aligned: {len(reference)} reference value(s) against "
            f"{len(other)} other value(s)"
        )
    pairs: list[tuple[Decimal, Decimal]] = []
    dropped = 0
    for left, right in zip(reference, other, strict=True):
        exact_ref = _decimal(left)
        exact_other = _decimal(right)
        if exact_ref is None or exact_other is None:
            dropped += 1
            continue
        pairs.append((exact_ref, exact_other))
    pairs.sort()
    return pairs[:limit], dropped


# --- the detector -----------------------------------------------------------------------------


def detect_conversion(
    reference: Sequence[object],
    other: Sequence[object],
    *,
    measure: str,
    reference_label: str,
    other_label: str,
    scales: Sequence[Decimal] = SCALES,
    steps: Sequence[int] = ROUNDING_STEPS,
    modes: Sequence[RoundingMode] = MODES,
    min_match: float = DEFAULT_MIN_MATCH,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    counter_examples: int = DEFAULT_COUNTER_EXAMPLES,
) -> PairDetection:
    """Find the conversion from ``other`` into ``reference``, or say that there is none.

    Scores every candidate of the declared space on the aligned pairs and reports the best, the
    rivals and the rows that do not fit. Below ``min_match`` (or with fewer than :data:`MIN_PAIRS`
    usable pairs) the outcome is :attr:`ConversionOutcome.NO_HYPOTHESIS` /
    :attr:`ConversionOutcome.INSUFFICIENT_DATA` -- a conversion is never invented to fill the gap.
    """
    if not 0.0 < min_match <= 1.0:
        raise ValueError(f"min_match must be in (0, 1], got {min_match}")
    if max_pairs < 1:
        raise ValueError(f"max_pairs must be at least 1, got {max_pairs}")
    pairs, dropped = _aligned(reference, other, limit=max_pairs)
    if len(pairs) < MIN_PAIRS:
        return PairDetection(
            measure=measure,
            reference=reference_label,
            other=other_label,
            outcome=ConversionOutcome.INSUFFICIENT_DATA,
            pairs_compared=len(pairs),
            pairs_dropped=dropped,
            notes=(
                f"only {len(pairs)} usable pair(s); at least {MIN_PAIRS} are needed to "
                "say anything",
            ),
        )
    compared = len(pairs)
    ranked = sorted(
        (_score(spec, pairs, compared=compared) for spec in _candidate_specs(scales, steps, modes)),
        key=candidate_key,
    )
    best = ranked[0]
    if best.match_rate < min_match:
        counter = _counter_examples(best, pairs, limit=counter_examples)
        return PairDetection(
            measure=measure,
            reference=reference_label,
            other=other_label,
            outcome=ConversionOutcome.NO_HYPOTHESIS,
            pairs_compared=compared,
            pairs_dropped=dropped,
            best=best,
            runner_ups=tuple(ranked[1:3]),
            counter_examples=counter,
            notes=(
                f"the best hypothesis explains only {best.match_rate:.1%} of the pairs, below the "
                f"{min_match:.1%} threshold, so no conversion is proposed",
            ),
        )
    best_claim = (best.template, best.mode)
    tied = sorted(
        {
            rival.template + (f" ({rival.mode.value})" if rival.mode is not None else "")
            for rival in ranked[1:]
            if rival.match_rate == best.match_rate and (rival.template, rival.mode) != best_claim
        }
    )
    notes: list[str] = []
    if tied:
        # A no-op rounding fits every column that already sits on that grid, so a tie is expected
        # and must be named rather than left for the reviewer to discover.
        notes.append(
            f"the sample cannot separate this from {len(tied)} hypothesis(es) explaining the same "
            f"pairs (e.g. {tied[0]}): the reviewer decides"
        )
    return PairDetection(
        measure=measure,
        reference=reference_label,
        other=other_label,
        outcome=ConversionOutcome.IDENTITY if best.identity else ConversionOutcome.CONVERTED,
        pairs_compared=compared,
        pairs_dropped=dropped,
        best=best,
        runner_ups=tuple(ranked[1:3]),
        counter_examples=(
            () if best.identity else _counter_examples(best, pairs, limit=counter_examples)
        ),
        notes=tuple(notes),
    )


def _score(spec: _Spec, pairs: Sequence[tuple[Decimal, Decimal]], *, compared: int) -> Hypothesis:
    """Count the pairs this candidate explains exactly and return it as a :class:`Hypothesis`."""
    matched = sum(1 for reference, other in pairs if _predicted(spec, other) == reference)
    return Hypothesis(
        scale=spec.scale,
        step=spec.step,
        mode=spec.mode,
        order=spec.order,
        template=_template(spec),
        matched=matched,
        compared=compared,
    )


def _counter_examples(
    hypothesis: Hypothesis, pairs: Sequence[tuple[Decimal, Decimal]], *, limit: int
) -> tuple[CounterExample, ...]:
    """The pairs the hypothesis misses, in a stable order, capped at ``limit``."""
    return tuple(
        CounterExample(
            other=str(other),
            reference=str(reference),
            predicted=str(hypothesis.apply(other)),
        )
        for reference, other in pairs
        if hypothesis.apply(other) != reference
    )[:limit]


def detect_pairs(
    pairs: pa.Table | pa.RecordBatch,
    *,
    measure: str,
    reference_label: str,
    other_label: str,
    reference_column: str = "reference",
    other_column: str = "other",
    **kwargs: Any,
) -> PairDetection:
    """Detect from a batch of aligned pairs (the shape D7/D5 hand over), column-wise.

    The two columns must be aligned by the entity key already: pairing rows is the join stage's job,
    and this function only reads numbers. Extra keyword arguments go to :func:`detect_conversion`.
    """
    for name in (reference_column, other_column):
        if name not in pairs.column_names:
            raise ValueError(
                f"the pair sample has no {name!r} column (it has {list(pairs.column_names)}); "
                f"name the two sides {reference_column!r} and {other_column!r}"
            )
    return detect_conversion(
        _values(pairs.column(reference_column)),
        _values(pairs.column(other_column)),
        measure=measure,
        reference_label=reference_label,
        other_label=other_label,
        **kwargs,
    )


def _values(column: pa.Array | pa.ChunkedArray) -> list[Any]:
    """One Arrow column as plain Python values."""
    return list(column.to_pylist())


def hypotheses(
    *,
    scales: Sequence[Decimal] = SCALES,
    steps: Sequence[int] = ROUNDING_STEPS,
    modes: Sequence[RoundingMode] = MODES,
) -> tuple[Hypothesis, ...]:
    """The whole declared space, unscored and in :func:`candidate_key` order without a rate.

    Exposed so a test -- or a human reading a digest -- can see exactly what D8 is allowed to
    propose. Every template here is validated on the way out, so a candidate that C06 would refuse
    cannot be scored in the first place.
    """
    out: list[Hypothesis] = []
    for spec in _candidate_specs(scales, steps, modes):
        template = _template(spec)
        check_normalize_template(template, where="a D8 candidate template")
        out.append(
            Hypothesis(
                scale=spec.scale,
                step=spec.step,
                mode=spec.mode,
                order=spec.order,
                template=template,
                matched=0,
                compared=0,
            )
        )
    return tuple(out)


# --- what a human is asked --------------------------------------------------------------------


class MeasureProposal(MDQMModel):
    """A conversion D8 found, with the evidence needed to approve it: the S12-ready output.

    ``template`` is exactly what belongs in ``measures.<name>.sources.<column>.normalize``; it is
    not written anywhere by this module. :meth:`to_proposal` builds the review-queue row, and only
    ``ReviewQueue.approve`` (a named human) ever writes curated YAML (DESIGN.md section 10).
    """

    measure: str = Field(min_length=1)
    source: str = Field(
        min_length=1, description="The column to convert, e.g. billing.credit_limit."
    )
    reference: str = Field(min_length=1, description="The column whose units are canonical.")
    outcome: ConversionOutcome
    template: str | None = Field(default=None, description="Empty when nothing is proposed.")
    step: int | None = None
    mode: RoundingMode | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: tuple[str, ...] = ()
    counter_examples: tuple[CounterExample, ...] = ()
    matched: int = Field(default=0, ge=0)
    compared: int = Field(default=0, ge=0)

    @classmethod
    def from_detection(cls, detection: PairDetection) -> MeasureProposal:
        """The proposal for a detection: a template only when a conversion was actually found."""
        best = detection.best if detection.proposes_anything else None
        return cls(
            measure=detection.measure,
            source=detection.other,
            reference=detection.reference,
            outcome=detection.outcome,
            template=None if best is None else best.template,
            step=None if best is None else best.step,
            mode=None if best is None else best.mode,
            confidence=detection.confidence if best is not None else 0.0,
            evidence=detection.evidence(),
            counter_examples=detection.counter_examples,
            matched=0 if best is None else best.matched,
            compared=detection.pairs_compared,
        )

    def to_proposal(self, *, proposed_at: datetime, proposed_by: str = "pipeline.D8") -> Proposal:
        """The review-queue row for this conversion. Raises when there is nothing to approve.

        The patch sits on the **measure item** (``field=""``) and merges ``sources.<column>`` there,
        not on ``sources.<column>`` directly: the queue attaches the approval provenance to the
        patch's own root, so a deeper field would record "approved_by" as if it were a source
        column. Known gap, reported in ``progress/S06.md``: approving a second source for a measure
        that already has one replaces ``sources`` instead of deep-merging it.
        """
        if self.template is None:
            raise ValueError(
                f"{self.measure}: {self.source} has outcome {self.outcome.value}, so there is no "
                "conversion to approve"
            )
        patch: dict[str, Any] = {"sources": {self.source: {"normalize": self.template}}}
        return Proposal(
            id=proposal_id(ProposalKind.MEASURE, "measures", self.measure, "", patch),
            kind=ProposalKind.MEASURE,
            section="measures",
            key=self.measure,
            field="",
            patch=patch,
            evidence=self.evidence,
            confidence=self.confidence,
            proposed_by=proposed_by,
            proposed_at=proposed_at,
        )


# --- the stage --------------------------------------------------------------------------------


class ColumnRef(MDQMModel):
    """One column of one table, plus the entity key that aligns it with the other side."""

    source: str = Field(min_length=1)
    table: str = Field(min_length=1)
    column: str = Field(min_length=1)
    key: str = Field(min_length=1, description="The entity key column in the same table.")

    @property
    def label(self) -> str:
        """``source.table.column``: the measure's own key for this source column."""
        return f"{self.source}.{self.table}.{self.column}"


class ColumnPair(MDQMModel):
    """Two columns that are believed to hold the same measure, one of them already canonical.

    ``reference`` is a **declaration**, not a detection: the pipeline says which column defines the
    measure's units (usually the one the semantic layer already names). Guessing it from the data
    would turn the symmetric case -- ``erp * 10`` and ``crm / 10`` fit the same rows -- into an
    invention.
    """

    reference: ColumnRef
    other: ColumnRef


@dataclass
class _Sample:
    """One bounded fetch: its rows and whether the fetch itself failed."""

    rows: list[Mapping[str, Any]]
    ok: bool


class ConversionStage:
    """Stage D8. Reads bounded samples, scores the declared space, proposes; never approves.

    The stage holds no source knowledge: :class:`ColumnPair` says where the two halves live, the
    bounded fetch is D5's (:func:`~mdqm.pipeline.stages.profiling.sample_query`), and the review
    queue is S12's. What it adds is the search and the honesty about its result.
    """

    name: str = "D8"
    heavy: bool = True

    def __init__(
        self,
        adapters: Mapping[str, SourceAdapter],
        pairs: Sequence[ColumnPair],
        *,
        measure: str,
        sample_rows: int = DEFAULT_SAMPLE_ROWS,
        min_match: float = DEFAULT_MIN_MATCH,
        max_pairs: int = DEFAULT_MAX_PAIRS,
        counter_examples: int = DEFAULT_COUNTER_EXAMPLES,
        steps: Sequence[int] = ROUNDING_STEPS,
        modes: Sequence[RoundingMode] = MODES,
        timeout_s: float = 30.0,
        propose: bool = True,
    ) -> None:
        if sample_rows < 1:
            raise ValueError(f"sample_rows must be at least 1, got {sample_rows}")
        self._adapters = dict(adapters)
        self._pairs = tuple(pairs)
        self._measure = measure
        self._sample_rows = sample_rows
        self._min_match = min_match
        self._max_pairs = max_pairs
        self._counter_examples = counter_examples
        self._steps = tuple(steps)
        self._modes = tuple(modes)
        self._timeout_s = timeout_s
        self._propose = propose

    @property
    def sources(self) -> tuple[str, ...]:
        """The sources this instance reads, sorted (what the runner gates and filters on)."""
        return tuple(sorted(self._adapters))

    @property
    def pairs(self) -> tuple[ColumnPair, ...]:
        """The declarations this instance tests, in the order it was given them.

        Read-only and public because it *is* the stage's input: a caller that built the pairs (the
        bootstrap wiring) and a test that asserts what was declared both need to see them, and
        reaching into ``_pairs`` for that is how internals leak into tests.
        """
        return self._pairs

    def inputs_hash(self, ctx: PipelineContext) -> str:
        """sha256 over the declared inputs: the pairs, the caps, the measure and the day.

        The **date** of ``exec_ts`` is deliberate, exactly as in D5: values that change every day
        are not the same fact tomorrow, so a nightly pass re-detects and two passes on one day
        agree.
        """
        return hash_inputs(
            self.name,
            STAGE_VERSION,
            self._measure,
            [pair.model_dump(mode="json") for pair in self._pairs],
            sorted(self._adapters),
            ctx.source,
            ctx.exec_ts.date().isoformat(),
            self._sample_rows,
            self._min_match,
            self._max_pairs,
            self._counter_examples,
            list(self._steps),
            [mode.value for mode in self._modes],
            self._propose,
        )

    def run(self, ctx: PipelineContext) -> StageReport:
        """Detect every declared pair and submit what was found to the review queue."""
        if not ctx.force and not quiet_allows(ctx.quiet_windows, ctx.exec_ts):
            raise QuietWindowClosed(self._refusal(ctx))
        counts: dict[str, int] = {
            "pairs": len(self._pairs),
            "compared": 0,
            "converted": 0,
            "identity": 0,
            "no_hypothesis": 0,
            "insufficient_data": 0,
            "unreadable": 0,
            "proposals": 0,
        }
        skipped: list[str] = []
        queue = ReviewQueue(ctx.store, ctx.semantic_root, clock=ctx.clock)
        for pair in self._pairs:
            detection = self.detect_pair(pair, ctx, skipped=skipped)
            if detection is None:
                counts["unreadable"] += 1
                continue
            counts["compared"] += detection.pairs_compared
            counts[detection.outcome.value] += 1
            proposal = MeasureProposal.from_detection(detection)
            if proposal.template is None:
                # Nothing to approve: say what was measured and why nothing was proposed.
                skipped.append(f"{proposal.measure}/{proposal.source}: {detection.outcome.value}")
                continue
            if not self._propose:
                continue
            result = queue.submit(proposal.to_proposal(proposed_at=ctx.exec_ts))
            counts["proposals"] += 1 if result.asks_a_human else 0
            if not result.asks_a_human:
                skipped.append(
                    f"{proposal.measure}/{proposal.source}: not asked again ({result.action.value})"
                )
        return StageReport(
            stage=self.name,
            status=StageStatus.RAN,
            counts=counts,
            skipped=tuple(skipped),
        )

    # -- internals -------------------------------------------------------------------------

    def _refusal(self, ctx: PipelineContext) -> str:
        """The message a direct caller sees: the same rule and hint the runner gives."""
        upcoming = next_quiet_start(ctx.quiet_windows, ctx.exec_ts)
        detail = (
            f"the next window opens at {upcoming[0].isoformat()}"
            if upcoming is not None
            else "no quiet window is configured so none ever opens"
        )
        return (
            f"stage {self.name} conversion detection is heavy and its quiet window is closed "
            f"({detail}); run it inside the window or pass --force to detect now"
        )

    def detect_pair(
        self,
        pair: ColumnPair,
        ctx: PipelineContext,
        *,
        skipped: list[str] | None = None,
    ) -> PairDetection | None:
        """Read both sides, align them on the entity key and score the declared space.

        Public so a caller (a digest, a test, an operator asking about one pair) can run the
        detection without the queue: ``None`` means the pair could not be read at all, and the
        reason is appended to ``skipped``.
        """
        skipped = skipped if skipped is not None else []
        missing = [
            ref.source for ref in (pair.reference, pair.other) if ref.source not in self._adapters
        ]
        if missing:
            skipped.append(
                f"{pair.reference.label} vs {pair.other.label}: no adapter for "
                f"{', '.join(sorted(set(missing)))}"
            )
            return None
        reference_rows = self._sample(pair.reference, ctx, skipped=skipped)
        other_rows = self._sample(pair.other, ctx, skipped=skipped)
        if reference_rows is None or other_rows is None:
            return None
        reference, other = self._join(pair, reference_rows, other_rows, skipped=skipped)
        if not reference:
            skipped.append(
                f"{pair.reference.label} vs {pair.other.label}: the two samples share no entity key"
            )
            return None
        batch = pa.table({"reference": pa.array(reference), "other": pa.array(other)})
        return detect_pairs(
            batch,
            measure=self._measure,
            reference_label=pair.reference.label,
            other_label=pair.other.label,
            steps=self._steps,
            modes=self._modes,
            min_match=self._min_match,
            max_pairs=self._max_pairs,
            counter_examples=self._counter_examples,
        )

    def _join(
        self,
        pair: ColumnPair,
        reference_rows: Sequence[Mapping[str, Any]],
        other_rows: Sequence[Mapping[str, Any]],
        *,
        skipped: list[str],
    ) -> tuple[list[Any], list[Any]]:
        """Align two samples on the entity key, comparing keys by their canonical text.

        ``0000000001`` and ``1`` are the same entity: a zero-padded key is the same format
        conversion D7 reports, and comparing the numbers instead of their spelling costs nothing.
        A duplicate key on the other side is one fact, so the first value wins and the duplicate is
        reported rather than double-counted.
        """
        lookup: dict[object, Any] = {}
        duplicates = 0
        for row in other_rows:
            key = _entity_key(row.get(pair.other.key))
            if key is None:
                continue
            if key in lookup:
                duplicates += 1
                continue
            lookup[key] = row.get(pair.other.column)
        reference: list[Any] = []
        other: list[Any] = []
        for row in reference_rows:
            key = _entity_key(row.get(pair.reference.key))
            if key is None or key not in lookup:
                continue
            reference.append(row.get(pair.reference.column))
            other.append(lookup[key])
        if duplicates:
            skipped.append(
                f"{pair.other.label}: {duplicates} duplicated key(s) in the sample were "
                "counted once"
            )
        return reference, other

    def _sample(
        self, ref: ColumnRef, ctx: PipelineContext, *, skipped: list[str]
    ) -> list[Mapping[str, Any]] | None:
        """One bounded fetch of ``ref.table``, through D5's own sampling statement."""
        adapter = self._adapters[ref.source]
        plan = sample_query(
            Table(name=ref.table),
            dialect=adapter.dialect(),
            rows=self._sample_rows,
            seed=0,
            native=False,
            row_estimate=None,
        )
        if plan is None:
            skipped.append(
                f"{ref.source}.{ref.table}: dialect {adapter.dialect()!r} cannot be "
                "sampled with SQL"
            )
            return None
        fragment = Fragment(
            source=ref.source,
            dialect=adapter.dialect(),
            sql=plan.sql,
            tables=(ref.table,),
            exec_ts=ctx.exec_ts,
        )
        limits = Limits(
            timeout_s=self._timeout_s,
            max_rows=self._sample_rows,
            max_bytes=_MAX_BYTES,
            max_partitions=1,
        )
        stream = adapter.execute(fragment, limits)
        try:
            rows = [row for batch in stream for row in batch.to_pylist()]
        except (NotSupported, TimeoutError) as exc:
            skipped.append(f"{ref.source}.{ref.table}: the bounded fetch stopped early ({exc})")
            return None
        finally:
            stream.close()
        missing = [name for name in (ref.key, ref.column) if rows and name not in rows[0]]
        if missing:
            skipped.append(
                f"{ref.source}.{ref.table}: the sample has no {', '.join(missing)} column, so the "
                "pair cannot be read"
            )
            return None
        if not rows:
            skipped.append(f"{ref.source}.{ref.table}: the bounded fetch returned no rows")
            return None
        return rows


def _entity_key(value: object) -> object | None:
    """A join key compared by meaning: ``'0000000001'`` and ``1`` are the same entity.

    Returns ``None`` for a missing or unusable key, so the row is dropped instead of joining every
    NULL to every other NULL.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            number = int(value)
        except (ValueError, OverflowError):  # NaN / +-Inf: no integer identity to join on
            return str(value)
        return number if number == value else str(value)
    if isinstance(value, Decimal):
        try:
            number = int(value)
        except (ArithmeticError, ValueError):  # NaN / +-Inf: no integer identity to join on
            return str(value)
        return number if number == value else str(value)
    text = str(value).strip()
    if not text:
        return None
    if not text.isdigit():
        return text
    try:
        return int(text)
    except ValueError:  # a digit int() cannot read (a superscript, or more digits than it parses)
        return text
