"""Tests for stage D8 conversion detection (S06).

The stage is the answer to "the bootstrap stages rediscover the 5000 / 5000-then-/10 conversions
from data": given the three storage forms of one credit limit, it must find the relationship itself,
report the rows that do not fit, and hand the result to the review queue -- never to curated YAML.

The acceptance criteria are the first group:

* **crm vs billing** -> ``ROUND(x / 5000) * 5000`` with ``half_up``, at least 99.5% of the pairs;
* **crm vs erp** -> ``x * 10``, with the planted erp exceptions reported as counter-examples;
* **boundary rows** (2500, 12500) are what separate ``half_up`` from ``half_even``, and the detector
  picks ``half_up``;
* **two unrelated columns** -> ``no_hypothesis``, not a weak match;
* the output is a :class:`MeasureProposal` with evidence, ready for the review queue.

Then the rules that keep it honest: the hypothesis space is finite and every template is one C06
accepts, nothing is written to ``curated/``, too few pairs is "insufficient data", and the stage
stays inside its quiet window.

No test sleeps, contacts a host or needs a container: the pairs come from the D02 world loaded into
F05's in-process DuckDB adapters, and the ``pg``/``oracle``/``mssql`` paths are not involved at all.

Only the two ``credit_limit`` pairs are declared here. Pairing the customers is D7's job, so the
test declares the entity key it expects (``crm.cust_id`` = ``erp.cust_no``; ``billing.client_no`` is
the same number zero-padded, which the stage compares by value) -- the *conversion* is what must be
discovered, and the assertions below never use the world's expected answers except to check that
what was found is right.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
import pytest

from mdqm.config.quiet import QuietWindow
from mdqm.pipeline.core import PipelineContext, StageStatus, run_pipeline
from mdqm.pipeline.stages.conversions import (
    ColumnPair,
    ColumnRef,
    ConversionOutcome,
    ConversionStage,
    QuietWindowClosed,
    detect_conversion,
    detect_pairs,
    hypotheses,
)
from mdqm.review.queue import ProposalKind, ReviewQueue
from mdqm.semantic.measures import check_normalize_template, render_normalize, round_to_multiple
from mdqm.semantic.models import Measure, MeasureSource, RoundingMode
from mdqm.store.db import Store
from mdqm.testing.fake_adapter import FakeAdapter
from mdqm.testing.world import WorldConfig, build_world

#: 2026-05-01 10:00 UTC = 13:00 in Asia/Jerusalem: inside the working day, so not quiet.
NOON = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)

#: 21:00-07:00 local: the window that makes a 13:00 pass a refusal.
NIGHT = QuietWindow(tz="Asia/Jerusalem", start="21:00", end="07:00")

#: The measure every pair below is about.
MEASURE = "credit_limit"

#: crm's rounded limit, the column the measure's units are declared in.
CRM = ColumnRef(source="crm", table="customers_crm", column="credit_limit", key="cust_id")
BILLING = ColumnRef(source="billing", table="accounts", column="credit_limit", key="client_no")
ERP = ColumnRef(source="erp", table="customers_erp", column="limit_k", key="cust_no")

BILLING_PAIR = ColumnPair(reference=CRM, other=BILLING)
ERP_PAIR = ColumnPair(reference=CRM, other=ERP)


@pytest.fixture(scope="module")
def world() -> Any:
    """One generated organisation, built once: the synthetic data every acceptance case uses."""
    return build_world(WorldConfig.small())


@pytest.fixture(scope="module")
def adapters(world: Any) -> Any:
    """The three storage forms as in-process DuckDB sources; closed at the end of the module."""
    built = {name: FakeAdapter.for_world(world, name=name) for name in ("crm", "billing", "erp")}
    yield built
    for adapter in built.values():
        adapter.close()


@pytest.fixture
def store(tmp_path: Any) -> Any:
    """An initialized metadata store under ``tmp_path``."""
    made = Store(tmp_path / "mdqm.sqlite3")
    made.initialize()
    return made


def _ctx(
    tmp_path: Any, store: Store, *, now: datetime = NOON, windows: Any = ()
) -> PipelineContext:
    """A context rooted in ``tmp_path``, with an injected instant and no model."""
    return PipelineContext(
        store=store,
        exec_ts=now,
        semantic_root=tmp_path / "semantic",
        quiet_windows=windows,
    )


def _stage(adapters: dict[str, FakeAdapter], pairs: Any) -> ConversionStage:
    """A D8 stage over the given pairs; the caps are the real defaults."""
    return ConversionStage(adapters, pairs, measure=MEASURE)


def _reference(world: Any) -> list[int]:
    """The canonical (crm) credit limit for every customer, in customer order."""
    return [truth.canonical_limit for truth in world.expected.customers]


def _planted_exception_pairs(world: Any) -> list[tuple[int, int]]:
    """The ``(raw limit, crm limit)`` the world deliberately keeps out of the conversion."""
    return [(item.erp_limit_k, item.crm_limit) for item in world.expected.erp_exceptions]


# --- acceptance: the three storage forms ------------------------------------------------------


def test_detects_all_three_forms_of_one_measure(
    tmp_path: Any, store: Store, adapters: dict[str, FakeAdapter], world: Any
) -> None:
    """billing needs ``ROUND(x/5000)*5000``, erp needs ``x*10``, crm is the declared canonical unit.

    The stage is run through :func:`run_pipeline`, so this is the whole contract (name, input hash,
    report) and not just a function call. The two conversions are proposed; the reference column
    needs nothing, and ``curated/`` is untouched.
    """
    stage = _stage(adapters, (BILLING_PAIR, ERP_PAIR))
    ctx = _ctx(tmp_path, store)
    result = run_pipeline([stage], ctx, force=True)
    assert result.ok, result.describe()
    report = result.reports[0]
    assert report.status is StageStatus.RAN
    assert report.counts["converted"] == 2
    assert report.counts["proposals"] == 2

    pending = ReviewQueue(store, tmp_path / "semantic").pending()
    templates = {
        column: source["normalize"]
        for proposal in pending
        for column, source in proposal.patch["sources"].items()
    }
    assert set(templates) == {
        "billing.accounts.credit_limit",
        "erp.customers_erp.limit_k",
    }
    assert templates["billing.accounts.credit_limit"] == "ROUND({col} / 5000) * 5000"
    assert templates["erp.customers_erp.limit_k"] == "{col} * 10"
    # The human gate is the only writer of curated/, so nothing exists there yet.
    assert not (tmp_path / "semantic" / "curated").exists()


def test_crm_vs_billing_recovers_half_up_rounding_to_5000(
    tmp_path: Any, store: Store, adapters: dict[str, FakeAdapter], world: Any
) -> None:
    """The acceptance case: ``ROUND(x/5000)*5000`` with ``half_up``, and every pair fits."""
    report = _stage(adapters, (BILLING_PAIR,)).run(_ctx(tmp_path, store))
    assert report.counts["converted"] == 1
    (proposal,) = ReviewQueue(store, tmp_path / "semantic").pending()
    assert proposal.patch == {
        "sources": {"billing.accounts.credit_limit": {"normalize": "ROUND({col} / 5000) * 5000"}}
    }
    assert proposal.confidence >= 0.995
    assert proposal.kind is ProposalKind.MEASURE

    detection = _detect_billing(adapters, tmp_path, store)
    assert detection.best is not None
    assert detection.best.step == 5000
    assert detection.best.mode is RoundingMode.HALF_UP
    assert detection.best.match_rate >= 0.995
    assert detection.counter_examples == ()
    # crm holds the rounded value and billing the raw one: the conversion explains every row that
    # has a value, and the planted NULL is dropped rather than counted as a miss.
    assert detection.pairs_dropped == len(world.expected.planted_nulls)
    assert detection.pairs_compared == len(world.expected.customers) - detection.pairs_dropped


def test_crm_vs_erp_recovers_x10_and_reports_the_planted_exceptions(
    tmp_path: Any, store: Store, adapters: dict[str, FakeAdapter], world: Any
) -> None:
    """erp stores the limit rounded and then divided by ten, so ``x * 10`` converts it back.

    The planted erp exceptions are exactly what must appear as counter-examples: one row holds a
    500 step too much, so ``x * 10`` overshoots by 5000 and the detector says so instead of
    rounding the fact away.
    """
    report = _stage(adapters, (ERP_PAIR,)).run(_ctx(tmp_path, store))
    assert report.counts["converted"] == 1
    (proposal,) = ReviewQueue(store, tmp_path / "semantic").pending()
    assert proposal.patch == {"sources": {"erp.customers_erp.limit_k": {"normalize": "{col} * 10"}}}

    detection = _detect_erp(adapters, tmp_path, store)
    assert detection.best is not None
    assert detection.best.template == "{col} * 10"
    assert detection.best.scale == 10
    assert detection.best.step is None, (
        "erp values are already rounded; a no-op ROUND is not claimed"
    )
    assert detection.counter_examples, "the planted exceptions must surface"
    found = {
        (int(item.other), int(item.reference), int(item.predicted))
        for item in detection.counter_examples
    }
    expected = {(raw, limit, limit + 5000) for raw, limit in _planted_exception_pairs(world)}
    assert found == expected
    assert detection.best.match_rate >= 0.995
    # The rounding on the erp side is a no-op on this sample, and the detector says so out loud
    # instead of pretending the sample proved there is no ROUND in the source.
    assert any("cannot separate" in note for note in detection.notes)


# --- acceptance: the boundary rows decide the mode ----------------------------------------------


def test_boundary_rows_decide_half_up_over_half_even() -> None:
    """2500 and 12500 are the values where the two modes disagree, so they decide the mode."""
    reference = [5000, 15000, 10000, 5000]
    other = [2500, 12500, 7500, 5000]
    detection = detect_conversion(
        reference,
        other,
        measure=MEASURE,
        reference_label="crm.credit_limit",
        other_label="billing.credit_limit",
    )
    assert detection.outcome is ConversionOutcome.CONVERTED
    assert detection.best is not None
    assert detection.best.mode is RoundingMode.HALF_UP
    assert detection.best.matched == 4
    # half_even is the runner-up and loses precisely on the ties: 2500 -> 0 and 12500 -> 10000.
    half_even = [rival for rival in detection.runner_ups if rival.mode is RoundingMode.HALF_EVEN]
    assert half_even, "the losing mode must be shown to the reviewer"
    assert half_even[0].matched == 2
    assert half_even[0].apply(2500) == 0
    assert half_even[0].apply(12500) == 10000


def test_without_a_boundary_row_the_two_modes_are_a_named_tie() -> None:
    """7500 rounds to 10000 under both modes, so the sample cannot choose and must say so."""
    detection = detect_conversion(
        [10000, 20000, 30000],
        [7500, 17500, 27500],
        measure=MEASURE,
        reference_label="crm.credit_limit",
        other_label="billing.credit_limit",
    )
    assert detection.best is not None
    assert detection.best.mode is RoundingMode.HALF_UP, "a tie resolves to the declared default"
    assert any("cannot separate" in note for note in detection.notes)


# --- acceptance: no weak matches ---------------------------------------------------------------


def test_two_unrelated_columns_yield_no_hypothesis(
    tmp_path: Any, store: Store, adapters: dict[str, FakeAdapter]
) -> None:
    """A customer id and an erp limit are not the same measure, so nothing is proposed."""
    decoy = ColumnPair(
        reference=ColumnRef(source="crm", table="customers_crm", column="cust_id", key="cust_id"),
        other=ERP,
    )
    report = _stage(adapters, (decoy,)).run(_ctx(tmp_path, store))
    assert report.counts["no_hypothesis"] == 1
    assert report.counts["converted"] == 0
    assert ReviewQueue(store, tmp_path / "semantic").pending() == ()
    assert any("no_hypothesis" in line for line in report.skipped)

    detection = detect_conversion(
        [1, 2, 3, 4],
        [500, 1000, 1500, 2000],
        measure=MEASURE,
        reference_label="crm.cust_id",
        other_label="erp.limit_k",
    )
    assert detection.outcome is ConversionOutcome.NO_HYPOTHESIS
    assert detection.best is not None, "the best attempt is still shown, with its poor rate"
    assert detection.best.match_rate < 0.99
    assert any("below the" in note for note in detection.notes)


def test_a_reference_column_needs_no_conversion_and_asks_nobody(
    tmp_path: Any, store: Store, adapters: dict[str, FakeAdapter], world: Any
) -> None:
    """crm is the declared canonical unit: identity, and identity is not a question for a human."""
    report = _stage(adapters, (ColumnPair(reference=CRM, other=CRM),)).run(_ctx(tmp_path, store))
    assert report.counts["identity"] == 1
    assert report.counts["proposals"] == 0
    assert ReviewQueue(store, tmp_path / "semantic").pending() == ()


def test_too_few_pairs_is_insufficient_data_not_a_guess() -> None:
    """Two pairs prove nothing, so the outcome says so instead of proposing a formula."""
    detection = detect_conversion(
        [5000, 10000],
        [2500, 7500],
        measure=MEASURE,
        reference_label="crm.credit_limit",
        other_label="billing.credit_limit",
    )
    assert detection.outcome is ConversionOutcome.INSUFFICIENT_DATA
    assert detection.best is None
    assert any("at least 3" in note for note in detection.notes)


# --- acceptance: the output is a review-queue-ready proposal ------------------------------------


def test_the_proposal_carries_its_evidence_and_targets_the_measure(
    tmp_path: Any, store: Store, adapters: dict[str, FakeAdapter]
) -> None:
    """A reviewer must be able to decide from the row alone: text, match rate, counter-examples."""
    _stage(adapters, (BILLING_PAIR,)).run(_ctx(tmp_path, store))
    (proposal,) = ReviewQueue(store, tmp_path / "semantic").pending()
    assert proposal.section == "measures"
    assert proposal.key == MEASURE
    assert proposal.collection is False
    assert proposal.location == ("measures", MEASURE)
    assert proposal.patch == {
        "sources": {"billing.accounts.credit_limit": {"normalize": "ROUND({col} / 5000) * 5000"}}
    }
    assert any("aligned pair" in line for line in proposal.evidence)
    assert any("unusable" in line for line in proposal.evidence)
    assert any("100.0%" in line for line in proposal.evidence)
    assert 0.0 < proposal.confidence <= 1.0
    assert proposal.proposed_by.startswith("pipeline")


def test_an_unproposed_detection_has_no_template_and_cannot_be_submitted(
    tmp_path: Any, store: Store, adapters: dict[str, FakeAdapter]
) -> None:
    """``no_hypothesis`` produces a proposal object with no template, and no queue row."""
    from mdqm.pipeline.stages.conversions import MeasureProposal

    detection = detect_conversion(
        [1, 2, 3, 4],
        [4000, 9000, 2000, 7000],
        measure=MEASURE,
        reference_label="crm.cust_id",
        other_label="erp.limit_k",
    )
    proposal = MeasureProposal.from_detection(detection)
    assert proposal.template is None
    with pytest.raises(ValueError, match="no conversion to approve"):
        proposal.to_proposal(proposed_at=NOON)


# --- the hypothesis space is the measure loader's vocabulary ------------------------------------


def test_every_candidate_is_a_template_the_measure_loader_accepts() -> None:
    """D8 may only propose what C06 can execute: each candidate is validated on the way out."""
    space = hypotheses()
    assert len(space) > 50, "the search must actually search"
    for candidate in space:
        check_normalize_template(candidate.template, where="a D8 candidate")
        measure = Measure(
            sources={"crm.customers_crm.credit_limit": MeasureSource(normalize=candidate.template)}
        )
        assert measure.sources["crm.customers_crm.credit_limit"].normalize == candidate.template


def test_the_recovered_rule_is_the_one_the_measure_module_renders(
    tmp_path: Any, store: Store, adapters: dict[str, FakeAdapter]
) -> None:
    """Cross-check against C06: the recovered template and the Python rule agree row by row."""
    detection = _detect_billing(adapters, tmp_path, store)
    assert detection.best is not None
    measure = Measure(
        canonical={"rounding": {"to": 5000, "mode": "half_up"}},
        sources={"billing.accounts.credit_limit": MeasureSource(normalize=detection.best.template)},
    )
    sql = render_normalize(
        measure, "billing.accounts.credit_limit", '"credit_limit"', dialect="postgres"
    )
    assert sql is not None
    assert "FLOOR" in sql, "ROUND must render as explicit arithmetic, never engine-lucky"
    for raw in (2500, 7500, 12500, 0, 60_000):
        assert detection.best.apply(raw) == round_to_multiple(raw, 5000, RoundingMode.HALF_UP)


def test_round_then_scale_is_found_when_the_order_matters() -> None:
    """``ROUND(x / g) * g * k`` and ``ROUND(x * k / g) * g`` differ; both are searched."""
    raw = [1234, 2871, 4321, 5001, 6789, 8999]
    reference = [round_to_multiple(item, 1000, RoundingMode.HALF_UP) * 10 for item in raw]
    detection = detect_conversion(
        reference,
        raw,
        measure=MEASURE,
        reference_label="crm.credit_limit",
        other_label="erp.limit_k",
    )
    assert detection.outcome is ConversionOutcome.CONVERTED
    assert detection.best is not None
    assert detection.best.order is not None
    assert detection.best.order.value == "round_then_scale"
    assert detection.best.template == "ROUND({col} / 1000) * 1000 * 10"
    assert detection.best.matched == len(raw)


def test_scales_and_pairs_are_checked_before_scoring() -> None:
    """A bad cap or misaligned columns is a caller error, not a silently wrong detection."""
    with pytest.raises(ValueError, match="aligned"):
        detect_conversion(
            [1, 2],
            [1],
            measure=MEASURE,
            reference_label="a",
            other_label="b",
        )
    with pytest.raises(ValueError, match="min_match"):
        detect_conversion(
            [1, 2, 3],
            [1, 2, 3],
            measure=MEASURE,
            reference_label="a",
            other_label="b",
            min_match=0.0,
        )


def test_values_that_are_not_numbers_are_dropped_and_counted() -> None:
    """A NULL or text value is not a counter-example: it is dropped, and the count says how many."""
    detection = detect_conversion(
        [5000, 10000, 15000, 20000],
        [2500, None, "n/a", 17500],
        measure=MEASURE,
        reference_label="crm.credit_limit",
        other_label="billing.credit_limit",
    )
    assert detection.pairs_dropped == 2
    assert detection.pairs_compared == 2
    assert detection.outcome is ConversionOutcome.INSUFFICIENT_DATA


def test_detection_is_deterministic() -> None:
    """Same input, same answer: no dict order, no clock, no randomness."""
    other = [2500, 7499, 12500, 1, 60_000]
    reference = [5000, 10000, 15000, 0, 60000]
    first = detect_conversion(
        reference, other, measure=MEASURE, reference_label="a", other_label="b"
    )
    second = detect_conversion(
        reference, other, measure=MEASURE, reference_label="a", other_label="b"
    )
    assert first == second


def test_an_arrow_batch_is_the_input_and_the_columns_must_be_named(
    tmp_path: Any, store: Store, adapters: dict[str, FakeAdapter]
) -> None:
    """D7/D5 hand over Arrow; only the ``reference``/``other`` columns are read."""
    batch = pa.table({"reference": [5000, 10000], "other": [2500, 7500]})
    detection = detect_pairs(
        batch,
        measure=MEASURE,
        reference_label="crm.credit_limit",
        other_label="billing.credit_limit",
    )
    assert detection.outcome is ConversionOutcome.INSUFFICIENT_DATA
    with pytest.raises(ValueError, match="no 'other' column"):
        detect_pairs(
            pa.table({"reference": [1], "right": [1]}),
            measure=MEASURE,
            reference_label="a",
            other_label="b",
        )


# --- the stage stays inside its quiet window ----------------------------------------------------


def test_a_direct_call_outside_the_quiet_window_is_refused(
    tmp_path: Any, store: Store, adapters: dict[str, FakeAdapter]
) -> None:
    """Heavy work waits for the window, exactly as D5 does; the message says when it opens."""
    stage = _stage(adapters, (BILLING_PAIR,))
    with pytest.raises(QuietWindowClosed, match="quiet window is closed"):
        stage.run(_ctx(tmp_path, store, windows=(NIGHT,)))


def test_forced_outside_the_window_it_runs(
    tmp_path: Any, store: Store, adapters: dict[str, FakeAdapter]
) -> None:
    """``--force`` is the operator's override, and the report records that it was forced."""
    stage = _stage(adapters, (BILLING_PAIR,))
    result = run_pipeline([stage], _ctx(tmp_path, store, windows=(NIGHT,)), force=True)
    assert result.ok, result.describe()
    assert result.reports[0].forced is True


# --- helpers -----------------------------------------------------------------------------------


def _detect_one(
    adapters: dict[str, FakeAdapter], pair: ColumnPair, tmp_path: Any, store: Store
) -> Any:
    """Detect one pair through the stage's own reading path (bounded sample -> Arrow -> search)."""
    detection = _stage(adapters, (pair,)).detect_pair(pair, _ctx(tmp_path, store))
    assert detection is not None, "the bounded sample must reach the detector"
    return detection


def _detect_billing(adapters: dict[str, FakeAdapter], tmp_path: Any, store: Store) -> Any:
    """The billing detection, through the stage's real reading path."""
    return _detect_one(adapters, BILLING_PAIR, tmp_path, store)


def _detect_erp(adapters: dict[str, FakeAdapter], tmp_path: Any, store: Store) -> Any:
    """The erp detection, through the same path."""
    return _detect_one(adapters, ERP_PAIR, tmp_path, store)
