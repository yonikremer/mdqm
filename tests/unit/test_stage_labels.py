"""Labels for tables, columns and measures: the model proposes words, code checks them, a human decides.

Section 6 of the handoff, and the reason this stage is wired to a model at all: ``SemanticLayer.
labels`` is the field the context index matches, so a Hebrew question resolves a table -- but
nothing proposed into it, so every Hebrew name was hand-written.

What this file proves, one class each:

* :class:`TestLabels` -- the accepted answer's words become one ``labels`` payload per table or
  column, keyed the way the layer keys them (``table:<source.table>``, ``column:<qualified>``); it
  carries the same evidence and confidence as the name; approving it writes ``curated/labels.yaml``
  and the layer loads back with those words.
* :class:`TestRefusals` -- a word that is not a word (JSON, SQL, a lone number, a control character,
  an over-long string) is refused with a code and a reason, and nothing is queued for that target.
* :class:`TestMeasureTarget` -- a measure is a target of its own: no table, its own namespace, its
  declared columns as the only ones it may name, and its labels reload from disk.
* :class:`TestTheStatedGap` -- a ``name``/``synonym`` payload still targets a section the layer does
  not declare, so approval refuses it. Pinned, not hidden.

Every test drives ``FakeLLM``: no socket, no key, no clock.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from mdqm.llm.fake import FakeLLM
from mdqm.pipeline.core import PipelineContext, run_pipeline
from mdqm.pipeline.stages.propose_names import (
    LABEL_WORDS_LIMIT,
    Labels,
    ProposeNamesStage,
    RejectionCode,
    Target,
    TargetKind,
    label_key,
    label_word_problem,
)
from mdqm.review.queue import Proposal, ProposalKind, ProposalStatus, ReviewError, ReviewQueue
from mdqm.semantic.loader import load
from mdqm.semantic.models import Measure
from mdqm.semantic.text import normalize_label
from mdqm.store.db import Store
from unit.test_stage_propose_names import (
    NOW,
    answers,
    both_runs,
    codes,
    make_catalog,
    make_profile,
    reply_from,
)

MEASURE = "credit_limit_r5000"


class Harness:
    """One stage and one context over a temp store, with the layer's measures as an input."""

    def __init__(
        self,
        root: Path,
        *,
        measures: dict[str, Measure] | None = None,
        known_terms: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.root = root
        self.store = Store(root / "metadata.sqlite")
        self.stage = ProposeNamesStage(
            {"crm": make_catalog()},
            profiles={"crm": make_profile()},
            measures=measures,
            known_terms=known_terms,
        )
        self.ctx = PipelineContext(store=self.store, exec_ts=NOW, semantic_root=root, llm=FakeLLM())

    def run(self, runs: Mapping[str, tuple[Labels, Labels] | Labels]) -> None:
        """Queue one scripted pass; the report is the stage's own, no assertion here."""
        llm = self.ctx.llm
        assert isinstance(llm, FakeLLM)
        llm.queue(*([reply_from(runs)] * (2 * len(self.stage.targets()))))
        run_pipeline([self.stage], self.ctx)

    def queue(self) -> ReviewQueue:
        return ReviewQueue(self.store, self.root)

    def pending(self, kind: ProposalKind | None = None) -> tuple[Proposal, ...]:
        return tuple(item for item in self.queue().pending() if kind is None or item.kind is kind)


def measure_stage_answers() -> dict[str, Labels]:
    """The catalog answers plus one for the measure, so every target has a scripted reply."""
    return {
        **answers(),
        MEASURE: Labels(
            name_he="מסגרת אשראי",
            name_en="Credit limit",
            synonyms_he=("מסגרת",),
            synonyms_en=("Limit",),
        ),
    }


def _measure() -> Measure:
    """One measure over one source column: what a layer declares before any data is read."""
    return Measure.model_validate(
        {"sources": {"crm.customers.credit_limit": {"normalize": "{col}"}}}
    )


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


@pytest.fixture
def measured(tmp_path: Path) -> Harness:
    """A harness whose layer declares one measure over one readable source column."""
    return Harness(
        tmp_path,
        measures={MEASURE: _measure()},
    )


# --- what a valid answer proposes -----------------------------------------------------------


class TestLabels:
    def test_table_and_column_words_are_keyed_the_way_the_layer_keys_them(
        self, harness: Harness
    ) -> None:
        harness.run(both_runs(answers()))

        labels = {item.key: item.patch["labels"] for item in harness.pending(ProposalKind.LABEL)}

        assert labels == {
            "table:crm.customers": ["לקוחות", "Customers", "אנשי קשר", "Clients"],
            "column:crm.customers.cust_id": [
                "מזהה לקוח",
                "Customer ID",
                "מספר לקוח",
                "Customer Number",
            ],
            "column:crm.customers.status_code": ["סטטוס לקוח", "Customer status", "State"],
            "column:crm.customers.email_address": ['כתובת דוא"ל', "Email address", "Email"],
        }
        assert {item.section for item in harness.pending(ProposalKind.LABEL)} == {"labels"}

    def test_a_label_carries_the_evidence_of_the_answer_it_came_from(
        self, harness: Harness
    ) -> None:
        harness.run(both_runs(answers()))

        pending = harness.queue().pending()
        label = next(
            item
            for item in pending
            if item.kind is ProposalKind.LABEL and item.key == "table:crm.customers"
        )
        named = next(
            item
            for item in pending
            if item.kind is ProposalKind.NAME and item.key == "crm.customers"
        )

        assert label.evidence == named.evidence
        assert label.confidence == named.confidence
        assert any("validators passed" in line for line in label.evidence)

    def test_approving_a_label_writes_a_layer_that_loads(self, harness: Harness) -> None:
        harness.run(both_runs(answers()))
        label = next(
            item
            for item in harness.pending(ProposalKind.LABEL)
            if item.key == "table:crm.customers"
        )

        approved = harness.queue().approve(label.id, by="data-team")

        assert approved.status is ProposalStatus.APPROVED
        layer = load(harness.root)
        assert layer.labels["table:crm.customers"] == (
            "לקוחות",
            "Customers",
            "אנשי קשר",
            "Clients",
        )

    def test_the_approved_list_holds_words_only(self, harness: Harness) -> None:
        """Every element is a search term, so the decision cannot be written inside the list."""
        harness.run(both_runs(answers()))
        label = next(item for item in harness.pending(ProposalKind.LABEL))
        harness.queue().approve(label.id, by="data-team")

        text = (harness.root / "curated" / "labels.yaml").read_text(encoding="utf-8")

        assert "approved_by" not in text
        assert "provenance" not in text
        assert all(isinstance(word, str) for word in load(harness.root).labels[label.key])

    def test_a_second_run_with_the_same_answers_writes_nothing_new(self, harness: Harness) -> None:
        harness.run(both_runs(answers()))
        first = len(harness.queue().pending())

        harness.ctx.force = True  # run again although the input hash is unchanged
        harness.run(both_runs(answers()))

        assert len(harness.queue().pending()) == first

    def test_the_per_key_cap_is_reported_instead_of_dropping_words_silently(self) -> None:
        wordy = Labels(
            name_he="סטטוס לקוח",
            name_en="Customer status",
            synonyms_he=("מצב", "מצב לקוח", "סטטוס", "עמדה", "הגדרה"),
            synonyms_en=("State", "Condition", "Situation", "Phase", "Flag"),
        )
        harness_stage = ProposeNamesStage({"crm": make_catalog()})
        target = Target(
            kind=TargetKind.COLUMN,
            key="crm.customers.status_code",
            source="crm",
            table="customers",
            column="status_code",
        )

        words, dropped = harness_stage.label_words(target, wordy)

        assert len(words) == LABEL_WORDS_LIMIT
        assert words[0] == "סטטוס לקוח"
        # 12 words offered, 8 kept: the four that went are counted, not silently dropped.
        assert dropped == 4

    def test_a_word_the_target_already_answers_to_is_not_reproposed(self, tmp_path: Path) -> None:
        harness = Harness(
            tmp_path,
            known_terms={"table:crm.customers": {normalize_label("לקוחות"): "crm.customers"}},
        )
        harness.run(both_runs(answers()))

        label = next(
            item
            for item in harness.pending(ProposalKind.LABEL)
            if item.key == "table:crm.customers"
        )

        assert "לקוחות" not in label.patch["labels"]
        assert "Customers" in label.patch["labels"]

    def test_a_value_gets_no_label_payload_because_its_value_map_holds_the_words(
        self, harness: Harness
    ) -> None:
        from unit.test_stage_propose_names import VALUE_MAP

        harness.stage = ProposeNamesStage(
            {"crm": make_catalog()}, profiles={"crm": make_profile()}, value_maps={"s": VALUE_MAP}
        )
        harness.run(both_runs({**answers(), "s::active": Labels(name_he="פעיל", name_en="Active")}))

        assert not [
            item
            for item in harness.pending(ProposalKind.LABEL)
            if item.key.startswith("table:") and "active" in item.patch["labels"]
        ]


# --- what a garbage answer looks like ------------------------------------------------------


class TestRefusals:
    """A model answers garbage; the validator must say which field and why, and queue nothing."""

    @pytest.mark.parametrize(
        ("overrides", "code", "field"),
        [
            ({"synonyms_en": ('{"a": 1}',)}, RejectionCode.NOT_WORDS, "synonyms_en"),
            (
                {"synonyms_en": ("SELECT * FROM customers;",)},
                RejectionCode.NOT_WORDS,
                "synonyms_en",
            ),
            ({"name_en": "12345"}, RejectionCode.NOT_WORDS, "name_en"),
            ({"synonyms_he": ("DROP{",)}, RejectionCode.NOT_WORDS, "synonyms_he"),
            ({"synonyms_en": ("a" * 65,)}, RejectionCode.TOO_LONG, "synonyms_en"),
            ({"synonyms_en": ("{",)}, RejectionCode.NOT_WORDS, "synonyms_en"),
        ],
    )
    def test_a_word_that_is_not_a_word_is_refused_with_its_field(
        self,
        harness: Harness,
        overrides: dict[str, object],
        code: RejectionCode,
        field: str,
    ) -> None:
        broken = Labels.model_validate(
            {"name_en": "Customer status", "name_he": "סטטוס לקוח", **overrides}
        )
        harness.run(both_runs(answers(status_code=broken)))

        assert code.value in codes(harness.stage.rejections)
        refusal = next(item for item in harness.stage.rejections if item.field == field)
        assert refusal.target == "crm.customers.status_code"
        assert refusal.detail
        assert all(item.key != "column:crm.customers.status_code" for item in harness.pending())

    def test_nothing_named_means_nothing_queued(self, harness: Harness) -> None:
        empty = Labels(description_en="Something about a customer.")
        harness.run(both_runs(answers(table=empty)))

        assert RejectionCode.NO_LABEL.value in codes(harness.stage.rejections)
        assert all(item.key != "table:crm.customers" for item in harness.pending())
        assert {item.key for item in harness.pending(ProposalKind.NAME)} == {
            "crm.customers.cust_id",
            "crm.customers.status_code",
            "crm.customers.email_address",
        }


# --- a measure is a target of its own ------------------------------------------------------


class TestMeasureTarget:
    def test_a_measure_is_targeted_and_its_labels_are_approvable(self, measured: Harness) -> None:
        measured.run(both_runs(measure_stage_answers()))

        label = next(
            item
            for item in measured.pending(ProposalKind.LABEL)
            if item.key == f"measure:{MEASURE}"
        )

        assert label.patch["labels"] == ["מסגרת אשראי", "Credit limit", "מסגרת", "Limit"]

        measured.queue().approve(label.id, by="data-team")

        assert load(measured.root).labels[f"measure:{MEASURE}"] == (
            "מסגרת אשראי",
            "Credit limit",
            "מסגרת",
            "Limit",
        )

    def test_a_measure_fact_sheet_lists_the_columns_it_declares(self, measured: Harness) -> None:
        sheet = measured.stage.fact_sheet(Target(kind=TargetKind.MEASURE, key=MEASURE))

        assert sheet.measure_sources == ("crm.customers.credit_limit",)
        assert sheet.kind is TargetKind.MEASURE
        assert sheet.source == ""

    def test_a_measure_may_not_reference_a_column_it_does_not_declare(
        self, measured: Harness
    ) -> None:
        invented = Labels(name_en="Credit limit of `limit_k`", name_he="מסגרת אשראי")
        measured.run(both_runs({**measure_stage_answers(), MEASURE: invented}))

        assert RejectionCode.MISSING_COLUMN.value in codes(measured.stage.rejections)

    def test_a_measure_may_reference_a_column_of_the_table_it_reads(
        self, measured: Harness
    ) -> None:
        """``credit_limit`` really is a column of ``crm.customers``; only inventions are refused."""
        harness = Harness(
            measured.root / "sibling",
            measures={MEASURE: _measure()},
        )
        real = Labels(
            name_en="Credit limit (`credit_limit`)", name_he="מסגרת אשראי", synonyms_en=("Limit",)
        )
        harness.run(both_runs({**measure_stage_answers(), MEASURE: real}))

        assert codes(harness.stage.rejections) == set()

    def test_the_measure_counters_say_what_was_targeted(self, measured: Harness) -> None:
        measured.run(both_runs(measure_stage_answers()))

        assert measured.stage.name == "A1+A2"
        pending = measured.pending(ProposalKind.LABEL)
        assert any(item.key == f"measure:{MEASURE}" for item in pending)


# --- the gap this stage cannot close -------------------------------------------------------


class TestTheStatedGap:
    def test_a_name_proposal_still_cannot_be_approved(self, harness: Harness) -> None:
        """``SemanticLayer`` declares no ``names`` section, so approval is refused and rolled back.

        Pinned so nobody believes a display name reaches the layer today: the labels payload is the
        one that works until the layer gains a place for a name and a description.
        """
        harness.run(both_runs(answers()))
        named = next(item for item in harness.pending(ProposalKind.NAME))

        with pytest.raises(ReviewError, match="approval refused"):
            harness.queue().approve(named.id, by="data-team")

        assert not (harness.root / "curated" / "names.yaml").exists()

    def test_a_synonym_proposal_for_a_table_shares_the_same_gap(self, harness: Harness) -> None:
        harness.run(both_runs(answers()))
        synonym = next(
            item for item in harness.pending(ProposalKind.SYNONYM) if item.section == "names"
        )

        with pytest.raises(ReviewError, match="approval refused"):
            harness.queue().approve(synonym.id, by="data-team")


# --- the pure rules, as a table ------------------------------------------------------------


class TestPureHelpers:
    @pytest.mark.parametrize(
        ("word", "accepted"),
        [
            ("לקוחות", True),
            ("Customers", True),
            ("credit limit", True),
            ("customer's status", True),
            ('כתובת דוא"ל', True),
            ("", False),
            ("   ", False),
            ("12345", False),
            ("--", False),
            ('{"a": 1}', False),
            ("SELECT * FROM x;", False),
            ("a{b}", False),
            ("a|b", False),
            ("new\nline", False),
            ("x" * 65, False),
        ],
    )
    def test_label_word_problem(self, word: str, accepted: bool) -> None:
        assert (label_word_problem(word) is None) is accepted
        if not accepted:
            assert label_word_problem(word)

    @pytest.mark.parametrize(
        ("kind", "key", "expected"),
        [
            (TargetKind.TABLE, "crm.customers", "table:crm.customers"),
            (TargetKind.COLUMN, "crm.customers.cust_id", "column:crm.customers.cust_id"),
            (TargetKind.MEASURE, "credit_limit", "measure:credit_limit"),
            (TargetKind.VALUE, "customer_status", None),
        ],
    )
    def test_label_key(self, kind: TargetKind, key: str, expected: str | None) -> None:
        assert label_key(Target(kind=kind, key=key)) == expected
