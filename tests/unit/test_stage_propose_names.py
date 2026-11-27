"""Stages A1+A2 (S11): the model proposes, these validators decide, a human approves.

The acceptance criteria, one class each:

* :class:`TestAccepted` -- with ``FakeLLM`` scripted answers, valid proposals are queued with their
  evidence, a computed confidence, and the model, prompt version and temperature they came from.
* :class:`TestRefusals` -- a colliding synonym, a description naming a column that does not exist, a
  Hebrew label that normalises to nothing, a copied observed value and a wrong-script label are all
  refused with a reason, counted, and not queued.
* :class:`TestConfidence` -- two runs that disagree score lower than two runs that agree, and the
  same scripted answers score the same score every time.
* :class:`TestIdempotence` -- a rerun with unchanged inputs is skipped by the runner and writes
  nothing; nothing under ``curated/`` is ever touched by the stage.

The pure helpers get a table of cases of their own (:class:`TestPureHelpers`), because they are what
the whole rule rests on.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mdqm.llm.fake import FakeLLM
from mdqm.models.catalog import CatalogSnapshot, Column, NormalizedType, Table
from mdqm.pipeline.core import PipelineContext, StageReport, StageStatus, run_pipeline
from mdqm.pipeline.stages.profiling import ColumnProfile, SourceProfile, TableProfile, TopValue
from mdqm.pipeline.stages.propose_names import (
    AGREEMENT_FIELDS,
    LABEL_LIMIT,
    PROMPT_VERSION,
    STAGE_NAME,
    TEMPERATURE,
    Labels,
    ProposeNamesStage,
    RejectionCode,
    Target,
    TargetKind,
    agreement,
    column_references,
    confidence,
    looks_personal,
    render_prompt,
    rotated,
)
from mdqm.review.queue import ProposalKind, ProposalStatus, ReviewQueue
from mdqm.semantic.models import ValueMap
from mdqm.store.db import Store

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)

#: One categorical field with two values, the first of which already answers to "פעיל".
VALUE_MAP = ValueMap(
    canonical=("active", "closed"),
    synonyms={"active": ("פעיל",)},
    sources={"crm.customers.status_code": {"A": "active", "C": "closed"}},
)


# --- fixtures ------------------------------------------------------------------------------


def make_catalog() -> CatalogSnapshot:
    """One table with three columns: a key, a categorical code, and a column of personal data.

    ``email`` is what the personal-data validator is about: a column the model must name without
    copying a row out of it.
    """
    return CatalogSnapshot(
        source="crm",
        captured_at=NOW,
        tables=(
            Table(
                name="customers",
                columns=(
                    Column(
                        name="cust_id",
                        native_type="integer",
                        normalized_type=NormalizedType.INTEGER,
                        nullable=False,
                    ),
                    Column(
                        name="status_code",
                        native_type="text",
                        normalized_type=NormalizedType.STRING,
                    ),
                    Column(
                        name="email_address",
                        native_type="text",
                        normalized_type=NormalizedType.STRING,
                    ),
                ),
            ),
        ),
    )


def make_profile() -> SourceProfile:
    """D5 evidence: ``cust_id`` values are identifiers, ``status_code`` is categorical."""
    return SourceProfile(
        source="crm",
        captured_at=NOW,
        tables=(
            TableProfile(
                table="customers",
                rows_estimate=100,
                sample_rows_cap=100,
                sample_complete=True,
                columns=(
                    ColumnProfile(
                        name="cust_id",
                        normalized_type=NormalizedType.INTEGER,
                        distinct=100,
                        null_rate=0.0,
                        top_values=(TopValue(value="100012345", count=3),),
                    ),
                    ColumnProfile(
                        name="status_code",
                        normalized_type=NormalizedType.STRING,
                        distinct=3,
                        null_rate=0.0,
                        top_values=(
                            TopValue(value="A", count=5),
                            TopValue(value="C", count=2),
                        ),
                    ),
                    ColumnProfile(
                        name="email_address",
                        normalized_type=NormalizedType.STRING,
                        distinct=95,
                        null_rate=0.05,
                        top_values=(TopValue(value="dana@example.com", count=1),),
                    ),
                ),
            ),
        ),
    )


def answers(
    table: Labels | None = None,
    cust_id: Labels | None = None,
    status_code: Labels | None = None,
    email_address: Labels | None = None,
) -> dict[str, Labels]:
    """The first run's answer per target; the second run repeats it unless a test says otherwise."""
    chosen = {
        "crm.customers": table
        or Labels(
            name_he="לקוחות",
            name_en="Customers",
            description_he="טבלת הלקוחות של המערכת.",
            description_en="One row per customer.",
            synonyms_he=("אנשי קשר",),
            synonyms_en=("Clients",),
        ),
        "crm.customers.cust_id": cust_id
        or Labels(
            name_he="מזהה לקוח",
            name_en="Customer ID",
            description_en="The unique number of a customer.",
            description_he="מספר הלקוח במערכת.",
            synonyms_he=("מספר לקוח",),
            synonyms_en=("Customer Number",),
        ),
        "crm.customers.status_code": status_code
        or Labels(
            name_he="סטטוס לקוח",
            name_en="Customer status",
            description_en="The state of the customer in the CRM.",
            description_he="מצב הלקוח במערכת.",
            synonyms_en=("State",),
        ),
        "crm.customers.email_address": email_address
        or Labels(
            name_he="כתובת דוא\"ל",
            name_en="Email address",
            description_en="The customer's contact address.",
            description_he="כתובת הדוא\"ל של הלקוח.",
            synonyms_en=("Email",),
        ),
    }
    return {key: value for key, value in chosen.items() if value is not None}


def reply_from(
    runs: Mapping[str, tuple[Labels, Labels] | Labels],
) -> Callable[[str], Labels]:
    """A FakeLLM reply that reads the target and the rotation out of the prompt it is given.

    Reading the prompt is the point: the test asserts on the *fact sheet the stage built*, so a
    reply cannot be served to the wrong target by accident. A value is keyed ``map::value``, because
    every value of one map shares the map's key.
    """

    def reply(prompt: str) -> Labels:
        sheet = json.loads(prompt[prompt.index("{"):])
        key = str(sheet["key"])
        if sheet.get("value") is not None:
            key = f"{key}::{sheet['value']}"
        scripted = runs[key]
        if isinstance(scripted, Labels):
            return scripted
        return scripted[int(sheet["rotation"])]

    return reply


def both_runs(runs: Mapping[str, Labels]) -> Mapping[str, tuple[Labels, Labels]]:
    """The same answer twice: perfect agreement, so confidence earns the agreement weight."""
    return {key: (value, value) for key, value in runs.items()}


class Harness:
    """One stage, one context: the store, the semantic root and the fake clock in a temp dir."""

    def __init__(
        self,
        root: Path,
        *,
        catalogs: Mapping[str, CatalogSnapshot] | None = None,
        profiles: Mapping[str, SourceProfile] | None = None,
        value_maps: Mapping[str, ValueMap] | None = None,
        known_terms: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        self.root = root
        self.store = Store(root / "metadata.sqlite")
        self.stage = ProposeNamesStage(
            catalogs if catalogs is not None else {"crm": make_catalog()},
            profiles=profiles if profiles is not None else {"crm": make_profile()},
            value_maps=value_maps,
            known_terms=known_terms,
        )
        self.ctx = PipelineContext(
            store=self.store,
            exec_ts=NOW,
            semantic_root=root,
            llm=FakeLLM(),
        )

    @property
    def llm(self) -> FakeLLM:
        llm = self.ctx.llm
        assert isinstance(llm, FakeLLM)
        return llm

    def run(self, runs: Mapping[str, tuple[Labels, Labels] | Labels]) -> StageReport:
        """Queue one scripted pass and return the pipeline's report for the stage."""
        calls = 2 * len(self.stage.targets())
        self.llm.queue(*([reply_from(runs)] * calls))
        result = run_pipeline([self.stage], self.ctx)
        return result.reports[0]

    def queue(self) -> ReviewQueue:
        return ReviewQueue(self.store, self.root)


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    yield Harness(tmp_path)


def codes(rejections: tuple[object, ...]) -> set[str]:
    """The rejection codes, so a test asserts on a set instead of on an ordering."""
    return {item.code.value for item in rejections}


# --- accepted proposals --------------------------------------------------------------------


class TestAccepted:
    """A valid answer reaches the queue with everything a human needs to decide."""

    def test_every_accepted_target_is_queued_as_pending(self, harness: Harness) -> None:
        report = harness.run(both_runs(answers()))
        assert report.status is StageStatus.RAN
        assert report.counts["created"] == 8, report.counts  # table + 3 columns, name + synonyms
        pending = harness.queue().pending()
        assert len(pending) == 8
        assert {item.status for item in pending} == {ProposalStatus.PENDING}
        assert {item.proposed_by for item in pending} == {STAGE_NAME}

    def test_the_patch_carries_both_languages_and_the_provenance(self, harness: Harness) -> None:
        harness.run(both_runs(answers()))
        named = next(
            item
            for item in harness.queue().pending()
            if item.key == "crm.customers.status_code" and item.kind is ProposalKind.NAME
        )
        assert named.patch["name"] == {"he": "סטטוס לקוח", "en": "Customer status"}
        assert named.patch["description"]["he"] == "מצב הלקוח במערכת."
        provenance = named.patch["provenance"]
        assert provenance["detected_by"] == STAGE_NAME
        assert 0.0 < provenance["match_rate"] <= 1.0

    def test_the_evidence_records_model_prompt_version_and_temperature(
        self, harness: Harness
    ) -> None:
        harness.run(both_runs(answers()))
        evidence = harness.queue().pending()[0].evidence
        assert f"model: {harness.llm.model_name}" in evidence
        assert f"prompt_version: {PROMPT_VERSION}" in evidence
        assert f"temperature: {TEMPERATURE}" in evidence
        assert any("label fields identical" in line for line in evidence)

    def test_a_profiled_target_earns_the_evidence_weight(self, harness: Harness) -> None:
        harness.run(both_runs(answers()))
        by_key = {item.key: item.confidence for item in harness.queue().pending()}
        assert by_key["crm.customers.status_code"] == 1.0, "D5 measured it and both runs agreed"
        assert by_key["crm.customers"] == 0.9, "the table itself was never profiled"

    def test_a_value_map_value_is_proposed_as_a_synonym_group(self, tmp_path: Path) -> None:
        harness = Harness(tmp_path, catalogs={}, value_maps={"crm.status": VALUE_MAP})
        harness.run(
            {
                "crm.status::active": Labels(
                    name_he="פעיל", name_en="Active", synonyms_en=("Current",)
                ),
                "crm.status::closed": Labels(
                    name_he="סגור", name_en="Closed", synonyms_he=("מוסגר",)
                ),
            }
        )
        pending = harness.queue().pending()
        assert len(pending) == 2
        assert {item.section for item in pending} == {"value_maps"}
        assert {item.kind for item in pending} == {ProposalKind.SYNONYM}
        closed = next(item for item in pending if "closed" in item.patch["synonyms"])
        # "Closed" is dropped: it is the canonical value itself, already this value's own term.
        assert closed.patch["synonyms"]["closed"] == ["מוסגר", "סגור"]


# --- refusals ------------------------------------------------------------------------------


class TestRefusals:
    """Every validator refuses with a reason and counts the refusal; nothing enters the queue."""

    def test_a_colliding_synonym_is_refused_and_names_the_collision(self, tmp_path: Path) -> None:
        harness = Harness(tmp_path, catalogs={}, value_maps={"crm.status": VALUE_MAP})
        report = harness.run(
            {
                "crm.status::active": Labels(
                    name_he="פעיל", name_en="Active", synonyms_en=("Current",)
                ),
                # "closed" tries to answer to the label that already means "active".
                "crm.status::closed": Labels(
                    name_he="סגור", name_en="Closed", synonyms_he=("פעיל",)
                ),
            }
        )
        assert codes(harness.stage.rejections) == {RejectionCode.COLLISION.value}
        detail = harness.stage.rejections[0].detail
        assert "פעיל" in detail and "crm.status::active" in detail
        assert report.counts["rejected"] == 1
        queued = harness.queue().pending()
        assert len(queued) == 1, "only the value that did not collide is queued"
        assert "closed" not in queued[0].patch["synonyms"]

    def test_a_description_naming_a_missing_column_is_refused(self, harness: Harness) -> None:
        harness.run(
            both_runs(
                answers(
                    cust_id=Labels(
                        name_he="מזהה לקוח",
                        name_en="Customer ID",
                        description_en="The customer key, taken from `customer_ref`.",
                    )
                )
            )
        )
        missing = [item for item in harness.stage.rejections
                   if item.code is RejectionCode.MISSING_COLUMN]
        assert missing, harness.stage.rejections
        assert "'customer_ref'" in missing[0].detail
        assert missing[0].target == "crm.customers.cust_id"
        keys = {item.key for item in harness.queue().pending()}
        assert "crm.customers.cust_id" not in keys, "nothing about the refused target is queued"

    def test_a_hebrew_label_that_normalises_to_nothing_is_refused(self, harness: Harness) -> None:
        harness.run(
            both_runs(
                answers(
                    status_code=Labels(
                        name_he="\u05b4\u05b0",  # niqqud only: Hebrew block, no letters
                        name_en="Customer status",
                        description_en="The state of the customer in the CRM.",
                    )
                )
            )
        )
        assert codes(harness.stage.rejections) == {
            RejectionCode.EMPTY_AFTER_NORMALISATION.value
        }
        assert "no letters left" in harness.stage.rejections[0].detail

    def test_a_copied_identifier_is_refused_as_personal_data(self, harness: Harness) -> None:
        harness.run(
            both_runs(
                answers(
                    email_address=Labels(
                        name_he="כתובת דוא\"ל",
                        name_en="Email address",
                        description_en="The customer's contact address.",
                        # a value D5 observed for this very column, copied into a synonym
                        synonyms_en=("dana@example.com",),
                    )
                )
            )
        )
        assert codes(harness.stage.rejections) == {RejectionCode.PERSONAL_DATA.value}
        assert "copied observed value" in harness.stage.rejections[0].detail

    @pytest.mark.parametrize(
        ("labels", "code"),
        [
            (Labels(name_he="Customer", name_en="Customers"), RejectionCode.NOT_HEBREW.value),
            (Labels(name_he="לקוחות", name_en="לקוחות"), RejectionCode.NOT_ENGLISH.value),
            (
                Labels(name_he="לקוחות", name_en="C" * (LABEL_LIMIT + 1)),
                RejectionCode.TOO_LONG.value,
            ),
            (Labels(name_he="", name_en=""), RejectionCode.NO_LABEL.value),
        ],
    )
    def test_a_bad_field_is_refused_with_its_own_code(
        self, harness: Harness, labels: Labels, code: str
    ) -> None:
        harness.run(both_runs(answers(table=labels)))
        assert code in codes(harness.stage.rejections)
        assert "crm.customers" not in {item.key for item in harness.queue().pending()}

    def test_the_report_names_each_refusal_in_words(self, harness: Harness) -> None:
        report = harness.run(
            both_runs(answers(table=Labels(name_he="Customers", name_en="Customers")))
        )
        assert any(line.startswith("rejected crm.customers") for line in report.skipped), (
            report.skipped
        )


# --- confidence ----------------------------------------------------------------------------


class TestConfidence:
    """Confidence comes from the validators and two runs, never from the model's own claim."""

    def test_disagreement_lowers_the_score_deterministically(self, tmp_path: Path) -> None:
        agree = Harness(tmp_path / "agree")
        agree_scores = agree_scores_of(agree)

        disagree = Harness(tmp_path / "disagree")
        differed = answers(
            status_code=Labels(
                name_he="מצב חשבון",
                name_en="Account state",
                description_en="The state of the customer in the CRM.",
                synonyms_en=("Condition",),
            )
        )
        runs = {
            key: (value, value) for key, value in answers().items()
        }
        runs["crm.customers.status_code"] = (answers()["crm.customers.status_code"],
                                            differed["crm.customers.status_code"])
        disagree_scores = agree_scores_of(disagree, runs=runs)

        assert disagree_scores["crm.customers.status_code"] < (
            agree_scores["crm.customers.status_code"]
        )
        # Same scripted answers, run again from scratch: exactly the same numbers.
        assert agree_scores_of(Harness(tmp_path / "agree-again")) == agree_scores

    def test_the_ladder_is_fixed(self) -> None:
        assert confidence(1.0, evidenced=True) == 1.0
        assert confidence(1.0, evidenced=False) == 0.9
        assert confidence(0.0, evidenced=False) == 0.5
        assert confidence(0.5, evidenced=True) == 0.8
        assert confidence(-3.0, evidenced=False) == 0.5
        assert confidence(4.0, evidenced=True) == 1.0


def agree_scores_of(
    harness: Harness, *, runs: Mapping[str, tuple[Labels, Labels]] | None = None
) -> dict[str, float]:
    """Run one pass and return the confidence per target key (the first proposal of each)."""
    harness.run(runs if runs is not None else both_runs(answers()))
    scores: dict[str, float] = {}
    for item in harness.queue().pending():
        scores.setdefault(item.key, item.confidence)
    return scores


# --- idempotence ---------------------------------------------------------------------------


class TestIdempotence:
    """A rerun with unchanged inputs writes nothing, and the curated layer is never touched."""

    def test_a_second_run_is_skipped_and_asks_no_model(self, harness: Harness) -> None:
        first = harness.run(both_runs(answers()))
        calls = len(harness.llm.calls)
        pending = len(harness.queue().pending())
        assert first.status is StageStatus.RAN

        second = harness.run(both_runs(answers()))
        assert second.status is StageStatus.SKIPPED
        assert "inputs unchanged" in second.skipped[0]
        assert len(harness.llm.calls) == calls, "a skipped stage must not call the model"
        assert len(harness.queue().pending()) == pending
        assert first.inputs_hash == second.inputs_hash

    def test_the_stage_writes_nothing_to_curated(self, harness: Harness) -> None:
        harness.run(both_runs(answers()))
        curated = harness.root / "curated"
        assert not curated.exists() or list(curated.iterdir()) == []
        assert harness.root.joinpath("generated").exists() is False or (
            list(harness.root.joinpath("generated").iterdir()) == []
        )

    def test_a_changed_catalog_changes_the_input_hash(self, harness: Harness) -> None:
        before = harness.stage.inputs_hash(harness.ctx)
        bigger = ProposeNamesStage(
            {
                "crm": CatalogSnapshot(
                    source="crm",
                    captured_at=NOW,
                    tables=(*make_catalog().tables, Table(name="orders")),
                )
            }
        )
        assert bigger.inputs_hash(harness.ctx) != before

    def test_without_a_model_nothing_is_proposed_and_it_says_so(self, tmp_path: Path) -> None:
        harness = Harness(tmp_path)
        context = PipelineContext(
            store=harness.store, exec_ts=NOW, semantic_root=tmp_path, llm=None
        )
        report = run_pipeline([harness.stage], context).reports[0]
        assert report.counts["proposals"] == 0
        assert any("no model client configured" in line for line in report.skipped)


# --- the pure helpers ----------------------------------------------------------------------


class TestPureHelpers:
    """The validators and the prompt help are pure, so they are tested as tables of cases."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("The key, in `cust_id`.", ("cust_id",)),
            ("Uses start_time and end_time.", ("start_time", "end_time")),
            ("One row per customer.", ()),
            ("See `cust_id`, then `status_code`.", ("cust_id", "status_code")),
            ("`cust_id` again as cust_id", ("cust_id",)),
        ],
    )
    def test_column_references(self, text: str, expected: tuple[str, ...]) -> None:
        assert column_references(text) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("100012345", True),
            ("dana@example.com", True),
            ("x" * 30, True),
            ("Active", False),
            ("פעיל", False),
            ("לקוח מזוהה", False),
        ],
    )
    def test_looks_personal(self, value: str, expected: bool) -> None:
        assert looks_personal(value) is expected

    @pytest.mark.parametrize(
        ("values", "rotation", "expected"),
        [
            ((), 3, ()),
            (("a",), 5, ("a",)),
            (("a", "b", "c"), 0, ("a", "b", "c")),
            (("a", "b", "c"), 1, ("b", "c", "a")),
            (("a", "b", "c"), 4, ("b", "c", "a")),
        ],
    )
    def test_rotated(
        self, values: tuple[str, ...], rotation: int, expected: tuple[str, ...]
    ) -> None:
        assert rotated(values, rotation) == expected

    def test_the_prompt_carries_the_sheet_and_never_a_braced_template(
        self, harness: Harness
    ) -> None:
        sheet = harness.stage.fact_sheet(
            Target(
                kind=TargetKind.COLUMN,
                key="crm.customers.cust_id",
                source="crm",
                table="customers",
                column="cust_id",
            )
        )
        text = render_prompt(sheet)
        assert text.endswith("}"), "the sheet is the trailing JSON object"
        document = json.loads(text[text.index("{"):])
        assert document["key"] == "crm.customers.cust_id"
        assert document["columns"] == ["cust_id", "status_code", "email_address"]
        assert document["observed_values"] == ["100012345"]
        assert document["rotation"] == 0

    @pytest.mark.parametrize("field", AGREEMENT_FIELDS)
    def test_agreement_folds_hebrew_and_case(self, field: str) -> None:
        first = Labels(name_he="פָּעִיל", name_en="Active", synonyms_en=("Live",))
        second = Labels(name_he="פעיל", name_en="active", synonyms_en=("live",))
        assert agreement(first, second) == 1.0, field

    def test_agreement_is_zero_for_opposite_names(self) -> None:
        first = Labels(
            name_he="לקוחות",
            name_en="Customers",
            synonyms_he=("אנשי קשר",),
            synonyms_en=("Clients",),
        )
        second = Labels(
            name_he="אנשים", name_en="People", synonyms_he=("אנש",), synonyms_en=("Folks",)
        )
        assert agreement(first, second) == 0.0

    def test_the_fact_sheet_rotates_its_lists_not_its_facts(self, harness: Harness) -> None:
        target = Target(
            kind=TargetKind.COLUMN,
            key="crm.customers.status_code",
            source="crm",
            table="customers",
            column="status_code",
        )
        first = harness.stage.fact_sheet(target)
        second = harness.stage.fact_sheet(target, rotation=1)
        assert first.columns == ("cust_id", "status_code", "email_address")
        assert second.columns == ("status_code", "email_address", "cust_id")
        assert first.observed_values == ("A", "C")
        assert second.observed_values == ("C", "A")
        assert first.existing_terms == second.existing_terms or sorted(
            first.existing_terms
        ) == sorted(second.existing_terms)


# --- a small guard on the store the queue uses ---------------------------------------------


def test_the_queue_table_exists_before_the_stage_writes(harness: Harness) -> None:
    """A stage must not depend on another stage having migrated the store first."""
    conn: sqlite3.Connection = harness.store.connect()
    assert conn is not None
    harness.run(both_runs(answers()))
    assert harness.queue().pending()
