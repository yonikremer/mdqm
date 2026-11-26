"""The review queue (S12): proposals wait, a human decides, and only approval writes curated YAML.

The acceptance criteria this file proves, one class each:

* :class:`TestApproveWritesCurated` -- approval reaches ``curated/*.yaml`` with provenance and the
  loader (C04) accepts the result.
* :class:`TestNegativeMemory` -- a rejection with identical evidence is not resurfaced, and is
  resurfaced once the evidence changes.
* :class:`TestExportPatch` -- the exported diff applies cleanly with ``git apply`` in a temp repo.
* :class:`TestOnlyApprovalWrites` -- no other entry point (submit, reject, never_ask, questions,
  approve that fails validation) touches the curated layer.
* :class:`TestAudit` -- an approval and a rejection are both audited, hash-chained.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner
from typer.testing import Result as CliRunnerResult

from mdqm.review.queue import (
    Proposal,
    ProposalKind,
    ProposalStatus,
    Question,
    ReviewError,
    ReviewQueue,
    SubmitAction,
    evidence_hash,
    proposal_id,
)
from mdqm.semantic.loader import load
from mdqm.store.audit import AuditLog
from mdqm.store.db import Store

#: What the CLI fixture seeds: the queue and the id of the one pending proposal.
Seeded = tuple[ReviewQueue, str]

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


class FakeClock:
    """A clock the test drives, so no test sleeps and every timestamp is deterministic."""

    def __init__(self, start: datetime = NOW) -> None:
        self._now = start
        self._step = timedelta(0)

    def now(self) -> datetime:
        return self._now

    def advance(self, by: timedelta) -> None:
        self._now += by

    def tick(self, by: timedelta = timedelta(seconds=1)) -> None:
        self._step = by
        self._now += by


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A semantic root with a curated directory, as the loader expects."""
    (tmp_path / "curated").mkdir()
    return tmp_path


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "metadata.db")


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def queue(store: Store, root: Path, clock: FakeClock) -> Iterator[ReviewQueue]:
    opened = ReviewQueue(store, root, clock=clock)
    yield opened
    store.close()


def make(
    *,
    kind: ProposalKind = ProposalKind.VALUE_MAP,
    section: str = "value_maps",
    key: str = "customer_status",
    field: str = "",
    patch: dict[str, object] | None = None,
    evidence: tuple[str, ...] = ("crm.customers.status has 3 values",),
    confidence: float = 0.95,
    collection: bool = False,
    at: datetime = NOW,
) -> Proposal:
    """Build a proposal through the real id function, as a pipeline would."""
    body = patch if patch is not None else {"canonical": ["active", "churned", "suspended"]}
    return Proposal(
        id=proposal_id(kind, section, key, field, body),
        kind=kind,
        section=section,
        key=key,
        field=field,
        collection=collection,
        patch=body,
        evidence=evidence,
        confidence=confidence,
        proposed_at=at,
    )


class TestIds:
    """The id is the content, so a rerun updates one row instead of filling the queue."""

    def test_same_content_same_id(self) -> None:
        assert make().id == make().id

    @pytest.mark.parametrize(
        "changed",
        [
            {"kind": ProposalKind.SYNONYM},
            {"section": "tables"},
            {"key": "other"},
            {"patch": {"canonical": ["active"]}},
        ],
    )
    def test_any_content_change_changes_the_id(self, changed: dict[str, object]) -> None:
        assert make().id != make(**changed).id  # type: ignore[arg-type]

    def test_evidence_is_not_part_of_the_id(self) -> None:
        """Otherwise a rerun with new evidence would open a second question about one fact."""
        assert make(evidence=("a",)).id == make(evidence=("b",)).id

    def test_evidence_hash_is_order_insensitive(self) -> None:
        assert evidence_hash(("a", "b")) == evidence_hash(("b", "a"))
        assert evidence_hash(("a",)) != evidence_hash(("a", "b"))


class TestSubmit:
    """Proposals accumulate, and resubmitting the same fact is not a new question."""

    def test_submit_creates_one_row(self, queue: ReviewQueue) -> None:
        result = queue.submit(make())
        assert result.action is SubmitAction.CREATED
        assert result.asks_a_human
        assert len(queue.pending()) == 1

    def test_resubmitting_identical_content_is_unchanged(self, queue: ReviewQueue) -> None:
        queue.submit(make())
        again = queue.submit(make())
        assert again.action is SubmitAction.UNCHANGED
        assert not again.asks_a_human
        assert len(queue.pending()) == 1

    def test_pending_with_new_evidence_keeps_newest(self, queue: ReviewQueue) -> None:
        first = queue.submit(make(evidence=("old",)))
        second = queue.submit(make(evidence=("new",)))
        assert second.action is SubmitAction.UNCHANGED
        assert queue.get(first.proposal.id).evidence == ("new",)

    def test_get_unknown_id_raises(self, queue: ReviewQueue) -> None:
        with pytest.raises(ReviewError, match="no proposal"):
            queue.get("0" * 64)


class TestNegativeMemory:
    """Acceptance: a rejected proposal is not resurfaced with identical evidence, and is when it
    changes."""

    def test_rejection_is_remembered(self, queue: ReviewQueue) -> None:
        created = queue.submit(make())
        queue.reject(created.proposal.id, by="data-team")
        assert queue.get(created.proposal.id).status is ProposalStatus.REJECTED
        assert queue.pending() == ()

    def test_identical_evidence_is_suppressed(self, queue: ReviewQueue) -> None:
        created = queue.submit(make(evidence=("evidence one",)))
        queue.reject(created.proposal.id, by="data-team")
        again = queue.submit(make(evidence=("evidence one",)))
        assert again.action is SubmitAction.SUPPRESSED
        assert not again.asks_a_human
        assert queue.pending() == ()

    def test_changed_evidence_resurfaces(self, queue: ReviewQueue) -> None:
        created = queue.submit(make(evidence=("evidence one",)))
        queue.reject(created.proposal.id, by="data-team")
        again = queue.submit(make(evidence=("new measurement: 99.8% match",)))
        assert again.action is SubmitAction.RESURFACED
        assert again.asks_a_human
        assert len(queue.pending()) == 1
        assert again.proposal.evidence == ("new measurement: 99.8% match",)

    def test_resurfaced_row_clears_the_old_decision(self, queue: ReviewQueue) -> None:
        created = queue.submit(make(evidence=("one",)))
        queue.reject(created.proposal.id, by="data-team")
        queue.submit(make(evidence=("two",)))
        resurfaced = queue.get(created.proposal.id)
        assert resurfaced.decided_by is None
        assert resurfaced.decided_at is None

    def test_never_ask_survives_new_evidence(self, queue: ReviewQueue) -> None:
        created = queue.submit(make(evidence=("one",)))
        queue.never_ask(created.proposal.id, by="data-team")
        again = queue.submit(make(evidence=("completely different proof",)))
        assert again.action is SubmitAction.SUPPRESSED
        assert queue.pending() == ()

    def test_approved_is_not_reopened(self, queue: ReviewQueue) -> None:
        created = queue.submit(make())
        queue.approve(created.proposal.id, by="data-team")
        again = queue.submit(make(evidence=("brand new",)))
        assert again.action is SubmitAction.ALREADY_APPROVED
        assert queue.pending() == ()


class TestApproveWritesCurated:
    """Acceptance: approval puts the item in curated YAML with provenance; the loader accepts it."""

    def test_approve_writes_the_patch_and_provenance(self, queue: ReviewQueue, root: Path) -> None:
        created = queue.submit(make())
        queue.approve(created.proposal.id, by="data-team")
        text = (root / "curated" / "value_maps.yaml").read_text(encoding="utf-8")
        assert "customer_status" in text
        assert "approved_by: data-team" in text
        assert "approved_at" in text

    def test_approved_layer_loads(self, queue: ReviewQueue, root: Path) -> None:
        created = queue.submit(make())
        queue.approve(created.proposal.id, by="data-team")
        layer = load(root)
        assert "customer_status" in layer.value_maps
        assert layer.value_maps["customer_status"].provenance is not None
        assert layer.value_maps["customer_status"].provenance.approved_by == "data-team"

    def test_approval_marks_the_row_and_stamps_the_decision(
        self, queue: ReviewQueue, clock: FakeClock
    ) -> None:
        created = queue.submit(make())
        clock.advance(timedelta(hours=2))
        approved = queue.approve(created.proposal.id, by="data-team")
        assert approved.status is ProposalStatus.APPROVED
        assert approved.decided_by == "data-team"
        assert approved.decided_at == clock.now()
        assert queue.pending() == ()

    def test_double_approval_is_refused(self, queue: ReviewQueue) -> None:
        created = queue.submit(make())
        queue.approve(created.proposal.id, by="data-team")
        with pytest.raises(ReviewError, match="already approved"):
            queue.approve(created.proposal.id, by="someone-else")

    def test_approval_preserves_existing_comments(self, queue: ReviewQueue, root: Path) -> None:
        """A human's comment in a curated file must survive the machine's edit around it."""
        curated = root / "curated" / "value_maps.yaml"
        curated.write_text(
            "# curated by the data team: do not lose this comment\n"
            "value_maps:\n"
            "  other_map:\n"
            "    # this map is unrelated and must keep its comment\n"
            "    canonical: [x, y]\n",
            encoding="utf-8",
        )
        created = queue.submit(make())
        queue.approve(created.proposal.id, by="data-team")
        text = curated.read_text(encoding="utf-8")
        assert "do not lose this comment" in text
        assert "this map is unrelated and must keep its comment" in text
        assert "customer_status" in text

    def test_approval_preserves_unrelated_keys(self, queue: ReviewQueue, root: Path) -> None:
        curated = root / "curated" / "value_maps.yaml"
        curated.write_text(
            "value_maps:\n  other_map:\n    canonical: [x, y]\n", encoding="utf-8"
        )
        created = queue.submit(make())
        queue.approve(created.proposal.id, by="data-team")
        layer = load(root)
        assert "other_map" in layer.value_maps
        assert "customer_status" in layer.value_maps

    def test_hebrew_survives_the_round_trip(self, queue: ReviewQueue, root: Path) -> None:
        """All files are UTF-8; the writer must not escape Hebrew into \\u sequences."""
        created = queue.submit(
            make(
                patch={
                    "canonical": ["active", "churned"],
                    "synonyms": {"active": ["פעיל"]},
                }
            )
        )
        queue.approve(created.proposal.id, by="data-team")
        text = (root / "curated" / "value_maps.yaml").read_text(encoding="utf-8")
        assert "פעיל" in text
        assert load(root).value_maps["customer_status"].synonyms["active"] == ("פעיל",)

    def test_mutability_approval_sets_approved(
        self, queue: ReviewQueue, root: Path
    ) -> None:
        created = queue.submit(
            make(
                kind=ProposalKind.MUTABILITY,
                section="tables",
                key="billing.usage_events",
                field="mutability",
                patch={"class": "append_rarely"},
            )
        )
        queue.approve(created.proposal.id, by="data-team")
        layer = load(root)
        declared = layer.tables["billing.usage_events"].mutability
        assert declared is not None
        assert declared.status.value == "approved"
        assert declared.class_ == "append_rarely"

    def test_join_is_appended_to_the_list(self, queue: ReviewQueue, root: Path) -> None:
        """``approved_joins`` is a list of named entries, matched by name, never duplicated."""
        join: dict[str, object] = {
            "name": "usage_to_customers",
            "left": "billing.usage_events.client_no",
            "right": "crm.customers.cust_id",
            "on": "LPAD(CAST({right} AS VARCHAR), 10, '0') = {left}",
            "cardinality": "many_to_one",
        }
        created = queue.submit(
            make(
                kind=ProposalKind.JOIN,
                section="approved_joins",
                key="usage_to_customers",
                patch=join,
                collection=True,
            )
        )
        queue.approve(created.proposal.id, by="data-team")
        layer = load(root)
        assert [j.name for j in layer.approved_joins] == ["usage_to_customers"]
        assert layer.approved_joins[0].provenance is not None

    def test_invalid_patch_stays_pending(self, queue: ReviewQueue, root: Path) -> None:
        """An approval that would make the layer unloadable must not land, and must not vanish."""
        bad = queue.submit(
            make(patch={"canonical": []})  # canonical needs at least one value
        )
        with pytest.raises(ReviewError, match="approval refused"):
            queue.approve(bad.proposal.id, by="data-team")
        assert not (root / "curated" / "value_maps.yaml").exists()
        assert queue.get(bad.proposal.id).status is ProposalStatus.PENDING

    def test_invalid_patch_rolls_back_existing(self, queue: ReviewQueue, root: Path) -> None:
        curated = root / "curated" / "value_maps.yaml"
        original = "value_maps:\n  other_map:\n    canonical: [x]\n"
        curated.write_text(original, encoding="utf-8")
        bad = queue.submit(make(patch={"canonical": []}))
        with pytest.raises(ReviewError, match="approval refused"):
            queue.approve(bad.proposal.id, by="data-team")
        assert curated.read_text(encoding="utf-8") == original

    def test_section_may_not_escape_the_directory(self, queue: ReviewQueue) -> None:
        with pytest.raises(ReviewError, match="not a section name"):
            queue.curated_path("../evil")


class TestOnlyApprovalWrites:
    """The heart of the gate: every other entry point leaves the curated layer untouched."""

    def curated_files(self, root: Path) -> list[Path]:
        return sorted((root / "curated").glob("*.yaml"))

    def test_submit_does_not_write(self, queue: ReviewQueue, root: Path) -> None:
        queue.submit(make())
        assert self.curated_files(root) == []

    def test_reject_does_not_write(self, queue: ReviewQueue, root: Path) -> None:
        created = queue.submit(make())
        queue.reject(created.proposal.id, by="data-team")
        assert self.curated_files(root) == []

    def test_never_ask_does_not_write(self, queue: ReviewQueue, root: Path) -> None:
        created = queue.submit(make())
        queue.never_ask(created.proposal.id, by="data-team")
        assert self.curated_files(root) == []

    def test_questions_do_not_write(self, queue: ReviewQueue, root: Path) -> None:
        queue.submit(make())
        queue.questions()
        assert self.curated_files(root) == []

    def test_pending_and_decided_do_not_write(self, queue: ReviewQueue, root: Path) -> None:
        created = queue.submit(make())
        queue.reject(created.proposal.id, by="data-team")
        queue.pending()
        queue.decided()
        queue.get(created.proposal.id)
        assert self.curated_files(root) == []

    def test_module_exposes_no_other_writer(self) -> None:
        """Guard against a future entry point that writes the curated layer without approval."""
        import mdqm.review.queue as module

        functions = {
            name
            for name, value in vars(module).items()
            if callable(value)
            and not name.startswith("__")
            and getattr(value, "__module__", None) == module.__name__
        }
        # `_write_text` is the module's single file-write; nothing else may reach the curated tree.
        assert sorted(name for name in functions if "write" in name) == ["_write_text"]
        # `_render` is pure: it returns text, so exporting a patch cannot write by accident.
        assert "_render" in functions

    def test_approve_of_a_rejected_row_still_writes(self, queue: ReviewQueue, root: Path) -> None:
        """A human may change their mind: rejection is remembered, not binding."""
        created = queue.submit(make())
        queue.reject(created.proposal.id, by="data-team")
        queue.approve(created.proposal.id, by="data-team")
        assert (root / "curated" / "value_maps.yaml").is_file()


class TestExportPatch:
    """Acceptance: export-patch output applies cleanly with ``git apply`` in a temp repo."""

    def _git(self, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - a fixed argv, no shell, in a test-only temp repo
            ["git", *args],  # noqa: S607 - git from PATH is the point of testing `git apply`
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
        )

    @pytest.fixture
    def repo(self, root: Path) -> Path:
        self._git("init", "-q", cwd=root)
        self._git("config", "user.email", "dev@example.invalid", cwd=root)
        self._git("config", "user.name", "dev", cwd=root)
        (root / "curated" / "value_maps.yaml").write_text(
            "# keeps the comment\nvalue_maps:\n  other_map:\n    canonical: [x]\n",
            encoding="utf-8",
        )
        self._git("add", ".", cwd=root)
        self._git("commit", "-q", "-m", "seed", cwd=root)
        return root

    def test_patch_applies_with_git_apply(self, queue: ReviewQueue, repo: Path) -> None:
        created = queue.submit(make())
        patch = queue.export_patch(created.proposal.id)
        assert patch, "export_patch must produce a diff"

        applied = self._git("apply", "--check", "-", cwd=repo) if False else None
        assert applied is None  # keep ruff happy about the unused variable

        patch_file = repo / "change.patch"
        patch_file.write_text(patch, encoding="utf-8")
        result = self._git("apply", "change.patch", cwd=repo)
        assert result.returncode == 0, result.stderr
        text = (repo / "curated" / "value_maps.yaml").read_text(encoding="utf-8")
        assert "customer_status" in text
        assert "keeps the comment" in text

    def test_patch_for_a_new_file_applies(self, queue: ReviewQueue, repo: Path) -> None:
        (repo / "curated" / "value_maps.yaml").unlink()
        self._git("add", "-A", cwd=repo)
        self._git("commit", "-q", "-m", "drop", cwd=repo)
        created = queue.submit(make())
        patch = queue.export_patch(created.proposal.id)
        (repo / "new.patch").write_text(patch, encoding="utf-8")
        result = self._git("apply", "new.patch", cwd=repo)
        assert result.returncode == 0, result.stderr
        assert (repo / "curated" / "value_maps.yaml").is_file()

    def test_export_does_not_write_the_curated_layer(self, queue: ReviewQueue, root: Path) -> None:
        created = queue.submit(make())
        queue.export_patch(created.proposal.id)
        assert not (root / "curated" / "value_maps.yaml").exists()

    def test_export_and_approve_agree(self, queue: ReviewQueue, root: Path) -> None:
        created = queue.submit(make())
        exported = queue.export_patch(created.proposal.id)
        queue.approve(created.proposal.id, by="data-team")
        written = (root / "curated" / "value_maps.yaml").read_text(encoding="utf-8")
        # Every added line of the diff is a line of the approved file.
        added = [
            line[1:] for line in exported.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]
        for line in added:
            assert line in written

    def test_export_of_an_approved_proposal_is_refused(self, queue: ReviewQueue) -> None:
        created = queue.submit(make())
        queue.approve(created.proposal.id, by="data-team")
        with pytest.raises(ReviewError, match="already approved"):
            queue.export_patch(created.proposal.id)


class TestBulkApproval:
    """Bulk approval of high-confidence items counts as one question."""

    def test_high_confidence_items_group_into_one_question(self, queue: ReviewQueue) -> None:
        for index in range(5):
            queue.submit(make(key=f"map_{index}", confidence=0.99))
        queue.submit(make(key="uncertain", confidence=0.4))
        questions = queue.questions(bulk_threshold=0.9)
        assert questions[0].bulk
        assert questions[0].count == 5
        assert len(questions) == 2  # the bulk question plus the uncertain single
        assert questions[0].count + sum(q.count for q in questions[1:]) == 6

    def test_bulk_approval_writes_every_selected_item(self, queue: ReviewQueue, root: Path) -> None:
        for index in range(3):
            queue.submit(make(key=f"map_{index}", confidence=0.99))
        queue.submit(make(key="unsure", confidence=0.2))
        approved = queue.approve_high_confidence(by="data-team", threshold=0.9)
        assert len(approved) == 3
        layer = load(root)
        assert set(layer.value_maps) == {"map_0", "map_1", "map_2"}
        assert [p.id for p in queue.pending()] == [make(key="unsure", confidence=0.2).id]

    def test_questions_are_deterministic(self, queue: ReviewQueue) -> None:
        for index in range(4):
            queue.submit(make(key=f"m{index}", confidence=0.5))
        assert [q.summary for q in queue.questions()] == [
            q.summary for q in queue.questions()
        ]

    def test_empty_queue_asks_nothing(self, queue: ReviewQueue) -> None:
        assert queue.questions() == ()
        assert isinstance(queue.questions(), tuple)


class TestAudit:
    """An approval and a rejection are both audited, and the chain stays intact."""

    @pytest.fixture
    def audit(self, store: Store) -> AuditLog:
        return AuditLog(store, clock=lambda: NOW)

    @pytest.fixture
    def audited(
        self, store: Store, root: Path, clock: FakeClock, audit: AuditLog
    ) -> ReviewQueue:
        return ReviewQueue(store, root, clock=clock, audit=audit)

    def test_approval_is_audited(self, audited: ReviewQueue, audit: AuditLog) -> None:
        created = audited.submit(make())
        audited.approve(created.proposal.id, by="data-team")
        rows = audit.rows()
        assert len(rows) == 1
        assert rows[0].tool == "review.approve"
        assert rows[0].user_id == "data-team"
        assert rows[0].version_hash == created.proposal.id

    def test_rejection_is_audited(self, audited: ReviewQueue, audit: AuditLog) -> None:
        created = audited.submit(make())
        audited.reject(created.proposal.id, by="data-team")
        rows = audit.rows()
        assert len(rows) == 1
        assert rows[0].tool == "review.reject"

    def test_chain_valid_across_decisions(
        self, audited: ReviewQueue, audit: AuditLog
    ) -> None:
        first = audited.submit(make(key="a", confidence=0.9))
        second = audited.submit(make(key="b", confidence=0.9))
        audited.approve(first.proposal.id, by="data-team")
        audited.reject(second.proposal.id, by="data-team")
        assert audit.verify_chain().ok
        assert len(audit.rows()) == 2

    def test_failed_approval_is_not_audited(self, audited: ReviewQueue, audit: AuditLog) -> None:
        """A refusal is not an approval; it must not leave a misleading audit row."""
        bad = audited.submit(make(patch={"canonical": []}))
        with pytest.raises(ReviewError):
            audited.approve(bad.proposal.id, by="data-team")
        assert audit.rows() == ()

    def test_queue_works_without_an_audit_log(self, queue: ReviewQueue) -> None:
        created = queue.submit(make())
        assert queue.approve(created.proposal.id, by="data-team").status is ProposalStatus.APPROVED


class TestPersistence:
    """The queue is a store, not process memory: a restart keeps every decision."""

    def test_rows_survive_reopening(self, store: Store, root: Path, clock: FakeClock) -> None:
        first = ReviewQueue(store, root, clock=clock)
        created = first.submit(make())
        first.reject(created.proposal.id, by="data-team")
        store.close()

        reopened_store = Store(store.path)
        try:
            second = ReviewQueue(reopened_store, root, clock=clock)
            reloaded = second.get(created.proposal.id)
            assert reloaded.status is ProposalStatus.REJECTED
            assert reloaded.decided_by == "data-team"
            assert reloaded.evidence == created.proposal.evidence
        finally:
            reopened_store.close()

    def test_hebrew_evidence_round_trips_through_sqlite(
        self, queue: ReviewQueue, store: Store, root: Path, clock: FakeClock
    ) -> None:
        created = queue.submit(make(evidence=("נמצאו 3 ערכים בעמודת סטטוס", "3 distinct values")))
        queue.reject(created.proposal.id, by="data-team")
        store.close()
        reopened = Store(store.path)
        try:
            again = ReviewQueue(reopened, root, clock=clock).get(created.proposal.id)
            assert again.evidence == ("נמצאו 3 ערכים בעמודת סטטוס", "3 distinct values")
        finally:
            reopened.close()


class TestCuratedPath:
    """A section is a file stem; a name that could escape the directory is refused."""

    @pytest.mark.parametrize("section", ["value_maps", "tables", "approved_joins", "a.b-c_1"])
    def test_allowed(self, queue: ReviewQueue, section: str) -> None:
        assert queue.curated_path(section).name == f"{section}.yaml"

    @pytest.mark.parametrize("section", ["../evil", "a/b", "", "a b", "a\\b"])
    def test_refused(self, queue: ReviewQueue, section: str) -> None:
        with pytest.raises(ReviewError):
            queue.curated_path(section)


def test_question_model_counts_its_proposals() -> None:
    question = Question(summary="s", proposals=("a" * 64, "b" * 64), bulk=True)
    assert question.count == 2


def test_review_error_is_a_runtime_error() -> None:
    assert issubclass(ReviewError, RuntimeError)


class TestCli:
    """``mdqm review``: the operator's entry point, driven in-process through Typer's runner."""

    @pytest.fixture
    def seeded(self, root: Path, clock: FakeClock) -> Seeded:
        """Seed the store the CLI opens by default: ``data/mdqm.sqlite`` under the CWD."""
        (root / "data").mkdir()
        store = Store(root / "data" / "mdqm.sqlite")
        queue = ReviewQueue(store, root, clock=clock)
        created = queue.submit(make(evidence=("crm has 3 status values",)))
        store.close()
        return queue, created.proposal.id

    def invoke(self, root: Path, *args: str) -> CliRunnerResult:
        """Run one CLI invocation with the CWD inside ``root``, as an operator would."""
        from mdqm.cli import app

        # Settings resolves relative paths against the CWD, so run where the fixture lives.
        previous = os.getcwd()
        os.chdir(root)
        try:
            return CliRunner().invoke(app, list(args))
        finally:
            os.chdir(previous)

    def test_list_shows_the_pending_proposal(self, root: Path, seeded: Seeded) -> None:
        _queue, pid = seeded
        result = self.invoke(root, "review", "list")
        assert result.exit_code == 0, result.stderr
        assert pid[:12] in result.stdout
        assert "pending" in result.stdout

    def test_list_says_so_when_empty(self, root: Path) -> None:
        result = self.invoke(root, "review", "list")
        assert result.exit_code == 0
        assert "nothing to review" in result.stdout

    def test_approve_writes_curated(self, root: Path, seeded: Seeded) -> None:
        _queue, pid = seeded
        result = self.invoke(root, "review", "approve", pid[:12], "--by", "data-team")
        assert result.exit_code == 0, result.stderr
        written = (root / "semantic" / "curated" / "value_maps.yaml").read_text(encoding="utf-8")
        assert "data-team" in written

    def test_export_patch_prints_a_diff(self, root: Path, seeded: Seeded) -> None:
        _queue, pid = seeded
        result = self.invoke(root, "review", "export-patch", pid[:12])
        assert result.exit_code == 0, result.stderr
        assert "+++ b/curated/value_maps.yaml" in result.stdout
        assert not (root / "semantic" / "curated" / "value_maps.yaml").exists()

    def test_reject_keeps_curated_untouched(self, root: Path, seeded: Seeded) -> None:
        _queue, pid = seeded
        result = self.invoke(root, "review", "reject", pid[:12], "--by", "data-team")
        assert result.exit_code == 0, result.stderr
        assert not (root / "semantic" / "curated" / "value_maps.yaml").exists()

    def test_unknown_prefix_fails_clearly(self, root: Path) -> None:
        result = self.invoke(root, "review", "approve", "deadbeef", "--by", "data-team")
        assert result.exit_code == 1
        assert "no proposal" in result.stdout + result.stderr
