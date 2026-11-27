"""Stages A1+A2 against the live catalog: proposals reach the queue, ``curated/`` stays untouched.

The unit tests prove the validators on scripted answers. This proves the two things only a real
database can: that the fact sheet is built from PostgreSQL's own catalog (tables, columns, types,
comments) without a single extra query per target, and that the whole pass end to end leaves the
human-owned layer alone -- a stage that wrote ``curated/`` would be a bug no unit test would catch,
because :class:`~mdqm.review.queue.ReviewQueue` is the only writer there and this stage never
approves anything.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.conftest import service_unavailable

from mdqm.llm.fake import FakeLLM
from mdqm.pipeline.core import PipelineContext, StageReport, StageStatus, run_pipeline
from mdqm.pipeline.stages.propose_names import (
    PROMPT_VERSION,
    STAGE_NAME,
    Labels,
    ProposeNamesStage,
)
from mdqm.review.queue import ProposalStatus, ReviewQueue
from mdqm.store import Store
from mdqm.testing.loaders.postgres import PgTarget, target_from_environ

pytestmark = pytest.mark.integration

NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def pg(require_services: bool) -> PgTarget:
    try:
        import psycopg

        target = target_from_environ()
        with psycopg.connect(target.dsn("billing"), connect_timeout=5):
            return target
    except Exception as exc:  # noqa: BLE001 - any failure means "no service here"
        service_unavailable(require_services, f"PostgreSQL: {exc}")
        raise AssertionError("unreachable: service_unavailable either fails or skips") from exc


def answer_from(prompt: str) -> Labels:
    """A deterministic, valid answer computed from the fact sheet the stage sent.

    No model and no network: what is under test here is the fact sheet, the validators and the
    queue, so the reply is derived from the sheet itself (a label per target, unique per table).
    """
    sheet = json.loads(prompt[prompt.index("{") :])
    subject = sheet.get("column") or sheet.get("table") or "record"
    if sheet["kind"] == "table":
        return Labels(
            name_he=f"טבלת {subject}",
            name_en="Business records",
            description_he="שורה לכל רשומה.",
            description_en=f"One row per record of `{subject}`.",
            synonyms_en=("Data set",),
        )
    words = str(subject).replace("_", " ").title()
    return Labels(
        name_he=f"שדה {words}",
        name_en=words,
        description_he="ערך עסקי של השדה.",
        description_en=f"The business value of `{subject}`.",
        synonyms_en=(f"{words} value",),
    )


def curated_snapshot(root: Path) -> dict[str, str]:
    """Every file under ``curated/`` and its hash: the layer that must not change."""
    curated = root / "curated"
    if not curated.is_dir():
        return {}
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(curated.glob("*.yaml"))
    }


@pytest.fixture(scope="module")
def live(
    pg: PgTarget, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, StageReport, int, int]:
    """Run A1+A2 once against the live billing catalog and report what it did."""
    from mdqm.adapters.postgresql import PostgresAdapter
    from mdqm.config.settings import SourceConfig

    class _Secrets:
        def get(self, name: str) -> str:
            return pg.ro_password

    adapter = PostgresAdapter(
        SourceConfig(
            name="billing",
            adapter="postgresql",
            secret="pg_ro_password",
            options={
                "host": pg.host,
                "port": pg.port,
                "dbname": "billing",
                "user": pg.ro_user,
            },
        ),
        _Secrets(),
    )
    root = tmp_path_factory.mktemp("semantic")
    llm = FakeLLM()
    context = PipelineContext(
        store=Store(root / "metadata.sqlite"),
        exec_ts=NOW,
        semantic_root=root,
        llm=llm,
    )
    try:
        stage = ProposeNamesStage({"billing": adapter.extract_catalog()})
        targets = len(stage.targets())
        # Two calls per target: the answer and its reordered replicate, both derived from the sheet.
        llm.queue(*([answer_from] * (2 * targets)))
        report = run_pipeline([stage], context).reports[0]
        calls = len(llm.calls)
    finally:
        adapter.close()
    return root, report, calls, targets


def test_the_live_catalog_produces_pending_proposals(
    live: tuple[Path, StageReport, int, int],
) -> None:
    root, report, calls, targets = live
    assert report.status is StageStatus.RAN, report
    assert targets > 10, f"the billing catalog should be bigger than a toy, got {targets}"
    assert calls == 2 * targets, "one call per target per run, and no call for a skipped target"
    assert report.counts["proposals"] > 0
    assert report.counts["created"] == report.counts["proposals"], report.counts
    pending = ReviewQueue(Store(root / "metadata.sqlite"), root).pending()
    assert pending, "A1+A2 proposed nothing against the live catalog"
    assert len(pending) == report.counts["proposals"]
    assert {item.status for item in pending} == {ProposalStatus.PENDING}
    assert {item.proposed_by for item in pending} == {STAGE_NAME}
    assert {item.section for item in pending} == {"names"}
    for item in pending:
        assert f"prompt_version: {PROMPT_VERSION}" in item.evidence
        assert 0.5 <= item.confidence <= 1.0


def test_the_stage_writes_no_curated_file(live: tuple[Path, StageReport, int, int]) -> None:
    root, report, _, _ = live
    assert report.outputs == (), "A1+A2 produces proposals, never YAML"
    assert curated_snapshot(root) == {}, "the curated layer belongs to humans"


def test_a_table_of_the_live_catalog_reaches_the_queue(
    live: tuple[Path, StageReport, int, int],
) -> None:
    root, _, _, _ = live
    keys = {item.key for item in ReviewQueue(Store(root / "metadata.sqlite"), root).pending()}
    assert "billing.usage_events" in keys
    assert any(key.startswith("billing.usage_events.") for key in keys)


def test_a_second_pass_over_an_unchanged_catalog_is_skipped(pg: PgTarget) -> None:
    """The input hash covers the catalog, so an unchanged schema must not pay for a second pass."""
    from mdqm.adapters.postgresql import PostgresAdapter
    from mdqm.config.settings import SourceConfig

    class _Secrets:
        def get(self, name: str) -> str:
            return pg.ro_password

    def build() -> PostgresAdapter:
        return PostgresAdapter(
            SourceConfig(
                name="billing",
                adapter="postgresql",
                secret="pg_ro_password",
                options={
                    "host": pg.host,
                    "port": pg.port,
                    "dbname": "billing",
                    "user": pg.ro_user,
                },
            ),
            _Secrets(),
        )

    context = PipelineContext(
        store=Store(Path.cwd() / "unused-names-metadata.sqlite"),
        exec_ts=NOW,
        semantic_root=Path.cwd(),
    )
    first_adapter, second_adapter = build(), build()
    try:
        first = ProposeNamesStage({"billing": first_adapter.extract_catalog()}).inputs_hash(context)
        second = ProposeNamesStage({"billing": second_adapter.extract_catalog()}).inputs_hash(
            context
        )
    finally:
        first_adapter.close()
        second_adapter.close()
    assert first == second
