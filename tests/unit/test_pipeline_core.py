"""Tests for the bootstrap pipeline runner (S01), the framework every D-stage runs on.

The four promises of the task are four groups of tests, plus the pieces the framework cannot get
wrong quietly:

* **idempotent** -- an unchanged input hash skips the stage, its writes and its model calls;
* **quiet windows** -- a heavy stage outside the window is refused with the next start in the
  message, and ``--force`` is the only way past it;
* **crash-safe** -- a stage that raises leaves ``generated/`` byte-identical (output is buffered and
  committed through the atomic writer only after the stage returns);
* **resumable + reported** -- a failure keeps every completed stage's output and its ledger row, and
  every report carries counts, skips and applied fixes;
* ``curated/`` has no write path at all, which one test proves by snapshotting mtimes.

No test sleeps: the clock is scripted and ``exec_ts`` is injected. Every store is in ``tmp_path``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from mdqm.cli import app
from mdqm.config.quiet import QuietWindow
from mdqm.llm.fake import FakeLLM
from mdqm.models.query import RawQuery
from mdqm.pipeline.core import (
    PIPELINE_RUN_TABLE,
    PipelineContext,
    PipelineResult,
    SourceBusy,
    SourceGate,
    SourceTooSlow,
    StageReport,
    StageStatus,
    available_stages,
    hash_inputs,
    is_heavy,
    next_quiet_start,
    quiet_allows,
    run_pipeline,
)
from mdqm.pipeline.stages.mining import MINED_QUERY_TABLE, batch_digest, ingest
from mdqm.store.db import Store

#: One instant for every test: 2026-05-01 10:00 UTC = 13:00 in Asia/Jerusalem (DST, so UTC+3).
NOON = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)

#: 21:00-07:00 local in the first organisation's zone: a window that crosses midnight.
NIGHT = QuietWindow(tz="Asia/Jerusalem", start="21:00", end="07:00")


class ScriptedClock:
    """A clock a test drives: ``monotonic`` advances by a fixed tick, ``now`` is one instant."""

    def __init__(self, tick: float = 0.25, now: datetime = NOON) -> None:
        self._tick = tick
        self._reading = 0.0
        self._now = now

    def monotonic(self) -> float:
        self._reading += self._tick
        return self._reading

    def now(self) -> datetime:
        return self._now


class Answer(BaseModel):
    """The schema a stage asks the model for, so FakeLLM has something to validate against."""

    names: list[str] = []


@dataclass
class FakeStage:
    """A stage with no domain: a version it hashes, an optional body, and a run counter."""

    name: str
    version: int = 1
    heavy: bool = False
    sources: tuple[str, ...] = ()
    body: Callable[[PipelineContext], StageReport] | None = None
    runs: int = 0

    def inputs_hash(self, ctx: PipelineContext) -> str:
        return hash_inputs(self.name, self.version)

    def run(self, ctx: PipelineContext) -> StageReport:
        self.runs += 1
        if self.body is None:
            return StageReport(stage=self.name, status=StageStatus.RAN)
        return self.body(ctx)


def reporting(counts: dict[str, int] | None = None) -> FakeStage:
    """A stage that writes one section and reports counters, a skip and an applied fix."""

    def body(ctx: PipelineContext) -> StageReport:
        ctx.write_generated("profile", {"tables": ["crm.customers"]})
        return StageReport(
            stage="D5",
            status=StageStatus.RAN,
            counts=counts or {"tables": 3, "columns": 21},
            skipped=("table crm.audit: unsupported type",),
            applied_fixes=("limit_k: read as x*10 (D8 conversion)",),
        )

    return FakeStage(name="D5", body=body)


def crashing(ctx: PipelineContext) -> StageReport:
    """Write a section, then fail: what a stage looks like when it hits a bad row half way."""
    ctx.write_generated("profile", {"tables": ["WRONG"]})
    raise RuntimeError("could not read crm.customers: connection reset")


@pytest.fixture
def store(tmp_path: Path) -> Store:
    """An initialized metadata store under ``tmp_path``."""
    opened = Store(tmp_path / "mdqm.sqlite")
    opened.initialize()
    try:
        yield opened
    finally:
        opened.close()


@pytest.fixture
def ctx(tmp_path: Path, store: Store) -> PipelineContext:
    """A context with no quiet window (unrestricted) and a scripted clock."""
    return PipelineContext(
        store=store,
        exec_ts=NOON,
        semantic_root=tmp_path / "semantic",
        clock=ScriptedClock(),
    )


def snapshot(root: Path) -> dict[str, tuple[int, bytes]]:
    """Every file under ``root``: its mtime and its bytes (what "written nothing" must preserve)."""
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): (path.stat().st_mtime_ns, path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def rows(store: Store, stage: str | None = None) -> list[Any]:
    """The pipeline ledger, oldest first, optionally for one stage."""
    sql = f"SELECT * FROM {PIPELINE_RUN_TABLE}"  # noqa: S608 - a module constant
    params: tuple[object, ...] = ()
    if stage is not None:
        sql += " WHERE stage = ?"
        params = (stage,)
    return list(store.connect().execute(sql + " ORDER BY id", params))


# --- hashing --------------------------------------------------------------------------------


def test_hash_inputs_is_canonical_and_stable() -> None:
    assert hash_inputs({"a": 1, "b": 2}) == hash_inputs({"b": 2, "a": 1})
    assert hash_inputs("1") != hash_inputs(1)
    digest = hash_inputs("D5", 1, ["crm", "billing"])
    assert len(digest) == 64
    assert digest == hash_inputs("D5", 1, ["crm", "billing"])


def test_hash_inputs_survives_types_json_cannot_encode() -> None:
    """Paths, datetimes and enums hash by their text form instead of raising."""
    assert len(hash_inputs(Path("semantic"), NOON, StageStatus.RAN)) == 64


# --- idempotence: unchanged input skips the stage and the model call (acceptance 1) ---------


def test_unchanged_input_skips_the_stage_and_its_model_call(
    ctx: PipelineContext, store: Store
) -> None:
    llm = FakeLLM(Answer(names=["credit limit"]))
    ctx.llm = llm

    def body(context: PipelineContext) -> StageReport:
        assert context.llm is not None
        answer = context.llm.structured("name these columns", Answer)
        context.write_generated("names", {"names": answer.value.names})
        return StageReport(stage="A1", status=StageStatus.RAN, counts={"names": 1})

    stage = FakeStage(name="A1", body=body)
    first = run_pipeline([stage], ctx)
    before = snapshot(ctx.semantic_root)

    second = run_pipeline([stage], ctx)

    assert [report.status for report in first.reports] == [StageStatus.RAN]
    assert [report.status for report in second.reports] == [StageStatus.SKIPPED]
    assert stage.runs == 1, "a skipped stage must not run"
    assert len(llm.calls) == 1, "a skipped stage must not ask the model again"
    assert snapshot(ctx.semantic_root) == before, "a skipped stage writes nothing"
    assert second.reports[0].skipped == ("inputs unchanged since the last successful run",)
    # The attempt is still recorded: an operator can see why the night did nothing.
    assert [row["status"] for row in rows(store, "A1")] == ["ran", "skipped"]


def test_a_changed_input_hash_reruns_the_stage(ctx: PipelineContext) -> None:
    llm = FakeLLM(Answer(names=["one"]), Answer(names=["two"]))
    ctx.llm = llm
    stage = reporting()
    run_pipeline([stage], ctx)

    stage.version = 2  # the stage's declared input moved, so its output is stale
    result = run_pipeline([stage], ctx)

    assert result.reports[0].status is StageStatus.RAN
    assert stage.runs == 2


def test_a_deleted_output_is_regenerated_instead_of_skipped(ctx: PipelineContext) -> None:
    """A hash match is only safe while the recorded output is still on disk."""
    stage = reporting()
    run_pipeline([stage], ctx)
    ctx.generated_path("profile").unlink()

    result = run_pipeline([stage], ctx)

    assert result.reports[0].status is StageStatus.RAN
    assert ctx.generated_path("profile").is_file()


def test_force_reruns_even_when_the_hash_matches(ctx: PipelineContext) -> None:
    stage = reporting()
    run_pipeline([stage], ctx)

    result = run_pipeline([stage], ctx, force=True)

    assert result.reports[0].status is StageStatus.RAN
    assert result.reports[0].forced is True
    assert stage.runs == 2


# --- quiet windows (acceptance 2) -----------------------------------------------------------


def test_quiet_allows_table() -> None:
    afternoon = datetime(2026, 5, 1, 13, 0, tzinfo=UTC)  # 16:00 local, outside the window
    evening = datetime(2026, 5, 1, 19, 0, tzinfo=UTC)  # 22:00 local, inside it
    never = QuietWindow(tz="UTC", start="21:00", end="21:00")
    assert quiet_allows((), afternoon) is True, "no window configured means unrestricted"
    assert quiet_allows((NIGHT,), afternoon) is False
    assert quiet_allows((NIGHT,), evening) is True
    assert quiet_allows((never,), evening) is False, "a zero-length window is never open"


def test_next_quiet_start_crosses_midnight_and_ignores_zero_length_windows() -> None:
    afternoon = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)  # 13:00 local
    upcoming = next_quiet_start((NIGHT,), afternoon)
    assert upcoming is not None
    instant, window = upcoming
    assert instant == datetime(2026, 5, 1, 18, 0, tzinfo=UTC)  # 21:00 Asia/Jerusalem
    assert window is NIGHT
    assert next_quiet_start((), afternoon) is None
    assert next_quiet_start((QuietWindow(tz="UTC", start="21:00", end="21:00"),), afternoon) is None


def test_heavy_stage_outside_the_window_is_refused_with_the_next_start(
    tmp_path: Path, store: Store
) -> None:
    ctx = PipelineContext(
        store=store,
        exec_ts=NOON,
        semantic_root=tmp_path / "semantic",
        quiet_windows=(NIGHT,),
        clock=ScriptedClock(),
    )
    before = reporting()
    before.heavy = True
    after = reporting()
    after.name = "D6"
    after.heavy = True

    result = run_pipeline([before, after], ctx)

    assert [report.status for report in result.reports] == [StageStatus.REFUSED]
    refusal = result.reports[0]
    assert refusal.error is not None
    assert "21:00 Asia/Jerusalem" in refusal.error, refusal.error
    assert "--force" in refusal.error
    assert refusal.deferred_until == datetime(2026, 5, 1, 18, 0, tzinfo=UTC)
    assert result.deferred_until == refusal.deferred_until
    assert result.aborted is False
    assert result.ok is False
    assert before.runs == 0
    assert after.runs == 0, "the whole pass waits for the window, not just the heavy stage"
    assert snapshot(ctx.semantic_root) == {}, "a refusal writes nothing"
    assert [row["status"] for row in rows(store, "D5")] == ["refused"]


def test_force_runs_a_heavy_stage_outside_the_window(tmp_path: Path, store: Store) -> None:
    ctx = PipelineContext(
        store=store,
        exec_ts=NOON,
        semantic_root=tmp_path / "semantic",
        quiet_windows=(NIGHT,),
        clock=ScriptedClock(),
    )
    stage = reporting()

    result = run_pipeline([stage], ctx, force=True)

    assert result.reports[0].status is StageStatus.RAN
    assert result.reports[0].forced is True
    assert stage.runs == 1
    assert ctx.generated_path("profile").is_file()


def test_heavy_stage_inside_the_window_runs(tmp_path: Path, store: Store) -> None:
    evening = datetime(2026, 5, 1, 19, 0, tzinfo=UTC)  # 22:00 local
    ctx = PipelineContext(
        store=store,
        exec_ts=evening,
        semantic_root=tmp_path / "semantic",
        quiet_windows=(NIGHT,),
        clock=ScriptedClock(now=evening),
    )

    result = run_pipeline([reporting()], ctx)

    assert result.reports[0].status is StageStatus.RAN


def test_a_light_stage_is_not_held_by_the_window(tmp_path: Path, store: Store) -> None:
    """Catalog reads are cheap: only ``heavy = True`` waits for the window."""
    ctx = PipelineContext(
        store=store,
        exec_ts=NOON,
        semantic_root=tmp_path / "semantic",
        quiet_windows=(NIGHT,),
        clock=ScriptedClock(),
    )

    def body(context: PipelineContext) -> StageReport:
        context.write_generated("catalog", {"tables": ["crm.customers"]})
        return StageReport(stage="D1", status=StageStatus.RAN, counts={"tables": 1})

    result = run_pipeline([FakeStage(name="D1", body=body)], ctx)

    assert result.reports[0].status is StageStatus.RAN
    assert is_heavy(FakeStage(name="D1")) is False


# --- crash safety (acceptance 3) ------------------------------------------------------------


def test_a_crashing_stage_leaves_generated_unchanged(ctx: PipelineContext) -> None:
    good = reporting()
    run_pipeline([good], ctx)
    before = snapshot(ctx.semantic_root)
    assert set(before) == {"generated/profile.yaml"}

    # Same stage name, different input: it runs, writes, and dies before returning.
    good.version = 2
    good.body = crashing
    result = run_pipeline([good], ctx)

    assert result.reports[0].status is StageStatus.FAILED
    assert result.reports[0].error is not None
    assert "RuntimeError" in result.reports[0].error
    assert "connection reset" in result.reports[0].error
    assert snapshot(ctx.semantic_root) == before, "a crash must not touch generated/"
    assert sorted(path.name for path in ctx.semantic_root.rglob("*") if path.is_file()) == [
        "profile.yaml"
    ], "no temp file is left behind"


def test_a_crashing_stage_loses_its_buffered_output_only(ctx: PipelineContext) -> None:
    """Two sections buffered: a crash after both writes leaves neither on disk."""

    def body(context: PipelineContext) -> StageReport:
        context.write_generated("joins", {"joins": []})
        context.write_generated("conversions", {"conversions": []})
        raise RuntimeError("bad row")

    result = run_pipeline([FakeStage(name="D7", body=body)], ctx)

    assert result.reports[0].status is StageStatus.FAILED
    assert snapshot(ctx.semantic_root) == {}


# --- resumability and reporting -------------------------------------------------------------


def test_a_failure_keeps_completed_stages_and_the_next_run_resumes(
    ctx: PipelineContext, store: Store
) -> None:
    first = reporting()
    second = FakeStage(name="D6", body=crashing)
    third = reporting()
    third.name = "D7"

    failed = run_pipeline([first, second, third], ctx)

    assert [report.status for report in failed.reports] == [StageStatus.RAN, StageStatus.FAILED]
    assert failed.aborted is True
    assert failed.failed is not None
    assert failed.failed.stage == "D6"
    assert third.runs == 0, "a failure stops the pass: later stages did not run"
    assert ctx.generated_path("profile").is_file(), (
        "the completed stage's output survives the failure"
    )

    second.body = None  # the bug is fixed
    resumed = run_pipeline([first, second, third], ctx)

    assert [report.status for report in resumed.reports] == [
        StageStatus.SKIPPED,
        StageStatus.RAN,
        StageStatus.RAN,
    ]
    assert first.runs == 1, "the completed stage was not redone"
    assert resumed.aborted is False
    assert [row["status"] for row in rows(store, "D6")] == ["failed", "ran"]


def test_every_report_carries_counts_skips_and_fixes(ctx: PipelineContext, store: Store) -> None:
    stage = reporting()

    result = run_pipeline([stage], ctx)

    report = result.reports[0]
    assert report.counts == {"tables": 3, "columns": 21}
    assert report.skipped == ("table crm.audit: unsupported type",)
    assert report.applied_fixes == ("limit_k: read as x*10 (D8 conversion)",)
    assert report.outputs == ("generated/profile.yaml",)
    assert report.inputs_hash == stage.inputs_hash(ctx)
    # The ledger keeps all of it, verbatim: the pipeline never paraphrases a stage's report.
    row = rows(store, "D5")[-1]
    assert row["counts"] == '{"columns": 21, "tables": 3}'
    assert "unsupported type" in row["skipped"]
    assert "x*10" in row["applied_fixes"]
    assert row["outputs"] == '["generated/profile.yaml"]'
    assert result.describe().startswith("D5   ran")


def test_the_runner_owns_the_status_of_a_successful_stage(ctx: PipelineContext) -> None:
    """A stage cannot excuse itself from the ledger by claiming it was skipped."""
    stage = FakeStage(
        name="D5",
        body=lambda context: StageReport(stage="D5", status=StageStatus.SKIPPED),
    )

    result = run_pipeline([stage], ctx)

    assert result.reports[0].status is StageStatus.RAN


def test_source_filter_skips_stages_of_other_sources(ctx: PipelineContext, store: Store) -> None:
    crm = reporting()
    crm.sources = ("crm",)
    everything = reporting()
    everything.name = "D1"

    result = run_pipeline([crm, everything], ctx, source="billing")

    assert [report.status for report in result.reports] == [
        StageStatus.SKIPPED,
        StageStatus.RAN,
    ]
    assert result.reports[0].skipped == ("does not read source billing",)
    assert crm.runs == 0
    assert rows(store, "D5") == [], "a stage filtered out by source is not in the ledger"
    assert report_of(result, "D1").status is StageStatus.RAN


def report_of(result: PipelineResult, stage: str) -> StageReport:
    """The single report of ``stage``."""
    matches = [report for report in result.reports if report.stage == stage]
    assert len(matches) == 1, f"expected one report for {stage}, got {len(matches)}"
    return matches[0]


def test_unknown_stage_name_is_a_value_error(ctx: PipelineContext) -> None:
    with pytest.raises(ValueError, match="no stage named 'D9'"):
        run_pipeline([reporting()], ctx, stage="D9")


def test_only_one_stage_runs_when_one_is_named(ctx: PipelineContext) -> None:
    first = reporting()
    second = FakeStage(name="D6")
    third = FakeStage(name="D7")

    result = run_pipeline([first, second, third], ctx, stage="D6")

    assert [report.stage for report in result.reports] == ["D6"]


def test_an_empty_stage_list_runs_nothing(ctx: PipelineContext) -> None:
    result = run_pipeline([], ctx)

    assert result.reports == ()
    assert result.ok is True
    assert result.describe() == "no stage ran"


# --- curated/ is never written (acceptance 4) -----------------------------------------------


def test_the_pipeline_never_writes_curated_files(tmp_path: Path, store: Store) -> None:
    root = tmp_path / "semantic"
    curated = root / "curated"
    curated.mkdir(parents=True)
    theirs = curated / "measures.yaml"
    theirs.write_text(
        "# a human wrote this: do not touch it\nmeasures:\n  credit_limit:\n"
        "    expression: round_half_up(limit / 5000) * 5000\n",
        encoding="utf-8",
    )
    ctx = PipelineContext(
        store=store,
        exec_ts=NOON,
        semantic_root=root,
        clock=ScriptedClock(),
    )
    curated_before = snapshot(curated)
    assert curated_before, "the fixture must have a curated file to protect"

    stages = [reporting(), FakeStage(name="D6", body=crashing)]
    run_pipeline(stages, ctx, force=True)
    run_pipeline(stages, ctx, force=True)

    assert snapshot(curated) == curated_before, "curated/ files and their mtimes must be untouched"
    assert ctx.generated_path("profile").is_file()
    assert ctx.curated_path("policy") == curated / "policy.yaml"
    assert not (curated / "policy.yaml").exists(), "no curated file was created"


def test_a_section_name_cannot_escape_generated(ctx: PipelineContext) -> None:
    with pytest.raises(ValueError, match="not a section name"):
        ctx.write_generated("../../etc/passwd", {"x": 1})
    with pytest.raises(ValueError, match="not a section name"):
        ctx.write_generated("nested/profile", {"x": 1})
    assert ctx.pending == ()


def test_a_context_can_read_a_committed_section(ctx: PipelineContext) -> None:
    stage = reporting()
    run_pipeline([stage], ctx)

    assert "crm.customers" in ctx.generated_path("profile").read_text(encoding="utf-8")
    assert ctx.outputs_present(("generated/profile.yaml",)) is True
    assert ctx.outputs_present(("generated/missing.yaml",)) is False


# --- per-source concurrency and latency -----------------------------------------------------


def test_gate_limits_concurrency_per_source() -> None:
    gate = SourceGate(max_per_source=1)

    assert gate.try_acquire("crm") is True
    assert gate.try_acquire("crm") is False, "the second call must wait or fail, not overlap"
    assert gate.try_acquire("billing") is True, "another source has its own budget"
    assert gate.in_flight("crm") == 1
    assert gate.peak("crm") == 1

    gate.release("crm")
    gate.release("billing")
    assert gate.try_acquire("crm") is True
    gate.release("crm")


def test_gate_allows_two_when_configured() -> None:
    gate = SourceGate(max_per_source=2)
    assert gate.try_acquire("crm") is True
    assert gate.try_acquire("crm") is True
    assert gate.try_acquire("crm") is False
    assert gate.peak("crm") == 2
    gate.release("crm")
    gate.release("crm")


def test_gate_rejects_a_nonsense_budget() -> None:
    with pytest.raises(ValueError, match="max_per_source"):
        SourceGate(max_per_source=0)
    with pytest.raises(ValueError, match="samples"):
        SourceGate(samples=0)


def test_gate_stops_when_latency_rises() -> None:
    gate = SourceGate(max_per_source=1, latency_ceiling_s=5.0, samples=3, clock=ScriptedClock())
    for seconds in (2.0, 9.0, 8.0):
        assert gate.try_acquire("crm") is True
        gate.release("crm", latency_s=seconds)
    assert gate.latency("crm") == (2.0, 9.0, 8.0)

    with pytest.raises(SourceTooSlow, match="median of the last 3 calls is 8.0s"):
        gate.try_acquire("crm")
    with pytest.raises(SourceTooSlow):
        gate.acquire("crm")
    assert gate.in_flight("crm") == 0, "a refused call takes no slot"


def test_gate_tolerates_one_slow_observation() -> None:
    """The median, not the maximum: one slow query must not stop the night."""
    gate = SourceGate(max_per_source=1, latency_ceiling_s=5.0, samples=3)
    for seconds in (1.0, 30.0, 1.0):
        gate.acquire("crm")
        gate.release("crm", latency_s=seconds)

    assert gate.try_acquire("crm") is True
    gate.release("crm")


def test_gate_access_times_the_body_with_the_injected_clock() -> None:
    gate = SourceGate(max_per_source=1, latency_ceiling_s=0.1, samples=1, clock=ScriptedClock(0.5))
    with gate.access("crm"):
        assert gate.in_flight("crm") == 1

    assert gate.in_flight("crm") == 0
    assert gate.latency("crm") == (0.5,)
    with pytest.raises(SourceTooSlow):
        gate.try_acquire("crm")


def test_gate_busy_refusal_names_the_source() -> None:
    gate = SourceGate(max_per_source=1, wait_s=0.0)
    gate.acquire("crm")
    try:
        with pytest.raises(SourceBusy, match="crm already has 1 call"):
            gate.acquire("crm")
    finally:
        gate.release("crm")


def test_a_slow_source_stops_the_stage(tmp_path: Path, store: Store) -> None:
    """The gate is on the context, so a heavy stage's own refusal fails the stage and the pass."""
    gate = SourceGate(max_per_source=1, latency_ceiling_s=1.0, samples=3)
    ctx = PipelineContext(
        store=store,
        exec_ts=NOON,
        semantic_root=tmp_path / "semantic",
        clock=ScriptedClock(),
        gate=gate,
    )
    for seconds in (10.0, 10.0, 10.0):
        gate.acquire("crm")
        gate.release("crm", latency_s=seconds)

    def body(context: PipelineContext) -> StageReport:
        with context.access("crm"):
            return StageReport(stage="D5", status=StageStatus.RAN)

    heavy = FakeStage(name="D5", heavy=True, sources=("crm",), body=body)
    result = run_pipeline([heavy], ctx)

    assert result.reports[0].status is StageStatus.FAILED
    assert result.reports[0].error is not None
    assert "SourceTooSlow" in result.reports[0].error


# --- a real stage: D9 mining is a first-class citizen --------------------------------------


def mining_stage(queries: Sequence[RawQuery]) -> FakeStage:
    """The stage shape DESIGN.md expects: hash the batch, let mining do the work, report counts."""

    def body(context: PipelineContext) -> StageReport:
        report = ingest(context.store, list(queries), source="crm", exec_ts=context.exec_ts)
        return StageReport(
            stage="D9",
            status=StageStatus.RAN,
            counts={
                "statements": report.statements,
                "parsed": report.parsed,
                "failed": report.failed,
                "fingerprints": len(report.queries),
            },
            skipped=("batch already mined",) if report.skipped else (),
        )

    class MiningStage(FakeStage):
        def inputs_hash(self, context: PipelineContext) -> str:
            return batch_digest(queries)

    return MiningStage(name="D9", heavy=True, body=body)


def count(store: Store, table: str) -> int:
    return int(store.connect().execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"])  # noqa: S608


def test_mining_runs_through_the_pipeline_and_skips_on_the_second_pass(
    ctx: PipelineContext, store: Store
) -> None:
    queries = (
        RawQuery(
            text="SELECT c.cust_id, c.limit_k FROM crm.customers AS c WHERE c.status = 'A'",
            dialect="postgres",
            calls=3,
            last_seen=datetime(2026, 4, 30, 8, 0, tzinfo=UTC),
            user="dana",
        ),
    )
    stage = mining_stage(queries)

    first = run_pipeline([stage], ctx)
    mined_rows = count(store, MINED_QUERY_TABLE)

    second = run_pipeline([stage], ctx)

    assert first.reports[0].status is StageStatus.RAN
    assert first.reports[0].counts["statements"] == 1
    assert first.reports[0].counts["failed"] == 0
    assert second.reports[0].status is StageStatus.SKIPPED
    assert count(store, MINED_QUERY_TABLE) == mined_rows == 1

    # Forced, the framework runs it again and mining's own batch hash keeps the counters honest.
    forced = run_pipeline([stage], ctx, force=True)
    assert forced.reports[0].skipped == ("batch already mined",)
    assert count(store, MINED_QUERY_TABLE) == 1


# --- CLI ------------------------------------------------------------------------------------


def test_cli_bootstrap_run_reports_an_empty_registry() -> None:
    assert available_stages() == ()
    result = CliRunner().invoke(app, ["bootstrap", "run"])
    assert result.exit_code == 0, result.output
    assert "no bootstrap stage is registered" in result.output


def test_cli_lists_bootstrap_commands() -> None:
    result = CliRunner().invoke(app, ["bootstrap", "--help"])
    assert result.exit_code == 0, result.output
    assert "run" in result.output
