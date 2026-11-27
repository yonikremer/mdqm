"""The bootstrap pipeline runner (task S01): ordered stages that skip, resume, report, stay quiet.

DESIGN.md section 10 says the semantic layer is built by scheduled stages whose *inputs are hashed,
so unchanged input => skip, including model calls*. This module is that rule and nothing else: it
holds no stage knowledge.

* a **stage** is any object with ``name``, ``inputs_hash(ctx)`` and ``run(ctx)`` (plus optional
  ``heavy = True`` and ``sources = ("crm", ...)`` declarations). :mod:`mdqm.pipeline.stages.mining`
  is the shape a real stage has: it hashes its own batch (:func:`mdqm.pipeline.stages.mining.
  batch_digest`), reports ``skipped`` when the batch was already mined, and writes only through the
  metadata store. A stage that needs a model takes :attr:`PipelineContext.llm`.
* :func:`run_pipeline` runs stages **in order** and records every attempt in the metadata store.
  A stage whose input hash equals its last successful run's -- and whose recorded outputs are still
  on disk -- is skipped without running, so a model is never asked again. A stage that raises stops
  the run: everything before it is already committed, so the next run resumes at the failure.
* **Heavy** stages (profiling, statistics, mining) run only inside a configured
  :class:`~mdqm.config.quiet.QuietWindow`; outside it the stage is refused, with the next window
  start in the message, unless ``force``. Heavy stages also take their sources' slots from
  :class:`SourceGate`: at most ``max_per_source`` (1-2) concurrent calls per source, and no new call
  when recent latency has risen.
* A stage's output reaches ``generated/`` **only if the stage returns**: :meth:`PipelineContext.
  write_generated` buffers a YAML section and the runner commits it afterwards, through the atomic
  writer of :mod:`mdqm.semantic.writer`. A crash therefore leaves ``generated/`` byte-identical.
* ``curated/`` has **no write path at all** in this module: humans own it and it wins over
  ``generated/`` (DESIGN.md section 10), so a rerun can never overwrite hand-made work.

Everything is injected -- store, execution instant, clock, semantic root, model -- so no test sleeps
and no stage reads the wall clock. Nothing here contacts a network host.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Protocol

from pydantic import Field

from mdqm.config.quiet import QuietWindow
from mdqm.governor.clock import Clock, SystemClock
from mdqm.llm.result import StructuredLLM
from mdqm.models.base import MDQMModel, UtcDatetime
from mdqm.semantic.loader import CURATED_DIR, GENERATED_DIR
from mdqm.semantic.writer import write_generated
from mdqm.sql.execution import run_local
from mdqm.store.db import Store

__all__ = [
    "ERROR_LIMIT",
    "PIPELINE_RUN_TABLE",
    "STAGE_ORDER",
    "PipelineContext",
    "PipelineResult",
    "SourceBusy",
    "SourceGate",
    "SourceTooSlow",
    "Stage",
    "StageReport",
    "StageStatus",
    "available_stages",
    "hash_inputs",
    "is_heavy",
    "next_quiet_start",
    "quiet_allows",
    "run_pipeline",
]

logger = logging.getLogger(__name__)


def _statement(*parts: str) -> str:
    """Assemble one SQL statement from module-level constants only (never from caller input)."""
    return "".join(parts)


#: One row per stage attempt: the ledger a rerun reads to decide what to skip.
PIPELINE_RUN_TABLE: Final[str] = "pipeline_run"

#: How much of a failing stage's message reaches the store: a report, never a stack trace.
ERROR_LIMIT: Final[int] = 400

#: A section is a file stem, not a path -- the same rule :mod:`mdqm.semantic.writer` enforces. It is
#: repeated here so a bad name fails *before* a commit has written anything.
_SECTION_NAME: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.-]+$")

_INSERT_RUN: Final[str] = _statement(
    "INSERT INTO ",
    PIPELINE_RUN_TABLE,
    " (stage, inputs_hash, status, started_at, finished_at, counts, skipped,"
    " applied_fixes, outputs, error, forced) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
)
_SELECT_LAST_RAN: Final[str] = _statement(
    "SELECT inputs_hash, outputs FROM ",
    PIPELINE_RUN_TABLE,
    " WHERE stage = ? AND status = 'ran' ORDER BY id DESC LIMIT 1",
)


class StageStatus(StrEnum):
    """The verdict of one stage attempt. Only :attr:`RAN` lets a later run skip the stage."""

    RAN = "ran"
    SKIPPED = "skipped"
    REFUSED = "refused"
    FAILED = "failed"


class StageReport(MDQMModel):
    """What one stage attempt did: the runner owns ``stage``, ``status``, ``inputs_hash`` and
    ``outputs``; the stage fills in everything else.

    A stage fills in ``counts`` (what it measured), ``skipped`` (what it deliberately left alone, in
    words) and ``applied_fixes`` (every silent auto-fix, principle 10). Missing any of the three is
    how a stage hides work, so the runner publishes them into the ledger unchanged.
    """

    stage: str
    status: StageStatus
    inputs_hash: str = Field(default="", description="sha256 of the inputs this attempt saw.")
    counts: dict[str, int] = Field(default_factory=dict)
    skipped: tuple[str, ...] = ()
    applied_fixes: tuple[str, ...] = ()
    outputs: tuple[str, ...] = Field(
        default=(), description="Paths written under generated/, relative to the semantic root."
    )
    forced: bool = False
    error: str | None = Field(default=None, description="Where it failed; never a stack trace.")
    deferred_until: UtcDatetime | None = Field(
        default=None, description="For a refusal: when the quiet window next opens."
    )

    @property
    def ok(self) -> bool:
        """True when the stage did not fail and was not deferred."""
        return self.status in (StageStatus.RAN, StageStatus.SKIPPED)

    def describe(self) -> str:
        """One line an operator can read: status, counters, skips, fixes, output, error."""
        bits = [f"{self.stage:<4} {self.status.value:<8}"]
        if self.counts:
            bits.append(" ".join(f"{name}={value}" for name, value in sorted(self.counts.items())))
        if self.skipped:
            # A skipped stage's reason is the whole line; a stage that ran names what it left alone.
            prefix = "" if self.status is StageStatus.SKIPPED else "skipped "
            bits.append(prefix + "; ".join(self.skipped))
        if self.applied_fixes:
            bits.append("fixes " + "; ".join(self.applied_fixes))
        if self.outputs:
            bits.append("wrote " + ", ".join(self.outputs))
        if self.forced:
            bits.append("(forced)")
        if self.error:
            bits.append(self.error)
        return "  ".join(bits)


class PipelineResult(MDQMModel):
    """Every attempt of one pass, in run order: what ran, what was skipped, where it stopped."""

    reports: tuple[StageReport, ...] = ()
    aborted: bool = Field(default=False, description="A stage failed; later stages did not run.")
    deferred_until: UtcDatetime | None = Field(
        default=None, description="A heavy stage was refused; the whole pass waits for this."
    )

    @property
    def failed(self) -> StageReport | None:
        """The stage that stopped the pass, or ``None``."""
        return next(
            (report for report in self.reports if report.status is StageStatus.FAILED), None
        )

    @property
    def ran(self) -> tuple[StageReport, ...]:
        """The stages that actually executed."""
        return tuple(report for report in self.reports if report.status is StageStatus.RAN)

    @property
    def ok(self) -> bool:
        """True when nothing failed and nothing was deferred."""
        return self.failed is None and self.deferred_until is None

    def describe(self) -> str:
        """One line per stage, plus the reason the pass stopped early (if it did)."""
        lines = [report.describe() for report in self.reports]
        if not lines:
            lines.append("no stage ran")
        if self.deferred_until is not None:
            lines.append(f"deferred until {self.deferred_until.isoformat()}")
        elif self.aborted and self.failed is not None:
            lines.append(f"stopped after {self.failed.stage}: later stages did not run")
        return "\n".join(lines)


class Stage(Protocol):
    """A bootstrap stage. Three members, no base class: a stage is anything shaped like this.

    ``inputs_hash`` must be a *pure function of declared inputs* (a path, a config value, a version,
    a sorted list of names) and must not include the stage's own output, or the stage would never
    skip. :func:`hash_inputs` is the canonical way to build it.
    """

    name: str

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub: pyright requires an ellipsis body
    def inputs_hash(self, ctx: PipelineContext) -> str:
        """sha256 over everything this stage's output depends on."""
        ...

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub: pyright requires an ellipsis body
    def run(self, ctx: PipelineContext) -> StageReport:
        """Do the work and report it. Raising is allowed: the runner records the failure."""
        ...


# --- input hashing --------------------------------------------------------------------------


def hash_inputs(*parts: object) -> str:
    """sha256 over the facts a stage's output depends on, in a canonical form.

    ``parts`` are JSON-shaped facts: paths, versions, counts, sorted name lists, config values.
    Mapping keys are sorted and separators are fixed, so two equal inputs hash equal whatever order
    they were assembled in. ``default=str`` lets a ``Path``, a ``datetime`` or an ``Enum`` be hashed
    by its text form instead of failing -- the hash is a change detector, not a serialisation.
    """
    payload = json.dumps(
        parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- quiet windows --------------------------------------------------------------------------


def quiet_allows(windows: Sequence[QuietWindow], now: datetime) -> bool:
    """True when heavy work may run at ``now``.

    No configured window means unrestricted (the same reading as
    :meth:`mdqm.config.settings.Settings.is_quiet`): a fresh install must be able to bootstrap on a
    laptop. The window arithmetic itself stays in :class:`~mdqm.config.quiet.QuietWindow`.
    """
    if not windows:
        return True
    return any(window.is_quiet(now) for window in windows)


def next_quiet_start(
    windows: Sequence[QuietWindow], now: datetime
) -> tuple[datetime, QuietWindow] | None:
    """When and in which window heavy work next becomes allowed; ``None`` when it never does.

    A zero-length window (``start == end``) is never quiet by design, so it opens nothing and is
    ignored here: promising a time that never arrives would be worse than saying there is none.
    """
    candidates: list[tuple[datetime, QuietWindow]] = []
    for window in windows:
        if window.start == window.end:
            continue
        local = now.astimezone(window.zone)
        opened = local.replace(
            hour=window.start.hour,
            minute=window.start.minute,
            second=window.start.second,
            microsecond=0,
        )
        if opened <= local:
            opened += timedelta(days=1)
        candidates.append((opened, window))
    if not candidates:
        return None
    (opened, window) = min(candidates, key=lambda item: item[0].astimezone(UTC))
    return (opened.astimezone(UTC), window)


def _describe_instant(instant: datetime, window: QuietWindow) -> str:
    """``2026-05-01 21:00 Asia/Jerusalem``: the wall clock a human sets the window by."""
    return f"{instant.astimezone(window.zone):%Y-%m-%d %H:%M} {window.tz}"


def _describe_delay(now: datetime, instant: datetime) -> str:
    """``in 2h 5m``: how long an operator waits, rounded down to a minute."""
    minutes = max(timedelta(0), instant - now) // timedelta(minutes=1)
    hours, minutes = divmod(minutes, 60)
    return f"in {hours}h {minutes}m" if hours else f"in {minutes}m"


def _refusal(
    stage: str, windows: Sequence[QuietWindow], now: datetime
) -> tuple[str, datetime | None]:
    """The refusal message (with the next window start in it) and that start as an instant."""
    configured = ", ".join(
        f"{window.start:%H:%M}-{window.end:%H:%M} {window.tz}" for window in windows
    )
    upcoming = next_quiet_start(windows, now)
    if upcoming is None:
        return (
            f"stage {stage} is heavy and no quiet window ever opens (configured: {configured}); "
            "give a window whose start and end differ, or pass --force to run it now",
            None,
        )
    instant, window = upcoming
    return (
        f"stage {stage} is heavy and its quiet window is closed; the next window opens at "
        f"{_describe_instant(instant, window)} ({_describe_delay(now, instant)}). "
        f"Configured: {configured}. Pass --force to run it now",
        instant,
    )


# --- per-source access ----------------------------------------------------------------------


class SourceBusy(RuntimeError):
    """Another call already holds this source's slot: ``max_per_source`` is reached."""


class SourceTooSlow(RuntimeError):
    """Recent calls to this source were slower than the ceiling, so no new call is made."""


@dataclass
class _Slot:
    """One source's budget: the semaphore, what is in flight, and the recent latencies."""

    semaphore: threading.BoundedSemaphore
    in_flight: int = 0
    peak: int = 0
    latencies: list[float] = field(default_factory=list)


class SourceGate:
    """Per-source access for heavy stages: at most ``max_per_source`` at once, and stop when slow.

    DESIGN.md section 10 allows 1-2 concurrent calls per source and says to stop if latency rises: a
    primary database with no replica pays for every extra concurrent profile scan. The gate lives on
    :class:`PipelineContext`, so two stages (or two threads of one stage) cannot exceed the limit
    between them.

    ``latency_ceiling_s`` is compared against the **median of the last ``samples`` finished calls**,
    not against one observation: a single slow query must not stop the night, but a source that has
    become uniformly slow must be left alone before it falls over. Latency is measured with the
    injected :class:`~mdqm.governor.clock.Clock`, so a test advances time instead of sleeping.
    """

    def __init__(
        self,
        *,
        max_per_source: int = 2,
        latency_ceiling_s: float = 30.0,
        samples: int = 3,
        wait_s: float = 300.0,
        clock: Clock | None = None,
    ) -> None:
        if max_per_source < 1:
            raise ValueError("max_per_source must be at least 1")
        if samples < 1:
            raise ValueError("samples must be at least 1")
        self.max_per_source = max_per_source
        self.latency_ceiling_s = latency_ceiling_s
        self.samples = samples
        self.wait_s = wait_s
        self._clock: Clock = clock or SystemClock()
        self._lock = threading.Lock()
        self._slots: dict[str, _Slot] = {}

    def _slot(self, source: str) -> _Slot:
        """The slot for ``source``, created on first use (caller holds the lock)."""
        slot = self._slots.get(source)
        if slot is None:
            slot = _Slot(semaphore=threading.BoundedSemaphore(self.max_per_source))
            self._slots[source] = slot
        return slot

    def in_flight(self, source: str) -> int:
        """How many calls to ``source`` are running right now."""
        with self._lock:
            return self._slots[source].in_flight if source in self._slots else 0

    def peak(self, source: str) -> int:
        """The highest concurrency ever observed for ``source`` (never above ``max_per_source``)."""
        with self._lock:
            return self._slots[source].peak if source in self._slots else 0

    def latency(self, source: str) -> tuple[float, ...]:
        """The completed latencies kept for ``source``, oldest first."""
        with self._lock:
            return tuple(self._slots[source].latencies) if source in self._slots else ()

    def _check_latency(self, source: str) -> None:
        """Refuse a new call when the median of the recent latencies is above the ceiling."""
        with self._lock:
            if source not in self._slots:
                return
            slot = self._slots[source]
            if len(slot.latencies) < self.samples:
                return
            ordered = sorted(slot.latencies)
            middle = len(ordered) // 2
            median = (
                ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
            )
            limit = self.latency_ceiling_s
        if median > limit:
            raise SourceTooSlow(
                f"{source} has slowed down: median of the last {len(ordered)} calls is "
                f"{median:.1f}s, above the {limit:.1f}s ceiling. Raise "
                f"SourceGate.latency_ceiling_s to keep going, or profile it later"
            )

    def try_acquire(self, source: str) -> bool:
        """Take a slot without waiting; ``True`` when it is yours (then :meth:`release` it)."""
        self._check_latency(source)
        with self._lock:
            slot = self._slot(source)
        if not slot.semaphore.acquire(blocking=False):
            return False
        self._enter(source)
        return True

    def acquire(self, source: str) -> None:
        """Take a slot, waiting at most ``wait_s``; raises :class:`SourceBusy` on timeout."""
        self._check_latency(source)
        with self._lock:
            slot = self._slot(source)
        if not slot.semaphore.acquire(timeout=self.wait_s):
            raise SourceBusy(
                f"{source} already has {self.max_per_source} call(s) in flight after "
                f"waiting {self.wait_s:.0f}s; the pipeline gives way instead of queueing"
            )
        self._enter(source)

    def _enter(self, source: str) -> None:
        with self._lock:
            slot = self._slots[source]
            slot.in_flight += 1
            slot.peak = max(slot.peak, slot.in_flight)

    def release(self, source: str, *, latency_s: float | None = None) -> None:
        """Give the slot back and record how long the call took (when the caller measured it)."""
        with self._lock:
            slot = self._slots.get(source)
            if slot is not None:
                slot.in_flight = max(0, slot.in_flight - 1)
                if latency_s is not None:
                    slot.latencies = [*slot.latencies, latency_s][-self.samples :]
        if slot is not None:
            slot.semaphore.release()

    @contextmanager
    def access(self, source: str) -> Iterator[None]:
        """Hold one of ``source``'s slots for the body, and time it on the way out."""
        self.acquire(source)
        started = self._clock.monotonic()
        try:
            yield
        finally:
            self.release(source, latency_s=self._clock.monotonic() - started)


# --- the context a stage runs in ------------------------------------------------------------


@dataclass
class PipelineContext:
    """What one pass shares with its stages: store, instant, dirs, model, budget.

    Mutable only through :meth:`write_generated`, which *buffers*: the runner commits the buffered
    sections after the stage returns, so a stage that fails half way leaves ``generated/`` alone.
    ``curated/`` is readable (:meth:`curated_path`) and has no writer here -- humans own it.
    """

    store: Store
    exec_ts: datetime
    semantic_root: Path
    quiet_windows: Sequence[QuietWindow] = ()
    llm: StructuredLLM | None = None
    source: str | None = None
    """``--source``: narrow every stage to one source (the runner sets it)."""
    force: bool = False
    """``--force``: set by the runner; a stage may read it to redo work it would otherwise reuse."""
    clock: Clock = field(default_factory=SystemClock)
    gate: SourceGate = field(default_factory=SourceGate)
    _pending: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    # -- output -------------------------------------------------------------------------
    def write_generated(self, section: str, payload: Mapping[str, Any]) -> str:
        """Buffer ``generated/<section>.yaml``; the runner writes it when the stage returns.

        Returns the path the file will have, relative to the semantic root, so a stage can name its
        own output in :attr:`StageReport.outputs` without knowing the root.
        """
        if not _SECTION_NAME.match(section):
            raise ValueError(
                f"{section!r} is not a section name: use letters, digits, '_', '-' or '.' "
                "(a section is a file stem under generated/, never a path)"
            )
        self._pending[section] = dict(payload)
        return f"{GENERATED_DIR}/{section}.yaml"

    def commit(self) -> tuple[str, ...]:
        """Write the buffered sections, each through a temp file and an atomic rename."""
        if not self._pending:
            return ()
        for section in self._pending:
            if not _SECTION_NAME.match(section):
                raise ValueError(f"{section!r} is not a section name")
        written = tuple(
            write_generated(self.semantic_root, section, self._pending[section]).relative_to(
                self.semantic_root
            )
            for section in sorted(self._pending)
        )
        self._pending.clear()
        return tuple(path.as_posix() for path in written)

    def discard(self) -> None:
        """Drop the buffered sections without writing anything (what a failed stage gets)."""
        self._pending.clear()

    @property
    def pending(self) -> tuple[str, ...]:
        """The section names buffered so far, sorted: what :meth:`commit` would write."""
        return tuple(sorted(self._pending))

    def outputs_present(self, outputs: Sequence[str]) -> bool:
        """True when every recorded output path still exists under ``generated/``.

        A hash-matched skip is only safe while the file it produced is still there: an operator who
        deleted a section expects the next pass to regenerate it, not to be told "unchanged".
        """
        return all((self.semantic_root / name).is_file() for name in outputs)

    # -- input --------------------------------------------------------------------------
    def generated_path(self, section: str) -> Path:
        """Where ``generated/<section>.yaml`` lives (readable whether or not it exists yet)."""
        return self.semantic_root / GENERATED_DIR / f"{section}.yaml"

    def curated_path(self, section: str) -> Path:
        """Where ``curated/<section>.yaml`` lives. Read it; the pipeline never writes it."""
        return self.semantic_root / CURATED_DIR / f"{section}.yaml"

    @contextmanager
    def access(self, *sources: str) -> Iterator[None]:
        """Hold one slot of every named source for the body, sorted so two cannot deadlock."""
        with ExitStack() as stack:
            for source in sorted(set(sources)):
                stack.enter_context(self.gate.access(source))
            yield


# --- the runner -----------------------------------------------------------------------------


def is_heavy(stage: Stage) -> bool:
    """True when the stage declares ``heavy = True``: it needs a quiet window and a source slot."""
    return bool(getattr(stage, "heavy", False))


def declared_sources(stage: Stage) -> tuple[str, ...]:
    """The sources a stage says it touches (``sources = ("crm", ...)``), empty when undeclared."""
    declared = getattr(stage, "sources", ())
    return tuple(str(source) for source in declared)


def _json_list(value: object) -> tuple[str, ...]:
    """A JSON string column as a tuple of strings; a damaged row reports nothing, never raises."""
    try:
        decoded = json.loads(value) if isinstance(value, str) else None
    except ValueError:
        logger.debug("a pipeline_run row holds unreadable JSON")
        return ()
    if not isinstance(decoded, list):
        return ()
    return tuple(str(item) for item in decoded)


def _last_success(store: Store, stage: str) -> tuple[str, tuple[str, ...]] | None:
    """The input hash and outputs of a stage's last successful run, or ``None``."""
    # Constant statement, value bound: nothing is interpolated into the SQL text.
    row = run_local(store.connect(), _SELECT_LAST_RAN, (stage,)).fetchone()
    if row is None:
        return None
    return (str(row["inputs_hash"]), _json_list(row["outputs"]))


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def _error_text(exc: BaseException) -> str:
    """The failure as one bounded line: no traceback, no newline, nothing unbounded in the store."""
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text[:ERROR_LIMIT]}"


def _record(ctx: PipelineContext, report: StageReport, *, started: datetime) -> None:
    """Write one attempt's row. One transaction per stage, so a failure keeps earlier rows."""
    with ctx.store.writing() as conn:
        # Constant statement, every value bound.
        run_local(
            conn,
            _INSERT_RUN,
            (
                report.stage,
                report.inputs_hash,
                report.status.value,
                _iso(started),
                _iso(ctx.clock.now()),
                json.dumps(report.counts, sort_keys=True),
                json.dumps(list(report.skipped), ensure_ascii=False),
                json.dumps(list(report.applied_fixes), ensure_ascii=False),
                json.dumps(list(report.outputs)),
                report.error,
                1 if report.forced else 0,
            ),
        )


def _run_stage(
    stage: Stage,
    ctx: PipelineContext,
    *,
    digest: str,
    force: bool,
) -> StageReport:
    """Execute one stage: source slots, buffered output, and a report whatever happens."""
    started = ctx.clock.now()
    sources = declared_sources(stage) if is_heavy(stage) else ()
    try:
        with ctx.access(*sources):
            report = stage.run(ctx)
        outputs = ctx.commit()
    except Exception as exc:  # a stage boundary: anything a stage raises is reported, not fatal
        ctx.discard()
        logger.warning(
            "bootstrap stage failed",
            exc_info=True,
            extra={"stage": stage.name, "source": ctx.source},
        )
        report = StageReport(
            stage=stage.name,
            status=StageStatus.FAILED,
            inputs_hash=digest,
            error=_error_text(exc),
        )
        _record(ctx, report, started=started)
        return report
    # The runner owns these fields: a stage reports what it measured, never how it "counts" as run.
    report = report.model_copy(
        update={
            "stage": stage.name,
            "status": StageStatus.RAN,
            "inputs_hash": digest,
            "outputs": outputs,
            "forced": force,
            "error": None,
        }
    )
    _record(ctx, report, started=started)
    return report


def run_pipeline(
    stages: Sequence[Stage],
    ctx: PipelineContext,
    *,
    stage: str | None = None,
    source: str | None = None,
    force: bool = False,
) -> PipelineResult:
    """Run ``stages`` in order and report every one of them.

    ``stage`` runs exactly one stage (``None`` runs them all), ``source`` narrows the pass to one
    source, and ``force`` overrides both the input-hash skip and the quiet-window refusal.

    The pass stops at the first failure (so a rerun resumes there) and at the first quiet-window
    refusal (so the whole pass waits for the window). A stage filtered out by ``source`` is reported
    as skipped and is *not* recorded in the ledger: it did not run against that source at all.
    """
    ctx.force = force
    ctx.source = source
    names = [candidate.name for candidate in stages]
    if stage is not None and stage not in names:
        raise ValueError(f"no stage named {stage!r}; this build has: {', '.join(names) or 'none'}")
    ctx.store.initialize()  # idempotent: a fresh deployment is migrated, a current one only read
    reports: list[StageReport] = []
    deferred: datetime | None = None
    aborted = False
    for candidate in stages:
        if stage is not None and candidate.name != stage:
            continue
        declared = declared_sources(candidate)
        if source is not None and declared and source not in declared:
            reports.append(
                StageReport(
                    stage=candidate.name,
                    status=StageStatus.SKIPPED,
                    skipped=(f"does not read source {source}",),
                )
            )
            continue
        digest = candidate.inputs_hash(ctx)
        previous = None if force else _last_success(ctx.store, candidate.name)
        if previous is not None and previous[0] == digest and ctx.outputs_present(previous[1]):
            report = StageReport(
                stage=candidate.name,
                status=StageStatus.SKIPPED,
                inputs_hash=digest,
                skipped=("inputs unchanged since the last successful run",),
            )
            _record(ctx, report, started=ctx.clock.now())
            reports.append(report)
            continue
        if is_heavy(candidate) and not force and not quiet_allows(ctx.quiet_windows, ctx.exec_ts):
            message, instant = _refusal(candidate.name, ctx.quiet_windows, ctx.exec_ts)
            report = StageReport(
                stage=candidate.name,
                status=StageStatus.REFUSED,
                inputs_hash=digest,
                error=message,
                deferred_until=instant,
            )
            _record(ctx, report, started=ctx.clock.now())
            reports.append(report)
            deferred = instant
            break
        report = _run_stage(candidate, ctx, digest=digest, force=force)
        reports.append(report)
        if report.status is StageStatus.FAILED:
            aborted = True
            break
    return PipelineResult(
        reports=tuple(reports),
        aborted=aborted,
        deferred_until=None if deferred is None else deferred,
    )


#: The run order of the bootstrap stages: read the catalogs, then everything measured or proposed
#: from them, and the name proposals last because they read the earlier stages' output. A stage a
#: build adds and does not list here runs after these, by name, so an unlisted stage can never
#: silently land in the middle of the sequence.
STAGE_ORDER: Final[tuple[str, ...]] = ("D1+D2", "D4", "D5", "D6", "D7", "D8", "D10", "A1+A2")


def available_stages(stages: Sequence[Stage]) -> tuple[Stage, ...]:
    """The stages a caller built, in run order.

    The caller -- ``mdqm.pipeline.cli`` -- is the one place with settings, secrets, a store and the
    adapters, so it builds the objects; this function only decides the order, which keeps this
    module free of any stage import (every stage imports it) and keeps the order in one reviewable
    place. Ties keep the caller's order, so two instances of one stage (D8 per measure) stay in the
    order they were given.
    """
    rank = {name: position for position, name in enumerate(STAGE_ORDER)}
    return tuple(sorted(stages, key=lambda stage: (rank.get(stage.name, len(rank)), stage.name)))
