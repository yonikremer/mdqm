"""Stage ``sizes``: the nightly measurement, as a bootstrap stage instead of a cron entry.

What each block proves:

* **derived presence** -- only an adapter that declares ``sizes()`` reaches the stage;
* **idempotency** -- a second pass on the same day skips, because the input hash covers the day;
  the next day re-measures, because sizes move;
* **the quiet window** -- a heavy stage refuses outside its window unless forced, and the direct
  caller gets the same rule as the runner;
* **one capture per pass** -- every source records under the single injected ``exec_ts``, so the
  newest capture is a set of rows measured together;
* **nothing silent** -- a source that cannot report sizes, or refuses to, is named in the report.

No network, no Docker, no sleeping: the adapters are doubles and the store is ``tmp_path``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import cast

import pytest

from mdqm.adapters.base import NotSupported, SourceAdapter
from mdqm.config.quiet import QuietWindow
from mdqm.models import PartitionSize, TableSize
from mdqm.pipeline.core import PipelineContext, StageStatus, run_pipeline
from mdqm.pipeline.stages.sizes import STAGE_NAME, QuietWindowClosed, SizesStage
from mdqm.store.db import Store
from mdqm.store.sizes import SizeStore

EXEC_TS = datetime(2026, 11, 28, 2, 0, tzinfo=UTC)
NIGHT = QuietWindow(tz="UTC", start=time(21, 0), end=time(7, 0))


class Clock:
    """A clock the test drives: ``now()`` always answers the instant the pass was given."""

    def __init__(self, instant: datetime = EXEC_TS) -> None:
        self._instant = instant

    def now(self) -> datetime:
        return self._instant

    def monotonic(self) -> float:
        return 0.0


def _sizes(source: str, *, partitions: int = 2) -> list[TableSize]:
    return [
        TableSize(
            source=source,
            table="usage_events",
            captured_at=EXEC_TS,
            partitions=tuple(
                PartitionSize(name=f"usage_events_p2026_{month:02d}", rows=1_000, bytes=1_000_000)
                for month in range(1, partitions + 1)
            ),
        )
    ]


class SizedAdapter:
    """An adapter that reports sizes and counts how often it was asked."""

    def __init__(self, sizes: list[TableSize] | None = None) -> None:
        self._sizes = sizes if sizes is not None else _sizes("billing")
        self.reads = 0

    def sizes(self) -> list[TableSize]:
        self.reads += 1
        return self._sizes


class UnsizedAdapter:
    """An adapter with no ``sizes()`` at all: the capability is absent, not a failure."""


class RaisingAdapter:
    """An adapter that declares ``sizes()`` and refuses, the way a read-only role would."""

    def sizes(self) -> list[TableSize]:
        raise NotSupported("size_stats", "ALL_SEGMENTS is not granted to this account")


def _stage(*adapters: tuple[str, object]) -> SizesStage:
    """A stage over doubles: the protocol is structural, so a double needs only ``sizes()``."""
    return SizesStage(cast(Mapping[str, SourceAdapter], dict(adapters)))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    opened = Store(tmp_path / "metadata.sqlite")
    opened.initialize()
    try:
        yield opened
    finally:
        opened.close()


def _ctx(store: Store, tmp_path: Path, *, instant: datetime = EXEC_TS) -> PipelineContext:
    return PipelineContext(
        store=store,
        exec_ts=instant,
        semantic_root=tmp_path / "semantic",
        clock=Clock(instant),
    )


# --- derived presence ---------------------------------------------------------------------


def test_only_sources_that_declare_sizes_are_measured(store: Store, tmp_path: Path) -> None:
    """``object()`` has no ``sizes()``: promising a measurement for it would be an invention."""
    stage = _stage(("billing", SizedAdapter()), ("api", UnsizedAdapter()), ("erp", object()))

    assert stage.sources == ("billing",)
    report = stage.run(_ctx(store, tmp_path))

    assert report.status is StageStatus.RAN
    assert report.counts == {"sources": 1, "tables": 1, "partitions": 2}


def test_a_source_that_refuses_is_reported_and_the_others_still_run(
    store: Store, tmp_path: Path
) -> None:
    stage = _stage(("billing", SizedAdapter()), ("oracle", RaisingAdapter()))

    report = stage.run(_ctx(store, tmp_path))

    assert report.status is StageStatus.RAN
    assert report.counts["sources"] == 1, "a refusal is a capability fact, not a measurement"
    assert report.skipped == (
        "oracle: sizes are not reported by this source (size_stats: ALL_SEGMENTS is not granted "
        "to this account)",
    )
    assert SizeStore(store).latest("oracle", "usage_events") is None


def test_an_empty_adapter_map_means_no_source() -> None:
    assert SizesStage({}).sources == ()


# --- one capture per pass -----------------------------------------------------------------


def test_every_source_records_under_the_single_exec_ts(store: Store, tmp_path: Path) -> None:
    """One capture for the pass: two per-source clocks would look like two different nights."""
    stage = _stage(("billing", SizedAdapter()), ("crm", SizedAdapter(_sizes("crm", partitions=1))))

    stage.run(_ctx(store, tmp_path))

    sizes = SizeStore(store)
    assert sizes.latest_capture() == EXEC_TS
    assert sizes.latest("crm", "usage_events") is not None
    assert sizes.latest("billing", "usage_events") is not None
    assert len(sizes.captures()) == 1


# --- skip, resume, quiet window -----------------------------------------------------------


def test_a_second_pass_on_the_same_day_skips_and_the_next_day_re_measures(
    store: Store, tmp_path: Path
) -> None:
    adapter = SizedAdapter()
    stage = _stage(("billing", adapter))
    ctx = _ctx(store, tmp_path)

    first = run_pipeline([stage], ctx)
    second = run_pipeline([stage], ctx)
    tomorrow = run_pipeline([stage], _ctx(store, tmp_path, instant=EXEC_TS + timedelta(days=1)))

    assert [report.status for report in first.reports] == [StageStatus.RAN]
    assert [report.status for report in second.reports] == [StageStatus.SKIPPED]
    assert [report.status for report in tomorrow.reports] == [StageStatus.RAN]
    assert adapter.reads == 2, "the source is read once per day, never twice"


def test_a_heavy_stage_is_refused_outside_its_quiet_window(store: Store, tmp_path: Path) -> None:
    stage = _stage(("billing", SizedAdapter()))
    daytime = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)
    ctx = PipelineContext(
        store=store,
        exec_ts=daytime,
        semantic_root=tmp_path / "semantic",
        quiet_windows=(NIGHT,),
        clock=Clock(daytime),
    )

    result = run_pipeline([stage], ctx)

    assert [report.status for report in result.reports] == [StageStatus.REFUSED]
    assert result.deferred_until is not None
    assert SizeStore(store).latest_capture() is None


def test_a_direct_caller_gets_the_same_quiet_window_rule(store: Store, tmp_path: Path) -> None:
    stage = _stage(("billing", SizedAdapter()))
    daytime = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)
    ctx = PipelineContext(
        store=store,
        exec_ts=daytime,
        semantic_root=tmp_path / "semantic",
        quiet_windows=(NIGHT,),
        clock=Clock(daytime),
    )

    with pytest.raises(QuietWindowClosed):
        stage.run(ctx)


def test_force_measures_inside_the_closed_window(store: Store, tmp_path: Path) -> None:
    stage = _stage(("billing", SizedAdapter()))
    daytime = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)
    ctx = PipelineContext(
        store=store,
        exec_ts=daytime,
        semantic_root=tmp_path / "semantic",
        quiet_windows=(NIGHT,),
        clock=Clock(daytime),
    )

    result = run_pipeline([stage], ctx, force=True)

    assert [report.status for report in result.reports] == [StageStatus.RAN]


# --- the wiring and the narrowing ---------------------------------------------------------


def test_the_stage_is_declared_in_the_run_order() -> None:
    """An unlisted stage runs after every listed one by name; ``sizes`` is declared, not implicit."""
    from mdqm.pipeline.core import STAGE_ORDER, available_stages

    assert STAGE_NAME in STAGE_ORDER
    ordered = available_stages([_stage(("billing", SizedAdapter()))])
    assert [stage.name for stage in ordered] == [STAGE_NAME]


def test_the_stage_inputs_hash_separates_two_days_and_two_source_lists(
    store: Store, tmp_path: Path
) -> None:
    stage = _stage(("billing", SizedAdapter()))
    today = _ctx(store, tmp_path)
    tomorrow = _ctx(store, tmp_path, instant=EXEC_TS + timedelta(days=1))

    assert stage.inputs_hash(today) != stage.inputs_hash(tomorrow)
    assert stage.inputs_hash(today) != _stage(("crm", SizedAdapter())).inputs_hash(today)


def test_narrowing_to_a_source_this_stage_cannot_measure_is_an_error(
    store: Store, tmp_path: Path
) -> None:
    stage = _stage(("billing", SizedAdapter()))
    ctx = _ctx(store, tmp_path)
    ctx.source = "api"

    with pytest.raises(ValueError, match="does not report sizes"):
        stage.run(ctx)
