"""Stage ``sizes``: measure every source's table and partition sizes, nightly, on a schedule.

The measurement itself existed as a command (``mdqm sizes measure``) and nothing called it, so the
profile a plan is estimated from went stale the moment the first nightly load landed and stayed
stale. This stage is that command as a **bootstrap stage**, so a nightly ``mdqm bootstrap run``
keeps the profile current without a second scheduler (cron, systemd timer, a container sidecar):
there is one place that decides what runs at night, and it is
:func:`~mdqm.pipeline.core.run_pipeline`.

The rules it inherits from the stage framework, and why each one is right here:

* **Derived presence.** A source appears only if its adapter declares ``sizes()``. An adapter that
  does not is named in the report instead of being measured as zero rows -- "not reported" is an
  answer, and a fabricated measurement is the one thing a size store must never contain.
* **Skippable when unchanged.** The input hash covers the sources **and the date** of ``exec_ts``,
  exactly as D5 and D6 do: sizes move every day, so tonight re-measures and a second pass on the
  same day skips. Nothing in the hash comes from the data or from this stage's own output.
* **Quiet-window aware.** The stage is heavy: ``pg_total_relation_size`` on every relation of a
  primary database is real work on a database that has no replica. The runner refuses it outside a
  configured window, and :meth:`SizesStage.run` repeats the check so a direct caller cannot skip it.
* **One capture per pass.** Every source records under one ``ctx.exec_ts``, so "the newest capture"
  is a set of rows that were measured together rather than a smear of per-source instants.

The store is the metadata SQLite beside the ledger, deliberately not the semantic layer: a nightly
size would change the layer's version hash nightly and invalidate the context index and every cache
key with it (``progress/SIZES.md``).

Deterministic: no wall clock is read (``exec_ts`` decides the day and the capture instant), every
collection is sorted, and a source that cannot answer is reported rather than retried.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Final, cast

from mdqm.adapters.base import NotSupported, SourceAdapter, supports_method
from mdqm.models.sizes import TableSize
from mdqm.pipeline.core import (
    PipelineContext,
    StageReport,
    StageStatus,
    hash_inputs,
    next_quiet_start,
    quiet_allows,
)
from mdqm.store.sizes import SizeStore

__all__ = ["STAGE_NAME", "STAGE_VERSION", "QuietWindowClosed", "SizesStage"]

logger = logging.getLogger(__name__)

STAGE_NAME: Final[str] = "sizes"

#: Bumped when the measurement itself changes, so a rerun replaces a reading taken the wrong way.
STAGE_VERSION: Final[int] = 1


class QuietWindowClosed(RuntimeError):
    """Raised when this heavy stage is called outside its quiet window and was not forced."""


class SizesStage:
    """Measure the tables and partitions of every source that can report them.

    ``sources`` is derived, not configured: an adapter without ``sizes()`` never appears, so the
    stage cannot promise a measurement for a source that has no way to take one.
    """

    name: str = STAGE_NAME
    heavy: bool = True

    def __init__(self, adapters: Mapping[str, SourceAdapter]) -> None:
        self._adapters = {
            name: adapter
            for name, adapter in dict(adapters).items()
            if supports_method(adapter, "sizes")
        }

    @property
    def sources(self) -> tuple[str, ...]:
        """The sources that declare ``sizes()``, sorted (what the runner gates and filters on)."""
        return tuple(sorted(self._adapters))

    def inputs_hash(self, ctx: PipelineContext) -> str:
        """sha256 over the declared inputs: the sources and the day being measured.

        The **date** is deliberate, as in D5 and D6: rows arrive every day, so a nightly pass must
        re-measure, while two passes on one day measure the same day and the second one skips.
        """
        return hash_inputs(
            self.name,
            STAGE_VERSION,
            sorted(self._adapters),
            ctx.source,
            ctx.exec_ts.date().isoformat(),
        )

    def run(self, ctx: PipelineContext) -> StageReport:
        """Read each selected source's sizes and record them as one capture."""
        if not ctx.force and not quiet_allows(ctx.quiet_windows, ctx.exec_ts):
            raise QuietWindowClosed(self._refusal(ctx))
        selected = (ctx.source,) if ctx.source is not None else self.sources
        if ctx.source is not None and ctx.source not in self._adapters:
            raise ValueError(
                f"source {ctx.source!r} does not report sizes; this stage measures: "
                f"{', '.join(self.sources) or 'no source'}"
            )
        store = SizeStore(ctx.store)
        counts: dict[str, int] = {"sources": 0, "tables": 0, "partitions": 0}
        skipped: list[str] = []
        for source in selected:
            measured = self._measure(source, skipped=skipped)
            if measured is None:
                continue
            store.record(measured, captured_at=ctx.exec_ts)
            counts["sources"] += 1
            counts["tables"] += len(measured)
            counts["partitions"] += sum(len(size.partitions) for size in measured)
            logger.info(
                "sizes measured",
                extra={
                    "source": source,
                    "tables": len(measured),
                    "captured_at": ctx.exec_ts.isoformat(),
                },
            )
        return StageReport(
            stage=self.name,
            status=StageStatus.RAN,
            counts=counts,
            skipped=tuple(skipped),
        )

    def _measure(self, source: str, *, skipped: list[str]) -> tuple[TableSize, ...] | None:
        """One source's reading, or ``None`` when it declined -- a capability fact, not a failure."""
        adapter = self._adapters[source]
        # The optional protocol member, reached through a plain lookup: absent on many adapters.
        read_sizes = getattr(adapter, "sizes")  # noqa: B009 - optional protocol member
        collect = cast(Callable[[], Sequence[TableSize]], read_sizes)
        try:
            sizes = tuple(collect())
        except NotSupported as exc:
            skipped.append(f"{source}: sizes are not reported by this source ({exc})")
            return None
        if not sizes:
            skipped.append(f"{source}: reported no tables to size")
            return None
        return sizes

    def _refusal(self, ctx: PipelineContext) -> str:
        """The message a direct caller sees: same rule and same next-window hint as the runner."""
        upcoming = next_quiet_start(ctx.quiet_windows, ctx.exec_ts)
        detail = (
            f"the next window opens at {upcoming[0].isoformat()}"
            if upcoming is not None
            else "no quiet window is configured so none ever opens"
        )
        return (
            f"stage {self.name} is heavy and its quiet window is closed ({detail}); run it inside "
            "the window or pass --force to measure sizes now"
        )
