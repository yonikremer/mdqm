"""Stage D6: how often each table changes, and how late its rows arrive -- both measured.

Two facts the rest of the product refuses to guess:

* **mutability** (``docs/DESIGN.md`` section 8) decides whether a cached segment may be served.
  It is measured from ``pg_stat_user_tables`` counters (A04) taken **twice**, with real work
  between the readings, so the evidence is a delta over a stated window rather than one cumulative
  number that says nothing about *when* the rows changed.
* **lag** (DESIGN section 6) is the bound the planner subtracts from an ingestion-time filter to
  derive an event-time one (C07). A derived bound with a guessed lag silently drops late rows, so
  ``lag_unverified`` may only be cleared by a measurement like the one this stage makes.

The rules that make a class a *measurement* and not a guess:

* **No counters, no class.** An adapter without the ``mutability_counters`` capability, a role that
  cannot read ``pg_stat_user_tables``, or a partitioned parent whose partition list we could not
  enumerate all leave the table ``unknown`` -- which behaves exactly like ``mutable_critical``.
  ``unknown`` is also the answer for a table whose counters are present but all zero: a table
  nothing has written yet proves nothing about whether it will change.
* **An update is never "just a new row".** ``n_tup_upd`` cannot say which row was updated, so
  ``updates > 0`` can never be read as "only the tail". Any update, any delete, or a changed
  checksum over the oldest sampled rows makes the table ``mutable_critical``.
* **A sealed range that can still gain a row is not append-only.** ``append_only`` keeps sealed
  segments until ``max_ttl``; that promise holds only when no row can appear inside an
  already-sealed event-time range. A table with a *measured* positive ingestion lag (a late row
  lands hours after its event) therefore gets ``append_rarely``, whose policy is tail refresh plus
  the nightly checksum this stage also records.
* **A measured lag is the only lag.** The stage writes ``lag: {min, max, verify: nightly}`` with the
  sample size and the window it measured over, and it re-measures on every pass: an observation
  above the declared maximum raises ``lag_violated``, keeps the declared maximum, and names the
  column in the stored ``lag_unverified`` set instead of silently widening the bound.

Deterministic: no wall clock is read except to time the counter window (``ctx.clock``), every
collection is sorted, and the sample sizes are inputs, so two passes on the same inputs agree.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import pyarrow as pa
from pydantic import Field
from sqlglot import exp

from mdqm.adapters.base import NotSupported, SourceAdapter, supports_method
from mdqm.cache.freshness import Mutability
from mdqm.models.base import MDQMModel, UtcDatetime
from mdqm.models.catalog import CatalogSnapshot, Table, TableKind
from mdqm.models.fragment import Fragment
from mdqm.models.limits import Limits
from mdqm.models.partitioning import Partitioning
from mdqm.pipeline.core import (
    PipelineContext,
    StageReport,
    StageStatus,
    hash_inputs,
    quiet_allows,
)
from mdqm.semantic.loader import load as load_layer
from mdqm.semantic.models import SemanticLayer
from mdqm.sql.guard import guard

__all__ = [
    "LAG_MARGIN",
    "LAG_UNVERIFIED_PREFIX",
    "STAGE_NAME",
    "STAGE_VERSION",
    "ClassifiedTable",
    "CounterSet",
    "LagMeasurement",
    "MutabilityStage",
    "OldRowCheck",
    "QuietWindowClosed",
    "TableEvidence",
    "classify",
    "format_duration",
    "percentiles",
    "read_lag_unverified",
]

logger = logging.getLogger(__name__)

STAGE_NAME: Final[str] = "D6"

#: Bumped when the measurement itself changes, so a rerun refreshes a fact sheet that was wrong.
STAGE_VERSION: Final[int] = 1

#: How many rows the lag distribution is measured over: a bounded sample, never a scan.
DEFAULT_LAG_SAMPLE: Final[int] = 5_000

#: How many of a table's *oldest* rows the cross-run checksum covers.
DEFAULT_OLD_ROW_SAMPLE: Final[int] = 1_000

#: Slack applied to a measured maximum lag before it is written as the bound: `max` must never be
#: short, because C07 subtracts it and a short bound drops late rows.
LAG_MARGIN: Final[float] = 1.2

#: A written bound is rounded up to this granularity, so a noisy second becomes a readable duration.
LAG_ROUND_SECONDS: Final[int] = 60

#: ``mdqm_meta`` key prefix holding the ingestion columns whose lag verification failed.
LAG_UNVERIFIED_PREFIX: Final[str] = "lag_unverified"

#: ``mdqm_meta`` key prefix holding the previous pass's checksums, lag maxima and window.
BASELINE_PREFIX: Final[str] = "mutability_baseline"

#: Column names a source uses for *when we learned about the row*. A recognised name is treated as
#: the ingestion-time candidate when the semantic layer does not declare one; see `_time_pair`.
INGESTION_NAME_HINTS: Final[tuple[str, ...]] = (
    "insertion_time",
    "insert_time",
    "ingestion_time",
    "ingested_at",
    "loaded_at",
    "load_time",
    "etl_time",
)

_SECTION_STEM: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.-]")


class QuietWindowClosed(RuntimeError):
    """Raised when a heavy stage is called outside its quiet window and was not forced."""


# --- the measured facts -------------------------------------------------------------------


class CounterSet(MDQMModel):
    """``pg_stat_user_tables`` counters at one instant. ``None`` means this source has no such
    counter, which is different from a zero."""

    inserted: int | None = None
    updated: int | None = None
    deleted: int | None = None
    hot_updated: int | None = None

    @property
    def available(self) -> bool:
        """True when every counter the classification reads is present."""
        return self.inserted is not None and self.updated is not None and self.deleted is not None

    @property
    def changed(self) -> bool:
        """True when any insert, update or delete is recorded (a delta, or a lifetime total)."""
        return any(value for value in (self.inserted, self.updated, self.deleted))

    def minus(self, earlier: CounterSet) -> CounterSet | None:
        """The change between two readings, or ``None`` when a counter went backwards.

        A counter that decreases means the server's statistics were reset between the readings:
        the delta is meaningless, and reporting it would be a fabricated measurement.
        """

        def delta(name: str) -> int | None:
            mine: int | None = getattr(self, name)
            theirs: int | None = getattr(earlier, name)
            if mine is None or theirs is None:
                return None
            return mine - theirs

        values = {name: delta(name) for name in ("inserted", "updated", "deleted", "hot_updated")}
        if any(value is not None and value < 0 for value in values.values()):
            return None
        return CounterSet(**values)


class LagMeasurement(MDQMModel):
    """The observed distribution of ``ingestion_time - event_time`` over one bounded sample."""

    event_column: str = Field(min_length=1)
    ingestion_column: str = Field(min_length=1)
    rows_measured: int = Field(ge=0)
    rows_skipped: int = Field(ge=0)
    minimum_s: float
    p50_s: float
    p99_s: float
    maximum_s: float
    window_from: UtcDatetime
    window_to: UtcDatetime

    @property
    def late_rows(self) -> int:
        """Rows whose ingestion time is later than their event time: the late-arrival count."""
        return self.rows_measured if self.maximum_s > 0 else 0

    @property
    def evidence(self) -> str:
        """One line an operator can read: the distribution, the sample and the window."""
        return (
            f"{self.ingestion_column} - {self.event_column} over {self.rows_measured} rows "
            f"({self.rows_skipped} skipped, NULL) between {self.window_from.isoformat()} and "
            f"{self.window_to.isoformat()}: min={self.minimum_s:.0f}s "
            f"p50={self.p50_s:.0f}s p99={self.p99_s:.0f}s max={self.maximum_s:.0f}s"
        )


class OldRowCheck(MDQMModel):
    """A checksum over the table's oldest sampled rows, compared with the previous pass."""

    columns: tuple[str, ...] = ()
    rows: int = Field(ge=0)
    checksum: str = Field(min_length=1)
    previous_checksum: str | None = None

    @property
    def changed(self) -> bool | None:
        """True when the oldest rows differ from the previous pass; ``None`` on a first pass."""
        if self.previous_checksum is None:
            return None
        return self.previous_checksum != self.checksum


class TableEvidence(MDQMModel):
    """Everything the classification of one table rests on, in one place.

    A class without one of these is a guess, so the stage refuses to emit a class without it.
    """

    source: str = Field(min_length=1)
    table: str = Field(min_length=1)
    lifetime: CounterSet | None = None
    delta: CounterSet | None = None
    window_from: UtcDatetime
    window_to: UtcDatetime
    partitions_rolled_up: int = Field(default=0, ge=0)
    row_estimate: int | None = Field(default=None, ge=0)
    lag: LagMeasurement | None = None
    old_rows: OldRowCheck | None = None
    notes: tuple[str, ...] = ()

    @property
    def window_seconds(self) -> float:
        return (self.window_to - self.window_from).total_seconds()

    def describe(self) -> tuple[str, ...]:
        """The evidence as YAML-friendly lines, next to the class they justify."""
        lines: list[str] = []
        counters = self.lifetime
        if counters is None:
            lines.append("no change counters were readable for this table")
        else:
            lines.append(
                f"lifetime counters since the server's statistics reset: "
                f"inserted={counters.inserted} updated={counters.updated} "
                f"deleted={counters.deleted} hot_updated={counters.hot_updated}"
            )
        delta = self.delta
        window = f"{self.window_seconds:.1f}s"
        if delta is None:
            lines.append(f"counter delta over {window}: unreadable (statistics were reset)")
        else:
            lines.append(
                f"delta over {window} from {self.window_from.isoformat()} to "
                f"{self.window_to.isoformat()}: inserted={delta.inserted} "
                f"updated={delta.updated} deleted={delta.deleted}"
            )
        if self.partitions_rolled_up:
            lines.append(f"counters summed over {self.partitions_rolled_up} partitions")
        if self.lag is not None:
            lines.append(f"lag: {self.lag.evidence}")
        if self.old_rows is not None:
            verdict = self.old_rows.changed
            state = "no previous pass" if verdict is None else ("changed" if verdict else "same")
            lines.append(
                f"oldest {self.old_rows.rows} rows checksum {self.old_rows.checksum[:16]}: {state}"
            )
        lines.extend(self.notes)
        return tuple(lines)


class ClassifiedTable(MDQMModel):
    """One table's class and the reasons that produced it, in the order they were checked."""

    class_: Mutability
    reasons: tuple[str, ...] = ()
    evidence: TableEvidence

    @property
    def label(self) -> str:
        return f"{self.evidence.source}.{self.evidence.table}"


# --- classification (pure) ----------------------------------------------------------------


def classify(evidence: TableEvidence) -> ClassifiedTable:
    """Turn measured evidence into a mutability class, conservatively and explainably.

    The order is the policy: every rule is checked in it, and the first that fires wins, so the
    reasons read as "why this class and not the safer-looking one".
    """
    lifetime = evidence.lifetime
    if lifetime is None or not lifetime.available:
        return _class(
            evidence,
            "unknown",
            "no change counters are readable, so nothing about how this table changes is known",
        )
    delta = evidence.delta
    if delta is None:
        return _class(
            evidence,
            "unknown",
            "the change counters were reset between the two readings, so no window was measured",
        )
    if (lifetime.deleted or 0) > 0:
        return _class(
            evidence,
            "mutable_critical",
            f"{lifetime.deleted} row(s) have been deleted: a deleted row invalidates a cached "
            "segment, so this table is never served from cache",
        )
    if (lifetime.updated or 0) > 0:
        return _class(
            evidence,
            "mutable_critical",
            f"{lifetime.updated} row(s) have been updated in place and the counters cannot say "
            "which rows those were, so an already-served range may have changed",
        )
    old_rows = evidence.old_rows
    if old_rows is not None and old_rows.changed is True:
        return _class(
            evidence,
            "mutable_critical",
            "the checksum of the oldest sampled rows changed since the previous pass: an old row "
            "was rewritten or a row older than the sample arrived",
        )
    lag = evidence.lag
    if lag is not None and lag.maximum_s > 0:
        return _class(
            evidence,
            "append_rarely",
            f"no update or delete was ever counted, but {lag.ingestion_column} - "
            f"{lag.event_column} measured up to {lag.maximum_s:.0f}s, so a row can still land "
            "inside an already-sealed event-time range; needs the nightly checksum",
        )
    if not lifetime.changed:
        return _class(
            evidence,
            "unknown",
            "the counters are present but record no insert, update or delete yet: a table nothing "
            "has written proves nothing about whether it will change",
        )
    return _class(
        evidence,
        "append_only",
        "inserts only, no update or delete ever counted, and no measured lag can place a row "
        "inside an already-sealed range",
    )


def _class(evidence: TableEvidence, name: Mutability, *reasons: str) -> ClassifiedTable:
    return ClassifiedTable(class_=name, reasons=reasons, evidence=evidence)


# --- small pure helpers -------------------------------------------------------------------


def format_duration(value: timedelta) -> str:
    """A duration as the semantic layer writes it: whole seconds, never a bare number."""
    return f"{round(value.total_seconds())}s"


def percentiles(values: Sequence[float]) -> tuple[float, float, float, float]:
    """``(min, p50, p99, max)`` by nearest rank, or zeros for an empty sample.

    Nearest rank rather than an interpolated percentile: the bound C07 derives must be an observed
    value rounded up, never a number a distribution model invented.
    """
    if not values:
        return (0.0, 0.0, 0.0, 0.0)
    ordered = sorted(values)

    def rank(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return ordered[index]

    return (ordered[0], rank(0.50), rank(0.99), ordered[-1])


def _slug(name: str) -> str:
    return _SECTION_STEM.sub("_", name).strip(".") or "source"


def _rows(batch: pa.RecordBatch) -> list[dict[str, Any]]:
    """One Arrow batch as plain dicts: the values are read, never the schema."""
    rows: list[dict[str, Any]] = batch.to_pylist()
    return rows


def _as_time(value: object) -> datetime | None:
    """A timestamp column value as an aware UTC datetime, or ``None`` when it is absent."""
    if not isinstance(value, datetime):
        return None
    return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    """sha256 over the sorted canonical text of the rows: order-insensitive, value-sensitive."""
    texts = sorted(
        json.dumps(_jsonable(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for row in rows
    )
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _jsonable(row: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): "" if value is None else str(value) for key, value in row.items()}


# --- the stage ----------------------------------------------------------------------------


class MutabilityStage:
    """Stage D6: measure every table's change rate and ingestion lag, then write both.

    The adapter port is the same one D5 and D7 use, so the stage works on anything that implements
    :class:`~mdqm.adapters.base.SourceAdapter`; ``mutability_stats`` missing is a capability verdict
    (every table becomes ``unknown``), never an exception.
    """

    name: str = STAGE_NAME
    heavy: bool = True

    def __init__(
        self,
        adapters: Mapping[str, SourceAdapter],
        *,
        tables: Sequence[str] | None = None,
        lag_sample: int = DEFAULT_LAG_SAMPLE,
        old_row_sample: int = DEFAULT_OLD_ROW_SAMPLE,
        timeout_s: float = 30.0,
        max_bytes: int = 8 * 1024 * 1024,
        layer_loader: Callable[[Any], SemanticLayer] | None = None,
        store_baselines: bool = True,
    ) -> None:
        if lag_sample < 1:
            raise ValueError(f"lag_sample must be at least 1, got {lag_sample}")
        if old_row_sample < 1:
            raise ValueError(f"old_row_sample must be at least 1, got {old_row_sample}")
        self._adapters = dict(adapters)
        self._tables = tuple(tables) if tables is not None else None
        self._lag_sample = lag_sample
        self._old_row_sample = old_row_sample
        self._timeout_s = timeout_s
        self._max_bytes = max_bytes
        self._layer_loader = layer_loader or load_layer
        self._store_baselines = store_baselines

    @property
    def sources(self) -> tuple[str, ...]:
        """The sources this instance measures, sorted (what the runner gates and filters on)."""
        return tuple(sorted(self._adapters))

    def inputs_hash(self, ctx: PipelineContext) -> str:
        """sha256 over the declared inputs: sources, tables, sample sizes and the day.

        The **date** is deliberate, exactly as in D5 and D7: counters and lag move every day, so a
        nightly pass must re-measure, while two passes on one day measure the same day.
        """
        return hash_inputs(
            self.name,
            STAGE_VERSION,
            sorted(self._adapters),
            self._tables,
            ctx.source,
            ctx.exec_ts.date().isoformat(),
            self._lag_sample,
            self._old_row_sample,
            self._timeout_s,
        )

    def run(self, ctx: PipelineContext) -> StageReport:
        """Measure every selected source and buffer one ``generated/mutability_<source>.yaml``."""
        if not ctx.force and not quiet_allows(ctx.quiet_windows, ctx.exec_ts):
            raise QuietWindowClosed(
                f"stage {self.name} is heavy and its quiet window is closed; rerun inside the "
                "window or pass --force"
            )
        selected = (ctx.source,) if ctx.source is not None else self.sources
        missing = [source for source in selected if source not in self._adapters]
        if missing:
            raise ValueError(f"no adapter for source(s) {', '.join(missing)}")
        counts: dict[str, int] = {
            "sources": 0,
            "tables": 0,
            "tables_classified": 0,
            "tables_unknown": 0,
            "lags_measured": 0,
            "lags_violated": 0,
            "old_row_checks": 0,
        }
        skipped: list[str] = []
        fixes: list[str] = []
        for source in selected:
            payload = self._measure_source(source, ctx, counts=counts, skipped=skipped, fixes=fixes)
            ctx.write_generated(f"mutability_{_slug(source)}", payload)
            counts["sources"] += 1
        return StageReport(
            stage=self.name,
            status=StageStatus.RAN,
            counts=counts,
            skipped=tuple(skipped),
            applied_fixes=tuple(fixes),
        )

    # -- measurement -----------------------------------------------------------------------

    def _measure_source(
        self,
        source: str,
        ctx: PipelineContext,
        *,
        counts: dict[str, int],
        skipped: list[str],
        fixes: list[str],
    ) -> dict[str, Any]:
        """One source: two counter readings around real work, then a class and a lag per table."""
        adapter = self._adapters[source]
        catalog = adapter.extract_catalog()
        partitions = self._partitions(source, adapter, catalog, skipped=skipped)
        layer = self._layer(ctx, skipped=skipped)
        window_from = ctx.clock.now()
        before = self._counters(source, adapter, skipped=skipped)
        # The work between the two readings is the measurement itself: bounded lag samples and
        # old-row checksums, one per table. That is what makes the delta a real window and not two
        # reads of the same instant.
        measured: dict[str, tuple[LagMeasurement | None, OldRowCheck | None]] = {}
        for table in self._selected(catalog):
            counts["tables"] += 1
            measured[table.name] = self._measure_table(
                source,
                adapter,
                table,
                layer=layer,
                partitions=partitions.get(table.name),
                exec_ts=ctx.exec_ts,
                counts=counts,
                skipped=skipped,
                fixes=fixes,
            )
        window_to = ctx.clock.now()
        after = self._counters(source, adapter, skipped=skipped)
        previous = self._read_baseline(ctx, source)
        lags: list[str] = []
        tables: dict[str, Any] = {}
        checksums: dict[str, str] = {}
        maxima: dict[str, float] = {}
        unverified: list[str] = []
        for table in self._selected(catalog):
            lag, old_rows = measured.get(table.name, (None, None))
            evidence = self._evidence(
                source,
                table,
                before=before,
                after=after,
                partitions=partitions.get(table.name),
                lag=lag,
                old_rows=self._with_previous(old_rows, table.name, previous.get("checksums", {})),
                window_from=window_from,
                window_to=window_to,
            )
            classified = classify(evidence)
            counts["tables_classified"] += 1
            if classified.class_ == "unknown":
                counts["tables_unknown"] += 1
            entry = _table_payload(classified, layer=layer)
            if lag is not None:
                key = f"{table.name}.{lag.ingestion_column}"
                declared = _declared_max_lag(layer, source, table.name, lag.ingestion_column)
                if declared is not None and lag.maximum_s > declared:
                    # The declared bound is short: keep it, say so, and stop deriving on
                    # it. Widening it here would quietly move a promise already relied on.
                    counts["lags_violated"] += 1
                    unverified.append(key)
                    fixes.append(
                        f"{source}.{key}: lag_violated: measured max {lag.maximum_s:.0f}s exceeds "
                        f"the declared {declared:.0f}s; the declared bound is kept and the column "
                        "is marked lag_unverified until a human reviews it"
                    )
                else:
                    entry["time_columns"] = _lag_payload(lag)
                    lags.append(evidence.lag.evidence if evidence.lag is not None else "")
                    counts["lags_measured"] += 1
                checksums[f"{table.name}.lag"] = f"{lag.rows_measured}:{lag.maximum_s:.3f}"
                maxima[key] = lag.maximum_s
            if old_rows is not None:
                counts["old_row_checks"] += 1
                checksums[f"{table.name}.old_rows"] = old_rows.checksum
            tables[f"{source}.{table.name}"] = entry
        if self._store_baselines:
            self._write_baseline(
                ctx,
                source,
                checksums=checksums,
                maxima=maxima,
                window_from=window_from,
                window_to=window_to,
            )
            self._write_unverified(ctx, source, unverified)
        return {"tables": tables}

    def _measure_table(
        self,
        source: str,
        adapter: SourceAdapter,
        table: Table,
        *,
        layer: SemanticLayer | None,
        partitions: Partitioning | None,
        exec_ts: datetime,
        counts: dict[str, int],
        skipped: list[str],
        fixes: list[str],
    ) -> tuple[LagMeasurement | None, OldRowCheck | None]:
        """The lag distribution and the old-row checksum for one table, both bounded."""
        pair = _time_pair(table, layer=layer, source=source, partitions=partitions)
        if pair is None:
            return (None, None)
        event_column, ingestion_column = pair
        lag = self._lag(
            source,
            adapter,
            table,
            event_column=event_column,
            ingestion_column=ingestion_column,
            exec_ts=exec_ts,
            skipped=skipped,
        )
        old_rows = self._old_rows(
            source,
            adapter,
            table,
            event_column=event_column,
            exec_ts=exec_ts,
            skipped=skipped,
        )
        if lag is None:
            fixes.append(
                f"{source}.{table.name}: no lag could be measured between {event_column} and "
                f"{ingestion_column}, so no derived time bound may rely on one"
            )
        return (lag, old_rows)

    def _lag(
        self,
        source: str,
        adapter: SourceAdapter,
        table: Table,
        *,
        event_column: str,
        ingestion_column: str,
        exec_ts: datetime,
        skipped: list[str],
    ) -> LagMeasurement | None:
        """Measure ``ingestion - event`` over the most recent bounded sample of rows."""
        dialect = adapter.dialect()
        sql = (
            exp.select(
                exp.column(ingestion_column).as_("ingestion"),
                exp.column(event_column).as_("event"),
            )
            .from_(exp.to_table(table.qualified_name))
            .where(exp.column(ingestion_column).is_(exp.not_(exp.Null())))
            .where(exp.column(event_column).is_(exp.not_(exp.Null())))
            .order_by(exp.column(ingestion_column).desc())
            .limit(self._lag_sample)
            .sql(dialect=dialect)
        )
        rows = self._read(
            source,
            adapter,
            table,
            sql=sql,
            dialect=dialect,
            exec_ts=exec_ts,
            skipped=skipped,
        )
        seconds: list[float] = []
        events: list[datetime] = []
        landed_at: list[datetime] = []
        for row in rows:
            start = _as_time(row.get("event"))
            landed = _as_time(row.get("ingestion"))
            if start is None or landed is None:
                continue
            seconds.append((landed - start).total_seconds())
            events.append(start)
            landed_at.append(landed)
        if not seconds:
            return None
        minimum, p50, p99, maximum = percentiles(seconds)
        return LagMeasurement(
            event_column=event_column,
            ingestion_column=ingestion_column,
            rows_measured=len(seconds),
            rows_skipped=len(rows) - len(seconds),
            minimum_s=minimum,
            p50_s=p50,
            p99_s=p99,
            maximum_s=maximum,
            # The window the distribution was measured over: the span of the sampled rows.
            window_from=min(events),
            window_to=max(landed_at),
        )

    def _old_rows(
        self,
        source: str,
        adapter: SourceAdapter,
        table: Table,
        *,
        event_column: str,
        exec_ts: datetime,
        skipped: list[str],
    ) -> OldRowCheck | None:
        """Checksum a table's oldest sampled rows, so a rewrite of an old row is visible."""
        dialect = adapter.dialect()
        sql = (
            exp.select(exp.Star())
            .from_(exp.to_table(table.qualified_name))
            .order_by(exp.column(event_column).asc())
            .limit(self._old_row_sample)
            .sql(dialect=dialect)
        )
        rows = self._read(
            source,
            adapter,
            table,
            sql=sql,
            dialect=dialect,
            exec_ts=exec_ts,
            skipped=skipped,
        )
        if not rows:
            return None
        return OldRowCheck(
            columns=tuple(sorted(str(key) for key in rows[0])),
            rows=len(rows),
            checksum=_checksum(rows),
        )

    def _read(
        self,
        source: str,
        adapter: SourceAdapter,
        table: Table,
        *,
        sql: str,
        dialect: str,
        exec_ts: datetime,
        skipped: list[str],
    ) -> list[dict[str, Any]]:
        """Run one bounded, guarded SELECT and materialise it; a refusal or miss yields no rows.

        The statement is guarded here for the same reason D5 guards its samples: this is a pipeline
        path that hands a fragment straight to an adapter with no governor in between.
        """
        checked = guard(sql, dialect)
        if not checked.ok:
            code = checked.error.code if checked.error is not None else "WRITE_BLOCKED"
            skipped.append(
                f"{source}.{table.name}: the measurement statement was refused by the read-only "
                f"guard ({code}); this table has no measured lag or checksum"
            )
            return []
        fragment = Fragment(
            source=source, dialect=dialect, sql=sql, tables=(table.name,), exec_ts=exec_ts
        )
        limits = Limits(
            timeout_s=self._timeout_s,
            max_rows=self._lag_sample,
            max_bytes=self._max_bytes,
            max_partitions=1,
        )
        # The adapter protocol method, not a SQL sink: the statement was guarded just above.
        stream = (  # pi-lens-ignore: python-sql-injection -- adapter protocol call
            adapter.execute(fragment, limits)
        )
        try:
            return [row for batch in stream for row in _rows(batch)]
        except (NotSupported, TimeoutError) as exc:
            skipped.append(
                f"{source}.{table.name}: the measurement fetch stopped early ({exc}); no lag or "
                "checksum was recorded for it"
            )
            return []
        finally:
            stream.close()

    # -- counters and evidence -------------------------------------------------------------

    def _counters(
        self, source: str, adapter: SourceAdapter, *, skipped: list[str]
    ) -> dict[str, CounterSet]:
        """One reading of every table's change counters, keyed by bare relation name.

        A source without the capability gives an empty reading: every table becomes ``unknown``,
        which is a verdict, not a failure.
        """
        if not supports_method(adapter, "mutability_stats"):
            skipped.append(
                f"{source}: no mutability counters capability, so its tables stay unknown "
                "(never guessed from row counts)"
            )
            return {}
        try:
            stats = adapter.mutability_stats()
        except NotSupported as exc:
            skipped.append(f"{source}: change counters are unreadable ({exc}); tables stay unknown")
            return {}
        read: dict[str, CounterSet] = {}
        for item in stats:
            read[_bare(item.table)] = CounterSet(
                inserted=item.rows_inserted,
                updated=item.rows_updated,
                deleted=item.rows_deleted,
                hot_updated=item.hot_updates,
            )
        return read

    def _partitions(
        self,
        source: str,
        adapter: SourceAdapter,
        catalog: CatalogSnapshot,
        *,
        skipped: list[str],
    ) -> dict[str, Partitioning]:
        """Partition metadata per partitioned table, or ``{}`` when it cannot be attributed.

        ``extract_partitions()`` reports partition metadata without the table it belongs to, so a
        set is attributed to a partitioned table only when **one** table's name is a prefix of
        every partition name in it. Two candidates, or none, means the attribution is a guess and
        the table gets no partition metadata -- and therefore no roll-up and no event column.
        """
        if not supports_method(adapter, "extract_partitions"):
            return {}
        # The optional protocol member, reached through a plain lookup: absent on many adapters,
        # the same shape D5 uses for `sample_sql`.
        build = getattr(adapter, "extract_partitions")  # noqa: B009 - optional protocol member
        try:
            reported = build()
        except NotSupported as exc:
            skipped.append(
                f"{source}: partition metadata is unavailable ({exc}); partitioned tables stay "
                "unknown because their own counters exclude their partitions"
            )
            return {}
        parents = [table for table in catalog.tables if table.kind is TableKind.PARTITIONED_TABLE]
        attributed: dict[str, Partitioning] = {}
        for item in reported:
            if not isinstance(item, Partitioning) or not item.bounds:
                continue
            candidates = [
                parent
                for parent in parents
                if all(bound.name.startswith(f"{parent.name}_") for bound in item.bounds)
            ]
            if len(candidates) == 1:
                attributed[candidates[0].name] = item
            else:
                skipped.append(
                    f"{source}: partition metadata for {len(item.bounds)} partition(s) of "
                    f"{item.bounds[0].name} matches {len(candidates)} tables, so it was not "
                    "attributed to any"
                )
        return attributed

    def _evidence(
        self,
        source: str,
        table: Table,
        *,
        before: dict[str, CounterSet],
        after: dict[str, CounterSet],
        partitions: Partitioning | None,
        lag: LagMeasurement | None,
        old_rows: OldRowCheck | None,
        window_from: datetime,
        window_to: datetime,
    ) -> TableEvidence:
        """Assemble a table's evidence, rolling a partitioned parent's counters up."""
        notes: list[str] = []
        lifetime, rolled = _roll_up(table, after, partitions)
        earlier, _ = _roll_up(table, before, partitions)
        delta = None if lifetime is None or earlier is None else lifetime.minus(earlier)
        if table.kind is TableKind.PARTITIONED_TABLE and partitions is None:
            notes.append(
                "this is a partitioned table and its partitions could not be listed, so its "
                "counters (which the server keeps per partition) are not a measurement"
            )
            lifetime, delta = (None, None)
        return TableEvidence(
            source=source,
            table=table.name,
            lifetime=lifetime,
            delta=delta,
            window_from=window_from,
            window_to=window_to,
            partitions_rolled_up=rolled,
            row_estimate=table.row_estimate,
            lag=lag,
            old_rows=old_rows,
            notes=tuple(notes),
        )

    def _selected(self, catalog: CatalogSnapshot) -> list[Table]:
        """The tables this stage measures, sorted: whole catalog, or the ``tables`` narrowing."""
        wanted = None if self._tables is None else set(self._tables)
        return [
            table
            for table in sorted(catalog.tables, key=lambda item: item.name)
            if table.kind in (TableKind.TABLE, TableKind.PARTITIONED_TABLE)
            and (wanted is None or table.name in wanted)
        ]

    def _layer(self, ctx: PipelineContext, *, skipped: list[str]) -> SemanticLayer | None:
        """The semantic layer, or ``None`` when it does not load: lag pairs then come from names."""
        try:
            return self._layer_loader(ctx.semantic_root)
        except Exception as exc:  # a broken layer is D10's error to report, not a crash here
            skipped.append(
                f"the semantic layer did not load ({type(exc).__name__}), so time columns were "
                "derived from the catalog only"
            )
            return None

    # -- baselines and verification --------------------------------------------------------

    def _with_previous(
        self, old_rows: OldRowCheck | None, table: str, previous: Mapping[str, Any]
    ) -> OldRowCheck | None:
        """Attach the previous pass's checksum, so "did an old row change?" is answerable."""
        if old_rows is None:
            return None
        found = previous.get(f"{table}.old_rows")
        if found is None:
            return old_rows
        return old_rows.model_copy(update={"previous_checksum": str(found)})

    def _read_baseline(self, ctx: PipelineContext, source: str) -> dict[str, Any]:
        """The previous pass's checksums and lag maxima for this source, or an empty mapping."""
        if not self._store_baselines:
            return {}
        raw = _meta(ctx, f"{BASELINE_PREFIX}.{source}")
        if raw is None:
            return {}
        try:
            document = json.loads(raw)
        except ValueError:
            logger.warning("a damaged mutability baseline was ignored", extra={"source": source})
            return {}
        return document if isinstance(document, dict) else {}

    def _write_baseline(
        self,
        ctx: PipelineContext,
        source: str,
        *,
        checksums: Mapping[str, str],
        maxima: Mapping[str, float],
        window_from: datetime,
        window_to: datetime,
    ) -> None:
        if not self._store_baselines:
            return
        payload = {
            "checksums": dict(sorted(checksums.items())),
            "maxima": dict(sorted(maxima.items())),
            "window_from": window_from.isoformat(),
            "window_to": window_to.isoformat(),
        }
        _write_meta(
            ctx,
            f"{BASELINE_PREFIX}.{source}",
            json.dumps(payload, sort_keys=True),
        )

    def _write_unverified(
        self, ctx: PipelineContext, source: str, unverified: Sequence[str]
    ) -> None:
        """Replace this source's ``lag_unverified`` set: a cleared entry had a measurement."""
        if not self._store_baselines:
            return
        _write_meta(
            ctx,
            f"{LAG_UNVERIFIED_PREFIX}.{source}",
            json.dumps(sorted(set(unverified))),
        )


# --- time-column discovery ----------------------------------------------------------------


def _time_pair(
    table: Table,
    *,
    layer: SemanticLayer | None,
    source: str,
    partitions: Partitioning | None,
) -> tuple[str, str] | None:
    """``(event_column, ingestion_column)`` for a table, from the layer first, the catalog second.

    The semantic layer's declared ``role`` is the authority: a human approved which column is the
    event time. Without it the pair is read from the catalog -- the partition key when it is a
    time column, plus a column whose name is a recognised ingestion-time name -- and the evidence
    says so, because a name is a weaker fact than an approval.
    """
    if layer is not None:
        semantics = layer.tables.get(f"{source}.{table.name}")
        if semantics is not None:
            declared = _declared_pair(semantics)
            if declared is not None:
                return declared
    event = _event_column(table, partitions)
    if event is None:
        return None
    for name in INGESTION_NAME_HINTS:
        column = table.column(name)
        if column is not None and column is not table.column(event):
            return (event, name)
    return None


def _declared_pair(semantics: Any) -> tuple[str, str] | None:
    """The pair a table's declared ``time_columns`` gives, when it declares both roles."""
    from mdqm.semantic.models import TimeRole

    event = next(
        (
            name
            for name, column in semantics.time_columns.items()
            if column.role is TimeRole.EVENT_TIME
        ),
        None,
    )
    ingestion = next(
        (
            name
            for name, column in semantics.time_columns.items()
            if column.role is TimeRole.INGESTION_TIME
        ),
        None,
    )
    if event is None or ingestion is None:
        return None
    return (event, ingestion)


def _event_column(table: Table, partitions: Partitioning | None) -> str | None:
    """The event-time candidate: the partition key when it is one column of this table."""
    if partitions is not None and len(partitions.key) == 1:
        name = partitions.key[0]
        if table.column(name) is not None:
            return name
    return None


def _declared_max_lag(
    layer: SemanticLayer | None, source: str, table: str, ingestion_column: str
) -> float | None:
    """The ``lag.max`` the layer declares for this column, in seconds, or ``None``."""
    if layer is None:
        return None
    semantics = layer.tables.get(f"{source}.{table}")
    if semantics is None:
        return None
    column = semantics.time_columns.get(ingestion_column)
    if column is None or column.lag is None:
        return None
    return column.lag.maximum.total_seconds()


def _lag_payload(lag: LagMeasurement) -> dict[str, Any]:
    """The ``time_columns`` block D6 writes: the ingestion column, its derivation and its lag.

    The event column is written with it because a derivation that names a column the same table
    does not list is invalid (`TableSemantics._time_columns_consistent`), and a partial block that
    only merges cleanly when another section happens to supply the rest is a trap.
    """
    return {
        lag.event_column: {"role": "event_time"},
        lag.ingestion_column: {
            "role": "ingestion_time",
            "derives_bound_on": lag.event_column,
            "lag": {
                "min": format_duration(timedelta(seconds=lag.minimum_s)),
                "max": format_duration(
                    timedelta(
                        seconds=math.ceil(lag.maximum_s * LAG_MARGIN / LAG_ROUND_SECONDS)
                        * LAG_ROUND_SECONDS
                    )
                ),
                "verify": "nightly",
            },
        },
    }


def _table_payload(classified: ClassifiedTable, *, layer: SemanticLayer | None) -> dict[str, Any]:
    """One ``tables.<source>.<table>`` block: the class, its evidence and (later) its lag."""
    evidence = classified.evidence
    lines = (*evidence.describe(), *classified.reasons)
    payload: dict[str, Any] = {
        "mutability": {
            "class": classified.class_,
            # A measurement is not an approval (DESIGN section 10): a human promotes a class to
            # `approved`, and until then the table behaves as mutable, so nothing is cached on it.
            "status": "proposed",
        },
        "provenance": {"detected_by": f"pipeline/{STAGE_NAME}", "evidence": list(lines)},
    }
    if classified.class_ == "unknown":
        payload["cache"] = {"mode": "none"}
    return payload


# --- roll-up, store access and naming -----------------------------------------------------


def _roll_up(
    table: Table,
    counters: Mapping[str, CounterSet],
    partitions: Partitioning | None,
) -> tuple[CounterSet | None, int]:
    """A table's counters plus, for a partitioned parent, every partition's.

    A partitioned parent's own counters are zero in ``pg_stat_user_tables`` however much its
    partitions hold, so reading only the parent would classify a busy table as untouched. The
    children are the partition names the probe reported, never a pattern of our own.
    """
    found = counters.get(_bare(table.name))
    if table.kind is not TableKind.PARTITIONED_TABLE or partitions is None:
        return (found, 0)
    total = found or CounterSet(inserted=0, updated=0, deleted=0, hot_updated=0)
    rolled = 0
    for bound in partitions.bounds:
        child = counters.get(_bare(bound.name))
        if child is None:
            continue
        rolled += 1
        total = CounterSet(
            inserted=(total.inserted or 0) + (child.inserted or 0),
            updated=(total.updated or 0) + (child.updated or 0),
            deleted=(total.deleted or 0) + (child.deleted or 0),
            hot_updated=(total.hot_updated or 0) + (child.hot_updated or 0),
        )
    return ((total if rolled else None), rolled)


def _bare(name: str) -> str:
    """The unqualified relation name: the server reports ``schema.table``, the catalog ``table``."""
    return name.rsplit(".", 1)[-1]


def read_lag_unverified(store: Any, source: str) -> frozenset[str]:
    """The ingestion columns whose last nightly verification failed, for the planner.

    C07 disables the derived bound for these: a violated lag means the written ``max`` is wrong for
    at least one observed row, and deriving on it would drop that row silently.
    """
    row = (
        store.connect()
        .execute(
            "SELECT value FROM mdqm_meta WHERE key = ?", (f"{LAG_UNVERIFIED_PREFIX}.{source}",)
        )
        .fetchone()
    )
    if row is None:
        return frozenset()
    try:
        values = json.loads(str(row["value"]))
    except ValueError:
        return frozenset()
    return frozenset(str(value) for value in values) if isinstance(values, list) else frozenset()


def _meta(ctx: PipelineContext, key: str) -> str | None:
    row = (
        ctx.store.connect().execute("SELECT value FROM mdqm_meta WHERE key = ?", (key,)).fetchone()
    )
    return None if row is None else str(row["value"])


def _write_meta(ctx: PipelineContext, key: str, value: str) -> None:
    """Upsert one ``mdqm_meta`` row through the store's own writer, so there is one write path."""
    with ctx.store.writing() as conn:
        conn.execute(
            "INSERT INTO mdqm_meta (key, value, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
            " updated_at = excluded.updated_at",
            (key, value, ctx.exec_ts.isoformat()),
        )
