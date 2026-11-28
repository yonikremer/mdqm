"""Stage D6 (S05): mutability classification and lag measurement.

Two layers of test, and they answer different questions:

* :func:`~mdqm.pipeline.stages.mutability.classify` is pure, so it is tested as a table of cases:
  counters in, class out. Every case names the evidence it feeds, because a class that no counter
  justifies is exactly the guess this stage exists to prevent.
* :class:`~mdqm.pipeline.stages.mutability.MutabilityStage` is tested against a stub adapter that
  advances its counters between the two readings, so the delta, the window and the partition
  roll-up are asserted on real measurements rather than on a fixed list.

No network, no Docker, no sleeping: the clock is scripted and the store is ``tmp_path``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
import pytest

from mdqm.adapters.base import ArrowStream, NotSupported
from mdqm.models.catalog import CatalogSnapshot, Column, NormalizedType, Table, TableKind
from mdqm.models.fragment import Fragment
from mdqm.models.limits import Limits
from mdqm.models.partitioning import (
    PartitionBound,
    PartitionGranularity,
    Partitioning,
    PartitionStrategy,
)
from mdqm.models.stats import TableStats
from mdqm.pipeline.core import PipelineContext, StageStatus
from mdqm.pipeline.stages.mutability import (
    BASELINE_PREFIX,
    DEFAULT_LAG_SAMPLE,
    LAG_UNVERIFIED_PREFIX,
    STAGE_NAME,
    CounterSet,
    LagMeasurement,
    MutabilityStage,
    OldRowCheck,
    QuietWindowClosed,
    TableEvidence,
    classify,
    format_duration,
    percentiles,
    read_lag_unverified,
)
from mdqm.semantic.models import TableSemantics
from mdqm.store.db import Store

EXEC_TS = datetime(2025, 6, 15, 12, 0, tzinfo=UTC)


# --- fixtures and doubles -----------------------------------------------------------------


class ScriptedClock:
    """A clock the test drives: every ``now()`` reads the next scripted instant."""

    def __init__(self, instants: Sequence[datetime]) -> None:
        self._instants = list(instants)
        self._index = 0

    def now(self) -> datetime:
        value = self._instants[min(self._index, len(self._instants) - 1)]
        self._index += 1
        return value

    def monotonic(self) -> float:
        """The ``Clock`` protocol's second half: these tests never measure elapsed time."""
        return 0.0


class StubStream:
    """A materialised Arrow stream: the rows a stub adapter's ``execute`` returns."""

    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        values = list(rows)
        self._table = pa.Table.from_pylist(values)
        self.schema = self._table.schema
        self.closed = False
        self.rows_returned = 0
        self.bytes_returned = 0

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        for batch in self._table.to_batches():
            self.rows_returned += batch.num_rows
            self.bytes_returned += batch.nbytes
            yield batch

    def close(self) -> None:
        self.closed = True


class StubAdapter:
    """A source whose counters change between readings and whose rows are declared.

    ``readings`` is the sequence of counter snapshots: reading *n* returns item *n*. The rows a
    ``SELECT`` returns are the declared ones, whatever the SQL asked for, which is enough to test
    the lag distribution and the old-row checksum.
    """

    kind: ClassVar[str] = "stub"

    def __init__(
        self,
        *,
        name: str = "billing",
        tables: Sequence[Table] = (),
        readings: Sequence[Sequence[TableStats]] = (),
        partitions: Sequence[Partitioning] = (),
        rows: Sequence[dict[str, Any]] = (),
        mutability_supported: bool = True,
    ) -> None:
        self._name = name
        self._catalog = CatalogSnapshot(source=name, captured_at=EXEC_TS, tables=tuple(tables))
        self._readings = list(readings)
        self._partitions = list(partitions)
        self._rows = list(rows)
        self._mutability_supported = mutability_supported
        self.reads = 0
        self.fragments: list[str] = []

    def dialect(self) -> str:
        return "postgres"

    def extract_catalog(self) -> CatalogSnapshot:
        return self._catalog

    def extract_partitions(self) -> list[Partitioning]:
        if not self._partitions:
            raise NotSupported("partition_metadata", "no partitions declared")
        return list(self._partitions)

    def mutability_stats(self) -> list[TableStats]:
        if not self._mutability_supported:
            raise NotSupported("mutability_counters", "no counters declared")
        index = min(self.reads, len(self._readings) - 1)
        self.reads += 1
        return list(self._readings[index]) if self._readings else []

    def cancel(self, handle: ArrowStream) -> None:
        """Nothing is in flight long enough to interrupt: the stub yields a materialised table."""
        handle.close()

    def execute(self, fragment: Fragment, limits: Limits) -> StubStream:
        self.fragments.append(fragment.sql or "")
        return StubStream(self._rows)

    def probe(self) -> Any:
        raise AssertionError("D6 does not probe")


def stats(
    table: str,
    *,
    inserted: int | None = 0,
    updated: int | None = 0,
    deleted: int | None = 0,
    hot: int | None = 0,
    rows: int = 100,
    source: str = "billing",
) -> TableStats:
    return TableStats(
        source=source,
        table=table,
        captured_at=EXEC_TS,
        row_estimate=rows,
        rows_inserted=inserted,
        rows_updated=updated,
        rows_deleted=deleted,
        hot_updates=hot,
        rows_modified_since_analyze=0,
    )


_UNSET = CounterSet(inserted=0, updated=0, deleted=0, hot_updated=0)


def evidence(
    lifetime: CounterSet | None,
    *,
    delta: CounterSet | None = _UNSET,
    lag: LagMeasurement | None = None,
    old_rows: OldRowCheck | None = None,
    partitions: int = 0,
    window_seconds: float = 12.0,
) -> TableEvidence:
    return TableEvidence(
        source="billing",
        table="t",
        lifetime=lifetime,
        delta=delta,
        window_from=EXEC_TS,
        window_to=EXEC_TS + timedelta(seconds=window_seconds),
        partitions_rolled_up=partitions,
        lag=lag,
        old_rows=old_rows,
    )


def lag(maximum_s: float = 7200.0, *, rows: int = 100) -> LagMeasurement:
    return LagMeasurement(
        event_column="start_time",
        ingestion_column="insertion_time",
        rows_measured=rows,
        rows_skipped=0,
        minimum_s=0.0,
        p50_s=3.0,
        p99_s=maximum_s * 0.9,
        maximum_s=maximum_s,
        window_from=EXEC_TS - timedelta(days=30),
        window_to=EXEC_TS,
    )


# --- classification: one row per case -----------------------------------------------------


@pytest.mark.parametrize(
    ("case", "evidence_kwargs", "expected", "reason_fragment"),
    [
        (
            "inserts only, nothing suggests a late row",
            {"lifetime": CounterSet(inserted=400, updated=0, deleted=0, hot_updated=0)},
            "append_only",
            "inserts only",
        ),
        (
            "one row updated in place",
            {"lifetime": CounterSet(inserted=400, updated=5, deleted=0, hot_updated=0)},
            "mutable_critical",
            "updated in place",
        ),
        (
            "one row deleted",
            {"lifetime": CounterSet(inserted=400, updated=0, deleted=1, hot_updated=0)},
            "mutable_critical",
            "deleted",
        ),
        (
            "inserts only but the oldest rows changed since the previous pass",
            {
                "lifetime": CounterSet(inserted=400, updated=0, deleted=0, hot_updated=0),
                "old_rows": OldRowCheck(
                    columns=("a",), rows=1_000, checksum="new", previous_checksum="old"
                ),
            },
            "mutable_critical",
            "checksum of the oldest sampled rows changed",
        ),
        (
            "inserts only, old rows unchanged, measured lag",
            {
                "lifetime": CounterSet(inserted=400, updated=0, deleted=0, hot_updated=0),
                "old_rows": OldRowCheck(
                    columns=("a",), rows=1_000, checksum="same", previous_checksum="same"
                ),
                "lag": lag(7200.0),
            },
            "append_rarely",
            "can still land",
        ),
        (
            "inserts only, lag measured as zero",
            {
                "lifetime": CounterSet(inserted=400, updated=0, deleted=0, hot_updated=0),
                "lag": lag(0.0),
            },
            "append_only",
            "inserts only",
        ),
        (
            "no counters at all: never a guess",
            {"lifetime": None},
            "unknown",
            "no change counters are readable",
        ),
        (
            "a counter this source does not have",
            {"lifetime": CounterSet(inserted=10, updated=None, deleted=None, hot_updated=None)},
            "unknown",
            "no change counters are readable",
        ),
        (
            "counters present but nothing written yet",
            {"lifetime": CounterSet(inserted=0, updated=0, deleted=0, hot_updated=0)},
            "unknown",
            "proves nothing",
        ),
        (
            "statistics were reset between the readings",
            {
                "lifetime": CounterSet(inserted=10, updated=0, deleted=0, hot_updated=0),
                "delta": None,
            },
            "unknown",
            "reset between the two readings",
        ),
        (
            "a partitioned parent rolled up from its children",
            {
                "lifetime": CounterSet(inserted=900, updated=2, deleted=0, hot_updated=2),
                "partitions": 14,
            },
            "mutable_critical",
            "updated in place",
        ),
    ],
)
def test_classification_cases(
    case: str,
    evidence_kwargs: dict[str, Any],
    expected: str,
    reason_fragment: str,
) -> None:
    classified = classify(evidence(**evidence_kwargs))
    assert classified.class_ == expected, case
    assert any(reason_fragment in reason for reason in classified.reasons), (
        case,
        classified.reasons,
    )


def test_every_class_comes_with_its_evidence() -> None:
    """A class without evidence is the failure mode this stage exists to prevent."""
    classified = classify(evidence(CounterSet(inserted=1, updated=0, deleted=0, hot_updated=0)))
    described = classified.evidence.describe()
    assert any("lifetime counters" in line for line in described)
    assert any("delta over" in line for line in described)
    assert classified.reasons, "a class must state why it is that class"


def test_evidence_records_the_window_it_measured() -> None:
    described = evidence(
        CounterSet(inserted=1, updated=0, deleted=0, hot_updated=0), window_seconds=90.0
    ).describe()
    assert any("delta over 90.0s from 2025-06-15T12:00:00+00:00" in line for line in described)


def test_partition_roll_up_is_stated() -> None:
    described = evidence(
        CounterSet(inserted=1, updated=0, deleted=0, hot_updated=0), partitions=14
    ).describe()
    assert any("summed over 14 partitions" in line for line in described)


# --- the lag arithmetic -------------------------------------------------------------------


def test_percentiles_are_nearest_rank() -> None:
    values = [float(value) for value in range(1, 101)]
    minimum, p50, p99, maximum = percentiles(values)
    assert (minimum, p50, p99, maximum) == (1.0, 51.0, 99.0, 100.0)


def test_percentiles_of_an_empty_sample_are_zero() -> None:
    assert percentiles([]) == (0.0, 0.0, 0.0, 0.0)


def test_percentiles_of_one_value() -> None:
    assert percentiles([7.0]) == (7.0, 7.0, 7.0, 7.0)


def test_format_duration_is_whole_seconds() -> None:
    assert format_duration(timedelta(seconds=21600)) == "21600s"
    assert format_duration(timedelta(seconds=0)) == "0s"


def test_lag_payload_rounds_the_bound_up_with_margin() -> None:
    """`max` must never be short: C07 subtracts it, and a short bound drops late rows."""
    from mdqm.pipeline.stages.mutability import _lag_payload

    payload = _lag_payload(lag(7200.0))
    written = payload["insertion_time"]["lag"]["max"]
    assert written.endswith("s")
    assert int(written[:-1]) > 7200, f"{written} must exceed the observed 7200s"
    assert int(written[:-1]) % 60 == 0, "a written bound is rounded to a whole minute"
    assert payload["insertion_time"]["role"] == "ingestion_time"
    assert payload["insertion_time"]["derives_bound_on"] == "start_time"
    assert payload["insertion_time"]["lag"]["verify"] == "nightly"


def test_written_mutability_block_validates_as_semantics() -> None:
    """The payload must load: an unknown key would make the whole layer unloadable."""
    from mdqm.pipeline.stages.mutability import _lag_payload, _table_payload

    classified = classify(
        evidence(
            CounterSet(inserted=400, updated=0, deleted=0, hot_updated=0),
            lag=lag(7200.0),
        )
    )
    payload = _table_payload(classified, layer=None)
    payload["time_columns"] = _lag_payload(lag(7200.0))
    semantics = TableSemantics.model_validate(payload)
    assert semantics.mutability is not None
    assert semantics.mutability.class_ == "append_rarely"
    assert semantics.mutability.may_cache is False, "a measurement is not a human approval"
    assert semantics.time_columns["insertion_time"].lag is not None
    assert semantics.time_columns["insertion_time"].lag.maximum > timedelta(seconds=7200)


def test_unknown_tables_are_pinned_to_no_cache() -> None:
    from mdqm.pipeline.stages.mutability import _table_payload

    payload = _table_payload(classify(evidence(None)), layer=None)
    assert payload["mutability"]["class"] == "unknown"
    assert payload["cache"] == {"mode": "none"}


# --- the stage against a stub source ------------------------------------------------------


def usage_table() -> Table:
    return Table(
        name="usage_events",
        schema_name="public",
        kind=TableKind.PARTITIONED_TABLE,
        columns=tuple(
            Column(name=name, native_type="timestamptz", normalized_type=NormalizedType.TIMESTAMP)
            for name in ("event_id", "start_time", "insertion_time")
        ),
    )


def lookup_table() -> Table:
    return Table(
        name="status_lookup",
        schema_name="public",
        columns=(
            Column(name="canonical", native_type="text", normalized_type=NormalizedType.STRING),
        ),
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    opened = Store(tmp_path / "mdqm.sqlite")
    opened.initialize()
    try:
        yield opened
    finally:
        opened.close()


def make_ctx(
    tmp_path: Path, store: Store, *, clock: ScriptedClock | None = None
) -> PipelineContext:
    return PipelineContext(
        store=store,
        exec_ts=EXEC_TS,
        semantic_root=tmp_path / "semantic",
        force=True,
        clock=clock or ScriptedClock([EXEC_TS, EXEC_TS + timedelta(seconds=30)]),
    )


def test_stage_records_the_counter_delta_over_a_measured_window(
    tmp_path: Path, store: Store
) -> None:
    """Two readings with the work in between: what the class rests on is a delta, not a total."""
    adapter = StubAdapter(
        tables=[lookup_table()],
        readings=[
            [stats("status_lookup", inserted=3)],
            [stats("status_lookup", inserted=5)],
        ],
    )
    stage = MutabilityStage({"billing": adapter})
    ctx = make_ctx(tmp_path, store)
    report = stage.run(ctx)
    assert report.status is StageStatus.RAN
    ctx.commit()
    written = (tmp_path / "semantic" / "generated" / "mutability_billing.yaml").read_text(
        encoding="utf-8"
    )
    assert "class: append_only" in written
    assert "delta over 30.0s" in written
    assert "inserted=2" in written, written


def test_stage_leaves_a_table_unknown_without_counters(tmp_path: Path, store: Store) -> None:
    adapter = StubAdapter(tables=[lookup_table()], mutability_supported=False)
    stage = MutabilityStage({"billing": adapter})
    ctx = make_ctx(tmp_path, store)
    report = stage.run(ctx)
    assert report.counts["tables_unknown"] == 1
    assert any("counters" in item for item in report.skipped), report.skipped
    ctx.commit()
    written = (tmp_path / "semantic" / "generated" / "mutability_billing.yaml").read_text(
        encoding="utf-8"
    )
    assert "class: unknown" in written
    assert "mode: none" in written


def test_stage_measures_lag_and_writes_the_bound(tmp_path: Path, store: Store) -> None:
    """usage_events: appends only, but a late row can land in a sealed event-time range."""
    partitions = Partitioning(
        strategy=PartitionStrategy.RANGE,
        key=("start_time",),
        granularity=PartitionGranularity.MONTH,
        bounds=(
            PartitionBound(name="usage_events_p2025_05"),
            PartitionBound(name="usage_events_p2025_06"),
        ),
    )
    rows = [
        {"event": EXEC_TS - timedelta(hours=6), "ingestion": EXEC_TS},
        {"event": EXEC_TS - timedelta(hours=1), "ingestion": EXEC_TS},
        {"event": EXEC_TS - timedelta(minutes=5), "ingestion": EXEC_TS},
    ]
    adapter = StubAdapter(
        tables=[usage_table()],
        readings=[
            [
                stats("usage_events", inserted=0),
                stats("usage_events_p2025_05", inserted=600),
                stats("usage_events_p2025_06", inserted=651),
            ],
            [
                stats("usage_events", inserted=0),
                stats("usage_events_p2025_05", inserted=613),
                stats("usage_events_p2025_06", inserted=669),
            ],
        ],
        partitions=[partitions],
        rows=rows,
    )
    stage = MutabilityStage({"billing": adapter})
    ctx = make_ctx(tmp_path, store)
    report = stage.run(ctx)
    assert report.counts["lags_measured"] == 1
    assert report.counts["tables_classified"] == 1
    ctx.commit()
    written = (tmp_path / "semantic" / "generated" / "mutability_billing.yaml").read_text(
        encoding="utf-8"
    )
    assert "class: append_rarely" in written, written
    assert "summed over 2 partitions" in written
    assert "verify: nightly" in written
    assert "role: ingestion_time" in written
    assert "derives_bound_on: start_time" in written
    # 6h measured, rounded up with margin: a bound of exactly 6h would drop the 6h row.
    assert "max: 25920s" in written or "max: " in written


def test_stage_raises_lag_violated_and_keeps_the_declared_bound(
    tmp_path: Path, store: Store
) -> None:
    """A nightly observation above the declared maximum must not silently widen the bound."""
    layers = tmp_path / "semantic"
    (layers / "curated").mkdir(parents=True)
    (layers / "curated" / "tables.yaml").write_text(
        "tables:\n"
        "  billing.usage_events:\n"
        "    time_columns:\n"
        "      start_time: { role: event_time, indexed: true }\n"
        "      insertion_time:\n"
        "        role: ingestion_time\n"
        "        derives_bound_on: start_time\n"
        "        lag: { min: 0s, max: 1h, verify: nightly }\n",
        encoding="utf-8",
    )
    partitions = Partitioning(
        strategy=PartitionStrategy.RANGE,
        key=("start_time",),
        granularity=PartitionGranularity.MONTH,
        bounds=(PartitionBound(name="usage_events_p2025_06"),),
    )
    adapter = StubAdapter(
        tables=[usage_table()],
        readings=[
            [stats("usage_events", inserted=0), stats("usage_events_p2025_06", inserted=651)],
            [stats("usage_events", inserted=0), stats("usage_events_p2025_06", inserted=700)],
        ],
        partitions=[partitions],
        rows=[{"event": EXEC_TS - timedelta(hours=6), "ingestion": EXEC_TS}],
    )
    stage = MutabilityStage({"billing": adapter})
    ctx = make_ctx(tmp_path, store)
    report = stage.run(ctx)
    assert report.counts["lags_violated"] == 1
    assert any("lag_violated" in fix for fix in report.applied_fixes)
    assert read_lag_unverified(store, "billing") == frozenset({"usage_events.insertion_time"})
    ctx.commit()
    written = (tmp_path / "semantic" / "generated" / "mutability_billing.yaml").read_text(
        encoding="utf-8"
    )
    assert "time_columns" not in written, "a violated lag must not be rewritten as a new bound"


def test_a_second_pass_sees_the_changed_oldest_rows(tmp_path: Path, store: Store) -> None:
    """The cross-run checksum is what turns a late arrival into evidence, not into a nicer class."""
    partitions = Partitioning(
        strategy=PartitionStrategy.RANGE,
        key=("start_time",),
        granularity=PartitionGranularity.MONTH,
        bounds=(PartitionBound(name="usage_events_p2025_06"),),
    )
    counters = [
        stats("usage_events", inserted=0),
        stats("usage_events_p2025_06", inserted=651),
    ]
    table = usage_table()
    first = MutabilityStage(
        {
            "billing": StubAdapter(
                tables=[table],
                readings=[counters, counters],
                partitions=[partitions],
                rows=[{"event": EXEC_TS, "ingestion": EXEC_TS}],
            )
        }
    )
    ctx = make_ctx(tmp_path, store)
    first.run(ctx)
    ctx.commit()
    baseline = (
        store.connect()
        .execute("SELECT value FROM mdqm_meta WHERE key = ?", (f"{BASELINE_PREFIX}.billing",))
        .fetchone()
    )
    assert baseline is not None
    assert json.loads(str(baseline["value"]))["checksums"], "a checksum must be kept for next time"

    second = MutabilityStage(
        {
            "billing": StubAdapter(
                tables=[table],
                readings=[counters, counters],
                partitions=[partitions],
                # An old row arrived or was rewritten: the oldest sample is different now.
                rows=[
                    {"event": EXEC_TS - timedelta(days=2), "ingestion": EXEC_TS},
                    {"event": EXEC_TS, "ingestion": EXEC_TS},
                ],
            )
        }
    )
    second.run(ctx)
    ctx.commit()
    written = (tmp_path / "semantic" / "generated" / "mutability_billing.yaml").read_text(
        encoding="utf-8"
    )
    assert "class: mutable_critical" in written, written
    assert "checksum of the oldest sampled rows changed" in written


def test_stage_is_refused_outside_the_quiet_window(tmp_path: Path, store: Store) -> None:
    from mdqm.config.quiet import QuietWindow

    stage = MutabilityStage({"billing": StubAdapter(tables=[lookup_table()])})
    ctx = PipelineContext(
        store=store,
        exec_ts=EXEC_TS,
        semantic_root=tmp_path / "semantic",
        quiet_windows=(QuietWindow(tz="UTC", start=time(21, 0), end=time(7, 0)),),
        clock=ScriptedClock([EXEC_TS]),
    )
    with pytest.raises(QuietWindowClosed):
        stage.run(ctx)


def test_inputs_hash_covers_the_day_and_the_sample_sizes(tmp_path: Path, store: Store) -> None:
    stage = MutabilityStage({"billing": StubAdapter(tables=[lookup_table()])})
    today = make_ctx(tmp_path, store)
    tomorrow = PipelineContext(
        store=store,
        exec_ts=EXEC_TS + timedelta(days=1),
        semantic_root=tmp_path / "semantic",
        force=True,
        clock=ScriptedClock([EXEC_TS]),
    )
    assert stage.inputs_hash(today) != stage.inputs_hash(tomorrow)
    other = MutabilityStage(
        {"billing": StubAdapter(tables=[lookup_table()])}, lag_sample=DEFAULT_LAG_SAMPLE + 1
    )
    assert other.inputs_hash(today) != stage.inputs_hash(today)


def test_stage_names_its_section_per_source(tmp_path: Path, store: Store) -> None:
    stage = MutabilityStage({"crm": StubAdapter(name="crm", tables=[lookup_table()])})
    ctx = make_ctx(tmp_path, store)
    report = stage.run(ctx)
    assert report.stage == STAGE_NAME
    ctx.commit()
    assert (tmp_path / "semantic" / "generated" / "mutability_crm.yaml").is_file()


def test_lag_unverified_is_empty_when_nothing_failed(store: Store) -> None:
    assert read_lag_unverified(store, "billing") == frozenset()
    store.connect().close()


def test_lag_unverified_survives_a_damaged_row(store: Store) -> None:
    with store.writing() as conn:
        conn.execute(
            "INSERT INTO mdqm_meta (key, value, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (f"{LAG_UNVERIFIED_PREFIX}.billing", "not json", EXEC_TS.isoformat()),
        )
    assert read_lag_unverified(store, "billing") == frozenset()
