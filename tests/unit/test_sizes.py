"""Sizes: the quantiles the planner reasons with, the parser, and the store round trip.

Three layers, each with the failure that matters:

* :func:`mdqm.planner.sizes.quantiles` is pure arithmetic. The cases pin nearest-rank behaviour
  (``p50`` of an even sample is a real sample, not an interpolation) and that ``None`` means "not
  reported", not zero -- a source that reports rows but not bytes must not drag the byte median down.
* :func:`mdqm.adapters.postgresql_plan.parse_sizes` turns ``pg_class`` rows into tables with their
  partitions. The cases pin what the query cannot promise: a partitioned parent has no rows of its
  own, ``reltuples`` of ``-1`` (never analyzed) is unknown, and a boolean is not a size.
* :class:`mdqm.store.sizes.SizeStore` keeps one row per sample per pass, and its newest capture is
  what answers a later plan.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mdqm.adapters.postgresql_plan import parse_sizes
from mdqm.compiler import CompileContext, CompiledQuery, compile_spec
from mdqm.models import PartitionSize, TableSize
from mdqm.planner.sizes import FAT_RATIO, fat_partitions, quantiles, summarise
from mdqm.semantic.models import SemanticLayer
from mdqm.spec import parse_spec
from mdqm.store.db import Store
from mdqm.store.sizes import SizeStore, as_json, describe

NOW = datetime(2026, 11, 28, tzinfo=UTC)


# --- the quantiles ----------------------------------------------------------------------------


def test_quantiles_are_nearest_rank_so_every_reported_number_is_an_observed_size() -> None:
    """An interpolated percentile invents a partition nobody has."""
    result = quantiles([10, 20, 30, 40])

    assert (result.minimum, result.p50, result.maximum) == (10, 20, 40)
    assert (result.p1, result.p25, result.p75, result.p99) == (10, 10, 30, 40)
    assert (result.count, result.total) == (4, 100)


def test_an_unreported_measure_is_absent_not_zero() -> None:
    """Oracle reports rows and not bytes; a zero byte median would be a lie about a real table."""
    result = quantiles([1_000, None, 3_000])

    assert (result.count, result.total) == (2, 4_000)
    assert result.p50 == 1_000


def test_nothing_reported_gives_zeros_and_a_zero_count() -> None:
    result = quantiles([])

    assert result.count == 0
    assert (result.minimum, result.maximum, result.p50) == (0, 0, 0)
    assert result.spread == 0.0, "no median means no spread, not a division by zero"


def test_one_sample_is_every_quantile() -> None:
    result = quantiles([7_777])

    assert {result.p1, result.p25, result.p50, result.p75, result.p99} == {7_777}


def test_a_fat_partition_is_named_and_a_balanced_table_names_nothing() -> None:
    balanced = {f"p{month:02d}": (40_000 + month, 14_000_000) for month in range(1, 13)}
    skewed = {**balanced, "p99_fat": (900_000, 300_000_000)}

    assert fat_partitions(balanced) == ()
    assert fat_partitions(skewed) == ("p99_fat",)
    assert summarise(skewed, partitions=True).unbalanced is True
    assert summarise(balanced, partitions=True).unbalanced is False


def test_the_spread_ratio_is_the_biggest_over_the_median() -> None:
    quantiles_of = summarise({"a": (100, 1), "b": (100, 1), "c": (400, 1)}, partitions=True)

    assert quantiles_of.rows.p50 == 100
    assert quantiles_of.spread_ratio == pytest.approx(4.0)
    assert quantiles_of.unbalanced is True
    assert FAT_RATIO == 4.0, "the hint threshold is a stated number, not a magic one"


# --- the parser -------------------------------------------------------------------------------


def _rows(*rows: tuple[object, ...]) -> list[tuple[object, ...]]:
    return list(rows)


def test_a_partitioned_parent_has_no_size_of_its_own() -> None:
    """Its storage belongs to the partitions; reporting the parent's 0 bytes would look empty."""
    sizes = parse_sizes(
        _rows(
            ("public", "usage_events", "usage_events", False, True, 0, 0),
            ("public", "usage_events", "usage_events_p2025_01", True, False, 41_000, 14_000_000),
            ("public", "usage_events", "usage_events_p2025_02", True, False, 50_006, 18_000_000),
        ),
        source="billing",
        captured_at=NOW,
    )

    assert len(sizes) == 1
    table = sizes[0]
    assert (table.table, table.rows, table.bytes) == ("usage_events", None, None)
    assert [partition.name for partition in table.partitions] == [
        "usage_events_p2025_01",
        "usage_events_p2025_02",
    ]
    assert table.partition_rows == {
        "usage_events_p2025_01": 41_000,
        "usage_events_p2025_02": 50_006,
    }


def test_a_plain_table_keeps_its_own_size() -> None:
    (table,) = parse_sizes(
        _rows(("public", "customers", "customers", False, False, 2_000, 1_100_000)),
        source="crm",
        captured_at=NOW,
    )

    assert (table.rows, table.bytes, table.partitions) == (2_000, 1_100_000, ())


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (-1, None),  # reltuples before any ANALYZE
        ("41000", 41_000),  # a driver handing back text
        (41000.7, 41_000),  # reltuples is a float
        (True, None),  # a boolean is not a size
        (None, None),
    ],
    ids=["never analyzed", "text", "float", "boolean", "absent"],
)
def test_a_size_the_server_did_not_really_report_is_unknown(
    raw: object, expected: int | None
) -> None:
    (table,) = parse_sizes(
        _rows(("public", "t", "t", False, False, raw, raw)),
        source="s",
        captured_at=NOW,
    )

    assert table.rows == expected
    assert table.bytes == expected


def test_the_same_bare_name_in_two_schemas_is_two_tables() -> None:
    sizes = parse_sizes(
        _rows(
            ("public", "customers", "customers", False, False, 10, 100),
            ("archive", "customers", "customers", False, False, 20, 200),
        ),
        source="crm",
        captured_at=NOW,
    )

    assert len(sizes) == 2, "merged by name, one table would hide the other"


# --- the store --------------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    opened = Store(tmp_path / "meta.sqlite")
    opened.initialize()
    try:
        yield opened
    finally:
        opened.close()


def _size(source: str = "billing", table: str = "usage_events") -> TableSize:
    return TableSize(
        source=source,
        table=table,
        captured_at=NOW,
        rows=None,
        bytes=None,
        partitions=(
            PartitionSize(name="usage_events_p2025_01", rows=41_000, bytes=14_000_000),
            PartitionSize(name="usage_events_p2025_02", rows=50_006, bytes=18_000_000),
        ),
    )


def test_a_capture_round_trips_with_its_partitions(store: Store) -> None:
    sizes = SizeStore(store, clock=lambda: NOW)

    captured_at = sizes.record([_size()])
    read = sizes.latest("billing", "usage_events")

    assert captured_at == NOW
    assert read is not None
    assert read.captured_at == NOW
    assert read.partition_rows == {
        "usage_events_p2025_01": 41_000,
        "usage_events_p2025_02": 50_006,
    }


def test_the_newest_capture_wins_and_the_older_one_is_kept(store: Store) -> None:
    """Two passes are two facts; overwriting would throw away the growth that matters."""
    later = datetime(2026, 11, 29, tzinfo=UTC)
    sizes = SizeStore(store, clock=lambda: NOW)
    sizes.record([_size()])
    SizeStore(store, clock=lambda: later).record(
        [
            TableSize(
                source="billing",
                table="usage_events",
                captured_at=later,
                partitions=(PartitionSize(name="usage_events_p2025_02", rows=99_000),),
            )
        ]
    )

    newest = sizes.latest("billing", "usage_events")
    assert newest is not None and newest.captured_at == later
    assert [capture for capture, _count in sizes.captures()] == [later, NOW]


def test_a_source_that_was_not_measured_last_night_still_answers(store: Store) -> None:
    """The newest capture *of that table*, not of the whole pass: older is better than nothing."""
    earlier = datetime(2026, 11, 20, tzinfo=UTC)
    SizeStore(store, clock=lambda: earlier).record([_size()])
    SizeStore(store, clock=lambda: NOW).record([_size(source="crm", table="customers")])

    assert SizeStore(store).latest("billing", "usage_events") is not None


def test_an_unmeasured_table_reports_nothing_rather_than_zero(store: Store) -> None:
    sizes = SizeStore(store)

    assert sizes.latest("billing", "usage_events") is None
    assert sizes.latest_all() == ()
    assert sizes.profile("billing", "usage_events") is None
    assert sizes.latest_capture() is None


def test_the_profile_of_a_partitioned_table_is_its_partitions(store: Store) -> None:
    sizes = SizeStore(store, clock=lambda: NOW)
    sizes.record([_size()])

    profile = sizes.profile("billing", "usage_events")

    assert profile is not None
    assert profile.partitions == 2
    assert profile.rows.p50 == 41_000
    assert profile.rows.maximum == 50_006


def test_the_report_is_one_line_per_table_and_says_when_a_table_is_uneven(store: Store) -> None:
    sizes = SizeStore(store, clock=lambda: NOW)
    sizes.record(
        [
            _size(),
            TableSize(
                source="billing",
                table="skewed",
                captured_at=NOW,
                partitions=(
                    PartitionSize(name="small", rows=100),
                    PartitionSize(name="big", rows=90_000),
                ),
            ),
        ]
    )

    rendered = describe(sizes.latest_all())

    assert "billing.skewed" in rendered
    assert "UNBALANCED" in rendered
    assert "usage_events" in rendered
    assert '"table": "usage_events"' in as_json(sizes.latest_all())


# --- the compiler path: a fat partition reaches the notes an agent reads -------------------

FAT_LAYER = SemanticLayer.model_validate(
    {
        "version": 1,
        "tables": {
            "billing.usage_events": {
                "key": ["event_id"],
                "partitioning": {
                    "key": ["start_time"],
                    "granularity": "month",
                    "require_key_filter": True,
                    "bounds": [
                        {
                            "name": "usage_events_p2026_10",
                            "lower": "2026-10-01T00:00:00Z",
                            "upper": "2026-11-01T00:00:00Z",
                        },
                        {
                            "name": "usage_events_p2026_11",
                            "lower": "2026-11-01T00:00:00Z",
                            "upper": "2026-12-01T00:00:00Z",
                        },
                        {
                            "name": "usage_events_p2026_12",
                            "lower": "2026-12-01T00:00:00Z",
                            "upper": "2027-01-01T00:00:00Z",
                        },
                    ],
                },
                "time_columns": {
                    "start_time": {"role": "event_time", "indexed": True, "use_by_default": True}
                },
            }
        },
    }
)

FAT_SIZES = {
    "usage_events_p2026_10": (1_000, 1_000_000),
    "usage_events_p2026_11": (1_000, 1_000_000),
    "usage_events_p2026_12": (200_000, 200_000_000),
}


def _compile_with_sizes(sizes: Mapping[str, tuple[int | None, int | None]]) -> CompiledQuery:
    """Compile a 40-day window ending mid-December, which spans the backfill and its neighbour."""
    return compile_spec(
        parse_spec(
            {
                "from": "billing.usage_events",
                "select": ["event_id"],
                "time": {"mode": "events_in_range", "last": "40d"},
            }
        ),
        CompileContext(
            layer=FAT_LAYER,
            source="billing",
            dialect="postgres",
            exec_ts=datetime(2026, 12, 15, tzinfo=UTC),
            columns={"billing.usage_events": ("event_id", "start_time", "insertion_time")},
            sizes=sizes,
        ),
    )


def test_the_note_and_the_names_reach_the_compiled_query() -> None:
    """`CompileContext.sizes` is the wiring: the planner's note must survive into the answer."""
    compiled = _compile_with_sizes(FAT_SIZES)

    assert compiled.error is None
    assert "fat_partition" in compiled.notes
    assert compiled.partition_plan is not None
    assert compiled.partition_plan.fat_partitions == ("usage_events_p2026_12",)


def test_without_a_profile_the_compiled_notes_are_exactly_what_they_were() -> None:
    """The invariant the whole feature rests on: no measurement, no change."""
    compiled = _compile_with_sizes({})

    assert compiled.error is None
    assert "fat_partition" not in compiled.notes
    assert compiled.partition_plan is not None
    assert compiled.partition_plan.fat_partitions == ()
