"""A03 acceptance: pure parsing of PostgreSQL partition metadata and EXPLAIN plans.

The canned plans in ``tests/data/plans/`` are not hand-written: each was captured from the compose
server with ``EXPLAIN (FORMAT JSON)`` and stored untouched, so the parser is tested on what
PostgreSQL 16 really emits (``Subplans Removed`` on an ``Append``, an ``InitPlan`` parameter ``$0``
in a filter, a one-partition ``Append`` that collapses into its child). Two of them are the same
window on two different columns, which is the whole point: event time prunes, ingestion time does
not.

Everything here is pure: no server, no socket (``tests/unit`` blocks them), so the verdict is
proven to be a function of the metadata and the plan, not of the machine it runs on.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pytest

from mdqm.adapters.base import NotSupported
from mdqm.adapters.postgresql_plan import (
    PgPlan,
    PruningVerdict,
    parse_partitions,
    parse_plan,
    parse_plan_text,
    partitions_in_window,
    pruning_verdict,
)
from mdqm.models import (
    PartitionBound,
    PartitionGranularity,
    Partitioning,
    PartitionStrategy,
    PlanSummary,
    PruningProof,
)
from mdqm.planner.partitions import Range, estimate_partition_scan

PLANS: Final = Path(__file__).resolve().parents[1] / "data" / "plans"

DECEMBER: Final = Range(datetime(2025, 12, 2, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))
"""The window the loader's data was anchored on: 30 days before 2026-01-01."""

DEC_PARTITION: Final = "usage_events_p2025_12"


def loaded(name: str) -> object:
    """One canned ``EXPLAIN (FORMAT JSON)`` document, decoded."""
    return json.loads((PLANS / name).read_text(encoding="utf-8"))


def parsed(name: str) -> PgPlan:
    """The parse of one canned plan, asserted to be readable."""
    plan = parse_plan(loaded(name))
    assert plan is not None, f"{name} is not a readable plan"
    return plan


def summary(name: str, total: int | None = 14) -> PlanSummary:
    """A ``PlanSummary`` as the adapter would return it for one canned plan."""
    plan = parsed(name)
    scanned = len(
        [relation for relation in plan.relations if relation.startswith("usage_events_p")]
    )
    return PlanSummary(
        partitions_scanned=scanned if total else None,
        partitions_total=total,
        uses_index=bool(plan.indexes),
        indexes_used=plan.indexes,
        est_rows=plan.plan_rows,
        est_cost=plan.total_cost,
        raw=(PLANS / name).read_text(encoding="utf-8"),
    )


def _month(year: int, month: int) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1, tzinfo=UTC)
    end = datetime(year + month // 12, month % 12 + 1, 1, tzinfo=UTC)
    return start, end


def _bound_rows(months: Sequence[tuple[int, int]]) -> list[list[str]]:
    """``PARTITIONS_SQL`` rows for a monthly RANGE table, as the server writes them."""
    rows: list[list[str]] = []
    for year, month in months:
        lower, upper = _month(year, month)
        rows.append(
            [
                "public",
                "usage_events",
                "r",
                "RANGE (start_time)",
                "public",
                f"usage_events_p{year}_{month:02d}",
                "FOR VALUES FROM ('{}') TO ('{}')".format(
                    lower.strftime("%Y-%m-%d %H:%M:%S+00"),
                    upper.strftime("%Y-%m-%d %H:%M:%S+00"),
                ),
                "586",
            ]
        )
    return rows


MONTHS: Final = [(2024, 11), (2024, 12)] + [(2025, month) for month in range(1, 13)]


@pytest.fixture(scope="module")
def monthly() -> Partitioning:
    """The loaded world's ``usage_events``: 14 monthly partitions, 2024-11 .. 2025-12."""
    tables = parse_partitions(_bound_rows(MONTHS))
    assert len(tables) == 1
    return tables[0]


# --------------------------------------------------------------------------------------
# Partition metadata: bounds are datetimes, not text
# --------------------------------------------------------------------------------------


def test_partitions_carry_parsed_bounds_key_and_granularity(monthly: Partitioning) -> None:
    """The rows become one table with 14 partitions, each a half-open UTC datetime range."""
    assert monthly.strategy is PartitionStrategy.RANGE
    assert monthly.key == ("start_time",)
    assert monthly.granularity is PartitionGranularity.MONTH
    assert len(monthly.bounds) == 14
    first = monthly.bounds[0]
    assert first.name == "usage_events_p2024_11"
    assert first.lower == datetime(2024, 11, 1, tzinfo=UTC)
    assert first.upper == datetime(2024, 12, 1, tzinfo=UTC)
    assert isinstance(first.lower, datetime) and isinstance(first.upper, datetime)
    assert first.row_estimate == 586
    last = monthly.bounds[-1]
    assert last.name == DEC_PARTITION
    assert last.upper == datetime(2026, 1, 1, tzinfo=UTC)


def test_a_default_partition_keeps_its_text_and_no_bounds() -> None:
    """``DEFAULT``, list and hash bounds are not datetimes: reporting none beats inventing one."""
    rows = [
        *_bound_rows([(2025, 12)]),
        [
            "public",
            "usage_events",
            "r",
            "RANGE (start_time)",
            "public",
            "usage_events_default",
            "DEFAULT",
            "-1",
        ],
    ]
    (table,) = parse_partitions(rows)
    default = table.bounds[-1]
    assert (default.name, default.lower, default.upper) == ("usage_events_default", None, None)
    assert default.row_estimate is None, "reltuples -1 means 'never analyzed', not 0 rows"


def test_minvalue_bounds_parse_to_none_and_granularity_stops_guessing() -> None:
    """An open-ended first partition is legal: its lower bound is unknown, not the epoch."""
    rows = _bound_rows([(2025, 12)])
    rows[0][6] = "FOR VALUES FROM (MINVALUE) TO ('2026-01-01 00:00:00+00')"
    (table,) = parse_partitions(rows)
    assert table.bounds[0].lower is None
    assert table.bounds[0].upper == datetime(2026, 1, 1, tzinfo=UTC)
    assert table.granularity is None, "with one open bound, the width cannot be inferred"


def test_a_partitioning_mechanism_we_cannot_read_is_not_supported() -> None:
    """Claiming a strategy we did not read would let the planner prune on fiction."""
    rows = [["public", "legacy", "x", "", "public", "legacy_2025", "FOR VALUES ...", "1"]]
    with pytest.raises(NotSupported) as caught:
        parse_partitions(rows)
    assert caught.value.capability == "partition_metadata"
    assert "legacy" in str(caught.value)


def test_hash_partitions_carry_their_text_and_no_granularity() -> None:
    """Hash bounds are a modulus/remainder, so there is no width to infer."""
    rows = [
        [
            "public",
            "accounts_hash",
            "h",
            "HASH (client_no)",
            "public",
            f"accounts_hash_p{index}",
            f"FOR VALUES WITH (MODULUS 4, REMAINDER {index})",
            "0",
        ]
        for index in range(4)
    ]
    (table,) = parse_partitions(rows)
    assert table.strategy is PartitionStrategy.HASH
    assert table.granularity is None
    assert [bound.values[0] for bound in table.bounds] == [
        f"FOR VALUES WITH (MODULUS 4, REMAINDER {index})" for index in range(4)
    ]


def test_partitioning_none_has_no_bounds_and_no_total() -> None:
    """The model's own shape check: an unpartitioned table reports nothing, not an empty list."""
    table = Partitioning(strategy=PartitionStrategy.NONE)
    assert table.partitions_total is None
    assert table.bounds == ()


# --------------------------------------------------------------------------------------
# The plan: six shapes, all captured from the server
# --------------------------------------------------------------------------------------


def test_the_pruned_plan_names_one_partition() -> None:
    """A constant event-time window: the Append collapses to the one partition it needs."""
    plan = parsed("usage_events_event_time_pruned.json")
    assert plan.relations == (DEC_PARTITION,)
    assert plan.indexes == ()
    assert plan.parameters == ()
    assert plan.plan_rows == 202.0
    assert plan.total_cost == pytest.approx(28.19)


def test_the_same_window_on_ingestion_time_names_every_partition() -> None:
    """``insertion_time`` is not the partition key, so no partition can be skipped."""
    plan = parsed("usage_events_ingestion_time_unpruned.json")
    assert len(plan.relations) == 14
    assert plan.relations[0] == "usage_events_p2024_11"
    assert plan.relations[-1] == DEC_PARTITION
    assert plan.subplans_removed == 0
    assert plan.pruned_at_plan_time is False


def test_the_parameterised_plan_is_pruned_at_plan_time() -> None:
    """A generic plan still prunes here: ``$1``/``$2`` are read from the index condition."""
    plan = parsed("usage_events_parameterised.json")
    assert plan.relations == (DEC_PARTITION,)
    assert plan.parameters == ("$1", "$2")
    assert plan.subplans_removed == 13
    assert plan.pruned_at_plan_time is True
    assert plan.indexes == ("usage_events_p2025_12_start_time_idx",)


def test_a_deferred_plan_names_every_partition_and_says_why() -> None:
    """An ``InitPlan`` value cannot be known at plan time: 14 subplans, filtered at run time."""
    plan = parsed("usage_events_deferred_pruning.json")
    assert len([name for name in plan.relations if name.startswith("usage_events_p")]) == 14
    assert "accounts_hist" in plan.relations, "the InitPlan's own relation is part of the plan"
    assert plan.parameters == ("$0", "$1")
    assert plan.subplans_removed == 0


def test_the_nested_loop_plan_lists_both_sides() -> None:
    """A join plan names two relations; the parser must not stop at the first scan it meets."""
    plan = parsed("usage_events_accounts_nested_loop.json")
    assert plan.relations == (DEC_PARTITION, "accounts")
    assert plan.plan_rows == 4.0


def test_the_index_scan_plan_reports_the_index() -> None:
    """Index use is reported from the node that names it, not from a text search."""
    plan = parsed("usage_events_index_scan.json")
    assert plan.indexes == ("usage_events_p2025_12_start_time_idx",)
    assert plan.relations == (DEC_PARTITION,)


def test_the_seq_scan_plan_reports_no_index() -> None:
    """A non-partitioned table: one relation, no index, no partitions to count."""
    plan = parsed("erp_customers_seq_scan.json")
    assert plan.relations == ("customers",)
    assert plan.indexes == ()
    assert plan.plan_rows == 7.0


def test_a_fully_pruned_plan_names_nothing() -> None:
    """``now()`` is past every partition: the planner removed all 14 and names no relation."""
    plan = parsed("usage_events_runtime_bound_unpruned.json")
    assert plan.relations == ()
    assert plan.subplans_removed == 14


@pytest.mark.parametrize(
    "document",
    [None, [], {}, {"Plan": {}}, "not a plan", [{"Plan": "nope"}]],
)
def test_anything_that_is_not_a_plan_parses_to_none(document: object) -> None:
    """``None`` is a real answer: garbage read as "zero partitions scanned" looks like a prune."""
    assert parse_plan(document) is None
    assert parse_plan_text(json.dumps(document)) is None


def test_plan_text_round_trips_through_a_summary() -> None:
    """``PlanSummary.raw`` is the plan text, so the verdict can be re-derived from history."""
    text = (PLANS / "usage_events_event_time_pruned.json").read_text(encoding="utf-8")
    assert parse_plan_text(text) == parsed("usage_events_event_time_pruned.json")
    assert parse_plan_text(None) is None


# --------------------------------------------------------------------------------------
# The verdict: expected vs named, and PROVEN only on an exact match
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "proof", "expected", "named"),
    [
        (
            "usage_events_event_time_pruned.json",
            PruningProof.PROVEN,
            (DEC_PARTITION,),
            (DEC_PARTITION,),
        ),
        ("usage_events_index_scan.json", PruningProof.PROVEN, (DEC_PARTITION,), (DEC_PARTITION,)),
        (
            "usage_events_parameterised.json",
            PruningProof.PROVEN,
            (DEC_PARTITION,),
            (DEC_PARTITION,),
        ),
        (
            "usage_events_accounts_nested_loop.json",
            PruningProof.PROVEN,
            (DEC_PARTITION,),
            (DEC_PARTITION,),
        ),
        (
            "usage_events_ingestion_time_unpruned.json",
            PruningProof.UNPROVEN,
            (DEC_PARTITION,),
            tuple(f"usage_events_p{year}_{month:02d}" for year, month in MONTHS),
        ),
        (
            "usage_events_deferred_pruning.json",
            PruningProof.UNPROVEN,
            (DEC_PARTITION,),
            tuple(f"usage_events_p{year}_{month:02d}" for year, month in MONTHS),
        ),
        (
            "usage_events_runtime_bound_unpruned.json",
            PruningProof.UNPROVEN,
            (DEC_PARTITION,),
            (),
        ),
    ],
)
def test_the_verdict_is_the_comparison_of_two_named_sets(
    monthly: Partitioning,
    name: str,
    proof: PruningProof,
    expected: tuple[str, ...],
    named: tuple[str, ...],
) -> None:
    """Every case names what it expected and what the plan named; the proof is their equality."""
    verdict = pruning_verdict(summary(name), monthly, DECEMBER)
    assert verdict.proof is proof
    assert verdict.expected == expected
    assert verdict.named == named
    if proof is PruningProof.PROVEN:
        assert verdict.unexpected == () and verdict.missing == ()
        assert verdict.ok


def test_an_unpruned_verdict_names_what_it_scanned_and_why(monthly: Partitioning) -> None:
    """The falsifiable part: 13 partitions named beyond the one the window spans."""
    verdict = pruning_verdict(
        summary("usage_events_ingestion_time_unpruned.json"), monthly, DECEMBER
    )
    assert len(verdict.unexpected) == 13
    assert DEC_PARTITION not in verdict.unexpected
    assert verdict.expected == (DEC_PARTITION,)
    assert verdict.missing == ()
    assert verdict.reason.startswith("scanned 13 partition(s) outside the window")
    assert "outside the window" in verdict.reason


def test_a_deferred_plan_says_that_pruning_is_deferred(monthly: Partitioning) -> None:
    """When the plan carries parameters, the reason names them instead of guessing."""
    verdict = pruning_verdict(summary("usage_events_deferred_pruning.json"), monthly, DECEMBER)
    assert verdict.proof is PruningProof.UNPROVEN
    assert "pruning is deferred to run time" in verdict.reason
    assert "$0" in verdict.reason


def test_proven_is_never_produced_without_a_window(monthly: Partitioning) -> None:
    """No window on the key means no partition list can be expected, so nothing can be proven."""
    verdict = pruning_verdict(summary("usage_events_event_time_pruned.json"), monthly, None)
    assert verdict.proof is PruningProof.UNKNOWN
    assert verdict.expected == () and verdict.named == ()
    assert "no window" in verdict.reason


def test_a_non_partitioned_table_is_unknown_not_a_fake_proven() -> None:
    """The answer the task forbids: a table with no partitions must never read as proven."""
    plan = summary("erp_customers_seq_scan.json", total=None)
    assert plan.partitions_total is None
    for partitioning in (None, Partitioning(strategy=PartitionStrategy.NONE)):
        verdict = pruning_verdict(plan, partitioning, DECEMBER)
        assert verdict.proof is PruningProof.UNKNOWN
        assert not verdict.ok


def test_a_table_windows_cannot_address_is_unknown_not_proven() -> None:
    """Hash partitions are chosen by a hash function: a window says nothing about which are read."""
    hashed = Partitioning(
        strategy=PartitionStrategy.HASH,
        key=("client_no",),
        bounds=tuple(PartitionBound(name=f"accounts_hash_p{index}") for index in range(4)),
    )
    verdict = pruning_verdict(
        summary("usage_events_ingestion_time_unpruned.json"), hashed, DECEMBER
    )
    assert verdict.proof is PruningProof.UNKNOWN
    assert "hash" in verdict.reason


def test_an_unreadable_plan_is_unknown(monthly: Partitioning) -> None:
    """A source whose plan cannot be read takes the statistics fallback, not a fake verdict."""
    for raw in (None, "not json at all", '{"Plan": {}}'):
        verdict = pruning_verdict(PlanSummary(raw=raw), monthly, DECEMBER)
        assert verdict.proof is PruningProof.UNKNOWN
        assert "no EXPLAIN document" in verdict.reason


def test_bounds_that_did_not_parse_are_unknown_not_unproven() -> None:
    """With no usable bounds there is nothing to expect, so the honest answer is UNKNOWN."""
    blank = Partitioning(strategy=PartitionStrategy.RANGE, key=("start_time",))
    verdict = pruning_verdict(summary("usage_events_event_time_pruned.json"), blank, DECEMBER)
    assert verdict.proof is PruningProof.UNKNOWN
    assert "no partition bounds" in verdict.reason


def test_stamp_fills_the_summary_the_planner_keeps(monthly: Partitioning) -> None:
    """The verdict is what history records: ``PlanSummary.pruning_proof`` stops being UNKNOWN."""
    original = summary("usage_events_event_time_pruned.json")
    assert original.pruning_proof is PruningProof.UNKNOWN
    stamped = pruning_verdict(original, monthly, DECEMBER).stamp(original)
    assert stamped.pruning_proof is PruningProof.PROVEN
    assert stamped.raw == original.raw
    assert stamped.partitions_scanned == 1


def test_other_relations_are_reported_and_not_counted_as_partitions(
    monthly: Partitioning,
) -> None:
    """The join's other side is visible evidence, not a partition of the table being judged."""
    verdict: PruningVerdict = pruning_verdict(
        summary("usage_events_accounts_nested_loop.json"), monthly, DECEMBER
    )
    assert verdict.other_relations == ("accounts",)
    assert verdict.proof is PruningProof.PROVEN


# --------------------------------------------------------------------------------------
# The window arithmetic agrees with the planner's static estimate
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        (
            Range(datetime(2025, 12, 2, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
            [DEC_PARTITION],
        ),
        (
            Range(datetime(2025, 11, 15, tzinfo=UTC), datetime(2025, 12, 15, tzinfo=UTC)),
            ["usage_events_p2025_11", DEC_PARTITION],
        ),
        (
            Range(datetime(2025, 12, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
            [DEC_PARTITION],
        ),
        (
            Range(datetime(2025, 12, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)),
            [DEC_PARTITION],
        ),
        (
            Range(datetime(2027, 1, 1, tzinfo=UTC), datetime(2027, 2, 1, tzinfo=UTC)),
            [],
        ),
        (
            Range(datetime(2024, 12, 1, tzinfo=UTC), datetime(2025, 2, 1, tzinfo=UTC)),
            [
                "usage_events_p2024_12",
                "usage_events_p2025_01",
            ],
        ),
    ],
)
def test_the_window_selects_partitions_and_matches_the_static_estimate(
    monthly: Partitioning, window: Range, expected: list[str]
) -> None:
    """Two implementations must agree, or the verdict and the estimate would tell two stories."""
    names = partitions_in_window(monthly, window)
    assert list(names) == expected
    assert len(names) == estimate_partition_scan(monthly, window).partitions


def test_the_window_bound_is_half_open_on_the_lower_edge(monthly: Partitioning) -> None:
    """``[2026-01-01, 2026-02-01)`` touches nothing: the last partition ends where it starts."""
    window = Range(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC))
    assert partitions_in_window(monthly, window) == ()
    assert estimate_partition_scan(monthly, window).partitions == 0


def test_an_empty_window_touches_nothing(monthly: Partitioning) -> None:
    """An inverted or zero-length window is a bug upstream, and reads nothing here."""
    moment = datetime(2025, 12, 5, tzinfo=UTC)
    assert partitions_in_window(monthly, Range(moment, moment)) == ()
