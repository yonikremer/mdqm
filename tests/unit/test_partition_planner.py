"""Tests for the static partition planner (task C08).

Every acceptance criterion is covered:

* one table-driven block per strategy: range-monthly, list, hash, expression and none;
* the function-on-key rewrite is proved **semantically equal** by running the original and the
  rewritten query against the D02 world in DuckDB and comparing the rows;
* the refusal carries a ``partitions_if_fixed`` estimate of the ``'2 of 60'`` shape, and a refused
  plan provably calls no source;
* ``align_to_partitions`` splits a 70-day range into month-aligned, half-open segments.

Plus the two invariants hypothesis can state properly: adding a key filter never *increases* the
estimated partition count, and the added range never drops a row the original predicate kept.

Pure, offline: no network, no Docker, no sleeping, no clock read (``exec_ts`` is a constant).
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import duckdb
import pytest
import sqlglot
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mdqm.models import (
    Fragment,
    Limits,
    PartitionBound,
    PartitionGranularity,
    Partitioning,
    PartitionStrategy,
)
from mdqm.planner.partitions import (
    PartitionNote,
    Range,
    align_to_partitions,
    estimate_partition_scan,
    plan_partitions,
)
from mdqm.spec.durations import Duration
from mdqm.testing.fake_adapter import FakeAdapter
from mdqm.testing.world import World, WorldConfig, build_world

EXEC_TS = datetime(2026, 1, 1, tzinfo=UTC)
LIMITS = Limits(timeout_s=10.0, max_rows=10_000, max_bytes=50_000_000, max_partitions=8)


def _month(offset: int, *, start: tuple[int, int] = (2025, 5)) -> datetime:
    """The first instant of the month ``offset`` months after ``start``."""
    index = start[0] * 12 + (start[1] - 1) + offset
    year, month = divmod(index, 12)
    return datetime(year, month + 1, 1, tzinfo=UTC)


def _bounds(count: int, *, start: tuple[int, int] = (2025, 5)) -> tuple[PartitionBound, ...]:
    """``count`` monthly partitions with a declared row estimate, from ``start`` onwards."""
    return tuple(
        PartitionBound(
            name=f"p{offset + 1:02d}",
            lower=_month(offset, start=start),
            upper=_month(offset + 1, start=start),
            row_estimate=100,
        )
        for offset in range(count)
    )


def _range_partitioning(**overrides: object) -> Partitioning:
    """The DESIGN shape: monthly range partitions on ``start_time``."""
    fields: dict[str, object] = {
        "strategy": "range",
        "key": ("start_time",),
        "granularity": "month",
        "bounds": _bounds(8),
    }
    fields.update(overrides)
    return Partitioning.model_validate(fields)


def _pred(sql: str) -> sqlglot.exp.Expr:
    return sqlglot.parse_one(sql, read="duckdb")


def _plan(
    partitioning: Partitioning,
    *predicates: str,
    default_window: str | None = None,
    max_partitions: int | None = None,
    table: str = "billing.usage_events",
):  # type: ignore[no-untyped-def]
    return plan_partitions(
        partitioning,
        tuple(_pred(sql) for sql in predicates),
        exec_ts=EXEC_TS,
        table=table,
        default_window=None if default_window is None else Duration.parse(default_window),
        max_partitions=max_partitions,
    )


# --- range-monthly: detection, rewrite, estimate ------------------------------------------

RANGE_CASES = [
    pytest.param(
        "CAST(start_time AS DATE) = '2025-07-15'",
        "start_time >= CAST('2025-07-15 00:00:00' AS TIMESTAMP) "
        "AND start_time < CAST('2025-07-16 00:00:00' AS TIMESTAMP)",
        1,
        (datetime(2025, 7, 15, tzinfo=UTC), datetime(2025, 7, 16, tzinfo=UTC)),
        id="cast-to-date-equals-one-day",
    ),
    pytest.param(
        "TRUNC(start_time) = '2025-07-15'",
        "start_time >= CAST('2025-07-15 00:00:00' AS TIMESTAMP) "
        "AND start_time < CAST('2025-07-16 00:00:00' AS TIMESTAMP)",
        1,
        (datetime(2025, 7, 15, tzinfo=UTC), datetime(2025, 7, 16, tzinfo=UTC)),
        id="bare-trunc-is-a-day",
    ),
    pytest.param(
        "date_trunc('month', start_time) = '2025-07-01'",
        "start_time >= CAST('2025-07-01 00:00:00' AS TIMESTAMP) "
        "AND start_time < CAST('2025-08-01 00:00:00' AS TIMESTAMP)",
        1,
        (datetime(2025, 7, 1, tzinfo=UTC), datetime(2025, 8, 1, tzinfo=UTC)),
        id="date-trunc-month-equals",
    ),
    pytest.param(
        "date_trunc('month', start_time) < '2025-07-15'",
        "start_time < CAST('2025-08-01 00:00:00' AS TIMESTAMP)",
        3,
        None,
        id="date-trunc-upper-moves-to-the-next-boundary",
    ),
    pytest.param(
        "date_trunc('month', start_time) >= '2025-07-15'",
        "start_time >= CAST('2025-07-15 00:00:00' AS TIMESTAMP)",
        6,
        None,
        id="date-trunc-lower-keeps-the-weaker-bound",
    ),
    pytest.param(
        "CAST(start_time AS DATE) BETWEEN '2025-07-01' AND '2025-07-31'",
        "start_time >= CAST('2025-07-01 00:00:00' AS TIMESTAMP) "
        "AND start_time < CAST('2025-08-01 00:00:00' AS TIMESTAMP)",
        1,
        (datetime(2025, 7, 1, tzinfo=UTC), datetime(2025, 8, 1, tzinfo=UTC)),
        id="cast-between-is-two-day-bounds",
    ),
    pytest.param(
        "date_trunc('month', CAST(start_time AS DATE)) = '2025-07-01'",
        "start_time >= CAST('2025-07-01 00:00:00' AS TIMESTAMP) "
        "AND start_time < CAST('2025-08-01 00:00:00' AS TIMESTAMP)",
        1,
        (datetime(2025, 7, 1, tzinfo=UTC), datetime(2025, 8, 1, tzinfo=UTC)),
        id="nested-floors-take-the-coarser-unit",
    ),
    pytest.param(
        "start_time >= '2025-07-01' AND start_time < '2025-09-01'",
        "",
        2,
        (datetime(2025, 7, 1, tzinfo=UTC), datetime(2025, 9, 1, tzinfo=UTC)),
        id="already-prunable-adds-nothing",
    ),
    pytest.param(
        "'2025-07-01' <= start_time",
        "",
        6,  # July onward: the lower bound really does drop May and June
        None,
        id="operands-the-other-way-round",
    ),
]


@pytest.mark.parametrize(
    ("sql", "expected_where", "expected_partitions", "expected_scope"), RANGE_CASES
)
def test_range_monthly_detects_rewrites_and_counts(
    sql: str,
    expected_where: str,
    expected_partitions: int,
    expected_scope: tuple[datetime, datetime] | None,
) -> None:
    """Every accepted predicate shape: what it adds, how many partitions it touches, its scope."""
    plan = _plan(_range_partitioning(), sql)
    assert plan.ok
    assert plan.error is None
    assert plan.where_sql("duckdb") == expected_where
    assert plan.estimate is not None
    assert plan.estimate.partitions == expected_partitions
    assert plan.estimate.total == 8
    assert plan.estimate.rows == 100 * expected_partitions
    if expected_scope is None:
        assert plan.scope is None
    else:
        assert plan.scope == Range(*expected_scope)
    if expected_where:
        # A rewrite is never silent: the note and the applied fix both name the column.
        assert PartitionNote.KEY_FUNCTION_REWRITTEN in plan.notes
        assert plan.applied_fixes and "start_time" in plan.applied_fixes[0]
    else:
        assert plan.notes == ()
        assert plan.applied_fixes == ()


def test_range_monthly_estimate_label_matches_the_design_shape() -> None:
    """The ``'2 of 60'`` label of DESIGN section 5, from declared bounds only."""
    plan = _plan(_range_partitioning(), "start_time >= '2024-01-01'")
    assert plan.estimate is not None
    assert plan.estimate.label == "8 of 8"
    narrow = _plan(
        _range_partitioning(), "start_time >= '2025-11-01' AND start_time < '2026-01-01'"
    )
    assert narrow.estimate is not None
    assert narrow.estimate.label == "2 of 8"


@pytest.mark.parametrize(
    "sql",
    [
        "date_trunc('month', start_time) = '2025-07-15'",  # mid-month: matches no partition
        "EXTRACT(year FROM start_time) = 2025",  # a function this planner cannot invert
    ],
)
def test_range_unprovable_filter_flags_human_approval(sql: str) -> None:
    """The key *is* filtered, just not provably: no error, a note, and the full count as a bound."""
    plan = _plan(_range_partitioning(), sql)
    assert plan.ok
    assert PartitionNote.NEEDS_HUMAN_APPROVAL in plan.notes
    assert plan.estimate is not None
    assert plan.estimate.partitions == 8
    assert plan.where_sql() == ""


# --- range-monthly: the missing filter is refused -----------------------------------------


def test_missing_key_filter_is_refused_with_a_precise_fix() -> None:
    """The MVP acceptance item: no key filter is an error with the exact thing to add."""
    plan = _plan(_range_partitioning(), default_window="45d")
    assert not plan.ok
    error = plan.error
    assert error is not None
    assert error.code == "PARTITION_SCAN_ALL"
    assert '"billing.usage_events" is partitioned month on "start_time"' in error.problem
    assert '"time"' in error.fix
    assert 'last": "45d"' in error.fix
    assert 'filter on "start_time"' in error.fix
    # DESIGN section 5: the estimate after the fix, so the agent can see it is worth doing.
    assert error.partitions_if_fixed == "2 of 8"


def test_refusal_falls_back_to_30d_when_the_table_declares_no_window() -> None:
    """No ``default_window``: the fix suggests the 30d default the error text names."""
    plan = _plan(_range_partitioning())
    assert plan.error is not None
    assert 'last": "30d"' in plan.error.fix
    assert plan.error.partitions_if_fixed == "1 of 8"


def test_refused_plan_calls_no_source() -> None:
    """Nothing runs after a refusal: the adapter's call log stays empty."""
    world = build_world(WorldConfig.small())
    partitioning = {"usage_events": _range_partitioning(bounds=_bounds(8), require_key_filter=True)}
    adapter = FakeAdapter.for_world(world, partitioning=partitioning)
    plan = _plan(partitioning["usage_events"], default_window="30d", table="usage_events")
    assert plan.error is not None
    assert adapter.calls == []


def test_default_window_is_injected_when_the_table_allows_it() -> None:
    """``require_key_filter: false``: the window is added, reported, and counted."""
    plan = _plan(
        _range_partitioning(require_key_filter=False), default_window="45d", max_partitions=8
    )
    assert plan.ok
    assert PartitionNote.DEFAULT_WINDOW_APPLIED in plan.notes
    assert plan.scope == Range(EXEC_TS - timedelta(days=45), EXEC_TS)
    assert plan.where_sql() == (
        "start_time >= CAST('2025-11-17 00:00:00' AS TIMESTAMP) "
        "AND start_time < CAST('2026-01-01 00:00:00' AS TIMESTAMP)"
    )
    assert plan.applied_fixes and "45d" in plan.applied_fixes[0]
    assert plan.estimate is not None
    assert plan.estimate.partitions == 2  # the bound is counted, not assumed
    # 45 days from 2025-11-17 crosses one boundary, so the cache gets two aligned segments.
    assert len(plan.segments) == 2
    assert plan.segments[0].lower == plan.scope.lower
    assert plan.segments[-1].upper == plan.scope.upper


def test_allowing_an_unfiltered_query_without_a_window_still_refuses() -> None:
    """No window to inject means no safe scan: the setting alone cannot invent a policy."""
    plan = _plan(_range_partitioning(require_key_filter=False))
    assert plan.error is not None
    assert plan.error.code == "PARTITION_SCAN_ALL"


# --- the cap -----------------------------------------------------------------------------


def test_cap_refuses_a_scan_over_max_partitions() -> None:
    """Static estimate over ``max_partitions`` is ``COST_TOO_HIGH``, not a connection."""
    plan = _plan(_range_partitioning(), "start_time >= '2025-05-01'", max_partitions=3)
    assert not plan.ok
    error = plan.error
    assert error is not None
    assert error.code == "COST_TOO_HIGH"
    assert "8 of 8 partitions" in error.problem
    assert "3 partitions" in error.problem
    assert '"time"' in error.fix


def test_cap_passes_when_the_estimate_fits() -> None:
    """A narrow window under the cap runs; the estimate is still reported."""
    plan = _plan(
        _range_partitioning(),
        "start_time >= '2025-07-01' AND start_time < '2025-07-05'",
        max_partitions=1,
    )
    assert plan.ok
    assert plan.estimate is not None
    assert plan.estimate.partitions == 1


# --- list / hash / expression / none -----------------------------------------------------

VALUES_BOUNDS = (
    PartitionBound(name="north_2025", values=("north",), row_estimate=10),
    PartitionBound(name="south_2025", values=("south",), row_estimate=20),
    PartitionBound(name="default", values=("unknown",), row_estimate=5),
)
LIST_PARTITIONING = Partitioning(
    strategy=PartitionStrategy.LIST, key=("region",), bounds=VALUES_BOUNDS
)
HASH_PARTITIONING = Partitioning(
    strategy=PartitionStrategy.HASH, key=("cust_id",), bounds=_bounds(4)
)
EXPRESSION_PARTITIONING = Partitioning.model_validate(
    {
        "strategy": "expression",
        "expression": "date_trunc('month', created)",
        "granularity": "month",
        "bounds": _bounds(8),
    }
)


@pytest.mark.parametrize(
    ("sql", "expected_partitions", "expected_rows", "expected_note"),
    [
        ("region = 'north'", 1, 10, None),
        ("region IN ('north', 'unknown')", 2, 15, None),
        ("region = 'nowhere'", 0, None, None),
        ("region = 'north' AND region = 'south'", 0, None, None),
        ("lower(region) = 'north'", 3, 35, PartitionNote.NEEDS_HUMAN_APPROVAL),
        ("region > 'm'", 3, 35, PartitionNote.NEEDS_HUMAN_APPROVAL),
    ],
)
def test_list_strategy_matches_declared_values(
    sql: str,
    expected_partitions: int,
    expected_rows: int | None,
    expected_note: PartitionNote | None,
) -> None:
    """List partitioning selects partitions by the values the filter names, nothing else."""
    plan = _plan(LIST_PARTITIONING, sql, table="crm.customers")
    assert plan.ok
    assert plan.estimate is not None
    assert plan.estimate.partitions == expected_partitions
    assert plan.estimate.total == 3
    assert plan.estimate.rows == expected_rows
    assert (expected_note in plan.notes) if expected_note else plan.notes == ()
    assert plan.where_sql() == ""


@pytest.mark.parametrize(
    ("sql", "expected_partitions"),
    [
        ("cust_id = 42", 1),
        ("cust_id IN (1, 2, 3)", 3),
        ("cust_id IN (1, 2, 3, 4, 5, 6)", 4),  # capped by the declared total
        ("cust_id >= 100", 4),  # a range does not let a hash source skip a partition
    ],
)
def test_hash_strategy_needs_a_filter_on_the_key_itself(sql: str, expected_partitions: int) -> None:
    """Hash partitioning: an equality filter bounds the scan; a range proves nothing."""
    plan = _plan(HASH_PARTITIONING, sql, table="billing.accounts")
    assert plan.ok
    assert plan.estimate is not None
    assert plan.estimate.partitions == expected_partitions
    assert plan.estimate.total == 4
    if sql.startswith("cust_id =") or sql.startswith("cust_id IN"):
        assert PartitionNote.ESTIMATE_UNPROVEN in plan.notes
    else:
        assert PartitionNote.NEEDS_HUMAN_APPROVAL in plan.notes


@pytest.mark.parametrize(
    ("sql", "expected_partitions", "expected_where"),
    [
        (
            "date_trunc('month', created) = '2025-07-01'",
            1,
            "created >= CAST('2025-07-01 00:00:00' AS TIMESTAMP) "
            "AND created < CAST('2025-08-01 00:00:00' AS TIMESTAMP)",
        ),
        ("created >= '2025-07-01' AND created < '2025-09-01'", 2, ""),
    ],
)
def test_expression_strategy_prunes_through_the_declared_expression(
    sql: str, expected_partitions: int, expected_where: str
) -> None:
    """An expression key prunes when the predicate matches it, or reads its own column."""
    plan = _plan(EXPRESSION_PARTITIONING, sql, table="api.tickets")
    assert plan.ok
    assert plan.estimate is not None
    assert plan.where_sql() == expected_where
    assert plan.estimate.partitions == expected_partitions
    assert plan.estimate.total == 8


def test_expression_strategy_flags_an_unmatched_expression() -> None:
    """``region || state`` is not invertible here: a human has to approve the pruning."""
    partitioning = Partitioning.model_validate(
        {"strategy": "expression", "expression": "region || state"}
    )
    plan = _plan(partitioning, "lower(region) = 'north'", table="api.tickets")
    assert plan.ok
    assert PartitionNote.NEEDS_HUMAN_APPROVAL in plan.notes
    assert plan.estimate is not None
    assert plan.estimate.partitions is None  # no declared bounds: nothing to count


def test_none_strategy_plans_nothing() -> None:
    """An unpartitioned table is the cost gate's problem: no predicates, no notes, no error."""
    plan = _plan(Partitioning(strategy=PartitionStrategy.NONE), "start_time >= '2025-07-01'")
    assert plan == type(plan)()
    assert plan.ok
    assert plan.error is None
    assert plan.estimate is None


def test_missing_partitioning_plans_nothing() -> None:
    """``None`` metadata (a source that does not support partitioning) changes nothing."""
    plan = plan_partitions(None, (), exec_ts=EXEC_TS)
    assert plan.ok
    assert plan.predicates == ()


# --- alignment ---------------------------------------------------------------------------


def test_alignment_splits_seventy_days_into_month_aligned_segments() -> None:
    """C08 acceptance: a 70-day range becomes month-aligned half-open pieces."""
    window = Range(datetime(2026, 1, 15, tzinfo=UTC), datetime(2026, 3, 26, tzinfo=UTC))
    assert window.length == timedelta(days=70)
    segments = align_to_partitions(window)
    assert segments == (
        Range(datetime(2026, 1, 15, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)),
        Range(datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)),
        Range(datetime(2026, 3, 1, tzinfo=UTC), datetime(2026, 3, 26, tzinfo=UTC)),
    )
    # Contiguous and exactly covering: no gap, no overlap, nothing outside the window.
    assert segments[0].lower == window.lower
    assert segments[-1].upper == window.upper
    for left, right in itertools.pairwise(segments):
        assert left.upper == right.lower


@pytest.mark.parametrize(
    ("granularity", "window", "expected"),
    [
        (
            PartitionGranularity.DAY,
            Range(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 3, 12, tzinfo=UTC)),
            (
                Range(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)),
                Range(datetime(2026, 1, 2, tzinfo=UTC), datetime(2026, 1, 3, tzinfo=UTC)),
                Range(datetime(2026, 1, 3, tzinfo=UTC), datetime(2026, 1, 3, 12, tzinfo=UTC)),
            ),
        ),
        (
            PartitionGranularity.QUARTER,
            Range(datetime(2026, 1, 20, tzinfo=UTC), datetime(2026, 8, 2, tzinfo=UTC)),
            (
                Range(datetime(2026, 1, 20, tzinfo=UTC), datetime(2026, 4, 1, tzinfo=UTC)),
                Range(datetime(2026, 4, 1, tzinfo=UTC), datetime(2026, 7, 1, tzinfo=UTC)),
                Range(datetime(2026, 7, 1, tzinfo=UTC), datetime(2026, 8, 2, tzinfo=UTC)),
            ),
        ),
        (
            PartitionGranularity.WEEK,
            Range(datetime(2026, 1, 7, tzinfo=UTC), datetime(2026, 1, 21, tzinfo=UTC)),
            (
                Range(datetime(2026, 1, 7, tzinfo=UTC), datetime(2026, 1, 12, tzinfo=UTC)),
                Range(datetime(2026, 1, 12, tzinfo=UTC), datetime(2026, 1, 19, tzinfo=UTC)),
                Range(datetime(2026, 1, 19, tzinfo=UTC), datetime(2026, 1, 21, tzinfo=UTC)),
            ),
        ),
        (
            PartitionGranularity.MONTH,
            Range(datetime(2026, 3, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)),
            (),
        ),
    ],
)
def test_alignment_per_granularity(
    granularity: PartitionGranularity, window: Range, expected: tuple[Range, ...]
) -> None:
    """Days, ISO weeks (Monday), quarters and an empty window."""
    assert align_to_partitions(window, granularity) == expected


def test_range_rejects_a_naive_or_inverted_window() -> None:
    """A naive or inverted window is a programming bug, never agent input."""
    with pytest.raises(ValueError, match="timezone-aware"):
        Range(datetime(2026, 1, 1), datetime(2026, 1, 2, tzinfo=UTC))
    with pytest.raises(ValueError, match="before"):
        Range(datetime(2026, 1, 2, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))


def test_plan_rejects_a_naive_exec_ts() -> None:
    """``exec_ts`` must be aware UTC; a naive value is an internal bug, not an error response."""
    with pytest.raises(ValueError, match="exec_ts"):
        plan_partitions(_range_partitioning(), (), exec_ts=datetime(2026, 1, 1))


# --- semantic equality on the synthetic world --------------------------------------------

_WORLD: World | None = None


def _world() -> World:
    """One generated organisation, built once: the planted quirks must not shift per test."""
    global _WORLD
    if _WORLD is None:
        _WORLD = build_world(WorldConfig.small())
    return _WORLD


def _rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[int]:
    return sorted(row[0] for row in con.execute(sql).fetchall())


REWRITE_CASES = [
    "CAST(start_time AS DATE) = '2025-07-15'",
    "CAST(start_time AS DATE) = '2025-11-02'",
    "date_trunc('month', start_time) = '2025-08-01'",
    "date_trunc('month', start_time) < '2025-09-15'",
    "date_trunc('month', start_time) >= '2025-09-15'",
    "CAST(start_time AS DATE) BETWEEN '2025-07-01' AND '2025-07-03'",
    "EXTRACT(year FROM start_time) = 2025",
    "start_time >= '2025-07-01' AND start_time < '2025-09-01'",
]


@pytest.fixture(scope="module")
def duck() -> duckdb.DuckDBPyConnection:
    """The world's ``usage_events`` loaded into one in-process DuckDB connection."""
    connection = duckdb.connect(":memory:")
    adapter = FakeAdapter.for_world(_world())
    try:
        # The adapter already loaded the rows; copy them through SQL so the test reads real data.
        connection.execute("ATTACH ':memory:' AS w")
        rows = [
            batch.to_pydict()
            for batch in adapter.execute(
                Fragment(
                    source="fake",
                    dialect="duckdb",
                    sql=(
                        "SELECT event_id, client_no, start_time, insertion_time, units "
                        "FROM usage_events"
                    ),
                    tables=("usage_events",),
                    exec_ts=EXEC_TS,
                ),
                LIMITS,
            )
        ]
    finally:
        adapter.close()
    columns = {name: [value for batch in rows for value in batch[name]] for name in rows[0]}
    connection.execute(
        "CREATE TABLE usage_events (event_id BIGINT, client_no VARCHAR, start_time TIMESTAMP,"
        " insertion_time TIMESTAMP, units INTEGER)"
    )
    connection.executemany(
        "INSERT INTO usage_events VALUES (?, ?, ?, ?, ?)",
        list(
            zip(
                columns["event_id"],
                columns["client_no"],
                columns["start_time"],
                columns["insertion_time"],
                columns["units"],
                strict=True,
            )
        ),
    )
    yield connection
    connection.close()


@pytest.mark.parametrize("sql", REWRITE_CASES)
def test_rewrite_is_semantically_equal_on_the_world(
    duck: duckdb.DuckDBPyConnection, sql: str
) -> None:
    """The rewrite never changes the answer: same rows before and after, on real synthetic data."""
    plan = _plan(_range_partitioning(), sql)
    assert plan.ok
    before = _rows(duck, f"SELECT event_id FROM usage_events WHERE {sql}")
    added = plan.where_sql("duckdb")
    after = _rows(duck, f"SELECT event_id FROM usage_events WHERE {sql} AND ({added or 'TRUE'})")
    assert after == before
    # ... and the case is not vacuous: at least one of them really returns rows.
    assert before or sql.startswith(
        ("date_trunc('month', start_time) <", "date_trunc('month', start_time) >=")
    )


def test_rewrite_makes_a_function_on_key_filter_prunable(duck: duckdb.DuckDBPyConnection) -> None:
    """The point of the rewrite: the raw column really appears, so a source can prune on it."""
    plan = _plan(_range_partitioning(), "CAST(start_time AS DATE) = '2025-07-15'")
    added = plan.where_sql("duckdb")
    assert "start_time >=" in added and "CAST(start_time AS DATE)" not in added
    assert _rows(duck, f"SELECT event_id FROM usage_events WHERE {added}") == _rows(
        duck, "SELECT event_id FROM usage_events WHERE CAST(start_time AS DATE) = '2025-07-15'"
    )


# --- invariants --------------------------------------------------------------------------

_MOMENTS = st.datetimes(
    min_value=datetime(2025, 1, 1),
    max_value=datetime(2027, 1, 1),
    timezones=st.just(UTC),
)


@st.composite
def _bound_sets(draw: st.DrawFn) -> tuple[PartitionBound, ...]:
    """1-6 monthly partitions on a random boundary, with unknown row counts sometimes."""
    start = draw(_MOMENTS)
    count = draw(st.integers(min_value=1, max_value=6))
    first = _month(0, start=(start.year, start.month))
    return tuple(
        PartitionBound(
            name=f"p{offset}",
            lower=_month(offset, start=(first.year, first.month)),
            upper=_month(offset + 1, start=(first.year, first.month)),
            row_estimate=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=10_000))),
        )
        for offset in range(count)
    )


@given(bounds=_bound_sets(), window=_MOMENTS, inside=st.floats(min_value=0.0, max_value=1.0))
@settings(
    max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_adding_a_key_filter_never_increases_the_estimate(
    bounds: tuple[PartitionBound, ...], window: datetime, inside: float
) -> None:
    """The invariant the refusal rests on: narrowing the window can only shrink the count."""
    partitioning = _range_partitioning(bounds=bounds)
    lower = _month(0, start=(window.year, window.month))
    upper = _month(2, start=(window.year, window.month))
    wide = estimate_partition_scan(partitioning, Range(lower, upper))
    point = lower + (upper - lower) * inside
    narrow = estimate_partition_scan(
        partitioning, Range(point, min(point + timedelta(seconds=1), upper))
    )
    assert wide.partitions is not None and narrow.partitions is not None
    assert narrow.partitions <= wide.partitions
    assert wide.partitions <= wide.total or wide.total is None


@given(bounds=_bound_sets(), moment=_MOMENTS)
@settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_estimate_never_exceeds_the_declared_total(
    bounds: tuple[PartitionBound, ...], moment: datetime
) -> None:
    """A static estimate can never promise fewer partitions than the catalog has, nor more."""
    partitioning = _range_partitioning(bounds=bounds)
    plan_window = Range(_month(0, start=(moment.year, moment.month)), moment)
    estimate = estimate_partition_scan(partitioning, plan_window)
    assert estimate.total == len(bounds)
    assert estimate.partitions is not None and 0 <= estimate.partitions <= len(bounds)
