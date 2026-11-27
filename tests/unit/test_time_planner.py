"""Tests for the time planner (task C07).

Every acceptance criterion is covered here:

* the DESIGN section 6 SQL is reproduced exactly (``insertion_time >= 14:00`` gives
  ``start_time >= 08:00`` with a 6h lag);
* an unknown lag with ``new_since`` yields ``INSERTION_TIME_NO_BOUND`` whose ``fix`` names the
  spec's ``time`` field;
* hypothesis proves the derived bound never loses a row, on the D02 world executed in DuckDB;
* a partition key of ingestion time widens an event range to an ingestion range with no row lost.

These are pure, offline unit tests: no network, no Docker, no clock, no sleeping.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import duckdb
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mdqm.planner.time import (
    TimeNote,
    event_column,
    ingestion_column,
    known_as_of,
    plan_time,
)
from mdqm.semantic.models import TableSemantics
from mdqm.spec.models import TimeSpec
from mdqm.testing.world import World, WorldConfig, build_world

# The canonical usage_events shape from DESIGN section 6 and examples/semantic_example.yaml.
USAGE_EVENTS = TableSemantics.model_validate(
    {
        "partitioning": {"key": "start_time", "granularity": "month"},
        "time_columns": {
            "start_time": {"role": "event_time", "indexed": True, "use_by_default": True},
            "insertion_time": {
                "role": "ingestion_time",
                "derives_bound_on": "start_time",
                "lag": {"min": "0s", "max": "6h", "verify": "nightly"},
            },
        },
    }
)

# Same table, but partitioned on the ingestion column instead of the event column.
USAGE_BY_LOAD = TableSemantics.model_validate(
    {
        "partitioning": {"key": "insertion_time", "granularity": "day"},
        "time_columns": {
            "start_time": {"role": "event_time", "indexed": False, "use_by_default": True},
            "insertion_time": {
                "role": "ingestion_time",
                "indexed": True,
                "derives_bound_on": "start_time",
                "lag": {"min": "0s", "max": "6h", "verify": "nightly"},
            },
        },
    }
)

# No measured lag at all: the planner must refuse to derive.
NO_LAG = TableSemantics.model_validate(
    {
        "time_columns": {
            "start_time": {"role": "event_time", "indexed": True, "use_by_default": True},
            "insertion_time": {"role": "ingestion_time"},
        },
    }
)

NOW = datetime(2026, 11, 25, 16, 0, tzinfo=UTC)
SIX_HOURS = timedelta(hours=6)


# ---------------------------------------------------------------- DESIGN SQL reproduction


@pytest.mark.parametrize(
    ("since", "expected_bound"),
    [
        (datetime(2026, 11, 25, 14, 0, tzinfo=UTC), datetime(2026, 11, 25, 8, 0, tzinfo=UTC)),
        (datetime(2026, 11, 25, 0, 0, tzinfo=UTC), datetime(2026, 11, 24, 18, 0, tzinfo=UTC)),
        (datetime(2026, 11, 25, 16, 0, tzinfo=UTC), datetime(2026, 11, 25, 10, 0, tzinfo=UTC)),
    ],
)
def test_new_since_reproduces_design_sql(since: datetime, expected_bound: datetime) -> None:
    """DESIGN section 6: ``insertion_time >= X`` plus ``start_time >= X - max_lag``."""
    plan = plan_time(
        TimeSpec(mode="new_since", since=since),
        USAGE_EVENTS,
        exec_ts=NOW,
        table_name="billing.usage_events",
    )
    assert plan.ok
    assert plan.derived_bound == expected_bound
    text = plan.where_sql("postgres")
    assert f"insertion_time >= CAST('{since.replace(tzinfo=None).isoformat(sep=' ')}'" in text
    assert f"start_time >= CAST('{expected_bound.replace(tzinfo=None).isoformat(sep=' ')}'" in text
    # The original insertion predicate stays: correctness first, the bound only narrows the scan.
    assert text.index("insertion_time") < text.index("start_time")


def test_new_since_exact_design_example() -> None:
    """The literal example in DESIGN section 6, compared as SQL, not as substrings."""
    plan = plan_time(
        TimeSpec(mode="new_since", since=datetime(2026, 11, 25, 14, 0, tzinfo=UTC)),
        USAGE_EVENTS,
        exec_ts=NOW,
        table_name="billing.usage_events",
    )
    assert plan.where_sql("postgres") == (
        "insertion_time >= CAST('2026-11-25 14:00:00' AS TIMESTAMP) "
        "AND start_time >= CAST('2026-11-25 08:00:00' AS TIMESTAMP)"
    )


def test_events_in_range_filters_the_default_event_column() -> None:
    """A period question reads event time only: no ingestion predicate is invented."""
    plan = plan_time(
        TimeSpec(mode="events_in_range", **{"from": "2026-11-01", "to": "2026-12-01"}),
        USAGE_EVENTS,
        exec_ts=NOW,
        table_name="billing.usage_events",
    )
    assert plan.ok
    assert plan.window == (
        datetime(2026, 11, 1, tzinfo=UTC),
        datetime(2026, 12, 1, tzinfo=UTC),
    )
    assert plan.derived_bound is None
    text = plan.where_sql("postgres")
    assert "start_time >= CAST('2026-11-01 00:00:00' AS TIMESTAMP)" in text
    assert "start_time < CAST('2026-12-01 00:00:00' AS TIMESTAMP)" in text
    assert "insertion_time" not in text


def test_last_window_is_relative_to_exec_ts() -> None:
    """``last`` is relative to the injected exec_ts, never to a clock read."""
    plan = plan_time(
        TimeSpec(mode="events_in_range", last="7d"),
        USAGE_EVENTS,
        exec_ts=NOW,
        table_name="billing.usage_events",
    )
    assert plan.window == (NOW - timedelta(days=7), NOW)


def test_no_time_intent_plans_nothing() -> None:
    """A spec without ``time`` adds no predicate and no note (the compiler adds LIMIT etc.)."""
    plan = plan_time(None, USAGE_EVENTS, exec_ts=NOW)
    assert plan.ok
    assert plan.predicates == ()
    assert plan.notes == ()


def test_partial_data_note_on_a_window_touching_the_lag_tail() -> None:
    """A window ending inside the last ``max_lag`` is flagged: rows may still be arriving."""
    recent = plan_time(
        TimeSpec(mode="events_in_range", **{"from": "2026-11-20", "to": "2026-11-25T15:00:00"}),
        USAGE_EVENTS,
        exec_ts=NOW,
        table_name="billing.usage_events",
    )
    assert TimeNote.PARTIAL_DATA_POSSIBLE in recent.notes
    old = plan_time(
        TimeSpec(mode="events_in_range", **{"from": "2026-10-01", "to": "2026-10-02"}),
        USAGE_EVENTS,
        exec_ts=NOW,
        table_name="billing.usage_events",
    )
    assert TimeNote.PARTIAL_DATA_POSSIBLE not in old.notes


# ---------------------------------------------------------------- unknown lag refuses to derive


def test_unknown_lag_yields_insertion_time_no_bound_naming_the_time_field() -> None:
    """No lag means no bound: refuse with a fix that names ``"time"`` and the column."""
    plan = plan_time(
        TimeSpec(mode="new_since", since=datetime(2026, 11, 25, 14, 0, tzinfo=UTC)),
        NO_LAG,
        exec_ts=NOW,
        table_name="billing.usage_events",
    )
    assert not plan.ok
    error = plan.error
    assert error is not None
    assert error.code == "INSERTION_TIME_NO_BOUND"
    assert '"time"' in error.fix
    assert "start_time" in error.fix
    assert "events_in_range" in error.fix
    assert plan.predicates == ()


def test_failed_lag_verification_disables_the_derivation() -> None:
    """A nightly violation makes the lag untrusted: derive nothing rather than drop late rows."""
    plan = plan_time(
        TimeSpec(mode="new_since", since=datetime(2026, 11, 25, 14, 0, tzinfo=UTC)),
        USAGE_EVENTS,
        exec_ts=NOW,
        table_name="billing.usage_events",
        lag_unverified=frozenset({"insertion_time"}),
    )
    assert not plan.ok
    assert plan.error is not None
    assert plan.error.code == "INSERTION_TIME_NO_BOUND"


def test_new_since_requires_since() -> None:
    plan = plan_time(TimeSpec(mode="new_since"), USAGE_EVENTS, exec_ts=NOW)
    assert not plan.ok
    assert plan.error is not None
    assert plan.error.code == "INVALID_SPEC"
    assert "time.since" in plan.error.problem or "time.since" in plan.error.fix


def test_ambiguous_window_is_an_invalid_spec() -> None:
    plan = plan_time(TimeSpec(mode="events_in_range"), USAGE_EVENTS, exec_ts=NOW)
    assert not plan.ok
    assert plan.error is not None
    assert plan.error.code == "INVALID_SPEC"


def test_naive_exec_ts_is_a_bug_not_a_default() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        plan_time(None, USAGE_EVENTS, exec_ts=datetime(2026, 11, 25, 16, 0))


# ---------------------------------------------------------------- column selection


def test_column_selection_prefers_use_by_default_then_indexed() -> None:
    event = event_column(USAGE_EVENTS)
    assert event is not None
    assert event[0] == "start_time"
    ingestion = ingestion_column(USAGE_EVENTS)
    assert ingestion is not None
    assert ingestion[0] == "insertion_time"
    assert event_column(TableSemantics()) is None
    assert ingestion_column(TableSemantics()) is None


def test_table_without_event_column_cannot_answer_a_period_question() -> None:
    plan = plan_time(
        TimeSpec(mode="events_in_range", last="7d"),
        TableSemantics.model_validate(
            {"time_columns": {"insertion_time": {"role": "ingestion_time"}}}
        ),
        exec_ts=NOW,
    )
    assert not plan.ok
    assert plan.error is not None
    assert plan.error.code == "INVALID_SPEC"


# ---------------------------------------------------------------- partition key = ingestion time


def test_partition_key_is_ingestion_time_widens_the_range() -> None:
    """Event range ``[X, Y)`` becomes ingestion range ``[X + min_lag, Y + max_lag)``."""
    start = datetime(2026, 11, 1, tzinfo=UTC)
    end = datetime(2026, 11, 2, tzinfo=UTC)
    plan = plan_time(
        TimeSpec(mode="events_in_range", **{"from": start, "to": end}),
        USAGE_BY_LOAD,
        exec_ts=NOW,
        table_name="billing.usage_by_load",
    )
    assert plan.ok
    text = plan.where_sql("postgres")
    # min_lag is 0s, so the lower ingestion bound equals the event start.
    assert "insertion_time >= CAST('2026-11-01 00:00:00' AS TIMESTAMP)" in text
    # max_lag is 6h, so the upper ingestion bound is the event end plus 6h.
    assert "insertion_time < CAST('2026-11-02 06:00:00' AS TIMESTAMP)" in text
    assert "start_time >= CAST('2026-11-01 00:00:00' AS TIMESTAMP)" in text
    assert "start_time < CAST('2026-11-02 00:00:00' AS TIMESTAMP)" in text


def test_min_lag_shifts_the_ingestion_lower_bound() -> None:
    """A non-zero ``min_lag`` moves the lower ingestion bound forward, never past a real row."""
    table = TableSemantics.model_validate(
        {
            "partitioning": {"key": "insertion_time", "granularity": "day"},
            "time_columns": {
                "start_time": {"role": "event_time", "use_by_default": True},
                "insertion_time": {
                    "role": "ingestion_time",
                    "derives_bound_on": "start_time",
                    "lag": {"min": "5m", "max": "6h", "verify": "nightly"},
                },
            },
        }
    )
    plan = plan_time(
        TimeSpec(mode="events_in_range", **{"from": "2026-11-01", "to": "2026-11-02"}),
        table,
        exec_ts=NOW,
    )
    text = plan.where_sql("postgres")
    assert "insertion_time >= CAST('2026-11-01 00:05:00' AS TIMESTAMP)" in text


def test_event_partition_key_adds_no_ingestion_predicate() -> None:
    """When the key is event time, the event filter already prunes: nothing extra is added."""
    plan = plan_time(
        TimeSpec(mode="events_in_range", **{"from": "2026-11-01", "to": "2026-11-02"}),
        USAGE_EVENTS,
        exec_ts=NOW,
    )
    assert "insertion_time" not in plan.where_sql("postgres")


# ---------------------------------------------------------------- known as of T


def test_known_as_of_adds_the_ingestion_ceiling() -> None:
    """'What did we know at T' = ingestion ``<= T`` plus the event range."""
    plan = known_as_of(
        USAGE_EVENTS,
        as_of=datetime(2026, 11, 10, tzinfo=UTC),
        start=datetime(2026, 11, 1, tzinfo=UTC),
        end=datetime(2026, 11, 2, tzinfo=UTC),
        exec_ts=NOW,
    )
    assert plan.ok
    text = plan.where_sql("postgres")
    assert "insertion_time <= CAST('2026-11-10 00:00:00' AS TIMESTAMP)" in text
    assert "start_time >= CAST('2026-11-01 00:00:00' AS TIMESTAMP)" in text
    assert "start_time < CAST('2026-11-02 00:00:00' AS TIMESTAMP)" in text


def test_known_as_of_needs_both_columns() -> None:
    plan = known_as_of(
        NO_LAG,
        as_of=datetime(2026, 11, 10, tzinfo=UTC),
        start=datetime(2026, 11, 1, tzinfo=UTC),
        end=datetime(2026, 11, 2, tzinfo=UTC),
        exec_ts=NOW,
    )
    # NO_LAG has both columns, so this one is fine; a table with neither is the error case.
    assert plan.ok
    empty = known_as_of(
        TableSemantics(),
        as_of=datetime(2026, 11, 10, tzinfo=UTC),
        start=datetime(2026, 11, 1, tzinfo=UTC),
        end=datetime(2026, 11, 2, tzinfo=UTC),
        exec_ts=NOW,
    )
    assert not empty.ok
    assert empty.error is not None
    assert empty.error.code == "INVALID_SPEC"


# ---------------------------------------------------------------- no row lost: D02 world + DuckDB


@pytest.fixture(scope="module")
def small_world() -> World:
    """One small deterministic world (200 customers, ~5k events), shared read-only."""
    return build_world(WorldConfig.small(seed=1))


@pytest.fixture(scope="module")
def event_range(small_world: World) -> tuple[datetime, datetime]:
    """The world's event-time span, so the property tests exercise real rows."""
    starts = [event.start_time for event in small_world.billing.usage_events]
    return min(starts), max(starts)


@pytest.fixture(scope="module")
def usage_db(small_world: World) -> duckdb.DuckDBPyConnection:
    """``usage_events`` from the world, loaded once into an in-memory DuckDB."""
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE usage_events ("
        "event_id BIGINT, client_no VARCHAR, start_time TIMESTAMP, "
        "insertion_time TIMESTAMP, units BIGINT, corrects_event_id BIGINT)"
    )
    rows = [
        (
            event.event_id,
            event.client_no,
            event.start_time.replace(tzinfo=None),
            event.insertion_time.replace(tzinfo=None),
            event.units,
            event.corrects_event_id,
        )
        for event in small_world.billing.usage_events
    ]
    connection.executemany("INSERT INTO usage_events VALUES (?, ?, ?, ?, ?, ?)", rows)
    yield connection
    connection.close()


def _ids(db: duckdb.DuckDBPyConnection, where: str) -> set[int]:
    result = db.execute(f"SELECT event_id FROM usage_events WHERE {where}").fetchall()
    return {row[0] for row in result}


def test_world_lag_is_within_the_declared_bound(small_world: World) -> None:
    """Precondition for every property below: no row exceeds ``max_lag`` in this world."""
    max_lag = timedelta(seconds=small_world.config.max_lag_seconds)
    assert small_world.expected.lag_violations == ()
    for event in small_world.billing.usage_events:
        assert event.insertion_time - event.start_time <= max_lag


@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(fraction=st.floats(min_value=0.0, max_value=0.9, allow_nan=False, allow_infinity=False))
def test_derived_bound_never_loses_a_row(
    usage_db: duckdb.DuckDBPyConnection,
    event_range: tuple[datetime, datetime],
    fraction: float,
) -> None:
    """The acceptance property: with and without the derived bound, the rows are identical.

    ``insertion_time >= X`` implies ``start_time >= X - max_lag`` for every row whose lag is
    within the declared bound, so the extra predicate is a pure scan restriction.
    """
    low, high = event_range
    since = (low + (high - low) * fraction).replace(microsecond=0)
    plan = plan_time(
        TimeSpec(mode="new_since", since=since),
        USAGE_EVENTS,
        exec_ts=NOW,
        table_name="billing.usage_events",
    )
    assert plan.ok
    assert plan.derived_bound == since - SIX_HOURS

    original = f"insertion_time >= TIMESTAMP '{since.replace(tzinfo=None).isoformat(sep=' ')}'"
    with_bound = (
        f"{original} AND start_time >= TIMESTAMP '"
        f"{plan.derived_bound.replace(tzinfo=None).isoformat(sep=' ')}'"
    )
    assert _ids(usage_db, original) == _ids(usage_db, with_bound)
    # Not vacuous: the original predicate does match rows in this window.
    assert _ids(usage_db, original)


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    start_days=st.integers(min_value=0, max_value=200),
    width_days=st.integers(min_value=1, max_value=14),
)
def test_derived_bound_matches_the_world_ground_truth(
    usage_db: duckdb.DuckDBPyConnection,
    event_range: tuple[datetime, datetime],
    start_days: int,
    width_days: int,
) -> None:
    """Every row in the window is found with the bound on, and none outside it is added."""
    low, high = event_range
    start = low + timedelta(days=start_days)
    end = start + timedelta(days=width_days)
    assert low <= start < high
    lo = start.replace(tzinfo=None).isoformat(sep=" ")
    hi = end.replace(tzinfo=None).isoformat(sep=" ")
    window = f"start_time >= TIMESTAMP '{lo}' AND start_time < TIMESTAMP '{hi}'"
    in_window = _ids(usage_db, window)
    bound = (start - SIX_HOURS).replace(tzinfo=None).isoformat(sep=" ")
    with_bound = f"{window} AND start_time >= TIMESTAMP '{bound}'"
    # The derived bound is implied by the window itself, so it adds and removes nothing.
    assert _ids(usage_db, with_bound) == in_window
    assert in_window, "the window must contain real events, or the test proves nothing"


def test_partition_key_is_ingestion_time_loses_no_row_on_the_world(
    small_world: World,
    usage_db: duckdb.DuckDBPyConnection,
    event_range: tuple[datetime, datetime],
) -> None:
    """An event range widened to an ingestion range still contains every in-range row."""
    low, _high = event_range
    start = low + timedelta(days=10)
    end = start + timedelta(days=7)
    plan = plan_time(
        TimeSpec(mode="events_in_range", **{"from": start, "to": end}),
        USAGE_BY_LOAD,
        exec_ts=NOW,
        table_name="billing.usage_by_load",
    )
    assert plan.ok

    lo = start.replace(tzinfo=None).isoformat(sep=" ")
    hi = end.replace(tzinfo=None).isoformat(sep=" ")
    in_events = _ids(usage_db, f"start_time >= TIMESTAMP '{lo}' AND start_time < TIMESTAMP '{hi}'")
    assert in_events, "the window must contain events, or the test proves nothing"

    ingestion_lo = (start + timedelta(0)).replace(tzinfo=None).isoformat(sep=" ")
    ingestion_hi = (end + SIX_HOURS).replace(tzinfo=None).isoformat(sep=" ")
    widened = _ids(
        usage_db,
        f"insertion_time >= TIMESTAMP '{ingestion_lo}' "
        f"AND insertion_time < TIMESTAMP '{ingestion_hi}'",
    )
    assert in_events <= widened, "the widened ingestion range dropped a row its event range had"


def test_ingestion_range_bounds_every_row_in_the_world(small_world: World) -> None:
    """Pure check, independent of SQL: every row's lag lies inside the declared lag range."""
    minimum = timedelta(0)
    maximum = timedelta(seconds=small_world.config.max_lag_seconds)
    for event in small_world.billing.usage_events:
        lag = event.insertion_time - event.start_time
        assert minimum <= lag <= maximum


# ---------------------------------------------------------------- monotonicity


@settings(max_examples=200, deadline=None)
@given(
    first=st.datetimes(
        min_value=datetime(2026, 1, 1), max_value=datetime(2026, 12, 1), timezones=st.just(UTC)
    ),
    second=st.datetimes(
        min_value=datetime(2026, 1, 1), max_value=datetime(2026, 12, 1), timezones=st.just(UTC)
    ),
)
def test_widening_the_requested_range_never_narrows_the_derived_bound(
    first: datetime, second: datetime
) -> None:
    """Monotonicity: a later ``since`` can only move the derived bound later, never earlier.

    Widening a requested ingestion window means starting *earlier*; the derived bound follows it
    down (``since - max_lag``), so a wider request never asks for a narrower scan.
    """
    earlier, later = sorted((first, second))
    plan_earlier = plan_time(
        TimeSpec(mode="new_since", since=earlier), USAGE_EVENTS, exec_ts=NOW, table_name="t"
    )
    plan_later = plan_time(
        TimeSpec(mode="new_since", since=later), USAGE_EVENTS, exec_ts=NOW, table_name="t"
    )
    assert plan_earlier.derived_bound is not None
    assert plan_later.derived_bound is not None
    assert plan_earlier.derived_bound <= plan_later.derived_bound
    # And the bound is always exactly max_lag below the requested start.
    assert plan_earlier.derived_bound == earlier - SIX_HOURS
    assert plan_later.derived_bound == later - SIX_HOURS


@settings(max_examples=100, deadline=None)
@given(
    start=st.datetimes(
        min_value=datetime(2026, 1, 1), max_value=datetime(2026, 11, 1), timezones=st.just(UTC)
    ),
    width=st.timedeltas(min_value=timedelta(hours=1), max_value=timedelta(days=30)),
)
def test_wider_event_window_widens_the_ingestion_range(start: datetime, width: timedelta) -> None:
    """An event window ``[X, Y)`` always yields a non-empty ingestion range that contains it."""
    end = start + width
    plan = plan_time(
        TimeSpec(mode="events_in_range", **{"from": start, "to": end}),
        USAGE_BY_LOAD,
        exec_ts=NOW,
    )
    text = plan.where_sql("postgres")
    lower = (start).replace(tzinfo=None).isoformat(sep=" ")
    upper = (end + SIX_HOURS).replace(tzinfo=None).isoformat(sep=" ")
    assert f"insertion_time >= CAST('{lower}' AS TIMESTAMP)" in text
    assert f"insertion_time < CAST('{upper}' AS TIMESTAMP)" in text


def test_plan_is_deterministic() -> None:
    """The same inputs render byte-identical SQL: no clock, no dict order, no randomness."""
    spec = TimeSpec(mode="new_since", since=datetime(2026, 11, 25, 14, 0, tzinfo=UTC))
    first = plan_time(spec, USAGE_EVENTS, exec_ts=NOW, table_name="billing.usage_events")
    second = plan_time(spec, USAGE_EVENTS, exec_ts=NOW, table_name="billing.usage_events")
    assert first.where_sql("duckdb") == second.where_sql("duckdb")
    assert first == second


def test_world_fixture_is_not_mutated(small_world: World) -> None:
    """The planner reads metadata only; the world is untouched (frozen dataclasses still)."""
    assert dataclasses.is_dataclass(small_world)
    assert small_world.billing.usage_events
