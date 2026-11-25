"""Tests for the cache freshness rule (C12).

Time is either an explicit ``age`` or an injected :class:`FixedClock`: nothing sleeps here.
Table-driven cases cover the documented examples and the boundaries; the hypothesis properties
cover the invariants the cache relies on (monotone in age, wider range never less fresh, an
unclassified table never cached).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mdqm.cache import (
    DEFAULT_MAX_TTL,
    DEFAULT_MIN_TTL,
    Clock,
    SystemClock,
    is_fresh,
    is_fresh_at,
    ttl_for,
)

MICROSECOND = timedelta(microseconds=1)
DAY = timedelta(days=1)
YEAR = timedelta(days=365)
END = datetime(2026, 3, 1, tzinfo=UTC)


class FixedClock:
    """Time under test control: no sleeping, no wall clock."""

    def __init__(self, now: datetime) -> None:
        self.current = now

    def now(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta) -> None:
        self.current += delta


def window(span: timedelta) -> tuple[datetime, datetime]:
    """A source-time window of ``span``, ending at ``END``."""
    return END - span, END


# --- acceptance: the worked example from the design ------------------------------------------


@pytest.mark.parametrize(
    ("span", "age", "expected"),
    [
        (YEAR, DAY, True),  # 1-year pull from yesterday: ttl 3.65 days
        (timedelta(days=7), DAY, False),  # 1-week pull from yesterday: ttl 1.68 hours
    ],
)
def test_design_example_year_versus_week_from_yesterday(
    span: timedelta, age: timedelta, expected: bool
) -> None:
    assert is_fresh(age, *window(span)) is expected


def test_design_defaults_are_the_documented_ones() -> None:
    assert timedelta(seconds=30) == DEFAULT_MIN_TTL
    assert timedelta(days=7) == DEFAULT_MAX_TTL
    assert ttl_for(YEAR) == timedelta(days=3, hours=15, minutes=36)  # 1% of 365 days


# --- acceptance: the min_ttl floor ----------------------------------------------------------


@pytest.mark.parametrize(
    ("span", "age", "expected"),
    [
        (timedelta(0), DEFAULT_MIN_TTL, True),  # exactly the floor: still fresh (<=)
        (timedelta(0), DEFAULT_MIN_TTL + MICROSECOND, False),
        (timedelta(seconds=1), timedelta(seconds=29), True),  # 1% of 1s is 10ms; floor wins
        (timedelta(seconds=1), timedelta(seconds=31), False),
        (timedelta(minutes=10), timedelta(seconds=29), True),  # 1% of 10min is 6s; floor wins
        (timedelta(minutes=10), timedelta(seconds=30), True),
        (timedelta(minutes=10), timedelta(seconds=31), False),
    ],
)
def test_min_ttl_is_the_floor_for_narrow_pulls(
    span: timedelta, age: timedelta, expected: bool
) -> None:
    assert is_fresh(age, *window(span)) is expected


# --- the max_ttl ceiling --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("span", "age", "expected"),
    [
        (timedelta(days=3650), DEFAULT_MAX_TTL, True),  # 1% is 36.5 days; ceiling wins
        (timedelta(days=3650), DEFAULT_MAX_TTL + MICROSECOND, False),
        (timedelta(days=3650), timedelta(days=3), True),
        (timedelta(days=3650), timedelta(days=30), False),  # 36.5 days uncapped, but 7 days rules
    ],
)
def test_max_ttl_caps_wide_pulls(span: timedelta, age: timedelta, expected: bool) -> None:
    assert is_fresh(age, *window(span)) is expected


# --- acceptance: exact boundaries (<= is fresh, < would not be) ------------------------------


@pytest.mark.parametrize(
    ("span", "expected_ttl"),
    [
        (timedelta(days=30), timedelta(days=30) * 0.01),  # ratio rule: 7.2h
        (timedelta(0), DEFAULT_MIN_TTL),  # floor
        (timedelta(days=3650), DEFAULT_MAX_TTL),  # ceiling
    ],
)
def test_ttl_boundary_is_inclusive(span: timedelta, expected_ttl: timedelta) -> None:
    assert ttl_for(span) == expected_ttl
    assert is_fresh(expected_ttl, *window(span)) is True
    assert is_fresh(expected_ttl + MICROSECOND, *window(span)) is False


# --- mutability policy ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutability", "short_ttl", "age", "expected"),
    [
        ("append_only", None, timedelta(hours=1), True),
        ("append_rarely", None, timedelta(hours=1), True),
        ("mutable_critical", None, timedelta(0), False),  # not even a just-fetched segment
        ("unknown", None, timedelta(0), False),
        ("mutable_critical", timedelta(seconds=60), timedelta(seconds=60), True),  # short TTL
        ("mutable_critical", timedelta(seconds=60), timedelta(seconds=61), False),
        ("unknown", timedelta(minutes=5), timedelta(minutes=1), True),
        ("mutable_critical", timedelta(days=30), DEFAULT_MAX_TTL, True),  # still capped by max_ttl
        ("mutable_critical", timedelta(days=30), DEFAULT_MAX_TTL + MICROSECOND, False),
    ],
)
def test_mutability_policy(
    mutability: str, short_ttl: timedelta | None, age: timedelta, expected: bool
) -> None:
    # A 1-year window gives a generous ratio TTL, so a False here comes from mutability alone.
    assert (
        is_fresh(age, *window(YEAR), mutability=mutability, short_ttl=short_ttl)  # type: ignore[arg-type]
        is expected
    )


# --- sealed (closed) partition segments -----------------------------------------------------


@pytest.mark.parametrize(
    ("age", "sealed", "verified", "mutability", "expected"),
    [
        (DAY, True, True, "append_only", True),  # ratio rule ignored: 1 week pull, 1 day old
        (DEFAULT_MAX_TTL, True, True, "append_rarely", True),
        (DEFAULT_MAX_TTL + MICROSECOND, True, True, "append_only", False),
        (timedelta(0), True, False, "append_only", False),  # nightly check saw a change
        (timedelta(0), True, True, "mutable_critical", False),  # mutability still wins
        (DAY, False, True, "append_only", False),  # the same segment, not sealed
    ],
)
def test_sealed_segments_are_bounded_only_by_max_ttl(
    age: timedelta, sealed: bool, verified: bool, mutability: str, expected: bool
) -> None:
    assert (
        is_fresh(
            age,
            *window(timedelta(days=7)),
            sealed=sealed,
            verified=verified,
            mutability=mutability,  # type: ignore[arg-type]
        )
        is expected
    )


# --- rejected inputs: a bad policy must fail loudly, never cache silently --------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"age": -timedelta(seconds=1)}, "age must not be negative"),
        ({"theta": 0.0}, "theta must be in"),
        ({"theta": 1.5}, "theta must be in"),
        ({"min_ttl": -timedelta(seconds=1)}, "min_ttl must not be negative"),
        ({"max_ttl": -timedelta(seconds=1)}, "max_ttl must not be negative"),
        ({"short_ttl": -timedelta(seconds=1)}, "short_ttl must not be negative"),
        ({"mutability": "sometimes"}, "unknown mutability class"),
    ],
)
def test_bad_policy_raises(kwargs: dict[str, object], message: str) -> None:
    call = {"age": timedelta(0), **kwargs}
    with pytest.raises(ValueError, match=message):
        is_fresh(call.pop("age"), *window(YEAR), **call)  # type: ignore[arg-type]


def test_inverted_range_raises() -> None:
    with pytest.raises(ValueError, match="is before range_from"):
        is_fresh(timedelta(0), END, END - DAY)


def test_mixed_aware_and_naive_range_raises() -> None:
    with pytest.raises(ValueError, match="both be aware or both naive"):
        is_fresh(timedelta(0), datetime(2026, 1, 1), END)


def test_ttl_for_rejects_a_bad_policy() -> None:
    with pytest.raises(ValueError, match="theta must be in"):
        ttl_for(YEAR, theta=0.0)


# --- clock-injected entry point: a cache hit carries fetched_at ------------------------------


def test_fetched_at_is_enough_to_decide_without_sleeping() -> None:
    fixed = FixedClock(END)
    clock: Clock = fixed
    fetched_at = END - DAY
    assert is_fresh_at(fetched_at, clock, *window(YEAR)) is True

    fixed.advance(timedelta(days=3))  # 4 days old now; the 1-year TTL is 3.65 days
    assert is_fresh_at(fetched_at, clock, *window(YEAR)) is False


def test_fetched_at_in_the_future_is_treated_as_stale() -> None:
    clock = FixedClock(END)
    assert is_fresh_at(END + MICROSECOND, clock, *window(YEAR)) is False


def test_naive_timestamps_are_rejected() -> None:
    clock = FixedClock(END)
    with pytest.raises(ValueError, match="fetched_at must be timezone-aware"):
        is_fresh_at(datetime(2026, 1, 1), clock, *window(YEAR))

    naive_clock = FixedClock(datetime(2026, 3, 1))
    with pytest.raises(ValueError, match=r"clock\.now\(\) must be timezone-aware"):
        is_fresh_at(END - DAY, naive_clock, *window(YEAR))


def test_system_clock_is_aware_utc() -> None:
    now = SystemClock().now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


# --- acceptance: hypothesis properties ------------------------------------------------------

ages = st.timedeltas(min_value=timedelta(0), max_value=timedelta(days=3650))
spans = st.timedeltas(min_value=timedelta(0), max_value=timedelta(days=3650))
ttls = st.timedeltas(min_value=timedelta(0), max_value=timedelta(days=3650))
thetas = st.floats(min_value=1e-6, max_value=1.0)


def fresh(
    age: timedelta,
    span: tuple[datetime, datetime],
    theta: float,
    min_ttl: timedelta,
    max_ttl: timedelta,
) -> bool:
    """``is_fresh`` for a hypothesis-generated policy, so the properties stay readable."""
    return is_fresh(age, span[0], span[1], theta=theta, min_ttl=min_ttl, max_ttl=max_ttl)


@given(span=spans, age=ages, older=ages, theta=thetas, min_ttl=ttls, max_ttl=ttls)
def test_freshness_is_monotone_in_age(
    span: timedelta,
    age: timedelta,
    older: timedelta,
    theta: float,
    min_ttl: timedelta,
    max_ttl: timedelta,
) -> None:
    """An older segment is never fresher than a younger one."""
    source_range = window(span)
    younger_fresh = fresh(age, source_range, theta, min_ttl, max_ttl)
    older_fresh = fresh(age + older, source_range, theta, min_ttl, max_ttl)
    assert not older_fresh or younger_fresh


@given(span=spans, extra=spans, age=ages, theta=thetas, min_ttl=ttls, max_ttl=ttls)
def test_a_wider_range_is_never_less_fresh(
    span: timedelta,
    extra: timedelta,
    age: timedelta,
    theta: float,
    min_ttl: timedelta,
    max_ttl: timedelta,
) -> None:
    """Widening the pull only raises the ratio TTL, so freshness can only improve."""
    narrow = window(span)
    wide = (narrow[1] - (span + extra), narrow[1])
    if fresh(age, narrow, theta, min_ttl, max_ttl):
        assert fresh(age, wide, theta, min_ttl, max_ttl)


@given(age=ages, span=spans, theta=thetas, min_ttl=ttls, max_ttl=ttls)
def test_unclassified_tables_are_never_fresh(
    age: timedelta, span: timedelta, theta: float, min_ttl: timedelta, max_ttl: timedelta
) -> None:
    range_from, range_to = window(span)
    for mutability in ("mutable_critical", "unknown"):
        assert (
            is_fresh(
                age,
                range_from,
                range_to,
                theta=theta,
                min_ttl=min_ttl,
                max_ttl=max_ttl,
                mutability=mutability,
            )
            is False
        )


@given(span=spans, theta=thetas, min_ttl=ttls, max_ttl=ttls)
def test_the_ttl_boundary_is_exactly_where_ttl_for_says(
    span: timedelta, theta: float, min_ttl: timedelta, max_ttl: timedelta
) -> None:
    """``ttl_for`` and ``is_fresh`` cannot drift apart: it is the reported boundary."""
    source_range = window(span)
    ttl = ttl_for(span, theta=theta, min_ttl=min_ttl, max_ttl=max_ttl)
    assert fresh(ttl, source_range, theta, min_ttl, max_ttl) is True
    assert fresh(ttl + MICROSECOND, source_range, theta, min_ttl, max_ttl) is False


@given(span=spans, theta=thetas, min_ttl=ttls, max_ttl=ttls)
def test_a_just_fetched_segment_is_always_fresh(
    span: timedelta, theta: float, min_ttl: timedelta, max_ttl: timedelta
) -> None:
    """The cache is usable at all: age 0 must be fresh for a cacheable class."""
    assert fresh(timedelta(0), window(span), theta, min_ttl, max_ttl) is True
