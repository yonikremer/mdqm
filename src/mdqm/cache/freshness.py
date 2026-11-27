"""Cache freshness: may a stored segment be reused, or must it be fetched again?

Pure logic: no I/O, no sleeping, and no global clock. The decision takes an ``age``; the one
convenience wrapper (:func:`is_fresh_at`) reads "now" from an injected :class:`Clock`, so tests
pin time instead of waiting for it.

The rule (``docs/DESIGN.md`` section 8)::

    fresh <=> age <= max(min_ttl, theta * (range_to - range_from))   and   age <= max_ttl
    age    = now - fetched_at

``min_ttl`` is a floor for narrow pulls: without it a one-minute query would be cached for
0.6 seconds and the cache would do nothing. ``max_ttl`` is the ceiling for wide pulls. The
comparison is inclusive (``age <= ttl``): a segment old enough to be exactly at its TTL is still
served, one microsecond older is not. Restating it: the TTL is ``min(max(min_ttl, theta * span),
max_ttl)``.

Mutability is measured, never declared (DESIGN section 8). ``append_only`` and ``append_rarely``
use the ratio rule; ``mutable_critical`` and ``unknown`` are never served from cache unless the
table has an explicit short TTL. A sealed (closed) partition segment cannot gain rows any more, so
it is bounded only by ``max_ttl`` and the nightly verification flag.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol, get_args

# Mutability classes from DESIGN section 8. A new class here means a new policy below.
Mutability = Literal["append_only", "append_rarely", "mutable_critical", "unknown"]
MUTABILITY_CLASSES: tuple[Mutability, ...] = get_args(Mutability)

# Classes whose segments may be served from cache on the ratio rule.
CACHEABLE_MUTABILITY: tuple[Mutability, ...] = ("append_only", "append_rarely")

DEFAULT_THETA = 0.01
DEFAULT_MIN_TTL = timedelta(seconds=30)
DEFAULT_MAX_TTL = timedelta(days=7)


class Clock(Protocol):
    """Time source, injected so tests never sleep and callers never read the wall clock."""

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub: pyright requires an ellipsis body
    def now(self) -> datetime:
        """Current instant, timezone-aware (UTC in production)."""
        ...


class SystemClock:
    """The production clock: UTC, aware."""

    def now(self) -> datetime:
        return datetime.now(UTC)


def _check_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            f"{name} must be timezone-aware (UTC internally), got a naive datetime {value!r}"
        )


def _check_policy(
    *,
    theta: float,
    min_ttl: timedelta,
    max_ttl: timedelta,
    short_ttl: timedelta | None,
) -> None:
    """Reject policy values that would silently cache too much, naming the offending one."""
    if not 0.0 < theta <= 1.0:
        raise ValueError(f"theta must be in (0, 1], got {theta!r} (it is a fraction of the range)")
    if min_ttl < timedelta(0):
        raise ValueError(f"min_ttl must not be negative, got {min_ttl}")
    if max_ttl < timedelta(0):
        raise ValueError(f"max_ttl must not be negative, got {max_ttl}")
    if short_ttl is not None and short_ttl < timedelta(0):
        raise ValueError(f"short_ttl must not be negative, got {short_ttl}")


def ttl_for(
    span: timedelta,
    *,
    theta: float = DEFAULT_THETA,
    min_ttl: timedelta = DEFAULT_MIN_TTL,
    max_ttl: timedelta = DEFAULT_MAX_TTL,
) -> timedelta:
    """The cache TTL for a pull covering ``span`` of source time: ``min(max(min_ttl, theta*span),
    max_ttl)``.

    Callers report this next to ``fetched_at`` so a refetch is explainable ("age 1d > ttl 1.68h").
    """
    _check_policy(theta=theta, min_ttl=min_ttl, max_ttl=max_ttl, short_ttl=None)
    return min(max(min_ttl, span * theta), max_ttl)


def is_fresh(
    age: timedelta,
    range_from: datetime,
    range_to: datetime,
    *,
    theta: float = DEFAULT_THETA,
    min_ttl: timedelta = DEFAULT_MIN_TTL,
    max_ttl: timedelta = DEFAULT_MAX_TTL,
    mutability: Mutability = "append_only",
    short_ttl: timedelta | None = None,
    sealed: bool = False,
    verified: bool = True,
) -> bool:
    """True when a segment of ``age`` may be served instead of fetched again.

    ``range_from``/``range_to`` are the source-time window the segment covers; only their
    difference matters, and they must be consistent (both aware or both naive).

    Boundaries are inclusive: ``age == ttl`` is fresh, ``age == ttl + 1µs`` is not.

    ``mutable_critical`` and ``unknown`` segments are never fresh on the ratio rule; only an
    explicit ``short_ttl`` (per table) allows caching, and even then ``max_ttl`` still caps it.
    ``sealed=True`` marks a closed partition segment: the ratio rule no longer applies, the
    segment lives until ``max_ttl`` unless the nightly check saw a change (``verified=False``).

    A negative ``age`` (a ``fetched_at`` in the future) is a caller bug, so it raises; the cache
    read path uses :func:`is_fresh_at`, which treats that case as stale.
    """
    if age < timedelta(0):
        raise ValueError(f"age must not be negative, got {age}: fetched_at is in the future")
    if (range_from.tzinfo is None) != (range_to.tzinfo is None):
        raise ValueError("range_from and range_to must both be aware or both naive, not mixed")
    if range_to < range_from:
        raise ValueError(f"range_to {range_to} is before range_from {range_from}")
    if mutability not in MUTABILITY_CLASSES:
        raise ValueError(
            f"unknown mutability class {mutability!r}: use one of {', '.join(MUTABILITY_CLASSES)}"
        )
    _check_policy(theta=theta, min_ttl=min_ttl, max_ttl=max_ttl, short_ttl=short_ttl)

    if mutability not in CACHEABLE_MUTABILITY:
        return short_ttl is not None and age <= min(short_ttl, max_ttl)
    if sealed:
        return age <= max_ttl and verified
    effective_ttl = ttl_for(range_to - range_from, theta=theta, min_ttl=min_ttl, max_ttl=max_ttl)
    return age <= effective_ttl


def is_fresh_at(
    fetched_at: datetime,
    clock: Clock,
    range_from: datetime,
    range_to: datetime,
    *,
    theta: float = DEFAULT_THETA,
    min_ttl: timedelta = DEFAULT_MIN_TTL,
    max_ttl: timedelta = DEFAULT_MAX_TTL,
    mutability: Mutability = "append_only",
    short_ttl: timedelta | None = None,
    sealed: bool = False,
    verified: bool = True,
) -> bool:
    """Convenience wrapper: :func:`is_fresh` with ``age = clock.now() - fetched_at``.

    ``fetched_at`` and the clock are both timezone-aware. A ``fetched_at`` in the future is clock
    skew or corrupt metadata, and the segment is treated as stale: refetching is cheap, serving a
    segment whose provenance we cannot trust is not.
    """
    _check_aware(fetched_at, "fetched_at")
    now = clock.now()
    _check_aware(now, "clock.now()")
    if fetched_at > now:
        return False
    return is_fresh(
        now - fetched_at,
        range_from,
        range_to,
        theta=theta,
        min_ttl=min_ttl,
        max_ttl=max_ttl,
        mutability=mutability,
        short_ttl=short_ttl,
        sealed=sealed,
        verified=verified,
    )
