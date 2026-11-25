"""Cache layer: freshness rules for stored segments and single-flight coalescing (DESIGN 8).

``freshness`` is pure logic (no I/O, injected clock); ``singleflight`` is async and only removes
duplicate concurrent work. Segments and DuckDB storage arrive with their own tasks.
"""

from __future__ import annotations

from mdqm.cache.freshness import (
    CACHEABLE_MUTABILITY,
    DEFAULT_MAX_TTL,
    DEFAULT_MIN_TTL,
    DEFAULT_THETA,
    MUTABILITY_CLASSES,
    Clock,
    Mutability,
    SystemClock,
    is_fresh,
    is_fresh_at,
    ttl_for,
)
from mdqm.cache.singleflight import KEY_VERSION, SingleFlight, flight_key

__all__ = [
    "CACHEABLE_MUTABILITY",
    "DEFAULT_MAX_TTL",
    "DEFAULT_MIN_TTL",
    "DEFAULT_THETA",
    "KEY_VERSION",
    "MUTABILITY_CLASSES",
    "Clock",
    "Mutability",
    "SingleFlight",
    "SystemClock",
    "flight_key",
    "is_fresh",
    "is_fresh_at",
    "ttl_for",
]
