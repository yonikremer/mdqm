"""Cache layer: freshness rules for stored segments (DESIGN section 8).

``freshness`` is pure logic (no I/O, injected clock); segments, single-flight and DuckDB storage
arrive with their own tasks.
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

__all__ = [
    "CACHEABLE_MUTABILITY",
    "DEFAULT_MAX_TTL",
    "DEFAULT_MIN_TTL",
    "DEFAULT_THETA",
    "MUTABILITY_CLASSES",
    "Clock",
    "Mutability",
    "SystemClock",
    "is_fresh",
    "is_fresh_at",
    "ttl_for",
]
