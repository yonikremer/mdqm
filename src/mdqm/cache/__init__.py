"""Cache layer: freshness rules, single-flight coalescing and the DuckDB segment store (DESIGN 8).

``freshness`` is pure logic (no I/O, injected clock); ``singleflight`` is async and only removes
duplicate concurrent work; ``segments`` (C14) stores partition-aligned result pieces in DuckDB and
refreshes the still-filling tail instead of trusting it.
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
from mdqm.cache.segments import (
    EvictionReport,
    SegmentCache,
    SegmentLookup,
    SegmentPart,
    SegmentSpec,
    SegmentStatus,
    TailFetch,
    TailPolicy,
    segment_key,
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
    "EvictionReport",
    "Mutability",
    "SegmentCache",
    "SegmentLookup",
    "SegmentPart",
    "SegmentSpec",
    "SegmentStatus",
    "SingleFlight",
    "SystemClock",
    "TailFetch",
    "TailPolicy",
    "flight_key",
    "is_fresh",
    "is_fresh_at",
    "segment_key",
    "ttl_for",
]
