"""``mdqm.semantic``: the business layer an agent queries instead of raw SQL.

One immutable :class:`SemanticLayer` is loaded from YAML that the bootstrap pipeline writes into
``generated/`` and humans write into ``curated/`` (curated wins, merged per key). It carries the
content hash that goes into cache keys and audit, so a changed meaning can never be served from an
old answer.
"""

from __future__ import annotations

from mdqm.semantic.durations import Duration, parse_duration
from mdqm.semantic.errors import SemanticError
from mdqm.semantic.loader import CURATED_DIR, GENERATED_DIR, load
from mdqm.semantic.models import (
    ApprovalStatus,
    ApprovedJoin,
    CacheDefaults,
    CacheOverride,
    Defaults,
    FrozenMapping,
    JoinCardinality,
    JoinUsage,
    Lag,
    Measure,
    MeasureCanonical,
    MeasureSource,
    MutabilityDecl,
    Provenance,
    RoundingMode,
    RoundingRule,
    SemanticLayer,
    SourceDecl,
    TableSemantics,
    TimeColumn,
    TimeRole,
    check_sql_template,
    compute_version_hash,
    render_template,
)
from mdqm.semantic.text import normalize_label
from mdqm.semantic.writer import update_curated, write_generated

__all__ = [
    "CURATED_DIR",
    "GENERATED_DIR",
    "ApprovalStatus",
    "ApprovedJoin",
    "CacheDefaults",
    "CacheOverride",
    "Defaults",
    "Duration",
    "FrozenMapping",
    "JoinCardinality",
    "JoinUsage",
    "Lag",
    "Measure",
    "MeasureCanonical",
    "MeasureSource",
    "MutabilityDecl",
    "Provenance",
    "RoundingMode",
    "RoundingRule",
    "SemanticError",
    "SemanticLayer",
    "SourceDecl",
    "TableSemantics",
    "TimeColumn",
    "TimeRole",
    "check_sql_template",
    "compute_version_hash",
    "load",
    "normalize_label",
    "parse_duration",
    "render_template",
    "update_curated",
    "write_generated",
]
