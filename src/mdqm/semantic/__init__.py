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
from mdqm.semantic.measures import (
    AppliedNormalization,
    MultiplicityReport,
    UndeclaredRounding,
    check_multiples,
    check_normalize_template,
    normalizations_for,
    render_normalize,
    round_to_multiple,
)
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
    ValueMap,
    check_sql_template,
    compute_version_hash,
    render_template,
)
from mdqm.semantic.text import normalize_label
from mdqm.semantic.values import (
    MAX_SUGGESTIONS,
    Resolution,
    ResolutionKind,
    ReverseResult,
    SourceLiteral,
    resolve_value,
    reverse_value,
)
from mdqm.semantic.writer import update_curated, write_generated

__all__ = [
    "CURATED_DIR",
    "GENERATED_DIR",
    "MAX_SUGGESTIONS",
    "AppliedNormalization",
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
    "MultiplicityReport",
    "MutabilityDecl",
    "Provenance",
    "Resolution",
    "ResolutionKind",
    "ReverseResult",
    "RoundingMode",
    "RoundingRule",
    "SemanticError",
    "SemanticLayer",
    "SourceDecl",
    "SourceLiteral",
    "TableSemantics",
    "TimeColumn",
    "TimeRole",
    "UndeclaredRounding",
    "ValueMap",
    "check_multiples",
    "check_normalize_template",
    "check_sql_template",
    "compute_version_hash",
    "load",
    "normalizations_for",
    "normalize_label",
    "parse_duration",
    "render_normalize",
    "render_template",
    "resolve_value",
    "reverse_value",
    "round_to_multiple",
    "update_curated",
    "write_generated",
]
