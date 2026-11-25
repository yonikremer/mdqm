"""The structured query spec: models, the duration parser, validation and its JSON Schema.

An agent sends a spec instead of SQL (docs/DESIGN.md section 5). Everything here is pure logic: no
source, no file, no clock.

``parse_spec`` is the entry point; ``spec_schema(semantic)`` is what the MCP tool advertises.
"""

from __future__ import annotations

from mdqm.spec.durations import Duration, parse_duration
from mdqm.spec.errors import SpecError, SpecIssue
from mdqm.spec.models import (
    Agg,
    Direction,
    Filter,
    FilterOp,
    MeasureSelect,
    OrderBy,
    QuerySpec,
    TimeMode,
    TimeSpec,
)
from mdqm.spec.schema import SemanticVocabulary, spec_schema, vocabulary_from_semantic
from mdqm.spec.validate import check_spec, parse_spec

__all__ = [
    "Agg",
    "Direction",
    "Duration",
    "Filter",
    "FilterOp",
    "MeasureSelect",
    "OrderBy",
    "QuerySpec",
    "SemanticVocabulary",
    "SpecError",
    "SpecIssue",
    "TimeMode",
    "TimeSpec",
    "check_spec",
    "parse_duration",
    "parse_spec",
    "spec_schema",
    "vocabulary_from_semantic",
]
