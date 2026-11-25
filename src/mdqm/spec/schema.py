"""JSON Schema for the query spec, with the organisation's names filled in at runtime.

The static schema is useless to a weak agent ("from" could be any string). :func:`spec_schema`
takes the semantic layer's vocabulary and turns the open string fields into enums, so the enum
*is* the documentation and a wrong table name is impossible to send.

The result is also usable as a structured-output ``response_format``: it stays valid JSON Schema,
every object is closed (``additionalProperties: false``), it has no ``$ref`` cycle, and it uses only
the keywords in :data:`ALLOWED_KEYWORDS` (pydantic's extra constraints are dropped, because an
endpoint that does not know ``minLength`` may reject the whole schema, and the server validates
those rules anyway).
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import Field

from mdqm.models.base import MDQMModel
from mdqm.spec.models import QuerySpec

#: Keywords a structured-output consumer understands. Anything else is stripped on export, so the
#: caller can pass the result to an endpoint that only knows this subset.
ALLOWED_KEYWORDS: frozenset[str] = frozenset(
    {
        "$defs",
        "$ref",
        "additionalProperties",
        "anyOf",
        "description",
        "enum",
        "items",
        "properties",
        "required",
        "title",
        "type",
    }
)

_KEYWORD_MAPS = frozenset({"properties", "$defs"})
_KEYWORD_LISTS = frozenset({"anyOf"})


class SemanticVocabulary(MDQMModel):
    """What :func:`spec_schema` needs from the semantic layer: the names it can enumerate.

    Kept tiny on purpose: the schema only needs *names*, and the semantic layer (C04) owns them.
    """

    tables: tuple[str, ...] = Field(default=(), description="Queryable table names.")
    measures: tuple[str, ...] = Field(default=(), description="Canonical measure names.")
    joins: tuple[str, ...] = Field(default=(), description="Approved join names.")


def vocabulary_from_semantic(semantic: Mapping[str, Any]) -> SemanticVocabulary:
    """Read the vocabulary from a loaded semantic layer (its YAML/JSON mapping form).

    Tables and measures are the mapping keys; an approved join is either a mapping with ``name``
    or a bare name. Names are sorted so the schema is byte-stable across runs.
    """
    tables = semantic.get("tables") or {}
    measures = semantic.get("measures") or {}
    joins = semantic.get("approved_joins") or ()
    if not isinstance(tables, Mapping) or not isinstance(measures, Mapping):
        raise ValueError("the semantic layer must hold `tables` and `measures` as mappings")
    if isinstance(joins, (str, bytes)) or not isinstance(joins, Sequence):
        raise ValueError("`approved_joins` must be a list")
    names: list[str] = []
    for join in joins:
        if isinstance(join, Mapping):
            name = join.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"an approved join has no name: {join!r}")
            names.append(name)
        elif isinstance(join, str) and join:
            names.append(join)
        else:
            raise ValueError(f"an approved join must be a mapping or a name, got {join!r}")
    return SemanticVocabulary(
        tables=tuple(sorted(str(name) for name in tables)),
        measures=tuple(sorted(str(name) for name in measures)),
        joins=tuple(sorted(names)),
    )


def spec_schema(semantic: SemanticVocabulary | Mapping[str, Any]) -> dict[str, Any]:
    """The query spec's JSON Schema with table/measure/join enums taken from ``semantic``.

    ``semantic`` is either a :class:`SemanticVocabulary` or a loaded semantic layer mapping (as
    :func:`vocabulary_from_semantic` reads it). An empty vocabulary leaves that field a plain
    string, which is what a spec validated without a semantic layer can honestly offer.
    """
    vocabulary = (
        semantic if isinstance(semantic, SemanticVocabulary) else vocabulary_from_semantic(semantic)
    )
    schema = copy.deepcopy(QuerySpec.model_json_schema())

    def _enum_field(properties: dict[str, Any], field: str, names: Sequence[str]) -> None:
        """Replace an open string field with an enum, keeping it a string."""
        if names:
            properties[field] = {"type": "string", "enum": list(names)}

    _enum_field(schema["properties"], "from", vocabulary.tables)
    _enum_field(_definitions(schema)["MeasureSelect"]["properties"], "measure", vocabulary.measures)
    if vocabulary.joins:
        schema["properties"]["joins"]["items"] = {
            "type": "string",
            "enum": list(vocabulary.joins),
        }
    _apply(schema, _drop_unsupported_keywords)
    _apply(schema, _close_objects)
    return schema


def _definitions(schema: Mapping[str, Any]) -> dict[str, Any]:
    """``$defs`` of a pydantic schema; fails loudly if pydantic stopped emitting them."""
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        raise KeyError(
            "the generated schema has no $defs; spec_schema() patches them by model name"
        )
    return definitions


def _apply(schema: dict[str, Any], visit: Callable[[dict[str, Any]], None]) -> None:
    """Run ``visit`` on every subschema of ``schema``."""
    _walk_schemas(schema, visit)


def _walk_schemas(node: Any, visit: Callable[[dict[str, Any]], None]) -> None:
    """Depth-first walk over subschemas only: a property *name* is not a keyword."""
    if isinstance(node, list):
        for item in node:
            _walk_schemas(item, visit)
        return
    if not isinstance(node, dict):
        return
    visit(node)
    for keyword, value in node.items():
        if keyword in _KEYWORD_MAPS and isinstance(value, dict):
            for subschema in value.values():
                _walk_schemas(subschema, visit)
        elif keyword in _KEYWORD_LISTS or keyword == "items":
            _walk_schemas(value, visit)


def _drop_unsupported_keywords(node: dict[str, Any]) -> None:
    for keyword in [name for name in node if name not in ALLOWED_KEYWORDS]:
        del node[keyword]


def _close_objects(node: dict[str, Any]) -> None:
    """An object schema must say ``additionalProperties: false``: unknown fields are rejected."""
    if "properties" in node:
        node["additionalProperties"] = False


def schema_keywords(schema: Mapping[str, Any]) -> frozenset[str]:
    """Every keyword used anywhere in a schema (the test compares this with ALLOWED_KEYWORDS)."""
    found: set[str] = set()

    def collect(node: dict[str, Any]) -> None:
        found.update(node)

    _walk_schemas(dict(schema), collect)
    return frozenset(found)


def ref_cycles(schema: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    """``$ref`` cycles, each as the chain of definitions that closes it (empty when there are none).

    A structured-output consumer rejects a recursive schema, so the tests assert this is empty.
    """
    definitions = _definitions(schema)
    graph: dict[str, set[str]] = {
        name: _refs_in(definition) for name, definition in definitions.items()
    }
    graph["(root)"] = _refs_in({key: value for key, value in schema.items() if key != "$defs"})
    cycles: list[tuple[str, ...]] = []

    def walk(name: str, trail: tuple[str, ...]) -> None:
        for next_name in sorted(graph.get(name, ())):
            if next_name in trail:
                cycles.append((*trail, next_name))
            elif next_name in graph:
                walk(next_name, (*trail, next_name))

    walk("(root)", ("(root)",))
    return tuple(cycles)


def _refs_in(node: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()

    def collect(subschema: dict[str, Any]) -> None:
        reference = subschema.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            names.add(reference.removeprefix("#/$defs/"))

    _walk_schemas(dict(node), collect)
    return names
