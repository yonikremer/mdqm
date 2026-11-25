"""JSON Schema export for the shared models.

One file per public model, deterministic bytes (so a regeneration shows a real diff), written with
``pathlib`` and UTF-8 so it behaves the same on Windows, Linux and macOS.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from mdqm.models.base import MDQMModel
from mdqm.models.capabilities import Capabilities, Capability
from mdqm.models.catalog import CatalogSnapshot, Column, ForeignKey, Index, Table
from mdqm.models.errors import ErrorResponse
from mdqm.models.fragment import Fragment
from mdqm.models.limits import Limits
from mdqm.models.partitioning import PartitionBound, Partitioning
from mdqm.models.plan import PlanSummary
from mdqm.models.query import RawQuery
from mdqm.models.stats import TableStats

#: Every public model, in export order. Enums are string vocabularies and get no file of their own.
PUBLIC_MODELS: tuple[type[MDQMModel], ...] = (
    Capability,
    Capabilities,
    Column,
    Index,
    ForeignKey,
    Table,
    CatalogSnapshot,
    Partitioning,
    PartitionBound,
    TableStats,
    RawQuery,
    PlanSummary,
    Limits,
    Fragment,
    ErrorResponse,
)

_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def schema_file_name(model: type[MDQMModel]) -> str:
    """``CatalogSnapshot`` -> ``catalog_snapshot.json``."""
    return f"{_CAMEL_BOUNDARY.sub('_', model.__name__).lower()}.json"


def export_json_schemas(directory: str | Path) -> Sequence[Path]:
    """Write one ``<model>.json`` JSON Schema per public model into ``directory``.

    Returns the written paths in model order. Existing files are overwritten; the bytes depend only
    on the models, so an unchanged model set produces an unchanged directory.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model in PUBLIC_MODELS:
        schema = model.model_json_schema()
        text = json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        path = target / schema_file_name(model)
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written
