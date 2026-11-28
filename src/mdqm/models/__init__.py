"""Shared frozen domain models.

Every lane imports its wire format from here: ``Capabilities`` from the probe, ``CatalogSnapshot``
and ``Partitioning`` from the adapters, ``Fragment`` from the compiler, ``PlanSummary`` from the
explain gate, ``Limits`` from the governor, ``ErrorResponse`` to the agent.
"""

from __future__ import annotations

from mdqm.models.base import JsonObject, MDQMModel, UtcDatetime
from mdqm.models.capabilities import Capabilities, Capability, CapabilitySource
from mdqm.models.catalog import (
    CatalogSnapshot,
    Column,
    ForeignKey,
    Index,
    NormalizedType,
    Table,
    TableKind,
    snapshot_hash,
)
from mdqm.models.errors import ErrorResponse
from mdqm.models.fragment import Fragment
from mdqm.models.limits import Limits
from mdqm.models.partitioning import (
    PartitionBound,
    PartitionGranularity,
    Partitioning,
    PartitionStrategy,
)
from mdqm.models.plan import PlanSummary, PruningProof
from mdqm.models.query import RawQuery
from mdqm.models.schema import PUBLIC_MODELS, export_json_schemas, schema_file_name
from mdqm.models.sizes import PartitionSize, TableSize
from mdqm.models.stats import TableStats

__all__ = [
    "PUBLIC_MODELS",
    "Capabilities",
    "Capability",
    "CapabilitySource",
    "CatalogSnapshot",
    "Column",
    "ErrorResponse",
    "ForeignKey",
    "Fragment",
    "Index",
    "JsonObject",
    "Limits",
    "MDQMModel",
    "NormalizedType",
    "PartitionBound",
    "PartitionGranularity",
    "PartitionSize",
    "PartitionStrategy",
    "Partitioning",
    "PlanSummary",
    "PruningProof",
    "RawQuery",
    "Table",
    "TableKind",
    "TableSize",
    "TableStats",
    "UtcDatetime",
    "export_json_schemas",
    "schema_file_name",
    "snapshot_hash",
]
