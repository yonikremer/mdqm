"""Catalog snapshot: tables, columns, indexes and keys as the probe found them.

Types are carried twice: ``native_type`` exactly as the source reports it (audit, debugging) and a
small normalised vocabulary (portable joins, conversions, drift comparison).
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from mdqm.models.base import MDQMModel, UtcDatetime


class NormalizedType(StrEnum):
    """Portable column type. Anything a source reports that has no home lands in ``OTHER``."""

    STRING = "string"
    INTEGER = "integer"
    DECIMAL = "decimal"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    TIMESTAMP = "timestamp"
    TIME = "time"
    BINARY = "binary"
    JSON = "json"
    ARRAY = "array"
    UUID = "uuid"
    OTHER = "other"


class TableKind(StrEnum):
    """What the "table" physically is. ES index families and REST endpoints are tables."""

    TABLE = "table"
    PARTITIONED_TABLE = "partitioned_table"
    VIEW = "view"
    MATERIALIZED_VIEW = "materialized_view"
    INDEX = "index"
    DATA_STREAM = "data_stream"
    ENDPOINT = "endpoint"
    OTHER = "other"


class Column(MDQMModel):
    """One column: native type for audit, normalised type for portable logic."""

    name: str = Field(min_length=1)
    native_type: str = Field(min_length=1)
    normalized_type: NormalizedType
    nullable: bool = True
    comment: str | None = None
    ordinal: int | None = Field(default=None, ge=1)


class Index(MDQMModel):
    """A source index. For an API source, the query parameters map onto this shape."""

    name: str = Field(min_length=1)
    columns: tuple[str, ...] = Field(min_length=1)
    unique: bool = False
    primary: bool = False
    index_type: str | None = None
    predicate: str | None = None


class ForeignKey(MDQMModel):
    """A declared foreign key: the only join shape that is auto-approved without a human."""

    columns: tuple[str, ...] = Field(min_length=1)
    ref_table: str = Field(min_length=1)
    ref_columns: tuple[str, ...] = Field(min_length=1)
    name: str | None = None

    @model_validator(mode="after")
    def _check_column_pairs(self) -> ForeignKey:
        if len(self.columns) != len(self.ref_columns):
            raise ValueError(
                "a foreign key maps column lists one to one: "
                f"{len(self.columns)} local columns vs {len(self.ref_columns)} referenced columns."
            )
        return self


class Table(MDQMModel):
    """One table in a source catalog."""

    name: str = Field(min_length=1)
    schema_name: str | None = None
    kind: TableKind = TableKind.TABLE
    columns: tuple[Column, ...] = ()
    indexes: tuple[Index, ...] = ()
    foreign_keys: tuple[ForeignKey, ...] = ()
    primary_key: tuple[str, ...] = ()
    row_estimate: int | None = Field(default=None, ge=0)
    comment: str | None = None

    @property
    def qualified_name(self) -> str:
        """``schema.name`` when a schema was reported, else ``name``."""
        return f"{self.schema_name}.{self.name}" if self.schema_name else self.name

    def column(self, name: str) -> Column | None:
        """Look a column up by exact name; ``None`` when it does not exist."""
        return next((c for c in self.columns if c.name == name), None)


class CatalogSnapshot(MDQMModel):
    """One source's catalog as extracted by the probe at ``captured_at``."""

    source: str = Field(min_length=1)
    captured_at: UtcDatetime
    tables: tuple[Table, ...] = ()

    def table(self, name: str) -> Table | None:
        """Look a table up by qualified or bare name; ``None`` when it does not exist."""
        return next(
            (t for t in self.tables if name in (t.name, t.qualified_name)),
            None,
        )


def _canonical(item: dict[str, Any]) -> str:
    return json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sorted(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order by canonical content, so input order can never change the hash."""
    return [json.loads(text) for text in sorted(_canonical(item) for item in items)]


def _table_content(table: Table) -> dict[str, Any]:
    return {
        "name": table.name,
        "schema": table.schema_name,
        "kind": table.kind.value,
        "comment": table.comment,
        "primary_key": sorted(table.primary_key),
        "columns": _sorted(
            [
                {
                    "name": c.name,
                    "native_type": c.native_type,
                    "normalized_type": c.normalized_type.value,
                    "nullable": c.nullable,
                    "comment": c.comment,
                }
                for c in table.columns
            ]
        ),
        "indexes": _sorted(
            [
                {
                    "name": i.name,
                    "columns": list(i.columns),
                    "unique": i.unique,
                    "primary": i.primary,
                    "index_type": i.index_type,
                    "predicate": i.predicate,
                }
                for i in table.indexes
            ]
        ),
        "foreign_keys": _sorted(
            [
                {
                    "name": fk.name,
                    "columns": list(fk.columns),
                    "ref_table": fk.ref_table,
                    "ref_columns": list(fk.ref_columns),
                }
                for fk in table.foreign_keys
            ]
        ),
    }


def snapshot_hash(snapshot: CatalogSnapshot) -> str:
    """Stable content hash of a catalog snapshot (sha256 hex).

    Used for schema drift detection and incremental pipeline stages. Order independent, and blind
    to what is not structure: ``captured_at`` (else every run would differ), ``row_estimate`` and
    column ``ordinal`` (statistics churn belongs to :class:`TableStats`, not to drift).
    """
    payload = {
        "source": snapshot.source,
        "tables": _sorted([_table_content(t) for t in snapshot.tables]),
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
