"""Stage D1+D2: what it records, what it deliberately refuses to record, and when it skips.

The property under test is *honesty of the generated facts*: the stage writes what the catalog and
the server's own partition metadata say, never a conclusion it cannot observe (time-column roles,
mutability, measures). Those belong to later stages and to the human approval gate, and a guess
written here would be indistinguishable from an approved fact.
"""

from __future__ import annotations

from pathlib import Path

from ruamel.yaml import YAML

from mdqm.models import (
    CatalogSnapshot,
    Column,
    Index,
    NormalizedType,
    PartitionBound,
    PartitionGranularity,
    Partitioning,
    PartitionStrategy,
    Table,
    TableKind,
)
from mdqm.pipeline.core import PipelineContext
from mdqm.pipeline.stages.catalog import (
    SECTION,
    CatalogStage,
    partition_owner,
    qualified_name,
)
from mdqm.store import Store


def _table(name: str, *, schema: str = "public", key: tuple[str, ...] = ()) -> Table:
    return Table(
        name=name,
        schema_name=schema,
        kind=TableKind.TABLE,
        columns=(
            Column(name="id", native_type="integer", normalized_type=NormalizedType.INTEGER),
            Column(
                name="created_at",
                native_type="timestamptz",
                normalized_type=NormalizedType.TIMESTAMP,
            ),
        ),
        indexes=(Index(name=f"{name}_pkey", columns=("id",), unique=True),),
        foreign_keys=(),
        primary_key=key or ("id",),
    )


class _StubAdapter:
    """The smallest thing that looks like a SourceAdapter for this stage."""

    kind = "stub"

    def __init__(
        self, tables: tuple[Table, ...], partitions: tuple[Partitioning, ...] = ()
    ) -> None:
        self._snapshot = CatalogSnapshot(
            source="stub", captured_at=__import__("datetime").datetime.now(
                __import__("datetime").UTC
            ), tables=tables
        )
        self._partitions = partitions
        self.calls = 0

    def extract_catalog(self) -> CatalogSnapshot:
        self.calls += 1
        return self._snapshot

    def extract_partitions(self) -> list[Partitioning]:
        return list(self._partitions)

    def dialect(self) -> str:  # pragma: no cover - not used by the stage
        return "postgres"


def _context(tmp_path: Path) -> PipelineContext:
    return PipelineContext(
        store=Store(tmp_path / "metadata.sqlite"),
        exec_ts=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").UTC),
        semantic_root=tmp_path / "semantic",
    )


def test_partition_owner_prefers_the_longest_table_name() -> None:
    """`customers_hist` must not be claimed by `customers`: a prefix is not an owner."""
    names = ["customers", "customers_hist", "usage_events"]
    assert partition_owner("customers_hist_p0", names) == "customers_hist"
    assert partition_owner("customers_p0", names) == "customers"
    assert partition_owner("usage_events_p2025_12", names) == "usage_events"
    assert partition_owner("something_else_p0", names) is None


def test_stage_records_columns_indexes_and_partitioning(tmp_path: Path) -> None:
    partitioning = Partitioning(
        strategy=PartitionStrategy.RANGE,
        key=("created_at",),
        granularity=PartitionGranularity.MONTH,
        bounds=(
            PartitionBound(name="events_p2025_12", row_estimate=10),
            PartitionBound(name="events_p2026_01", row_estimate=0),
        ),
    )
    adapter = _StubAdapter((_table("events"),), (partitioning,))
    stage = CatalogStage({"crm": adapter})
    context = _context(tmp_path)

    digest = stage.inputs_hash(context)
    report = stage.run(context)
    context.commit()  # the runner does this; the stage only buffers

    written = YAML(typ="safe").load(
        (tmp_path / "semantic" / "generated" / f"{SECTION}.yaml").read_text(encoding="utf-8")
    )
    entry = written["tables"]["crm.events"]
    assert [column["name"] for column in entry["columns"]] == ["id", "created_at"]
    assert entry["key"] == ["id"]
    assert entry["indexes"][0]["name"] == "events_pkey"
    assert entry["partitioning"]["key"] == ["created_at"]
    assert entry["partitioning"]["granularity"] == "month"
    assert len(entry["partitioning"]["bounds"]) == 2
    assert "mutability" not in entry, "mutability is classified by S05, not assumed here"
    assert "time_columns" not in entry, "time roles are conclusions, not catalog facts"
    assert report.counts["crm.partitions"] == 2
    assert digest and len(digest) == 64


def test_the_same_catalog_gives_the_same_input_hash(tmp_path: Path) -> None:
    """Idempotence depends on this: an unchanged catalog must skip, a changed one must not."""
    adapter = _StubAdapter((_table("events"),))
    stage = CatalogStage({"crm": adapter})
    context = _context(tmp_path)
    first = stage.inputs_hash(context)

    fresh = CatalogStage({"crm": _StubAdapter((_table("events"),))})
    assert fresh.inputs_hash(context) == first

    changed = CatalogStage({"crm": _StubAdapter((_table("events"), _table("extra")))}) 
    assert changed.inputs_hash(context) != first


def test_a_partition_with_no_matching_table_is_not_attached(tmp_path: Path) -> None:
    """Better a missing partition than one attached to the wrong table."""
    orphan = Partitioning(
        strategy=PartitionStrategy.RANGE,
        key=("created_at",),
        granularity=PartitionGranularity.MONTH,
        bounds=(PartitionBound(name="ghost_p2025_12"),),
    )
    adapter = _StubAdapter((_table("events"),), (orphan,))
    stage = CatalogStage({"crm": adapter})
    context = _context(tmp_path)
    stage.inputs_hash(context)
    report = stage.run(context)
    context.commit()

    written = YAML(typ="safe").load(
        (tmp_path / "semantic" / "generated" / f"{SECTION}.yaml").read_text(encoding="utf-8")
    )
    assert "partitioning" not in written["tables"]["crm.events"]
    assert report.counts["crm.partitioned_tables"] == 0


def test_qualified_name_handles_a_schemaless_source() -> None:
    assert qualified_name(_table("events"), "crm") == "crm.events"
    assert qualified_name(_table("events", schema="other"), "erp") == "erp.events"
