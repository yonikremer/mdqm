"""Stage D1+D2: the catalog and its partitioning, written as generated semantic facts.

What this stage is allowed to claim, and why it stops there:

* it records what the source *says* about itself -- tables, columns with native and normalised
  types, primary keys, indexes, and the real partition bounds (A03 parses them from the server);
* it does **not** record time-column roles (`event_time` vs `ingestion_time`), mutability, measures
  or value maps. Those are conclusions, not catalog facts: the first two are classified from data
  (S05) and the rest are proposed, validated and approved (D7-D10, the review queue). A guess
  written here would be indistinguishable from an approved fact;
* it writes to ``generated/`` only, through :meth:`PipelineContext.write_generated`, which buffers
  and commits atomically: a stage that fails half way leaves the semantic tree untouched.

Idempotence is the runner's contract (input hash + output present => ``skipped``), and this stage
makes it honest by hashing the catalog it just read: the catalog *is* the input, so a hash built
from anything else could let a changed schema be skipped.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final, Protocol

from mdqm.adapters.base import NotSupported
from mdqm.models import CatalogSnapshot, Partitioning, Table
from mdqm.pipeline.core import PipelineContext, StageReport, StageStatus

STAGE_NAME: Final = "D1+D2"
SECTION: Final = "catalog"
"""``generated/catalog.yaml``: one file, because the loader merges every generated file into one
document and a per-source split would collide on the same top-level ``tables`` key."""


class CatalogReader(Protocol):
    """What this stage needs from a source: its catalog, and its partitions if it declares any.

    Narrower than :class:`~mdqm.adapters.base.SourceAdapter` on purpose: a stage that only reads
    metadata should not require its input to be able to execute queries, and a test double should
    not have to implement `execute` to prove what the stage writes.
    """

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub: pyright requires an ellipsis body
    def extract_catalog(self) -> CatalogSnapshot:
        """Tables, columns, keys and indexes as the source reports them."""
        ...

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub: pyright requires an ellipsis body
    def extract_partitions(self) -> list[Partitioning]:
        """Partition metadata, or raise :class:`NotSupported` when the source has none to give."""
        ...


def qualified_name(table: Table, source: str) -> str:
    """The semantic key: ``<source>.<table>``.

    The namespace is the *source*, not the database schema. ``crm.customers`` and
    ``billing.accounts`` are two tables of one organisation living in two servers; keying them by
    ``public.customers`` would collide the moment two sources both use a ``public`` schema, and it
    would not match the keys the curated layer and the query specs already use.
    """
    return f"{source}.{table.name}"


def partition_owner(bound_name: str, table_names: Sequence[str]) -> str | None:
    """The table a partition bound name belongs to, or ``None`` when it is ambiguous.

    ``Partitioning`` (F02) carries no table name -- the protocol returns one flat list -- so the
    only link is the bound's relation name: ``usage_events_p2025_12`` belongs to ``usage_events``.
    The longest matching table name wins, so ``customers_hist`` cannot be claimed by ``customers``.
    """
    best: str | None = None
    for name in table_names:
        matches = bound_name == name or bound_name.startswith(f"{name}_")
        if matches and (best is None or len(name) > len(best)):
            best = name
    return best


class CatalogStage:
    """Stage D1+D2: read the adapters, write ``generated/catalog.yaml``."""

    name: str = STAGE_NAME
    heavy: bool = True

    def __init__(
        self,
        adapters: Mapping[str, CatalogReader],
        *,
        tables: Sequence[str] | None = None,
    ) -> None:
        self._adapters: dict[str, CatalogReader] = dict(adapters)
        self._only = frozenset(tables) if tables else None
        self._snapshots: dict[str, CatalogSnapshot] = {}
        self._partitions: dict[str, tuple[Partitioning, ...]] = {}

    # -- inputs --------------------------------------------------------------------------

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def inputs_hash(self, ctx: PipelineContext) -> str:
        """sha256 over the catalogs and partitions actually read, so a schema change cannot skip.

        The read happens here, once: :meth:`run` reuses what this cached, so the hash and the work
        cannot disagree about what the input was.
        """
        self._snapshots.clear()
        self._partitions.clear()
        parts: list[Any] = [STAGE_NAME]
        for source in self._selected(ctx):
            adapter = self._adapters[source]
            snapshot = adapter.extract_catalog()
            self._snapshots[source] = snapshot
            try:
                partitions = tuple(adapter.extract_partitions())
            except NotSupported:
                partitions = ()
            self._partitions[source] = partitions
            parts.append(
                {
                    "source": source,
                    "catalog": _catalog_digest(snapshot),
                    "partitions": [p.model_dump(mode="json") for p in partitions],
                }
            )
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()

    # -- output --------------------------------------------------------------------------

    def run(self, ctx: PipelineContext) -> StageReport:
        """Build the tables payload from the catalogs and buffer it for the runner to commit."""
        counts: dict[str, int] = {}
        skipped: list[str] = []
        fixes: list[str] = []
        tables: dict[str, Any] = {}

        for source in self._selected(ctx):
            snapshot = self._snapshots.get(source) or self._adapters[source].extract_catalog()
            partitions = self._partitions.get(source, ())
            by_table = _partitions_by_table(partitions, snapshot, source)
            if not partitions:
                skipped.append(f"{source}: no partition metadata (unsupported or none declared)")

            for table in snapshot.tables:
                name = qualified_name(table, source)
                entry: dict[str, Any] = {
                    "key": list(table.primary_key),
                    "columns": [column.model_dump(mode="json") for column in table.columns],
                    "indexes": [index.model_dump(mode="json") for index in table.indexes],
                }
                partitioning = by_table.get(name)
                if partitioning is not None:
                    entry["partitioning"] = partitioning.model_dump(mode="json", exclude_none=True)
                tables[name] = entry

            counts[f"{source}.tables"] = len(snapshot.tables)
            counts[f"{source}.columns"] = sum(len(table.columns) for table in snapshot.tables)
            counts[f"{source}.partitioned_tables"] = len(by_table)
            counts[f"{source}.partitions"] = sum(len(p.bounds) for p in by_table.values())

        if not tables:
            skipped.append("no tables were reported by any source")
        ctx.write_generated(SECTION, {"version": 1, "tables": tables})
        return StageReport(
            stage=STAGE_NAME,
            status=StageStatus.RAN,
            counts=counts,
            skipped=tuple(skipped),
            applied_fixes=tuple(fixes),
        )

    # -- helpers -------------------------------------------------------------------------

    def _selected(self, ctx: PipelineContext) -> list[str]:
        chosen = [source for source in self.sources if ctx.source is None or ctx.source == source]
        if self._only is not None:
            chosen = [source for source in chosen if source in self._only]
        return chosen


def _catalog_digest(snapshot: CatalogSnapshot) -> str:
    """A stable digest of the schema: no ``captured_at``, no row estimates, sorted by table name."""
    payload = [
        table.model_dump(mode="json", exclude={"row_estimate", "comment"})
        for table in sorted(snapshot.tables, key=lambda t: t.name)
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _partitions_by_table(
    partitions: Sequence[Partitioning], snapshot: CatalogSnapshot, source: str
) -> dict[str, Partitioning]:
    """Map each ``Partitioning`` to its table by bound name, dropping anything ambiguous.

    A partition whose bounds match no table (or match several equally well) is reported by its
    absence: attaching it to the wrong table would put a wrong partition key into the semantic
    layer, which is worse than a missing one.
    """
    # Partition bounds carry the *unqualified* relation name (``events_p2025_12``) while the catalog
    # keys tables by ``schema.table``. Map through the bare name, and only when it is unambiguous:
    # two schemas holding the same table name make the owner a guess, and a guess is worse than a
    # missing entry. Anything that matches no table is also dropped, never attached to a neighbour.
    by_bare: dict[str, str] = {}
    ambiguous: set[str] = set()
    for table in snapshot.tables:
        if table.name in by_bare:
            ambiguous.add(table.name)
        by_bare[table.name] = qualified_name(table, source)
    for name in ambiguous:
        by_bare.pop(name, None)

    grouped: dict[str, list[Any]] = {}
    for partitioning in partitions:
        if not partitioning.bounds:
            continue
        owners: set[str] = set()
        for bound in partitioning.bounds:
            if not bound.name:
                continue
            bare = partition_owner(bound.name, list(by_bare))
            if bare is not None:
                owners.add(by_bare[bare])
        if len(owners) != 1:
            continue
        (owner,) = owners
        grouped.setdefault(owner, []).append(partitioning)
    result: dict[str, Partitioning] = {}
    for owner, items in grouped.items():
        # A table has exactly one partition scheme; two would mean a self-contradictory catalog.
        if len(items) == 1:
            result[owner] = items[0]
    return result
