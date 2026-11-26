"""Stage D10: schema drift. Code, not a model (task S09).

A source's catalog changes under the semantic layer's feet: a column disappears, a type is
redefined, an index that made the partition bound cheap is dropped, a table is re-partitioned. Any
of those silently breaks a measure's normalisation, an approved join, or the planner's pruning
assumption -- the failure DESIGN.md section 2 names as one this product exists to prevent.

This module answers one question, deterministically: **what changed between two catalog snapshots,
and what depends on it?** It is not a second comparison engine: change detection is gated on
:func:`~mdqm.models.catalog.snapshot_hash` (F02), the sha256 over canonical, content-sorted
structure that already excludes ``captured_at``, ``row_estimate`` and column ``ordinal``. Equal
hashes therefore mean "statistics churn, not drift", and this module returns an empty report without
looking further -- a nightly statistics refresh can never look like a schema change.

What a reader gets:

* :class:`DriftReport` -- every change, classified breaking or not, sorted so two runs on the same
  inputs produce identical output. Nothing is applied silently (principle 10): a non-breaking change
  is still listed, with the exact ``fix`` a weak agent needs.
* :attr:`DriftReport.refs` -- the semantic YAML entries each change reaches, as
  ``<section>.<key>[.<field>]`` (``measures.credit_limit_r5000``,
  ``tables.billing.usage_events.time_columns.start_time``). This mapping is what D10 exists for.
* :attr:`DriftReport.blocked` -- the entries an **unapproved breaking** change makes unusable. A
  dropped column a measure reads blocks that measure; :func:`blocked_references` is how the linter
  or the server reads the same fact back from the store, so an unapproved destructive change cannot
  be quietly absorbed.
* :class:`AdminNotification` -- the immediate payload for the admin (DESIGN.md section 10: breaking
  changes do not wait for the weekly digest). The stage stores it; delivery is another task's job,
  so nothing here sends mail or touches a network.

Decisions that are not obvious from the code:

* **A rename is a labelled guess, never a silent one.** A column removed and a column added in the
  same table with the same portable shape (and unknown-or-equal ordinals) is reported as
  ``column_renamed`` -- still breaking -- only when the pairing is unambiguous: exactly one removed
  and one added. With two candidates it degrades to ``column_removed`` + ``column_added``, because
  the wrong pairing would send a human to the wrong measure.
* **Nullability is breaking in both directions.** A column becoming nullable changes what an
  aggregate means (a join key that starts holding NULLs silently drops rows); becoming ``NOT NULL``
  changes what a filter means. "Conservative" means report it and let a human decide.
* **A dropped index is breaking only when it carries an access guarantee** -- it covers a
  ``use_by_default``/``indexed`` time column or a partition key. Any other dropped index is
  reported as non-breaking: it is a cost fact, and calling it breaking would train the reader to
  ignore the report.
* **The baseline does not advance while breaking changes are undecided.** That is what "never
  silently absorbed" means mechanically: the next run compares against the same baseline and reports
  the same drift until a human decides, instead of absorbing it into "last night's catalog".
* **Breaking changes write no ``generated/`` section.** They are a question for a human, so they go
  to :class:`~mdqm.review.queue.ReviewQueue` (S12) as a proposal on the affected table, and approval
  records the acknowledgement in ``curated/``. Inventing a ``generated/`` section is not an option:
  the semantic loader validates the merged document and an unknown top-level key makes the whole
  layer unloadable.
* **The baseline lives in the metadata store** (``mdqm_meta``, migration 0001): one document per
  source, no new file, no new table.

Deterministic: no clock read (``ctx.exec_ts`` is injected), no ``hash()``, sorted output.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from pydantic import Field

from mdqm.models.base import MDQMModel, UtcDatetime
from mdqm.models.catalog import CatalogSnapshot, Column, Index, Table, snapshot_hash
from mdqm.models.partitioning import Partitioning
from mdqm.pipeline.core import PipelineContext, StageReport, StageStatus, hash_inputs
from mdqm.review.queue import (
    Proposal,
    ProposalKind,
    ProposalStatus,
    ReviewError,
    ReviewQueue,
    SubmitAction,
    proposal_id,
)
from mdqm.semantic.errors import SemanticError
from mdqm.semantic.loader import load
from mdqm.semantic.models import SemanticLayer
from mdqm.store.db import Store

__all__ = [
    "BASELINE_PREFIX",
    "BLOCKED_PREFIX",
    "NOTIFICATION_PREFIX",
    "PARTITIONS_PREFIX",
    "STAGE_NAME",
    "AdminNotification",
    "ChangeKind",
    "DriftChange",
    "DriftReport",
    "DriftStage",
    "SemanticRef",
    "Severity",
    "blocked_references",
    "diff_snapshots",
    "read_baseline",
    "read_notification",
    "semantic_index",
    "write_baseline",
]

logger = logging.getLogger(__name__)

#: The stage's name in the pipeline ledger and in ``--stage``.
STAGE_NAME: Final[str] = "D10"

#: ``mdqm_meta`` key prefix holding one source's last accepted catalog (JSON).
BASELINE_PREFIX: Final[str] = "catalog_snapshot"

#: ``mdqm_meta`` key prefix holding the partition metadata that came with that catalog (JSON).
PARTITIONS_PREFIX: Final[str] = "catalog_partitions"

#: ``mdqm_meta`` key prefix holding the semantic entries an undecided drift blocks.
BLOCKED_PREFIX: Final[str] = "drift_blocked"

#: ``mdqm_meta`` key prefix holding the admin notification waiting to be delivered.
NOTIFICATION_PREFIX: Final[str] = "drift_notification"

#: How many evidence lines one proposal carries: enough to decide, short enough to read.
EVIDENCE_LIMIT: Final[int] = 12

#: Who a drift proposal says proposed it (the queue stores it verbatim in ``proposed_by``).
PROPOSED_BY: Final[str] = "pipeline/D10"


def _statement(*parts: str) -> str:
    """Assemble one SQL statement from module-level constants only (never from caller input)."""
    return "".join(parts)  # noqa: S608 - fragments are constants; values are bound parameters


_SELECT_META: Final[str] = _statement("SELECT value FROM mdqm_meta WHERE key = ?")
_SELECT_META_LIKE: Final[str] = _statement(
    "SELECT 'key' AS marker, value FROM mdqm_meta WHERE key LIKE ?"
)
_UPSERT_META: Final[str] = _statement(
    "INSERT INTO mdqm_meta (key, value, updated_at) VALUES (?, ?, ?)"
    " ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at"
)


class Severity(StrEnum):
    """Whether a change can silently change an answer, or is only a fact worth reporting."""

    BREAKING = "breaking"
    NON_BREAKING = "non_breaking"


class ChangeKind(StrEnum):
    """One kind of catalog change. Every kind a source can produce has a name here."""

    TABLE_ADDED = "table_added"
    TABLE_REMOVED = "table_removed"
    TABLE_KIND_CHANGED = "table_kind_changed"
    COLUMN_ADDED = "column_added"
    COLUMN_REMOVED = "column_removed"
    COLUMN_RENAMED = "column_renamed"
    COLUMN_TYPE_CHANGED = "column_type_changed"
    COLUMN_NULLABILITY_CHANGED = "column_nullability_changed"
    INDEX_ADDED = "index_added"
    INDEX_REMOVED = "index_removed"
    INDEX_CHANGED = "index_changed"
    PRIMARY_KEY_CHANGED = "primary_key_changed"
    FOREIGN_KEY_ADDED = "foreign_key_added"
    FOREIGN_KEY_REMOVED = "foreign_key_removed"
    PARTITION_ADDED = "partition_added"
    PARTITION_REMOVED = "partition_removed"
    PARTITION_STRATEGY_CHANGED = "partition_strategy_changed"
    PARTITION_KEY_CHANGED = "partition_key_changed"
    PARTITION_GRANULARITY_CHANGED = "partition_granularity_changed"


#: The kinds that can change what an answer means, i.e. that need a human before anything uses the
#: affected semantic entry again. Absent from this set: added tables/columns/partitions, a wider
#: native type, and an index that carries no access guarantee -- all reported, none blocking.
BREAKING_KINDS: Final[frozenset[ChangeKind]] = frozenset(
    {
        ChangeKind.TABLE_REMOVED,
        ChangeKind.TABLE_KIND_CHANGED,
        ChangeKind.COLUMN_REMOVED,
        ChangeKind.COLUMN_RENAMED,
        ChangeKind.COLUMN_TYPE_CHANGED,
        ChangeKind.COLUMN_NULLABILITY_CHANGED,
        ChangeKind.INDEX_CHANGED,
        ChangeKind.PRIMARY_KEY_CHANGED,
        ChangeKind.FOREIGN_KEY_REMOVED,
        ChangeKind.PARTITION_REMOVED,
        ChangeKind.PARTITION_STRATEGY_CHANGED,
        ChangeKind.PARTITION_KEY_CHANGED,
        ChangeKind.PARTITION_GRANULARITY_CHANGED,
    }
)


class SemanticRef(MDQMModel):
    """One semantic YAML entry: a section, the item inside it, and an optional field path.

    ``label`` is the identity a linter, a server or a human uses (``measures.credit_limit_r5000``).
    ``reference`` is what it depends on in the source (``billing.accounts.credit_limit``, or the
    ``source.table`` a table-level entry is about), so a reader sees the link without opening YAML.
    """

    section: str = Field(min_length=1)
    key: str = ""
    field: str = ""
    reference: str = ""

    @property
    def label(self) -> str:
        """``section.key`` or ``section.key.field``: the entry this change touches."""
        return ".".join(part for part in (self.section, self.key, self.field) if part)

    @property
    def sort_key(self) -> tuple[str, str, str, str]:
        return (self.section, self.key, self.field, self.reference)


class DriftChange(MDQMModel):
    """One change between two snapshots, and what in the semantic layer it reaches."""

    kind: ChangeKind
    severity: Severity
    table: str = Field(min_length=1, description="``source.table``, the name the YAML uses.")
    column: str | None = None
    index: str | None = None
    partition: str | None = None
    before: str | None = None
    after: str | None = None
    detail: str = ""
    refs: tuple[SemanticRef, ...] = ()
    fix: str = Field(default="", description="The exact next action, for a weak agent.")

    @property
    def breaking(self) -> bool:
        """True when this change can silently change an answer."""
        return self.severity is Severity.BREAKING

    @property
    def sort_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.kind.value,
            self.table,
            self.column or "",
            self.index or "",
            self.partition or "",
        )

    def describe(self) -> str:
        """One line a human reads: severity, what changed, on what, and the entries it reaches."""
        target = self.table
        if self.column:
            target += f".{self.column}"
        if self.index:
            target += f" index {self.index}"
        if self.partition:
            target += f" partition {self.partition}"
        arrow = ""
        if self.before is not None or self.after is not None:
            arrow = f": {self.before or 'absent'} -> {self.after or 'absent'}"
        refs = f" [{' '.join(ref.label for ref in self.refs)}]" if self.refs else ""
        return f"{self.severity.value} {self.kind.value} {target}{arrow}{refs}"


class AdminNotification(MDQMModel):
    """The payload an admin is told about immediately (not in the weekly digest)."""

    source: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    body: str
    blocked: tuple[str, ...] = ()
    created_at: UtcDatetime

    def payload(self) -> dict[str, Any]:
        """JSON-ready mapping with a stable shape, for the metadata store or a mail body."""
        return {
            "source": self.source,
            "subject": self.subject,
            "body": self.body,
            "blocked": list(self.blocked),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AdminNotification | None:
        """Rebuild from :meth:`payload`; ``None`` when the stored document does not fit."""
        try:
            return cls(
                source=str(payload["source"]),
                subject=str(payload["subject"]),
                body=str(payload["body"]),
                blocked=tuple(str(label) for label in payload.get("blocked", ())),
                created_at=str(payload["created_at"]),
            )
        except (KeyError, ValueError):
            return None


class DriftReport(MDQMModel):
    """What changed for one source: every change, the refs it reaches, and what it blocks."""

    source: str
    previous_hash: str
    current_hash: str
    changes: tuple[DriftChange, ...] = ()

    @property
    def changed(self) -> bool:
        """True when anything at all differs structurally."""
        return bool(self.changes)

    @property
    def breaking(self) -> tuple[DriftChange, ...]:
        """The changes that need a human before the affected entries are used again."""
        return tuple(change for change in self.changes if change.breaking)

    @property
    def non_breaking(self) -> tuple[DriftChange, ...]:
        """The changes that are pure facts: reported, blocking nothing."""
        return tuple(change for change in self.changes if not change.breaking)

    @property
    def refs(self) -> tuple[SemanticRef, ...]:
        """Every semantic entry any change touches, deduplicated and sorted."""
        return _sorted_refs(ref for change in self.changes for ref in change.refs)

    @property
    def blocked(self) -> tuple[SemanticRef, ...]:
        """The entries an **unapproved breaking** change makes unusable, deduplicated and sorted."""
        return _sorted_refs(ref for change in self.breaking for ref in change.refs)

    def is_blocked(self, label: str) -> bool:
        """True when ``label`` (or a field of it) is blocked by this report."""
        return any(label == ref.label or label.startswith(f"{ref.label}.") for ref in self.blocked)

    def notification(self, *, created_at: datetime) -> AdminNotification | None:
        """The immediate admin payload, or ``None`` when nothing breaking changed."""
        breaking = self.breaking
        if not breaking:
            return None
        tables = sorted({change.table for change in breaking})
        lines = [change.describe() for change in breaking]
        blocked = tuple(ref.label for ref in self.blocked)
        if blocked:
            lines.append("blocked until a human decides: " + ", ".join(blocked))
        lines.append("decide in `mdqm review list`; nothing was changed automatically.")
        return AdminNotification(
            source=self.source,
            subject=f"{self.source}: {len(breaking)} breaking catalog change(s) in "
            + ", ".join(tables),
            body="\n".join(lines),
            blocked=blocked,
            created_at=created_at,
        )

    def describe(self) -> str:
        """One line per change, breaking first: what an operator reads after a run."""
        if not self.changes:
            return f"{self.source}: no schema drift ({self.current_hash[:12]})"
        ordered = sorted(self.changes, key=_describe_key)
        lines = [change.describe() for change in ordered]
        lines.append(
            f"{self.source}: {len(self.breaking)} breaking, {len(self.non_breaking)} non-breaking"
        )
        return "\n".join(lines)


def _change_key(change: DriftChange) -> tuple[str, str, str, str, str]:
    """Order changes by kind and target: one ordering for every report this module writes."""
    return change.sort_key


def _describe_key(change: DriftChange) -> tuple[bool, tuple[str, str, str, str, str]]:
    """Breaking changes first, then by target, for the text an operator reads."""
    return (not change.breaking, change.sort_key)


def _sorted_refs(refs: Iterable[SemanticRef]) -> tuple[SemanticRef, ...]:
    """Deduplicate refs by label and sort them, so output bytes never depend on input order."""
    by_label: dict[str, SemanticRef] = {}
    for ref in refs:
        by_label.setdefault(ref.label, ref)
    return tuple(by_label[label] for label in sorted(by_label))


# --------------------------------------------------------------------------------------------------
# The semantic index: what depends on which source column
# --------------------------------------------------------------------------------------------------


def _split_reference(reference: str) -> tuple[str, str] | None:
    """``source.table.column`` -> ``(source.table, column)``; ``None`` when it is not 3 parts."""
    parts = reference.split(".")
    if len(parts) < 3 or not all(parts):
        return None
    return (".".join(parts[:-1]), parts[-1])


class _Index:
    """The semantic layer as two lookups: by source column, and by source table."""

    def __init__(self) -> None:
        self.by_column: dict[str, list[SemanticRef]] = {}
        self.by_table: dict[str, list[SemanticRef]] = {}

    def add_column(self, reference: str, ref: SemanticRef) -> None:
        self.by_column.setdefault(reference, []).append(ref)

    def add_table(self, table: str, ref: SemanticRef) -> None:
        self.by_table.setdefault(table, []).append(ref)

    def columns_of(self, table: str) -> list[SemanticRef]:
        """Every column-level ref of ``table``: what a table-level change invalidates too."""
        found: list[SemanticRef] = []
        for reference in sorted(self.by_column):
            split = _split_reference(reference)
            if split is not None and split[0] == table:
                found.extend(self.by_column[reference])
        return found


def semantic_index(layer: SemanticLayer | None, *, source: str) -> _Index:
    """Map every semantic entry onto the ``source.table[.column]`` it reads.

    ``source`` qualifies the bare table keys the YAML may use, so ``tables.usage_events`` and a
    measure's ``billing.usage_events.credit_limit`` land on the same ``billing.usage_events``.
    An absent layer yields an empty index: drift is then reported without a mapping, which is honest
    (nothing in the layer is known to depend on the change) and never an error.
    """
    index = _Index()
    if layer is None:
        return index
    for name in sorted(layer.measures):
        for reference in sorted(layer.measures[name].sources):
            index.add_column(
                reference, SemanticRef(section="measures", key=name, reference=reference)
            )
    for name in sorted(layer.value_maps):
        for reference in sorted(layer.value_maps[name].sources):
            index.add_column(
                reference, SemanticRef(section="value_maps", key=name, reference=reference)
            )
    for join in layer.approved_joins:
        for reference in (join.left, join.right):
            index.add_column(reference, SemanticRef(section="approved_joins", key=join.name))
    for key in sorted(layer.tables):
        table = _qualify(key, source)
        index.add_table(table, SemanticRef(section="tables", key=key, reference=table))
        semantics = layer.tables[key]
        for column in sorted(semantics.time_columns):
            index.add_column(
                f"{table}.{column}",
                SemanticRef(
                    section="tables", key=key, field=f"time_columns.{column}", reference=table
                ),
            )
        partitioning = semantics.partitioning
        if partitioning is not None:
            for column in sorted(partitioning.key):
                index.add_column(
                    f"{table}.{column}",
                    SemanticRef(
                        section="tables",
                        key=key,
                        field=f"partitioning.key.{column}",
                        reference=table,
                    ),
                )
        for column in sorted(semantics.key):
            index.add_column(
                f"{table}.{column}",
                SemanticRef(section="tables", key=key, field=f"key.{column}", reference=table),
            )
    return index


def _qualify(table_key: str, source: str) -> str:
    """``usage_events`` -> ``billing.usage_events``; an already qualified key is left alone."""
    return table_key if table_key.startswith(f"{source}.") else f"{source}.{table_key}"


def _guarded_columns(layer: SemanticLayer | None, *, source: str) -> frozenset[str]:
    """``source.table.column`` of every default time column and partition key column.

    These are the columns whose index the planner and the cache rely on; dropping that index is a
    breaking change, dropping any other index is not.
    """
    if layer is None:
        return frozenset()
    guarded: set[str] = set()
    for key in sorted(layer.tables):
        semantics = layer.tables[key]
        table = _qualify(key, source)
        for column, decl in sorted(semantics.time_columns.items()):
            if decl.use_by_default or decl.indexed:
                guarded.add(f"{table}.{column}")
        if semantics.partitioning is not None:
            for column in semantics.partitioning.key:
                guarded.add(f"{table}.{column}")
    return frozenset(guarded)


# --------------------------------------------------------------------------------------------------
# The diff
# --------------------------------------------------------------------------------------------------


def diff_snapshots(
    previous: CatalogSnapshot,
    current: CatalogSnapshot,
    *,
    layer: SemanticLayer | None = None,
    partitions_before: Mapping[str, Partitioning] | None = None,
    partitions_after: Mapping[str, Partitioning] | None = None,
) -> DriftReport:
    """Classify every structural change between two snapshots of one source.

    Gated on :func:`~mdqm.models.catalog.snapshot_hash`: equal structure (and equal partition
    metadata) short-circuits to an empty report, which is what makes statistics churn --
    ``captured_at``, ``row_estimate``, column ``ordinal`` -- structurally unable to look like drift.

    Partition metadata is not part of a :class:`CatalogSnapshot` (it comes from
    ``SourceAdapter.extract_partitions``), so it travels separately, keyed by table name; omitting
    it means no partition change is reported.
    """
    if previous.source != current.source:
        raise ValueError(
            f"cannot diff two sources: previous is {previous.source!r}, "
            f"current is {current.source!r}"
        )
    source = current.source
    before_partitions = _keyed_partitions(partitions_before, source)
    after_partitions = _keyed_partitions(partitions_after, source)
    before_hash, after_hash = snapshot_hash(previous), snapshot_hash(current)
    if before_hash == after_hash and before_partitions == after_partitions:
        return DriftReport(source=source, previous_hash=before_hash, current_hash=after_hash)

    index = semantic_index(layer, source=source)
    guarded = _guarded_columns(layer, source=source)
    changes: list[DriftChange] = []
    before = {table.name: table for table in previous.tables}
    after = {table.name: table for table in current.tables}
    for name in sorted(set(before) | set(after)):
        old, new = before.get(name), after.get(name)
        table = f"{source}.{name}"
        if old is None and new is not None:
            changes.append(
                _change(
                    ChangeKind.TABLE_ADDED,
                    table=table,
                    detail="a table the semantic layer does not know yet",
                    refs=index.by_table.get(table, []),
                    fix="run `mdqm probe` to refresh the catalog; add it to curated/ if a "
                    "question needs it.",
                )
            )
        elif new is None and old is not None:
            changes.append(
                _change(
                    ChangeKind.TABLE_REMOVED,
                    table=table,
                    before=old.kind.value,
                    detail="the table the semantic layer refers to is gone",
                    refs=index.by_table.get(table, []) + index.columns_of(table),
                    fix=f"remove {table} from curated/tables.yaml and every entry that reads it.",
                )
            )
        elif old is not None and new is not None:
            changes.extend(_table_changes(old, new, table=table, index=index, guarded=guarded))
    changes.extend(
        _partition_changes(
            before_partitions,
            after_partitions,
            _missing=_missing_partition_entries(partitions_before, partitions_after, source),
            source=source,
            index=index,
        )
    )
    ordered = tuple(sorted(changes, key=_change_key))
    return DriftReport(
        source=source, previous_hash=before_hash, current_hash=after_hash, changes=ordered
    )


def _change(
    kind: ChangeKind,
    *,
    table: str,
    refs: Iterable[SemanticRef] = (),
    detail: str = "",
    before: str | None = None,
    after: str | None = None,
    column: str | None = None,
    index_name: str | None = None,
    partition: str | None = None,
    fix: str = "",
    severity: Severity | None = None,
) -> DriftChange:
    """Build one change; the severity follows :data:`BREAKING_KINDS` unless an override is given."""
    return DriftChange(
        kind=kind,
        severity=severity
        or (Severity.BREAKING if kind in BREAKING_KINDS else Severity.NON_BREAKING),
        table=table,
        column=column,
        index=index_name,
        partition=partition,
        before=before,
        after=after,
        detail=detail,
        refs=_sorted_refs(refs),
        fix=fix,
    )


def _table_changes(
    old: Table, new: Table, *, table: str, index: _Index, guarded: frozenset[str]
) -> list[DriftChange]:
    """Every change inside one table that exists on both sides."""
    changes: list[DriftChange] = []
    if old.kind is not new.kind:
        changes.append(
            _change(
                ChangeKind.TABLE_KIND_CHANGED,
                table=table,
                before=old.kind.value,
                after=new.kind.value,
                detail="how the source serves the table changed (table, view, endpoint, ...)",
                refs=index.by_table.get(table, []),
                fix=f"re-check freshness and mutability for {table} in curated/tables.yaml.",
            )
        )
    if tuple(old.primary_key) != tuple(new.primary_key):
        changes.append(
            _change(
                ChangeKind.PRIMARY_KEY_CHANGED,
                table=table,
                before=", ".join(old.primary_key) or "none",
                after=", ".join(new.primary_key) or "none",
                detail="the key a join or a deduplication relies on moved",
                refs=index.by_table.get(table, []),
                fix=f"re-check every approved_join and key that names {table}.",
            )
        )
    changes.extend(_column_changes(old, new, table=table, index=index))
    changes.extend(_index_changes(old, new, table=table, index=index, guarded=guarded))
    changes.extend(_foreign_key_changes(old, new, table=table, index=index))
    return changes


def _column_changes(old: Table, new: Table, *, table: str, index: _Index) -> list[DriftChange]:
    """Added, removed, renamed and redefined columns, with the unambiguous-rename rule."""
    changes: list[DriftChange] = []
    before = {column.name: column for column in old.columns}
    after = {column.name: column for column in new.columns}
    removed = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    if len(removed) == 1 and len(added) == 1 and _same_shape(before[removed[0]], after[added[0]]):
        gone, came = removed[0], added[0]
        changes.append(
            _change(
                ChangeKind.COLUMN_RENAMED,
                table=table,
                column=gone,
                before=gone,
                after=came,
                detail="one column left and one arrived with the same shape: a rename is a guess",
                refs=index.by_column.get(f"{table}.{gone}", [])
                + index.by_column.get(f"{table}.{came}", []),
                fix=f"confirm the rename, then point every entry that read {table}.{gone} at "
                f"{came}.",
            )
        )
        removed, added = [], []
    for name in removed:
        changes.append(
            _change(
                ChangeKind.COLUMN_REMOVED,
                table=table,
                column=name,
                before=name,
                detail="a column the semantic layer may read no longer exists",
                refs=index.by_column.get(f"{table}.{name}", []),
                fix=f"point the entries that read {table}.{name} at the new column, "
                "or remove them.",
            )
        )
    for name in added:
        changes.append(
            _change(
                ChangeKind.COLUMN_ADDED,
                table=table,
                column=name,
                after=name,
                detail="a column nothing in the semantic layer refers to yet",
                refs=index.by_column.get(f"{table}.{name}", []),
                fix="nothing blocking; describe it if a question needs it.",
            )
        )
    for name in sorted(set(before) & set(after)):
        old_column, new_column = before[name], after[name]
        reference = f"{table}.{name}"
        refs = index.by_column.get(reference, [])
        if old_column.normalized_type is not new_column.normalized_type:
            changes.append(
                _change(
                    ChangeKind.COLUMN_TYPE_CHANGED,
                    table=table,
                    column=name,
                    before=old_column.normalized_type.value,
                    after=new_column.normalized_type.value,
                    detail=f"native type {old_column.native_type} -> {new_column.native_type}",
                    refs=refs,
                    fix=f"re-check every measure and conversion that reads {reference}.",
                )
            )
        elif old_column.native_type != new_column.native_type:
            # Re-spelling the type without changing its portable meaning is a fact, not a break.
            changes.append(
                _change(
                    ChangeKind.COLUMN_TYPE_CHANGED,
                    table=table,
                    column=name,
                    before=old_column.native_type,
                    after=new_column.native_type,
                    detail="the native type changed without changing the portable type",
                    refs=refs,
                    severity=Severity.NON_BREAKING,
                    fix="nothing; the portable type is unchanged.",
                )
            )
        if old_column.nullable is not new_column.nullable:
            changes.append(
                _change(
                    ChangeKind.COLUMN_NULLABILITY_CHANGED,
                    table=table,
                    column=name,
                    before="nullable" if old_column.nullable else "not null",
                    after="nullable" if new_column.nullable else "not null",
                    detail="NULLs can appear or vanish, which changes aggregates and joins",
                    refs=refs,
                    fix=f"re-check the aggregates and joins that read {reference}.",
                )
            )
    return changes


def _same_shape(left: Column, right: Column) -> bool:
    """True when two columns are plausibly the same thing under another name."""
    ordinals_agree = left.ordinal is None or right.ordinal is None or left.ordinal == right.ordinal
    return (
        left.normalized_type is right.normalized_type
        and left.native_type == right.native_type
        and left.nullable is right.nullable
        and ordinals_agree
    )


def _index_changes(
    old: Table, new: Table, *, table: str, index: _Index, guarded: frozenset[str]
) -> list[DriftChange]:
    """Added, dropped and redefined indexes, breaking only when an access guarantee is lost."""
    changes: list[DriftChange] = []
    before = {item.name: item for item in old.indexes}
    after = {item.name: item for item in new.indexes}
    for name in sorted(set(after) - set(before)):
        item = after[name]
        changes.append(
            _change(
                ChangeKind.INDEX_ADDED,
                table=table,
                index_name=name,
                after=", ".join(item.columns),
                detail="a new index appeared",
                refs=_index_refs(table, item, index),
                fix="nothing blocking; the planner may prune more cheaply now.",
            )
        )
    for name in sorted(set(before) - set(after)):
        item = before[name]
        carries_guarantee = _carries_guarantee(table, item, guarded)
        changes.append(
            _change(
                ChangeKind.INDEX_REMOVED,
                table=table,
                index_name=name,
                before=", ".join(item.columns),
                detail=(
                    "the index a default time column or partition key relies on is gone"
                    if carries_guarantee
                    else "an index that carries no default-time or partition guarantee is gone"
                ),
                refs=_index_refs(table, item, index),
                fix=(
                    f"restore the index on {table}, or stop relying on it for pruning."
                    if carries_guarantee
                    else "nothing blocking; a query that used it now scans more rows."
                ),
                severity=Severity.BREAKING if carries_guarantee else Severity.NON_BREAKING,
            )
        )
    for name in sorted(set(before) & set(after)):
        old_index, new_index = before[name], after[name]
        if (
            tuple(old_index.columns) == tuple(new_index.columns)
            and old_index.unique is new_index.unique
            and old_index.predicate == new_index.predicate
        ):
            continue
        changes.append(
            _change(
                ChangeKind.INDEX_CHANGED,
                table=table,
                index_name=name,
                before=_index_text(old_index),
                after=_index_text(new_index),
                detail="the definition behind a name changed, so a plan that trusted it "
                "changes too",
                refs=_index_refs(table, old_index, index) + _index_refs(table, new_index, index),
                fix=f"re-run `mdqm probe` for {table.split('.')[0]} and re-check pruning.",
            )
        )
    return changes


def _index_text(item: Index) -> str:
    """``start_time unique``: the definition as one readable token for a ``before -> after``."""
    text = ", ".join(item.columns)
    if item.unique:
        text += " unique"
    if item.predicate:
        text += f" where {item.predicate}"
    return text


def _carries_guarantee(table: str, item: Index, guarded: frozenset[str]) -> bool:
    """True when the index covers a default time column or a partition key (or is the PK)."""
    if item.primary:
        return True
    return any(f"{table}.{column}" in guarded for column in item.columns)


def _index_refs(table: str, item: Index, index: _Index) -> list[SemanticRef]:
    """Every semantic entry that reads a column this index covers."""
    found: list[SemanticRef] = []
    for column in item.columns:
        found.extend(index.by_column.get(f"{table}.{column}", []))
    return found


def _foreign_key_changes(old: Table, new: Table, *, table: str, index: _Index) -> list[DriftChange]:
    """Added and removed foreign keys: a declared FK is the only auto-approved join shape."""
    changes: list[DriftChange] = []

    def signature(columns: tuple[str, ...], ref_table: str, ref_columns: tuple[str, ...]) -> str:
        return f"{','.join(columns)}->{ref_table}({','.join(ref_columns)})"

    before = {signature(fk.columns, fk.ref_table, fk.ref_columns) for fk in old.foreign_keys}
    after = {signature(fk.columns, fk.ref_table, fk.ref_columns) for fk in new.foreign_keys}
    for added in sorted(after - before):
        changes.append(
            _change(
                ChangeKind.FOREIGN_KEY_ADDED,
                table=table,
                after=added,
                detail="a declared foreign key appeared; a join may now be approved by evidence",
                refs=index.by_table.get(table, []),
                fix="nothing blocking; the declared FK may justify a new approved_join.",
            )
        )
    for removed in sorted(before - after):
        changes.append(
            _change(
                ChangeKind.FOREIGN_KEY_REMOVED,
                table=table,
                before=removed,
                detail="a declared foreign key disappeared: the containment that justified a join "
                "is gone",
                refs=index.by_table.get(table, []) + index.columns_of(table),
                fix=f"re-verify the approved_joins that relied on {removed}.",
            )
        )
    return changes


def _keyed_partitions(
    partitions: Mapping[str, Partitioning] | None, source: str
) -> dict[str, Partitioning]:
    """One key shape for both sides of the diff: ``billing.usage_events`` -> ``usage_events``."""
    prefix = f"{source}."
    return {
        (name[len(prefix) :] if name.startswith(prefix) else name): value
        for name, value in (partitions or {}).items()
    }


def _missing_partition_entries(
    before: Mapping[str, Partitioning] | None,
    after: Mapping[str, Partitioning] | None,
    source: str,
) -> tuple[str, ...]:
    """Tables whose partition metadata appeared or disappeared entirely (not a shape change)."""
    left = _keyed_partitions(before, source)
    right = _keyed_partitions(after, source)
    appeared = sorted(set(right) - set(left))
    vanished = sorted(set(left) - set(right))
    return tuple(f"{'+' if name in appeared else '-'}{name}" for name in (*appeared, *vanished))


def _partition_changes(
    before: Mapping[str, Partitioning],
    after: Mapping[str, Partitioning],
    *,
    _missing: Sequence[str],
    source: str,
    index: _Index,
) -> list[DriftChange]:
    """Strategy, key, granularity and individual partition changes, per table."""
    changes: list[DriftChange] = []
    for entry in _missing:
        marker, name = entry[0], entry[1:]
        table = f"{source}.{name}"
        if marker == "+":
            changes.append(
                _change(
                    ChangeKind.PARTITION_ADDED,
                    table=table,
                    after=after[name].strategy.value,
                    detail="partition metadata appeared for a table that had none reported",
                    refs=index.by_table.get(table, []),
                    fix="nothing blocking; record the partitioning in curated/tables.yaml.",
                )
            )
        else:
            changes.append(
                _change(
                    ChangeKind.PARTITION_REMOVED,
                    table=table,
                    before=before[name].strategy.value,
                    detail="partition metadata disappeared, so pruning has to fall back",
                    refs=index.by_table.get(table, []),
                    fix=f"restore partition metadata for {table}, or expect full scans.",
                )
            )
    for name in sorted(set(before) & set(after)):
        old, new = before[name], after[name]
        table = f"{source}.{name}"
        if old.strategy is not new.strategy:
            changes.append(
                _change(
                    ChangeKind.PARTITION_STRATEGY_CHANGED,
                    table=table,
                    before=old.strategy.value,
                    after=new.strategy.value,
                    detail="the table is partitioned a different way, so pruning rules change",
                    refs=index.by_table.get(table, []),
                    fix=f"re-check the partitioning and the enforced key filter for {table}.",
                )
            )
        if tuple(old.key) != tuple(new.key):
            changes.append(
                _change(
                    ChangeKind.PARTITION_KEY_CHANGED,
                    table=table,
                    before=", ".join(old.key) or "none",
                    after=", ".join(new.key) or "none",
                    detail="a filter on the old key no longer prunes, so the planner would scan",
                    refs=index.by_table.get(table, []) + index.columns_of(table),
                    fix=f"re-check partitioning.key for {table} and every query that filtered it.",
                )
            )
        if old.granularity is not new.granularity:
            changes.append(
                _change(
                    ChangeKind.PARTITION_GRANULARITY_CHANGED,
                    table=table,
                    before=old.granularity.value if old.granularity else "none",
                    after=new.granularity.value if new.granularity else "none",
                    detail="partition width changed, so cache segments align differently",
                    refs=index.by_table.get(table, []),
                    fix=f"re-check partitioning.granularity for {table}.",
                )
            )
        old_bounds = {bound.name: bound for bound in old.bounds}
        new_bounds = {bound.name: bound for bound in new.bounds}
        for bound in sorted(set(new_bounds) - set(old_bounds)):
            changes.append(
                _change(
                    ChangeKind.PARTITION_ADDED,
                    table=table,
                    partition=bound,
                    after=_bound_text(new_bounds[bound]),
                    detail="a new partition exists; no definition changed with it",
                    refs=index.by_table.get(table, []),
                    fix="nothing blocking; a new time range is queryable now.",
                )
            )
        for bound in sorted(set(old_bounds) - set(new_bounds)):
            changes.append(
                _change(
                    ChangeKind.PARTITION_REMOVED,
                    table=table,
                    partition=bound,
                    before=_bound_text(old_bounds[bound]),
                    detail="a partition was dropped, so rows an answer covered may be gone",
                    refs=index.by_table.get(table, []),
                    fix=f"confirm the retention policy that dropped {bound} on {table}.",
                )
            )
    return changes


def _bound_text(bound: object) -> str | None:
    """A partition bound as one readable token: its upper edge, else its listed values."""
    upper = getattr(bound, "upper", None)
    if upper is not None:
        return str(upper.isoformat())
    values = getattr(bound, "values", ())
    return ", ".join(values) or None


# --------------------------------------------------------------------------------------------------
# Where the baseline, the blocks and the open questions live
# --------------------------------------------------------------------------------------------------


def _meta_value(store: Store, key: str) -> str | None:
    row = store.connect().execute(_SELECT_META, (key,)).fetchone()
    return None if row is None else str(row["value"])


def _write_meta(store: Store, key: str, value: str, *, at: datetime) -> None:
    """Upsert one ``mdqm_meta`` row through the store's own writer, so there is one write path."""
    with store.writing() as conn:
        conn.execute(_UPSERT_META, (key, value, at.isoformat()))


def read_baseline(store: Store, source: str) -> CatalogSnapshot | None:
    """The catalog snapshot this source was last reconciled against, or ``None`` on a first run.

    ``None`` means "nothing to compare": D10 records the snapshot and reports no drift. A damaged
    row is treated as absent (and logged) rather than crashing the pass; the next accepted snapshot
    replaces it.
    """
    raw = _meta_value(store, f"{BASELINE_PREFIX}.{source}")
    if raw is None:
        return None
    try:
        return CatalogSnapshot.model_validate_json(raw)
    except (ValueError, sqlite3.Error):
        logger.warning("stored catalog baseline is unreadable", extra={"source": source})
        return None


def write_baseline(
    store: Store,
    snapshot: CatalogSnapshot,
    *,
    at: datetime,
    partitions: Mapping[str, Partitioning] | None = None,
) -> None:
    """Record ``snapshot`` (and its partition metadata) as the baseline the next run compares to."""
    _write_meta(store, f"{BASELINE_PREFIX}.{snapshot.source}", snapshot.model_dump_json(), at=at)
    _write_meta(
        store,
        f"{PARTITIONS_PREFIX}.{snapshot.source}",
        json.dumps(
            {name: value.model_dump(mode="json") for name, value in (partitions or {}).items()},
            sort_keys=True,
        ),
        at=at,
    )


def read_partitions(store: Store, source: str) -> dict[str, Partitioning]:
    """The partition metadata stored with this source's baseline; empty when none was recorded."""
    raw = _meta_value(store, f"{PARTITIONS_PREFIX}.{source}")
    if raw is None:
        return {}
    try:
        decoded = json.loads(raw)
    except ValueError:
        logger.warning("stored partition metadata is unreadable", extra={"source": source})
        return {}
    if not isinstance(decoded, dict):
        return {}
    try:
        return {str(name): Partitioning.model_validate(value) for name, value in decoded.items()}
    except ValueError:
        logger.warning("stored partition metadata does not validate", extra={"source": source})
        return {}


def blocked_references(store: Store, *, source: str | None = None) -> tuple[str, ...]:
    """The semantic entries currently blocked by an undecided breaking change.

    This is the enforcement seam: a linter, the server or a CLI reads it before answering from an
    entry, so a dropped column cannot be used until a human approved the change. ``source=None``
    reads every source.
    """
    pattern = f"{BLOCKED_PREFIX}.%" if source is None else f"{BLOCKED_PREFIX}.{source}"
    labels: set[str] = set()
    rows = store.connect().execute(_SELECT_META_LIKE, (pattern,)).fetchall()
    for row in rows:
        try:
            decoded = json.loads(str(row["value"]))
        except ValueError:
            logger.warning("a stored drift block list is unreadable")
            continue
        if isinstance(decoded, list):
            labels.update(str(item) for item in decoded)
    return tuple(sorted(labels))


def read_notification(store: Store, source: str) -> AdminNotification | None:
    """The admin notification this source's last run produced and nobody has delivered yet."""
    raw = _meta_value(store, f"{NOTIFICATION_PREFIX}.{source}")
    if raw is None:
        return None
    try:
        decoded = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(decoded, dict):
        return None
    return AdminNotification.from_payload(decoded)


# --------------------------------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------------------------------

#: What the stage calls to read a source's catalog, and its partition metadata.
CatalogReader = Callable[[str], CatalogSnapshot]
PartitionReader = Callable[[str], Mapping[str, Partitioning]]


class DriftStage:
    """Stage D10: diff each source's catalog against the baseline and record what changed.

    Dependencies are injected (a stage is given no adapters): ``extract`` reads one source's catalog
    and ``extract_partitions`` reads that source's partition metadata, both keyed by source name.
    ``queue`` is where a breaking change becomes a question a human answers (S12); without one the
    change is still reported and blocked, and the report says no question was filed.

    A stage object memoises the catalogs read during :meth:`inputs_hash` for one pass, because a
    catalog extraction is a metadata read -- cheap, but it must happen *before* the input hash so a
    changed catalog can never be skipped as "unchanged input".
    """

    name: str = STAGE_NAME

    def __init__(
        self,
        *,
        extract: CatalogReader,
        sources: Sequence[str],
        extract_partitions: PartitionReader | None = None,
        layer_loader: Callable[[Path], SemanticLayer] | None = None,
        queue: ReviewQueue | None = None,
    ) -> None:
        if not sources:
            raise ValueError("D10 needs at least one source to diff")
        self.sources = tuple(sources)
        self._extract = extract
        self._extract_partitions = extract_partitions
        self._layer_loader = layer_loader or load
        self._queue = queue
        self._fresh: dict[str, CatalogSnapshot] = {}

    def _read(self, source: str) -> CatalogSnapshot:
        """Read one source's catalog once in this pass, and keep it for :meth:`run`.

        The memo is cleared at the start of every :meth:`inputs_hash` (the first thing the runner
        calls): a catalog kept from the previous night would make the hash -- and therefore the
        skip decision -- describe a catalog that no longer exists.
        """
        snapshot = self._fresh.get(source)
        if snapshot is None:
            snapshot = self._extract(source)
            self._fresh[source] = snapshot
        return snapshot

    def _layer(self, ctx: PipelineContext) -> SemanticLayer | None:
        """The semantic layer, or ``None`` when it does not load yet.

        A layer that fails to load is another task's error to report; refusing to diff because of it
        would hide the drift as well, so D10 diffs without a mapping and says so in its report.
        """
        try:
            return self._layer_loader(ctx.semantic_root)
        except SemanticError:
            return None

    def inputs_hash(self, ctx: PipelineContext) -> str:
        """sha256 over each source's fresh catalog hash, its baseline hash and the layer version.

        Reading the catalog here is what makes the hash honest: the catalog *is* this stage's input,
        so a hash built from anything else would let a changed catalog be skipped.
        """
        self._fresh.clear()
        parts: list[object] = [STAGE_NAME]
        for source in self.sources:
            if ctx.source is not None and ctx.source != source:
                continue
            baseline = read_baseline(ctx.store, source)
            parts.append(
                {
                    "source": source,
                    "current": snapshot_hash(self._read(source)),
                    "baseline": snapshot_hash(baseline) if baseline is not None else None,
                    "partitions": len(read_partitions(ctx.store, source)),
                }
            )
        layer = self._layer(ctx)
        parts.append({"layer": None if layer is None else layer.version_hash})
        return hash_inputs(*parts)

    def run(self, ctx: PipelineContext) -> StageReport:
        """Diff, block, propose and notify -- then advance the baseline when nothing is breaking."""
        counts: dict[str, int] = {}
        skipped: list[str] = []
        fixes: list[str] = []
        layer = self._layer(ctx)
        if layer is None:
            skipped.append(
                "the semantic layer did not load: drift is classified without a YAML mapping"
            )
        for source in self.sources:
            if ctx.source is not None and ctx.source != source:
                continue
            snapshot = self._read(source)
            after_partitions = (
                dict(self._extract_partitions(source))
                if self._extract_partitions is not None
                else read_partitions(ctx.store, source)
            )
            baseline = read_baseline(ctx.store, source)
            if baseline is None:
                write_baseline(ctx.store, snapshot, at=ctx.exec_ts, partitions=after_partitions)
                counts["baselines"] = counts.get("baselines", 0) + 1
                skipped.append(f"{source}: first snapshot recorded, nothing to compare")
                continue
            report = diff_snapshots(
                baseline,
                snapshot,
                layer=layer,
                partitions_before=read_partitions(ctx.store, source),
                partitions_after=after_partitions,
            )
            breaking = report.breaking
            undecided = tuple(change for change in breaking if not _decided(change, self._queue))
            settled = report.model_copy(update={"changes": (*report.non_breaking, *undecided)})
            counts["changes"] = counts.get("changes", 0) + len(report.changes)
            counts["breaking"] = counts.get("breaking", 0) + len(breaking)
            counts["undecided"] = counts.get("undecided", 0) + len(undecided)
            counts["non_breaking"] = counts.get("non_breaking", 0) + len(report.non_breaking)
            counts["blocked"] = counts.get("blocked", 0) + len(settled.blocked)
            _write_meta(
                ctx.store,
                f"{BLOCKED_PREFIX}.{source}",
                json.dumps([ref.label for ref in settled.blocked]),
                at=ctx.exec_ts,
            )
            if undecided:
                self._propose(settled, ctx=ctx, counts=counts)
                notification = settled.notification(created_at=ctx.exec_ts)
                if notification is not None:
                    _write_meta(
                        ctx.store,
                        f"{NOTIFICATION_PREFIX}.{source}",
                        json.dumps(notification.payload(), sort_keys=True),
                        at=ctx.exec_ts,
                    )
                    counts["notifications"] = counts.get("notifications", 0) + 1
                fixes.append(
                    f"{source}: baseline kept at the previous catalog until {len(breaking)} "
                    "breaking change(s) are decided, so the drift cannot be absorbed"
                )
            else:
                write_baseline(ctx.store, snapshot, at=ctx.exec_ts, partitions=after_partitions)
                fixes.append(f"{source}: baseline advanced past {len(report.changes)} change(s)")
            logger.info(
                "drift compared",
                extra={
                    "source": source,
                    "changes": len(report.changes),
                    "breaking": len(breaking),
                    "blocked": len(report.blocked),
                },
            )
        return StageReport(
            stage=STAGE_NAME,
            status=StageStatus.RAN,
            counts=counts,
            skipped=tuple(skipped),
            applied_fixes=tuple(fixes),
        )

    def _propose(
        self, report: DriftReport, *, ctx: PipelineContext, counts: dict[str, int]
    ) -> None:
        """File one review-queue proposal per breaking change, on the table that changed."""
        for change in report.breaking:
            patch = _proposal_patch(change)
            counts["proposals"] = counts.get("proposals", 0) + 1
            if self._queue is None:
                counts["unfiled"] = counts.get("unfiled", 0) + 1
                continue
            result = self._queue.submit(
                Proposal(
                    id=_proposal_identity(change),
                    kind=ProposalKind.OTHER,
                    section="tables",
                    key=change.table,
                    field="",
                    patch=patch,
                    evidence=_evidence(change),
                    confidence=1.0,
                    proposed_by=PROPOSED_BY,
                    proposed_at=ctx.exec_ts,
                )
            )
            if result.action in (SubmitAction.CREATED, SubmitAction.RESURFACED):
                counts["questions"] = counts.get("questions", 0) + 1


def _evidence_text(change: DriftChange) -> str:
    """The change plus its fix as one provenance string, for the table an approval records."""
    return f"{change.describe()} | fix: {change.fix}" if change.fix else change.describe()


def _decided(change: DriftChange, queue: ReviewQueue | None) -> bool:
    """True when a human **approved** this exact change, so it stops blocking.

    Only ``approved`` settles a change, and only a settled set lets the baseline advance. A
    rejection or a "never ask" leaves it blocking: the source still changed, so the entries that
    read the gone column are still wrong, and the queue already remembers nobody wants asking again.
    """
    if queue is None:
        return False
    try:
        return queue.get(_proposal_identity(change)).status is ProposalStatus.APPROVED
    except ReviewError:
        return False


def _proposal_identity(change: DriftChange) -> str:
    """The review queue's content id for this change: one row per (table, change), rerun-safe."""
    return proposal_id(
        ProposalKind.OTHER, "tables", change.table, "", _proposal_patch(change)
    )


def _proposal_patch(change: DriftChange) -> dict[str, Any]:
    """The curated patch an approval writes: an acknowledgement on the table's ``provenance``.

    ``provenance`` is a field :class:`~mdqm.semantic.models.TableSemantics` already has, so an
    approved proposal leaves a curated file the loader accepts. D10 never invents a fix: the human
    acts on the evidence and the ``fix`` text.
    """
    return {
        "provenance": {
            "detected_by": PROPOSED_BY,
            "evidence": _evidence_text(change),
        }
    }


def _evidence(change: DriftChange) -> tuple[str, ...]:
    """What a human reads in the queue: the change, the fix, and the entries it reaches."""
    lines = [change.describe()]
    if change.detail:
        lines.append(f"detail: {change.detail}")
    if change.fix:
        lines.append(f"fix: {change.fix}")
    lines.extend(f"blocked: {ref.label}" for ref in change.refs)
    return tuple(lines[:EVIDENCE_LIMIT])
