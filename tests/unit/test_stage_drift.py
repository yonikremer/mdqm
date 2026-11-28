"""Tests for stage D10 schema drift (S09).

Four promises, and one test group per promise:

* **Statistics churn is not drift.** ``captured_at``, ``row_estimate`` and column ``ordinal`` are
  excluded from ``snapshot_hash`` (F02), so changing only them must produce an empty report and no
  question for anyone.
* **A real structural change is drift, and breaking ones block.** A dropped column, a type change,
  a dropped index on a default time column, a partition-key change: each one is classified, the
  semantic entries that read it are named, and those entries are listed as blocked until a human
  decides.
* **The rename rule is narrow.** One removed column plus one added column with the same shape is
  reported as a rename (still breaking); two candidates degrade to removed + added.
* **The stage files the question and keeps the drift visible.** A breaking change becomes a review
  queue proposal and an admin notification payload, and the baseline does *not* advance; a
  non-breaking change advances the baseline and asks nothing.

No clock is read (``exec_ts`` is injected) and every store test uses ``tmp_path``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

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
from mdqm.models.partitioning import (
    PartitionBound,
    PartitionGranularity,
    Partitioning,
    PartitionStrategy,
)
from mdqm.pipeline.core import PipelineContext, StageStatus, run_pipeline
from mdqm.pipeline.stages.drift import (
    BLOCKED_PREFIX,
    NOTIFICATION_PREFIX,
    STAGE_NAME,
    AdminNotification,
    ChangeKind,
    DriftChange,
    DriftReport,
    DriftStage,
    Severity,
    blocked_references,
    diff_snapshots,
    read_baseline,
    read_notification,
    semantic_index,
    write_baseline,
)
from mdqm.review.queue import ProposalStatus, ReviewQueue
from mdqm.semantic.models import (
    ApprovedJoin,
    JoinCardinality,
    Measure,
    MeasureSource,
    SemanticLayer,
    TableSemantics,
    TimeColumn,
    TimeRole,
    ValueMap,
)
from mdqm.store.db import Store

EXEC_TS = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)
"""One execution instant for every case: drift detection must not read the clock."""

SOURCE = "billing"


# --------------------------------------------------------------------------------------------------
# Building snapshots and a semantic layer without a database
# --------------------------------------------------------------------------------------------------


def column(
    name: str,
    *,
    native: str = "varchar(10)",
    normalized: NormalizedType = NormalizedType.STRING,
    nullable: bool = True,
    ordinal: int | None = None,
) -> Column:
    return Column(
        name=name,
        native_type=native,
        normalized_type=normalized,
        nullable=nullable,
        ordinal=ordinal,
    )


def index(name: str, *columns: str, unique: bool = False, primary: bool = False) -> Index:
    return Index(name=name, columns=columns, unique=unique, primary=primary)


def table(
    name: str = "accounts",
    *,
    columns: tuple[Column, ...] | None = None,
    indexes: tuple[Index, ...] = (),
    foreign_keys: tuple[ForeignKey, ...] = (),
    primary_key: tuple[str, ...] = (),
    kind: TableKind = TableKind.TABLE,
    row_estimate: int | None = 100,
) -> Table:
    return Table(
        name=name,
        kind=kind,
        columns=columns or (column("client_no"), column("credit_limit")),
        indexes=indexes,
        foreign_keys=foreign_keys,
        primary_key=primary_key,
        row_estimate=row_estimate,
    )


def snapshot(
    *tables: Table,
    source: str = SOURCE,
    captured_at: datetime = EXEC_TS,
) -> CatalogSnapshot:
    return CatalogSnapshot(source=source, captured_at=captured_at, tables=tables)


def monthly(key: str = "start_time", *bounds: str) -> Partitioning:
    return Partitioning(
        strategy=PartitionStrategy.RANGE,
        key=(key,),
        granularity=PartitionGranularity.MONTH,
        bounds=tuple(PartitionBound(name=name, lower=EXEC_TS, upper=EXEC_TS) for name in bounds),
    )


def credit_layer(*, table_key: str = "billing.accounts") -> SemanticLayer:
    """A layer whose measure, value map, time column and join all read the same two tables."""
    # A stated version_hash keeps the layer's own validator quiet: it copies the model when the
    # hash is empty, and creating a layer is not what this file tests.
    return SemanticLayer(
        version_hash="credit-layer",
        measures={
            "credit_limit_r5000": Measure(
                sources={
                    "billing.accounts.credit_limit": MeasureSource(normalize="{col}"),
                    "billing.customers.limit_k": MeasureSource(normalize="{col} * 10"),
                }
            )
        },
        value_maps={
            "customer_status": ValueMap(
                canonical=("active", "churned"),
                sources={"billing.accounts.state": {"A": "active", "C": "churned"}},
            )
        },
        tables={
            table_key: TableSemantics(
                key=("client_no",),
                partitioning=monthly(),
                time_columns={"start_time": TimeColumn(role=TimeRole.EVENT_TIME, indexed=True)},
            )
        },
        approved_joins=(
            ApprovedJoin(
                name="usage_to_accounts",
                left="billing.usage_events.client_no",
                right="billing.accounts.client_no",
                on="{left} = {right}",
                cardinality=JoinCardinality.MANY_TO_ONE,
            ),
        ),
    )


def only_change(report: DriftReport, kind: ChangeKind) -> DriftChange:
    """The single change of ``kind`` in ``report``, so a test names what it asserts on."""
    matches = [change for change in report.changes if change.kind is kind]
    assert len(matches) == 1, f"expected one {kind} in {[c.kind for c in report.changes]}"
    return matches[0]


# --------------------------------------------------------------------------------------------------
# Statistically churn is not drift (acceptance: statistics-only changes are not drift)
# --------------------------------------------------------------------------------------------------


def test_identical_snapshots_produce_no_report() -> None:
    report = diff_snapshots(snapshot(table()), snapshot(table()))
    assert report.changes == ()
    assert not report.changed
    assert report.previous_hash == report.current_hash
    assert report.describe() == f"{SOURCE}: no schema drift ({report.current_hash[:12]})"


@pytest.mark.parametrize(
    "later_table",
    [
        pytest.param(table(row_estimate=999_999), id="row_estimate"),
        pytest.param(
            table(columns=(column("client_no", ordinal=9), column("credit_limit", ordinal=2))),
            id="column_ordinal",
        ),
        pytest.param(
            table(columns=(column("credit_limit", ordinal=1), column("client_no", ordinal=2))),
            id="column_order",
        ),
    ],
)
def test_statistics_churn_is_not_drift(later_table: Table) -> None:
    """row_estimate, ordinal and column order are excluded from snapshot_hash, so they are churn."""
    before = snapshot(table())
    after = snapshot(later_table, captured_at=datetime(2026, 12, 25, tzinfo=UTC))
    assert snapshot_hash(before) == snapshot_hash(after)
    assert diff_snapshots(before, after).changes == ()


def test_partition_row_estimate_churn_is_not_drift() -> None:
    """A partition's row_estimate moves every night; the bound it names did not change."""
    before = Partitioning(
        strategy=PartitionStrategy.RANGE,
        key=("start_time",),
        granularity=PartitionGranularity.MONTH,
        bounds=(PartitionBound(name="p2026_11", lower=EXEC_TS, upper=EXEC_TS, row_estimate=10),),
    )
    after = before.model_copy(
        update={
            "bounds": (
                PartitionBound(name="p2026_11", lower=EXEC_TS, upper=EXEC_TS, row_estimate=9999),
            )
        }
    )
    report = diff_snapshots(
        snapshot(table()),
        snapshot(table()),
        partitions_before={SOURCE: before},
        partitions_after={SOURCE: after},
    )
    assert report.changes == ()


# --------------------------------------------------------------------------------------------------
# 12+ change kinds, table driven (acceptance 1)
# --------------------------------------------------------------------------------------------------


def _expect(kind: ChangeKind, severity: Severity) -> tuple[ChangeKind, Severity]:
    return (kind, severity)


CHANGE_CASES: list[Any] = [
    pytest.param(
        snapshot(),
        snapshot(table()),
        _expect(ChangeKind.TABLE_ADDED, Severity.NON_BREAKING),
        id="table-added",
    ),
    pytest.param(
        snapshot(table()),
        snapshot(),
        _expect(ChangeKind.TABLE_REMOVED, Severity.BREAKING),
        id="table-removed",
    ),
    pytest.param(
        snapshot(table()),
        snapshot(table(kind=TableKind.VIEW)),
        _expect(ChangeKind.TABLE_KIND_CHANGED, Severity.BREAKING),
        id="table-kind-changed",
    ),
    pytest.param(
        snapshot(table()),
        snapshot(table(columns=(column("client_no"), column("credit_limit"), column("state")))),
        _expect(ChangeKind.COLUMN_ADDED, Severity.NON_BREAKING),
        id="column-added",
    ),
    pytest.param(
        snapshot(table(columns=(column("client_no"), column("credit_limit"), column("state")))),
        snapshot(table()),
        _expect(ChangeKind.COLUMN_REMOVED, Severity.BREAKING),
        id="column-removed",
    ),
    pytest.param(
        snapshot(table()),
        snapshot(
            table(
                columns=(
                    column("client_no"),
                    column("credit_limit", normalized=NormalizedType.INTEGER, native="int"),
                )
            )
        ),
        _expect(ChangeKind.COLUMN_TYPE_CHANGED, Severity.BREAKING),
        id="column-type-changed",
    ),
    pytest.param(
        snapshot(table()),
        snapshot(
            table(
                columns=(
                    column("client_no"),
                    column("credit_limit", native="varchar(40)"),
                )
            )
        ),
        _expect(ChangeKind.COLUMN_TYPE_CHANGED, Severity.NON_BREAKING),
        id="native-type-widened",
    ),
    pytest.param(
        snapshot(table()),
        snapshot(
            table(
                columns=(
                    column("client_no"),
                    column("credit_limit", nullable=False),
                )
            )
        ),
        _expect(ChangeKind.COLUMN_NULLABILITY_CHANGED, Severity.BREAKING),
        id="column-became-not-null",
    ),
    pytest.param(
        snapshot(table(indexes=(index("accounts_pk", "client_no", primary=True),))),
        snapshot(
            table(
                indexes=(
                    index("accounts_pk", "client_no", primary=True),
                    index("accounts_limit", "credit_limit"),
                )
            )
        ),
        _expect(ChangeKind.INDEX_ADDED, Severity.NON_BREAKING),
        id="index-added",
    ),
    pytest.param(
        snapshot(
            table(
                indexes=(
                    index("accounts_pk", "client_no", primary=True),
                    index("accounts_limit", "credit_limit"),
                )
            )
        ),
        snapshot(table(indexes=(index("accounts_pk", "client_no", primary=True),))),
        _expect(ChangeKind.INDEX_REMOVED, Severity.NON_BREAKING),
        id="index-removed-no-guarantee",
    ),
    pytest.param(
        snapshot(table(indexes=(index("accounts_state", "credit_limit"),))),
        snapshot(table(indexes=(index("accounts_state", "credit_limit", unique=True),))),
        _expect(ChangeKind.INDEX_CHANGED, Severity.BREAKING),
        id="index-changed",
    ),
    pytest.param(
        snapshot(table(primary_key=("client_no",))),
        snapshot(table(primary_key=("client_no", "credit_limit"))),
        _expect(ChangeKind.PRIMARY_KEY_CHANGED, Severity.BREAKING),
        id="primary-key-changed",
    ),
    pytest.param(
        snapshot(table()),
        snapshot(
            table(
                foreign_keys=(
                    ForeignKey(columns=("client_no",), ref_table="customers", ref_columns=("id",)),
                )
            )
        ),
        _expect(ChangeKind.FOREIGN_KEY_ADDED, Severity.NON_BREAKING),
        id="foreign-key-added",
    ),
    pytest.param(
        snapshot(
            table(
                foreign_keys=(
                    ForeignKey(columns=("client_no",), ref_table="customers", ref_columns=("id",)),
                )
            )
        ),
        snapshot(table()),
        _expect(ChangeKind.FOREIGN_KEY_REMOVED, Severity.BREAKING),
        id="foreign-key-removed",
    ),
]


@pytest.mark.parametrize("before, after, expected", CHANGE_CASES)
def test_each_change_kind_is_classified(
    before: CatalogSnapshot, after: CatalogSnapshot, expected: tuple[ChangeKind, Severity]
) -> None:
    report = diff_snapshots(before, after)
    kinds = {change.kind for change in report.changes}
    kind, severity = expected
    assert kind in kinds, f"{kind} missing from {sorted(k.value for k in kinds)}"
    for change in report.changes:
        if change.kind is kind:
            assert change.severity is severity
            assert change.breaking is (severity is Severity.BREAKING)
            assert change.describe().startswith(severity.value)


def test_table_removed_reports_every_column_of_that_table() -> None:
    report = diff_snapshots(snapshot(table()), snapshot())
    change = only_change(report, ChangeKind.TABLE_REMOVED)
    assert change.table == "billing.accounts"
    assert change.before == "table"
    assert "remove billing.accounts" in change.fix


# --------------------------------------------------------------------------------------------------
# Rename detection is narrow on purpose
# --------------------------------------------------------------------------------------------------


def test_a_single_shape_matched_pair_is_reported_as_a_rename() -> None:
    before = snapshot(table(columns=(column("client_no"), column("credit_limit"))))
    after = snapshot(table(columns=(column("client_no"), column("credit_limit_r5000"))))
    report = diff_snapshots(before, after)
    kinds = {change.kind for change in report.changes}
    assert kinds == {ChangeKind.COLUMN_RENAMED}
    change = only_change(report, ChangeKind.COLUMN_RENAMED)
    assert change.before == "credit_limit"
    assert change.after == "credit_limit_r5000"
    assert change.breaking
    assert "confirm the rename" in change.fix


def test_two_candidates_are_not_paired_as_a_rename() -> None:
    """Ambiguity degrades to removed + added: guessing the pairing would name the wrong measure."""
    before = snapshot(table(columns=(column("client_no"), column("a"), column("b"))))
    after = snapshot(table(columns=(column("client_no"), column("c"), column("d"))))
    report = diff_snapshots(before, after)
    kinds = {change.kind for change in report.changes}
    assert ChangeKind.COLUMN_RENAMED not in kinds
    assert kinds == {ChangeKind.COLUMN_REMOVED, ChangeKind.COLUMN_ADDED}


def test_a_different_shape_is_not_a_rename() -> None:
    before = snapshot(table(columns=(column("client_no"), column("credit_limit"))))
    after = snapshot(
        table(
            columns=(
                column("client_no"),
                column("credit_limit_k", native="int", normalized=NormalizedType.INTEGER),
            )
        )
    )
    report = diff_snapshots(before, after)
    kinds = {change.kind for change in report.changes}
    assert ChangeKind.COLUMN_RENAMED not in kinds
    assert ChangeKind.COLUMN_REMOVED in kinds


# --------------------------------------------------------------------------------------------------
# The mapping to semantic YAML, and what an unapproved change blocks (acceptance 2)
# --------------------------------------------------------------------------------------------------


def test_a_dropped_column_used_by_a_measure_lists_that_measure() -> None:
    layer = credit_layer()
    before = snapshot(table(columns=(column("client_no"), column("credit_limit"))))
    after = snapshot(table(columns=(column("client_no"),)))
    report = diff_snapshots(before, after, layer=layer)
    change = only_change(report, ChangeKind.COLUMN_REMOVED)
    assert [ref.label for ref in change.refs] == ["measures.credit_limit_r5000"]
    assert change.refs[0].reference == "billing.accounts.credit_limit"
    assert report.blocked == change.refs
    assert report.is_blocked("measures.credit_limit_r5000")
    notification = report.notification(created_at=EXEC_TS)
    assert notification is not None
    assert "measures.credit_limit_r5000" in notification.body


def test_a_dropped_column_reaches_the_value_map_and_the_join_that_read_it() -> None:
    layer = credit_layer()
    before = snapshot(table(columns=(column("client_no"), column("state"), column("credit_limit"))))
    after = snapshot(table(columns=(column("state"), column("credit_limit"))))
    report = diff_snapshots(before, after, layer=layer)
    blocked = {ref.label for ref in report.blocked}
    assert blocked == {"approved_joins.usage_to_accounts", "tables.billing.accounts.key.client_no"}


def test_a_non_breaking_change_blocks_nothing() -> None:
    layer = credit_layer()
    before = snapshot(table())
    after = snapshot(
        table(columns=(column("client_no"), column("credit_limit"), column("state_new")))
    )
    report = diff_snapshots(before, after, layer=layer)
    assert report.breaking == ()
    assert report.blocked == ()
    assert report.notification(created_at=EXEC_TS) is None


def test_a_removed_table_blocks_every_entry_that_reads_it() -> None:
    layer = credit_layer()
    report = diff_snapshots(snapshot(table()), snapshot(), layer=layer)
    blocked = {ref.label for ref in report.blocked}
    assert "measures.credit_limit_r5000" in blocked
    assert "value_maps.customer_status" in blocked
    assert "approved_joins.usage_to_accounts" in blocked


def test_drift_without_a_layer_is_reported_without_a_mapping() -> None:
    """No layer means nothing is known to depend on the change: report it, block nothing."""
    report = diff_snapshots(snapshot(table()), snapshot())
    assert [change.kind for change in report.changes] == [ChangeKind.TABLE_REMOVED]
    assert report.refs == ()
    assert report.blocked == ()


def test_semantic_index_qualifies_bare_table_keys() -> None:
    layer = credit_layer(table_key="accounts")
    index = semantic_index(layer, source=SOURCE)
    assert "billing.accounts.credit_limit" in index.by_column
    assert "billing.accounts" in index.by_table


def test_diff_of_two_sources_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot diff two sources"):
        diff_snapshots(snapshot(table(), source="crm"), snapshot(table()))


# --------------------------------------------------------------------------------------------------
# Partitioning: a new monthly partition is not breaking, a key change is (acceptance 3)
# --------------------------------------------------------------------------------------------------


def test_a_new_monthly_partition_is_not_breaking() -> None:
    before = {SOURCE: monthly("start_time", "p2026_10")}
    after = {SOURCE: monthly("start_time", "p2026_10", "p2026_11")}
    report = diff_snapshots(
        snapshot(table()), snapshot(table()), partitions_before=before, partitions_after=after
    )
    change = only_change(report, ChangeKind.PARTITION_ADDED)
    assert change.severity is Severity.NON_BREAKING
    assert change.partition == "p2026_11"
    assert report.breaking == ()
    assert report.blocked == ()
    assert report.notification(created_at=EXEC_TS) is None


def test_a_dropped_partition_is_breaking() -> None:
    before = {SOURCE: monthly("start_time", "p2026_10", "p2026_11")}
    after = {SOURCE: monthly("start_time", "p2026_11")}
    report = diff_snapshots(
        snapshot(table()), snapshot(table()), partitions_before=before, partitions_after=after
    )
    change = only_change(report, ChangeKind.PARTITION_REMOVED)
    assert change.breaking
    assert change.partition == "p2026_10"


def test_a_partition_key_change_is_breaking() -> None:
    layer = credit_layer()
    before = {SOURCE: monthly("start_time", "p2026_11")}
    after = {SOURCE: monthly("insertion_time", "p2026_11")}
    report = diff_snapshots(
        snapshot(table()),
        snapshot(table()),
        layer=layer,
        partitions_before=before,
        partitions_after=after,
    )
    change = only_change(report, ChangeKind.PARTITION_KEY_CHANGED)
    assert change.breaking
    assert (change.before, change.after) == ("start_time", "insertion_time")
    assert "partitioning.key" in change.fix


def test_a_partition_strategy_and_granularity_change_are_breaking() -> None:
    before = {
        SOURCE: Partitioning(
            strategy=PartitionStrategy.RANGE,
            key=("start_time",),
            granularity=PartitionGranularity.MONTH,
        )
    }
    after = {
        SOURCE: Partitioning(
            strategy=PartitionStrategy.HASH,
            key=("start_time",),
            granularity=PartitionGranularity.DAY,
        )
    }
    report = diff_snapshots(
        snapshot(table()), snapshot(table()), partitions_before=before, partitions_after=after
    )
    kinds = {change.kind for change in report.changes}
    assert kinds == {
        ChangeKind.PARTITION_STRATEGY_CHANGED,
        ChangeKind.PARTITION_GRANULARITY_CHANGED,
    }
    assert all(change.breaking for change in report.changes)


def test_partition_metadata_appearing_or_vanishing_is_reported() -> None:
    report = diff_snapshots(
        snapshot(table()),
        snapshot(table()),
        partitions_after={SOURCE: monthly()},
    )
    assert only_change(report, ChangeKind.PARTITION_ADDED).severity is Severity.NON_BREAKING
    report = diff_snapshots(
        snapshot(table()), snapshot(table()), partitions_before={SOURCE: monthly()}
    )
    assert only_change(report, ChangeKind.PARTITION_REMOVED).breaking


def test_an_index_on_a_default_time_column_is_breaking_when_dropped() -> None:
    """A dropped index that carries the pruning guarantee is breaking; another one is not."""
    layer = SemanticLayer(
        version_hash="usage-layer",
        tables={
            "billing.usage_events": TableSemantics(
                time_columns={
                    "start_time": TimeColumn(role=TimeRole.EVENT_TIME, use_by_default=True)
                }
            )
        },
    )
    usage = "usage_events"
    before = snapshot(
        table(
            usage,
            columns=(
                column("start_time", native="timestamptz", normalized=NormalizedType.TIMESTAMP),
            ),
            indexes=(index("usage_start", "start_time"), index("usage_other", "start_time")),
        )
    )
    after = snapshot(
        table(
            usage,
            columns=(
                column("start_time", native="timestamptz", normalized=NormalizedType.TIMESTAMP),
            ),
            indexes=(index("usage_other", "start_time"),),
        )
    )
    report = diff_snapshots(before, after, layer=layer)
    change = only_change(report, ChangeKind.INDEX_REMOVED)
    assert change.breaking
    assert change.index == "usage_start"
    assert "tables.billing.usage_events.time_columns.start_time" in {
        ref.label for ref in change.refs
    }


# --------------------------------------------------------------------------------------------------
# The stage: proposals, notification, and a baseline that does not absorb the drift
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    opened = Store(tmp_path / "mdqm.sqlite")
    opened.initialize()
    try:
        yield opened
    finally:
        opened.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    semantic = tmp_path / "semantic"
    (semantic / "generated").mkdir(parents=True)
    return semantic


def write_layer(root: Path, layer: SemanticLayer) -> None:
    """Write the layer as one generated file, the way the pipeline's own writer would."""
    payload = layer.model_dump(mode="json", exclude={"version_hash"})
    (root / "generated" / "tables.yaml").write_text(json.dumps(payload), encoding="utf-8")


class Reader:
    """A catalog reader a test drives: one snapshot per source, counting how often it is read."""

    def __init__(self, snapshot_for: dict[str, CatalogSnapshot]) -> None:
        self.snapshots = dict(snapshot_for)
        self.reads = 0

    def __call__(self, source: str) -> CatalogSnapshot:
        self.reads += 1
        return self.snapshots[source]


def ctx_for(store: Store, root: Path) -> PipelineContext:
    return PipelineContext(store=store, exec_ts=EXEC_TS, semantic_root=root)


def test_first_run_records_the_baseline_and_asks_nothing(store: Store, root: Path) -> None:
    stage = DriftStage(extract=Reader({SOURCE: snapshot(table())}), sources=[SOURCE])
    ctx = ctx_for(store, root)
    result = run_pipeline([stage], ctx)
    report = result.reports[0]
    assert report.status is StageStatus.RAN
    assert report.counts == {"baselines": 1}
    assert read_baseline(store, SOURCE) is not None
    assert blocked_references(store) == ()
    assert read_notification(store, SOURCE) is None


def test_an_unchanged_catalog_asks_nothing_and_writes_nothing(store: Store, root: Path) -> None:
    """The acceptance for "an unchanged catalog produces no proposal"."""
    first = snapshot(table())
    write_baseline(store, first, at=EXEC_TS)
    queue = ReviewQueue(store, root)
    stage = DriftStage(extract=Reader({SOURCE: snapshot(table())}), sources=[SOURCE], queue=queue)
    result = run_pipeline([stage], ctx_for(store, root), force=True)
    assert result.reports[0].counts.get("changes", 0) == 0
    assert queue.pending() == ()
    assert read_notification(store, SOURCE) is None


def test_statistics_churn_raises_no_question(store: Store, root: Path) -> None:
    write_baseline(store, snapshot(table()), at=EXEC_TS)
    queue = ReviewQueue(store, root)
    stage = DriftStage(
        extract=Reader({SOURCE: snapshot(table(row_estimate=777_777))}),
        sources=[SOURCE],
        queue=queue,
    )
    run_pipeline([stage], ctx_for(store, root), force=True)
    assert queue.pending() == ()


def test_a_breaking_change_files_a_proposal_and_a_notification_and_keeps_the_baseline(
    store: Store, root: Path
) -> None:
    layer = credit_layer()
    write_layer(root, layer)
    before = snapshot(table(columns=(column("client_no"), column("credit_limit"))))
    write_baseline(store, before, at=EXEC_TS)
    queue = ReviewQueue(store, root)
    stage = DriftStage(
        extract=Reader({SOURCE: snapshot(table(columns=(column("client_no"),)))}),
        sources=[SOURCE],
        queue=queue,
    )
    result = run_pipeline([stage], ctx_for(store, root), force=True)
    report = result.reports[0]
    assert report.counts["breaking"] == 1
    assert report.counts["questions"] == 1

    pending = queue.pending()
    assert len(pending) == 1
    proposal = pending[0]
    assert proposal.section == "tables"
    assert proposal.key == "billing.accounts"
    assert proposal.field == "", "the patch IS the table's provenance: no nested field path"
    assert proposal.proposed_by == "pipeline/D10"
    assert any("measures.credit_limit_r5000" in line for line in proposal.evidence)

    notification = read_notification(store, SOURCE)
    assert notification is not None
    assert "breaking catalog change" in notification.subject
    assert "measures.credit_limit_r5000" in notification.body

    # The baseline did not advance: the next run still sees the same drift.
    assert read_baseline(store, SOURCE) == before
    assert "measures.credit_limit_r5000" in blocked_references(store, source=SOURCE)
    again = run_pipeline([stage], ctx_for(store, root), force=True)
    assert again.reports[0].counts["breaking"] == 1
    assert len(queue.pending()) == 1, "the same drift must not open a second question"


def test_approving_the_proposal_lets_the_baseline_advance(store: Store, root: Path) -> None:
    """The approval path leaves a curated file the loader accepts, then the drift is settled."""
    layer = credit_layer()
    write_layer(root, layer)
    write_baseline(store, snapshot(table()), at=EXEC_TS)
    queue = ReviewQueue(store, root)
    after = snapshot(table(columns=(column("client_no"),)))
    stage = DriftStage(extract=Reader({SOURCE: after}), sources=[SOURCE], queue=queue)
    run_pipeline([stage], ctx_for(store, root), force=True)
    proposal = queue.pending()[0]
    queue.approve(proposal.id, by="data-team")
    assert queue.get(proposal.id).status is ProposalStatus.APPROVED

    # A human decided; the next pass records the new catalog and blocks nothing new.
    stage = DriftStage(extract=Reader({SOURCE: after}), sources=[SOURCE], queue=queue)
    result = run_pipeline([stage], ctx_for(store, root), force=True)
    assert result.reports[0].counts["breaking"] == 1, "the catalog is still different"
    assert result.reports[0].counts["undecided"] == 0, "the approval settled it"
    assert read_baseline(store, SOURCE) == after, "a decided change lets the baseline advance"
    assert blocked_references(store, source=SOURCE) == ()


def test_a_non_breaking_change_advances_the_baseline_and_asks_nothing(
    store: Store, root: Path
) -> None:
    write_baseline(store, snapshot(table()), at=EXEC_TS)
    queue = ReviewQueue(store, root)
    after = snapshot(table(columns=(column("client_no"), column("credit_limit"), column("state"))))
    stage = DriftStage(extract=Reader({SOURCE: after}), sources=[SOURCE], queue=queue)
    result = run_pipeline([stage], ctx_for(store, root), force=True)
    assert result.reports[0].counts["non_breaking"] == 1
    assert queue.pending() == ()
    assert read_baseline(store, SOURCE) == after
    assert any("baseline advanced" in fix for fix in result.reports[0].applied_fixes)


def test_a_breaking_change_without_a_queue_is_still_reported_and_blocked(
    store: Store, root: Path
) -> None:
    write_baseline(store, snapshot(table()), at=EXEC_TS)
    stage = DriftStage(
        extract=Reader({SOURCE: snapshot(table(columns=(column("client_no"),)))}),
        sources=[SOURCE],
    )
    result = run_pipeline([stage], ctx_for(store, root), force=True)
    assert result.reports[0].counts["unfiled"] == 1
    assert read_notification(store, SOURCE) is not None
    assert read_baseline(store, SOURCE) == snapshot(table()), "the baseline did not absorb it"


def test_the_stage_hashes_the_catalog_so_a_change_is_never_skipped(
    store: Store, root: Path
) -> None:
    """inputs_hash reads the catalog, so a changed catalog cannot compare equal to the last run."""
    reader = Reader({SOURCE: snapshot(table())})
    stage = DriftStage(extract=reader, sources=[SOURCE])
    ctx = ctx_for(store, root)
    first = stage.inputs_hash(ctx)
    reader.snapshots[SOURCE] = snapshot(table(row_estimate=5))
    assert stage.inputs_hash(ctx) == first, "row_estimate is not an input"
    reader.snapshots[SOURCE] = snapshot(table(columns=(column("client_no"),)))
    assert stage.inputs_hash(ctx) != first, "a dropped column must change the hash"


def test_a_damaged_baseline_is_treated_as_a_first_run(store: Store, root: Path) -> None:
    with store.writing() as conn:
        conn.execute(
            "INSERT INTO mdqm_meta (key, value, updated_at) VALUES (?, ?, ?)",
            ("catalog_snapshot.billing", "{not json", EXEC_TS.isoformat()),
        )
    assert read_baseline(store, SOURCE) is None
    stage = DriftStage(extract=Reader({SOURCE: snapshot(table())}), sources=[SOURCE])
    result = run_pipeline([stage], ctx_for(store, root))
    assert result.reports[0].status is StageStatus.RAN
    assert read_baseline(store, SOURCE) is not None


def test_an_unloadable_layer_does_not_stop_the_diff(store: Store, root: Path) -> None:
    (root / "generated" / "broken.yaml").write_text("sources: 7\n", encoding="utf-8")
    stage = DriftStage(extract=Reader({SOURCE: snapshot(table())}), sources=[SOURCE])
    result = run_pipeline([stage], ctx_for(store, root))
    report = result.reports[0]
    assert report.status is StageStatus.RAN
    assert any("did not load" in line for line in report.skipped)


def test_the_stage_requires_a_source() -> None:
    with pytest.raises(ValueError, match="at least one source"):
        DriftStage(extract=Reader({}), sources=[])


def test_notification_payload_round_trips() -> None:
    report = diff_snapshots(snapshot(table()), snapshot())
    notification = report.notification(created_at=EXEC_TS)
    assert notification is not None
    payload = notification.payload()
    rebuilt = AdminNotification.from_payload(payload)
    assert rebuilt == notification
    assert payload["blocked"] == list(notification.blocked)


def test_stage_name_matches_the_design() -> None:
    assert STAGE_NAME == "D10"
    assert DriftStage(extract=Reader({SOURCE: snapshot(table())}), sources=[SOURCE]).name == "D10"


def test_blocked_and_notification_keys_are_namespaced() -> None:
    assert BLOCKED_PREFIX.startswith("drift_")
    assert NOTIFICATION_PREFIX.startswith("drift_")


def test_a_report_with_no_changes_describes_itself(store: Store) -> None:
    report = diff_snapshots(snapshot(table()), snapshot(table()))
    assert "no schema drift" in report.describe()
    assert blocked_references(store) == ()
