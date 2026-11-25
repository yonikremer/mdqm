"""Tests for the shared domain models (task F02).

Table-driven: one sample per public model drives the cross-cutting invariants (frozen, no unknown
fields, JSON round-trip), then the specific contracts (stable snapshot hash, schema export, error
shape, and the validation each model owes its callers).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError

from mdqm.models import (
    PUBLIC_MODELS,
    Capabilities,
    Capability,
    CapabilitySource,
    CatalogSnapshot,
    Column,
    ErrorResponse,
    ForeignKey,
    Fragment,
    Index,
    Limits,
    NormalizedType,
    PartitionBound,
    PartitionGranularity,
    Partitioning,
    PartitionStrategy,
    PlanSummary,
    PruningProof,
    RawQuery,
    Table,
    TableKind,
    TableStats,
    export_json_schemas,
    schema_file_name,
    snapshot_hash,
)

EXEC_TS = datetime(2026, 11, 25, 10, 0, tzinfo=UTC)


def _column(name: str, native_type: str = "varchar(10)", nullable: bool = True) -> Column:
    return Column(
        name=name,
        native_type=native_type,
        normalized_type=NormalizedType.STRING,
        nullable=nullable,
        comment=f"{name} comment",
    )


def _table(
    name: str = "customers",
    schema_name: str | None = "crm",
    columns: tuple[Column, ...] | None = None,
) -> Table:
    return Table(
        name=name,
        schema_name=schema_name,
        kind=TableKind.TABLE,
        columns=columns
        if columns is not None
        else (_column("cust_id", "integer"), _column("status")),
        indexes=(Index(name="customers_pkey", columns=("cust_id",), unique=True, primary=True),),
        foreign_keys=(
            ForeignKey(columns=("cust_id",), ref_table="erp.customers", ref_columns=("cust_no",)),
        ),
        primary_key=("cust_id",),
        row_estimate=1200,
        comment="master customers",
    )


def _snapshot(
    tables: tuple[Table, ...] | None = None, captured_at: datetime = EXEC_TS
) -> CatalogSnapshot:
    return CatalogSnapshot(
        source="crm",
        captured_at=captured_at,
        tables=tables if tables is not None else (_table(),),
    )


SAMPLES: dict[str, BaseModel] = {
    "Capability": Capability(supported=True, detail="EXPLAIN (FORMAT JSON) is readable"),
    "Capabilities": Capabilities(
        source="crm",
        adapter="postgresql",
        version="16.4",
        retrieved_at=EXEC_TS,
        capabilities={
            "explain_plan": Capability(supported=True, detail="plan read as JSON"),
            "query_store": Capability(
                supported=False,
                detail="pg_stat_statements not loaded",
                source=CapabilitySource.PROBED,
            ),
            "structured_output": Capability(),
        },
    ),
    "Column": _column("status", "varchar(10)"),
    "Index": Index(
        name="customers_status_idx",
        columns=("status", "cust_id"),
        unique=False,
        index_type="btree",
        predicate=None,
    ),
    "ForeignKey": ForeignKey(
        name="accounts_customer_fk",
        columns=("cust_id",),
        ref_table="crm.customers",
        ref_columns=("cust_id",),
    ),
    "Table": _table(),
    "CatalogSnapshot": _snapshot(),
    "Partitioning": Partitioning(
        strategy=PartitionStrategy.RANGE,
        key=("start_time",),
        granularity=PartitionGranularity.MONTH,
        bounds=(
            PartitionBound(
                name="usage_events_2026_10",
                lower=datetime(2026, 10, 1, tzinfo=UTC),
                upper=datetime(2026, 11, 1, tzinfo=UTC),
                row_estimate=500,
            ),
        ),
        require_key_filter=True,
    ),
    "PartitionBound": PartitionBound(name="usage_events_2026_11", lower=EXEC_TS, upper=None),
    "TableStats": TableStats(
        source="billing",
        table="billing.accounts_hist",
        captured_at=EXEC_TS,
        row_estimate=900,
        rows_inserted=10,
        rows_updated=4,
        rows_deleted=0,
        hot_updates=2,
        rows_modified_since_analyze=14,
        last_analyze=EXEC_TS - timedelta(hours=6),
        last_autovacuum=None,
        checksum="sha256:abcd",
    ),
    "RawQuery": RawQuery(
        text="SELECT status, avg(credit_limit) FROM customers GROUP BY status",
        dialect="postgres",
        calls=1840,
        mean_time_ms=12.5,
        last_seen=EXEC_TS,
        user="analyst@example.invalid",
    ),
    "PlanSummary": PlanSummary(
        partitions_scanned=2,
        partitions_total=60,
        uses_index=True,
        indexes_used=("usage_events_pkey",),
        est_rows=120.0,
        est_cost=8.5,
        pruning_proof=PruningProof.PROVEN,
        raw='[{"Plan": {"Node Type": "Append"}}]',
    ),
    "Limits": Limits(timeout_s=30.0, max_rows=1000, max_bytes=5_000_000, max_partitions=4),
    "Fragment": Fragment(
        source="billing",
        dialect="postgres",
        sql="SELECT \"client_no\" FROM usage_events WHERE start_time >= TIMESTAMP '2026-11-01'",
        params={"region": "north", "limit": 100},
        tables=("billing.usage_events",),
        exec_ts=EXEC_TS,
    ),
    "ErrorResponse": ErrorResponse(
        code="PARTITION_SCAN_ALL",
        problem="usage_events is partitioned monthly on start_time and has no filter on it.",
        fix="add time: { mode: 'events_in_range', last: '30d' }",
        partitions_if_fixed="2 of 60",
    ),
}


# --------------------------------------------------------------------------------------------------
# Cross-cutting invariants: frozen, no unknown fields, JSON round-trip
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_model_is_frozen(name: str) -> None:
    model = SAMPLES[name]
    field = next(iter(type(model).model_fields))
    with pytest.raises(ValidationError):
        setattr(model, field, getattr(model, field))


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_model_rejects_unknown_fields(name: str) -> None:
    model = SAMPLES[name]
    payload = {**model.model_dump(), "unexpected_field": 1}
    with pytest.raises(ValidationError):
        type(model).model_validate(payload)


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_model_json_round_trip(name: str) -> None:
    model = SAMPLES[name]
    assert type(model).model_validate_json(model.model_dump_json()) == model


def test_public_models_cover_every_sample_and_are_exported() -> None:
    assert {model.__name__ for model in PUBLIC_MODELS} == set(SAMPLES)
    import mdqm.models as models

    for exported in models.__all__:
        assert hasattr(models, exported), exported


# --------------------------------------------------------------------------------------------------
# snapshot_hash: order independent, structural only
# --------------------------------------------------------------------------------------------------


def test_snapshot_hash_is_order_independent() -> None:
    left_columns = (
        _column("cust_id", "integer"),
        _column("status"),
        _column("credit_limit", "numeric"),
    )
    right_columns = (_column("client_no"), _column("state"), _column("credit_limit", "numeric"))
    first = _snapshot(
        tables=(
            _table(columns=left_columns),
            _table(name="accounts", schema_name="billing", columns=right_columns),
        )
    )
    shuffled = _snapshot(
        tables=(
            _table(name="accounts", schema_name="billing", columns=tuple(reversed(right_columns))),
            _table(columns=tuple(reversed(left_columns))),
        )
    )
    assert snapshot_hash(first) == snapshot_hash(shuffled)


def test_snapshot_hash_ignores_volatile_fields() -> None:
    base = _snapshot()
    later = _snapshot(captured_at=EXEC_TS + timedelta(days=3))
    reordered = _snapshot(
        tables=(
            Table.model_validate(
                {
                    **_table().model_dump(),
                    "columns": list(reversed(_table().columns)),
                    "row_estimate": 999_999,
                }
            ),
        )
    )
    assert snapshot_hash(base) == snapshot_hash(later) == snapshot_hash(reordered)


@pytest.mark.parametrize(
    "change",
    [
        {"comment": "renamed in a later release"},
        {"columns": (_column("status", "text"),)},
        {"indexes": ()},
        {"foreign_keys": ()},
        {"primary_key": ()},
        {"kind": TableKind.VIEW},
    ],
)
def test_snapshot_hash_changes_with_structure(change: dict[str, Any]) -> None:
    changed = Table.model_validate({**_table().model_dump(), **change})
    assert snapshot_hash(_snapshot()) != snapshot_hash(_snapshot(tables=(changed,)))


@settings(max_examples=25, deadline=None)
@given(
    columns=st.lists(
        st.builds(
            Column,
            name=st.text(alphabet="abcdefgh", min_size=1, max_size=6),
            native_type=st.just("integer"),
            normalized_type=st.just(NormalizedType.INTEGER),
        ),
        max_size=4,
        unique_by=lambda column: column.name,
    ),
    reverse=st.booleans(),
)
def test_snapshot_hash_is_invariant_under_column_shuffle(
    columns: list[Column], reverse: bool
) -> None:
    ordered = tuple(reversed(columns)) if reverse else tuple(columns)
    left = _snapshot(tables=(Table(name="t", columns=tuple(columns)),))
    right = _snapshot(tables=(Table(name="t", columns=ordered),))
    assert snapshot_hash(left) == snapshot_hash(right)


# --------------------------------------------------------------------------------------------------
# Schema export
# --------------------------------------------------------------------------------------------------


def test_export_json_schemas_writes_one_valid_schema_per_model(tmp_path: Path) -> None:
    written = export_json_schemas(tmp_path)
    assert [path.name for path in written] == [schema_file_name(model) for model in PUBLIC_MODELS]
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        schema_file_name(model) for model in PUBLIC_MODELS
    )
    for model, path in zip(PUBLIC_MODELS, written, strict=True):
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        assert schema["title"] == model.__name__
        assert set(schema["properties"]) == set(model.model_fields)
        assert set(schema.get("required", ())) <= set(schema["properties"])
        assert schema["additionalProperties"] is False


def test_export_json_schemas_is_deterministic(tmp_path: Path) -> None:
    first = export_json_schemas(tmp_path / "a")
    second = export_json_schemas(tmp_path / "b")
    assert [p.read_bytes() for p in first] == [p.read_bytes() for p in second]


def test_schema_file_name_is_snake_case() -> None:
    assert schema_file_name(CatalogSnapshot) == "catalog_snapshot.json"
    assert schema_file_name(Limits) == "limits.json"


# --------------------------------------------------------------------------------------------------
# ErrorResponse: the exact shape from docs/DESIGN.md
# --------------------------------------------------------------------------------------------------


def test_error_response_serialises_to_design_shape() -> None:
    error = ErrorResponse(
        code="PARTITION_SCAN_ALL",
        problem=(
            "usage_events is partitioned monthly on start_time and has no filter on it. "
            "The whole table would be scanned"
        ),
        fix="add time: { mode: 'events_in_range', last: '30d' }",
        partitions_if_fixed="2 of 60",
    )
    assert json.loads(error.model_dump_json()) == {
        "status": "error",
        "code": "PARTITION_SCAN_ALL",
        "problem": (
            "usage_events is partitioned monthly on start_time and has no filter on it. "
            "The whole table would be scanned"
        ),
        "fix": "add time: { mode: 'events_in_range', last: '30d' }",
        "partitions_if_fixed": "2 of 60",
    }


def test_error_response_omits_unset_optionals() -> None:
    error = ErrorResponse(code="WRITE_BLOCKED", problem="DELETE is not allowed.", fix="use query")
    assert error.model_dump() == {
        "status": "error",
        "code": "WRITE_BLOCKED",
        "problem": "DELETE is not allowed.",
        "fix": "use query",
    }
    assert "null" not in error.model_dump_json()


def test_error_response_optional_fields_appear_when_set() -> None:
    payload = json.loads(
        ErrorResponse(
            code="AMBIGUOUS_VALUE",
            problem="'active' matches two values.",
            fix="use one of valid_values",
            valid_values=("active", "churned"),
            did_you_mean=("status", "state"),
        ).model_dump_json()
    )
    assert payload["valid_values"] == ["active", "churned"]
    assert payload["did_you_mean"] == ["status", "state"]


def test_error_response_status_is_fixed() -> None:
    with pytest.raises(ValidationError):
        ErrorResponse.model_validate({"status": "ok", "code": "X", "problem": "p", "fix": "f"})


# --------------------------------------------------------------------------------------------------
# Capabilities: tri-state and fallback reporting
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("supported", "state", "supports", "is_unknown"),
    [
        (True, "supported", True, False),
        (False, "unsupported", False, False),
        (None, "unknown", False, True),
    ],
)
def test_capability_tristate(
    supported: bool | None, state: str, supports: bool, is_unknown: bool
) -> None:
    capability = Capability(supported=supported)
    assert capability.state == state
    capabilities = Capabilities(
        source="es",
        adapter="elastic",
        retrieved_at=EXEC_TS,
        capabilities={"explain_plan": capability},
    )
    assert capabilities.supports("explain_plan") is supports
    assert capabilities.is_unknown("explain_plan") is is_unknown


def test_missing_capability_reads_as_unknown_not_as_failure() -> None:
    capabilities = Capabilities(source="api", adapter="openapi", retrieved_at=EXEC_TS)
    assert capabilities.supports("partition_metadata") is False
    assert capabilities.is_unknown("partition_metadata") is True
    assert capabilities.get("partition_metadata").state == "unknown"


def test_capability_report_lists_fallbacks() -> None:
    capabilities = Capabilities(
        source="crm",
        adapter="postgresql",
        retrieved_at=EXEC_TS,
        capabilities={
            "explain_plan": Capability(supported=True),
            "query_store": Capability(supported=False, detail="extension not loaded"),
            "mutability_counters": Capability(),
        },
    )
    assert capabilities.unsupported() == ("query_store",)
    assert capabilities.unknown() == ("mutability_counters",)


def test_override_is_recorded_as_override() -> None:
    capability = Capability(
        supported=True, source=CapabilitySource.OVERRIDE, detail="admin override"
    )
    assert capability.source is CapabilitySource.OVERRIDE
    assert capability.state == "supported"


# --------------------------------------------------------------------------------------------------
# Timestamps: aware only, normalised to UTC
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: TableStats.model_validate(
            {"source": "s", "table": "t", "captured_at": "2026-11-25T10:00:00"}
        ),
        lambda: RawQuery.model_validate(
            {"text": "select 1", "dialect": "postgres", "last_seen": "2026-11-25T10:00:00"}
        ),
        lambda: CatalogSnapshot.model_validate(
            {"source": "s", "captured_at": "2026-11-25T10:00:00"}
        ),
        lambda: Fragment.model_validate(
            {
                "source": "s",
                "dialect": "postgres",
                "sql": "SELECT 1",
                "exec_ts": "2026-11-25T10:00:00",
            }
        ),
    ],
)
def test_naive_datetime_is_rejected(build: Callable[[], object]) -> None:
    with pytest.raises(ValidationError):
        build()


def test_offsets_are_normalised_to_utc() -> None:
    jerusalem = datetime(2026, 11, 25, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    stats = TableStats(source="s", table="t", captured_at=jerusalem)
    assert stats.captured_at == EXEC_TS
    assert stats.captured_at.utcoffset() == timedelta(0)
    assert TableStats.model_validate_json(stats.model_dump_json()) == stats


# --------------------------------------------------------------------------------------------------
# Per-model validation
# --------------------------------------------------------------------------------------------------


def test_table_qualified_name_and_lookup() -> None:
    table = _table()
    assert table.qualified_name == "crm.customers"
    assert Table(name="plain").qualified_name == "plain"
    assert table.column("cust_id") is not None
    assert table.column("missing") is None
    snapshot = _snapshot(tables=(table,))
    assert snapshot.table("crm.customers") is table
    assert snapshot.table("customers") is table
    assert snapshot.table("nope") is None


def test_index_requires_columns() -> None:
    with pytest.raises(ValidationError):
        Index(name="idx", columns=())


@pytest.mark.parametrize(
    ("columns", "ref_columns"),
    [(("a", "b"), ("a",)), (("a",), ("a", "b"))],
)
def test_foreign_key_sides_must_match(
    columns: tuple[str, ...], ref_columns: tuple[str, ...]
) -> None:
    with pytest.raises(ValidationError):
        ForeignKey(columns=columns, ref_table="t", ref_columns=ref_columns)


@pytest.mark.parametrize(
    "payload",
    [
        {"strategy": PartitionStrategy.RANGE.value},
        {"strategy": PartitionStrategy.HASH.value},
        {"strategy": PartitionStrategy.LIST.value},
        {"strategy": PartitionStrategy.EXPRESSION.value},
        {
            "strategy": PartitionStrategy.RANGE.value,
            "key": ["start_time"],
            "expression": "date_trunc('month', start_time)",
        },
        {"strategy": PartitionStrategy.NONE.value, "key": ["start_time"]},
    ],
)
def test_partitioning_shape_is_checked(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Partitioning.model_validate(payload)


def test_partitioning_none_and_total() -> None:
    assert Partitioning(strategy=PartitionStrategy.NONE).partitions_total is None
    monthly = SAMPLES["Partitioning"]
    assert isinstance(monthly, Partitioning)
    assert monthly.partitions_total == 1
    assert Partitioning(strategy=PartitionStrategy.HASH, key=("cust_id",)).partitions_total is None


def test_plan_summary_rejects_impossible_counts() -> None:
    with pytest.raises(ValidationError):
        PlanSummary(partitions_scanned=7, partitions_total=3)


@pytest.mark.parametrize(
    "counters",
    [
        {"timeout_s": 0},
        {"timeout_s": -1},
        {"max_rows": 0},
        {"max_bytes": 0},
        {"max_partitions": 0},
    ],
)
def test_limits_must_be_positive(counters: dict[str, float]) -> None:
    payload: dict[str, Any] = {
        "timeout_s": 30.0,
        "max_rows": 1000,
        "max_bytes": 1024,
        "max_partitions": 2,
    }
    payload.update(counters)
    with pytest.raises(ValidationError):
        Limits.model_validate(payload)


def test_fragment_carries_exactly_one_payload() -> None:
    common: dict[str, Any] = {"source": "billing", "dialect": "postgres", "exec_ts": EXEC_TS}
    assert Fragment(sql="SELECT 1", **common).kind == "sql"
    assert Fragment(dsl={"query": {"match_all": {}}}, **{**common, "dialect": "dsl"}).kind == "dsl"
    assert Fragment(request={"path": "/rates"}, **{**common, "dialect": "dsl"}).kind == "request"
    with pytest.raises(ValidationError):
        Fragment(**common)
    with pytest.raises(ValidationError):
        Fragment(sql="SELECT 1", dsl={"query": {}}, **common)


@pytest.mark.parametrize(
    "payload",
    [
        {"path": "/rates", "at": EXEC_TS},
        {"path": "/rates", "rows": [(1, 2)]},
        {"path": "/rates", "score": float("nan")},
        {"path": "/rates", 3: "non-string key"},
    ],
)
def test_fragment_rejects_values_that_cannot_round_trip(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Fragment(source="api_fx", dialect="dsl", request=payload, exec_ts=EXEC_TS)


def test_fragment_params_hold_json_scalars() -> None:
    fragment = Fragment(
        source="billing",
        dialect="postgres",
        sql="SELECT 1",
        params={"client_no": "0000001234", "limit": 10, "ratio": 0.5, "flag": True, "none": None},
        exec_ts=EXEC_TS,
    )
    assert Fragment.model_validate_json(fragment.model_dump_json()) == fragment


def test_table_stats_changes_needs_every_counter() -> None:
    complete = TableStats(
        source="s", table="t", captured_at=EXEC_TS, rows_inserted=3, rows_updated=2, rows_deleted=1
    )
    assert complete.changes == 6
    partial = TableStats(source="s", table="t", captured_at=EXEC_TS, rows_inserted=3)
    assert partial.changes is None
