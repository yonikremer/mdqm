"""Column statistics: measurement, associations, the category gate, and the store.

The category gate has two paths that must not be confused. A complete reading rejects a value
the column never held, and it does not touch a source. A reading that missed a partition, or
that is only a sample, asks a probe first and rejects only when that probe says the value is
absent.
"""

from __future__ import annotations

from datetime import UTC, datetime, time

import duckdb
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlglot import exp

from mdqm.compiler import CompileContext, compile_spec
from mdqm.config.quiet import QuietWindow
from mdqm.config.settings import Settings
from mdqm.context.pack import ContextPack, PackItem, PackKind
from mdqm.semantic.models import SemanticLayer
from mdqm.spec.validate import parse_spec
from mdqm.sql.guard import guard
from mdqm.stats.association import choose_pairs, cramers_v, measure_association, mutual_information
from mdqm.stats.catalog import (
    parse_es_index_stats,
    parse_oracle_column_stats,
    parse_pg_class,
    parse_pg_stats,
    parse_sqlserver_column_stats,
)
from mdqm.stats.collect import collect_duckdb, collect_rows, distribution_drift
from mdqm.stats.dictionary import load_sentinels
from mdqm.stats.estimate import estimate_result_rows, narrow_window
from mdqm.stats.guard import Decision, decide, gate_spec, match_value
from mdqm.stats.measure import (
    ColumnRequest,
    is_categorical,
    measure_column,
    measure_distribution,
    top_values,
)
from mdqm.stats.models import (
    CategoryColumnStats,
    CollectionMethod,
    ObservedValue,
    PartitionExtent,
    PartitionStamp,
)
from mdqm.stats.plan import (
    collection_open,
    partitions_to_collect,
    unprofiled_partitions,
)
from mdqm.stats.similarity import equivalent, suggest
from mdqm.stats.sqlgen import existence_sql, nullif_expression, sample_sql, sentinel_sentence
from mdqm.stats.store import StatsStore
from mdqm.stats.surface import attach_briefs, column_line
from mdqm.store.db import Store

WHEN = datetime(2026, 11, 1, tzinfo=UTC)
ROWS = (
    {"status": "Active", "amount": 10, "note": "ok", "bucket": "2026-01"},
    {"status": "Active", "amount": 0, "note": "", "bucket": "2026-01"},
    {"status": "Closed", "amount": None, "note": "   ", "bucket": "2026-02"},
    {"status": None, "amount": 5, "note": "ok", "bucket": "2026-02"},
)


def _layer() -> SemanticLayer:
    return SemanticLayer.model_validate(
        {
            "version": 1,
            "value_maps": {
                "customer_status": {
                    "canonical": ["active", "churned"],
                    "synonyms": {"active": ["פעיל", "active"]},
                    "sources": {
                        "billing.accounts.state": {"Active": "active", "Closed": "churned"}
                    },
                }
            },
            "tables": {"billing.accounts": {"key": ["client_no"]}},
        }
    )


def _stats(**overrides: object) -> CategoryColumnStats:
    payload: dict[str, object] = {
        "source": "billing",
        "table": "accounts",
        "column": "state",
        "values": (
            ObservedValue(value="Active", count=64),
            ObservedValue(value="Closed", count=30),
        ),
        "other_count": 0,
        "rows_seen": 100,
        "collected_at": WHEN,
        "method": CollectionMethod.EXACT,
        "distribution_complete": True,
    }
    payload.update(overrides)
    return CategoryColumnStats.model_validate(payload)


class _Probe:
    def __init__(self, answer: bool | None) -> None:
        self.answer = answer
        self.calls: list[str] = []

    def exists(self, *, source: str, table: str, column: str, value: str) -> bool | None:
        self.calls.append(value)
        return self.answer


def _gate(value: str, stats: CategoryColumnStats, probe: _Probe | None = None):
    spec = parse_spec(
        {
            "from": "billing.accounts",
            "select": ["client_no"],
            "filters": [{"column": "state", "op": "eq", "value": value}],
            "limit": 5,
        }
    )
    return gate_spec(
        spec,
        layer=_layer(),
        columns={"billing.accounts": ("client_no", "state")},
        stats={"billing.accounts.state": stats},
        probe=probe,
    )


# --- measurement -----------------------------------------------------------------------


def test_column_shares_distinguish_null_zero_empty_and_whitespace() -> None:
    quality = measure_column(
        [None, 0, 0, "", "   ", False, "0", 5],
        ColumnRequest("amount", numeric=True, text=True),
        source="crm",
        table="customers",
        collected_at=WHEN,
        method=CollectionMethod.EXACT,
    )
    assert quality.null_share == pytest.approx(1 / 8)
    assert quality.zero_share == pytest.approx(2 / 8)
    assert quality.empty_share == pytest.approx(1 / 8)
    assert quality.whitespace_share == pytest.approx(1 / 8)
    assert quality.distinct_count == 6
    assert quality.minimum == "0"
    assert quality.maximum == "5"


def test_empty_column_has_no_share() -> None:
    quality = measure_column(
        [],
        ColumnRequest("amount", numeric=True),
        source="crm",
        table="customers",
        collected_at=WHEN,
        method=CollectionMethod.EXACT,
    )
    assert quality.null_share is None
    assert quality.rows_seen == 0


def test_false_is_not_zero_and_bool_is_not_the_integer() -> None:
    quality = measure_column(
        [False, True],
        ColumnRequest("flag", numeric=True),
        source="crm",
        table="customers",
        collected_at=WHEN,
        method=CollectionMethod.EXACT,
    )
    assert quality.zero_share == 0
    assert quality.distinct_count == 2


def test_top_values_break_ties_by_name_and_keep_the_tail() -> None:
    head, other = top_values({"b": 2, "a": 2, "c": 1}, 2)
    assert [item.value for item in head] == ["a", "b"]
    assert other == 1


def test_categorical_detection_bounds() -> None:
    assert is_categorical(distinct=3, non_null=1000)
    assert not is_categorical(distinct=0, non_null=10)
    assert not is_categorical(distinct=40, non_null=100)
    assert is_categorical(distinct=40, non_null=10_000)


def test_distribution_is_complete_only_when_the_table_was_covered() -> None:
    exact = measure_distribution(
        ["A", "A", "B"],
        source="crm",
        table="customers",
        column="status",
        collected_at=WHEN,
        method=CollectionMethod.EXACT,
        covers_table=True,
    )
    sampled = measure_distribution(
        ["A", "B"],
        source="crm",
        table="customers",
        column="status",
        collected_at=WHEN,
        method=CollectionMethod.SAMPLE,
        covers_table=False,
    )
    assert exact.distribution_complete
    assert not sampled.distribution_complete


def test_collect_rows_records_per_bucket_drift_and_association() -> None:
    collected = collect_rows(
        ROWS,
        (
            ColumnRequest("status", text=True),
            ColumnRequest("amount", numeric=True),
            ColumnRequest("note", text=True),
        ),
        source="billing",
        table="accounts",
        collected_at=WHEN,
        method=CollectionMethod.EXACT,
        bucket_column="bucket",
        co_used=(("status", "amount"),),
    )
    status = next(item for item in collected.quality if item.column == "status")
    assert status.null_share == pytest.approx(0.25)
    buckets = [item.bucket for item in collected.distributions if item.column == "status"]
    assert buckets == ["", "2026-01", "2026-02"]
    assert frozenset(
        (collected.associations[0].column_x, collected.associations[0].column_y)
    ) == frozenset(("status", "amount"))
    january = next(item for item in collected.distributions if item.bucket == "2026-01")
    february = next(
        item
        for item in collected.distributions
        if item.bucket == "2026-02" and item.column == "status"
    )
    moved = distribution_drift(january, february)
    assert moved


def test_duckdb_sample_records_the_rows_it_actually_read() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute("CREATE TABLE probe (status VARCHAR, amount INTEGER)")
    connection.executemany(
        "INSERT INTO probe VALUES (?, ?)",
        [("Active", index) for index in range(200)] + [("Closed", 0)],
    )
    collected = collect_duckdb(
        connection,
        "probe",
        (ColumnRequest("status", text=True), ColumnRequest("amount", numeric=True)),
        source="billing",
        table="accounts",
        collected_at=WHEN,
        sample_rows=30,
        seed=7,
    )
    assert collected.quality[0].method is CollectionMethod.SAMPLE
    assert collected.quality[0].sample_rows == 30
    assert collected.quality[0].rows_seen == 30


def test_duckdb_full_read_is_exact() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute("CREATE TABLE probe (status VARCHAR)")
    connection.executemany("INSERT INTO probe VALUES (?)", [("Active",), ("",), (None,)])
    collected = collect_duckdb(
        connection,
        "probe",
        (ColumnRequest("status", text=True),),
        source="billing",
        table="accounts",
        collected_at=WHEN,
    )
    status = collected.quality[0]
    assert status.method is CollectionMethod.EXACT
    assert status.null_share == pytest.approx(1 / 3)
    assert status.empty_share == pytest.approx(1 / 3)
    assert status.sample_rows is None


# --- association and pairs --------------------------------------------------------------


def test_cramers_v_is_zero_when_independent_and_one_when_determined() -> None:
    independent = [("A", "1"), ("A", "2"), ("B", "1"), ("B", "2")]
    determined = [("A", "1"), ("A", "1"), ("B", "2"), ("B", "2")]
    assert cramers_v(independent) == pytest.approx(0.0)
    assert cramers_v(determined) == pytest.approx(1.0)
    assert mutual_information(independent) == pytest.approx(0.0)
    assert mutual_information(determined) > 0


def test_choose_pairs_prefers_columns_mined_queries_used_together() -> None:
    columns = [("status", 3, True), ("amount", 100, False), ("region", 4, True)]
    pairs = choose_pairs(columns, co_used=(("region", "amount"),), max_pairs=1)
    assert pairs == (("region", "amount"),)


def test_numeric_association_reports_quantiles_per_category() -> None:
    association = measure_association(
        [("Active", 1), ("Active", 3), ("Closed", 0), ("Closed", None)],
        source="billing",
        table="accounts",
        column_x="status",
        column_y="amount",
        y_numeric=True,
        collected_at=WHEN,
        method=CollectionMethod.EXACT,
    )
    by_name = {item.category: item for item in association.by_category}
    assert by_name["Active"].mean == pytest.approx(2)
    assert by_name["Closed"].null_share == pytest.approx(0.5)
    assert by_name["Closed"].zero_share == pytest.approx(0.5)


@given(
    st.lists(st.tuples(st.sampled_from(["A", "B", "C"]), st.sampled_from(["x", "y"])), max_size=30)
)
@settings(max_examples=25, deadline=None)
def test_cramers_v_stays_inside_zero_and_one(pairs: list[tuple[str, str]]) -> None:
    score = cramers_v(pairs)
    assert 0.0 <= score <= 1.0


# --- similarity and the gate ------------------------------------------------------------


def test_equivalent_folds_case_whitespace_hebrew_and_transliteration() -> None:
    assert equivalent(" Active ", "active")
    assert equivalent("פָּעִיל", "פעיל")
    assert equivalent("שלום", "shalom")
    assert not equivalent("activ", "active")


def test_suggestions_are_deterministic_and_carry_frequency() -> None:
    observed = (("active", 64), ("churned", 30), ("suspended", 6))
    first = suggest("activ", observed, rows=100)
    second = suggest("activ", observed, rows=100)
    assert [item.value for item in first] == [item.value for item in second]
    assert first[0].value == "active"
    assert first[0].frequency() == "64 of 100, 64%"


def test_fresh_stats_reject_without_calling_the_probe() -> None:
    probe = _Probe(True)
    result = _gate("activ", _stats(), probe)
    assert result.error is not None
    assert result.error.code == "VALUE_NOT_IN_DATA"
    assert "64 of 100, 64%" in result.error.fix
    assert result.error.did_you_mean == ("active (64 of 100, 64%)",)
    assert probe.calls == []


def test_hebrew_synonym_of_a_stored_code_is_allowed() -> None:
    assert match_value(
        "פָּעִיל", _stats(), _layer().value_maps["customer_status"], "billing.accounts.state"
    )
    result = _gate("פעיל", _stats(), _Probe(False))
    assert result.ok
    assert result.error is None


def test_stale_partition_is_looked_up_before_a_rejection() -> None:
    stale = _stats(unprofiled_partitions=("2026-11",), distribution_complete=False)
    found = _Probe(True)
    allowed = _gate("activ", stale, found)
    assert allowed.ok
    assert found.calls
    assert "bounded lookup found" in allowed.notes[0]

    missed = _Probe(False)
    rejected = _gate("activ", stale, missed)
    assert rejected.error is not None
    assert rejected.error.code == "VALUE_NOT_IN_DATA"
    assert "bounded lookup found no row" in rejected.error.problem
    assert "2026-11" in rejected.error.problem

    unknown = _Probe(None)
    unverified = _gate("activ", stale, unknown)
    assert unverified.ok
    assert "could not run" in unverified.notes[0]


def test_stale_stats_without_a_probe_are_not_rejected() -> None:
    stale = _stats(distribution_complete=False)
    result = _gate("activ", stale, None)
    assert result.ok
    assert "not rejected" in result.notes[0]
    assert decide("activ", stale) is Decision.VERIFY
    assert decide("activ", _stats()) is Decision.REJECT


def test_a_value_in_the_tail_is_verified_rather_than_rejected() -> None:
    partial = _stats(other_count=5, distribution_complete=False)
    probe = _Probe(False)
    result = _gate("nope", partial, probe)
    assert probe.calls
    assert result.error is not None


# --- plan, catalog, sql, surface, store -------------------------------------------------


def test_incremental_plan_skips_unchanged_partitions() -> None:
    previous = (PartitionStamp(name="2026-01", rows=10, bytes=100),)
    current = (
        PartitionStamp(name="2026-01", rows=10, bytes=100),
        PartitionStamp(name="2026-02", rows=4, bytes=40),
    )
    plan = partitions_to_collect(current, previous)
    assert plan.collect == ("2026-02",)
    assert plan.unchanged == ("2026-01",)
    grown = partitions_to_collect((PartitionStamp(name="2026-01", rows=11, bytes=100),), previous)
    assert grown.collect == ("2026-01",)


def test_quiet_window_defaults_to_jerusalem_overnight() -> None:
    settings = Settings()
    noon = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)  # 15:00 in Jerusalem (IDT, UTC+3)
    night = datetime(2026, 6, 1, 20, 0, tzinfo=UTC)  # 23:00 in Jerusalem
    assert settings.stats_collection_open(noon) is False
    assert settings.stats_collection_open(night) is True
    assert settings.stats_collection_open(noon, force=True) is True
    closed = QuietWindow(tz="UTC", start=time(0, 0), end=time(0, 0))
    assert collection_open(noon, (closed,)) is False


def test_catalog_parsers_use_the_source_views() -> None:
    quality, categories = parse_pg_stats(
        [
            {
                "attname": "status",
                "null_frac": 0.1,
                "n_distinct": 2,
                "most_common_vals": ["A", "C"],
                "most_common_freqs": [0.6, 0.3],
                "histogram_bounds": None,
            }
        ],
        source="crm",
        table="customers",
        row_estimate=100,
        collected_at=WHEN,
    )
    assert quality[0].null_share == pytest.approx(0.1)
    assert quality[0].method is CollectionMethod.CATALOG
    assert categories[0].distribution_complete
    assert categories[0].values[0].count == 60
    negative, none = parse_pg_stats(
        [{"attname": "id", "null_frac": 0.0, "n_distinct": -0.5, "histogram_bounds": ["1", "9"]}],
        source="crm",
        table="customers",
        row_estimate=100,
        collected_at=WHEN,
    )
    assert negative[0].distinct_count == 50
    assert none == ()
    assert parse_pg_class({"reltuples": 12.4, "relpages": 2}) == (12, 16384)
    oracle = parse_oracle_column_stats(
        [{"column_name": "STATUS", "num_nulls": 5, "num_distinct": 3, "num_rows": 50}],
        source="erp",
        table="customers",
        collected_at=WHEN,
    )
    assert oracle[0].null_share == pytest.approx(0.1)
    server = parse_sqlserver_column_stats(
        [
            {
                "name": "status",
                "rows": 40,
                "null_count": 4,
                "distinct": 3,
                "minimum": "A",
                "maximum": "S",
            }
        ],
        source="erp",
        table="customers",
        collected_at=WHEN,
    )
    assert server[0].null_share == pytest.approx(0.1)
    extents = parse_es_index_stats(
        {
            "events-2026-02": {"primaries": {"docs": {"count": 3}, "store": {"size_in_bytes": 90}}},
            "events-2026-01": {"primaries": {"docs": {"count": 2}, "store": {"size_in_bytes": 80}}},
        },
        source="es",
        table="events",
        collected_at=WHEN,
    )
    assert [item.name for item in extents] == ["events-2026-01", "events-2026-02"]
    assert extents[0].rows == 2
    assert extents[1].bytes == 90


def test_existence_and_sample_sql_are_single_selects() -> None:
    probe = existence_sql(table="accounts", column="state", value="activ", dialect="postgres")
    assert guard(probe, dialect="postgres").ok
    assert "LIMIT" in probe.upper()
    sampled = sample_sql(
        table="accounts", columns=("state", "amount"), dialect="duckdb", rows=20, seed=3
    )
    assert "SAMPLE" in sampled
    assert "20" in sampled


def test_nullif_nests_empty_string_and_zero() -> None:
    expression = nullif_expression(exp.column("note"), ("0", ""))
    rendered = expression.sql(dialect="postgres")
    assert rendered.count("NULLIF") == 2
    assert sentinel_sentence("note", ("", "0")).startswith("column note is read through NULLIF")


def test_sentinels_yaml_names_the_file_when_a_key_is_wrong() -> None:
    parsed = load_sentinels("sentinels:\n  billing.accounts.note: ['', 0]\n", path="sentinels.yaml")
    assert parsed == {"billing.accounts.note": ("", "0")}
    with pytest.raises(ValueError, match="sentinels.yaml"):
        load_sentinels("sentinels:\n  note: ['']\n", path="sentinels.yaml")


def test_narrow_window_keeps_the_newest_partitions_under_the_cap() -> None:
    extents = (
        PartitionExtent(
            source="billing",
            table="usage_events",
            name="2026-01",
            rows=80,
            time_min="2026-01-01",
            time_max="2026-02-01",
            method=CollectionMethod.CATALOG,
            collected_at=WHEN,
        ),
        PartitionExtent(
            source="billing",
            table="usage_events",
            name="2026-02",
            rows=30,
            time_min="2026-02-01",
            time_max="2026-03-01",
            method=CollectionMethod.CATALOG,
            collected_at=WHEN,
        ),
    )
    window = narrow_window(extents, 40)
    assert window == ("2026-02-01", "2026-03-01", 30)
    assert (
        estimate_result_rows(table_rows=1000, selectivity=0.1, grouped=True, group_cardinality=4)
        == 4
    )


def test_unprofiled_partitions_are_the_ones_without_a_distribution() -> None:
    extents = (
        PartitionExtent(
            source="b",
            table="t",
            name="2026-02",
            rows=1,
            method=CollectionMethod.CATALOG,
            collected_at=WHEN,
        ),
        PartitionExtent(
            source="b",
            table="t",
            name="2026-01",
            rows=1,
            method=CollectionMethod.CATALOG,
            collected_at=WHEN,
        ),
    )
    assert unprofiled_partitions(extents, ("2026-01",)) == ("2026-02",)


def test_stats_round_trip_through_the_metadata_store(tmp_path) -> None:  # type: ignore[no-untyped-def]
    opened = Store(tmp_path / "meta.sqlite")
    opened.initialize()
    store = StatsStore(opened)
    collected = collect_rows(
        ROWS,
        (ColumnRequest("status", text=True), ColumnRequest("amount", numeric=True)),
        source="billing",
        table="accounts",
        collected_at=WHEN,
        method=CollectionMethod.EXACT,
        bucket_column="bucket",
    )
    store.record_quality(collected.quality)
    store.record_categories(collected.distributions)
    store.record_associations(collected.associations)
    store.record_extents(
        (
            PartitionExtent(
                source="billing",
                table="accounts",
                name="2026-01",
                rows=2,
                bytes=40,
                time_column="bucket",
                time_min="2026-01-01",
                time_max="2026-02-01",
                method=CollectionMethod.EXACT,
                collected_at=WHEN,
            ),
        )
    )
    assert store.latest_quality("billing", "accounts")[0].column == "amount"
    categories = store.latest_categories("billing", "accounts")
    assert {item.bucket for item in categories if item.column == "status"} == {
        "",
        "2026-01",
        "2026-02",
    }
    saved = store.latest_associations("billing", "accounts")[0]
    assert frozenset((saved.column_x, saved.column_y)) == frozenset(("status", "amount"))
    assert store.latest_extents("billing", "accounts")[0].rows == 2
    later = collected.quality[0].model_copy(
        update={"null_share": 0.5, "collected_at": datetime(2026, 11, 2, tzinfo=UTC)}
    )
    store.record_quality((later,))
    newest = next(
        item for item in store.latest_quality("billing", "accounts") if item.column == later.column
    )
    assert newest.null_share == pytest.approx(0.5)
    assert newest.collected_at == datetime(2026, 11, 2, tzinfo=UTC)


def test_briefs_attach_only_to_a_pack_that_already_named_the_table() -> None:
    pack = ContextPack(
        query="status",
        semantic_version="abc",
        budget_tokens=200,
        budget_clamped=False,
        tokens_estimated=10,
        items=(
            PackItem(
                kind=PackKind.TABLE,
                ref="billing.accounts",
                text="billing.accounts: key client_no",
                matched_terms=1,
            ),
        ),
        text="mdqm context\ntable:\n  billing.accounts: key client_no",
    )
    line = column_line(
        collect_rows(
            ROWS,
            (ColumnRequest("status", text=True),),
            source="billing",
            table="accounts",
            collected_at=WHEN,
            method=CollectionMethod.EXACT,
        ).quality[0]
    )
    attached = attach_briefs(pack, {"billing.accounts": line, "crm.customers": "customers: status"})
    assert "stats:" in attached.text
    assert "crm.customers" not in attached.text
    empty = ContextPack(
        query="weather",
        semantic_version="abc",
        budget_tokens=200,
        budget_clamped=False,
        tokens_estimated=0,
    )
    assert attach_briefs(empty, {"billing.accounts": line}).is_empty


def test_compiler_wraps_a_marked_column_in_nullif() -> None:
    layer = SemanticLayer.model_validate(
        {"version": 1, "tables": {"crm.customers": {"key": ["cust_id"]}}}
    )
    compiled = compile_spec(
        parse_spec({"from": "crm.customers", "select": ["credit_limit"], "limit": 5}),
        CompileContext(
            layer=layer,
            source="crm",
            dialect="postgres",
            exec_ts=WHEN,
            columns={"crm.customers": ("cust_id", "credit_limit")},
            sentinels={"crm.customers.credit_limit": ("", "0")},
        ),
    )
    assert compiled.ok
    assert compiled.fragment is not None
    sql = compiled.fragment.sql
    assert sql is not None
    assert "NULLIF" in sql
    assert any("NULLIF" in fix for fix in compiled.applied_fixes)
