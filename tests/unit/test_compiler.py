"""Tests for the single-source compiler (task C09).

Four kinds of evidence, because "it compiles" is not one of them:

* **golden SQL** for every ``valid`` spec in ``examples/query_specs.json`` — a change to the emitted
  statement shows up as a reviewable diff in ``tests/data/golden/``;
* **the guard and the frozen clock** on every compiled fragment, so the SQL this module produces is
  held to the same read-only and determinism rules as SQL an agent writes;
* **table-driven refusals**, so every error an agent can hit names the path and the fix;
* **end to end on the DuckDB fake adapter**, where the compiled SQL really runs and its rows are
  compared with the answers task E01 derived independently (``tests/eval/questions.yaml``):
  the 5000 half-up rounding, the erp ``limit_k * 10`` form, a Hebrew label, a partition window and
  the approved cross-source join.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pytest
from ruamel.yaml import YAML

from mdqm.compiler import CompileContext, CompiledQuery, compile_spec
from mdqm.errors import ErrorCode
from mdqm.models import Fragment, Limits
from mdqm.semantic.loader import load
from mdqm.semantic.models import SemanticLayer
from mdqm.spec.models import QuerySpec
from mdqm.spec.validate import parse_spec
from mdqm.sql.exec_ts import rewrite_sql
from mdqm.sql.guard import guard
from mdqm.testing.fake_adapter import FakeAdapter, table_from_rows
from mdqm.testing.world import World, WorldConfig, build_world

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "examples" / "query_specs.json"
LAYER_FILE = Path(__file__).resolve().parents[1] / "data" / "semantic" / "compiler_semantic.yaml"
QUESTIONS = ROOT / "tests" / "eval" / "questions.yaml"

#: One execution timestamp for every compilation in this module (C02: one value per execution).
EXEC_TS = datetime(2026, 11, 25, 15, 0, tzinfo=UTC)

#: The catalog as the adapters report it. Only these spellings may reach the SQL.
COLUMNS: Mapping[str, tuple[str, ...]] = {
    "crm.customers": ("cust_id", "status", "credit_limit", "region"),
    "billing.accounts": ("client_no", "state", "credit_limit", "region", "opened_at"),
    "billing.usage_events": ("event_id", "client_no", "units", "start_time", "insertion_time"),
    "erp.customers": ("cust_no", "status_code", "limit_k", "name"),
}

#: The valid example specs, loaded once: their names become golden file names and test ids.
VALID_EXAMPLES: tuple[Mapping[str, Any], ...] = tuple(
    json.loads(SPECS.read_text(encoding="utf-8"))["valid"]
)
EXAMPLE_IDS = [entry["name"] for entry in VALID_EXAMPLES]


@pytest.fixture(scope="module")
def layer() -> SemanticLayer:
    """The test semantic layer, loaded through the real loader (so its validation runs too)."""
    return load(LAYER_FILE)


@pytest.fixture(scope="module")
def questions() -> Mapping[str, Any]:
    """The task E01 question set: independently derived answers, and the world they belong to."""
    document = YAML(typ="safe").load(QUESTIONS.read_text(encoding="utf-8"))
    return cast(Mapping[str, Any], document)


@pytest.fixture(scope="module")
def world(questions: Mapping[str, Any]) -> World:
    """The seeded synthetic organisation the E01 answers are computed on (task D02)."""
    config = dict(cast(Mapping[str, Any], questions["world"]))
    anchor = config["anchor"]
    return build_world(
        WorldConfig(
            seed=int(config["seed"]),
            customers=int(config["customers"]),
            months=int(config["months"]),
            events_per_customer=int(config["events_per_customer"]),
            anchor=anchor if isinstance(anchor, datetime) else datetime.fromisoformat(str(anchor)),
            max_lag_seconds=int(config["max_lag_seconds"]),
        )
    )


# --- helpers -----------------------------------------------------------------------------


def context(
    layer: SemanticLayer,
    *,
    source: str,
    dialect: str = "postgres",
    exec_ts: datetime = EXEC_TS,
    **overrides: Any,
) -> CompileContext:
    """A compile context with the test catalog, overridable per test."""
    return CompileContext(
        layer=layer,
        source=source,
        dialect=dialect,
        exec_ts=exec_ts,
        columns=COLUMNS,
        **overrides,
    )


def compiled(
    layer: SemanticLayer, spec: Mapping[str, Any] | QuerySpec, **kwargs: Any
) -> CompiledQuery:
    """Parse and compile one spec mapping, for a source derived from its ``from`` table."""
    parsed = spec if isinstance(spec, QuerySpec) else parse_spec(spec)
    return compile_spec(parsed, context(layer, **kwargs))


def fragment_of(result: CompiledQuery) -> Fragment:
    """The fragment, asserting the compilation succeeded (a readable failure when it did not)."""
    assert result.error is None, f"unexpected refusal: {result.error.problem}"
    assert result.fragment is not None
    return result.fragment


def e01_spec(questions: Mapping[str, Any], question_id: str) -> QuerySpec:
    """The canonical spec of one E01 question, validated by the real spec parser."""
    for entry in cast(Sequence[Mapping[str, Any]], questions["questions"]):
        if entry["id"] == question_id:
            return parse_spec(cast(Mapping[str, Any], entry["spec"]))
    raise AssertionError(f"question {question_id!r} is not in {QUESTIONS.name}")


def e01_answer(questions: Mapping[str, Any], question_id: str) -> Mapping[str, Any]:
    """The expected answer of one E01 question, as data."""
    for entry in cast(Sequence[Mapping[str, Any]], questions["questions"]):
        if entry["id"] == question_id:
            return cast(Mapping[str, Any], entry["answer"])
    raise AssertionError(f"question {question_id!r} is not in {QUESTIONS.name}")


def _rows(stream: Iterator[pa.RecordBatch]) -> pa.Table:
    return pa.Table.from_batches(list(stream))


# --- golden SQL --------------------------------------------------------------------------


@pytest.mark.parametrize("case", VALID_EXAMPLES, ids=EXAMPLE_IDS)
def test_every_valid_example_compiles_to_golden_sql(
    case: Mapping[str, Any], layer: SemanticLayer, golden: Any
) -> None:
    """Each valid example compiles for postgres, and the statement is committed as a golden file."""
    result = compiled(layer, cast(Mapping[str, Any], case["spec"]), source="billing")
    fragment = fragment_of(result)
    assert fragment.dialect == "postgres"
    golden(f"{case['name']}.sql", fragment.sql or "")


@pytest.mark.parametrize("case", VALID_EXAMPLES, ids=EXAMPLE_IDS)
def test_every_compiled_fragment_passes_the_guard(
    case: Mapping[str, Any], layer: SemanticLayer
) -> None:
    """The SQL we compile is guarded exactly like SQL an agent writes (C01)."""
    fragment = fragment_of(compiled(layer, cast(Mapping[str, Any], case["spec"]), source="billing"))
    result = guard(fragment.sql or "", "postgres")
    assert result.ok, result.error.problem if result.error else "guard refused the SQL"


@pytest.mark.parametrize("case", VALID_EXAMPLES, ids=EXAMPLE_IDS)
def test_every_compiled_fragment_has_no_now_like_function(
    case: Mapping[str, Any], layer: SemanticLayer
) -> None:
    """``exec_ts`` is fully substituted: nothing in the statement reads a source's own clock."""
    fragment = fragment_of(compiled(layer, cast(Mapping[str, Any], case["spec"]), source="billing"))
    rewritten = rewrite_sql(fragment.sql or "", dialect="postgres", exec_ts=EXEC_TS)
    assert rewritten.replacements == ()
    folded = (fragment.sql or "").upper()
    for now_like in ("NOW(", "CURRENT_TIMESTAMP", "CURRENT_DATE", "LOCALTIMESTAMP"):
        assert now_like not in folded


def test_compile_is_deterministic(layer: SemanticLayer) -> None:
    """Same spec, same context, byte-identical SQL: a diff can only come from a real change."""
    for case in VALID_EXAMPLES:
        spec = cast(Mapping[str, Any], case["spec"])
        first = fragment_of(compiled(layer, spec, source="billing"))
        second = fragment_of(compiled(layer, spec, source="billing"))
        assert first.sql == second.sql, case["name"]
        assert first.tables == second.tables
        assert first.model_dump() == second.model_dump()


# --- refusals ----------------------------------------------------------------------------


def test_unapproved_join_names_the_approved_on_clause(layer: SemanticLayer) -> None:
    """A column that only an approved join reaches is reported with that join's full ON clause."""
    spec = {
        "from": "usage_events",
        "select": [{"measure": "usage_units", "agg": "sum"}],
        "filters": [{"column": "customer_status", "op": "eq", "value": "פעיל"}],
        "time": {"mode": "events_in_range", "from": "2025-11-01", "to": "2026-01-01"},
    }
    result = compiled(layer, spec, source="billing")
    assert result.error is not None
    assert result.error.code == ErrorCode.JOIN_NOT_APPROVED.value
    assert result.fragment is None
    assert '"billing.usage_events"' in result.error.problem
    assert '"crm.customers"' in result.error.problem
    assert result.error.fix is not None
    assert "LPAD(CAST(customers.cust_id AS VARCHAR), 10, '0') = usage_events.client_no" in (
        result.error.fix
    )


@pytest.mark.parametrize(
    ("spec", "source", "code", "needle"),
    [
        (
            {"from": "nowhere", "select": ["event_id"]},
            "billing",
            ErrorCode.UNKNOWN_TABLE,
            '"nowhere"',
        ),
        (
            {"from": "usage_events", "select": ["no_such_column"]},
            "billing",
            ErrorCode.UNKNOWN_COLUMN,
            '"no_such_column"',
        ),
        (
            {"from": "customers", "select": ["region"]},
            "crm",
            ErrorCode.INVALID_SPEC,
            "crm.customers",
        ),
        (
            {"from": "usage_events", "select": ["event_id"], "joins": ["invented_join"]},
            "billing",
            ErrorCode.JOIN_NOT_APPROVED,
            '"invented_join"',
        ),
        (
            {"from": "usage_events", "select": [{"measure": "nope", "agg": "sum"}]},
            "billing",
            ErrorCode.INVALID_SPEC,
            "nope",
        ),
        (
            {
                "from": "usage_events",
                "select": [{"measure": "usage_units", "agg": "sum"}],
                "filters": [
                    {
                        "column": "customer_status",
                        "op": "range",
                        "value": ["2025-01-01", "2025-06-01"],
                    }
                ],
                "joins": ["usage_to_customers"],
            },
            "billing",
            ErrorCode.INVALID_SPEC,
            "categorical",
        ),
        (
            {
                "from": "usage_events",
                "select": [{"measure": "usage_units", "agg": "sum"}],
                "filters": [{"column": "customer_status", "op": "eq", "value": "אין ערך כזה"}],
                "joins": ["usage_to_customers"],
                "time": {"mode": "events_in_range", "last": "30d"},
            },
            "billing",
            ErrorCode.UNKNOWN_VALUE,
            "filters.0.value",
        ),
    ],
)
def test_refusals_name_the_path_and_the_fix(
    layer: SemanticLayer,
    spec: Mapping[str, Any],
    source: str,
    code: ErrorCode,
    needle: str,
) -> None:
    """Every refusal is a coded ``ErrorResponse`` whose text names the offending name."""
    result = compiled(layer, spec, source=source)
    assert result.error is not None, "expected a refusal"
    assert result.error.code == code.value
    assert needle in f"{result.error.problem} {result.error.fix}"
    assert result.fragment is None


def test_missing_partition_filter_is_refused_without_sql(layer: SemanticLayer) -> None:
    """No window on the partition key: refuse (C08), and produce no statement at all."""
    spec = {
        "from": "usage_events",
        "select": [{"measure": "usage_units", "agg": "count"}],
        "limit": 10,
    }
    result = compiled(layer, spec, source="billing")
    assert result.error is not None
    assert result.error.code == ErrorCode.PARTITION_SCAN_ALL.value
    assert result.fragment is None


def test_cross_source_anchor_is_refused(layer: SemanticLayer) -> None:
    """A fragment targets one source: the ``from`` table must belong to it."""
    result = compiled(layer, {"from": "crm.customers", "select": ["region"]}, source="billing")
    assert result.error is not None
    assert result.error.code == ErrorCode.INVALID_SPEC.value
    assert "crm" in result.error.problem


def test_unsupported_dialect_is_a_programming_error(layer: SemanticLayer) -> None:
    """A dialect the exec_ts rewriter cannot handle is our bug: it raises, it does not answer."""
    with pytest.raises(ValueError, match="unsupported target dialect"):
        compiled(
            layer,
            {"from": "usage_events", "select": ["event_id"]},
            source="billing",
            dialect="cosmosdb",
        )


# --- what the SQL says -------------------------------------------------------------------


def test_row_cap_is_injected_and_reported(layer: SemanticLayer) -> None:
    """A spec without a limit still cannot read a whole table, and the cap is reported."""
    result = compiled(
        layer,
        {
            "from": "usage_events",
            "select": ["event_id"],
            "time": {"mode": "events_in_range", "last": "7d"},
        },
        source="billing",
        max_rows=250,
    )
    fragment = fragment_of(result)
    assert (fragment.sql or "").endswith("LIMIT 250")
    assert any("row cap of 250" in fix for fix in result.applied_fixes)


def test_trailing_window_uses_exec_ts_not_the_clock(layer: SemanticLayer) -> None:
    """``last: 7d`` is two absolute literals derived from ``exec_ts``: re-runs see one window."""
    result = compiled(
        layer,
        {
            "from": "usage_events",
            "select": ["event_id"],
            "time": {"mode": "events_in_range", "last": "7d"},
        },
        source="billing",
    )
    sql = fragment_of(result).sql or ""
    assert "2026-11-18 15:00:00" in sql
    assert "2026-11-25 15:00:00" in sql


def test_new_since_adds_the_derived_event_time_bound(layer: SemanticLayer) -> None:
    """The ingestion window keeps its implied event-time bound, so late rows are not missed."""
    result = compiled(
        layer,
        {
            "from": "usage_events",
            "select": ["event_id"],
            "time": {"mode": "new_since", "since": "2026-11-25T14:00:00Z"},
        },
        source="billing",
    )
    sql = fragment_of(result).sql or ""
    assert "insertion_time >=" in sql
    # max_lag is 6h, so the indexed event column is bounded at 08:00 that day.
    assert "start_time >= CAST('2026-11-25 08:00:00' AS TIMESTAMP)" in sql


def test_measure_normalisation_is_reported(layer: SemanticLayer) -> None:
    """The 5000 half-up rounding is in the SQL *and* in ``applied_fixes``, never silent."""
    result = compiled(
        layer,
        {
            "from": "accounts",
            "select": [{"measure": "credit_limit_r5000", "agg": "avg"}],
        },
        source="billing",
    )
    sql = fragment_of(result).sql or ""
    assert "FLOOR(ABS(CAST(credit_limit AS DECIMAL(38, 6))" in sql
    assert len(result.normalizations) == 1
    applied = result.normalizations[0]
    assert applied.column == "billing.accounts.credit_limit"
    assert applied.identity is False
    assert applied.mode is not None and applied.mode.value == "half_up"


@pytest.mark.parametrize(
    ("spec", "source", "needles"),
    [
        (
            {"from": "crm.customers", "select": [{"measure": "credit_limit_r5000", "agg": "avg"}]},
            "crm",
            ["AVG(credit_limit)"],
        ),
        (
            {"from": "erp.customers", "select": [{"measure": "credit_limit_r5000", "agg": "sum"}]},
            "erp",
            ["SUM(limit_k * 10)"],
        ),
        (
            {
                "from": "crm.customers",
                "select": [{"measure": "credit_limit_r5000", "agg": "count_distinct"}],
            },
            "crm",
            ["COUNT(DISTINCT credit_limit)"],
        ),
        (
            {
                "from": "usage_events",
                "select": [{"measure": "usage_units", "agg": "sum"}],
                "time": {"mode": "events_in_range", "last": "30d"},
            },
            "billing",
            ["SUM(units)"],
        ),
    ],
)
def test_aggregates_and_identity_normalisation(
    layer: SemanticLayer, spec: Mapping[str, Any], source: str, needles: Sequence[str]
) -> None:
    """Each aggregate renders as itself, and an identity normalisation adds no arithmetic."""
    sql = fragment_of(compiled(layer, spec, source=source)).sql or ""
    for needle in needles:
        assert needle in sql


def test_omitted_agg_defaults_to_sum_and_says_so(layer: SemanticLayer) -> None:
    """The spec allows ``agg`` to be omitted; the compiler picks sum and reports the choice."""
    result = compiled(
        layer,
        {"from": "crm.customers", "select": [{"measure": "credit_limit_r5000"}]},
        source="crm",
    )
    fragment = fragment_of(result)
    assert "SUM(credit_limit)" in (fragment.sql or "")
    assert any("used sum" in fix for fix in result.applied_fixes)


@pytest.mark.parametrize(
    ("filter_spec", "needles"),
    [
        ({"column": "customer_status", "op": "eq", "value": "פעיל"}, ["status = 'A'"]),
        ({"column": "customer_status", "op": "ne", "value": "active"}, ["status <> 'A'"]),
        (
            {"column": "customer_status", "op": "in", "value": ["עזב", "מושעה"]},
            ["status IN ('C', 'S')"],
        ),
        ({"column": "units", "op": "gt", "value": 10}, ["units > 10"]),
        ({"column": "units", "op": "range", "value": [5, 9]}, ["units BETWEEN 5 AND 9"]),
        (
            {"column": "client_no", "op": "contains", "value": "1486"},
            ["POSITION('1486' IN usage_events.client_no) > 0"],
        ),
        (
            {"column": "units", "op": "eq", "value": 7},
            ["units = 7"],
        ),
    ],
)
def test_filter_operators(
    layer: SemanticLayer, filter_spec: Mapping[str, Any], needles: Sequence[str]
) -> None:
    """Every operator maps to one sqlglot node; a value map applies to the categorical ones."""
    result = compiled(
        layer,
        {
            "from": "usage_events",
            "select": [{"measure": "usage_units", "agg": "sum"}],
            "filters": [filter_spec],
            "joins": ["usage_to_customers"],
            "time": {"mode": "events_in_range", "last": "30d"},
        },
        source="billing",
    )
    sql = fragment_of(result).sql or ""
    for needle in needles:
        assert needle in sql


def test_categorical_group_returns_canonical_values(layer: SemanticLayer) -> None:
    """A grouped status is the canonical word, not the source's ``A``/``Active``/``1``."""
    result = compiled(
        layer,
        {
            "from": "crm.customers",
            "select": [{"measure": "credit_limit_r5000", "agg": "avg"}],
            "group_by": ["customer_status"],
            "order_by": [{"by": "customer_status", "direction": "asc"}],
        },
        source="crm",
    )
    sql = fragment_of(result).sql or ""
    assert (
        "CASE status WHEN 'A' THEN 'active' WHEN 'C' THEN 'churned' WHEN 'S' THEN 'suspended'"
        in sql
    )
    assert "GROUP BY CASE status WHEN 'A' THEN 'active'" in sql
    assert "ORDER BY CASE status WHEN 'A' THEN 'active'" in sql


def test_time_bucket_groups_the_event_column(layer: SemanticLayer) -> None:
    """``month`` is a date bucket on the event-time column, and needs no calendar table."""
    result = compiled(
        layer,
        {
            "from": "usage_events",
            "select": [{"measure": "event_count", "agg": "count"}],
            "group_by": ["month"],
            "time": {"mode": "events_in_range", "from": "2026-06-01", "to": "2026-11-01"},
        },
        source="billing",
    )
    sql = fragment_of(result).sql or ""
    assert "DATE_TRUNC('MONTH', start_time)" in sql
    assert "GROUP BY DATE_TRUNC('MONTH', start_time)" in sql


def test_an_instant_filter_on_a_time_column_is_a_timestamp(layer: SemanticLayer) -> None:
    """A bound on a declared time column compares as a timestamp, never as text."""
    result = compiled(
        layer,
        {
            "from": "usage_events",
            "select": [{"measure": "usage_units", "agg": "count"}],
            "time": {"mode": "events_in_range", "from": "2025-12-01", "to": "2026-01-01"},
            "filters": [{"column": "insertion_time", "op": "lte", "value": "2025-12-15T00:00:00Z"}],
        },
        source="billing",
    )
    sql = fragment_of(result).sql or ""
    assert "insertion_time <= CAST('2025-12-15 00:00:00' AS TIMESTAMP)" in sql


# --- end to end on DuckDB ----------------------------------------------------------------


def _adapter(name: str, tables: Mapping[str, pa.Table]) -> FakeAdapter:
    """One fake source holding exactly the tables a question reads (names are physical).

    The session time zone is pinned to UTC. ``start_time`` is a ``timestamptz`` column and the
    time predicate's literal is a naive UTC timestamp (task C07 renders it that way), so a source
    whose session zone is not UTC would move the window by its offset. Pinning it here is the
    documented deployment assumption, not a workaround for the compiler.
    """
    adapter = FakeAdapter(name=name, tables=dict(tables))
    # The fake exposes no session knob, and its connection is Optional until it is closed.
    connection = adapter._con
    assert connection is not None
    connection.execute("SET TimeZone='UTC'")
    return adapter


def _deduped_erp(world: World) -> pa.Table:
    """``erp.customers`` deduped on its key, which is the view task E01's oracle reads.

    The world plants duplicate customer records that differ only in their name, so the same
    ``cust_no`` can appear twice with one limit. E01's oracle dedupes on
    ``(cust_no, status_code, limit_k)`` before averaging; the query spec has no equivalent, so the
    fixture hands the compiler that view and ``test_erp_duplicates_are_a_documented_spec_gap``
    keeps the difference visible instead of hiding it.
    """
    rows = sorted({(row.cust_no, row.status_code, row.limit_k) for row in world.erp.customers})
    return pa.table(
        {
            "cust_no": [row[0] for row in rows],
            "status_code": [row[1] for row in rows],
            "limit_k": [row[2] for row in rows],
        }
    )


def _execute(adapter: FakeAdapter, fragment: Fragment) -> tuple[list[str], list[tuple[Any, ...]]]:
    limits = Limits(
        timeout_s=60.0, max_rows=1_000_000, max_bytes=256 * 1024 * 1024, max_partitions=64
    )
    table = _rows(iter(adapter.execute(fragment, limits)))
    return list(table.column_names), [tuple(row.values()) for row in table.to_pylist()]


def _assert_answer(
    got: tuple[list[str], list[tuple[Any, ...]]],
    answer: Mapping[str, Any],
    tolerance: float,
) -> None:
    """Compare to an E01 answer: exact for names/strings/integers, relative tolerance for floats."""
    columns, rows = got
    assert columns == list(cast(Sequence[str], answer["columns"]))
    expected = [tuple(row) for row in cast(Sequence[Sequence[Any]], answer["rows"])]
    assert len(rows) == len(expected), f"row count {len(rows)} != {len(expected)}: {rows}"
    for row, wanted in zip(rows, expected, strict=True):
        assert len(row) == len(wanted)
        for value, target in zip(row, wanted, strict=True):
            if isinstance(target, float):
                assert math.isclose(float(value), target, rel_tol=tolerance, abs_tol=0.0), (
                    f"{value!r} != {target!r}"
                )
            else:
                assert value == target, f"{value!r} != {target!r}"


@pytest.mark.parametrize(
    ("question_id", "source", "tables"),
    [
        # billing stores the raw limit: the 5000 half-up rounding must happen in the SQL.
        ("avg_limit_by_status_billing", "billing", (("accounts", "billing.accounts"),)),
        # erp stores the rounded limit divided by ten: the *10 must happen in the SQL.
        ("avg_limit_by_status_erp", "erp", (("customers", "erp.customers"),)),
        # a Hebrew label must resolve to the source's code on the filter side.
        ("churned_customers_hebrew", "crm", (("customers", "crm.customers"),)),
        # a window on the partition key is required, and it changes the row count.
        ("events_count_window", "billing", (("usage_events", "billing.usage_events"),)),
        # the approved join owns the zero-padded key conversion; the group must be canonical.
        (
            "usage_units_by_status_window",
            "billing",
            (("usage_events", "billing.usage_events"), ("customers", "crm.customers")),
        ),
    ],
)
def test_e01_questions_execute_on_duckdb(
    questions: Mapping[str, Any],
    world: World,
    layer: SemanticLayer,
    question_id: str,
    source: str,
    tables: Sequence[tuple[str, str]],
) -> None:
    """The compiled SQL runs on DuckDB and returns the answer E01 derived independently."""
    wanted = {
        physical: (
            _deduped_erp(world)
            if dataset == "erp.customers"
            else table_from_rows(cast(Any, _dataset(world, dataset)))
        )
        for physical, dataset in tables
    }
    adapter = _adapter(source, wanted)
    try:
        spec = e01_spec(questions, question_id)
        result = compile_spec(
            spec,
            CompileContext(
                layer=layer,
                source=source,
                dialect="duckdb",
                exec_ts=EXEC_TS,
                columns=_catalog(adapter),
            ),
        )
        fragment = fragment_of(result)
        got = _execute(adapter, fragment)
        _assert_answer(got, e01_answer(questions, question_id), float(questions["float_tolerance"]))
    finally:
        adapter.close()


def test_e01_refusal_question_calls_no_source(
    questions: Mapping[str, Any], layer: SemanticLayer
) -> None:
    """The partition question's correct answer is a refusal, and the compiler never emits SQL."""
    spec = e01_spec(questions, "usage_events_all_partitions_refused")
    result = compile_spec(spec, context(layer, source="billing"))
    assert result.fragment is None
    assert result.error is not None
    assert result.error.code == ErrorCode.PARTITION_SCAN_ALL.value
    assert result.error.fix is not None
    assert '"time":' in result.error.fix


def test_erp_duplicate_records_are_deduped_on_the_declared_key(
    questions: Mapping[str, Any], world: World, layer: SemanticLayer
) -> None:
    """E01 dedupes ``erp.customers`` on its key before averaging; so must the compiler.

    The world plants customer records that repeat a ``cust_no``, so averaging the raw table weights
    those customers twice. E01's oracle dedupes on ``(cust_no, status_code, limit_k)``; the compiler
    now does the same for an aggregate over a table whose layer entry declares a key.

    FAILS IF: the compiled SQL does not dedupe, or the answer drifts from E01's truth. That is
    what happened before the fix: the tool answered 29719.63 where E01 says 29748.72.
    """
    adapter = _adapter("erp", {"customers": table_from_rows(world.erp.customers)})
    try:
        spec = e01_spec(questions, "avg_limit_by_status_erp")
        result = compile_spec(
            spec,
            CompileContext(
                layer=layer,
                source="erp",
                dialect="duckdb",
                exec_ts=EXEC_TS,
                columns=_catalog(adapter),
            ),
        )
        fragment = fragment_of(result)
        columns, rows = _execute(adapter, fragment)
    finally:
        adapter.close()

    assert fragment.sql is not None
    assert "DISTINCT" in fragment.sql.upper(), fragment.sql
    assert columns == ["customer_status", "credit_limit_r5000_avg"]
    expected_rows = cast(
        Sequence[Sequence[Any]], e01_answer(questions, "avg_limit_by_status_erp")["rows"]
    )
    answer = {str(row[0]): float(row[1]) for row in expected_rows}
    produced = {row[0]: float(row[1]) for row in rows}
    assert set(produced) == set(answer)
    for status, expected in answer.items():
        assert produced[status] == pytest.approx(expected, rel=1e-9), status


def _dataset(world: World, key: str) -> Sequence[Any]:
    """One world dataset by its semantic table key."""
    match key:
        case "billing.usage_events":
            return world.billing.usage_events
        case "billing.accounts":
            return world.billing.accounts
        case "crm.customers":
            return world.crm.customers
        case "erp.customers":
            return world.erp.customers
        case _:
            raise AssertionError(f"no world dataset for {key!r}")


def _catalog(adapter: FakeAdapter) -> Mapping[str, tuple[str, ...]]:
    """The columns an adapter really has, keyed by the semantic table keys this test layer uses."""
    snapshot = adapter.extract_catalog()
    by_table = {
        table.name: tuple(column.name for column in table.columns) for table in snapshot.tables
    }
    return {
        key: by_table[key.rpartition(".")[2]]
        for key in COLUMNS
        if key.rpartition(".")[2] in by_table
    }
