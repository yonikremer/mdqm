"""Oracle and T-SQL rendering of the single-source compiler (task C10).

sqlglot makes a target dialect look free: it renders something for every AST and, where a construct
has no equivalent, it passes the name through unchanged (``LPAD`` on T-SQL, ``TIMESTAMP_TRUNC`` on
Oracle) or drops the part it cannot write. The result often still *parses* as sqlglot, so a round
trip alone proves nothing about whether the engine accepts it. These tests pin four things instead:

* **golden SQL** per dialect for the same specs task C09 pinned for PostgreSQL, so every change to
  the emitted statement is a reviewable diff in ``tests/data/golden/``;
* **the dialect's own spelling**, asserted by name (``FETCH FIRST``/``TOP``, ``TO_TIMESTAMP``/
  ``DATETIME2``, ``"alias"``/``[alias]``, ``NUMBER``/``NUMERIC``, ``TRUNC``/``DATETRUNC``), so a
  silent fall-back to PostgreSQL spelling, or a construct the engine does not have (``LPAD`` on
  T-SQL, ``TIMESTAMP_TRUNC`` on Oracle), fails here;
* **refusals**: a construct one dialect cannot express with the same meaning raises a typed error
  with a fix and no SQL, instead of a best-effort translation;
* **parse-back in the target dialect**, which proves at least that the statement matches the syntax
  sqlglot models for that dialect.

Nothing here executes on a real Oracle or SQL Server: that gap is recorded in ``progress/C10.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel, UnsupportedError

from mdqm.compiler import CompileContext, CompiledQuery, compile_spec
from mdqm.compiler.dialects import render, rewrite, trunc_expression
from mdqm.errors import ErrorCode
from mdqm.semantic.loader import load
from mdqm.semantic.models import SemanticLayer
from mdqm.spec.validate import parse_spec
from mdqm.sql.exec_ts import rewrite_sql
from mdqm.sql.guard import guard

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "examples" / "query_specs.json"
LAYER_FILE = Path(__file__).resolve().parents[1] / "data" / "semantic" / "compiler_semantic.yaml"

#: One execution timestamp for every compilation in this module (C02: one value per execution).
EXEC_TS = datetime(2026, 11, 25, 15, 0, tzinfo=UTC)

#: The catalog as the adapters report it (as in the C09 module: only these spellings may reach SQL).
COLUMNS: Mapping[str, tuple[str, ...]] = {
    "crm.customers": ("cust_id", "status", "credit_limit", "region"),
    "billing.accounts": ("client_no", "state", "credit_limit", "region", "opened_at"),
    "billing.usage_events": ("event_id", "client_no", "units", "start_time", "insertion_time"),
    "erp.customers": ("cust_no", "status_code", "limit_k", "name"),
}

#: The dialects this task adds. ``postgres`` appears only as a control.
DIALECTS: tuple[str, ...] = ("oracle", "tsql")

VALID_EXAMPLES: tuple[Mapping[str, Any], ...] = tuple(
    json.loads(SPECS.read_text(encoding="utf-8"))["valid"]
)
EXAMPLE_IDS = [entry["name"] for entry in VALID_EXAMPLES]

#: A spec whose only job is the time bucket, so the bucket rules are tested on their own.
BUCKET_SPEC: Mapping[str, Any] = {
    "from": "usage_events",
    "select": [{"measure": "event_count", "agg": "count"}],
    "time": {"mode": "events_in_range", "from": "2026-06-01", "to": "2026-08-01"},
    "group_by": ["month"],
    "limit": 20,
}

#: What each dialect writes for the start of a unit, and what it cannot write at all.
#: The bucket is unqualified because a single-table fragment does not qualify its columns.
BUCKETS: tuple[tuple[str, str | None, str | None], ...] = (
    # unit,    oracle                              tsql           (None = must refuse)
    ("day", "TRUNC(start_time, 'DD')", "DATETRUNC(DAY, start_time)"),
    # A T-SQL week starts wherever DATEFIRST says, so the same query means a different week.
    ("week", "TRUNC(start_time, 'IW')", None),
    # 'MONTH'/'YEAR' are sqlglot's spelling of the 'MM'/'YYYY' masks we emit: both are valid Oracle.
    ("month", "TRUNC(start_time, 'MONTH')", "DATETRUNC(MONTH, start_time)"),
    ("quarter", "TRUNC(start_time, 'Q')", "DATETRUNC(QUARTER, start_time)"),
    ("year", "TRUNC(start_time, 'YEAR')", "DATETRUNC(YEAR, start_time)"),
)


@pytest.fixture(scope="module")
def layer() -> SemanticLayer:
    """The test semantic layer, loaded through the real loader (so its validation runs too)."""
    return load(LAYER_FILE)


# --- helpers -----------------------------------------------------------------------------


def compile_for(layer: SemanticLayer, spec: Mapping[str, Any], dialect: str) -> CompiledQuery:
    """Compile one spec mapping for ``dialect`` against the test catalog."""
    context = CompileContext(
        layer=layer,
        source="billing",
        dialect=dialect,
        exec_ts=EXEC_TS,
        columns=COLUMNS,
    )
    return compile_spec(parse_spec(spec), context)


def sql_of(result: CompiledQuery) -> str:
    """The statement, asserting the compilation succeeded (a readable failure when it did not)."""
    assert result.error is None, f"unexpected refusal: {result.error.problem}"
    assert result.fragment is not None
    return result.fragment.sql or ""


def statements(layer: SemanticLayer, dialect: str) -> list[tuple[str, str]]:
    """``(example name, sql)`` for every example in the golden set, compiled for one dialect."""
    return [
        (
            str(case["name"]),
            sql_of(compile_for(layer, cast(Mapping[str, Any], case["spec"]), dialect)),
        )
        for case in VALID_EXAMPLES
    ]


def spec_named(name: str) -> Mapping[str, Any]:
    """One example spec by its golden name."""
    for entry in VALID_EXAMPLES:
        if entry["name"] == name:
            return cast(Mapping[str, Any], entry["spec"])
    raise AssertionError(f"{name!r} is not a valid example in {SPECS.name}")


# --- golden SQL --------------------------------------------------------------------------


@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize("case", VALID_EXAMPLES, ids=EXAMPLE_IDS)
def test_every_valid_example_compiles_to_golden_sql(
    case: Mapping[str, Any], dialect: str, layer: SemanticLayer, golden: Any
) -> None:
    """Each C09 spec compiles for oracle and tsql; the statement is a committed golden file."""
    result = compile_for(layer, cast(Mapping[str, Any], case["spec"]), dialect)
    assert result.fragment is not None, result.error.problem if result.error else ""
    assert result.fragment.dialect == dialect
    golden(f"{case['name']}_{dialect}.sql", result.fragment.sql or "")


# --- the engine's own spelling, not PostgreSQL's -----------------------------------------


@pytest.mark.parametrize("dialect", DIALECTS)
def test_the_dialect_markers_are_present_and_the_foreign_ones_absent(
    dialect: str, layer: SemanticLayer
) -> None:
    """Every statement carries its dialect's spelling and none of the other dialect's."""
    present = {
        "oracle": ("FETCH FIRST", "TO_TIMESTAMP("),
        "tsql": ("TOP ", "DATETIME2"),
    }[dialect]
    absent = {
        # LPAD stays: Oracle has it. TIMESTAMP_TRUNC does not exist in Oracle at all.
        "oracle": ("DATETRUNC(", "DATETIME2", "LIMIT ", "NUMERIC(", "TIMESTAMP_TRUNC(", "["),
        # LPAD does NOT exist in T-SQL, and brackets are its quoting, never a double quote.
        "tsql": (
            "FETCH FIRST",
            "TO_TIMESTAMP(",
            "NUMBER(",
            "LIMIT ",
            "LPAD(",
            "TIMESTAMP_TRUNC(",
            '"',
        ),
    }[dialect]
    for name, sql in statements(layer, dialect):
        for marker in present:
            assert marker in sql, f"{name} [{dialect}] lost {marker!r}: {sql}"
        for marker in absent:
            assert marker not in sql, f"{name} [{dialect}] kept {marker!r}: {sql}"


@pytest.mark.parametrize(
    ("dialect", "alias"),
    [
        ("oracle", '"credit_limit_r5000_avg"'),
        ("tsql", "[credit_limit_r5000_avg]"),
    ],
)
def test_identifiers_use_the_dialects_quote_style(
    dialect: str, alias: str, layer: SemanticLayer
) -> None:
    """A quoted alias keeps its spelling: unquoted, Oracle upper-cases it and T-SQL does not."""
    sql = sql_of(compile_for(layer, spec_named("avg_credit_limit_by_status_3mo"), dialect))
    assert alias in sql, sql


@pytest.mark.parametrize(
    ("dialect", "type_name"),
    [("oracle", "NUMBER(38, 6)"), ("tsql", "NUMERIC(38, 6)")],
)
def test_the_normalisation_cast_is_typed_by_the_dialect(
    dialect: str, type_name: str, layer: SemanticLayer
) -> None:
    """The measure's ``DECIMAL(38, 6)`` becomes the dialect's own exact numeric type."""
    sql = sql_of(compile_for(layer, spec_named("avg_credit_limit_by_status_3mo"), dialect))
    assert f"CAST(credit_limit AS {type_name})" in sql
    assert "DECIMAL(" not in sql.replace(f"{type_name}", ""), sql


@pytest.mark.parametrize("dialect", DIALECTS)
def test_every_rendered_statement_parses_back_in_its_dialect(
    dialect: str, layer: SemanticLayer
) -> None:
    """The statement is valid syntax *for that dialect* and re-renders to itself.

    Idempotence is the useful half: a second read/render cycle through sqlglot's dialect grammar
    returning different text would mean we wrote something that dialect reads differently.
    """
    for name, sql in statements(layer, dialect):
        parsed = sqlglot.parse_one(sql, read=dialect)
        assert parsed is not None, f"{name} [{dialect}] does not parse back"
        again = parsed.sql(dialect=dialect, unsupported_level=ErrorLevel.RAISE)
        assert again == sql, f"{name} [{dialect}] is not stable under a read/render cycle:\n{again}"


@pytest.mark.parametrize("dialect", DIALECTS)
def test_the_compiled_fragments_pass_the_guard(dialect: str, layer: SemanticLayer) -> None:
    """The SQL we compile for a dialect is guarded exactly like SQL an agent writes (C01)."""
    for name, sql in statements(layer, dialect):
        checked = guard(sql, dialect)
        assert checked.ok, f"{name} [{dialect}]: " + (
            checked.error.problem if checked.error else "guard refused the SQL"
        )


@pytest.mark.parametrize("dialect", DIALECTS)
def test_no_clock_function_survives_in_any_dialect(dialect: str, layer: SemanticLayer) -> None:
    """``exec_ts`` is substituted before rendering, so no dialect's own clock is read."""
    now_like = ("CURRENT_TIMESTAMP", "CURRENT_DATE", "LOCALTIMESTAMP", "SYSDATE", "SYSTIMESTAMP")
    for name, sql in statements(layer, dialect):
        assert rewrite_sql(sql, dialect=dialect, exec_ts=EXEC_TS).replacements == (), name
        for function in now_like:
            assert function not in sql.upper(), f"{name} [{dialect}] kept {function}"


@pytest.mark.parametrize("dialect", DIALECTS)
def test_compile_is_deterministic_per_dialect(dialect: str, layer: SemanticLayer) -> None:
    """Same spec, same dialect, byte-identical SQL: a golden diff means a real change."""
    for case in VALID_EXAMPLES:
        spec = cast(Mapping[str, Any], case["spec"])
        assert sql_of(compile_for(layer, spec, dialect)) == sql_of(
            compile_for(layer, spec, dialect)
        ), f"{case['name']} [{dialect}]"


# --- the timestamp literal: one instant, spelled so the session cannot misread it -------


def test_oracle_reads_no_session_format_mask(layer: SemanticLayer) -> None:
    """Oracle gets ``TO_TIMESTAMP`` with an explicit mask: ``CAST(... AS TIMESTAMP)`` reads NLS."""
    sql = sql_of(compile_for(layer, spec_named("avg_credit_limit_by_status_3mo"), "oracle"))
    assert "TO_TIMESTAMP('2026-08-25 15:00:00.000000', 'YYYY-MM-DD HH24:MI:SS.FF6')" in sql
    assert "AS TIMESTAMP" not in sql


@pytest.mark.parametrize("name", EXAMPLE_IDS)
def test_tsql_writes_the_iso_8601_form_set_dateformat_cannot_reread(
    name: str, layer: SemanticLayer
) -> None:
    """T-SQL gets ``CAST('2026-08-25T15:00:00.000000' AS DATETIME2)``: a ``T``, never a space."""
    sql = sql_of(compile_for(layer, spec_named(name), "tsql"))
    for literal in (
        "2026-08-25T15",
        "2026-08-25T14",
        "2026-11-25T15",
        "2026-11-25T08",
        "2026-06-01T00",
        "2026-11-01T00",
    ):
        if literal in ("2026-08-25T15", "2026-11-25T15"):
            assert f"CAST('{literal}:00:00.000000' AS DATETIME2)" in sql or literal not in sql, sql
    assert " AS DATETIME2)" in sql


# --- the join key: T-SQL has neither LPAD nor RPAD ---------------------------------------


def test_the_padded_join_key_is_translated_for_tsql_and_kept_for_oracle(
    layer: SemanticLayer,
) -> None:
    """A template's ``LPAD`` becomes ``RIGHT(REPLICATE(...))`` on T-SQL; Oracle keeps ``LPAD``."""
    spec = spec_named("events_per_month_hebrew_filter")
    oracle = sql_of(compile_for(layer, spec, "oracle"))
    tsql = sql_of(compile_for(layer, spec, "tsql"))
    assert "LPAD(CAST(customers.cust_id AS VARCHAR2), 10, '0')" in oracle
    assert (
        "RIGHT(REPLICATE('0', 10 - LEN(CAST(customers.cust_id AS VARCHAR))) "
        "+ CAST(customers.cust_id AS VARCHAR), 10)" in tsql
    ), tsql
    assert "LPAD" not in tsql and "RPAD" not in tsql


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        ("LPAD", "RIGHT(REPLICATE('0', 10 - LEN(cust_id)) + cust_id, 10)"),
        ("RPAD", "LEFT(cust_id + REPLICATE('0', 10 - LEN(cust_id)), 10)"),
    ],
)
def test_pad_rewrite_matches_the_sql_server_idiom(function: str, expected: str) -> None:
    """The LPAD/RPAD rewrite is the textbook T-SQL equivalent, not something approximate."""
    padded = cast(
        exp.Expression, sqlglot.parse_one(f"SELECT {function}(cust_id, 10, '0')", read="postgres")
    )
    assert rewrite(padded, "tsql").sql(dialect="tsql") == f"SELECT {expected}"


def test_pad_rewrite_is_tsql_only() -> None:
    """The other dialects have LPAD: rewriting there would only make the SQL harder to read."""
    padded = cast(
        exp.Expression, sqlglot.parse_one("SELECT LPAD(cust_id, 10, '0')", read="postgres")
    )
    for dialect in ("postgres", "oracle", "duckdb", "mysql"):
        assert rewrite(padded, dialect).sql(dialect=dialect) == "SELECT LPAD(cust_id, 10, '0')"


# --- buckets, and the one that must refuse ----------------------------------------------


@pytest.mark.parametrize(("unit", "oracle", "tsql"), BUCKETS, ids=[bucket[0] for bucket in BUCKETS])
def test_time_buckets_are_written_by_the_dialect(
    unit: str, oracle: str | None, tsql: str | None, layer: SemanticLayer
) -> None:
    """Each unit is the dialect's own truncation form, or a refusal where it has no faithful one."""
    spec = {**BUCKET_SPEC, "group_by": [unit]}
    for dialect, expected in (("oracle", oracle), ("tsql", tsql)):
        result = compile_for(layer, spec, dialect)
        if expected is None:
            assert result.error is not None, f"{unit} [{dialect}] should have been refused"
            assert result.fragment is None
            continue
        assert result.error is None, (
            f"{unit} [{dialect}]: {result.error.problem if result.error else ''}"
        )
        sql = sql_of(result)
        # the bucket must be the same expression in SELECT and GROUP BY, or the engine rejects it
        assert sql.count(expected) == 2, f"{unit} [{dialect}]: {sql}"


def test_a_tsql_week_bucket_refuses_instead_of_guessing(layer: SemanticLayer) -> None:
    """A T-SQL week starts wherever ``DATEFIRST`` says, so the same query means a different week."""
    result = compile_for(layer, {**BUCKET_SPEC, "group_by": ["week"]}, "tsql")
    error = result.error
    assert error is not None
    assert result.fragment is None, "a refusal must not carry SQL"
    assert error.code == ErrorCode.INVALID_SPEC.value
    assert "week" in error.problem and "tsql" in error.problem
    assert "day" in error.fix and "month" in error.fix, error.fix


@pytest.mark.parametrize(
    ("dialect", "expected"),
    [
        ("postgres", "DATE_TRUNC('WEEK', start_time)"),
        ("oracle", "TRUNC(start_time, 'IW')"),
    ],
)
def test_a_week_bucket_is_a_fixed_iso_week_elsewhere(dialect: str, expected: str) -> None:
    """PostgreSQL and Oracle both truncate to the ISO Monday: only T-SQL's week is ambiguous."""
    column = sqlglot.parse_one("SELECT start_time").find(sqlglot.exp.Column)
    assert column is not None
    truncated = trunc_expression("week", column, dialect)
    assert truncated is not None
    assert truncated.sql(dialect=dialect, unsupported_level=ErrorLevel.RAISE) == expected


# --- nothing degrades silently ----------------------------------------------------------


def test_sqlglot_refuses_instead_of_degrading() -> None:
    """``unsupported_level=RAISE`` is wired into every render, not only the paths we test.

    Without the flag sqlglot happily writes this statement for T-SQL, dropping the LATERAL and
    changing the answer. That is the failure mode this compiler refuses to inherit.
    """
    statement = cast(
        exp.Expression,
        sqlglot.parse_one("SELECT * FROM t, LATERAL (SELECT 1) AS s", read="postgres"),
    )
    assert statement is not None
    with pytest.raises(UnsupportedError):
        render(statement, "tsql")
    # the same AST renders for a dialect that has LATERAL: RAISE is not a blunt switch
    assert "LATERAL" in render(statement, "postgres")


def test_a_refused_compilation_has_no_sql_and_a_fix(layer: SemanticLayer) -> None:
    """Every refusal path leaves ``fragment`` empty and fills ``code``, ``problem`` and ``fix``."""
    result = compile_for(layer, {**BUCKET_SPEC, "group_by": ["week"]}, "tsql")
    error = result.error
    assert error is not None and result.fragment is None
    assert error.code and error.problem and error.fix
