"""Tests for ``mdqm.semantic.measures`` (task C06): normalisation and explicit rounding.

The credit limit means ``round_half_up(x / 5000) * 5000`` and the synthetic world (task D02) stores
it three ways: already rounded, raw, and rounded-then-divided-by-ten. The planted boundary values
(2500, 7500, 12500) sit exactly on the ``.5`` tie, which is the only place half-up and banker's
rounding disagree -- so a test that passes at 2500 and 7500 proves the mode is real and not an
engine default that happened to be lenient.

Table-driven where the cases are data (rounding modes, boundary values, broken templates) and a
real DuckDB evaluation where the case is "does this SQL actually compute that number".
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import duckdb
import pytest
import sqlglot
from hypothesis import given, settings
from hypothesis import strategies as st

from mdqm.semantic import SemanticError, load
from mdqm.semantic.measures import (
    AppliedNormalization,
    check_multiples,
    check_normalize_template,
    normalizations_for,
    render_normalize,
    round_to_multiple,
)
from mdqm.semantic.models import Measure, RoundingMode
from mdqm.testing.world import WorldConfig, build_world

STEP = 5000
"""The canonical granularity from ``docs/DESIGN.md`` section 12."""

DIALECTS = ("postgres", "oracle", "tsql")
"""The three dialects the acceptance criteria name, plus duckdb for local evaluation."""

CREDIT_LIMIT = Measure.model_validate(
    {
        "canonical": {"rounding": {"to": STEP, "mode": "half_up"}},
        "sources": {
            "crm.customers.credit_limit": {"normalize": "{col}"},
            "billing.accounts.credit_limit": {"normalize": "ROUND({col} / 5000) * 5000"},
            "erp.customers.limit_k": {"normalize": "{col} * 10"},
        },
    }
)
"""The exact measure from ``examples/semantic_example.yaml``, three storage forms."""


def _measure(mode: str = "half_up", to: int = STEP) -> Measure:
    """A one-source measure with the given canonical rounding, for focused cases."""
    return Measure.model_validate(
        {
            "canonical": {"rounding": {"to": to, "mode": mode}},
            "sources": {"a.b.c": {"normalize": "ROUND({col} / 5000) * 5000"}},
        }
    )


def evaluate(sql: str, value: int | None = None, *, column: str = "x") -> Any:
    """Run one rendered expression on DuckDB, optionally binding ``value`` for ``column``.

    Goes through an explicit connection: ``duckdb.sql`` is typed ``Optional``, and a test helper
    that hides a possible None behind ``# type: ignore`` would fail confusingly instead of loudly.
    """
    connection = duckdb.connect()
    try:
        if value is None:
            return connection.execute(f"SELECT {sql}").fetchall()
        # A one-row subquery binds the operand, so ``sql`` keeps referring to ``column``.
        query = f"SELECT {sql} FROM (SELECT {value} AS {column}) AS operand"
        return connection.execute(query).fetchall()[0][0]
    finally:
        connection.close()


def evaluate_in_dialect(sql: str, dialect: str, value: int | None = None) -> Any:
    """Transpile rendered SQL back to DuckDB and evaluate it, so the text is what is tested."""
    back = sqlglot.transpile(sql, read=dialect, write="duckdb")[0]
    return evaluate(back, value)


# --- rounding: the reference implementation, table-driven ---------------------------------

#: (value, mode, expected) -- the boundaries are the whole point; 2500/12500 are where the modes
#: part company, and the negative rows prove "half up" means away from zero, not toward +infinity.
ROUNDING_CASES: tuple[tuple[int, RoundingMode, int], ...] = (
    (0, RoundingMode.HALF_UP, 0),
    (1, RoundingMode.HALF_UP, 0),
    (2499, RoundingMode.HALF_UP, 0),
    (2500, RoundingMode.HALF_UP, 5000),
    (2501, RoundingMode.HALF_UP, 5000),
    (4999, RoundingMode.HALF_UP, 5000),
    (5000, RoundingMode.HALF_UP, 5000),
    (5001, RoundingMode.HALF_UP, 5000),
    (7499, RoundingMode.HALF_UP, 5000),
    (7500, RoundingMode.HALF_UP, 10000),
    (12500, RoundingMode.HALF_UP, 15000),
    (5000, RoundingMode.HALF_EVEN, 5000),
    (2500, RoundingMode.HALF_EVEN, 0),
    (7500, RoundingMode.HALF_EVEN, 10000),
    (12500, RoundingMode.HALF_EVEN, 10000),
    (2500, RoundingMode.HALF_DOWN, 0),
    (7500, RoundingMode.HALF_DOWN, 5000),
    (12500, RoundingMode.HALF_DOWN, 10000),
    (2500, RoundingMode.FLOOR, 0),
    (2501, RoundingMode.FLOOR, 0),
    (7499, RoundingMode.FLOOR, 5000),
    (2500, RoundingMode.CEIL, 5000),
    (5000, RoundingMode.CEIL, 5000),
    (5001, RoundingMode.CEIL, 10000),
    (-2500, RoundingMode.HALF_UP, -5000),
    (-2501, RoundingMode.HALF_UP, -5000),
    (-2499, RoundingMode.HALF_UP, 0),
    (-7500, RoundingMode.HALF_UP, -10000),
    (-2500, RoundingMode.HALF_EVEN, 0),
    (-7500, RoundingMode.HALF_EVEN, -10000),
    (-2500, RoundingMode.FLOOR, -5000),
    (-2500, RoundingMode.CEIL, 0),
)


@pytest.mark.parametrize(("value", "mode", "expected"), ROUNDING_CASES)
def test_round_to_multiple(value: int, mode: RoundingMode, expected: int) -> None:
    assert round_to_multiple(value, STEP, mode) == Decimal(expected)


@pytest.mark.parametrize("value", (2500, 12500))
def test_half_up_is_not_bankers(value: int) -> None:
    """The two plausible rules disagree exactly on the planted boundaries."""
    assert round_to_multiple(value, STEP, RoundingMode.HALF_UP) != round_to_multiple(
        value, STEP, RoundingMode.HALF_EVEN
    )


def test_half_up_matches_the_world_ground_truth() -> None:
    """The reference implementation agrees with D02's independently computed truth."""
    world = build_world(WorldConfig(customers=200, events_per_customer=0))
    for boundary in world.expected.boundary_rows:
        assert round_to_multiple(boundary.raw_limit, STEP) == boundary.half_up_limit
    for truth in world.expected.customers:
        assert round_to_multiple(truth.raw_limit, STEP) == truth.canonical_limit


def test_step_must_be_positive() -> None:
    with pytest.raises(ValueError, match="step must be positive"):
        round_to_multiple(2500, 0)


@given(st.integers(min_value=-(10**9), max_value=10**9), st.integers(min_value=1, max_value=10**6))
@settings(max_examples=200)
def test_half_up_is_always_a_multiple_of_the_step(value: int, step: int) -> None:
    """Property: whatever goes in, the canonical value sits exactly on the grid."""
    result = round_to_multiple(value, step)
    assert result % step == 0


@given(st.integers(min_value=-(10**9), max_value=10**9), st.integers(min_value=1, max_value=10**6))
@settings(max_examples=200)
def test_rounding_is_idempotent(value: int, step: int) -> None:
    """Property: rounding an already-rounded value cannot move it again."""
    once = round_to_multiple(value, step)
    assert round_to_multiple(once, step) == once


@given(
    st.integers(min_value=-(10**9), max_value=10**9),
    st.integers(min_value=1, max_value=10**6),
    st.sampled_from(list(RoundingMode)),
)
@settings(max_examples=200)
def test_rounding_moves_by_at_most_half_a_step(value: int, step: int, mode: RoundingMode) -> None:
    """Property: `HALF_*` never moves further than half a step; floor/ceil stay within one."""
    result = round_to_multiple(value, step, mode)
    limit = Decimal(step) if mode in (RoundingMode.FLOOR, RoundingMode.CEIL) else Decimal(step) / 2
    assert abs(result - Decimal(value)) <= limit


# --- rendered SQL: explicit, per dialect -------------------------------------------------


def test_half_up_sql_never_uses_a_bare_round() -> None:
    """The rendered SQL carries the rule, so no engine gets to pick the tie behaviour."""
    sql = render_normalize(_measure(), "a.b.c", "x", dialect="postgres")
    assert sql is not None
    assert "ROUND" not in sql.upper()
    assert "FLOOR" in sql.upper()
    assert "SIGN" in sql.upper()


def test_integer_division_is_cast_so_boundaries_survive() -> None:
    """The cast is load-bearing: without it 2500 / 5000 truncates to 0 on PostgreSQL and SQL Server.

    DuckDB divides integers as floats by default, so the truncating behaviour is reproduced with
    its ``integer_division`` setting rather than assumed.
    """
    sql = render_normalize(_measure(), "a.b.c", "x", dialect="postgres")
    assert sql is not None
    assert "CAST" in sql.upper()

    connection = duckdb.connect()
    try:
        connection.execute("SET integer_division = true")
        uncast = connection.execute("SELECT 2500 / 5000").fetchall()[0][0]
        cast = connection.execute(
            "SELECT CAST(2500 AS DECIMAL(38,6)) / CAST(5000 AS DECIMAL(38,6))"
        ).fetchall()[0][0]
    finally:
        connection.close()
    assert uncast == 0, "integer division truncates, which is why the template casts"
    assert cast == Decimal("0.5")


@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize(("value", "expected"), ((2500, 5000), (7500, 10000), (12500, 15000)))
def test_boundary_rounds_up_in_every_dialect(dialect: str, value: int, expected: int) -> None:
    """Acceptance: 2500 and 7500 round up (half_up) for postgres, oracle and tsql."""
    sql = render_normalize(_measure(), "a.b.c", "x", dialect=dialect)
    assert sql is not None
    assert sql == render_normalize(_measure(), "a.b.c", "x", dialect=dialect)
    assert evaluate_in_dialect(sql, dialect, value) == expected


@pytest.mark.parametrize("dialect", DIALECTS)
def test_half_even_dialect_sql_differs_on_the_boundary(dialect: str) -> None:
    """The declared mode really changes the rendered SQL, so the mode is not cosmetic."""
    half_up = render_normalize(_measure("half_up"), "a.b.c", "x", dialect=dialect)
    half_even = render_normalize(_measure("half_even"), "a.b.c", "x", dialect=dialect)
    assert half_up is not None and half_even is not None
    assert half_up != half_even
    assert evaluate_in_dialect(half_up, dialect, 2500) == 5000
    assert evaluate_in_dialect(half_even, dialect, 2500) == 0


@pytest.mark.parametrize("dialect", DIALECTS)
def test_negative_half_up_moves_away_from_zero(dialect: str) -> None:
    sql = render_normalize(_measure(), "a.b.c", "x", dialect=dialect)
    assert sql is not None
    assert evaluate_in_dialect(sql, dialect, -2500) == -5000


def test_identity_source_renders_the_column_alone() -> None:
    sql = render_normalize(
        CREDIT_LIMIT, "crm.customers.credit_limit", "credit_limit", dialect="postgres"
    )
    assert sql == "credit_limit"


def test_unlisted_column_has_no_normalisation() -> None:
    assert render_normalize(CREDIT_LIMIT, "nope.nope.nope", "x", dialect="postgres") is None


def test_round_without_a_declared_mode_is_an_error() -> None:
    """Guessing the mode would silently answer with another organisation's number."""
    measure = Measure.model_validate({"sources": {"a.b.c": {"normalize": "ROUND({col} / 5000)"}}})
    with pytest.raises(ValueError, match="declares no canonical rounding"):
        render_normalize(measure, "a.b.c", "x", dialect="postgres")


def test_round_to_a_scale_is_rejected() -> None:
    """A scaled ROUND cannot express the canonical rule, so it fails at load time."""
    with pytest.raises(ValueError, match="whole number"):
        check_normalize_template("ROUND({col}, 2) * 5000", where="a normalize template")


# --- templates: allowlist at load time ---------------------------------------------------


@pytest.mark.parametrize(
    "template",
    (
        "(SELECT max({col}) FROM t)",
        "{col} + (SELECT 1)",
        "FOO({col})",
        "COUNT(*)",
        "LPAD({col}, 3, '0')",
        "SUM({col})",
    ),
)
def test_bad_templates_are_rejected(template: str) -> None:
    with pytest.raises(ValueError):
        check_normalize_template(template, where="a normalize template")


@pytest.mark.parametrize(
    "template",
    (
        "{col}",
        "{col} * 10",
        "ROUND({col} / 5000) * 5000",
        "FLOOR({col} / 100)",
        "CAST({col} AS DECIMAL(38, 6)) / 7",
    ),
)
def test_good_templates_are_accepted(template: str) -> None:
    check_normalize_template(template, where="a normalize template")


@pytest.mark.parametrize(
    ("template", "fragment"),
    (
        ("FOO({col})", "'FOO'"),
        ("(SELECT max({col}) FROM t)", "subquery"),
    ),
)
def test_rejection_names_the_template_and_the_reason(template: str, fragment: str) -> None:
    with pytest.raises(ValueError) as raised:
        check_normalize_template(template, where="a normalize template")
    assert fragment.lower() in str(raised.value).lower()


def test_unknown_function_is_rejected_when_the_file_loads(tmp_path: Any) -> None:
    """Acceptance: rejected at load time, and the message carries the template's path."""
    document = tmp_path / "layer.yaml"
    document.write_text(
        "measures:\n"
        "  credit_limit_r5000:\n"
        "    canonical: { rounding: { to: 5000, mode: half_up } }\n"
        "    sources:\n"
        '      crm.customers.credit_limit: { normalize: "NOPE({col})" }\n',
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(SemanticError) as raised:
        load(document)
    message = str(raised.value)
    assert "NOPE" in message
    assert "measures.credit_limit_r5000.sources" in message
    assert "normalize" in message


def test_subquery_is_rejected_when_the_file_loads(tmp_path: Any) -> None:
    document = tmp_path / "layer.yaml"
    document.write_text(
        "measures:\n"
        "  credit_limit_r5000:\n"
        "    canonical: { rounding: { to: 5000, mode: half_up } }\n"
        "    sources:\n"
        '      crm.customers.credit_limit: { normalize: "(SELECT max({col}) FROM t)" }\n',
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(SemanticError) as raised:
        load(document)
    assert "subquery" in str(raised.value).lower()


# --- the motivating case, evaluated on DuckDB --------------------------------------------


def _normalised_by_source(world: Any) -> dict[str, dict[int, Decimal]]:
    """Normalise all three storage forms of the credit limit with the rendered SQL, on DuckDB."""
    connection = duckdb.connect()
    try:
        connection.execute("CREATE TABLE crm (cust_id BIGINT, credit_limit BIGINT)")
        connection.executemany(
            "INSERT INTO crm VALUES (?, ?)",
            [(row.cust_id, row.credit_limit) for row in world.crm.customers],
        )
        connection.execute("CREATE TABLE billing (cust_id BIGINT, credit_limit BIGINT)")
        connection.executemany(
            "INSERT INTO billing VALUES (?, ?)",
            [
                (int(row.client_no), row.credit_limit)
                for row in world.billing.accounts
                if row.credit_limit is not None
            ],
        )
        connection.execute("CREATE TABLE erp (cust_id BIGINT, limit_k BIGINT)")
        connection.executemany(
            "INSERT INTO erp VALUES (?, ?)",
            [(row.cust_no, row.limit_k) for row in world.erp.customers],
        )

        sources = {
            "crm": ("crm.customers.credit_limit", "crm", "credit_limit"),
            "billing": ("billing.accounts.credit_limit", "billing", "credit_limit"),
            "erp": ("erp.customers.limit_k", "erp", "limit_k"),
        }
        values: dict[str, dict[int, Decimal]] = {}
        for name, (column, table, field) in sources.items():
            sql = render_normalize(CREDIT_LIMIT, column, field, dialect="duckdb")
            assert sql is not None
            rows = connection.execute(f"SELECT cust_id, {sql} FROM {table}").fetchall()
            values[name] = {
                int(key): Decimal(str(value)) for key, value in rows if value is not None
            }
        return values
    finally:
        connection.close()


def test_three_storage_forms_normalise_to_the_same_canonical_value() -> None:
    """Acceptance: >= 99.5% of shared customers agree, on real DuckDB arithmetic.

    The world plants a deliberate erp exception (a limit one step off) in about 4 per mille of
    customers, so 100% is the wrong assertion: the honest one is "everyone except the planted
    exceptions", which also proves the exceptions are still visible rather than rounded away.
    """
    world = build_world(WorldConfig(customers=2000, months=1, events_per_customer=0))
    values = _normalised_by_source(world)
    canonical = {
        truth.cust_id: Decimal(truth.canonical_limit) for truth in world.expected.customers
    }
    planted = {row.cust_id for row in world.expected.erp_exceptions}

    shared = set(values["crm"]) & set(values["billing"]) & set(values["erp"]) & set(canonical)
    assert shared, "the three forms share no customers"

    for name in ("crm", "billing"):
        agree = sum(1 for key in shared if values[name][key] == canonical[key])
        assert agree / len(shared) >= 0.995, f"{name} normalised only {agree}/{len(shared)}"

    erp_agree = sum(1 for key in shared if values["erp"][key] == canonical[key])
    assert erp_agree / len(shared) >= 0.995, f"erp normalised only {erp_agree}/{len(shared)}"
    # And the disagreement is exactly the planted exception, not a rounding accident.
    disagree = {key for key in shared if values["erp"][key] != canonical[key]}
    assert disagree == planted & shared


def test_normalisation_is_idempotent_on_duckdb() -> None:
    """Property: applying the same normalisation twice cannot move a value off the grid."""
    sql = render_normalize(CREDIT_LIMIT, "billing.accounts.credit_limit", "x", dialect="duckdb")
    assert sql is not None
    for value in (0, 2499, 2500, 2501, 5000, 7500, 12500, 60000):
        once = evaluate(sql, value)
        twice = evaluate(sql.replace("x", f"({once})"), None)
        assert Decimal(str(twice[0][0])) == Decimal(str(once))


# --- result sanity check -----------------------------------------------------------------


def test_sanity_check_flags_a_non_multiple_and_counts_it() -> None:
    """Acceptance: a value off the canonical grid is flagged and counted, never raised."""
    report = check_multiples([0, 5000, 10000, 12345, 2500], STEP)
    assert report.checked == 5
    assert report.violation_count == 2
    assert report.violations == (Decimal(12345), Decimal(2500))
    assert not report.ok


def test_sanity_check_passes_clean_values() -> None:
    report = check_multiples([0, 5000, 10000, Decimal(15000)], STEP)
    assert report.ok
    assert report.violation_count == 0
    assert report.violation_rate == 0.0


def test_sanity_check_skips_nulls_and_bounds_the_sample() -> None:
    report = check_multiples([None, 1, 2, 3, 4, 5, None], STEP, limit=2)
    assert report.checked == 5
    assert report.violation_count == 2
    assert report.violations == (Decimal(1), Decimal(2))


def test_sanity_check_with_nothing_checked_has_no_rate() -> None:
    report = check_multiples([None, None], STEP)
    assert report.checked == 0
    assert report.violation_rate == 0.0
    assert report.ok


def test_sanity_check_rejects_a_bad_step() -> None:
    with pytest.raises(ValueError, match="step must be positive"):
        check_multiples([1], 0)


def test_sanity_check_cannot_see_a_value_that_is_off_by_a_whole_step() -> None:
    """The erp exceptions are off by exactly one step, so they are still ON the 5000 grid.

    This is the honest limit of the multiplicity check: it catches a value that is not a multiple
    of the canonical step (a wrong template, or an unrounded raw value), **not** a value that is a
    correct multiple of the wrong amount. Those are caught by comparing the three storage forms
    against each other, which is what the cross-source agreement test below does.
    """
    world = build_world(WorldConfig(customers=500, events_per_customer=0))
    values = [row.limit_k * 10 for row in world.erp.customers]
    report = check_multiples(values, STEP)
    assert report.checked == len(values)
    assert report.violation_count == 0
    assert world.expected.erp_exceptions, "the world must still plant its exceptions"
    for exception in world.expected.erp_exceptions:
        assert exception.erp_limit_times_10 % STEP == 0
        assert exception.erp_limit_times_10 != exception.crm_limit


# --- reporting: nothing silent -----------------------------------------------------------


def test_applied_normalisations_are_reported() -> None:
    """Acceptance (principle 10): every conversion shows up for ``applied_fixes``."""
    applied = normalizations_for(
        CREDIT_LIMIT,
        {
            "crm.customers.credit_limit": "c.credit_limit",
            "billing.accounts.credit_limit": "b.credit_limit",
            "erp.customers.limit_k": "e.limit_k",
        },
        dialect="duckdb",
    )
    assert [item.column for item in applied] == [
        "crm.customers.credit_limit",
        "billing.accounts.credit_limit",
        "erp.customers.limit_k",
    ]
    by_column = {item.column: item for item in applied}
    assert by_column["crm.customers.credit_limit"].identity is True
    assert by_column["billing.accounts.credit_limit"].identity is False
    assert by_column["erp.customers.limit_k"].identity is False
    assert all(item.mode is RoundingMode.HALF_UP for item in applied)
    assert all(isinstance(item, AppliedNormalization) for item in applied)


def test_reporting_skips_columns_the_measure_does_not_list() -> None:
    applied = normalizations_for(CREDIT_LIMIT, {"nope.nope.nope": "x"}, dialect="duckdb")
    assert applied == ()
