"""Tests for stage D5 profiling (S04): statistics first, a bounded sample second, provenance always.

The task's acceptance criteria are the first group, then the four rules that make the stage safe to
run against a primary database every night:

* **statistics answer first** -- a table whose optimizer statistics are fresh and complete is
  distributed from those statistics, and with ``shape=False`` costs *no data query at all*;
* **stale statistics are not trusted** -- an old ``analyze`` time, or none at all, falls back to the
  sample instead of publishing a number nobody can date;
* **the fetch is bounded and reproducible** -- every statement carries a ``LIMIT`` or a seeded
  ``TABLESAMPLE``, the same seed produces the same statement and the same fact sheet, and a table
  over the cap is reported as a sample rather than the table;
* **nothing runs outside a quiet window** unless forced, proven against ``FakeAdapter.calls``.

No test sleeps, no test contacts a host: the sample comes from ``FakeAdapter``'s in-process DuckDB
and the ``pg_stats`` shape is a fixture. There is no PostgreSQL server on this machine, so the
"live statistics" path is covered by the pure parser plus the injected
:class:`~mdqm.pipeline.stages.profiling.OptimizerStatistics` port.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pytest
from ruamel.yaml import YAML

from mdqm.config.quiet import QuietWindow
from mdqm.models.catalog import Column, NormalizedType, Table, TableKind
from mdqm.pipeline.core import PipelineContext, StageStatus, run_pipeline
from mdqm.pipeline.stages.profiling import (
    GRANULARITY_LADDER,
    ColumnStat,
    FormatKind,
    ProfilingStage,
    Provenance,
    QuietWindowClosed,
    StaticStatistics,
    TopValue,
    detect_format,
    detect_granularity,
    parse_pg_stats,
    sample_query,
)
from mdqm.store.db import Store
from mdqm.testing.fake_adapter import FakeAdapter
from mdqm.testing.world import WorldConfig, build_world

#: 2026-05-01 10:00 UTC = 13:00 in Asia/Jerusalem: inside the working day, so not quiet.
NOON = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)

#: 21:00-07:00 local: the window that makes a 13:00 pass a refusal.
NIGHT = QuietWindow(tz="Asia/Jerusalem", start=time(21, 0), end=time(7, 0))

#: Bigger than every small-world table except ``usage_events``, so most reads are complete.
FULL_ROWS = 10_000


class ScriptedClock:
    """A clock a test drives: ``monotonic`` ticks, ``now`` is one injected instant."""

    def __init__(self, tick: float = 0.25, now: datetime = NOON) -> None:
        self._tick = tick
        self._reading = 0.0
        self._now = now

    def monotonic(self) -> float:
        self._reading += self._tick
        return self._reading

    def now(self) -> datetime:
        return self._now


@pytest.fixture(scope="module")
def world() -> Any:
    """One generated organisation, built once: the synthetic data every acceptance case uses."""
    return build_world(WorldConfig.small())


@pytest.fixture
def store(tmp_path: Path) -> Any:
    """An initialized metadata store under ``tmp_path``."""
    opened = Store(tmp_path / "mdqm.sqlite")
    opened.initialize()
    try:
        yield opened
    finally:
        opened.close()


def _ctx(
    tmp_path: Path, store: Store, *, quiet: tuple[QuietWindow, ...] = (), now: datetime = NOON
) -> PipelineContext:
    """A context rooted in ``tmp_path`` with an injected clock and no model."""
    return PipelineContext(
        store=store,
        exec_ts=now,
        semantic_root=tmp_path / "semantic",
        quiet_windows=quiet,
        clock=ScriptedClock(now=now),
    )


def _sheets(
    tmp_path: Path,
    store: Store,
    adapters: dict[str, FakeAdapter],
    *,
    quiet: tuple[QuietWindow, ...] = (),
    now: datetime = NOON,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Run the stage through the runner and return each source's generated fact sheet."""
    stage = ProfilingStage(adapters, **kwargs)
    ctx = _ctx(tmp_path, store, quiet=quiet, now=now)
    result = run_pipeline([stage], ctx, force=True)
    assert result.ok, result.describe()
    assert result.reports[0].status is StageStatus.RAN, result.describe()
    sheets: dict[str, dict[str, Any]] = {}
    for source in adapters:
        path = tmp_path / "semantic" / "generated" / f"profile_{source}.yaml"
        sheets[source] = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    return sheets


def _table(sheet: dict[str, Any], name: str) -> dict[str, Any]:
    """One table's entry in a fact sheet."""
    return next(entry for entry in sheet["tables"] if entry["table"] == name)


def _column(sheet: dict[str, Any], table: str, column: str) -> dict[str, Any]:
    """One column's entry in a fact sheet."""
    return next(item for item in _table(sheet, table)["columns"] if item["name"] == column)


# --- acceptance: numeric granularity on the synthetic world -----------------------------------


def test_crm_credit_limit_is_a_multiple_of_5000(tmp_path: Path, store: Store, world: Any) -> None:
    """crm rounds the limit to 5000, so every value fits and the granularity is 5000 at 100%."""
    adapter = FakeAdapter.for_world(world, name="crm")
    try:
        sheets = _sheets(
            tmp_path, store, {"crm": adapter}, sample_rows=FULL_ROWS, tables=("customers_crm",)
        )
        column = _column(sheets["crm"], "customers_crm", "credit_limit")
        assert column["granularity"]["value"] == 5000
        assert column["granularity"]["share"] == pytest.approx(1.0)
        assert column["value_shape_provenance"] == Provenance.SAMPLED
    finally:
        adapter.close()


def test_erp_limit_k_is_a_multiple_of_500(tmp_path: Path, store: Store, world: Any) -> None:
    """erp stores the rounded limit divided by ten, so the granularity is 500, not 5000."""
    adapter = FakeAdapter.for_world(world, name="erp")
    try:
        sheets = _sheets(
            tmp_path, store, {"erp": adapter}, sample_rows=FULL_ROWS, tables=("customers_erp",)
        )
        column = _column(sheets["erp"], "customers_erp", "limit_k")
        assert column["granularity"]["value"] == 500
        assert column["granularity"]["share"] == pytest.approx(1.0)
    finally:
        adapter.close()


def test_billing_raw_limit_has_granularity_one(tmp_path: Path, store: Store, world: Any) -> None:
    """billing keeps the raw limit: granularity 1, and 5000 fits only a rounding-error share.

    This is the case D8 must not confuse with crm: the *candidate* 5000 is reported alongside, so a
    reader sees that almost nothing fits it.
    """
    adapter = FakeAdapter.for_world(world, name="billing")
    try:
        sheets = _sheets(
            tmp_path,
            store,
            {"billing": adapter},
            sample_rows=FULL_ROWS,
            tables=("accounts",),
        )
        column = _column(sheets["billing"], "accounts", "credit_limit")
        assert column["granularity"]["value"] == 1
        shares = {item["value"]: item["share"] for item in column["granularity"]["candidates"]}
        assert set(shares) == set(GRANULARITY_LADDER)
        assert shares[5000] <= 0.05
    finally:
        adapter.close()


def test_client_no_is_detected_as_zero_padded_ten_digits(
    tmp_path: Path, store: Store, world: Any
) -> None:
    """``client_no`` is a join key on both sides: its shape must survive as a fact, not a guess."""
    adapter = FakeAdapter.for_world(world, name="billing")
    try:
        sheets = _sheets(
            tmp_path, store, {"billing": adapter}, sample_rows=FULL_ROWS, tables=("accounts",)
        )
        shape = _column(sheets["billing"], "accounts", "client_no")["format"]
        assert shape["kind"] == FormatKind.DIGITS
        assert shape["digits"] == 10
        assert shape["zero_padded"] is True
        assert shape["share"] == pytest.approx(1.0)
    finally:
        adapter.close()


def test_hebrew_text_is_profiled_without_mojibake(tmp_path: Path, store: Store, world: Any) -> None:
    """The world holds Hebrew names, so the sheet must keep them readable (UTF-8 end to end)."""
    adapter = FakeAdapter.for_world(world, name="crm")
    try:
        sheets = _sheets(
            tmp_path, store, {"crm": adapter}, sample_rows=FULL_ROWS, tables=("customers_crm",)
        )
        shape = _column(sheets["crm"], "customers_crm", "full_name")["format"]
        assert shape["kind"] == FormatKind.LETTERS
        rendered = (tmp_path / "semantic" / "generated" / "profile_crm.yaml").read_text(
            encoding="utf-8"
        )
        assert "\\u" not in rendered
    finally:
        adapter.close()


# --- acceptance: a table over the cap is a sample, and says so --------------------------------


def test_a_table_over_the_row_cap_is_sampled_and_says_so(
    tmp_path: Path, store: Store, world: Any
) -> None:
    """200 rows against a 50-row cap: never a scan, and ``sample_complete`` tells the truth."""
    adapter = FakeAdapter.for_world(world, name="crm")
    try:
        sheets = _sheets(
            tmp_path, store, {"crm": adapter}, sample_rows=50, tables=("customers_crm",)
        )
        table = _table(sheets["crm"], "customers_crm")
        assert table["rows_estimate"] == 200
        assert table["sample_complete"] is False
        assert table["rows_sampled"] <= 50
        assert table["sample_method"] in {"limit", "tablesample"}
        assert table["sample_rows_cap"] == 50
    finally:
        adapter.close()


def test_a_table_within_the_cap_is_reported_complete(
    tmp_path: Path, store: Store, world: Any
) -> None:
    """The same table under a larger cap is complete, so the two cases cannot be confused."""
    adapter = FakeAdapter.for_world(world, name="crm")
    try:
        sheets = _sheets(
            tmp_path, store, {"crm": adapter}, sample_rows=FULL_ROWS, tables=("customers_crm",)
        )
        table = _table(sheets["crm"], "customers_crm")
        assert table["sample_complete"] is True
        assert table["rows_sampled"] == 200
    finally:
        adapter.close()


def test_every_statement_is_bounded(tmp_path: Path, store: Store, world: Any) -> None:
    """Principle 6: no profile statement may be a scan. Each one ends in a bound."""
    adapter = FakeAdapter.for_world(world, name="crm")
    try:
        _sheets(tmp_path, store, {"crm": adapter}, sample_rows=40)
        assert adapter.calls, "the stage must read some rows to shape a column"
        for sql in adapter.calls:
            assert "LIMIT" in sql.upper() or "TABLESAMPLE" in sql.upper(), sql
            assert "WHERE" not in sql.upper(), "profiling never filters; a filter invites a scan"
    finally:
        adapter.close()


# --- acceptance: nothing runs outside a quiet window ------------------------------------------


def test_no_query_outside_the_quiet_window(tmp_path: Path, store: Store, world: Any) -> None:
    """The runner refuses the heavy stage, so the adapter is never touched at 13:00."""
    adapter = FakeAdapter.for_world(world, name="crm")
    try:
        stage = ProfilingStage({"crm": adapter})
        ctx = _ctx(tmp_path, store, quiet=(NIGHT,))
        result = run_pipeline([stage], ctx)
        assert result.reports[0].status is StageStatus.REFUSED
        assert result.reports[0].deferred_until is not None
        assert adapter.calls == []
    finally:
        adapter.close()


def test_a_forced_pass_profiles_outside_the_window(
    tmp_path: Path, store: Store, world: Any
) -> None:
    """``--force`` is the only way past the window, and it is recorded on the report."""
    adapter = FakeAdapter.for_world(world, name="crm")
    try:
        stage = ProfilingStage({"crm": adapter})
        ctx = _ctx(tmp_path, store, quiet=(NIGHT,))
        result = run_pipeline([stage], ctx, force=True)
        assert result.reports[0].status is StageStatus.RAN
        assert result.reports[0].forced is True
        assert adapter.calls
    finally:
        adapter.close()


def test_a_direct_caller_cannot_skip_the_quiet_window(
    tmp_path: Path, store: Store, world: Any
) -> None:
    """Bypassing the runner must not become a way to profile a primary database in the afternoon."""
    adapter = FakeAdapter.for_world(world, name="crm")
    try:
        stage = ProfilingStage({"crm": adapter})
        with pytest.raises(QuietWindowClosed, match="quiet window is closed"):
            stage.run(_ctx(tmp_path, store, quiet=(NIGHT,)))
        assert adapter.calls == []
    finally:
        adapter.close()


# --- statistics are preferred, and only sampling fills the gaps -------------------------------


def _stats(table: str, column: str, *, analyzed_at: datetime | None) -> ColumnStat:
    """One fresh-looking statistics row, as a vendor adapter would hand it over."""
    return ColumnStat(
        table=table,
        column=column,
        null_fraction=0.02,
        distinct=4,
        minimum="0",
        maximum="60000",
        top_values=(TopValue(value="5000", count=100),),
        analyzed_at=analyzed_at,
    )


def test_fresh_statistics_answer_the_distribution(tmp_path: Path, store: Store, world: Any) -> None:
    """null rate, distinct, bounds and top values come from the statistics, not from rows."""
    adapter = FakeAdapter.for_world(world, name="billing")
    try:
        fresh = StaticStatistics(
            {
                "accounts": (
                    _stats("accounts", "credit_limit", analyzed_at=NOON - timedelta(days=1)),
                )
            }
        )
        sheets = _sheets(
            tmp_path,
            store,
            {"billing": adapter},
            statistics={"billing": fresh},
            sample_rows=FULL_ROWS,
            tables=("accounts",),
        )
        column = _column(sheets["billing"], "accounts", "credit_limit")
        assert column["distribution_provenance"] == Provenance.STATS
        assert column["null_rate"] == pytest.approx(0.02)
        assert column["distinct"] == 4
        assert column["minimum"] == "0"
        assert column["maximum"] == "60000"
        assert column["top_values"] == [{"value": "5000", "count": 100}]
        assert column["stats_analyzed_at"] is not None
        # Value shape still needs values: no statistics row can say what 5000 means.
        assert column["value_shape_provenance"] == Provenance.SAMPLED
        assert column["granularity"]["value"] == 1
    finally:
        adapter.close()


def test_a_stats_only_pass_costs_no_data_query(tmp_path: Path, store: Store, world: Any) -> None:
    """``shape=False`` with complete fresh statistics: the cheapest possible profile, zero rows."""
    adapter = FakeAdapter.for_world(world, name="billing")
    try:
        names = tuple(
            column.name
            for table in adapter.extract_catalog().tables
            if table.name == "accounts"
            for column in table.columns
        )
        stats = StaticStatistics(
            {
                "accounts": tuple(
                    _stats("accounts", name, analyzed_at=NOON - timedelta(days=1)) for name in names
                )
            }
        )
        sheets = _sheets(
            tmp_path,
            store,
            {"billing": adapter},
            statistics={"billing": stats},
            shape=False,
            tables=("accounts",),
        )
        assert adapter.calls == [], "fresh statistics must save the data query entirely"
        table = _table(sheets["billing"], "accounts")
        assert table["rows_sampled"] == 0
        assert table["sample_method"] == "none"
        assert table["rows_estimate_provenance"] == Provenance.STATS
        for column in table["columns"]:
            assert column["distribution_provenance"] == Provenance.STATS
            assert column["value_shape_provenance"] == Provenance.ABSENT
    finally:
        adapter.close()


def test_stale_statistics_fall_back_to_the_sample(tmp_path: Path, store: Store, world: Any) -> None:
    """An ``analyze`` older than the bound is not trusted, so the sample answers instead."""
    adapter = FakeAdapter.for_world(world, name="billing")
    try:
        stale = StaticStatistics(
            {
                "accounts": (
                    _stats("accounts", "credit_limit", analyzed_at=NOON - timedelta(days=30)),
                )
            }
        )
        sheets = _sheets(
            tmp_path,
            store,
            {"billing": adapter},
            statistics={"billing": stale},
            sample_rows=FULL_ROWS,
            tables=("accounts",),
        )
        column = _column(sheets["billing"], "accounts", "credit_limit")
        assert column["distribution_provenance"] == Provenance.SAMPLED
        assert column["top_values"] != [{"value": "5000", "count": 100}]
    finally:
        adapter.close()


def test_statistics_without_an_analyze_time_are_treated_as_stale(
    tmp_path: Path, store: Store, world: Any
) -> None:
    """Freshness that cannot be dated is not freshness: the conservative reading, not the cheap
    one."""
    adapter = FakeAdapter.for_world(world, name="billing")
    try:
        undated = StaticStatistics(
            {"accounts": (_stats("accounts", "credit_limit", analyzed_at=None),)}
        )
        sheets = _sheets(
            tmp_path,
            store,
            {"billing": adapter},
            statistics={"billing": undated},
            sample_rows=FULL_ROWS,
            tables=("accounts",),
        )
        column = _column(sheets["billing"], "accounts", "credit_limit")
        assert column["distribution_provenance"] == Provenance.SAMPLED
        assert column["stats_analyzed_at"] is None
    finally:
        adapter.close()


def test_a_sample_only_column_reports_no_statistics_at_all(
    tmp_path: Path, store: Store, world: Any
) -> None:
    """A source with no statistics reader gets ``sampled``, never a number claiming to be stats."""
    adapter = FakeAdapter.for_world(world, name="crm")
    try:
        sheets = _sheets(
            tmp_path, store, {"crm": adapter}, sample_rows=FULL_ROWS, tables=("customers_crm",)
        )
        column = _column(sheets["crm"], "customers_crm", "credit_limit")
        assert column["distribution_provenance"] == Provenance.SAMPLED
        assert column["stats_analyzed_at"] is None
        assert column["distinct_exact"] is True
    finally:
        adapter.close()


# --- the pg_stats shape, as a pure function ---------------------------------------------------


def test_parse_pg_stats_reads_the_real_column_names() -> None:
    """``pg_stats`` in, source-neutral facts out: the one place the mapping lives."""
    rows = [
        {
            "schemaname": "public",
            "tablename": "customers",
            "attname": "credit_limit",
            "null_frac": 0.0,
            "n_distinct": 12.0,
            "most_common_vals": ["5000", "10000"],
            "most_common_freqs": [0.25, 0.125],
            "histogram_bounds": ["0", "60000"],
        },
        {
            "tablename": "customers",
            "attname": "client_no",
            "null_frac": 0.01,
            "n_distinct": -0.5,
            "most_common_vals": None,
            "most_common_freqs": None,
            "histogram_bounds": ["0000000001", "0000000200"],
        },
    ]
    parsed = parse_pg_stats(rows, row_estimate=200, analyzed_at=NOON)
    assert [stat.column for stat in parsed] == ["client_no", "credit_limit"]
    credit, client = parsed[1], parsed[0]
    assert credit.distinct == 12
    assert credit.minimum == "0"
    assert credit.maximum == "60000"
    assert credit.top_values == (
        TopValue(value="5000", count=50),
        TopValue(value="10000", count=25),
    )
    # A negative n_distinct is a fraction of the rows: "too many distinct values to count".
    assert client.distinct == 100
    assert client.null_fraction == pytest.approx(0.01)
    assert client.answers_distribution() is True


def test_parse_pg_stats_falls_back_to_the_common_values_for_bounds() -> None:
    """A low-cardinality column has no histogram; the MCVs carry its bounds."""
    parsed = parse_pg_stats(
        [
            {
                "tablename": "customers",
                "attname": "status",
                "null_frac": 0.0,
                "n_distinct": 3.0,
                "most_common_vals": ["A", "C", "S"],
                "most_common_freqs": [0.5, 0.3, 0.2],
                "histogram_bounds": None,
            }
        ],
        row_estimate=100,
    )
    assert parsed[0].minimum == "A"
    assert parsed[0].maximum == "S"


def test_parse_pg_stats_skips_a_row_without_a_column_name() -> None:
    """A malformed row is dropped, never published under an empty name."""
    assert parse_pg_stats([{"tablename": "t", "attname": None}], row_estimate=1) == ()


def test_static_statistics_returns_none_for_an_unknown_table() -> None:
    """The port answers ``None`` for a table it does not describe, which means "sample it"."""
    provider = StaticStatistics({"a": ()})
    assert provider.column_stats("b") is None
    assert provider.column_stats("a") == ()


# --- value shaping, as pure functions ---------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "granularity", "share"),
    [
        ([0, 5000, 10000, 60000], 5000, 1.0),
        ([0, 500, 1000], 500, 1.0),
        ([0, 500, 1000, 1500], 500, 1.0),
        ([1, 2, 3], 1, 1.0),
        ([0, 5000, 4133], 1, 1.0),
        ([None, None], None, None),
        ([], None, None),
        (["5000", "10000"], 5000, 1.0),
    ],
)
def test_detect_granularity_tables(
    values: list[Any], granularity: int | None, share: float | None
) -> None:
    """The ladder table, including NULLs, strings and a column that is not uniform."""
    detected = detect_granularity(values)
    if granularity is None:
        assert detected is None
    else:
        assert detected is not None
        assert detected.value == granularity
        assert detected.share == pytest.approx(share)  # type: ignore[arg-type]


def test_detect_granularity_ignores_a_single_stray_value() -> None:
    """99 of 100 multiples of 5000 stays 5000: the threshold exists so one dirty row cannot lie."""
    detected = detect_granularity([*([5000] * 99), 4321])
    assert detected is not None
    assert detected.value == 5000
    assert detected.share == pytest.approx(0.99)


def test_detect_granularity_rejects_a_decimal_column() -> None:
    """Nothing on the integer ladder fits a fractional column, so it reports no granularity."""
    assert detect_granularity([1.5, 2.25, 3.75]) is None


@pytest.mark.parametrize(
    ("values", "kind", "digits", "zero_padded"),
    [
        (["0000000001", "0000000200"], FormatKind.DIGITS, 10, True),
        (["1", "22", "333"], FormatKind.DIGITS, None, False),
        (["A", "C", "S"], FormatKind.LETTERS, 1, False),
        (["David Cohen", "Yoni Levi"], FormatKind.LETTERS, None, False),
        (["a@b.com", "x1@y.org"], FormatKind.ALPHANUMERIC, None, False),
    ],
)
def test_detect_format_tables(
    values: list[str], kind: FormatKind, digits: int | None, zero_padded: bool
) -> None:
    """The shape table: uniform width, mixed width, letters and a non-letter alphanumeric."""
    detected = detect_format(values)
    assert detected is not None
    assert detected.kind is kind
    assert detected.digits == digits
    assert detected.zero_padded is zero_padded


def test_detect_format_reports_a_minority_shape_as_a_share() -> None:
    """A mostly-numeric column says so with a share below one instead of hiding the exceptions."""
    detected = detect_format(["123", "456", "abc"])
    assert detected is not None
    assert detected.kind is FormatKind.DIGITS
    assert detected.share == pytest.approx(2 / 3)


def test_detect_format_has_no_answer_for_empty_values() -> None:
    """All-NULL produces no shape rather than a fabricated one."""
    assert detect_format([None, None]) is None


# --- the statement: bounded, dialect-correct, seeded -----------------------------------------


def _catalog_table(name: str = "customers", *, schema: str | None = None) -> Table:
    """The smallest catalog table the SQL builder needs."""
    return Table(
        name=name,
        schema_name=schema,
        kind=TableKind.TABLE,
        columns=(Column(name="a", native_type="INT", normalized_type=NormalizedType.INTEGER),),
    )


def test_sample_query_bounds_with_limit_by_default() -> None:
    """Without the source's own sampling, a plain LIMIT is the cheap, portable bound."""
    plan = sample_query(
        _catalog_table(), dialect="duckdb", rows=100, seed=7, native=False, row_estimate=None
    )
    assert plan is not None
    assert plan.method == "limit"
    assert "LIMIT 100" in plan.sql


@pytest.mark.parametrize(
    ("dialect", "expected"),
    [("oracle", "FETCH FIRST 100 ROWS ONLY"), ("tsql", "TOP 100")],
)
def test_sample_query_uses_each_dialects_own_row_bound(dialect: str, expected: str) -> None:
    """sqlglot renders the bound, so no dialect table lives in this module."""
    plan = sample_query(
        _catalog_table(), dialect=dialect, rows=100, seed=7, native=False, row_estimate=None
    )
    assert plan is not None
    assert expected in plan.sql


def test_sample_query_uses_seeded_native_sampling_when_declared() -> None:
    """With the ``sampling`` capability, a seeded reservoir sample replaces the plain bound."""
    plan = sample_query(
        _catalog_table(), dialect="duckdb", rows=100, seed=7, native=True, row_estimate=100_000
    )
    assert plan is not None
    assert plan.method == "tablesample"
    assert "TABLESAMPLE" in plan.sql.upper()
    assert "REPEATABLE" in plan.sql.upper()
    again = sample_query(
        _catalog_table(), dialect="duckdb", rows=100, seed=7, native=True, row_estimate=100_000
    )
    assert again is not None
    assert again.sql == plan.sql, "the same seed must build the same statement"
    other = sample_query(
        _catalog_table(), dialect="duckdb", rows=100, seed=8, native=True, row_estimate=100_000
    )
    assert other is not None
    assert other.sql != plan.sql


def test_sample_query_keeps_a_small_table_on_limit() -> None:
    """A percentage sample of a small Postgres table can return nothing: it falls back to LIMIT."""
    plan = sample_query(
        _catalog_table(), dialect="postgres", rows=100, seed=7, native=True, row_estimate=200
    )
    assert plan is not None
    assert plan.method == "limit"


def test_sample_query_refuses_a_name_that_is_not_an_identifier() -> None:
    """The catalog is trusted for the name, and still gated: nothing weird reaches SQL."""
    with pytest.raises(ValueError, match="plain identifier"):
        sample_query(
            _catalog_table("customers; DROP TABLE x"),
            dialect="duckdb",
            rows=10,
            seed=0,
            native=False,
            row_estimate=None,
        )


def test_sample_query_has_no_statement_for_a_non_sql_dialect() -> None:
    """Elasticsearch and API sources carry no SQL: the stage then reads statistics only."""
    assert (
        sample_query(
            _catalog_table(), dialect="dsl", rows=10, seed=0, native=False, row_estimate=None
        )
        is None
    )


def test_sample_query_qualifies_a_schema() -> None:
    """A schema-qualified catalog table keeps its schema in the statement."""
    plan = sample_query(
        _catalog_table("customers", schema="public"),
        dialect="postgres",
        rows=10,
        seed=0,
        native=False,
        row_estimate=None,
    )
    assert plan is not None
    assert '"public"' in plan.sql


# --- determinism, the input hash and reporting -----------------------------------------------


def test_two_runs_on_the_same_data_produce_the_same_fact_sheet(tmp_path: Path, world: Any) -> None:
    """Reproducible by construction: same seed, same rows, byte-identical output."""
    rendered: list[str] = []
    for run in ("first", "second"):
        root = tmp_path / run
        root.mkdir()
        opened = Store(root / "mdqm.sqlite")
        opened.initialize()
        adapter = FakeAdapter.for_world(world, name="crm")
        try:
            _sheets(
                root,
                opened,
                {"crm": adapter},
                sample_rows=FULL_ROWS,
                tables=("customers_crm",),
            )
            rendered.append(
                (root / "semantic" / "generated" / "profile_crm.yaml").read_text(encoding="utf-8")
            )
        finally:
            adapter.close()
            opened.close()
    assert rendered[0] == rendered[1]


def test_the_input_hash_moves_with_the_day_and_the_seed(
    tmp_path: Path, store: Store, world: Any
) -> None:
    """A profile is a fact about a day: tomorrow re-measures, today skips, the seed matters."""
    adapter = FakeAdapter.for_world(world, name="crm")
    try:
        stage = ProfilingStage({"crm": adapter}, seed=1)
        today = stage.inputs_hash(_ctx(tmp_path, store))
        tomorrow = stage.inputs_hash(_ctx(tmp_path, store, now=NOON + timedelta(days=1)))
        assert today != tomorrow
        assert stage.inputs_hash(_ctx(tmp_path, store)) == today
        assert ProfilingStage({"crm": adapter}, seed=2).inputs_hash(_ctx(tmp_path, store)) != today
    finally:
        adapter.close()


def test_an_unchanged_input_skips_the_second_pass(tmp_path: Path, store: Store, world: Any) -> None:
    """The framework's skip applies: one day's profile is not recomputed on every pass."""
    adapter = FakeAdapter.for_world(world, name="crm")
    try:
        stage = ProfilingStage({"crm": adapter}, sample_rows=FULL_ROWS)
        first = run_pipeline([stage], _ctx(tmp_path, store), force=True)
        assert first.reports[0].status is StageStatus.RAN
        calls_after_first = len(adapter.calls)
        second = run_pipeline([stage], _ctx(tmp_path, store))
        assert second.reports[0].status is StageStatus.SKIPPED
        assert len(adapter.calls) == calls_after_first
    finally:
        adapter.close()


def test_the_report_counts_what_was_measured_and_what_was_left_out(
    tmp_path: Path, store: Store, world: Any
) -> None:
    """Principle 10: counters, skips and fixes travel with the run, not into the log file only."""
    adapter = FakeAdapter.for_world(world, name="crm")
    try:
        stage = ProfilingStage({"crm": adapter}, sample_rows=FULL_ROWS)
        result = run_pipeline([stage], _ctx(tmp_path, store), force=True)
        report = result.reports[0]
        assert report.counts["tables"] >= 1
        assert report.counts["columns"] >= 1
        assert report.counts["sampled_tables"] >= 1
        assert report.outputs == ("generated/profile_crm.yaml",)
        assert report.inputs_hash
    finally:
        adapter.close()


def test_an_unknown_source_fails_loudly(tmp_path: Path, store: Store, world: Any) -> None:
    """``--source`` naming no adapter is a configuration error, not a silent empty sheet."""
    adapter = FakeAdapter.for_world(world, name="crm")
    try:
        stage = ProfilingStage({"crm": adapter})
        ctx = _ctx(tmp_path, store)
        ctx.source = "billing"
        with pytest.raises(ValueError, match="no adapter for source"):
            stage.run(ctx)
    finally:
        adapter.close()


def test_only_the_named_table_is_profiled(tmp_path: Path, store: Store, world: Any) -> None:
    """``tables=`` narrows the pass, which is how an operator re-profiles one broken table."""
    adapter = FakeAdapter.for_world(world, name="crm")
    try:
        sheets = _sheets(
            tmp_path, store, {"crm": adapter}, sample_rows=FULL_ROWS, tables=("customers_crm",)
        )
        assert [entry["table"] for entry in sheets["crm"]["tables"]] == ["customers_crm"]
    finally:
        adapter.close()


def test_a_non_table_kind_is_skipped_in_words(tmp_path: Path, store: Store, world: Any) -> None:
    """An endpoint or a data stream is not a D5 subject; the report says so instead of guessing."""
    adapter = FakeAdapter.for_world(world, name="crm")
    try:
        table = Table(
            name="search_things",
            kind=TableKind.DATA_STREAM,
            columns=(Column(name="a", native_type="TEXT", normalized_type=NormalizedType.STRING),),
        )
        stage = ProfilingStage({"crm": adapter})
        skipped: list[str] = []
        assert (
            stage._profile_table(
                "crm", adapter, table, exec_ts=NOON, skipped=skipped, fixes=[], notes=[]
            )
            is None
        )
        assert skipped == ["crm.search_things: data_stream is not profiled by D5"]
    finally:
        adapter.close()
