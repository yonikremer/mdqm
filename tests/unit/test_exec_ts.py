"""Tests for the ``exec_ts`` rewriter (task C02).

The contract: one ``exec_ts`` per execution, injected; every non-deterministic time function
and the interval arithmetic on it becomes an absolute literal; rewriting twice changes nothing;
and the audit trail names every substitution with its position.
"""

from __future__ import annotations

import copy
import re
from datetime import UTC, datetime, timedelta, timezone

import pytest
import sqlglot
from hypothesis import given, settings
from hypothesis import strategies as st

from mdqm.sql.exec_ts import (
    SUPPORTED_DIALECTS,
    ExecTsMode,
    resolve_exec_ts,
    rewrite_es_date_math,
    rewrite_es_query,
    rewrite_sql,
)
from unit.test_query_corpus import NOW_FUNCTIONS, CorpusStatement, load_corpus

JERUSALEM = "Asia/Jerusalem"
EXEC_TS = datetime(2026, 11, 25, 10, 0, tzinfo=UTC)
TS = "2026-11-25 10:00:00"
DATE = "2026-11-25"

#: Names this module knows beyond the corpus' closed set (sqlglot normalises most of them, so
#: they only appear when written for a dialect that lacks them).
EXTRA_NOW_NAMES = frozenset(
    {
        "curdate",
        "curtime",
        "getutcdate",
        "localtime",
        "sysdatetimeoffset",
        "sysutcdatetime",
        "utc_date",
        "utc_timestamp",
    }
)
ALL_NOW_NAMES = NOW_FUNCTIONS | EXTRA_NOW_NAMES

CORPUS = load_corpus()[0]


def now_names(sql: str) -> set[str]:
    """Every now-like function name in ``sql``, lower case."""
    return {
        name
        for name in ALL_NOW_NAMES
        if re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", sql, re.IGNORECASE)
    }


# ---------------------------------------------------------------------------------------------
# resolve_exec_ts: frozen vs relative, one value per execution
# ---------------------------------------------------------------------------------------------


def test_frozen_mode_uses_the_stored_value() -> None:
    stored = datetime(2026, 11, 25, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    resolved = resolve_exec_ts(ExecTsMode.FROZEN, stored=stored)
    assert resolved == EXEC_TS
    assert resolved.tzinfo is UTC


def test_relative_mode_asks_the_injected_clock() -> None:
    calls: list[int] = []

    def clock() -> datetime:
        calls.append(1)
        return EXEC_TS

    assert resolve_exec_ts(ExecTsMode.RELATIVE, clock=clock) == EXEC_TS
    assert calls == [1], "the clock is read exactly once per execution"


@pytest.mark.parametrize(
    ("mode", "kwargs"),
    [
        (ExecTsMode.FROZEN, {}),
        (ExecTsMode.RELATIVE, {}),
        (ExecTsMode.FROZEN, {"stored": datetime(2026, 11, 25, 10, 0)}),
        (ExecTsMode.RELATIVE, {"clock": lambda: datetime(2026, 11, 25, 10, 0)}),
    ],
    ids=["frozen-without-stored", "relative-without-clock", "naive-stored", "naive-clock"],
)
def test_resolve_exec_ts_rejects_missing_or_naive_input(
    mode: ExecTsMode, kwargs: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="timezone-aware|needs"):
        resolve_exec_ts(mode, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# One table of every rewrite, per dialect
# ---------------------------------------------------------------------------------------------

REWRITES: tuple[tuple[str, str, str], ...] = (
    # PostgreSQL / DuckDB
    ("postgres", "now()", f"CAST('{TS}' AS TIMESTAMP)"),
    ("postgres", "CURRENT_TIMESTAMP", f"CAST('{TS}' AS TIMESTAMP)"),
    ("postgres", "CURRENT_DATE", f"CAST('{DATE}' AS DATE)"),
    ("postgres", "LOCALTIMESTAMP", f"CAST('{TS}' AS TIMESTAMP)"),
    ("duckdb", "now()", f"CAST('{TS}' AS TIMESTAMP)"),
    ("duckdb", "CURRENT_DATE", f"CAST('{DATE}' AS DATE)"),
    # interval arithmetic folds into one literal
    ("postgres", "now() - interval '7 days'", f"CAST('2026-11-18 {TS[11:]}' AS TIMESTAMP)"),
    ("postgres", "now() - interval '30 days'", f"CAST('2026-10-26 {TS[11:]}' AS TIMESTAMP)"),
    ("postgres", "now() + interval '2 hours'", "CAST('2026-11-25 12:00:00' AS TIMESTAMP)"),
    ("postgres", "CURRENT_DATE - interval '7 days'", f"CAST('2026-11-18 {TS[11:]}' AS TIMESTAMP)"),
    ("postgres", "now() - interval '1 year'", f"CAST('2025-11-25 {TS[11:]}' AS TIMESTAMP)"),
    # wrappers keep their meaning; only the now value inside them is replaced
    (
        "postgres",
        "DATE_TRUNC('month', now())",
        f"DATE_TRUNC('MONTH', CAST('{TS}' AS TIMESTAMP))",
    ),
    # Oracle
    ("oracle", "SYSDATE", f"CAST('{TS}' AS TIMESTAMP)"),
    ("oracle", "SYSTIMESTAMP", f"CAST('{TS}' AS TIMESTAMP)"),
    ("oracle", "CURRENT_DATE", f"CAST('{DATE}' AS DATE)"),
    ("oracle", "LOCALTIMESTAMP", f"CAST('{TS}' AS TIMESTAMP)"),
    ("oracle", "SYSDATE - 7", f"CAST('2026-11-18 {TS[11:]}' AS TIMESTAMP)"),
    ("oracle", "SYSDATE - INTERVAL '6' HOUR", "CAST('2026-11-25 04:00:00' AS TIMESTAMP)"),
    ("oracle", "SYSDATE - 1 / 24", "CAST('2026-11-25 09:00:00' AS TIMESTAMP)"),
    (
        "oracle",
        "ADD_MONTHS(SYSDATE, -3)",
        f"ADD_MONTHS(CAST('{TS}' AS TIMESTAMP), -3)",
    ),
    (
        "oracle",
        "TRUNC(SYSDATE, 'MM')",
        f"TRUNC(CAST('{TS}' AS TIMESTAMP), 'MONTH')",
    ),
    # T-SQL
    ("tsql", "GETDATE()", f"CAST('{TS}' AS DATETIME2)"),
    ("tsql", "SYSDATETIME()", f"CAST('{TS}' AS DATETIME2)"),
    ("tsql", "CURRENT_TIMESTAMP", f"CAST('{TS}' AS DATETIME2)"),
    ("tsql", "GETDATE() - 7", f"CAST('2026-11-18 {TS[11:]}' AS DATETIME2)"),
    (
        "tsql",
        "DATEADD(day, -7, GETDATE())",
        f"CAST('2026-11-18 {TS[11:]}' AS DATETIME2)",
    ),
    (
        "tsql",
        "DATEADD(minute, -90, GETDATE())",
        "CAST('2026-11-25 08:30:00' AS DATETIME2)",
    ),
    # MySQL
    ("mysql", "NOW()", f"CAST('{TS}' AS DATETIME)"),
    ("mysql", "CURDATE()", f"CAST('{DATE}' AS DATE)"),
    ("mysql", "DATE_SUB(NOW(), INTERVAL 7 DAY)", f"CAST('2026-11-18 {TS[11:]}' AS DATETIME)"),
)

REWRITE_IDS = [f"{dialect}:{source}" for dialect, source, _ in REWRITES]


@pytest.mark.parametrize(("dialect", "source", "expected"), REWRITES, ids=REWRITE_IDS)
def test_rewrite_replaces_every_now_like_spelling(dialect: str, source: str, expected: str) -> None:
    result = rewrite_sql(f"SELECT 1 WHERE ts >= {source}", dialect=dialect, exec_ts=EXEC_TS)
    assert result.sql == f"SELECT 1 WHERE ts >= {expected}"


@pytest.mark.parametrize(("dialect", "source", "expected"), REWRITES, ids=REWRITE_IDS)
def test_rewrite_reports_what_it_changed(dialect: str, source: str, expected: str) -> None:
    sql = f"SELECT 1 WHERE ts >= {source}"
    result = rewrite_sql(sql, dialect=dialect, exec_ts=EXEC_TS)
    assert len(result.replacements) == 1
    replacement = result.replacements[0]
    # A wrapper keeps its function, so only the inner now value is replaced there.
    assert replacement.replaced_with in expected
    assert now_names(replacement.original), "the audit names the expression that was replaced"


POSITION_CASES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("postgres", "SELECT 1 WHERE ts >= now()", ("now",)),
    ("postgres", "SELECT 1 WHERE ts >= now() - interval '7 days'", ("now",)),
    ("postgres", "SELECT now(), now() FROM t", ("now", "now")),
    ("duckdb", "SELECT 1 WHERE ts >= now() + interval '1 hour'", ("now",)),
    ("tsql", "SELECT 1 WHERE ts >= SYSDATETIME()", ("SYSDATETIME",)),
    ("tsql", "SELECT 1 WHERE ts >= DATEADD(day, -7, GETDATE())", ("DATEADD",)),
    ("tsql", "SELECT 1 WHERE ts >= GETDATE() - 7 AND u < GETDATE()", ("GETDATE", "GETDATE")),
    ("mysql", "SELECT 1 WHERE ts >= NOW()", ("NOW",)),
)


def _offsets(sql: str, tokens: tuple[str, ...]) -> tuple[int, ...]:
    """Where those spellings literally start in ``sql``, found without sqlglot."""
    found: list[int] = []
    cursor = 0
    for token in tokens:
        at = sql.index(token, cursor)
        found.append(at)
        cursor = at + 1
    return tuple(found)


@pytest.mark.parametrize(
    ("dialect", "sql", "tokens"),
    POSITION_CASES,
    ids=[sql[:44] for _, sql, _ in POSITION_CASES],
)
def test_positions_point_at_the_replaced_function(
    dialect: str, sql: str, tokens: tuple[str, ...]
) -> None:
    result = rewrite_sql(sql, dialect=dialect, exec_ts=EXEC_TS)
    assert tuple(item.position for item in result.replacements) == _offsets(sql, tokens)


BARE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("postgres", "SELECT 1 WHERE ts >= CURRENT_DATE"),
    ("oracle", "SELECT 1 WHERE ts >= SYSDATE - 7"),
    ("oracle", "SELECT 1 WHERE ts >= TRUNC(SYSDATE, 'MM')"),
    ("oracle", "SELECT 1 WHERE ts >= SYSTIMESTAMP - 7"),
    ("tsql", "SELECT 1 WHERE ts >= CURRENT_TIMESTAMP"),
)


@pytest.mark.parametrize(
    ("dialect", "sql"), BARE_KEYWORDS, ids=[sql[:40] for _, sql in BARE_KEYWORDS]
)
def test_bare_keywords_carry_no_parser_position(dialect: str, sql: str) -> None:
    """sqlglot records no offset for keyword tokens; the audit still names the expression."""
    result = rewrite_sql(sql, dialect=dialect, exec_ts=EXEC_TS)
    assert len(result.replacements) == 1
    replacement = result.replacements[0]
    assert replacement.position is None
    assert now_names(replacement.original)


def test_one_exec_ts_value_for_every_occurrence() -> None:
    """AGENTS.md section 5: one exec_ts per execution, not one clock read per function."""
    sql = "SELECT now(), now() AS b, CURRENT_TIMESTAMP, CURRENT_DATE FROM t WHERE ts >= now()"
    result = rewrite_sql(sql, dialect="postgres", exec_ts=EXEC_TS)
    literals = re.findall(r"CAST\('([^']+)' AS", result.sql)
    assert len(literals) == 5
    assert {value[:10] for value in literals} == {DATE}
    assert literals[0] == literals[1] == literals[2] == literals[4] == TS
    assert len(result.replacements) == 5


def test_a_timestamp_with_microseconds_keeps_them() -> None:
    exec_ts = datetime(2026, 11, 25, 10, 0, 0, 123456, tzinfo=UTC)
    result = rewrite_sql("SELECT now()", dialect="postgres", exec_ts=exec_ts)
    assert result.sql == "SELECT CAST('2026-11-25 10:00:00.123456' AS TIMESTAMP)"


# ---------------------------------------------------------------------------------------------
# Time zones, including the DST edge the acceptance asks for
# ---------------------------------------------------------------------------------------------

TIME_ZONE_CASES: tuple[tuple[str, datetime, str], ...] = (
    # Asia/Jerusalem ends DST on Sunday 2026-10-25 at 02:00 local (UTC+3 -> UTC+2).
    (
        "before the change (UTC+3)",
        datetime(2026, 10, 24, 12, 0, tzinfo=UTC),
        "2026-10-24 15:00:00",
    ),
    (
        "one hour before the change (UTC+3)",
        datetime(2026, 10, 24, 22, 30, tzinfo=UTC),
        "2026-10-25 01:30:00",
    ),
    (
        "after the change (UTC+2)",
        datetime(2026, 10, 25, 12, 0, tzinfo=UTC),
        "2026-10-25 14:00:00",
    ),
    (
        "a fixed instant, no DST in UTC",
        datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        "2026-01-15 14:00:00",
    ),
)


@pytest.mark.parametrize(
    ("exec_ts", "expected"),
    [(exec_ts, expected) for _, exec_ts, expected in TIME_ZONE_CASES],
    ids=[label for label, _, _ in TIME_ZONE_CASES],
)
def test_literal_is_rendered_in_the_session_time_zone(exec_ts: datetime, expected: str) -> None:
    result = rewrite_sql(
        "SELECT CURRENT_TIMESTAMP", dialect="postgres", exec_ts=exec_ts, session_time_zone=JERUSALEM
    )
    assert result.sql == f"SELECT CAST('{expected}' AS TIMESTAMP)"


def test_elapsed_interval_crosses_the_dst_change_as_an_instant() -> None:
    """24h before 10:00Z on the day after the change is 13:00 local, not 12:00."""
    exec_ts = datetime(2026, 10, 25, 10, 0, tzinfo=UTC)
    result = rewrite_sql(
        "SELECT CURRENT_TIMESTAMP - interval '1 day'",
        dialect="postgres",
        exec_ts=exec_ts,
        session_time_zone=JERUSALEM,
    )
    assert result.sql == "SELECT CAST('2026-10-24 13:00:00' AS TIMESTAMP)"


def test_calendar_months_are_added_in_the_session_zone() -> None:
    exec_ts = datetime(2026, 3, 31, 12, 0, tzinfo=UTC)
    result = rewrite_sql(
        "SELECT CURRENT_TIMESTAMP + interval '1 month'",
        dialect="postgres",
        exec_ts=exec_ts,
        session_time_zone=JERUSALEM,
    )
    # 31 March + 1 month clamps to 30 April, and Israel is on UTC+3 in April (12:00Z = 15:00).
    assert result.sql == "SELECT CAST('2026-04-30 15:00:00' AS TIMESTAMP)"


def test_utc_now_functions_ignore_the_session_time_zone() -> None:
    exec_ts = datetime(2026, 11, 25, 10, 0, tzinfo=UTC)
    tsql = rewrite_sql(
        "SELECT GETUTCDATE()", dialect="tsql", exec_ts=exec_ts, session_time_zone=JERUSALEM
    )
    mysql = rewrite_sql(
        "SELECT UTC_TIMESTAMP()", dialect="mysql", exec_ts=exec_ts, session_time_zone=JERUSALEM
    )
    assert tsql.sql == f"SELECT CAST('{TS}' AS DATETIME2)"
    assert mysql.sql == f"SELECT CAST('{TS}' AS DATETIME)"


# ---------------------------------------------------------------------------------------------
# Nothing else may change
# ---------------------------------------------------------------------------------------------

UNTOUCHED: tuple[tuple[str, str], ...] = (
    ("postgres", "SELECT COUNT(*) FROM usage_events WHERE start_time >= DATE '2026-11-01'"),
    (
        "postgres",
        "SELECT cust_id, ROUND(credit_limit / 5000) * 5000 AS bucket FROM crm.customers",
    ),
    ("postgres", "SELECT updated_at, deleted_at FROM crm.customers"),
    ("oracle", "SELECT cust_id FROM crm.customers WHERE status = 'פעיל'"),
    ("tsql", "SELECT TOP 10 [cust_id] FROM [dbo].[customers] WITH (NOLOCK)"),
    # A timestamp literal is already absolute.
    ("postgres", "SELECT 1 FROM t WHERE ts >= TIMESTAMP '2026-11-25 14:00:00'"),
)


@pytest.mark.parametrize(("dialect", "sql"), UNTOUCHED, ids=[sql[:40] for _, sql in UNTOUCHED])
def test_sql_without_a_now_function_is_only_normalised(dialect: str, sql: str) -> None:
    result = rewrite_sql(sql, dialect=dialect, exec_ts=EXEC_TS)
    assert result.replacements == ()
    assert now_names(result.sql) == set()
    # The statement is re-rendered by sqlglot, so compare the parse, not the spelling.
    assert rewrite_sql(result.sql, dialect=dialect, exec_ts=EXEC_TS).sql == result.sql


def test_a_user_function_with_a_now_name_in_another_dialect_survives() -> None:
    """`getdate()` is only a clock in T-SQL; in PostgreSQL it is a user function."""
    sql = "SELECT getdate(updated_at) FROM crm.customers"
    result = rewrite_sql(sql, dialect="postgres", exec_ts=EXEC_TS)
    assert result.replacements == ()
    assert "GETDATE" in result.sql.upper()


# ---------------------------------------------------------------------------------------------
# The golden corpus: every statement rewritten, and the `expect` blocks honoured
# ---------------------------------------------------------------------------------------------


def test_corpus_is_not_empty() -> None:
    assert len(CORPUS) >= 100


@pytest.mark.parametrize("stmt", CORPUS, ids=[stmt.id for stmt in CORPUS])
def test_corpus_now_functions_are_all_replaced(stmt: CorpusStatement) -> None:
    assert stmt.expect is not None
    result = rewrite_sql(stmt.sql, dialect=stmt.dialect, exec_ts=EXEC_TS)
    assert now_names(result.sql) == set(), f"{stmt.id} kept a now-like function"
    # The audit names the expression as sqlglot renders it (`now()` becomes CURRENT_TIMESTAMP),
    # so compare against the same normalised spelling of the source statement.
    normalised = sqlglot.parse_one(
        stmt.sql, read=stmt.dialect, error_level=sqlglot.ErrorLevel.RAISE
    ).sql(dialect=stmt.dialect)
    audited = now_names("\n".join(item.original for item in result.replacements))
    assert now_names(normalised) <= audited, (
        f"{stmt.id}: a now-like function was replaced without an audit entry"
    )
    # The acceptance criterion: every function the expect block lists appears in the statement
    # as written, and everything the statement contains is reported by the audit trail.
    assert set(stmt.expect.now_functions) <= now_names(stmt.sql)
    assert now_names(normalised) <= audited


def test_corpus_rewrite_golden(golden) -> None:  # type: ignore[no-untyped-def]
    """Every corpus statement, rewritten with one fixed exec_ts (review the diff)."""
    for dialect in ("postgres", "oracle", "tsql"):
        lines = [
            f"-- exec_ts: {EXEC_TS.isoformat()}   session time zone: UTC   dialect: {dialect}",
            "-- Generated by tests/unit/test_exec_ts.py; rewrite with --update-golden.",
        ]
        for stmt in CORPUS:
            if stmt.dialect != dialect:
                continue
            result = rewrite_sql(stmt.sql, dialect=dialect, exec_ts=EXEC_TS)
            lines.append("")
            lines.append(f"-- {stmt.id}")
            lines.append(result.sql)
        golden(f"exec_ts_{dialect}.sql", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------------------
# Idempotence and determinism (property test)
# ---------------------------------------------------------------------------------------------

SPELLINGS: dict[str, tuple[str, ...]] = {
    "postgres": ("now()", "CURRENT_TIMESTAMP", "CURRENT_DATE", "LOCALTIMESTAMP"),
    "duckdb": ("now()", "CURRENT_TIMESTAMP", "CURRENT_DATE"),
    "oracle": (
        "SYSDATE",
        "SYSTIMESTAMP",
        "CURRENT_DATE",
        "CURRENT_TIMESTAMP",
        "LOCALTIMESTAMP",
    ),
    "tsql": (
        "GETDATE()",
        "GETUTCDATE()",
        "SYSDATETIME()",
        "SYSDATETIMEOFFSET()",
        "CURRENT_TIMESTAMP",
    ),
    "mysql": ("NOW()", "CURDATE()", "UTC_TIMESTAMP()", "CURRENT_TIMESTAMP"),
}

WRAPPERS: dict[str, tuple[str, ...]] = {
    "postgres": (
        "{e}",
        "{e} - interval '7 days'",
        "{e} + interval '3 hours'",
        "DATE_TRUNC('month', {e})",
    ),
    "duckdb": ("{e}", "{e} - interval '1 day'", "DATE_TRUNC('month', {e})"),
    "oracle": ("{e}", "{e} - 7", "{e} - 1 / 24", "TRUNC({e}, 'MM')", "ADD_MONTHS({e}, -3)"),
    "tsql": ("{e}", "DATEADD(day, -7, {e})", "DATEADD(minute, -90, {e})", "EOMONTH({e}, -3)"),
    "mysql": ("{e}", "DATE_SUB({e}, INTERVAL 7 DAY)"),
}

ZONES = ("UTC", JERUSALEM, "America/New_York", "Australia/Sydney")


def _sql_for(dialect: str, spelling: str, wrapper: str, column: str) -> str:
    expression = wrapper.format(e=spelling)
    return f"SELECT COUNT(*) AS n FROM t WHERE {column} >= {expression}"


@pytest.mark.parametrize("dialect", sorted(SUPPORTED_DIALECTS))
def test_rewriting_twice_is_a_no_op_property(dialect: str) -> None:
    """A second rewrite must find nothing to do and must not change one character."""

    examples = st.sampled_from(
        [(spelling, wrapper) for spelling in SPELLINGS[dialect] for wrapper in WRAPPERS[dialect]]
    )

    @settings(deadline=None, max_examples=50)
    @given(
        choice=examples,
        exec_ts=st.datetimes(
            min_value=datetime(2000, 1, 1),
            max_value=datetime(2100, 1, 1),
            timezones=st.just(UTC),
        ),
        zone=st.sampled_from(ZONES),
        column=st.sampled_from(("start_time", "insertion_time", "updated_at")),
    )
    def property_(choice: tuple[str, str], exec_ts: datetime, zone: str, column: str) -> None:
        spelling, wrapper = choice
        sql = _sql_for(dialect, spelling, wrapper, column)
        once = rewrite_sql(sql, dialect=dialect, exec_ts=exec_ts, session_time_zone=zone)
        again = rewrite_sql(once.sql, dialect=dialect, exec_ts=exec_ts, session_time_zone=zone)
        assert again.sql == once.sql
        assert again.replacements == ()
        assert now_names(once.sql) == set()
        assert once.replacements, "the generated statement always had a now-like function"

    property_()


@pytest.mark.parametrize("dialect", sorted(SUPPORTED_DIALECTS))
def test_rewriting_is_deterministic(dialect: str) -> None:
    sql = _sql_for(dialect, SPELLINGS[dialect][0], WRAPPERS[dialect][0], "start_time")
    first = rewrite_sql(sql, dialect=dialect, exec_ts=EXEC_TS, session_time_zone=JERUSALEM)
    second = rewrite_sql(sql, dialect=dialect, exec_ts=EXEC_TS, session_time_zone=JERUSALEM)
    assert first == second


# ---------------------------------------------------------------------------------------------
# Errors the caller can act on
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dialect": "spark", "exec_ts": EXEC_TS}, "unsupported dialect"),
        ({"dialect": "postgres", "exec_ts": datetime(2026, 11, 25, 10, 0)}, "timezone-aware"),
        (
            {"dialect": "postgres", "exec_ts": EXEC_TS, "session_time_zone": "Mars/Olympus"},
            "unknown time zone",
        ),
        ({"dialect": "postgres", "exec_ts": EXEC_TS}, "exactly one statement"),
    ],
    ids=["dialect", "naive-exec-ts", "time-zone", "two-statements"],
)
def test_rewrite_sql_rejects_input_it_cannot_handle(
    kwargs: dict[str, object], message: str
) -> None:
    sql = "SELECT 1; SELECT 2" if message == "exactly one statement" else "SELECT 1"
    with pytest.raises(ValueError, match=message):
        rewrite_sql(sql, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# Elasticsearch date math
# ---------------------------------------------------------------------------------------------

ES_CASES: tuple[tuple[str, str], ...] = (
    ("now", "2026-11-25T10:00:00.000Z"),
    ("now-7d", "2026-11-18T10:00:00.000Z"),
    ("now-7d/d", "2026-11-18T00:00:00.000Z"),
    ("now+1h", "2026-11-25T11:00:00.000Z"),
    ("now-1d+2h", "2026-11-24T12:00:00.000Z"),
    ("now/d", "2026-11-25T00:00:00.000Z"),
    ("now-1M/M", "2026-10-01T00:00:00.000Z"),
    ("now-1y/y", "2025-01-01T00:00:00.000Z"),
    ("now/w", "2026-11-23T00:00:00.000Z"),
    ("now-7d/h", "2026-11-18T10:00:00.000Z"),
    ("now-7d/m", "2026-11-18T10:00:00.000Z"),
    ("now-90m/m", "2026-11-25T08:30:00.000Z"),
)


@pytest.mark.parametrize(("expression", "expected"), ES_CASES, ids=[e for e, _ in ES_CASES])
def test_es_date_math_becomes_an_absolute_instant(expression: str, expected: str) -> None:
    assert rewrite_es_date_math(expression, exec_ts=EXEC_TS) == expected


def test_es_rounding_happens_after_the_arithmetic() -> None:
    """``now-7d/d`` is 18 November 00:00, not 18 November 10:00 minus a day boundary."""
    assert rewrite_es_date_math("now-7d/d", exec_ts=EXEC_TS) == "2026-11-18T00:00:00.000Z"
    assert rewrite_es_date_math("now-168h/d", exec_ts=EXEC_TS) == "2026-11-18T00:00:00.000Z"


def test_es_date_math_rounds_in_the_query_time_zone() -> None:
    # 10:00Z is 12:00 in Jerusalem (UTC+2 in November): minus 7 days, floored to that local day.
    assert (
        rewrite_es_date_math("now-7d/d", exec_ts=EXEC_TS, time_zone=JERUSALEM)
        == "2026-11-17T22:00:00.000Z"
    )
    assert (
        rewrite_es_date_math("now/d", exec_ts=EXEC_TS, time_zone=JERUSALEM)
        == "2026-11-24T22:00:00.000Z"
    )


def test_es_date_math_rounds_correctly_across_the_dst_change() -> None:
    exec_ts = datetime(2026, 10, 25, 10, 0, tzinfo=UTC)  # 12:00 local, UTC+2
    # One day earlier is 13:00 local (UTC+3, before the change); floored to that local day.
    assert (
        rewrite_es_date_math("now-1d/d", exec_ts=exec_ts, time_zone=JERUSALEM)
        == "2026-10-23T21:00:00.000Z"
    )


@pytest.mark.parametrize("expression", ["7d", "now-7", "now-7q", "2026-11-25||-7d/d", ""])
def test_es_date_math_rejects_what_is_not_date_math(expression: str) -> None:
    with pytest.raises(ValueError, match="not Elasticsearch date math"):
        rewrite_es_date_math(expression, exec_ts=EXEC_TS)


ES_QUERY = {
    "query": {
        "bool": {
            "filter": [
                {
                    "range": {
                        "start_time": {
                            "gte": "now-7d/d",
                            "lt": "now",
                            "time_zone": JERUSALEM,
                        }
                    }
                },
                {"range": {"updated_at": {"gte": "2026-01-01"}}},
                {"term": {"status": "now-7d/d"}},
            ]
        }
    },
    "_source": ["start_time", "now"],
}


def test_es_query_rewrites_only_date_math_range_bounds() -> None:
    original = copy.deepcopy(ES_QUERY)
    rewritten, replacements = rewrite_es_query(ES_QUERY, exec_ts=EXEC_TS)
    assert original == ES_QUERY, "the input document must not be mutated"
    ranges = rewritten["query"]["bool"]["filter"]
    assert ranges[0]["range"]["start_time"] == {
        "gte": "2026-11-17T22:00:00.000Z",
        "lt": "2026-11-25T10:00:00.000Z",
        "time_zone": JERUSALEM,
    }
    assert ranges[1]["range"]["updated_at"] == {"gte": "2026-01-01"}
    assert ranges[2]["term"] == {"status": "now-7d/d"}
    assert rewritten["_source"] == ["start_time", "now"]
    assert [(item.original, item.replaced_with) for item in replacements] == [
        ("now-7d/d", "2026-11-17T22:00:00.000Z"),
        ("now", "2026-11-25T10:00:00.000Z"),
    ]
    assert all(item.position is None for item in replacements)


def test_es_query_uses_the_default_time_zone_without_one_in_the_range() -> None:
    query = {"range": {"start_time": {"gte": "now/d"}}}
    rewritten, _ = rewrite_es_query(query, exec_ts=EXEC_TS, time_zone=JERUSALEM)
    assert rewritten == {"range": {"start_time": {"gte": "2026-11-24T22:00:00.000Z"}}}
