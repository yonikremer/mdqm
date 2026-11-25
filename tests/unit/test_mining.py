"""Tests for stage D9 mining (S08) and the deterministic extractor behind it.

Three kinds of test, because mining has three promises:

* **The corpus is the oracle.** ``tests/data/queries/*.sql`` carries an ``expect`` block per
  statement and is validated by ``test_query_corpus.py``; here every statement must be extracted
  exactly as its block says. Failures are printed per statement, so a mismatch names itself. The
  comparison parses both sides in the statement's dialect first: whether a conversion is written
  ``::varchar`` or ``CAST(... AS VARCHAR)``, or spaced two ways, is not a difference in meaning --
  and a group-by term is compared on its column and expression, not on the alias it was written
  through (the corpus itself writes ``c.status`` in one block and plain ``status`` in another).
* **Mining is deterministic and idempotent.** The same batch twice gives byte-identical output, and
  re-delivering a batch already mined writes nothing and moves no watermark.
* **A hint is a hypothesis with evidence.** Every conversion mined out of history names its source,
  its fingerprint and how often the statement ran, and no formula appears that the statement did not
  contain: mining raises a prior, it never invents a conversion.

No clock is read: ``exec_ts`` is injected everywhere, and every store test uses ``tmp_path``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlglot
from sqlglot import exp

from mdqm.models.query import RawQuery
from mdqm.pipeline.stages.mining import (
    DEFAULT_WEIGHTS,
    MINED_QUERY_TABLE,
    WATERMARK_TABLE,
    IngestReport,
    MinedQuery,
    MiningWeights,
    ParseFailure,
    age_days,
    batch_digest,
    ingest,
    mine,
    rank,
    read,
    score,
    text_hash,
    watermark,
)
from mdqm.sql.fingerprint import (
    AGGREGATE_FUNCTIONS,
    LITERAL_MARKER,
    NOW_MARKER,
    ConversionHint,
    HintKind,
    QueryFacts,
    analyze,
    fingerprint,
    normalize,
    now_functions,
)
from mdqm.store.db import Store

# The corpus loader lives with the corpus validator, so both read the file format the same way.
from .test_query_corpus import PARSED, CorpusStatement, ExpectedExtraction

EXEC_TS = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)
"""One execution time for every case: mining must not read the clock."""

HALF_DAY = timedelta(hours=12)


# --------------------------------------------------------------------------------------------------
# The corpus: extraction equals the expect blocks
# --------------------------------------------------------------------------------------------------


def _canonical(text: str, dialect: str, *, drop_qualifiers: bool = False) -> str:
    """Compare two spellings of the same expression: parse both, render one way, lowercase."""
    tree = sqlglot.parse_one(f"SELECT {text}", read=dialect)
    if drop_qualifiers:
        tree = tree.transform(
            lambda node: exp.column(node.name.lower()) if isinstance(node, exp.Column) else node
        )
    rendered = tree.expressions[0].sql(dialect=dialect)
    return " ".join(rendered.split()).lower()


def _mismatches(stmt: CorpusStatement, expect: ExpectedExtraction, facts: QueryFacts) -> list[str]:
    """Every way ``facts`` disagrees with one expect block, as printable lines."""
    problems: list[str] = []
    dialect = stmt.dialect

    def compare(name: str, got: Any, want: Any) -> None:
        if got != want:
            problems.append(f"{expect.id}: {name} {got!r} != expected {want!r}")

    compare("tables", sorted(facts.tables), sorted(expect.tables))
    compare("filters", sorted(facts.filters), sorted(expect.filters))
    compare("time_columns", sorted(facts.time_columns), sorted(expect.time_columns))
    compare("aggregations", sorted(facts.aggregations), sorted(expect.aggregations))
    compare("now_functions", sorted(facts.now_functions), sorted(expect.now_functions))
    compare(
        "group_by",
        sorted(_canonical(term, dialect, drop_qualifiers=True) for term in facts.group_by),
        sorted(_canonical(term, dialect, drop_qualifiers=True) for term in expect.group_by),
    )
    compare(
        "rounding",
        sorted(_canonical(term, dialect) for term in facts.rounding),
        sorted(_canonical(term, dialect) for term in expect.rounding),
    )
    got_joins = [
        (
            edge.tables,
            edge.columns,
            edge.conversion,
            None if edge.conversion_expr is None else _canonical(edge.conversion_expr, dialect),
        )
        for edge in facts.joins
    ]
    want_joins = [
        (
            edge.tables,
            edge.columns,
            edge.conversion,
            None if edge.conversion_expr is None else _canonical(edge.conversion_expr, dialect),
        )
        for edge in expect.joins
    ]
    compare("joins", got_joins, want_joins)
    return problems


@pytest.mark.parametrize("stmt", PARSED, ids=[stmt.id for stmt in PARSED])
def test_extraction_matches_the_corpus_expect_block(stmt: CorpusStatement) -> None:
    """One statement, one expect block: extraction equals it, or the message says how not."""
    assert stmt.expect is not None
    facts = analyze(stmt.sql, stmt.dialect)
    assert facts.ok, f"{stmt.id}: mining could not parse it: {facts.error}"
    problems = _mismatches(stmt, stmt.expect, facts)
    assert not problems, "\n".join(problems)


def test_corpus_extraction_rate() -> None:
    """The acceptance budget: at least 95% exactly, with every failure named in the message."""
    matched = 0
    failures: list[str] = []
    for stmt in PARSED:
        assert stmt.expect is not None
        facts = analyze(stmt.sql, stmt.dialect)
        problems = _mismatches(stmt, stmt.expect, facts)
        if problems:
            failures.extend(problems)
        else:
            matched += 1
    rate = matched / len(PARSED)
    assert rate >= 0.95, f"{matched}/{len(PARSED)} = {rate:.1%}\n" + "\n".join(failures)


# --------------------------------------------------------------------------------------------------
# Fingerprints: same shape => same fingerprint, different shape => different
# --------------------------------------------------------------------------------------------------

SAME_FINGERPRINT_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "postgres",
        "SELECT cust_id FROM crm.customers WHERE status = 'A'",
        "select cust_id\nfrom crm.customers\n  where status = 'S'",
    ),
    (
        "postgres",
        "SELECT c.cust_id FROM crm.customers AS c WHERE c.status = 'A'",
        "SELECT k.cust_id FROM crm.customers k WHERE k.status = 'B'",
    ),
    (
        "postgres",
        "SELECT CAST(credit_limit AS numeric) FROM crm.customers",
        "SELECT credit_limit::numeric FROM crm.customers",
    ),
    (
        "postgres",
        "SELECT COUNT(*) FROM billing.usage_events WHERE start_time >= now() - interval '7 days'",
        "SELECT COUNT(*) FROM billing.usage_events WHERE start_time >= now() - interval '90 days'",
    ),
    (
        "oracle",
        "SELECT cust_id FROM crm.customers WHERE name = 'O''Brien'",
        "SELECT cust_id FROM crm.customers WHERE name = 'בנק לאומי'",
    ),
    (
        "tsql",
        "SELECT TOP 10 cust_id FROM dbo.customers WHERE status = 'A'",
        "select top 10 cust_id from dbo.customers where status = N'C'",
    ),
)


@pytest.mark.parametrize(("dialect", "left", "right"), SAME_FINGERPRINT_CASES)
def test_literals_whitespace_and_aliases_do_not_change_the_fingerprint(
    dialect: str, left: str, right: str
) -> None:
    assert fingerprint(left, dialect) == fingerprint(right, dialect)


DIFFERENT_FINGERPRINT_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "postgres",
        "SELECT c.cust_id FROM crm.customers c JOIN billing.accounts e ON c.cust_id = e.client_no",
        "SELECT c.cust_id FROM crm.customers c JOIN billing.accounts e "
        "ON LPAD(c.cust_id::varchar, 10, '0') = e.client_no",
    ),
    (
        "postgres",
        "SELECT COUNT(*) FROM billing.usage_events WHERE start_time >= now()",
        "SELECT COUNT(*) FROM billing.usage_events WHERE insertion_time >= now()",
    ),
    (
        "postgres",
        "SELECT cust_id FROM crm.customers WHERE start_time >= now()",
        "SELECT cust_id FROM crm.customers WHERE start_time >= DATE '2026-01-01'",
    ),
    (
        "tsql",
        "SELECT client_no FROM dbo.accounts WHERE status = 'Active'",
        "SELECT client_no FROM dbo.usage_events WHERE status = 'Active'",
    ),
)


@pytest.mark.parametrize(("dialect", "left", "right"), DIFFERENT_FINGERPRINT_CASES)
def test_a_different_join_or_column_changes_the_fingerprint(
    dialect: str, left: str, right: str
) -> None:
    assert fingerprint(left, dialect) != fingerprint(right, dialect)


def test_fingerprint_is_sha256_of_the_normalised_statement() -> None:
    sql = "SELECT cust_id FROM crm.customers WHERE status = 'A'"
    digest = fingerprint(sql, "postgres")
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)
    assert digest == fingerprint(sql, "postgres")
    assert digest != fingerprint(sql, "oracle")  # the dialect is part of identity


def test_normalised_sql_keeps_no_literal_and_no_clock() -> None:
    sql = (
        "SELECT cust_id, name FROM crm.customers "
        "WHERE name = 'O''Brien' AND status = 'A' AND credit_limit > 42 "
        "AND start_time >= now() - interval '7 days'"
    )
    normalized = normalize(sql, "postgres")
    for literal in ("O'Brien", "'A'", "42", "7 days"):
        assert literal not in normalized
    assert LITERAL_MARKER in normalized
    assert NOW_MARKER in normalized
    for alias in (" c1", "c1."):
        assert alias not in normalized


def test_normalisation_is_idempotent_for_a_normalised_statement() -> None:
    sql = "SELECT COUNT(*) FROM billing.usage_events WHERE start_time >= now() - interval '7 days'"
    once = normalize(sql, "postgres")
    assert normalize(sql, "postgres") == once


def test_now_functions_are_read_from_the_statement_text() -> None:
    assert now_functions("SELECT COUNT(*) FROM t WHERE a >= now()", "postgres") == ("now",)
    assert now_functions("SELECT SYSDATE FROM dual", "oracle") == ("sysdate",)
    assert now_functions("SELECT 1 FROM t WHERE a >= DATEADD(day,-1,GETDATE())", "tsql") == (
        "getdate",
    )
    assert now_functions("SELECT 'now' AS label FROM t", "postgres") == ()


# --------------------------------------------------------------------------------------------------
# Facts: joins, filters, aggregations, time columns, conversion hints
# --------------------------------------------------------------------------------------------------

LPAD_JOIN = (
    "SELECT c.cust_id, e.client_no FROM crm.customers c "
    "JOIN billing.accounts e ON LPAD(c.cust_id::varchar, 10, '0') = e.client_no "
    "WHERE c.status = 'A'"
)

FACT_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "postgres",
        LPAD_JOIN,
        "join:customers>accounts:cust_id>client_no:lpad",
    ),
    (
        "postgres",
        "SELECT c.cust_id FROM crm.customers c JOIN billing.accounts e "
        "ON CAST(c.cust_id AS VARCHAR(10)) = e.client_no",
        "join:customers>accounts:cust_id>client_no:cast",
    ),
    (
        "oracle",
        "SELECT c.cust_id FROM crm.customers c JOIN billing.accounts e "
        "ON c.cust_id = TO_NUMBER(e.client_no)",
        "join:customers>accounts:cust_id>client_no:to_number",
    ),
    (
        "tsql",
        "SELECT c.cust_id FROM dbo.customers c JOIN dbo.accounts e "
        "ON TRY_CAST(e.client_no AS int) = c.cust_id",
        "join:customers>accounts:cust_id>client_no:try_cast",
    ),
    (
        "postgres",
        "SELECT cust_id FROM crm.customers WHERE status = 'A' ORDER BY cust_id",
        "no-join",
    ),
    (
        "postgres",
        # a comma join in Oracle is still a join; compared columns are not filters
        "SELECT c.cust_id, e.client_no FROM crm.customers c, billing.accounts e "
        "WHERE LPAD(CAST(c.cust_id AS VARCHAR(10)), 10, '0') = e.client_no",
        "join:customers>accounts:cust_id>client_no:lpad",
    ),
)


@pytest.mark.parametrize(("dialect", "sql", "expected"), FACT_CASES)
def test_join_edges(dialect: str, sql: str, expected: str) -> None:
    facts = analyze(sql, dialect)
    assert facts.ok, facts.error
    if expected == "no-join":
        assert facts.joins == ()
        return
    kind, tables, columns, conversion = expected.split(":")
    left, right = tables.split(">")
    left_column, right_column = columns.split(">")
    assert [(edge.tables, edge.columns, edge.conversion) for edge in facts.joins] == [
        ((left, right), (left_column, right_column), conversion)
    ]
    assert facts.joins[0].conversion_expr is not None


def test_filters_exclude_join_columns_and_aggregated_columns() -> None:
    """A join key is not a filter, and ``HAVING SUM(x) > n`` filters the sum, not the column."""
    facts = analyze(
        "SELECT status, SUM(credit_limit) AS total FROM crm.customers "
        "WHERE status <> 'C' GROUP BY status HAVING SUM(credit_limit) > 100000",
        "postgres",
    )
    assert facts.filters == ("status",)
    assert facts.group_by == ("status",)
    assert facts.aggregations == ("sum",)

    joined = analyze(LPAD_JOIN, "postgres")
    assert joined.filters == ("status",)


def test_time_columns_need_a_range_predicate_and_a_time_name() -> None:
    ranged = analyze(
        "SELECT cust_id FROM billing.usage_events WHERE start_time >= now() - interval '1 day'",
        "postgres",
    )
    assert ranged.time_columns == ("start_time",)
    flag = analyze("SELECT cust_id FROM crm.customers WHERE deleted_at IS NULL", "postgres")
    assert flag.filters == ("deleted_at",)
    assert flag.time_columns == ()


def test_aggregations_are_a_closed_set() -> None:
    facts = analyze(
        "SELECT status, COUNT(*), RATIO_TO_REPORT(COUNT(*)) OVER () FROM crm.customers "
        "GROUP BY status",
        "oracle",
    )
    assert facts.aggregations == ("count",)
    assert set(facts.aggregations) <= AGGREGATE_FUNCTIONS


def test_aggregate_in_a_cte_and_a_derived_table_is_reported() -> None:
    facts = analyze(
        "WITH recent AS (SELECT cust_id, SUM(bytes) AS b FROM billing.usage_events "
        "GROUP BY cust_id) SELECT COUNT(*) FROM recent r JOIN crm.customers c "
        "ON c.cust_id = r.cust_id",
        "postgres",
    )
    assert set(facts.tables) == {"usage_events", "customers"}  # the CTE is not a base table
    assert facts.aggregations == ("count", "sum")
    assert facts.group_by == ("cust_id",)
    assert [edge.tables for edge in facts.joins] == [("recent", "customers")]


def test_unparseable_statement_is_reported_not_raised() -> None:
    facts = analyze("SELECT FROM WHERE ((", "postgres")
    assert not facts.ok
    assert facts.error
    assert facts.fingerprint is None
    assert facts.tables == ()
    assert "Traceback" not in (facts.error or "")


def test_unknown_dialect_is_reported_not_raised() -> None:
    facts = analyze("SELECT 1", "not-a-dialect")
    assert not facts.ok
    assert facts.error


# --------------------------------------------------------------------------------------------------
# Conversion hints: typed, with the parameters a validator needs
# --------------------------------------------------------------------------------------------------


def _hint(sql: str, dialect: str = "postgres", index: int = 0) -> ConversionHint:
    facts = analyze(sql, dialect)
    assert facts.ok, facts.error
    assert facts.hints, f"no hint mined from {sql}"
    return facts.hints[index]


def test_lpad_join_hint_carries_width_and_fill() -> None:
    hint = _hint(LPAD_JOIN)
    assert hint.kind is HintKind.JOIN_KEY
    assert hint.function == "lpad"
    assert (hint.tables, hint.columns) == (("customers", "accounts"), ("cust_id", "client_no"))
    assert hint.width == 10
    assert hint.fill == "0"
    assert "LPAD" in hint.expression.upper()


def test_to_char_join_hint_carries_the_format() -> None:
    hint = _hint(
        "SELECT c.cust_id FROM crm.customers c JOIN billing.accounts e "
        "ON TO_CHAR(c.cust_id, 'FM0000000000') = e.client_no",
        "oracle",
    )
    assert hint.function == "to_char"
    assert hint.pattern == "FM0000000000"


def test_cast_join_hint_carries_the_target_type() -> None:
    hint = _hint(
        "SELECT c.cust_id FROM dbo.customers c JOIN dbo.accounts e "
        "ON CAST(c.cust_id AS varchar(10)) = e.client_no",
        "tsql",
    )
    assert hint.function == "cast"
    assert hint.target_type == "VARCHAR(10)"


ROUNDING_CASES: tuple[tuple[str, str, HintKind, float | None, float | None], ...] = (
    (
        "postgres",
        "SELECT ROUND(credit_limit / 5000) * 5000 AS c FROM crm.customers",
        HintKind.ROUNDING,
        5000.0,
        None,
    ),
    (
        "oracle",
        "SELECT ROUND(c.credit_limit / 5000) * 5000 AS c FROM crm.customers",
        HintKind.ROUNDING,
        5000.0,
        None,
    ),
    (
        "tsql",
        "SELECT ROUND(credit_limit, -3) AS c FROM dbo.customers",
        HintKind.ROUNDING,
        1000.0,
        None,
    ),
    (
        "postgres",
        "SELECT FLOOR(credit_limit / 1000) AS c FROM crm.customers",
        HintKind.ROUNDING,
        1000.0,
        None,
    ),
    (
        "postgres",
        "SELECT cust_no, limit_k * 10 AS credit_limit FROM erp.customers",
        HintKind.SCALE,
        None,
        10.0,
    ),
    (
        "postgres",
        "SELECT cust_id, credit_limit::numeric / 1000 AS k FROM crm.customers",
        HintKind.SCALE,
        None,
        0.001,
    ),
)


@pytest.mark.parametrize(("dialect", "sql", "kind", "granularity", "scale"), ROUNDING_CASES)
def test_rounding_and_scale_hints(
    dialect: str, sql: str, kind: HintKind, granularity: float | None, scale: float | None
) -> None:
    hint = _hint(sql, dialect)
    assert hint.kind is kind
    assert hint.granularity == granularity
    assert hint.scale == scale


def test_arithmetic_that_is_not_a_conversion_is_not_a_hint() -> None:
    facts = analyze("SELECT SUM(credit_limit) / COUNT(*) AS average FROM crm.customers", "postgres")
    assert facts.rounding == ()
    assert facts.hints == ()


def test_hint_expression_came_from_the_statement() -> None:
    """No invented formula: a hint's expression is the statement's own expression, respelled."""
    for statement in PARSED:
        assert statement.expect is not None
        facts = analyze(statement.sql, statement.dialect)
        assert facts.ok, facts.error
        whole = " ".join(
            sqlglot.parse_one(statement.sql, read=statement.dialect)
            .sql(dialect=statement.dialect)
            .split()
        ).lower()
        for hint in facts.hints:
            expression = _canonical(hint.expression, statement.dialect)
            assert expression in whole, f"{statement.id}: {expression!r} is not in the statement"


# --------------------------------------------------------------------------------------------------
# Mining: merge, rank, determinism
# --------------------------------------------------------------------------------------------------

BATCH: tuple[RawQuery, ...] = (
    RawQuery(
        text="SELECT COUNT(*) FROM billing.usage_events "
        "WHERE start_time >= now() - interval '7 days'",
        dialect="postgres",
        calls=40,
        user="dana",
        last_seen=EXEC_TS - timedelta(days=1),
    ),
    RawQuery(
        text="SELECT COUNT(*) FROM billing.usage_events "
        "WHERE start_time >= now() - interval '30 days'",
        dialect="postgres",
        calls=10,
        user="yossi",
        last_seen=EXEC_TS - HALF_DAY,
    ),
    RawQuery(
        text="SELECT c.cust_id FROM crm.customers c JOIN billing.accounts e "
        "ON LPAD(c.cust_id::varchar, 10, '0') = e.client_no",
        dialect="postgres",
        calls=5,
        user="dana",
        last_seen=EXEC_TS - timedelta(days=10),
    ),
    RawQuery(text="SELECT FROM WHERE ((", dialect="postgres", calls=1, last_seen=EXEC_TS),
)


def test_mine_merges_statements_that_share_a_fingerprint() -> None:
    report = mine(BATCH, source="crm", exec_ts=EXEC_TS)
    assert report.statements == 4
    assert report.parsed == 3
    assert report.failed == 1
    assert len(report.queries) == 2  # the two Windows differ only in the literal

    usage = next(
        query
        for query in report.queries
        if "usage_events" in (query.facts.normalized or "")
    )
    assert usage.calls == 50
    assert usage.statements == 2
    assert usage.users == ("dana", "yossi")
    assert usage.first_seen == EXEC_TS - timedelta(days=1)
    assert usage.last_seen == EXEC_TS - HALF_DAY
    assert usage.source == "crm"


def test_mine_reports_the_failure_without_storing_the_text() -> None:
    report = mine(BATCH, source="crm", exec_ts=EXEC_TS)
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert isinstance(failure, ParseFailure)
    assert failure.text_hash == text_hash("SELECT FROM WHERE ((")
    assert "SELECT FROM WHERE" not in failure.error
    assert failure.count == 1


def test_mine_is_deterministic_byte_for_byte() -> None:
    first = mine(BATCH, source="crm", exec_ts=EXEC_TS).model_dump_json()
    second = mine(BATCH, source="crm", exec_ts=EXEC_TS).model_dump_json()
    assert first == second


def test_mine_is_independent_of_the_input_order() -> None:
    forward = mine(BATCH, source="crm", exec_ts=EXEC_TS).model_dump_json()
    backward = mine(tuple(reversed(BATCH)), source="crm", exec_ts=EXEC_TS).model_dump_json()
    assert forward == backward


def test_mine_attaches_evidence_to_every_hint() -> None:
    report = mine(BATCH, source="crm", exec_ts=EXEC_TS)
    assert report.hints
    for hint in report.hints:
        assert hint.evidence is not None
        assert hint.evidence.source == "crm"
        assert len(hint.evidence.fingerprint) == 64
        assert hint.evidence.calls == 5


def test_mine_of_an_empty_batch_writes_nothing() -> None:
    report = mine((), source="crm", exec_ts=EXEC_TS)
    assert report.statements == 0
    assert report.queries == ()
    assert report.failures == ()
    assert report.batch_hash == batch_digest(())


SCORE_CASES: tuple[tuple[str, dict[str, Any], float], ...] = (
    ("volume", {"calls": 100}, 1.0 * 4.605170185988092),
    ("breadth", {"distinct_users": 3}, 3.0),
    ("recency_at_zero", {"age": 0.0}, 1.0),
    ("recency_at_half_life", {"age": 30.0}, 0.5),
    ("unknown_age", {"age": None}, 0.0),
    ("all_terms", {"calls": 100, "distinct_users": 2, "age": 30.0}, 4.605170 + 2.0 + 0.5),
)


@pytest.mark.parametrize(("label", "kwargs", "expected"), SCORE_CASES)
def test_score_formula(label: str, kwargs: dict[str, Any], expected: float) -> None:
    assert score(**kwargs) == pytest.approx(expected, abs=1e-5), label


def test_score_falls_back_to_statements_when_calls_are_unknown() -> None:
    assert score(calls=0, statements=10) == pytest.approx(1.0 * 2.302585, abs=1e-5)


def test_score_weights_are_configurable() -> None:
    weights = MiningWeights(calls=0.0, recency=0.0, users=5.0)
    assert score(distinct_users=2, weights=weights) == pytest.approx(10.0)
    assert DEFAULT_WEIGHTS.calls != weights.calls


def test_rank_breaks_ties_by_fingerprint() -> None:
    facts = analyze("SELECT cust_id FROM crm.customers", "postgres")
    left = MinedQuery(source="a", fingerprint="a" * 64, dialect="postgres", facts=facts)
    right = MinedQuery(source="a", fingerprint="b" * 64, dialect="postgres", facts=facts)
    ranked = rank((right, left), exec_ts=EXEC_TS)
    assert [query.fingerprint for query in ranked] == ["a" * 64, "b" * 64]


def test_rank_prefers_a_fresh_statement_over_a_stale_one() -> None:
    facts = analyze("SELECT cust_id FROM crm.customers", "postgres")
    fresh = MinedQuery(
        source="a",
        fingerprint="a" * 64,
        dialect="postgres",
        calls=1,
        last_seen=EXEC_TS,
        facts=facts,
    )
    stale = MinedQuery(
        source="a",
        fingerprint="b" * 64,
        dialect="postgres",
        calls=1,
        last_seen=EXEC_TS - timedelta(days=90),
        facts=facts,
    )
    ranked = rank((stale, fresh), exec_ts=EXEC_TS)
    assert [query.fingerprint for query in ranked] == ["a" * 64, "b" * 64]
    assert ranked[0].score > ranked[1].score


def test_a_statement_without_last_seen_gets_no_recency_credit() -> None:
    """An unknown age is not "fresh": the recency term is dropped, not maximised."""
    assert score(age=None) == 0.0
    assert score(age=age_days(None, EXEC_TS)) == 0.0
    assert score(age=age_days(EXEC_TS, EXEC_TS)) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------------------
# Ingestion: idempotence, watermark, stored counters
# --------------------------------------------------------------------------------------------------


def open_store(tmp_path: Path) -> Store:
    """A migrated metadata store on a throwaway file."""
    store = Store(tmp_path / "meta.sqlite")
    store.initialize()
    return store


def stored_counts(store: Store) -> dict[str, tuple[int, int, tuple[str, ...]]]:
    rows = (
        store.connect()
        .execute(
            f"SELECT fingerprint, calls, statements, users FROM {MINED_QUERY_TABLE}"  # noqa: S608
        )
        .fetchall()
    )
    return {
        str(row["fingerprint"]): (
            int(row["calls"]),
            int(row["statements"]),
            tuple(json.loads(row["users"])),
        )
        for row in rows
    }


def test_ingest_is_idempotent_for_the_same_batch(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    first = ingest(store, BATCH, source="crm", exec_ts=EXEC_TS)
    before = stored_counts(store)
    second = ingest(store, BATCH, source="crm", exec_ts=EXEC_TS)
    assert first.skipped is False
    assert second.skipped is True
    assert stored_counts(store) == before


def test_the_watermark_advances_and_never_moves_back(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    assert watermark(store, "crm") is None
    first = ingest(store, BATCH, source="crm", exec_ts=EXEC_TS)
    assert first.watermark_before is None
    assert first.watermark_after == EXEC_TS
    assert watermark(store, "crm") == EXEC_TS

    older = (
        RawQuery(
            text="SELECT cust_id FROM crm.customers WHERE status = 'C'",
            dialect="postgres",
            calls=3,
            last_seen=EXEC_TS - timedelta(days=30),
        ),
    )
    second = ingest(store, older, source="crm", exec_ts=EXEC_TS)
    assert second.watermark_before == EXEC_TS
    assert second.watermark_after == EXEC_TS  # monotonic: an older row does not rewind it
    assert watermark(store, "crm") == EXEC_TS


def test_ingest_accumulates_a_new_batch_for_the_same_fingerprint(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    ingest(store, BATCH, source="crm", exec_ts=EXEC_TS)
    usage_fingerprint = next(
        query.fingerprint
        for query in mine(BATCH, source="crm", exec_ts=EXEC_TS).queries
        if "usage_events" in (query.facts.normalized or "")
    )

    later = (
        RawQuery(
            text="SELECT COUNT(*) FROM billing.usage_events "
            "WHERE start_time >= now() - interval '21 days'",
            dialect="postgres",
            calls=7,
            user="michal",
            last_seen=EXEC_TS - timedelta(hours=1),
        ),
    )
    report = ingest(store, later, source="crm", exec_ts=EXEC_TS)
    assert report.skipped is False
    calls, statements, users = stored_counts(store)[usage_fingerprint]
    assert (calls, statements) == (57, 3)
    assert users == ("dana", "michal", "yossi")
    assert watermark(store, "crm") == EXEC_TS  # 50 was mined at EXEC_TS already


def test_ingest_keeps_sources_apart(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    ingest(store, BATCH, source="crm", exec_ts=EXEC_TS)
    other = (RawQuery(text="SELECT 1 FROM dual", dialect="oracle", calls=2, last_seen=EXEC_TS),)
    ingest(store, other, source="oracle", exec_ts=EXEC_TS)
    assert watermark(store, "crm") == EXEC_TS
    assert watermark(store, "oracle") == EXEC_TS
    assert len(read(store, exec_ts=EXEC_TS, source="crm")) == 2
    assert len(read(store, exec_ts=EXEC_TS)) == 3


def test_read_returns_the_ranked_candidates_with_their_hints(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    ingest(store, BATCH, source="crm", exec_ts=EXEC_TS)
    queries = read(store, exec_ts=EXEC_TS, source="crm")
    assert [query.score for query in queries] == sorted(
        (query.score for query in queries), reverse=True
    )
    assert any(query.facts.hints for query in queries)
    stored = queries[0].facts.normalized or ""
    assert NOW_MARKER in stored or LITERAL_MARKER in stored  # the shape, never the raw text
    assert "interval '7 days'" not in stored


def test_read_rescores_at_the_injected_time(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    ingest(store, BATCH, source="crm", exec_ts=EXEC_TS)
    now = read(store, exec_ts=EXEC_TS)
    later = read(store, exec_ts=EXEC_TS + timedelta(days=365))
    assert len(now) == len(later)
    for before, after in zip(now, later, strict=True):
        assert after.score <= before.score  # everything got older


def test_ingest_ignores_an_empty_batch(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    report = ingest(store, (), source="crm", exec_ts=EXEC_TS)
    assert isinstance(report, IngestReport)
    assert report.queries == ()
    assert watermark(store, "crm") is None
    assert (
        store.connect()
        .execute(
            f"SELECT COUNT(*) AS n FROM {WATERMARK_TABLE}"  # noqa: S608
        )
        .fetchone()["n"]
        == 0
    )
