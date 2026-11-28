"""Validate the golden SQL corpus in ``tests/data/queries``.

The corpus is the hand-written oracle for the SQL meters: the read-only guard (C01), the
``exec_ts`` rewriter (C02) and the deterministic query miner (S08). This module checks the
corpus *itself* and never implements an extraction of its own: a statement either parses in
its dialect, and its ``-- expect`` block either validates against :class:`ExpectedExtraction`,
or the corpus is wrong.

File format (``<dialect>.sql``, dialect dir names are sqlglot dialect names)::

    -- expect: {"id": "pg-01", "tables": ["customers"], "filters": ["status"]}
    SELECT cust_id FROM crm.customers WHERE status = 'A'
    -- ---
    -- expect: {...}
    SELECT ...

* Blocks are separated by a line whose stripped text is exactly ``-- ---``.
* Each block starts with one ``-- expect: <single-line JSON>`` line; the rest is one statement.
  Comment lines that appear before the first ``-- expect`` line belong to the file header.
* Statements are read-only SELECTs only, and each is short enough to review by eye.

Expect-block semantics (the contract the miners match against):

``tables``
    Base table names, bare (schema and alias stripped), unique. CTE names and subquery aliases
    are *not* base tables. A name that appears both as a table and as a CTE must not be used.
``joins``
    Equality edges, listed in the order the ON clauses appear. ``tables`` is ``(left, right)``
    in FROM order and ``columns[i]`` is the column of ``tables[i]`` used in the equality (also
    for correlated ``EXISTS`` edges and Oracle ``(+)`` joins). ``conversion`` is the lowercase
    function name of a format/scale conversion applied to one side (``lpad``, ``to_char``,
    ``cast``, ``convert``, ``to_number``, ``try_cast``); ``conversion_expr`` is that side's
    expression copied verbatim from the statement. A plain equality has both ``None``.
``filters``
    Columns named in WHERE/HAVING predicates, as written (bare names; a subquery alias such as
    ``t.n`` counts as ``n``). Columns that only appear inside a JOIN ON clause belong to
    ``joins``, not here.
``time_columns``
    The subset of ``filters`` that are time-typed columns used in a range predicate: event time
    (``start_time``), ingestion time (``insertion_time``), ``updated_at``.
``group_by``
    GROUP BY terms as written, whitespace collapsed (a column name or an expression such as
    ``DATE_TRUNC('month', start_time)``).
``aggregations``
    Lowercase aggregate function names used anywhere in the statement.
``rounding``
    Rounding and scale expressions copied verbatim from the statement, e.g.
    ``ROUND(credit_limit / 5000) * 5000`` or ``limit_k * 10``. D8 has to rediscover both the
    rounding granularity and the order of operations, so scale conversions are recorded here.
``now_functions``
    Lowercase names of the non-deterministic time functions that C02 must rewrite, without
    parentheses or arguments (``now``, ``current_timestamp``, ``sysdate``, ``getdate``, ...).
    Wrappers around them (``ADD_MONTHS``, ``EOMONTH``) are not now-like and are not listed.

Comparisons are order-insensitive for every list; the corpus is not required to be sorted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import sqlglot
import sqlglot.errors as sqlglot_errors
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlglot import exp

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "queries"
SEPARATOR = "-- ---"
EXPECT_PREFIX = "-- expect:"
DIALECTS = ("postgres", "oracle", "tsql")
MIN_STATEMENTS = 40
MIN_CONVERSIONS = 10

#: Closed set so a typo in the corpus fails instead of silently disagreeing with the miner.
AGGREGATIONS = frozenset({"avg", "count", "max", "min", "stddev", "sum", "variance"})
NOW_FUNCTIONS = frozenset(
    {
        "current_date",
        "current_timestamp",
        "getdate",
        "getutcdate",
        "localtimestamp",
        "now",
        "sysdate",
        "sysdatetime",
        "systimestamp",
    }
)
CONVERSIONS = frozenset({"cast", "convert", "lpad", "to_char", "to_number", "try_cast"})


class JoinEdge(BaseModel):
    """One equality edge of a join, with the conversion applied to one of its sides."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tables: tuple[str, str]
    columns: tuple[str, str]
    conversion: str | None = None
    conversion_expr: str | None = None

    @model_validator(mode="after")
    def _check(self) -> JoinEdge:
        if not self.tables[0] or not self.tables[1]:
            raise ValueError(f"join edge tables must be named: {self.tables}")
        if not self.columns[0] or not self.columns[1]:
            raise ValueError(f"join edge columns must be named: {self.columns}")
        if (self.conversion is None) != (self.conversion_expr is None):
            raise ValueError(
                f"join edge {self.tables}: conversion and conversion_expr go together "
                f"(got {self.conversion!r} / {self.conversion_expr!r})"
            )
        if self.conversion is not None and self.conversion not in CONVERSIONS:
            raise ValueError(
                f"join edge {self.tables}: unknown conversion {self.conversion!r}, "
                f"use one of {sorted(CONVERSIONS)}"
            )
        return self


class ExpectedExtraction(BaseModel):
    """What a miner must report for one corpus statement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^(pg|or|ts)-\d{2}$")
    tables: tuple[str, ...] = Field(min_length=1)
    joins: tuple[JoinEdge, ...] = ()
    filters: tuple[str, ...] = ()
    time_columns: tuple[str, ...] = ()
    group_by: tuple[str, ...] = ()
    aggregations: tuple[str, ...] = ()
    rounding: tuple[str, ...] = ()
    now_functions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> ExpectedExtraction:
        for name in (
            "tables",
            "filters",
            "time_columns",
            "group_by",
            "aggregations",
            "rounding",
            "now_functions",
        ):
            values = getattr(self, name)
            dupes = sorted({value for value in values if values.count(value) > 1})
            if dupes:
                raise ValueError(f"{self.id}: {name} repeats {dupes}")
        unknown = sorted(set(self.aggregations) - AGGREGATIONS)
        if unknown:
            raise ValueError(f"{self.id}: unknown aggregations {unknown}")
        unknown_now = sorted(set(self.now_functions) - NOW_FUNCTIONS)
        if unknown_now:
            raise ValueError(f"{self.id}: unknown now functions {unknown_now}")
        stray = sorted(set(self.time_columns) - set(self.filters))
        if stray:
            raise ValueError(f"{self.id}: time_columns not in filters: {stray}")
        return self

    @property
    def has_conversion(self) -> bool:
        """True when a miner should surface at least one ConversionHint for this statement."""
        return bool(self.rounding) or any(edge.conversion for edge in self.joins)


class CorpusStatement(NamedTuple):
    """One parsed corpus block."""

    dialect: str
    id: str
    sql: str
    expect: ExpectedExtraction | None
    problems: tuple[str, ...]
    lineno: int


def _flatten(text: str) -> str:
    """Whitespace- and case-insensitive form used for verbatim-substring checks."""
    return re.sub(r"\s+", "", text).lower()


def _blocks(text: str) -> list[tuple[int, list[str]]]:
    """Split a corpus file into (first line number, lines) blocks."""
    blocks: list[tuple[int, list[str]]] = []
    current: list[str] = []
    start = 1
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.strip() == SEPARATOR:
            blocks.append((start, current))
            current = []
            start = lineno + 1
        else:
            current.append(line)
    blocks.append((start, current))
    return [(first, lines) for first, lines in blocks if any(line.strip() for line in lines)]


def _parse_block(dialect: str, lineno: int, lines: list[str]) -> CorpusStatement:
    expect_at = next(
        (index for index, line in enumerate(lines) if line.strip().startswith(EXPECT_PREFIX)),
        None,
    )
    if expect_at is None:
        return CorpusStatement(
            dialect,
            "",
            "",
            None,
            (f"{dialect}.sql:{lineno}: block has no '{EXPECT_PREFIX}' line",),
            lineno,
        )
    problems: list[str] = []
    header = lines[:expect_at]
    for offset, line in enumerate(header):
        if line.strip() and not line.lstrip().startswith("--"):
            problems.append(f"{dialect}.sql:{lineno + offset}: text before the expect line")

    raw_expect = lines[expect_at].strip()[len(EXPECT_PREFIX) :].strip()
    try:
        payload: Any = json.loads(raw_expect)
    except json.JSONDecodeError as exc:
        return CorpusStatement(
            dialect,
            "",
            "",
            None,
            (f"{dialect}.sql:{lineno + expect_at}: expect is not valid JSON: {exc}",),
            lineno,
        )

    expect: ExpectedExtraction | None = None
    try:
        expect = ExpectedExtraction.model_validate(payload)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()
        )
        problems.append(f"{dialect}.sql:{lineno + expect_at}: invalid expect block: {details}")

    sql = "\n".join(lines[expect_at + 1 :]).strip()
    if not sql:
        problems.append(f"{dialect}.sql:{lineno + expect_at}: expect block has no statement")
    return CorpusStatement(
        dialect,
        expect.id if expect else "",
        sql,
        expect,
        tuple(problems),
        lineno + expect_at,
    )


def load_corpus() -> tuple[list[CorpusStatement], list[str]]:
    """Parse every corpus file; return the statements plus every structural problem found."""
    statements: list[CorpusStatement] = []
    problems: list[str] = []
    for dialect in DIALECTS:
        path = CORPUS_DIR / f"{dialect}.sql"
        if not path.exists():
            problems.append(f"missing corpus file: {path}")
            continue
        data = path.read_bytes()
        if b"\r\n" in data:
            problems.append(f"{path.name}: CRLF line endings, the corpus is LF only")
        text = data.decode("utf-8")
        for lineno, lines in _blocks(text):
            statement = _parse_block(dialect, lineno, lines)
            statements.append(statement)
            problems.extend(statement.problems)
    return statements, problems


STATEMENTS, PARSE_PROBLEMS = load_corpus()
PARSED = [stmt for stmt in STATEMENTS if stmt.expect is not None and stmt.sql]
IDS = [stmt.id for stmt in PARSED]


def _word(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w]){re.escape(name)}(?![\w])", re.IGNORECASE)


def _consistency_problems(stmt: CorpusStatement) -> list[str]:
    """Cheap cross-checks: does the expect block describe this statement at all?"""
    assert stmt.expect is not None
    expect = stmt.expect
    sql = stmt.sql
    flat = _flatten(sql)
    problems: list[str] = []
    where = f"{stmt.dialect}.sql:{stmt.lineno} [{expect.id}]"

    for table in expect.tables:
        if not _word(table).search(sql):
            problems.append(f"{where}: table {table!r} does not appear in the statement")
    for column in expect.filters:
        if not _word(column).search(sql):
            problems.append(f"{where}: filter column {column!r} does not appear")
    for agg in expect.aggregations:
        if not re.search(rf"\b{re.escape(agg)}\s*\(", sql, re.IGNORECASE):
            problems.append(f"{where}: aggregation {agg!r} is not called")
    for now_function in expect.now_functions:
        if not _word(now_function).search(sql):
            problems.append(f"{where}: now function {now_function!r} does not appear")
    for field_name in ("group_by", "rounding"):
        for item in getattr(expect, field_name):
            if _flatten(item) not in flat:
                problems.append(f"{where}: {field_name} item {item!r} is not in the statement")
    for edge in expect.joins:
        for table in edge.tables:
            if not _word(table).search(sql):
                problems.append(f"{where}: join table {table!r} does not appear")
        for column in edge.columns:
            if not _word(column).search(sql):
                problems.append(f"{where}: join column {column!r} does not appear")
        if edge.conversion is not None:
            if not re.search(rf"\b{re.escape(edge.conversion)}\s*\(", sql, re.IGNORECASE):
                problems.append(f"{where}: conversion {edge.conversion!r} is not called")
            assert edge.conversion_expr is not None
            if _flatten(edge.conversion_expr) not in flat:
                problems.append(
                    f"{where}: conversion_expr {edge.conversion_expr!r} is not in the statement"
                )
            try:
                sqlglot.parse_one(f"SELECT {edge.conversion_expr}", read=stmt.dialect)
            except sqlglot_errors.SqlglotError as exc:
                problems.append(
                    f"{where}: conversion_expr {edge.conversion_expr!r} is not valid SQL: {exc}"
                )
    for item in expect.rounding:
        try:
            sqlglot.parse_one(f"SELECT {item}", read=stmt.dialect)
        except sqlglot_errors.SqlglotError as exc:
            problems.append(f"{where}: rounding item {item!r} is not a valid expression: {exc}")
    return problems


@pytest.fixture(scope="module")
def corpus() -> tuple[list[CorpusStatement], list[str]]:
    """Every parsed statement plus every structural problem (asserted empty by the caller)."""
    return PARSED, PARSE_PROBLEMS


def test_corpus_files_parse_correctly(corpus: tuple[list[CorpusStatement], list[str]]) -> None:
    statements, problems = corpus
    assert statements, "the corpus is empty"
    assert not problems, "corpus problems:\n" + "\n".join(problems)


def test_corpus_is_large_enough(corpus: tuple[list[CorpusStatement], list[str]]) -> None:
    statements, _ = corpus
    per_dialect = {
        dialect: [stmt for stmt in statements if stmt.dialect == dialect] for dialect in DIALECTS
    }
    for dialect, stmts in per_dialect.items():
        assert len(stmts) >= MIN_STATEMENTS, (
            f"{dialect}.sql has {len(stmts)} < {MIN_STATEMENTS} statements"
        )


def test_corpus_has_conversion_and_rounding_cases(
    corpus: tuple[list[CorpusStatement], list[str]],
) -> None:
    statements, _ = corpus
    for dialect in DIALECTS:
        stmts = [stmt for stmt in statements if stmt.dialect == dialect]
        conversion_cases = [stmt.id for stmt in stmts if stmt.expect and stmt.expect.has_conversion]
        assert len(conversion_cases) >= MIN_CONVERSIONS, (
            f"{dialect}.sql has {len(conversion_cases)} conversion/rounding statements "
            f"(< {MIN_CONVERSIONS}): {conversion_cases}"
        )


def test_statement_ids_are_unique_and_dialect_tagged(
    corpus: tuple[list[CorpusStatement], list[str]],
) -> None:
    statements, _ = corpus
    duplicates = sorted({stmt_id for stmt_id in IDS if IDS.count(stmt_id) > 1})
    assert not duplicates, f"duplicate expect ids: {duplicates}"
    prefixes = {"postgres": "pg", "oracle": "or", "tsql": "ts"}
    for stmt in statements:
        assert stmt.expect is not None
        assert stmt.id.startswith(prefixes[stmt.dialect] + "-"), (
            f"{stmt.dialect}.sql:{stmt.lineno}: id {stmt.id!r} does not use the "
            f"{prefixes[stmt.dialect]!r} prefix"
        )


def test_no_statement_appears_twice(corpus: tuple[list[CorpusStatement], list[str]]) -> None:
    statements, _ = corpus
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for stmt in statements:
        key = _flatten(stmt.sql)
        if key in seen:
            duplicates.append(f"{stmt.id} duplicates {seen[key]}")
        else:
            seen[key] = stmt.id
    assert not duplicates, "copy-pasted statements:\n" + "\n".join(duplicates)


def test_every_statement_is_read_only(corpus: tuple[list[CorpusStatement], list[str]]) -> None:
    statements, _ = corpus
    offenders = [
        stmt.id
        for stmt in statements
        if not isinstance(sqlglot.parse_one(stmt.sql, read=stmt.dialect), (exp.Select, exp.Union))
    ]
    assert not offenders, f"not a plain SELECT: {offenders}"


@pytest.mark.parametrize("stmt", PARSED, ids=IDS or None)
def test_statement_parses_in_its_dialect(stmt: CorpusStatement) -> None:
    expressions = sqlglot.parse(stmt.sql, read=stmt.dialect, error_level=sqlglot.ErrorLevel.RAISE)
    assert len(expressions) == 1, f"{stmt.id}: {len(expressions)} statements in one block"


@pytest.mark.parametrize("stmt", PARSED, ids=IDS or None)
def test_expect_block_matches_statement(stmt: CorpusStatement) -> None:
    problems = _consistency_problems(stmt)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    ("dialect", "snippet"),
    [
        ("postgres", "::"),
        ("postgres", "interval '"),
        ("postgres", "DISTINCT ON"),
        ("postgres", "WITH "),
        ("oracle", "SYSDATE"),
        ("oracle", "ROWNUM"),
        ("oracle", "FETCH FIRST"),
        ("oracle", "TO_CHAR"),
        ("oracle", "(+)"),
        ("tsql", "[client_no]"),
        ("tsql", "TOP 100"),
        ("tsql", "GETDATE()"),
        ("tsql", "OFFSET "),
        ("tsql", "WITH (NOLOCK)"),
    ],
)
def test_dialect_specific_features_are_covered(dialect: str, snippet: str) -> None:
    text = (CORPUS_DIR / f"{dialect}.sql").read_text(encoding="utf-8")
    assert snippet in text, f"{dialect}.sql no longer covers {snippet!r}"


def test_corpus_contains_hebrew_literals() -> None:
    for dialect in DIALECTS:
        text = (CORPUS_DIR / f"{dialect}.sql").read_text(encoding="utf-8")
        assert re.search(r"[\u0590-\u05ff]", text), f"{dialect}.sql has no Hebrew literal"
