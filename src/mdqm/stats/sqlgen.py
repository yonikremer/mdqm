"""Read-only SQL for statistics: a one-row existence probe, a sample, and NULLIF sentinels.

Identifiers are checked against a narrow pattern and then quoted by sqlglot. Values become
sqlglot literals. Nothing here is assembled by concatenating a user string into the statement
text. The sample clause is dialect syntax sqlglot does not always model; its only interpolated
pieces are integers this module itself checked.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final, cast

import sqlglot
from sqlglot import exp

__all__ = [
    "existence_sql",
    "nullif_expression",
    "require_ident",
    "sample_sql",
    "sentinel_sentence",
]

_IDENT: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")

_DIALECT: Final[dict[str, str]] = {
    "postgres": "postgres",
    "postgresql": "postgres",
    "duckdb": "duckdb",
    "oracle": "oracle",
    "tsql": "tsql",
    "mssql": "tsql",
}


def require_ident(name: str) -> exp.Identifier:
    """``name`` as a quoted identifier, or a :class:`ValueError` when it is not a plain name.

    A stats reader quotes names that came from a catalog. A name with a quote, a space or a
    comment does not come from a catalog we trust, and it never reaches the SQL text.
    """
    if not isinstance(name, str) or _IDENT.fullmatch(name) is None:
        raise ValueError(f"{name!r} is not an identifier the stats reader will quote")
    return exp.to_identifier(name, quoted=True)


def _dialect(dialect: str) -> str:
    try:
        return _DIALECT[dialect]
    except KeyError:
        known = ", ".join(sorted(_DIALECT))
        raise ValueError(
            f"unsupported dialect {dialect!r}; the stats reader knows {known}"
        ) from None


def sentinel_sentence(column: str, sentinels: Sequence[str]) -> str:
    """The one sentence both the linter and the compiler report, so a retry does not double it."""
    shown = ", ".join(repr(item) for item in _sentinel_order(sentinels))
    return (
        f"column {column} is read through NULLIF so {shown} counts as missing, "
        "because the dictionary marks it"
    )


def _sentinel_order(sentinels: Sequence[str]) -> tuple[str, ...]:
    """Empty string first, then ``0``, then the rest: the NULLIF nest is stable."""
    unique = set(sentinels)
    return tuple(sorted(unique, key=lambda item: (item != "", item != "0", item)))


def nullif_expression(column: exp.Expression, sentinels: Sequence[str]) -> exp.Expression:
    """``NULLIF(NULLIF(column, ''), 0)`` for the sentinels the dictionary listed.

    ``"0"`` is the number 0. Every other sentinel, including ``""`` and a whitespace string, is
    a string literal. An empty sentinel list returns ``column`` unchanged.
    """
    expression = column
    for sentinel in _sentinel_order(sentinels):
        literal: exp.Expression = (
            exp.Literal.number(0) if sentinel == "0" else exp.Literal.string(sentinel)
        )
        expression = cast(exp.Expression, exp.func("NULLIF", expression, literal))
    return expression


def existence_sql(*, table: str, column: str, value: str, dialect: str) -> str:
    """``SELECT 1 AS hit FROM table WHERE column = value LIMIT 1``.

    One row, one comparison, no function that could write. The caller runs it under the
    read-only account and the governor's timeout. ``value`` is a literal, never interpolated.
    """
    query = (
        exp.select(exp.alias_(exp.Literal.number(1), "hit"))
        .from_(require_ident(table))
        .where(exp.EQ(this=exp.column(require_ident(column)), expression=exp.Literal.string(value)))
        .limit(1)
    )
    rendered = query.sql(dialect=_dialect(dialect))
    parsed = sqlglot.parse_one(rendered, read=_dialect(dialect))
    if not isinstance(parsed, (exp.Select, exp.Union)):
        raise ValueError("the existence probe did not render as a single SELECT")
    return rendered


def sample_sql(
    *,
    table: str,
    columns: Sequence[str],
    dialect: str,
    rows: int,
    seed: int,
    percent: int = 1,
) -> str:
    """A bounded sample of ``columns`` from ``table``.

    DuckDB uses a repeatable reservoir sample, which is deterministic for a seed. PostgreSQL
    uses ``TABLESAMPLE SYSTEM ... REPEATABLE``. Oracle and SQL Server have no repeatable
    system sample; the percent is still an integer and the statement stays a single SELECT.
    ``rows`` and ``seed`` and ``percent`` are integers this function checks before they appear
    in the text.
    """
    if rows < 1:
        raise ValueError(f"sample rows must be >= 1, got {rows}")
    if seed < 0:
        raise ValueError(f"sample seed must be >= 0, got {seed}")
    if not 1 <= percent <= 100:
        raise ValueError(f"sample percent must be in 1..100, got {percent}")
    if not columns:
        raise ValueError("a sample names at least one column")
    selected = [exp.column(require_ident(name)) for name in columns]
    body = exp.select(*selected).from_(require_ident(table)).sql(dialect=_dialect(dialect))
    name = _dialect(dialect)
    # rows, seed and percent are ints checked above. `body` is sqlglot output of quoted
    # identifiers. The sample clause is dialect syntax sqlglot does not render.
    if name == "duckdb":
        rendered = f"{body} USING SAMPLE {rows} ROWS (reservoir, {seed})"
    elif name == "postgres":
        rendered = f"{body} TABLESAMPLE SYSTEM ({percent}) REPEATABLE ({seed}) LIMIT {rows}"
    elif name == "oracle":
        rendered = f"{body} SAMPLE ({percent}) FETCH FIRST {rows} ROWS ONLY"
    else:
        rendered = (
            f"{body} TABLESAMPLE ({percent} PERCENT) "
            f"ORDER BY (SELECT NULL) OFFSET 0 ROWS FETCH NEXT {rows} ROWS ONLY"
        )
    return rendered
