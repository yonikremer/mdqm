"""Result summariser: a bounded summary, a small sample and an actionable handle (task C16).

The calling agent is weak and pays per token, so it never receives a whole result set
(``docs/DESIGN.md`` section 9: "result = summary + sample + handle"). This module turns an Arrow
table (or an adapter's lazy batch stream) into a fixed-shape answer:

* ``rows_total`` and, when rows were left out, ``rows_not_shown`` plus ``truncated`` -- drops are
  counted, never silent;
* per-column statistics (nulls, distinct, min/max) for the leading columns, each carrying how exact
  it is;
* a sample of at most ``sample_rows`` rows, labelled as a sample;
* a handle plus a ready-to-copy ``next_call`` for the rest of the rows.

Everything is deterministic: column order follows the schema, sample rows are the first ones in
source order, numbers are never formatted, and no clock or randomness is read.

Token estimation has no dependency and is deliberately conservative (a weak model's tokenizer is
unknown): ``chars / 4`` for ASCII and ``chars / 2.5`` for everything else. Hebrew is mostly two-byte
UTF-8 and tokenises worse than English, so counting it at 2.5 characters per token over-estimates
rather than under-estimates. The estimate is applied to the JSON the agent actually receives, and
the summary shrinks itself (statistics first, then sample rows) until it fits the budget.

Cost is bounded by Arrow C++ kernels, not Python loops: ``pyarrow`` reads 10 million rows for the
statistics in well under a second, and ``distinct`` is computed over at most
:data:`DISTINCT_SCAN_ROWS` rows -- reported as ``lower_bound`` when that is less than the row count.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable
from datetime import date, datetime, time
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
from pydantic import Field

from mdqm.models.base import JsonObject, MDQMModel

# pyarrow ships no type stubs (pyproject sets `ignore_missing_imports` for mypy for that reason), so
# pyright reports every attribute of a pyarrow type or module -- ``pa.Table.schema``,
# ``pc.count_distinct``, ``pc.min_max`` -- as unknown. Those calls are real and covered by the unit
# tests below; pyright is not part of scripts/check.py. Suppressed here only, not project-wide.
# pyright: reportAttributeAccessIssue=false

__all__ = [
    "DEFAULT_PAGE_ROWS",
    "DEFAULT_SAMPLE_ROWS",
    "DEFAULT_STAT_COLUMNS",
    "DEFAULT_TOKEN_BUDGET",
    "DISTINCT_SCAN_ROWS",
    "HARD_TOKEN_BUDGET",
    "MIN_TOKEN_BUDGET",
    "ColumnStats",
    "DistinctKind",
    "ResultSummary",
    "effective_budget",
    "estimate_tokens",
    "summarise",
]

#: Default share of an answer's budget (DESIGN section 9: warn above 10,000, hard cap 25,000).
DEFAULT_TOKEN_BUDGET = 4_000
#: Ceiling the configuration can never raise, because the agent's context is the real limit.
HARD_TOKEN_BUDGET = 25_000
#: Floor of the budget. Below this the envelope alone (row count, handle, next call) no longer fits,
#: so a smaller request is raised to it and the answer says so (``budget_clamped``).
MIN_TOKEN_BUDGET = 512
#: Sample rows offered to the agent unless configuration says otherwise.
DEFAULT_SAMPLE_ROWS = 10
#: Columns that get statistics unless configuration says otherwise. Bounds the cost of a wide table.
DEFAULT_STAT_COLUMNS = 8
#: Rows scanned for ``distinct``. Keeps a high-cardinality 10-million-row column off the hot path.
DISTINCT_SCAN_ROWS = 200_000
#: Rows the ``next_call`` hint suggests per page.
DEFAULT_PAGE_ROWS = 1_000

_CHARS_PER_TOKEN_ASCII = 4.0
_CHARS_PER_TOKEN_OTHER = 2.5

#: ``tokens_estimated`` is part of the JSON it measures; a few passes reach the fixpoint.
_TOKEN_SETTLE_PASSES = 8

DistinctKind = Literal["exact", "lower_bound", "unavailable"]
"""How much to trust ``ColumnStats.distinct``: a count over a row sample is only a lower bound."""


class ColumnStats(MDQMModel):
    """Statistics for one column, with every approximation named."""

    name: str
    type: str = Field(description="Arrow type, e.g. 'int64' or 'timestamp[us]'.")
    nulls: int = Field(ge=0)
    distinct: int = Field(ge=0)
    distinct_kind: DistinctKind = Field(
        description="'exact', 'lower_bound' (row sample) or 'unavailable' (type has no kernel)."
    )
    min_value: str | int | float | bool | None = Field(
        default=None, description="JSON scalar, or null when the type has no ordering."
    )
    max_value: str | int | float | bool | None = None


class ResultSummary(MDQMModel):
    """What the agent gets instead of the result set: summary, sample, handle."""

    status: Literal["ok"] = "ok"
    rows_total: int = Field(ge=0)
    handle: str = Field(min_length=1, description="Opaque id for the stored result.")
    sample: tuple[JsonObject, ...] = Field(description="The first rows, in source order.")
    sample_rows_wanted: int = Field(ge=0)
    sample_label: str = Field(description="Human label: a sample, and of what.")
    rows_not_shown: int = Field(ge=0, description="Rows the agent has not seen; 0 means complete.")
    truncated: bool = Field(description="True when rows were left out of this answer.")
    columns: tuple[ColumnStats, ...] = ()
    columns_total: int = Field(ge=0)
    columns_without_stats: int = Field(
        ge=0, description="Columns with no statistics here (not covered, or dropped for budget)."
    )
    tokens_estimated: int = Field(ge=0, description="Conservative token estimate of this answer.")
    budget_tokens: int = Field(gt=0, description="Budget actually applied after clamping.")
    budget_clamped: bool = Field(description="True when the requested budget was not used as is.")
    next_call: str = Field(description="Exact call that pages the remaining rows.")

    @property
    def is_complete(self) -> bool:
        """True when the sample holds every row, so no ``next_call`` is needed."""
        return not self.truncated

    def column_by_name(self, name: str) -> ColumnStats:
        """Statistics for column ``name``.

        Raises ``KeyError`` when the column has none here; ``columns_without_stats`` says how many
        were omitted (not covered, or dropped for budget).
        """
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)


def estimate_tokens(text: str) -> int:
    """Conservatively estimate the tokens ``text`` costs, rounded up.

    ASCII is counted at :data:`_CHARS_PER_TOKEN_ASCII` characters per token, anything else at
    :data:`_CHARS_PER_TOKEN_OTHER`, because Hebrew (and other non-Latin scripts) tokenise worse
    than English. Over-estimating is the safe direction: the budget is a ceiling.
    """
    if text.isascii():
        return math.ceil(len(text) / _CHARS_PER_TOKEN_ASCII)
    non_ascii = sum(1 for char in text if not char.isascii())
    ascii_chars = len(text) - non_ascii
    return math.ceil(ascii_chars / _CHARS_PER_TOKEN_ASCII + non_ascii / _CHARS_PER_TOKEN_OTHER)


def effective_budget(requested: int) -> tuple[int, bool]:
    """Clamp ``requested`` into ``[MIN_TOKEN_BUDGET, HARD_TOKEN_BUDGET]``; also report a change.

    The second element is the ``budget_clamped`` flag the answer carries, so a configuration that
    asks for more (or less) than we may deliver is visible instead of silently ignored.
    """
    if requested < 1:
        raise ValueError(f"budget_tokens must be positive, got {requested}")
    clamped = min(max(requested, MIN_TOKEN_BUDGET), HARD_TOKEN_BUDGET)
    return clamped, clamped != requested


def summarise(
    data: pa.Table | pa.RecordBatch | Iterable[pa.RecordBatch],
    *,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    page_rows: int = DEFAULT_PAGE_ROWS,
    budget_tokens: int = DEFAULT_TOKEN_BUDGET,
    max_stat_columns: int = DEFAULT_STAT_COLUMNS,
    handle: str | None = None,
) -> ResultSummary:
    """Summarise ``data`` into a budget-bounded answer the agent can act on.

    ``data`` may be a :class:`pyarrow.Table`, a single batch, or anything yielding record batches --
    including an adapter's :class:`~mdqm.adapters.base.ArrowStream`, which is just an iterable. The
    caller keeps ownership of the stream (closing it is the governor's job).

    ``sample_rows``, ``page_rows``, ``budget_tokens`` and ``max_stat_columns`` come from
    configuration; the defaults are generic product defaults. ``handle`` is the caller's registry
    key for the stored result; when omitted a deterministic one is derived from the schema and the
    row count (fine for a first answer, useless for paging a stored result).

    The returned summary never exceeds ``budget_tokens`` once clamped: statistics are dropped first,
    then sample rows, so the row count and the handle always survive.
    """
    if sample_rows < 0:
        raise ValueError(f"sample_rows must not be negative, got {sample_rows}")
    if page_rows < 1:
        raise ValueError(f"page_rows must be positive, got {page_rows}")
    if max_stat_columns < 0:
        raise ValueError(f"max_stat_columns must not be negative, got {max_stat_columns}")

    budget, clamped = effective_budget(budget_tokens)
    table, schema = _as_table(data)
    rows_total = 0 if table is None else table.num_rows

    stats: list[ColumnStats] = []
    if table is not None:
        stats = [_column_stats(table, i) for i in range(min(len(schema), max_stat_columns))]
    sample: list[JsonObject] = [] if table is None else _sample(table, sample_rows)
    result_handle = handle or _derived_handle(schema, rows_total)

    shown_stats = list(stats)
    shown_sample = list(sample)

    def build(tokens: int) -> ResultSummary:
        shown = len(shown_sample)
        return ResultSummary(
            rows_total=rows_total,
            handle=result_handle,
            sample=tuple(shown_sample),
            sample_rows_wanted=sample_rows,
            sample_label=_sample_label(shown, rows_total),
            rows_not_shown=rows_total - shown,
            truncated=rows_total > shown,
            columns=tuple(shown_stats),
            columns_total=len(schema),
            columns_without_stats=len(schema) - len(shown_stats),
            tokens_estimated=tokens,
            budget_tokens=budget,
            budget_clamped=clamped,
            next_call=_next_call(result_handle, shown, rows_total - shown, page_rows),
        )

    while True:
        summary = _settled(build)
        if summary.tokens_estimated <= budget:
            return summary
        # Statistics first: a sample row carries more meaning per token than a min/max pair.
        if shown_stats:
            shown_stats.pop()
        elif shown_sample:
            shown_sample.pop()
        else:  # pragma: no cover - the envelope is smaller than MIN_TOKEN_BUDGET by design
            raise RuntimeError(
                f"summary envelope needs {summary.tokens_estimated} tokens, "
                f"more than the floor budget {MIN_TOKEN_BUDGET}"
            )


def _settled(build: Callable[[int], ResultSummary]) -> ResultSummary:
    """Build until ``tokens_estimated`` equals the measured size of the built model."""
    tokens = 0
    for _ in range(_TOKEN_SETTLE_PASSES):
        summary = build(tokens)
        measured = estimate_tokens(summary.model_dump_json())
        if measured == tokens:
            return summary
        tokens = measured
    raise RuntimeError("token estimate did not settle; the summary model is self-referential")


def _as_table(
    data: pa.Table | pa.RecordBatch | Iterable[pa.RecordBatch],
) -> tuple[pa.Table | None, pa.Schema]:
    """Normalise the input to one table view plus the schema to report on.

    ``pa.Table.from_batches`` references the batches instead of copying, so a stream costs no
    extra memory. An empty stream still has a schema, which is what an empty answer reports.
    """
    if isinstance(data, pa.Table):
        return data, data.schema
    if isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data]), data.schema
    batches = list(data)
    if batches:
        return pa.Table.from_batches(batches), batches[0].schema
    schema = getattr(data, "schema", None)
    return None, schema if isinstance(schema, pa.Schema) else pa.schema([])


def _sample(table: pa.Table, sample_rows: int) -> list[JsonObject]:
    """The first ``sample_rows`` rows as JSON-native objects, in schema (and so source) order."""
    if sample_rows == 0 or table.num_rows == 0:
        return []
    rows = table.slice(0, sample_rows).to_pylist()
    return [{key: _json_scalar(value) for key, value in row.items()} for row in rows]


def _column_stats(table: pa.Table, index: int) -> ColumnStats:
    field = table.schema.field(index)
    column = table.column(index)
    distinct, kind = _distinct(column)
    min_value, max_value = _min_max(column)
    return ColumnStats(
        name=field.name,
        type=str(field.type),
        nulls=column.null_count,
        distinct=distinct,
        distinct_kind=kind,
        min_value=min_value,
        max_value=max_value,
    )


def _distinct(column: pa.ChunkedArray) -> tuple[int, DistinctKind]:
    """Distinct non-null values, over a bounded slice so a 10-million-row column stays cheap.

    A count over a slice can only under-count the truth, which is why it is labelled
    ``lower_bound`` rather than passed off as exact.
    """
    scanned = column
    if column.length() > DISTINCT_SCAN_ROWS:
        scanned = column.slice(0, DISTINCT_SCAN_ROWS)
    kind: DistinctKind = "exact" if scanned.length() == column.length() else "lower_bound"
    if pa.types.is_dictionary(scanned.type):
        # count_distinct has no dictionary kernel; the value type is the same data, decoded.
        scanned = scanned.cast(scanned.type.value_type)
    try:
        count = pc.count_distinct(scanned)
    except (pa.ArrowNotImplementedError, pa.ArrowInvalid):
        return 0, "unavailable"
    return int(count.as_py()), kind


def _min_max(column: pa.ChunkedArray) -> tuple[Any, Any]:
    """Min and max as JSON scalars, or ``(None, None)`` when the type has no ordering."""
    try:
        bounds = pc.min_max(column)
    except (pa.ArrowNotImplementedError, pa.ArrowInvalid):
        return None, None
    return _json_scalar(bounds["min"].as_py()), _json_scalar(bounds["max"].as_py())


def _json_scalar(value: Any) -> Any:
    """Convert one Arrow-decoded value to something JSON can hold, without formatting numbers.

    Floats stay floats (a float is not rounded or stringified, so nothing drifts), ``NaN`` and
    infinity become ``null`` because JSON has no such numbers, and exact decimals stay strings
    because converting them to float would lose digits a data analyst cares about.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return bytes(value).hex()
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    return str(value)


def _sample_label(shown: int, rows_total: int) -> str:
    """Say out loud that this is a sample, and of what."""
    if rows_total == 0:
        return "the result is empty: no rows"
    if shown >= rows_total:
        return f"all {rows_total} rows are included"
    return f"first {shown} of {rows_total} rows: a sample, not the full result"


def _next_call(handle: str, offset: int, remaining: int, page_rows: int) -> str:
    """The exact call that pages the rest, so a weak agent does not have to deduce it."""
    if remaining <= 0:
        return "nothing to page: the sample holds every row"
    limit = min(remaining, page_rows)
    return f"get_more(handle={handle!r}, offset={offset}, limit={limit})"


def _derived_handle(schema: pa.Schema, rows_total: int) -> str:
    """A deterministic handle from schema and row count: no clock, no randomness."""
    names = ",".join(field.name for field in schema)
    return f"r_{hashlib.sha256(f'{rows_total}|{names}'.encode()).hexdigest()[:12]}"
