"""Tests for the result summariser (C16).

Table-driven cases cover the shape the agent receives, the statistics, the token heuristic and the
budget rules; hypothesis properties cover the invariants the server relies on (the answer never
exceeds its budget, the sample never exceeds ``sample_rows``, drops are always accounted for, and a
bigger input never reports a smaller count).

Nothing here touches the network, Docker or the wall clock.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime
from decimal import Decimal

import pyarrow as pa
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mdqm.results import (
    DEFAULT_STAT_COLUMNS,
    DEFAULT_TOKEN_BUDGET,
    DISTINCT_SCAN_ROWS,
    HARD_TOKEN_BUDGET,
    MIN_TOKEN_BUDGET,
    effective_budget,
    estimate_tokens,
    summarise,
)

HEBREW = "סטטוס הלקוח פעיל"


def table(rows: int, *, columns: dict[str, list[object]] | None = None) -> pa.Table:
    """A table of ``rows`` rows, ``id`` first, options from ``columns``."""
    data: dict[str, list[object]] = {"id": list(range(rows))}
    if columns:
        data.update(columns)
    return pa.table(data)


# --- acceptance: conservative token heuristic that handles Hebrew -----------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("a", 1),  # 1/4 rounds up
        ("abcd", 1),
        ("abcde", 2),
        ("a" * 100, 25),
        ("א", 1),  # 1/2.5 rounds up
        ("א" * 5, 2),
        ("א" * 10, 4),
        ("abc" + "א", 2),  # 3/4 + 1/2.5 = 1.15
        (HEBREW, 7),  # 17 Hebrew chars + 2 spaces = 19 chars
        ("日本", 1),  # any non-ASCII script counts conservatively, not just Hebrew
    ],
)
def test_estimate_tokens_cases(text: str, expected: int) -> None:
    assert estimate_tokens(text) == expected


@pytest.mark.parametrize("size", [4, 40, 100, 1_000])
def test_hebrew_is_counted_more_conservatively_than_ascii(size: int) -> None:
    """Acceptance: Hebrew-heavy output is counted more conservatively than ASCII."""
    hebrew = estimate_tokens("א" * size)
    ascii_tokens = estimate_tokens("a" * size)
    assert hebrew > ascii_tokens
    assert hebrew >= 1.6 * ascii_tokens  # 2.5 vs 4 characters per token, rounded up


def test_hebrew_summary_fits_budget_and_is_reported_as_hebrew() -> None:
    """A Hebrew result keeps its text (no escaping drift) and still fits the budget."""
    rows = table(50, columns={"status": [HEBREW] * 50})
    summary = summarise(rows, sample_rows=5, budget_tokens=600)
    assert all(row["status"] == HEBREW for row in summary.sample)
    assert summary.tokens_estimated <= summary.budget_tokens
    assert estimate_tokens(summary.model_dump_json()) == summary.tokens_estimated


# --- shape: rows, sample, handle --------------------------------------------------------------


def test_small_table_is_complete() -> None:
    summary = summarise(table(3), sample_rows=10)
    assert summary.rows_total == 3
    assert len(summary.sample) == 3
    assert summary.rows_not_shown == 0
    assert summary.truncated is False
    assert summary.is_complete is True
    assert summary.next_call == "nothing to page: the sample holds every row"
    assert summary.sample_label == "all 3 rows are included"


def test_large_table_reports_the_drop_and_an_actionable_handle() -> None:
    summary = summarise(table(10_000), sample_rows=4, page_rows=500)
    assert summary.rows_total == 10_000
    assert len(summary.sample) == 4
    assert summary.rows_not_shown == 9_996
    assert summary.truncated is True
    assert "sample" in summary.sample_label
    assert "4 of 10000" in summary.sample_label
    assert summary.handle
    assert summary.next_call == f"get_more(handle={summary.handle!r}, offset=4, limit=500)"


def test_empty_table_is_reported_as_empty() -> None:
    summary = summarise(table(0), sample_rows=5)
    assert summary.rows_total == 0
    assert summary.sample == ()
    assert summary.rows_not_shown == 0
    assert summary.truncated is False
    assert summary.sample_label == "the result is empty: no rows"
    assert summary.column_by_name("id").nulls == 0


@pytest.mark.parametrize(("rows", "sample_rows", "shown"), [(100, 0, 0), (100, 1, 1), (100, 7, 7)])
def test_sample_never_exceeds_sample_rows(rows: int, sample_rows: int, shown: int) -> None:
    """Acceptance: the summary never includes more than ``sample_rows`` rows."""
    summary = summarise(table(rows), sample_rows=sample_rows)
    assert len(summary.sample) == shown
    assert len(summary.sample) <= sample_rows
    assert summary.rows_not_shown == rows - shown


def test_handle_is_deterministic_without_a_clock() -> None:
    assert summarise(table(5)).handle == summarise(table(5)).handle
    assert summarise(table(5)).handle != summarise(table(6)).handle


def test_caller_handle_wins() -> None:
    summary = summarise(table(9), handle="q_abc123", sample_rows=1)
    assert summary.handle == "q_abc123"
    assert summary.next_call.startswith("get_more(handle='q_abc123', offset=1")


def test_output_is_byte_identical_across_calls() -> None:
    """Determinism: no dict ordering, no float formatting, no clock."""
    a = summarise(table(20, columns={"tag": ["x", "y"] * 10}), sample_rows=3)
    b = summarise(table(20, columns={"tag": ["x", "y"] * 10}), sample_rows=3)
    assert a.model_dump_json() == b.model_dump_json()
    dumped = json.loads(a.model_dump_json())
    assert list(dumped) == [
        "status",
        "rows_total",
        "handle",
        "sample",
        "sample_rows_wanted",
        "sample_label",
        "rows_not_shown",
        "truncated",
        "columns",
        "columns_total",
        "columns_without_stats",
        "tokens_estimated",
        "budget_tokens",
        "budget_clamped",
        "next_call",
    ]
    assert [column["name"] for column in dumped["columns"]] == ["id", "tag"]
    assert list(dumped["sample"][0]) == ["id", "tag"]


# --- column statistics ------------------------------------------------------------------------


def test_column_stats_cover_nulls_distinct_and_bounds() -> None:
    data = pa.table(
        {
            "num": pa.array([3, None, 1, 3], type=pa.int64()),
            "word": ["b", None, "a", "c"],
        }
    )
    summary = summarise(data)
    num = summary.column_by_name("num")
    word = summary.column_by_name("word")
    assert (num.type, num.nulls, num.distinct, num.distinct_kind) == ("int64", 1, 2, "exact")
    assert (num.min_value, num.max_value) == (1, 3)
    assert (word.nulls, word.distinct, word.min_value, word.max_value) == (1, 3, "a", "c")


def test_all_null_column_has_no_bounds() -> None:
    data = pa.table({"gap": pa.array([None, None], type=pa.int64())})
    gap = summarise(data).column_by_name("gap")
    assert (gap.nulls, gap.distinct, gap.min_value, gap.max_value) == (2, 0, None, None)


def test_dictionary_column_counts_distinct_values() -> None:
    """Categorical columns arrive as dictionaries; count_distinct has no kernel for them."""
    data = pa.table(
        {"status": pa.array(["A", "C", "A"], type=pa.dictionary(pa.int32(), pa.string()))}
    )
    status = summarise(data).column_by_name("status")
    assert (status.distinct, status.distinct_kind) == (2, "exact")


def test_unordered_type_gets_no_bounds_and_no_distinct() -> None:
    data = pa.table({"tags": pa.array([["a"], ["b"]]), "x": [1, 2]})
    tags = summarise(data).column_by_name("tags")
    assert (tags.min_value, tags.max_value) == (None, None)
    assert (tags.distinct, tags.distinct_kind) == (0, "unavailable")


def test_distinct_over_a_long_column_is_labelled_a_lower_bound() -> None:
    rows = DISTINCT_SCAN_ROWS + 1
    column = pa.chunked_array([pa.array(range(rows), type=pa.int64())])
    data = pa.table({"n": column})
    stats = summarise(data, sample_rows=0).column_by_name("n")
    assert stats.distinct_kind == "lower_bound"
    assert stats.distinct <= rows


def test_statistics_cover_only_the_leading_columns() -> None:
    data = pa.table({f"c{i}": [i] for i in range(DEFAULT_STAT_COLUMNS + 4)})
    summary = summarise(data)
    assert summary.columns_total == DEFAULT_STAT_COLUMNS + 4
    assert len(summary.columns) == DEFAULT_STAT_COLUMNS
    assert summary.columns_without_stats == 4
    assert [column.name for column in summary.columns] == [
        f"c{i}" for i in range(DEFAULT_STAT_COLUMNS)
    ]


def test_max_stat_columns_zero_still_reports_the_total() -> None:
    summary = summarise(table(2), max_stat_columns=0)
    assert summary.columns == ()
    assert (summary.columns_total, summary.columns_without_stats) == (1, 1)


# --- value conversion: no formatting drift ----------------------------------------------------


def test_floats_keep_exact_values_and_never_break_json() -> None:
    data = pa.table({"f": [0.1, 1.5, float("nan"), float("inf"), float("-inf")]})
    summary = summarise(data, sample_rows=10)
    assert [row["f"] for row in summary.sample] == [0.1, 1.5, None, None, None]
    json.loads(summary.model_dump_json())  # NaN/inf would make this fail


def test_time_and_exact_numbers_are_rendered_deterministically() -> None:
    data = pa.table(
        {
            "ts": pa.array(
                [datetime(2026, 3, 1, 12, 30, tzinfo=UTC)], type=pa.timestamp("us", "UTC")
            ),
            "day": pa.array([date(2026, 3, 1)]),
            "money": pa.array([Decimal("12.50")], type=pa.decimal128(10, 2)),
        }
    )
    row = summarise(data, sample_rows=1).sample[0]
    assert row["ts"] == "2026-03-01T12:30:00+00:00"
    assert row["day"] == "2026-03-01"
    assert row["money"] == "12.50"


def test_binary_is_decoded_or_hexed_rather_than_dropped() -> None:
    data = pa.table({"raw": pa.array([b"abc", b"\xff\xfe"])})
    sample = summarise(data, sample_rows=2).sample
    assert sample[0]["raw"] == "abc"
    assert sample[1]["raw"] == "fffe"


# --- budget -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "applied", "clamped"),
    [
        (1, MIN_TOKEN_BUDGET, True),
        (MIN_TOKEN_BUDGET, MIN_TOKEN_BUDGET, False),
        (DEFAULT_TOKEN_BUDGET, DEFAULT_TOKEN_BUDGET, False),
        (10_000, 10_000, False),
        (HARD_TOKEN_BUDGET, HARD_TOKEN_BUDGET, False),
        (HARD_TOKEN_BUDGET + 1, HARD_TOKEN_BUDGET, True),
        (10**9, HARD_TOKEN_BUDGET, True),
    ],
)
def test_effective_budget_clamps_and_reports(requested: int, applied: int, clamped: bool) -> None:
    assert effective_budget(requested) == (applied, clamped)


@pytest.mark.parametrize("requested", [0, -5])
def test_budget_must_be_positive(requested: int) -> None:
    with pytest.raises(ValueError, match="budget_tokens must be positive"):
        effective_budget(requested)


def test_default_budget_is_well_below_the_warning_threshold_and_capped() -> None:
    assert DEFAULT_TOKEN_BUDGET < 10_000
    assert HARD_TOKEN_BUDGET == 25_000


def test_over_budget_request_is_clamped_and_says_so() -> None:
    summary = summarise(table(3), budget_tokens=HARD_TOKEN_BUDGET * 1000)
    assert summary.budget_tokens == HARD_TOKEN_BUDGET
    assert summary.budget_clamped is True


def test_tight_budget_drops_statistics_before_sample_rows() -> None:
    """The sample survives longest: it carries more meaning per token than a min/max pair."""
    wide = table(
        40,
        columns={f"c{i}": [f"value-{i}-{row}" for row in range(40)] for i in range(20)},
    )
    summary = summarise(wide, sample_rows=6, budget_tokens=MIN_TOKEN_BUDGET)
    assert summary.tokens_estimated <= summary.budget_tokens
    assert summary.columns == ()
    assert summary.columns_without_stats == summary.columns_total
    assert len(summary.sample) > 0


def test_absurdly_narrow_budget_still_answers_and_counts_the_drops() -> None:
    wide = table(100, columns={"v": ["x" * 200] * 100})
    summary = summarise(wide, sample_rows=10, budget_tokens=1)
    assert summary.budget_tokens == MIN_TOKEN_BUDGET
    assert summary.tokens_estimated <= summary.budget_tokens
    assert summary.rows_total == 100
    assert summary.rows_not_shown == 100 - len(summary.sample)
    assert summary.truncated is (summary.rows_not_shown > 0)


def test_reported_token_count_equals_the_measured_size() -> None:
    summary = summarise(table(30, columns={"s": ["abc"] * 30}), sample_rows=3)
    assert summary.tokens_estimated == estimate_tokens(summary.model_dump_json())


# --- input shapes -----------------------------------------------------------------------------


def test_record_batch_and_batch_stream_match_the_table_answer() -> None:
    data = table(12, columns={"tag": ["a", "b"] * 6})
    expected = summarise(data, sample_rows=3, handle="h").model_dump_json()
    one_batch = summarise(
        pa.RecordBatch.from_pydict(data.to_pydict()), sample_rows=3, handle="h"
    ).model_dump_json()
    stream = summarise(
        data.to_batches(max_chunksize=5), sample_rows=3, handle="h"
    ).model_dump_json()
    assert one_batch == expected
    assert stream == expected


def test_stream_object_with_schema_is_accepted() -> None:
    """An adapter's ArrowStream is just an iterable of batches plus a schema."""

    class FakeStream:
        schema = table(0).schema

        def __iter__(self) -> object:
            yield pa.RecordBatch.from_pydict({"id": [1, 2]})

        def close(self) -> None:  # the caller owns the stream; we must not close it
            raise AssertionError("summarise must not close the caller's stream")

    summary = summarise(FakeStream(), sample_rows=5)
    assert summary.rows_total == 2
    assert summary.rows_not_shown == 0


def test_empty_stream_reports_its_schema() -> None:
    class EmptyStream:
        schema = pa.schema([pa.field("id", pa.int64())])

        def __iter__(self) -> object:
            return iter(())

    summary = summarise(EmptyStream())
    assert summary.rows_total == 0
    assert summary.columns_total == 1
    assert summary.columns_without_stats == 1


def test_table_without_columns_does_not_crash() -> None:
    summary = summarise(pa.table({}), sample_rows=5)
    assert (summary.rows_total, summary.columns_total, summary.columns) == (0, 0, ())


# --- performance ------------------------------------------------------------------------------


def test_ten_million_rows_summarise_under_a_second_and_under_the_budget() -> None:
    """Acceptance: a 10-million-row table summarises in under 1 second, within the budget."""
    rows = 10_000_000
    big = pa.table(
        {
            "cust_id": pa.array(range(rows), type=pa.int64()),
            "status": pa.array(["Active", "Closed", "Hold"] * (rows // 3 + 1)).slice(0, rows),
            "credit": pa.array(range(rows), type=pa.float64()),
        }
    )
    started = time.perf_counter()
    summary = summarise(big, sample_rows=10)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"summarising {rows} rows took {elapsed:.2f}s"
    assert summary.rows_total == rows
    assert summary.tokens_estimated <= summary.budget_tokens
    assert summary.column_by_name("cust_id").distinct_kind == "lower_bound"


# --- properties -------------------------------------------------------------------------------


@st.composite
def any_result(draw: st.DrawFn) -> tuple[pa.Table, int, int, int]:
    rows = draw(st.integers(min_value=0, max_value=40))
    columns = draw(st.integers(min_value=0, max_value=4))
    data: dict[str, list[object]] = {"id": list(range(rows))}
    for index in range(columns):
        width = draw(st.integers(min_value=1, max_value=40))
        data[f"c{index}"] = [("x" * width) for _ in range(rows)]
    sample_rows = draw(st.integers(min_value=0, max_value=12))
    budget = draw(st.integers(min_value=1, max_value=HARD_TOKEN_BUDGET + 5_000))
    return pa.table(data), sample_rows, budget, draw(st.integers(min_value=1, max_value=1_000))


@settings(max_examples=120, deadline=None)
@given(case=any_result())
def test_property_budget_sample_and_drop_invariants(
    case: tuple[pa.Table, int, int, int],
) -> None:
    data, sample_rows, budget, page_rows = case
    summary = summarise(data, sample_rows=sample_rows, budget_tokens=budget, page_rows=page_rows)

    # never over budget, and never silently so
    assert summary.tokens_estimated <= summary.budget_tokens
    assert estimate_tokens(summary.model_dump_json()) <= summary.budget_tokens
    assert summary.budget_tokens == min(max(budget, MIN_TOKEN_BUDGET), HARD_TOKEN_BUDGET)

    # the sample is a sample
    assert len(summary.sample) <= sample_rows
    assert len(summary.sample) <= summary.rows_total
    assert summary.rows_total == len(summary.sample) + summary.rows_not_shown
    assert summary.truncated == (summary.rows_not_shown > 0)
    assert (summary.rows_not_shown == 0) == summary.is_complete
    assert summary.columns_total == len(data.column_names)
    assert len(summary.columns) + summary.columns_without_stats == summary.columns_total
    if summary.rows_not_shown == 0:
        assert summary.next_call.startswith("nothing to page")
    else:
        assert summary.handle in summary.next_call


@settings(max_examples=60, deadline=None)
@given(
    small=st.integers(min_value=0, max_value=30),
    extra=st.integers(min_value=0, max_value=30),
    sample_rows=st.integers(min_value=0, max_value=8),
    budget=st.integers(min_value=MIN_TOKEN_BUDGET, max_value=HARD_TOKEN_BUDGET),
)
def test_property_a_bigger_input_never_reports_a_smaller_count(
    small: int, extra: int, sample_rows: int, budget: int
) -> None:
    smaller = summarise(table(small), sample_rows=sample_rows, budget_tokens=budget)
    bigger = summarise(table(small + extra), sample_rows=sample_rows, budget_tokens=budget)
    assert bigger.rows_total >= smaller.rows_total
    assert bigger.rows_not_shown >= smaller.rows_not_shown
    assert bigger.tokens_estimated <= bigger.budget_tokens


@settings(max_examples=40, deadline=None)
@given(rows=st.integers(min_value=0, max_value=20), repeat=st.booleans())
def test_property_summarising_twice_is_identical(rows: int, repeat: bool) -> None:
    data = table(rows, columns={"s": ["hé" if repeat else "he"] * rows})
    first = summarise(data, sample_rows=4)
    second = summarise(data, sample_rows=4)
    assert first.model_dump_json() == second.model_dump_json()
