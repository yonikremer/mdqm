"""Result summariser: summary + sample + handle under a token budget (DESIGN section 9).

``summary`` is pure logic over Arrow data: no I/O, no clock, no randomness. The server hands it
the adapter's batch stream and gets back a budget-bounded answer it can send to a weak agent.
"""

from __future__ import annotations

from mdqm.results.summary import (
    DEFAULT_PAGE_ROWS,
    DEFAULT_SAMPLE_ROWS,
    DEFAULT_STAT_COLUMNS,
    DEFAULT_TOKEN_BUDGET,
    DISTINCT_SCAN_ROWS,
    HARD_TOKEN_BUDGET,
    MIN_TOKEN_BUDGET,
    ColumnStats,
    DistinctKind,
    ResultSummary,
    effective_budget,
    estimate_tokens,
    summarise,
)

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
