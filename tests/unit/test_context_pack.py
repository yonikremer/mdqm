"""Tests for the Context Pack (task H05): budgeted, deterministic, and silent when irrelevant.

The pack is what a weak agent actually receives instead of documentation, so three promises are
tested here and nothing else matters as much:

* **It is relevant or it is empty.** ``average credit limit by status`` brings the value map, the
  measure, the approved join and an example spec — in Hebrew too. ``what is the weather`` brings
  nothing: an irrelevant prompt must cost zero tokens.
* **It never exceeds the budget**, for Hebrew and English alike, and a tighter budget drops the
  lowest ranked facts first and says so.
* **It is deterministic**: the same question over the same semantic version produces a
  byte-identical block, because the hook injects it on every prompt and a reviewer must be able to
  diff it.

The Hebrew strings are real Hebrew (with niqqud where the organisation writes it), matching the
fixtures of the search index the pack reads from.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mdqm.context.index import ApprovedExample, ContextIndex
from mdqm.context.pack import (
    DEFAULT_PACK_BUDGET,
    MAX_PACK_BUDGET,
    MIN_PACK_BUDGET,
    OmissionReason,
    PackKind,
    budget_for_context,
    build_pack,
    content_terms,
)
from mdqm.results.summary import estimate_tokens
from mdqm.semantic import SemanticLayer
from mdqm.store import Store

NOW = datetime(2026, 3, 1, 9, 30, tzinfo=UTC)

#: The layer the pack is built from: one measure stored three ways, one categorical value map with
#: Hebrew synonyms, one partitioned event table with a default event-time column and an
#: ingestion-time column, and two approved joins (one used much more than the other, so the pack's
#: "ordered by usage" rule is observable).
SEMANTIC: dict[str, Any] = {
    "version": 7,
    "sources": {
        "crm": {"adapter": "postgresql"},
        "billing": {"adapter": "postgresql"},
        "erp": {"adapter": "postgresql"},
    },
    "value_maps": {
        "customer_status": {
            "canonical": ["active", "churned", "suspended"],
            "synonyms": {
                "active": ["פעיל", "פָּעִיל"],
                "churned": ["עזב"],
                "suspended": ["מושהה"],
            },
            "sources": {
                "crm.customers.status": {"A": "active", "C": "churned", "S": "suspended"},
                "billing.accounts.state": {
                    "Active": "active",
                    "Closed": "churned",
                    "Hold": "suspended",
                },
            },
        }
    },
    "measures": {
        "credit_limit_r5000": {
            "canonical": {"rounding": {"to": 5000, "mode": "half_up"}},
            "sources": {
                "crm.customers.credit_limit": {"normalize": "{col}"},
                "billing.accounts.credit_limit": {"normalize": "ROUND({col} / 5000) * 5000"},
                "erp.customers.limit_k": {"normalize": "{col} * 10"},
            },
        },
        # Two more measures on the same business words: a real layer has many, and the budget cut
        # has to choose between them.
        "credit_limit_total": {
            "sources": {
                "crm.customers.credit_limit": {"normalize": "{col}"},
                "billing.accounts.credit_limit": {"normalize": "ROUND({col} / 5000) * 5000"},
                "erp.customers.limit_k": {"normalize": "{col} * 10"},
            }
        },
        "credit_limit_customer_count": {
            "sources": {
                "crm.customers.credit_limit": {"normalize": "{col}"},
                "billing.accounts.credit_limit": {"normalize": "ROUND({col} / 5000) * 5000"},
                "erp.customers.limit_k": {"normalize": "{col} * 10"},
            }
        },
    },
    "tables": {
        "billing.usage_events": {
            "key": ["event_id"],
            "partitioning": {"key": "start_time", "granularity": "month"},
            "time_columns": {
                "start_time": {"role": "event_time", "indexed": True, "use_by_default": True},
                "insertion_time": {
                    "role": "ingestion_time",
                    "derives_bound_on": "start_time",
                    "lag": {"min": "0s", "max": "6h"},
                },
            },
            "mutability": {"class": "append_rarely", "status": "approved"},
        }
    },
    "approved_joins": [
        {
            "name": "usage_to_customers",
            "left": "billing.usage_events.client_no",
            "right": "crm.customers.cust_id",
            "on": "LPAD(CAST({right} AS VARCHAR), 10, '0') = {left}",
            "cardinality": "many_to_one",
            "usage": {"count_90d": 12, "users_90d": 3},
        },
        {
            "name": "usage_to_accounts",
            "left": "billing.usage_events.client_no",
            "right": "billing.accounts.client_no",
            "on": "CAST({right} AS VARCHAR) = {left}",
            "cardinality": "many_to_one",
            "usage": {"count_90d": 1, "users_90d": 1},
        },
    ],
}

#: The deployment's words for things the layer names in one language only.
LABELS: dict[str, tuple[str, ...]] = {
    "measure:credit_limit_r5000": ("credit limit", "credit limits", "מסגרת אשראי"),
    "table:crm.customers": ("לקוחות", "customers"),
    "table:billing.accounts": ("חשבונות", "accounts"),
    "table:billing.usage_events": ("events", "אירועי שימוש"),
}

EXAMPLES: tuple[ApprovedExample, ...] = (
    ApprovedExample(
        name="avg_credit_limit_by_status_3mo",
        question="מהי מסגרת האשראי הממוצעת לפי סטטוס ב-3 החודשים האחרונים",
        ref="specs/avg_credit_limit_by_status_3mo",
    ),
    ApprovedExample(
        name="credit_limit_by_status",
        question="average credit limit by customer status",
        ref="specs/credit_limit_by_status",
    ),
)

EN_QUESTION = "average credit limit by status"
HE_QUESTION = "מהי מסגרת האשראי לפי סטטוס לקוח פעיל"
#: Questions that must all fit the budget: an English one, a Hebrew one, and two short ones.
QUESTIONS = (EN_QUESTION, HE_QUESTION, "usage events", "לקוחות")


def layer() -> SemanticLayer:
    """The test layer."""
    return SemanticLayer.model_validate(SEMANTIC)


@pytest.fixture()
def store(tmp_path: Path) -> Iterator[Store]:
    """A migrated metadata DB on a throwaway file (SQLite runs in-process, no Docker)."""
    opened = Store(tmp_path / "meta.sqlite")
    opened.initialize()
    try:
        yield opened
    finally:
        opened.close()


def indexed(
    store: Store, *, examples: tuple[ApprovedExample, ...] = EXAMPLES
) -> tuple[ContextIndex, SemanticLayer]:
    """An index built over :func:`layer`, ready to answer pack questions."""
    semantic = layer()
    index = ContextIndex(store, clock=lambda: NOW)
    index.build(semantic, labels=LABELS, examples=examples)
    return index, semantic


def pack(
    store: Store,
    question: str,
    *,
    budget_tokens: int = DEFAULT_PACK_BUDGET,
    examples: tuple[ApprovedExample, ...] = EXAMPLES,
) -> Any:
    """Build the pack for ``question`` over the standard fixture."""
    index, semantic = indexed(store, examples=examples)
    return build_pack(
        question,
        layer=semantic,
        index=index,
        examples=examples,
        budget_tokens=budget_tokens,
    )


# --- relevance: the right facts, or nothing ----------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "what is the weather in Tel Aviv",
        "write me a poem about the sea",
        "מה מזג האוויר בתל אביב",
        "how do I install a python package",
    ],
)
def test_an_unrelated_prompt_gets_an_empty_pack(store: Store, question: str) -> None:
    """A prompt about the weather (or anything else) adds no tokens at all."""
    built = pack(store, question)
    assert built.is_empty
    assert built.text == ""
    assert built.tokens_estimated == 0
    assert built.items == ()


@pytest.mark.parametrize("question", [EN_QUESTION, HE_QUESTION])
def test_credit_limit_question_brings_the_four_facts(store: Store, question: str) -> None:
    """The measure, the value map, the approved join and an example spec — in both languages."""
    built = pack(store, question)
    refs = set(built.refs())
    assert "measure:credit_limit_r5000" in refs, sorted(refs)
    assert "value:customer_status" in refs, sorted(refs)
    assert "join:usage_to_customers" in refs, sorted(refs)
    assert any(ref.startswith("example:") for ref in refs), sorted(refs)
    # The per-source conversions are the whole reason the pack exists: the ERP stores the limit in
    # thousands, so the agent must be told to multiply, not guess.
    measure = built.items_of_kind(PackKind.MEASURE)[0]
    assert "erp.customers.limit_k={col} * 10" in measure.text
    assert "round half_up to 5000" in measure.text
    # And the categorical codes, so "status" is answerable in either source.
    value = built.items_of_kind(PackKind.VALUE)[0]
    assert "crm.customers.status:A" in value.text
    assert "billing.accounts.state:Active" in value.text
    assert "פעיל" in value.text


def test_the_pack_carries_the_default_time_column_and_partition_key(store: Store) -> None:
    """A table arrives with its pruning facts: no query may scan every partition."""
    built = pack(store, "usage events")
    tables = {item.ref: item.text for item in built.items_of_kind(PackKind.TABLE)}
    assert "billing.usage_events" in tables, sorted(tables)
    line = tables["billing.usage_events"]
    assert "time start_time (event_time, use_by_default)" in line
    assert "partition key start_time (range, month)" in line
    # The ingestion-time column and its derived bound travel with it: late rows are handled, not
    # silently dropped.
    assert "time insertion_time (ingestion_time, derives bound on start_time, max lag 6h)" in line


def test_joins_are_ordered_by_usage(store: Store) -> None:
    """Two approved joins match; the one the organisation actually uses comes first."""
    built = pack(store, "usage events customers accounts")
    joins = [item.ref for item in built.items_of_kind(PackKind.JOIN)]
    assert joins[:2] == ["usage_to_customers", "usage_to_accounts"], joins


def test_at_most_three_examples_and_the_rest_are_reported(store: Store) -> None:
    """Examples are capped, and a capped example appears in ``omitted`` with reason ``limit``."""
    many = tuple(
        ApprovedExample(
            name=f"credit_limit_by_status_{index}",
            question="average credit limit by status",
            ref=f"specs/credit_limit_by_status_{index}",
        )
        for index in range(5)
    )
    built = pack(store, EN_QUESTION, examples=many)
    assert len(built.items_of_kind(PackKind.EXAMPLE)) == 3
    limited = [item for item in built.omitted if item.reason is OmissionReason.LIMIT]
    assert len(limited) == 2
    assert {item.kind for item in limited} == {PackKind.EXAMPLE}


# --- the budget -------------------------------------------------------------------------


@given(
    question=st.sampled_from(QUESTIONS),
    budget=st.integers(min_value=MIN_PACK_BUDGET, max_value=DEFAULT_PACK_BUDGET),
)
@settings(max_examples=12, deadline=None)
def test_the_pack_never_exceeds_its_budget(question: str, budget: int) -> None:
    """The rendered block fits the budget, the estimate is honest, and it holds some value."""
    built = pack_for(question, budget)
    assert built.budget_tokens == budget
    assert built.tokens_estimated <= budget
    assert estimate_tokens(built.text) == built.tokens_estimated
    if not built.is_empty:
        assert built.text.startswith("mdqm context")


def pack_for(question: str, budget: int) -> Any:
    """A pack over a throwaway store, for the property test (which may not use a fixture)."""
    with TemporaryDirectory() as directory:
        opened = Store(Path(directory) / "meta.sqlite")
        opened.initialize()
        try:
            return pack(opened, question, budget_tokens=budget)
        finally:
            opened.close()


def test_a_tighter_budget_drops_the_lowest_ranked_first(store: Store) -> None:
    """Same ranking, shorter cut: what survives a tight budget is the head of the loose pack."""
    question = "usage events status credit limit customers accounts"
    loose = pack(store, question, budget_tokens=DEFAULT_PACK_BUDGET)
    tight = pack(store, question, budget_tokens=MIN_PACK_BUDGET)
    assert loose.refs()
    assert tight.tokens_estimated <= MIN_PACK_BUDGET
    assert len(tight.items) < len(loose.items), "the fixture must not fit in the floor budget"
    assert tight.refs() == loose.refs()[: len(tight.items)]
    dropped = [item for item in tight.omitted if item.reason is OmissionReason.BUDGET]
    dropped_refs = tuple(f"{item.kind.value}:{item.ref}" for item in dropped)
    assert dropped_refs == loose.refs()[len(tight.items) :]
    assert "do not fit" in tight.text


def test_a_pack_that_fits_nothing_is_empty_and_says_why(store: Store) -> None:
    """A budget of 1 is clamped to the floor; whatever does not fit is reported, never hidden."""
    built = pack(store, "usage events status credit limit customers accounts", budget_tokens=1)
    assert built.budget_clamped
    assert built.tokens_estimated <= built.budget_tokens
    assert built.items or built.text == ""
    for item in built.omitted:
        assert item.reason in {OmissionReason.BUDGET, OmissionReason.LIMIT}


def test_a_store_without_fts5_builds_the_same_facts(store: Store, tmp_path: Path) -> None:
    """The LIKE fallback ranks nothing (score is null) but must not lose or reorder the facts."""
    semantic = layer()
    ranked = ContextIndex(store, clock=lambda: NOW)
    ranked.build(semantic, labels=LABELS, examples=EXAMPLES)
    plain_store = Store(tmp_path / "like.sqlite")
    plain_store.initialize()
    try:
        plain = ContextIndex(plain_store, clock=lambda: NOW, fts5_available=False)
        plain.build(semantic, labels=LABELS, examples=EXAMPLES)
        with_fts = build_pack(EN_QUESTION, layer=semantic, index=ranked, examples=EXAMPLES)
        without = build_pack(EN_QUESTION, layer=semantic, index=plain, examples=EXAMPLES)
    finally:
        plain_store.close()
    assert {item.ref for item in with_fts.items} == {item.ref for item in without.items}
    assert all(item.score is None for item in without.items)
    assert without.tokens_estimated <= without.budget_tokens


@pytest.mark.parametrize(
    ("requested", "applied", "clamped"),
    [
        (DEFAULT_PACK_BUDGET, DEFAULT_PACK_BUDGET, False),
        (MIN_PACK_BUDGET, MIN_PACK_BUDGET, False),
        (10, MIN_PACK_BUDGET, True),
        (MAX_PACK_BUDGET * 10, MAX_PACK_BUDGET, True),
    ],
)
def test_the_budget_is_clamped_and_says_so(
    store: Store, requested: int, applied: int, clamped: bool
) -> None:
    """A configured budget outside the sane range is clamped, visibly."""
    built = pack(store, EN_QUESTION, budget_tokens=requested)
    assert built.budget_tokens == applied
    assert built.budget_clamped is clamped


def test_a_non_positive_budget_is_refused(store: Store) -> None:
    """A zero budget is a configuration bug, not an empty pack."""
    index, semantic = indexed(store)
    with pytest.raises(ValueError, match="budget_tokens must be positive"):
        build_pack(EN_QUESTION, layer=semantic, index=index, budget_tokens=0)


@pytest.mark.parametrize(
    ("context_tokens", "expected"),
    [
        (None, DEFAULT_PACK_BUDGET),
        (100_000, DEFAULT_PACK_BUDGET),
        (1_000_000, MAX_PACK_BUDGET),
        (8_000, MIN_PACK_BUDGET),
        (10_000, 400),
    ],
)
def test_the_budget_scales_with_the_probed_context_window(
    context_tokens: int | None, expected: int
) -> None:
    """The pack is a fraction of the model's window, clamped into the sane range."""
    applied = min(max(expected, MIN_PACK_BUDGET), MAX_PACK_BUDGET)
    assert budget_for_context(context_tokens) == applied


def test_a_context_window_must_be_positive() -> None:
    """A probed window of zero is a bug in the probe, and it is loud."""
    with pytest.raises(ValueError, match="context_tokens must be positive"):
        budget_for_context(0)


# --- determinism ------------------------------------------------------------------------


@pytest.mark.parametrize("question", [EN_QUESTION, HE_QUESTION, "usage events"])
def test_the_same_question_gives_a_byte_identical_pack(
    store: Store, tmp_path: Path, question: str
) -> None:
    """Two independent indexes over the same semantic version produce the same bytes."""
    first = pack(store, question)
    other = Store(tmp_path / "second.sqlite")
    other.initialize()
    try:
        second = pack(other, question)
    finally:
        other.close()
    assert first.semantic_version == second.semantic_version
    assert first.text == second.text
    assert first.model_dump_json() == second.model_dump_json()


def test_the_pack_survives_a_json_round_trip(store: Store) -> None:
    """The pack is a wire model: what a client dumps validates back to the same value."""
    built = pack(store, EN_QUESTION)
    assert type(built).model_validate_json(built.model_dump_json()) == built


# --- the question words ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("average credit limit by status", ("average", "credit", "limit", "status")),
        (
            "Credit Limits",
            (
                "credit",
                "limits",
            ),
        ),
        ("מהי מסגרת האשראי הממוצעת לפי סטטוס", ("מסגרת", "האשראי", "הממוצעת", "סטטוס")),
        ("?!", ()),
        ("   ", ()),
    ],
)
def test_only_content_words_are_searched(question: str, expected: tuple[str, ...]) -> None:
    """Stopwords and punctuation carry no meaning, so they never become searches."""
    assert content_terms(question) == expected


def test_a_prompt_with_no_words_is_an_empty_pack(store: Store) -> None:
    """The hook runs on every prompt: punctuation alone must not raise or inject anything."""
    built = pack(store, "?! ...")
    assert built.is_empty
    assert built.text == ""


# --- the shipped example: one layer, two languages -----------------------------------------


def test_the_examples_hebrew_question_gets_a_pack(tmp_path: Path) -> None:
    """The gap this closes: an English prompt answered 1100 characters of facts and the same
    question in Hebrew answered nothing, because a measure had no Hebrew name anywhere.
    """
    from mdqm.semantic.loader import load
    from mdqm.server.context_endpoint import ContextPackSource

    root = Path(__file__).resolve().parents[2] / "examples" / "semantic_example.yaml"
    store = Store(tmp_path / "meta.sqlite")
    store.initialize()
    try:
        source = ContextPackSource.build(load(root), store=store)
        hebrew = source.context_for("מסגרת אשראי ממוצעת לפי סטטוס")
        english = source.context_for("average credit limit by customer status")
        unrelated = source.context_for("leaderboard")
    finally:
        store.close()

    assert "credit_limit_r5000" in hebrew.text
    assert "credit_limit_r5000" in english.text
    assert unrelated.text == "", "a prompt that names nothing in the layer injects nothing"
