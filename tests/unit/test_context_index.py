"""Tests for the context index (task H04): Hebrew and English, synonyms, and the honest fallback.

What is under test is one promise: a weak agent that types a business word — in Hebrew or English,
with niqqud, with a Hebrew definite article, as a prefix — is handed the right measure, table,
value or join. The Hebrew strings below are real Hebrew, not transliterations: ``פָּעִיל`` with
niqqud, ``מִסְגֶּרֶת`` with niqqud, and the plain spellings a data team actually uses.

Two search paths exist and both are tested. FTS5 is present in the development interpreter, so the
ranked path is what normally runs; the ``LIKE`` fallback is driven by injecting the capability
probe's verdict (``fts5_available=False``), which is exactly the situation of a Python build whose
SQLite lacks FTS5.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from mdqm.context.index import (
    DOC_TABLE,
    EMPTY_SEARCH_TABLE,
    FTS_TABLE,
    ApprovedExample,
    ContextHit,
    ContextIndex,
    ContextIndexError,
    DocKind,
    IndexReport,
    expand_terms,
    prefix_variants,
)
from mdqm.models import CatalogSnapshot
from mdqm.semantic import SemanticLayer
from mdqm.semantic.text import normalize_label
from mdqm.store import Store, check_fts5

NOW = datetime(2026, 3, 1, 9, 30, tzinfo=UTC)
FTS5_HERE = check_fts5().available

#: The semantic layer as the tests need it: one measure stored three ways, one categorical value map
#: with Hebrew synonyms (two of them with niqqud), one partitioned table and one approved join. The
#: shape follows ``examples/semantic_example.yaml``; it is inline so a change there cannot silently
#: change what these tests assert.
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
                "active": ["פעיל", "active", "פָּעִיל"],
                "churned": ["עזב", "churned"],
                "suspended": ["מושהה", "suspended"],
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
        }
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
        }
    ],
}

#: The deployment's own words for things the layer does not name in both languages. This is where
#: "credit limit" and "מסגרת אשראי" come from; both spellings of the Hebrew phrase are supplied
#: because a data team writes it both ways.
LABELS: dict[str, tuple[str, ...]] = {
    "measure:credit_limit_r5000": ("credit limit", "מסגרת אשראי", "מִסְגֶּרֶת אַשְׁרַאי"),
    "table:crm.customers": ("לקוחות", "customers"),
    "table:billing.accounts": ("חשבונות", "accounts"),
}

CATALOG = CatalogSnapshot.model_validate(
    {
        "source": "crm",
        "captured_at": "2026-01-01T00:00:00Z",
        "tables": [
            {
                "name": "customers",
                "row_estimate": 1200,
                "comment": "רייקרד לקוחות",
                "columns": [
                    {"name": "cust_id", "native_type": "int4", "normalized_type": "integer"},
                    {
                        "name": "status",
                        "native_type": "char(1)",
                        "normalized_type": "string",
                        "comment": "מצב הלקוח",
                    },
                ],
            }
        ],
    }
)

EXAMPLES = (
    ApprovedExample(
        name="avg_credit_limit_by_status_3mo",
        question="מהי מסגרת האשראי הממוצעת לפי סטטוס ב-3 החודשים האחרונים",
        ref="specs/avg_credit_limit_by_status_3mo",
    ),
)


def layer(**overrides: Any) -> SemanticLayer:
    """The test layer, optionally with one top-level key replaced (for the rebuild tests)."""
    payload: dict[str, Any] = {**SEMANTIC, **overrides}
    return SemanticLayer.model_validate(payload)


@pytest.fixture()
def store(tmp_path: Path) -> Iterator[Store]:
    """A migrated metadata DB on a throwaway file; the migrations are the shipped ones."""
    opened = Store(tmp_path / "meta.sqlite")
    opened.initialize()
    try:
        yield opened
    finally:
        opened.close()


def index(store: Store, **kwargs: Any) -> ContextIndex:
    """An index with a fixed clock, so a logged search has a predictable timestamp."""
    kwargs.setdefault("clock", lambda: NOW)
    return ContextIndex(store, **kwargs)


def build(
    store: Store,
    *,
    semantic: SemanticLayer | None = None,
    labels: dict[str, tuple[str, ...]] | None = LABELS,
    catalogs: tuple[CatalogSnapshot, ...] = (CATALOG,),
    examples: tuple[ApprovedExample, ...] = EXAMPLES,
    **kwargs: Any,
) -> IndexReport:
    """The common case: index the test layer and report what happened."""
    return index(store, **kwargs).build(
        semantic if semantic is not None else layer(),
        catalogs=catalogs,
        labels=labels,
        examples=examples,
    )


def refs(hits: tuple[ContextHit, ...]) -> set[str]:
    """The documents a search returned, as ``<kind>:<ref>`` ids."""
    return {hit.doc_id for hit in hits}


def search(store: Store, text: str, **kwargs: Any) -> tuple[ContextHit, ...]:
    """Search with the default index (logging left on, as the server has it)."""
    return index(store).search(text, **kwargs)


def table_names(store: Store) -> set[str]:
    rows = store.connect().execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    return {str(row[0]) for row in rows}


# --- the acceptance criteria: Hebrew and English find credit_limit ---------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("credit limit", "measure:credit_limit_r5000"),
        ("Credit Limit", "measure:credit_limit_r5000"),
        ("credit_limit", "measure:credit_limit_r5000"),
        ("credi", "measure:credit_limit_r5000"),  # prefix search
        ("מסגרת אשראי", "measure:credit_limit_r5000"),
        ("מִסְגֶּרֶת אַשְׁרַאי", "measure:credit_limit_r5000"),  # with niqqud
        ("מסגרת", "measure:credit_limit_r5000"),  # one word of the phrase
        ("אשראי", "measure:credit_limit_r5000"),  # the other word
    ],
)
def test_credit_limit_is_found_in_both_languages(store: Store, query: str, expected: str) -> None:
    """The Hebrew phrase for 'credit limit' and the English one both find the measure."""
    build(store)
    found = refs(search(store, query))
    assert expected in found, f"{query!r} did not find {expected}: got {sorted(found)}"
    # And the table that stores it, because the measure's Hebrew name is a business name of its
    # columns: a weak agent that asked for "credit limit" must be able to write `from customers`.
    assert "table:crm.customers" in found, sorted(found)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("customers", "table:crm.customers"),
        ("customer", "table:crm.customers"),  # prefix of the plural
        ("cust", "table:crm.customers"),  # a half-typed word
        ("לקוחות", "table:crm.customers"),
        ("הלקוחות", "table:crm.customers"),  # the definite article: ה + לקוחות
        ("ללקוחות", "table:crm.customers"),  # the preposition: ל + לקוחות
        ("חשבונות", "table:billing.accounts"),
        ("החשבונות", "table:billing.accounts"),  # ה + חשבונות
    ],
)
def test_a_hebrew_prefix_form_finds_the_same_table(store: Store, query: str, expected: str) -> None:
    """'the customers' in Hebrew finds 'customer' through the prefix variants, not by luck."""
    build(store)
    found = refs(search(store, query))
    assert expected in found, f"{query!r} did not find {expected}: got {sorted(found)}"


def test_every_indexed_entity_kind_is_present(store: Store) -> None:
    """The task's list: tables, columns, measures, values, approved joins, approved examples."""
    build(store)
    kinds = {
        str(row[0])
        for row in store.connect().execute(f"SELECT DISTINCT kind FROM {DOC_TABLE}")
    }
    assert kinds == {kind.value for kind in DocKind}


def test_a_value_is_found_by_any_of_its_spellings(store: Store) -> None:
    """Normalisation is what makes niqqud, geresh and case irrelevant — on both sides."""
    build(store)
    spellings = ["active", "ACTIVE", "פעיל", "פָּעִיל", "  active  "]
    results = {spelling: refs(search(store, spelling)) for spelling in spellings}
    baseline = results["active"]
    assert "value:customer_status:active" in baseline, sorted(baseline)
    for spelling, found in results.items():
        assert found == baseline, f"{spelling!r} returned {sorted(found)} not {sorted(baseline)}"


def test_the_english_and_hebrew_phrases_agree_on_the_documents(store: Store) -> None:
    """Two spellings of one business word are two ways to ask one question, not two answers."""
    build(store)
    hebrew = refs(search(store, "מסגרת אשראי"))
    english = refs(search(store, "credit limit"))
    assert hebrew == english, f"he={sorted(hebrew)} en={sorted(english)}"


# --- the honest miss: an unanswered question is data ----------------------------------------


def test_a_known_miss_is_logged_to_empty_searches(store: Store) -> None:
    """A question the layer has no word for is logged for the data team, normalised."""
    build(store)
    missed = "תחזית מזג האוויר להיום"
    assert search(store, missed) == ()
    logged = index(store).unmatched()
    assert len(logged) == 1
    entry = logged[0]
    assert entry.query == normalize_label(missed)
    assert entry.searched_at == NOW
    assert entry.documents > 0, "the log must say what the index held when it missed"
    # A search that did answer is not a gap in the semantic layer.
    assert search(store, "credit limit")
    assert len(index(store).unmatched()) == 1


def test_an_empty_search_can_be_kept_out_of_the_log(store: Store) -> None:
    """A caller that is only probing the index (a lint, a test) can say so."""
    build(store)
    assert search(store, "zzz", log_empty=False) == ()
    rows = store.connect().execute(f"SELECT COUNT(*) FROM {EMPTY_SEARCH_TABLE}").fetchone()
    assert int(rows[0]) == 0


# --- incremental rebuild --------------------------------------------------------------------


def test_rebuild_with_an_unchanged_version_hash_does_no_work(store: Store) -> None:
    """The same semantics twice: the second build writes nothing and touches no row."""
    first = build(store)
    assert first.rebuilt and first.written == first.documents > 0
    conn = store.connect()
    before = [tuple(row) for row in conn.execute(f"SELECT id, doc_id FROM {DOC_TABLE} ORDER BY id")]
    stamped = conn.execute("SELECT indexed_at FROM context_index_state WHERE id = 1").fetchone()

    second = build(store)

    after = [tuple(row) for row in conn.execute(f"SELECT id, doc_id FROM {DOC_TABLE} ORDER BY id")]
    assert second.already_current and not second.rebuilt
    assert second.written == 0
    assert second.version_hash == first.version_hash
    assert after == before, "an unchanged version_hash must not rewrite the documents"
    assert (
        conn.execute("SELECT indexed_at FROM context_index_state WHERE id = 1").fetchone()[0]
        == stamped[0]
    )


def test_a_changed_semantic_layer_is_reindexed(store: Store) -> None:
    """A new fact changes version_hash, and the last word about a value is the new one."""
    build(store)
    changed = layer(version=8, value_maps={})
    report = build(store, semantic=changed)
    assert report.rebuilt and report.written == report.documents
    assert report.version_hash != layer().version_hash
    assert search(store, "פעיל") == (), "the removed value map must not still be indexed"


def test_search_reports_that_the_fallback_ran(store: Store) -> None:
    """With FTS5 reported unavailable, hits carry no score, and no virtual table is created."""
    report = build(store, fts5_available=False)
    assert report.rebuilt
    assert FTS_TABLE not in table_names(store)
    hits = index(store, fts5_available=False).search("מסגרת אשראי")
    assert hits, "the LIKE fallback must answer the same question"
    assert all(hit.score is None for hit in hits)
    assert "measure:credit_limit_r5000" in refs(hits)


@pytest.mark.parametrize(
    "query",
    ["credit limit", "מסגרת אשראי", "מִסְגֶּרֶת אַשְׁרַאי", "הלקוחות", "פעיל"],
)
def test_the_fallback_and_the_fts_path_find_the_same_documents(store: Store, query: str) -> None:
    """Degrading the search engine must not silently change the answer."""
    build(store)
    ranked = refs(index(store).search(query))
    fallback = refs(index(store, fts5_available=False).search(query))
    assert ranked == fallback, f"{query!r}: fts={sorted(ranked)} like={sorted(fallback)}"


@pytest.mark.skipif(not FTS5_HERE, reason="this Python build has no FTS5")
def test_the_fts_path_ranks_the_measure_above_a_prose_mention(store: Store) -> None:
    """bm25 weights the title above the body, and equal scores keep a deterministic order."""
    build(store)
    hits = index(store).search("credit limit")
    assert FTS_TABLE in table_names(store)
    assert hits[0].doc_id == "measure:credit_limit_r5000", [hit.doc_id for hit in hits]
    assert hits[0].score is not None and hits[0].score < 0
    assert index(store).search("credit limit") == hits, "the same question must not re-order"


# --- pure query building: variants and synonyms ---------------------------------------------


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("הלקוחות", ("הלקוחות", "לקוחות")),
        ("ללקוחות", ("ללקוחות", "לקוחות")),
        ("customers", ("customers",)),
        ("", ("",)),
        ("של", ("של",)),  # two letters: stripping would leave a syllable, not a word
        ("שלום", ("שלום", "לום")),  # "ש" here is part of the word, so both are searched
        ("אשראי", ("אשראי",)),  # א is not a prefix
    ],
)
def test_prefix_variants_never_lose_the_original(term: str, expected: tuple[str, ...]) -> None:
    """Prefix stripping adds a variant; it never replaces what the user typed."""
    assert prefix_variants(term) == expected


def test_expand_terms_ors_each_word_and_ands_nothing_away() -> None:
    """Each query word becomes one OR-group; a synonym group joins it, it does not replace it."""
    groups = (("active", "פעיל"), ("customers", "לקוחות"))
    assert expand_terms(["פעיל", "customers"], groups) == (
        ("פעיל", "active"),
        ("customers", "לקוחות"),
    )
    assert expand_terms(["הלקוחות"], groups) == (("הלקוחות", "לקוחות"),)


# --- refusals: a wrong question is answered with a fix, never with a guess -------------------


@pytest.mark.parametrize("text", ["", "   ", "?!\u200f"])
def test_a_search_without_a_word_is_refused(store: Store, text: str) -> None:
    """Nothing to look for is not "no results" — it is a bug in the caller."""
    build(store)
    with pytest.raises(ContextIndexError, match="at least one word"):
        search(store, text)


def test_a_non_positive_limit_is_refused(store: Store) -> None:
    build(store)
    with pytest.raises(ContextIndexError, match="limit must be at least 1"):
        search(store, "credit limit", limit=0)


def test_a_label_key_that_names_nothing_is_refused(store: Store) -> None:
    """An ignored label is a search that mysteriously misses, so it fails at build time."""
    with pytest.raises(ContextIndexError, match="unknown label key"):
        build(store, labels={"measure:nope": ("nonsense",)})


def test_searching_a_store_that_was_never_migrated_names_the_fix(tmp_path: Path) -> None:
    """The error an operator sees must say what to do, not only what failed."""
    unopened = Store(tmp_path / "raw.sqlite")
    try:
        with pytest.raises(ContextIndexError, match="Store.initialize"):
            index(unopened).search("credit limit")
    finally:
        unopened.close()


@pytest.mark.parametrize(
    "weights", [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0, -1.0), (1.0, 1.0, 1.0, 1e309)]
)
def test_unusable_weights_are_refused(store: Store, weights: tuple[float, ...]) -> None:
    """The weights are rendered into SQL, so only validated finite numbers reach it."""
    with pytest.raises(ContextIndexError, match="weights"):
        ContextIndex(store, weights=weights)  # type: ignore[arg-type]


def test_empty_searches_are_read_newest_first(store: Store) -> None:
    """The data team reads the queue newest first, and each row says what was searched."""
    build(store)
    search(store, "the first miss")
    search(store, "the second miss")
    logged = index(store).unmatched()
    assert [entry.query for entry in logged] == ["the second miss", "the first miss"]
    assert all(entry.user_id is None for entry in logged)
    assert index(store).unmatched(limit=1)[0].query == "the second miss"
