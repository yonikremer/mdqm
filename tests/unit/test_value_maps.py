"""Tests for the value-map resolver (task C05): labels in both languages, and literals back.

Table-driven where the cases are data — what a user or a weak model actually types, in Hebrew and
in English, with niqqud, geresh/gershayim, stray spaces and the wrong case — and a property test
where the invariant is real: normalisation is idempotent, a canonical value always resolves to
itself, and a literal this module hands back always maps to the canonical value it came from.

The Hebrew strings are real Hebrew. ``פָּעִיל`` with niqqud and ``צה״ל`` with a Hebrew gershayim are
what a data team types, and they must resolve exactly like their plain forms.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mdqm.errors import ErrorCode
from mdqm.semantic import (
    ResolutionKind,
    SemanticLayer,
    SourceLiteral,
    ValueMap,
    load,
    normalize_label,
    resolve_value,
    reverse_value,
)

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "semantic_example.yaml"

#: The three storage forms of one categorical field, as ``examples/semantic_example.yaml`` has
#: them: a single-letter code, a capitalised word and an integer.
STATUS = ValueMap.model_validate(
    {
        "canonical": ["active", "churned", "suspended"],
        "synonyms": {
            "active": ["פעיל", "active"],
            "churned": ["עזב", "churned"],
            "suspended": ["מושעה", "suspended"],
        },
        "sources": {
            "crm.customers.status": {"A": "active", "C": "churned", "S": "suspended"},
            "billing.accounts.state": {
                "Active": "active",
                "Closed": "churned",
                "Hold": "suspended",
            },
            "erp.customers.status_code": {1: "active", 2: "churned", 3: "suspended"},
        },
    }
)

#: A Hebrew label that only works with geresh folding: the map spells it with a Hebrew gershayim.
SEGMENT = ValueMap.model_validate(
    {
        "canonical": ["gov", "private"],
        "synonyms": {"gov": ["צה״ל", "IDF"], "private": ["פרטי"]},
        "sources": {"crm.customers.segment": {"G": "gov", "P": "private"}},
    }
)

#: A label that is one canonical value and also a synonym of another. The loader cannot reject it
#: (it only compares synonyms with synonyms), so the resolver must report it instead of picking.
OVERLAP = ValueMap.model_validate(
    {
        "canonical": ["active", "open"],
        "synonyms": {"active": ["open"]},
    }
)

#: One source code, two meanings: legal in a layer, fatal as a silent guess.
COLLIDING_CODES = ValueMap.model_validate(
    {
        "canonical": ["active", "churned"],
        "sources": {
            "crm.customers.status": {"A": "active"},
            "erp.customers.status_code": {"A": "churned"},
        },
    }
)

#: A zero-padded code is an identifier, not the number 1: the literal must stay text.
PADDED = ValueMap.model_validate(
    {
        "canonical": ["active"],
        "sources": {"billing.accounts.legacy_code": {"01": "active"}},
    }
)


@pytest.fixture(scope="module")
def example_layer() -> SemanticLayer:
    return load(EXAMPLE)


# --- forward: any spelling a human or a model writes ------------------------------------

LABEL_CASES: dict[str, tuple[str | int, str]] = {
    "hebrew": ("פעיל", "active"),
    "hebrew with niqqud": ("פָּעִיל", "active"),
    "hebrew suspended": ("מושעה", "suspended"),
    "english lower": ("active", "active"),
    "english padded mixed case": (" Active ", "active"),
    "english upper": ("ACTIVE", "active"),
    "english collapsed whitespace": ("  churned\n", "churned"),
    "canonical itself": ("active", "active"),
    "crm code": ("A", "active"),
    "crm code lower case": ("a", "active"),
    "billing code": ("Hold", "suspended"),
    "billing code lower case": ("hold", "suspended"),
    "erp integer code": (1, "active"),
    "erp integer code as text": ("3", "suspended"),
}


@pytest.mark.parametrize("case", LABEL_CASES, ids=list(LABEL_CASES))
def test_labels_resolve_in_both_languages(case: str) -> None:
    label, expected = LABEL_CASES[case]
    result = resolve_value(STATUS, label, field="filters.0.value")
    assert result.ok, (case, result.error)
    assert result.canonical == expected
    assert result.kind is ResolutionKind.RESOLVED
    assert result.error is None


NORMALISATION_VARIANTS: dict[str, tuple[ValueMap, str, str]] = {
    "gershayim typographic quote": (SEGMENT, 'צה"ל', "gov"),
    "gershayim hebrew mark": (SEGMENT, "צה״ל", "gov"),
    # U+2033 DOUBLE PRIME is deliberately absent: NFKC expands it into two U+2032 PRIMEs before the
    # geresh/gershayim folding, so it cannot match. Reported as a gap in semantic/text.py.
    "niqqud stripped": (STATUS, "פָּעִיל", "active"),
    "niqqud and padding": (STATUS, " פָּעִיל ", "active"),
}


@pytest.mark.parametrize("case", NORMALISATION_VARIANTS, ids=list(NORMALISATION_VARIANTS))
def test_niqqud_and_gershayim_variants_resolve_like_the_plain_form(case: str) -> None:
    value_map, label, expected = NORMALISATION_VARIANTS[case]
    assert resolve_value(value_map, label).canonical == expected


def test_example_layer_resolves_the_documented_values(example_layer: SemanticLayer) -> None:
    """The shipped example's map answers the same way through the resolver and the model."""
    status = example_layer.value_maps["customer_status"]
    for label, expected in (
        ("פעיל", "active"),
        (" Active ", "active"),
        ("active", "active"),
        ("A", "active"),
        ("Hold", "suspended"),
    ):
        assert resolve_value(status, label).canonical == expected


# --- ambiguity is reported, never guessed ----------------------------------------------


def test_a_label_that_is_a_canonical_value_and_a_synonym_is_ambiguous() -> None:
    result = resolve_value(OVERLAP, "open", field="filters.0.value")
    assert result.kind is ResolutionKind.AMBIGUOUS
    assert result.canonical is None
    assert result.candidates == ("active", "open")
    error = result.error
    assert error is not None
    assert error.code == ErrorCode.AMBIGUOUS_VALUE.value
    assert error.valid_values == ("active", "open")


def test_one_code_in_two_sources_is_ambiguous() -> None:
    result = resolve_value(COLLIDING_CODES, "A", field="filters.0.value")
    assert result.kind is ResolutionKind.AMBIGUOUS
    assert result.candidates == ("active", "churned")
    assert result.error is not None
    assert result.error.code == ErrorCode.AMBIGUOUS_VALUE.value


def test_naming_the_column_resolves_a_scoped_code() -> None:
    """The compiler knows which source it compiles for, so it may say so and get an answer."""
    scoped = resolve_value(COLLIDING_CODES, "A", column="erp.customers.status_code")
    assert scoped.ok
    assert scoped.canonical == "churned"


def test_ambiguous_error_lists_the_matched_values_not_every_value() -> None:
    error = resolve_value(COLLIDING_CODES, "a").error
    assert error is not None
    assert error.valid_values == ("active", "churned")
    assert "matches several canonical values" in error.problem


# --- unknown is reported with the valid set and the nearest spellings -------------------

UNKNOWN_CASES: dict[str, tuple[ValueMap, str | int, tuple[str, ...]]] = {
    "english typo": (STATUS, "avtive", ("active",)),
    "hebrew not in the map": (STATUS, "לא קיים", ()),
    "empty": (STATUS, "", ()),
    "whitespace only": (STATUS, "   ", ()),
    "code not listed": (STATUS, "Z", ()),
    "number not listed": (STATUS, 99, ()),
}


@pytest.mark.parametrize("case", UNKNOWN_CASES, ids=list(UNKNOWN_CASES))
def test_unknown_values_are_reported_with_the_valid_set(case: str) -> None:
    value_map, label, expected_suggestions = UNKNOWN_CASES[case]
    result = resolve_value(value_map, label, field="filters.0.value")
    assert result.kind is ResolutionKind.UNKNOWN
    assert result.candidates == value_map.canonical
    assert result.did_you_mean == expected_suggestions
    error = result.error
    assert error is not None
    assert error.code == ErrorCode.UNKNOWN_VALUE.value
    assert error.valid_values == value_map.canonical
    assert (error.did_you_mean or ()) == expected_suggestions


def test_unknown_value_fix_mentions_both_languages() -> None:
    """The fix tells a weak model that Hebrew works, because that is the mistake it makes."""
    error = resolve_value(STATUS, "nope").error
    assert error is not None
    assert "Hebrew and English" in error.fix


def test_a_code_from_another_column_is_unknown_when_the_column_is_named() -> None:
    result = resolve_value(STATUS, "Hold", column="crm.customers.status")
    assert result.kind is ResolutionKind.UNKNOWN
    assert result.candidates == STATUS.canonical


# --- reverse: canonical (or any label) -> the literal each source stores ----------------

EXPECTED_ACTIVE_LITERALS = (
    SourceLiteral(column="billing.accounts.state", literal="Active"),
    SourceLiteral(column="crm.customers.status", literal="A"),
    SourceLiteral(column="erp.customers.status_code", literal=1),
)


def test_reverse_maps_to_every_source_in_a_deterministic_order() -> None:
    result = reverse_value(STATUS, "active")
    assert result.ok
    assert result.canonical == "active"
    assert result.literals == EXPECTED_ACTIVE_LITERALS
    assert result.literals == reverse_value(STATUS, "active").literals


def test_reverse_keeps_the_type_each_source_expects() -> None:
    kinds = {
        literal.column: (literal.literal, literal.is_integer)
        for literal in reverse_value(STATUS, "suspended").literals
    }
    assert kinds == {
        "billing.accounts.state": ("Hold", False),
        "crm.customers.status": ("S", False),
        "erp.customers.status_code": (3, True),
    }


REVERSE_LABEL_CASES: dict[str, str] = {
    "canonical": "active",
    "english synonym upper": "ACTIVE",
    "hebrew synonym with niqqud": "פָּעִיל",
    "crm code": "A",
    "billing code": "Active",
    "erp integer code": "1",
}


@pytest.mark.parametrize("case", REVERSE_LABEL_CASES, ids=list(REVERSE_LABEL_CASES))
def test_any_spelling_reverses_to_the_same_literals(case: str) -> None:
    result = reverse_value(STATUS, REVERSE_LABEL_CASES[case], field="filters.0.value")
    assert result.error is None
    assert result.canonical == "active"
    assert result.literals == EXPECTED_ACTIVE_LITERALS


def test_reverse_restricted_to_one_column() -> None:
    result = reverse_value(STATUS, "churned", column="crm.customers.status")
    assert result.literals == (SourceLiteral(column="crm.customers.status", literal="C"),)


def test_reverse_of_a_zero_padded_code_stays_text() -> None:
    """``"01"`` is an identifier: turning it into ``1`` would filter the wrong rows."""
    result = reverse_value(PADDED, "active")
    assert result.literals == (SourceLiteral(column="billing.accounts.legacy_code", literal="01"),)
    assert result.literals[0].is_integer is False


def test_reverse_of_an_unknown_label_carries_the_forward_error() -> None:
    result = reverse_value(STATUS, "avtive", field="filters.0.value")
    assert not result.ok
    assert result.canonical is None
    assert result.literals == ()
    assert result.error is not None
    assert result.error.code == ErrorCode.UNKNOWN_VALUE.value
    assert result.error.did_you_mean == ("active",)


def test_reverse_of_an_ambiguous_label_does_not_pick_a_source() -> None:
    result = reverse_value(COLLIDING_CODES, "A", field="filters.0.value")
    assert not result.ok
    assert result.literals == ()
    assert result.error is not None
    assert result.error.code == ErrorCode.AMBIGUOUS_VALUE.value


def test_reverse_of_a_value_no_source_lists_is_valid_but_empty() -> None:
    """The value exists in the layer and simply is not stored in that source: not a bad filter."""
    result = reverse_value(PADDED, "active", column="crm.customers.status")
    assert result.error is None
    assert result.canonical == "active"
    assert result.literals == ()


# --- properties ------------------------------------------------------------------------

#: Every character class normalisation touches, so hypothesis writes the inputs a user writes.
HEBREW_LABELS = st.text(
    alphabet="אבגדהוזחטיכלמנסעפצקרשת ׳״'\"\u05b0\u05b4\u05b8\u05bc\ufb01 ",
    max_size=10,
)


@given(HEBREW_LABELS)
def test_normalisation_is_idempotent(label: str) -> None:
    once = normalize_label(label)
    assert normalize_label(once) == once


@given(st.sampled_from(STATUS.canonical))
def test_a_canonical_value_always_resolves_to_itself(canonical: str) -> None:
    for spelling in (canonical, canonical.upper(), f" {canonical} ", f"\t{canonical}\n"):
        result = resolve_value(STATUS, spelling)
        assert result.ok
        assert result.canonical == canonical


@given(st.sampled_from([*STATUS.canonical, "פעיל", "עזב", "מושעה", "A", "a", "Hold", "3", 2]))
def test_every_resolved_label_round_trips_through_the_source_codes(label: str | int) -> None:
    """The invariant that matters: a literal handed to a driver maps back to where it came from."""
    result = reverse_value(STATUS, label)
    assert result.ok
    assert result.canonical is not None
    assert result.literals, "a resolved value that a source stores must produce a literal"
    for literal in result.literals:
        assert STATUS.canonical_for(literal.column, literal.literal) == result.canonical


@given(HEBREW_LABELS)
def test_a_resolution_never_invents_a_value(label: str) -> None:
    result = resolve_value(STATUS, label)
    if result.ok:
        assert result.canonical in STATUS.canonical
        assert result.candidates == ()
    else:
        assert result.candidates == STATUS.canonical
        assert result.error is not None
