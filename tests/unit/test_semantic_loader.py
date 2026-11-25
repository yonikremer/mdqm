"""Tests for ``mdqm.semantic`` (task C04): models, merge, errors, version hash, writer.

Table-driven where the cases are data (labels, durations, broken documents, mutability classes) and
a real document where the case is the whole flow. Hebrew strings are real Hebrew: niqqud, geresh
and gershayim are what a data team actually types.
"""

from __future__ import annotations

import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mdqm.models.partitioning import PartitionGranularity, PartitionStrategy
from mdqm.semantic import (
    JoinCardinality,
    SemanticError,
    SemanticLayer,
    check_sql_template,
    compute_version_hash,
    load,
    normalize_label,
    parse_duration,
    render_template,
    update_curated,
    write_generated,
)
from mdqm.semantic.writer import dump

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "semantic_example.yaml"
OVERLAY = Path(__file__).resolve().parents[1] / "data" / "semantic" / "overlay"
BASE_TEXT = (OVERLAY / "generated" / "base.yaml").read_text(encoding="utf-8")


def write_layer(tmp_path: Path, text: str, name: str = "layer.yaml") -> Path:
    """Write a semantic document with LF endings and return its path."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def load_text(tmp_path: Path, text: str, name: str = "layer.yaml") -> SemanticLayer:
    return load(write_layer(tmp_path, text, name))


# --- the worked example ------------------------------------------------------------------


def test_example_yaml_loads() -> None:
    layer = load(EXAMPLE)

    assert layer.version == 17
    assert set(layer.sources) == {"crm", "billing", "erp", "api_fx"}
    assert layer.sources["crm"].load.max_concurrency == 3
    assert layer.sources["api_fx"].rate_limit is not None
    assert layer.sources["api_fx"].rate_limit.rps == 2
    assert layer.sources["api_fx"].auth is not None
    assert layer.sources["api_fx"].auth.header == "X-API-Key"

    assert layer.defaults.cache.staleness_ratio == 0.01
    assert layer.defaults.cache.min_ttl == timedelta(seconds=30)
    assert layer.defaults.cache.max_ttl == timedelta(days=7)
    assert [window.tz for window in layer.defaults.quiet_windows] == ["Asia/Jerusalem"]

    status = layer.value_maps["customer_status"]
    assert status.canonical == ("active", "churned", "suspended")
    assert status.canonical_for("crm.customers.status", "A") == "active"
    assert status.canonical_for("erp.customers.status_code", 3) == "suspended"

    measure = layer.measures["credit_limit_r5000"]
    assert measure.canonical.rounding is not None
    assert measure.canonical.rounding.to == 5000
    assert measure.provenance is not None
    assert measure.provenance.detected_by == "pipeline/D8"
    assert measure.provenance.match_rate == 0.998
    assert measure.provenance.approved_by == "data-team"
    assert measure.normalize_for("erp.customers.limit_k", "limit_k") == "limit_k * 10"
    assert (
        measure.normalize_for("billing.accounts.credit_limit", "credit_limit")
        == "ROUND(credit_limit / 5000) * 5000"
    )
    assert measure.normalize_for("crm.customers.credit_limit", "credit_limit") == "credit_limit"
    assert measure.normalize_for("crm.customers.other", "other") is None
    assert not measure.is_identity_everywhere

    usage = layer.tables["billing.usage_events"]
    assert usage.key == ("event_id",)
    assert usage.default_window == timedelta(days=7)
    assert usage.partitioning is not None
    assert usage.partitioning.strategy is PartitionStrategy.RANGE
    assert usage.partitioning.key == ("start_time",)
    assert usage.partitioning.granularity is PartitionGranularity.MONTH
    assert usage.partitioning.require_key_filter is True
    assert usage.time_columns["insertion_time"].derives_bound_on == "start_time"
    assert usage.time_columns["insertion_time"].lag is not None
    assert usage.time_columns["insertion_time"].lag.maximum == timedelta(hours=6)
    assert usage.may_cache is True

    assert layer.tables["billing.accounts_hist"].may_cache is False

    join = layer.approved_joins[0]
    assert join.name == "usage_to_customers"
    assert join.cardinality is JoinCardinality.MANY_TO_ONE
    assert join.usage is not None
    assert join.usage.count_90d == 1840
    assert join.usage.users_90d == 37
    assert join.on == "LPAD(CAST({right} AS VARCHAR), 10, '0') = {left}"

    assert len(layer.version_hash) == 64
    assert compute_version_hash(layer) == layer.version_hash


def test_a_document_of_comments_only_loads_as_an_empty_layer(tmp_path: Path) -> None:
    layer = load_text(tmp_path, "# nothing yet\n")
    assert layer.tables == {}
    assert len(layer.version_hash) == 64


# --- immutability ------------------------------------------------------------------------


def test_a_loaded_layer_cannot_be_mutated() -> None:
    layer = load(EXAMPLE)
    with pytest.raises(ValidationError):
        layer.version = 18  # type: ignore[misc]
    with pytest.raises(TypeError):
        layer.tables["billing.usage_events"] = layer.tables["billing.accounts_hist"]
    with pytest.raises(TypeError):
        layer.tables.clear()
    with pytest.raises(TypeError):
        layer.value_maps["customer_status"].synonyms.update({"active": ("פעיל",)})
    with pytest.raises(TypeError):
        layer.value_maps["customer_status"].sources["crm.customers.status"]["A"] = "churned"
    assert isinstance(layer.approved_joins, tuple)


# --- curated overrides generated ---------------------------------------------------------


def test_curated_wins_per_key_and_generated_facts_survive() -> None:
    layer = load(OVERLAY)

    table = layer.tables["crm.customers"]
    assert table.mutability is not None
    assert table.mutability.class_ == "mutable_critical"
    # everything the curated file does not mention still comes from generated/
    assert table.key == ("cust_id",)
    assert table.default_window == timedelta(days=30)
    assert table.partitioning is not None
    assert table.partitioning.granularity is PartitionGranularity.MONTH
    assert set(table.time_columns) == {"created_at"}

    status = layer.value_maps["customer_status"]
    assert status.canonical_for("erp.customers.status_code", 2) == "churned"
    assert status.canonical_for("crm.customers.status", "A") == "active"
    assert status.resolve("פעיל") == "active"

    joins = {join.name: join for join in layer.approved_joins}
    assert list(joins) == ["usage_to_customers", "hist_to_customers"]
    assert joins["usage_to_customers"].cardinality is JoinCardinality.ONE_TO_MANY
    assert joins["usage_to_customers"].left == "billing.usage_events.client_no"
    assert joins["usage_to_customers"].usage is not None
    assert joins["usage_to_customers"].usage.count_90d == 12

    assert layer.tables["billing.accounts_hist"].may_cache is False


def test_removing_a_curated_key_restores_the_generated_value(tmp_path: Path) -> None:
    root = tmp_path / "semantic"
    shutil.copytree(OVERLAY, root)
    curated = root / "curated" / "human.yaml"
    text = curated.read_text(encoding="utf-8")
    override = (
        "    # the nightly job rewrites rows in place, so a cached answer can be wrong\n"
        "    mutability: { class: mutable_critical, status: approved }\n"
    )
    assert override in text
    curated.write_text(text.replace(override, ""), encoding="utf-8", newline="\n")

    layer = load(root)
    table = layer.tables["crm.customers"]
    assert table.mutability is not None
    assert table.mutability.class_ == "append_rarely"
    # the other curated key is untouched
    assert layer.value_maps["customer_status"].canonical_for("erp.customers.status_code", 1) == (
        "active"
    )


def test_removing_the_curated_file_restores_every_generated_value(tmp_path: Path) -> None:
    root = tmp_path / "semantic"
    shutil.copytree(OVERLAY, root)
    (root / "curated" / "human.yaml").unlink()

    layer = load(root)
    table = layer.tables["crm.customers"]
    assert table.mutability is not None
    assert table.mutability.class_ == "append_rarely"
    assert layer.value_maps["customer_status"].canonical_for("erp.customers.status_code", 1) is None
    assert [join.name for join in layer.approved_joins] == ["usage_to_customers"]


def test_a_curated_list_replaces_the_generated_list(tmp_path: Path) -> None:
    root = tmp_path / "semantic"
    (root / "generated").mkdir(parents=True)
    (root / "curated").mkdir()
    write_layer(
        root / "generated",
        'defaults:\n  quiet_windows:\n    - { tz: UTC, start: "02:00", end: "03:00" }\n',
        name="a.yaml",
    )
    write_layer(
        root / "curated",
        'defaults:\n  quiet_windows:\n    - { tz: Asia/Jerusalem, start: "21:00", end: "07:00" }\n',
        name="b.yaml",
    )
    layer = load(root)
    assert [window.tz for window in layer.defaults.quiet_windows] == ["Asia/Jerusalem"]


# --- version hash ------------------------------------------------------------------------

LAYOUT_ONLY_VARIANTS = {
    "same file": BASE_TEXT,
    "extra comment": BASE_TEXT.replace(
        "# generated base: the pipeline writes this file, a human never edits it",
        "# generated base\n# a second comment line that changes nothing semantic",
    ),
    "extra whitespace": BASE_TEXT.replace("min_ttl: 30s", "min_ttl:    30s"),
    "flow to block style": BASE_TEXT.replace(
        "  crm: { adapter: postgresql, secret: crm_ro }",
        "  crm:\n    adapter: postgresql\n    secret: crm_ro",
    ),
}


@pytest.mark.parametrize("variant", LAYOUT_ONLY_VARIANTS, ids=list(LAYOUT_ONLY_VARIANTS))
def test_version_hash_ignores_comments_and_layout(tmp_path: Path, variant: str) -> None:
    reference = load_text(tmp_path, BASE_TEXT, name="reference.yaml")
    edited = load_text(tmp_path, LAYOUT_ONLY_VARIANTS[variant], name="edited.yaml")
    assert reference.version_hash == edited.version_hash


CONTENT_CHANGES = {
    "measure rounding mode": ("mode: half_up", "mode: half_even"),
    "measure rounding granularity": ("to: 5000", "to: 1000"),
    "cache ttl": ("min_ttl: 30s", "min_ttl: 60s"),
    "mutability class": ("class: append_rarely", "class: append_only"),
    "hebrew synonym": ('"עזב"', '"נטש"'),
    "layer version": ("version: 7", "version: 8"),
    "join cardinality": ("cardinality: many_to_one", "cardinality: one_to_many"),
    "ingestion lag": ("usage: { count_90d: 12", "usage: { count_90d: 13"),
}


@pytest.mark.parametrize("change", CONTENT_CHANGES, ids=list(CONTENT_CHANGES))
def test_version_hash_changes_with_semantic_content(tmp_path: Path, change: str) -> None:
    old, new = CONTENT_CHANGES[change]
    assert old in BASE_TEXT
    reference = load_text(tmp_path, BASE_TEXT, name="reference.yaml")
    edited = load_text(tmp_path, BASE_TEXT.replace(old, new), name="edited.yaml")
    assert reference.version_hash != edited.version_hash


def test_version_hash_ignores_key_order(tmp_path: Path) -> None:
    first = load_text(
        tmp_path, "version: 1\nvalue_maps:\n  s:\n    canonical: [active]\n", name="a.yaml"
    )
    second = load_text(
        tmp_path, "value_maps:\n  s:\n    canonical: [active]\nversion: 1\n", name="b.yaml"
    )
    assert first.version_hash == second.version_hash


def test_an_overridden_layer_hashes_differently_from_the_generated_one(tmp_path: Path) -> None:
    root = tmp_path / "semantic"
    shutil.copytree(OVERLAY, root)
    overlaid = load(root)
    (root / "curated" / "human.yaml").unlink()
    generated_only = load(root)
    assert overlaid.version_hash != generated_only.version_hash


# --- validation errors: file, path, line -------------------------------------------------

ERROR_CASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "typo in a field name": (
        "version: 1\n"
        "tables:\n"
        "  crm.customers:\n"
        "    key: [cust_id]\n"
        "    mutability:\n"
        "      class: append_rarely\n"
        "      stauts: approved\n",
        (
            "typo.yaml",
            "tables.crm.customers.mutability.stauts",
            "line 7",
            "did you mean 'status'",
        ),
    ),
    "unknown mutability class": (
        "version: 1\ntables:\n  crm.customers:\n    mutability:\n      class: append_nightly\n",
        (
            "unknown class.yaml",
            "tables.crm.customers.mutability.class",
            "line 5",
            "unknown mutability class 'append_nightly'",
            "unknown",
        ),
    ),
    "broken yaml": (
        "version: 1\ntables: [\n",
        ("broken.yaml", "invalid YAML", "line"),
    ),
    "bad duration": (
        "version: 1\ntables:\n  crm.customers:\n    default_window: 7\n",
        (
            "bad window.yaml",
            "tables.crm.customers.default_window",
            "line 4",
            "expected a duration string",
        ),
    ),
    "derivation without a measured lag": (
        "version: 1\n"
        "tables:\n"
        "  billing.usage_events:\n"
        "    time_columns:\n"
        "      start_time: { role: event_time, indexed: true }\n"
        "      insertion_time: { role: ingestion_time, derives_bound_on: start_time }\n",
        ("no lag.yaml", "tables.billing.usage_events.time_columns.insertion_time", "measured lag"),
    ),
    "derivation onto a column that is not listed": (
        "version: 1\n"
        "tables:\n"
        "  billing.usage_events:\n"
        "    time_columns:\n"
        "      insertion_time:\n"
        "        role: ingestion_time\n"
        "        derives_bound_on: start_time\n"
        "        lag: { min: 0s, max: 6h }\n",
        (
            "missing column.yaml",
            "tables.billing.usage_events",
            "line 3",
            "not listed under time_columns",
        ),
    ),
    "measure template without the column placeholder": (
        "version: 1\n"
        "measures:\n"
        "  credit_limit:\n"
        "    sources:\n"
        '      crm.customers.credit_limit: { normalize: "ROUND(x)" }\n',
        ("no col.yaml", "normalize", "must use the placeholder"),
    ),
    "measure template with an unknown placeholder": (
        "version: 1\n"
        "measures:\n"
        "  credit_limit:\n"
        "    sources:\n"
        '      crm.customers.credit_limit: { normalize: "{column} * 10" }\n',
        ("wrong col.yaml", "unknown placeholder", "column"),
    ),
    "measure template that is not sql": (
        "version: 1\n"
        "measures:\n"
        "  credit_limit:\n"
        "    sources:\n"
        '      crm.customers.credit_limit: { normalize: "ROUND({col} / 5000" }\n',
        ("broken sql.yaml", "is not valid SQL"),
    ),
    "join without a left placeholder": (
        "version: 1\n"
        "approved_joins:\n"
        "  - name: usage_to_customers\n"
        "    left: billing.usage_events.client_no\n"
        "    right: crm.customers.cust_id\n"
        '    on: "{right} = 1"\n'
        "    cardinality: many_to_one\n",
        ("join.yaml", "approved join", "left"),
    ),
    "unqualified join side": (
        "version: 1\n"
        "approved_joins:\n"
        "  - name: usage_to_customers\n"
        "    left: client_no\n"
        "    right: crm.customers.cust_id\n"
        '    on: "{left} = {right}"\n'
        "    cardinality: many_to_one\n",
        ("unqualified.yaml", "approved_joins[0]", "line 3", "source.table.column"),
    ),
    "value map code outside the canonical list": (
        "version: 1\n"
        "value_maps:\n"
        "  customer_status:\n"
        "    canonical: [active]\n"
        "    sources:\n"
        "      crm.customers.status: { A: active, X: closed }\n",
        ("stray code.yaml", "value_maps.customer_status", "not canonical values"),
    ),
    "ambiguous synonym": (
        "version: 1\n"
        "value_maps:\n"
        "  customer_status:\n"
        "    canonical: [active, churned]\n"
        "    synonyms:\n"
        '      active: ["פעיל"]\n'
        '      churned: ["פָּעִיל"]\n',
        ("ambiguous.yaml", "value_maps.customer_status", "synonym of both"),
    ),
}


@pytest.mark.parametrize("case", ERROR_CASES, ids=list(ERROR_CASES))
def test_a_broken_document_names_file_path_and_line(tmp_path: Path, case: str) -> None:
    text, expected = ERROR_CASES[case]
    name = expected[0]
    path = write_layer(tmp_path, text, name=name)
    with pytest.raises(SemanticError) as caught:
        load(path)
    message = str(caught.value)
    assert message.startswith(str(path))
    for fragment in expected:
        assert fragment in message, f"{fragment!r} missing from: {message}"


def test_an_error_names_the_curated_file_that_holds_the_bad_value(tmp_path: Path) -> None:
    root = tmp_path / "semantic"
    shutil.copytree(OVERLAY, root)
    curated = root / "curated" / "human.yaml"
    curated.write_text(
        curated.read_text(encoding="utf-8").replace(
            "class: mutable_critical", "class: mutable_always"
        ),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(SemanticError) as caught:
        load(root)
    message = str(caught.value)
    assert "human.yaml" in message
    assert "tables.crm.customers.mutability.class" in message
    assert "line 5" in message


def test_a_missing_path_and_an_empty_directory_are_reported(tmp_path: Path) -> None:
    with pytest.raises(SemanticError, match="not found"):
        load(tmp_path / "nope.yaml")
    empty = tmp_path / "semantic"
    empty.mkdir()
    with pytest.raises(SemanticError, match="no semantic files found"):
        load(empty)


def test_a_document_that_is_not_a_mapping_is_reported(tmp_path: Path) -> None:
    path = write_layer(tmp_path, "- just\n- a list\n")
    with pytest.raises(SemanticError, match="must be a mapping"):
        load(path)


# --- mutability is conservative ----------------------------------------------------------

MUTABILITY_CASES = {
    "approved append_only": ({"class": "append_only", "status": "approved"}, True),
    "approved append_rarely": ({"class": "append_rarely", "status": "approved"}, True),
    "proposed append_only": ({"class": "append_only", "status": "proposed"}, False),
    "rejected append_only": ({"class": "append_only", "status": "rejected"}, False),
    "approved mutable_critical": ({"class": "mutable_critical", "status": "approved"}, False),
    "approved unknown": ({"class": "unknown", "status": "approved"}, False),
}


@pytest.mark.parametrize("case", MUTABILITY_CASES, ids=list(MUTABILITY_CASES))
def test_mutability_cache_permission_is_conservative(case: str) -> None:
    declaration, expected = MUTABILITY_CASES[case]
    layer = SemanticLayer.model_validate({"tables": {"crm.customers": {"mutability": declaration}}})
    table = layer.tables["crm.customers"]
    assert table.mutability is not None
    assert table.may_cache is expected
    assert table.mutability.behaves_as_mutable is not expected


def test_unknown_mutability_is_accepted_and_treated_as_mutable() -> None:
    layer = SemanticLayer.model_validate(
        {"tables": {"crm.customers": {"mutability": {"class": "unknown"}}}}
    )
    table = layer.tables["crm.customers"]
    assert table.mutability is not None
    assert table.mutability.class_ == "unknown"
    assert table.mutability.status.value == "proposed"
    assert table.may_cache is False


def test_a_cache_override_of_none_disables_caching() -> None:
    layer = SemanticLayer.model_validate(
        {
            "tables": {
                "crm.customers": {
                    "mutability": {"class": "append_only", "status": "approved"},
                    "cache": {"mode": "none"},
                }
            }
        }
    )
    assert layer.tables["crm.customers"].may_cache is False


# --- partitioning written the short way --------------------------------------------------

PARTITIONING_CASES: dict[str, tuple[dict[str, Any], PartitionStrategy, tuple[str, ...]]] = {
    "bare key": (
        {"key": "start_time", "granularity": "month", "require_key_filter": True},
        PartitionStrategy.RANGE,
        ("start_time",),
    ),
    "key list": ({"key": ["a", "b"]}, PartitionStrategy.RANGE, ("a", "b")),
    "expression": (
        {"expression": "date_trunc('month', ts)"},
        PartitionStrategy.EXPRESSION,
        (),
    ),
    "explicit hash": (
        {"strategy": "hash", "key": "tenant_id", "require_key_filter": False},
        PartitionStrategy.HASH,
        ("tenant_id",),
    ),
}


@pytest.mark.parametrize("case", PARTITIONING_CASES, ids=list(PARTITIONING_CASES))
def test_partitioning_short_form_is_normalised(case: str) -> None:
    section, strategy, key = PARTITIONING_CASES[case]
    layer = SemanticLayer.model_validate(
        {"tables": {"billing.usage_events": {"partitioning": section}}}
    )
    partitioning = layer.tables["billing.usage_events"].partitioning
    assert partitioning is not None
    assert partitioning.strategy is strategy
    assert partitioning.key == key


def test_an_empty_partitioning_section_is_not_partitioned() -> None:
    layer = SemanticLayer.model_validate(
        {"tables": {"crm.customers": {"partitioning": {"key": []}}}}
    )
    partitioning = layer.tables["crm.customers"].partitioning
    assert partitioning is not None
    assert partitioning.strategy is PartitionStrategy.NONE


# --- value maps in Hebrew and English ----------------------------------------------------


@pytest.fixture(scope="module")
def example_layer() -> SemanticLayer:
    return load(EXAMPLE)


LABEL_CASES = {
    "hebrew": ("פעיל", "active"),
    "hebrew with niqqud": ("פָּעִיל", "active"),
    "hebrew churned": ("עזב", "churned"),
    "english": ("Active", "active"),
    "english padded": ("  CHURNED  ", "churned"),
    "unknown label": ("לא קיים", None),
    "empty label": ("", None),
}


@pytest.mark.parametrize("case", LABEL_CASES, ids=list(LABEL_CASES))
def test_value_map_resolves_labels_in_both_languages(
    example_layer: SemanticLayer, case: str
) -> None:
    label, expected = LABEL_CASES[case]
    assert example_layer.value_maps["customer_status"].resolve(label) == expected


CODE_CASES: dict[str, tuple[str, object, str | None]] = {
    "crm code": ("crm.customers.status", "A", "active"),
    "billing label": ("billing.accounts.state", "Hold", "suspended"),
    "erp integer code": ("erp.customers.status_code", 1, "active"),
    "erp integer code as text": ("erp.customers.status_code", "3", "suspended"),
    "unknown code": ("crm.customers.status", "Z", None),
    "unknown column": ("crm.customers.unknown", "A", None),
}


@pytest.mark.parametrize("case", CODE_CASES, ids=list(CODE_CASES))
def test_value_map_maps_source_codes(example_layer: SemanticLayer, case: str) -> None:
    column, code, expected = CODE_CASES[case]
    assert example_layer.value_maps["customer_status"].canonical_for(column, code) == expected


NORMALISE_CASES = {
    "plain": ("שלום", "שלום"),
    "niqqud stripped": ("פָּעִיל", "פעיל"),
    "geresh hebrew": ("ג׳", "ג'"),
    "geresh typographic": ("ג’", "ג'"),
    "gershayim hebrew": ("צה״ל", 'צה"ל'),
    "gershayim typographic": ("צה”ל", 'צה"ל'),
    "case folded": ("Active Customer", "active customer"),
    "whitespace collapsed": ("  active \n customer ", "active customer"),
    "nfkc ligature": ("ﬁle", "file"),
    "nfkc compatibility": ("Ⅻ", "xii"),
    "zero width removed": ("\u200bact\u200bive", "active"),
}


@pytest.mark.parametrize("case", NORMALISE_CASES, ids=list(NORMALISE_CASES))
def test_normalize_label(case: str) -> None:
    raw, expected = NORMALISE_CASES[case]
    assert normalize_label(raw) == expected


# --- durations ---------------------------------------------------------------------------


DURATION_CASES = {
    "seconds": ("30s", timedelta(seconds=30)),
    "minutes": ("90m", timedelta(minutes=90)),
    "hours": ("6h", timedelta(hours=6)),
    "days": ("7d", timedelta(days=7)),
    "weeks": ("2w", timedelta(weeks=2)),
    "milliseconds": ("500ms", timedelta(milliseconds=500)),
    "padded": (" 1d ", timedelta(days=1)),
    "fractional": ("1.5h", timedelta(minutes=90)),
    "iso 8601": ("P7D", timedelta(days=7)),
    "iso 8601 time part": ("PT6H", timedelta(hours=6)),
    "already a timedelta": (timedelta(0), timedelta(0)),
}


@pytest.mark.parametrize("case", DURATION_CASES, ids=list(DURATION_CASES))
def test_parse_duration(case: str) -> None:
    raw, expected = DURATION_CASES[case]
    assert parse_duration(raw) == expected


@pytest.mark.parametrize("raw", ["7", "5x", "-1d", "", "  ", "seven days", 7, True, 1.5], ids=repr)
def test_parse_duration_rejects_ambiguous_input(raw: object) -> None:
    with pytest.raises(ValueError, match="duration"):
        parse_duration(raw)


# --- SQL templates -----------------------------------------------------------------------

TEMPLATE_CASES = {
    "identity": ("{col}", True),
    "rounding": ("ROUND({col} / 5000) * 5000", True),
    "scaling": ("{col} * 10", True),
    "no placeholder": ("ROUND(x)", False),
    "unknown placeholder": ("{column} * 10", False),
    "unbalanced": ("ROUND({col} / 5000", False),
    "two statements": ("{col}; DROP TABLE t", False),
}


@pytest.mark.parametrize("case", TEMPLATE_CASES, ids=list(TEMPLATE_CASES))
def test_measure_template_is_checked_at_load_time(case: str) -> None:
    template, expected = TEMPLATE_CASES[case]
    if expected:
        assert (
            check_sql_template(
                template,
                allowed=frozenset({"col"}),
                required=frozenset({"col"}),
                what="a measure normalize template",
            )
            == template
        )
    else:
        with pytest.raises(ValueError):
            check_sql_template(
                template,
                allowed=frozenset({"col"}),
                required=frozenset({"col"}),
                what="a measure normalize template",
            )


def test_render_template_substitutes_only_declared_placeholders() -> None:
    assert (
        render_template(
            "LPAD(CAST({right} AS VARCHAR), 10, '0') = {left}", {"right": "b", "left": "a"}
        )
        == "LPAD(CAST(b AS VARCHAR), 10, '0') = a"
    )
    with pytest.raises(ValueError, match="no value for the placeholder"):
        render_template("{col} * 10", {})


# --- the writer --------------------------------------------------------------------------


def test_write_generated_leaves_curated_files_untouched(tmp_path: Path) -> None:
    root = tmp_path / "semantic"
    shutil.copytree(OVERLAY, root)
    curated = root / "curated" / "human.yaml"
    before = curated.read_bytes()

    written = write_generated(root, "tables", {"tables": {"crm.customers": {"key": ["cust_id"]}}})

    assert written == root / "generated" / "tables.yaml"
    assert curated.read_bytes() == before
    assert "Do not edit" in written.read_text(encoding="utf-8")


def test_written_generated_facts_load_back(tmp_path: Path) -> None:
    write_generated(
        tmp_path,
        "tables",
        {"tables": {"crm.customers": {"mutability": {"class": "unknown"}}}},
    )
    layer = load(tmp_path)
    assert layer.tables["crm.customers"].mutability is not None
    assert layer.tables["crm.customers"].may_cache is False


def test_write_generated_refuses_a_name_that_escapes_the_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a file name"):
        write_generated(tmp_path, "../escape", {})


def test_update_curated_preserves_comments(tmp_path: Path) -> None:
    root = tmp_path / "semantic"
    curated = root / "curated" / "human.yaml"
    curated.parent.mkdir(parents=True)
    curated.write_text(
        "# kept at the top\n"
        "value_maps:\n"
        "  # kept above the value map\n"
        "  customer_status:\n"
        "    canonical: [active]\n"
        "    sources:\n"
        "      crm.customers.status: { A: active }\n",
        encoding="utf-8",
        newline="\n",
    )

    update_curated(
        curated,
        {
            "value_maps": {
                "customer_status": {
                    "canonical": ["active", "churned"],
                    "sources": {"crm.customers.status": {"A": "active", "C": "churned"}},
                }
            }
        },
    )

    text = curated.read_text(encoding="utf-8")
    assert "# kept at the top" in text
    assert "# kept above the value map" in text
    layer = load(root)
    assert layer.value_maps["customer_status"].canonical == ("active", "churned")
    assert layer.value_maps["customer_status"].canonical_for("crm.customers.status", "C") == (
        "churned"
    )


def test_update_curated_creates_a_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "semantic" / "curated" / "new.yaml"
    update_curated(path, {"version": 3})
    assert load(path.parent.parent).version == 3


def test_dump_writes_hebrew_as_text(tmp_path: Path) -> None:
    text = dump({"value_maps": {"s": {"synonyms": {"active": ["פעיל"]}}}})
    assert "פעיל" in text
    assert "\\u" not in text
