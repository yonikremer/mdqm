"""C03: query spec models, the duration parser, and the JSON Schema export.

Everything here is pure logic: no source, no file written, no clock. The table-driven cases come
first (a weak agent's failure modes are concrete), the property tests cover the two real invariants
(spec -> JSON -> spec, and a positive duration always moves a timestamp forward).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from hypothesis import given
from hypothesis import strategies as st
from ruamel.yaml import YAML

from mdqm.spec import (
    Duration,
    QuerySpec,
    SemanticVocabulary,
    SpecError,
    SpecIssue,
    TimeMode,
    check_spec,
    parse_duration,
    parse_spec,
    spec_schema,
    vocabulary_from_semantic,
)
from mdqm.spec.schema import ALLOWED_KEYWORDS, ref_cycles, schema_keywords

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples" / "query_specs.json"
SEMANTIC = REPO / "examples" / "semantic_example.yaml"

#: Cases are always written the way an agent writes them: as JSON-ready mappings.
Spec = dict[str, Any]


#: The canonical examples file is the contract: every entry must behave as it is labelled.
_EXAMPLES: dict[str, list[dict[str, Any]]] = json.loads(EXAMPLES.read_text(encoding="utf-8"))
VALID_EXAMPLES = [pytest.param(case, id=case["name"]) for case in _EXAMPLES["valid"]]
INVALID_EXAMPLES = [pytest.param(case, id=case["name"]) for case in _EXAMPLES["invalid"]]


# --------------------------------------------------------------------------------------------------
# Duration parser
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("45s", (0, 0, 45)),
        ("90m", (0, 0, 5400)),
        ("6h", (0, 0, 21_600)),
        ("2w", (0, 14, 0)),
        ("30d", (0, 30, 0)),
        ("3mo", (3, 0, 0)),
        ("1y", (12, 0, 0)),
        ("007d", (0, 7, 0)),
        ("0s", (0, 0, 0)),
        ("P1M", (1, 0, 0)),
        ("P1Y", (12, 0, 0)),
        ("PT6H", (0, 0, 21_600)),
        ("P2W", (0, 14, 0)),
        ("P1Y2M3DT4H5M6S", (14, 3, 14_706)),
        ("P3M", (3, 0, 0)),
    ],
)
def test_duration_parses_compact_and_iso(text: str, expected: tuple[int, int, int]) -> None:
    duration = parse_duration(text)
    assert (duration.months, duration.days, duration.seconds) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "1",
        "d",
        "1D",  # case carries meaning in ISO, so the compact form may not be case-folded
        "1M",  # minutes or months? never guess
        "1MO",
        "1.5d",
        "1 d",
        " 3mo",
        "3mo ",
        "1d12h",  # several components: use ISO when you mean a sum
        "1min",
        "1mon",
        "-1d",
        "+1d",
        "1e3d",
        "\u0661d",  # Arabic-Indic digit 1: locale-free means ASCII digits only
        "P",
        "PT",
        "P1YT",
        "P1.5D",
        "P1D2H",  # a time part must follow T
        "9999999d",
    ],
)
def test_duration_rejections_say_what_to_send(text: str) -> None:
    with pytest.raises(ValueError) as caught:
        parse_duration(text)
    message = str(caught.value)
    assert message
    assert "Traceback" not in message


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("1M", "'m' is minutes and 'mo' is months"),  # case decides ISO's M; never guess
        ("1D", "lowercase"),
        ("1.5d", "P1DT12H"),  # points at ISO for a fractional or compound span
        ("1d12h", "ISO-8601"),
        ("3 months", "ISO-8601"),
        ("1min", "ambiguous"),
    ],
)
def test_duration_rejection_message_is_actionable(text: str, fragment: str) -> None:
    """A weak agent must be able to fix the duration from the message alone."""
    with pytest.raises(ValueError) as caught:
        parse_duration(text)
    assert fragment in str(caught.value)


def test_duration_text_round_trips() -> None:
    for text in ("45s", "90m", "6h", "2w", "30d", "3mo", "1y", "P1Y2M3DT4H5M6S"):
        duration = parse_duration(text)
        assert parse_duration(duration.to_text()) == duration


@pytest.mark.parametrize(
    ("start", "text", "expected"),
    [
        # The documented month-end rule: keep the day of month, clamp to the last day of a short
        # month. Adding 28 days would silently give a different window, and rolling into March
        # would skip February, so clamping is the only deterministic, never-wider answer.
        ("2026-01-31T00:00:00", "1mo", "2026-02-28T00:00:00"),
        ("2024-01-31T00:00:00", "1mo", "2024-02-29T00:00:00"),  # leap year
        ("2026-03-31T00:00:00", "1mo", "2026-04-30T00:00:00"),
        ("2026-11-30T00:00:00", "3mo", "2027-02-28T00:00:00"),
        ("2024-02-29T00:00:00", "1y", "2025-02-28T00:00:00"),
        ("2026-01-31T00:00:00", "P1M", "2026-02-28T00:00:00"),  # ISO month == calendar month
        ("2026-01-15T10:30:00", "30d", "2026-02-14T10:30:00"),
        ("2026-01-15T10:30:00", "90m", "2026-01-15T12:00:00"),
        ("2026-01-01T00:00:00", "1y", "2027-01-01T00:00:00"),
    ],
)
def test_add_to_is_calendar_aware(start: str, text: str, expected: str) -> None:
    moment = datetime.fromisoformat(start).replace(tzinfo=UTC)
    assert parse_duration(text).add_to(moment) == datetime.fromisoformat(expected).replace(
        tzinfo=UTC
    )


@pytest.mark.parametrize(
    ("start", "text", "expected"),
    [
        ("2026-01-31T00:00:00", "1mo", "2025-12-31T00:00:00"),
        ("2026-03-31T00:00:00", "1mo", "2026-02-28T00:00:00"),
        ("2026-01-01T00:00:00", "1y", "2025-01-01T00:00:00"),
        ("2026-01-01T00:00:00", "6h", "2025-12-31T18:00:00"),
    ],
)
def test_subtract_from_is_calendar_aware(start: str, text: str, expected: str) -> None:
    moment = datetime.fromisoformat(start).replace(tzinfo=UTC)
    assert parse_duration(text).subtract_from(moment) == datetime.fromisoformat(expected).replace(
        tzinfo=UTC
    )


def test_zero_duration_is_zero() -> None:
    assert parse_duration("0s").is_zero
    assert not parse_duration("1s").is_zero


def test_duration_rejects_a_window_of_zero_in_a_spec() -> None:
    with pytest.raises(SpecError) as caught:
        parse_spec(
            {
                "from": "usage_events",
                "select": ["event_id"],
                "time": {"mode": "events_in_range", "last": "0d"},
            }
        )
    assert [issue.path for issue in caught.value.issues] == ["time.last"]


@given(
    months=st.integers(min_value=0, max_value=200),
    days=st.integers(min_value=0, max_value=400),
    seconds=st.integers(min_value=0, max_value=100_000),
    offset_days=st.integers(min_value=0, max_value=3_000),
)
def test_positive_duration_always_moves_forward(
    months: int, days: int, seconds: int, offset_days: int
) -> None:
    """An invariant a bound depends on: a positive span never collapses or reverses a timestamp."""
    duration = Duration(months=months, days=days, seconds=seconds)
    moment = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=offset_days)
    later = duration.add_to(moment)
    earlier = duration.subtract_from(moment)
    if duration.is_zero:
        assert later == moment == earlier
    else:
        assert later > moment > earlier
        assert duration.add_to(later) >= later


# --------------------------------------------------------------------------------------------------
# The canonical examples file
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", VALID_EXAMPLES)
def test_valid_examples_parse(case: dict[str, Any]) -> None:
    spec = parse_spec(case["spec"])
    assert spec.from_table == case["spec"]["from"]
    assert spec.select
    # The wire format keeps the agent's own spelling ("from"), so it can be re-sent verbatim.
    assert json.loads(spec.model_dump_json())["from"] == case["spec"]["from"]


@pytest.mark.parametrize("case", VALID_EXAMPLES)
def test_valid_examples_round_trip_through_json(case: dict[str, Any]) -> None:
    spec = parse_spec(case["spec"])
    assert QuerySpec.model_validate_json(spec.model_dump_json()) == spec
    assert parse_spec(json.dumps(case["spec"])) == spec


@pytest.mark.parametrize("case", INVALID_EXAMPLES)
def test_invalid_examples_fail_with_the_listed_path(case: dict[str, Any]) -> None:
    with pytest.raises(SpecError) as caught:
        parse_spec(case["spec"])
    paths = [issue.path for issue in caught.value.issues]
    assert case["error_path"] in paths, paths
    for issue in caught.value.issues:
        assert issue.problem and issue.fix
        assert "Traceback" not in issue.problem


# --------------------------------------------------------------------------------------------------
# Filters: one table of accepted and refused value shapes
# --------------------------------------------------------------------------------------------------


def _filter_spec(**filter_kwargs: Any) -> Spec:
    return {
        "from": "usage_events",
        "select": ["event_id"],
        "filters": [{"column": "customer_status", **filter_kwargs}],
    }


@pytest.mark.parametrize(
    ("op", "value"),
    [
        ("eq", "פעיל"),  # a Hebrew label is a value, not SQL
        ("eq", " Active "),
        ("eq", "A"),
        ("eq", 1),
        ("eq", 1.5),
        ("eq", True),
        ("ne", "closed"),
        ("in", ["active", "churned"]),
        ("in", [1, 2, 3]),
        ("contains", "abc"),
        ("gt", 5),
        ("gte", 5.5),
        ("lt", 100),
        ("lte", "2026-06-01"),
        ("range", [1, 10]),
        ("range", ["2026-01-01", "2026-06-01"]),
    ],
)
def test_filter_shapes_accepted(op: str, value: Any) -> None:
    spec = parse_spec(_filter_spec(op=op, value=value))
    assert spec.filters[0].op == op


@pytest.mark.parametrize(
    ("op", "value", "path", "fix_fragment"),
    [
        ("eq", ["a", "b"], "filters.0.value", "'in'"),
        ("eq", {"sql": "1; DROP TABLE x"}, "filters.0.value", "no way to express SQL"),
        ("eq", None, "filters.0.value", "no way to express SQL"),
        ("eq", "", "filters.0.value", "empty string"),
        ("in", "active", "filters.0.value", "takes a list"),
        ("in", [], "filters.0.value", "at least one"),
        ("range", [1], "filters.0.value", "exactly two"),
        ("range", [1, 2, 3], "filters.0.value", "exactly two"),
        ("range", [10, 1], "filters.0.value", "low < high"),
        ("range", [1, "a"], "filters.0.value", "two numbers or two dates"),
        ("gt", True, "filters.0.value", "not a boolean"),
        ("gt", "abc", "filters.0.value", "ISO-8601"),
        ("contains", 5, "filters.0.value", "text value"),
        ("eq", "a\tb", "filters.0.value", "control character"),
        ("nope", "x", "filters.0.op", "one of"),
    ],
)
def test_filter_shapes_rejected(op: str, value: Any, path: str, fix_fragment: str) -> None:
    with pytest.raises(SpecError) as caught:
        parse_spec(_filter_spec(op=op, value=value))
    issue = next(issue for issue in caught.value.issues if issue.path == path)
    assert fix_fragment in (issue.fix + issue.problem), (issue, fix_fragment)


# --------------------------------------------------------------------------------------------------
# Time spec: intent, never a column
# --------------------------------------------------------------------------------------------------


def _time_spec(**time_kwargs: Any) -> Spec:
    return {"from": "usage_events", "select": ["event_id"], "time": time_kwargs}


@pytest.mark.parametrize(
    "time_kwargs",
    [
        {"mode": "events_in_range", "last": "3mo"},
        {"mode": "events_in_range", "last": "7d"},
        {"mode": "events_in_range", "from": "2026-06-01"},
        {"mode": "events_in_range", "to": "2026-11-01"},
        {"mode": "events_in_range", "from": "2026-06-01", "to": "2026-11-01T00:00:00"},
        {"mode": "new_since", "since": "2026-11-25T14:00:00"},
        {"mode": "new_since", "since": "2026-11-25T16:00:00+02:00"},  # → 14:00Z
    ],
)
def test_time_shapes_accepted(time_kwargs: dict[str, Any]) -> None:
    spec = parse_spec(_time_spec(**time_kwargs))
    assert spec.time is not None
    assert spec.time.mode is TimeMode(time_kwargs["mode"])


def test_time_instants_are_normalised_to_utc() -> None:
    spec = parse_spec(_time_spec(mode="new_since", since="2026-11-25T16:00:00+02:00"))
    assert spec.time is not None
    assert spec.time.since == datetime(2026, 11, 25, 14, tzinfo=UTC)


@pytest.mark.parametrize(
    ("time_kwargs", "path"),
    [
        ({"mode": "whenever"}, "time.mode"),
        ({"mode": "events_in_range"}, "time"),
        ({"mode": "events_in_range", "last": "3mo", "from": "2026-06-01"}, "time.last"),
        (
            {"mode": "events_in_range", "from": "2026-11-01", "to": "2026-06-01"},
            "time.to",
        ),
        ({"mode": "events_in_range", "last": "3mo", "since": "2026-06-01"}, "time.since"),
        ({"mode": "new_since"}, "time.since"),
        ({"mode": "new_since", "since": "2026-11-25", "last": "7d"}, "time.last"),
        ({"mode": "new_since", "since": "2026-11-25", "to": "2026-11-26"}, "time.to"),
        ({"mode": "events_in_range", "from": "2026-13-01"}, "time.from"),
        ({"mode": "events_in_range", "last": "3 months"}, "time.last"),
    ],
)
def test_time_shapes_rejected(time_kwargs: dict[str, Any], path: str) -> None:
    with pytest.raises(SpecError) as caught:
        parse_spec(_time_spec(**time_kwargs))
    assert path in [issue.path for issue in caught.value.issues]


# --------------------------------------------------------------------------------------------------
# Top-level rules
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "path"),
    [
        ({"select": ["event_id"]}, "from"),
        ({"from": "usage_events"}, "select"),
        ({"from": "", "select": ["event_id"]}, "from"),
        ({"from": "usage_events", "select": []}, "select"),
        ({"from": "usage_events", "select": ["event_id"], "limit": 0}, "limit"),
        ({"from": "usage_events", "select": ["event_id"], "limit": "10"}, "limit"),
        ({"from": "usage_events", "select": ["event_id"], "sql": "SELECT 1"}, "sql"),
        (
            {"from": "usage_events", "select": ["event_id"], "group_by": ["a", "a"]},
            "group_by.1",
        ),
        (
            {"from": "usage_events", "select": ["event_id"], "group_by": [""]},
            "group_by.0",
        ),
        (
            {
                "from": "usage_events",
                "select": ["units", {"measure": "event_count", "agg": "count"}],
            },
            "select.0",
        ),
        (
            {
                "from": "usage_events",
                "select": ["event_id"],
                "order_by": [{"by": "missing"}],
            },
            "order_by.0.by",
        ),
        (
            {"from": "usage_events", "select": ["event_id"], "joins": ["a", "a"]},
            "joins.1",
        ),
        (
            {
                "from": "usage_events",
                "select": [{"measure": "event_count", "agg": "median"}],
            },
            "select.0.agg",
        ),
        ({"from": "usage_events", "select": ["event_id"], "dry_run": "maybe"}, "dry_run"),
        ({"from": "usage_events", "select": ["event_id"], "time": "3mo"}, "time"),
    ],
)
def test_top_level_rules(spec: Spec, path: str) -> None:
    with pytest.raises(SpecError) as caught:
        parse_spec(spec)
    assert path in [issue.path for issue in caught.value.issues]


def test_group_by_is_required_for_a_plain_column_beside_an_aggregate() -> None:
    spec = {
        "from": "usage_events",
        "select": ["customer_status", {"measure": "event_count", "agg": "count"}],
        "group_by": ["customer_status"],
    }
    assert parse_spec(spec).group_by == ("customer_status",)


def test_ordering_by_a_select_alias_is_allowed() -> None:
    spec = {
        "from": "usage_events",
        "select": [{"measure": "event_count", "agg": "count", "alias": "events"}],
        "order_by": [{"by": "events", "direction": "desc"}],
    }
    parsed = parse_spec(spec)
    assert parsed.order_by[0].by == "events"


def test_select_names_expose_the_output_columns() -> None:
    spec = parse_spec(
        {
            "from": "accounts",
            "select": [
                "customer_status",
                {"measure": "credit_limit_r5000", "agg": "avg"},
                {"measure": "event_count", "alias": "events"},
                {"measure": "event_count", "agg": "count"},
            ],
            "group_by": ["customer_status"],
        }
    )
    assert spec.selected_names() == (
        "customer_status",
        "credit_limit_r5000_avg",
        "events",
        "event_count_count",
    )


def test_spec_is_frozen() -> None:
    spec = parse_spec({"from": "usage_events", "select": ["event_id"]})
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError for frozen models
        spec.limit = 5  # type: ignore[misc]


def test_check_spec_is_pure_and_accepts_the_parsed_model() -> None:
    spec = QuerySpec.model_validate({"from": "usage_events", "select": ["event_id"]})
    assert check_spec(spec) == ()
    assert check_spec(spec) == ()


def test_invalid_json_and_wrong_root_type() -> None:
    with pytest.raises(SpecError) as caught:
        parse_spec("{not json")
    assert caught.value.issues[0].path == "(root)"
    with pytest.raises(SpecError) as caught:
        parse_spec(["from", "usage_events"])
    assert caught.value.issues[0].path == "(root)"


def test_parse_spec_accepts_bytes() -> None:
    spec = parse_spec(b'{"from": "usage_events", "select": ["event_id"]}')
    assert spec.from_table == "usage_events"


def test_error_response_shape_has_no_nulls() -> None:
    with pytest.raises(SpecError) as caught:
        parse_spec(
            {
                "from": "usage_events",
                "select": ["event_id"],
                "limit": 0,
                "sql": "SELECT 1",
            }
        )
    response = caught.value.to_error_response()
    assert response.status == "error"
    assert response.code == "INVALID_SPEC"
    assert response.problem and response.fix
    dumped = response.model_dump_json()
    assert "null" not in dumped
    assert "Traceback" not in dumped


def test_error_response_is_capped_and_says_how_many_were_hidden() -> None:
    issues = (SpecIssue(path=f"select.{index}", problem="bad", fix="fix it") for index in range(7))
    error = SpecError(list(issues))
    response = error.to_error_response()
    assert len(error.issues) == 7
    assert "and 2 more issue(s)" in response.problem
    assert response.fix.count("fix it") == 5


# --------------------------------------------------------------------------------------------------
# Hypothesis: spec -> JSON -> spec
# --------------------------------------------------------------------------------------------------

_names = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12).filter(
    lambda value: not value.endswith("_") or len(value) > 1
)
_labels = st.sampled_from(["פעיל", "active", " Active ", "A", "סגור"])
# JSON has no NaN/Infinity, and the spec rejects them, so the strategy stays finite.
_scalars = st.one_of(
    _labels,
    st.integers(min_value=-1000, max_value=1000),
    st.booleans(),
    st.floats(width=32, allow_nan=False, allow_infinity=False),
)
_durations = st.sampled_from(["45s", "90m", "6h", "2w", "30d", "3mo", "1y", "P1Y2M3D"])


@st.composite
def _specs(draw: st.DrawFn) -> Spec:
    columns = draw(st.lists(_names, min_size=1, max_size=3, unique=True))
    measures = draw(st.lists(_names, min_size=0, max_size=2, unique=True))
    aggregated = [
        {"measure": measure, "agg": draw(st.sampled_from(["avg", "sum", "count"]))}
        for measure in measures
    ]
    select: list[Any] = [*columns, *aggregated]
    time: Spec | None = draw(
        st.one_of(
            st.none(),
            st.just({"mode": "events_in_range", "last": draw(_durations)}),
            st.just({"mode": "new_since", "since": "2026-11-25T14:00:00"}),
        )
    )
    filters = [
        {
            "column": draw(_names),
            "op": "eq",
            "value": draw(_scalars),
        }
        for _ in range(draw(st.integers(min_value=0, max_value=2)))
    ]
    return {
        "from": draw(_names),
        "select": select,
        # A plain column beside an aggregate needs a grouping key, exactly as the rule says.
        "group_by": columns if aggregated else [],
        "filters": filters,
        "time": time,
        "limit": draw(st.one_of(st.none(), st.integers(min_value=1, max_value=1000))),
        "joins": draw(st.lists(_names, max_size=2, unique=True)),
        "dry_run": draw(st.booleans()),
        "fresh": draw(st.booleans()),
    }


@given(spec=_specs())
def test_any_valid_spec_round_trips_through_json(spec: Spec) -> None:
    parsed = parse_spec(spec)
    assert QuerySpec.model_validate_json(parsed.model_dump_json()) == parsed
    assert parse_spec(json.loads(parsed.model_dump_json())) == parsed


@given(spec=_specs())
def test_dumping_never_leaks_an_object_pydantic_cannot_reload(spec: Spec) -> None:
    dumped = json.loads(parse_spec(spec).model_dump_json())
    assert isinstance(dumped, dict)
    assert dumped["from"] == spec["from"]


# --------------------------------------------------------------------------------------------------
# JSON Schema export
# --------------------------------------------------------------------------------------------------


def _vocabulary() -> SemanticVocabulary:
    return SemanticVocabulary(
        tables=("accounts", "usage_events"),
        measures=("credit_limit_r5000", "event_count"),
        joins=("usage_to_customers",),
    )


def test_schema_enums_come_from_the_vocabulary() -> None:
    schema = spec_schema(_vocabulary())
    assert schema["properties"]["from"]["enum"] == ["accounts", "usage_events"]
    assert schema["$defs"]["MeasureSelect"]["properties"]["measure"]["enum"] == [
        "credit_limit_r5000",
        "event_count",
    ]
    assert schema["properties"]["joins"]["items"]["enum"] == ["usage_to_customers"]


def test_schema_without_a_vocabulary_leaves_open_strings() -> None:
    schema = spec_schema(SemanticVocabulary())
    assert "enum" not in schema["properties"]["from"]
    assert schema["properties"]["from"]["type"] == "string"


def test_schema_is_valid_json_schema_and_closed() -> None:
    schema = spec_schema(_vocabulary())
    jsonschema.Draft202012Validator.check_schema(schema)
    assert ref_cycles(schema) == ()
    for node in _object_schemas(schema):
        assert node["additionalProperties"] is False


def test_schema_uses_only_portable_keywords() -> None:
    schema = spec_schema(_vocabulary())
    unexpected = schema_keywords(schema) - ALLOWED_KEYWORDS
    assert not unexpected, f"structured-output endpoints may reject {sorted(unexpected)}"


def test_schema_is_deterministic() -> None:
    assert json.dumps(spec_schema(_vocabulary())) == json.dumps(spec_schema(_vocabulary()))


@pytest.mark.parametrize("case", VALID_EXAMPLES)
def test_valid_examples_validate_against_the_schema(case: dict[str, Any]) -> None:
    vocabulary = SemanticVocabulary(tables=(case["spec"]["from"],))
    jsonschema.validate(case["spec"], spec_schema(vocabulary))


def test_schema_refs_all_resolve_inside_defs() -> None:
    """A dangling ``$ref`` is invalid in any consumer, and check_schema does not catch it."""
    schema = spec_schema(_vocabulary())
    definitions = set(schema["$defs"])
    for target in _refs(schema):
        assert target in definitions, target


def test_schema_rejects_a_table_outside_the_vocabulary() -> None:
    schema = spec_schema(_vocabulary())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"from": "secrets", "select": ["x"]}, schema)


def test_vocabulary_from_the_example_semantic_layer() -> None:
    loaded = YAML(typ="safe").load(SEMANTIC.read_text(encoding="utf-8"))
    vocabulary = vocabulary_from_semantic(loaded)
    assert "billing.usage_events" in vocabulary.tables
    assert "credit_limit_r5000" in vocabulary.measures
    assert vocabulary.joins == ("usage_to_customers",)
    schema = spec_schema(loaded)
    assert "billing.usage_events" in schema["properties"]["from"]["enum"]


@pytest.mark.parametrize(
    "broken",
    [
        {"tables": ["not", "a", "mapping"]},
        {"approved_joins": "usage_to_customers"},
        {"approved_joins": [{"left": "a"}]},
    ],
)
def test_vocabulary_rejects_a_malformed_semantic_layer(broken: Mapping[str, Any]) -> None:
    with pytest.raises(ValueError):
        vocabulary_from_semantic(broken)


def _refs(node: Any) -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        reference = node.get("$ref")
        if isinstance(reference, str):
            found.append(reference.removeprefix("#/$defs/"))
        for value in node.values():
            found.extend(_refs(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_refs(item))
    return found


def _object_schemas(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every subschema that describes an object (it has `properties`)."""
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "properties" in node:
                found.append(dict(node))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    return found
