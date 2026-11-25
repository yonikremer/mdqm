"""Tests for the error taxonomy (F06): closed set, one builder per code, actionable fixes, no leaks.

Three things are proved here:

1. **Exhaustiveness.** Every :class:`~mdqm.errors.ErrorCode` has exactly one builder in
   :data:`mdqm.errors.BUILDERS` and at least one case below, and the code values are the frozen
   list the task published (a rename is a deliberate edit of this file, not an accident).
2. **Actionability.** Each builder's ``fix`` names the exact spec field or value to change, in one
   line, and is never truncated.
3. **No leaks.** No response carries a stack trace, a secret or a statement, whether it came from an
   exception (including its ``__cause__``/``__context__`` chain), from a hostile value the agent
   pasted, or from a code outside the taxonomy.

The mapping from real failure modes to codes is table-tested in both directions: a real failure
(guard rejection, timeout, adapter error, connection failure, malformed spec, anything else) yields
its documented code, and every such code is in the closed set and rebuilds to itself.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import mdqm.errors as errors
from mdqm.adapters.base import NotSupported
from mdqm.config import ConfigError
from mdqm.errors import BUILDERS, ErrorCode
from mdqm.models import ErrorResponse
from mdqm.sql.guard import guard

# --- the frozen taxonomy ----------------------------------------------------------------

#: The published code list. Stable by contract: changing a value breaks stored history and agents.
EXPECTED_CODES = frozenset(
    {
        "PARSE_ERROR",
        "WRITE_BLOCKED",
        "MULTI_STATEMENT",
        "INVALID_SPEC",
        "UNKNOWN_TABLE",
        "UNKNOWN_COLUMN",
        "AMBIGUOUS_VALUE",
        "UNKNOWN_VALUE",
        "JOIN_NOT_APPROVED",
        "INSERTION_TIME_NO_BOUND",
        "PARTITION_SCAN_ALL",
        "COST_TOO_HIGH",
        "QUOTA_EXCEEDED",
        "TIMEOUT",
        "RESULT_TOO_LARGE",
        "SOURCE_UNAVAILABLE",
        "NOT_SUPPORTED",
        "INTERNAL_ERROR",
    }
)

#: The codes C01's guard already publishes. F06 reuses them and must never rename them.
GUARD_CODES = ("PARSE_ERROR", "WRITE_BLOCKED", "MULTI_STATEMENT")

ON_CLAUSE = "crm.customers.cust_id = billing.accounts.client_no"


@dataclass(frozen=True)
class Case:
    """One builder invocation and what its response must say."""

    code: ErrorCode
    kwargs: dict[str, Any]
    fix_must: tuple[str, ...]
    problem_must: tuple[str, ...] = ()
    valid_values: tuple[str, ...] | None = None
    did_you_mean: tuple[str, ...] | None = None
    partitions_if_fixed: str | None = None


CASES: tuple[Case, ...] = (
    # Guard / compiler.
    Case(
        ErrorCode.PARSE_ERROR,
        {"dialect": "postgres"},
        fix_must=('"sql"', '"from"', '"select"'),
        problem_must=("postgres",),
    ),
    Case(
        ErrorCode.WRITE_BLOCKED,
        {"operation": "INSERT"},
        fix_must=('"INSERT"', '"sql"', "SELECT"),
        problem_must=("INSERT", "can only read"),
    ),
    Case(
        ErrorCode.MULTI_STATEMENT,
        {"count": 2},
        fix_must=("first ';'", "separate call"),
        problem_must=("2 statements",),
    ),
    # Spec and semantic layer.
    Case(
        ErrorCode.INVALID_SPEC,
        {
            "field": "time.mode",
            "reason": "it is not one of the allowed values",
            "expected": ("events_in_range", "new_since"),
        },
        fix_must=('"time.mode"', '"events_in_range"', '"new_since"'),
        problem_must=('"time.mode"', "not one of the allowed values"),
        valid_values=("events_in_range", "new_since"),
    ),
    Case(
        ErrorCode.INVALID_SPEC,
        {"field": "filters.0.value", "reason": "it is not a valid value"},
        fix_must=('"filters.0.value"', "examples/query_specs.json"),
    ),
    Case(
        ErrorCode.UNKNOWN_TABLE,
        {"name": "account", "known": ("accounts", "customers", "usage_events")},
        fix_must=('"from"', '"accounts"', '"usage_events"'),
        problem_must=('"account"', "semantic layer"),
        did_you_mean=("accounts",),
    ),
    Case(
        ErrorCode.UNKNOWN_COLUMN,
        {"table": "accounts", "name": "status", "known": ("customer_status", "client_no")},
        fix_must=('"select"', '"customer_status"'),
        problem_must=('"accounts"', '"status"'),
        did_you_mean=("customer_status",),
    ),
    Case(
        ErrorCode.AMBIGUOUS_VALUE,
        {"field": "filters.0.value", "value": "פעיל", "canonical": ("active", "Active")},
        fix_must=('"filters.0.value"', '"active"'),
        problem_must=("פעיל",),
        valid_values=("active", "Active"),
    ),
    Case(
        ErrorCode.UNKNOWN_VALUE,
        {"field": "group_by.0", "value": "nonsense", "valid": ("A", "Active")},
        fix_must=('"group_by.0"', '"A"', '"Active"'),
        valid_values=("A", "Active"),
    ),
    Case(
        ErrorCode.JOIN_NOT_APPROVED,
        {"left": "crm.customers", "right": "billing.accounts", "approved_on": ON_CLAUSE},
        fix_must=(ON_CLAUSE, "ON clause"),
        problem_must=("crm.customers", "billing.accounts", "not approved"),
    ),
    Case(
        ErrorCode.JOIN_NOT_APPROVED,
        {"left": "crm.customers", "right": "erp.customers"},
        fix_must=('"joins"', "curated/joins.yaml"),
    ),
    # Planning.
    Case(
        ErrorCode.INSERTION_TIME_NO_BOUND,
        {"table": "usage_events", "event_column": "start_time", "max_lag": "6h"},
        fix_must=('"time"', '"new_since"', '"start_time"', "6h"),
        problem_must=("usage_events", "not indexed"),
    ),
    Case(
        ErrorCode.INSERTION_TIME_NO_BOUND,
        {"table": "usage_events", "event_column": "start_time"},
        fix_must=('"time"', "events_in_range", '"start_time"', "narrow window"),
    ),
    Case(
        ErrorCode.PARTITION_SCAN_ALL,
        {
            "table": "usage_events",
            "key": "start_time",
            "granularity": "monthly",
            "default_window": "90d",
            "partitions_if_fixed": "3 of 24",
        },
        fix_must=('"time"', '"90d"', '"start_time"'),
        problem_must=("monthly", '"start_time"', "every partition"),
        partitions_if_fixed="3 of 24",
    ),
    Case(
        ErrorCode.PARTITION_SCAN_ALL,
        {"table": "usage_events", "key": "start_time", "granularity": "monthly"},
        fix_must=('"time"', '"30d"', '"start_time"'),
    ),
    Case(
        ErrorCode.COST_TOO_HIGH,
        {"table": "usage_events", "estimated": "12,000,000 rows", "limit": "1,000,000 rows"},
        fix_must=('"time"', '"limit"', '"group_by"'),
        problem_must=("12,000,000 rows", "1,000,000 rows"),
    ),
    # Governor and results.
    Case(
        ErrorCode.QUOTA_EXCEEDED,
        {"quota": "queries per minute", "limit": "60", "resets_at": "2026-11-25T15:00:00Z"},
        fix_must=("2026-11-25T15:00:00Z", "retry"),
        problem_must=("queries per minute", "60"),
    ),
    Case(
        ErrorCode.TIMEOUT,
        {"seconds": 30.0, "target": "usage_events"},
        fix_must=('"time"', '"group_by"'),
        problem_must=("usage_events", "30s", "cancelled"),
    ),
    Case(
        ErrorCode.TIMEOUT,
        {"seconds": None, "target": None},
        fix_must=('"time"', "server limit"),
        problem_must=("time limit",),
    ),
    Case(
        ErrorCode.RESULT_TOO_LARGE,
        {"max_rows": 10000, "rows": 42000},
        fix_must=('"limit": 10000', '"group_by"'),
        problem_must=("42000 rows", "10000"),
    ),
    Case(
        ErrorCode.RESULT_TOO_LARGE,
        {"max_rows": 10000},
        fix_must=('"limit": 10000',),
        problem_must=("cap is 10000 rows",),
    ),
    # Sources.
    Case(
        ErrorCode.SOURCE_UNAVAILABLE,
        {"source": "crm", "secret_name": "crm_ro"},
        fix_must=('"crm"', '"crm_ro"', "never logged"),
        problem_must=("crm", "not answering"),
    ),
    Case(
        ErrorCode.SOURCE_UNAVAILABLE,
        {"source": "crm"},
        fix_must=('"crm"', "reachable"),
    ),
    Case(
        ErrorCode.NOT_SUPPORTED,
        {"source": "crm", "capability": "explain_plan", "fallback": "index and cost gate only"},
        fix_must=('"explain_plan"', "index and cost gate only"),
        problem_must=("crm", "does not support"),
    ),
    Case(
        ErrorCode.NOT_SUPPORTED,
        {"source": "crm", "capability": "query_store"},
        fix_must=('"query_store"', "no fallback"),
    ),
    # Our own bugs.
    Case(
        ErrorCode.INTERNAL_ERROR,
        {"correlation_id": "9f2c1d4e5a6b7c8d"},
        fix_must=("9f2c1d4e5a6b7c8d", "administrator"),
        problem_must=("did not run",),
    ),
)

#: Text shapes that must never appear in an agent-facing error, with the value that must vanish.
HOSTILE_TEXTS = (
    (
        "password",
        "postgresql://mdqm_ro:hunter2@db.internal:5432/crm?password=hunter2",
        ("hunter2",),
    ),
    ("key value", "token=hunter2 api_key: hunter2 secret = hunter2", ("hunter2",)),
    (
        "bearer",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature",
        ("eyJhbGciOiJIUzI1NiJ9",),
    ),
    ("bare jwt", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJkYW5hIn0.c2lnbmF0dXJl", ("eyJzdWIiOiJkYW5hIn0",)),
    (
        "private key",
        "-----BEGIN RSA PRIVATE KEY-----\nhunter2\n-----END RSA PRIVATE KEY-----",
        ("hunter2",),
    ),
    ("aws key", "AKIAIOSFODNN7EXAMPLE", ("AKIAIOSFODNN7EXAMPLE",)),
    (
        "stack trace",
        'Traceback (most recent call last):\n  File "/srv/mdqm/run.py", line 42, in go\n'
        "    boom()\nhunter2",
        ("hunter2",),
    ),
    ("control bytes", "a\x00b\x1bc\x07d", ("\x00", "\x1b", "\x07")),
    ("newlines", "line one\nline two\r\nline three\ttab", ("\n", "\r\n")),
)


# --- 1. exhaustiveness ------------------------------------------------------------------


def test_codes_are_exactly_the_published_closed_set() -> None:
    assert {code.value for code in ErrorCode} == EXPECTED_CODES
    assert len(EXPECTED_CODES) == len(ErrorCode)


def test_every_code_value_is_stable_upper_snake_case() -> None:
    for code in ErrorCode:
        assert code.value == code.name, "the wire value and the member name must not drift"
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", code.value), code


def test_every_code_has_exactly_one_builder_and_at_least_one_case() -> None:
    assert set(BUILDERS) == set(ErrorCode), "a code without a builder could never be raised"
    assert all(callable(builder) for builder in BUILDERS.values())
    assert {case.code for case in CASES} == set(ErrorCode), "a code without a case is untested"


def test_the_guard_codes_are_reused_and_not_renamed() -> None:
    for code in GUARD_CODES:
        assert ErrorCode[code].value == code


def test_builder_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        BUILDERS[ErrorCode.TIMEOUT] = errors.parse_error  # type: ignore[index]


def test_builders_take_keyword_only_arguments() -> None:
    for code, builder in BUILDERS.items():
        for parameter in inspect.signature(builder).parameters.values():
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (code, parameter.name)


def test_builders_take_identifiers_never_a_statement() -> None:
    """No builder can be handed free text that could be another user's SQL.

    This is the structural half of "no SQL with literals from other users": there is no parameter
    that carries a statement, a message or a detail - only identifiers, spec fields and enums.
    """
    forbidden = {"sql", "statement", "query", "text", "detail", "message", "payload", "body"}
    for code, builder in BUILDERS.items():
        names = set(inspect.signature(builder).parameters)
        assert not names & forbidden, (code, sorted(names & forbidden))


# --- 2. actionability ------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[f"{index}-{case.code.value}" for index, case in enumerate(CASES)],
)
def test_builder_returns_its_code_with_one_line_actionable_text(case: Case) -> None:
    response = BUILDERS[case.code](**case.kwargs)

    assert isinstance(response, ErrorResponse)
    assert response.status == "error"
    assert response.code == case.code.value

    for text in (response.problem, response.fix):
        assert text and text == text.strip()
        assert "\n" not in text, "error text is one line"
        assert len(text) <= 512
        for marker in ("Traceback", "[empty]"):
            assert marker not in text

    assert "..." not in response.fix, "a builder must never truncate its own fix"
    for token in case.fix_must:
        assert token in response.fix, (case.code, token, response.fix)
    for token in case.problem_must:
        assert token in response.problem, (case.code, token, response.problem)

    assert response.valid_values == case.valid_values
    if case.did_you_mean is not None:
        assert response.did_you_mean == case.did_you_mean
    assert response.partitions_if_fixed == case.partitions_if_fixed


@pytest.mark.parametrize(
    "builder", list(BUILDERS.values()), ids=lambda builder: str(builder.__name__)
)
def test_every_builder_is_covered_by_cases(builder: Callable[..., ErrorResponse]) -> None:
    """Each builder is exercised by at least one case, so no code ships untested."""
    assert [case for case in CASES if BUILDERS[case.code] is builder]


def test_dump_omits_unset_optional_fields() -> None:
    response = errors.timeout(seconds=5.0, target="accounts")
    payload = response.model_dump()
    assert set(payload) == {"status", "code", "problem", "fix"}
    assert "partitions_if_fixed" not in response.model_dump_json()
    assert ErrorResponse.model_validate_json(response.model_dump_json()) == response


@pytest.mark.parametrize(
    "case", CASES, ids=[f"{index}-{case.code.value}" for index, case in enumerate(CASES)]
)
def test_every_response_has_only_the_published_keys(case: Case) -> None:
    """The agent-facing shape stays ``status``/``code``/``problem``/``fix`` plus three optionals."""
    published = {
        "status",
        "code",
        "problem",
        "fix",
        "partitions_if_fixed",
        "valid_values",
        "did_you_mean",
    }
    payload = BUILDERS[case.code](**case.kwargs).model_dump()
    assert set(payload) <= published
    assert {"status", "code", "problem", "fix"} <= set(payload)
    assert payload["status"] == "error"


def test_hebrew_values_survive_and_are_not_ascii_mangled() -> None:
    response = errors.unknown_value(field="filters.0.value", value="לא ידוע", valid=("A", "Active"))
    assert "לא ידוע" in response.problem
    assert response.valid_values == ("A", "Active")


# --- 3. no leaks ------------------------------------------------------------------------


@pytest.mark.parametrize("hostile", HOSTILE_TEXTS, ids=lambda item: item[0])
def test_a_pasted_credential_or_stack_trace_cannot_reach_an_error(
    hostile: tuple[str, str, tuple[str, ...]],
) -> None:
    _, text, must_vanish = hostile
    for case in CASES:
        kwargs = {
            name: (text,)
            if isinstance(value, tuple)
            else (text if isinstance(value, str) else value)
            for name, value in case.kwargs.items()
        }
        response = BUILDERS[case.code](**kwargs)
        dump = response.model_dump_json()
        for unwanted in (*must_vanish, "Traceback"):
            assert unwanted not in dump, (case.code, unwanted, dump)
        for text_field in (response.problem, response.fix):
            assert "\x00" not in text_field and "\x1b" not in text_field
            assert "\n" not in text_field and "\r" not in text_field
            assert "\t" not in text_field


def test_a_long_identifier_is_truncated_not_passed_through() -> None:
    response = errors.unknown_table(name="x" * 5000, known=("accounts",))
    assert "..." in response.problem
    assert "x" * 200 not in response.problem
    assert len(response.problem) < 512


def test_a_statement_from_another_user_cannot_reach_the_agent() -> None:
    """The failure the agent sees says nothing about the statement that failed.

    Another user's SQL can only arrive through an exception (history replay, query store, cache);
    ``from_exception`` never reads the exception text, so its literals cannot travel.
    """
    foreign = "SELECT cust_id, credit_limit FROM crm.customers WHERE name = 'dana' AND limit_k > 42"
    try:
        try:
            raise ValueError(f"replay failed: {foreign}")
        except ValueError as cause:
            raise RuntimeError(
                f"wrap {foreign}\nTraceback (most recent call last): File x"
            ) from cause
    except RuntimeError as exc:
        assert exc.__cause__ is not None
        assert exc.__context__ is not None
        response = errors.from_exception(exc, correlation_id="cid-1")

    dump = response.model_dump_json()
    assert response.code == "INTERNAL_ERROR"
    for leak in (
        "dana",
        "crm.customers",
        "limit_k",
        "credit_limit",
        "SELECT",
        "Traceback",
        "ValueError",
    ):
        assert leak not in dump, leak
    assert "cid-1" in response.fix


def test_internal_errors_are_logged_with_the_correlation_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nothing is silent: the operator gets the exception, the agent gets only the id."""
    with caplog.at_level(logging.ERROR, logger="mdqm.errors"):
        response = errors.from_exception(
            RuntimeError("connection dropped while reading payroll"), correlation_id="cid-9"
        )
    assert response.code == "INTERNAL_ERROR"
    record = caplog.records[-1]
    assert record.levelno == logging.ERROR
    assert getattr(record, "correlation_id", None) == "cid-9"
    assert record.exc_info is not None
    assert "payroll" in caplog.text, "the operator log keeps what the agent must not see"


def test_a_configuration_failure_is_internal_not_echoed() -> None:
    problem = ConfigError("mdqm.yaml", ["server.port: password=hunter2"])
    response = errors.from_exception(problem, correlation_id="cid-2")
    assert response.code == "INTERNAL_ERROR"
    assert "mdqm.yaml" not in response.model_dump_json()
    assert "hunter2" not in response.model_dump_json()


# --- 4. mapping real failures onto the taxonomy, both directions -------------------------


class _Spec(BaseModel):
    """A stand-in for C03's spec model: the same pydantic error shapes."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(gt=0)
    mode: Literal["events_in_range", "new_since"]


def _spec_error(**payload: Any) -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        _Spec.model_validate(payload)
    return caught.value


@dataclass(frozen=True)
class FailureMode:
    """A real failure and the code the agent must see for it."""

    name: str
    code: ErrorCode | None
    produce: Callable[[], ErrorResponse | None] = field(repr=False)


FAILURE_MODES: tuple[FailureMode, ...] = (
    FailureMode(
        "guard blocks a write",
        ErrorCode.WRITE_BLOCKED,
        lambda: errors.from_guard(guard("INSERT INTO customers VALUES (1)", "postgres")),
    ),
    FailureMode(
        "guard blocks two statements",
        ErrorCode.MULTI_STATEMENT,
        lambda: errors.from_guard(guard("SELECT 1; SELECT 2", "postgres")),
    ),
    FailureMode(
        "guard blocks unparseable SQL",
        ErrorCode.PARSE_ERROR,
        lambda: errors.from_guard(guard("SELECT FROM WHERE (", "postgres")),
    ),
    FailureMode(
        "guard allows a query",
        None,
        lambda: errors.from_guard(guard("SELECT 1 AS one", "postgres")),
    ),
    FailureMode(
        "a timeout",
        ErrorCode.TIMEOUT,
        lambda: errors.from_exception(TimeoutError("cancelled"), timeout_s=30),
    ),
    FailureMode(
        "an unsupported capability",
        ErrorCode.NOT_SUPPORTED,
        lambda: errors.from_exception(
            NotSupported("explain_plan", "no plan endpoint"), source="crm"
        ),
    ),
    FailureMode(
        "a source refusing the connection",
        ErrorCode.SOURCE_UNAVAILABLE,
        lambda: errors.from_exception(ConnectionRefusedError("refused"), source="crm"),
    ),
    FailureMode(
        "a malformed spec",
        ErrorCode.INVALID_SPEC,
        lambda: errors.from_validation_error(_spec_error(limit=0, mode="events_in_range")),
    ),
    FailureMode(
        "a bad enum value in the spec",
        ErrorCode.INVALID_SPEC,
        lambda: errors.from_validation_error(_spec_error(limit=5, mode="whenever")),
    ),
    FailureMode(
        "a used-up quota",
        ErrorCode.QUOTA_EXCEEDED,
        lambda: errors.quota_exceeded(
            quota="rows per hour", limit="1,000,000", resets_at="2026-11-25T15:00:00Z"
        ),
    ),
    FailureMode(
        "anything else",
        ErrorCode.INTERNAL_ERROR,
        lambda: errors.from_exception(KeyError("boom"), correlation_id="cid-3"),
    ),
)


@pytest.mark.parametrize("mode", FAILURE_MODES, ids=lambda mode: mode.name)
def test_a_real_failure_maps_to_its_documented_code(mode: FailureMode) -> None:
    response = mode.produce()
    if mode.code is None:
        assert response is None, "an allowed statement must not carry an error"
        return
    assert response is not None
    assert response.code == mode.code.value
    assert response.problem and response.fix
    assert mode.code in BUILDERS, "the mapped code must have a builder"
    assert ErrorCode(response.code) is mode.code


def test_every_mapped_code_round_trips_failure_to_code_to_builder_to_code() -> None:
    mapped = {mode.code for mode in FAILURE_MODES if mode.code is not None}
    assert mapped <= set(ErrorCode)
    cases = {case.code: case for case in CASES}
    for code in mapped:
        case = cases[code]
        assert BUILDERS[code](**case.kwargs).code == code.value


def test_the_guard_text_is_kept_and_only_resanitised() -> None:
    result = guard("DELETE FROM accounts", "postgres")
    assert result.error is not None
    response = errors.from_guard(result)
    assert response is not None
    assert response.code == result.error.code
    assert response.problem == result.error.problem
    assert response.fix == result.error.fix


def test_a_guard_code_outside_the_taxonomy_becomes_internal_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The closed set is enforced at runtime: a renamed code cannot leak to an agent."""

    @dataclass(frozen=True)
    class _Renamed:
        ok: bool = False
        error: ErrorResponse | None = ErrorResponse(
            code="WRITE_BLOCKED_LEGACY", problem="blocked", fix="do not do that"
        )

    with caplog.at_level(logging.ERROR, logger="mdqm.errors"):
        response = errors.from_guard(_Renamed())
    assert response is not None
    assert response.code == "INTERNAL_ERROR"
    assert "WRITE_BLOCKED_LEGACY" not in response.model_dump_json()
    assert "WRITE_BLOCKED_LEGACY" in caplog.text


def test_a_guard_rejection_without_a_response_becomes_internal_error() -> None:
    @dataclass(frozen=True)
    class _Broken:
        ok: bool = False
        error: ErrorResponse | None = None

    response = errors.from_guard(_Broken())
    assert response is not None
    assert response.code == "INTERNAL_ERROR"


def test_a_validation_error_never_echoes_the_offending_value() -> None:
    """An agent can put a credential in a spec field; the error must not repeat it."""
    exc = _spec_error(limit=5, mode="events_in_range", password="hunter2")
    response = errors.from_validation_error(exc)
    dump = response.model_dump_json()
    assert response.code == "INVALID_SPEC"
    assert "password" in response.fix
    assert "hunter2" not in dump


def test_validation_paths_match_the_spec_examples() -> None:
    response = errors.from_validation_error(_spec_error(limit=0, mode="events_in_range"))
    assert '"limit"' in response.fix
    assert "below the minimum" in response.problem

    response = errors.from_validation_error(_spec_error(limit=5, mode="whenever"))
    assert '"mode"' in response.fix
    assert response.valid_values == ("events_in_range", "new_since")


def test_a_nested_spec_path_is_dotted_like_the_examples() -> None:
    class _Time(BaseModel):
        model_config = ConfigDict(extra="forbid")
        mode: Literal["events_in_range", "new_since"]

    class _Nested(BaseModel):
        model_config = ConfigDict(extra="forbid")
        time: _Time

    with pytest.raises(ValidationError) as caught:
        _Nested.model_validate({"time": {"mode": "whenever"}})
    response = errors.from_validation_error(caught.value)
    assert '"time.mode"' in response.fix
    assert response.valid_values == ("events_in_range", "new_since")


def test_a_validation_error_with_no_field_errors_still_names_a_field() -> None:
    response = errors.from_validation_error(ValidationError.from_exception_data("_Spec", []))
    assert response.code == "INVALID_SPEC"
    assert "spec" in response.fix


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", "[empty]"),
        ("  spaced\ttext  ", "spaced text"),
        ("a\x00b", "ab"),
        ("hunter2" * 200, "hunter2" * 200),
    ],
)
def test_safe_text_is_one_line_and_never_empty(value: str, expected: str) -> None:
    cleaned = errors._safe(value)
    assert "\n" not in cleaned and "\x00" not in cleaned
    if expected == "hunter2" * 200:
        assert cleaned.endswith("...") and len(cleaned) == 512
    else:
        assert cleaned == expected


def test_safe_is_idempotent() -> None:
    for _, text, _ in HOSTILE_TEXTS:
        once = errors._safe(text)
        assert errors._safe(once) == once


def test_candidate_lists_are_bounded_and_deduplicated() -> None:
    listed = errors._names(tuple(f"t{index}" for index in range(40)))
    assert listed.startswith('"t0", "t1"') and listed.endswith("(+32 more)")
    assert errors._candidates(("accounts", "accounts", "customers")) == ("accounts", "customers")


def test_sequence_arguments_are_accepted_as_any_sequence() -> None:
    known: Sequence[str] = ["accounts", "customers"]
    assert errors.unknown_table(name="account", known=known).did_you_mean == ("accounts",)
