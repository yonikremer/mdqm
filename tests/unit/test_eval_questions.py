"""The Phase-0 question set and its ground truth (task E01).

`tests/eval/questions.yaml` holds 15 analyst questions with a Hebrew and an English phrasing each,
the canonical spec an agent should send, and the expected answer as data. This module proves the
fixture is worth trusting:

* **shape**: 15 questions, unique ids, 8 Hebrew and 7 English, both phrasings present and in the
  right script, and the difficulty counts the task asks for (>=3 conversion, >=3 value map,
  >=2 insertion time);
* **validation**: every `spec` passes :func:`mdqm.spec.parse_spec`, and the answer's columns are the
  names that spec produces;
* **two independent derivations of every answer, neither from the code under test**:
  - :func:`test_answers_match_duckdb_oracle` runs the hand-written ``oracle_sql`` over the three
    storage forms the world writes, using explicit per-source normalisation (no compiler, no
    executor, no semantic layer);
  - :func:`test_answers_match_world_ground_truth` recomputes the same answers in plain Python from
    the world's rows and its ``expected``/``World.expected`` ground truth (task D02);
  If those two disagree with each other, one of them is wrong and the question is not trustworthy;
* **refusals**: the partition question really has no window on the partition key, and the SQL
  statement in the write question really is refused by the read-only guard.

Executing the *spec* itself is task E02's job (the harness drives the MCP server, and the compiler
does not exist yet). What is frozen here is the answer: `oracle_sql` is what that spec means,
written by hand so a compiler bug cannot silently redefine the expected result.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pytest
from ruamel.yaml import YAML

from mdqm.errors import ErrorCode
from mdqm.models import ErrorResponse
from mdqm.spec import QuerySpec, parse_spec
from mdqm.sql.guard import WRITE_BLOCKED, guard
from mdqm.testing.world import (
    GroundTruth,
    World,
    WorldConfig,
    build_world,
    round_half_up,
    world_hash,
)

QUESTIONS_FILE = Path(__file__).resolve().parents[1] / "eval" / "questions.yaml"

#: Storage-form code -> canonical value, spelled out here on purpose. The value map lives in the
#: semantic layer (the code under test); the fixture must not borrow it.
CRM_STATUS = {"A": "active", "C": "churned", "S": "suspended"}
BILLING_STATE = {"Active": "active", "Closed": "churned", "Hold": "suspended"}
ERP_CODE = {1: "active", 2: "churned", 3: "suspended"}

#: Priorities a question may carry. Closed on purpose: a typo in a tag must fail, not pass silently.
DIFFICULTY_TAGS = frozenset(
    {
        "single_table",
        "cross_source",
        "needs_conversion",
        "needs_value_map",
        "needs_partition_pruning",
        "needs_insertion_time",
        "needs_join",
        "expected_refusal",
    }
)

#: Task E01 requires at least this many questions per tag.
REQUIRED_MINIMUMS = {"needs_conversion": 3, "needs_value_map": 3, "needs_insertion_time": 2}

#: 8 Hebrew and 7 English questions: the mix the task asks for, so a new question keeps it honest.
INITIAL_LANGUAGES = {"he": 8, "en": 7}

HEBREW_LETTERS = range(0x0590, 0x05EB)


# --------------------------------------------------------------------------------------
# Fixtures: the document, the world it pins, and the storage forms in DuckDB.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Question:
    """One question record, typed, so the assertions below read like the YAML they check."""

    id: str
    language: str
    difficulty: tuple[str, ...]
    text: Mapping[str, str]
    answer: Mapping[str, Any]
    spec: Mapping[str, Any] | None
    statement: Mapping[str, Any] | None
    oracle_sql: str | None
    refusal_check: Mapping[str, Any] | None
    note: str


def read_questions() -> tuple[Mapping[str, Any], tuple[Question, ...]]:
    """Load the fixture, keeping every field a test needs and rejecting nothing here."""
    yaml = YAML(typ="safe")
    document: Mapping[str, Any] = yaml.load(QUESTIONS_FILE.read_text(encoding="utf-8"))
    questions = tuple(
        Question(
            id=entry["id"],
            language=entry["language"],
            difficulty=tuple(entry["difficulty"]),
            text=entry["text"],
            answer=entry["answer"],
            spec=entry.get("spec"),
            statement=entry.get("statement"),
            oracle_sql=entry.get("oracle_sql"),
            refusal_check=entry.get("refusal_check"),
            note=entry.get("note", ""),
        )
        for entry in document["questions"]
    )
    return document, questions


DOCUMENT, QUESTIONS = read_questions()
BY_ID = {question.id: question for question in QUESTIONS}


def build_pinned_world() -> World:
    """Build exactly the world the fixture pins, so the answers belong to these bytes."""
    world = DOCUMENT["world"]
    return build_world(
        WorldConfig(
            seed=world["seed"],
            customers=world["customers"],
            months=world["months"],
            events_per_customer=world["events_per_customer"],
            anchor=datetime.fromisoformat(world["anchor"]),
            max_lag_seconds=world["max_lag_seconds"],
        )
    )


def utc_naive(moment: datetime) -> datetime:
    """DuckDB's TIMESTAMP has no zone; the world is UTC, so dropping it loses nothing."""
    return moment.astimezone(UTC).replace(tzinfo=None)


def load_table(
    connection: duckdb.DuckDBPyConnection, name: str, columns: Mapping[str, Sequence[Any]]
) -> None:
    """Bulk-load one storage form through Arrow: ``executemany`` per row would take ~30 s here."""
    temporary = f"_{name.replace('.', '_')}"
    connection.register(temporary, pa.table(dict(columns)))
    connection.execute(f"CREATE TABLE {name} AS SELECT * FROM {temporary}")
    connection.unregister(temporary)


def load_storage_forms(world: World) -> duckdb.DuckDBPyConnection:
    """Write the world's rows into DuckDB under the source names the questions use.

    This is the eval's own loader: it reads the frozen dataclasses and touches no adapter.
    """
    connection = duckdb.connect()
    for schema in ("crm", "billing", "erp"):
        connection.execute(f"CREATE SCHEMA {schema}")
    load_table(
        connection,
        "crm.customers",
        {
            "cust_id": [row.cust_id for row in world.crm.customers],
            "full_name": [row.full_name for row in world.crm.customers],
            "status": [row.status for row in world.crm.customers],
            "credit_limit": [row.credit_limit for row in world.crm.customers],
            "region": [row.region for row in world.crm.customers],
            "signup_date": [row.signup_date for row in world.crm.customers],
        },
    )
    load_table(
        connection,
        "billing.accounts",
        {
            "client_no": [row.client_no for row in world.billing.accounts],
            "state": [row.state for row in world.billing.accounts],
            "credit_limit": [row.credit_limit for row in world.billing.accounts],
            "region": [row.region for row in world.billing.accounts],
            "opened_date": [row.opened_date for row in world.billing.accounts],
        },
    )
    load_table(
        connection,
        "billing.usage_events",
        {
            "event_id": [row.event_id for row in world.billing.usage_events],
            "client_no": [row.client_no for row in world.billing.usage_events],
            "start_time": [utc_naive(row.start_time) for row in world.billing.usage_events],
            "insertion_time": [utc_naive(row.insertion_time) for row in world.billing.usage_events],
            "units": [row.units for row in world.billing.usage_events],
            "corrects_event_id": [row.corrects_event_id for row in world.billing.usage_events],
        },
    )
    load_table(
        connection,
        "billing.accounts_hist",
        {
            "hist_id": [row.hist_id for row in world.billing.accounts_hist],
            "client_no": [row.client_no for row in world.billing.accounts_hist],
            "state": [row.state for row in world.billing.accounts_hist],
            "credit_limit": [row.credit_limit for row in world.billing.accounts_hist],
            "updated_at": [utc_naive(row.updated_at) for row in world.billing.accounts_hist],
        },
    )
    load_table(
        connection,
        "erp.customers",
        {
            "cust_no": [row.cust_no for row in world.erp.customers],
            "status_code": [row.status_code for row in world.erp.customers],
            "limit_k": [row.limit_k for row in world.erp.customers],
            "name": [row.name for row in world.erp.customers],
        },
    )
    return connection


@pytest.fixture(scope="module")
def world() -> World:
    return build_pinned_world()


@pytest.fixture(scope="module")
def truth(world: World) -> GroundTruth:
    return world.expected


@pytest.fixture(scope="module")
def duck(world: World) -> Iterator[duckdb.DuckDBPyConnection]:
    connection = load_storage_forms(world)
    try:
        yield connection
    finally:
        connection.close()


# --------------------------------------------------------------------------------------
# Comparing an answer to a computation.
# --------------------------------------------------------------------------------------


def assert_same_rows(
    question_id: str, answer: Mapping[str, Any], actual: Sequence[Sequence[Any]]
) -> None:
    """Compare ``actual`` with the stored rows, per cell, with the fixture's float tolerance."""
    tolerance = DOCUMENT["float_tolerance"]
    expected_rows = answer["rows"]
    assert len(actual) == len(expected_rows), f"{question_id}: row count {len(actual)}"
    for index, (expected_row, actual_row) in enumerate(zip(expected_rows, actual, strict=True)):
        assert len(actual_row) == len(answer["columns"]), f"{question_id}: row {index} width"
        for column, expected_cell, actual_cell in zip(
            answer["columns"], expected_row, actual_row, strict=True
        ):
            where = f"{question_id}: row {index}, column {column}"
            if isinstance(expected_cell, (bool, str)):
                assert actual_cell == expected_cell, where
            elif isinstance(expected_cell, int):
                assert int(actual_cell) == expected_cell, where
            else:
                assert math.isclose(
                    float(actual_cell), float(expected_cell), rel_tol=tolerance, abs_tol=tolerance
                ), f"{where}: {actual_cell} != {expected_cell}"


def expected_columns(spec: QuerySpec) -> list[str]:
    """The output names a spec produces: unselected group keys first, then the select list."""
    selected = spec.selected_names()
    return [key for key in spec.group_by if key not in selected] + list(selected)


# --------------------------------------------------------------------------------------
# 1. The fixture's own shape.
# --------------------------------------------------------------------------------------


def test_fifteen_questions_with_unique_ids() -> None:
    assert len(QUESTIONS) == 15
    assert len({question.id for question in QUESTIONS}) == 15


def test_language_mix_and_scripts() -> None:
    counted: dict[str, int] = {}
    for question in QUESTIONS:
        counted[question.language] = counted.get(question.language, 0) + 1

        assert set(question.text) == {"he", "en"}, question.id
        hebrew, english = question.text["he"], question.text["en"]
        assert hebrew.strip() and english.strip(), question.id
        assert any(ord(character) in HEBREW_LETTERS for character in hebrew), question.id
        assert not any(ord(character) in HEBREW_LETTERS for character in english), question.id
        # One record, one answer: the two phrasings cannot drift apart, they are the same question.
        assert question.answer, question.id
    assert counted == INITIAL_LANGUAGES


def test_difficulty_tags_and_required_counts() -> None:
    counted: dict[str, int] = {}
    for question in QUESTIONS:
        assert question.difficulty, question.id
        assert set(question.difficulty) <= DIFFICULTY_TAGS, question.id
        assert len(set(question.difficulty)) == len(question.difficulty), question.id
        for tag in question.difficulty:
            counted[tag] = counted.get(tag, 0) + 1
    for tag, minimum in REQUIRED_MINIMUMS.items():
        assert counted.get(tag, 0) >= minimum, f"{tag}: {counted.get(tag, 0)} < {minimum}"


def test_exactly_one_of_spec_and_statement() -> None:
    for question in QUESTIONS:
        assert (question.spec is None) != (question.statement is None), question.id


def test_answer_shapes() -> None:
    for question in QUESTIONS:
        kind = question.answer["kind"]
        assert kind in {"rows", "error"}, question.id
        if kind == "rows":
            assert question.answer["columns"], question.id
            assert question.answer["rows"] is not None, question.id
            assert question.oracle_sql, f"{question.id}: a rows answer needs an oracle_sql"
            assert "expected_refusal" not in question.difficulty, question.id
        else:
            assert question.answer["code"] in set(ErrorCode), question.id
            assert question.answer["problem"] and question.answer["fix"], question.id
            assert "expected_refusal" in question.difficulty, question.id
            assert question.oracle_sql is None, f"{question.id}: a refusal has nothing to compute"


def test_the_mvp_scenarios_are_present() -> None:
    """The five situations the MVP acceptance calls out must be in the set, by id."""
    assert BY_ID["avg_limit_by_status_crm"].answer["kind"] == "rows"
    assert "needs_conversion" in BY_ID["avg_limit_by_status_erp"].difficulty
    assert "needs_insertion_time" in BY_ID["new_since_resync"].difficulty
    assert BY_ID["usage_events_all_partitions_refused"].answer["code"] == "PARTITION_SCAN_ALL"
    assert BY_ID["delete_usage_events"].answer["code"] == "WRITE_BLOCKED"
    assert BY_ID["usage_units_by_status_window"].spec is not None
    assert "cross_source" in BY_ID["usage_units_by_status_window"].difficulty


# --------------------------------------------------------------------------------------
# 2. Every spec validates, and the answer speaks the spec's language.
# --------------------------------------------------------------------------------------


def test_every_spec_validates() -> None:
    for question in QUESTIONS:
        if question.spec is None:
            assert question.statement is not None
            assert question.statement["kind"] == "sql", question.id
            assert question.statement["sql"].strip(), question.id
            continue
        spec = parse_spec(question.spec)
        assert isinstance(spec, QuerySpec), question.id
        if question.answer["kind"] == "rows":
            assert list(question.answer["columns"]) == expected_columns(spec), question.id


def test_count_of_usage_units_is_a_row_count(world: World) -> None:
    """Events questions count rows as `count(usage_units)`: only true while units is never NULL."""
    assert all(event.units is not None and event.units >= 1 for event in world.billing.usage_events)
    assert all(customer.credit_limit is not None for customer in world.crm.customers)


def test_the_world_is_the_pinned_one(world: World) -> None:
    assert world_hash(world) == DOCUMENT["world"]["sha256"]
    pinned = DOCUMENT["world"]
    assert len(world.crm.customers) == pinned["customers"]
    assert world.expected.max_lag_seconds == pinned["max_lag_seconds"]


# --------------------------------------------------------------------------------------
# 3. First independent derivation: hand-written SQL over the storage forms, on DuckDB.
# --------------------------------------------------------------------------------------


def test_answers_match_duckdb_oracle(duck: duckdb.DuckDBPyConnection) -> None:
    computed = 0
    for question in QUESTIONS:
        if question.answer["kind"] != "rows":
            continue
        assert question.oracle_sql is not None
        actual = duck.execute(question.oracle_sql).fetchall()
        assert_same_rows(question.id, question.answer, actual)
        computed += 1
    assert computed == 13, "every rows answer must be exercised"


# --------------------------------------------------------------------------------------
# 4. Second independent derivation: plain Python over the world's own rows and ground truth.
# --------------------------------------------------------------------------------------


def group_average(pairs: Mapping[str, list[int]]) -> list[list[Any]]:
    """``{key: values}`` -> ``[[key, mean], ...]``, sorted by key."""
    return [[key, sum(values) / len(values)] for key, values in sorted(pairs.items())]


def independent_answers(world: World) -> dict[str, list[list[Any]]]:
    """Recompute the rows answers from the world, with no DuckDB and no mdqm query code.

    The three credit-limit questions intentionally start from a different place for each source:
    the canonical truth for crm (which stores it), the raw rows for billing (which must be rounded
    and has NULLs), and the erp rows (which must be deduplicated and multiplied by 10).
    """
    truth = world.expected

    crm: dict[str, list[int]] = {}
    regions: dict[str, int] = {}
    for customer in truth.customers:
        crm.setdefault(customer.canonical_status, []).append(customer.canonical_limit)
        regions[customer.region] = regions.get(customer.region, 0) + 1

    billing: dict[str, list[int]] = {}
    billing_regions: dict[str, list[int]] = {}
    for account in world.billing.accounts:
        if account.credit_limit is None:
            continue
        rounded = round_half_up(account.credit_limit)
        billing.setdefault(BILLING_STATE[account.state], []).append(rounded)
        billing_regions.setdefault(account.region, []).append(rounded)

    erp: dict[str, list[int]] = {}
    seen: set[int] = set()
    for row in world.erp.customers:
        if row.cust_no in seen:
            continue
        seen.add(row.cust_no)
        erp.setdefault(ERP_CODE[row.status_code], []).append(row.limit_k * 10)

    status_by_client = {
        customer.client_no: customer.canonical_status for customer in truth.customers
    }
    units_by_status: dict[str, int] = {}
    december: dict[str, int] = {}
    quarter_events = 0
    new_since_events = 0
    new_since_units = 0
    known_as_of = 0
    quarter_start = datetime(2025, 10, 1, tzinfo=UTC)
    november = datetime(2025, 11, 1, tzinfo=UTC)
    december_start = datetime(2025, 12, 1, tzinfo=UTC)
    mid_november = datetime(2025, 11, 15, tzinfo=UTC)
    known_at = datetime(2025, 12, 15, tzinfo=UTC)
    anchor = world.config.anchor
    for event in world.billing.usage_events:
        if november <= event.start_time < anchor:
            status = status_by_client.get(event.client_no)
            if status is not None:  # the join drops orphan client numbers
                units_by_status[status] = units_by_status.get(status, 0) + event.units
        if quarter_start <= event.start_time < anchor:
            quarter_events += 1
        if event.insertion_time >= mid_november:
            new_since_units += event.units
        if event.insertion_time >= december_start:
            new_since_events += 1
        if december_start <= event.start_time < anchor:
            december[event.client_no] = december.get(event.client_no, 0) + event.units
            if event.insertion_time <= known_at:
                known_as_of += 1

    top = sorted(december.items(), key=lambda item: (-item[1], item[0]))[:5]
    return {
        "avg_limit_by_status_crm": group_average(crm),
        "avg_limit_by_status_billing": group_average(billing),
        "avg_limit_by_status_erp": group_average(erp),
        "avg_limit_by_region_billing": group_average(billing_regions),
        "sum_limit_by_status_erp": [[key, sum(values)] for key, values in sorted(erp.items())],
        "usage_units_by_status_window": [
            [key, value] for key, value in sorted(units_by_status.items())
        ],
        "events_count_window": [[quarter_events]],
        "new_since_resync": [[new_since_events]],
        "units_new_since": [[new_since_units]],
        "known_as_of_snapshot": [[known_as_of]],
        "churned_customers_hebrew": [
            [sum(1 for customer in truth.customers if customer.canonical_status == "churned")]
        ],
        "customers_by_region_crm": [[region, count] for region, count in sorted(regions.items())],
        "top_clients_by_usage_december": [[client, units] for client, units in top],
    }


def test_answers_match_world_ground_truth(world: World) -> None:
    recomputed = independent_answers(world)
    rows_questions = {q.id for q in QUESTIONS if q.answer["kind"] == "rows"}
    assert set(recomputed) == rows_questions, "every rows question needs a second derivation"
    for question in QUESTIONS:
        if question.answer["kind"] != "rows":
            continue
        assert_same_rows(question.id, question.answer, recomputed[question.id])


def test_the_two_derivations_cover_every_question() -> None:
    """Guard the guard: one derivation for a question means one implementation is trusted blind."""
    for question in QUESTIONS:
        if question.answer["kind"] == "rows":
            assert question.oracle_sql
        else:
            assert question.refusal_check, f"{question.id}: a refusal needs a precondition check"


# --------------------------------------------------------------------------------------
# 5. The refusals are real.
# --------------------------------------------------------------------------------------


def test_partition_refusal_precondition() -> None:
    question = BY_ID["usage_events_all_partitions_refused"]
    assert question.spec is not None and question.refusal_check is not None
    check = question.refusal_check
    assert check["kind"] == "no_partition_key_filter"
    partition_key: str = check["partition_key"]

    spec = parse_spec(question.spec)
    assert spec.from_table == check["table"]
    assert spec.time is None, "a window would make the scan safe, so the refusal would be wrong"
    assert all(entry.column != partition_key for entry in spec.filters)
    assert question.answer["code"] == ErrorCode.PARTITION_SCAN_ALL


def test_write_statement_is_blocked() -> None:
    question = BY_ID["delete_usage_events"]
    assert question.statement is not None and question.refusal_check is not None
    assert question.refusal_check["kind"] == "write_statement"
    result = guard(question.statement["sql"], "postgres")
    assert not result.ok
    assert result.ast is None
    error: ErrorResponse | None = result.error
    assert error is not None
    assert error.code == question.answer["code"] == WRITE_BLOCKED
    # The refusal keeps the same shape the fixture promises an agent: code + problem + fix.
    assert error.problem.strip() and error.fix.strip()
