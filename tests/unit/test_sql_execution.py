"""Tests for the execution chokepoint (:mod:`mdqm.sql.execution`).

The chokepoint is the last thing between a compiled statement and a source, so these tests prove
the refusal happens *there* -- after the statement was built, immediately before it would be sent --
and that nothing is handed to the driver when it does.

The spy cursor counts every ``execute`` call. A refusal that still executed would show up as one
call, which is why the assertion is on the count and not only on the exception.
"""

from __future__ import annotations

from typing import Any

import pytest

from mdqm.sql.execution import (
    LocalStatement,
    PreparedStatement,
    StatementRefused,
    local,
    prepare,
    run,
    run_local,
)
from mdqm.sql.guard import MULTI_STATEMENT, WRITE_BLOCKED

#: (case id, statement, expected guard code). Each one is valid SQL that a compiler bug, a
#: hand-written fragment or a future code path could plausibly produce.
REFUSED_CASES: tuple[tuple[str, str, str], ...] = (
    ("delete", "DELETE FROM customers WHERE cust_id = 1", WRITE_BLOCKED),
    ("update", "UPDATE customers SET status = 'A'", WRITE_BLOCKED),
    ("insert", "INSERT INTO customers (cust_id) VALUES (1)", WRITE_BLOCKED),
    ("truncate", "TRUNCATE customers", WRITE_BLOCKED),
    ("drop", "DROP TABLE customers", WRITE_BLOCKED),
    # A second statement after a `;` is how a stacked write is smuggled onto a read.
    (
        "stacked",
        "SELECT 1 FROM dual; DELETE FROM customers WHERE 1 = 1",
        MULTI_STATEMENT,
    ),
    ("stacked_comment", "SELECT 1; -- and then\nDELETE FROM customers", MULTI_STATEMENT),
    # Denylisted functions: they read files, sleep or write state. The guard's own list decides,
    # so the case asserts a refusal, not a particular spelling of it.
    ("denied_file_read", "SELECT pg_read_file('/etc/passwd')", WRITE_BLOCKED),
    ("denied_sleep", "SELECT pg_sleep(10)", WRITE_BLOCKED),
    ("denied_nextval", "SELECT nextval('customers_cust_id_seq')", WRITE_BLOCKED),
)


class SpyCursor:
    """A cursor that records what it was asked to run, and never talks to a server."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def execute(self, text: str, params: Any = None) -> SpyCursor:
        self.calls.append((text, params))
        return self


@pytest.mark.parametrize(
    ("case_id", "statement", "code"), REFUSED_CASES, ids=[case[0] for case in REFUSED_CASES]
)
def test_prepare_refuses_at_execution_time(case_id: str, statement: str, code: str) -> None:
    """The guard runs when the statement is about to run, and its code reaches the caller."""
    with pytest.raises(StatementRefused) as caught:
        prepare(statement, dialect="postgres")

    refused = caught.value
    assert refused.code == code, case_id
    assert refused.result.ok is False
    # The refusal carries the fix, so an agent gets something it can act on.
    assert refused.result.error is not None
    assert refused.result.error.fix


@pytest.mark.parametrize(
    ("case_id", "statement"),
    [
        ("delete", "DELETE FROM customers WHERE cust_id = 1"),
        ("stacked", "SELECT 1 FROM dual; DELETE FROM customers WHERE 1 = 1"),
        ("denied_sleep", "SELECT pg_sleep(10)"),
    ],
    ids=["delete", "stacked", "denied_sleep"],
)
def test_refused_statement_never_reaches_the_driver(case_id: str, statement: str) -> None:
    """Nothing is sent: the refusal is raised, so no cursor is touched at all."""
    cursor = SpyCursor()

    with pytest.raises(StatementRefused):
        run(cursor, prepare(statement, dialect="postgres"))

    assert cursor.calls == [], f"{case_id} was sent to the driver"


def test_a_read_reaches_the_driver_once_through_the_chokepoint() -> None:
    """The happy path: one statement, values bound, nothing interpolated."""
    cursor = SpyCursor()

    returned = run(
        cursor,
        prepare(
            "SELECT cust_id FROM customers WHERE status = %(status)s",
            dialect="postgres",
            params={"status": "A"},
        ),
    )

    assert returned is cursor
    assert cursor.calls == [
        ("SELECT cust_id FROM customers WHERE status = %(status)s", {"status": "A"})
    ]


def test_an_unbound_read_passes_no_parameters() -> None:
    """psycopg wants ``None``, not ``()``, when a statement takes no value."""
    cursor = SpyCursor()

    run(cursor, prepare("SELECT 1", dialect="postgres"))

    assert cursor.calls == [("SELECT 1", None)]


def test_a_prefix_cannot_launder_a_write() -> None:
    """``prefix`` exists for EXPLAIN; it must not let a write through behind a command word."""
    with pytest.raises(StatementRefused) as caught:
        prepare(
            "DELETE FROM customers",
            dialect="postgres",
            prefix="EXPLAIN (FORMAT JSON) ",
        )

    assert caught.value.code == WRITE_BLOCKED


def test_the_prefix_is_added_after_the_guard_inspected_the_statement() -> None:
    """The guarded text is the fragment; the prefix is the server command around it."""
    prepared = prepare("SELECT 1", dialect="postgres", prefix="EXPLAIN (FORMAT JSON) ")

    assert prepared.text == "EXPLAIN (FORMAT JSON) SELECT 1"
    assert prepared.dialect == "postgres"


def test_a_prepared_statement_cannot_be_built_by_hand_into_run_local() -> None:
    """The two doors stay apart: a source statement is not a local statement and vice versa."""
    assert not isinstance(prepare("SELECT 1", dialect="postgres"), LocalStatement)

    cursor = SpyCursor()
    run_local(cursor, local("INSERT INTO query_history (id) VALUES (?)", ("q1",)))

    assert cursor.calls == [("INSERT INTO query_history (id) VALUES (?)", ("q1",))]


def test_local_sql_writes_and_is_not_guarded() -> None:
    """Our own metadata SQL writes on purpose; the guard would rightly refuse it."""
    statement = local("DELETE FROM query_history WHERE id = ?", ("q1",))

    assert isinstance(statement, LocalStatement)
    assert statement.text == "DELETE FROM query_history WHERE id = ?"
    assert statement.params == ("q1",)


def test_run_rejects_nothing_but_takes_only_prepared_statements() -> None:
    """``run`` cannot be handed a raw string: the type is the guarantee, not the comment."""
    assert isinstance(prepare("SELECT 1", dialect="postgres"), PreparedStatement)
