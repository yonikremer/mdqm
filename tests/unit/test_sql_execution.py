"""The execution chokepoint (:mod:`mdqm.sql.execution`) refuses *at the driver boundary*.

``tests/unit/test_sql_injection_protection.py`` proves the guard refuses the injection payloads.
This file proves the other half: that the refusal happens at the chokepoint -- after a statement was
built, immediately before it would be sent -- and that nothing at all is handed to the driver when it
does. The spy cursor records every ``execute`` call, so a refusal that still executed would fail here
even though the exception was raised.
"""

from __future__ import annotations

from typing import Any

import pytest

from mdqm.sql.execution import PreparedStatement, StatementRefused, execute_one, prepare, wrapped
from mdqm.sql.guard import MULTI_STATEMENT, WRITE_BLOCKED

#: The three shapes a statement injection takes, each arriving the way a compiled fragment does: a
#: write, a second statement behind a `;`, and a denylisted server function.
REFUSED_CASES: tuple[tuple[str, str, str], ...] = (
    ("write", "DELETE FROM customers WHERE cust_id = 1", WRITE_BLOCKED),
    ("stacked", "SELECT 1 FROM dual; DELETE FROM customers WHERE 1 = 1", MULTI_STATEMENT),
    ("denied_function", "SELECT pg_sleep(10)", WRITE_BLOCKED),
)


class SpyCursor:
    """A cursor that records what it was asked to run, and never talks to a server."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def execute(self, *args: Any) -> SpyCursor:
        self.calls.append(args)
        return self


@pytest.mark.parametrize(
    ("case_id", "statement", "code"), REFUSED_CASES, ids=[case[0] for case in REFUSED_CASES]
)
def test_a_refused_statement_raises_before_anything_is_sent(
    case_id: str, statement: str, code: str
) -> None:
    """The guard's own code reaches the caller, with its fix, and no cursor is touched."""
    cursor = SpyCursor()

    with pytest.raises(StatementRefused) as caught:
        execute_one(cursor, prepare(statement, dialect="postgres"))

    refused = caught.value
    assert refused.result.ok is False
    assert refused.result.error is not None
    assert refused.result.error.code == code, case_id
    assert refused.result.error.fix
    assert cursor.calls == [], f"{case_id} reached the driver"


def test_a_read_reaches_the_driver_once_through_the_chokepoint() -> None:
    """The happy path: one statement, values bound, nothing interpolated."""
    cursor = SpyCursor()

    returned = execute_one(
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


def test_an_unbound_read_gets_one_argument() -> None:
    """A driver is handed what the statement needs and nothing more.

    mssql-python takes a different path as soon as it is given a parameter argument, and that path
    returns no result set under ``SET SHOWPLAN_XML ON``: no values means one argument.
    """
    cursor = SpyCursor()

    execute_one(cursor, prepare("SELECT 1", dialect="postgres"))

    assert cursor.calls == [("SELECT 1",)]


def test_a_template_cannot_launder_a_write() -> None:
    """``wrapped`` only ever wraps an already-guarded statement, so a write cannot get behind it."""
    with pytest.raises(StatementRefused) as caught:
        wrapped(
            prepare("DELETE FROM customers", dialect="postgres"),
            template="EXPLAIN (FORMAT JSON) {sql}",
        )

    assert caught.value.result.error is not None
    assert caught.value.result.error.code == WRITE_BLOCKED


def test_the_template_joins_the_command_and_the_guarded_statement() -> None:
    """``EXPLAIN {sql}``: the command word is a literal here, the statement was inspected above."""
    prepared = wrapped(
        prepare("SELECT 1", dialect="postgres"), template="EXPLAIN (FORMAT JSON) {sql}"
    )

    assert prepared.text == "EXPLAIN (FORMAT JSON) SELECT 1"
    assert isinstance(prepared, PreparedStatement)
