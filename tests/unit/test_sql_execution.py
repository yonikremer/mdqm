"""The execution chokepoint (:mod:`mdqm.sql.execution`) refuses *at the driver boundary*.

``tests/unit/test_sql_injection_protection.py`` proves the guard refuses the injection payloads.
This file proves the other half: that the refusal happens at the chokepoint -- after a statement was
built, immediately before it would be sent -- and that nothing at all is handed to the driver when it
does. The spy cursor counts every ``execute`` call, so a refusal that still executed would fail here
even though the exception was raised.
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

#: The three shapes a statement injection takes, each reaching the driver as a *compiled fragment*
#: would: a write, a second statement behind a `;`, and a denylisted server function.
REFUSED_CASES: tuple[tuple[str, str, str], ...] = (
    ("write", "DELETE FROM customers WHERE cust_id = 1", WRITE_BLOCKED),
    ("stacked", "SELECT 1 FROM dual; DELETE FROM customers WHERE 1 = 1", MULTI_STATEMENT),
    ("denied_function", "SELECT pg_sleep(10)", WRITE_BLOCKED),
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
def test_a_refused_statement_raises_before_anything_is_sent(
    case_id: str, statement: str, code: str
) -> None:
    """The guard's own code reaches the caller, with its fix, and no cursor is touched."""
    cursor = SpyCursor()

    with pytest.raises(StatementRefused) as caught:
        run(cursor, prepare(statement, dialect="postgres"))

    refused = caught.value
    assert refused.code == code, case_id
    assert refused.result.ok is False
    assert refused.result.error is not None
    assert refused.result.error.fix
    assert cursor.calls == [], f"{case_id} reached the driver"


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
    """``prefix`` exists for EXPLAIN; a write behind a command word is still refused."""
    with pytest.raises(StatementRefused) as caught:
        prepare("DELETE FROM customers", dialect="postgres", prefix="EXPLAIN (FORMAT JSON) ")

    assert caught.value.code == WRITE_BLOCKED


def test_the_prefix_is_added_after_the_guard_inspected_the_statement() -> None:
    """The guarded text is the fragment; the prefix is the server command around it."""
    prepared = prepare("SELECT 1", dialect="postgres", prefix="EXPLAIN (FORMAT JSON) ")

    assert prepared.text == "EXPLAIN (FORMAT JSON) SELECT 1"
    assert prepared.dialect == "postgres"


def test_local_sql_writes_and_is_not_guarded() -> None:
    """Our own metadata SQL writes on purpose: the guard would rightly refuse it."""
    statement = local("DELETE FROM query_history WHERE id = ?", ("q1",))

    assert isinstance(statement, LocalStatement)
    assert statement.text == "DELETE FROM query_history WHERE id = ?"
    assert statement.params == ("q1",)


def test_the_two_doors_stay_apart() -> None:
    """A source statement cannot be passed to ``run_local``, or a local one to ``run``."""
    assert not isinstance(prepare("SELECT 1", dialect="postgres"), LocalStatement)
    assert isinstance(prepare("SELECT 1", dialect="postgres"), PreparedStatement)

    cursor = SpyCursor()
    run_local(cursor, local("INSERT INTO query_history (id) VALUES (?)", ("q1",)))

    assert cursor.calls == [("INSERT INTO query_history (id) VALUES (?)", ("q1",))]
