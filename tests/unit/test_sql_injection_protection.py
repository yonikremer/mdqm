"""Read-only SQL injection: the cases that must be refused, and the value that must stay a value.

The product runs SQL that this codebase compiles, plus SQL built from what an agent asks for. There are
two distinct risks and they need two distinct answers:

1. **Statement injection** — an attempt to turn one read into a write, a second statement, or a call to a
   server-side function that reads files or sleeps. The AST guard must refuse it *before* it reaches a
   driver, and it must refuse it on the parsed tree, not on the text (a comment or a string literal
   containing `;` is not a second statement).
2. **Value injection** — a value that contains SQL syntax. It must never be interpolated; it travels as
   a bound parameter and comes back as data.

These are the tests that make the claim checkable. They are deliberately adversarial: every payload here
is one a hostile agent would actually try.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from mdqm.sql.execution import (
    PreparedStatement,
    StatementRefused,
    local,
    prepare,
    run,
    run_local,
)
from mdqm.sql.guard import guard

POSTGRES = "postgres"


@pytest.fixture
def local_db() -> Iterator[sqlite3.Connection]:
    """A scratch database with one table, so a successful injection would be visible."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE victim (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO victim (id, name) VALUES (1, 'keep me')")
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


# --- 1. statement injection is refused by the guard ---------------------------------------

WRITES = (
    "INSERT INTO accounts (client_no) VALUES ('x')",
    "UPDATE accounts SET state = 'Hold'",
    "DELETE FROM accounts",
    "DROP TABLE accounts",
    "TRUNCATE TABLE accounts",
    "CREATE TABLE evil (x INTEGER)",
    "ALTER TABLE accounts ADD COLUMN evil INTEGER",
    "GRANT ALL ON accounts TO PUBLIC",
    "COPY accounts TO '/tmp/dump.csv'",
    "VACUUM FULL accounts",
)

STACKED = (
    "SELECT 1; DROP TABLE accounts",
    "SELECT 1; DELETE FROM accounts",
    "SELECT 1 /* leading */ ; DROP TABLE accounts",
    "SELECT 1 WHERE 1=1; UPDATE accounts SET state = 'Hold'",
)

DANGEROUS_FUNCTIONS = (
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT pg_ls_dir('.')",
    "SELECT pg_sleep(10)",
    "SELECT lo_import('/etc/passwd')",
)


@pytest.mark.parametrize("statement", WRITES)
def test_a_write_is_refused(statement: str) -> None:
    result = guard(statement, POSTGRES)
    assert not result.ok, f"{statement!r} must be refused"
    assert result.error is not None and result.error.fix


@pytest.mark.parametrize("statement", STACKED)
def test_a_stacked_statement_is_refused(statement: str) -> None:
    """The second statement is the attack; parsing is what catches it, not a `;` scan."""
    result = guard(statement, POSTGRES)
    assert not result.ok, f"{statement!r} must be refused"


@pytest.mark.parametrize("statement", DANGEROUS_FUNCTIONS)
def test_a_file_reading_or_sleeping_function_is_refused(statement: str) -> None:
    result = guard(statement, POSTGRES)
    assert not result.ok, f"{statement!r} must be refused"


def test_a_semicolon_inside_a_string_or_comment_is_not_a_second_statement() -> None:
    """A text scan would refuse this and be useless; the parser knows better, and that is the point."""
    for innocuous in (
        "SELECT '; DROP TABLE accounts' AS note",
        "SELECT 1 /* ; DROP TABLE accounts */ AS note",
        "SELECT 1 -- ; DROP TABLE accounts",
    ):
        assert guard(innocuous, POSTGRES).ok, f"{innocuous!r} is one read statement"


def test_a_read_is_allowed() -> None:
    result = guard("SELECT cust_id, credit_limit FROM customers", POSTGRES)
    assert result.ok
    assert result.ast is not None, "an allowed statement must carry its parsed tree"


# --- 2. a value never becomes SQL ---------------------------------------------------------


HOSTILE_VALUES = (
    "'; DROP TABLE victim; --",
    "' OR 1=1 --",
    "1; DELETE FROM victim",
    '"; DROP TABLE victim; --',
    "' UNION SELECT name FROM victim --",
    "\\'; DROP TABLE victim; --",
)


@pytest.mark.parametrize("hostile", HOSTILE_VALUES)
def test_a_hostile_value_comes_back_as_data(local_db: sqlite3.Connection, hostile: str) -> None:
    """The payload must round-trip as a string and leave the table alone."""
    cur = local_db.cursor()
    before = local_db.execute("SELECT count(*) FROM victim").fetchone()[0]

    result = run_local(cur, local("SELECT ? AS echoed", (hostile,)))
    echoed = result.fetchone()[0]

    after = local_db.execute("SELECT count(*) FROM victim").fetchone()[0]
    assert echoed == hostile, "a bound value must come back byte-for-byte"
    assert after == before, "the value must not have reached the parser as SQL"
    assert local_db.execute("SELECT name FROM victim WHERE id = 1").fetchone()[0] == "keep me"


def test_the_chokepoint_refuses_an_unguarded_statement_before_execution(
    local_db: sqlite3.Connection,
) -> None:
    """`prepare` is the only way to get a `PreparedStatement`, and it refuses a write."""
    with pytest.raises(StatementRefused) as caught:
        prepare("DELETE FROM victim", dialect=POSTGRES)
    assert caught.value.result.error is not None
    assert caught.value.result.error.code == "WRITE_BLOCKED"

    rows = local_db.execute("SELECT count(*) FROM victim").fetchone()[0]
    assert rows == 1, "nothing may have executed"


def test_a_prepared_statement_keeps_its_values_bound(local_db: sqlite3.Connection) -> None:
    """Even after preparation, the value travels as a parameter, not as text."""
    statement = prepare(
        "SELECT ? AS echoed",
        dialect=POSTGRES,
        params=("'; DROP TABLE victim; --",),
    )
    assert isinstance(statement, PreparedStatement)
    assert statement.params == ("'; DROP TABLE victim; --",)
    assert "DROP TABLE" not in statement.text, "the value must not be in the statement text"

    cur = local_db.cursor()
    assert run(cur, statement).fetchone()[0] == "'; DROP TABLE victim; --"
    assert local_db.execute("SELECT count(*) FROM victim").fetchone()[0] == 1


def test_preparation_does_not_mutate_the_callers_values() -> None:
    """A tuple in, a tuple out: a prepared statement is a value object, not a live handle."""
    values = ("a", "b")
    statement = prepare("SELECT ?, ?", dialect=POSTGRES, params=values)
    assert statement.params == values
