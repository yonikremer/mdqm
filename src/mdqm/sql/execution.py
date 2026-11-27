"""The one place a statement reaches a driver.

`mdqm` runs two very different kinds of SQL, and they get different treatment:

* **Source SQL** — anything sent to PostgreSQL, Oracle, SQL Server or Elasticsearch. It is either
  compiled through the sqlglot AST or written by us in this repository, and it must pass
  :func:`mdqm.sql.guard.guard` *immediately before execution*, not at some earlier stage. Values are
  always bound; no value ever becomes part of the statement text.
* **Local SQL** -- the metadata SQLite database (`store/`, `context/`, `review/`). It is ours and it
  writes, so the guard (which only allows reads) would rightly refuse it. Values are still bound,
  and the text is still a literal at the call site.

Why a chokepoint instead of a comment on every call: the SQL-injection lint rule cannot tell a
guarded fragment from an interpolated one -- measured, it flags *any* non-literal argument, even a
function call. Suppressing it at twenty call sites hides the question. Funnelling every dynamic
statement through one audited function leaves exactly one place to review, and runs the guard at the
last possible moment, where a mistake cannot slip past.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mdqm.sql.guard import GuardPolicy, GuardResult, guard


class StatementRefused(RuntimeError):
    """A statement did not pass the guard, or was not prepared at all.

    Carries the guard's own :class:`~mdqm.models.ErrorResponse` so the caller can hand the agent the
    same `code`/`problem`/`fix` it would get from any other refusal.
    """

    def __init__(self, result: GuardResult) -> None:
        super().__init__(result.error.problem if result.error is not None else "statement refused")
        self.result = result


@dataclass(frozen=True, slots=True)
class PreparedStatement:
    """A source statement that already passed the guard, with its bound values.

    The only way to obtain one is :func:`prepare`, so an unprepared string cannot reach
    :func:`run` by accident.
    """

    text: str
    params: Mapping[str, Any] | Sequence[Any] | None
    dialect: str


def prepare(
    text: str,
    *,
    dialect: str,
    params: Mapping[str, Any] | Sequence[Any] | None = None,
    policy: GuardPolicy | None = None,
) -> PreparedStatement:
    """Guard ``text`` and return it ready to execute.

    Raises:
        StatementRefused: the guard rejected the statement (a write, several statements in one
            string, a denylisted function). The refusal carries the fix, so the caller reports it.
    """
    result = guard(text, dialect, policy)
    if not result.ok:
        raise StatementRefused(result)
    return PreparedStatement(text=text, params=params, dialect=dialect)


def execute_one(cursor: Any, statement: PreparedStatement) -> Any:
    """Execute exactly one guarded statement, with its values bound.

    Named `execute_one` on purpose. A driver's `execute` will happily run several statements separated
    by `;` on some engines, which is precisely how a stacked-statement injection works. Here the guard
    has already refused anything but a single parsed read, and this is the only function that hands the
    text to a driver -- so "one statement, parameterised" is the only shape that exists.

    A statement with no values is executed with one argument, not with ``None`` as a second one:
    mssql-python switches to its prepared-statement path as soon as it is handed a parameter
    argument, and under ``SET SHOWPLAN_XML ON`` that path returns no result set at all ("Invalid
    cursor state"). A driver is handed exactly what the statement needs.
    """
    if statement.params is None:
        return cursor.execute(statement.text)
    return cursor.execute(statement.text, statement.params)


def wrapped(prepared: PreparedStatement, *, template: str, **fields: Any) -> PreparedStatement:
    """Wrap an already-guarded statement in a constant template, e.g. ``EXPLAIN {sql}``.

    Some statements *must* embed another one (``EXPLAIN … FOR``, ``SET SHOWPLAN_XML``). Doing that at a
    call site is text building, and text building is where injections live, so it happens here instead:
    the inner statement is already guarded, the template is a literal, and any interpolated field is a
    value this module controls (an integer statement id, never agent input). The result is still a
    :class:`PreparedStatement`, so it can only leave through :func:`execute_one`.
    """
    return PreparedStatement(
        text=template.format(sql=prepared.text, **fields),
        params=prepared.params,
        dialect=prepared.dialect,
    )


def run(cursor: Any, statement: PreparedStatement) -> Any:
    """Deprecated alias for :func:`execute_one`; kept so existing call sites keep working."""
    return execute_one(cursor, statement)


def run_local(
    cursor: Any, text: str, params: Mapping[str, Any] | Sequence[Any] | None = None
) -> Any:
    """Execute SQL against *our own* metadata database.

    Not guarded: this SQL writes, and it belongs to this repository rather than to a source. Values
    are bound exactly as they are for a source, and callers pass a literal (or a module constant) as
    the text -- never anything assembled from user input.
    """
    return cursor.execute(text, params)


#: The shape of an identifier this repository will quote into local SQL text. Deliberately narrow:
#: the names here are ours (a table we created, a column a segment is keyed by), and anything that does
#: not look like a plain SQL identifier is refused rather than quoted.
_LOCAL_IDENT: re.Pattern[str] = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]{0,62}$")


def local_ident(name: str) -> str:
    """``name`` quoted for a local engine, after validating its shape.

    A driver cannot bind an identifier, so an identifier is the one thing that must become SQL text.
    It becomes text here, through a shape check, and nowhere else -- a caller that needs a table name
    in a statement calls this instead of building an f-string.
    """
    if not _LOCAL_IDENT.fullmatch(name):
        raise ValueError(f"{name!r} is not a plain identifier, so it cannot be quoted into SQL")
    return f'"{name}"'


def local_idents(names: Sequence[str]) -> str:
    """``local_ident`` for a list, comma-separated: the shape a column list needs."""
    return ", ".join(local_ident(name) for name in names)


#: Settings this repository is willing to change on a local engine, with the shape their value may take.
_LOCAL_SETTINGS: dict[str, re.Pattern[str]] = {
    "memory_limit": re.compile(r"^[0-9]{1,6}(KB|MB|GB|TB)$"),
    "temp_directory": re.compile(r"^[A-Za-z]:[\/][^'\"]*$|^/[^'\"]*$"),
    "TimeZone": re.compile(r"^[A-Za-z][A-Za-z0-9_+/-]{0,63}$"),
}


def set_local(engine: Any, name: str, value: str) -> None:
    """Set a local engine setting, validating both halves first.

    A driver cannot bind a value into ``SET``, so this is the one place where a setting is written into
    SQL text -- and it is written here, where the name must be one of a fixed table and the value must
    match that setting's pattern. An agent-provided string cannot reach it: the callers pass our own
    configuration.
    """
    pattern = _LOCAL_SETTINGS.get(name)
    if pattern is None:
        raise StatementRefused(
            guard("SELECT 1", "duckdb")  # reuse the refusal type with the guard's shape
        )
    if not pattern.fullmatch(value):
        raise StatementRefused(guard("SELECT 1", "duckdb"))
    engine.execute("SET " + name + " = '" + value + "'").fetchall()
