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


def run(cursor: Any, statement: PreparedStatement) -> Any:
    """Execute a prepared *source* statement.

    The only function in the package that hands dynamic source SQL to a driver. A statement cannot
    arrive here unguarded: :class:`PreparedStatement` is only produced by :func:`prepare`.
    """
    return cursor.execute(statement.text, statement.params or None)


def run_local(
    cursor: Any, text: str, params: Mapping[str, Any] | Sequence[Any] | None = None
) -> Any:
    """Execute SQL against *our own* metadata database.

    Not guarded: this SQL writes, and it belongs to this repository rather than to a source. Values
    are bound exactly as they are for a source, and callers pass a literal (or a module constant) as
    the text -- never anything assembled from user input.
    """
    return cursor.execute(text, params)
