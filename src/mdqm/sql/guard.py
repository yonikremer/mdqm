"""The read-only SQL guard: the AST layer of the three-layer read-only defence.

Every statement the server sends to a source passes through :func:`guard` first, including the SQL
the server compiled itself (DESIGN section 2.4: read-only account + AST guard + read-only
transaction).
The guard is deliberately paranoid:

* exactly one statement, whose root is a ``SELECT``, a ``WITH ... SELECT`` or a ``UNION``;
* no DML, DDL, transaction control, ``COPY``, ``CALL``/``EXEC``, ``SELECT ... INTO``, ``FOR UPDATE``
  or locking table hint anywhere in the tree, so a Postgres data-modifying CTE
  (``WITH d AS (DELETE ...) SELECT ...``) is refused even though its root looks like a ``SELECT``;
* a configurable denylist of functions that read files, sleep, run remote commands or write
  sequences (``pg_read_file``, ``lo_*``, ``dblink``, ``xp_*``, ``DBMS_*``, ``UTL_*``, ...), matched
  on the function name and, for package-qualified calls, on the qualifier;
* a parse failure is a rejection, never an allow: SQL the parser does not understand is never handed
  to a database.

The module is pure (no I/O, no logging, no clock) and never raises, so a caller can always turn the
result into an ``ErrorResponse``.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, TypeAlias

import sqlglot
from sqlglot import exp

from mdqm.models import ErrorResponse

__all__ = [
    "DEFAULT_ALLOWED_TABLE_HINTS",
    "DEFAULT_DENIED_FUNCTIONS",
    "DEFAULT_DENIED_FUNCTION_PREFIXES",
    "MULTI_STATEMENT",
    "PARSE_ERROR",
    "WRITE_BLOCKED",
    "GuardPolicy",
    "GuardResult",
    "guard",
]

#: The SQL does not parse (or is empty, or carries bytes no statement should contain).
PARSE_ERROR: Final[str] = "PARSE_ERROR"
#: The statement would modify a source, lock rows, or run code: never sent anywhere.
WRITE_BLOCKED: Final[str] = "WRITE_BLOCKED"
#: More than one statement was sent; the guard never hands a batch to a source.
MULTI_STATEMENT: Final[str] = "MULTI_STATEMENT"

_FIX_ONE_SELECT: Final[str] = (
    "Send exactly one read-only statement: SELECT ... (optionally WITH ... SELECT, UNION, "
    "INTERSECT or EXCEPT). This server cannot modify a source, so a write has no equivalent."
)

#: Functions that read files, inspect the server, sleep, run remote commands or write sequences.
#: Matched case-insensitively against the function name (and against the qualifier of a
#: package-qualified call). Deliberately narrow: ordinary read-only functions such as
#: ``pg_typeof``, ``pg_size_pretty`` or ``pg_catalog.count`` are not listed.
DEFAULT_DENIED_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {
        # Postgres: file system, large objects, sleep, admin, sequences.
        "pg_read_file",
        "pg_read_binary_file",
        "pg_stat_file",
        "pg_ls_dir",
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_rotate_logfile",
        "pg_switch_wal",
        "pg_create_restore_point",
        "pg_promote",
        "pg_drop_replication_slot",
        "pg_create_physical_replication_slot",
        "pg_create_logical_replication_slot",
        "pg_wal_replay_pause",
        "pg_wal_replay_resume",
        "set_config",
        "nextval",
        "setval",
        "dblink",
        "dblink_exec",
        "dblink_connect",
        "dblink_connect_u",
        "dblink_disconnect",
        "query_to_xml",
        # T-SQL: command shell, OLE automation, ad-hoc remote and dynamic SQL.
        "xp_cmdshell",
        "sp_executesql",
        "sp_oacreate",
        "sp_oamethod",
        "sp_oagetproperty",
        "sp_oasetproperty",
        "openrowset",
        "openquery",
        "opendatasource",
        "openxml",
        # Oracle: scheduler, Java, files, network and arbitrary PL/SQL entry points.
        "dbms_scheduler",
        "dbms_java",
        "dbms_pipe",
        "dbms_utility",
        "utl_http",
        "utl_inaddr",
        "utl_file",
        "utl_tcp",
        "utl_smtp",
        # Other engines: sleep, locks, file access and shell-out user-defined functions.
        "sleep",
        "benchmark",
        "get_lock",
        "load_file",
        "sys_exec",
        "sys_eval",
    }
)

#: Namespace prefixes whose members are denied wholesale, because the member list differs per
#: engine version. ``pg_`` and ``sys_`` are *not* blanket prefixes: ``pg_typeof``, ``SYS_CONTEXT``
#: and ``SYS_GUID`` are ordinary read-only functions, so only the write/exec namespaces are listed.
DEFAULT_DENIED_FUNCTION_PREFIXES: Final[tuple[str, ...]] = (
    "__",
    "dbms_",
    "utl_",
    "owa_",
    "xp_",
    "sp_",
    "dblink_",
    "lo_",
    "pg_read_",
    "pg_ls_",
    "pg_sleep",
    "pg_stat_reset",
    "pg_terminate_",
    "pg_cancel_",
    "pg_reload_",
    "pg_rotate_",
    "pg_switch_",
    "pg_create_",
    "pg_drop_",
    "pg_wal_",
    "pg_promote",
)

#: Table hints that only weaken isolation or skip locked rows. Anything else (UPDLOCK, XLOCK,
#: TABLOCK, TABLOCKX, HOLDLOCK, ...) takes a write lock on a primary database, so it is refused.
DEFAULT_ALLOWED_TABLE_HINTS: Final[frozenset[str]] = frozenset(
    {"nolock", "readpast", "readuncommitted"}
)

# Statement nodes that modify data or schema, wherever they appear (so a data-modifying CTE is
# caught). The order of the groups is the order the messages are produced in.
_WRITE_NODES: Final[tuple[type[exp.Expression], ...]] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
)
_DDL_NODES: Final[tuple[type[exp.Expression], ...]] = (
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Comment,
    exp.Analyze,
    exp.Refresh,
)
_CONTROL_NODES: Final[tuple[type[exp.Expression], ...]] = (
    exp.Grant,
    exp.Revoke,
    exp.Set,
    exp.Use,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Pragma,
    exp.Attach,
    exp.Detach,
    exp.Kill,
    exp.Describe,
    exp.Export,
)
_EXECUTION_NODES: Final[tuple[type[exp.Expression], ...]] = (
    exp.Command,
    exp.Copy,
    exp.Execute,
    exp.ExecuteSql,
    exp.LoadData,
)

#: Statement roots that are read-only queries. ``WITH ... SELECT`` is a ``Select`` carrying a
#: ``with`` argument; ``UNION``, ``INTERSECT`` and ``EXCEPT`` (Oracle ``MINUS``) are
#: ``SetOperation`` nodes.
_READ_ROOTS: Final[tuple[type[exp.Expression], ...]] = (exp.Select, exp.SetOperation)

_NUL: Final[str] = "\x00"
#: MySQL runs the body of ``/*! ... */`` as SQL; a write hidden there would survive an AST check.
_EXECUTABLE_COMMENT: Final[str] = "/*!"


@dataclass(frozen=True, slots=True)
class GuardPolicy:
    """What the guard refuses beyond the structural rules. Extend or narrow per deployment."""

    denied_functions: frozenset[str] = DEFAULT_DENIED_FUNCTIONS
    denied_function_prefixes: tuple[str, ...] = DEFAULT_DENIED_FUNCTION_PREFIXES
    allowed_table_hints: frozenset[str] = DEFAULT_ALLOWED_TABLE_HINTS


@dataclass(frozen=True, slots=True)
class GuardResult:
    """Outcome of :func:`guard`.

    ``ok`` is True only for a single read-only query; ``ast`` is then the parsed tree. A rejection
    never carries an AST (a caller must not be able to run a blocked statement by accident) and
    always carries an :class:`ErrorResponse` with an exact ``fix``.
    """

    ok: bool
    dialect: str
    ast: exp.Expr | None = None
    error: ErrorResponse | None = None


#: (code, problem, fix) for the first thing the guard refuses in a statement.
_Finding: TypeAlias = tuple[str, str, str]


def guard(sql: str, dialect: str, policy: GuardPolicy | None = None) -> GuardResult:
    """Check that ``sql`` is one read-only query in ``dialect`` and never raises.

    Args:
        sql: the statement to check. The server calls this for free SQL *and* for the SQL it
            compiled itself.
        dialect: a sqlglot dialect name (``postgres``, ``oracle``, ``tsql``, ...).
        policy: what to refuse beyond the structural rules; defaults to :class:`GuardPolicy`.

    Returns:
        ``GuardResult(ok=True, ast=...)`` for a single read-only query, otherwise
        ``GuardResult(ok=False, error=ErrorResponse(...))``.
    """
    active = policy if policy is not None else GuardPolicy()
    text = sql if isinstance(sql, str) else ""
    if not text.strip():
        return _reject(dialect, PARSE_ERROR, "The statement is empty.", _FIX_ONE_SELECT)
    if _NUL in text:
        return _reject(
            dialect,
            PARSE_ERROR,
            "The statement contains a NUL byte, which no driver accepts.",
            "Remove the NUL byte from the SQL.",
        )
    if _EXECUTABLE_COMMENT in text:
        return _reject(
            dialect,
            WRITE_BLOCKED,
            "The statement uses a MySQL executable comment '/*! ... */', whose body some engines "
            "run as SQL.",
            "Remove the '/*! ... */' comment and send the statement plainly.",
        )

    try:
        parsed = sqlglot.parse(text, read=dialect, error_level=sqlglot.ErrorLevel.RAISE)
    except Exception as exc:  # noqa: BLE001 - a guard must never raise at its caller
        # sqlglot raises ParseError, TokenError, UnsupportedError and ValueError (unknown dialect).
        detail = str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__
        return _reject(
            dialect,
            PARSE_ERROR,
            f"The statement is not valid {dialect} SQL: {detail}",
            _FIX_ONE_SELECT,
        )

    # sqlglot yields None for an empty statement, so `SELECT 1;;` is one statement, not two.
    statements = [statement for statement in parsed if statement is not None]
    if not statements:
        return _reject(
            dialect,
            PARSE_ERROR,
            "The input holds no statement (only comments or separators).",
            _FIX_ONE_SELECT,
        )
    if len(statements) > 1:
        return _reject(
            dialect,
            MULTI_STATEMENT,
            f"The input holds {len(statements)} statements; exactly one is allowed per call.",
            "Send one statement per call and delete everything after the first ';'.",
        )

    root = statements[0]
    for node in root.walk():
        finding = _forbidden(node, active)
        if finding is not None:
            code, problem, fix = finding
            return _reject(dialect, code, problem, fix)

    if not _is_read_query(root):
        return _reject(
            dialect,
            WRITE_BLOCKED,
            f"A {type(root).__name__.upper()} statement is not a read-only query.",
            _FIX_ONE_SELECT,
        )
    return GuardResult(ok=True, dialect=dialect, ast=root)


def _reject(dialect: str, code: str, problem: str, fix: str) -> GuardResult:
    """Build a rejection. Rejections never carry an AST."""
    return GuardResult(
        ok=False,
        dialect=dialect,
        error=ErrorResponse(code=code, problem=problem, fix=fix),
    )


def _is_read_query(node: exp.Expr) -> bool:
    """True for ``SELECT``, ``WITH ... SELECT``, ``UNION`` and a parenthesised query."""
    if isinstance(node, _READ_ROOTS):
        return True
    if isinstance(node, exp.Subquery):
        return isinstance(node.this, _READ_ROOTS)
    return False


def _forbidden(node: exp.Expr, policy: GuardPolicy) -> _Finding | None:
    """Return the first reason ``node`` may not appear in a read-only statement."""
    if isinstance(node, _WRITE_NODES):
        return (
            WRITE_BLOCKED,
            f"The statement writes data ({type(node).__name__.upper()}), including inside a CTE.",
            "Replace the write with a SELECT; this server cannot modify a source.",
        )
    if isinstance(node, _DDL_NODES):
        return (
            WRITE_BLOCKED,
            f"The statement changes schema or metadata ({type(node).__name__.upper()}).",
            "Schema changes are done by a DBA, never through this server. Send a SELECT.",
        )
    if isinstance(node, _EXECUTION_NODES):
        return (
            WRITE_BLOCKED,
            f"The statement runs code or moves data ({type(node).__name__.upper()}): CALL, EXEC, "
            "COPY, LOAD, EXPLAIN and other server commands are refused.",
            "Send a single read-only SELECT. EXPLAIN, CALL, COPY, SET and SHOW are not supported.",
        )
    if isinstance(node, _CONTROL_NODES):
        return (
            WRITE_BLOCKED,
            f"The statement changes session or transaction state ({type(node).__name__.upper()}).",
            "Remove the statement: each query already runs in its own read-only transaction. "
            "Send a SELECT.",
        )
    if isinstance(node, exp.Into):
        return (
            WRITE_BLOCKED,
            "SELECT ... INTO creates a table and moves rows into it.",
            "Remove the INTO clause and read the rows directly; add a LIMIT if the result is big.",
        )
    if isinstance(node, exp.Lock):
        return (
            WRITE_BLOCKED,
            "The statement takes row locks (FOR UPDATE / FOR SHARE / FOR NO KEY UPDATE).",
            "Remove the locking clause: every query already runs in a read-only transaction.",
        )
    if isinstance(node, exp.WithTableHint):
        blocked = sorted(
            hint
            for hint in (_hint_name(var) for var in node.find_all(exp.Var))
            if hint and hint not in policy.allowed_table_hints
        )
        if blocked:
            return (
                WRITE_BLOCKED,
                f"The table hint(s) {', '.join(blocked)} take locks on the source.",
                "Remove the hint, or use only NOLOCK, READPAST or READUNCOMMITTED.",
            )
        return None
    if isinstance(node, exp.Func):
        denied = _denied_name(_function_name(node), policy)
        if denied is not None:
            return _denied_function_finding(denied)
    elif isinstance(node, exp.Dot) and isinstance(node.expression, exp.Func):
        # A package-qualified call: `DBMS_LOCK.SLEEP(5)` keeps only SLEEP in the function node, so
        # the qualifier is the part that names the dangerous package.
        for qualifier in _qualifier_names(node.this):
            denied = _denied_name(qualifier, policy)
            if denied is not None:
                return _denied_function_finding(denied)
    return None


def _denied_function_finding(name: str) -> _Finding:
    """The rejection for a denied function, naming it so the agent can remove it."""
    return (
        WRITE_BLOCKED,
        f"The function {name} is not allowed: it reads files, sleeps, locks, runs remote commands "
        "or writes state.",
        f"Remove {name}(...) from the query. Only plain read-only functions are allowed.",
    )


def _function_name(node: exp.Func) -> str:
    """The name of the called function as written.

    A ``Func`` subclass knows its name (``SUM`` -> ``sql_name() == 'SUM'``), while an unknown
    function is an ``Anonymous`` node that keeps the spelling in ``this`` and reports
    ``sql_name() == 'ANONYMOUS'``. ``Func.name`` is *not* the function name: it is the name of the
    first argument.
    """
    if isinstance(node, exp.Anonymous):
        name = node.name
        return name if isinstance(name, str) else ""
    return node.sql_name()


def _hint_name(var: exp.Expr) -> str:
    """Lower-case table-hint name of an ``exp.Var`` in ``WITH ( ... )`` (``NOLOCK`` -> nolock)."""
    return _normalise(var.name)


def _qualifier_names(node: exp.Expr) -> Iterator[str]:
    """Every identifier-ish name in the qualifier part of a dotted call (``SYS.UTL_FILE``)."""
    for sub in node.walk():
        if isinstance(sub, (exp.Identifier, exp.Column, exp.Dot)):
            name = sub.name
            if name:
                yield name


def _normalise(name: str) -> str:
    """Case-fold and NFKC-fold a name so full-width or quoted spellings match the denylist."""
    return unicodedata.normalize("NFKC", name).strip().casefold()


def _denied_name(name: str, policy: GuardPolicy) -> str | None:
    """Return the matched denial (as written in the policy) or None when the name is allowed."""
    normalised = _normalise(name)
    if not normalised:
        return None
    if normalised in policy.denied_functions:
        return normalised
    for prefix in policy.denied_function_prefixes:
        if normalised.startswith(prefix):
            return normalised
    return None
