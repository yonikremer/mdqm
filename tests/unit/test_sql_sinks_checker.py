"""`scripts/check_sql_sinks.py` is a security tool, so it needs cases of its own.

The checker decides whether a *value* can reach a statement's text. Its job is to be satisfiable by
safe code and unforgiving on unsafe code, which is exactly what the off-the-shelf rule got wrong (it
flags every non-literal argument, so it catches nothing that matters and explains nothing). Every
case below names the rule it pins; a case that would let an injection through is a bug in the tool,
so the cases that must be **flagged** are the important half.

The fixtures are written to `tmp_path` and the checker is run over them as a command, so the test
pins the tool's real behaviour (files in, findings and exit code out), not its internals.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

CHECKER = Path(__file__).resolve().parents[2] / "scripts" / "check_sql_sinks.py"


def _checker() -> ModuleType:
    """The script, imported by path: `scripts/` is untyped by policy and is not a package."""
    spec = importlib.util.spec_from_file_location("check_sql_sinks", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # `@dataclass(slots=True)` looks its own module up in `sys.modules` while the body runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _checker()


def _findings(tmp_path: Path, source: str, *, name: str = "module.py") -> list[str]:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return [finding.reason for finding in MODULE.check_source(path, source)]


# --- must be flagged: a value can reach the statement -------------------------------------------

UNSAFE: list[tuple[str, str]] = [
    (
        "f-string interpolation",
        'def go(cursor, table):\n    cursor.execute(f"SELECT * FROM {table}")\n',
    ),
    (
        "concatenation with +",
        'def go(cursor, value):\n    cursor.execute("SELECT * FROM t WHERE c = " + value)\n',
    ),
    (
        "percent formatting",
        'def go(cursor, value):\n    cursor.execute("SELECT * FROM t WHERE c = %s" % value)\n',
    ),
    (
        "format()",
        'def go(cursor, table):\n    cursor.execute("SELECT * FROM {}".format(table))\n',
    ),
    (
        "join()",
        'def go(cursor, parts):\n    cursor.execute(" ".join(parts))\n',
    ),
    (
        "conditional expression",
        'def go(cursor, flag):\n    cursor.execute("SELECT 1" if flag else "SELECT 2")\n',
    ),
    (
        "name assigned from an f-string",
        "def go(cursor, table):\n"
        '    statement = f"SELECT * FROM {table}"\n'
        "    cursor.execute(statement)\n",
    ),
    (
        "module-level interpolated constant",
        "TABLE = 'customers'\n"
        'STATEMENT = f"SELECT * FROM {TABLE}"\n'
        "def go(cursor):\n"
        "    cursor.execute(STATEMENT)\n",
    ),
    (
        "executemany",
        "def go(cursor, rows, table):\n"
        '    cursor.executemany(f"INSERT INTO {table} VALUES (?)", rows)\n',
    ),
    (
        "copy",
        'def go(cursor, table):\n    cursor.copy(f"COPY {table} FROM STDIN")\n',
    ),
    (
        "a constant assembled with + into a name",
        "TABLE = 'customers'\n"
        "STATEMENT = 'SELECT * FROM ' + TABLE\n"
        "def go(cursor):\n"
        "    cursor.execute(STATEMENT)\n",
    ),
    (
        "a name reassigned from an f-string after a literal",
        "STATEMENT = 'SELECT 1'\n"
        "def go(cursor, table):\n"
        "    STATEMENT = f'SELECT * FROM {table}'\n"
        "    cursor.execute(STATEMENT)\n",
    ),
]


@pytest.mark.parametrize(("case", "source"), UNSAFE, ids=[case for case, _ in UNSAFE])
def test_an_unsafe_statement_is_flagged(tmp_path: Path, case: str, source: str) -> None:
    assert _findings(tmp_path, source), f"{case} must be flagged"


# --- must be clean: no value can reach the statement --------------------------------------------


SAFE: dict[str, str] = {
    "string literal": 'def go(cursor):\n    cursor.execute("SELECT 1")\n',
    "module constant": ("STATEMENT = 'SELECT 1'\ndef go(cursor):\n    cursor.execute(STATEMENT)\n"),
    "prepare()": (
        'def go(cursor, text):\n    cursor.execute(MODULE_prepare(text, "postgres", ()))\n'
    ),
    "execute_one()": ("def go(cursor, statement):\n    execute_one(cursor, statement)\n"),
    "run_local()": "def go(cursor, text):\n    run_local(cursor, text, ())\n",
    "sql.SQL composition": (
        "import psycopg.sql as sql\n"
        "def go(cursor, table):\n"
        "    cursor.execute(sql.SQL('SELECT * FROM {}').format(sql.Identifier(table)))\n"
    ),
    "a statement produced elsewhere": (
        "def go(cursor, fragment):\n    cursor.execute(fragment.sql)\n"
    ),
    "a compiled fragment annotated as one": (
        "def go(adapter, fragment: Fragment, limits):\n    adapter.execute(fragment, limits)\n"
    ),
    "a compiled plan read off a source": (
        "def go(executor, federated, user):\n"
        "    plan = federated.plan\n"
        "    executor.execute(plan, user=user)\n"
    ),
    "a bare parameter, the documented limit": ("def go(cursor, text):\n    cursor.execute(text)\n"),
}


@pytest.mark.parametrize("case", sorted(SAFE), ids=sorted(SAFE))
def test_a_safe_statement_is_not_flagged(tmp_path: Path, case: str) -> None:
    assert _findings(tmp_path, SAFE[case]) == [], f"{case} must be clean"


def test_prepare_is_the_guarded_chokepoint(tmp_path: Path) -> None:
    """`prepare(...)` inline as the statement argument: the guard runs, so the checker passes it."""
    source = 'def go(cursor, text):\n    cursor.execute(prepare(text, "postgres", ()))\n'
    assert _findings(tmp_path, source) == []


# --- the trusted assembler ----------------------------------------------------------------------

ASSEMBLES_TEXT = 'def go(cursor, table):\n    cursor.execute(f"SELECT * FROM {table}")\n'


def test_the_trusted_assembler_is_exempt_by_path(tmp_path: Path) -> None:
    """`mdqm/sql/execution.py` builds statement text on purpose; the exemption is by path."""
    assert _findings(tmp_path, ASSEMBLES_TEXT, name="mdqm/sql/execution.py") == []


def test_the_same_code_is_flagged_anywhere_else(tmp_path: Path) -> None:
    """The exemption must not leak: identical text two directories over is a finding."""
    assert _findings(tmp_path, ASSEMBLES_TEXT, name="mdqm/adapters/postgresql.py")


def test_the_synthetic_data_loaders_are_exempt_by_path(tmp_path: Path) -> None:
    """They write DDL, which the read-only guard refuses, and the query path never runs them."""
    assert _findings(tmp_path, ASSEMBLES_TEXT, name="mdqm/testing/loaders/postgres.py") == []


def test_a_pipeline_stage_is_checked(tmp_path: Path) -> None:
    """`pipeline/` builds SQL for sources, so it is runtime and must be checked."""
    assert _findings(tmp_path, ASSEMBLES_TEXT, name="mdqm/pipeline/stages/mining.py")


# --- the command contract the pre-commit hook reads ---------------------------------------------


def test_main_exits_clean_and_says_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "clean.py").write_text("cursor.execute('SELECT 1')\n", encoding="utf-8")

    assert MODULE.main([str(tmp_path)]) == 0
    assert capsys.readouterr().err == ""


def test_main_exits_one_with_every_finding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "bad.py").write_text(
        'cursor.execute(f"SELECT * FROM {table}")\ncursor.execute("SELECT 1" + tail)\n',
        encoding="utf-8",
    )

    assert MODULE.main([str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "bad.py:1:" in err
    assert "bad.py:2:" in err
    assert "2 statement(s)" in err


def test_a_file_that_does_not_parse_is_skipped_not_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Syntax errors belong to the parsers and linters; a false finding here would hide the real one."""
    (tmp_path / "broken.py").write_text("def go(:\n", encoding="utf-8")

    assert MODULE.main([str(tmp_path)]) == 0
    assert capsys.readouterr().err == ""


def test_iter_python_files_skips_virtualenvs_and_worktrees(tmp_path: Path) -> None:
    for relative in ("src/a.py", ".venv/lib/b.py", ".worktrees/x/c.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    found = [path.name for path in MODULE.iter_python_files([str(tmp_path)])]

    assert found == ["a.py"]
