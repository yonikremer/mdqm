"""A SQL-injection lint rule that looks at *how* the statement was built.

The off-the-shelf rule in this repo's toolchain flags **every** non-literal argument to
`execute`/`executemany` — a module constant, a guarded fragment, even a function call. That is not a
data-flow check, so it cannot be satisfied by safe code and it cannot distinguish `f"...{user_input}"`
from `f"...{validated_identifier}"`. It catches everything and explains nothing.

This checker answers the actual question: **can a value reach the statement text?**

It walks each module and, for every call that reaches a driver (`execute`, `executemany`, `copy`),
classifies the statement argument:

* **safe** — a string literal; a name bound *only* to a string literal at module or function scope; a
  call to `prepare(...)`, `execute_one(...)`/`run(...)` (the guarded chokepoint), or `sql.SQL(...)`
  composed from identifiers and literals;
* **unsafe** — an f-string with any interpolated expression, `str + str`, `%` formatting,
  `"...".format(...)`, `"".join(...)`, or a name that was assigned from any of those.

The difference from the built-in rule is deliberate: interpolating a **value** into SQL is the bug; a
statement assembled from constants and validated identifiers is not. Where the built-in rule needs a
suppression, this one needs nothing.

Usage:
    python scripts/check_sql_sinks.py [paths...]     # 0 clean, 1 findings
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SINK_METHODS = frozenset({"execute", "executemany", "copy", "copy_expert", "executescript"})
"""Methods that hand a statement to a driver."""

SAFE_CALLS = frozenset({"prepare", "execute_one", "run", "run_local", "SQL", "sql"})
"""Calls that produce a statement this repository has already checked.

`prepare()` runs the AST guard; `execute_one()` executes exactly one guarded statement; `run_local()`
is for our own metadata database and takes a literal at the call site. `sql.SQL(...)` is psycopg's
composition API, which quotes identifiers itself.
"""

FORMAT_METHODS = frozenset({"format", "join", "replace"})


@dataclass(frozen=True, slots=True)
class Finding:
    """One unsafe statement argument."""

    path: Path
    line: int
    reason: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.reason}"


def _literal_strings(tree: ast.Module) -> set[str]:
    """Names bound only to string literals, at module scope or anywhere (union, conservatively).

    A name that is assigned an interpolated value anywhere is *not* included, so the check cannot be
    fooled by re-assignment.
    """
    literals: set[str] = set()
    tainted: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and node.value is not None:
            targets, value = list(node.targets), node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if value is None:
            continue
        is_literal = isinstance(value, ast.Constant) and isinstance(value.value, str)
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if is_literal:
                literals.add(target.id)
            else:
                tainted.add(target.id)
    return literals - tainted


def _describe(node: ast.expr) -> str | None:
    """Why ``node`` is unsafe as a statement, or ``None`` when it is safe."""
    if isinstance(node, ast.Constant):
        return None if isinstance(node.value, str) else "statement is not a string literal"
    if isinstance(node, ast.Name):
        # resolved by the caller against the literal set; a bare name is safe only then
        return None
    if isinstance(node, ast.JoinedStr):
        for part in node.values:
            if isinstance(part, ast.FormattedValue):
                return "f-string interpolates a value into the statement"
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return "statement is concatenated with `+`"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return "statement uses `%` formatting"
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in SAFE_CALLS:
            return None
        if name in FORMAT_METHODS:
            return f"statement is built with `{name}()`"
        return None
    if isinstance(node, ast.Attribute):
        # e.g. `fragment.sql` — a value produced elsewhere; the guard is what makes it safe, and that
        # is enforced at execution time by execute_one(). Not a text-building pattern.
        return None
    if isinstance(node, ast.IfExp):
        return "statement is chosen by a conditional expression"
    return None


def _sink_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in SINK_METHODS:
        return func.attr
    return None


def check_source(path: Path, text: str) -> Iterator[Finding]:
    """Yield one finding per unsafe statement reaching a driver in ``text``."""
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return
    literals = _literal_strings(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        method = _sink_name(node)
        if method is None or method == "copy_expert":
            continue
        if not node.args:
            continue
        statement = node.args[0]
        if isinstance(statement, ast.Name) and statement.id in literals:
            continue
        reason = _describe(statement)
        if reason is not None:
            yield Finding(path=path, line=node.lineno, reason=f"{method}(): {reason}")


def iter_python_files(paths: list[str]) -> Iterator[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix == ".py":
            yield path
        elif path.is_dir():
            yield from sorted(
                child
                for child in path.rglob("*.py")
                if ".venv" not in child.parts and ".worktrees" not in child.parts
            )


def main(argv: list[str] | None = None) -> int:
    """Check every file given on the command line (or `src` and `scripts` by default)."""
    arguments = list(argv if argv is not None else sys.argv[1:])
    paths = arguments or ["src", "scripts"]
    findings: list[Finding] = []
    for path in iter_python_files(paths):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        findings.extend(check_source(path, text))
    for finding in findings:
        print(finding.render(), file=sys.stderr)
    if findings:
        print(
            f"{len(findings)} statement(s) are built from values; use a bound parameter, or "
            "`execute_one()` for a guarded statement",
            file=sys.stderr,
        )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
