"""Fail when a YAML or TOML file does not parse.

Pre-commit hands this the staged files that match `\\.(toml|ya?ml)$`. A malformed `pyproject.toml`
or compose file usually fails much later and much less clearly (a pip resolver error, a container that
will not start), so parsing it at commit time is cheap insurance.

YAML is parsed with ruamel's **safe** loader: this checks syntax, it must never construct anything.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError


def check_toml(path: Path) -> str | None:
    """The problem with ``path`` as TOML, or ``None`` when it parses."""
    try:
        tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return f"invalid TOML: {exc}"
    except OSError as exc:
        return f"cannot read: {exc.strerror or exc}"
    return None


def check_yaml(path: Path) -> str | None:
    """The problem with ``path`` as YAML, or ``None`` when it parses."""
    loader = YAML(typ="safe")
    try:
        list(loader.load_all(path.read_text(encoding="utf-8")))
    except YAMLError as exc:
        # The message can quote the offending line, which may hold a credential; the position is
        # enough to find it.
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1} column {mark.column + 1}" if mark is not None else ""
        return f"invalid YAML{where}: {type(exc).__name__}"
    except OSError as exc:
        return f"cannot read: {exc.strerror or exc}"
    return None


def main(argv: list[str] | None = None) -> int:
    """Check every file given on the command line; returns 0 when all of them parse."""
    paths = [Path(name) for name in (argv if argv is not None else sys.argv[1:])]
    problems = 0
    for path in paths:
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".toml":
            problem = check_toml(path)
        elif suffix in {".yaml", ".yml"}:
            problem = check_yaml(path)
        else:
            continue
        if problem is not None:
            print(f"{path}: {problem}", file=sys.stderr)
            problems += 1
    if problems:
        print(f"{problems} file(s) do not parse", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
