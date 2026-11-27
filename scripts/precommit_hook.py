"""Run a pre-commit hook with the project's own interpreter.

`language: system` hooks run whatever `python` is on PATH, which is how a hook ends up importing the
*global* environment instead of the project's: on this machine `python scripts/check_configs.py`
failed with `No module named 'ruamel'` for exactly that reason.

So every hook entry goes through this launcher: it finds the repository's virtual environment
(`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` elsewhere) and re-executes the requested
module or script with it. When there is no `.venv`, it falls back to the interpreter running this file,
which is the best guess available and keeps the hook usable in a container or CI where the environment
is global.

Usage:
    python scripts/precommit_hook.py -m ruff check --fix --exit-non-zero-on-fix
    python scripts/precommit_hook.py -m pytest tests/unit -x -q
    python scripts/precommit_hook.py -m scripts.check_configs <files...>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def project_python() -> Path:
    """The interpreter of the repository's virtual environment, or this one as a fallback."""
    candidates = (
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv" / "bin" / "python",
        REPO_ROOT / ".venv" / "bin" / "python3",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def main(argv: list[str] | None = None) -> int:
    """Execute the requested command with the project interpreter and return its exit code."""
    arguments = list(argv if argv is not None else sys.argv[1:])
    if not arguments:
        print("usage: precommit_hook.py -m <module> [args...]", file=sys.stderr)
        return 2
    interpreter = project_python()
    command = [str(interpreter), *arguments]
    # `-m` is passed through as-is; anything else is a script path, which we hand to the interpreter.
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
