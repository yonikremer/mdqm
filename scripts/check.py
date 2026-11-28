"""One command for 'is the repo healthy': ruff, mypy, pyright, unit tests. Cross-platform.

    python scripts/check.py [--fast]

``--fast`` skips the two type checkers (mypy and pyright) and runs the fast half only.

pyright is in the gate on purpose: it reported nothing while it was a command somebody remembered to
run, and that is exactly how the cleared backlog grows back. It is slower than the rest of the gate
(--fast exists for that), so a change to a type annotation is worth a plain run.
"""

import subprocess
import sys

TYPE_CHECKS = {"mypy", "pyright"}

STEPS = [
    ("ruff", [sys.executable, "-m", "ruff", "check", "."]),
    ("mypy", [sys.executable, "-m", "mypy"]),
    ("pyright", [sys.executable, "-m", "pyright"]),
    ("unit tests", [sys.executable, "-m", "pytest", "tests/unit", "-q"]),
]

if __name__ == "__main__":
    fast = "--fast" in sys.argv
    steps = [s for s in STEPS if not (fast and s[0] in TYPE_CHECKS)]
    failed = []
    for name, cmd in steps:
        print(f"== {name}", flush=True)
        try:
            code = subprocess.run(cmd).returncode
        except FileNotFoundError:
            print(f"   {name} is not installed: pip install -e .[dev]", flush=True)
            code = 1
        if code != 0:
            failed.append(name)
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)
    print("all checks passed")
