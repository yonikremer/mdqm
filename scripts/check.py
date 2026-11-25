"""One command for 'is the repo healthy': ruff, mypy, unit tests. Cross-platform.  python scripts/check.py [--fast]"""

import subprocess
import sys

STEPS = [
    ("ruff", [sys.executable, "-m", "ruff", "check", "."]),
    ("mypy", [sys.executable, "-m", "mypy"]),
    ("unit tests", [sys.executable, "-m", "pytest", "tests/unit", "-q"]),
]

if __name__ == "__main__":
    steps = [s for s in STEPS if not (s[0] == "mypy" and "--fast" in sys.argv)]
    failed = []
    for name, cmd in steps:
        print(f"== {name}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)
    print("all checks passed")
