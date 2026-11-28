"""The CI gate itself: which checks `scripts/check.py` runs, and that the type checkers stay in it.

Why a test for a shell-shaped script: adding pyright to the gate is the only thing that stops the
cleared type debt from growing back silently, and deleting one line from a list is exactly how that
protection disappears without anybody noticing. So the *composition* of the gate is pinned here, and
the commands it runs are pinned to this interpreter (``sys.executable``), which is what makes it
portable to a Windows checkout without a ``python3`` on the PATH.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]


def _gate() -> ModuleType:
    """Import ``scripts/check.py`` by path: ``scripts/`` is not a package."""
    path = REPO_ROOT / "scripts" / "check.py"
    spec = importlib.util.spec_from_file_location("mdqm_check_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gate_runs_ruff_mypy_pyright_and_the_unit_tests() -> None:
    steps = _gate().STEPS

    assert [name for name, _command in steps] == ["ruff", "mypy", "pyright", "unit tests"]


def test_every_step_runs_through_this_interpreter() -> None:
    """A bare ``python``/``pyright`` would depend on the PATH, and Windows has no ``python3``."""
    for name, command in _gate().STEPS:
        assert command[0] == sys.executable, name
        assert command[1] == "-m", name


def test_the_unit_tests_step_runs_the_unit_folder_quietly() -> None:
    (unit,) = [command for name, command in _gate().STEPS if name == "unit tests"]

    assert unit[2:] == ["pytest", "tests/unit", "-q"]


def test_fast_skips_the_type_checkers_and_keeps_the_rest() -> None:
    """``--fast`` is the small loop; it must not quietly drop the tests as well."""
    gate = _gate()

    fast = [name for name, _command in gate.STEPS if name not in gate.TYPE_CHECKS]

    assert fast == ["ruff", "unit tests"]
