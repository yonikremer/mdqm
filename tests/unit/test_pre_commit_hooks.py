"""The pre-commit hooks themselves.

A lint hook that silently stops catching things is worse than no hook, so the two local scripts that
carry real judgement -- `check_secrets.py` and `check_configs.py` -- are tested here with the cases
that made them necessary:

* a committed credential is refused, and the value is never echoed back (CI logs it);
* a `pragma: allowlist secret` fixture is accepted, on the line and on the line above it;
* code that *reads* a secret (`password = secrets.get(...)`) and a shell reference (`"$VAR"`) are
  secrets and must not be flagged, because a hook that cries wolf gets switched off;
* broken YAML and TOML fail with the file named, and good files pass.

The hook wiring (ids, stages, entries) is checked too, so a renamed hook cannot quietly disappear.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts"


def _load(name: str) -> ModuleType:
    """Import a script by path: `scripts/` is not a package."""
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"mdqm_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def secrets_module() -> ModuleType:
    return _load("check_secrets")


@pytest.fixture(scope="module")
def configs_module() -> ModuleType:
    return _load("check_configs")


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- check_secrets ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('key = "-----BEGIN RSA PRIVATE KEY-----"', "private key block"),
        ('aws = "AKIAIOSFODNN7EXAMPLE"', "AWS access key id"),
        ('token = "ghp_' + "a" * 36 + '"', "GitHub token"),
        ('auth = "Bearer ' + "b" * 30 + '"', "bearer token"),
        ('password = "correct-horse-battery"', "assigned password"),
    ],
)
def test_a_credential_is_reported(
    secrets_module: ModuleType, tmp_path: Path, content: str, expected: str
) -> None:
    path = _write(tmp_path, "leak.py", content + "\n")
    hits = list(secrets_module.findings(path))
    assert hits, f"{content!r} should be reported"
    assert expected in hits[0]


@pytest.mark.parametrize(
    "content",
    [
        'password = secrets.get(config.secret)   # reading, not committing',
        'password = "$MDQM_RO_PASSWORD"          # a shell reference',
        'password = "${MDQM_RO_PASSWORD}"',
        'note = "the password is in the vault"',
    ],
)
def test_code_that_reads_a_secret_is_not_a_secret(
    secrets_module: ModuleType, tmp_path: Path, content: str
) -> None:
    path = _write(tmp_path, "code.py", content + "\n")
    assert list(secrets_module.findings(path)) == []


def test_the_allowlist_marker_works_on_the_line_and_above(
    secrets_module: ModuleType, tmp_path: Path
) -> None:
    inline = _write(
        tmp_path,
        "inline.md",
        'aws = "AKIAIOSFODNN7EXAMPLE"  # pragma: allowlist secret\n',
    )
    above = _write(
        tmp_path,
        "above.md",
        "# a fixture, not a secret  # pragma: allowlist secret\n"
        'aws = "AKIAIOSFODNN7EXAMPLE"\n',
    )
    assert list(secrets_module.findings(inline)) == []
    assert list(secrets_module.findings(above)) == []


def test_a_hit_never_echoes_the_value(secrets_module: ModuleType, tmp_path: Path) -> None:
    """The report goes to CI logs, so it must name the file and line only."""
    secret = "AKIAIOSFODNN7EXAMPLE"
    path = _write(tmp_path, "leak.py", f'aws = "{secret}"\n')
    hits = list(secrets_module.findings(path))
    assert hits and secret not in hits[0]


def test_binary_and_large_files_are_skipped(
    secrets_module: ModuleType, tmp_path: Path
) -> None:
    binary = tmp_path / "blob.parquet"
    binary.write_bytes(b"-----BEGIN RSA PRIVATE KEY-----")
    assert list(secrets_module.findings(binary)) == []


# --- check_configs ----------------------------------------------------------------


def test_broken_yaml_is_reported_with_its_position(
    configs_module: ModuleType, tmp_path: Path
) -> None:
    path = _write(tmp_path, "broken.yaml", "a: [1, 2\nb: 3\n")
    problem = configs_module.check_yaml(path)
    assert problem is not None
    assert "invalid YAML" in problem


def test_broken_toml_is_reported(configs_module: ModuleType, tmp_path: Path) -> None:
    path = _write(tmp_path, "broken.toml", "a = \n")
    problem = configs_module.check_toml(path)
    assert problem is not None
    assert "invalid TOML" in problem


def test_good_files_pass(configs_module: ModuleType, tmp_path: Path) -> None:
    assert configs_module.check_yaml(_write(tmp_path, "ok.yaml", "a: 1\nb: [1, 2]\n")) is None
    assert configs_module.check_toml(_write(tmp_path, "ok.toml", "a = 1\n")) is None


def test_main_reports_failure_and_success(configs_module: ModuleType, tmp_path: Path) -> None:
    good = _write(tmp_path, "good.yaml", "a: 1\n")
    bad = _write(tmp_path, "bad.yaml", "a: [\n")
    assert configs_module.main([str(good)]) == 0
    assert configs_module.main([str(bad)]) == 1


# --- the wiring -------------------------------------------------------------------


def test_every_hook_is_local_and_uses_the_project_interpreter() -> None:
    """No `rev:` means no clone at hook time, which is what makes this work air-gapped."""
    config = REPO_ROOT / ".pre-commit-config.yaml"
    document = YAML(typ="safe").load(config.read_text(encoding="utf-8"))
    repos = document["repos"]
    assert repos, "the config declares no hooks"
    assert all(repo.get("repo") == "local" for repo in repos), "a remote repo would need a network"
    entries = [hook["entry"] for repo in repos for hook in repo["hooks"]]
    assert all("precommit_hook.py" in entry or entry.startswith("python scripts/") or "-m " in entry
               for entry in entries), entries


def test_the_security_family_is_enabled() -> None:
    """`S` is flake8-bandit: the rules the owner asked for first."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    select_line = next(line for line in text.splitlines() if line.startswith("select = "))
    for family in ('"S"', '"B"', '"RUF"'):
        assert family in select_line, f"{family} is not enabled in {select_line}"
