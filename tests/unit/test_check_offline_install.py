"""Tests for the offline-install check (task M08).

The script lives outside the package (``scripts/``), so it is loaded by path. Nothing here touches
the network or Docker: the allowlist checks read files, and the heavy steps (wheelhouse download,
fresh venv) are the script's job, run by hand or in the deployment pipeline.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_offline_install.py"


@pytest.fixture(scope="module")
def offline() -> ModuleType:
    """The script as a module. Registered in ``sys.modules`` first: dataclasses need that."""
    spec = importlib.util.spec_from_file_location("check_offline_install", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_offline_install"] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_requirements_are_the_shipped_ones(offline: ModuleType) -> None:
    requirements = offline.runtime_requirements()
    names = {offline.canonical(offline.requirement_name(item)) for item in requirements}

    # Core runtime: every one of these is imported by src/mdqm on the server path.
    assert {
        "mcp",
        "pydantic",
        "pydantic-settings",
        "sqlglot",
        "duckdb",
        "pyarrow",
        "httpx",
        "ruamel-yaml",
        "typer",
        "tzdata",
    } <= names
    # The `pg` extra ships; dev-only tooling never reaches the image.
    assert {"psycopg", "psycopg-pool"} <= names
    assert not names & {"pytest", "ruff", "mypy", "hypothesis", "fastapi", "uvicorn"}


def test_runtime_requirements_rejects_an_unknown_extra(offline: ModuleType) -> None:
    with pytest.raises(ValueError, match="no extra 'nosuch'"):
        offline.runtime_requirements(extras=("nosuch",))


@pytest.mark.parametrize(
    ("requirement", "expected"),
    [
        ("psycopg[binary]>=3.2", "psycopg"),
        ("ruamel.yaml>=0.18", "ruamel.yaml"),
        ("mcp>=2,<3", "mcp"),
        ("duckdb", "duckdb"),
        ("pkg ; python_version < '3.13'", "pkg"),
    ],
)
def test_requirement_name(offline: ModuleType, requirement: str, expected: str) -> None:
    assert offline.requirement_name(requirement) == expected


class TestWheelhouse:
    def test_classifies_wheels_and_source_archives(
        self, offline: ModuleType, tmp_path: Path
    ) -> None:
        (tmp_path / "django_pkg-1.0-py3-none-any.whl").write_text("", encoding="utf-8")
        binary = "psycopg_binary-3.3.9-cp312-cp312-win_amd64.whl"
        (tmp_path / binary).write_text("", encoding="utf-8")
        (tmp_path / "legacy-0.1.tar.gz").write_text("", encoding="utf-8")
        (tmp_path / "requirements.txt").write_text("", encoding="utf-8")

        staged = offline.scan_wheelhouse(tmp_path)

        assert set(staged.wheels) == {"django-pkg", "psycopg-binary"}
        assert staged.source_archives == ("legacy-0.1.tar.gz",)

    def test_missing_is_canonical_and_sorted(self, offline: ModuleType, tmp_path: Path) -> None:
        (tmp_path / "ruamel_yaml-0.19.1-py3-none-any.whl").write_text("", encoding="utf-8")
        staged = offline.scan_wheelhouse(tmp_path)

        missing = staged.missing(["ruamel.yaml>=0.18", "psycopg[binary]>=3.2", "duckdb>=1.1"])

        assert missing == ["duckdb>=1.1", "psycopg[binary]>=3.2"]

    def test_absent_directory_is_empty_not_an_error(
        self, offline: ModuleType, tmp_path: Path
    ) -> None:
        staged = offline.scan_wheelhouse(tmp_path / "nope")
        assert staged.wheels == {}
        assert staged.source_archives == ()


class TestDockerfile:
    def test_this_repository_passes(self, offline: ModuleType) -> None:
        text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert offline.check_dockerfile(text) == []

    @pytest.mark.parametrize(
        ("snippet", "expected"),
        [
            ("ENV PGPASSWORD=hunter2\n", "literal credential"),
            ("ARG MDQM_MODEL__API_KEY=sk-live\n", "literal credential"),
            ("RUN python -m pip install requests\n", "installs packages"),
        ],
    )
    def test_flags_a_broken_recipe(self, offline: ModuleType, snippet: str, expected: str) -> None:
        recipe = (
            "FROM python:3.12-slim-bookworm AS builder\n"
            "RUN pip install .\n"
            "FROM python:3.12-slim-bookworm AS runtime\n"
            "USER 10001:10001\n"
            "HEALTHCHECK CMD python -c 'print(1)'\n"
            f"{snippet}"
        )
        problems = offline.check_dockerfile(recipe)
        assert any(expected in problem for problem in problems), problems

    def test_missing_healthcheck_is_a_problem(self, offline: ModuleType) -> None:
        recipe = "FROM python:3.12-slim-bookworm\nUSER 10001:10001\n"
        assert any("no HEALTHCHECK" in problem for problem in offline.check_dockerfile(recipe))

    def test_missing_user_is_a_problem(self, offline: ModuleType) -> None:
        recipe = "FROM python:3.12-slim-bookworm\nHEALTHCHECK CMD python -c 'print(1)'\n"
        assert any("no USER" in problem for problem in offline.check_dockerfile(recipe))

    @pytest.mark.parametrize(
        ("image", "pinned"),
        [
            ("python:3.12-slim-bookworm", True),
            ("python:latest", False),
            ("python", False),
            ("scratch", True),
            ("alpine@sha256:" + "a" * 64, True),
        ],
    )
    def test_base_image_pinning(self, offline: ModuleType, image: str, pinned: bool) -> None:
        assert (offline._is_pinned(image) is None) is pinned

    def test_running_as_root_is_a_problem(self, offline: ModuleType) -> None:
        recipe = "FROM python:3.12-slim-bookworm\nUSER root\nHEALTHCHECK CMD python -c 'print(1)'\n"
        assert any("runs as root" in problem for problem in offline.check_dockerfile(recipe))

    @pytest.mark.parametrize(
        ("instruction", "leaks"),
        [
            ("ENV MDQM_MODEL__API_KEY=", False),
            ("ENV MDQM_TOKENS_FILE=/run/secrets/tokens.json", False),
            ("ENV MDQM_SECRETS__DIR=/run/secrets", False),
            ("ARG DB_TOKEN=$STAGED_TOKEN", False),
            ("ENV DB_PASSWORD=hunter2", True),
            ("ENV DB_PASSWD s3cret", True),
            ("ENV PGPASSWORD=hunter2", True),
            # Accepted false positive: a suffix match is how PGPASSWORD is caught too.
            ("ENV MONKEY=1", True),
        ],
    )
    def test_credential_literals(self, offline: ModuleType, instruction: str, leaks: bool) -> None:
        assert (offline._credential_literal(instruction) is not None) is leaks
