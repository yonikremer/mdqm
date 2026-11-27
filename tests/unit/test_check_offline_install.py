"""Tests for the offline-install check (task M08).

The script lives outside the package (``scripts/``), so it is loaded by path. Nothing here touches
the network or Docker: the allowlist checks read files, and the heavy steps (wheelhouse download,
fresh venv) are the script's job, run by hand or in the deployment pipeline.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
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


# --- B1: the build backend is a dependency of the build -------------------------------


def test_the_build_backend_is_in_the_closure(offline: ModuleType) -> None:
    """`pip install .[pg]` builds this project, so the build backend has to be staged too.

    The reviewer's reproduction: `pip install --no-index --find-links /wheelhouse ".[pg]"` ->
    "ERROR: Could not find a version that satisfies the requirement hatchling". A wheelhouse check
    that does not look for `hatchling` reports success for a build that cannot succeed.
    """
    names = {
        offline.canonical(offline.requirement_name(item)) for item in offline.runtime_requirements()
    }

    assert {"hatchling", "mcp", "psycopg"} <= names
    assert offline.canonical("mdqm") not in names, "the project cannot build itself"


def test_build_requirements_come_from_build_system(offline: ModuleType, tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "demo"\ndependencies = ["duckdb"]\n'
        '[build-system]\nrequires = ["hatchling", "demo"]\n',
        encoding="utf-8",
    )

    # The self-reference is dropped; every other backend requirement is kept.
    assert offline.build_requirements(pyproject) == ["hatchling"]
    assert offline.runtime_requirements(pyproject, extras=()) == ["duckdb", "hatchling"]


def test_missing_hatchling_is_reported_as_missing(offline: ModuleType, tmp_path: Path) -> None:
    """The wheelhouse path must fail on it, not print "offline install check passed"."""
    (tmp_path / "mcp-2.0.0-py3-none-any.whl").write_text("", encoding="utf-8")
    staged = offline.scan_wheelhouse(tmp_path, target_platform="manylinux2014_x86_64")

    missing = staged.missing(["mcp>=2,<3", "hatchling"])

    assert missing == ["hatchling"]


# --- B2: staged wheels are for the image's platform, not the developer's --------------


def test_the_default_target_platform_is_the_images(offline: ModuleType) -> None:
    assert offline.DEFAULT_TARGET_PLATFORM == "manylinux2014_x86_64"


@pytest.mark.parametrize(
    ("platform_tag", "installable"),
    [
        ("manylinux2014_x86_64", True),
        ("manylinux_2_17_x86_64", True),
        ("linux_x86_64", True),
        ("any", True),
        # The bug: this is what a wheelhouse staged on a Windows dev box holds.
        ("win_amd64", False),
        ("macosx_11_0_arm64", False),
        ("manylinux2014_aarch64", False),
        ("musllinux_1_2_x86_64", False),
    ],
)
def test_platform_families_for_a_manylinux_image(
    offline: ModuleType, platform_tag: str, installable: bool
) -> None:
    assert offline.satisfies_platform(platform_tag, "manylinux2014_x86_64") is installable, (
        platform_tag
    )


def test_win_amd64_is_accepted_when_explicitly_asked_for(offline: ModuleType) -> None:
    """Staging for a Windows target stays possible; it just is not the default."""
    assert offline.satisfies_platform("win_amd64", "win_amd64") is True
    assert offline.satisfies_platform("manylinux2014_x86_64", "win_amd64") is False


@pytest.mark.parametrize(
    ("filename", "tags"),
    [
        ("duckdb-1.1.0-cp312-cp312-manylinux_2_17_x86_64.whl", ("manylinux_2_17_x86_64",)),
        ("ruamel_yaml-0.19.1-py3-none-any.whl", ("any",)),
        (
            "duckdb-1.1.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
            ("manylinux2014_x86_64", "manylinux_2_17_x86_64"),
        ),
    ],
)
def test_wheel_platform_tags(offline: ModuleType, filename: str, tags: tuple[str, ...]) -> None:
    assert offline.wheel_platform_tags(filename) == tags


def test_a_windows_wheelhouse_fails_the_linux_requirement(
    offline: ModuleType, tmp_path: Path
) -> None:
    """The proven bug: a `win_amd64` wheel must not "satisfy" a manylinux requirement."""
    binary = "psycopg_binary-3.3.9-cp312-cp312-win_amd64.whl"
    (tmp_path / binary).write_text("", encoding="utf-8")
    (tmp_path / "ruamel_yaml-0.19.1-py3-none-any.whl").write_text("", encoding="utf-8")

    staged = offline.scan_wheelhouse(tmp_path, target_platform="manylinux2014_x86_64")

    assert staged.wheels == {"ruamel-yaml": "ruamel_yaml-0.19.1-py3-none-any.whl"}
    assert staged.incompatible == (f"{binary} (not manylinux2014_x86_64)",)
    assert staged.missing(["psycopg[binary]>=3.2"]) == ["psycopg[binary]>=3.2"]


def test_the_download_asks_pip_for_the_image_platform(offline: ModuleType, tmp_path: Path) -> None:
    """Without these, pip stages wheels for the running interpreter (cp312-win_amd64 here)."""
    argv = offline.download_argv(tmp_path, tmp_path / "requirements.txt", "manylinux2014_x86_64")

    def value_of(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert value_of("--platform") == "manylinux2014_x86_64"
    assert value_of("--implementation") == "cp"
    assert value_of("--python-version") == "3.12"
    assert value_of("--abi") == "cp312"
    # pip rejects --platform without it, and it is the wheels-only contract anyway.
    assert "--only-binary=:all:" in argv


# --- B3: a Linux wheelhouse cannot be staged natively from Windows --------------------
#
# `--platform` filters which wheels are eligible; environment markers are still evaluated for the
# interpreter that runs pip. On Windows `mcp` requires `pywin32` for `sys_platform == "win32"` and
# there is no manylinux wheel of it: ResolutionImpossible, whatever the flags say.


@pytest.mark.parametrize(
    ("host", "target_platform", "needs_container"),
    [
        # The reproduced bug: Linux target, Windows host.
        ("win32", "manylinux2014_x86_64", True),
        ("darwin", "manylinux_2_17_x86_64", True),
        ("linux", "manylinux2014_x86_64", False),
        # Same-family staging resolves its own markers correctly, and is explicitly allowed.
        ("win32", "win_amd64", False),
    ],
)
def test_a_linux_target_from_a_non_linux_host_needs_a_container(
    offline: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    host: str,
    target_platform: str,
    needs_container: bool,
) -> None:
    monkeypatch.setattr(offline, "host_os", lambda: host)

    reason = offline.staging_container_reason(target_platform)

    assert (reason is not None) is needs_container
    if needs_container:
        # The message has to name the mechanism, or the operator cannot tell why it refuses.
        assert "environment markers" in reason
        assert host in reason


def test_the_container_download_is_the_same_contract_plus_the_target_os(
    offline: ModuleType, tmp_path: Path
) -> None:
    argv = offline.container_download_argv(
        tmp_path, tmp_path / "requirements.txt", "manylinux2014_x86_64", "python:3.12-slim"
    )

    assert argv[:4] == ["docker", "run", "--rm", "--volume"]
    assert "python:3.12-slim" in argv
    # The container's own python runs pip: the host's interpreter would bring the host's markers.
    assert sys.executable not in argv
    assert "--only-binary=:all:" in argv
    assert argv[argv.index("--platform") + 1] == "manylinux2014_x86_64"
    assert argv[argv.index("--abi") + 1] == "cp312"
    # Mounted, so the wheels land on the host: the wheelhouse and the requirement list.
    mounts = [value for flag, value in zip(argv, argv[1:], strict=False) if flag == "--volume"]
    assert any(mount.endswith(":/wheelhouse") for mount in mounts)
    assert any(mount.endswith(":/requirements.txt:ro") for mount in mounts)
    assert argv[argv.index("--dest") + 1] == "/wheelhouse"
    assert argv[argv.index("--requirement") + 1] == "/requirements.txt"


def test_a_mirror_reaches_the_staging_container(
    offline: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A site that can only see an internal index must stage through it, not through PyPI."""
    monkeypatch.setenv("PIP_INDEX_URL", "https://pypi.internal/simple")
    monkeypatch.setenv("PIP_TRUSTED_HOST", "pypi.internal")

    argv = offline.container_download_argv(
        tmp_path, tmp_path / "requirements.txt", "manylinux2014_x86_64", "python:3.12-slim"
    )

    assert "PIP_INDEX_URL=https://pypi.internal/simple" in argv
    assert "PIP_TRUSTED_HOST=pypi.internal" in argv


def test_the_container_install_uses_no_index_and_can_lose_the_network(
    offline: ModuleType, tmp_path: Path
) -> None:
    """`--no-download` runs air-gapped for real: no index flag, and no network at all."""
    argv = offline.container_install_argv(tmp_path, tmp_path, "python:3.12-slim", network_free=True)

    assert argv[argv.index("--network") + 1] == "none"
    joined = " ".join(argv)
    assert "--no-index" in joined
    assert "--find-links /wheelhouse" in joined
    assert "mdqm[pg]" in joined
    # The image's interpreter, not the host's: a manylinux wheelhouse has no win_amd64 binaries.
    assert sys.executable not in argv
    assert "/venv/bin/mdqm --version" in joined


def test_the_container_is_online_while_downloading(offline: ModuleType, tmp_path: Path) -> None:
    """Staging still has to reach an index; only the air-gapped re-run may not."""
    argv = offline.container_install_argv(tmp_path, tmp_path, "python:3.12-slim")
    assert "--network" not in argv


def test_no_download_never_runs_a_pip_download(
    offline: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The air-gapped path must not touch an index -- the build backend included."""
    ran: list[list[str]] = []
    monkeypatch.setattr(offline, "_run", lambda command, cwd=None: ran.append(list(command)))

    staged = offline.stage_wheelhouse(tmp_path, download=False, target_platform="win_amd64")

    assert staged.wheels == {}
    assert ran, "the project wheel still has to be built"
    assert not [command for command in ran if "download" in command]
    assert all("--no-index" in command for command in ran)


# --- B4: a staged wheelhouse is closed, not just named --------------------------------


def _fake_wheel(path: Path, name: str, *, requires: tuple[str, ...] = ()) -> Path:
    """A real zip with a dist-info/METADATA, so the closure check reads what pip would read."""
    metadata = ["Metadata-Version: 2.1", f"Name: {name}", "Version: 1.0"]
    metadata += [f"Requires-Dist: {requirement}" for requirement in requires]
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"{name.replace('-', '_')}-1.0-py3-none-any.whl"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr(
            f"{name.replace('-', '_')}-1.0.dist-info/METADATA",
            "\n".join(metadata) + "\n",
        )
    return target


def test_the_build_backend_closure_is_checked_not_assumed(
    offline: ModuleType, tmp_path: Path
) -> None:
    """`hatchling` alone cannot build this project; packaging/pathspec/pluggy/trove-classifiers
    are unmarked requirements of it, so an absent wheel is a hole."""
    _fake_wheel(
        tmp_path,
        "hatchling",
        requires=(
            "packaging>=24",
            "pathspec>=0.12",
            "pluggy>=1.0",
            'trove-classifiers; extra == "metadata"',  # marked: the target decides, not us
        ),
    )
    staged = offline.scan_wheelhouse(tmp_path, target_platform="manylinux2014_x86_64")

    gaps = offline.closure_gaps(tmp_path, staged)

    assert gaps == [
        "hatchling requires 'packaging>=24', which is not staged",
        "hatchling requires 'pathspec>=0.12', which is not staged",
        "hatchling requires 'pluggy>=1.0', which is not staged",
    ]


def test_a_closed_wheelhouse_has_no_gaps(offline: ModuleType, tmp_path: Path) -> None:
    _fake_wheel(tmp_path, "hatchling", requires=("packaging>=24",))
    _fake_wheel(tmp_path, "packaging")
    staged = offline.scan_wheelhouse(tmp_path, target_platform="manylinux2014_x86_64")

    assert offline.closure_gaps(tmp_path, staged) == []


def test_a_wheel_that_is_not_a_zip_is_not_a_crash(offline: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "broken-1.0-py3-none-any.whl").write_text("not a zip", encoding="utf-8")
    staged = offline.scan_wheelhouse(tmp_path, target_platform="manylinux2014_x86_64")

    assert offline.closure_gaps(tmp_path, staged) == []


# --- the operator can refuse the container --------------------------------------------


def test_no_docker_fails_loudly_instead_of_staging_for_the_wrong_os(
    offline: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(offline, "host_os", lambda: "win32")
    monkeypatch.setattr(offline, "_run", lambda command, cwd=None: None)

    code = offline.main(["--no-download", "--no-docker", "--wheelhouse", str(tmp_path)])
    printed = capsys.readouterr().out

    assert code == 1
    assert "--no-docker" in printed
    assert "pywin32" in printed
    assert "offline install check passed" not in printed


def test_a_missing_docker_is_a_named_failure(
    offline: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(offline, "host_os", lambda: "win32")
    monkeypatch.setattr(offline.shutil, "which", lambda name: None)
    monkeypatch.setattr(offline, "_run", lambda command, cwd=None: None)

    code = offline.main(["--no-download", "--wheelhouse", str(tmp_path)])
    printed = capsys.readouterr().out

    assert code == 1
    assert "no 'docker' on PATH" in printed
    assert "offline install check passed" not in printed


# --- H4: the no-download path never reaches an index ----------------------------------


def test_building_the_project_wheel_uses_no_index(offline: ModuleType, tmp_path: Path) -> None:
    """`pip wheel` creates an isolated build env, so without --no-index it resolves hatchling
    from the configured index even when the operator asked for the air-gapped path."""
    argv = offline.wheel_argv(tmp_path)

    assert "--no-index" in argv
    assert argv[argv.index("--find-links") + 1] == str(tmp_path)
    assert argv[argv.index("--wheel-dir") + 1] == str(tmp_path)
    assert str(offline.REPO_ROOT) == argv[-1]


# --- the recipe: two install modes, group-writable trees, a small build context -------


def test_the_recipe_supports_a_mirror_and_a_wheelhouse(offline: ModuleType) -> None:
    """The target network has internal mirrors; a fully disconnected site has none."""
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "ARG INSTALL_MODE=mirror" in text
    assert "--no-index --find-links /wheelhouse" in text
    assert "--index-url" in text
    assert "PIP_TRUSTED_HOST" in text
    # Neither mode may need a compiler: an sdist would be built next to the index.
    assert text.count("--only-binary=:all:") >= 2


def test_writable_trees_are_group_owned_not_chowned_to_a_fixed_uid(offline: ModuleType) -> None:
    """OpenShift's restricted-v2 SCC runs an arbitrary namespace UID in GID 0."""
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "chgrp -R 0 /data /cache /semantic" in text
    assert "chmod -R g=u /data /cache /semantic" in text
    assert "chown 10001" not in text


def test_dockerignore_keeps_transcripts_and_secrets_out_of_the_context(
    offline: ModuleType,
) -> None:
    """Every COPY is explicit today, but the context is uploaded to an in-cluster builder."""
    entries = {
        line.strip()
        for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert {".pi", "deploy", ".env", ".env.*", "AGENTS.md", "CLAUDE.md"} <= entries
    assert {"data", "*.duckdb", "*.sqlite"} <= entries
    # The wheelhouse install mode reads it, so it must stay in the context.
    assert "wheelhouse" not in entries


def test_the_wheelhouse_directory_ships_a_placeholder(offline: ModuleType) -> None:
    """`COPY wheelhouse/` fails the build outright when the directory is absent."""
    assert (REPO_ROOT / "wheelhouse" / ".gitkeep").is_file()


def test_a_pip_failure_is_reported_as_a_problem_not_a_traceback(
    offline: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: object
) -> None:
    """`--no-download` on an incomplete wheelhouse must fail loudly and say which command failed.

    Reviewer's bug: it printed "offline install check passed" instead. A CalledProcessError out of
    pip is the normal shape of that failure, so it is what this simulates -- no pip is run here.
    """

    def refuse(command: list[str], cwd: Path | None = None) -> None:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(offline, "_run", refuse)

    code = offline.main(["--no-download", "--wheelhouse", str(tmp_path)])
    printed = capsys.readouterr().out  # type: ignore[attr-defined]

    assert code == 1
    assert "offline install check passed" not in printed
    assert "FAILED" in printed
    assert "--no-index" in printed, "the message must name the command that failed"


def test_a_wrong_platform_wheel_fails_the_run_and_is_named(
    offline: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The reproduction end to end: a win_amd64 wheel in a Linux wheelhouse is a failed check."""
    (tmp_path / "psycopg_binary-3.3.9-cp312-cp312-win_amd64.whl").write_text("", encoding="utf-8")
    monkeypatch.setattr(offline, "host_os", lambda: "linux")
    monkeypatch.setattr(offline, "_run", lambda command, cwd=None: None)
    monkeypatch.setattr(offline, "fresh_install", lambda *args, **kwargs: None)

    code = offline.main(["--no-download", "--wheelhouse", str(tmp_path)])
    printed = capsys.readouterr().out

    assert code == 1
    assert "wheels staged for another platform" in printed
    assert "psycopg_binary-3.3.9-cp312-cp312-win_amd64.whl" in printed
    # And the build backend is reported by name, not swallowed by the wrong-platform complaint: an
    # empty-of-Linux-wheelhouse means hatchling is missing too, and the operator must see both.
    assert "no wheel staged for:" in printed
    assert "hatchling" in printed
    assert "offline install check passed" not in printed


class TestWheelhouse:
    def test_classifies_wheels_and_source_archives(
        self, offline: ModuleType, tmp_path: Path
    ) -> None:
        (tmp_path / "django_pkg-1.0-py3-none-any.whl").write_text("", encoding="utf-8")
        binary = "psycopg_binary-3.3.9-cp312-cp312-win_amd64.whl"
        (tmp_path / binary).write_text("", encoding="utf-8")
        (tmp_path / "legacy-0.1.tar.gz").write_text("", encoding="utf-8")
        (tmp_path / "requirements.txt").write_text("", encoding="utf-8")

        # Staged for Windows on purpose: a win_amd64 wheel is installable only when asked for.
        staged = offline.scan_wheelhouse(tmp_path, target_platform="win_amd64")

        assert set(staged.wheels) == {"django-pkg", "psycopg-binary"}
        assert staged.source_archives == ("legacy-0.1.tar.gz",)
        assert staged.incompatible == ()

    def test_missing_is_canonical_and_sorted(self, offline: ModuleType, tmp_path: Path) -> None:
        (tmp_path / "ruamel_yaml-0.19.1-py3-none-any.whl").write_text("", encoding="utf-8")
        staged = offline.scan_wheelhouse(tmp_path, target_platform="manylinux2014_x86_64")

        missing = staged.missing(["ruamel.yaml>=0.18", "psycopg[binary]>=3.2", "duckdb>=1.1"])

        assert missing == ["duckdb>=1.1", "psycopg[binary]>=3.2"]

    def test_absent_directory_is_empty_not_an_error(
        self, offline: ModuleType, tmp_path: Path
    ) -> None:
        staged = offline.scan_wheelhouse(tmp_path / "nope", target_platform="win_amd64")
        assert staged.wheels == {}
        assert staged.source_archives == ()
        assert staged.incompatible == ()


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
