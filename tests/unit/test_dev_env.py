"""Tests for the dev environment files: `docker-compose.yml`, `.env.example` and `scripts/dev.py`.

Task D01. Everything here is a *file* check — no container runtime, no socket, no network — because
the compose services cannot be started on a machine without Docker. Starting them, the `pg` seed,
`pg_stat_statements`, the `mdqm_ro` role and the healthchecks are therefore NOT covered by tests;
`progress/D01.md` says so explicitly.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"
GITIGNORE_PATH = REPO_ROOT / ".gitignore"


@dataclass(frozen=True)
class ServiceSpec:
    """What the tests claim about one compose service.

    ``container_port is None`` marks a one-shot service that publishes nothing (it runs, reports and
    exits), so the loopback/healthcheck rules below do not apply to it; the one-shot rules do.
    """

    profiles: tuple[str, ...]
    image_prefix: str
    container_port: int | None = None
    healthcheck: bool = True


# service -> spec. `pg` and `api` also belong to `oss`: the open-source-only stack is those two plus
# MariaDB and OpenSearch (owner request, see progress/OSS-mocks.md).
SERVICES: dict[str, ServiceSpec] = {
    "pg": ServiceSpec(("pg", "oss"), "postgres:", 5432),
    "oracle": ServiceSpec(("oracle",), "gvenzl/oracle-free:", 1521),
    "mssql": ServiceSpec(("mssql",), "mcr.microsoft.com/mssql/server:", 1433),
    "es": ServiceSpec(("es",), "docker.elastic.co/elasticsearch/elasticsearch:", 9200),
    "api": ServiceSpec(("api", "oss"), "mdqm/mock-api:", 8000),
    "mariadb": ServiceSpec(("oss",), "mariadb:", 3306),
    "opensearch": ServiceSpec(("oss",), "opensearchproject/opensearch:", 9200),
    "opensearch-init": ServiceSpec(("oss",), "curlimages/curl:", None, healthcheck=False),
    # The server itself, from this repo's Dockerfile: the only service that runs mdqm, so
    # `--profile server` is what makes a local end-to-end deployment possible.
    "server": ServiceSpec(("server",), "mdqm/server:", 8765),
}

OSS_PROFILE = {"pg", "mariadb", "opensearch", "opensearch-init", "api"}
"""What `docker compose --profile oss up -d` starts: no Oracle, no SQL Server, no Elasticsearch."""

VAR_REFERENCE = re.compile(r"\$\{([A-Z0-9_]+)")


def load_yaml(path: Path) -> Any:
    yaml = YAML(typ="safe")
    with path.open(encoding="utf-8") as handle:
        return yaml.load(handle)


def load_script(name: str) -> types.ModuleType:
    """Import `scripts/<name>.py` as a module. `scripts/` is not a package, so do it by path."""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"dev_env_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def env_example_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip()
    return values


def compose_text() -> str:
    return COMPOSE_PATH.read_text(encoding="utf-8")


def services() -> dict[str, Any]:
    document = load_yaml(COMPOSE_PATH)
    return dict(document["services"])


def test_compose_parses_and_declares_every_service() -> None:
    document = load_yaml(COMPOSE_PATH)
    assert document["name"] == "mdqm", "the project name keeps container/volume names in mdqm_*"
    assert set(document["services"]) == set(SERVICES)
    assert set(document["volumes"]) >= {
        "pg-data",
        "oracle-data",
        "mssql-data",
        "es-data",
        "mariadb-data",
        "opensearch-data",
        "server-data",
        "server-cache",
    }


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_each_service_declares_the_profiles_we_claim(service: str) -> None:
    assert services()[service]["profiles"] == list(SERVICES[service].profiles)


def test_oss_profile_is_the_open_source_stack_only() -> None:
    claimed = {name for name, definition in services().items() if "oss" in definition["profiles"]}
    assert claimed == OSS_PROFILE
    for name, spec in SERVICES.items():
        if name in OSS_PROFILE:
            continue
        assert "oss" not in spec.profiles, f"{name} must stay out of the open-source-only stack"


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_each_image_is_pinned_and_never_latest(service: str) -> None:
    definition = services()[service]
    if "build" in definition:
        # Images this repo builds itself are tagged by us, so they carry no upstream version.
        assert service in {"api", "server"}, f"{service} builds an image nobody expected"
        assert (REPO_ROOT / definition["build"]["dockerfile"]).is_file()
    image = definition["image"]
    assert image.startswith(SERVICES[service].image_prefix), f"{service}: unexpected image {image}"
    tag = image.rsplit(":", 1)[1]
    assert tag not in {"latest", "stable", "main"}, f"{service}: floating tag {tag}"
    if "build" not in definition:
        # A published image must name a version; the locally built one is tagged by us.
        assert re.search(r"\d", tag), f"{service}: tag {tag} does not name a version"
    assert "@" not in image, "a digest cannot be updated together with the code that pins it"


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_each_service_has_a_healthcheck(service: str) -> None:
    definition = services()[service]
    if not SERVICES[service].healthcheck:
        # A one-shot loader has nothing to poll; it must instead not restart and publish nothing.
        assert definition["restart"] == "no"
        assert "ports" not in definition
        return
    healthcheck = definition["healthcheck"]
    assert healthcheck["test"], "an empty healthcheck is not a healthcheck"
    assert healthcheck["retries"] > 0
    assert int(str(healthcheck["interval"]).rstrip("s")) > 0


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_ports_are_loopback_only_and_not_the_engine_default(service: str) -> None:
    # 127.0.0.1:${MDQM_PG_PORT:-55432}:5432 — the host port is the only variable part.
    pattern = re.compile(r"^127\.0\.0\.1:\$\{(?P<var>[A-Z0-9_]+)(?::-\d+)?\}:(?P<container>\d+)$")
    published = services()[service].get("ports")
    if SERVICES[service].container_port is None:
        # A one-shot service publishes nothing, which the healthcheck test already asserts.
        assert published is None
        return
    assert published is not None and len(published) == 1
    match = pattern.match(str(published[0]))
    assert match is not None, f"{service}: {published[0]!r} is not loopback+variable+non-default"
    assert match.group("container") == str(SERVICES[service].container_port)
    assert match.group("var") in env_example_values(), "the host port is overridable from .env"


def test_every_variable_compose_needs_is_documented_in_env_example() -> None:
    needed = set(VAR_REFERENCE.findall(compose_text()))
    assert needed, "the compose file is expected to take its credentials from the environment"
    missing = needed - set(env_example_values())
    assert not missing, f"variables without a value in .env.example: {sorted(missing)}"


def test_env_example_holds_only_throwaway_dev_credentials() -> None:
    values = env_example_values()
    secrets = {k: v for k, v in values.items() if "PASSWORD" in k or k.endswith("_KEY")}
    assert set(secrets) == {
        "MDQM_PG_PASSWORD",
        "MDQM_RO_PASSWORD",
        "MDQM_ORACLE_PASSWORD",
        "MDQM_MSSQL_PASSWORD",
        "MDQM_MOCK_API_KEY",
        "MDQM_MARIADB_ROOT_PASSWORD",
    }
    for key, value in secrets.items():
        assert value.startswith("MqmDev"), f"{key} does not look like a throwaway dev value"
        assert len(value) >= 8, f"{key} is too short for every engine's password policy"
    # SQL Server demands 3 of 4 character classes; Oracle rejects a leading digit.
    assert re.search(r"[A-Z].*[a-z0-9]", values["MDQM_MSSQL_PASSWORD"])
    assert re.search(r"[^A-Za-z0-9]", values["MDQM_MSSQL_PASSWORD"])
    assert not values["MDQM_ORACLE_PASSWORD"][0].isdigit()


def test_env_creates_server_secrets_once_and_does_not_overwrite(tmp_path: Path) -> None:
    """`dev.py env` must produce the files the server profile mounts, and leave a rotation alone."""
    dev = load_script("dev")
    server = tmp_path / "docker" / "server"
    server.mkdir(parents=True)
    example = server / "tokens.example.json"
    example.write_text('{"tokens":[{"token":"throwaway","user_id":"dev"}]}\n', encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("MDQM_RO_PASSWORD=MqmDevReadOnly2026\n", encoding="utf-8")
    tokens = server / "tokens.json"
    secret = server / "pg_ro_password"
    originals = {
        "SERVER_DIR": dev.SERVER_DIR,
        "TOKENS_EXAMPLE": dev.TOKENS_EXAMPLE,
        "TOKENS_FILE": dev.TOKENS_FILE,
        "PG_RO_SECRET": dev.PG_RO_SECRET,
        "ENV_FILE": dev.ENV_FILE,
    }
    for name, value in (
        ("SERVER_DIR", server),
        ("TOKENS_EXAMPLE", example),
        ("TOKENS_FILE", tokens),
        ("PG_RO_SECRET", secret),
        ("ENV_FILE", env_file),
    ):
        setattr(dev, name, value)
    try:
        dev.ensure_server_secrets()
        assert json.loads(tokens.read_text(encoding="utf-8"))["tokens"][0]["user_id"] == "dev"
        assert secret.read_text(encoding="utf-8").strip() == "MqmDevReadOnly2026"
        tokens.write_text("rotated\n", encoding="utf-8")
        secret.write_text("rotated-password\n", encoding="utf-8")
        dev.ensure_server_secrets()
        assert tokens.read_text(encoding="utf-8") == "rotated\n"
        assert secret.read_text(encoding="utf-8") == "rotated-password\n"
    finally:
        for name, value in originals.items():
            setattr(dev, name, value)


def test_env_file_itself_is_git_ignored() -> None:
    lines = {line.strip() for line in GITIGNORE_PATH.read_text(encoding="utf-8").splitlines()}
    assert ".env" in lines, "a real .env must never be committable"


def test_no_windows_specific_paths_in_the_compose_file() -> None:
    text = compose_text()
    assert "C:\\" not in text and "C:/" not in text and "%USERPROFILE%" not in text
    for definition in services().values():
        for volume in definition.get("volumes", []):
            source = str(volume).split(":")[0]
            assert source.startswith("./") or not source.startswith(("/", ".")), (
                f"bind mount {source!r} must be repo-relative"
            )


def declared_profiles() -> set[str]:
    """Every profile name the compose file actually uses, in a stable order for assertions."""
    return {profile for spec in SERVICES.values() for profile in spec.profiles}


def test_dev_py_help_lists_every_profile() -> None:
    """`--help` must tell a reader which profiles exist, without reading the compose file."""
    up_help = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(REPO_ROOT / "scripts" / "dev.py"), "up", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert up_help.returncode == 0, up_help.stderr
    for profile in declared_profiles():
        assert profile in up_help.stdout
    top_help = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(REPO_ROOT / "scripts" / "dev.py"), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert top_help.returncode == 0, top_help.stderr
    for action in ("up", "down", "status"):
        assert action in top_help.stdout


def test_dev_py_explains_a_missing_container_runtime(tmp_path: Path) -> None:
    """The message a developer on a machine without Docker actually sees."""
    dev = load_script("dev")
    env = dict(os.environ, PATH=str(tmp_path))
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(REPO_ROOT / "scripts" / "dev.py"), "status"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        env=env,
    )
    assert result.returncode == 2
    assert "docker was not found on PATH" in result.stderr
    assert "docs/dev-env.md" in result.stderr
    assert dev.MISSING_DOCKER in result.stderr


def test_status_rejects_an_unknown_profile() -> None:
    """Profile names are still checked, including on Python 3.11 where argparse cannot."""
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(REPO_ROOT / "scripts" / "dev.py"), "status", "nope"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 2
    assert "nope" in result.stderr
    assert "unknown profile" in result.stderr


def test_find_docker_reports_none_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = load_script("dev")
    monkeypatch.setattr(dev.shutil, "which", lambda _name: None)
    assert dev.find_docker() is None
    monkeypatch.setattr(dev.shutil, "which", lambda _name: "/usr/bin/docker")
    assert dev.find_docker() == "/usr/bin/docker"


def test_up_waits_for_health_and_down_removes_volumes() -> None:
    """The two flags the acceptance criteria turn on, asserted on the command builder."""
    dev = load_script("dev")
    assert "--wait" in dev.compose(*dev.profile_args(["pg"]), "up", "--detach", "--wait")
    assert dev.profile_args(["pg", "es"]) == ["--profile", "pg", "--profile", "es"]
    down = dev.compose(*dev.profile_args(dev.PROFILES), "down", "--volumes", "--remove-orphans")
    assert "--volumes" in down
    assert declared_profiles() <= set(dev.PROFILES), (
        "dev.py down must name every profile, or a started service survives it"
    )


@pytest.mark.parametrize(
    ("target", "expected"),
    [("127.0.0.1:55432", ("127.0.0.1", 55432)), ("localhost:80", ("localhost", 80))],
)
def test_wait_for_parses_host_port(target: str, expected: tuple[str, int]) -> None:
    wait_for = load_script("wait_for")
    assert wait_for.parse_host_port(target) == expected


@pytest.mark.parametrize("target", ["127.0.0.1", "127.0.0.1:", ":55432", "host:port"])
def test_wait_for_rejects_a_bad_target(target: str) -> None:
    wait_for = load_script("wait_for")
    with pytest.raises(ValueError):
        wait_for.parse_host_port(target)


def test_wait_until_gives_up_at_the_deadline() -> None:
    wait_for = load_script("wait_for")
    ready, detail = wait_for.wait_until(lambda: (False, "still down"), 0.0, 0.0)
    assert ready is False
    assert detail == "still down"
