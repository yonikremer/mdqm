"""Start, stop and inspect the mdqm docker-compose services. Standard library only.

    python scripts/dev.py up pg              # start PostgreSQL, exit 0 only once it is healthy
    python scripts/dev.py up pg es           # several profiles at once
    python scripts/dev.py status             # what is running, per profile
    python scripts/dev.py down               # stop everything and remove the volumes
    python scripts/dev.py env                # create .env from .env.example (throwaway values)

`up` uses `docker compose up --wait`, which blocks until every healthcheck passes and exits
non-zero when a service never becomes healthy, so a green exit really means "ready". `down` uses
`--volumes`, so a rerun starts from the init scripts again.

Windows-first: no bash, no shell pipeline, paths via pathlib, UTF-8 everywhere. Read docs/dev-env.md
first; it lists the RAM each profile needs.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
ENV_FILE = REPO_ROOT / ".env"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
PROFILES = ("pg", "oracle", "mssql", "es", "api", "oss", "server")
"""Compose profiles. `oss` is the open-source-only stack (Postgres + MariaDB + OpenSearch + api)."""
HELP_PROFILES = ", ".join(PROFILES)

MISSING_DOCKER = """docker was not found on PATH, so the compose profiles cannot be used.

  install Docker Desktop (Linux containers, WSL2 backend): https://docs.docker.com/desktop/
  then check: docker compose version   (needs Compose v2.20 or newer for `up --wait`)

Everything that does not need a database still works: python scripts/check.py
See docs/dev-env.md for the full picture."""


def find_docker() -> str | None:
    """Path to the `docker` executable, or None when no container runtime is installed."""
    return shutil.which("docker")


def ensure_env() -> None:
    """Create `.env` from `.env.example` once, so a fresh clone does not fail on missing values."""
    if ENV_FILE.exists() or not ENV_EXAMPLE.exists():
        return
    ENV_FILE.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    print(f"created {ENV_FILE.name} from {ENV_EXAMPLE.name} (throwaway dev credentials)")


def compose(*args: str) -> list[str]:
    """The `docker compose` command line for this project, with the file pinned explicitly."""
    return ["docker", "compose", "--file", str(COMPOSE_FILE), *args]


def profile_args(profiles: tuple[str, ...] | list[str]) -> list[str]:
    args: list[str] = []
    for profile in profiles:
        args.extend(["--profile", profile])
    return args


def run(command: list[str]) -> int:
    print(" ".join(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT).returncode


def cmd_up(profiles: list[str]) -> int:
    return run(compose(*profile_args(profiles), "up", "--detach", "--wait"))


def cmd_down() -> int:
    # Every profile is named so services started with any profile are stopped too.
    return run(compose(*profile_args(PROFILES), "down", "--volumes", "--remove-orphans"))


def cmd_status(profiles: list[str]) -> int:
    if not profiles:
        profiles = list(PROFILES)
    return run(compose(*profile_args(profiles), "ps", "--all"))


def cmd_env() -> int:
    if ENV_FILE.exists():
        print(f"{ENV_FILE.name} already exists; not touching it")
        return 0
    ensure_env()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/dev.py",
        description="Run the mdqm docker-compose services, one profile per task.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    up = sub.add_parser("up", help="start the given profiles and wait until they are healthy")
    up.add_argument(
        "profiles", nargs="+", choices=PROFILES, metavar="profile", help=f"one or more of: {HELP_PROFILES}"
    )

    sub.add_parser("down", help="stop every service and remove the volumes")

    status = sub.add_parser("status", help="show the state of the services")
    status.add_argument(
        "profiles", nargs="*", choices=PROFILES, metavar="profile", help=f"default: all ({HELP_PROFILES})"
    )

    sub.add_parser("env", help="create .env from .env.example if it is missing")

    args = parser.parse_args(argv)

    if args.action == "env":
        return cmd_env()

    if find_docker() is None:
        print(MISSING_DOCKER, file=sys.stderr)
        return 2

    ensure_env()
    if args.action == "up":
        return cmd_up(list(args.profiles))
    if args.action == "down":
        return cmd_down()
    return cmd_status(list(args.profiles))


if __name__ == "__main__":
    raise SystemExit(main())
