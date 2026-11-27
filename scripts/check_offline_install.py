"""Offline-install check (task M08): can this project be installed with no network at all?

The target network is air-gapped (AGENTS.md principle 7 and 8): every dependency must arrive as a
wheel from the internal PyPI mirror, nothing may be compiled at build time, and nothing may be
downloaded at runtime. This script proves that on the current machine:

1. **Allowlist** - read the requirements the image ships (``pyproject.toml``: core plus the extras
   in :data:`IMAGE_EXTRAS`) and inspect the *Dockerfile* for a pinned base image, a non-root user, a
   ``HEALTHCHECK``, no literal credential and no ``pip install`` in the runtime stage. Pure file
   reading, so it runs with no network and is unit-tested (``tests/unit/test_check_offline_install.py``).
2. **Wheelhouse** - build this project's wheel and download the whole dependency closure into a
   wheelhouse with ``--only-binary=:all:``. A dependency that only has a source distribution fails
   here: on the air-gapped network there is no compiler and no network either.
3. **Fresh install** - create a new virtual environment, install ``mdqm[pg]`` from that wheelhouse
   with ``--no-index``, and run a smoke import plus a real SQLite store migration. This is what the
   container's builder stage does, just without a container.

Usage (OS agnostic, no shell built-ins):

    python scripts/check_offline_install.py                      # download, install, smoke
    python scripts/check_offline_install.py --no-download         # air-gapped: use the staged wheels
    python scripts/check_offline_install.py --wheelhouse staged   # stage somewhere else
    python scripts/check_offline_install.py --no-docker           # fail instead of using a container
    python scripts/check_offline_install.py --docker-image registry.internal/mirror/python:3.12-slim

The staged wheels are always for the **image's** platform (:data:`DEFAULT_TARGET_PLATFORM`), never
for the machine that stages them: a ``win_amd64`` wheel cannot install in a ``manylinux`` image, so a
wheel whose platform tag is not in the requested family counts as absent here.

Staging for Linux from a non-Linux host runs pip **inside a Linux container**. ``--platform`` only
filters which wheels are eligible; environment markers are still evaluated for the interpreter that
runs pip. On Windows ``sys_platform == "win32"`` is true, so pip insists on ``pywin32`` for ``mcp``
-- and no manylinux wheel of it exists. Inside ``python:3.12-slim`` the markers resolve for Linux, so
the same command succeeds. See :func:`container_download_argv` and :func:`staging_container_reason`.

For the same reason the **install and smoke steps** then run in that container too: a manylinux
wheelhouse holds no ``win_amd64`` binaries, so installing it into a Windows venv fails on ``duckdb``
and would prove nothing about the image anyway. With ``--no-download`` that container gets
``--network none``, so the air-gapped claim is enforced rather than promised.

Exit code 0 means every check passed. Anything else means the deployment would fail on the
air-gapped network, and the printed problems say what to fix.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import venv
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
DOCKERFILE = REPO_ROOT / "Dockerfile"

#: Extras the image installs. `pg` is the MVP target; `oracle` and `mssql` ship only after their
#: driver verdicts (task A08), so they are not part of the offline contract yet.
IMAGE_EXTRAS: tuple[str, ...] = ("pg",)

#: Names whose last word marks a credential slot: a non-empty literal value there is a leak.
CREDENTIAL_WORDS: frozenset[str] = frozenset(
    {"password", "passwd", "pwd", "token", "secret", "key", "credentials"}
)

#: Suffixes of a source archive: these need a build, so they can never be in the wheelhouse.
SDIST_SUFFIXES: tuple[str, ...] = (".tar.gz", ".tgz", ".zip", ".tar.bz2")

#: The platform the image runs on, and therefore the one every staged wheel must be built for.
#: ``manylinux2014_x86_64`` is the portable glibc tag, so one wheel set serves every Debian-based
#: x86_64 host; a ``win_amd64`` wheel staged on a developer's box is a wheelhouse that cannot
#: install in the image at all.
DEFAULT_TARGET_PLATFORM = "manylinux2014_x86_64"

#: Interpreter the image runs. With ``--platform`` pip demands ``--only-binary=:all:``, which is
#: the contract anyway: a source archive needs a compiler the target does not have.
TARGET_PYTHON_VERSION = "3.12"
TARGET_IMPLEMENTATION = "cp"

#: Container that stages a Linux wheelhouse from a non-Linux host: same interpreter as the image, so
#: the markers and the ``cp312`` ABI resolve for the target instead of the developer's machine. It is
#: only a staging tool -- it never ships -- and any registry mirror of ``python:3.12-slim`` works, by
#: tag or by digest (``--docker-image``).
DEFAULT_DOCKER_IMAGE = "python:3.12-slim"

#: Mount points inside the staging container.
_CONTAINER_WHEELHOUSE = "/wheelhouse"
_CONTAINER_REQUIREMENTS = "/requirements.txt"

#: pip settings a mirror needs. Forwarded from the host env into the container when set, because a
#: machine that can only reach an internal index must stage through the same index.
_FORWARDED_PIP_ENV: tuple[str, ...] = (
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "PIP_TRUSTED_HOST",
    "PIP_CERT",
)

#: Architectures a wheel platform tag can end in, longest first so ``x86_64`` beats ``64``.
_ARCH_SUFFIXES: tuple[str, ...] = (
    "x86_64",
    "aarch64",
    "ppc64le",
    "s390x",
    "i686",
    "win32",
    "amd64",
    "arm64",
)

_CANONICAL_RE = re.compile(r"[-_.]+")
_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

SMOKE = """
import pathlib
import tempfile

import mdqm
import mdqm.adapters.openapi
import mdqm.cli
import mdqm.executor
import mdqm.pipeline
import mdqm.server.app
from mdqm.store.db import Store

with tempfile.TemporaryDirectory() as tmp:
    store = Store(pathlib.Path(tmp) / "mdqm.sqlite")
    store.initialize()  # applies migrations 0001..0008: they must be inside the wheel
    store.close()
print("smoke ok", mdqm.__version__)
"""


# --- pure logic: requirements, wheelhouse, Dockerfile -----------------------------------


def canonical(name: str) -> str:
    """Canonical distribution name (PEP 503), so ``ruamel.yaml`` and ``ruamel-yaml`` match."""
    return _CANONICAL_RE.sub("-", name).lower()


def requirement_name(requirement: str) -> str:
    """The distribution a requirement string names: ``psycopg[binary]>=3.2`` -> ``psycopg``."""
    match = _NAME_RE.match(requirement)
    if match is None:
        raise ValueError(f"not a requirement: {requirement!r}")
    return match.group(1)


def _pyproject(pyproject: Path) -> dict[str, Any]:
    """The parsed `pyproject.toml`.

    Typed as a mapping of ``Any``: ``tomllib`` returns nested tables, so ``object`` would be a lie
    that pushes a ``.get`` on ``object`` into every caller (which is exactly what pyright flagged).
    """
    data: dict[str, Any] = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data


def build_requirements(pyproject: Path = PYPROJECT) -> list[str]:
    """``build-system.requires``: what pip needs in its isolated env to build this project.

    A local directory such as ``.[pg]`` is built from source, so pip creates a build environment
    and installs the build backend (``hatchling``) into it. Offline that backend has to be staged
    like every other wheel, or the install fails with "Could not find a version that satisfies the
    requirement hatchling". A self-reference is dropped: the project cannot build itself.
    """
    data = _pyproject(pyproject)
    build = data.get("build-system", {})
    requires = build.get("requires", []) if isinstance(build, dict) else []
    project = data.get("project", {})
    own = canonical(str(project.get("name", ""))) if isinstance(project, dict) else ""
    return [item for item in requires if canonical(requirement_name(item)) != own]


def runtime_requirements(
    pyproject: Path = PYPROJECT, extras: Sequence[str] = IMAGE_EXTRAS
) -> list[str]:
    """Every requirement the image installs: core, ``extras`` and the build backend.

    Reads ``pyproject.toml`` rather than the installed metadata, because the installed metadata of
    an editable checkout goes stale the moment somebody adds a dependency. The build backend is
    part of the closure: with no network and no ``hatchling`` wheel staged, ``pip install .[pg]``
    fails before it installs anything.
    """
    data = _pyproject(pyproject)
    project = data.get("project", {})
    requirements: list[str] = list(project.get("dependencies", []))
    declared = project.get("optional-dependencies", {})
    for extra in extras:
        if extra not in declared:
            raise ValueError(f"pyproject.toml declares no extra {extra!r}")
        requirements.extend(declared[extra])
    return requirements + build_requirements(pyproject)


def is_sdist(filename: str) -> bool:
    """True for a source archive: something that needs a compiler, which the target cannot do."""
    return filename.endswith(SDIST_SUFFIXES)


def wheel_platform_tags(filename: str) -> tuple[str, ...]:
    """The platform tags of a wheel file, e.g. ``('manylinux_2_17_x86_64',)``.

    ``{name}-{version}(-{build})?-{python}-{abi}-{platform}.whl``: the name and version never
    contain ``-``, so the last ``-`` field is the platform. A platform may compress several tags
    with ``.`` (``manylinux2014_x86_64.manylinux_2_17_x86_64``).
    """
    return tuple(filename[: -len(".whl")].rsplit("-", 1)[-1].split("."))


def platform_family(tag: str) -> str:
    """The OS family a platform tag names: ``manylinux_2_17_x86_64`` -> ``manylinux``."""
    return tag.split("_", 1)[0].lower()


def platform_arch(tag: str) -> str | None:
    """The architecture a platform tag names, or ``None`` for an architecture-free tag."""
    lowered = tag.lower()
    for arch in _ARCH_SUFFIXES:
        if lowered == arch or lowered.endswith("_" + arch):
            return arch
    return None


def satisfies_platform(platform_tag: str, target_platform: str) -> bool:
    """True when a wheel built for ``platform_tag`` can be installed on ``target_platform``.

    Architecture matters as much as the OS: a ``manylinux2014_aarch64`` wheel does not install on
    an ``x86_64`` image. ``any`` is a pure-Python wheel and installs everywhere.
    """
    if platform_tag.lower() == "any":
        return True
    family = platform_family(platform_tag)
    wanted = platform_family(target_platform)
    if wanted.startswith("manylinux"):
        # Any manylinux generation, plus a plain linux wheel (pip accepts those locally). Never
        # musl, Windows or macOS: this image is glibc Linux.
        if not (family.startswith("manylinux") or family == "linux"):
            return False
    elif family != wanted:
        return False
    wheel_arch, wanted_arch = platform_arch(platform_tag), platform_arch(target_platform)
    return wanted_arch is None or wheel_arch is None or wheel_arch == wanted_arch


def host_os() -> str:
    """The OS family staging runs on: ``linux``, ``win32``, ``darwin`` or whatever ``sys.platform``."""
    return "linux" if sys.platform.startswith("linux") else sys.platform


def staging_container_reason(target_platform: str) -> str | None:
    """Why a Linux wheelhouse cannot be staged on this host, or ``None`` when pip can do it here.

    ``--platform`` only decides which *wheels* are eligible. Environment markers are still
    evaluated for the interpreter running pip, so on Windows ``mcp`` pulls ``pywin32``
    (``sys_platform == "win32"``) and the resolve dies with ``ResolutionImpossible``: there is no
    manylinux wheel of ``pywin32``. The only way to get the target's markers evaluated is to run
    pip on the target's OS, which is what the staging container is for.
    """
    wanted = platform_family(target_platform)
    target_is_linux = wanted.startswith("manylinux") or wanted == "linux"
    if not target_is_linux or host_os() == "linux":
        return None
    return (
        f"target {target_platform} is Linux but this host is {host_os()}: pip would evaluate "
        "environment markers for this machine (win_amd64 staging insists on pywin32, which has "
        "no manylinux wheel), so the wheels must be staged inside a Linux container"
    )


@dataclass(frozen=True)
class Wheelhouse:
    """What a directory of staged artefacts contains, keyed by canonical distribution name.

    ``wheels`` holds only what the target can install. A wheel for another platform is not a
    staged wheel: it goes to ``incompatible``, and :meth:`missing` therefore reports its
    requirement as absent. A check that counted a ``win_amd64`` wheel as satisfying a
    ``manylinux`` requirement would pass on a wheelhouse that cannot install in the image.
    """

    wheels: dict[str, str]
    source_archives: tuple[str, ...] = ()
    incompatible: tuple[str, ...] = ()

    def missing(self, requirements: Iterable[str]) -> list[str]:
        """The requirements with no installable wheel staged. Empty means the closure is complete."""
        absent = [
            requirement
            for requirement in requirements
            if canonical(requirement_name(requirement)) not in self.wheels
        ]
        return sorted(absent)


def scan_wheelhouse(directory: Path, *, target_platform: str) -> Wheelhouse:
    """Classify every file in ``directory``: installable wheels, wrong-platform wheels, sdists."""
    wheels: dict[str, str] = {}
    incompatible: list[str] = []
    sdists: list[str] = []
    if not directory.is_dir():
        return Wheelhouse(wheels={}, source_archives=(), incompatible=())
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.name.endswith(".whl"):
            # The name never contains '-', so the first field is the distribution.
            name = canonical(path.name.split("-")[0])
            if any(
                satisfies_platform(tag, target_platform) for tag in wheel_platform_tags(path.name)
            ):
                wheels[name] = path.name
            else:
                incompatible.append(f"{path.name} (not {target_platform})")
        elif is_sdist(path.name):
            sdists.append(path.name)
    return Wheelhouse(
        wheels=wheels, source_archives=tuple(sdists), incompatible=tuple(incompatible)
    )


def _metadata(wheel: Path) -> str:
    """The text of a wheel's ``*.dist-info/METADATA``, or ``""`` for a wheel without one."""
    try:
        with zipfile.ZipFile(wheel) as archive:
            entry = next(
                (name for name in archive.namelist() if name.endswith(".dist-info/METADATA")), None
            )
            return "" if entry is None else archive.read(entry).decode("utf-8", "replace")
    except (OSError, zipfile.BadZipFile):
        return ""


def unmarked_requirements(wheel: Path) -> list[str]:
    """A wheel's ``Requires-Dist`` entries that carry no environment marker at all.

    Only these are checked for closure: a marked requirement depends on the *target's* environment,
    and pip inside the staging container is the authority on that. An entry without a marker is
    required everywhere, so an absent wheel for it is a hole in the wheelhouse whatever the target.
    """
    return [
        line.split(":", 1)[1].strip()
        for line in _metadata(wheel).splitlines()
        if line.lower().startswith("requires-dist:") and ";" not in line
    ]


def closure_gaps(directory: Path, staged: Wheelhouse) -> list[str]:
    """Staged wheels whose *unmarked* requirements are missing from the same wheelhouse.

    This is what makes "the build backend is staged" mean its closure too: ``hatchling`` alone
    cannot build this project, it needs ``packaging``, ``pathspec``, ``pluggy`` and
    ``trove-classifiers``, and none of those carries a platform marker.
    """
    gaps: list[str] = []
    for name, filename in sorted(staged.wheels.items()):
        for requirement in unmarked_requirements(directory / filename):
            if canonical(requirement_name(requirement)) not in staged.wheels:
                gaps.append(f"{name} requires {requirement!r}, which is not staged")
    return sorted(gaps)


def _instructions(text: str) -> list[str]:
    """Dockerfile instructions with comments dropped and ``\\`` continuations joined."""
    joined: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        buffer += line
        joined.append(buffer.strip())
        buffer = ""
    if buffer.strip():
        joined.append(buffer.strip())
    return joined


def _is_pinned(image: str) -> str | None:
    """Reason a ``FROM`` reference is not reproducible, or ``None`` when it is."""
    if image == "scratch" or "@sha256:" in image:
        return None
    tag_part = image.rsplit("/", 1)[-1]
    if ":" not in tag_part:
        return f"{image}: no tag, so a rebuild silently changes the base image"
    if tag_part.rsplit(":", 1)[-1] == "latest":
        return f"{image}: 'latest' moves under the image"
    return None


def _credential_literal(instruction: str) -> str | None:
    """A credential-shaped name given a literal value, e.g. ``ARG PGPASSWORD=hunter2``.

    Names are matched by suffix, not by word: `PGPASSWORD` has no separator. That makes
    `MONKEY=1` a false positive, which is the safe side for a leak check -- and the value must
    still be non-empty and a literal, so an empty slot or a `/run/secrets` path never trips it.
    """
    keyword, _, rest = instruction.partition(" ")
    if keyword.upper() not in {"ENV", "ARG"}:
        return None
    pairs: list[tuple[str, str]] = []
    tokens = rest.split()
    if keyword.upper() == "ENV" and tokens and "=" not in tokens[0] and len(tokens) >= 2:
        pairs.append((tokens[0], tokens[1]))  # legacy `ENV NAME value` form
    else:
        for token in tokens:
            name, _, value = token.partition("=")
            pairs.append((name, value))
    for name, value in pairs:
        # `API_KEY` ends in a credential word, and so does `PGPASSWORD` with no separator at all.
        letters = name.lower().replace("_", "")
        if not any(letters.endswith(word) for word in CREDENTIAL_WORDS):
            continue
        if not value or value.startswith("$") or "/" in value or "\\" in value:
            continue  # empty, build-arg/env indirection, or a path under /run/secrets
        return f"{keyword} {name} carries a literal credential"
    return None


def check_dockerfile(text: str) -> list[str]:
    """Problems with the image recipe that make it unpinnable, unsafe or secretly online."""
    problems: list[str] = []
    instructions = _instructions(text)
    if not instructions:
        return ["Dockerfile is empty"]

    stages = [index for index, line in enumerate(instructions) if line.upper().startswith("FROM ")]
    if not stages:
        problems.append("no FROM: the image has no base")
    for index in stages:
        image = instructions[index].split()[1]
        reason = _is_pinned(image)
        if reason:
            problems.append(f"base image not pinned: {reason}")

    if not any(
        line.upper().startswith("HEALTHCHECK ") and "NONE" not in line.upper()
        for line in instructions
    ):
        problems.append("no HEALTHCHECK: nothing tells the platform the server answers")

    users = [line for line in instructions if line.upper().startswith("USER ")]
    if not users:
        problems.append("no USER: the container would run as root")
    else:
        user = users[-1].split()[1]
        if user in {"root", "0", "0:0", "root:root"} or user.startswith("0:"):
            problems.append(f"final stage runs as root (USER {user})")

    if stages:
        runtime_stage = instructions[stages[-1] + 1 :]
        offenders = [
            line
            for line in runtime_stage
            if re.search(r"\bpip3? install\b|python[0-9.]* -m pip install\b", line)
        ]
        if offenders:
            problems.append(
                "the runtime stage installs packages: nothing may download at runtime "
                f"({offenders[0]})"
            )

    for line in instructions:
        leak = _credential_literal(line)
        if leak:
            problems.append(leak)

    return problems


# --- effects: wheelhouse, fresh venv, smoke ---------------------------------------------


def _run(command: Sequence[str], cwd: Path | None = None) -> None:
    """Run a command, echoing it. A non-zero exit is fatal: the check must not pass on a warning."""
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(list(command), cwd=None if cwd is None else str(cwd), check=True)


def venv_python(venv_dir: Path) -> Path:
    """The interpreter inside a virtual environment, on any OS."""
    candidate = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not candidate.is_file():
        raise SystemExit(f"no interpreter in the fresh venv at {candidate}")
    return candidate


def download_argv(wheelhouse: Path, requirements_file: Path, target_platform: str) -> list[str]:
    """The ``pip download`` command: the closure for the *image's* platform, never this machine's.

    Without ``--platform`` / ``--python-version`` / ``--abi`` pip stages wheels for the interpreter
    that runs the check (``cp312-win_amd64`` on a developer's Windows box), and the wheelhouse then
    cannot install in a ``manylinux`` image. pip allows these flags only with ``--only-binary=:all:``.
    """
    version = TARGET_PYTHON_VERSION.replace(".", "")
    return [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "--platform",
        target_platform,
        "--implementation",
        TARGET_IMPLEMENTATION,
        "--python-version",
        TARGET_PYTHON_VERSION,
        "--abi",
        f"{TARGET_IMPLEMENTATION}{version}",
        "--dest",
        str(wheelhouse),
        "--requirement",
        str(requirements_file),
    ]


def _docker_path(path: Path) -> str:
    """An absolute path in the form the Docker CLI parses inside ``--volume`` (``C:/x``, never `C:\\x`)."""
    return path.resolve().as_posix()


def container_download_argv(
    wheelhouse: Path, requirements_file: Path, target_platform: str, image: str
) -> list[str]:
    """The same ``pip download``, but run inside ``image`` so the markers resolve for the target.

    Only ``--dest`` and ``--requirement`` change: they point at the two mounts. Everything else is
    :func:`download_argv`'s contract, so a container run and a native run stage the same set.
    """
    version = TARGET_PYTHON_VERSION.replace(".", "")
    command = [
        "docker",
        "run",
        "--rm",
        "--volume",
        f"{_docker_path(wheelhouse)}:{_CONTAINER_WHEELHOUSE}",
        "--volume",
        f"{_docker_path(requirements_file)}:{_CONTAINER_REQUIREMENTS}:ro",
    ]
    for name in _FORWARDED_PIP_ENV:
        value = os.environ.get(name)
        if value:
            command.extend(["--env", f"{name}={value}"])
    command.extend(
        [
            image,
            "python",
            "-m",
            "pip",
            "download",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--only-binary=:all:",
            "--platform",
            target_platform,
            "--implementation",
            TARGET_IMPLEMENTATION,
            "--python-version",
            TARGET_PYTHON_VERSION,
            "--abi",
            f"{TARGET_IMPLEMENTATION}{version}",
            "--dest",
            _CONTAINER_WHEELHOUSE,
            "--requirement",
            _CONTAINER_REQUIREMENTS,
        ]
    )
    return command


def wheel_argv(wheelhouse: Path) -> list[str]:
    """The ``pip wheel`` command: this project's own wheel, built with no index at all.

    ``--no-index`` matters as much here as on the install: pip builds the project in an isolated
    environment and resolves the build backend there, so without it a ``--no-download`` run still
    reaches the configured package index. It works because ``hatchling`` is staged in the
    wheelhouse (see :func:`build_requirements`).
    """
    return [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-index",
        "--find-links",
        str(wheelhouse),
        "--wheel-dir",
        str(wheelhouse),
        str(REPO_ROOT),
    ]


def stage_wheelhouse(
    wheelhouse: Path,
    download: bool,
    *,
    target_platform: str,
    use_docker: bool = True,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
) -> Wheelhouse:
    """Stage the dependency closure (unless ``download`` is off) and this project's own wheel.

    When the target platform is Linux and this host is not, the download runs in a Linux container
    (:func:`container_download_argv`); the project's own wheel is built here, because it is a
    ``py3-none-any`` wheel and needs no target interpreter.
    """
    wheelhouse.mkdir(parents=True, exist_ok=True)
    if download:
        requirements_file = wheelhouse / "requirements.txt"
        requirements_file.write_text("\n".join(runtime_requirements()) + "\n", encoding="utf-8")
        # `--only-binary=:all:` is the contract: an sdist fails here instead of failing silently on
        # a machine with no compiler and no network.
        reason = staging_container_reason(target_platform)
        if reason is not None and use_docker:
            _run(
                container_download_argv(
                    wheelhouse, requirements_file, target_platform, docker_image
                )
            )
        else:
            _run(download_argv(wheelhouse, requirements_file, target_platform))
    # After the download, so `--no-index` can resolve the build backend from the wheelhouse itself.
    _run(wheel_argv(wheelhouse))
    return scan_wheelhouse(wheelhouse, target_platform=target_platform)


def container_install_argv(
    wheelhouse: Path, workdir: Path, image: str, *, network_free: bool = False
) -> list[str]:
    """Install ``mdqm[pg]`` from the wheelhouse and smoke it inside ``image``, with no index.

    A manylinux wheelhouse holds no ``win_amd64`` binaries, so installing it into a Windows venv
    dies on the first compiled wheel (``duckdb``) and would prove nothing about the image even if it
    did not. The venv lives in the container's own filesystem; only the wheelhouse and the smoke
    program are mounted, so nothing is written back to the host.

    ``network_free`` runs the container with ``--network none``: an air-gapped install needs no
    network at all, so removing it turns "we did not use an index" into a fact the kernel enforces.
    """
    target = "mdqm" if not IMAGE_EXTRAS else f"mdqm[{','.join(IMAGE_EXTRAS)}]"
    script = "\n".join(
        [
            "set -eu",
            "python -m venv /venv",
            "/venv/bin/python -m pip install --disable-pip-version-check --no-index "
            f'--find-links {_CONTAINER_WHEELHOUSE} --only-binary=:all: "{target}"',
            "/venv/bin/python /work/smoke.py",
            "/venv/bin/mdqm --version",
        ]
    )
    command = ["docker", "run", "--rm"]
    if network_free:
        command.extend(["--network", "none"])
    command.extend(
        [
            "--volume",
            f"{_docker_path(wheelhouse)}:{_CONTAINER_WHEELHOUSE}:ro",
            "--volume",
            f"{_docker_path(workdir / 'smoke.py')}:/work/smoke.py:ro",
            image,
            "sh",
            "-c",
            script,
        ]
    )
    return command


def fresh_install(
    wheelhouse: Path,
    workdir: Path,
    *,
    in_container: bool = False,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    network_free: bool = False,
) -> Path | None:
    """Install ``mdqm[pg]`` into a brand-new venv from the wheelhouse, with no index at all.

    When the wheelhouse is for a Linux image and this host is not Linux, the venv is built and
    installed *inside* the container (:func:`container_install_argv`), which also runs the smoke
    test: there is no host interpreter that could import those wheels. ``None`` means "installed and
    smoke-tested in the container, nothing left to do here".
    """
    if in_container:
        (workdir / "smoke.py").write_text(SMOKE, encoding="utf-8")
        _run(container_install_argv(wheelhouse, workdir, docker_image, network_free=network_free))
        return None
    venv_dir = workdir / "venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
    interpreter = venv_python(venv_dir)
    target = "mdqm" if not IMAGE_EXTRAS else f"mdqm[{','.join(IMAGE_EXTRAS)}]"
    _run(
        [
            str(interpreter),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            "--only-binary=:all:",
            target,
        ]
    )
    return interpreter


def smoke(interpreter: Path | None) -> None:
    """Prove the installed wheel imports, its console script runs, and its migrations ship.

    ``None`` means the container install already ran this, on the interpreter those wheels target.
    """
    if interpreter is None:
        return
    _run([str(interpreter), "-c", SMOKE])
    console = interpreter.parent / ("mdqm.exe" if sys.platform == "win32" else "mdqm")
    _run([str(console), "--version"])


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checks. Returns the exit code; 0 means the offline install is sound."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "check the offline install").splitlines()[0]
    )
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        default=REPO_ROOT / "wheelhouse",
        help="where the staged wheels live (default: <repo>/wheelhouse, also the image's build input)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="air-gapped run: use the wheels already staged, never touch a package index",
    )
    parser.add_argument(
        "--target-platform",
        default=DEFAULT_TARGET_PLATFORM,
        help=(
            "platform the staged wheels must be built for, i.e. the image's, not this machine's "
            f"(default: {DEFAULT_TARGET_PLATFORM}); pass win_amd64 only to stage for Windows"
        ),
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the temporary venv for inspection"
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help=(
            "never use a staging container: staging a Linux wheelhouse from a non-Linux host then "
            "fails loudly instead of running pip with the wrong environment markers"
        ),
    )
    parser.add_argument(
        "--docker-image",
        default=DEFAULT_DOCKER_IMAGE,
        help=(
            "image that runs pip for a Linux target when this host is not Linux "
            f"(default: {DEFAULT_DOCKER_IMAGE}); point it at the internal mirror in an air-gapped site"
        ),
    )
    options = parser.parse_args(argv)

    problems: list[str] = []
    requirements = runtime_requirements()
    build = build_requirements()
    print(
        f"runtime requirements ({', '.join(IMAGE_EXTRAS)} extras): "
        f"{len(requirements) - len(build)} + {len(build)} build backend",
        flush=True,
    )
    for requirement in requirements:
        print(f"  {requirement}", flush=True)
    abi = f"{TARGET_IMPLEMENTATION}{TARGET_PYTHON_VERSION.replace('.', '')}"
    print(f"target platform: {options.target_platform} ({abi})", flush=True)

    # How the closure gets staged and installed: here, or in a container that has the target's
    # environment markers *and* the target's binaries. `--no-download` does not change that: a
    # manylinux wheelhouse still cannot install into a Windows venv.
    stage_in_container = False
    reason = staging_container_reason(options.target_platform)
    if reason is not None:
        if options.no_docker:
            problems.append(f"--no-docker: {reason}")
        elif shutil.which("docker") is None:
            problems.append(f"{reason}; no 'docker' on PATH and --no-docker was not given")
        else:
            stage_in_container = True
            print(f"container: {options.docker_image} ({reason})", flush=True)

    dockerfile_problems = check_dockerfile(DOCKERFILE.read_text(encoding="utf-8"))
    problems.extend(dockerfile_problems)
    print(
        f"Dockerfile: {'ok' if not dockerfile_problems else 'FAILED'}",
        flush=True,
    )

    wheelhouse = (
        options.wheelhouse if options.wheelhouse.is_absolute() else REPO_ROOT / options.wheelhouse
    )
    # A wheelhouse without the closure's own dependencies is not a wheelhouse, whatever the tags say.
    # A pip failure is this script's failure: report it as a problem, not as a traceback. The case
    # that matters is `--no-download` against an incomplete wheelhouse (a Windows-staged one, or a
    # missing build backend): it must fail here, loudly, and never print "check passed".
    try:
        staged = stage_wheelhouse(
            wheelhouse,
            download=not options.no_download,
            target_platform=options.target_platform,
            use_docker=stage_in_container,
            docker_image=options.docker_image,
        )
    except subprocess.CalledProcessError as exc:
        print("\nFAILED:", flush=True)
        for problem in problems:
            print(f"  - {problem}", flush=True)
        print(f"  - {' '.join(str(part) for part in exc.cmd)} exited {exc.returncode}", flush=True)
        return 1
    missing = staged.missing(requirements)
    if missing:
        problems.append(f"no wheel staged for: {', '.join(missing)}")
    if staged.incompatible:
        problems.append(
            f"wheels staged for another platform than {options.target_platform} "
            "(they cannot install in the image): " + ", ".join(staged.incompatible)
        )
    if staged.source_archives:
        problems.append(
            "source archives in the wheelhouse (they need a compiler the target does not have): "
            + ", ".join(staged.source_archives)
        )
    print(
        f"wheelhouse {wheelhouse}: {len(staged.wheels)} installable wheel(s), "
        f"{len(staged.incompatible)} wrong-platform, "
        f"{len(staged.source_archives)} source archive(s)",
        flush=True,
    )
    gaps = closure_gaps(wheelhouse, staged)
    problems.extend(gaps)
    if gaps:
        print(f"closure: {len(gaps)} hole(s) in the staged set", flush=True)
    else:
        print("closure: every unmarked requirement is staged", flush=True)

    workdir = Path(tempfile.mkdtemp(prefix="mdqm-offline-"))
    try:
        interpreter = fresh_install(
            wheelhouse,
            workdir,
            in_container=stage_in_container,
            docker_image=options.docker_image,
            network_free=options.no_download,
        )
        smoke(interpreter)
    except subprocess.CalledProcessError as exc:
        print("\nFAILED:", flush=True)
        for problem in problems:
            print(f"  - {problem}", flush=True)
        print(f"  - {' '.join(str(part) for part in exc.cmd)} exited {exc.returncode}", flush=True)
        return 1
    finally:
        if options.keep:
            print(f"kept {workdir}", flush=True)
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    if problems:
        print("\nFAILED:", flush=True)
        for problem in problems:
            print(f"  - {problem}", flush=True)
        return 1
    print("\noffline install check passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
