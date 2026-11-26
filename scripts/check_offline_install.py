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

Exit code 0 means every check passed. Anything else means the deployment would fail on the
air-gapped network, and the printed problems say what to fix.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import venv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

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


def runtime_requirements(
    pyproject: Path = PYPROJECT, extras: Sequence[str] = IMAGE_EXTRAS
) -> list[str]:
    """The direct requirements the image installs: core dependencies plus ``extras``.

    Reads ``pyproject.toml`` rather than the installed metadata, because the installed metadata of
    an editable checkout goes stale the moment somebody adds a dependency.
    """
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    requirements: list[str] = list(project.get("dependencies", []))
    declared = project.get("optional-dependencies", {})
    for extra in extras:
        if extra not in declared:
            raise ValueError(f"pyproject.toml declares no extra {extra!r}")
        requirements.extend(declared[extra])
    return requirements


def is_sdist(filename: str) -> bool:
    """True for a source archive: something that needs a compiler, which the target cannot do."""
    return filename.endswith(SDIST_SUFFIXES)


@dataclass(frozen=True)
class Wheelhouse:
    """What a directory of staged artefacts contains, keyed by canonical distribution name."""

    wheels: dict[str, str]
    source_archives: tuple[str, ...] = ()

    def missing(self, requirements: Iterable[str]) -> list[str]:
        """The requirements with no wheel staged. Empty means the closure is complete."""
        absent = [
            requirement
            for requirement in requirements
            if canonical(requirement_name(requirement)) not in self.wheels
        ]
        return sorted(absent)


def scan_wheelhouse(directory: Path) -> Wheelhouse:
    """Classify every file in ``directory`` into wheels and source archives."""
    wheels: dict[str, str] = {}
    sdists: list[str] = []
    if not directory.is_dir():
        return Wheelhouse(wheels={}, source_archives=())
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.name.endswith(".whl"):
            # {name}-{version}(-{build})?-{python}-{abi}-{platform}.whl; the name never has '-'.
            wheels[canonical(path.name.split("-")[0])] = path.name
        elif is_sdist(path.name):
            sdists.append(path.name)
    return Wheelhouse(wheels=wheels, source_archives=tuple(sdists))


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


def stage_wheelhouse(wheelhouse: Path, download: bool) -> Wheelhouse:
    """Build this project's wheel and, unless ``download`` is off, the whole dependency closure."""
    wheelhouse.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheelhouse),
            str(REPO_ROOT),
        ]
    )
    if download:
        requirements_file = wheelhouse / "requirements.txt"
        requirements_file.write_text("\n".join(runtime_requirements()) + "\n", encoding="utf-8")
        # `--only-binary=:all:` is the contract: an sdist fails here instead of failing silently on
        # a machine with no compiler and no network.
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--only-binary=:all:",
                "--dest",
                str(wheelhouse),
                "--requirement",
                str(requirements_file),
            ]
        )
    return scan_wheelhouse(wheelhouse)


def fresh_install(wheelhouse: Path, workdir: Path) -> Path:
    """Install ``mdqm[pg]`` into a brand-new venv from the wheelhouse, with no index at all."""
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


def smoke(interpreter: Path) -> None:
    """Prove the installed wheel imports, its console script runs, and its migrations ship."""
    _run([str(interpreter), "-c", SMOKE])
    console = interpreter.parent / ("mdqm.exe" if sys.platform == "win32" else "mdqm")
    _run([str(console), "--version"])


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checks. Returns the exit code; 0 means the offline install is sound."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
        "--keep", action="store_true", help="keep the temporary venv for inspection"
    )
    options = parser.parse_args(argv)

    problems: list[str] = []
    requirements = runtime_requirements()
    print(
        f"runtime requirements ({', '.join(IMAGE_EXTRAS)} extras): {len(requirements)}", flush=True
    )
    for requirement in requirements:
        print(f"  {requirement}", flush=True)

    dockerfile_problems = check_dockerfile(DOCKERFILE.read_text(encoding="utf-8"))
    problems.extend(dockerfile_problems)
    print(
        f"Dockerfile: {'ok' if not dockerfile_problems else 'FAILED'}",
        flush=True,
    )

    wheelhouse = (
        options.wheelhouse if options.wheelhouse.is_absolute() else REPO_ROOT / options.wheelhouse
    )
    staged = stage_wheelhouse(wheelhouse, download=not options.no_download)
    missing = staged.missing(requirements)
    if missing:
        problems.append(f"no wheel staged for: {', '.join(missing)}")
    if staged.source_archives:
        problems.append(
            "source archives in the wheelhouse (they need a compiler the target does not have): "
            + ", ".join(staged.source_archives)
        )
    print(
        f"wheelhouse {wheelhouse}: {len(staged.wheels)} wheel(s), "
        f"{len(staged.source_archives)} source archive(s)",
        flush=True,
    )

    workdir = Path(tempfile.mkdtemp(prefix="mdqm-offline-"))
    try:
        interpreter = fresh_install(wheelhouse, workdir)
        smoke(interpreter)
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
