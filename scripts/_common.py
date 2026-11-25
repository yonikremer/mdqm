"""Shared helpers for the task scripts. Standard library only, so they run before anything is installed."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

HERE = Path(__file__).resolve().parent
REPO_FILES_ROOT = HERE.parent


def git(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    res = subprocess.run(
        ["git", *args], cwd=cwd or REPO_FILES_ROOT, capture_output=True, text=True, check=False
    )
    if check and res.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout.strip()


def common_dir() -> Path:
    """The shared .git directory (same for every worktree). Falls back to a local dir without git."""
    try:
        out = git("rev-parse", "--path-format=absolute", "--git-common-dir")
        return Path(out)
    except SystemExit:
        d = REPO_FILES_ROOT / ".git-fallback"
        d.mkdir(exist_ok=True)
        return d


def main_root() -> Path:
    """Root of the main worktree (the one on `main`). Progress files are read from there."""
    cd = common_dir()
    return cd.parent if cd.name == ".git" else REPO_FILES_ROOT


def load_tasks() -> dict[str, dict]:
    path = REPO_FILES_ROOT / "tasks.json"
    if not path.exists():
        path = main_root() / "tasks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {t["id"]: t for t in data["tasks"]}


def done_ids() -> set[str]:
    pdir = main_root() / "progress"
    out: set[str] = set()
    if not pdir.exists():
        return out
    for f in pdir.glob("*.md"):
        head = f.read_text(encoding="utf-8").splitlines()[:12]
        if any(line.strip().lower() == "status: done" for line in head):
            out.add(f.stem)
    return out


def die(msg: str, code: int = 1) -> NoReturn:
    print(msg, file=sys.stderr)
    raise SystemExit(code)
