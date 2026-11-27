"""Merge a finished task branch into main, safely, one at a time.

    python scripts/integrate.py T-ID

Preconditions: the task was done on branch `task/<ID>` (ideally in its own git worktree), committed,
and includes progress/<ID>.md with `status: done`. This script:
  1. takes a global lock (only one integration at a time),
  2. rebases task/<ID> onto main (aborts and reports on conflicts),
  3. runs the fast checks again in the task worktree (unit tests + ruff),
  4. fast-forwards main to the branch, releases the claim.
If anything fails, main is untouched and the exit code is non-zero.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from _common import common_dir, die, git, main_root

CHECKS = [
    [sys.executable, "-m", "pytest", "tests/unit", "-q", "-x"],
    [sys.executable, "-m", "ruff", "check", "."],
]


def worktree_for(branch: str):
    out = git("worktree", "list", "--porcelain")
    cur = None
    for line in out.splitlines():
        if line.startswith("worktree "):
            cur = line.split(" ", 1)[1]
        elif line == f"branch refs/heads/{branch}" and cur:
            return cur
    return None


def acquire_lock(timeout=900):
    lock = common_dir() / "mdqm-integrate.lock"
    start = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return lock  # noqa: TRY300 - the lock is the success path of the try
        except FileExistsError:
            if time.time() - lock.stat().st_mtime > 1800:  # stale lock
                lock.unlink(missing_ok=True)
                continue
            if time.time() - start > timeout:
                die("could not get integration lock", 4)
            time.sleep(3)


def main():
    if len(sys.argv) != 2:
        die(__doc__ or "", 1)
    tid = sys.argv[1]
    branch = f"task/{tid}"
    root = main_root()
    if git("rev-parse", "--verify", branch, cwd=root, check=False) == "":
        die(f"branch {branch} not found", 1)
    wt = worktree_for(branch) or str(root)
    lock = acquire_lock()
    try:
        if git("rev-parse", "--abbrev-ref", "HEAD", cwd=root) != "main":
            die("main worktree must be on branch main", 5)
        if git("status", "--porcelain", cwd=root):
            die("main worktree has uncommitted changes", 5)
        r = subprocess.run(["git", "rebase", "main"], cwd=wt, capture_output=True, text=True)
        if r.returncode != 0:
            subprocess.run(["git", "rebase", "--abort"], cwd=wt, capture_output=True)
            die(
                f"rebase of {branch} onto main failed (resolve conflicts in the worktree, then retry):\n{r.stdout}{r.stderr}",
                2,
            )
        progress = os.path.join(wt, "progress", f"{tid}.md")
        if not os.path.exists(progress):
            die(f"progress/{tid}.md missing on {branch}", 6)
        for cmd in CHECKS:
            r = subprocess.run(cmd, cwd=wt)
            if r.returncode != 0:
                die(f"check failed after rebase: {' '.join(cmd)}", 3)
        git("merge", "--ff-only", branch, cwd=root)
        claim = common_dir() / "mdqm-claims" / f"{tid}.json"
        claim.unlink(missing_ok=True)
        print(f"integrated {branch} into main")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
