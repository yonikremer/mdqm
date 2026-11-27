"""Fail when a diff looks like it carries a credential.

The product's premise includes "secrets never in code, YAML, logs, fixtures or error messages"
(AGENTS.md principle 11). ruff's `S105`/`S106`/`S107` catch literals assigned to obviously named
variables; this hook catches the shapes that matter in a *diff* and that a linter cannot reason about,
because they are strings in data files, docs or tests:

* a PEM private key block, anywhere in any file;
* a known token prefix (AWS, GitHub, Slack, API-key styles);
* a `Bearer <long token>` or `password: <long literal>` in a text file.

A line carrying `pragma: allowlist secret` is skipped: that is the `detect-secrets` convention, and the
only legitimate use here is a fixture that exists to prove redaction works. The marker is visible in
review, which is the point — an allowlist file would hide it.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from pathlib import Path

#: (name, pattern). Every pattern needs a long, unambiguous literal: a short or fuzzy one would fire on
#: ordinary text and the hook would end up disabled, which is worse than not having it.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("API key with sk- prefix", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{24,}")),
    # A shell or template reference ("$VAR", "${VAR}") is not a secret: the value is not in the file.
    # Only a literal of some length counts.
    (
        "assigned password",
        # A quoted literal only: `password = secrets.get(...)` is code reading a
        # secret, not a secret committed to the file, and flagging it is how a hook
        # gets switched off. Shell/template references ("$VAR") are excluded too.
        re.compile(r"(?i)password\s*[:=]\s*['\"](?!\$)[^'\"$]{12,}['\"]"),
    ),
)

ALLOWLIST_MARKER = "pragma: allowlist secret"
"""A line carrying this marker is a deliberate fixture that exists to be redacted."""

#: Only text files are scanned; a wheel or a PNG cannot carry a match we care about.
TEXT_SUFFIXES = frozenset(
    {
        ".py", ".toml", ".yaml", ".yml", ".json", ".md", ".txt", ".cfg", ".ini", ".env", ".sql",
        ".sh", ".ps1", ".bat", ".ndjson", ".example", ".ts", ".js", ".css", ".html",
    }
)

MAX_BYTES = 2 * 1024 * 1024
"""Skip anything larger: a generated data file is not where a hand-written secret hides."""


def findings(path: Path) -> Iterator[str]:
    """Yield one line per suspected credential in ``path``."""
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".env":
        return
    try:
        if path.stat().st_size > MAX_BYTES:
            return
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    lines = text.splitlines()
    for number, line in enumerate(lines, start=1):
        # The marker may sit on the line or on the line above it: a multi-line fixture (a PEM block
        # split across string literals) cannot carry a trailing comment on every piece.
        previous = lines[number - 2] if number >= 2 else ""
        if ALLOWLIST_MARKER in line or ALLOWLIST_MARKER in previous:
            continue
        for name, pattern in PATTERNS:
            if pattern.search(line):
                # Never echo the match itself: the point is to keep the secret out of logs.
                yield f"{path}:{number}: looks like a {name}"


def main(argv: list[str] | None = None) -> int:
    """Scan every file given on the command line; returns 1 when anything looks like a secret."""
    paths = [Path(name) for name in (argv if argv is not None else sys.argv[1:])]
    hits = [hit for path in paths if path.is_file() for hit in findings(path)]
    for hit in hits:
        print(hit, file=sys.stderr)
    if hits:
        print(
            f"{len(hits)} possible secret(s); remove them, or mark a deliberate fixture with "
            f"'{ALLOWLIST_MARKER}', and rotate anything real",
            file=sys.stderr,
        )
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
