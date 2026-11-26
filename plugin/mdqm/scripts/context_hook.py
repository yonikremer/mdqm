#!/usr/bin/env python3
"""``UserPromptSubmit`` hook: inject the mdqm Context Pack into the agent's prompt.

Claude Code runs this before the model sees a prompt. It posts the prompt to the mdqm server's
``POST /context`` and prints the pack back as ``hookSpecificOutput.additionalContext``, so a weak
agent starts the turn already knowing the measures, value maps, approved joins, tables and example
specs that match the question (``docs/DESIGN.md`` section 9).

Verified against the Claude Code docs, not memory (AGENTS.md section 9):

* ``UserPromptSubmit`` input carries ``prompt`` on stdin as JSON.
* Exit 0 plus ``{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
  "additionalContext": "..."}}`` is the documented way to add context; plain stdout would also be
  injected, but the JSON form names the event and is what a reviewer can parse.
* The event has no matcher and a lowered default timeout, because it blocks the prompt. The hook is
  therefore **fail-open and silent**: any error prints nothing and exits 0, so a stopped server, a
  missing token or a slow network costs the user nothing but the context.

Only the standard library, and Python 3.9 syntax, because this runs on user desktops that are not
this repository's venv. No shell is involved: ``hooks/hooks.json`` spawns the interpreter directly
in exec form, so Windows needs no bash and no quoting.

Environment (first one set wins):

* ``CLAUDE_PLUGIN_OPTION_MDQM_URL`` / ``MDQM_URL`` -- the server's base URL, e.g.
  ``https://mdqm.internal``. A trailing ``/`` or a pasted ``.../mcp`` is accepted.
* ``CLAUDE_PLUGIN_OPTION_MDQM_TOKEN`` / ``MDQM_TOKEN`` -- the personal bearer token.

Nothing is logged and no token is ever printed: a hook's stdout becomes model context.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

#: Wall-clock budget for the whole call. ``UserPromptSubmit`` blocks the prompt, so the budget is
#: the product requirement: with a dead server this hook must still finish well under a second, and
#: a connect that is silently dropped (not refused) waits out this timeout. Measured on Windows:
#: ~0.14s of interpreter start plus this value, against ~0.12s when nothing is attempted.
TIMEOUT_SECONDS = 0.45

#: ``/context`` refuses a prompt longer than the server's cap; reading the body is not the point.
MAX_RESPONSE_BYTES = 4 * 1024 * 1024

#: Env var names, in the order they are tried.
URL_VARS = ("CLAUDE_PLUGIN_OPTION_MDQM_URL", "MDQM_URL")
TOKEN_VARS = ("CLAUDE_PLUGIN_OPTION_MDQM_TOKEN", "MDQM_TOKEN")


def _first_env(names: tuple[str, ...]) -> str | None:
    """The first non-empty value among ``names``, trimmed."""
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _base_url(raw: str) -> str | None:
    """The server base URL, or ``None`` when it is not an http(s) URL.

    A user who pastes the MCP endpoint (``.../mcp``) means the server, so that suffix is dropped
    instead of turning a one-character paste into a 404.
    """
    candidate = raw.strip().rstrip("/")
    if candidate.endswith("/mcp"):
        candidate = candidate[: -len("/mcp")]
    parsed = urllib.parse.urlsplit(candidate)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    return candidate


def _prompt() -> str | None:
    """The submitted prompt from stdin, or ``None`` when the hook payload is not what we expect."""
    raw = sys.stdin.buffer.read()
    try:
        payload: Any = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    prompt = payload.get("prompt")
    return prompt if isinstance(prompt, str) and prompt.strip() else None


def _fetch(url: str, token: str, prompt: str) -> str | None:
    """The Context Pack for ``prompt``, or ``None`` on any failure at all."""
    body = json.dumps({"prompt": prompt}).encode("utf-8")
    # The scheme is already narrowed to http/https by `_base_url`, so no `file:` URL can arrive.
    request = urllib.request.Request(  # noqa: S310 - scheme validated by _base_url
        url + "/context",
        data=body,
        method="POST",
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    # Same reason as above: http/https only.
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
        if getattr(response, "status", 200) != 200:
            return None
        raw = response.read(MAX_RESPONSE_BYTES)
    try:
        answer: Any = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return None
    if not isinstance(answer, dict):
        return None
    text = answer.get("additionalContext")
    if not isinstance(text, str) or not text.strip():
        return None
    return text


def _emit(text: str) -> None:
    """Write the documented envelope as UTF-8 bytes.

    Bytes, not ``print``: for accents in the Hebrew pack to survive it must not go through the
    console's legacy code page on Windows.
    """
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": text,
        }
    }
    sys.stdout.buffer.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    sys.stdout.buffer.flush()


def main() -> int:
    """Print the injected context, or nothing. Always exit 0: this hook never blocks a prompt."""
    try:
        prompt = _prompt()
        if prompt is None:
            return 0
        raw_url = _first_env(URL_VARS)
        token = _first_env(TOKEN_VARS)
        if raw_url is None or token is None:
            return 0
        url = _base_url(raw_url)
        if url is None:
            return 0
        pack = _fetch(url, token, prompt)
        if pack is not None:
            _emit(pack)
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        # Fail open and silent. The prompt proceeds without injected context, which is exactly the
        # state before this plugin was installed, so there is nothing to warn the user about.
        return 0
    except Exception:  # noqa: BLE001 - a hook must never block a prompt, whatever went wrong
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
