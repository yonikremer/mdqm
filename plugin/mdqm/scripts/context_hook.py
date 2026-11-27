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
  therefore **fail-open**: any error prints nothing on stdout and exits 0, so a stopped server, a
  missing token or a slow network costs the user nothing but the context. It is **not silent** any
  more: a failure that was ours to fix writes one line to stderr, which reaches the user without
  polluting the model's context (AGENTS.md principle 10: everything reported, nothing silent).

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

import http.client
import json
import os
import ssl
import sys
import urllib.parse
from typing import Any

#: Wall-clock budget for the round trip *once connected*: bearer check, full-text search, pack
#: render. The old value (0.45 s) covered the whole call and was measured on loopback; behind a
#: Route a prompt pays DNS + TCP + TLS + the round trip, so on every prompt off a loopback the
#: context was dropped while looking like "the server has nothing". ``hooks/hooks.json`` sets a 3 s
#: hard timeout, which stays the bound on this one.
TIMEOUT_SECONDS = 2.0

#: Budget for getting a connection at all. Kept short on purpose: a stopped server, or a port whose
#: SYN is dropped and never refused, must cost the prompt nothing. Sharing one long budget for both
#: phases is what makes "server down" feel like a hang.
CONNECT_TIMEOUT_SECONDS = 0.45

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
    """The Context Pack for ``prompt``, or ``None`` on any failure at all.

    Two phases with two budgets, which is the whole point of not using ``urlopen``: connecting gets
    :data:`CONNECT_TIMEOUT_SECONDS` so a dead server is free, and the round trip after that gets
    :data:`TIMEOUT_SECONDS` so a slow-but-alive one still injects. ``urlopen`` applies one timeout
    to every socket operation, so it can only offer one of the two.
    """
    body = json.dumps({"prompt": prompt}).encode("utf-8")
    # The scheme is already narrowed to http/https by `_base_url`, so no `file:` URL can arrive.
    parts = urllib.parse.urlsplit(url + "/context")
    if parts.scheme == "https":
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            parts.hostname or "",
            parts.port,
            timeout=CONNECT_TIMEOUT_SECONDS,
            context=ssl.create_default_context(),  # the system trust store, no proxy of our own
        )
    else:
        connection = http.client.HTTPConnection(
            parts.hostname or "", parts.port, timeout=CONNECT_TIMEOUT_SECONDS
        )
    path = parts.path or "/context"
    if parts.query:
        path = path + "?" + parts.query
    try:
        connection.connect()  # TCP, plus the TLS handshake on https: short budget
        if connection.sock is not None:
            connection.sock.settimeout(TIMEOUT_SECONDS)  # the round trip: longer budget
        connection.request(
            "POST",
            path,
            body=body,
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        response = connection.getresponse()
        if response.status != 200:
            return None
        raw = response.read(MAX_RESPONSE_BYTES)
    finally:
        connection.close()
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


def _warn(reason: str) -> None:
    """One line on stderr: it reaches the user, and it never becomes model context.

    stdout is the injection channel, so a diagnostic must not go there. ASCII only: a Windows
    console with a legacy code page would raise on anything else, inside the very handler that is
    supposed to fail open.
    """
    sys.stderr.write("mdqm: no context injected: " + reason + "\n")
    sys.stderr.flush()


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
    """Print the injected context, or one diagnostic line. Always exit 0: never block a prompt."""
    url = ""
    try:
        prompt = _prompt()
        if prompt is None:
            _warn("the UserPromptSubmit payload carried no usable prompt")
            return 0
        raw_url = _first_env(URL_VARS)
        token = _first_env(TOKEN_VARS)
        if raw_url is None or token is None:
            # Not configured is not a failure: the plugin is simply not set up on this desktop.
            return 0
        url = _base_url(raw_url) or ""
        if not url:
            _warn("MDQM_URL is not an http(s) URL with a host")
            return 0
        pack = _fetch(url, token, prompt)
        if pack is not None:
            _emit(pack)
            return 0
        _warn(url + "/context returned no usable additionalContext")
    except OSError as exc:
        # A refused, dropped or timed-out connection: the server is down or behind a broken proxy.
        _warn("could not reach " + (url or "the server") + ": " + type(exc).__name__)
        return 0
    except http.client.HTTPException as exc:
        _warn("bad HTTP response from " + (url or "the server") + ": " + type(exc).__name__)
        return 0
    except (ValueError, KeyError) as exc:
        _warn("unexpected response from the server: " + type(exc).__name__)
        return 0
    except Exception as exc:  # noqa: BLE001 - a hook must never block a prompt, whatever went wrong
        _warn("unexpected failure: " + type(exc).__name__)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
