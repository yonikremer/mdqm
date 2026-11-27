"""M07: the Claude Code plugin's ``UserPromptSubmit`` hook, and the plugin files around it.

The hook is a desktop process, so it is tested as one: spawned with ``sys.executable``, fed a JSON
payload on stdin, and pointed at a **stub** HTTP server that records the request it receives. That
proves the three promises a hook must keep, in the order they matter:

* **It works when the server does.** A relevant prompt produces exactly the JSON shape Claude Code
  documents for ``UserPromptSubmit`` -- ``hookSpecificOutput.additionalContext`` -- and nothing else
  on stdout. The body the hook sends is validated against the server's own
  :class:`~mdqm.server.context_endpoint.ContextRequest`, so the stub cannot drift from the real
  contract (M06).
* **It fails open.** Server down, ``401``, garbage response, garbage stdin, missing token: empty
  output, exit ``0``, and (for the down case, which the acceptance criterion names) under a second.
  A hook that blocks a prompt is worse than a hook that adds nothing.
* **It leaks nothing and hard-codes nothing.** No token or host or port literal in the shipped
  plugin directory: both come from ``MDQM_URL`` / ``MDQM_TOKEN`` at the user's machine. The hook is
  spawned in exec form, so Windows needs no bash, and the script's own syntax is parsed at the
  Python 3.9 level it promises to support.

Formats are the current Claude Code ones, read from code.claude.com on the day this was written and
recorded in ``progress/M07.md``; the assertions here pin the parts that would silently break.

The loop opt-in below is for the stub server's loopback socket only. Nothing reaches a data source,
the network or the internet, and the real server is not started.
"""

from __future__ import annotations

import ast
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest
import pytest_socket
from ruamel.yaml import YAML

from mdqm.server.context_endpoint import ContextRequest
from mdqm.server.tools import TOOL_NAMES

#: ``plugin/mdqm`` -- the directory Claude Code loads, and the one that must hold no secret.
PLUGIN = Path(__file__).resolve().parents[2] / "plugin" / "mdqm"

#: The hook, run by the interpreter that runs the tests (the plugin's own is chosen by the user).
HOOK = PLUGIN / "scripts" / "context_hook.py"

#: A token value used only here. It never appears in a committed plugin file.
TOKEN = "tok-plugin-hook-test"

#: The loop opt-in is for the stub server binding ``127.0.0.1``; see the module docstring.
pytestmark = pytest.mark.enable_socket


@pytest.fixture(autouse=True)
def loopback_only() -> Iterator[None]:
    """``127.0.0.1`` and nothing else: the stub server, never a source or the internet."""
    pytest_socket.socket_allow_hosts(["127.0.0.1"])
    yield
    pytest_socket.socket_allow_hosts(["127.0.0.1"])


class Stub:
    """A recording HTTP server: every request is kept so a test can assert on the wire bytes."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.status = 200
        self.body = json.dumps({"additionalContext": "MEASURE credit_limit_r5000 = avg(limit)"})
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                stub.requests.append(
                    {
                        "path": self.path,
                        "headers": {key.lower(): value for key, value in self.headers.items()},
                        "body": self.rfile.read(length),
                    }
                )
                payload = stub.body.encode("utf-8")
                self.send_response(stub.status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the base class's names
                """Silence the default stderr logging: a test run should not look like a server."""

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        # `server_address` is typed loosely; a bytes host would f-string to "b'127.0.0.1'" and every
        # URL assertion below would fail for a reason that has nothing to do with the hook.
        assert isinstance(host, str) and isinstance(port, int)
        return f"http://{host}:{port}"

    def __enter__(self) -> Stub:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _closed_port() -> str:
    """A loopback URL nothing is listening on: the "server is down" case, without a race."""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return f"http://127.0.0.1:{port}"


def run_hook(
    stdin: bytes,
    *,
    url: str | None,
    token: str | None = TOKEN,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Spawn the hook exactly as ``hooks/hooks.json`` does: exec form, no shell."""
    env = {
        key: value
        for key, value in {
            "MDQM_URL": url,
            "MDQM_TOKEN": token,
            "SYSTEMROOT": _system_root(),
        }.items()
        if value is not None
    }
    env.update(extra_env or {})
    # Fixed argv and no shell: the interpreter and the script path are ours, not user input.
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(HOOK)],
        input=stdin,
        capture_output=True,
        env=env,
        timeout=30,
        check=False,
    )


def _system_root() -> str | None:
    """Windows needs ``SYSTEMROOT`` to start a process at all; other platforms do not have it."""
    return os.environ.get("SYSTEMROOT")


def payload(prompt: str) -> bytes:
    """The ``UserPromptSubmit`` stdin payload, as the docs describe it."""
    return json.dumps(
        {"hook_event_name": "UserPromptSubmit", "session_id": "s1", "cwd": ".", "prompt": prompt}
    ).encode("utf-8")


def injected(stdout: bytes) -> str:
    """The ``additionalContext`` the hook asked Claude Code to inject."""
    parsed = json.loads(stdout.decode("utf-8"))
    assert set(parsed) == {"hookSpecificOutput"}
    output = parsed["hookSpecificOutput"]
    assert output["hookEventName"] == "UserPromptSubmit"
    context = output["additionalContext"]
    # A non-string here would still be JSON-valid and Claude Code would inject nothing useful.
    assert isinstance(context, str)
    return context


# --------------------------------------------------------------------------------------------------
# It works when the server does
# --------------------------------------------------------------------------------------------------


def test_relevant_prompt_becomes_additional_context() -> None:
    with Stub() as stub:
        result = run_hook(payload("average credit limit by status"), url=stub.url)

    assert result.returncode == 0
    assert result.stderr == b""
    assert injected(result.stdout) == "MEASURE credit_limit_r5000 = avg(limit)"


def test_request_matches_the_servers_own_context_contract() -> None:
    """The stub cannot drift: the body is parsed by the model ``POST /context`` uses."""
    with Stub() as stub:
        run_hook(payload("how many active customers"), url=stub.url)

    (request,) = stub.requests
    assert request["path"] == "/context"
    assert request["headers"]["authorization"] == f"Bearer {TOKEN}"
    assert request["headers"]["content-type"] == "application/json"
    # Validates, or raises: the one field the endpoint requires, and exactly one.
    parsed = ContextRequest.model_validate_json(request["body"])
    assert parsed.prompt == "how many active customers"


@pytest.mark.parametrize(
    "url_form",
    ["{url}", "{url}/", "{url}/mcp"],
    ids=["bare", "trailing-slash", "pasted-mcp-endpoint"],
)
def test_url_forms_all_reach_the_context_endpoint(url_form: str) -> None:
    """A pasted ``.../mcp`` or a trailing slash is a one-character mistake, not a 404."""
    with Stub() as stub:
        run_hook(payload("credit limit"), url=url_form.format(url=stub.url))

    assert [request["path"] for request in stub.requests] == ["/context"]


def test_hebrew_pack_survives_as_utf8() -> None:
    """The pack is Hebrew and English; the console code page must not eat it."""
    pack = "מידה: credit_limit_r5000 — ערך סטטוס: 'פעיל'"
    with Stub() as stub:
        stub.body = json.dumps({"additionalContext": pack}, ensure_ascii=False)
        result = run_hook(payload("כמה לקוחות פעילים"), url=stub.url)

    assert injected(result.stdout) == pack


def test_plugin_option_env_vars_win_over_the_plain_names() -> None:
    """``CLAUDE_PLUGIN_OPTION_*`` is what a plugin hook is documented to receive."""
    with Stub() as stub:
        result = run_hook(
            payload("credit limit"),
            url=None,
            token=None,
            extra_env={
                "CLAUDE_PLUGIN_OPTION_MDQM_URL": stub.url,
                "CLAUDE_PLUGIN_OPTION_MDQM_TOKEN": TOKEN,
            },
        )

    assert injected(result.stdout) == "MEASURE credit_limit_r5000 = avg(limit)"
    assert stub.requests[0]["headers"]["authorization"] == f"Bearer {TOKEN}"


# --------------------------------------------------------------------------------------------------
# It fails open, silently, and fast
# --------------------------------------------------------------------------------------------------


def test_server_down_prints_nothing_exits_zero_under_a_second() -> None:
    """The acceptance criterion: a stopped server costs the user no time and no error.

    Two bounds on purpose. The absolute one is the product promise (well under a second, because the
    hook blocks the prompt). The relative one is what this test is really about: a dead server must
    not be *waited for*, so the cost of reaching an unreachable port stays far below the hook's own
    timeout. That second assertion isolates the behaviour from how fast the box starts Python, which
    a busy CI run is free to change. The measured numbers printed by the run are in progress/M07.md.
    """
    baseline = time.monotonic()
    warm = run_hook(payload("credit limit"), url=None)
    baseline = time.monotonic() - baseline

    started = time.monotonic()
    result = run_hook(payload("average credit limit by status"), url=_closed_port())
    elapsed = time.monotonic() - started

    assert warm.returncode == 0
    assert result.returncode == 0
    assert result.stdout == b""
    assert elapsed < 1.0, f"hook took {elapsed:.3f}s with the server down"
    assert elapsed - baseline < 0.7, (
        f"a dead server cost {elapsed - baseline:.3f}s over a run that makes no call "
        f"(baseline {baseline:.3f}s): the hook is waiting out its timeout instead of giving up"
    )


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (401, json.dumps({"error": "invalid_token"})),
        (503, json.dumps({"code": "NOT_SUPPORTED", "problem": "no", "fix": "none"})),
        (200, "not json at all"),
        (200, json.dumps({})),
        (200, json.dumps({"additionalContext": ""})),
        (200, json.dumps({"additionalContext": "   "})),
        (200, json.dumps({"additionalContext": 42})),
    ],
    ids=[
        "unauthorized",
        "not-supported",
        "not-json",
        "no-field",
        "empty-pack",
        "blank-pack",
        "wrong-type",
    ],
)
def test_a_useless_answer_injects_nothing(status: int, body: str) -> None:
    """Every non-answer is the same answer for a hook: nothing, so the prompt goes as typed."""
    with Stub() as stub:
        stub.status = status
        stub.body = body
        result = run_hook(payload("credit limit"), url=stub.url)

    assert result.returncode == 0
    assert result.stdout == b""


@pytest.mark.parametrize(
    "stdin",
    [
        b"",
        b"{not json",
        b'{"prompt": ""}',
        b'{"prompt": "   "}',
        b'{"prompt": 7}',
        b'["a list, not the payload"]',
        b'{"no_prompt_field": "x"}',
    ],
    ids=[
        "empty",
        "broken-json",
        "empty-prompt",
        "blank-prompt",
        "not-a-string",
        "not-an-object",
        "no-prompt-field",
    ],
)
def test_an_unusable_payload_prints_nothing(stdin: bytes) -> None:
    with Stub() as stub:
        result = run_hook(stdin, url=stub.url)

    assert result.returncode == 0
    assert result.stdout == b""
    assert stub.requests == []


@pytest.mark.parametrize(
    ("url", "token"),
    [
        (None, TOKEN),
        ("http://127.0.0.1:1", None),
        ("", TOKEN),
        ("not-a-url", TOKEN),
        ("ftp://127.0.0.1/data", TOKEN),
    ],
    ids=["no-url", "no-token", "empty-url", "garbage-url", "wrong-scheme"],
)
def test_without_a_url_or_token_it_never_calls(url: str | None, token: str | None) -> None:
    """A call that cannot authenticate is a wasted round trip on every prompt: skip it."""
    with Stub() as stub:
        result = run_hook(payload("credit limit"), url=url, token=token)

    assert result.returncode == 0
    assert result.stdout == b""
    assert stub.requests == []


# --------------------------------------------------------------------------------------------------
# The plugin files: shape, no secrets, no hard-coded endpoint
# --------------------------------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def shipped_files() -> list[Path]:
    """Every file Claude Code loads from ``plugin/mdqm``, in a stable order."""
    return sorted(path for path in PLUGIN.rglob("*") if path.is_file())


def test_shipped_files_are_the_documented_ones() -> None:
    """The four component locations the docs name, and nothing extra to puzzle over."""
    relative = {path.relative_to(PLUGIN).as_posix() for path in shipped_files()}
    assert relative == {
        ".claude-plugin/plugin.json",
        ".mcp.json",
        "hooks/hooks.json",
        "scripts/context_hook.py",
        "skills/mdqm-query/SKILL.md",
    }


def test_all_plugin_text_files_are_utf8_with_lf_endings() -> None:
    """Hebrew in the skill, LF everywhere: a CRLF hook script is a Windows surprise waiting."""
    for path in shipped_files():
        raw = path.read_bytes()
        raw.decode("utf-8")  # raises if not UTF-8
        assert b"\r\n" not in raw, f"{path.name} has CRLF endings"


def test_manifest_declares_name_and_the_python_option_the_hook_needs() -> None:
    manifest = load_json(PLUGIN / ".claude-plugin" / "plugin.json")

    assert manifest["name"] == "mdqm"
    options = manifest["userConfig"]
    assert set(options) == {"mdqm_python"}
    # The hook's `command` is exactly this option, so an unknown key or a missing field is a
    # plugin that cannot start rather than a clear validation error.
    assert options["mdqm_python"]["type"] == "string"
    assert options["mdqm_python"]["title"] and options["mdqm_python"]["description"]
    assert options["mdqm_python"]["default"]


def test_mcp_json_points_at_the_server_through_environment_variables() -> None:
    servers = load_json(PLUGIN / ".mcp.json")["mcpServers"]

    (entry,) = servers.values()
    assert entry["type"] == "http"
    # Documented `${VAR}` expansion in a server's `url` and `headers`: the plugin file holds no
    # token, no host and no port, so one checkout serves every environment.
    assert entry["url"] == "${MDQM_URL}/mcp"
    assert entry["headers"]["Authorization"] == "Bearer ${MDQM_TOKEN}"
    # Five coarse tools that must be present on turn one, not behind a tool-search step.
    assert entry["alwaysLoad"] is True


def test_hook_uses_exec_form_so_no_shell_is_involved() -> None:
    """Exec form is the documented cross-platform path: Windows has no bash to shell out to."""
    hooks = load_json(PLUGIN / "hooks" / "hooks.json")["hooks"]

    assert list(hooks) == ["UserPromptSubmit"]
    (group,) = hooks["UserPromptSubmit"]
    # `UserPromptSubmit` has no matcher support, so a matcher here would be dead configuration.
    assert "matcher" not in group
    (hook,) = group["hooks"]
    assert hook["type"] == "command"
    assert hook["args"] == ["${CLAUDE_PLUGIN_ROOT}/scripts/context_hook.py"]
    assert hook["command"] == "${user_config.mdqm_python}"
    assert 0 < hook["timeout"] <= 5, "this hook blocks the prompt; a long timeout stalls a turn"


def test_no_token_host_or_port_literal_in_any_shipped_file() -> None:
    """AGENTS.md: no secrets, and nothing an organisation would have to edit to install."""
    for path in shipped_files():
        text = path.read_text(encoding="utf-8")
        assert re.search(r"\b(?:127\.0\.0\.1|localhost|0\.0\.0\.0)\b", text) is None, (
            f"{path.name} hard-codes a host"
        )
        assert re.search(r"https?://[^\"'\s]+:\d{2,5}", text) is None, (
            f"{path.name} hard-codes a port"
        )
        # A token value would be long and a variable reference is not.
        assert re.search(r"Bearer\s+(?!\$\{)[A-Za-z0-9._-]{8,}", text) is None, (
            f"{path.name} looks like it holds a bearer token"
        )


def test_hook_is_python_39_syntax_and_stdlib_only() -> None:
    """The hook runs on a user desktop's interpreter, not this repository's venv."""
    source = HOOK.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(HOOK), feature_version=(3, 9))

    imported = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    # Standard library only: no dependency is installed for the plugin's own interpreter.
    assert imported <= set(sys.stdlib_module_names), imported


def test_skill_frontmatter_is_first_line_and_names_the_five_real_tools() -> None:
    """The skill must open with ``---`` on line 1 or Claude Code treats it all as body text."""
    text = (PLUGIN / "skills" / "mdqm-query" / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")

    _, frontmatter, body = text.split("---", 2)
    fields = YAML(typ="safe").load(frontmatter)
    assert fields["name"] == "mdqm-query"
    assert fields["description"]
    # Plugin MCP tools are `mcp__plugin_<plugin>_<server>__<tool>`; the server here is `mdqm`.
    allowed = fields["allowed-tools"].split()
    assert allowed == [f"mcp__plugin_mdqm_mdqm__{name}" for name in TOOL_NAMES]
    for name in TOOL_NAMES:
        assert name in body
    # Three worked examples, all of them real spec shapes.
    assert body.count("```json") >= 3


def test_skill_examples_are_valid_query_specs() -> None:
    """A weak agent copies the example; an example that fails validation teaches the wrong spec."""
    from mdqm.spec.models import QuerySpec

    text = (PLUGIN / "skills" / "mdqm-query" / "SKILL.md").read_text(encoding="utf-8")
    blocks = [block.split("\n", 1)[1].rsplit("```", 1)[0] for block in text.split("```json")[1:]]
    parsed = [QuerySpec.model_validate(json.loads(block)) for block in blocks]

    assert len(parsed) >= 3
    assert {spec.from_table for spec in parsed} >= {"accounts", "usage_events"}
    # The Hebrew filter value is a business label, which the server maps to the source value.
    assert any(filter_.value == "פעיל" for spec in parsed for filter_ in spec.filters)
