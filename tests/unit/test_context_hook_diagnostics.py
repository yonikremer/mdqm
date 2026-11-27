"""M4/M5: the plugin hook's timeout, and the CLI minimum its exec form requires.

Two findings from the deployment review, both about a hook that fails invisibly:

* ``TIMEOUT_SECONDS`` was 0.45 s, measured on loopback. Behind a Route a prompt pays DNS + TCP +
  TLS + the bearer check + FTS + pack render, so the context was dropped on every prompt off a
  loopback. The budget is now 2.0 s, with ``hooks/hooks.json``'s 3 s as the hard bound.
* ``hooks/hooks.json`` uses exec form (``command`` + ``args``), which needs Claude Code
  >= 2.1.139. An older CLI runs the interpreter with no script and hangs in a REPL on every prompt,
  so the minimum is declared in the manifest and in the README.

The hook is spawned as the CLI spawns it -- exec form, no shell -- and only stderr/exit code are
asserted here; the injection contract itself is pinned by ``test_plugin_hook.py``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from mdqm.server.app import DEFAULT_TOKENS_FILE  # noqa: F401 - proves the plugin imports nothing odd

PLUGIN = Path(__file__).resolve().parents[2] / "plugin" / "mdqm"
HOOK = PLUGIN / "scripts" / "context_hook.py"
README = PLUGIN.parent / "README.md"

#: A port nothing listens on. No in-process socket is opened, so ``pytest-socket`` stays happy:
#: the connection attempt happens inside the spawned hook process.
REFUSED = "http://127.0.0.1:1"

#: Not a credential: a value the hook never sends anywhere in these tests.
TOKEN = "tok-hook-diagnostics"

PROMPT = json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": "credit limit"}).encode(
    "utf-8"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_hook(
    stdin: bytes, *, url: str | None, token: str | None = TOKEN
) -> subprocess.CompletedProcess[bytes]:
    """Spawn the hook the way ``hooks/hooks.json`` does: exec form, fixed argv, no shell."""
    env = {
        key: value
        for key, value in {
            "MDQM_URL": url,
            "MDQM_TOKEN": token,
            "SYSTEMROOT": os.environ.get("SYSTEMROOT"),
        }.items()
        if value is not None
    }
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(HOOK)],
        input=stdin,
        capture_output=True,
        env=env,
        timeout=60,
        check=False,
    )


def hook_source() -> str:
    return HOOK.read_text(encoding="utf-8")


def test_the_budget_survives_a_route_not_just_loopback() -> None:
    """0.45 s was a loopback number; the failure it caused was invisible to the user."""
    source = hook_source()
    match = re.search(r"^TIMEOUT_SECONDS = ([0-9.]+)$", source, re.MULTILINE)
    connect = re.search(r"^CONNECT_TIMEOUT_SECONDS = ([0-9.]+)$", source, re.MULTILINE)
    assert match is not None, "TIMEOUT_SECONDS must stay a module-level constant"
    assert connect is not None, "the connect phase needs its own, shorter budget"
    budget = float(match.group(1))

    assert budget >= 2.0
    # A dead server must stay free: the connect budget is what keeps "server down" from hanging.
    assert float(connect.group(1)) < 1.0
    # The hard bound in hooks.json still bounds the prompt: the two must not cross over.
    (hook,) = load_json(PLUGIN / "hooks" / "hooks.json")["hooks"]["UserPromptSubmit"][0]["hooks"]
    assert hook["timeout"] > budget, "the hook's own budget must expire before the hard timeout"
    assert hook["timeout"] <= 5


def test_an_unreachable_server_says_so_on_stderr() -> None:
    """Fail open, but not silently: principle 10, without polluting the model's context."""
    result = run_hook(PROMPT, url=REFUSED)

    assert result.returncode == 0
    assert result.stdout == b"", "stdout is the injection channel: a diagnostic must not go there"
    lines = result.stderr.decode("utf-8").splitlines()
    assert len(lines) == 1, f"expected exactly one diagnostic line, got {lines}"
    assert lines[0].startswith("mdqm: no context injected:")
    assert "127.0.0.1" in lines[0], "the line must name what could not be reached"


def test_a_garbled_payload_says_so_on_stderr() -> None:
    """A CLI that changes the payload shape is a real failure, not "nothing to inject"."""
    result = run_hook(b'{"no_prompt_field": "x"}', url=REFUSED)

    assert result.returncode == 0
    assert result.stdout == b""
    assert result.stderr.decode("utf-8").startswith("mdqm: no context injected:")


def test_a_url_with_no_scheme_says_so_on_stderr() -> None:
    result = run_hook(PROMPT, url="mdqm.internal")

    assert result.returncode == 0
    assert result.stdout == b""
    assert "MDQM_URL" in result.stderr.decode("utf-8")


@pytest.mark.parametrize(
    ("url", "token"),
    [(None, "tok"), ("http://127.0.0.1:1", None)],
    ids=["no-url", "no-token"],
)
def test_a_plugin_that_is_not_configured_stays_quiet(url: str | None, token: str | None) -> None:
    """Not configured is not a failure: a stderr line on every prompt would be noise."""
    result = run_hook(PROMPT, url=url, token=token)

    assert result.returncode == 0
    assert result.stdout == b""
    assert result.stderr == b""


def test_the_stdout_channel_stays_an_envelope_or_nothing() -> None:
    """Guards the split above: a diagnostic must never be written where context is expected."""
    for stdin, url in [(PROMPT, REFUSED), (b"", REFUSED), (PROMPT, None)]:
        result = run_hook(stdin, url=url)
        assert result.stdout in (b"",) or json.loads(result.stdout).keys() == {"hookSpecificOutput"}


def test_the_manifest_declares_the_cli_minimum_the_exec_form_needs() -> None:
    manifest = load_json(PLUGIN / ".claude-plugin" / "plugin.json")

    assert manifest["requires"]["claudeCode"] == ">=2.1.139"


def test_the_readme_states_the_minimum_before_the_install_steps() -> None:
    """A requirement buried after the install steps is a requirement nobody reads."""
    text = README.read_text(encoding="utf-8")
    install = text.index("## Install")

    assert "2.1.139" in text[:install]
