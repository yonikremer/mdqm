"""M01: the MCP server skeleton -- five tools, health, context stub, and the auth hook.

The strongest evidence available on this box is a real roundtrip: a uvicorn server on an ephemeral
port behind the SDK's own client, over streamable HTTP. The SDK's in-memory transport would skip the
bearer middleware entirely, so it could not show that an unauthenticated request is refused.

Sockets: ``tests/unit`` is socket-blocked by ``conftest.py``. This module opts in, then narrows the
permission to ``127.0.0.1`` so the loopback server can be reached and nothing else can.
"""

from __future__ import annotations

import importlib.metadata
import json
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx2
import pytest
import pytest_socket
import uvicorn
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client

from mdqm.server.app import SERVER_INSTRUCTIONS, create_server
from mdqm.server.auth import (
    READ_SCOPE,
    AuthConfigError,
    StaticTokenVerifier,
    TokenEntry,
    identity_from_access_token,
)
from mdqm.server.schemas import TOOL_SCHEMAS_PATH, render_tool_schemas, tool_schemas
from mdqm.server.tools import TOOL_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The whole module may open loopback sockets (the server listens on 127.0.0.1).
pytestmark = pytest.mark.enable_socket

TOKENS = (
    TokenEntry(token="tok-alice-9f2c", user_id="alice", display_name="Alice"),
    TokenEntry(token="tok-bob-7a41", user_id="bob", display_name="Bob"),
)

#: One valid call per tool: the stub runs no source, so the arguments only have to validate.
TOOL_CALLS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("find_data", {"question": "credit limit by customer status", "limit": 3}),
    (
        "query",
        {"spec": {"from": "customers", "select": ["customer_status"], "limit": 5}},
    ),
    ("get_more", {"handle": "h_1", "offset": 0, "limit": 10}),
    ("history", {"limit": 5}),
    ("feedback", {"query_id": "q_1", "rating": 5, "comment": "rounding was right"}),
)


@pytest.fixture(autouse=True)
def loopback_only() -> Iterator[None]:
    """Allow ``127.0.0.1`` only: anything leaving the box stays blocked for this module."""
    pytest_socket.socket_allow_hosts(["127.0.0.1"])
    yield


@pytest.fixture(scope="module")
def endpoint() -> Iterator[str]:
    """Run the real server on an ephemeral loopback port and yield its ``/mcp`` URL."""
    server = create_server(verifier=StaticTokenVerifier(TOKENS))
    config = uvicorn.Config(
        server.streamable_http_app(), host="127.0.0.1", port=0, log_level="warning"
    )
    running = uvicorn.Server(config)
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while not running.started:
            if time.monotonic() > deadline:  # pragma: no cover - only on a wedged box
                raise RuntimeError("the test server did not start within 30s")
            time.sleep(0.01)
        port = running.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        running.should_exit = True
        thread.join(timeout=10)


@asynccontextmanager
async def mcp_client(url: str, token: str | None = None) -> AsyncIterator[Client]:
    """An SDK client against ``url``, with the bearer header when ``token`` is given."""
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    http_client = httpx2.AsyncClient(headers=headers, timeout=30.0)
    try:
        async with Client(streamable_http_client(url, http_client=http_client)) as client:
            yield client
    finally:
        await http_client.aclose()


def base_url(endpoint: str) -> str:
    """``http://127.0.0.1:PORT/mcp`` -> ``http://127.0.0.1:PORT`` for the custom routes."""
    return endpoint.removesuffix("/mcp")


# --------------------------------------------------------------------------------------------------
# The pinned MCP major version
# --------------------------------------------------------------------------------------------------


def test_installed_mcp_sdk_is_the_pinned_major() -> None:
    # MCP 2.x renamed FastMCP to MCPServer and changed handler APIs; 1.x examples must not apply.
    assert importlib.metadata.version("mcp").split(".")[0] == "2"

    from mcp.server.auth.provider import TokenVerifier  # noqa: F401 - the verifier protocol we implement
    from mcp.server.mcpserver import MCPServer  # noqa: F401 - the 2.x entry point must exist


# --------------------------------------------------------------------------------------------------
# Tool surface
# --------------------------------------------------------------------------------------------------


async def test_lists_exactly_the_five_coarse_tools(endpoint: str) -> None:
    async with mcp_client(endpoint, TOKENS[0].token) as client:
        listed = await client.list_tools()

    assert tuple(tool.name for tool in listed.tools) == TOOL_NAMES
    for tool in listed.tools:
        assert tool.description, f"{tool.name} has no description for a weak model"
        assert tool.input_schema["type"] == "object"
        assert tool.output_schema is not None, f"{tool.name} advertises no structured output"
        assert tool.output_schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("tool_name", "arguments"), TOOL_CALLS, ids=[name for name, _ in TOOL_CALLS]
)
async def test_calls_each_tool_and_gets_a_typed_stub(
    endpoint: str, tool_name: str, arguments: dict[str, Any]
) -> None:
    async with mcp_client(endpoint, TOKENS[0].token) as client:
        result = await client.call_tool(tool_name, arguments)

    assert result.is_error is False
    structured = result.structured_content
    assert structured is not None
    assert structured["status"] == "not_implemented"
    assert structured["surface"] == tool_name
    assert structured["problem"] and structured["fix"], (
        "a stub must say what is missing and what to do"
    )
    assert structured["user_id"] == "alice"


def test_tool_descriptions_state_what_when_and_an_example() -> None:
    # A weak model reads the description, not the docs: it must be short and carry an example.
    from mdqm.server import tools

    for name in TOOL_NAMES:
        function = getattr(tools, name)
        doc = (function.__doc__ or "").strip()
        assert "Example:" in doc, f"{name} has no example"
        assert len(doc) <= 600, f"{name}'s description is too long for a weak model"
        assert doc.splitlines()[0].endswith("."), f"{name} has no one-line summary"


# --------------------------------------------------------------------------------------------------
# Auth hook
# --------------------------------------------------------------------------------------------------


async def _raw_initialize(url: str, token: str | None) -> httpx2.Response:
    """POST a JSON-RPC ``initialize`` straight at ``/mcp``: no SDK client in the way."""
    headers = {"Accept": "application/json, text/event-stream"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "unit-test", "version": "0"},
        },
    }
    async with httpx2.AsyncClient(timeout=30.0) as http:
        return await http.post(url, json=body, headers=headers)


@pytest.mark.parametrize("token", [None, "not-a-real-token"], ids=["missing", "invalid"])
async def test_bad_token_is_401_and_never_reaches_a_handler(
    endpoint: str, token: str | None
) -> None:
    response = await _raw_initialize(endpoint, token)

    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")
    # No MCP session was opened, so no tool handler ran: the middleware refused the request first.
    assert "mcp-session-id" not in response.headers
    assert "result" not in response.text


async def test_a_valid_token_opens_a_session(endpoint: str) -> None:
    # The same raw request with a valid token is not refused, so the 401s above are about the token.
    async with mcp_client(endpoint, TOKENS[0].token) as client:
        listed = await client.list_tools()

    assert len(listed.tools) == len(TOOL_NAMES)


@pytest.mark.parametrize(
    ("token", "user_id"), [(TOKENS[0].token, "alice"), (TOKENS[1].token, "bob")]
)
async def test_a_valid_token_reaches_the_handler_as_an_identity(
    endpoint: str, token: str, user_id: str
) -> None:
    async with mcp_client(endpoint, token) as client:
        result = await client.call_tool("find_data", {"question": "customers"})

    assert result.structured_content is not None
    assert result.structured_content["user_id"] == user_id


def test_identity_comes_from_the_token_file() -> None:
    verifier = StaticTokenVerifier(TOKENS)
    entry = TOKENS[0]

    import anyio

    access = anyio.run(verifier.verify_token, entry.token)
    assert access is not None
    identity = identity_from_access_token(access)
    assert identity.user_id == "alice"
    assert identity.display_name == "Alice"
    assert access.scopes == [READ_SCOPE]
    # The session is owned by the person, so one user cannot ride another user's session.
    assert access.client_id != StaticTokenVerifier(TOKENS[1:]).users[0]


def test_unknown_token_is_refused_without_echoing_it() -> None:
    import anyio

    verifier = StaticTokenVerifier(TOKENS)
    assert anyio.run(verifier.verify_token, "tok-nope") is None
    assert verifier.users == ("alice", "bob")


# --------------------------------------------------------------------------------------------------
# The dev token file
# --------------------------------------------------------------------------------------------------


def write_tokens(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_token_file_roundtrip(tmp_path: Path) -> None:
    path = write_tokens(
        tmp_path / "tokens.json",
        {"tokens": [{"token": "tok-x", "user_id": "dana", "display_name": "Dana"}]},
    )
    verifier = StaticTokenVerifier.from_file(path)
    assert verifier.users == ("dana",)


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        ("{not json", "not valid JSON"),
        ({"tokens": [{"user_id": "no-token"}]}, "does not match its shape"),
        (
            {"tokens": [{"token": "same", "user_id": "a"}, {"token": "same", "user_id": "b"}]},
            "same token",
        ),
    ],
    ids=["broken-json", "missing-token", "duplicate-token"],
)
def test_bad_token_file_names_the_problem_and_no_token(
    tmp_path: Path, payload: object, needle: str
) -> None:
    path = tmp_path / "tokens.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")

    with pytest.raises(AuthConfigError) as excinfo:
        StaticTokenVerifier.from_file(path)

    assert needle in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_missing_token_file_is_a_clean_error(tmp_path: Path) -> None:
    missing = tmp_path / "absent.json"
    with pytest.raises(AuthConfigError) as excinfo:
        StaticTokenVerifier.from_file(missing)
    assert str(missing) in str(excinfo.value)


# --------------------------------------------------------------------------------------------------
# Health and the /context stub
# --------------------------------------------------------------------------------------------------


async def test_healthz_is_public_and_reports_ok(endpoint: str) -> None:
    async with httpx2.AsyncClient(timeout=30.0) as http:
        response = await http.get(f"{base_url(endpoint)}/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize("token", [None, "not-a-real-token"], ids=["missing", "invalid"])
async def test_context_requires_a_token(endpoint: str, token: str | None) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx2.AsyncClient(timeout=30.0) as http:
        response = await http.post(
            f"{base_url(endpoint)}/context", json={"prompt": "hi"}, headers=headers
        )

    assert response.status_code == 401
    assert response.json()["error"] == "invalid_token"


async def test_context_stub_names_the_caller(endpoint: str) -> None:
    async with httpx2.AsyncClient(timeout=30.0) as http:
        response = await http.post(
            f"{base_url(endpoint)}/context",
            json={"prompt": "כמה לקוחות פעילים יש?"},  # Hebrew prompt: no ASCII assumption
            headers={"Authorization": f"Bearer {TOKENS[1].token}"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_implemented"
    assert body["surface"] == "context"
    assert body["user_id"] == "bob"


# --------------------------------------------------------------------------------------------------
# Generated schema document
# --------------------------------------------------------------------------------------------------


async def test_docs_tool_schemas_json_is_up_to_date() -> None:
    document = await tool_schemas(create_server(verifier=StaticTokenVerifier(TOKENS)))
    on_disk = (REPO_ROOT / TOOL_SCHEMAS_PATH).read_text(encoding="utf-8")

    assert render_tool_schemas(document) == on_disk, (
        "docs/tool-schemas.json is stale: run python scripts/gen_tool_schemas.py"
    )


async def test_exported_schemas_cover_every_tool_and_are_deterministic() -> None:
    server = create_server(verifier=StaticTokenVerifier(TOKENS))
    first = await tool_schemas(server)
    second = await tool_schemas(server)

    assert render_tool_schemas(first) == render_tool_schemas(second)
    assert [tool["name"] for tool in first["tools"]] == sorted(TOOL_NAMES)
    assert first["generated_by"] == "scripts/gen_tool_schemas.py"


# --------------------------------------------------------------------------------------------------
# The server object itself
# --------------------------------------------------------------------------------------------------


def test_server_instructions_are_short_and_point_at_the_tools() -> None:
    assert len(SERVER_INSTRUCTIONS) < 800
    for name in TOOL_NAMES:
        assert f"`{name}`" in SERVER_INSTRUCTIONS


def test_cli_serve_exposes_host_and_port() -> None:
    from typer.testing import CliRunner

    from mdqm.cli import app

    result = CliRunner().invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output
