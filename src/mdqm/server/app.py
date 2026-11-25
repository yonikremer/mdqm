"""Assemble the MCP server: five tools, ``GET /healthz``, ``POST /context`` and the auth hook.

Transport: streamable HTTP (``docs/DESIGN.md`` section 9: one central HTTP server, TLS at the edge).
Auth: the SDK's bearer middleware runs in front of every MCP request, so an unauthenticated or
unknown token is refused with ``401`` before a tool handler is reached. The two custom routes are
exempt from that middleware (``MCPServer.custom_route`` documents this), so ``/healthz`` is public
and ``/context`` checks the bearer token itself.

Nothing here reads a data source: the tool bodies are stubs (``mdqm.server.tools``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from mdqm import __version__
from mdqm.server.auth import (
    READ_SCOPE,
    AuthConfigError,
    StaticTokenVerifier,
    TokenVerifier,
    UserIdentity,
    identity_from_access_token,
)
from mdqm.server.context import RequestContextUnavailable
from mdqm.server.tools import NotImplementedResult, register_tools

__all__ = [
    "DEFAULT_PUBLIC_URL",
    "DEFAULT_TOKENS_FILE",
    "SERVER_INSTRUCTIONS",
    "TOKENS_FILE_ENV",
    "create_server",
    "serve",
    "tokens_file_path",
]

logger = logging.getLogger(__name__)

#: Base URL the server advertises in its auth metadata. The edge terminates TLS, so plain HTTP here.
DEFAULT_PUBLIC_URL = "http://127.0.0.1:8765"

#: Env var pointing at the dev token file (see :mod:`mdqm.server.auth`).
TOKENS_FILE_ENV = "MDQM_TOKENS_FILE"
DEFAULT_TOKENS_FILE = "tokens.json"

#: Sent to every client at initialize. Short and imperative: the calling model is weak.
SERVER_INSTRUCTIONS = (
    "Query organisational data without writing SQL. "
    "1) `find_data` when you are unsure of a table, column or value name. "
    "2) `query` with a JSON spec -- never SQL -- to get a summary, a sample and a handle. "
    "3) `get_more` with that handle for more rows. "
    "4) `history` for your own recent queries, `feedback` to rate an answer. "
    "Errors carry `code`, `problem` and an exact `fix`: follow the `fix` and retry once."
)


def tokens_file_path() -> Path:
    """Where the dev token file lives: ``MDQM_TOKENS_FILE``, else ``tokens.json`` in the cwd."""
    return Path(os.environ.get(TOKENS_FILE_ENV) or DEFAULT_TOKENS_FILE)


async def _bearer_identity(
    verifier: TokenVerifier, authorization: str | None
) -> UserIdentity | None:
    """Verify an ``Authorization: Bearer`` header for a custom route the SDK's middleware skips."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    access = await verifier.verify_token(token.strip())
    return None if access is None else identity_from_access_token(access)


def _unauthorized() -> Response:
    """The same 401 shape the SDK's middleware sends, for the routes that check auth themselves."""
    return JSONResponse(
        {"error": "invalid_token", "error_description": "Authentication required"},
        status_code=401,
        headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
    )


def create_server(
    *,
    verifier: TokenVerifier,
    public_url: str = DEFAULT_PUBLIC_URL,
    version: str = __version__,
) -> MCPServer:
    """Build the server object. No I/O and no clock: the caller supplies the verifier.

    Args:
        verifier: Who a bearer token belongs to. Required: a server with no auth is never served.
        public_url: Base URL advertised as the issuer and resource, e.g. ``https://mdqm.internal``.
        version: Reported at initialize; defaults to this distribution's version.
    """
    server: MCPServer = MCPServer(
        name="mdqm",
        version=version,
        instructions=SERVER_INSTRUCTIONS,
        token_verifier=verifier,
        auth=AuthSettings(
            issuer_url=f"{public_url.rstrip('/')}/",
            resource_server_url=f"{public_url.rstrip('/')}/mcp",
            required_scopes=[READ_SCOPE],
            # Tokens are opaque personal tokens, not RFC 8707 resource-bound grants.
            validate_token_resource=False,
        ),
    )
    register_tools(server)

    # `custom_route` has no return annotation in the SDK, so mypy cannot type the decorator.
    @server.custom_route("/healthz", methods=["GET"])  # type: ignore[untyped-decorator]
    async def healthz(request: Request) -> Response:  # noqa: ARG001 - the SDK's handler signature
        """Liveness probe: public on purpose, so a load balancer needs no token."""
        return JSONResponse({"status": "ok", "name": "mdqm", "version": version})

    @server.custom_route("/context", methods=["POST"])  # type: ignore[untyped-decorator]
    async def context_endpoint(request: Request) -> Response:
        """Context Pack endpoint for the client hook (stub in this build)."""
        identity = await _bearer_identity(verifier, request.headers.get("authorization"))
        if identity is None:
            return _unauthorized()
        stub = NotImplementedResult(
            surface="context",
            user_id=identity.user_id,
            problem="the Context Pack is not built yet, so no context is injected for this prompt",
            fix="carry on without injected context; ask the user for table and column names",
        )
        return JSONResponse(stub.model_dump())

    return server


def serve(*, host: str, port: int) -> None:
    """Run over streamable HTTP until interrupted (``mdqm serve``).

    Raises:
        AuthConfigError: the token file is missing or unreadable. Serving without auth is not an
            option, so the caller must fail rather than start an open server.
    """
    verifier = StaticTokenVerifier.from_file(tokens_file_path())
    logger.info(
        "serving mdqm %s on %s:%d for %d token(s)", __version__, host, port, len(verifier.users)
    )
    server = create_server(verifier=verifier, public_url=f"http://{host}:{port}")
    server.run("streamable-http", host=host, port=port)


# Re-exported for the CLI, which must report a bad token file as a clean error.
__all__ += ["AuthConfigError", "RequestContextUnavailable"]
