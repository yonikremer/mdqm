"""Assemble the MCP server: five tools, ``GET /healthz``, ``POST /context`` and the auth hook.

Transport: streamable HTTP (``docs/DESIGN.md`` section 9: one central HTTP server, TLS at the edge).
Auth: the SDK's bearer middleware runs in front of every MCP request, so an unauthenticated or
unknown token is refused with ``401`` before a tool handler is reached. The two custom routes are
exempt from that middleware (``MCPServer.custom_route`` documents this), so ``/healthz`` is public
and ``/context`` checks the bearer token itself.

Nothing here reads a data source: the tool bodies are stubs (``mdqm.server.tools``). ``/context``
reads the semantic layer only (``mdqm.server.context_endpoint``).
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
from mdqm.config.settings import Settings
from mdqm.governor.handles import ResultStore
from mdqm.server.auth import (
    READ_SCOPE,
    AuthConfigError,
    StaticTokenVerifier,
    TokenVerifier,
    UserIdentity,
    identity_from_access_token,
)
from mdqm.server.context import RequestContextUnavailable
from mdqm.server.context_endpoint import (
    ContextPackSource,
    answer_context,
    context_source_from_settings,
)
from mdqm.server.tools import (
    FindDataServices,
    configure_find_data,
    configure_query_services,
    configure_services,
    register_tools,
)
from mdqm.server.wiring import WiringError, query_services, tool_services

__all__ = [
    "DEFAULT_PUBLIC_URL",
    "DEFAULT_TOKENS_FILE",
    "SERVER_INSTRUCTIONS",
    "TOKENS_FILE_ENV",
    "advertised_url",
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


def advertised_url(*, host: str, port: int, public_url: str | None = None) -> str:
    """The base URL the auth metadata advertises: the configured one, else ``http://host:port``.

    ``public_url`` exists because the derived URL is wrong for a proxied deployment, which is the
    normal one: the image binds ``0.0.0.0``, an edge Route or Ingress terminates TLS, and a client
    that reads ``http://0.0.0.0:8765`` as its issuer cannot follow it. See
    ``MDQM_SERVER__PUBLIC_URL``.
    """
    if public_url:
        return public_url.rstrip("/")
    return f"http://{host}:{port}"


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
    context_source: ContextPackSource | None = None,
) -> MCPServer:
    """Build the server object. No I/O and no clock: the caller supplies the verifier.

    Args:
        verifier: Who a bearer token belongs to. Required: a server with no auth is never served.
        public_url: Base URL advertised as the issuer and resource, e.g. ``https://mdqm.internal``.
        version: Reported at initialize; defaults to this distribution's version.
        context_source: The indexed semantic layer ``POST /context`` answers from. ``None`` is a
            supported state: the route then refuses with ``NOT_SUPPORTED`` and a ``fix``.
    """
    server: MCPServer = MCPServer(
        name="mdqm",
        version=version,
        instructions=SERVER_INSTRUCTIONS,
        token_verifier=verifier,
        auth=AuthSettings(
            # AuthSettings is a pydantic model whose fields are AnyHttpUrl: pydantic coerces these
            # strings on validation (mypy --strict accepts it), but pyright only sees the declared
            # field type. False positive, so editors stay green.
            issuer_url=f"{public_url.rstrip('/')}/",  # pyright: ignore[reportArgumentType]
            resource_server_url=f"{public_url.rstrip('/')}/mcp",  # pyright: ignore[reportArgumentType]
            required_scopes=[READ_SCOPE],
            # Tokens are opaque personal tokens, not RFC 8707 resource-bound grants.
            validate_token_resource=False,
        ),
    )
    register_tools(server)

    # The tools and the /context route must answer from the SAME semantic layer: an agent whose
    # hook got a Context Pack and whose `find_data` then says "no dictionary is loaded" has been
    # told two different things by the same server. Wiring it here makes that impossible.
    # `custom_route` has no return annotation in the SDK, so mypy cannot type the decorator.
    @server.custom_route("/healthz", methods=["GET"])  # type: ignore[untyped-decorator]
    async def healthz(request: Request) -> Response:
        """Liveness probe: public on purpose, so a load balancer needs no token."""
        return JSONResponse({"status": "ok", "name": "mdqm", "version": version})

    @server.custom_route("/context", methods=["POST"])  # type: ignore[untyped-decorator]
    async def context_endpoint(request: Request) -> Response:
        """Context Pack for the client hook: authenticated like the tools, never echoing a token."""
        identity = await _bearer_identity(verifier, request.headers.get("authorization"))
        if identity is None:
            return _unauthorized()
        return await answer_context(request, source=context_source)

    return server


def serve(*, host: str, port: int, public_url: str | None = None) -> None:
    """Run over streamable HTTP until interrupted (``mdqm serve``).

    Args:
        host: bind address.
        port: bind port.
        public_url: URL clients reach this server on, advertised as the auth issuer. ``None``
            derives it from ``host``/``port``, which is only right when nothing proxies the server.

    Raises:
        AuthConfigError: the token file is missing or unreadable. Serving without auth is not an
            option, so the caller must fail rather than start an open server.
        SemanticError: a semantic layer exists but does not load. Starting without it would serve
            guesses, so the server refuses to start instead.
    """
    verifier = StaticTokenVerifier.from_file(tokens_file_path())
    base_url = advertised_url(host=host, port=port, public_url=public_url)
    logger.info(
        "serving mdqm %s on %s:%d, advertised as %s for %d token(s)",
        __version__,
        host,
        port,
        base_url,
        len(verifier.users),
    )
    settings = Settings.load()
    context_source = context_source_from_settings(settings)
    if context_source is not None:
        # The tools and /context must answer from the SAME layer: otherwise an agent whose hook got
        # a Context Pack finds `find_data` claiming no dictionary is loaded, on the same server.
        # Wired here, where the layer is built -- `create_server` stays a pure constructor with no
        # process-global side effect, which is what keeps the tests honest.
        configure_find_data(
            FindDataServices(
                layer=context_source.layer,
                index=context_source.index,
                examples=tuple(context_source.examples),
                budget_tokens=context_source.budget_tokens,
            )
        )
    if context_source is not None:
        # The `query` tool refuses with "no pipeline is configured" until this exists. Wiring it
        # here -- rather than in `create_server` -- keeps the constructor pure and makes a failure
        # visible at startup instead of on the first agent's question.
        try:
            # One result store for the whole server: `query` fills it, `get_more` pages it.
            handles = ResultStore()
            configure_services(tool_services(settings, handles=handles))
            configure_query_services(
                query_services(settings, context_source.layer, handles=handles)
            )
        except WiringError:
            logger.exception("query pipeline not wired")
    server = create_server(
        verifier=verifier,
        public_url=base_url,
        context_source=context_source,
    )
    server.run("streamable-http", host=host, port=port)


# Re-exported for the CLI, which must report a bad token file as a clean error.
__all__ += ["AuthConfigError", "RequestContextUnavailable"]
