"""The MCP server: five coarse tools, health and context endpoints, pluggable auth.

``create_server`` builds the object; ``serve`` runs it over streamable HTTP; ``StaticTokenVerifier``
is the dev identity implementation (the real identity provider is still unknown, AGENTS.md §9).
"""

from __future__ import annotations

from mdqm.server.app import (
    DEFAULT_PUBLIC_URL,
    DEFAULT_TOKENS_FILE,
    SERVER_INSTRUCTIONS,
    TOKENS_FILE_ENV,
    create_server,
    serve,
    tokens_file_path,
)
from mdqm.server.auth import (
    READ_SCOPE,
    AuthConfigError,
    StaticTokenVerifier,
    TokenEntry,
    TokenVerifier,
    UserIdentity,
    identity_from_access_token,
)
from mdqm.server.context import RequestContext, RequestContextUnavailable, current_context
from mdqm.server.schemas import (
    TOOL_SCHEMAS_PATH,
    render_tool_schemas,
    tool_schemas,
    write_tool_schemas,
)
from mdqm.server.tools import TOOL_NAMES, NotImplementedResult, register_tools

__all__ = [
    "DEFAULT_PUBLIC_URL",
    "DEFAULT_TOKENS_FILE",
    "READ_SCOPE",
    "SERVER_INSTRUCTIONS",
    "TOKENS_FILE_ENV",
    "TOOL_NAMES",
    "TOOL_SCHEMAS_PATH",
    "AuthConfigError",
    "NotImplementedResult",
    "RequestContext",
    "RequestContextUnavailable",
    "StaticTokenVerifier",
    "TokenEntry",
    "TokenVerifier",
    "UserIdentity",
    "create_server",
    "current_context",
    "identity_from_access_token",
    "register_tools",
    "render_tool_schemas",
    "serve",
    "tool_schemas",
    "tokens_file_path",
    "write_tool_schemas",
]
