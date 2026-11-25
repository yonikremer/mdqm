"""The request context a tool handler receives: who is calling, and with which scopes.

The SDK's bearer middleware authenticates the HTTP request and the MCP session keeps that token in
an ambient contextvar for the lifetime of the session (verified in 2.3.0:
``mcp.server.auth.middleware.auth_context.get_access_token`` is populated inside a tool handler over
streamable HTTP). This module turns that into a typed, token-free
:class:`RequestContext` so handlers never touch the raw token themselves.
"""

from __future__ import annotations

from mcp.server.auth.middleware.auth_context import get_access_token
from pydantic import Field

from mdqm.models.base import MDQMModel
from mdqm.server.auth import UserIdentity, identity_from_access_token

__all__ = ["RequestContext", "RequestContextUnavailable", "current_context"]


class RequestContextUnavailable(RuntimeError):
    """A tool ran without an authenticated identity: the transport is missing its auth hook."""


class RequestContext(MDQMModel):
    """The authenticated caller. Holds no token: a token must never reach a log line or a result."""

    identity: UserIdentity = Field(description="Who is asking.")
    scopes: tuple[str, ...] = Field(default=(), description="Scopes the token carries.")

    @property
    def user_id(self) -> str:
        """The user id, as history and audit record it."""
        return self.identity.user_id


def current_context() -> RequestContext:
    """The context of the message being handled.

    Raises:
        RequestContextUnavailable: no identity is bound to this message, which means the server was
            built without a verifier. That is a wiring bug, not something the agent can fix.
    """
    token = get_access_token()
    if token is None:
        raise RequestContextUnavailable(
            "no authenticated identity on this message: build the server with a token verifier "
            "and serve it with auth enabled"
        )
    return RequestContext(identity=identity_from_access_token(token), scopes=tuple(token.scopes))
