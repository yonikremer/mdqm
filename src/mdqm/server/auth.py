"""Identity for the MCP server: a pluggable verifier and the static-token dev implementation.

How personal identity is issued in the first organisation is **unknown** (AGENTS.md section 9), so
only the interface and a static-token dev implementation live here. The interface is deliberately
the MCP SDK's own shape -- ``async verify_token(token) -> AccessToken | None`` -- so a verifier can
be handed straight to ``MCPServer(token_verifier=...)`` and the SDK's bearer middleware refuses
unauthenticated requests before a tool handler runs.

The identity of a caller travels in the SDK's ``AccessToken``: ``subject`` is the ``user_id``,
``claims["display_name"]`` is the human name. A token value is never logged, never put in an error
message, and never reaches a tool handler (see :mod:`mdqm.server.context`).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from mcp.server.auth.provider import AccessToken
from pydantic import Field, ValidationError

from mdqm.models.base import MDQMModel

__all__ = [
    "READ_SCOPE",
    "AuthConfigError",
    "StaticTokenVerifier",
    "TokenEntry",
    "TokenFile",
    "TokenVerifier",
    "UserIdentity",
    "identity_from_access_token",
]

logger = logging.getLogger(__name__)

#: The one scope every token must carry. Coarse on purpose: the five tools are all read-only.
READ_SCOPE = "mdqm.read"

#: Claim name that carries the human display name on the SDK access token.
DISPLAY_NAME_CLAIM = "display_name"


class AuthConfigError(ValueError):
    """The token file cannot be used. Never contains a token value."""


class UserIdentity(MDQMModel):
    """Who is asking. ``user_id`` is what history and audit record: a stable id, not an email."""

    user_id: str = Field(min_length=1, description="Stable per-person id, e.g. 'yoni'.")
    display_name: str | None = Field(default=None, description="Human name, for the digest.")


class TokenEntry(MDQMModel):
    """One row of the dev token file: a bearer token and the identity it stands for."""

    token: str = Field(min_length=1, description="The bearer token value. Dev only; never commit.")
    user_id: str = Field(min_length=1)
    display_name: str | None = None
    scopes: tuple[str, ...] = (READ_SCOPE,)


class TokenFile(MDQMModel):
    """The dev token file's shape: ``{"tokens": [{...}, ...]}``."""

    tokens: tuple[TokenEntry, ...] = ()


class TokenVerifier(Protocol):
    """Turn a bearer token into an :class:`AccessToken`, or ``None`` when it is not valid.

    Structurally the MCP SDK's ``mcp.server.auth.provider.TokenVerifier``, so an implementation can
    be passed to ``MCPServer(token_verifier=...)`` without an adapter.
    """

    async def verify_token(self, token: str) -> AccessToken | None: ...


def _digest(token: str) -> str:
    """Compare tokens by digest: the file lookup then never compares secret bytes."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def identity_from_access_token(token: AccessToken) -> UserIdentity:
    """Read :class:`UserIdentity` out of a verified access token.

    ``subject`` carries the user id; ``client_id`` is the fallback so a verifier that only sets
    ``client_id`` still works instead of authenticating an empty identity.
    """
    claims = token.claims or {}
    display_name = claims.get(DISPLAY_NAME_CLAIM)
    return UserIdentity(
        user_id=token.subject or token.client_id,
        display_name=display_name if isinstance(display_name, str) else None,
    )


class StaticTokenVerifier:
    """Dev/test identity: bearer tokens loaded once from a JSON file, mapped to a user identity.

    The file holds literal tokens, so it is a deployment secret (mounted file or an env-pointed
    path), never a repository file. Anything the first organisation does instead -- an internal
    directory, ``headersHelper`` short-lived tokens, OAuth/OIDC, a reverse proxy identity header --
    is a different :class:`TokenVerifier`, not a change here.
    """

    def __init__(self, entries: Iterable[TokenEntry] = ()) -> None:
        self._by_digest: dict[str, TokenEntry] = {}
        for entry in entries:
            digest = _digest(entry.token)
            if digest in self._by_digest:
                raise AuthConfigError(
                    f"two token entries share the same token for user {entry.user_id!r}; "
                    "fix: give each person their own token"
                )
            self._by_digest[digest] = entry

    @property
    def users(self) -> tuple[str, ...]:
        """The known user ids, sorted: enough for a startup log without leaking a token."""
        return tuple(sorted({entry.user_id for entry in self._by_digest.values()}))

    @classmethod
    def from_file(cls, path: str | Path) -> StaticTokenVerifier:
        """Load the token file. Every failure names the path and never echoes a token."""
        target = Path(path)
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise AuthConfigError(
                f"cannot read the token file {target}: {exc.strerror}. "
                f"fix: create it (see .env.example) or point the tokens-file setting at one"
            ) from exc
        try:
            raw: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AuthConfigError(
                f"token file {target} is not valid JSON: {exc.msg} at line {exc.lineno}"
            ) from exc
        try:
            parsed = TokenFile.model_validate(raw)
        except ValidationError as exc:
            raise AuthConfigError(f"token file {target} does not match its shape: {exc}") from exc
        try:
            verifier = cls(parsed.tokens)
        except AuthConfigError as exc:
            # Name the file too: "which token file is wrong" is the first thing to know.
            raise AuthConfigError(f"token file {target}: {exc}") from exc
        logger.info("loaded static tokens for %d user(s)", len(parsed.tokens))
        return verifier

    async def verify_token(self, token: str) -> AccessToken | None:
        """Return the access token, or ``None`` for an unknown token. Never logs the value."""
        entry = self._by_digest.get(_digest(token))
        if entry is None:
            logger.info("rejected a bearer token: not in the token file")
            return None
        claims: dict[str, Any] = {DISPLAY_NAME_CLAIM: entry.display_name}
        return AccessToken(
            token=token,
            client_id=f"mdqm:{entry.user_id}",
            scopes=list(entry.scopes),
            subject=entry.user_id,
            claims=claims,
        )
