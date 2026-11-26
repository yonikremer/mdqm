"""``POST /context``: the Context Pack a client hook injects before the agent answers (task M06).

The Claude Code plugin's ``UserPromptSubmit`` hook posts the user's message here and puts the
answer into ``additionalContext``, so a weak agent starts every turn knowing the measures, value
maps, approved joins, tables and example specs that match the question (``docs/DESIGN.md`` section
9). ``mdqm find_data`` (M02) builds the same block from the same call, so the two cannot disagree.

What this module owns, and what it deliberately does not:

* **The wire contract**: ``POST {"prompt": str}`` -> ``{"additionalContext": str}``. Nothing else is
  in a success body. ``additionalContext`` is ``""`` when nothing is relevant, because the block is
  injected automatically and an irrelevant prompt must cost zero tokens: empty is a normal answer,
  not an error (see :mod:`mdqm.context.pack`).
* **The refusals are the agent-facing :class:`~mdqm.models.ErrorResponse`**, never a stack trace:
  ``400`` + ``INVALID_SPEC`` for a body that is not ``{"prompt": <string>}``, and ``503`` +
  ``NOT_SUPPORTED`` when this deployment has no semantic layer to read (a wiring state, not the
  caller's mistake -- the ``fix`` says so). Authentication is *not* here: the transport verifies the
  bearer token before this handler runs and answers ``401`` itself.
* **No token, no prompt echo, no data values.** A token never enters this module (the caller passes
  only the identity it already verified); the prompt is an input, never output; and the block is
  built from the semantic layer alone, so a result row cannot leak through the hook.
* **Determinism**: the same question over the same semantic version is a byte-identical body, which
  is what lets a reviewer diff injected context and a test assert it.

The indexing itself belongs to :class:`~mdqm.context.index.ContextIndex` and the budget to
:func:`~mdqm.context.pack.build_pack`; this module only decides *which* layer answers and shapes the
HTTP response.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import Field, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from mdqm.config.settings import Settings
from mdqm.context.index import ApprovedExample, ContextIndex
from mdqm.context.pack import DEFAULT_PACK_BUDGET, ContextPack, build_pack
from mdqm.errors import from_validation_error, not_supported
from mdqm.models import ErrorResponse
from mdqm.models.base import MDQMModel
from mdqm.semantic import SemanticLayer, load
from mdqm.store import Store

__all__ = [
    "MAX_PROMPT_CHARS",
    "ContextPackSource",
    "ContextRequest",
    "answer_context",
    "context_source_from_settings",
]

logger = logging.getLogger(__name__)

#: Longest prompt accepted. A user message is short; a bigger body is a bug or an attack, and
#: refusing names it instead of quietly reading the first N characters.
MAX_PROMPT_CHARS: Final[int] = 4_000

#: The refusal when this deployment has no semantic layer to read. It says what to do instead of
#: pretending the prompt was irrelevant, because "empty" already means "nothing matched".
_NO_LAYER_FALLBACK: Final[str] = (
    "carry on without injected context and ask the user for table and column names"
)


class ContextRequest(MDQMModel):
    """The hook's request body. Unknown fields are refused, so a typo is visible, not dropped."""

    prompt: str = Field(
        max_length=MAX_PROMPT_CHARS,
        description="The user's message, as typed: Hebrew or English, any length up to the cap.",
    )


@dataclass(frozen=True)
class ContextPackSource:
    """Everything the endpoint needs to answer: one indexed semantic layer.

    Args:
        layer: the approved semantic facts the block is rendered from.
        index: the search index built over ``layer`` (same layer, same examples).
        examples: the approved example specs the index was built with.
        budget_tokens: the block's ceiling; the same default the ``query`` tool's answers use.
    """

    layer: SemanticLayer
    index: ContextIndex
    examples: Sequence[ApprovedExample] = ()
    budget_tokens: int = DEFAULT_PACK_BUDGET

    @classmethod
    def build(
        cls,
        layer: SemanticLayer,
        *,
        store: Store,
        labels: Mapping[str, Sequence[str]] | None = None,
        examples: Sequence[ApprovedExample] = (),
        budget_tokens: int = DEFAULT_PACK_BUDGET,
    ) -> ContextPackSource:
        """Index ``layer`` into ``store`` and return the source that answers from it.

        Raises:
            ContextIndexError: a label names nothing in the layer, or ``store`` was never migrated.
        """
        index = ContextIndex(store)
        index.build(layer, labels=labels, examples=examples)
        return cls(
            layer=layer,
            index=index,
            examples=tuple(examples),
            budget_tokens=budget_tokens,
        )

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        *,
        store: Store,
        labels: Mapping[str, Sequence[str]] | None = None,
        examples: Sequence[ApprovedExample] = (),
        budget_tokens: int = DEFAULT_PACK_BUDGET,
    ) -> ContextPackSource:
        """Load a semantic directory (``generated/`` + ``curated/``) and index it.

        Raises:
            SemanticError: the directory has no semantic files, or one of them is invalid.
        """
        return cls.build(
            load(root),
            store=store,
            labels=labels,
            examples=examples,
            budget_tokens=budget_tokens,
        )

    def context_for(self, prompt: str) -> ContextPack:
        """The block for ``prompt``: the same call ``find_data`` (M02) makes, so the two agree."""
        return build_pack(
            prompt,
            layer=self.layer,
            index=self.index,
            examples=self.examples,
            budget_tokens=self.budget_tokens,
        )


def context_source_from_settings(settings: Settings | None = None) -> ContextPackSource | None:
    """The pack source this deployment serves from, or ``None`` when it has no semantic layer.

    ``None`` is a real state, not a failure: a fresh installation serves the MCP tools before the
    bootstrap pipeline has approved any semantic facts. The endpoint then answers ``NOT_SUPPORTED``
    with a ``fix``, so the state is visible to the caller and in the log rather than pretending the
    prompt matched nothing. A layer that exists but does not load raises: absence of facts is
    normal, corruption is not, and a started server must not hide it.
    """
    resolved = settings if settings is not None else Settings()
    root = resolved.paths.semantic_dir
    if not root.is_dir():
        logger.warning(
            "no semantic layer, so /context answers NOT_SUPPORTED",
            extra={"semantic_dir": str(root)},
        )
        return None
    store = Store(resolved.paths.metadata_db)
    store.initialize()
    source = ContextPackSource.from_directory(root, store=store)
    logger.info(
        "context endpoint ready",
        extra={"semantic_dir": str(root), "semantic_version": source.layer.version_hash},
    )
    return source


async def answer_context(
    request: Request,
    *,
    source: ContextPackSource | None,
) -> Response:
    """Shape the HTTP answer for an already-authenticated request.

    The caller (the transport) has verified the bearer token, so an identity exists by the time this
    runs and none is needed here: the block is the same for every caller who can read the layer.

    Args:
        request: the ``POST /context`` request; only its body is read.
        source: the indexed semantic layer, or ``None`` when this server has none.
    """
    body = await _request(request)
    if isinstance(body, ErrorResponse):
        return _refusal(body, status_code=400)
    if source is None:
        return _refusal(
            not_supported(
                source="the mdqm server",
                capability="context injection",
                fallback=_NO_LAYER_FALLBACK,
            ),
            status_code=503,
        )
    pack = source.context_for(body.prompt)
    logger.info(
        "context pack built",
        extra={
            "items": len(pack.items),
            "omitted": len(pack.omitted),
            "tokens_estimated": pack.tokens_estimated,
            "budget_tokens": pack.budget_tokens,
            "semantic_version": pack.semantic_version,
        },
    )
    return JSONResponse({"additionalContext": pack.text})


async def _request(request: Request) -> ContextRequest | ErrorResponse:
    """The validated body, or the agent-facing refusal that says what the hook must send."""
    raw = await request.body()
    try:
        return ContextRequest.model_validate_json(raw)
    except ValidationError as exc:
        return from_validation_error(exc, root="context")


def _refusal(error: ErrorResponse, *, status_code: int) -> Response:
    """A refusal body: the shared error shape, and no ``additionalContext`` for a hook to inject."""
    return JSONResponse(error.model_dump(), status_code=status_code)
