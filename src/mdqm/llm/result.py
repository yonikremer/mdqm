"""What a model call returns: the validated object, plus every attempt it took.

An agent-facing pipeline must be able to say *how* an answer came about: how many calls it
needed, why the earlier ones were rejected, and whether it came from the cache (``cached`` and
``fetched_at``; principle 10, "everything reported, nothing silent"). The attempt records hold
short, sanitised problems only: never the prompt, the completion or a header.
"""

from __future__ import annotations

from typing import Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, Field

from mdqm.models.base import MDQMModel, UtcDatetime

#: The way one HTTP call (or its validation) ended.
AttemptOutcome = Literal[
    "ok",
    "invalid_json",
    "invalid_schema",
    "empty_response",
    "timeout",
    "transport_error",
    "http_error",
    "response_format_rejected",
    "non_json_body",
]


ModelT = TypeVar("ModelT", bound=BaseModel)


class Attempt(MDQMModel):
    """One try at answering: its number, how it ended and the short reason."""

    number: int = Field(ge=1)
    outcome: AttemptOutcome
    problem: str | None = Field(
        default=None, max_length=400, description="Sanitised reason; never a prompt or a key."
    )
    status_code: int | None = None
    error_code: str | None = None
    retry_delay_s: float | None = Field(
        default=None, ge=0, description="Seconds slept before the next attempt; None for the last."
    )


class LLMResult(MDQMModel, Generic[ModelT]):
    """A validated model answer and the trail that produced it."""

    value: ModelT
    attempts: tuple[Attempt, ...] = ()
    model: str
    prompt_version: str
    cached: bool = Field(default=False, description="True when no HTTP call was made.")
    fetched_at: UtcDatetime = Field(
        description="When the answer was produced (a cache hit keeps it)."
    )


#: The shape every model backend in this project offers (real client, :class:`FakeLLM`).
DEFAULT_PROMPT_VERSION = "1"


class StructuredLLM(Protocol):
    """Ask the model for one closed JSON object and get it validated in code."""

    def structured(
        self,
        prompt: str,
        schema: type[ModelT],
        *,
        system: str | None = None,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> LLMResult[ModelT]: ...
