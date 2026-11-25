"""A scripted model for tests and for a deployment with no endpoint: no socket, no key, no clock.

``FakeLLM`` answers from a queue of scripted replies and records every call it received, so a
pipeline test can assert on the exact prompt it built without a network. It implements
:class:`~mdqm.llm.result.StructuredLLM`, which is the only thing the pipeline may depend on.

A scripted item is one of:

* an instance of the requested schema: returned as the value,
* a mapping, or a JSON string: validated against the schema, so a wrong shape fails the test,
* an exception: raised as-is (to script a failure after a success),
* a callable ``(prompt) -> item``: computed at call time, for replies that echo the input.

Replies are consumed in order. Running out raises :class:`AssertionError`, not a made-up answer:
a test that asks more questions than it scripted is a broken test.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypeAlias

from pydantic import BaseModel

from mdqm.llm.result import DEFAULT_PROMPT_VERSION, Attempt, LLMResult, ModelT
from mdqm.models.base import MDQMModel

__all__ = ["FakeCall", "FakeLLM", "Scripted"]

#: One scripted reply; see the module docstring for the accepted shapes.
Scripted: TypeAlias = BaseModel | dict[str, Any] | str | BaseException | Callable[[str], Any]


class FakeCall(MDQMModel):
    """One recorded call: what the pipeline asked for (never an answer, which the test scripted)."""

    prompt: str
    system: str | None = None
    schema_name: str
    prompt_version: str


class FakeLLM:
    """Scripted :class:`~mdqm.llm.result.StructuredLLM` for tests."""

    def __init__(
        self,
        *replies: Scripted,
        model: str = "fake-llm",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._replies: list[Scripted] = list(replies)
        self._model = model
        self._now = clock or (lambda: datetime.now(UTC))
        self.calls: list[FakeCall] = []

    @property
    def model_name(self) -> str:
        """The name recorded in every result (tests assert it reaches the audit trail)."""
        return self._model

    def queue(self, *replies: Scripted) -> None:
        """Append more scripted replies (after the ones already queued)."""
        self._replies.extend(replies)

    def structured(
        self,
        prompt: str,
        schema: type[ModelT],
        *,
        system: str | None = None,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> LLMResult[ModelT]:
        """Return the next scripted reply, validated against ``schema``."""
        self.calls.append(
            FakeCall(
                prompt=prompt,
                system=system,
                schema_name=schema.__name__,
                prompt_version=prompt_version,
            )
        )
        if not self._replies:
            raise AssertionError(
                f"FakeLLM has no reply left for {schema.__name__} (call {len(self.calls)})"
            )
        item = self._replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item(prompt)
        return LLMResult(
            value=_as_schema(item, schema),
            attempts=(Attempt(number=1, outcome="ok"),),
            model=self._model,
            prompt_version=prompt_version,
            cached=False,
            fetched_at=self._now(),
        )

    def __repr__(self) -> str:
        return (
            f"FakeLLM(model={self._model!r}, queued={len(self._replies)}, calls={len(self.calls)})"
        )


def _as_schema(item: Any, schema: type[ModelT]) -> ModelT:
    if isinstance(item, schema):
        return item
    if isinstance(item, dict):
        return schema.model_validate(item)
    if isinstance(item, str):
        return schema.model_validate_json(item)
    raise AssertionError(
        f"FakeLLM was scripted with a {type(item).__name__}, which is not a {schema.__name__}"
    )
