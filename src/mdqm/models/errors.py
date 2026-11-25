"""The error an agent sees. Always ``code`` + ``problem`` + a concrete ``fix``, never a stack trace.

Shape is fixed by docs/DESIGN.md "Error shape"; optional fields are omitted when unset so the agent
never chews on ``null`` noise.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from mdqm.models.base import MDQMModel


class ErrorResponse(MDQMModel):
    """A blockable/repairable failure, addressed to a weak agent."""

    status: Literal["error"] = "error"
    code: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    fix: str = Field(min_length=1)
    partitions_if_fixed: str | None = Field(
        default=None, description="Scan estimate after the fix, e.g. '2 of 60'."
    )
    valid_values: tuple[str, ...] | None = None
    did_you_mean: tuple[str, ...] | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Dump without unset optional fields: the agent-facing shape carries no nulls."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        """Dump without unset optional fields (see :meth:`model_dump`)."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(**kwargs)
