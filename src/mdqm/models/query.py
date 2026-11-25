"""Past queries mined from a source's own query statistics.

Used to rank joins/examples and to raise the prior of conversion hypotheses. Mining never approves
anything: "common" is not "correct".
"""

from __future__ import annotations

from pydantic import Field

from mdqm.models.base import MDQMModel, UtcDatetime


class RawQuery(MDQMModel):
    """One mined query as the source reported it, before parsing or normalisation."""

    text: str = Field(min_length=1)
    dialect: str = Field(min_length=1)
    calls: int | None = Field(default=None, ge=0)
    mean_time_ms: float | None = Field(default=None, ge=0)
    last_seen: UtcDatetime | None = None
    user: str | None = None
