"""What a source's own plan says, which is the pruning proof the planner demands before running.

Without a readable plan the verdict is ``unknown``: the filter is injected anyway and the scan is
capped by statistics instead.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from mdqm.models.base import MDQMModel


class PruningProof(StrEnum):
    """Whether the plan *proves* that only the intended partitions are scanned."""

    PROVEN = "proven"
    UNPROVEN = "unproven"
    UNKNOWN = "unknown"


class PlanSummary(MDQMModel):
    """Plan-level facts extracted from ``EXPLAIN`` or an engine equivalent.

    ``raw`` holds the plan as text (JSON plans included): keeping it a string is what makes the
    summary comparable, hashable and JSON round-trippable across engines.
    """

    partitions_scanned: int | None = Field(default=None, ge=0)
    partitions_total: int | None = Field(default=None, ge=0)
    uses_index: bool | None = None
    indexes_used: tuple[str, ...] = ()
    est_rows: float | None = Field(default=None, ge=0)
    est_cost: float | None = Field(default=None, ge=0)
    pruning_proof: PruningProof = PruningProof.UNKNOWN
    raw: str | None = None

    @model_validator(mode="after")
    def _check_partition_counts(self) -> PlanSummary:
        scanned, total = self.partitions_scanned, self.partitions_total
        if scanned is not None and total is not None and scanned > total:
            raise ValueError(
                f"partitions_scanned ({scanned}) cannot exceed partitions_total ({total})."
            )
        return self
