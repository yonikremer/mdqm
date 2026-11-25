"""Limits the governor enforces on one execution.

No defaults here on purpose: every number is an organisation's decision and comes from config, so
the core never hard-codes a timeout, a row cap or a partition cap.
"""

from __future__ import annotations

from pydantic import Field

from mdqm.models.base import MDQMModel


class Limits(MDQMModel):
    """Hard caps for a single fragment execution."""

    timeout_s: float = Field(gt=0, description="Wall-clock timeout in seconds.")
    max_rows: int = Field(gt=0, description="Row cap; the linter injects the same LIMIT.")
    max_bytes: int = Field(gt=0, description="Byte cap on the fetched result.")
    max_partitions: int = Field(gt=0, description="Partition scan cap for one table.")
