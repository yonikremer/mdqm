"""The federated executor (task C17): fragments per source, one local join, one quality report.

:class:`~mdqm.executor.federate.FederatedExecutor` runs a
:class:`~mdqm.executor.models.FederatedPlan` of already-compiled fragments through the governor and
the fragment cache, lands their Arrow rows in its own DuckDB, joins and aggregates them there, and
returns the answer with a report that counts every row a join dropped or multiplied.
"""

from __future__ import annotations

from mdqm.executor.cache import FragmentCache, FragmentLookup, fragment_key
from mdqm.executor.federate import DEFAULT_MEMORY_LIMIT, FederatedExecutor, FederatedResult
from mdqm.executor.models import (
    SPEC_SAMPLE_KEYS,
    ColumnCheck,
    FederatedOutput,
    FederatedPlan,
    FragmentStep,
    GroupColumn,
    JoinReport,
    JoinStep,
    LocalAggregate,
    NullGroup,
    QualityReport,
    SanityViolation,
    StepCache,
    StepReport,
)

__all__ = [
    "DEFAULT_MEMORY_LIMIT",
    "SPEC_SAMPLE_KEYS",
    "ColumnCheck",
    "FederatedExecutor",
    "FederatedOutput",
    "FederatedPlan",
    "FederatedResult",
    "FragmentCache",
    "FragmentLookup",
    "FragmentStep",
    "GroupColumn",
    "JoinReport",
    "JoinStep",
    "LocalAggregate",
    "NullGroup",
    "QualityReport",
    "SanityViolation",
    "StepCache",
    "StepReport",
    "fragment_key",
]
