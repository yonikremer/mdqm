"""The governor: concurrency, quotas, timeouts and result handles (task C15).

One import surface for the layer that protects primary databases with no replicas
(``AGENTS.md`` principle 6). The pieces are small and separate on purpose -- admission, the circuit
breaker, the handle store, and the :class:`Governor` that composes them -- because each has one job
and its own tests.

Every refusal an agent can see is a :class:`GovernorRefusal` carrying a
:class:`~mdqm.models.ErrorResponse`; nothing here returns a traceback, and nothing is refused,
clamped or cancelled without being reported.
"""

from __future__ import annotations

from mdqm.governor.admission import (
    Admission,
    AdmissionPolicy,
    Lease,
    Peak,
    Quota,
    QuotaPolicy,
    QuotaState,
    Scope,
)
from mdqm.governor.breaker import BreakerPolicy, BreakerReport, BreakerState, CircuitBreaker
from mdqm.governor.clock import Clock, SystemClock
from mdqm.governor.core import ExecutionOutcome, Governance, Governor, GovernorReport
from mdqm.governor.handles import (
    DEFAULT_MAX_HANDLES,
    DEFAULT_TTL_S,
    HandleNotFound,
    Page,
    ResultHandle,
    ResultStore,
    StoreStats,
)
from mdqm.governor.refusal import GovernorRefusal

__all__ = [
    "DEFAULT_MAX_HANDLES",
    "DEFAULT_TTL_S",
    "Admission",
    "AdmissionPolicy",
    "BreakerPolicy",
    "BreakerReport",
    "BreakerState",
    "CircuitBreaker",
    "Clock",
    "ExecutionOutcome",
    "Governance",
    "Governor",
    "GovernorRefusal",
    "GovernorReport",
    "HandleNotFound",
    "Lease",
    "Page",
    "Peak",
    "Quota",
    "QuotaPolicy",
    "QuotaState",
    "ResultHandle",
    "ResultStore",
    "Scope",
    "StoreStats",
    "SystemClock",
]
