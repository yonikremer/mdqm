"""The adapter seam: the only thing the core knows about a source.

``docs/DESIGN.md`` "Adapter interface" is the contract. An adapter is a *plugin*: core code talks
to :class:`SourceAdapter` and never to a vendor driver. Because the protocol is structural, an
adapter class needs no base class from us; the optional methods are optional in practice (the
registry requires only the five below) and an unsupported one must raise :class:`NotSupported`
naming the capability, which is how the probe learns to report it as unsupported and the feature
falls back.

Adapters are synchronous on purpose: the MCP server runs them in a bounded thread pool behind the
governor, so adapters stay simple blocking code. An adapter never substitutes ``now()``: the
fragment it receives already carries the single ``exec_ts`` of the execution.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from mdqm.config.secrets import SecretProvider
from mdqm.config.settings import SourceConfig
from mdqm.models import (
    Capabilities,
    CatalogSnapshot,
    Fragment,
    Limits,
    Partitioning,
    PlanSummary,
    RawQuery,
    TableStats,
)

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = [
    "OPTIONAL_METHODS",
    "REQUIRED_METHODS",
    "AdapterFactory",
    "ArrowStream",
    "NotSupported",
    "SourceAdapter",
    "supports_method",
]

#: Methods the registry demands of every adapter (DESIGN: required).
REQUIRED_METHODS: tuple[str, ...] = ("dialect", "probe", "extract_catalog", "execute", "cancel")

#: Optional method -> the capability name it provides. One table, so the probe and the conformance
#: kit can never disagree about which capability an unimplemented method means.
OPTIONAL_METHODS: dict[str, str] = {
    "extract_partitions": "partition_metadata",
    "explain": "explain_plan",
    "mine_queries": "query_store",
    "mutability_stats": "mutability_counters",
}


class NotSupported(Exception):
    """An optional capability this source cannot provide.

    ``capability`` is a key of :data:`OPTIONAL_METHODS` values, so the caller can record the
    verdict in :class:`~mdqm.models.Capabilities` and run the declared fallback instead. It is a
    normal answer, not a failure: it must never be logged as an error.
    """

    def __init__(self, capability: str, detail: str | None = None) -> None:
        self.capability = capability
        self.detail = detail
        super().__init__(f"{capability}: {detail}" if detail else capability)


class ArrowStream(Protocol):
    """What :meth:`SourceAdapter.execute` returns: batches, read lazily.

    Lazy on purpose: the governor stops at ``max_rows``/``max_bytes`` without fetching the rest,
    and ``cancel`` has something to interrupt while a long query is still returning batches.
    """

    #: Arrow schema of the batches, known before the first batch arrives.
    schema: pa.Schema

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        """Yield ``pyarrow.RecordBatch`` values until exhausted."""
        ...

    def close(self) -> None:
        """Release the cursor/connection. Idempotent."""
        ...


class SourceAdapter(Protocol):
    """One source, in the vocabulary of ``docs/DESIGN.md`` "Adapter interface"."""

    #: Adapter kind, also the entry-point name and ``SourceConfig.adapter`` (``"postgresql"``).
    kind: str

    def dialect(self) -> str:
        """sqlglot dialect name for this source, or ``"dsl"`` for Elasticsearch and APIs."""
        ...

    def probe(self) -> Capabilities:
        """Detect what this source supports. Optional features that raise :class:`NotSupported`
        must come back as unsupported here (never as supported)."""
        ...

    def extract_catalog(self) -> CatalogSnapshot:
        """Read tables, columns, indexes and keys. Never empty for a healthy source."""
        ...

    def execute(self, fragment: Fragment, limits: Limits) -> ArrowStream:
        """Run one already-compiled fragment, respecting ``limits`` (rows, bytes, timeout)."""
        ...

    def cancel(self, handle: ArrowStream) -> None:
        """Abort the query behind ``handle`` (the stream ``execute`` returned). Idempotent."""
        ...

    def extract_partitions(self) -> list[Partitioning]:
        """Partition metadata per table, or raise :class:`NotSupported`."""
        ...

    def explain(self, fragment: Fragment) -> PlanSummary:
        """The source's own plan for ``fragment``, or raise :class:`NotSupported`."""
        ...

    def mine_queries(self, since: datetime) -> Iterator[RawQuery]:
        """Past queries from the source's query store, or raise :class:`NotSupported`."""
        ...

    def mutability_stats(self) -> list[TableStats]:
        """Change counters per table, or raise :class:`NotSupported`."""
        ...


#: How the registry builds an adapter: it hands over the source config and the secret provider
#: (``None`` when no provider is configured). Connections are the adapter's business; it may be
#: lazy, and it must never log or embed a secret value.
AdapterFactory = Callable[[SourceConfig, SecretProvider | None], SourceAdapter]


def supports_method(candidate: object, method: str) -> bool:
    """True when ``candidate`` defines ``method``.

    Optional methods may simply be absent (``getattr`` would raise ``AttributeError`` mid-flow),
    so every caller asks this first instead of assuming the whole protocol is implemented.
    """
    return callable(getattr(candidate, method, None))
