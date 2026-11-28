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
from typing import TYPE_CHECKING, ClassVar, Protocol

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
    TableSize,
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
    "OptionalSourceMethods",
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
    "sizes": "size_stats",
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

    #: Rows and bytes this stream has produced so far. Universal: the governor's row and byte caps
    #: are enforced with them, so every implementation counts, whatever it reads from. The
    #: source-specific counters (a cache hit, an Elasticsearch index set) stay on their own class.
    rows_returned: int
    bytes_returned: int

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub; "..." is the unimplemented body
    def __iter__(self) -> Iterator[pa.RecordBatch]:
        """Yield ``pyarrow.RecordBatch`` values until exhausted."""
        ...

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub; "..." is the unimplemented body
    def close(self) -> None:
        """Release the cursor/connection. Idempotent."""
        ...


class OptionalSourceMethods(Protocol):
    """The four methods a source **may not implement at all**, documented rather than required.

    They are optional in practice, not just in theory: :data:`REQUIRED_METHODS` excludes them,
    :data:`OPTIONAL_METHODS` names them, the probe reports an absent one as an unsupported
    capability, and every caller asks :func:`supports_method` before reaching for one. Declaring them
    as required members of :class:`SourceAdapter` would make the type model contradict that contract
    and force a test double to implement four methods it exists to prove can be missing.
    """

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub; "..." is the unimplemented body
    def extract_partitions(self) -> list[Partitioning]:
        """Partition metadata per table, or raise :class:`NotSupported`."""
        ...

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub; "..." is the unimplemented body
    def explain(self, fragment: Fragment) -> PlanSummary:
        """The source's own plan for ``fragment``, or raise :class:`NotSupported`."""
        ...

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub; "..." is the unimplemented body
    def mine_queries(self, since: datetime) -> Iterator[RawQuery]:
        """Past queries from the source's query store, or raise :class:`NotSupported`."""
        ...

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub; "..." is the unimplemented body
    def mutability_stats(self) -> list[TableStats]:
        """Change counters per table, or raise :class:`NotSupported`."""
        ...

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub; "..." is the unimplemented body
    def sizes(self) -> list[TableSize]:
        """Rows and bytes per table and partition, or raise :class:`NotSupported`."""
        ...


class SourceAdapter(Protocol):
    """One source, in the vocabulary of ``docs/DESIGN.md`` "Adapter interface"."""

    #: Adapter kind, also the entry-point name and ``SourceConfig.adapter`` (``"postgresql"``).
    #: A ``ClassVar``: the kind is a property of the implementation, not something a caller may
    #: assign per instance. Declaring it mutable here made every adapter that (correctly) defined a
    #: ``ClassVar`` fail the protocol check.
    kind: ClassVar[str]

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub; "..." is the unimplemented body
    def dialect(self) -> str:
        """sqlglot dialect name for this source, or ``"dsl"`` for Elasticsearch and APIs."""
        ...

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub; "..." is the unimplemented body
    def probe(self) -> Capabilities:
        """Detect what this source supports. Optional features that raise :class:`NotSupported`
        must come back as unsupported here (never as supported)."""
        ...

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub; "..." is the unimplemented body
    def extract_catalog(self) -> CatalogSnapshot:
        """Read tables, columns, indexes and keys. Never empty for a healthy source."""
        ...

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub; "..." is the unimplemented body
    def execute(self, fragment: Fragment, limits: Limits) -> ArrowStream:
        """Run one already-compiled fragment, respecting ``limits`` (rows, bytes, timeout)."""
        ...

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub; "..." is the unimplemented body
    def cancel(self, handle: ArrowStream) -> None:
        """Abort the query behind ``handle`` (the stream ``execute`` returned). Idempotent."""
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
