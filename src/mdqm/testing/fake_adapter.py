"""A real :class:`~mdqm.adapters.base.SourceAdapter` backed by in-process DuckDB (task F05).

Other lanes need to exercise execution, planning and the governor **without Postgres, Oracle,
SQL Server or Elasticsearch** — this machine often has no container runtime at all. So the fake is
not a stub with a side channel: it holds the rows in a real DuckDB database, runs the fragment SQL
there, streams ``pyarrow.RecordBatch`` values, and enforces ``Limits`` itself. It passes
:func:`~mdqm.testing.conformance.run_conformance`, which is the only definition of "a real
adapter" this project has.

Everything optional is *declared*, never assumed: construct it with partition metadata, a fake
``explain`` or a query history and :meth:`FakeAdapter.probe` reports those capabilities; omit them
and the same methods raise :class:`~mdqm.adapters.base.NotSupported`, which is what the probe then
reports. Nothing here writes to a source, so the read-only principle is trivially kept.

Registration: the fake is **not** declared as an ``mdqm.adapters`` entry point. A test registers it
explicitly (``registry.register("fake", FakeAdapter)``) or builds it directly. Auto-registering it
would let a production ``mdqm serve`` whose YAML names ``adapter: fake`` silently answer from
fabricated data — the failure mode principle 10 ("everything reported, nothing silent") exists to
prevent. See ``progress/F05.md`` for the full rationale.

Usage::

    world = build_world(WorldConfig.small())
    adapter = FakeAdapter.for_world(world, name="crm")
    run_conformance(lambda: FakeAdapter.for_world(world, name="crm"))
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import duckdb
import pyarrow as pa

from mdqm.adapters.base import ArrowStream, NotSupported
from mdqm.config.secrets import SecretProvider
from mdqm.config.settings import SourceConfig
from mdqm.models import (
    Capabilities,
    Capability,
    CatalogSnapshot,
    Column,
    Fragment,
    Limits,
    NormalizedType,
    Partitioning,
    PlanSummary,
    PruningProof,
    RawQuery,
    Table,
    TableKind,
    TableStats,
)

if TYPE_CHECKING:
    from mdqm.testing.world import World

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MEMORY_LIMIT",
    "FakeAdapter",
    "FakeAdapterTimeout",
    "FakeStream",
    "table_from_rows",
]

#: Batch size the stream asks DuckDB for. Small on purpose: a row/byte cap can then stop mid-scan
#: instead of after one huge batch.
DEFAULT_BATCH_SIZE: Final = 1024

#: Where DuckDB spills when a query outgrows memory. Its own spilling: no extension, no network.
DEFAULT_MEMORY_LIMIT: Final = "512MB"

#: How often a simulated delay re-checks the deadline, so a timeout is honoured while waiting
#: rather than reported afterwards.
_POLL_SLICE_S: Final = 0.005


class FakeAdapterTimeout(TimeoutError):
    """The simulated deadline passed. A ``TimeoutError``, never a ``NotSupported``."""

    _closed: bool


def table_from_rows(rows: Sequence[Any]) -> pa.Table:
    """Build an Arrow table from frozen dataclass rows (the ``World`` datasets' row types).

    Keeps :class:`~mdqm.testing.world.World` as plain Python dataclasses: the loader converts at
    the edge instead of the generator importing Arrow.
    """
    return pa.Table.from_pylist([asdict(row) for row in rows])


def _normalized_type(pa_type: pa.DataType) -> NormalizedType:
    """Map an Arrow type onto the project's small portable vocabulary."""
    if pa.types.is_boolean(pa_type):
        return NormalizedType.BOOLEAN
    if pa.types.is_integer(pa_type):
        return NormalizedType.INTEGER
    if pa.types.is_floating(pa_type):
        return NormalizedType.FLOAT
    if pa.types.is_decimal(pa_type):
        return NormalizedType.DECIMAL
    if pa.types.is_timestamp(pa_type):
        return NormalizedType.TIMESTAMP
    if pa.types.is_date(pa_type):
        return NormalizedType.DATE
    if pa.types.is_time(pa_type):
        return NormalizedType.TIME
    if pa.types.is_binary(pa_type) or pa.types.is_large_binary(pa_type):
        return NormalizedType.BINARY
    if pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
        return NormalizedType.STRING
    if pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type):
        return NormalizedType.ARRAY
    if pa.types.is_struct(pa_type) or pa.types.is_map(pa_type):
        return NormalizedType.JSON
    return NormalizedType.OTHER


class FakeStream:
    """The ``ArrowStream`` a fake ``execute`` returns: lazy batches, caps and deadline enforced.

    Lazy because ``Limits`` is the adapter's promise, not the caller's: the stream stops at
    ``max_rows``/``max_bytes`` and never fetches the rest, and :meth:`FakeAdapter.cancel` has a
    live reader to interrupt.
    """

    _closed: bool

    def __init__(
        self,
        reader: pa.RecordBatchReader,
        *,
        limits: Limits,
        deadline: float,
        latency_s: float,
        clock: Callable[[], float],
    ) -> None:
        self.schema: pa.Schema = reader.schema
        self._reader = reader
        self._max_rows = limits.max_rows
        self._max_bytes = limits.max_bytes
        self._deadline = deadline
        self._latency_s = latency_s
        self._clock = clock
        self._timeout_s = limits.timeout_s
        self._closed: bool = False
        self.cancelled = False
        self.rows_returned = 0
        self.bytes_returned = 0

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        """Yield batches until the source, a cap or a cancellation stops the scan."""
        if self.cancelled:
            return
        for batch in self._reader:
            if self.cancelled:
                return
            self._wait()
            trimmed = self._trim(batch)
            if trimmed.num_rows == 0:
                return
            self.rows_returned += trimmed.num_rows
            self.bytes_returned += trimmed.nbytes
            yield trimmed
            if self.rows_returned >= self._max_rows or self.bytes_returned >= self._max_bytes:
                return

    def _trim(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        """Cut the last batch at the row cap, so the cap is exact rather than batch-rounded."""
        remaining = self._max_rows - self.rows_returned
        if batch.num_rows > remaining:
            return batch.slice(0, remaining)
        return batch

    def _wait(self) -> None:
        """Spend the simulated per-batch latency, raising the moment the deadline passes."""
        remaining = self._latency_s
        while remaining > 0:
            self._check_deadline()
            step = min(_POLL_SLICE_S, remaining)
            time.sleep(step)
            remaining -= step
        self._check_deadline()

    def _check_deadline(self) -> None:
        if self._clock() > self._deadline:
            raise FakeAdapterTimeout(f"simulated query exceeded the {self._timeout_s}s timeout")

    def cancel(self) -> None:
        """Stop the scan. Idempotent: a second call does nothing."""
        self.cancelled = True
        self.close()

    def close(self) -> None:
        """Release the DuckDB reader. Idempotent."""
        if not self._closed:
            self._closed = True
            self._reader.close()


class FakeAdapter:
    """A configurable, in-memory ``SourceAdapter`` over DuckDB (dialect ``duckdb``).

    ``tables`` maps a table name to an Arrow table (use :func:`table_from_rows` or
    :meth:`for_world`); ``partitioning`` declares partition metadata per table; ``query_history``,
    ``explain_enabled`` and ``mutability`` declare the optional capabilities. Call counters,
    latency and injected failures make the governor, the timeout path and the retry path testable
    without a real database.

    The default ``tables`` holds one generic one-row table so a bare instance is still a valid
    source for :func:`~mdqm.testing.conformance.run_conformance`; load real rows with
    :meth:`for_world`.
    """

    kind: Final[str] = "fake"

    def __init__(
        self,
        config: SourceConfig | None = None,
        secrets: SecretProvider | None = None,
        *,
        name: str | None = None,
        tables: Mapping[str, pa.Table] | None = None,
        partitioning: Mapping[str, Partitioning] | None = None,
        explain_enabled: bool = True,
        query_history: Sequence[RawQuery] | None = None,
        mutability: Sequence[TableStats] | None = None,
        extra_capabilities: Mapping[str, Capability] | None = None,
        latency_s: float = 0.0,
        fail_with: Exception | None = None,
        fail_after_calls: int | None = None,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        spill_dir: Path | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        """Build a ready adapter. Every argument has a safe default, so a bare instance works."""
        if latency_s < 0:
            raise ValueError(f"latency_s must be >= 0, got {latency_s}")
        self._name = name or (config.name if config is not None else self.kind)
        self._secrets = secrets
        if config is not None and config.adapter != self.kind:
            # The registry already proves this; doing it here keeps direct construction honest.
            raise ValueError(
                f"config names adapter {config.adapter!r} but this is the {self.kind!r} adapter"
            )
        self._partitioning = dict(partitioning or {})
        self._explain_enabled = explain_enabled
        self._query_history = tuple(query_history or ())
        self._mutability = tuple(mutability or ())
        self._extra_capabilities = dict(extra_capabilities or {})
        self._latency_s = latency_s
        self._fail_with = fail_with
        self._fail_after_calls = fail_after_calls
        self._batch_size = batch_size
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic

        #: Every ``execute`` call, in order: the fragment's SQL. What "was the source called?" tests
        #: (cache/freshness/governor) assert on.
        self.calls: list[str] = []

        self._con = duckdb.connect(":memory:")
        # A source session is UTC: comparing a naive TIMESTAMP literal to a TIMESTAMPTZ column reads
        # the literal in the session zone, so the host's own zone would shift every window.
        self._con.execute("SET TimeZone='UTC'").fetchall()
        self._con.execute(f"SET memory_limit='{memory_limit}'")
        if spill_dir is not None:
            spill_dir.mkdir(parents=True, exist_ok=True)
            self._con.execute(f"SET temp_directory='{spill_dir.as_posix()}'")
        self._tables: dict[str, pa.Table] = {}
        source_tables = dict(tables) if tables is not None else _default_tables()
        for table_name, arrow_table in source_tables.items():
            self._load(table_name, arrow_table)

    # --- construction helpers -----------------------------------------------------------

    @classmethod
    def for_world(
        cls,
        world: World,
        *,
        name: str = "fake",
        partitioning: Mapping[str, Partitioning] | None = None,
        latency_s: float = 0.0,
    ) -> FakeAdapter:
        """Build an adapter holding one :class:`~mdqm.testing.world.World`'s three storage forms.

        Tables are ``customers_crm``, ``accounts``, ``usage_events``, ``accounts_hist`` and
        ``customers_erp``; the world's planted quirks (NULLs, orphans, duplicates) come along, so a
        test can assert on them without recreating them.
        """
        return cls(
            name=name,
            tables=world_tables(world),
            partitioning=partitioning,
            latency_s=latency_s,
        )

    def _load(self, table_name: str, arrow_table: pa.Table) -> None:
        """Create the DuckDB table from the Arrow data, so ``execute`` really scans it."""
        if not table_name or not table_name.replace("_", "").isalnum():
            raise ValueError(f"unsafe table name {table_name!r}")
        self._con.register("_mdqm_load", arrow_table)
        try:
            # table_name is checked to be [A-Za-z0-9_] above, so the only interpolation into this
            # DDL is that validated identifier; DuckDB cannot bind an identifier as a parameter.
            self._con.execute(
                f'CREATE OR REPLACE TABLE main."{table_name}" AS SELECT * FROM _mdqm_load'  # noqa: S608
            )
        finally:
            self._con.unregister("_mdqm_load")
        self._tables[table_name] = arrow_table

    def close(self) -> None:
        """Close the DuckDB database. Idempotent.

        The connection object is kept -- DuckDB's ``close`` tolerates a second call -- instead of
        being set to ``None``: nulling it forced an ``Optional`` annotation on every use, so each
        later call had to re-prove what the constructor already guarantees.
        """
        # No `_closed` flag: DuckDB's ``close`` is idempotent, and a boolean only the closer reads
        # was another piece of state for a type checker to argue about.
        with contextlib.suppress(Exception):
            self._con.close()

    # --- SourceAdapter: identity ---------------------------------------------------------

    def dialect(self) -> str:
        """``duckdb``: fragments are SQL this adapter really runs."""
        return "duckdb"

    @property
    def source(self) -> str:
        """The configured source name; the catalog and the probe both report it."""
        return self._name

    # --- SourceAdapter: probe ------------------------------------------------------------

    def probe(self) -> Capabilities:
        """Report exactly the optional capabilities this instance was given."""
        verdicts: dict[str, Capability] = {
            "partition_metadata": _verdict(bool(self._partitioning)),
            "explain_plan": _verdict(self._explain_enabled),
            "query_store": _verdict(bool(self._query_history)),
            "mutability_counters": _verdict(bool(self._mutability)),
        }
        verdicts.update(self._extra_capabilities)
        return Capabilities(
            source=self._name,
            adapter=self.kind,
            version=f"duckdb-{duckdb.__version__}",
            retrieved_at=self._clock(),
            capabilities=verdicts,
        )

    # --- SourceAdapter: catalog ----------------------------------------------------------

    def extract_catalog(self) -> CatalogSnapshot:
        """Tables and columns as they exist in DuckDB, with partition metadata folded in."""
        tables = tuple(
            Table(
                name=table_name,
                kind=(
                    TableKind.PARTITIONED_TABLE
                    if table_name in self._partitioning
                    else TableKind.TABLE
                ),
                columns=tuple(
                    Column(
                        name=field.name,
                        native_type=str(field.type),
                        normalized_type=_normalized_type(field.type),
                        nullable=field.nullable,
                        ordinal=position,
                    )
                    for position, field in enumerate(arrow.schema, start=1)
                ),
                row_estimate=arrow.num_rows,
            )
            for table_name, arrow in sorted(self._tables.items())
        )
        return CatalogSnapshot(source=self._name, captured_at=self._clock(), tables=tables)

    # --- SourceAdapter: execute / cancel -------------------------------------------------

    def execute(self, fragment: Fragment, limits: Limits) -> ArrowStream:
        """Run ``fragment.sql`` in DuckDB and stream the batches under ``limits``.

        The row and byte caps are enforced while streaming (never by the caller) and the timeout
        is checked while waiting, so a deadline is honoured rather than reported afterwards.
        """
        if fragment.sql is None:
            raise ValueError(
                f"the fake adapter runs SQL; fragment kind {fragment.kind!r} is not supported"
            )
        self._record_call(fragment)
        self._maybe_fail()
        deadline = self._monotonic() + limits.timeout_s
        reader = self._con.sql(fragment.sql).to_arrow_reader(self._batch_size)
        return FakeStream(
            reader,
            limits=limits,
            deadline=deadline,
            latency_s=self._latency_s,
            clock=self._monotonic,
        )

    def cancel(self, handle: ArrowStream) -> None:
        """Cancel the stream ``execute`` returned. Idempotent."""
        close = getattr(handle, "cancel", None)
        if callable(close):
            close()
        else:
            handle.close()

    def _record_call(self, fragment: Fragment) -> None:
        self.calls.append(fragment.sql or "")

    def _maybe_fail(self) -> None:
        """Raise the injected failure on the configured call number."""
        if self._fail_with is None:
            return
        if self._fail_after_calls is None or len(self.calls) >= self._fail_after_calls:
            raise self._fail_with

    # --- optional capabilities ------------------------------------------------------------

    def extract_partitions(self) -> list[Partitioning]:
        """Declared partition metadata, or :class:`NotSupported` when none was given."""
        if not self._partitioning:
            raise NotSupported("partition_metadata", "no partitioning was declared")
        return [self._partitioning[name] for name in sorted(self._partitioning)]

    def explain(self, fragment: Fragment) -> PlanSummary:
        """A plan derived from the declared partitions, or :class:`NotSupported` when disabled.

        The scan count comes from the declared bounds and whether the SQL names the partition key.
        That is a deliberate fake, not a planner: it proves the explain gate's control flow, never
        a real pruning decision.
        """
        if not self._explain_enabled:
            raise NotSupported("explain_plan", "explain was disabled at construction")
        target = self._explain_target(fragment)
        if target is None:
            return PlanSummary(
                pruning_proof=PruningProof.UNKNOWN,
                raw="FAKE-SCAN no declared partitions",
            )
        partitioning = self._partitioning[target]
        total = len(partitioning.bounds) or None
        # shortcut: key-name substring, not a real predicate parse — enough to drive the gate.
        mentions_key = any(key.lower() in (fragment.sql or "").lower() for key in partitioning.key)
        scanned = 1 if (mentions_key and total) else total
        proof = PruningProof.UNKNOWN
        if total is not None and total > 1:
            proof = PruningProof.PROVEN if scanned == 1 else PruningProof.UNPROVEN
        est_rows = None
        if scanned is not None and partitioning.bounds:
            est_rows = sum(b.row_estimate or 0 for b in partitioning.bounds[:scanned]) or None
        return PlanSummary(
            partitions_scanned=scanned,
            partitions_total=total,
            est_rows=est_rows,
            pruning_proof=proof,
            raw=f"FAKE-SCAN table={target} partitions={scanned}/{total}",
        )

    def _explain_target(self, fragment: Fragment) -> str | None:
        """Which declared table the fake plan describes: the fragment's table, else the only one."""
        for table_name in fragment.tables:
            if table_name in self._partitioning:
                return table_name
        if len(self._partitioning) == 1:
            return next(iter(self._partitioning))
        return None

    def mine_queries(self, since: datetime) -> Iterator[RawQuery]:
        """Declared query history newer than ``since``, or :class:`NotSupported`."""
        if not self._query_history:
            raise NotSupported("query_store", "no query history was declared")
        for query in self._query_history:
            if query.last_seen is None or query.last_seen >= since:
                yield query

    def mutability_stats(self) -> list[TableStats]:
        """Declared change counters, or :class:`NotSupported` when none was given."""
        if not self._mutability:
            raise NotSupported("mutability_counters", "no mutability stats were declared")
        return list(self._mutability)


def _verdict(supported: bool) -> Capability:
    """A probed verdict, so ``probe()`` never reports an optional feature it cannot serve."""
    return Capability(supported=supported)


def _default_tables() -> dict[str, pa.Table]:
    """One generic one-row table, so a bare fake is still a conforming source."""
    return {"mdqm_fake_probe": pa.table({"one": [1]})}


def world_tables(world: World) -> dict[str, pa.Table]:
    """The three storage forms of a :class:`~mdqm.testing.world.World` as Arrow tables."""
    return {
        "customers_crm": table_from_rows(world.crm.customers),
        "accounts": table_from_rows(world.billing.accounts),
        "usage_events": table_from_rows(world.billing.usage_events),
        "accounts_hist": table_from_rows(world.billing.accounts_hist),
        "customers_erp": table_from_rows(world.erp.customers),
    }
