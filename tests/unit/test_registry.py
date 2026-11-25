"""Tests for the adapter seam (task F04): registry discovery, protocol validation, conformance kit.

Entry-point discovery is driven by ``importlib.metadata`` test doubles: no installation, no
network, no Docker. The fake adapters below are the reference a real adapter is compared against;
the broken ones prove the conformance kit actually fails instead of rubber-stamping.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime

import pyarrow as pa
import pytest

from mdqm.adapters.base import NotSupported, SourceAdapter, supports_method
from mdqm.adapters.registry import (
    ADAPTER_ENTRY_POINT_GROUP,
    AdapterNotRegistered,
    AdapterRegistrationError,
    AdapterRegistry,
)
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
    PartitionStrategy,
    PlanSummary,
    PruningProof,
    RawQuery,
    Table,
    TableStats,
)
from mdqm.testing.conformance import run_conformance

NOW = datetime(2026, 1, 1, tzinfo=UTC)
LIMITS = Limits(timeout_s=10.0, max_rows=5, max_bytes=1_000_000, max_partitions=4)


def config(kind: str = "fake", name: str = "src") -> SourceConfig:
    return SourceConfig(name=name, adapter=kind)


def fragment(adapter: SourceAdapter, sql: str = "SELECT 1 AS one") -> Fragment:
    return Fragment(source=adapter.kind, dialect=adapter.dialect(), sql=sql, exec_ts=NOW)


# --- fake adapters -----------------------------------------------------------------------


class Stream:
    """A minimal ArrowStream: what ``execute`` must return."""

    def __init__(self, table: pa.Table) -> None:
        self.schema = table.schema
        self._batches = table.to_batches()
        self.cancelled = False
        self.closed = False

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        for batch in self._batches:
            if self.cancelled:
                return
            yield batch

    def close(self) -> None:
        self.closed = True


class FakeAdapter:
    """A compliant in-memory adapter: same shape a real plugin must have."""

    kind = "fake"

    def __init__(
        self,
        source_config: SourceConfig,
        secrets: SecretProvider | None = None,
        *,
        rows: int = 3,
        explain_supported: bool = False,
    ) -> None:
        self.config = source_config
        self.secrets = secrets
        self.source = source_config.name
        self.rows = rows
        self.executions = 0
        self.cancelled: list[Stream] = []
        self._explain_supported = explain_supported

    def dialect(self) -> str:
        return "duckdb"

    def probe(self) -> Capabilities:
        return Capabilities(
            source=self.source,
            adapter=self.kind,
            retrieved_at=NOW,
            capabilities={
                "partition_metadata": Capability(supported=False),
                "explain_plan": Capability(supported=self._explain_supported),
                "query_store": Capability(supported=False),
                "mutability_counters": Capability(supported=False),
            },
        )

    def extract_catalog(self) -> CatalogSnapshot:
        return CatalogSnapshot(
            source=self.source,
            captured_at=NOW,
            tables=(
                Table(
                    name="t",
                    columns=(
                        Column(
                            name="one",
                            native_type="INTEGER",
                            normalized_type=NormalizedType.INTEGER,
                        ),
                    ),
                ),
            ),
        )

    def execute(self, query: Fragment, limits: Limits) -> Stream:
        self.executions += 1
        if "-- slow" in (query.sql or ""):
            deadline = time.monotonic() + limits.timeout_s
            while time.monotonic() < deadline:
                time.sleep(0.005)
            raise TimeoutError(f"query exceeded timeout_s={limits.timeout_s}")
        count = min(self.rows, limits.max_rows)
        return Stream(pa.table({"one": list(range(count))}))

    def cancel(self, handle: Stream) -> None:
        handle.cancelled = True
        self.cancelled.append(handle)

    def explain(self, query: Fragment) -> PlanSummary:
        if not self._explain_supported:
            raise NotSupported("explain_plan", "no access to plans on this account")
        return PlanSummary(pruning_proof=PruningProof.PROVEN)


class RowCapIgnoringAdapter(FakeAdapter):
    """Returns every row, ignoring ``Limits.max_rows``."""

    def execute(self, query: Fragment, limits: Limits) -> Stream:
        return Stream(pa.table({"one": list(range(limits.max_rows + 3))}))


class GarbageOptionalAdapter(FakeAdapter):
    """``mutability_stats`` returns a dict instead of ``TableStats`` models."""

    def mutability_stats(self) -> list[TableStats]:  # type: ignore[return-value]
        return [{"rows_inserted": 1}]  # type: ignore[list-item]


class CancelRaisingAdapter(FakeAdapter):
    def cancel(self, handle: Stream) -> None:
        raise RuntimeError("driver cannot cancel")


class TimeoutIgnoringAdapter(FakeAdapter):
    """Finishes a slow query instead of raising at the deadline."""

    def execute(self, query: Fragment, limits: Limits) -> Stream:
        if "-- slow" in (query.sql or ""):
            return Stream(pa.table({"one": [1]}))
        return super().execute(query, limits)


class EmptyCatalogAdapter(FakeAdapter):
    def extract_catalog(self) -> CatalogSnapshot:
        return CatalogSnapshot(source=self.source, captured_at=NOW)


class LyingProbeAdapter(FakeAdapter):
    """Claims a capability it does not implement."""

    def probe(self) -> Capabilities:
        base = super().probe()
        return base.model_copy(
            update={
                "capabilities": {
                    **base.capabilities,
                    "query_store": Capability(supported=True),
                }
            }
        )


class UnsupportedPartitionsAdapter(FakeAdapter):
    """Implements the optional methods, honestly reporting one as unsupported."""

    def mutability_stats(self) -> list[TableStats]:
        return [TableStats(source=self.source, table="t", captured_at=NOW, rows_inserted=2)]

    def mine_queries(self, since: datetime) -> Iterator[RawQuery]:
        return iter([RawQuery(text="select 1", dialect="duckdb")])

    def probe(self) -> Capabilities:
        base = super().probe()
        return base.model_copy(
            update={
                "capabilities": {
                    **base.capabilities,
                    "mutability_counters": Capability(supported=True),
                }
            }
        )


# --- entry-point doubles -----------------------------------------------------------------


class EntryPoint:
    """Stands in for ``importlib.metadata.EntryPoint`` (same two members the registry uses)."""

    def __init__(self, name: str, target: object = None, error: Exception | None = None) -> None:
        self.name = name
        self._target = target
        self._error = error

    def load(self) -> object:
        if self._error is not None:
            raise self._error
        return self._target


def registry_of(
    *entry_points: EntryPoint, secrets: SecretProvider | None = None
) -> AdapterRegistry:
    return AdapterRegistry(secrets=secrets, entry_points=lambda: list(entry_points))


# --- discovery ---------------------------------------------------------------------------

#: Discovery must use this group, and a plugin package declares it; the fake EP names it too.


def test_discovery_uses_the_mdq_adapters_entry_point_group() -> None:
    assert ADAPTER_ENTRY_POINT_GROUP == "mdqm.adapters"
    # No adapter distribution is installed for mdqm itself, so real discovery is empty but works.
    assert AdapterRegistry().discover() == ()


def test_discover_registers_entry_points() -> None:
    class SecondAdapter(FakeAdapter):
        kind = "second"

    registry = registry_of(
        EntryPoint("fake", FakeAdapter),
        EntryPoint("second", SecondAdapter),
    )
    assert registry.discover() == ("fake", "second")
    adapter = registry.create(config("second"))
    assert isinstance(adapter, SecondAdapter)
    assert adapter.config.adapter == "second"


def test_discover_is_idempotent() -> None:
    registry = registry_of(EntryPoint("fake", FakeAdapter))
    assert registry.discover() == ("fake",)
    assert registry.discover() == ("fake",)


def test_discover_reports_a_broken_entry_point() -> None:
    registry = registry_of(EntryPoint("broken", error=ImportError("no driver installed")))
    with pytest.raises(AdapterRegistrationError) as caught:
        registry.discover()
    message = str(caught.value)
    assert "mdqm.adapters:broken" in message
    assert "ImportError" in message


def test_discover_rejects_an_entry_point_that_is_not_an_adapter() -> None:
    class NotAnAdapter:
        def probe(self) -> None: ...

    registry = registry_of(EntryPoint("fake", NotAnAdapter))
    with pytest.raises(AdapterRegistrationError) as caught:
        registry.discover()
    assert "mdqm.adapters:fake" in str(caught.value)


# --- registration validates the protocol -------------------------------------------------

_REQUIRED_IMPLS: dict[str, object] = {
    "kind": "fake",
    "dialect": FakeAdapter.dialect,
    "probe": FakeAdapter.probe,
    "extract_catalog": FakeAdapter.extract_catalog,
    "execute": FakeAdapter.execute,
    "cancel": FakeAdapter.cancel,
}


def adapter_class_missing(method: str) -> type:
    namespace = dict(_REQUIRED_IMPLS)
    namespace.pop(method)
    return type("PartialAdapter", (), namespace)


@pytest.mark.parametrize("method", ["dialect", "probe", "extract_catalog", "execute", "cancel"])
def test_registering_an_adapter_without_a_required_method_fails(method: str) -> None:
    registry = AdapterRegistry()
    with pytest.raises(AdapterRegistrationError) as caught:
        registry.register("fake", adapter_class_missing(method))  # type: ignore[arg-type]
    message = str(caught.value)
    assert method in message
    assert "SourceAdapter protocol" in message
    assert registry.names() == ()


def test_the_registration_error_names_every_missing_method() -> None:
    registry = AdapterRegistry()
    with pytest.raises(AdapterRegistrationError) as caught:
        registry.register("fake", type("Empty", (), {}))  # type: ignore[arg-type]
    message = str(caught.value)
    for method in ("dialect", "probe", "extract_catalog", "execute", "cancel"):
        assert method in message


def test_duplicate_registration_is_refused() -> None:
    registry = AdapterRegistry()
    registry.register("fake", FakeAdapter)
    with pytest.raises(AdapterRegistrationError, match="already registered"):
        registry.register("fake", FakeAdapter)


@pytest.mark.parametrize("kind", ["", "   "])
def test_an_empty_adapter_kind_is_refused(kind: str) -> None:
    with pytest.raises(AdapterRegistrationError, match="non-empty"):
        AdapterRegistry().register(kind, FakeAdapter)


def test_an_unknown_kind_lists_what_is_registered() -> None:
    registry = AdapterRegistry()
    registry.register("fake", FakeAdapter)
    with pytest.raises(AdapterNotRegistered) as caught:
        registry.factory("postgresql")
    message = str(caught.value)
    assert "postgresql" in message
    assert "fake" in message
    assert ADAPTER_ENTRY_POINT_GROUP in message


# --- creation ----------------------------------------------------------------------------


def test_create_passes_config_and_secret_provider_to_the_factory() -> None:
    secrets = _NullSecrets()
    registry = AdapterRegistry(secrets=secrets)
    registry.register("fake", FakeAdapter)
    source_config = config()
    adapter = registry.create(source_config)
    assert isinstance(adapter, FakeAdapter)
    assert adapter.config is source_config
    assert adapter.secrets is secrets


def test_create_validates_a_factory_function() -> None:
    registry = AdapterRegistry()
    registry.register("fake", lambda cfg, sec: object())  # type: ignore[arg-type,return-value]
    with pytest.raises(AdapterRegistrationError, match="execute"):
        registry.create(config())


def test_create_rejects_a_kind_mismatch() -> None:
    class WrongKind(FakeAdapter):
        kind = "other"

    registry = AdapterRegistry()
    registry.register("fake", WrongKind)
    with pytest.raises(AdapterRegistrationError, match="kind 'other'"):
        registry.create(config())


class _NullSecrets:
    def get(self, name: str) -> str:
        raise AssertionError("no secret is needed in these tests")


# --- NotSupported <-> capability ---------------------------------------------------------


def test_unimplemented_optional_method_raises_not_supported() -> None:
    adapter = FakeAdapter(config())
    with pytest.raises(NotSupported) as caught:
        adapter.explain(fragment(adapter))
    assert caught.value.capability == "explain_plan"
    assert adapter.probe().supports("explain_plan") is False


def test_absent_optional_methods_are_not_supported_by_the_probe() -> None:
    adapter = FakeAdapter(config())
    capabilities = adapter.probe()
    for method, capability in (
        ("extract_partitions", "partition_metadata"),
        ("mine_queries", "query_store"),
        ("mutability_stats", "mutability_counters"),
    ):
        assert not supports_method(adapter, method)
        assert capabilities.supports(capability) is False
    assert capabilities.unsupported() == (
        "explain_plan",
        "mutability_counters",
        "partition_metadata",
        "query_store",
    )


def test_not_supported_is_a_normal_answer_not_an_error() -> None:
    error = NotSupported("query_store", "V$SQL is not readable")
    assert error.capability == "query_store"
    assert "V$SQL" in str(error)
    assert str(NotSupported("sampling")) == "sampling"


def test_dialect_and_kind_are_part_of_the_contract() -> None:
    adapter = FakeAdapter(config())
    assert adapter.kind == "fake"
    assert adapter.dialect() == "duckdb"


# --- conformance kit ---------------------------------------------------------------------


def test_conformance_kit_passes_a_compliant_adapter() -> None:
    run_conformance(lambda: FakeAdapter(config()))


def test_conformance_kit_passes_with_explicit_fragment_and_slow_query() -> None:
    adapter = FakeAdapter(config())
    run_conformance(
        lambda: FakeAdapter(config()),
        fragment=fragment(adapter),
        slow_fragment=fragment(adapter, "SELECT 1 -- slow"),
        limits=LIMITS,
    )


def test_conformance_kit_passes_optional_methods_that_work() -> None:
    run_conformance(lambda: FakeAdapter(config(), explain_supported=True))
    run_conformance(lambda: UnsupportedPartitionsAdapter(config()))


def test_conformance_accepts_a_partitioning_result() -> None:
    class PartitionedAdapter(FakeAdapter):
        def extract_partitions(self) -> list[Partitioning]:
            return [Partitioning(strategy=PartitionStrategy.HASH, key=("one",))]

        def probe(self) -> Capabilities:
            base = super().probe()
            return base.model_copy(
                update={
                    "capabilities": {
                        **base.capabilities,
                        "partition_metadata": Capability(supported=True),
                    }
                }
            )

    run_conformance(lambda: PartitionedAdapter(config()))


BROKEN_CASES: tuple[tuple[str, Callable[[], SourceAdapter], str], ...] = (
    ("row cap ignored", lambda: RowCapIgnoringAdapter(config()), "max_rows"),
    ("garbage optional result", lambda: GarbageOptionalAdapter(config()), "TableStats"),
    ("cancel raises", lambda: CancelRaisingAdapter(config()), "cancel() raised RuntimeError"),
    ("empty catalog", lambda: EmptyCatalogAdapter(config()), "no tables"),
    ("probe over-claims", lambda: LyingProbeAdapter(config()), "has no"),
)


@pytest.mark.parametrize(("label", "factory", "expected"), BROKEN_CASES)
def test_conformance_kit_fails_a_broken_adapter(
    label: str, factory: Callable[[], SourceAdapter], expected: str
) -> None:
    with pytest.raises(AssertionError) as caught:
        run_conformance(factory)
    message = str(caught.value)
    assert "adapter conformance:" in message, label
    assert expected in message, label


def test_conformance_kit_fails_an_adapter_that_ignores_the_timeout() -> None:
    adapter = FakeAdapter(config())
    with pytest.raises(AssertionError, match="ignored timeout_s"):
        run_conformance(
            lambda: TimeoutIgnoringAdapter(config()),
            slow_fragment=fragment(adapter, "SELECT 1 -- slow"),
            limits=LIMITS,
        )


def test_conformance_kit_requires_a_fragment_for_non_sql_sources() -> None:
    class DslAdapter(FakeAdapter):
        def dialect(self) -> str:
            return "dsl"

    with pytest.raises(AssertionError, match="pass fragment="):
        run_conformance(lambda: DslAdapter(config()))


def test_conformance_kit_flags_a_probe_for_another_adapter() -> None:
    class MisnamedAdapter(FakeAdapter):
        def probe(self) -> Capabilities:
            base = super().probe()
            return base.model_copy(update={"adapter": "duckdb"})

    with pytest.raises(AssertionError, match="adapter's kind"):
        run_conformance(lambda: MisnamedAdapter(config()))


def test_conformance_kit_flags_a_missing_not_supported_declaration() -> None:
    class LyingExplainAdapter(FakeAdapter):
        """explain() raises NotSupported while the probe claims explain_plan."""

        def probe(self) -> Capabilities:
            base = super().probe()
            return base.model_copy(
                update={
                    "capabilities": {
                        **base.capabilities,
                        "explain_plan": Capability(supported=True),
                    }
                }
            )

    with pytest.raises(AssertionError, match="must agree"):
        run_conformance(lambda: LyingExplainAdapter(config()))
