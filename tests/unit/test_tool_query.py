"""M03: the ``query`` tool body -- lint, compile, plan, cache, execute, summarise, record.

Five claims, each tested as a behaviour rather than against a helper name, and all five with one
hard assertion in common: **what reached the source**. A fake adapter's ``calls`` list is the
evidence, because "we refused before the source" and "we refused after asking it" read the same in
an error message and differ by a database's worth of load.

* a valid spec answers with a summary, a sample and a handle, and writes one history row and one
  audit row (with the semantic version, so an answer is attributable to the rules that made it);
* an unknown column comes back as the linter's own ``UNKNOWN_COLUMN`` -- with its ``did_you_mean``
  and its ``fix`` -- and costs zero source calls;
* a spec over the partitioned table with no time window is refused with ``PARTITION_SCAN_ALL``,
  ``partitions_if_fixed`` and a ``fix``, and the adapter is never called;
* the same spec twice answers from the cache the second time: zero new calls, ``cache_hit`` true,
  the first call's ``fetched_at`` reported, and ``fresh=true`` forces the call;
* a ``DELETE`` sent as raw SQL is refused with ``WRITE_BLOCKED`` and lands in the audit chain, which
  then still verifies.

The pipeline is driven through the tool function itself with the SDK's auth context bound (the same
pattern ``tests/unit/test_tools_misc.py`` uses): one process, no transport, no Docker. The three
sources are F05 fake adapters, each a real in-process DuckDB, and they hold the D02 world's three
storage forms, so the compile-plan-execute path is the real one.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import duckdb
import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

from mdqm.errors import ErrorCode
from mdqm.executor.cache import FragmentCache
from mdqm.governor import Governor
from mdqm.models import Limits
from mdqm.semantic.loader import load
from mdqm.semantic.models import SemanticLayer
from mdqm.server.auth import READ_SCOPE
from mdqm.server.tools import (
    QueryAnswer,
    QueryServices,
    configure_query_services,
    query,
    reset_query_services,
)
from mdqm.spec.models import QuerySpec
from mdqm.store import Store
from mdqm.store.audit import AuditLog, AuditOutcome
from mdqm.store.history import HistoryStore
from mdqm.testing.fake_adapter import FakeAdapter, table_from_rows
from mdqm.testing.world import World, WorldConfig, build_world

#: One execution timestamp for the whole module (C02: one value per execution).
EXEC_TS: Final = datetime(2026, 11, 25, 15, 0, tzinfo=UTC)

#: Caps the tests explicitly expect: only the partition cap can trip, and it must not.
LIMITS: Final = Limits(timeout_s=30.0, max_rows=50_000, max_bytes=64 << 20, max_partitions=64)

#: The layer these tests run under. It declares what the pipeline reads -- measures across three
#: storage forms, the customer-status value map, the monthly partitions of ``usage_events`` and one
#: approved mutability class per table -- and nothing else, so a refusal here is a refusal about the
#: thing the test names.
LAYER_YAML: Final = """
version: 1

sources:
  crm: { adapter: postgresql, secret: crm_ro }
  billing: { adapter: postgresql, secret: billing_ro }
  erp: { adapter: postgresql, secret: erp_ro }

value_maps:
  customer_status:
    canonical: [active, churned, suspended]
    synonyms:
      active: ["פעיל", "active"]
      churned: ["עזב", "churned"]
      suspended: ["מושעה", "suspended"]
    sources:
      crm.customers.status: { A: active, C: churned, S: suspended }
      billing.accounts.state: { Active: active, Closed: churned, Hold: suspended }
      erp.customers.status_code: { 1: active, 2: churned, 3: suspended }

measures:
  credit_limit_r5000:
    canonical: { rounding: { to: 5000, mode: half_up } }
    sources:
      crm.customers.credit_limit: { normalize: "{col}" }
      billing.accounts.credit_limit: { normalize: "ROUND({col} / 5000) * 5000" }
      erp.customers.limit_k: { normalize: "{col} * 10" }
  usage_units:
    sources:
      billing.usage_events.units: { normalize: "{col}" }

tables:
  crm.customers:
    key: [cust_id]
    mutability: { class: append_rarely, status: approved }
  billing.accounts:
    key: [client_no]
    mutability: { class: append_rarely, status: approved }
  erp.customers:
    key: [cust_no]
    mutability: { class: append_rarely, status: approved }
  billing.usage_events:
    key: [event_id]
    default_window: 7d
    partitioning:
      key: start_time
      granularity: month
      require_key_filter: true
      bounds:
        - { name: p2026_09, lower: 2026-09-01T00:00:00Z, upper: 2026-10-01T00:00:00Z }
        - { name: p2026_10, lower: 2026-10-01T00:00:00Z, upper: 2026-11-01T00:00:00Z }
        - { name: p2026_11, lower: 2026-11-01T00:00:00Z, upper: 2026-12-01T00:00:00Z }
    time_columns:
      start_time: { role: event_time, indexed: true, use_by_default: true }
      insertion_time:
        {
          role: ingestion_time,
          derives_bound_on: start_time,
          lag: { min: 0s, max: 6h, verify: nightly },
        }
    mutability: { class: append_rarely, status: approved }
"""

#: The question E01 answers for billing, and the shape a weak model actually sends.
BILLING_SPEC: Final[dict[str, Any]] = {
    "from": "billing.accounts",
    "select": [{"measure": "credit_limit_r5000", "agg": "avg"}],
    "group_by": ["customer_status"],
    "order_by": [{"by": "customer_status", "direction": "asc"}],
    "limit": 50,
}


# --------------------------------------------------------------------------------------------------
# deterministic time, a throwaway database, an authenticated caller
# --------------------------------------------------------------------------------------------------


class ScriptedClock:
    """A clock the test drives: ``exec_ts``, cache TTLs and handle expiry never cost a sleep."""

    def __init__(self, now: datetime = EXEC_TS) -> None:
        self._now = now
        self._mono = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def __call__(self) -> datetime:
        """The plain-callable shape history and the audit log take."""
        return self._now

    def advance(self, delta: timedelta) -> datetime:
        self._now += delta
        self._mono += delta.total_seconds()
        return self._now


@pytest.fixture(scope="module")
def world() -> World:
    """The seeded D02 world in its smallest declared size: same generator, fewer rows."""
    return build_world(WorldConfig.small())


@pytest.fixture(scope="module")
def layer(tmp_path_factory: pytest.TempPathFactory) -> SemanticLayer:
    """The M03 layer, written once and loaded through the real loader."""
    path = tmp_path_factory.mktemp("layer") / "m03.yaml"
    path.write_text(LAYER_YAML, encoding="utf-8")
    return load(path)


@pytest.fixture
def adapters(world: World) -> Iterator[dict[str, FakeAdapter]]:
    """Three fake adapters, one per logical source, each holding that source's storage form."""
    built = {
        "crm": FakeAdapter(
            name="crm", tables={"customers": table_from_rows(world.crm.customers)}
        ),
        "billing": FakeAdapter(
            name="billing",
            tables={
                "accounts": table_from_rows(world.billing.accounts),
                "usage_events": table_from_rows(world.billing.usage_events),
            },
        ),
        "erp": FakeAdapter(
            name="erp", tables={"customers": table_from_rows(world.erp.customers)}
        ),
    }
    try:
        yield built
    finally:
        for adapter in built.values():
            adapter.close()


@pytest.fixture
def clock() -> ScriptedClock:
    return ScriptedClock()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    """A migrated metadata DB on a throwaway file: history and the audit chain live here."""
    opened = Store(tmp_path / "meta.sqlite")
    opened.initialize()
    try:
        yield opened
    finally:
        opened.close()


@pytest.fixture
def catalog(adapters: Mapping[str, FakeAdapter]) -> dict[str, tuple[str, ...]]:
    """The columns the adapters report: the only spellings a fragment may use."""
    columns: dict[str, tuple[str, ...]] = {}
    for source, adapter in adapters.items():
        for table in adapter.extract_catalog().tables:
            columns[f"{source}.{table.name}"] = tuple(column.name for column in table.columns)
    return columns


@pytest.fixture
def services(
    adapters: Mapping[str, FakeAdapter],
    layer: SemanticLayer,
    clock: ScriptedClock,
    store: Store,
    catalog: Mapping[str, tuple[str, ...]],
    tmp_path: Path,
) -> Iterator[QueryServices]:
    """The whole pipeline: governor, cache, history and audit over the three fake sources."""
    con = duckdb.connect(":memory:")
    con.execute("SET TimeZone='UTC'").fetchall()
    try:
        built = QueryServices(
            layer=layer,
            governor=Governor(dict(adapters), clock=clock),
            history=HistoryStore(store, clock=clock),
            audit=AuditLog(store, clock=clock),
            adapters=dict(adapters),
            columns=catalog,
            cache=FragmentCache(con, clock=clock),
            clock=clock,
            limits=LIMITS,
        )
        yield built
    finally:
        con.close()


@pytest.fixture(autouse=True)
def configured(services: QueryServices) -> Iterator[None]:
    """Inject the pipeline for each test and clear the process-wide slot afterwards."""
    reset_query_services()
    configure_query_services(services)
    try:
        yield
    finally:
        reset_query_services()


def access_token(user_id: str = "alice") -> AccessToken:
    """The verified access token the SDK's middleware binds for one message."""
    return AccessToken(
        token="throwaway-not-a-secret",
        client_id=f"mdqm:{user_id}",
        scopes=[READ_SCOPE],
        subject=user_id,
        claims={"display_name": user_id.title()},
    )


@contextmanager
def as_user(user_id: str = "alice") -> Iterator[None]:
    """Run the block as ``user_id``: what the auth middleware binds inside a tool handler."""
    reset = auth_context_var.set(AuthenticatedUser(access_token(user_id)))
    try:
        yield
    finally:
        auth_context_var.reset(reset)


def ask(
    spec: Mapping[str, Any] | None = None,
    *,
    sql: str | None = None,
    source: str | None = None,
) -> QueryAnswer:
    """Call the tool the way the MCP SDK does: the arguments are validated into the wire models.

    Over the transport the SDK validates ``arguments`` against the tool's input schema before the
    handler runs, so a handler only ever sees a ``QuerySpec``. Calling ``query`` with a raw dict
    would test a path that does not exist in production.
    """
    validated = None if spec is None else QuerySpec.model_validate(spec)
    return query(validated, sql=sql, source=source)


def calls(adapters: Mapping[str, FakeAdapter]) -> dict[str, list[str]]:
    """Every ``execute`` call per source: the evidence that a stage stopped before the source."""
    return {source: list(adapter.calls) for source, adapter in adapters.items()}


def total_calls(trail: Mapping[str, list[str]]) -> int:
    return sum(len(statements) for statements in trail.values())


# --------------------------------------------------------------------------------------------------
# a valid spec: the answer, and the two records
# --------------------------------------------------------------------------------------------------


def test_a_valid_spec_answers_with_a_summary_sample_and_handle(
    services: QueryServices, adapters: Mapping[str, FakeAdapter]
) -> None:
    with as_user():
        answer = ask(BILLING_SPEC)

    assert answer.status == "ok", answer.error
    assert answer.error is None
    assert answer.user_id == "alice"
    assert answer.summary is not None
    assert answer.result_handle and answer.result_handle.startswith("h_")
    assert answer.result_handle == answer.summary.handle
    # The three statuses of the world, averaged after the conversion and read back by name.
    assert [row["customer_status"] for row in answer.sample] == ["active", "churned", "suspended"]
    assert answer.rows_total == 3
    assert answer.cache_hit is False
    # The billed source is the one that was asked, and only it.
    assert total_calls(calls(adapters)) == 1
    assert calls(adapters)["billing"]
    # The plan came from the source's own EXPLAIN; ``accounts`` declares no partitioning, so the
    # honest verdict is "unknown", never a guess that the scan prunes.
    assert answer.plan is not None
    assert answer.plan.pruning_proof.value == "unknown"
    # This call fetched the rows, so ``fetched_at`` is this execution's instant.
    assert answer.freshness[0].status == "miss"
    assert answer.freshness[0].fetched_at == EXEC_TS

    records = services.history.recent(user_id="alice")
    assert len(records) == 1
    run = records[0]
    assert run.tool == "query" and run.exec_ts == EXEC_TS
    assert run.spec is not None and run.spec.from_table == "billing.accounts"
    assert run.fragments and run.fragments[0].source == "billing"
    assert run.error is None and run.cache is not None and run.cache.hit is False
    assert run.rerunnable

    audit = services.audit.rows()
    assert [(row.user_id, row.tool, row.outcome) for row in audit] == [
        ("alice", "query", AuditOutcome.OK)
    ]
    assert audit[0].version_hash == services.layer.version_hash
    assert services.audit.verify_chain().ok


def test_a_dry_run_returns_the_plan_and_touches_nothing(
    services: QueryServices, adapters: Mapping[str, FakeAdapter]
) -> None:
    with as_user():
        answer = ask({**BILLING_SPEC, "dry_run": True})

    assert answer.status == "dry_run"
    assert answer.summary is None and answer.result_handle is None
    assert answer.plan is not None
    assert total_calls(calls(adapters)) == 0
    assert services.history.count() == 0


# --------------------------------------------------------------------------------------------------
# a refused spec: the linter's words, and no source call
# --------------------------------------------------------------------------------------------------


def test_an_unknown_column_returns_the_linters_error_and_calls_no_source(
    services: QueryServices, adapters: Mapping[str, FakeAdapter]
) -> None:
    with as_user():
        answer = ask({**BILLING_SPEC, "select": ["credit_limitt"], "group_by": [], "order_by": []})

    assert answer.status == "error"
    assert answer.error is not None
    assert answer.error.code == ErrorCode.UNKNOWN_COLUMN.value
    assert "credit_limit" in (answer.error.did_you_mean or ())
    assert "credit_limit" in answer.error.fix
    assert total_calls(calls(adapters)) == 0
    # A refusal is a run: it is in the audit chain and in history, with the code the agent saw.
    audit = services.audit.rows()
    assert [row.outcome for row in audit] == [AuditOutcome.BLOCKED]
    assert audit[0].error_code is ErrorCode.UNKNOWN_COLUMN
    assert services.audit.verify_chain().ok
    records = services.history.recent(user_id="alice")
    assert [None if run.error is None else run.error.code for run in records] == [
        ErrorCode.UNKNOWN_COLUMN.value
    ]


def test_a_partitioned_table_without_a_time_filter_is_refused_before_the_source(
    services: QueryServices, adapters: Mapping[str, FakeAdapter]
) -> None:
    with as_user():
        answer = ask({"from": "billing.usage_events", "select": ["units"], "limit": 5})

    assert answer.status == "error"
    assert answer.error is not None
    assert answer.error.code == ErrorCode.PARTITION_SCAN_ALL.value
    assert answer.error.fix
    # The estimate the agent needs in order to see the fix is worth making.
    assert answer.error.partitions_if_fixed
    assert total_calls(calls(adapters)) == 0


# --------------------------------------------------------------------------------------------------
# the cache: an identical call costs nothing, and ``fresh`` pays again
# --------------------------------------------------------------------------------------------------


def test_the_same_spec_twice_is_served_from_the_cache_with_zero_new_calls(
    services: QueryServices, adapters: Mapping[str, FakeAdapter]
) -> None:
    with as_user():
        first = ask(BILLING_SPEC)
        assert first.status == "ok", first.error
        after_first = total_calls(calls(adapters))
        second = ask(BILLING_SPEC)

    assert after_first == 1
    assert total_calls(calls(adapters)) == after_first, "the second call must not reach a source"
    assert second.status == "ok", second.error
    assert second.cache_hit is True
    assert second.rows_total == first.rows_total
    assert second.sample == first.sample
    # Same rows, same answer, and the freshness says when the rows were learned -- not "now".
    assert second.freshness[0].status == "hit"
    assert second.freshness[0].fetched_at == first.freshness[0].fetched_at
    assert second.freshness[0].age_s is not None and second.freshness[0].age_s >= 0
    # Neither call was refused, so both are in the chain, and the second one is a cache hit.
    assert [run.cache.hit for run in services.history.recent(user_id="alice") if run.cache] == [
        True,
        False,
    ]


def test_fresh_forces_the_source_even_when_the_cache_holds_the_answer(
    services: QueryServices, adapters: Mapping[str, FakeAdapter]
) -> None:
    with as_user():
        assert ask(BILLING_SPEC).status == "ok"
        before = total_calls(calls(adapters))
        again = ask({**BILLING_SPEC, "fresh": True})

    assert again.status == "ok", again.error
    assert total_calls(calls(adapters)) == before + 1
    assert again.cache_hit is False
    assert again.freshness[0].status == "miss"


def test_a_stale_entry_is_refetched_instead_of_served(
    services: QueryServices, adapters: Mapping[str, FakeAdapter], clock: ScriptedClock
) -> None:
    with as_user():
        assert ask(BILLING_SPEC).status == "ok"
        before = total_calls(calls(adapters))
        # Well past the floor TTL (30s): an append_rarely table may live that long, not for ever.
        clock.advance(timedelta(hours=1))
        stale = ask(BILLING_SPEC)

    assert stale.status == "ok", stale.error
    assert total_calls(calls(adapters)) == before + 1
    assert stale.cache_hit is False


# --------------------------------------------------------------------------------------------------
# the emergency exit: raw SQL is guarded, and the block is audited
# --------------------------------------------------------------------------------------------------


def test_a_delete_sent_as_raw_sql_is_blocked_and_audited(
    services: QueryServices, adapters: Mapping[str, FakeAdapter]
) -> None:
    with as_user():
        answer = ask(sql="DELETE FROM accounts WHERE client_no = '0000001234'", source="billing")

    assert answer.status == "error"
    assert answer.error is not None
    assert answer.error.code == ErrorCode.WRITE_BLOCKED.value
    assert "SELECT" in answer.error.fix
    assert total_calls(calls(adapters)) == 0

    audit = services.audit.rows()
    assert len(audit) == 1
    row = audit[0]
    assert row.outcome is AuditOutcome.BLOCKED
    assert row.error_code is ErrorCode.WRITE_BLOCKED
    assert row.blocked_attempts == 1
    assert row.tool == "query"
    verification = services.audit.verify_chain()
    assert verification.ok, verification.describe()


def test_raw_sql_needs_an_explicit_configured_source() -> None:
    with as_user():
        missing = ask(sql="SELECT 1")
        unknown = ask(sql="SELECT 1", source="nowhere")

    for answer in (missing, unknown):
        assert answer.status == "error"
        assert answer.error is not None
        assert answer.error.code == ErrorCode.INVALID_SPEC.value
        assert answer.error.fix


def test_a_read_only_statement_runs_through_the_governor(
    adapters: Mapping[str, FakeAdapter],
) -> None:
    with as_user():
        answer = ask(sql="SELECT status FROM customers LIMIT 3", source="crm")

    assert answer.status == "ok", answer.error
    assert answer.summary is not None
    # No table was resolved, so no partition plan and no EXPLAIN: the answer says so by "unknown".
    assert answer.plan is not None
    assert answer.freshness[0].status == "uncached"
    assert len(adapters["crm"].calls) == 1


# --------------------------------------------------------------------------------------------------
# the server without a pipeline still answers
# --------------------------------------------------------------------------------------------------


def test_the_tool_says_not_implemented_when_no_pipeline_was_configured() -> None:
    reset_query_services()
    with as_user():
        answer = ask(BILLING_SPEC)

    assert isinstance(answer, QueryAnswer)
    assert answer.status == "not_implemented"
    assert answer.surface == "query"
    assert answer.user_id == "alice"
    assert answer.problem and answer.fix
