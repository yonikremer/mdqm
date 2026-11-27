"""M04: the ``query`` tool's federated branch -- a spec that spans sources, and its receipt.

Five claims, and the last one is the reason the task exists:

* a spec whose approved join reaches another source answers what task E01 derived independently,
  through the tool, with the join evaluated locally: the cross-source usage question (billing's
  events, crm's status) and the cross-source credit-limit question (crm's canonical value, billing
  and erp proving the customer exists);
* the *report* carries the bad news: the planted ``client_no`` rows that have no customer are
  counted as drops, with the planted keys sampled;
* the protections still apply: a fragment over its row cap is ``COST_TOO_HIGH`` before the join, a
  partitioned table with no window is ``PARTITION_SCAN_ALL`` before any source is asked, a join that
  is not approved is ``JOIN_NOT_APPROVED`` with the approved ON clause, and a source that is down
  yields ``complete=false`` and no rows at all;
* a second identical cross-source query makes **zero** new source calls and reports the first call's
  ``fetched_at`` per source;
* a single-source spec keeps taking M03's path unchanged: no quality report, no executor, one
  freshness row.

Everything runs in one process: the three sources are F05 fake adapters (each a real in-process
DuckDB holding one storage form of the D02 world), and the tool is called the way the MCP SDK calls
it, with the auth context bound and the arguments validated into the wire models.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, cast

import duckdb
import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from ruamel.yaml import YAML

from mdqm.errors import ErrorCode
from mdqm.executor.cache import FragmentCache
from mdqm.executor.federate import FederatedExecutor
from mdqm.governor import Governor
from mdqm.models import ErrorResponse, Limits
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
from mdqm.server.tools.query_federated import federated_plan
from mdqm.spec.models import QuerySpec
from mdqm.store import Store
from mdqm.store.audit import AuditLog
from mdqm.store.history import HistoryStore
from mdqm.testing.fake_adapter import FakeAdapter, table_from_rows
from mdqm.testing.world import World, WorldConfig, build_world

ROOT: Final = Path(__file__).resolve().parents[2]
QUESTIONS: Final = ROOT / "tests" / "eval" / "questions.yaml"

#: The world's own anchor: a trailing "2mo" window from here is exactly E01's November+December one.
EXEC_TS: Final = datetime(2026, 1, 1, tzinfo=UTC)

#: Caps the tests expect: only the explicit row-cap test trips them.
LIMITS: Final = Limits(timeout_s=30.0, max_rows=50_000, max_bytes=64 << 20, max_partitions=64)

#: The layer these tests federate under: the M03 layer plus the two customer-identity joins the D02
#: world needs (crm<->billing by padded key, crm<->erp by integer key), and an approved mutability
#: class per table so the cache is allowed to work at all.
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
        - { name: p2025_10, lower: 2025-10-01T00:00:00Z, upper: 2025-11-01T00:00:00Z }
        - { name: p2025_11, lower: 2025-11-01T00:00:00Z, upper: 2025-12-01T00:00:00Z }
        - { name: p2025_12, lower: 2025-12-01T00:00:00Z, upper: 2026-01-01T00:00:00Z }
    time_columns:
      start_time: { role: event_time, indexed: true, use_by_default: true }
      insertion_time:
        {
          role: ingestion_time,
          derives_bound_on: start_time,
          lag: { min: 0s, max: 6h, verify: nightly },
        }
    mutability: { class: append_rarely, status: approved }

approved_joins:
  - name: usage_to_customers
    left: billing.usage_events.client_no
    right: crm.customers.cust_id
    on: "LPAD(CAST({right} AS VARCHAR), 10, '0') = {left}"
    cardinality: many_to_one
  - name: crm_accounts
    left: billing.accounts.client_no
    right: crm.customers.cust_id
    on: "LPAD(CAST({right} AS VARCHAR), 10, '0') = {left}"
    cardinality: many_to_one
  - name: crm_erp
    left: crm.customers.cust_id
    right: erp.customers.cust_no
    on: "{left} = {right}"
    cardinality: one_to_many
"""

#: The one cross-source question the MVP must answer as one answer (E01's usage question).
USAGE_SPEC: Final[dict[str, Any]] = {
    "from": "billing.usage_events",
    "select": [{"measure": "usage_units", "agg": "sum"}],
    "joins": ["usage_to_customers"],
    "time": {"mode": "events_in_range", "from": "2025-11-01", "to": "2026-01-01"},
    "group_by": ["customer_status"],
    "order_by": [{"by": "customer_status", "direction": "asc"}],
    "limit": 50,
}

#: The cross-source credit-limit question: crm's canonical value, billing+erp proving identity.
#: The grouping column is written as ``status`` rather than the value map's field name
#: ``customer_status``: with three tables in scope that field is ambiguous, and the server refuses
#: it rather than guessing which source's status the analyst meant (see the ambiguity test below).
CREDIT_SPEC: Final[dict[str, Any]] = {
    "from": "crm.customers",
    "select": [{"measure": "credit_limit_r5000", "agg": "avg"}],
    "joins": ["crm_accounts", "crm_erp"],
    "group_by": ["status"],
    "order_by": [{"by": "status", "direction": "asc"}],
    "limit": 50,
}


# --- fixtures ------------------------------------------------------------------------------


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
        return self._now

    def advance(self, delta: timedelta) -> datetime:
        self._now += delta
        self._mono += delta.total_seconds()
        return self._now


def read_questions() -> Mapping[str, Mapping[str, Any]]:
    """The E01 document's questions by id."""
    document = cast(Mapping[str, Any], YAML(typ="safe").load(QUESTIONS.read_text(encoding="utf-8")))
    return {str(question["id"]): question for question in document["questions"]}


@pytest.fixture(scope="module")
def pinned_world() -> World:
    """Exactly the seeded world the E01 answers describe (task D02)."""
    document = cast(Mapping[str, Any], YAML(typ="safe").load(QUESTIONS.read_text(encoding="utf-8")))
    world = cast(Mapping[str, Any], document["world"])
    anchor = world["anchor"]
    return build_world(
        WorldConfig(
            seed=int(world["seed"]),
            customers=int(world["customers"]),
            months=int(world["months"]),
            events_per_customer=int(world["events_per_customer"]),
            anchor=anchor if isinstance(anchor, datetime) else datetime.fromisoformat(anchor),
            max_lag_seconds=int(world["max_lag_seconds"]),
        )
    )


@pytest.fixture(scope="module")
def layer(tmp_path_factory: pytest.TempPathFactory) -> SemanticLayer:
    path = tmp_path_factory.mktemp("layer") / "m04.yaml"
    path.write_text(LAYER_YAML, encoding="utf-8")
    return load(path)


@pytest.fixture
def adapters(pinned_world: World) -> Iterator[dict[str, FakeAdapter]]:
    """Three fake adapters, one per logical source, holding that source's storage form."""
    built = {
        "crm": FakeAdapter(
            name="crm", tables={"customers": table_from_rows(pinned_world.crm.customers)}
        ),
        "billing": FakeAdapter(
            name="billing",
            tables={
                "accounts": table_from_rows(pinned_world.billing.accounts),
                "usage_events": table_from_rows(pinned_world.billing.usage_events),
            },
        ),
        "erp": FakeAdapter(
            name="erp", tables={"customers": table_from_rows(pinned_world.erp.customers)}
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
    opened = Store(tmp_path / "meta.sqlite")
    opened.initialize()
    try:
        yield opened
    finally:
        opened.close()


@pytest.fixture
def catalog(adapters: Mapping[str, FakeAdapter]) -> dict[str, tuple[str, ...]]:
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
    """The pipeline plus a shared federated executor, so its cache survives between calls."""
    con = duckdb.connect(":memory:")
    con.execute("SET TimeZone='UTC'").fetchall()
    runner = FederatedExecutor(
        Governor(dict(adapters), clock=clock),
        layer=layer,
        clock=clock,
        spill_dir=tmp_path / "spill",
    )
    try:
        yield QueryServices(
            layer=layer,
            governor=Governor(dict(adapters), clock=clock),
            history=HistoryStore(store, clock=clock),
            audit=AuditLog(store, clock=clock),
            adapters=dict(adapters),
            columns=catalog,
            cache=FragmentCache(con, clock=clock),
            clock=clock,
            limits=LIMITS,
            executor=runner,
            spill_dir=tmp_path / "spill",
        )
    finally:
        runner.close()
        con.close()


@pytest.fixture(autouse=True)
def configured(services: QueryServices) -> Iterator[None]:
    reset_query_services()
    configure_query_services(services)
    try:
        yield
    finally:
        reset_query_services()


def access_token(user_id: str = "alice") -> AccessToken:
    return AccessToken(
        token="test-token",
        client_id=f"mdqm:{user_id}",
        scopes=[READ_SCOPE],
        subject=user_id,
        claims={"display_name": user_id.title()},
    )


@contextmanager
def as_user(user_id: str = "alice") -> Iterator[None]:
    reset = auth_context_var.set(AuthenticatedUser(access_token(user_id)))
    try:
        yield
    finally:
        auth_context_var.reset(reset)


def ask(spec: Mapping[str, Any]) -> QueryAnswer:
    """Call the tool the way the SDK does: the arguments are validated into the wire models."""
    return query(QuerySpec.model_validate(spec))


def calls(adapters: Mapping[str, FakeAdapter]) -> int:
    return sum(len(adapter.calls) for adapter in adapters.values())


def orphan_events_in_window(world: World, *, start: datetime, end: datetime) -> int:
    """The planted ``client_no`` rows the join cannot match, counted from the world itself."""
    known = {customer.client_no for customer in world.expected.customers}
    return sum(
        1
        for event in world.billing.usage_events
        if start <= event.start_time < end and event.client_no not in known
    )


# --- the answer ----------------------------------------------------------------------------


def test_the_cross_source_usage_question_answers_the_e01_ground_truth(
    services: QueryServices, adapters: Mapping[str, FakeAdapter], pinned_world: World
) -> None:
    """billing's events joined to crm's status: one answer, and it is the document's answer."""
    expected = read_questions()["usage_units_by_status_window"]["answer"]
    assert expected["columns"] == ["customer_status", "usage_units_sum"]
    with as_user():
        answer = ask(USAGE_SPEC)
    assert answer.status == "ok", answer.error
    assert answer.quality is not None and answer.quality.complete is True
    assert answer.summary is not None
    # The summary's sample is JSON, so numbers arrive as text: compare the values, not the spelling.
    rows = [
        [row["customer_status"], int(cast(Any, row["usage_units_sum"]))] for row in answer.sample
    ]
    assert rows == [list(row) for row in expected["rows"]]
    assert answer.rows_total == len(expected["rows"])
    assert answer.result_handle is not None
    assert answer.cache_hit is False
    assert answer.status == "ok"

    # Two fragments, two sources, each with its own fetched_at.
    assert {step.source for step in answer.quality.steps} == {"billing", "crm"}
    assert {step.table for step in answer.quality.steps} == {
        "billing.usage_events",
        "crm.customers",
    }
    assert {fresh.source for fresh in answer.freshness} == {"billing", "crm"}
    assert all(fresh.fetched_at == EXEC_TS for fresh in answer.freshness)
    for step in answer.quality.steps:
        assert step.fetched is True
        assert step.rows > 0
        assert step.sql is not None

    # The join is the approved one, and the drops are the planted orphans in this window.
    (join,) = answer.quality.joins
    assert join.join == "usage_to_customers"
    assert join.how == "inner"
    assert "LPAD" in join.on
    planted = orphan_events_in_window(
        pinned_world,
        start=datetime(2025, 11, 1, tzinfo=UTC),
        end=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert join.dropped == planted
    assert 0 < join.dropped <= pinned_world.expected.orphan_event_count
    assert answer.quality.rows_dropped == planted
    assert set(join.dropped_keys) <= set(pinned_world.expected.orphan_client_nos)
    assert any("dropped" in note for note in answer.quality.notes)
    # Every source was really asked: one fragment each, three sources idle.
    assert calls(adapters) == 2


def test_a_trailing_window_federates_the_same_way_as_an_absolute_one(
    services: QueryServices,
) -> None:
    """``last: 2mo`` from the world's anchor is E01's window, resolved by the tool."""
    relative = dict(USAGE_SPEC)
    relative["time"] = {"mode": "events_in_range", "last": "2mo"}
    with as_user():
        trailing = ask(relative)
        absolute = ask(USAGE_SPEC)
    assert trailing.status == "ok", trailing.error
    assert trailing.sample == absolute.sample


def test_the_cross_source_credit_question_matches_the_crm_ground_truth(
    services: QueryServices,
) -> None:
    """Three storage forms, one answer: crm holds the value, billing and erp prove identity."""
    expected = read_questions()["avg_limit_by_status_crm"]["answer"]
    with as_user():
        answer = ask(CREDIT_SPEC)
    assert answer.status == "ok", answer.error
    assert answer.quality is not None and answer.quality.complete is True
    assert answer.sample and len(answer.sample) == len(expected["rows"])
    for row, values in zip(answer.sample, expected["rows"], strict=True):
        assert row["status"] == values[0]
        measured = float(cast(Any, row["credit_limit_r5000_avg"]))
        assert math.isclose(measured, cast(float, values[1]), rel_tol=1e-15), (
            f"{row['status']}: {measured} != {values[1]}"
        )
    assert {step.source for step in answer.quality.steps} == {"crm", "billing", "erp"}
    # The three sources were joined through the approved joins, and the tool said so.
    assert any("crm_accounts" in fix for fix in answer.applied_fixes)
    assert any("crm_erp" in fix for fix in answer.applied_fixes)
    by_join = {join.join: join for join in answer.quality.joins}
    # Both partners are the *many* side of their join and neither reaches an output column, so
    # neither may fan the canonical crm rows out: membership without multiplication.
    assert by_join["crm_accounts"].how == "semi"
    assert by_join["crm_erp"].how == "semi"
    for join in by_join.values():
        assert join.matched + join.dropped == join.rows_left
        assert join.multiplied == 0


def test_a_single_source_spec_keeps_the_m03_path(
    services: QueryServices, adapters: Mapping[str, FakeAdapter], layer: SemanticLayer
) -> None:
    """No join, one source: no quality report, one freshness row, and M03's own fragment cache."""
    spec = QuerySpec.model_validate(
        {
            "from": "crm.customers",
            "select": [{"measure": "credit_limit_r5000", "agg": "avg"}],
            "group_by": ["customer_status"],
            "limit": 50,
        }
    )
    exec_ts = services.now()
    assert federated_plan(services, spec, exec_ts) is None
    with as_user():
        answer = ask(spec.model_dump(mode="json", by_alias=True, exclude_none=True))
    assert answer.status == "ok", answer.error
    assert answer.quality is None
    assert len(answer.freshness) == 1
    assert answer.freshness[0].source == "crm"
    # Only crm was asked: the federated builder never ran, so nothing else was touched.
    assert calls(adapters) == 1
    assert "crm.customers" in answer.freshness[0].table


# --- the bad news the report must carry -----------------------------------------------------


def test_rows_dropped_by_the_join_are_reported_with_counts_and_a_sample(
    services: QueryServices, pinned_world: World
) -> None:
    """A join that loses rows says how many and which keys; silence is the failure mode."""
    with as_user():
        answer = ask(USAGE_SPEC)
    assert answer.quality is not None
    (join,) = answer.quality.joins
    assert join.rows_left > 0
    assert join.matched + join.dropped == join.rows_left
    assert 0 < join.dropped < join.rows_left
    assert 1 <= len(join.dropped_keys) <= 5
    assert join.reason.startswith("left rows whose billing.usage_events.client_no")
    assert join.dropped == orphan_events_in_window(
        pinned_world, start=datetime(2025, 11, 1, tzinfo=UTC), end=datetime(2026, 1, 1, tzinfo=UTC)
    )


def test_an_unapproved_join_is_refused_with_the_approved_on_clause(
    services: QueryServices, adapters: Mapping[str, FakeAdapter]
) -> None:
    """A join name the layer does not have is a refusal with the fix, not a guess.

    The *ON clause* of an approved join is quoted by the test below, where the spec asks for a
    column that only a join reaches: that is the case where the agent has to copy a condition.
    """
    spec = {
        "from": "billing.usage_events",
        "select": [{"measure": "usage_units", "agg": "sum"}],
        "joins": ["usage_to_clients"],
        "time": {"mode": "events_in_range", "last": "2mo"},
        "limit": 5,
    }
    with as_user():
        answer = ask(spec)
    assert answer.status == "error"
    assert answer.error is not None
    assert answer.error.code == ErrorCode.JOIN_NOT_APPROVED
    assert answer.error.fix and "curated/joins.yaml" in answer.error.fix
    assert "usage_to_clients" in answer.error.problem
    assert answer.summary is None and answer.sample == () and answer.rows_total == 0
    assert calls(adapters) == 0


def test_a_join_that_does_not_touch_the_read_table_is_refused(
    services: QueryServices, adapters: Mapping[str, FakeAdapter]
) -> None:
    """erp has no approved join to usage: the refusal names the joins it could use instead."""
    spec = {
        "from": "erp.customers",
        "select": ["cust_no"],
        "joins": ["usage_to_customers"],
        "limit": 5,
    }
    with as_user():
        answer = ask(spec)
    assert answer.status == "error"
    assert answer.error is not None
    assert answer.error.code == ErrorCode.INVALID_SPEC
    assert answer.error.fix and "joins.0" in answer.error.fix
    assert calls(adapters) == 0


def test_a_grouping_column_stored_in_three_sources_is_refused_as_ambiguous(
    services: QueryServices, adapters: Mapping[str, FakeAdapter]
) -> None:
    """The value map's field is ambiguous across three sources: refused, not guessed."""
    ambiguous = dict(CREDIT_SPEC)
    ambiguous["group_by"] = ["customer_status"]
    ambiguous["order_by"] = [{"by": "customer_status", "direction": "asc"}]
    with as_user():
        answer = ask(ambiguous)
    assert answer.status == "error"
    assert answer.error is not None
    assert answer.error.code == ErrorCode.INVALID_SPEC
    assert answer.error.fix and "group_by.0" in answer.error.fix
    assert calls(adapters) == 0


def test_a_column_that_needs_an_approved_join_answers_with_its_on_clause(
    services: QueryServices, adapters: Mapping[str, FakeAdapter]
) -> None:
    """The join-aware hint: "that column lives in another source, here is the ON clause"."""
    spec = {
        "from": "billing.usage_events",
        "select": ["units", "cust_id"],
        "time": {"mode": "events_in_range", "last": "2mo"},
        "limit": 5,
    }
    with as_user():
        answer = ask(spec)
    assert answer.status == "error"
    assert answer.error is not None
    assert answer.error.code == ErrorCode.JOIN_NOT_APPROVED
    assert answer.error.fix and "LPAD" in answer.error.fix
    assert calls(adapters) == 0


# --- the protections ------------------------------------------------------------------------


def test_a_fragment_over_its_row_cap_is_refused_before_the_join(
    services: QueryServices, adapters: Mapping[str, FakeAdapter]
) -> None:
    """COST_TOO_HIGH with a fix, from the frame that really caps a fragment."""
    tight = LIMITS.model_copy(update={"max_rows": 10})
    services = services.__class__(
        layer=services.layer,
        governor=services.governor,
        history=services.history,
        audit=services.audit,
        adapters=services.adapters,
        columns=services.columns,
        cache=services.cache,
        clock=services.clock,
        limits=tight,
        executor=services.executor,
    )
    with as_user():
        configure_query_services(services)
        answer = ask(USAGE_SPEC)
    assert answer.status == "error"
    assert answer.error is not None
    assert answer.error.code == ErrorCode.COST_TOO_HIGH
    assert answer.error.fix
    assert answer.quality is not None and answer.quality.complete is False
    assert answer.sample == () and answer.rows_total == 0


def test_a_partitioned_table_with_no_window_is_refused_before_any_source(
    services: QueryServices, adapters: Mapping[str, FakeAdapter]
) -> None:
    spec = {
        "from": "billing.usage_events",
        "select": [{"measure": "usage_units", "agg": "sum"}],
        "joins": ["usage_to_customers"],
        "group_by": ["customer_status"],
        "limit": 5,
    }
    with as_user():
        answer = ask(spec)
    assert answer.status == "error"
    assert answer.error is not None
    assert answer.error.code == ErrorCode.PARTITION_SCAN_ALL
    assert answer.error.fix
    assert calls(adapters) == 0


def test_a_source_that_fails_yields_complete_false_and_no_rows(
    layer: SemanticLayer,
    pinned_world: World,
    clock: ScriptedClock,
    store: Store,
    catalog: Mapping[str, tuple[str, ...]],
    tmp_path: Path,
) -> None:
    """Half a join is not an answer: the report says incomplete and there is nothing to page."""
    broken = {
        "crm": FakeAdapter(
            name="crm", tables={"customers": table_from_rows(pinned_world.crm.customers)}
        ),
        "billing": FakeAdapter(
            name="billing",
            tables={
                "accounts": table_from_rows(pinned_world.billing.accounts),
                "usage_events": table_from_rows(pinned_world.billing.usage_events),
            },
            fail_with=ConnectionError("connection refused: port 1 is closed"),
        ),
        "erp": FakeAdapter(
            name="erp", tables={"customers": table_from_rows(pinned_world.erp.customers)}
        ),
    }
    con = duckdb.connect(":memory:")
    con.execute("SET TimeZone='UTC'").fetchall()
    runner = FederatedExecutor(
        Governor(dict(broken), clock=clock),
        layer=layer,
        clock=clock,
        spill_dir=tmp_path / "spill",
    )
    try:
        built = QueryServices(
            layer=layer,
            governor=Governor(dict(broken), clock=clock),
            history=HistoryStore(store, clock=clock),
            audit=AuditLog(store, clock=clock),
            adapters=dict(broken),
            columns=catalog,
            cache=FragmentCache(con, clock=clock),
            clock=clock,
            limits=LIMITS,
            executor=runner,
        )
        with as_user():
            configure_query_services(built)
            answer = ask(USAGE_SPEC)
    finally:
        runner.close()
        con.close()
        for adapter in broken.values():
            adapter.close()
    assert answer.status == "error"
    assert answer.error is not None
    assert answer.error.code == ErrorCode.SOURCE_UNAVAILABLE
    assert answer.quality is not None
    assert answer.quality.complete is False
    assert answer.summary is None
    assert answer.sample == ()
    assert answer.rows_total == 0
    assert answer.result_handle is None
    assert any("no answer" in note for note in answer.quality.notes)


# --- the cache ------------------------------------------------------------------------------


def test_a_repeated_cross_source_query_makes_no_new_source_calls(
    services: QueryServices, adapters: Mapping[str, FakeAdapter], clock: ScriptedClock
) -> None:
    """The second identical call answers from the fragments' cache and says so."""
    with as_user():
        first = ask(USAGE_SPEC)
        after_first = calls(adapters)
        clock.advance(timedelta(seconds=5))
        second = ask(USAGE_SPEC)
    assert first.status == "ok" and second.status == "ok"
    assert after_first == 2
    assert calls(adapters) == after_first  # zero new statements on any source
    assert second.cache_hit is True
    assert first.cache_hit is False
    assert second.sample == first.sample
    assert second.quality is not None
    assert all(step.fetched is False for step in second.quality.steps)
    assert all(step.cache_status == "hit" for step in second.quality.steps)
    # The rows are reported at the time they were really read, not at the time of this call.
    assert {fresh.fetched_at for fresh in second.freshness} == {EXEC_TS}
    assert {fresh.status for fresh in second.freshness} == {"hit"}
    assert all(fresh.age_s == 5.0 for fresh in second.freshness)


def test_a_plan_can_always_be_derived_for_a_cross_source_spec(services: QueryServices) -> None:
    """The derivation is itself testable: steps, joins, output, and the reported joins."""
    spec = QuerySpec.model_validate(CREDIT_SPEC)
    derived = federated_plan(services, spec, EXEC_TS)
    assert derived is not None and not isinstance(derived, ErrorResponse)
    plan = derived.plan
    assert plan.names == ("crm_customers", "billing_accounts", "erp_customers")
    assert [(join.join, join.how) for join in plan.joins] == [
        ("crm_accounts", "semi"),
        ("crm_erp", "semi"),
    ]
    assert plan.output.columns == ("status", "credit_limit_r5000_avg")
    assert plan.output.sort_keys == ("status",)
    assert all(step.fragment.sql is not None for step in plan.steps)
    # Every fragment names one source's tables only: that is what makes it runnable there.
    by_source = {step.source: step.fragment.tables for step in plan.steps}
    assert by_source["crm"] == ("crm.customers",)
    assert by_source["billing"] == ("billing.accounts",)
    assert by_source["erp"] == ("erp.customers",)
