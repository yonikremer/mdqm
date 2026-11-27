"""End-to-end proof of the claims in ``README.md`` "The problems it solves" (E2E).

Each test below is one claim, driven through the product's own path -- a validated spec, the
compiler, the guard, the governor, a real adapter, the federated executor, the metadata store --
with the seeded synthetic organisation of task D02 behind the adapters and the independently
derived answers of task E01 as the only checker. Nothing here asserts that "it ran": every
docstring names the exact README sentence and what would make the assertion false.

Why these live under ``tests/integration``: they are the acceptance slice, not unit tests of one
module. They need no Docker (the three sources are the F05 fake adapters, each a real in-process
DuckDB holding one storage form of the world), so they run everywhere; ``test_postgres_e2e.py`` is
the twin that does need the real server.

The semantic layer is spelled out here instead of imported from ``tests/unit/test_executor.py``:
``tests/`` is not a package, so an integration test cannot import a unit-test fixture. It is the
same federation layer C17 uses (identity / ``ROUND(x/5000)*5000`` / ``x*10``), plus the measures
claim (f) needs to overflow a small context budget.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pytest
from ruamel.yaml import YAML

from mdqm.adapters.base import ArrowStream, SourceAdapter
from mdqm.compiler import CompileContext, CompiledQuery, compile_spec
from mdqm.context.index import ApprovedExample, ContextIndex
from mdqm.context.pack import MAX_EXAMPLES, MIN_PACK_BUDGET, OmissionReason, PackKind, build_pack
from mdqm.errors import ErrorCode
from mdqm.executor import (
    FederatedExecutor,
    FederatedOutput,
    FederatedPlan,
    FragmentStep,
    GroupColumn,
    JoinStep,
    LocalAggregate,
    StepCache,
)
from mdqm.governor import Admission, AdmissionPolicy, Governor, GovernorRefusal
from mdqm.models import ErrorResponse, Fragment, Limits
from mdqm.results.summary import estimate_tokens
from mdqm.semantic import SemanticLayer
from mdqm.semantic.loader import load
from mdqm.spec import parse_spec
from mdqm.spec.models import Agg
from mdqm.sql.guard import guard
from mdqm.store import Store
from mdqm.store.audit import AuditLog, AuditOutcome
from mdqm.testing.fake_adapter import FakeAdapter, table_from_rows
from mdqm.testing.world import World, WorldConfig, build_world, round_half_even, round_half_up

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_FILE = ROOT / "tests" / "eval" / "questions.yaml"

#: One execution timestamp for every call in this module (C02: one value per execution).
EXEC_TS = datetime(2026, 11, 25, 15, 0, tzinfo=UTC)

#: Who is asking. The audit row and the result handle are owned by this id.
USER = "e2e"

#: Caps for every execution: high enough that only a genuine problem trips them.
LIMITS = Limits(timeout_s=30.0, max_rows=50_000, max_bytes=64 << 20, max_partitions=64)

#: The catalog as the three adapters report it: only these spellings may reach a fragment.
COLUMNS: Mapping[str, tuple[str, ...]] = {
    "crm.customers": ("cust_id", "full_name", "status", "credit_limit", "region", "signup_date"),
    "billing.accounts": ("client_no", "state", "credit_limit", "region", "opened_date"),
    "billing.usage_events": (
        "event_id",
        "client_no",
        "start_time",
        "insertion_time",
        "units",
        "corrects_event_id",
    ),
    "erp.customers": ("cust_no", "status_code", "limit_k", "name"),
}

#: The storage-form status codes, spelled out here on purpose: the fixture must not borrow the
#: value map from the code under test (task E01 does the same).
BILLING_STATE: Mapping[str, str] = {"Active": "active", "Closed": "churned", "Hold": "suspended"}

#: The federation semantic layer. Six measures on the same columns, because a real layer has many
#: and claim (f) needs a pack that cannot fit a small budget.
LAYER_YAML = """
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
  credit_limit_r5000_total:
    canonical: { rounding: { to: 5000, mode: half_up } }
    sources:
      crm.customers.credit_limit: { normalize: "{col}" }
      billing.accounts.credit_limit: { normalize: "ROUND({col} / 5000) * 5000" }
      erp.customers.limit_k: { normalize: "{col} * 10" }
  credit_limit_r5000_count:
    canonical: { rounding: { to: 5000, mode: half_up } }
    sources:
      crm.customers.credit_limit: { normalize: "{col}" }
      billing.accounts.credit_limit: { normalize: "ROUND({col} / 5000) * 5000" }
      erp.customers.limit_k: { normalize: "{col} * 10" }
  credit_limit_r5000_max:
    canonical: { rounding: { to: 5000, mode: half_up } }
    sources:
      crm.customers.credit_limit: { normalize: "{col}" }
      billing.accounts.credit_limit: { normalize: "ROUND({col} / 5000) * 5000" }
      erp.customers.limit_k: { normalize: "{col} * 10" }
  credit_limit_r5000_min:
    canonical: { rounding: { to: 5000, mode: half_up } }
    sources:
      crm.customers.credit_limit: { normalize: "{col}" }
      billing.accounts.credit_limit: { normalize: "ROUND({col} / 5000) * 5000" }
      erp.customers.limit_k: { normalize: "{col} * 10" }
  credit_limit_r5000_median:
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
  billing.accounts:
    key: [client_no]
  erp.customers:
    key: [cust_no]
  billing.usage_events:
    key: [event_id]
    default_window: 7d
    partitioning: { key: start_time, granularity: month, require_key_filter: true }
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
  - name: usage_to_customers
    left: billing.usage_events.client_no
    right: crm.customers.cust_id
    on: "LPAD(CAST({right} AS VARCHAR), 10, '0') = {left}"
    cardinality: many_to_one
"""

#: Six examples that all match the same question, so the per-kind cap must drop some and say so.
EXAMPLES: tuple[ApprovedExample, ...] = tuple(
    ApprovedExample(
        name=f"credit_limit_by_status_variant_{index}",
        question="average credit limit by customer status",
        ref=f"specs/credit_limit_by_status_variant_{index}",
    )
    for index in range(MAX_EXAMPLES * 2)
)

#: The question claim (f) asks about: both languages, because the pack must work in both.
PACK_QUESTION = "average credit limit by customer status"


# --- fixtures -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Outcome:
    """One agent request, end to end: the rows, or the refusal, plus the source call trail."""

    columns: tuple[str, ...] = ()
    rows: tuple[Mapping[str, Any], ...] = ()
    error: ErrorResponse | None = None
    calls: Mapping[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the request produced rows instead of a refusal."""
        return self.error is None


class ManualClock:
    """A clock a test drives, so cache staleness never costs a sleep."""

    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta


def read_questions() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """The task E01 document and its questions by id."""
    loaded = YAML(typ="safe").load(QUESTIONS_FILE.read_text(encoding="utf-8"))
    document = cast(Mapping[str, Any], loaded)
    by_id = {question["id"]: question for question in document["questions"]}
    return document, by_id


@pytest.fixture(scope="module")
def pinned_world() -> World:
    """Exactly the seeded world the E01 answers belong to (task D02)."""
    document, _ = read_questions()
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
    """The federation semantic layer, written once and loaded through the real loader."""
    path = tmp_path_factory.mktemp("layer") / "federation.yaml"
    path.write_text(LAYER_YAML, encoding="utf-8")
    return load(path)


def _sources(world: World) -> dict[str, FakeAdapter]:
    """Three fake adapters, one per logical source, each holding that source's storage form."""
    return {
        "crm": FakeAdapter(name="crm", tables={"customers": table_from_rows(world.crm.customers)}),
        "billing": FakeAdapter(
            name="billing",
            tables={
                "accounts": table_from_rows(world.billing.accounts),
                "usage_events": table_from_rows(world.billing.usage_events),
            },
        ),
        "erp": FakeAdapter(name="erp", tables={"customers": table_from_rows(world.erp.customers)}),
    }


@pytest.fixture
def sources(pinned_world: World) -> Iterator[dict[str, FakeAdapter]]:
    built = _sources(pinned_world)
    try:
        yield built
    finally:
        for adapter in built.values():
            adapter.close()


def adapters(sources: Mapping[str, Any]) -> Mapping[str, SourceAdapter]:
    """The sources as a governor takes them.

    A cast, not a wrapper: ``FakeAdapter.kind`` is an instance attribute while the protocol declares
    it as a class variable, so a strict type checker cannot see that the fake is an adapter. F05's
    conformance run is the real proof that it is one.
    """
    return cast(Mapping[str, SourceAdapter], dict(sources))


@pytest.fixture
def governed(sources: Mapping[str, FakeAdapter]) -> Governor:
    """The governor over the three sources: the only path a test has to an adapter."""
    return Governor(adapters(sources))


# --- helpers ------------------------------------------------------------------------------


def call_counts(sources: Mapping[str, FakeAdapter]) -> dict[str, int]:
    """How many times each adapter was asked for rows."""
    return {name: len(adapter.calls) for name, adapter in sources.items()}


def compile_one(
    layer: SemanticLayer,
    source: str,
    spec: Mapping[str, Any],
    *,
    max_rows: int = 50_000,
) -> CompiledQuery:
    """Compile one spec for one source with the known catalog; the fragment or the refusal."""
    context = CompileContext(
        layer=layer,
        source=source,
        dialect="duckdb",
        exec_ts=EXEC_TS,
        columns=COLUMNS,
        max_rows=max_rows,
    )
    return compile_spec(parse_spec(spec), context)


def execute_spec(
    *,
    governor: Governor,
    layer: SemanticLayer,
    sources: Mapping[str, FakeAdapter],
    source: str,
    spec: Mapping[str, Any],
) -> Outcome:
    """The product path for one spec: compile, then run it through the governor, then page it.

    A refusal at either stage is returned, never raised, because that is what an agent sees.
    """
    compiled = compile_one(layer, source, spec)
    if not compiled.ok or compiled.fragment is None:
        return Outcome(error=compiled.error, calls=call_counts(sources))
    try:
        outcome = governor.execute(
            source=source, user=USER, fragment=compiled.fragment, limits=LIMITS
        )
    except GovernorRefusal as refusal:
        return Outcome(error=refusal.response, calls=call_counts(sources))
    page = governor.get_more(
        outcome.handle.handle, user=USER, offset=0, limit=max(1, outcome.report.rows)
    )
    columns = tuple(page.rows[0]) if page.rows else ()
    return Outcome(columns=columns, rows=page.rows, calls=call_counts(sources))


def run_sql(
    governor: Governor, source: str, sql: str, *, tables: tuple[str, ...] = ("usage_events",)
) -> tuple[Mapping[str, Any], ...]:
    """Run one hand-written read query through the governor; rows only, for a counterfactual."""
    fragment = Fragment(source=source, dialect="duckdb", sql=sql, tables=tables, exec_ts=EXEC_TS)
    outcome = governor.execute(source=source, user=USER, fragment=fragment, limits=LIMITS)
    page = governor.get_more(
        outcome.handle.handle, user=USER, offset=0, limit=max(1, outcome.report.rows)
    )
    return cast(tuple[Mapping[str, Any], ...], page.rows)


def assert_e01_answer(
    outcome: Outcome, answer: Mapping[str, Any], tolerance: float = 1e-15
) -> None:
    """Compare an answer with an E01 one: exact for names/codes, relative tolerance for floats."""
    assert outcome.error is None, outcome.error
    assert list(outcome.columns) == list(answer["columns"]), outcome.columns
    expected = [tuple(row) for row in cast(Sequence[Sequence[Any]], answer["rows"])]
    assert len(outcome.rows) == len(expected), outcome.rows
    for row, wanted in zip(outcome.rows, expected, strict=True):
        values = tuple(row[name] for name in outcome.columns)
        for value, target in zip(values, wanted, strict=True):
            if isinstance(target, float):
                assert math.isclose(float(value), target, rel_tol=tolerance), (
                    f"{value!r} != {target!r}"
                )
            else:
                assert value == target, f"{value!r} != {target!r}"


def _step(
    layer: SemanticLayer,
    *,
    name: str,
    source: str,
    table: str,
    spec: Mapping[str, Any],
    dedupe_on: tuple[str, ...] = (),
) -> FragmentStep:
    """One plan step: a compiled fragment plus the rules that apply to running it."""
    compiled = compile_one(layer, source, spec)
    assert compiled.ok, compiled.error
    assert compiled.fragment is not None
    return FragmentStep(
        name=name,
        source=source,
        table=table,
        fragment=compiled.fragment,
        limits=LIMITS,
        cache=StepCache(version_hash=layer.version_hash, mutability="append_rarely"),
        dedupe_on=dedupe_on,
    )


def credit_plan(layer: SemanticLayer) -> FederatedPlan:
    """``avg credit_limit_r5000 by customer_status`` across the three storage forms.

    crm is the identity, billing and erp prove the customer exists there; erp joins as a semi join
    because the world plants duplicate ``cust_no`` rows that a row-producing join would double.
    """
    return FederatedPlan(
        exec_ts=EXEC_TS,
        steps=(
            _step(
                layer,
                name="crm",
                source="crm",
                table="crm.customers",
                spec={
                    "from": "crm.customers",
                    "select": ["cust_id", "customer_status", "credit_limit"],
                    "limit": 5_000,
                },
            ),
            _step(
                layer,
                name="billing",
                source="billing",
                table="billing.accounts",
                spec={
                    "from": "billing.accounts",
                    "select": ["client_no", "customer_status", "credit_limit"],
                    "limit": 5_000,
                },
            ),
            _step(
                layer,
                name="erp",
                source="erp",
                table="erp.customers",
                spec={
                    "from": "erp.customers",
                    "select": ["cust_no", "customer_status", "limit_k"],
                    "limit": 5_000,
                },
                dedupe_on=("cust_no",),
            ),
        ),
        joins=(
            JoinStep(join="crm_accounts", left="crm", right="billing", how="left"),
            JoinStep(join="crm_erp", left="crm", right="erp", how="semi"),
        ),
        output=FederatedOutput(
            group_by=(GroupColumn(step="crm", column="customer_status"),),
            aggregates=(
                LocalAggregate(
                    step="crm",
                    column="credit_limit",
                    agg=Agg.AVG,
                    alias="credit_limit_r5000_avg",
                    measure="credit_limit_r5000",
                ),
            ),
            order_by=("customer_status",),
            limit=50,
        ),
    )


def usage_plan(layer: SemanticLayer, *, start: str, end: str) -> FederatedPlan:
    """``sum usage_units by customer status`` for one window: the E01 cross-source question."""
    return FederatedPlan(
        exec_ts=EXEC_TS,
        steps=(
            _step(
                layer,
                name="usage",
                source="billing",
                table="billing.usage_events",
                spec={
                    "from": "billing.usage_events",
                    "select": ["client_no", "units"],
                    "time": {"mode": "events_in_range", "from": start, "to": end},
                    "limit": 20_000,
                },
            ),
            _step(
                layer,
                name="customers",
                source="crm",
                table="crm.customers",
                spec={
                    "from": "crm.customers",
                    "select": ["cust_id", "customer_status"],
                    "limit": 10_000,
                },
            ),
        ),
        joins=(JoinStep(join="usage_to_customers", left="usage", right="customers", how="inner"),),
        output=FederatedOutput(
            group_by=(GroupColumn(step="customers", column="customer_status"),),
            aggregates=(
                LocalAggregate(
                    step="usage",
                    column="units",
                    agg=Agg.SUM,
                    alias="usage_units_sum",
                    measure="usage_units",
                ),
            ),
            order_by=("customer_status",),
            limit=50,
        ),
    )


def average_by_status(
    world: World, rounding: Any, status_of: Mapping[str, str] = BILLING_STATE
) -> dict[str, float]:
    """The billing average one rounding rule produces, computed from the world, not from us."""
    totals: dict[str, int] = {}
    counts: dict[str, int] = {}
    for account in world.billing.accounts:
        if account.credit_limit is None:
            continue
        status = status_of[account.state]
        totals[status] = totals.get(status, 0) + int(rounding(account.credit_limit))
        counts[status] = counts.get(status, 0) + 1
    return {status: totals[status] / counts[status] for status in totals}


# --------------------------------------------------------------------------------------
# (a) "The same number means different things in different systems."
# --------------------------------------------------------------------------------------


def test_claim_a_same_number_different_meaning(
    governed: Governor,
    layer: SemanticLayer,
    sources: Mapping[str, FakeAdapter],
    pinned_world: World,
    tmp_path: Path,
) -> None:
    """README: a limit rounded to 5000, stored raw, and stored divided by ten must join correctly.

    FAILS IF: the cross-source average differs from E01's independently derived answer; or any
    single source's normalisation is wrong (billing's raw column not rounded half-up, erp's
    ``limit_k`` not multiplied by ten); or the fixture stops discriminating -- a half-even
    implementation must produce a *different* number for the same data, which is asserted below.
    """
    _, questions = read_questions()

    # 1. The cross-source answer: crm is the identity, billing and erp prove the customer exists.
    runner = FederatedExecutor(
        governed, layer=layer, clock=ManualClock(EXEC_TS), spill_dir=tmp_path / "crm_spill"
    )
    with runner:
        result = runner.execute(credit_plan(layer), user=USER)
    assert result.ok, result.error
    assert result.table is not None
    assert result.quality.rows_dropped == 0, "every customer exists in all three sources"
    federated = {
        str(row["customer_status"]): float(row["credit_limit_r5000_avg"])
        for row in result.table.to_pylist()
    }
    eval_answer = questions["avg_limit_by_status_crm"]["answer"]
    assert federated == {
        str(row[0]): float(row[1]) for row in cast(Sequence[Sequence[Any]], eval_answer["rows"])
    }

    # 2. Each storage form, through its own compiled SQL, lands on its own E01 answer.
    erp_view = sorted(
        {(row.cust_no, row.status_code, row.limit_k) for row in pinned_world.erp.customers}
    )
    deduped = FakeAdapter(
        name="erp",
        tables={
            "customers": pa.table(
                {
                    "cust_no": [row[0] for row in erp_view],
                    "status_code": [row[1] for row in erp_view],
                    "limit_k": [row[2] for row in erp_view],
                }
            )
        },
    )
    per_source: dict[str, FakeAdapter] = {
        "crm": sources["crm"],
        "billing": sources["billing"],
        "erp": deduped,
    }
    try:
        for question_id in (
            "avg_limit_by_status_crm",
            "avg_limit_by_status_billing",
            "avg_limit_by_status_erp",
        ):
            question = questions[question_id]
            source = cast(str, question["spec"]["from"]).split(".")[0]
            outcome = execute_spec(
                governor=Governor(adapters(per_source)),
                layer=layer,
                sources=per_source,
                source=source,
                spec=question["spec"],
            )
            assert_e01_answer(outcome, question["answer"])
    finally:
        deduped.close()

    # 3. The test discriminates: half-even is a different number for this data, and the code
    #    produced half-up. Without this, a wrong rounding rule could pass just as easily.
    half_up = average_by_status(pinned_world, round_half_up)
    half_even = average_by_status(pinned_world, round_half_even)
    differing = [
        status
        for status in half_up
        if not math.isclose(half_up[status], half_even[status], rel_tol=1e-15)
    ]
    assert differing, "the world has no .5 boundary row: this test would not discriminate"
    assert any(row.rounds_differently for row in pinned_world.expected.boundary_rows)
    billing = questions["avg_limit_by_status_billing"]["answer"]
    assert half_up == {
        str(row[0]): float(row[1]) for row in cast(Sequence[Sequence[Any]], billing["rows"])
    }


# --------------------------------------------------------------------------------------
# (b) "Agents write SQL that melts the primary."
# --------------------------------------------------------------------------------------


def test_claim_b_a_partition_scanning_query_never_reaches_the_source(
    governed: Governor, layer: SemanticLayer, sources: Mapping[str, FakeAdapter]
) -> None:
    """README: "refuses a query with no partition key, tells the agent exactly what to add ... and
    never reaches the source until the query is acceptable".

    FAILS IF: the query is accepted (it would scan every monthly partition); or a refusal comes
    back without an actionable ``fix``; or the adapter was called anyway; or the fix the error
    suggests does not actually work.
    """
    _, questions = read_questions()
    refused = execute_spec(
        governor=governed,
        layer=layer,
        sources=sources,
        source="billing",
        spec=questions["usage_events_all_partitions_refused"]["spec"],
    )
    assert refused.error is not None, "a query with no window on the partition key must be refused"
    assert refused.error.code == ErrorCode.PARTITION_SCAN_ALL.value
    assert "billing.usage_events" in refused.error.problem
    assert refused.error.fix and '"time"' in refused.error.fix
    # Zero adapter calls: nothing was compiled into SQL, so nothing could be sent.
    assert refused.calls == {"crm": 0, "billing": 0, "erp": 0}, refused.calls
    compiled = compile_one(
        layer, "billing", questions["usage_events_all_partitions_refused"]["spec"]
    )
    assert compiled.fragment is None, "a refused spec must not produce SQL"

    # The fix the refusal named works, and it answers E01's question for the same table.
    accepted = execute_spec(
        governor=governed,
        layer=layer,
        sources=sources,
        source="billing",
        spec=questions["events_count_window"]["spec"],
    )
    assert accepted.ok, accepted.error
    assert_e01_answer(accepted, questions["events_count_window"]["answer"])
    assert accepted.calls == {"crm": 0, "billing": 1, "erp": 0}, accepted.calls


# --------------------------------------------------------------------------------------
# (c) "The same fetch happens twenty times."
# --------------------------------------------------------------------------------------


def test_claim_c_twenty_identical_fetches_cost_one_source_call(
    governed: Governor,
    layer: SemanticLayer,
    sources: Mapping[str, FakeAdapter],
    tmp_path: Path,
) -> None:
    """README: identical calls repeat nothing, and each served piece carries its ``fetched_at``.

    FAILS IF: the second..twenty-first identical call makes any new source call; or a repeated
    answer reports no ``fetched_at`` per source; or a *changed* parameter is wrongly served from
    the cache.
    """
    clock = ManualClock(EXEC_TS)
    runner = FederatedExecutor(
        governed, layer=layer, clock=clock, spill_dir=tmp_path / "cache_spill"
    )
    plan = usage_plan(layer, start="2025-11-01", end="2026-01-01")
    with runner:
        first = runner.execute(plan, user=USER)
        assert first.ok, first.error
        after_first = call_counts(sources)
        assert after_first == {"crm": 1, "billing": 1, "erp": 0}, after_first
        stamps = {step.name: step.fetched_at for step in first.quality.steps}
        assert set(stamps) == {"usage", "customers"}
        assert all(stamp == EXEC_TS for stamp in stamps.values())

        for _ in range(20):
            again = runner.execute(plan, user=USER)
            assert again.ok, again.error
            assert call_counts(sources) == after_first, "a cache hit must not call the source"
            for step in again.quality.steps:
                assert step.fetched is False
                assert step.cache_status == "hit"
                assert step.fetched_at == stamps[step.name]

        # A changed parameter is a different fragment: billing is fetched again, and the unchanged
        # crm fragment is still a hit. Mixed freshness, reported per step rather than averaged.
        moved = runner.execute(usage_plan(layer, start="2025-12-01", end="2026-01-01"), user=USER)
        assert moved.ok, moved.error
        after_move = call_counts(sources)
        assert after_move["billing"] == after_first["billing"] + 1
        assert after_move["crm"] == after_first["crm"]
        statuses = {step.name: step.cache_status for step in moved.quality.steps}
        assert statuses == {"usage": "miss", "customers": "hit"}, statuses
        assert {step.name: step.fetched_at for step in moved.quality.steps}["usage"] == EXEC_TS


# --------------------------------------------------------------------------------------
# (d) "Late-arriving rows silently vanish."
# --------------------------------------------------------------------------------------


def test_claim_d_late_arriving_rows_are_counted_not_dropped(
    governed: Governor,
    layer: SemanticLayer,
    sources: Mapping[str, FakeAdapter],
    pinned_world: World,
) -> None:
    """README: the ingestion-time bound is derived from the measured lag and the event-time
    predicate stays, "so no late row is dropped".

    Proven in both directions on the same fixture: the product's query counts the rows that
    happened before the requested instant and arrived after it, and the naive
    ``start_time >= since`` query run on the same adapter loses exactly those rows.

    FAILS IF: the product's answer equals the naive one (a late row vanished); or the derived bound
    would exclude a late row; or the compiled SQL lost the ingestion-time predicate.
    """
    _, questions = read_questions()

    # The shipped question still answers what E01 derived for it.
    shipped = execute_spec(
        governor=governed,
        layer=layer,
        sources=sources,
        source="billing",
        spec=questions["new_since_resync"]["spec"],
    )
    assert_e01_answer(shipped, questions["new_since_resync"]["answer"])

    # The instant that question cannot show: one where the world plants a row whose event time is
    # before it and whose insertion time is not. Taking the earliest such insertion is deterministic
    # and needs no assumption about the generator beyond the lag it declares.
    events = pinned_world.billing.usage_events
    lagged = [
        event for event in events if event.insertion_time - event.start_time > timedelta(seconds=60)
    ]
    assert lagged, "the world plants no delayed row: this claim could not be tested here"
    since = min(event.insertion_time for event in lagged)
    late = [event for event in events if event.start_time < since <= event.insertion_time]
    assert late, "the chosen instant has no row straddling it"

    spec = {
        "from": "billing.usage_events",
        "select": [{"measure": "usage_units", "agg": "count"}],
        "time": {"mode": "new_since", "since": since.strftime("%Y-%m-%dT%H:%M:%SZ")},
        "limit": 10,
    }
    outcome = execute_spec(
        governor=governed, layer=layer, sources=sources, source="billing", spec=spec
    )
    assert outcome.ok, outcome.error
    delivered = int(outcome.rows[0]["usage_units_count"])
    by_insertion = sum(1 for event in events if event.insertion_time >= since)
    by_event = sum(1 for event in events if event.start_time >= since)
    assert delivered == by_insertion
    assert by_event == delivered - len(late), (by_event, delivered, len(late))

    # The same fixture, the naive filter: it really loses rows.
    naive = run_sql(
        governed,
        "billing",
        "SELECT count(units) AS n FROM usage_events "
        f"WHERE start_time >= CAST('{since.strftime('%Y-%m-%d %H:%M:%S')}' AS TIMESTAMP)",
    )
    assert int(naive[0]["n"]) == by_event
    assert int(naive[0]["n"]) < delivered

    # Why it loses them: the derived bound the planner adds is wide enough, and the event-time
    # predicate is still there rather than replaced by an ingestion-time range.
    compiled = compile_one(layer, "billing", spec)
    assert compiled.ok, compiled.error
    sql = cast(str, compiled.fragment.sql if compiled.fragment else "")
    assert "insertion_time >=" in sql, sql
    assert "start_time >=" in sql, "the original event-time predicate must stay"
    assert compiled.time_plan is not None
    derived = compiled.time_plan.derived_bound
    assert derived is not None, "a 'new_since' question must derive an event-time bound"
    assert derived == since - timedelta(seconds=pinned_world.expected.max_lag_seconds)
    assert all(event.start_time >= derived for event in late), (
        "the derived bound would drop a late-arriving row"
    )


# --------------------------------------------------------------------------------------
# (e) "Read-only is not a promise, it is three layers."
# --------------------------------------------------------------------------------------


def test_claim_e_a_write_is_refused_by_the_guard_and_the_refusal_is_audited(
    governed: Governor, sources: Mapping[str, FakeAdapter], layer: SemanticLayer, tmp_path: Path
) -> None:
    """README: the AST guard refuses writes, "also on SQL the server compiles", and refusals are in
    the hash-chained audit log.

    Layer 1 (read-only account) and layer 3 (read-only transaction) are asserted against real
    PostgreSQL in ``test_postgres_e2e.py``; layer 2 and the audit trail are here.

    FAILS IF: a DELETE/UPDATE/DDL statement reaches the adapter; or the refusal is not an
    ``ErrorResponse`` with a ``fix``; or the audit row is missing, or its chain does not verify.
    """
    _, questions = read_questions()
    statement = cast(str, questions["delete_usage_events"]["statement"]["sql"])
    writes = (
        statement,
        "UPDATE billing.usage_events SET units = 0",
        "INSERT INTO billing.usage_events (event_id) VALUES (1)",
        "DROP TABLE billing.usage_events",
        "CREATE TABLE scratch (i INTEGER)",
        "TRUNCATE billing.usage_events",
    )
    for write in writes:
        blocked = guard(write, "duckdb")
        assert not blocked.ok, write
        assert blocked.error is not None
        assert blocked.error.code == ErrorCode.WRITE_BLOCKED.value, write
        assert blocked.error.fix, f"{write}: a refusal must carry a fix"

    # Through the governor: the refusal is the same object shape the agent sees, and no source ran.
    fragment = Fragment(
        source="billing",
        dialect="duckdb",
        sql=statement,
        tables=("billing.usage_events",),
        exec_ts=EXEC_TS,
    )
    with pytest.raises(GovernorRefusal) as excinfo:
        governed.execute(source="billing", user=USER, fragment=fragment, limits=LIMITS)
    response = excinfo.value.response
    assert response.code == ErrorCode.WRITE_BLOCKED.value
    assert response.fix
    assert sources["billing"].calls == [], "a write must never reach an adapter"

    # The SQL we compile ourselves passes the same guard: the second layer is not only for agents.
    compiled = compile_one(
        layer,
        "billing",
        {"from": "billing.accounts", "select": ["client_no"], "limit": 10},
    )
    assert compiled.ok, compiled.error
    assert compiled.fragment is not None
    passed = guard(cast(str, compiled.fragment.sql), "duckdb")
    assert passed.ok, passed.error

    # The trail: a blocked attempt is a row, and the chain verifies.
    store = Store(tmp_path / "meta.sqlite")
    store.initialize()
    try:
        audit = AuditLog(store, clock=lambda: EXEC_TS)
        record = audit.record_refusal(
            user_id=USER, tool="query", error=response, sources=("billing",)
        )
        assert record.outcome is AuditOutcome.BLOCKED
        assert record.error_code is ErrorCode.WRITE_BLOCKED
        assert record.blocked_attempts == 1
        verdict = audit.verify_chain()
        assert verdict.ok, verdict.describe()
        assert verdict.checked == 1
    finally:
        store.close()


# --------------------------------------------------------------------------------------
# (f) "A weak model cannot choose between 40 tools or read a schema."
# --------------------------------------------------------------------------------------


def test_claim_f_the_context_pack_is_small_deterministic_and_says_what_it_dropped(
    layer: SemanticLayer, tmp_path: Path
) -> None:
    """README: "the context the agent needs ... is injected automatically, inside a token budget,
    with what was left out reported".

    FAILS IF: the rendered block exceeds its budget; or two builds for the same question differ by
    one byte; or an item that matched is missing without appearing in ``omitted``; or the facts the
    answer depends on (the measure and its per-source conversions, the value map) are not there.
    """
    store = Store(tmp_path / "meta.sqlite")
    store.initialize()
    try:
        index = ContextIndex(store, clock=lambda: EXEC_TS)
        index.build(layer, examples=EXAMPLES)

        first = build_pack(PACK_QUESTION, layer=layer, index=index, examples=EXAMPLES)
        second = build_pack(PACK_QUESTION, layer=layer, index=index, examples=EXAMPLES)
        assert first.text, "a question about a measure in the layer must get context"
        assert first.text == second.text, "the pack must be byte-identical for the same question"
        assert first.refs() == second.refs()
        assert first.tokens_estimated <= first.budget_tokens
        assert estimate_tokens(first.text) == first.tokens_estimated

        refs = set(first.refs())
        assert "measure:credit_limit_r5000" in refs, sorted(refs)
        assert "value:customer_status" in refs, sorted(refs)
        measure = next(
            item
            for item in first.items
            if item.kind is PackKind.MEASURE and item.ref == "credit_limit_r5000"
        )
        assert "erp.customers.limit_k={col} * 10" in measure.text
        assert "round half_up to 5000" in measure.text

        # The per-kind cap dropped examples, and every dropped item is reported with its reason.
        assert len(first.items_of_kind(PackKind.EXAMPLE)) <= MAX_EXAMPLES
        assert first.omitted, "the example cap must report what it left out"
        for item in first.omitted:
            assert item.reason in {OmissionReason.BUDGET, OmissionReason.LIMIT}

        # A budget too small for the layer drops the lowest ranked facts, and the block says how
        # many, so the agent is never told it has the whole picture when it does not.
        tight = build_pack(
            PACK_QUESTION,
            layer=layer,
            index=index,
            examples=EXAMPLES,
            budget_tokens=MIN_PACK_BUDGET,
        )
        assert tight.tokens_estimated <= tight.budget_tokens
        dropped = [item for item in tight.omitted if item.reason is OmissionReason.BUDGET]
        assert dropped, f"the layer is too small to prove the budget cut: {tight.refs()}"
        assert f"omitted: {len(dropped)}" in tight.text, tight.text
        assert set(tight.refs()) <= set(first.refs()), "a tighter budget may only drop facts"
        assert len(tight.items) < len(first.items), "a tighter budget keeps fewer facts"
    finally:
        store.close()


# --------------------------------------------------------------------------------------
# (g) "Agents write SQL that melts the primary" -- the protections, measured
# --------------------------------------------------------------------------------------


class _Gate:
    """Holds every caller inside the adapter until the test releases them.

    The governor's slots are the only thing that can let a caller through, so the number of
    callers inside is exactly the number of slots held -- that is what makes the peak an
    observation of admission rather than of the thread scheduler.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._entered = 0
        self.peak = 0
        self._released = threading.Event()

    def enter(self) -> None:
        with self._cond:
            self._entered += 1
            self.peak = max(self.peak, self._entered)
            self._cond.notify_all()
        self._released.wait()

    def leave(self) -> None:
        with self._cond:
            self._entered -= 1

    def wait_entered(self, count: int, timeout: float = 10.0) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: self._entered >= count, timeout)

    def release(self) -> None:
        self._released.set()


class _GatedSource:
    """A real adapter whose ``execute`` parks inside :class:`_Gate`; one DuckDB per call."""

    kind = "fake"

    def __init__(self, gate: _Gate, table: pa.Table) -> None:
        self._gate = gate
        self._table = table
        self.calls: list[str] = []

    def dialect(self) -> str:
        return "duckdb"

    def probe(self) -> Any:
        return FakeAdapter(name="fake", tables={"t": self._table}).probe()

    def extract_catalog(self) -> Any:
        return FakeAdapter(name="fake", tables={"t": self._table}).extract_catalog()

    def execute(self, fragment: Fragment, limits: Limits) -> ArrowStream:
        self._gate.enter()
        try:
            self.calls.append(fragment.sql or "")
            return FakeAdapter(name="fake", tables={"t": self._table}).execute(fragment, limits)
        finally:
            self._gate.leave()

    def cancel(self, handle: ArrowStream) -> None:
        close = getattr(handle, "cancel", None)
        if callable(close):
            close()
        else:
            handle.close()


class _ScriptedClock:
    """A clock that moves only when the test (or a stream) moves it: no sleeps."""

    def __init__(self) -> None:
        self._mono = 0.0
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        with self._lock:
            return self._mono

    def now(self) -> datetime:
        return EXEC_TS

    def jump(self, to: float) -> None:
        with self._lock:
            self._mono = to


class _DeadlinePassingStream:
    """A stream that moves the clock past the deadline right after the first batch."""

    def __init__(self, inner: ArrowStream, clock: _ScriptedClock, jump_to: float) -> None:
        self.schema = inner.schema
        self._inner = inner
        self._clock = clock
        self._jump_to = jump_to
        self.cancelled = False

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        for index, batch in enumerate(self._inner):
            yield batch
            if index == 0:
                self._clock.jump(self._jump_to)

    def cancel(self) -> None:
        self.cancelled = True
        self._inner.close()

    def close(self) -> None:
        self._inner.close()


class _CancelSpy:
    """A real adapter that records every ``cancel`` it is told to perform."""

    kind = "fake"

    def __init__(self, clock: _ScriptedClock) -> None:
        self._clock = clock
        self._inner = FakeAdapter(name="fake", tables={"t": pa.table({"n": list(range(2_000))})})
        self.cancels: list[Any] = []

    def dialect(self) -> str:
        return self._inner.dialect()

    def probe(self) -> Any:
        return self._inner.probe()

    def extract_catalog(self) -> Any:
        return self._inner.extract_catalog()

    def execute(self, fragment: Fragment, limits: Limits) -> ArrowStream:
        stream = self._inner.execute(fragment, limits)
        return _DeadlinePassingStream(stream, self._clock, jump_to=limits.timeout_s * 10)

    def cancel(self, handle: ArrowStream) -> None:
        self.cancels.append(handle)
        self._inner.cancel(handle)


def _bare_fragment() -> Fragment:
    return Fragment(
        source="fake", dialect="duckdb", sql="SELECT * FROM t", tables=("t",), exec_ts=EXEC_TS
    )


def test_claim_g_the_governor_caps_concurrency_and_really_cancels() -> None:
    """README: "Concurrency, quotas, timeouts and a circuit breaker sit in front of every source."

    Two observations, both measured rather than asserted from configuration:

    * nine concurrent callers, one source, a per-source ceiling of three -- the peak held inside
      the adapter must be exactly three, and the admission report must agree;
    * a stream that passes its deadline must be *cancelled at the adapter* (its own ``cancel`` is
      called on the live stream), not abandoned, and the slot must be released.

    FAILS IF: the peak exceeds the ceiling (the primary is being hammered); or the timeout returns
    without the adapter's cancel being invoked; or the slot stays held afterwards.
    """
    # --- the concurrency ceiling -----------------------------------------------------
    gate = _Gate()
    table = pa.table({"n": list(range(5))})
    admission = Admission(AdmissionPolicy(max_global=9, per_user=9, per_source=3, max_wait_s=20.0))
    source = _GatedSource(gate, table)
    burst = Governor(adapters({"fake": source}), admission=admission)

    outcomes: list[str] = []
    lock = threading.Lock()

    def ask() -> None:
        try:
            burst.execute(source="fake", user="alice", fragment=_bare_fragment(), limits=LIMITS)
            with lock:
                outcomes.append("ok")
        except GovernorRefusal as refusal:
            with lock:
                outcomes.append(refusal.code)

    threads = [threading.Thread(target=ask) for _ in range(9)]
    for thread in threads:
        thread.start()
    assert gate.wait_entered(3), "the governor let fewer than per_source callers through"
    assert gate.peak == 3, f"per-source ceiling of 3 was breached: peak {gate.peak}"
    gate.release()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()

    assert outcomes == ["ok"] * 9, outcomes
    peaks = {peak.name: peak for peak in admission.peaks()}
    assert peaks["fake"].peak == 3
    assert peaks["fake"].limit == 3
    assert peaks["fake"].in_use == 0

    # --- the deadline cancels, it does not abandon -----------------------------------
    clock = _ScriptedClock()
    spy = _CancelSpy(clock)
    timed = Governor(adapters({"fake": spy}), clock=clock)
    with pytest.raises(GovernorRefusal) as excinfo:
        timed.execute(
            source="fake",
            user="alice",
            fragment=_bare_fragment(),
            limits=Limits(timeout_s=2.0, max_rows=50_000, max_bytes=1 << 20, max_partitions=8),
        )
    refusal = excinfo.value
    assert refusal.code == ErrorCode.TIMEOUT.value
    assert refusal.response.fix
    assert len(spy.cancels) == 1, "the adapter's cancel path was not used: the call was abandoned"
    handle = spy.cancels[0]
    assert getattr(handle, "cancelled", False) is True
    assert all(peak.in_use == 0 for peak in timed.governance().peaks), "the slot leaked"
