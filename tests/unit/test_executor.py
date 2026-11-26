"""Tests for the federated executor (task C17).

Four kinds of evidence, because "it joined" is not one of them:

* **the answer**, compared with the ground truth task E01 derived independently for the same seeded
  world: the cross-source average credit limit by status (the canonical value recovered from the
  source that stores it canonically, with the other two sources proving identity) and the
  cross-source usage sum, whose expected numbers are in ``tests/eval/questions.yaml``;
* **the bad news**, which is the point of the report: the planted orphan ``client_no`` rows a join
  cannot match are counted and sampled, a partner's duplicate keys are reported as multiplication
  rather than silently inflating an average, an unmapped categorical code shows up as a NULL group;
* **the protections**: a fragment over its row cap is refused with ``COST_TOO_HIGH`` before the join
  runs, and a failing source produces no answer at all instead of half a join;
* **the cache**: a second identical call touches no adapter and reports per-source ``fetched_at``,
  and the same call after the floor TTL refetches.

The three sources are three F05 fake adapters (one per logical source), each a real in-process
DuckDB holding one storage form of the D02 world. The executor has its own DuckDB for the join; no
assertion here reaches into an adapter's tables.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from ruamel.yaml import YAML

from mdqm.compiler import CompileContext, CompiledQuery, compile_spec
from mdqm.errors import ErrorCode
from mdqm.executor import (
    ColumnCheck,
    FederatedExecutor,
    FederatedOutput,
    FederatedPlan,
    FragmentStep,
    GroupColumn,
    JoinStep,
    LocalAggregate,
    StepCache,
)
from mdqm.executor.models import QualityReport
from mdqm.governor.core import Governor
from mdqm.models import Limits
from mdqm.semantic.loader import load
from mdqm.semantic.models import SemanticLayer
from mdqm.spec.models import Agg
from mdqm.spec.validate import parse_spec
from mdqm.testing.fake_adapter import FakeAdapter, table_from_rows
from mdqm.testing.world import CrmCustomer, World, WorldConfig, build_world

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "tests" / "eval" / "questions.yaml"

#: One execution timestamp for every plan in this module (C02: one value per execution).
EXEC_TS = datetime(2026, 11, 25, 15, 0, tzinfo=UTC)

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

#: Caps for every test step: high enough that only the cap test trips them.
LIMITS = Limits(timeout_s=30.0, max_rows=50_000, max_bytes=64 << 20, max_partitions=64)

#: The semantic layer these tests federate under: the compiler-test layer plus the two customer
#: identity joins the D02 world needs (crm<->billing by padded key, crm<->erp by integer key).
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


# --- fixtures ----------------------------------------------------------------------------


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
    document = cast(Mapping[str, Any], YAML(typ="safe").load(QUESTIONS.read_text(encoding="utf-8")))
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


@pytest.fixture
def sources(pinned_world: World) -> dict[str, FakeAdapter]:
    """Three fake adapters, one per logical source, each holding that source's storage form."""
    return {
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


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(EXEC_TS)


@pytest.fixture
def executor(
    sources: Mapping[str, FakeAdapter],
    layer: SemanticLayer,
    clock: ManualClock,
    tmp_path: Path,
) -> Iterator[FederatedExecutor]:
    """An executor over the three sources, with its own DuckDB and a spill dir under tmp_path."""
    governor = Governor(dict(sources))
    runner = FederatedExecutor(
        governor, layer=layer, clock=clock, spill_dir=tmp_path / "duckdb_spill"
    )
    with runner:
        yield runner


# --- helpers -----------------------------------------------------------------------------


def compile_for(
    layer: SemanticLayer,
    source: str,
    spec: Mapping[str, Any],
    *,
    exec_ts: datetime = EXEC_TS,
    max_rows: int = 50_000,
) -> CompiledQuery:
    """Parse and compile one spec for one source; the fragment or the refusal."""
    context = CompileContext(
        layer=layer,
        source=source,
        dialect="duckdb",
        exec_ts=exec_ts,
        columns=COLUMNS,
        max_rows=max_rows,
    )
    return compile_spec(parse_spec(spec), context)


def fragment_of(layer: SemanticLayer, source: str, spec: Mapping[str, Any]) -> Any:
    """The compiled fragment, failing the test with the refusal when the compiler said no."""
    compiled = compile_for(layer, source, spec)
    assert compiled.ok, compiled.error
    return compiled.fragment


def step(
    layer: SemanticLayer,
    *,
    name: str,
    source: str,
    table: str,
    spec: Mapping[str, Any],
    limits: Limits = LIMITS,
    cache: bool = True,
    checks: tuple[ColumnCheck, ...] = (),
    dedupe_on: tuple[str, ...] = (),
) -> FragmentStep:
    """One plan step: a compiled fragment plus the rules that apply to running it."""
    return FragmentStep(
        name=name,
        source=source,
        table=table,
        fragment=fragment_of(layer, source, spec),
        limits=limits,
        cache=(
            StepCache(version_hash=layer.version_hash, mutability="append_rarely")
            if cache
            else None
        ),
        checks=checks,
        dedupe_on=dedupe_on,
    )


def credit_steps(
    layer: SemanticLayer,
    *,
    crm_limits: Limits = LIMITS,
    checks: tuple[ColumnCheck, ...] = (),
    erp_dedupe: bool = True,
) -> tuple[FragmentStep, ...]:
    """The three storage forms of the same customers: crm (canonical), billing (raw), erp (/10)."""
    return (
        step(
            layer,
            name="crm",
            source="crm",
            table="crm.customers",
            spec={
                "from": "crm.customers",
                "select": ["cust_id", "customer_status", "credit_limit"],
                "limit": 5_000,
            },
            limits=crm_limits,
            checks=checks,
        ),
        step(
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
        step(
            layer,
            name="erp",
            source="erp",
            table="erp.customers",
            spec={
                "from": "erp.customers",
                "select": ["cust_no", "customer_status", "limit_k"],
                "limit": 5_000,
            },
            dedupe_on=("cust_no",) if erp_dedupe else (),
        ),
    )


def credit_output() -> FederatedOutput:
    """``avg credit_limit_r5000 by customer_status``, the question the MVP must answer."""
    return FederatedOutput(
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
    )


def credit_plan(
    layer: SemanticLayer,
    *,
    crm_limits: Limits = LIMITS,
    checks: tuple[ColumnCheck, ...] = (),
    erp_how: str = "semi",
    erp_dedupe: bool = True,
) -> FederatedPlan:
    """The cross-source plan: crm is the identity, billing and erp prove the customer exists.

    ``erp_how="semi"`` keeps one row per customer: the erp world has duplicate rows on purpose, and
    a row-producing join would double count them (which the ``left`` variant of this plan reports).
    """
    return FederatedPlan(
        exec_ts=EXEC_TS,
        steps=credit_steps(layer, crm_limits=crm_limits, checks=checks, erp_dedupe=erp_dedupe),
        joins=(
            JoinStep(join="crm_accounts", left="crm", right="billing", how="left"),
            JoinStep(join="crm_erp", left="crm", right="erp", how=cast(Any, erp_how)),
        ),
        output=credit_output(),
    )


def usage_plan(layer: SemanticLayer) -> FederatedPlan:
    """``usage units by customer status, November and December 2025``: the E01 cross-source join."""
    return FederatedPlan(
        exec_ts=EXEC_TS,
        steps=(
            step(
                layer,
                name="usage",
                source="billing",
                table="billing.usage_events",
                spec={
                    "from": "billing.usage_events",
                    "select": ["client_no", "units"],
                    "time": {"mode": "events_in_range", "from": "2025-11-01", "to": "2026-01-01"},
                    "limit": 20_000,
                },
            ),
            step(
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


def rows_of(report: QualityReport) -> dict[str, Any]:
    """One step's report by name, for a terse assertion."""
    return {step.name: step for step in report.steps}


def calls(sources: Mapping[str, FakeAdapter]) -> dict[str, int]:
    """How many times each adapter was asked for rows."""
    return {name: len(adapter.calls) for name, adapter in sources.items()}


# --- the answer --------------------------------------------------------------------------


def test_cross_source_average_matches_ground_truth(
    executor: FederatedExecutor, layer: SemanticLayer, pinned_world: World
) -> None:
    """The federated average by status equals the canonical answer, exactly.

    The limit comes from the source that stores the canonical rounded value; billing's raw column
    and erp's ``limit_k`` prove the customer exists in both, so identity across three sources is
    what is being tested, not a mixed average of three storage forms (which would be wrong).
    """
    result = executor.execute(credit_plan(layer), user="c17")
    assert result.ok, result.error
    assert result.table is not None
    assert result.table.column_names == ["customer_status", "credit_limit_r5000_avg"]
    answer = result.table.to_pylist()
    truth = pinned_world.expected.avg_rounded_limit_by_status
    assert [row["customer_status"] for row in answer] == sorted(truth)
    for row in answer:
        expected = truth[cast(str, row["customer_status"])]
        # ``MeasureAnswer.avg`` is rounded for reading; total/count is the exact value, and E01's
        # independently derived answer for the same question is the second opinion.
        assert math.isclose(
            cast(float, row["credit_limit_r5000_avg"]),
            expected.total / expected.count,
            rel_tol=1e-15,
        ), f"{row['customer_status']}: {row['credit_limit_r5000_avg']} != {expected.total}"
    _, questions = read_questions()
    eval_answer = questions["avg_limit_by_status_crm"]["answer"]
    assert result.table.to_pylist() == [
        dict(zip(eval_answer["columns"], values, strict=True)) for values in eval_answer["rows"]
    ]
    # Every customer exists in all three sources, so the report must say nothing was lost.
    assert result.quality.rows_dropped == 0
    assert {join.join for join in result.quality.joins} == {"crm_accounts", "crm_erp"}
    # The value map is part of the answer: a source code ('A') never leaks into a group.
    assert all(isinstance(row["customer_status"], str) for row in answer)


#: One plan per storage form: the source, its fragment's columns, and the column the measure lives
#: on. The expected answer for each is E01's independently derived one.
NORMALISATION_CASES: tuple[tuple[str, str, str, tuple[str, ...], str], ...] = (
    (
        "avg_limit_by_status_crm",
        "crm",
        "crm.customers",
        ("cust_id", "customer_status", "credit_limit"),
        "credit_limit",
    ),
    (
        "avg_limit_by_status_billing",
        "billing",
        "billing.accounts",
        ("client_no", "customer_status", "credit_limit"),
        "credit_limit",
    ),
    (
        "avg_limit_by_status_erp",
        "erp",
        "erp.customers",
        ("cust_no", "customer_status", "limit_k"),
        "limit_k",
    ),
)


@pytest.mark.parametrize(
    ("question", "step_name", "table", "columns", "measure_column"),
    NORMALISATION_CASES,
    ids=[case[0] for case in NORMALISATION_CASES],
)
def test_local_normalisation_reproduces_each_eval_answer(
    executor: FederatedExecutor,
    layer: SemanticLayer,
    pinned_world: World,
    question: str,
    step_name: str,
    table: str,
    columns: tuple[str, ...],
    measure_column: str,
) -> None:
    """Each storage form normalised locally lands on E01's independently derived average.

    The compiler normalises an aggregate it emits; a federated plan hands rows between sources, so
    the unit conversion of a row value happens here, through the same semantic template
    (identity / ``ROUND(x/5000)*5000`` / ``x*10``). erp also has to dedupe on its declared key, or
    its six planted duplicate rows would double count.
    """
    _, questions = read_questions()
    expected = questions[question]["answer"]
    plan = FederatedPlan(
        exec_ts=EXEC_TS,
        steps=(
            step(
                layer,
                name=step_name,
                source=step_name,
                table=table,
                spec={"from": table, "select": list(columns), "limit": 5_000},
                dedupe_on=("cust_no",) if step_name == "erp" else (),
            ),
        ),
        output=FederatedOutput(
            group_by=(GroupColumn(step=step_name, column="customer_status"),),
            aggregates=(
                LocalAggregate(
                    step=step_name,
                    column=measure_column,
                    agg=Agg.AVG,
                    alias="credit_limit_r5000_avg",
                    measure="credit_limit_r5000",
                ),
            ),
            order_by=("customer_status",),
            limit=50,
        ),
    )
    result = executor.execute(plan, user="c17")
    assert result.ok, result.error
    assert result.table is not None
    assert result.table.column_names == list(expected["columns"])
    for row, values in zip(result.table.to_pylist(), expected["rows"], strict=True):
        assert row["customer_status"] == values[0]
        assert math.isclose(
            cast(float, row["credit_limit_r5000_avg"]), cast(float, values[1]), rel_tol=1e-15
        ), f"{row['customer_status']}: {row['credit_limit_r5000_avg']} != {values[1]}"
    (report,) = result.quality.steps
    if step_name == "erp":
        assert report.deduplicated == len(pinned_world.expected.duplicate_cust_nos)
    else:
        assert report.deduplicated == 0


def test_report_keeps_a_fetched_at_per_source(
    executor: FederatedExecutor, layer: SemanticLayer, clock: ManualClock
) -> None:
    """Each source reports when its rows were fetched, and the fragment SQL that produced them."""
    result = executor.execute(credit_plan(layer), user="c17")
    assert result.ok
    report = rows_of(result.quality)
    assert set(report) == {"crm", "billing", "erp"}
    assert {step.source for step in result.quality.steps} == {"crm", "billing", "erp"}
    for entry in result.quality.steps:
        assert entry.fetched_at == clock.now()
        assert entry.fetched is True
        assert entry.cache_status == "miss"
        assert entry.rows > 0
        assert entry.sql is not None


def test_usage_by_status_matches_the_eval_answer(
    executor: FederatedExecutor, layer: SemanticLayer, pinned_world: World
) -> None:
    """The cross-source usage question answers exactly what E01 derived, and counts the orphans.

    ``billing.usage_events.client_no`` is a padded string and ``crm.customers.cust_id`` an integer:
    the approved join owns the conversion, and the events whose client has no customer are dropped
    by the inner join and reported as drops instead of disappearing silently.
    """
    _, questions = read_questions()
    expected = questions["usage_units_by_status_window"]["answer"]
    result = executor.execute(usage_plan(layer), user="c17")
    assert result.ok, result.error
    assert result.table is not None
    assert result.table.column_names == list(expected["columns"])
    assert result.table.to_pylist() == [
        dict(zip(expected["columns"], row, strict=True)) for row in expected["rows"]
    ]

    (join,) = result.quality.joins
    assert join.join == "usage_to_customers"
    # The window holds 45 of the 200 planted orphan events; count them from the world, not from us.
    known = {customer.client_no for customer in pinned_world.expected.customers}
    window_from, window_to = datetime(2025, 11, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)
    orphans = [
        event
        for event in pinned_world.billing.usage_events
        if window_from <= event.start_time < window_to and event.client_no not in known
    ]
    assert join.dropped == len(orphans)
    assert 0 < join.dropped <= pinned_world.expected.orphan_event_count
    assert join.reason.startswith("left rows whose billing.usage_events.client_no")
    # The sample is the planted orphans, never a customer who merely failed a conversion.
    assert set(join.dropped_keys) <= set(pinned_world.expected.orphan_client_nos)
    assert 0 < len(join.dropped_keys) <= 5


# --- the bad news the report must carry ---------------------------------------------------


def test_join_reports_multiplied_rows_from_duplicate_erp_keys(
    executor: FederatedExecutor, layer: SemanticLayer, pinned_world: World
) -> None:
    """A row-producing join onto erp's planted duplicates reports the inflation, not hides it.

    This is why the shipped plan joins erp as a semi join: a ``left`` join doubles those customers
    and would answer a different average. The report says so in rows.
    """
    result = executor.execute(credit_plan(layer, erp_how="left", erp_dedupe=False), user="c17")
    assert result.ok, result.error
    (_, erp) = result.quality.joins
    assert erp.join == "crm_erp"
    assert erp.multiplied == len(pinned_world.expected.duplicate_cust_nos)
    assert erp.dropped == 0
    assert "multiplied" in " ".join(result.quality.notes)


def test_unmapped_categorical_value_becomes_a_null_group(
    layer: SemanticLayer, pinned_world: World, tmp_path: Path
) -> None:
    """A status code the value map does not cover is reported as a NULL group, not misfiled."""
    rows = list(pinned_world.crm.customers)
    # The compiler's CASE falls through to NULL for a code no value map declares.
    replaced = [
        CrmCustomer(
            cust_id=row.cust_id,
            full_name=row.full_name,
            status="X",
            credit_limit=row.credit_limit,
            region=row.region,
            signup_date=row.signup_date,
        )
        for row in rows[:3]
    ]
    runner = FederatedExecutor(
        Governor(
            {
                "crm": FakeAdapter(
                    name="crm", tables={"customers": table_from_rows([*replaced, *rows[3:]])}
                ),
                "billing": FakeAdapter(
                    name="billing",
                    tables={"accounts": table_from_rows(pinned_world.billing.accounts)},
                ),
                "erp": FakeAdapter(
                    name="erp", tables={"customers": table_from_rows(pinned_world.erp.customers)}
                ),
            }
        ),
        layer=layer,
        spill_dir=tmp_path / "duckdb_spill",
    )
    with runner:
        result = runner.execute(credit_plan(layer), user="c17")
    assert result.ok, result.error
    assert result.table is not None
    (group,) = result.quality.null_groups
    assert group.column == "customer_status"
    assert group.step == "crm"
    assert group.rows == 3
    assert "value map" in group.reason


def test_sanity_check_flags_a_normalisation_off_the_rounding_step(
    executor: FederatedExecutor, layer: SemanticLayer, pinned_world: World
) -> None:
    """erp stores ``limit_k`` in units of 500, so a check for multiples of 5000 must fire.

    The rule under test: the semantic layer says the measure is rounded to 5000, so a source whose
    normalised column is not a multiple of 5000 is a conversion bug — invisible inside an average,
    counted here.
    """
    check = ColumnCheck(column="limit_k", multiple_of=5_000)
    # erp's own fragment selects only the key, so the check runs on a plan that keeps limit_k.
    erp_step = FragmentStep(
        name="erp",
        source="erp",
        table="erp.customers",
        fragment=fragment_of(
            layer,
            "erp",
            {"from": "erp.customers", "select": ["cust_no", "limit_k"], "limit": 5_000},
        ),
        limits=LIMITS,
        checks=(check,),
    )
    plan = credit_plan(layer).model_copy(
        update={"steps": (*credit_plan(layer).steps[:2], erp_step)}
    )
    result = executor.execute(plan, user="c17")
    assert result.ok, result.error
    (violation,) = result.quality.sanity
    expected = sum(1 for row in pinned_world.erp.customers if row.limit_k % 5_000 != 0)
    assert violation.step == "erp"
    assert violation.column == "limit_k"
    assert violation.multiple_of == 5_000
    assert violation.offending == expected
    assert violation.examples


# --- the protections ----------------------------------------------------------------------


def test_fragment_over_its_row_cap_is_refused_before_the_join(
    executor: FederatedExecutor,
    layer: SemanticLayer,
    sources: Mapping[str, FakeAdapter],
) -> None:
    """A fragment past its cap is ``COST_TOO_HIGH`` with a narrowing fix, and no join runs."""
    tiny = Limits(timeout_s=30.0, max_rows=10, max_bytes=1 << 20, max_partitions=64)
    result = executor.execute(credit_plan(layer, crm_limits=tiny), user="c17")
    assert result.error is not None
    assert result.error.code == ErrorCode.COST_TOO_HIGH.value
    assert result.table is None
    assert result.sql is None
    assert not result.quality.complete
    assert "Narrow the filter" in result.error.fix
    # The refusal happened on the first step, so the later sources were never called at all.
    assert calls(sources) == {"crm": 1, "billing": 0, "erp": 0}
    refused = rows_of(result.quality)["crm"]
    assert refused.row_cap_hit is True
    assert refused.rows > tiny.max_rows


def test_a_failing_source_produces_no_answer_rather_than_half_a_join(
    layer: SemanticLayer, pinned_world: World, tmp_path: Path
) -> None:
    """One source down: no table, the code says which failure, and the report stays honest."""
    broken = FakeAdapter(
        name="billing",
        tables={"accounts": table_from_rows(pinned_world.billing.accounts)},
        fail_with=OSError("connection refused"),
    )
    runner = FederatedExecutor(
        Governor(
            {
                "crm": FakeAdapter(
                    name="crm", tables={"customers": table_from_rows(pinned_world.crm.customers)}
                ),
                "billing": broken,
                "erp": FakeAdapter(
                    name="erp", tables={"customers": table_from_rows(pinned_world.erp.customers)}
                ),
            }
        ),
        layer=layer,
        spill_dir=tmp_path / "duckdb_spill",
    )
    with runner:
        result = runner.execute(credit_plan(layer), user="c17")
    assert result.table is None
    assert result.error is not None
    assert result.error.code == ErrorCode.SOURCE_UNAVAILABLE.value
    assert not result.quality.complete
    assert [step.name for step in result.quality.steps] == ["crm", "billing"]
    assert "no answer is produced" in result.quality.notes[0]


def test_a_plan_naming_another_join_fails_loudly(
    executor: FederatedExecutor, layer: SemanticLayer
) -> None:
    """A join attached to the wrong steps is a plan bug, and it says so instead of counting 0."""
    plan = credit_plan(layer)
    wrong = plan.model_copy(
        update={"joins": (JoinStep(join="crm_accounts", left="crm", right="erp", how="left"),)}
    )
    with pytest.raises(ValueError, match="connects billing.accounts and crm"):
        executor.execute(wrong, user="c17")


# --- the cache ----------------------------------------------------------------------------


def test_second_identical_call_touches_no_source(
    executor: FederatedExecutor,
    layer: SemanticLayer,
    sources: Mapping[str, FakeAdapter],
    clock: ManualClock,
) -> None:
    """The same plan twice: one fetch, then zero adapter calls, with per-source freshness."""
    first = executor.execute(credit_plan(layer), user="c17")
    assert first.ok, first.error
    fetched = calls(sources)
    assert fetched == {"crm": 1, "billing": 1, "erp": 1}
    stamp = {step.name: step.fetched_at for step in first.quality.steps}

    second = executor.execute(credit_plan(layer), user="c17")
    assert second.ok, second.error
    assert calls(sources) == fetched, "a cache hit must not call the source"
    assert second.table is not None and first.table is not None
    assert second.table.to_pylist() == first.table.to_pylist()
    for entry in second.quality.steps:
        assert entry.fetched is False
        assert entry.cache_status == "hit"
        assert entry.fetched_at == stamp[entry.name]
        assert entry.age_s == 0.0
    assert executor.cache.size() == 3

    # Past the floor TTL a whole-fragment read is stale again, and the source is asked once more.
    clock.advance(timedelta(seconds=31))
    third = executor.execute(credit_plan(layer), user="c17")
    assert third.ok, third.error
    assert calls(sources) == {"crm": 2, "billing": 2, "erp": 2}
    assert {entry.cache_status for entry in third.quality.steps} == {"stale"}


def test_a_step_without_a_cache_policy_always_fetches(
    executor: FederatedExecutor, layer: SemanticLayer, sources: Mapping[str, FakeAdapter]
) -> None:
    """No mutability class, no reuse: a table nobody classified is treated as mutable."""
    plan = credit_plan(layer)
    uncached = plan.model_copy(
        update={"steps": tuple(step.model_copy(update={"cache": None}) for step in plan.steps)}
    )
    executor.execute(uncached, user="c17")
    executor.execute(uncached, user="c17")
    assert calls(sources) == {"crm": 2, "billing": 2, "erp": 2}
    assert executor.cache.size() == 0
