"""Measurements for the claims in ``README.md``: how long each stage actually takes.

    .venv/Scripts/python.exe scripts/bench.py
    .venv/Scripts/python.exe scripts/bench.py --json
    .venv/Scripts/python.exe scripts/bench.py --runs 50

This prints numbers; it does not gate on them. Wall-clock time depends on the machine, the Python
build and whatever else is running, so an assertion on a millisecond value is a flaky test wearing
a benchmark's clothes. What is asserted here is only what must hold for the measurements to mean
anything:

* the compiler produced SQL, and it carries the two conversions the layer declares (``ROUND(`` for
  billing's raw limit, ``* 10`` for erp's divided one) -- otherwise "compile latency" measured a
  failure path;
* the cache hit really made zero source calls, and the miss really made one -- otherwise "cache-hit
  latency" measured a second fetch;
* the context pack stayed inside its token budget and was byte-identical when built twice;
* the federated join returned rows and the same numbers twice;
* all 10 000 round-trips and roundings came back exactly -- otherwise the throughput is a
  measurement of the wrong arithmetic.

No Docker, no network, no clock as an input: the world is the seeded one of task D02 and every
random choice comes from a fixed seed, so two runs on the same machine measure the same work.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from mdqm.compiler import CompileContext, CompiledQuery, compile_spec
from mdqm.context.index import ContextIndex
from mdqm.context.pack import DEFAULT_PACK_BUDGET, build_pack
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
from mdqm.governor import Governor
from mdqm.models import Limits
from mdqm.semantic.loader import load
from mdqm.semantic.models import SemanticLayer
from mdqm.semantic.values import resolve_value, reverse_value
from mdqm.spec import parse_spec
from mdqm.spec.models import Agg
from mdqm.store import Store
from mdqm.testing.fake_adapter import FakeAdapter, table_from_rows
from mdqm.testing.world import World, WorldConfig, build_world, round_half_up

#: Fixed seed: the same work every run (no clock, no environment in the inputs).
SEED: Final = 20260101

#: One execution timestamp for every compile, as the server does (C02).
EXEC_TS: Final = datetime(2026, 11, 25, 15, 0, tzinfo=UTC)

#: The catalog the compiler may resolve names against.
COLUMNS: Final[dict[str, tuple[str, ...]]] = {
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

#: The layer the numbers are produced under: three storage forms of one measure, one value map with
#: both languages, one partitioned event table.
LAYER_YAML: Final = """
version: 1

sources:
  crm: { adapter: postgresql }
  billing: { adapter: postgresql }
  erp: { adapter: postgresql }

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
  - name: usage_to_customers
    left: billing.usage_events.client_no
    right: crm.customers.cust_id
    on: "LPAD(CAST({right} AS VARCHAR), 10, '0') = {left}"
    cardinality: many_to_one
"""

#: The windowed aggregate the compile/plan/execute/join numbers use.
USAGE_SPEC: Final[dict[str, Any]] = {
    "from": "billing.usage_events",
    "select": [{"measure": "usage_units", "agg": "sum"}],
    "time": {"mode": "events_in_range", "from": "2025-11-01", "to": "2026-01-01"},
    "limit": 1_000,
}

#: The same window without an aggregate, so the execute number includes fetching real rows.
EXECUTE_SPEC: Final[dict[str, Any]] = {
    "from": "billing.usage_events",
    "select": ["event_id", "client_no", "units"],
    "time": {"mode": "events_in_range", "from": "2025-11-01", "to": "2026-01-01"},
    "limit": 50_000,
}

#: The label the value map must resolve in either language, and the source column it is asked for.
VALUE_COLUMN: Final = "crm.customers.status"


# --- timing -------------------------------------------------------------------------------


@dataclass(slots=True)
class Measurement:
    """One row of the printed table: what was measured, and its distribution."""

    name: str
    samples: list[float] = field(default_factory=list)
    note: str = ""

    def add(self, seconds: float) -> None:
        self.samples.append(seconds)

    @property
    def p50(self) -> float:
        return _percentile(self.samples, 50.0)

    @property
    def p95(self) -> float:
        return _percentile(self.samples, 95.0)

    @property
    def n(self) -> int:
        return len(self.samples)

    def as_json(self) -> dict[str, Any]:
        """Milliseconds, the unit the printed table uses, so the two outputs cannot disagree."""
        return {
            "name": self.name,
            "n": self.n,
            "p50_ms": self.p50 * 1000,
            "p95_ms": self.p95 * 1000,
            "min_ms": (min(self.samples) if self.samples else 0.0) * 1000,
            "max_ms": (max(self.samples) if self.samples else 0.0) * 1000,
            "note": self.note,
        }


def _percentile(values: Sequence[float], q: float) -> float:
    """The nearest-rank percentile, so no sample is invented between two measurements."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round(q / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[rank]


def _time(call: Callable[[], Any], runs: int) -> list[float]:
    """Run ``call`` ``runs`` times, returning one wall-clock sample per run."""
    samples: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started)
    return samples


def _record(target: Measurement, samples: Sequence[float]) -> Measurement:
    for sample in samples:
        target.add(sample)
    return target


# --- the fixtures -------------------------------------------------------------------------


def _layer(directory: Path) -> SemanticLayer:
    path = directory / "layer.yaml"
    path.write_text(LAYER_YAML, encoding="utf-8")
    return load(path)


def _compile(layer: SemanticLayer, source: str, spec: dict[str, Any]) -> CompiledQuery:
    return compile_spec(
        parse_spec(spec),
        CompileContext(
            layer=layer,
            source=source,
            dialect="duckdb",
            exec_ts=EXEC_TS,
            columns=COLUMNS,
            max_rows=50_000,
        ),
    )


def _step(
    layer: SemanticLayer, *, name: str, source: str, table: str, spec: dict[str, Any]
) -> FragmentStep:
    compiled = _compile(layer, source, spec)
    assert compiled.ok and compiled.fragment is not None, compiled.error
    return FragmentStep(
        name=name,
        source=source,
        table=table,
        fragment=compiled.fragment,
        limits=Limits(timeout_s=60.0, max_rows=100_000, max_bytes=64 << 20, max_partitions=64),
        cache=StepCache(version_hash=layer.version_hash, mutability="append_rarely"),
    )


def _join_plan(layer: SemanticLayer) -> FederatedPlan:
    """The cross-source question: usage events joined to crm customers, summed by status."""
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
                    "time": {"mode": "events_in_range", "from": "2025-11-01", "to": "2026-01-01"},
                    "limit": 50_000,
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


def _credit_plan(layer: SemanticLayer) -> FederatedPlan:
    """The single-source plan the cache number uses: enough work to be worth caching."""
    return FederatedPlan(
        exec_ts=EXEC_TS,
        steps=(
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
        ),
        output=FederatedOutput(
            group_by=(GroupColumn(step="billing", column="customer_status"),),
            aggregates=(
                LocalAggregate(
                    step="billing",
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


def _sources(world: World) -> dict[str, FakeAdapter]:
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


# --- the measurements ---------------------------------------------------------------------


def bench_compile(layer: SemanticLayer, runs: int) -> Measurement:
    """Parse + resolve + normalise + plan time and partitions + guard: one spec to one fragment."""
    measured = Measurement("compile+plan")
    compiled = _compile(layer, "billing", USAGE_SPEC)
    assert compiled.ok and compiled.fragment is not None, compiled.error
    sql = compiled.fragment.sql or ""
    assert "start_time" in sql, "the window did not reach the SQL"
    assert compiled.partition_plan is not None and compiled.partition_plan.ok

    # The two conversions of claim (a), compiled rather than assumed.
    billing = _compile(
        layer,
        "billing",
        {"from": "billing.accounts", "select": [{"measure": "credit_limit_r5000", "agg": "avg"}]},
    )
    erp = _compile(
        layer,
        "erp",
        {"from": "erp.customers", "select": [{"measure": "credit_limit_r5000", "agg": "avg"}]},
    )
    # Half-up is rendered explicitly (FLOOR(ABS(q) + 0.5) * SIGN(q)), not as a bare ROUND(), so the
    # rule is pinned in the SQL rather than left to the engine's default.
    for result, needle in ((billing, "FLOOR(ABS("), (erp, "* 10")):
        assert result.ok and result.fragment is not None, result.error
        assert needle in (result.fragment.sql or ""), (needle, result.fragment.sql)

    _record(measured, _time(lambda: _compile(layer, "billing", USAGE_SPEC), runs))
    time_predicates = 0 if compiled.time_plan is None else len(compiled.time_plan.predicates)
    partition_predicates = (
        0 if compiled.partition_plan is None else len(compiled.partition_plan.predicates)
    )
    measured.note = f"{len(sql)} chars SQL, {time_predicates}+{partition_predicates} predicates"
    return measured


def bench_execute(
    sources: Mapping[str, FakeAdapter], layer: SemanticLayer, runs: int
) -> Measurement:
    """Adapter execute + full drain on the fake DuckDB sources: the source-side cost."""
    measured = Measurement("execute (fake source)")
    compiled = _compile(layer, "billing", EXECUTE_SPEC)
    assert compiled.ok and compiled.fragment is not None, compiled.error
    limits = Limits(timeout_s=60.0, max_rows=100_000, max_bytes=64 << 20, max_partitions=64)
    adapter = sources["billing"]

    def once() -> int:
        stream = adapter.execute(compiled.fragment, limits)
        try:
            rows = sum(batch.num_rows for batch in stream)
        finally:
            stream.close()
        assert rows > 0
        return rows

    rows_per_run = once()
    _record(measured, _time(lambda: once(), runs))
    measured.note = f"{rows_per_run} rows fetched per run"
    return measured


def bench_cache(
    sources: dict[str, FakeAdapter], layer: SemanticLayer, runs: int, directory: Path
) -> tuple[Measurement, Measurement, int]:
    """The same fragment served twice: the miss, then the hits, with the source-call count."""
    miss = Measurement("cache miss (fetch)")
    hit = Measurement("cache hit")
    governor = Governor(dict(sources))
    billing = sources["billing"]
    runner = FederatedExecutor(
        governor, layer=layer, spill_dir=directory / "cache_spill", path=directory / "cache.duckdb"
    )
    plan = _credit_plan(layer)
    try:
        before = len(billing.calls)
        started = time.perf_counter()
        runner.execute(plan, user="bench")
        miss.add(time.perf_counter() - started)
        calls_on_miss = len(billing.calls) - before
        assert calls_on_miss >= 1, "the first execution did not reach the source at all"
        after_miss = len(billing.calls)

        def one_hit() -> None:
            outcome = runner.execute(plan, user="bench")
            assert outcome.ok, outcome.error

        _record(hit, _time(one_hit, runs))
        assert len(billing.calls) == after_miss, "a cache hit must not call the source"
        # The engine may split a fragment into segments, so the count is what it is; what must
        # hold is that the hits add nothing to it.
        miss.note = f"{calls_on_miss} source call(s)"
        hit.note = f"{runs} hits, 0 new calls"
        return miss, hit, calls_on_miss
    finally:
        runner.close()


def bench_join(
    sources: dict[str, FakeAdapter], layer: SemanticLayer, runs: int, directory: Path
) -> Measurement:
    """The federated join: two fragments, one DuckDB join, the aggregate, the quality report."""
    measured = Measurement("federated join")
    governor = Governor(dict(sources))
    runner = FederatedExecutor(governor, layer=layer, spill_dir=directory / "join_spill")
    answers: list[list[dict[str, Any]]] = []
    try:
        plan = _join_plan(layer)

        def once() -> None:
            outcome = runner.execute(plan, user="bench")
            assert outcome.ok, outcome.error
            assert outcome.table is not None and outcome.table.num_rows > 0
            answers.append(outcome.table.to_pylist())

        _record(measured, _time(once, runs))
        assert all(answer == answers[0] for answer in answers), "the join is not deterministic"
        measured.note = f"{len(answers[0])} groups, {answers[0][0]['customer_status']} first"
        return measured
    finally:
        runner.close()


def bench_context(layer: SemanticLayer, runs: int, directory: Path) -> tuple[Measurement, int, int]:
    """Building the Context Pack for one question: latency and the tokens it costs."""
    measured = Measurement("context pack build")
    store = Store(directory / "meta.sqlite")
    store.initialize()
    try:
        index = ContextIndex(store, clock=lambda: EXEC_TS)
        index.build(layer)
        question = "average credit limit by customer status"
        tokens: list[int] = []

        def once() -> None:
            pack = build_pack(question, layer=layer, index=index, budget_tokens=DEFAULT_PACK_BUDGET)
            assert pack.text, "the pack is empty: nothing to measure"
            assert pack.tokens_estimated <= pack.budget_tokens, pack.tokens_estimated
            tokens.append(pack.tokens_estimated)

        _record(measured, _time(once, runs))
        first = build_pack(question, layer=layer, index=index, budget_tokens=DEFAULT_PACK_BUDGET)
        second = build_pack(question, layer=layer, index=index, budget_tokens=DEFAULT_PACK_BUDGET)
        assert first.text == second.text, "the pack is not deterministic"
        assert len(set(tokens)) == 1, "the token count moved between runs"
        measured.note = f"{first.tokens_estimated} tokens, {len(first.items)} facts"
        return measured, first.tokens_estimated, len(first.items)
    finally:
        store.close()


def bench_roundtrip(
    layer: SemanticLayer, values: int = 10_000, passes: int = 3
) -> tuple[Measurement, int]:
    """Label -> canonical -> the source's code -> canonical, for 10k values, and the same for a
    round-half-up applied twice."""
    measured = Measurement("round/normalise round-trip")
    value_map = layer.value_maps["customer_status"]
    rng = random.Random(SEED)  # noqa: S311 - a benchmark needs reproducible inputs, not secrets
    # Every spelling of the crm column's three values: canonical, Hebrew synonym, source code.
    labels = ["active", "פעיל", "A", "Active", "churned", "עזב", "C", "suspended", "מושעה", "S"]
    pool = [rng.choice(labels) for _ in range(values)]
    amounts = [rng.randrange(0, 5_000_000) for _ in range(values)]

    resolved = 0
    for _ in range(passes):
        started = time.perf_counter()
        for label in pool:
            forward = resolve_value(value_map, label, column=VALUE_COLUMN)
            assert forward.ok, label
            canonical = forward.canonical
            assert canonical is not None
            backward = reverse_value(value_map, canonical, column=VALUE_COLUMN)
            assert backward.ok and backward.literals, canonical
            code = backward.literals[0].literal
            again = resolve_value(value_map, code, column=VALUE_COLUMN)
            assert again.canonical == canonical, (label, code, again)
            resolved += 1
        for amount in amounts:
            once = round_half_up(amount)
            assert round_half_up(once) == once, amount
            assert once % 5_000 == 0, amount
        measured.add(time.perf_counter() - started)

    assert resolved == values * passes
    per_value = statistics.median(measured.samples) / values * 1e6
    measured.note = f"{values} labels + {values} roundings/pass, {per_value:.2f} us/value"
    return measured, resolved


# --- output -------------------------------------------------------------------------------


def _print_table(measurements: Sequence[Measurement]) -> None:
    width = max(len(m.name) for m in measurements)
    print(f"{'stage'.ljust(width)}  {'p50':>10}  {'p95':>10}  {'n':>4}  note")
    print(f"{'-' * width}  {'-' * 10}  {'-' * 10}  {'-' * 4}  {'-' * 24}")
    for measurement in measurements:
        print(
            f"{measurement.name.ljust(width)}  "
            f"{measurement.p50 * 1000:>8.3f}ms  "
            f"{measurement.p95 * 1000:>8.3f}ms  "
            f"{measurement.n:>4}  {measurement.note}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run every measurement, print them, and exit non-zero only on a broken invariant."""
    parser = argparse.ArgumentParser(description="mdqm stage benchmarks (no Docker, no network).")
    parser.add_argument("--runs", type=int, default=30, help="samples per timed stage (default 30)")
    parser.add_argument("--json", action="store_true", help="machine-readable output instead")
    arguments = parser.parse_args(argv)
    if arguments.runs < 1:
        parser.error("--runs must be at least 1")

    world = build_world(WorldConfig.small(seed=SEED))
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        layer = _layer(directory)
        sources = _sources(world)
        try:
            compile_measurement = bench_compile(layer, arguments.runs)
            execute_measurement = bench_execute(sources, layer, max(3, arguments.runs // 3))
            miss_measurement, cache_measurement, calls_on_miss = bench_cache(
                sources, layer, arguments.runs, directory
            )
            context_measurement, tokens, facts = bench_context(layer, arguments.runs, directory)
            join_measurement = bench_join(sources, layer, max(3, arguments.runs // 3), directory)
            roundtrip_measurement, resolved = bench_roundtrip(layer)
        finally:
            for adapter in sources.values():
                adapter.close()

    measurements = [
        compile_measurement,
        execute_measurement,
        miss_measurement,
        cache_measurement,
        context_measurement,
        join_measurement,
        roundtrip_measurement,
    ]
    # Invariants that make the numbers mean something; never a threshold on a duration.
    assert compile_measurement.n and compile_measurement.p95 > 0
    assert miss_measurement.n == 1 and cache_measurement.n and cache_measurement.p95 > 0
    assert calls_on_miss >= 1
    assert tokens > 0 and facts > 0
    assert resolved == 10_000 * 3

    if arguments.json:
        print(
            json.dumps(
                {
                    "runs": arguments.runs,
                    "seed": SEED,
                    "world": {
                        "customers": world.config.customers,
                        "events": world.config.event_count,
                    },
                    "measurements": [measurement.as_json() for measurement in measurements],
                    "context": {"tokens": tokens, "facts": facts},
                    "cache": {"source_calls_on_first_fetch": calls_on_miss},
                    "values_round_tripped": resolved,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"mdqm benchmarks: seed {SEED}, {arguments.runs} runs")
        print(
            f"world: {world.config.customers} customers, {world.config.event_count} usage events\n"
        )
        _print_table(measurements)
        print(
            "\nread these as shapes, not targets: same machine, same seed, same work."
            f"\ncontext pack: {tokens} tokens / {facts} facts; values round-tripped: {resolved}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
