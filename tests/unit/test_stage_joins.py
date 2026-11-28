"""Tests for stage D7 join candidates (S07): name, type, containment, and the format in between.

The task's acceptance criteria come first, then the rules that make a proposal trustworthy:

* **the planted conversion is recovered** -- ``crm.customers.cust_id`` against
  ``billing.accounts.client_no`` is proposed *with* the ten-character zero pad, both containments
  at 1.0, and the ``on`` clause that an approval would write;
* **orphans are reported, never hidden** -- the child table carrying the planted orphan keys has
  them counted *and* named, in distinct keys and in rows, against the world's ground truth;
* **a similar name is not a join** -- a name-similar pair with unrelated values, two integer
  columns whose ranges merely overlap, and a three-value code column that is fully contained in
  every id column are all rejected, because a wrong join is worse than no join;
* **nothing is measured dishonestly** -- a value set that hit the distinct cap makes its pairs
  unmeasurable and says so, and the stage refuses to run outside a quiet window;
* **approval is a copy, not a rewrite** -- a proposal is an ``approved_joins`` entry that the
  review queue can write and the loader can load back.

No test sleeps and no test contacts a host: values come from ``FakeAdapter``'s in-process DuckDB
over the D02 world, and the synthetic sources are Arrow tables built in the test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from mdqm.config.quiet import QuietWindow
from mdqm.models.catalog import CatalogSnapshot, Column, ForeignKey, NormalizedType, Table
from mdqm.pipeline.core import PipelineContext, StageStatus, run_pipeline
from mdqm.pipeline.stages.joins import (
    Containment,
    ConversionKind,
    FormatConversion,
    JoinCandidate,
    JoinMeasurement,
    JoinStage,
    ValueSet,
    conversions,
    distinct_query,
    measure,
    name_similarity,
)
from mdqm.pipeline.stages.profiling import QuietWindowClosed
from mdqm.review.queue import ProposalKind, ProposalStatus, ReviewQueue
from mdqm.semantic.loader import load
from mdqm.semantic.models import JoinCardinality
from mdqm.store.db import Store
from mdqm.testing.fake_adapter import FakeAdapter, world_tables
from mdqm.testing.world import WorldConfig, build_world

#: 2026-05-01 10:00 UTC = 13:00 in Asia/Jerusalem: inside the working day, so not quiet.
NOON = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)

#: 21:00-07:00 local: the window that makes a 13:00 pass a refusal.
NIGHT = QuietWindow(tz="Asia/Jerusalem", start=time(21, 0), end=time(7, 0))

CRM_CUST_ID = "crm.customers_crm.cust_id"
BILLING_CLIENT_NO = "billing.accounts.client_no"
ERP_CUST_NO = "erp.customers_erp.cust_no"
EVENTS_CLIENT_NO = "billing.usage_events.client_no"


class ScriptedClock:
    """A clock a test drives: ``monotonic`` ticks, ``now`` is one injected instant."""

    def __init__(self, tick: float = 0.25, now: datetime = NOON) -> None:
        self._tick = tick
        self._reading = 0.0
        self._now = now

    def monotonic(self) -> float:
        self._reading += self._tick
        return self._reading

    def now(self) -> datetime:
        return self._now


def _table(**columns: Sequence[Any]) -> pa.Table:
    """One Arrow table from keyword columns: the shape every synthetic source in this file has."""
    return pa.table({name: list(values) for name, values in columns.items()})


@pytest.fixture(scope="module")
def world() -> Any:
    """One generated organisation, built once: the synthetic data every acceptance case uses."""
    return build_world(WorldConfig.small())


@pytest.fixture
def store(tmp_path: Path) -> Any:
    """An initialized metadata store under ``tmp_path``."""
    opened = Store(tmp_path / "mdqm.sqlite")
    opened.initialize()
    try:
        yield opened
    finally:
        opened.close()


@pytest.fixture
def adapters(world: Any) -> Any:
    """Three sources, each holding only its own dataset -- the way the real organisation looks."""
    tables = world_tables(world)
    billing = ("accounts", "usage_events", "accounts_hist")
    built = {
        "crm": FakeAdapter(name="crm", tables={"customers_crm": tables["customers_crm"]}),
        "billing": FakeAdapter(name="billing", tables={name: tables[name] for name in billing}),
        "erp": FakeAdapter(name="erp", tables={"customers_erp": tables["customers_erp"]}),
    }
    try:
        yield built
    finally:
        for adapter in built.values():
            adapter.close()


def _ctx(
    tmp_path: Path, store: Store, *, quiet: tuple[QuietWindow, ...] = (), now: datetime = NOON
) -> PipelineContext:
    """A context rooted in ``tmp_path`` with an injected clock and no model."""
    return PipelineContext(
        store=store,
        exec_ts=now,
        semantic_root=tmp_path / "semantic",
        quiet_windows=quiet,
        clock=ScriptedClock(now=now),
    )


def _measure(
    ctx: PipelineContext, adapters: Mapping[str, FakeAdapter], **kwargs: Any
) -> JoinMeasurement:
    """Measure every source and return what the stage found."""
    return JoinStage(adapters, **kwargs).measure(ctx)


def _pair(measurement: JoinMeasurement, left: str, right: str) -> JoinCandidate:
    """The candidate for two labels in either direction: the sort order of a pair is not a fact."""
    found = next(
        (item for item in measurement.candidates if {item.left, item.right} == {left, right}),
        None,
    )
    assert found is not None, sorted(item.name for item in measurement.candidates)
    return found


def _orphan_side(containment: Containment) -> Containment:
    """The containment as measured; the helper exists so the assertion below reads in one line."""
    return containment


def _value_set(
    values: Sequence[str],
    *,
    label: tuple[str, str, str] = ("s", "t", "k"),
    counts: Sequence[int] | None = None,
    normalized_type: NormalizedType = NormalizedType.STRING,
    unique: bool = False,
) -> ValueSet:
    """A hand-built value set, for the pure conversion and measurement tests."""
    source, table, column = label
    return ValueSet(
        source=source,
        table=table,
        column=column,
        normalized_type=normalized_type,
        values=tuple(values),
        counts=tuple(counts or [1] * len(values)),
        declared_unique=unique,
    )


class DeclaredKeys(FakeAdapter):
    """A fake source whose catalog also declares a primary key and a foreign key.

    ``FakeAdapter`` reports what Arrow knows, and Arrow has no keys, so the simulated source says
    so itself. That is all D7 needs to tell "the catalog promises this" from "the values suggest
    it".
    """

    def extract_catalog(self) -> CatalogSnapshot:
        catalog = super().extract_catalog()
        tables: list[Table] = []
        for table in catalog.tables:
            if table.name == "orders":
                tables.append(
                    table.model_copy(
                        update={
                            "foreign_keys": (
                                ForeignKey(
                                    columns=("customer_id",),
                                    ref_table="customers",
                                    ref_columns=("cust_id",),
                                    name="orders_customer_fk",
                                ),
                            )
                        }
                    )
                )
            elif table.name == "customers":
                tables.append(table.model_copy(update={"primary_key": ("cust_id",)}))
            else:
                tables.append(table)
        return catalog.model_copy(update={"tables": tuple(tables)})


# --- acceptance: the planted cross-source key ------------------------------------------------


def test_cust_id_joins_client_no_with_the_ten_character_zero_pad(
    tmp_path: Path, store: Store, adapters: Mapping[str, FakeAdapter]
) -> None:
    """crm's integer key reaches billing's VARCHAR(10): the proposal carries LPAD, width 10."""
    measurement = _measure(_ctx(tmp_path, store), adapters)
    candidate = _pair(measurement, CRM_CUST_ID, BILLING_CLIENT_NO)
    conversion = candidate.containment.conversion
    assert conversion.kind is ConversionKind.ZERO_PAD
    assert conversion.width == 10
    assert "LPAD" in conversion.on
    assert candidate.containment.containment_left == pytest.approx(1.0)
    assert candidate.containment.containment_right == pytest.approx(1.0)
    assert candidate.containment.jaccard == pytest.approx(1.0)
    assert candidate.containment.unmatched_left_distinct == 0
    assert candidate.containment.unmatched_right_distinct == 0


def test_the_zero_pad_proposal_is_an_approved_join_entry(
    tmp_path: Path, store: Store, adapters: Mapping[str, FakeAdapter]
) -> None:
    """A proposal is only useful if approval is a copy: the real model must accept it."""
    candidate = _pair(_measure(_ctx(tmp_path, store), adapters), CRM_CUST_ID, BILLING_CLIENT_NO)
    join = candidate.as_approved_join()
    assert join.left == candidate.left
    assert join.right == candidate.right
    assert join.on == candidate.containment.conversion.on
    assert join.provenance is not None
    assert join.provenance.match_rate == pytest.approx(1.0)
    assert any("containment" in line for line in candidate.evidence)


def test_cust_id_joins_erp_cust_no_with_no_conversion(
    tmp_path: Path, store: Store, adapters: Mapping[str, FakeAdapter]
) -> None:
    """The erp key is the same integer: identity, both containments at 1.0."""
    candidate = _pair(_measure(_ctx(tmp_path, store), adapters), CRM_CUST_ID, ERP_CUST_NO)
    assert candidate.containment.conversion.kind is ConversionKind.IDENTITY
    assert candidate.containment.conversion.on == "{left} = {right}"
    assert candidate.containment.containment_left == pytest.approx(1.0)
    assert candidate.containment.containment_right == pytest.approx(1.0)


def test_planted_orphans_are_counted_and_named(
    tmp_path: Path, store: Store, adapters: Mapping[str, FakeAdapter], world: Any
) -> None:
    """``usage_events`` carries keys with no customer: the proposal reports them, never hides them.

    Both the distinct count and the row count are asserted against the world's own ground truth, so
    the numbers are checked against an independent computation rather than against the stage.
    """
    measurement = _measure(_ctx(tmp_path, store), adapters)
    candidate = _pair(measurement, EVENTS_CLIENT_NO, CRM_CUST_ID)
    containment = _orphan_side(candidate.containment)
    orphans = sorted(world.expected.orphan_client_nos)
    assert world.expected.orphan_event_count > 0
    assert containment.jaccard < 1.0
    # One side is the orphan-bearing event table, whichever orientation the sort produced.
    assert {containment.unmatched_left_distinct, containment.unmatched_right_distinct} == {
        0,
        len(orphans),
    }
    assert {containment.unmatched_left_rows, containment.unmatched_right_rows} == {
        0,
        world.expected.orphan_event_count,
    }
    named = containment.unmatched_left_examples or containment.unmatched_right_examples
    assert set(named) <= set(orphans)
    assert any("unmatched keys" in line for line in candidate.evidence)
    assert containment.containment_left < 1.0 or containment.containment_right < 1.0


# --- acceptance: false positives stay out -----------------------------------------------------


def test_a_three_value_code_column_contained_in_an_id_is_not_proposed(
    tmp_path: Path, store: Store, adapters: Mapping[str, FakeAdapter]
) -> None:
    """erp's 1/2/3 status code is *fully contained* in crm's id range: still not a join.

    This is the coincidence the distinct-size ratio exists for. Containment alone would propose it.
    """
    measurement = _measure(_ctx(tmp_path, store), adapters)
    status = "erp.customers_erp.status_code"
    assert measurement.candidate(status, CRM_CUST_ID) is None
    assert measurement.candidate(CRM_CUST_ID, status) is None


def test_overlapping_integer_ranges_are_not_proposed(tmp_path: Path, store: Store) -> None:
    """Two integer columns whose ranges merely intersect score about 0.33, far under the floor."""
    alpha = FakeAdapter(name="alpha", tables={"orders": _table(order_no=range(1, 501))})
    beta = FakeAdapter(name="beta", tables={"tickets": _table(order_no=range(250, 751))})
    try:
        measurement = _measure(_ctx(tmp_path, store), {"alpha": alpha, "beta": beta})
    finally:
        alpha.close()
        beta.close()
    assert measurement.candidates == ()
    assert measurement.counts["pairs_rejected"] == 1


def test_a_similar_name_with_unrelated_values_is_not_proposed(tmp_path: Path, store: Store) -> None:
    """``customer_id`` on both sides, disjoint values: the name alone approves nothing."""
    left = FakeAdapter(name="crm", tables={"customers": _table(customer_id=range(1, 201))})
    right = FakeAdapter(
        name="warehouse", tables={"customers": _table(customer_id=range(10_001, 10_201))}
    )
    try:
        measurement = _measure(_ctx(tmp_path, store), {"crm": left, "warehouse": right})
    finally:
        left.close()
        right.close()
    assert name_similarity("customer_id", "customer_id") == pytest.approx(1.0)
    assert measurement.candidates == ()


# --- conversions measured, not assumed --------------------------------------------------------


def test_conversions_are_offered_only_when_the_values_support_them() -> None:
    """Case table: what each value shape offers, so an inapplicable conversion never gets scored."""
    padded = _value_set([f"{n:010d}" for n in range(1, 4)])
    narrow = _value_set(["1", "2", "3"], normalized_type=NormalizedType.INTEGER)
    letters = _value_set(["acme", "globex"])
    prefixed = _value_set(["CUST-000001", "CUST-000002"])

    offered = {item.kind for item in conversions(narrow, padded, dialect="postgres")}
    assert ConversionKind.IDENTITY in offered
    assert ConversionKind.ZERO_PAD in offered
    pad = next(item for item in conversions(narrow, padded) if item.kind is ConversionKind.ZERO_PAD)
    assert pad.width == 10
    assert "VARCHAR" in pad.on

    assert ConversionKind.CASE_FOLD in {
        item.kind for item in conversions(letters, _value_set(["ACME", "GLOBEX"]))
    }
    prefix = next(
        item for item in conversions(narrow, prefixed) if item.kind is ConversionKind.PREFIX_STRIP
    )
    assert prefix.prefix_length == len("CUST-")
    assert "SUBSTR" in prefix.on

    # A padded numeric string cannot be compared with letters, so no padding is offered for them.
    assert ConversionKind.ZERO_PAD not in {item.kind for item in conversions(letters, padded)}


def test_prefix_strip_conversion_is_measured_and_proposed(tmp_path: Path, store: Store) -> None:
    """``CUST-000123`` against ``123``: the constant prefix is detected, not assumed."""
    numbers = FakeAdapter(name="crm", tables={"customers": _table(cust_no=range(1, 51))})
    prefixed = FakeAdapter(
        name="api",
        tables={"tickets": _table(ticket_ref=[f"CUST-{n:06d}" for n in range(1, 51)])},
    )
    try:
        measurement = _measure(_ctx(tmp_path, store), {"crm": numbers, "api": prefixed})
    finally:
        numbers.close()
        prefixed.close()
    candidate = _pair(measurement, "crm.customers.cust_no", "api.tickets.ticket_ref")
    assert candidate.containment.conversion.kind is ConversionKind.PREFIX_STRIP
    assert candidate.containment.jaccard == pytest.approx(1.0)


def test_case_fold_conversion_is_measured_and_proposed(tmp_path: Path, store: Store) -> None:
    """Two text keys differing only in case are a real join, with ``UPPER`` on both sides."""
    lower = FakeAdapter(
        name="crm",
        tables={"accounts": _table(account_code=[f"acme{n}" for n in range(1, 21)])},
    )
    upper = FakeAdapter(
        name="billing",
        tables={"accounts": _table(account_code=[f"ACME{n}" for n in range(1, 21)])},
    )
    try:
        measurement = _measure(_ctx(tmp_path, store), {"crm": lower, "billing": upper})
    finally:
        lower.close()
        upper.close()
    candidate = _pair(measurement, "crm.accounts.account_code", "billing.accounts.account_code")
    assert candidate.containment.conversion.kind is ConversionKind.CASE_FOLD
    assert candidate.containment.conversion.on == "UPPER({left}) = UPPER({right})"
    assert candidate.containment.jaccard == pytest.approx(1.0)


def test_measure_reports_containment_both_ways_and_in_rows() -> None:
    """The pure measurement: containment both ways, unmatched keys and unmatched rows."""
    left = _value_set(["1", "2", "3"], counts=[5, 5, 5])
    right = _value_set(["0000000001", "0000000002", "0000000003", "0000000004"])
    identity = measure(
        left, right, FormatConversion(kind=ConversionKind.IDENTITY, on="{left} = {right}")
    )
    padded = measure(
        left,
        right,
        FormatConversion(
            kind=ConversionKind.ZERO_PAD,
            on="{left} = LPAD({right}, 10, '0')",
            width=10,
            converted="right",
        ),
    )
    assert identity.jaccard == 0.0
    assert identity.unmatched_left_rows == 15
    assert padded.jaccard == pytest.approx(0.75)
    assert padded.containment_left == pytest.approx(1.0)
    assert padded.unmatched_left_rows == 0
    assert padded.unmatched_right_examples == ("4",)


# --- honesty: what could not be measured ------------------------------------------------------


def test_a_value_set_over_the_cap_is_reported_and_not_proposed(
    tmp_path: Path, store: Store, adapters: Mapping[str, FakeAdapter]
) -> None:
    """Above the distinct cap the containment is unknown, so nothing is proposed from it."""
    measurement = _measure(_ctx(tmp_path, store), adapters, value_cap=2)
    assert measurement.candidates == ()
    assert measurement.counts["pairs_unmeasurable"] > 0
    assert any("distinct values" in line for line in measurement.skipped)


def test_only_key_shaped_columns_are_read(
    tmp_path: Path, store: Store, adapters: Mapping[str, FakeAdapter]
) -> None:
    """A measure or a text column is not a join key, and its values are never read."""
    measurement = _measure(_ctx(tmp_path, store), adapters)
    read = {item.label for item in measurement.columns}
    assert CRM_CUST_ID in read
    assert BILLING_CLIENT_NO in read
    assert not any(label.endswith(".credit_limit") for label in read)
    assert not any(label.endswith(".full_name") for label in read)
    assert measurement.counts["columns_skipped"] > 0


def test_the_same_input_measures_the_same_way(
    tmp_path: Path, store: Store, adapters: Mapping[str, FakeAdapter]
) -> None:
    """Deterministic: two passes over unchanged data produce identical measurements."""
    ctx = _ctx(tmp_path, store)
    first = _measure(ctx, adapters).model_dump(mode="json")
    second = _measure(ctx, adapters).model_dump(mode="json")
    assert first == second


def test_the_stage_refuses_outside_a_quiet_window(
    tmp_path: Path, store: Store, adapters: Mapping[str, FakeAdapter]
) -> None:
    """A value read per column is heavy: outside the window the stage refuses, and says when."""
    stage = JoinStage(adapters)
    ctx = _ctx(tmp_path, store, quiet=(NIGHT,))
    with pytest.raises(QuietWindowClosed) as raised:
        stage.run(ctx)
    assert "quiet window" in str(raised.value)
    forced = run_pipeline([stage], ctx, force=True)
    assert forced.ok, forced.describe()
    assert forced.reports[0].status is StageStatus.RAN


# --- the human gate ---------------------------------------------------------------------------


def test_candidates_are_proposed_through_the_review_queue(
    tmp_path: Path, store: Store, adapters: Mapping[str, FakeAdapter]
) -> None:
    """approve() writes ``curated/approved_joins.yaml`` and the loader reads the join back."""
    queue = ReviewQueue(store, tmp_path / "semantic", clock=ScriptedClock())
    stage = JoinStage(adapters, queue=queue)
    result = run_pipeline([stage], _ctx(tmp_path, store), force=True)
    assert result.ok, result.describe()
    assert result.reports[0].counts["proposals_created"] > 0

    pending = queue.pending()
    assert pending
    target = next(
        proposal
        for proposal in pending
        if {proposal.patch["left"], proposal.patch["right"]} == {CRM_CUST_ID, BILLING_CLIENT_NO}
    )
    assert target.kind is ProposalKind.JOIN
    assert target.section == "approved_joins"
    assert target.status is ProposalStatus.PENDING
    assert any("LPAD" in line for line in target.evidence)

    queue.approve(target.id, by="yoni")
    layer = load(tmp_path / "semantic")
    approved = next(join for join in layer.approved_joins if join.name == target.key)
    assert "LPAD" in approved.on
    assert {approved.left, approved.right} == {CRM_CUST_ID, BILLING_CLIENT_NO}


def test_a_declared_foreign_key_with_total_containment_is_auto_approvable(
    tmp_path: Path, store: Store
) -> None:
    """DESIGN section 10's auto gate: a declared FK *and* full containment, nothing less."""
    adapter = DeclaredKeys(
        name="erp",
        tables={
            "customers": _table(cust_id=list(range(1, 51))),
            "orders": _table(
                customer_id=[n % 50 + 1 for n in range(1, 201)],
                order_id=list(range(1, 201)),
            ),
        },
    )
    try:
        measurement = _measure(_ctx(tmp_path, store), {"erp": adapter})
    finally:
        adapter.close()
    candidate = _pair(measurement, "erp.orders.customer_id", "erp.customers.cust_id")
    assert candidate.declared_fk is True
    assert candidate.auto_approvable is True
    assert candidate.containment.total is True
    # Cardinality follows the proposal's own orientation: one customer, many orders.
    expected = (
        JoinCardinality.ONE_TO_MANY
        if candidate.left == "erp.customers.cust_id"
        else JoinCardinality.MANY_TO_ONE
    )
    assert candidate.cardinality is expected


# --- the statement, and the name similarity --------------------------------------------------


def test_distinct_query_is_bounded_and_ordered() -> None:
    """The read is one ``GROUP BY``, ordered for determinism and limited one past the cap."""
    column = Column(name="client_no", native_type="varchar", normalized_type=NormalizedType.STRING)
    table = Table(name="accounts", columns=(column,))
    sql = distinct_query(table, column, dialect="postgres", cap=7)
    assert "GROUP BY" in sql
    assert "ORDER BY" in sql
    assert "LIMIT 8" in sql
    assert sql.count("client_no") >= 3


@pytest.mark.parametrize("name", ["bad name", "x;drop", "9lives"])
def test_distinct_query_refuses_an_identifier_it_did_not_expect(name: str) -> None:
    """A catalog that reports a wild name must not reach the query text."""
    table = Table(name="accounts", columns=())
    column = Column(name=name, native_type="varchar", normalized_type=NormalizedType.STRING)
    with pytest.raises(ValueError):
        distinct_query(table, column, dialect="postgres", cap=5)


def test_distinct_query_refuses_a_wild_table_name() -> None:
    """The table name goes into the statement too, so it is checked the same way."""
    column = Column(name="client_no", native_type="varchar", normalized_type=NormalizedType.STRING)
    table = Table(name="accounts; drop table x", columns=(column,))
    with pytest.raises(ValueError):
        distinct_query(table, column, dialect="postgres", cap=5)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("cust_id", "cust_no", 1.0),
        ("cust_id", "cust_id", 1.0),
        ("Cust_Id", "cust_id", 1.0),
        ("client_no", "client_no", 1.0),
    ],
)
def test_name_similarity_on_shared_stems(left: str, right: str, expected: float) -> None:
    """The generic key tokens are dropped, so the entity stem is what is compared."""
    assert name_similarity(left, right) == pytest.approx(expected)


def test_name_similarity_is_low_for_two_different_entities() -> None:
    """``cust_id`` and ``client_no`` are different words: the score is why a name cannot gate."""
    assert name_similarity("cust_id", "client_no") < 0.5
    assert name_similarity("account_id", "customer_id") < 0.6
