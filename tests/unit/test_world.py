"""Tests for the seeded synthetic world (task D02).

Every acceptance criterion has a test here. The world is plain Python rows, so these are
fast, offline unit tests: no database, no network, no clock.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

import pytest

from mdqm.testing.world import (
    CANONICAL_STATUSES,
    ERP_SCALE,
    ROUNDING_STEP,
    GroundTruth,
    World,
    WorldConfig,
    build_world,
    render_json,
    round_half_even,
    round_half_up,
    world_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "world"

STATUS_CANONICAL = {"A": "active", "C": "churned", "S": "suspended"}
STATE_CANONICAL = {"Active": "active", "Closed": "churned", "Hold": "suspended"}
CODE_CANONICAL = {1: "active", 2: "churned", 3: "suspended"}


def replace_small(*, seed: int, lag_violations: int) -> WorldConfig:
    """A small world with a planted lag violation, for drift tests."""
    return dataclasses.replace(WorldConfig.small(seed=seed), lag_violations=lag_violations)


@pytest.fixture(scope="module")
def small() -> World:
    """One small world (200 customers, 5k events) shared by the read-only tests."""
    return build_world(WorldConfig.small(seed=1))


# ---------------------------------------------------------------- determinism


def test_same_seed_is_byte_identical() -> None:
    first = build_world(WorldConfig.small(seed=7))
    second = build_world(WorldConfig.small(seed=7))
    assert render_json(first) == render_json(second)
    assert world_hash(first) == world_hash(second)


def test_different_seed_gives_different_rows() -> None:
    one = build_world(WorldConfig.small(seed=1))
    two = build_world(WorldConfig.small(seed=2))
    assert world_hash(one) != world_hash(two)
    assert one.crm.customers != two.crm.customers
    assert one.billing.usage_events != two.billing.usage_events


def test_builds_do_not_share_state() -> None:
    """No global mutable state: an interleaved build cannot change a later result."""
    first = world_hash(build_world(WorldConfig.small(seed=1)))
    build_world(WorldConfig(seed=99, customers=40, months=3, events_per_customer=3))
    again = world_hash(build_world(WorldConfig.small(seed=1)))
    assert first == again


def test_render_is_canonical_text() -> None:
    text = render_json(build_world(WorldConfig.small(seed=1)))
    assert "\r" not in text
    assert text.endswith("\n")
    parsed = json.loads(text)
    assert set(parsed) == {"config", "crm", "billing", "erp", "expected"}
    assert parsed["expected"]["customers"], "ground truth must carry the canonical customers"
    assert text == json.dumps(parsed, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    assert "דנה" in text or "כהן" in text, "Hebrew names must survive as UTF-8, not \\u escapes"


def test_client_no_is_zero_padded_decimal(small: World) -> None:
    for truth in small.expected.customers:
        assert truth.client_no == f"{truth.cust_id:010d}"
        assert len(truth.client_no) == 10
        assert truth.client_no.isdigit()


# ---------------------------------------------------------------- cross-source invariants


def test_crm_limits_are_all_multiples_of_5000(small: World) -> None:
    limits = [row.credit_limit for row in small.crm.customers]
    assert limits
    assert all(limit % ROUNDING_STEP == 0 for limit in limits)
    assert all(limit >= 0 for limit in limits)


def test_erp_values_are_all_multiples_of_500(small: World) -> None:
    values = [row.limit_k for row in small.erp.customers]
    assert values
    assert all(value % 500 == 0 for value in values)


def test_erp_matches_crm_for_at_least_99_5_percent() -> None:
    """The default world: planted erp exceptions stay under the 0.5% budget."""
    world = build_world()
    truth = world.expected
    erp_by_cust = {}
    for row in world.erp.customers:
        erp_by_cust.setdefault(row.cust_no, row)
    crm_by_cust = {row.cust_id: row for row in world.crm.customers}
    mismatched = {
        cust_no
        for cust_no, row in erp_by_cust.items()
        if row.limit_k * ERP_SCALE != crm_by_cust[cust_no].credit_limit
    }
    assert mismatched == {exception.cust_id for exception in truth.erp_exceptions}
    assert len(mismatched) * 1000 <= len(crm_by_cust) * 5  # >= 99.5% agree
    assert len(mismatched) < len(crm_by_cust) // 100


def test_erp_exceptions_are_described_in_ground_truth(small: World) -> None:
    crm_by_cust = {row.cust_id: row for row in small.crm.customers}
    for exception in small.expected.erp_exceptions:
        assert exception.crm_limit == crm_by_cust[exception.cust_id].credit_limit
        assert exception.erp_limit_times_10 != exception.crm_limit
        assert exception.erp_limit_k == exception.crm_limit // ERP_SCALE + 500
        assert exception.reason


def test_billing_raw_is_mostly_unrounded(small: World) -> None:
    raw = [row.credit_limit for row in small.billing.accounts if row.credit_limit is not None]
    multiples = [value for value in raw if value % ROUNDING_STEP == 0]
    assert len(multiples) * 100 <= len(raw) * 5, f"{len(multiples)}/{len(raw)} multiples of 5000"


def test_the_three_sources_hold_the_same_customers(small: World) -> None:
    crm_ids = {row.cust_id for row in small.crm.customers}
    billing_ids = {int(row.client_no) for row in small.billing.accounts}
    erp_ids = {row.cust_no for row in small.erp.customers}
    truth_ids = {truth.cust_id for truth in small.expected.customers}
    assert crm_ids == truth_ids
    assert billing_ids == truth_ids
    assert erp_ids == truth_ids, "duplicate erp rows repeat an id, they never add one"


def test_billing_raw_equals_truth_except_planted_nulls(small: World) -> None:
    nulled = {null.key for null in small.expected.planted_nulls}
    truth_by_client = {truth.client_no: truth for truth in small.expected.customers}
    for row in small.billing.accounts:
        if row.client_no in nulled:
            assert row.credit_limit is None
        else:
            assert row.credit_limit == truth_by_client[row.client_no].raw_limit


def test_status_vocabularies_agree_across_sources(small: World) -> None:
    """'A' / 'Active' / 1 are the same canonical status: one value map, three codes."""
    erp_by_cust = {row.cust_no: row for row in small.erp.customers}
    crm_by_cust = {row.cust_id: row for row in small.crm.customers}
    billing_by_client = {int(row.client_no): row for row in small.billing.accounts}
    for truth in small.expected.customers:
        canonical = truth.canonical_status
        assert canonical in CANONICAL_STATUSES
        assert STATUS_CANONICAL[crm_by_cust[truth.cust_id].status] == canonical
        assert STATE_CANONICAL[billing_by_client[truth.cust_id].state] == canonical
        assert CODE_CANONICAL[erp_by_cust[truth.cust_id].status_code] == canonical


# ---------------------------------------------------------------- lag / insertion time


def test_insertion_time_is_start_time_plus_lag(small: World) -> None:
    for event in small.billing.usage_events:
        assert event.insertion_time >= event.start_time
        lag = int((event.insertion_time - event.start_time).total_seconds())
        assert 0 <= lag <= small.config.max_lag_seconds


def test_lag_distribution_respects_max_lag(small: World) -> None:
    truth = small.expected
    lags = [
        int((event.insertion_time - event.start_time).total_seconds())
        for event in small.billing.usage_events
    ]
    assert truth.max_lag_seconds == small.config.max_lag_seconds
    assert truth.lag_max_seconds == max(lags) <= small.config.max_lag_seconds
    assert truth.lag_p99_seconds <= small.config.max_lag_seconds
    assert truth.late_arriving_events == sum(1 for lag in lags if lag > 0)
    assert truth.lag_violations == ()


def test_lag_violations_can_be_planted() -> None:
    config = replace_small(seed=3, lag_violations=5)
    world = build_world(config)
    lags = {
        event.event_id: int((event.insertion_time - event.start_time).total_seconds())
        for event in world.billing.usage_events
    }
    violations = world.expected.lag_violations
    assert len(violations) == 5
    assert world.expected.lag_max_seconds > config.max_lag_seconds
    for violation in violations:
        assert lags[violation.event_id] == violation.lag_seconds
        assert violation.lag_seconds > config.max_lag_seconds


def test_events_are_ordered_by_event_time_and_have_unique_ids(small: World) -> None:
    events = small.billing.usage_events
    ids = [event.event_id for event in events]
    assert len(ids) == len(set(ids))
    regular = [event for event in events if event.corrects_event_id is None]
    assert [event.start_time for event in regular] == sorted(event.start_time for event in regular)


# ---------------------------------------------------------------- rounding


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, 0),
        (1, 0),
        (2499, 0),
        (2500, 5000),
        (2501, 5000),
        (4999, 5000),
        (5000, 5000),
        (7500, 10000),
        (10000, 10000),
        (12500, 15000),
        (17500, 20000),
        (60000, 60000),
    ],
)
def test_round_half_up_never_uses_bankers_rounding(raw: int, expected: int) -> None:
    assert round_half_up(raw) == expected


@pytest.mark.parametrize(
    ("raw", "half_even"),
    [(2500, 0), (7500, 10000), (12500, 10000), (5000, 5000), (12501, 15000)],
)
def test_round_half_even_documents_the_wrong_rule(raw: int, half_even: int) -> None:
    assert round_half_even(raw) == half_even


def test_rounding_rejects_negative_and_zero_step() -> None:
    with pytest.raises(ValueError):
        round_half_up(-2500)
    with pytest.raises(ValueError):
        round_half_up(2500, step=0)


def test_boundary_rows_exist_and_ground_truth_is_half_up(small: World) -> None:
    """2500 -> 5000 and 7500 -> 10000 are exactly where banker's rounding disagrees."""
    truth = small.expected
    crm_by_cust = {row.cust_id: row for row in small.crm.customers}
    raw_values = {row.raw_limit for row in truth.boundary_rows if row.planted}
    assert {2500, 7500, 12500} <= raw_values
    for row in truth.boundary_rows:
        assert crm_by_cust[row.cust_id].credit_limit == row.half_up_limit
    half_up = {(row.raw_limit, row.half_up_limit) for row in truth.boundary_rows}
    assert (2500, 5000) in half_up
    assert (7500, 10000) in half_up
    assert (12500, 15000) in half_up
    differs = {
        (row.raw_limit, row.half_even_limit)
        for row in truth.boundary_rows
        if row.rounds_differently
    }
    assert (2500, 0) in differs, "half-up must not behave like banker's rounding"
    assert (12500, 10000) in differs


# ---------------------------------------------------------------- planted facts


def test_nulls_are_planted_and_listed(small: World) -> None:
    nulled = {null.key for null in small.expected.planted_nulls}
    accounts = {row.client_no: row for row in small.billing.accounts}
    observed = {client_no for client_no, row in accounts.items() if row.credit_limit is None}
    assert observed == nulled
    assert nulled, "at least one NULL raw limit must be planted"
    for null in small.expected.planted_nulls:
        assert (null.dataset, null.table, null.column) == ("billing", "accounts", "credit_limit")
        assert null.reason


def test_orphan_events_are_counted_as_join_drops(small: World) -> None:
    truth = small.expected
    known = {row.client_no for row in small.billing.accounts}
    orphan_nos = set(truth.orphan_client_nos)
    assert orphan_nos and orphan_nos.isdisjoint(known)
    observed = Counter(
        event.client_no for event in small.billing.usage_events if event.client_no not in known
    )
    assert truth.orphan_event_count == sum(observed.values())
    assert set(observed) == orphan_nos
    assert list(truth.orphan_client_nos) == sorted(orphan_nos)


def test_duplicate_erp_rows_are_planted(small: World) -> None:
    counts = Counter(row.cust_no for row in small.erp.customers)
    duplicated = sorted(cust_no for cust_no, count in counts.items() if count > 1)
    assert duplicated == list(small.expected.duplicate_cust_nos)
    by_cust = {}
    for row in small.erp.customers:
        by_cust.setdefault(row.cust_no, []).append(row)
    for cust_no in duplicated:
        rows = by_cust[cust_no]
        assert len(rows) == 2
        assert rows[0] == rows[1], "a duplicate must be identical, so a naive AVG double counts"


def test_backdated_corrections_are_appended_rarely(small: World) -> None:
    truth = small.expected
    assert truth.corrections, "at least one correction must be planted"
    assert len(truth.corrections) * 100 < truth.events_total, "append-rarely, not append-always"
    by_id = {event.event_id: event for event in small.billing.usage_events}
    for correction in truth.corrections:
        original = by_id[correction.original_event_id]
        appended = by_id[correction.correction_event_id]
        assert appended.corrects_event_id == original.event_id
        assert appended.start_time == original.start_time == correction.start_time
        assert appended.client_no == original.client_no
        assert appended.insertion_time > original.insertion_time, "the correction arrives later"
        assert appended.units == correction.units_after != correction.units_before
        assert original.units == correction.units_before
        assert appended.event_id > original.event_id, "appended, never rewritten in place"


def test_accounts_hist_is_frequently_updated(small: World) -> None:
    rows = small.billing.accounts_hist
    truth = small.expected.accounts_hist
    counts = Counter(row.client_no for row in rows)
    assert truth.rows_total == len(rows)
    assert truth.clients_with_multiple_revisions == sum(1 for n in counts.values() if n > 1)
    assert truth.clients_with_multiple_revisions > 0
    assert truth.superseded_rows == len(rows) - truth.clients
    assert truth.mutability_class == "mutable_critical"
    for client_no, revisions in counts.items():
        moments = [row.updated_at for row in rows if row.client_no == client_no]
        assert len(moments) == revisions
        assert moments == sorted(set(moments)), f"{client_no} revisions must be distinguishable"


# ---------------------------------------------------------------- ground truth answers


def test_ground_truth_average_by_status_matches_the_crm_rows(small: World) -> None:
    """Recompute from the stored (rounded) crm values through the value map, independently."""
    totals: dict[str, int] = {}
    counts: dict[str, int] = {}
    for row in small.crm.customers:
        canonical = STATUS_CANONICAL[row.status]
        totals[canonical] = totals.get(canonical, 0) + row.credit_limit
        counts[canonical] = counts.get(canonical, 0) + 1
    for status, answer in small.expected.avg_rounded_limit_by_status.items():
        assert answer.total == totals[status]
        assert answer.count == counts[status]
        assert answer.avg == round(totals[status] / counts[status], 6)
    overall = small.expected.avg_rounded_limit_all
    assert overall.total == sum(totals.values())
    assert overall.count == len(small.crm.customers)
    assert overall.total == sum(
        a.total for a in small.expected.avg_rounded_limit_by_status.values()
    )


def test_ground_truth_events_per_month_matches_the_rows(small: World) -> None:
    truth = small.expected
    observed = Counter(event.start_time.strftime("%Y-%m") for event in small.billing.usage_events)
    assert truth.events_per_month == dict(sorted(observed.items()))
    assert sum(truth.events_per_month.values()) == truth.events_total
    assert len(truth.events_per_month) == small.config.months
    assert truth.regular_events == truth.events_total - len(truth.corrections)


def test_ground_truth_canonical_limits_are_half_up_of_raw(small: World) -> None:
    for truth in small.expected.customers:
        assert truth.canonical_limit == round_half_up(truth.raw_limit)
        assert truth.canonical_limit % ROUNDING_STEP == 0


# ---------------------------------------------------------------- the CLI


def _run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "mdqm.testing.world", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        env=env,
        check=False,
    )


def test_cli_writes_world_and_ground_truth(tmp_path: Path) -> None:
    out = tmp_path / "data" / "world.json"
    result = _run_cli(tmp_path, "--seed", "1", "--out", str(out), "--small")
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    world = json.loads(out.read_text(encoding="utf-8"))
    assert len(world["expected"]["customers"]) == WorldConfig.small(seed=1).customers
    assert world["billing"]["usage_events"]
    assert world["expected"]["avg_rounded_limit_by_status"]
    assert str(out) in result.stdout


def test_cli_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    assert _run_cli(tmp_path, "--seed", "4", "--out", str(first), "--small").returncode == 0
    assert _run_cli(tmp_path, "--seed", "4", "--out", str(second), "--small").returncode == 0
    assert first.read_bytes() == second.read_bytes()


def test_cli_can_plant_lag_violations(tmp_path: Path) -> None:
    out = tmp_path / "violations.json"
    result = _run_cli(
        tmp_path, "--seed", "2", "--out", str(out), "--small", "--lag-violations", "3"
    )
    assert result.returncode == 0, result.stderr
    world = json.loads(out.read_text(encoding="utf-8"))
    assert len(world["expected"]["lag_violations"]) == 3


# ---------------------------------------------------------------- committed fixture


def test_committed_tiny_world_is_unchanged(pytestconfig: pytest.Config) -> None:
    """A tiny world is committed, so a change in generation shows up as a reviewable diff."""
    config = WorldConfig(seed=1, customers=12, months=2, events_per_customer=4)
    text = render_json(build_world(config))
    path = FIXTURE_DIR / "tiny_seed1.json"
    if pytestconfig.getoption("--update-golden"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        return
    assert path.is_file(), f"missing fixture {path}; run with --update-golden"
    assert path.read_text(encoding="utf-8") == text


# ---------------------------------------------------------------- config guards


@pytest.mark.parametrize(
    "kwargs",
    [
        {"customers": 0},
        {"months": 0},
        {"events_per_customer": -1},
        {"max_lag_seconds": -1},
    ],
)
def test_config_rejects_impossible_values(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        WorldConfig(**kwargs)  # type: ignore[arg-type]


def test_config_rejects_naive_anchor() -> None:
    from datetime import datetime

    with pytest.raises(ValueError):
        WorldConfig(anchor=datetime(2026, 1, 1))


def test_tiny_world_still_has_one_of_each_plant() -> None:
    world = build_world(WorldConfig(seed=5, customers=6, months=1, events_per_customer=5))
    truth: GroundTruth = world.expected
    assert truth.boundary_rows
    assert truth.planted_nulls
    assert truth.erp_exceptions
    assert truth.duplicate_cust_nos
    assert truth.corrections


def test_window_is_exactly_the_requested_months(small: World) -> None:
    starts = [event.start_time for event in small.billing.usage_events]
    anchor = small.config.anchor
    assert max(starts) < anchor, "no event starts after the anchor"
    first_month = min(small.expected.events_per_month)
    assert min(starts).strftime("%Y-%m") == first_month
    assert len(small.expected.events_per_month) == small.config.months
    window_days = (anchor - timedelta(days=1)).date() - min(starts).date()
    assert window_days.days >= (small.config.months - 1) * 28
