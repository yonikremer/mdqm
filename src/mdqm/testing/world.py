"""Deterministic synthetic organisation for tests, loaders and the eval harness (task D02).

One small organisation, the same customers in three storage forms, so the cross-source
conversion problem is testable without any real engine::

    crm.customers      status 'A'|'C'|'S'      credit_limit = round_half_up(raw, 5000)
    billing.accounts   state 'Active'|...      credit_limit = raw, unrounded (may be NULL)
    erp.customers      status_code 1|2|3       limit_k = round_half_up(raw, 5000) // 10

The canonical measure is therefore ``round_half_up(x / 5000) * 5000`` and the three
normalisations are identity / ``ROUND(x / 5000) * 5000`` / ``x * 10``.

``build_world`` computes the canonical customer records *first* and then derives the three
storage forms from them, so ``World.expected`` is ground truth by construction and never a
recomputation of the code under test. Everything is plain Python rows: no database, no
network, no clock. The same seed gives the same bytes on every OS (``render_json`` sorts
keys and writes LF), and Python's ``random.Random`` is used rather than numpy.

Planted on purpose:

* credit limits exactly on the .5 boundary (2500, 7500, 12500) so half-up and banker's
  rounding disagree (2500 -> 5000 vs 0, 12500 -> 15000 vs 10000);
* NULL raw limits, orphan ``client_no`` values with no customer, duplicate erp rows;
* a few backdated corrections in ``usage_events`` (append-rarely) and an
  ``accounts_hist`` table with several revisions per account (mutable_critical).

Derived/aggregate tables (``usage_by_load``, the hash-partitioned table, lookup tables) are
built from these rows by the loader (task D03), not stored here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

ROUNDING_STEP = 5000
"""Granularity of the canonical credit limit (see DESIGN section 12)."""

ERP_SCALE = 10
"""``erp.customers.limit_k`` is the rounded limit divided by this."""

CANONICAL_STATUSES = ("active", "churned", "suspended")
"""Canonical status vocabulary; each source stores its own code for it."""

_STATUS_CODES = {
    "active": ("A", "Active", 1),
    "churned": ("C", "Closed", 2),
    "suspended": ("S", "Hold", 3),
}
_STATUS_WEIGHTS = (60, 30, 10)
"""Deterministic population weights for active/churned/suspended."""

_REGIONS = ("north", "center", "tel_aviv", "south")
_FIRST_NAMES = (
    "דנה",
    "יוסי",
    "מיכל",
    "אחמד",
    "שרה",
    "דוד",
    "נועה",
    "עומר",
    "יעל",
    "רן",
)
_LAST_NAMES = ("כהן", "לוי", "אבו סאלם", "מזרחי", "פרידמן", "חדאד")

ANCHOR = datetime(2026, 1, 1, tzinfo=UTC)
"""End of the event window. A constant, never ``now()``, so runs are reproducible."""

BOUNDARY_VALUES = (2500, 7500, 12500)
"""Midpoints between two multiples of 5000: exactly where rounding modes disagree."""

ORPHAN_BASE = 9_000_000_000
"""First of the planted client numbers that have no customer anywhere."""


def round_half_up(value: int, step: int = ROUNDING_STEP) -> int:
    """Round ``value`` to the nearest multiple of ``step``, ties away from zero.

    Python's built-in ``round`` uses banker's rounding (2.5 -> 2), which would give
    2500 -> 0 and 12500 -> 10000. The canonical measure is half-up (DESIGN section 12),
    so this is the one place the rule lives. Integer arithmetic only: no float drift.
    """
    if value < 0:
        raise ValueError(f"credit limits are non-negative, got {value}")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    return ((value + step // 2) // step) * step


def round_half_even(value: int, step: int = ROUNDING_STEP) -> int:
    """Round ``value`` to the nearest multiple of ``step``, ties to even (banker's).

    Used only by the ground truth to record what the *wrong* rule would have produced;
    never used to build stored data.
    """
    if value < 0:
        raise ValueError(f"credit limits are non-negative, got {value}")
    quotient, remainder = divmod(value, step)
    doubled = remainder * 2
    if doubled > step or (doubled == step and quotient % 2 == 1):
        quotient += 1
    return quotient * step


@dataclass(frozen=True, slots=True)
class WorldConfig:
    """Shape of one generated organisation. ``WorldConfig.small()`` is the fast variant."""

    seed: int = 1
    customers: int = 2000
    months: int = 8
    events_per_customer: int = 25
    """Regular usage events per customer: 2000 * 25 = ~50k over 8 months."""

    anchor: datetime = ANCHOR
    max_lag_seconds: int = 6 * 3600
    """Declared ingestion lag bound for ``usage_events`` (DESIGN section 6)."""

    lag_violations: int = 0
    """Plant rows whose lag exceeds ``max_lag_seconds`` (for drift/lag-verification tests)."""

    boundary_per_value: int = 0
    """Customers per .5 boundary value; 0 means auto (5 per mille, at least 1)."""

    null_rows: int = 0
    """Planted NULL raw limits; 0 means auto (3 per mille, at least 1)."""

    orphan_events: int = 0
    """Events carrying a ``client_no`` that has no customer; 0 means auto (~0.4%)."""

    duplicate_customers: int = 0
    """Duplicate rows planted in ``erp.customers``; 0 means auto (~3 per mille, at least 1)."""

    erp_exceptions: int = 0
    """Customers whose erp limit deliberately disagrees with crm; 0 means auto (4 per mille)."""

    corrections: int = 0
    """Backdated correction rows appended to ``usage_events``; 0 means auto (~1 per 300)."""

    @classmethod
    def small(cls, seed: int = 1) -> WorldConfig:
        """2000 customers / 50k events shrunk to 200 / 5k: fast unit and integration runs."""
        return cls(seed=seed, customers=200)

    def __post_init__(self) -> None:
        if self.customers < 1:
            raise ValueError(f"customers must be >= 1, got {self.customers}")
        if self.months < 1:
            raise ValueError(f"months must be >= 1, got {self.months}")
        if self.events_per_customer < 0:
            raise ValueError(f"events_per_customer must be >= 0, got {self.events_per_customer}")
        if self.max_lag_seconds < 0:
            raise ValueError(f"max_lag_seconds must be >= 0, got {self.max_lag_seconds}")
        if self.anchor.tzinfo is None:
            raise ValueError("anchor must be timezone-aware UTC")

    @property
    def event_count(self) -> int:
        """Total number of rows written to ``billing.usage_events``, corrections included."""
        return self.regular_events + self.corrections_count

    @property
    def regular_events(self) -> int:
        return self.customers * self.events_per_customer

    @property
    def corrections_count(self) -> int:
        return self.corrections or max(1, self.customers // 300)

    @property
    def boundary_count(self) -> int:
        per_value = self.boundary_per_value or max(1, self.customers * 5 // 1000)
        return min(3 * per_value, self.customers)

    @property
    def null_count(self) -> int:
        return min(self.null_rows or max(1, self.customers * 3 // 1000), self.customers)

    @property
    def orphan_event_count(self) -> int:
        return min(
            self.orphan_events or max(1, self.regular_events // 250),
            self.regular_events,
        )

    @property
    def duplicate_count(self) -> int:
        return min(
            self.duplicate_customers or max(1, self.customers * 3 // 1000),
            self.customers,
        )

    @property
    def erp_exception_count(self) -> int:
        return min(
            self.erp_exceptions or max(1, self.customers * 4 // 1000),
            self.customers,
        )


# --------------------------------------------------------------------------------------
# Rows: the three storage forms plus the history table. Frozen dataclasses so a loader
# cannot mutate the world it was handed.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CrmCustomer:
    """``crm.customers``: rounded limit, single-letter status."""

    cust_id: int
    full_name: str
    status: str
    credit_limit: int
    region: str
    signup_date: date


@dataclass(frozen=True, slots=True)
class BillingAccount:
    """``billing.accounts``: raw (unrounded) limit, word status, zero-padded key."""

    client_no: str
    state: str
    credit_limit: int | None
    region: str
    opened_date: date


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """``billing.usage_events``: event time indexed/partitioned, ingestion time is not."""

    event_id: int
    client_no: str
    start_time: datetime
    insertion_time: datetime
    units: int
    corrects_event_id: int | None


@dataclass(frozen=True, slots=True)
class AccountsHistRow:
    """``billing.accounts_hist``: frequently updated rows, latest ``updated_at`` wins."""

    hist_id: int
    client_no: str
    state: str
    credit_limit: int
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ErpCustomer:
    """``erp.customers``: integer status code, limit in thousands of the rounded limit."""

    cust_no: int
    status_code: int
    limit_k: int
    name: str


# --------------------------------------------------------------------------------------
# Ground truth. One record per planted fact, so a test can assert the plant exists and a
# consumer can subtract it instead of pretending the data is clean.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CustomerTruth:
    """Canonical customer record; the three storage forms are views of this."""

    cust_id: int
    client_no: str
    canonical_status: str
    raw_limit: int
    canonical_limit: int
    region: str


@dataclass(frozen=True, slots=True)
class BoundaryRow:
    """A customer whose raw limit is exactly on a .5 boundary."""

    cust_id: int
    raw_limit: int
    half_up_limit: int
    half_even_limit: int
    planted: bool
    rounds_differently: bool


@dataclass(frozen=True, slots=True)
class ErpException:
    """A customer whose erp ``limit_k`` deliberately disagrees with the crm rounding."""

    cust_id: int
    crm_limit: int
    erp_limit_k: int
    erp_limit_times_10: int
    reason: str


@dataclass(frozen=True, slots=True)
class PlantedNull:
    """A deliberately missing value, so null handling is exercised on purpose."""

    dataset: str
    table: str
    column: str
    key: str
    reason: str


@dataclass(frozen=True, slots=True)
class EventCorrection:
    """A backdated correction appended later (append-rarely table)."""

    original_event_id: int
    correction_event_id: int
    client_no: str
    start_time: datetime
    units_before: int
    units_after: int
    insertion_time: datetime


@dataclass(frozen=True, slots=True)
class LagRow:
    """An individual usage event's lag, as measured by ``insertion_time - start_time``."""

    event_id: int
    lag_seconds: int


@dataclass(frozen=True, slots=True)
class MeasureAnswer:
    """An aggregate answer: keep sum and count so a consumer can compare exactly."""

    total: int
    count: int
    avg: float


@dataclass(frozen=True, slots=True)
class AccountHistoryTruth:
    """What ``accounts_hist`` holds, including how many rows are superseded."""

    rows_total: int
    clients: int
    clients_with_multiple_revisions: int
    superseded_rows: int
    mutability_class: str


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Precomputed answers and planted facts. Computed from truth, not from the storage forms."""

    customers: tuple[CustomerTruth, ...]
    avg_rounded_limit_by_status: dict[str, MeasureAnswer]
    avg_rounded_limit_all: MeasureAnswer
    events_per_month: dict[str, int]
    events_total: int
    regular_events: int
    lag_p99_seconds: int
    lag_max_seconds: int
    late_arriving_events: int
    lag_violations: tuple[LagRow, ...]
    max_lag_seconds: int
    boundary_rows: tuple[BoundaryRow, ...]
    erp_exceptions: tuple[ErpException, ...]
    duplicate_cust_nos: tuple[int, ...]
    orphan_client_nos: tuple[str, ...]
    orphan_event_count: int
    planted_nulls: tuple[PlantedNull, ...]
    corrections: tuple[EventCorrection, ...]
    accounts_hist: AccountHistoryTruth


@dataclass(frozen=True, slots=True)
class CrmDataset:
    customers: tuple[CrmCustomer, ...]


@dataclass(frozen=True, slots=True)
class BillingDataset:
    accounts: tuple[BillingAccount, ...]
    usage_events: tuple[UsageEvent, ...]
    accounts_hist: tuple[AccountsHistRow, ...]


@dataclass(frozen=True, slots=True)
class ErpDataset:
    customers: tuple[ErpCustomer, ...]


@dataclass(frozen=True, slots=True)
class World:
    """One generated organisation: config, the three datasets, and the ground truth."""

    config: WorldConfig
    crm: CrmDataset
    billing: BillingDataset
    erp: ErpDataset
    expected: GroundTruth


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


def build_world(config: WorldConfig | None = None) -> World:
    """Build the world for ``config`` (default: 2000 customers, ~50k events, seed 1).

    Deterministic: one ``random.Random(seed)`` consumed in a fixed order, no clock, no
    dict/set iteration over unordered keys, integer arithmetic for all money.
    """
    cfg = config or WorldConfig()
    rng = random.Random(cfg.seed)  # noqa: S311 - reproducibility, not secrecy
    window_start = _months_before(cfg.anchor, cfg.months)
    window_seconds = int((cfg.anchor - window_start).total_seconds())
    cust_ids = list(range(1, cfg.customers + 1))

    planted_boundary = _planted_boundaries(cfg, rng, cust_ids)
    truths = [
        _customer_truth(cfg, rng, cust_id, planted_boundary.get(cust_id)) for cust_id in cust_ids
    ]
    client_nos = [truth.client_no for truth in truths]

    null_ids, exception_ids, duplicate_ids = _planted_quirks(cfg, rng, cust_ids, planted_boundary)
    duplicate_set = set(duplicate_ids)
    truth_by_id = {truth.cust_id: truth for truth in truths}

    crm = CrmDataset(
        customers=tuple(
            CrmCustomer(
                cust_id=truth.cust_id,
                full_name=_full_name(rng),
                status=_STATUS_CODES[truth.canonical_status][0],
                credit_limit=truth.canonical_limit,
                region=truth.region,
                signup_date=_signup_date(rng, cfg),
            )
            for truth in truths
        )
    )
    accounts = tuple(
        BillingAccount(
            client_no=truth.client_no,
            state=_STATUS_CODES[truth.canonical_status][1],
            credit_limit=None if truth.cust_id in null_ids else truth.raw_limit,
            region=truth.region,
            opened_date=_signup_date(rng, cfg),
        )
        for truth in truths
    )
    erp_rows: list[ErpCustomer] = []
    for truth in truths:
        limit_k = (
            truth.canonical_limit // ERP_SCALE + 500
            if truth.cust_id in exception_ids
            else truth.canonical_limit // ERP_SCALE
        )
        row = ErpCustomer(
            cust_no=truth.cust_id,
            status_code=_STATUS_CODES[truth.canonical_status][2],
            limit_k=limit_k,
            name=_full_name(rng),
        )
        erp_rows.append(row)
        if truth.cust_id in duplicate_set:
            erp_rows.append(row)  # an identical duplicate row, so a naive AVG double counts

    events, lags, corrections = _usage_events(cfg, rng, client_nos, window_start, window_seconds)
    hist, hist_truth = _accounts_hist(cfg, rng, truths)

    known_clients = set(client_nos)
    orphan_events = [event for event in events if event.client_no not in known_clients]

    expected = GroundTruth(
        customers=tuple(truths),
        avg_rounded_limit_by_status=_by_status(truths),
        avg_rounded_limit_all=_measure([truth.canonical_limit for truth in truths]),
        events_per_month=_events_per_month(events),
        events_total=len(events),
        regular_events=len(events) - len(corrections),
        lag_p99_seconds=_percentile99(lags),
        lag_max_seconds=max(lags) if lags else 0,
        late_arriving_events=sum(1 for lag in lags if lag > 0),
        lag_violations=tuple(
            LagRow(event_id=events[index].event_id, lag_seconds=lag)
            for index, lag in enumerate(lags)
            if lag > cfg.max_lag_seconds
        ),
        max_lag_seconds=cfg.max_lag_seconds,
        boundary_rows=tuple(
            BoundaryRow(
                cust_id=truth.cust_id,
                raw_limit=truth.raw_limit,
                half_up_limit=truth.canonical_limit,
                half_even_limit=round_half_even(truth.raw_limit),
                planted=truth.cust_id in planted_boundary,
                rounds_differently=round_half_even(truth.raw_limit) != truth.canonical_limit,
            )
            for truth in truths
            if truth.raw_limit % ROUNDING_STEP * 2 == ROUNDING_STEP
        ),
        erp_exceptions=tuple(
            ErpException(
                cust_id=cust_id,
                crm_limit=truth_by_id[cust_id].canonical_limit,
                erp_limit_k=truth_by_id[cust_id].canonical_limit // ERP_SCALE + 500,
                erp_limit_times_10=(truth_by_id[cust_id].canonical_limit // ERP_SCALE + 500) * 10,
                reason="planted: erp holds one 500 step too much, so crm and erp disagree",
            )
            for cust_id in sorted(exception_ids)
        ),
        duplicate_cust_nos=tuple(duplicate_ids),
        orphan_client_nos=tuple(sorted({event.client_no for event in orphan_events})),
        orphan_event_count=len(orphan_events),
        planted_nulls=tuple(
            PlantedNull(
                dataset="billing",
                table="accounts",
                column="credit_limit",
                key=truth_by_id[cust_id].client_no,
                reason="planted NULL raw limit; crm keeps the canonical rounded value",
            )
            for cust_id in sorted(null_ids)
        ),
        corrections=tuple(corrections),
        accounts_hist=hist_truth,
    )
    return World(
        config=cfg,
        crm=crm,
        billing=BillingDataset(
            accounts=accounts, usage_events=tuple(events), accounts_hist=tuple(hist)
        ),
        erp=ErpDataset(customers=tuple(erp_rows)),
        expected=expected,
    )


def _months_before(moment: datetime, months: int) -> datetime:
    """``moment`` shifted back by whole calendar months, keeping the day and time."""
    total = moment.year * 12 + (moment.month - 1) - months
    return moment.replace(year=total // 12, month=total % 12 + 1)


def _weighted_index(rng: random.Random, weights: Sequence[int]) -> int:
    pick = rng.randrange(sum(weights))
    for index, weight in enumerate(weights):
        pick -= weight
        if pick < 0:
            return index
    return len(weights) - 1  # pragma: no cover - unreachable


def _full_name(rng: random.Random) -> str:
    first = _FIRST_NAMES[rng.randrange(len(_FIRST_NAMES))]
    last = _LAST_NAMES[rng.randrange(len(_LAST_NAMES))]
    return f"{first} {last}"


def _signup_date(rng: random.Random, cfg: WorldConfig) -> date:
    return (cfg.anchor - timedelta(days=rng.randrange(30, 3000))).date()


def _planted_quirks(
    cfg: WorldConfig,
    rng: random.Random,
    cust_ids: Sequence[int],
    planted_boundary: dict[int, int],
) -> tuple[set[int], set[int], list[int]]:
    """Give each customer at most one planted quirk, so every fact has one cause.

    Boundary customers are already picked; NULLs, erp exceptions and duplicates are dealt
    from the remaining ids without replacement. Returns (nulls, exceptions, duplicates).
    """
    pool = rng.sample(
        [cust_id for cust_id in cust_ids if cust_id not in planted_boundary],
        max(0, len(cust_ids) - len(planted_boundary)),
    )
    wanted = cfg.null_count + cfg.erp_exception_count + cfg.duplicate_count
    dealt = pool[:wanted]
    null_ids = set(dealt[: cfg.null_count])
    exception_ids = set(dealt[cfg.null_count : cfg.null_count + cfg.erp_exception_count])
    duplicate_ids = sorted(dealt[cfg.null_count + cfg.erp_exception_count :])
    return null_ids, exception_ids, duplicate_ids


def _planted_boundaries(
    cfg: WorldConfig, rng: random.Random, cust_ids: Sequence[int]
) -> dict[int, int]:
    """Choose the customers whose raw limit sits exactly between two multiples of 5000."""
    count = cfg.boundary_count
    chosen = sorted(rng.sample(list(cust_ids), count))
    per_value = max(1, count // len(BOUNDARY_VALUES))
    return {
        cust_id: BOUNDARY_VALUES[index // per_value % len(BOUNDARY_VALUES)]
        for index, cust_id in enumerate(chosen)
    }


def _customer_truth(
    cfg: WorldConfig, rng: random.Random, cust_id: int, planted_raw: int | None
) -> CustomerTruth:
    raw = planted_raw if planted_raw is not None else rng.randrange(0, 60_001)
    status = CANONICAL_STATUSES[_weighted_index(rng, _STATUS_WEIGHTS)]
    return CustomerTruth(
        cust_id=cust_id,
        client_no=f"{cust_id:010d}",
        canonical_status=status,
        raw_limit=raw,
        canonical_limit=round_half_up(raw),
        region=_REGIONS[rng.randrange(len(_REGIONS))],
    )


def _usage_events(
    cfg: WorldConfig,
    rng: random.Random,
    client_nos: Sequence[str],
    window_start: datetime,
    window_seconds: int,
) -> tuple[list[UsageEvent], list[int], list[EventCorrection]]:
    """Build ``usage_events`` sorted by event time, then append late corrections.

    Lag: 99% of rows land a few seconds after the event, 1% arrive up to ``max_lag``
    later, nothing exceeds ``max_lag`` unless ``lag_violations`` asks for it.
    """
    orphan_total = cfg.orphan_event_count
    orphan_ids = [
        f"{ORPHAN_BASE + offset:010d}" for offset in range(max(1, orphan_total // 20 + 1))
    ]
    drafts: list[tuple[datetime, str, int, int]] = []
    lags: list[int] = []
    for index in range(cfg.regular_events):
        client_no = (
            orphan_ids[index % len(orphan_ids)]
            if index < orphan_total
            else (client_nos[rng.randrange(len(client_nos))])
        )
        start_time = window_start + timedelta(seconds=rng.randrange(window_seconds))
        lag = _lag_seconds(cfg, rng, index)
        drafts.append((start_time, client_no, lag, rng.randrange(1, 501)))
    drafts.sort()

    events: list[UsageEvent] = []
    for event_id, (start_time, client_no, lag, units) in enumerate(drafts, start=1):
        events.append(
            UsageEvent(
                event_id=event_id,
                client_no=client_no,
                start_time=start_time,
                insertion_time=start_time + timedelta(seconds=lag),
                units=units,
                corrects_event_id=None,
            )
        )
        lags.append(lag)

    corrections: list[EventCorrection] = []
    next_id = len(events) + 1
    correction_count = cfg.corrections_count
    if correction_count and events:
        picks = sorted(rng.sample(range(len(events)), min(correction_count, len(events))))
        pending: list[UsageEvent] = []
        for offset, index in enumerate(picks):
            original = events[index]
            correction_lag = cfg.max_lag_seconds // 2 + rng.randrange(
                max(1, cfg.max_lag_seconds - cfg.max_lag_seconds // 2 + 1)
            )
            corrected_units = max(1, original.units + rng.randrange(-100, 101))
            correction = UsageEvent(
                event_id=next_id + offset,
                client_no=original.client_no,
                start_time=original.start_time,
                insertion_time=original.start_time + timedelta(seconds=correction_lag),
                units=corrected_units,
                corrects_event_id=original.event_id,
            )
            pending.append(correction)
            corrections.append(
                EventCorrection(
                    original_event_id=original.event_id,
                    correction_event_id=correction.event_id,
                    client_no=original.client_no,
                    start_time=original.start_time,
                    units_before=original.units,
                    units_after=corrected_units,
                    insertion_time=correction.insertion_time,
                )
            )
            lags.append(correction_lag)
        pending.sort(key=lambda event: (event.insertion_time, event.event_id))
        events.extend(pending)
    return events, lags, corrections


def _lag_seconds(cfg: WorldConfig, rng: random.Random, index: int) -> int:
    if cfg.lag_violations and index < cfg.lag_violations:
        return cfg.max_lag_seconds + rng.randrange(1, 3601)
    if rng.randrange(0, 100) == 0:
        return rng.randrange(min(60, cfg.max_lag_seconds), cfg.max_lag_seconds + 1)
    return rng.randrange(0, min(60, cfg.max_lag_seconds + 1))


def _accounts_hist(
    cfg: WorldConfig, rng: random.Random, truths: Sequence[CustomerTruth]
) -> tuple[list[AccountsHistRow], AccountHistoryTruth]:
    """A history table that is updated often: several revisions per account."""
    rows: list[AccountsHistRow] = []
    multi = 0
    for truth in truths:
        revisions = rng.randrange(1, 4)
        multi += 1 if revisions > 1 else 0
        moment = cfg.anchor - timedelta(days=rng.randrange(5, 400))
        limit = truth.raw_limit
        for _ in range(revisions):
            rows.append(
                AccountsHistRow(
                    hist_id=len(rows) + 1,
                    client_no=truth.client_no,
                    state=_STATUS_CODES[truth.canonical_status][1],
                    credit_limit=max(0, limit),
                    updated_at=moment,
                )
            )
            # strictly later each revision: "latest updated_at wins" must be unambiguous
            moment += timedelta(days=rng.randrange(1, 40))
            limit += rng.randrange(-2, 3) * ROUNDING_STEP
    return rows, AccountHistoryTruth(
        rows_total=len(rows),
        clients=len(truths),
        clients_with_multiple_revisions=multi,
        superseded_rows=len(rows) - len(truths),
        mutability_class="mutable_critical",
    )


def _measure(values: Sequence[int]) -> MeasureAnswer:
    total = sum(values)
    return MeasureAnswer(
        total=total,
        count=len(values),
        avg=round(total / len(values), 6) if values else 0.0,
    )


def _by_status(truths: Sequence[CustomerTruth]) -> dict[str, MeasureAnswer]:
    buckets: dict[str, list[int]] = {status: [] for status in CANONICAL_STATUSES}
    for truth in truths:
        buckets[truth.canonical_status].append(truth.canonical_limit)
    return {status: _measure(values) for status, values in buckets.items()}


def _events_per_month(events: Sequence[UsageEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        key = event.start_time.strftime("%Y-%m")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _percentile99(lags: Sequence[int]) -> int:
    """Nearest-rank p99: the smallest lag that at least 99% of rows do not exceed."""
    if not lags:
        return 0
    ordered = sorted(lags)
    rank = max(1, -(-len(ordered) * 99 // 100))
    return ordered[rank - 1]


# --------------------------------------------------------------------------------------
# Serialisation: one canonical byte form, so two runs (and two OSes) compare equal.
# --------------------------------------------------------------------------------------


def _encode(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def render_json(world: World) -> str:
    """The canonical text form: sorted keys, two-space indent, UTF-8, trailing LF."""
    return json.dumps(_encode(asdict(world)), sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def world_hash(world: World) -> str:
    """SHA-256 of ``render_json``: equal hashes mean byte-identical worlds."""
    return hashlib.sha256(render_json(world).encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m mdqm.testing.world --seed 1 --out data/world.json``."""
    parser = argparse.ArgumentParser(
        prog="python -m mdqm.testing.world",
        description="Write the deterministic synthetic world and its ground truth as JSON.",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--out", type=Path, required=True, help="output JSON file")
    parser.add_argument("--small", action="store_true", help="200 customers / 5k events")
    parser.add_argument("--customers", type=int, default=None, help="override customer count")
    parser.add_argument("--months", type=int, default=None, help="override event window months")
    parser.add_argument(
        "--events-per-customer", type=int, default=None, help="override events per customer"
    )
    parser.add_argument(
        "--max-lag-seconds", type=int, default=None, help="override the declared ingestion lag"
    )
    parser.add_argument(
        "--lag-violations",
        type=int,
        default=0,
        help="plant this many rows whose lag exceeds --max-lag-seconds",
    )
    args = parser.parse_args(argv)

    base = WorldConfig.small(seed=args.seed) if args.small else WorldConfig(seed=args.seed)
    overrides: dict[str, Any] = {"lag_violations": args.lag_violations}
    for attribute in ("customers", "months", "events_per_customer", "max_lag_seconds"):
        value = getattr(args, attribute)
        if value is not None:
            overrides[attribute] = value
    config = replace(base, **overrides)

    world = build_world(config)
    text = render_json(world)
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    sys.stdout.write(
        f"world: {config.customers} customers, {world.expected.events_total} usage_events, "
        f"seed {config.seed}, sha256 {world_hash(world)}\nwritten to {out}\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
