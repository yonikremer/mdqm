"""A03 acceptance: partition metadata and the EXPLAIN pruning proof against the live server.

Needs the ``pg`` profile and the loaded world (docs/dev-env.md)::

    python scripts/dev.py up pg
    python -m mdqm.testing.loaders.postgres --small
    python -m pytest tests/integration/test_pg_partitions.py --require-services -q

Every number here is the server's, not the adapter's:

* the plan comes from the server's own ``EXPLAIN (FORMAT JSON)``, obtained without ``ANALYZE`` so
  nothing is executed to ask about it;
* the partition list the verdict is checked against is parsed in this file from the server's
  ``pg_get_expr(relpartbound)`` text, independently of ``extract_partitions``;
* the window is anchored on ``2026-01-01T00:00Z``, the day the loader's data ends. A ``now()``-
  based window would prove nothing here: the machine clock is past the data, so ``now()`` prunes
  every partition away and every query would "pass".
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Final

import psycopg
import pytest

from mdqm.adapters.postgresql import PostgresAdapter
from mdqm.adapters.postgresql_plan import pruning_verdict
from mdqm.config.settings import SourceConfig
from mdqm.errors import ErrorCode
from mdqm.models import Fragment, Partitioning, PruningProof
from mdqm.planner.partitions import Range, estimate_partition_scan, gate_plan
from mdqm.testing.loaders.postgres import PgTarget, target_from_environ

ANCHOR: Final = datetime(2026, 1, 1, tzinfo=UTC)
"""The end of the loader's window. Fixed, so the test does not depend on the machine clock."""

WINDOW: Final = Range(ANCHOR - timedelta(days=30), ANCHOR)
"""``[2025-12-02, 2026-01-01)``: the 30 days before the anchor."""

NOVEMBER: Final = Range(datetime(2025, 11, 5, tzinfo=UTC), datetime(2025, 12, 1, tzinfo=UTC))
"""A second window, used to show the verdict follows the window instead of a constant."""

EVENT_SQL: Final = f"""
SELECT client_no, sum(units) AS units
  FROM public.usage_events
 WHERE start_time >= '{WINDOW.lower.isoformat()}'::timestamptz
   AND start_time <  '{WINDOW.upper.isoformat()}'::timestamptz
 GROUP BY client_no ORDER BY client_no
"""

INGESTION_SQL: Final = f"""
SELECT client_no, sum(units) AS units
  FROM public.usage_events
 WHERE insertion_time >= '{WINDOW.lower.isoformat()}'::timestamptz
   AND insertion_time <  '{WINDOW.upper.isoformat()}'::timestamptz
 GROUP BY client_no ORDER BY client_no
"""

UNBOUNDED_SQL: Final = """
SELECT client_no, count(*) AS events FROM public.usage_events GROUP BY client_no
"""


SECRET_REF: Final = "mdqm_a03_read_only"
"""The secret *name* the config carries; the value comes from ``.env``."""


class _StaticSecrets:
    """A secret provider returning the throwaway dev password (the name is what config carries)."""

    def __init__(self, password: str) -> None:
        self._password = password

    def get(self, name: str) -> str:
        """Resolve any reference to the throwaway dev password."""
        assert name, "a secret reference must be a name, never an empty string"
        return self._password


def _unavailable(request: pytest.FixtureRequest, reason: str) -> None:
    """Skip, or fail when ``--require-services`` was given (a skipped test is not a pass)."""
    if request.config.getoption("--require-services"):
        pytest.fail(f"required service unavailable: {reason}")
    pytest.skip(f"service unavailable: {reason}")


@pytest.fixture(scope="module")
def target(request: pytest.FixtureRequest) -> PgTarget:
    """Where the compose PostgreSQL lives, from ``.env`` (never a hard-coded credential)."""
    try:
        found = target_from_environ()
    except Exception as exc:
        _unavailable(request, f"PostgreSQL connection settings: {exc}")
        raise AssertionError("unreachable") from exc
    try:
        with psycopg.connect(found.dsn("crm"), connect_timeout=5):
            pass
    except psycopg.Error as exc:
        _unavailable(request, f"PostgreSQL at {found.host}:{found.port}: {exc}")
        raise AssertionError("unreachable") from exc
    return found


@pytest.fixture(scope="module")
def secrets(target: PgTarget) -> _StaticSecrets:
    return _StaticSecrets(target.ro_password)


@pytest.fixture(scope="module")
def billing(target: PgTarget, secrets: _StaticSecrets) -> Iterator[PostgresAdapter]:
    adapter = PostgresAdapter(
        SourceConfig(
            name="billing",
            adapter="postgresql",
            secret=SECRET_REF,
            options={
                "host": target.host,
                "port": target.port,
                "dbname": "billing",
                "user": target.ro_user,
                "connect_timeout": 5,
            },
        ),
        secrets,
    )
    try:
        yield adapter
    finally:
        adapter.close()


@pytest.fixture(scope="module")
def erp(target: PgTarget, secrets: _StaticSecrets) -> Iterator[PostgresAdapter]:
    adapter = PostgresAdapter(
        SourceConfig(
            name="erp",
            adapter="postgresql",
            secret=SECRET_REF,
            options={
                "host": target.host,
                "port": target.port,
                "dbname": "erp",
                "user": target.ro_user,
                "connect_timeout": 5,
            },
        ),
        secrets,
    )
    try:
        yield adapter
    finally:
        adapter.close()


@pytest.fixture(scope="module")
def usage_events(billing: PostgresAdapter) -> Partitioning:
    """The table's declared metadata, as the adapter reads it from the catalog."""
    found = [entry for entry in billing.extract_partitions() if entry.key == ("start_time",)]
    assert len(found) == 1, (
        f"no RANGE partitioning on start_time among {billing.extract_partitions()}"
    )
    return found[0]


def _fragment(adapter: PostgresAdapter, statement: str) -> Fragment:
    return Fragment(source=adapter.source, dialect="postgres", sql=statement, exec_ts=ANCHOR)


_BOUND_RE: Final = re.compile(r"FOR VALUES FROM \('(?P<lower>[^']+)'\) TO \('(?P<upper>[^']+)'\)")


def _server_partitions(target: PgTarget) -> list[tuple[str, datetime, datetime]]:
    """``(name, lower, upper)`` from the server's bound text, parsed here, not by the adapter."""
    with psycopg.connect(target.dsn("billing")) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT c.relname, pg_get_expr(c.relpartbound, c.oid) "
            "FROM pg_inherits i "
            "JOIN pg_class p ON p.oid = i.inhparent "
            "JOIN pg_class c ON c.oid = i.inhrelid "
            "WHERE p.relname = 'usage_events' ORDER BY c.relname"
        )
        rows = cur.fetchall()
    found: list[tuple[str, datetime, datetime]] = []
    for name, definition in rows:
        match = _BOUND_RE.search(str(definition))
        assert match is not None, f"unexpected bound text: {definition}"
        lower = datetime.fromisoformat(match.group("lower"))
        upper = datetime.fromisoformat(match.group("upper"))
        found.append(
            (
                str(name),
                lower if lower.tzinfo else lower.replace(tzinfo=UTC),
                upper if upper.tzinfo else upper.replace(tzinfo=UTC),
            )
        )
    return found


def _spans(target: PgTarget, window: Range) -> list[str]:
    """The partitions whose declared range overlaps ``window``, computed from the server's text."""
    return [
        name
        for name, lower, upper in _server_partitions(target)
        if lower < window.upper and upper > window.lower
    ]


# --------------------------------------------------------------------------------------
# Acceptance 1: an event-time window prunes, and the verdict says so
# --------------------------------------------------------------------------------------


def test_an_event_time_window_touches_at_most_two_of_the_partitions(
    billing: PostgresAdapter, usage_events: Partitioning, target: PgTarget
) -> None:
    """The indexed, partitioned column: one partition named, and the server's text agrees."""
    declared = _server_partitions(target)
    assert len(declared) == len(usage_events.bounds) == 14

    plan = billing.explain(_fragment(billing, EVENT_SQL))
    assert plan.partitions_total == len(declared) == 14
    assert plan.partitions_scanned is not None
    assert plan.partitions_scanned <= 2, f"the plan named {plan.partitions_scanned} partitions"
    assert plan.raw is not None

    verdict = pruning_verdict(plan, usage_events, WINDOW)
    assert verdict.proof is PruningProof.PROVEN
    assert verdict.expected == verdict.named
    assert list(verdict.named) == _spans(target, WINDOW) == ["usage_events_p2025_12"]
    assert verdict.ok


def test_the_verdict_follows_the_window_instead_of_a_constant(
    billing: PostgresAdapter, usage_events: Partitioning, target: PgTarget
) -> None:
    """A different month proves a different partition: the answer is derived, not remembered."""
    statement = (
        "SELECT count(*) FROM public.usage_events "
        f"WHERE start_time >= '{NOVEMBER.lower.isoformat()}'::timestamptz "
        f"AND start_time < '{NOVEMBER.upper.isoformat()}'::timestamptz"
    )
    plan = billing.explain(_fragment(billing, statement))
    verdict = pruning_verdict(plan, usage_events, NOVEMBER)
    assert verdict.proof is PruningProof.PROVEN
    assert verdict.named == ("usage_events_p2025_11",)
    assert list(verdict.named) == _spans(target, NOVEMBER)


# --------------------------------------------------------------------------------------
# Acceptance 2: the same window on ingestion time prunes nothing
# --------------------------------------------------------------------------------------


def test_the_same_window_on_ingestion_time_touches_every_partition(
    billing: PostgresAdapter, usage_events: Partitioning, target: PgTarget
) -> None:
    """``insertion_time`` is not the partition key: all 14 partitions are named, and it shows."""
    plan = billing.explain(_fragment(billing, INGESTION_SQL))
    total = len(_server_partitions(target))
    assert plan.partitions_total == total
    assert plan.partitions_scanned == total, (
        f"an ingestion-time filter must not prune: {plan.partitions_scanned} of {total}"
    )

    verdict = pruning_verdict(plan, usage_events, WINDOW)
    assert verdict.proof is PruningProof.UNPROVEN
    assert len(verdict.named) == total
    assert len(verdict.unexpected) == total - 1
    assert verdict.missing == ()
    assert "outside the window" in verdict.reason


def test_a_fragment_with_no_time_predicate_touches_every_partition(
    billing: PostgresAdapter, usage_events: Partitioning, target: PgTarget
) -> None:
    """No window in the fragment: every partition is scanned, and no proof can be claimed."""
    plan = billing.explain(_fragment(billing, UNBOUNDED_SQL))
    total = len(_server_partitions(target))
    assert plan.partitions_scanned == plan.partitions_total == total

    without_window = pruning_verdict(plan, usage_events, None)
    assert without_window.proof is PruningProof.UNKNOWN
    assert "no window" in without_window.reason

    with_window = pruning_verdict(plan, usage_events, WINDOW)
    assert with_window.proof is PruningProof.UNPROVEN
    assert len(with_window.named) == total == len(with_window.unexpected) + 1


# --------------------------------------------------------------------------------------
# Acceptance 3: a table with no partitions is UNKNOWN, never a fake PROVEN
# --------------------------------------------------------------------------------------


def test_a_non_partitioned_table_reports_no_partition_count_and_no_proof(
    erp: PostgresAdapter,
) -> None:
    """``erp.customers`` has no partitions: there is no count and nothing to prove."""
    plan = erp.explain(_fragment(erp, "SELECT cust_no, name FROM public.customers"))
    assert plan.partitions_total is None
    assert plan.partitions_scanned is None

    verdict = pruning_verdict(plan, None, WINDOW)
    assert verdict.proof is PruningProof.UNKNOWN
    assert not verdict.ok
    assert verdict.expected == verdict.named == ()


# --------------------------------------------------------------------------------------
# Acceptance 4: the planner consumes the verdict, and the cap refuses an unpruned plan
# --------------------------------------------------------------------------------------


def test_the_static_estimate_and_the_plan_agree_for_a_good_window(
    billing: PostgresAdapter, usage_events: Partitioning
) -> None:
    """The static count and the source's own plan agree: the plan gate has nothing to add."""
    plan = billing.explain(_fragment(billing, EVENT_SQL))
    estimate = estimate_partition_scan(usage_events, WINDOW)
    assert estimate.partitions == 1, "the declared bounds put the window in one partition"
    assert plan.partitions_scanned == estimate.partitions
    assert (
        gate_plan(
            plan,
            table="public.usage_events",
            partitioning=usage_events,
            max_partitions_scanned=8,
        )
        is None
    )


def test_a_plan_over_the_cap_is_refused_with_partition_scan_all(
    billing: PostgresAdapter, usage_events: Partitioning
) -> None:
    """The wrong column passes every static check; only the plan proves the scan is too wide."""
    plan = billing.explain(_fragment(billing, INGESTION_SQL))
    refusal = gate_plan(
        plan,
        table="public.usage_events",
        partitioning=usage_events,
        max_partitions_scanned=8,
    )
    assert refusal is not None
    assert refusal.code == ErrorCode.PARTITION_SCAN_ALL.value
    assert "14 of 14" in refusal.problem
    assert "start_time" in refusal.problem
    assert '"time"' in refusal.fix

    # The same plan under a cap that allows it is not refused: the gate is the number, not a mood.
    assert (
        gate_plan(
            plan,
            table="public.usage_events",
            partitioning=usage_events,
            max_partitions_scanned=plan.partitions_total or 0,
        )
        is None
    )
