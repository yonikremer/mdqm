"""ASYNC-QUERY: the asynchronous query job -- start, poll, stop, and the blocking default.

What these tests are for, in the owner's words: *return feedback about a query fast, before it
completes*. The claims, each asserted as a behaviour:

* ``query(mode="async")`` answers at once with a ``query_id`` **and the plan**, and the work
  finishes on its own; every refusal that needs no source still comes back inline.
* ``get_more(query_id=...)`` reports the job from its stored row, so a second message (or a second
  process) sees the same state, and a finished job hands back the summary, sample and
  ``result_handle`` a blocking call would have returned.
* the numbers are measured, never invented: ``rows_read`` is what the governor drained and
  ``partitions_done`` is what the source's own plan said, and neither moves until something reports.
* ``cancel`` reaches the driver: the governor calls the adapter's own ``cancel`` on the live stream,
  the run ends ``cancelled`` with ``CANCELLED`` (not ``TIMEOUT``), and the circuit breaker -- which
  counts a slow *source* -- is told nothing.
* the **default is unchanged**: without ``mode`` the tool blocks and no job row is written, because
  five acceptance tests and every existing caller depend on that shape.

The pipeline is the real one -- an F05 fake adapter over in-process DuckDB, the real compiler,
planner, governor, history and audit -- driven through the tool functions with the SDK's auth context
bound, exactly as ``tests/unit/test_tool_query.py`` does. No transport, no Docker, and no sleeping
except where a slow source is the thing under test.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import duckdb
import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

from mdqm.errors import ErrorCode
from mdqm.errors import cancelled as cancelled_error
from mdqm.errors import timeout as timeout_error
from mdqm.executor.cache import FragmentCache
from mdqm.governor import BreakerState, Governor
from mdqm.models import (
    Limits,
    PartitionBound,
    PartitionGranularity,
    Partitioning,
    PartitionStrategy,
)
from mdqm.semantic.loader import load
from mdqm.semantic.models import SemanticLayer
from mdqm.server.auth import READ_SCOPE
from mdqm.server.jobs import JobHooks, JobOutcome, JobProgress, JobQueueFull, JobRegistry
from mdqm.server.tools import (
    GetMoreAnswer,
    QueryAnswer,
    QueryServices,
    ToolServices,
    get_more,
    query,
    use_query_services,
    use_services,
)
from mdqm.spec.models import QuerySpec
from mdqm.store import Store
from mdqm.store.audit import AuditLog, AuditOutcome
from mdqm.store.feedback import FeedbackStore
from mdqm.store.history import HistoryStore
from mdqm.store.jobs import JobState, QueryJobStore
from mdqm.testing.fake_adapter import FakeAdapter, table_from_rows
from mdqm.testing.world import World, WorldConfig, build_world

#: One execution timestamp for the whole module (C02: one value per execution), inside the window
#: the world's own ``usage_events`` rows cover.
EXEC_TS: Final = datetime(2025, 12, 15, 12, 0, tzinfo=UTC)

#: Caps the tests expect: nothing here should trip a cap except the job ceiling, which is its own.
LIMITS: Final = Limits(timeout_s=30.0, max_rows=50_000, max_bytes=64 << 20, max_partitions=64)

#: The layer these tests run under: one source, one measure, one partitioned table with three
#: monthly bounds, so the plan has a partition total a poll can report.
LAYER_YAML: Final = """
version: 1

sources:
  billing: { adapter: postgresql, secret: billing_ro }

measures:
  usage_units:
    sources:
      billing.usage_events.units: { normalize: "{col}" }

tables:
  billing.accounts:
    key: [client_no]
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
    mutability: { class: append_rarely, status: approved }
"""

#: A one-source spec with no time window: it pages a plain table and nothing is pruned.
PLAIN_SPEC: Final[dict[str, Any]] = {
    "from": "billing.accounts",
    "select": ["client_no"],
    "order_by": [{"by": "client_no", "direction": "asc"}],
    "limit": 50,
}

#: The partitioned spec: three monthly partitions in the window, so the plan has a total.
WINDOW_SPEC: Final[dict[str, Any]] = {
    "from": "billing.usage_events",
    "select": [{"measure": "usage_units", "agg": "count"}],
    "time": {"mode": "events_in_range", "from": "2025-10-01", "to": "2026-01-01"},
    "limit": 10,
}

#: The partition metadata the fake source reports: the same three months the layer declares, so a
#: plan really does say "one partition of three" and the poll can show the number it was told.
USAGE_PARTITIONING: Final = Partitioning(
    strategy=PartitionStrategy.RANGE,
    key=("start_time",),
    granularity=PartitionGranularity.MONTH,
    bounds=(
        PartitionBound(
            name="p2025_10",
            lower=datetime(2025, 10, 1, tzinfo=UTC),
            upper=datetime(2025, 11, 1, tzinfo=UTC),
        ),
        PartitionBound(
            name="p2025_11",
            lower=datetime(2025, 11, 1, tzinfo=UTC),
            upper=datetime(2025, 12, 1, tzinfo=UTC),
        ),
        PartitionBound(
            name="p2025_12",
            lower=datetime(2025, 12, 1, tzinfo=UTC),
            upper=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    ),
)


class ScriptedClock:
    """A clock the test drives, so no assertion depends on wall-clock time."""

    def __init__(self, now: datetime = EXEC_TS) -> None:
        self._now = now
        self._mono = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def __call__(self) -> datetime:
        """The plain-callable shape history, feedback and the audit log take."""
        return self._now

    def advance(self, delta: timedelta) -> datetime:
        self._now += delta
        self._mono += delta.total_seconds()
        return self._now


# --------------------------------------------------------------------------------------------------
# a harness: the real pipeline plus one registry, in a throwaway database
# --------------------------------------------------------------------------------------------------


@dataclass
class Harness:
    """One configured server: the query tool's services and the lookup tools' services."""

    services: QueryServices
    tool: ToolServices
    registry: JobRegistry
    adapters: Mapping[str, FakeAdapter]
    store: Store
    clock: ScriptedClock


@pytest.fixture(scope="module")
def world() -> World:
    """The seeded D02 world, smallest size: the same generator, fewer rows."""
    return build_world(WorldConfig.small())


@pytest.fixture(scope="module")
def layer(tmp_path_factory: pytest.TempPathFactory) -> SemanticLayer:
    """The layer above, written once and loaded through the real loader."""
    path = tmp_path_factory.mktemp("layer") / "async.yaml"
    path.write_text(LAYER_YAML, encoding="utf-8")
    return load(path)


@pytest.fixture
def clock() -> ScriptedClock:
    return ScriptedClock()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    """A migrated metadata DB on a throwaway file: jobs, history and the audit chain live here."""
    opened = Store(tmp_path / "meta.sqlite")
    opened.initialize()
    try:
        yield opened
    finally:
        opened.close()


@pytest.fixture
def make(
    world: World, layer: SemanticLayer, clock: ScriptedClock, store: Store
) -> Iterator[Callable[..., Harness]]:
    """Build a harness per call, so a test can vary the fake source's speed and the job ceiling."""
    cleanup: list[Callable[[], None]] = []

    def factory(
        *,
        latency_s: float = 0.0,
        batch_size: int = 1024,
        max_jobs: int = 8,
        with_registry: bool = True,
    ) -> Harness:
        adapter = FakeAdapter(
            name="billing",
            tables={
                "accounts": table_from_rows(world.billing.accounts),
                "usage_events": table_from_rows(world.billing.usage_events),
            },
            latency_s=latency_s,
            batch_size=batch_size,
            partitioning={"usage_events": USAGE_PARTITIONING},
        )
        con = duckdb.connect(":memory:")
        con.execute("SET TimeZone='UTC'").fetchall()
        adapters: dict[str, FakeAdapter] = {"billing": adapter}
        columns: dict[str, tuple[str, ...]] = {}
        for source, each in adapters.items():
            for table in each.extract_catalog().tables:
                columns[f"{source}.{table.name}"] = tuple(column.name for column in table.columns)
        registry = JobRegistry(
            QueryJobStore(store, clock=clock),
            clock=clock,
            max_jobs=max_jobs,
            progress_interval_s=0.0,
        )
        governor = Governor(adapters, clock=clock)
        history = HistoryStore(store, clock=clock)
        audit = AuditLog(store, clock=clock)
        services = QueryServices(
            layer=layer,
            governor=governor,
            history=history,
            audit=audit,
            adapters=adapters,
            columns=columns,
            cache=FragmentCache(con, clock=clock),
            clock=clock,
            limits=LIMITS,
            jobs=registry if with_registry else None,
        )
        tool = ToolServices(
            handles=governor.store,
            history=history,
            feedback=FeedbackStore(store, clock=clock),
            audit=audit,
            jobs=registry if with_registry else None,
        )

        def release() -> None:
            registry.shutdown(timeout_s=10.0)
            con.close()
            adapter.close()

        cleanup.append(release)
        return Harness(
            services=services,
            tool=tool,
            registry=registry,
            adapters=adapters,
            store=store,
            clock=clock,
        )

    try:
        yield factory
    finally:
        for step in reversed(cleanup):
            step()


@contextmanager
def configured(harness: Harness) -> Iterator[Harness]:
    """Install both process-wide slots for the block: the query tool and the lookup tools."""
    with use_query_services(harness.services), use_services(harness.tool):
        yield harness


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


def ask(spec: Mapping[str, Any], *, mode: str = "blocking") -> QueryAnswer:
    """Call ``query`` the way the SDK does: the arguments are validated into the wire models."""
    return query(QuerySpec.model_validate(spec), mode=mode)  # type: ignore[arg-type]


def poll(query_id: str, *, cancel: bool = False) -> GetMoreAnswer:
    """Call ``get_more`` with a job id, as the agent would."""
    return get_more(query_id=query_id, cancel=cancel)


def count_jobs(store: Store) -> int:
    """How many job rows exist: the evidence that the blocking path wrote none."""
    row = store.connect().execute("SELECT COUNT(*) FROM query_job").fetchone()
    return int(row[0]) if row is not None else -1


def wait_for_running(harness: Harness, query_id: str, *, timeout_s: float = 5.0) -> JobProgress:
    """Poll the registry until the job's worker has started, or fail: no bare sleeps."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        progress = harness.registry.status(query_id, owner="alice")
        if progress is not None and progress.state is JobState.RUNNING:
            return progress
        time.sleep(0.01)
    raise AssertionError("the job never reached running")


def poll_until_finished(
    harness: Harness, query_id: str, *, timeout_s: float = 15.0
) -> GetMoreAnswer:
    """Wait for the job's thread, then poll once: no retry loop, no flakiness."""
    assert harness.registry.wait(query_id, timeout_s=timeout_s), "the job never finished"
    return poll(query_id)


# --------------------------------------------------------------------------------------------------
# the stored row: created, visible, owned
# --------------------------------------------------------------------------------------------------


def test_a_started_job_is_a_row_before_its_worker_does_anything(
    store: Store, clock: ScriptedClock
) -> None:
    """The poll answers from the database, so the row must exist the moment `start` returns."""
    registry = JobRegistry(QueryJobStore(store, clock=clock), clock=clock)
    gate = threading.Event()
    entered = threading.Event()

    def work(hooks: JobHooks) -> JobOutcome:
        entered.set()
        gate.wait(5.0)
        return JobOutcome(rows_total=0)

    job = registry.start(user_id="alice", source="billing", work=work)
    try:
        assert job.state is JobState.QUEUED
        assert job.query_id == f"j_{job.id}"
        assert count_jobs(store) == 1
        assert entered.wait(5.0), "the worker never started"
        progress = registry.status(job.query_id, owner="alice")
        assert progress is not None
        assert progress.state is JobState.RUNNING
        assert progress.finished is False
    finally:
        gate.set()
        registry.shutdown()


def test_a_finished_job_reports_the_handle_it_earned(store: Store, clock: ScriptedClock) -> None:
    """`done` means the poll carries the handle, and a second poll says exactly the same thing."""
    registry = JobRegistry(QueryJobStore(store, clock=clock), clock=clock)
    job = registry.start(
        user_id="alice",
        source="billing",
        work=lambda hooks: JobOutcome(result_handle="h_abc", rows_total=3),
    )
    assert registry.wait(job.query_id)

    first = registry.status(job.query_id, owner="alice")
    second = registry.status(job.query_id, owner="alice")
    assert first is not None and second is not None
    assert first.state is JobState.DONE
    assert first.result_handle == "h_abc" and first.rows_total == 3
    assert first.finished_at is not None
    assert second.model_dump() == first.model_dump()
    assert first.finished is True


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (JobOutcome(result_handle="h_1", rows_total=1), JobState.DONE),
        (JobOutcome(error=timeout_error(seconds=1.0)), JobState.FAILED),
        (JobOutcome(error=cancelled_error()), JobState.CANCELLED),
    ],
    ids=["answered", "failed", "cancelled"],
)
def test_the_terminal_state_follows_the_run(
    store: Store, clock: ScriptedClock, outcome: JobOutcome, expected: JobState
) -> None:
    """A run that answers is `done`; one carrying an error is `failed`; a cancellation is neither."""
    registry = JobRegistry(QueryJobStore(store, clock=clock), clock=clock)
    job = registry.start(user_id="alice", source="billing", work=lambda hooks: outcome)
    assert registry.wait(job.query_id)
    progress = registry.status(job.query_id, owner="alice")
    assert progress is not None
    assert progress.state is expected
    assert progress.error == outcome.error


def test_a_raised_exception_becomes_a_structured_error_not_a_traceback(
    store: Store, clock: ScriptedClock
) -> None:
    """A bug inside the run is reported as `INTERNAL_ERROR`, scrubbed, with nothing leaking."""
    registry = JobRegistry(QueryJobStore(store, clock=clock), clock=clock)

    def explode(hooks: JobHooks) -> JobOutcome:
        raise RuntimeError("the driver exploded with a secret: hunter2")

    job = registry.start(user_id="alice", source="billing", work=explode)
    assert registry.wait(job.query_id)
    progress = registry.status(job.query_id, owner="alice")
    assert progress is not None and progress.error is not None
    assert progress.state is JobState.FAILED
    assert progress.error.code == ErrorCode.INTERNAL_ERROR.value
    assert "hunter2" not in progress.error.problem + progress.error.fix
    assert "RuntimeError" not in progress.error.problem


def test_progress_is_what_was_measured_and_nothing_else(store: Store, clock: ScriptedClock) -> None:
    """Before a batch is drained the counters are zero; after, they are the numbers the sink got."""
    registry = JobRegistry(QueryJobStore(store, clock=clock), clock=clock, progress_interval_s=0.0)
    gate = threading.Event()
    entered = threading.Event()

    def work(hooks: JobHooks) -> JobOutcome:
        entered.set()
        gate.wait(5.0)
        hooks.plan(2, 5)
        hooks.progress(40, 640)
        return JobOutcome(result_handle="h_9", rows_total=40)

    job = registry.start(user_id="alice", source="billing", work=work, est_rows=100.0)
    try:
        assert entered.wait(5.0)
        before = registry.status(job.query_id, owner="alice")
        assert before is not None
        assert (before.rows_read, before.bytes_read) == (0, 0)
        assert (before.partitions_done, before.partitions_total) == (None, None)
        assert before.est_rows == 100.0
        gate.set()
        assert registry.wait(job.query_id)
    finally:
        gate.set()
        registry.shutdown()
    after = registry.status(job.query_id, owner="alice")
    assert after is not None
    assert (after.rows_read, after.bytes_read) == (40, 640)
    assert (after.partitions_done, after.partitions_total) == (2, 5)


def test_another_users_job_is_invisible_and_looks_unknown(
    store: Store, clock: ScriptedClock
) -> None:
    """Ownership is checked on the row: bob cannot tell alice's job from an id nobody issued."""
    registry = JobRegistry(QueryJobStore(store, clock=clock), clock=clock)
    job = registry.start(
        user_id="alice", source="billing", work=lambda hooks: JobOutcome(rows_total=0)
    )
    assert registry.wait(job.query_id)
    assert registry.status("j_99999", owner="bob") is None
    assert registry.status(job.query_id, owner="bob") is None
    assert registry.cancel(job.query_id, owner="bob") is None
    assert registry.status(job.query_id, owner="alice") is not None


def test_a_row_left_running_by_a_restart_is_reported_as_lost(
    store: Store, clock: ScriptedClock
) -> None:
    """A `running` row with no worker is marked failed with a reason, never left lying."""
    live = JobRegistry(QueryJobStore(store, clock=clock), clock=clock)
    gate = threading.Event()

    def work(hooks: JobHooks) -> JobOutcome:
        gate.wait(5.0)
        return JobOutcome(rows_total=0)

    job = live.start(user_id="alice", source="billing", work=work)
    # A second registry over the same database stands in for a restarted process.
    restarted = JobRegistry(QueryJobStore(store, clock=clock), clock=clock)
    try:
        assert live.status(job.query_id, owner="alice") is not None
        progress = restarted.status(job.query_id, owner="alice")
        assert progress is not None
        assert progress.state is JobState.FAILED
        assert "restarted" in progress.note
        assert progress.error is not None
        assert progress.error.code == ErrorCode.INTERNAL_ERROR.value
    finally:
        gate.set()
        live.shutdown()
        restarted.shutdown()


def test_the_ceiling_refuses_one_more_job(store: Store, clock: ScriptedClock) -> None:
    """The ceiling is this process's thread budget, and it refuses out loud rather than queuing."""
    registry = JobRegistry(QueryJobStore(store, clock=clock), clock=clock, max_jobs=1)
    gate = threading.Event()
    try:
        registry.start(
            user_id="alice",
            source="billing",
            work=lambda hooks: (gate.wait(5.0), JobOutcome(rows_total=0))[1],
        )
        with pytest.raises(JobQueueFull):
            registry.start(user_id="alice", source="billing", work=lambda hooks: JobOutcome())
    finally:
        gate.set()
        registry.shutdown()


def test_a_finished_job_survives_a_new_registry(store: Store, clock: ScriptedClock) -> None:
    """The row is the durable answer: a fresh registry still reports `done` and the handle."""
    first = JobRegistry(QueryJobStore(store, clock=clock), clock=clock)
    job = first.start(
        user_id="alice",
        source="billing",
        work=lambda hooks: JobOutcome(result_handle="h_kept", rows_total=4),
    )
    assert first.wait(job.query_id)
    second = JobRegistry(QueryJobStore(store, clock=clock), clock=clock)
    reported = second.status(job.query_id, owner="alice")
    assert reported is not None
    assert reported.state is JobState.DONE
    assert reported.result_handle == "h_kept"
    assert reported.rows_total == 4


def test_the_job_table_refuses_a_state_it_does_not_declare(
    store: Store, clock: ScriptedClock
) -> None:
    """The CHECK is the state machine: a row cannot invent a sixth state."""
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        store.connect().execute(
            "INSERT INTO query_job (user_id, query_id, state, created_at) VALUES (?, ?, ?, ?)",
            ("alice", "j_bad", "zombie", clock.now().isoformat()),
        )


def test_mark_finished_refuses_a_state_a_running_job_cannot_reach(
    store: Store, clock: ScriptedClock
) -> None:
    """A "finished" write must be terminal, or the row could go backwards unnoticed."""
    jobs = QueryJobStore(store, clock=clock)
    job = jobs.create(user_id="alice", source="billing")
    with pytest.raises(Exception, match="not a terminal state"):
        jobs.mark_finished(job.query_id, state=JobState.RUNNING)


# --------------------------------------------------------------------------------------------------
# the tool: async answers at once, and the blocking default is untouched
# --------------------------------------------------------------------------------------------------


def test_an_async_query_answers_twice_about_the_same_job(
    make: Callable[..., Harness],
) -> None:
    """The whole path: start, poll, poll again, then page the handle -- one answer, two reads."""
    harness = make()
    with configured(harness), as_user():
        started = ask(WINDOW_SPEC, mode="async")
        assert started.status == "started", started.error
        assert started.query_id is not None
        assert started.query_id.startswith("j_")
        assert started.summary is None and started.result_handle is None
        # The fast feedback: the plan (partitions and rows the window touches) is in the answer.
        assert started.plan is not None
        assert started.plan.partitions_total == 3

        finished = poll_until_finished(harness, started.query_id)
        again = poll(started.query_id)
        page = get_more(result_handle=finished.result_handle or "", limit=5)

    assert finished.status == "done", finished.error
    assert finished.error is None
    assert finished.result_handle is not None
    assert finished.result_handle.startswith("h_")
    assert finished.rows_total > 0
    assert finished.progress is not None
    assert finished.progress.state is JobState.DONE
    assert finished.progress.partitions_total == 3
    # The fake source's own plan says one partition of the three: reported, not guessed.
    assert finished.progress.partitions_done == 1
    assert finished.progress.rows_read > 0
    assert finished.summary is not None
    assert finished.summary.rows_total == finished.rows_total
    assert again.status == "done"
    assert again.result_handle == finished.result_handle
    assert again.sample == finished.sample
    assert page.status == "ok", page.error
    assert page.page is not None and page.page.rows_total == finished.rows_total


def test_the_blocking_default_is_unchanged_and_writes_no_job(make: Callable[..., Harness]) -> None:
    """Without `mode` the tool waits, answers with rows, and records no job at all."""
    harness = make()
    with configured(harness), as_user():
        answer = ask(PLAIN_SPEC)
    assert answer.status == "ok", answer.error
    assert answer.result_handle is not None
    assert answer.rows_total == 50
    assert answer.query_id is None
    assert count_jobs(harness.store) == 0


def test_a_refusal_in_async_mode_still_comes_back_inline(make: Callable[..., Harness]) -> None:
    """A typo costs no job: every check that needs no source happens before the job is started."""
    harness = make()
    with configured(harness), as_user():
        answer = ask(
            {"from": "billing.accounts", "select": ["client_noo"], "limit": 5}, mode="async"
        )
    assert answer.status == "error"
    assert answer.error is not None
    assert answer.error.code == ErrorCode.UNKNOWN_COLUMN.value
    assert answer.query_id is None
    assert count_jobs(harness.store) == 0


def test_a_dry_run_in_async_mode_answers_inline(make: Callable[..., Harness]) -> None:
    """`dry_run` asks for the plan, which is what the started answer carries: no job is created."""
    harness = make()
    with configured(harness), as_user():
        answer = ask({**WINDOW_SPEC, "dry_run": True}, mode="async")
    assert answer.status == "dry_run"
    assert answer.query_id is None
    assert answer.plan is not None and answer.plan.partitions_total == 3
    assert count_jobs(harness.store) == 0


def test_async_without_a_registry_names_the_fallback(make: Callable[..., Harness]) -> None:
    """A build with no registry refuses with NOT_SUPPORTED and says what to do instead."""
    harness = make(with_registry=False)
    with configured(harness), as_user():
        answer = ask(PLAIN_SPEC, mode="async")
    assert answer.status == "error"
    assert answer.error is not None
    assert answer.error.code == ErrorCode.NOT_SUPPORTED.value
    assert "mode" in answer.error.fix
    assert count_jobs(harness.store) == 0


def test_the_job_ceiling_refuses_with_a_quota_error(make: Callable[..., Harness]) -> None:
    """A full registry is a quota refusal with a fix, not a queued job nobody will run."""
    harness = make(max_jobs=1)
    gate = threading.Event()
    try:
        with configured(harness), as_user():
            first = ask(PLAIN_SPEC, mode="async")
            assert first.status == "started", first.error
            assert first.query_id is not None
            poll_until_finished(harness, first.query_id)
            harness.registry.start(
                user_id="alice",
                source="billing",
                work=lambda hooks: (gate.wait(5.0), JobOutcome(rows_total=0))[1],
            )
            second = ask(PLAIN_SPEC, mode="async")
    finally:
        gate.set()
    assert second.status == "error"
    assert second.error is not None
    assert second.error.code == ErrorCode.QUOTA_EXCEEDED.value
    assert "retry" in second.error.fix


# --------------------------------------------------------------------------------------------------
# stopping a job
# --------------------------------------------------------------------------------------------------


def test_cancel_stops_the_source_and_ends_the_job_as_cancelled(
    make: Callable[..., Harness],
) -> None:
    """The stop reaches the driver's own cancel, and the agent sees `cancelled`, not `timeout`."""
    harness = make(latency_s=0.05, batch_size=1)
    with configured(harness), as_user():
        started = ask(PLAIN_SPEC, mode="async")
        assert started.status == "started", started.error
        assert started.query_id is not None
        wait_for_running(harness, started.query_id)
        requested = poll(started.query_id, cancel=True)
        assert requested.status in {"queued", "running"}
        assert requested.progress is not None
        assert "caller" in requested.progress.note
        finished = poll_until_finished(harness, started.query_id)

    assert finished.status == "cancelled", finished.error
    assert finished.error is not None
    assert finished.error.code == ErrorCode.CANCELLED.value
    assert finished.result_handle is None
    # The run was recorded: a stopped query is an audited refusal, not a gap.
    outcomes = [row.outcome for row in harness.services.audit.rows()]
    assert AuditOutcome.BLOCKED in outcomes


def test_a_cancelled_job_does_not_blame_the_source(make: Callable[..., Harness]) -> None:
    """The breaker counts a slow source; a caller's change of mind must leave it closed."""
    harness = make(latency_s=0.05, batch_size=1)
    with configured(harness), as_user():
        started = ask(PLAIN_SPEC, mode="async")
        assert started.query_id is not None
        wait_for_running(harness, started.query_id)
        poll(started.query_id, cancel=True)
        poll_until_finished(harness, started.query_id)
    assert harness.services.governor.breaker.report("billing").state is BreakerState.CLOSED


def test_cancelling_a_finished_job_changes_nothing(make: Callable[..., Harness]) -> None:
    """A stop that arrives too late is reported as the answer the job already produced."""
    harness = make()
    with configured(harness), as_user():
        started = ask(PLAIN_SPEC, mode="async")
        assert started.query_id is not None
        done = poll_until_finished(harness, started.query_id)
        late = poll(started.query_id, cancel=True)
    assert done.status == "done"
    assert late.status == "done"
    assert late.result_handle == done.result_handle


# --------------------------------------------------------------------------------------------------
# the two references: a handle pages, a job id watches
# --------------------------------------------------------------------------------------------------


def test_get_more_needs_exactly_one_reference(make: Callable[..., Harness]) -> None:
    """`result_handle` and `query_id` are different reference kinds, so sending both is refused."""
    harness = make()
    with configured(harness), as_user():
        both = get_more(result_handle="h_1", query_id="j_1")
        neither = get_more()
    for answer in (both, neither):
        assert answer.status == "error"
        assert answer.error is not None
        assert answer.error.code == ErrorCode.INVALID_SPEC.value


def test_an_unknown_job_and_another_users_job_read_the_same(make: Callable[..., Harness]) -> None:
    """No id probe: alice's job must look to bob exactly like one nobody issued."""
    harness = make()
    with configured(harness), as_user():
        started = ask(PLAIN_SPEC, mode="async")
        assert started.query_id is not None
        poll_until_finished(harness, started.query_id)
    with configured(harness), as_user("bob"):
        stolen = poll(started.query_id)
        unknown = poll("j_999999")
    assert stolen.status == "error" and unknown.status == "error"
    assert stolen.error is not None and unknown.error is not None
    assert stolen.error.code == ErrorCode.NOT_FOUND.value
    assert unknown.error.code == ErrorCode.NOT_FOUND.value
    assert stolen.error.problem == unknown.error.problem


def test_get_more_without_a_registry_is_honest(make: Callable[..., Harness]) -> None:
    """A build that runs no jobs says so instead of reporting a job it cannot see."""
    harness = make(with_registry=False)
    with configured(harness), as_user():
        answer = poll("j_1")
    assert answer.status == "not_implemented"
    assert answer.problem is not None and "query_id" in answer.problem
    assert answer.fix is not None and "result_handle" in answer.fix


# --------------------------------------------------------------------------------------------------
# the cross-source path: deferred whole, because it is several fragments and no single plan
# --------------------------------------------------------------------------------------------------


def test_a_cross_source_spec_is_also_deferred_whole(
    make: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A federated spec becomes one job too: the started answer says so, the poll carries the rows.

    The federated engine is tested where it lives; this test owns only the branch that hands it to a
    background thread, so the engine is replaced by a stand-in with the same surface.
    """
    from mdqm.server.tools import query_federated

    class _Plan:
        steps = ("a", "b")

    class _Federated:
        plan = _Plan()

    def fake_plan(services: Any, spec: Any, exec_ts: Any) -> _Federated:
        return _Federated()

    def fake_run(services: Any, user: str, **kwargs: Any) -> QueryAnswer:
        return QueryAnswer(user_id=user, result_handle="h_fed", rows_total=7)

    monkeypatch.setattr(query_federated, "federated_plan", fake_plan)
    monkeypatch.setattr(query_federated, "run_federated", fake_run)

    harness = make()
    with configured(harness), as_user():
        started = ask(PLAIN_SPEC, mode="async")
        assert started.status == "started", started.error
        assert started.plan is not None
        assert "no per-partition progress" in " ".join(started.plan.notes)
        assert started.query_id is not None
        finished = poll_until_finished(harness, started.query_id)

    assert finished.status == "done", finished.error
    assert finished.result_handle == "h_fed"
    assert finished.rows_total == 7
    assert finished.progress is not None
    # No plan was read, so no partition number is claimed: null means "not reported".
    assert finished.progress.partitions_done is None
