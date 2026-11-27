"""M03 against live PostgreSQL: the ``query`` tool, not the pieces under it.

Three claims, each one only a real server can settle:

* **the tool's answer is E01's answer.** The billing question ("average credit limit by account
  state") goes through the tool -- lint, compile, EXPLAIN, execute through the governor, summarise,
  record -- and the rows it returns are compared with the ground truth task E01 derived
  independently, inside the tolerance that document declares. A pipeline that quietly answered a
  different question fails here and nowhere in the unit tests.
* **the run is in history afterwards.** The tool's own row exists, carries the spec, the source and
  the exec_ts, and the audit chain still verifies -- so an answer is attributable to the rules and
  the request that produced it.
* **a missing time bound never reaches the database.** A query over the monthly-partitioned
  ``usage_events`` with no window is refused, and the refusal is checked against the server itself
  (``pg_stat_activity``), not against the tool's own message. A positive control proves the
  instrument works: the same measurement does see the bounded query that really ran.

Run it with::

    python scripts/dev.py up pg
    python -m pytest tests/integration/test_tool_query_e2e.py --require-services -q
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Final, cast

import duckdb
import psycopg
import pytest
from ruamel.yaml import YAML

from mdqm.adapters.registry import AdapterRegistry
from mdqm.config.settings import SourceConfig
from mdqm.errors import ErrorCode
from mdqm.executor.cache import FragmentCache
from mdqm.governor import Governor
from mdqm.models import Limits
from mdqm.server.tools import QueryServices, configure_query_services, query, reset_query_services
from mdqm.store import Store
from mdqm.store.audit import AuditLog
from mdqm.store.history import HistoryStore
from mdqm.testing.loaders.postgres import PgTarget, load, target_from_environ
from mdqm.testing.world import ANCHOR, WorldConfig

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
QUESTIONS: Final = REPO_ROOT / "tests" / "eval" / "questions.yaml"

#: The question checked against real data: billing stores the credit limit raw.
BILLING_QUESTION: Final = "avg_limit_by_status_billing"

#: One execution timestamp, inside the loaded world's window.
EXEC_TS: Final = datetime(2026, 11, 25, 15, 0, tzinfo=ANCHOR.tzinfo)

LIMITS: Final = Limits(timeout_s=60.0, max_rows=50_000, max_bytes=64 << 20, max_partitions=64)

#: A window that really filters the partition key: the last 30 days of event time.
BOUNDED_SPEC: Final[dict[str, Any]] = {
    "from": "billing.usage_events",
    "select": ["units"],
    "time": {"mode": "events_in_range", "last": "30d"},
    "limit": 10,
}

#: The same table with no window at all: all 14 monthly partitions would be read.
UNBOUNDED_SPEC: Final[dict[str, Any]] = {
    "from": "billing.usage_events",
    "select": ["units"],
    "limit": 10,
}


class FixedClock:
    """One instant for the whole run, and the plain-callable shape the stores take."""

    def __init__(self, moment: datetime = EXEC_TS) -> None:
        self._moment = moment

    def now(self) -> datetime:
        return self._moment

    def monotonic(self) -> float:
        return 0.0

    def __call__(self) -> datetime:
        return self._moment


class _PasswordSecrets:
    """Resolves the read-only role's password: the secret is the password and nothing else."""

    def __init__(self, target: PgTarget) -> None:
        self._target = target

    def get(self, name: str) -> str:
        assert name, "a secret reference is a name, never an empty string"
        return self._target.ro_password


def _unavailable(request: pytest.FixtureRequest, reason: str) -> None:
    """Skip, or fail when ``--require-services`` was given."""
    if request.config.getoption("--require-services"):
        pytest.fail(f"required service unavailable: {reason}")
    pytest.skip(f"service unavailable: {reason}")


@pytest.fixture(scope="module")
def pg(request: pytest.FixtureRequest) -> Iterator[PgTarget]:
    """The seeded PostgreSQL stack, or a clean skip (a failure with ``--require-services``)."""
    try:
        target = target_from_environ()
    except Exception as exc:  # noqa: BLE001 - any settings failure means "not configured here"
        _unavailable(request, f"PostgreSQL connection settings: {exc}")
        raise AssertionError("unreachable") from exc
    try:
        with psycopg.connect(target.dsn("billing"), connect_timeout=5):
            pass
    except psycopg.Error as exc:
        _unavailable(request, f"PostgreSQL at {target.host}:{target.port}: {exc}")
        raise AssertionError("unreachable") from exc
    yield target


@pytest.fixture(scope="module")
def seeded(pg: PgTarget) -> None:
    """Load the **default** world: E01's answers describe 2000 customers, not the small world."""
    load(pg, WorldConfig())


@pytest.fixture(scope="module")
def adapters(pg: PgTarget, seeded: None) -> Iterator[Mapping[str, Any]]:
    """The three logical sources, each a real ``PostgresAdapter`` over its own database."""
    from mdqm.adapters.postgresql import PostgresAdapter

    registry = AdapterRegistry(secrets=_PasswordSecrets(pg), entry_points=lambda: ())
    registry.register("postgresql", PostgresAdapter)

    def build(source: str) -> Any:
        return registry.create(
            SourceConfig(
                name=source,
                adapter="postgresql",
                secret=f"pg_{source}_ro",
                options={
                    "host": pg.host,
                    "port": pg.port,
                    "dbname": source,
                    "user": pg.ro_user,
                },
            )
        )

    sources = {source: build(source) for source in ("crm", "billing", "erp")}
    try:
        yield sources
    finally:
        for adapter in sources.values():
            adapter.close()


@pytest.fixture(scope="module")
def services(
    adapters: Mapping[str, Any], pg: PgTarget, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[QueryServices]:
    """The pipeline the server would build: layer, governor, cache, history and audit."""
    from mdqm.semantic.loader import load as load_layer

    layer = load_layer(REPO_ROOT / "examples" / "semantic_example.yaml")
    columns: dict[str, tuple[str, ...]] = {}
    for source, adapter in adapters.items():
        snapshot = adapter.extract_catalog()
        for table in snapshot.tables:
            columns[f"{source}.{table.name}"] = tuple(column.name for column in table.columns)

    path = tmp_path_factory.mktemp("m03-e2e")
    store = Store(path / "meta.sqlite")
    store.initialize()
    con = duckdb.connect(":memory:")
    con.execute("SET TimeZone='UTC'").fetchall()
    clock = FixedClock()
    built = QueryServices(
        layer=layer,
        governor=Governor(dict(adapters), clock=clock),
        history=HistoryStore(store, clock=clock),
        audit=AuditLog(store, clock=clock),
        adapters=dict(adapters),
        columns=columns,
        cache=FragmentCache(con, clock=clock),
        clock=clock,
        limits=LIMITS,
    )
    reset_query_services()
    configure_query_services(built)
    try:
        yield built
    finally:
        reset_query_services()
        con.close()
        store.close()


@pytest.fixture
def as_alice() -> Iterator[None]:
    """Bind the authenticated caller the tool reads, the way the auth middleware does."""
    from mcp.server.auth.middleware.auth_context import auth_context_var
    from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
    from mcp.server.auth.provider import AccessToken

    from mdqm.server.auth import READ_SCOPE

    token = AccessToken(
        token="throwaway-not-a-secret",
        client_id="mdqm:alice",
        scopes=[READ_SCOPE],
        subject="alice",
        claims={"display_name": "Alice"},
    )
    reset = auth_context_var.set(AuthenticatedUser(token))
    try:
        yield
    finally:
        auth_context_var.reset(reset)


def question(question_id: str) -> Mapping[str, Any]:
    """One question from the E01 document."""
    document = cast(
        Mapping[str, Any],
        YAML(typ="safe").load(QUESTIONS.read_text(encoding="utf-8")),
    )
    by_id: Mapping[str, Any] = {
        str(entry["id"]): entry for entry in cast(list[Mapping[str, Any]], document["questions"])
    }
    return by_id[question_id]


def statements_on_usage_events(pg: PgTarget) -> int:
    """How many statements the server has executed against ``usage_events``, counted by itself.

    ``pg_stat_statements`` aggregates the statements the shared library saw, so it counts a query
    that really ran and nothing else. ``pg_stat_activity`` was the obvious witness and the wrong
    one: the adapter streams through a named cursor, so the last statement a backend reports is
    ``FETCH FORWARD ...``, which names no table. The bound parameter keeps this probe's own text
    (``$1``) out of the match, so it cannot count itself.
    """
    with psycopg.connect(pg.dsn("billing"), connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT coalesce(sum(calls), 0) FROM pg_stat_statements WHERE query ILIKE %s",
            ("%usage_events%",),
        )
        row = cur.fetchone()
    return int(0 if row is None else row[0])


def ask(spec: Mapping[str, Any]) -> Any:
    """Call the tool the way the SDK does: arguments validated into the wire model first."""
    from mdqm.spec.models import QuerySpec

    return query(QuerySpec.model_validate(spec))


# --- claim (a): the tool's answer is the document's answer ----------------------------------


def test_the_billing_question_through_the_tool_matches_the_e01_ground_truth(
    services: QueryServices, as_alice: None
) -> None:
    """E01's cross-source average, produced by the whole pipeline against live PostgreSQL."""
    entry = question(BILLING_QUESTION)
    document = cast(
        Mapping[str, Any],
        YAML(typ="safe").load(QUESTIONS.read_text(encoding="utf-8")),
    )
    tolerance = float(document["float_tolerance"])

    answer = ask(cast(Mapping[str, Any], entry["spec"]))

    assert answer.status == "ok", answer.error
    assert answer.error is None and answer.summary is not None
    assert answer.result_handle and answer.result_handle == answer.summary.handle
    columns = list(cast(list[str], entry["answer"]["columns"]))
    produced = {
        str(row[columns[0]]): float(row[columns[1]])
        for row in answer.sample
        if row[columns[0]] is not None
    }
    expected = {
        str(row[0]): float(row[1]) for row in cast(list[list[Any]], entry["answer"]["rows"])
    }
    assert set(produced) == set(expected), produced
    for status, wanted in expected.items():
        assert produced[status] == pytest.approx(wanted, rel=tolerance), status
    # The plan is the server's, and it says whether the scan prunes: ``accounts`` declares no
    # partitioning, so the honest verdict is "unknown" rather than a claim.
    assert answer.plan is not None
    assert answer.plan.pruning_proof.value in {"unknown", "proven", "unproven"}


def test_the_run_is_in_history_and_the_audit_chain_verifies(
    services: QueryServices, as_alice: None
) -> None:
    """The tool's own row: who asked, with which spec, on which source, at which exec_ts."""
    answer = ask(cast(Mapping[str, Any], question(BILLING_QUESTION)["spec"]))
    assert answer.status == "ok", answer.error

    runs = services.history.recent(user_id="alice")
    assert runs, "the tool wrote no history row"
    run = runs[0]
    assert run.user_id == "alice" and run.tool == "query"
    assert run.exec_ts == EXEC_TS
    assert run.spec is not None and run.spec.from_table == "billing.accounts"
    assert [fragment.source for fragment in run.fragments] == ["billing"]
    assert run.error is None
    assert run.rerunnable, "a stored run must be replayable, and this one redacted something"
    verification = services.audit.verify_chain()
    assert verification.ok, verification.describe()


# --- claim (b): an unbounded scan never reaches the database --------------------------------


def test_an_unbounded_partitioned_query_is_refused_and_never_reaches_the_server(
    services: QueryServices, pg: PgTarget, as_alice: None
) -> None:
    """The refusal is measured on the server, and a control proves the measurement works."""
    before = statements_on_usage_events(pg)

    refused = ask(UNBOUNDED_SPEC)

    assert refused.status == "error"
    assert refused.error is not None
    assert refused.error.code == ErrorCode.PARTITION_SCAN_ALL.value
    assert refused.error.fix
    assert statements_on_usage_events(pg) == before, (
        "the refused query reached the server: a partition scan the planner refused must never be "
        "sent"
    )
    # It is a refusal, so it is a row: the audit chain says which code the agent was given.
    blocked = [
        row for row in services.audit.rows() if row.error_code is ErrorCode.PARTITION_SCAN_ALL
    ]
    assert blocked, "the refusal was not audited"
    assert blocked[-1].outcome.value == "blocked"

    # The positive control: a bounded window over the same table really does reach the server, so
    # the measurement above is not blind to a query that runs.
    bounded = ask(BOUNDED_SPEC)
    assert bounded.status == "ok", bounded.error
    assert statements_on_usage_events(pg) > before, (
        "a query that really ran is not visible in pg_stat_statements: the check above proves "
        "nothing"
    )
