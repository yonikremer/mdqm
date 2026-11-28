"""E03: an MCP client, three live PostgreSQL databases, answers checked against the data.

The client is the MCP SDK's own streamable-HTTP client. The server is the product's
``MCPServer``, wired the way ``mdqm serve`` wires it (governor, fragment cache, shared federated
executor, history, audit). The three databases are the seeded organisation (crm, billing, erp).
Nothing in the assertions is a number copied out of the tool's own report: each expected value is
computed here from the rows PostgreSQL returns, with the rounding spelled out in this file.

What one cross-source credit-limit question can and cannot prove:

* the spec that names both approved identity joins starts at ``crm.customers``, so the measure is
  crm's column (already the canonical rounded limit) and billing/erp are semi-joins -- membership
  without letting erp's duplicate rows inflate the average;
* the billing ``ROUND({col}/5000)*5000`` and erp ``{col}*10`` conversions are therefore proved by
  asking those storage forms through the same client, and by stage D8 rediscovering both formulas
  from the live rows.

Run it with the ``pg`` profile up and the world loaded (the fixture loads the world itself)::

    python -m pytest tests/integration/test_e2e.py --require-services -q
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import duckdb
import httpx2
import psycopg
import pytest
import uvicorn
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from ruamel.yaml import YAML

from mdqm.adapters.postgresql import PostgresAdapter
from mdqm.adapters.registry import AdapterRegistry
from mdqm.config.settings import SourceConfig
from mdqm.executor.cache import FragmentCache
from mdqm.executor.federate import FederatedExecutor
from mdqm.governor import Governor
from mdqm.models import Fragment, Limits
from mdqm.models.partitioning import PartitionGranularity
from mdqm.pipeline.core import PipelineContext
from mdqm.pipeline.stages.catalog import CatalogStage
from mdqm.pipeline.stages.conversions import ColumnPair, ColumnRef, ConversionStage
from mdqm.pipeline.stages.profiling import ProfilingStage
from mdqm.semantic.loader import load as load_layer
from mdqm.semantic.models import (
    ApprovedJoin,
    JoinCardinality,
    SemanticLayer,
    compute_version_hash,
)
from mdqm.server.app import create_server
from mdqm.server.auth import READ_SCOPE, StaticTokenVerifier, TokenEntry
from mdqm.server.tools.misc import ToolServices
from mdqm.server.tools.query import QueryServices, configure_query_services, reset_query_services
from mdqm.sql.execution import StatementRefused
from mdqm.store.audit import AuditLog
from mdqm.store.db import Store
from mdqm.store.feedback import FeedbackStore
from mdqm.store.history import HistoryStore
from mdqm.testing.loaders.postgres import PgTarget, load, target_from_environ
from mdqm.testing.world import WorldConfig

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = REPO_ROOT / "tests" / "eval" / "questions.yaml"
EXAMPLE = REPO_ROOT / "examples" / "semantic_example.yaml"
EXEC_TS = datetime(2026, 1, 1, tzinfo=UTC)
LIMITS = Limits(timeout_s=60.0, max_rows=100_000, max_bytes=64 << 20, max_partitions=32)
TOKEN = "tok-alice-9f2c"
ROUNDING_STEP = 5000
STATUS = {"A": "active", "C": "churned", "S": "suspended"}
BILLING_STATE = {"Active": "active", "Closed": "churned", "Hold": "suspended"}
ERP_CODE = {1: "active", 2: "churned", 3: "suspended"}

_YAML = YAML(typ="safe")


class _PasswordSecrets:
    """The read-only password, resolved by name. The value never appears in a log line here."""

    def __init__(self, target: PgTarget) -> None:
        self._target = target

    def get(self, name: str) -> str:
        if not name:
            raise AssertionError("a secret reference is a name")
        return self._target.ro_password


def _unavailable(request: pytest.FixtureRequest, reason: str) -> None:
    if request.config.getoption("--require-services"):
        pytest.fail(f"required service unavailable: {reason}")
    pytest.skip(f"service unavailable: {reason}")


def _questions() -> dict[str, dict[str, Any]]:
    document = _YAML.load(QUESTIONS.read_text(encoding="utf-8"))
    return {str(entry["id"]): dict(entry) for entry in document["questions"]}


def _round_half_up(value: int, step: int = ROUNDING_STEP) -> int:
    """Half up, for a non-negative integer. Ties go away from zero, which is up."""
    if value < 0:
        raise AssertionError(f"the synthetic limits are non-negative, got {value}")
    return ((value + step // 2) // step) * step


def _average(values: list[int]) -> float:
    if not values:
        raise AssertionError("an average over no rows is not an answer")
    return sum(values) / len(values)


def _published(question_id: str) -> dict[str, float]:
    """E01's stored rows, used only to confirm the loaded world is the one that document describes."""
    answer = _questions()[question_id]["answer"]
    rows = answer["rows"]
    key_at = answer["columns"].index(
        "customer_status" if "customer_status" in answer["columns"] else answer["columns"][0]
    )
    value_at = 1
    return {str(row[key_at]): float(row[value_at]) for row in rows}


def _close(produced: Mapping[str, float], expected: Mapping[str, float]) -> None:
    assert set(produced) == set(expected), produced
    for status, wanted in expected.items():
        assert produced[status] == pytest.approx(wanted, rel=1e-9), status


# --- the live world ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg(request: pytest.FixtureRequest) -> Iterator[PgTarget]:
    try:
        target = target_from_environ()
    except Exception as exc:
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
    """The default world. E01's answers describe 2000 customers, not the small one."""
    load(pg, WorldConfig())


def _adapter(pg: PgTarget, source: str) -> PostgresAdapter:
    registry = AdapterRegistry(secrets=_PasswordSecrets(pg), entry_points=lambda: ())
    registry.register("postgresql", PostgresAdapter)
    created = registry.create(
        SourceConfig(
            name=source,
            adapter="postgresql",
            secret=f"{source}_ro",
            options={"host": pg.host, "port": pg.port, "dbname": source, "user": pg.ro_user},
        )
    )
    assert isinstance(created, PostgresAdapter)
    return created


def _layer_with_identity_joins() -> SemanticLayer:
    """The shipped example plus the two identity joins the credit question needs.

    ``examples/semantic_example.yaml`` approves ``usage_to_customers`` only. The credit-limit
    question also needs billing and erp joined to crm; those joins are the same key conversions
    the unit suite already uses, added here on a copy so the example file stays what it is.
    """
    layer = load_layer(EXAMPLE)
    extra = (
        ApprovedJoin(
            name="crm_accounts",
            left="billing.accounts.client_no",
            right="crm.customers.cust_id",
            on="LPAD(CAST({right} AS VARCHAR), 10, '0') = {left}",
            cardinality=JoinCardinality.MANY_TO_ONE,
        ),
        ApprovedJoin(
            name="crm_erp",
            left="crm.customers.cust_id",
            right="erp.customers.cust_no",
            on="{left} = {right}",
            cardinality=JoinCardinality.ONE_TO_MANY,
        ),
    )
    updated = layer.model_copy(
        update={"approved_joins": layer.approved_joins + extra, "version_hash": "recompute"}
    )
    return updated.model_copy(update={"version_hash": compute_version_hash(updated)})


@pytest.fixture(scope="module")
def services(
    pg: PgTarget, seeded: None, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[QueryServices, Mapping[str, PostgresAdapter]]]:
    path = tmp_path_factory.mktemp("e03")
    adapters = {source: _adapter(pg, source) for source in ("crm", "billing", "erp")}
    layer = _layer_with_identity_joins()
    columns: dict[str, tuple[str, ...]] = {}
    for source, adapter in adapters.items():
        for table in adapter.extract_catalog().tables:
            columns[f"{source}.{table.name}"] = tuple(column.name for column in table.columns)
    store = Store(path / "meta.sqlite")
    store.initialize()
    governor = Governor(dict(adapters))
    connection = duckdb.connect(str(path / "fragments.duckdb"))
    connection.execute("SET enable_external_access=false")
    executor = FederatedExecutor(governor, layer=layer, spill_dir=path / "spill")
    built = QueryServices(
        layer=layer,
        governor=governor,
        history=HistoryStore(store),
        audit=AuditLog(store),
        adapters=dict(adapters),
        columns=columns,
        cache=FragmentCache(connection),
        limits=LIMITS,
        executor=executor,
        spill_dir=path / "spill",
    )
    configure_query_services(built)
    from mdqm.server.tools import configure_services

    configure_services(
        ToolServices(
            handles=governor.store,
            history=built.history,
            feedback=FeedbackStore(store),
            audit=built.audit,
        )
    )
    try:
        yield built, adapters
    finally:
        reset_query_services()
        from mdqm.server.tools import reset_services

        reset_services()
        executor.close()
        connection.close()
        for adapter in adapters.values():
            adapter.close()


@pytest.fixture(scope="module")
def endpoint(services: tuple[QueryServices, Mapping[str, PostgresAdapter]]) -> Iterator[str]:
    """The real server on an ephemeral loopback port. ``services`` is required so it is wired first."""
    del services
    server = create_server(
        verifier=StaticTokenVerifier(
            (TokenEntry(token=TOKEN, user_id="alice", display_name="Alice", scopes=(READ_SCOPE,)),)
        )
    )
    config = uvicorn.Config(
        server.streamable_http_app(), host="127.0.0.1", port=0, log_level="warning"
    )
    running = uvicorn.Server(config)
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while not running.started:
            if time.monotonic() > deadline:
                raise RuntimeError("the test server did not start within 30s")
            time.sleep(0.01)
        port = running.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        running.should_exit = True
        thread.join(timeout=10)


def _client(endpoint: str) -> Any:
    """The SDK client, opened inside the test.

    A pytest async fixture tears this transport down in a different task than the one that opened
    it, and anyio then refuses to exit the cancel scope. The unit suite opens the client in the
    test body for the same reason.
    """

    @asynccontextmanager
    async def opened() -> AsyncIterator[Client]:
        http_client = httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {TOKEN}"}, timeout=120.0
        )
        try:
            async with Client(streamable_http_client(endpoint, http_client=http_client)) as mcp:
                yield mcp
        finally:
            await http_client.aclose()

    return opened()


async def _ask(client: Client, arguments: dict[str, Any]) -> dict[str, Any]:
    result = await client.call_tool("query", arguments)
    body = result.structured_content
    assert isinstance(body, dict), result
    return body


def _rows_by_status(body: Mapping[str, Any], status_key: str, value_key: str) -> dict[str, float]:
    assert body["status"] == "ok", body.get("error")
    sample = body.get("sample") or []
    assert sample, body
    return {str(row[status_key]): float(row[value_key]) for row in sample}


# --- ground truth, computed from the databases and not from the tool ------------------------


def _crm_average(pg: PgTarget) -> dict[str, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    with psycopg.connect(pg.dsn("crm")) as conn, conn.cursor() as cur:
        cur.execute("SELECT status, credit_limit FROM customers")
        for status, limit in cur.fetchall():
            grouped[STATUS[str(status)]].append(int(limit))
    return {status: _average(values) for status, values in grouped.items()}


def _billing_average(pg: PgTarget) -> dict[str, float]:
    """Raw limit, rounded half-up to 5000 before the average. NULL limits are not zeros."""
    grouped: dict[str, list[int]] = defaultdict(list)
    with psycopg.connect(pg.dsn("billing")) as conn, conn.cursor() as cur:
        cur.execute("SELECT state, credit_limit FROM accounts WHERE credit_limit IS NOT NULL")
        for state, limit in cur.fetchall():
            grouped[BILLING_STATE[str(state)]].append(_round_half_up(int(limit)))
    return {status: _average(values) for status, values in grouped.items()}


def _erp_average(pg: PgTarget) -> dict[str, float]:
    """``limit_k`` is the rounded limit divided by 10. Identical duplicate rows are counted once."""
    grouped: dict[str, list[int]] = defaultdict(list)
    with psycopg.connect(pg.dsn("erp")) as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT cust_no, status_code, limit_k FROM customers")
        for _cust_no, status_code, limit_k in cur.fetchall():
            grouped[ERP_CODE[int(status_code)]].append(int(limit_k) * 10)
    return {status: _average(values) for status, values in grouped.items()}


def _scalar(pg: PgTarget, database: str, statement: str) -> int:
    with psycopg.connect(pg.dsn(database)) as conn, conn.cursor() as cur:
        cur.execute(statement)
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


# --- the five claims, through the client ---------------------------------------------------


async def test_three_database_credit_limit_matches_independent_ground_truth(
    endpoint: str, pg: PgTarget
) -> None:
    """One spec joins crm, billing and erp. The average is the canonical (crm) figure.

    FAILS IF: a source is missing from the receipt, a semi-join multiplies rows, or the average
    disagrees with the average computed from crm's own rows.
    """
    independent = _crm_average(pg)
    _close(independent, _published("avg_limit_by_status_crm"))
    async with _client(endpoint) as client:
        body = await _ask(
            client,
            {
                "spec": {
                    "from": "crm.customers",
                    "select": [{"measure": "credit_limit_r5000", "agg": "avg"}],
                    "joins": ["crm_accounts", "crm_erp"],
                    "group_by": ["status"],
                    "order_by": [{"by": "status", "direction": "asc"}],
                    "limit": 50,
                }
            },
        )
    produced = _rows_by_status(body, "status", "credit_limit_r5000_avg")
    _close(produced, independent)
    quality = body.get("quality") or {}
    steps = quality.get("steps") or []
    assert {step["source"] for step in steps} == {"crm", "billing", "erp"}
    assert all(step["rows"] > 0 for step in steps)
    by_join = {join["join"]: join for join in quality.get("joins") or []}
    assert set(by_join) == {"crm_accounts", "crm_erp"}
    for join in by_join.values():
        assert join["how"] == "semi"
        assert join["multiplied"] == 0
        assert join["matched"] + join["dropped"] == join["rows_left"]
    fixes = " ".join(body.get("applied_fixes") or [])
    assert "crm_accounts" in fixes and "crm_erp" in fixes


async def test_each_storage_form_applies_its_own_rounding(endpoint: str, pg: PgTarget) -> None:
    """Billing rounds half-up to 5000; erp multiplies the already-rounded-then-divided value by 10.

    crm stores the rounded limit, so its normalisation is the identity. Each answer is compared to
    the same arithmetic run in this process over the rows the database returned.
    """
    cases = (
        ("avg_limit_by_status_billing", _billing_average(pg), "customer_status"),
        ("avg_limit_by_status_erp", _erp_average(pg), "customer_status"),
        ("avg_limit_by_status_crm", _crm_average(pg), "customer_status"),
    )
    async with _client(endpoint) as client:
        for question_id, independent, status_key in cases:
            _close(independent, _published(question_id))
            body = await _ask(client, {"spec": dict(_questions()[question_id]["spec"])})
            produced = _rows_by_status(body, status_key, "credit_limit_r5000_avg")
            _close(produced, independent)


async def test_new_since_keeps_late_rows_and_adds_the_derived_event_bound(
    endpoint: str, pg: PgTarget
) -> None:
    """'What is new since 1 Dec' counts by ingestion time, and the scan is also bounded on event time.

    The derived bound is ``since - max_lag`` (6h in the example layer): it can only drop partitions,
    never a row whose ingestion time qualifies. The history re-run is frozen at the original
    ``exec_ts``, so asking again does not move the window.
    """
    spec = dict(_questions()["new_since_resync"]["spec"])
    async with _client(endpoint) as client:
        body = await _ask(client, {"spec": spec})
        assert body["status"] == "ok", body.get("error")
        counted = int(body["sample"][0]["usage_units_count"])
        listed = await client.call_tool("history", {"limit": 5})
        runs = (listed.structured_content or {}).get("runs") or []
        assert runs, listed.structured_content
        rerun = await client.call_tool(
            "history", {"rerun_id": runs[0]["id"], "rerun_mode": "frozen"}
        )
    plan = (rerun.structured_content or {}).get("rerun") or {}
    direct = _scalar(
        pg,
        "billing",
        "SELECT COUNT(units) FROM usage_events "
        "WHERE insertion_time >= TIMESTAMPTZ '2025-12-01 00:00:00+00'",
    )
    assert counted == direct
    assert counted == int(_questions()["new_since_resync"]["answer"]["rows"][0][0])
    assert plan.get("mode") == "frozen"
    assert plan.get("exec_ts") == runs[0]["exec_ts"]
    sql = " ".join(str(fragment.get("sql") or "") for fragment in plan.get("fragments") or [])
    assert "insertion_time" in sql
    assert "start_time" in sql
    assert "2025-11-30" in sql, sql  # 2025-12-01 minus the approved 6h lag


async def test_a_query_without_a_time_bound_is_refused_with_a_fix(endpoint: str) -> None:
    question = _questions()["usage_events_all_partitions_refused"]
    async with _client(endpoint) as client:
        body = await _ask(client, {"spec": dict(question["spec"])})
    assert body["status"] == "error", body
    assert body["error"]["code"] == question["answer"]["code"]
    assert body["error"]["fix"]


async def test_a_repeated_query_is_served_from_cache_with_fetched_at(endpoint: str) -> None:
    """``billing.usage_events`` is ``append_rarely``, so the second identical call is a cache hit."""
    spec = dict(_questions()["events_count_window"]["spec"])
    async with _client(endpoint) as client:
        first = await _ask(client, {"spec": spec})
        assert first["status"] == "ok", first.get("error")
        second = await _ask(client, {"spec": spec})
    assert second["status"] == "ok", second.get("error")
    assert second["cache_hit"] is True, second
    assert second["sample"] == first["sample"]
    freshness = second.get("freshness") or []
    assert freshness and all(entry.get("fetched_at") for entry in freshness)


async def test_delete_is_blocked_by_the_guard_and_by_the_database_when_the_guard_is_bypassed(
    endpoint: str,
    pg: PgTarget,
    services: tuple[QueryServices, Mapping[str, PostgresAdapter]],
) -> None:
    """Two different layers, both required.

    The tool refuses ``DELETE`` before a statement is sent (the guard). A cursor that never goes
    through the guard -- a plain ``psycopg`` connection as ``mdqm_ro`` -- is still refused by
    PostgreSQL: ``SELECT nextval`` is allowed by privilege and writes, so only the read-only
    transaction (SQLSTATE 25006) can refuse it; ``DELETE`` after forcing the transaction read-write
    is refused by the missing privilege (42501).
    """
    _built, adapters = services
    before = _scalar(
        pg,
        "billing",
        "SELECT coalesce(sum(calls), 0) FROM pg_stat_statements WHERE upper(query) LIKE '%DELETE%'",
    )
    async with _client(endpoint) as client:
        body = await _ask(client, {"sql": "DELETE FROM public.accounts", "source": "billing"})
    assert body["status"] == "error", body
    assert body["error"]["code"] == "WRITE_BLOCKED"
    assert body["error"]["fix"]
    after = _scalar(
        pg,
        "billing",
        "SELECT coalesce(sum(calls), 0) FROM pg_stat_statements WHERE upper(query) LIKE '%DELETE%'",
    )
    assert after == before, "the guarded DELETE reached PostgreSQL"

    billing = adapters["billing"]
    with pytest.raises(StatementRefused):
        stream = billing.execute(
            Fragment(
                source="billing",
                dialect="postgres",
                sql="DELETE FROM public.accounts WHERE false",
                exec_ts=EXEC_TS,
            ),
            LIMITS,
        )
        stream.close()
    session = billing.execute(
        Fragment(
            source="billing",
            dialect="postgres",
            sql=(
                "SELECT current_setting('transaction_read_only') AS ro, "
                "current_setting('statement_timeout') AS timeout_setting"
            ),
            exec_ts=EXEC_TS,
        ),
        LIMITS,
    )
    try:
        batch = next(iter(session))
        row = {name: batch.column(name).to_pylist()[0] for name in batch.schema.names}
    finally:
        session.close()
    assert row["ro"] == "on"
    assert row["timeout_setting"] not in {"0", "0ms"}

    with psycopg.connect(pg.dsn("billing")) as admin, admin.cursor() as cur:
        cur.execute("DROP SEQUENCE IF EXISTS public.mdqm_e2e_write_probe")
        cur.execute("CREATE SEQUENCE public.mdqm_e2e_write_probe")
        cur.execute("GRANT USAGE, SELECT ON SEQUENCE public.mdqm_e2e_write_probe TO mdqm_ro")
    try:
        with psycopg.connect(pg.dsn("billing", read_only=True)) as conn, conn.cursor() as cur:
            cur.execute("SHOW transaction_read_only")
            assert cur.fetchone() == ("on",)
            with pytest.raises(psycopg.errors.ReadOnlySqlTransaction) as excinfo:
                cur.execute("SELECT nextval('public.mdqm_e2e_write_probe')")
            assert excinfo.value.sqlstate == "25006"
        # A second session, so the aborted nextval transaction is not the one that issues DELETE.
        with psycopg.connect(pg.dsn("billing", read_only=True)) as conn, conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = off")
            conn.commit()
            with pytest.raises(psycopg.errors.InsufficientPrivilege) as denied:
                cur.execute("DELETE FROM public.accounts WHERE false")
            assert denied.value.sqlstate == "42501"
    finally:
        with psycopg.connect(pg.dsn("billing")) as admin, admin.cursor() as cur:
            cur.execute("DROP SEQUENCE IF EXISTS public.mdqm_e2e_write_probe")


async def test_the_audit_chain_verifies_after_the_run(
    endpoint: str, services: tuple[QueryServices, Mapping[str, PostgresAdapter]]
) -> None:
    built, _adapters = services
    async with _client(endpoint) as client:
        body = await _ask(client, {"spec": dict(_questions()["avg_limit_by_status_crm"]["spec"])})
    assert body["status"] == "ok", body.get("error")
    verdict = built.audit.verify_chain()
    assert verdict.ok, verdict.describe()
    assert verdict.checked >= 1


def test_bootstrap_rediscovers_conversions_and_monthly_granularity(
    pg: PgTarget,
    seeded: None,
    services: tuple[QueryServices, Mapping[str, PostgresAdapter]],
    tmp_path: Path,
) -> None:
    """D1+D2, D5 and D8 over the live databases.

    The catalog YAML loads, ``usage_events`` is monthly on ``start_time`` (the example's
    granularity), and D8's best template for each pair is the formula the example already states.
    """
    del pg, seeded
    _built, adapters = services
    root = tmp_path / "semantic"
    root.mkdir()
    store = Store(tmp_path / "pipeline.sqlite")
    store.initialize()
    ctx = PipelineContext(store=store, exec_ts=EXEC_TS, semantic_root=root, force=True)

    catalog = CatalogStage(adapters)
    catalog.inputs_hash(ctx)
    assert catalog.run(ctx).status.value == "ran"
    profile = ProfilingStage(adapters, tables=("customers", "accounts"))
    assert profile.run(ctx).status.value == "ran"
    ctx.commit()

    loaded = load_layer(root)
    usage = loaded.tables["billing.usage_events"]
    assert usage.partitioning is not None
    assert usage.partitioning.granularity is PartitionGranularity.MONTH
    assert usage.partitioning.key == ("start_time",)
    assert len(usage.partitioning.bounds) >= 12

    crm = ColumnRef(source="crm", table="customers", column="credit_limit", key="cust_id")
    stage = ConversionStage(
        adapters,
        (
            ColumnPair(
                reference=crm,
                other=ColumnRef(
                    source="billing", table="accounts", column="credit_limit", key="client_no"
                ),
            ),
            ColumnPair(
                reference=crm,
                other=ColumnRef(source="erp", table="customers", column="limit_k", key="cust_no"),
            ),
        ),
        measure="credit_limit_r5000",
    )
    example = load_layer(EXAMPLE).measures["credit_limit_r5000"]
    detected: dict[str, str] = {}
    for pair in stage.pairs:
        found = stage.detect_pair(pair, ctx)
        assert found is not None and found.best is not None, pair.other.label
        assert found.best.match_rate >= 0.95, (pair.other.label, found.best.describe())
        detected[pair.other.label] = found.best.template
        assert found.best.template == example.sources[pair.other.label].normalize

    proposal = root / "curated"
    proposal.mkdir()
    document = {
        "version": 1,
        "measures": {
            "credit_limit_r5000": {
                "sources": {
                    "crm.customers.credit_limit": {"normalize": "{col}"},
                    "billing.accounts.credit_limit": {
                        "normalize": detected["billing.accounts.credit_limit"]
                    },
                    "erp.customers.limit_k": {"normalize": detected["erp.customers.limit_k"]},
                }
            }
        },
    }
    buffer = StringIO()
    YAML().dump(document, buffer)
    (proposal / "conversions.yaml").write_text(buffer.getvalue(), encoding="utf-8")
    # A directory load merges generated + curated. The conversions file must be accepted, and the
    # curated formulas are the ones D8 just proposed.
    merged = load_layer(root)
    sources = merged.measures["credit_limit_r5000"].sources
    assert sources["billing.accounts.credit_limit"].normalize == "ROUND({col} / 5000) * 5000"
    assert sources["erp.customers.limit_k"].normalize == "{col} * 10"
