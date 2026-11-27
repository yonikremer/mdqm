"""The real-database twin of ``test_claims_e2e.py``: the same claims against live PostgreSQL.

``test_claims_e2e.py`` proves the claims with in-process DuckDB sources, which is the only way to
run them on a machine without a container runtime. This module proves the parts that only a real
server can decide:

* **the conversions are still right when the data is real** -- the ``billing`` question answered by
  SQL PostgreSQL actually ran, compared with an independent half-up derivation run by the server
  itself and with the answer task E01 derived offline;
* **the catalog and the partitions are discovered, not assumed** -- ``usage_events`` comes back as a
  partitioned table with its monthly set;
* **the right partition key prunes** -- an event-time bound touches one or two partitions while the
  identical bound on the ingestion time cannot prune at all;
* **read-only is the server's own refusal** -- ``mdqm_ro`` cannot write, in both layers, and the
  SQLSTATE is PostgreSQL's;
* **a timeout really cancels** -- a ``pg_sleep`` past its deadline is cancelled on the server, not
  left running in the background.

The PostgreSQL adapter is task A02 and is being written in a sibling lane. Until
``mdqm.adapters.postgresql`` exports a class this module can hand to the registry, every test that
needs it skips with that reason; the ``psycopg``-only tests (pruning, read-only) run as soon as the
compose stack is up. ``--require-services`` turns a *missing service* into a failure; a missing
adapter stays a skip because no environment can supply it yet.

Run it with::

    python scripts/dev.py up pg
    python -m pytest tests/integration/test_postgres_e2e.py --require-services -q
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import psycopg
import pytest
from psycopg import sql

from mdqm.models import Fragment, Limits
from mdqm.testing.loaders.postgres import PgTarget, target_from_environ
from mdqm.testing.world import ANCHOR, WorldConfig

#: The world window's last 30 days: one event-time partition, every ingestion-time partition.
WINDOW_START: datetime = ANCHOR - timedelta(days=30)

#: 25006 (read-only transaction) and 42501 (missing privilege): the two read-only layers.
READ_ONLY_SQLSTATES = frozenset(
    {
        psycopg.errors.ReadOnlySqlTransaction.sqlstate,
        psycopg.errors.InsufficientPrivilege.sqlstate,
    }
)

#: The question checked against real data: billing stores the limit raw.
BILLING_QUESTION = "avg_limit_by_status_billing"

#: Storage-form code -> canonical value, spelled out here on purpose (the fixture must not read the
#: value map from the code under test).
BILLING_STATE = {"Active": "active", "Closed": "churned", "Hold": "suspended"}

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    """Load the **default** world before the claims are checked.

    E01's answers describe the default world (2000 customers); the D03 tests use the small one for
    speed. A test that asserts against E01 while the database holds the small world compares two
    different datasets and fails for a reason that has nothing to do with the code -- which is
    exactly what happened before this fixture existed.
    """
    from mdqm.testing.loaders.postgres import load

    load(pg, WorldConfig())


# --- building the adapter, however task A02 spells it --------------------------------------


class _PasswordSecrets:
    """A secret provider that resolves a secret name to the read-only password.

    A02's contract is explicit: ``options`` carries host/port/dbname/user and the secret is the
    **password** only. An earlier version of this test handed the whole DSN over as the secret,
    which the adapter then sent as a password -- the pool could never connect.
    """

    def __init__(self, target: PgTarget) -> None:
        self._target = target

    def get(self, name: str) -> str:
        return self._target.ro_password


def postgres_adapter(target: PgTarget, source: str) -> Any:
    """Build the A02 adapter for one source, or skip with a precise reason when it is absent."""
    module = pytest.importorskip(
        "mdqm.adapters.postgresql",
        reason="task A02 (the PostgreSQL adapter) has not landed in this branch yet",
    )
    factory = getattr(module, "PostgresAdapter", None)
    if factory is None:
        pytest.skip(
            "mdqm.adapters.postgresql exists but exports no PostgresAdapter; "
            "point this helper at the class task A02 shipped"
        )
    from mdqm.adapters.registry import AdapterRegistry
    from mdqm.config.settings import SourceConfig

    registry = AdapterRegistry(secrets=_PasswordSecrets(target), entry_points=lambda: ())
    registry.register("postgresql", factory)
    return registry.create(
        SourceConfig(
            name=source,
            adapter="postgresql",
            secret=source + "_ro",
            # A02 refuses a source with no database user: the server needs a per-person
            # identity, and the secret provider resolves it. Tests use the read-only role.
            options={
                "host": target.host,
                "port": target.port,
                "dbname": source,
                "user": target.ro_user,
            },
        )
    )


def _drained(stream: Any) -> list[dict[str, Any]]:
    """Drain an adapter stream into plain rows."""
    batches = list(stream)
    if not batches:
        return []
    import pyarrow as pa

    return pa.Table.from_batches(batches).to_pylist()


def _limits(timeout_s: float = 30.0) -> Limits:
    return Limits(timeout_s=timeout_s, max_rows=100_000, max_bytes=64 << 20, max_partitions=64)


# --- claim (a): the conversions, on data PostgreSQL really holds ---------------------------


def test_real_billing_average_matches_two_independent_derivations(
    pg: PgTarget, seeded: None
) -> None:
    """Claim (a) against a real server: the raw limit rounded half-up to 5000, then averaged.

    FAILS IF: the SQL the compiler emitted produces a number other than (1) the half-up arithmetic
    the server computes when asked directly and (2) E01's offline answer -- which would mean the
    conversion is right only on DuckDB.
    """
    from ruamel.yaml import YAML

    from mdqm.compiler import CompileContext, compile_spec
    from mdqm.semantic.loader import load
    from mdqm.spec import parse_spec

    document = cast(
        dict[str, Any],
        YAML(typ="safe").load(
            (REPO_ROOT / "tests" / "eval" / "questions.yaml").read_text(encoding="utf-8")
        ),
    )
    by_id = {question["id"]: question for question in document["questions"]}
    spec = by_id[BILLING_QUESTION]["spec"]

    adapter = postgres_adapter(pg, "billing")
    try:
        snapshot = adapter.extract_catalog()
        columns = {
            f"billing.{table.name}": tuple(column.name for column in table.columns)
            for table in snapshot.tables
        }
        layer = load(REPO_ROOT / "examples" / "semantic_example.yaml")
        compiled = compile_spec(
            parse_spec(spec),
            CompileContext(
                layer=layer,
                source="billing",
                dialect="postgres",
                exec_ts=datetime(2026, 11, 25, 15, 0, tzinfo=ANCHOR.tzinfo),
                columns=columns,
                max_rows=10_000,
            ),
        )
        assert compiled.ok, compiled.error
        assert compiled.fragment is not None
        got = _drained(adapter.execute(compiled.fragment, _limits()))
    finally:
        adapter.close()

    # (1) the same arithmetic, written by hand, executed by the server.
    with psycopg.connect(pg.dsn("billing")) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT state, AVG(floor((credit_limit + 2500) / 5000) * 5000) FROM public.accounts "
            "WHERE credit_limit IS NOT NULL GROUP BY 1"
        )
        oracle = {
            BILLING_STATE[str(row[0])]: float(row[1])
            for row in cur.fetchall()
            if str(row[0]) in BILLING_STATE
        }

    # (2) E01's offline answer.
    expected = {
        str(row[0]): float(row[1])
        for row in cast(list[list[Any]], by_id[BILLING_QUESTION]["answer"]["rows"])
    }
    tolerance = float(document["float_tolerance"])
    assert set(oracle) == set(expected)
    produced = {str(row["customer_status"]): float(row["credit_limit_r5000_avg"]) for row in got}
    assert set(produced) == set(expected), produced
    for status, wanted in expected.items():
        assert produced[status] == pytest.approx(oracle[status], rel=tolerance), status
        assert produced[status] == pytest.approx(wanted, rel=tolerance), status


def test_catalog_finds_usage_events_and_its_monthly_partitions(pg: PgTarget, seeded: None) -> None:
    """Claim: capability detection, not assumptions -- the partition set is discovered.

    FAILS IF: ``usage_events`` is reported as a plain table; or its discovered partition bounds do
    not match the monthly set the DDL created.
    """
    from mdqm.models import PartitionGranularity, TableKind
    from mdqm.testing.loaders.postgres import monthly_partitions

    adapter = postgres_adapter(pg, "billing")
    try:
        snapshot = adapter.extract_catalog()
        table = snapshot.table("usage_events")
        assert table is not None, [t.name for t in snapshot.tables]
        assert table.kind is TableKind.PARTITIONED_TABLE
        assert {"client_no", "start_time", "insertion_time", "units"} <= {
            column.name for column in table.columns
        }
        declared = adapter.extract_partitions()
    finally:
        adapter.close()

    # The bounds carry the physical partition names, which is what identifies the right set.
    monthly = [
        partitioning
        for partitioning in declared
        if any(bound.name.startswith("usage_events_p") for bound in partitioning.bounds)
    ]
    assert len(monthly) == 1, [p.key for p in declared]
    partitioning = monthly[0]
    assert "start_time" in partitioning.key
    assert partitioning.granularity is PartitionGranularity.MONTH
    expected = monthly_partitions(ANCHOR, forward=0)
    assert len(partitioning.bounds) == len(expected) >= 12
    # `suffix` already carries the `p` (e.g. "p2024_11"), so prefixing it again would name a
    # relation that does not exist and the test would fail against correct code.
    assert [bound.name for bound in partitioning.bounds] == [
        f"usage_events_{partition.suffix}" for partition in expected
    ]


# --- claim (d): the right partition key prunes -------------------------------------------


def _planned_relations(target: PgTarget, database: str, query: sql.Composable) -> list[str]:
    """Base relations the planner will read, from ``EXPLAIN (FORMAT JSON)``."""
    import json

    with psycopg.connect(target.dsn(database)) as conn, conn.cursor() as cur:
        cur.execute(sql.SQL("EXPLAIN (FORMAT JSON) {}").format(query))
        row = cur.fetchone()
    assert row is not None
    document = row[0]
    if isinstance(document, str):
        document = json.loads(document)
    names: list[str] = []

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        name = node.get("Relation Name")
        if isinstance(name, str):
            names.append(name)
        for child in node.get("Plans", []) or []:
            walk(child)

    for entry in document:
        walk(entry.get("Plan") if isinstance(entry, dict) else None)
    return names


def test_event_time_prunes_and_ingestion_time_does_not(pg: PgTarget) -> None:
    """Claim (d): the event-time predicate is the one that prunes; ingestion time cannot.

    FAILS IF: an event-time window touches more than two partitions (the bound would be wasting the
    primary), or the identical bound on the ingestion time prunes at all (the two columns would be
    interchangeable, and "what is new" would need no derived bound).
    """
    with psycopg.connect(pg.dsn("billing")) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM pg_inherits i JOIN pg_class parent ON parent.oid = i.inhparent "
            "WHERE parent.relname = 'usage_events'"
        )
        counted = cur.fetchone()
    assert counted is not None
    total = int(counted[0])
    assert total >= 12, total

    event_time = _planned_relations(
        pg,
        "billing",
        sql.SQL("SELECT count(*) FROM public.usage_events WHERE start_time >= {}").format(
            sql.Literal(WINDOW_START)
        ),
    )
    event_partitions = [name for name in event_time if name.startswith("usage_events_p")]
    assert len(event_partitions) <= 2, event_partitions

    ingestion_time = _planned_relations(
        pg,
        "billing",
        sql.SQL("SELECT count(*) FROM public.usage_events WHERE insertion_time >= {}").format(
            sql.Literal(WINDOW_START)
        ),
    )
    ingestion_partitions = [name for name in ingestion_time if name.startswith("usage_events_p")]
    assert len(ingestion_partitions) == total, ingestion_partitions


# --- claim (e): read-only is the server's own refusal ------------------------------------


def test_read_only_is_enforced_by_the_server_not_by_us(pg: PgTarget) -> None:
    """Claim (e) layers 1 and 3: the account is read-only and the transaction is read-only.

    FAILS IF: a write succeeds; or it is refused with something other than PostgreSQL's own
    SQLSTATE (a client-side check would prove nothing); or forcing the transaction read-write finds
    the privilege waiting there.
    """
    attempts = (
        "DELETE FROM public.usage_events WHERE start_time < TIMESTAMP '2026-01-01'",
        "UPDATE public.accounts SET state = 'Hold'",
        "DROP TABLE public.accounts_hist",
    )
    for statement in attempts:
        with (
            psycopg.connect(pg.dsn("billing", read_only=True)) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SHOW transaction_read_only")
            assert cur.fetchone() == ("on",)
            with pytest.raises(psycopg.errors.Error) as excinfo:
                cur.execute(statement)
            conn.rollback()
        error = excinfo.value
        assert error.sqlstate in READ_ONLY_SQLSTATES, f"{statement}: {error.sqlstate} {error}"
        assert str(error).strip(), "a refusal must carry the server's message"

    # Layer 1 alone: forcing the transaction read-write still cannot write.
    with psycopg.connect(pg.dsn("billing", read_only=True)) as conn, conn.cursor() as cur:
        cur.execute("SET default_transaction_read_only = off")
        conn.commit()
        with pytest.raises(psycopg.errors.Error) as excinfo:
            cur.execute("DELETE FROM public.accounts_hist")
        conn.rollback()
    assert excinfo.value.sqlstate == psycopg.errors.InsufficientPrivilege.sqlstate


# --- claim (g): a timeout cancels the work on the server ---------------------------------


def test_timeout_cancels_a_real_pg_sleep(pg: PgTarget) -> None:
    """Claim (g) on a real server: past the deadline the query is cancelled, not abandoned.

    FAILS IF: the call returns without a ``TIMEOUT`` refusal; or a backend is still running the
    sleep afterwards (the connection was dropped while the server kept working); or the call takes
    the whole sleep, which means the deadline was reported and nothing was cancelled.
    """
    from mdqm.governor import Governor, GovernorRefusal

    adapter = postgres_adapter(pg, "billing")
    governor = Governor({"billing": adapter})
    fragment = Fragment(
        source="billing",
        dialect="postgres",
        sql="SELECT count(*) AS n FROM usage_events a CROSS JOIN usage_events b",
        exec_ts=datetime(2026, 11, 25, 15, 0, tzinfo=ANCHOR.tzinfo),
    )
    started = time.monotonic()
    try:
        with pytest.raises(GovernorRefusal) as excinfo:
            governor.execute(source="billing", user="e2e", fragment=fragment, limits=_limits(0.5))
        elapsed = time.monotonic() - started
    finally:
        adapter.close()

    assert excinfo.value.code == "TIMEOUT", excinfo.value.response
    assert excinfo.value.response.fix

    with psycopg.connect(pg.dsn("billing")) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE state = 'active' "
            "AND query LIKE '%usage_events b%' AND pid <> pg_backend_pid()"
        )
        row = cur.fetchone()
    assert row is not None
    assert int(row[0]) == 0, "a backend is still running the cancelled query"
    assert elapsed < 5.0, f"the call waited {elapsed:.1f}s: nothing was cancelled"
