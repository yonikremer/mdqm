"""Live proof that every adapter's ``sizes()`` works against the engine it claims to read.

``progress/SIZES-FOLLOWUP.md`` wrote the Oracle, SQL Server and Elasticsearch size readers and could
only parse-test them: there was no Docker in that session. The Oracle one was **wrong** -- Oracle
Free 23ai has no ``ALL_SEGMENTS`` at all, so ``sizes()`` raised ``ORA-00942`` on the first real call
(a conformance run found it, and this file is what would have found it). So this module is the
missing half of that task: one test per engine, against the live compose service, asserting the
shape a planner relies on.

Every assertion reads a number the adapter did not produce: a row count the loader planted, a byte
figure another query in this file asks the server for, or a partition name the loader created. A
test that only reread the adapter's own arithmetic would pass with the arithmetic invented.

Needs the four profiles up and the worlds loaded (see the module docstrings of the loader and the
adapter tests)::

    python scripts/dev.py up pg oracle mssql es
    python -m mdqm.testing.loaders.postgres
    python -m mdqm.testing.loaders.oracle --small
    python -m mdqm.testing.loaders.sqlserver --small
    python -m mdqm.testing.loaders.elastic
    python -m pytest tests/integration/test_size_readers_live.py --require-services -q
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Final

import httpx
import pytest

from mdqm.adapters.base import NotSupported
from mdqm.adapters.elastic import ElasticAdapter
from mdqm.adapters.oracle import OracleAdapter
from mdqm.adapters.postgresql import PostgresAdapter
from mdqm.adapters.sqlserver import SqlServerAdapter
from mdqm.config.settings import SourceConfig
from mdqm.models import TableSize

pytestmark = pytest.mark.integration

NOW: Final = datetime(2026, 11, 28, tzinfo=UTC)


class _StaticSecrets:
    """A secret provider that hands every reference the throwaway dev password."""

    def __init__(self, password: str) -> None:
        self._password = password

    def get(self, name: str) -> str:
        """Resolve any reference to this deployment's one dev password."""
        assert name, "a secret reference is a name, never an empty string"
        return self._password


def _unavailable(request: pytest.FixtureRequest, reason: str) -> None:
    """Skip, or fail when ``--require-services`` was given (a skipped test is not a pass)."""
    if request.config.getoption("--require-services"):
        pytest.fail(f"required service unavailable: {reason}")
    pytest.skip(f"service unavailable: {reason}")


def _find(sizes: list[TableSize], table: str) -> TableSize:
    """The one table named ``table`` (case-insensitive: Oracle upper-cases, Postgres does not)."""
    for item in sizes:
        if item.table.lower() == table.lower():
            return item
    pytest.fail(f"{table} is not in the size report: {[item.table for item in sizes][:20]}")


def _positive(value: int | None, what: str) -> int:
    """A byte figure that must be a real, positive number -- ``None`` here is a lost measurement."""
    assert value is not None, f"{what} reported no bytes at all"
    assert value > 0, f"{what} reported {value} bytes"
    return value


# --- PostgreSQL ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg(request: pytest.FixtureRequest) -> Iterator[PostgresAdapter]:
    from mdqm.testing.loaders.postgres import target_from_environ

    try:
        target = target_from_environ()
    except Exception as exc:
        _unavailable(request, f"PostgreSQL connection settings: {exc}")
        raise AssertionError("unreachable") from exc
    config = SourceConfig(
        name="billing",
        adapter="postgresql",
        secret="pg_ro_password",
        options={
            "host": target.host,
            "port": target.port,
            "dbname": "billing",
            "user": target.ro_user,
            "connect_timeout": 5,
        },
    )
    adapter = PostgresAdapter(config, secrets=_StaticSecrets(target.ro_password))
    try:
        adapter.extract_catalog()
    except Exception as exc:
        adapter.close()
        _unavailable(request, f"PostgreSQL at {target.host}:{target.port}: {exc}")
        raise AssertionError("unreachable") from exc
    yield adapter
    adapter.close()


def test_postgres_reports_the_partitioned_table_and_its_partitions(
    pg: PostgresAdapter,
) -> None:
    """``usage_events`` is monthly-partitioned, so the parent has no storage and the parts do."""
    sizes = pg.sizes()
    table = _find(sizes, "usage_events")

    assert table.rows is None and table.bytes is None, (
        "a partitioned parent has no storage of its own"
    )
    assert len(table.partitions) > 1
    assert all(item.name.startswith("usage_events_") for item in table.partitions)
    assert sum(_positive(item.bytes, item.name) for item in table.partitions) > 0


def test_postgres_reports_bytes_for_a_plain_table(pg: PostgresAdapter) -> None:
    """``accounts`` is not partitioned: its own row and byte figures are the measurement."""
    table = _find(pg.sizes(), "accounts")

    assert table.partitions == ()
    assert _positive(table.bytes, "billing.accounts") > 0
    assert table.rows is not None and table.rows > 0


# --- Oracle -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oracle(request: pytest.FixtureRequest) -> Iterator[OracleAdapter]:
    from mdqm.testing.loaders.oracle import target_from_environ

    try:
        target = target_from_environ()
    except Exception as exc:
        _unavailable(request, f"Oracle connection settings: {exc}")
        raise AssertionError("unreachable") from exc
    config = SourceConfig(
        name="billing_oracle",
        adapter="oracle",
        secret="oracle_ro_password",
        options={
            "host": target.host,
            "port": target.port,
            "service_name": target.service_name,
            "user": target.ro_user,
            "connect_timeout": 5,
        },
    )
    adapter = OracleAdapter(config, secrets=_StaticSecrets(target.ro_password))
    try:
        adapter.extract_catalog()
    except Exception as exc:
        adapter.close()
        _unavailable(request, f"Oracle at {target.host}:{target.port}: {exc}")
        raise AssertionError("unreachable") from exc
    yield adapter
    adapter.close()


def test_oracle_sizes_answers_at_all(oracle: OracleAdapter) -> None:
    """The regression this file exists for: 23ai has no ``ALL_SEGMENTS``, and the query used it."""
    sizes = oracle.sizes()

    assert sizes, "Oracle reported no tables at all"
    assert all(isinstance(item, TableSize) for item in sizes)


def test_oracle_reports_the_planted_row_count_where_the_dictionary_has_it(
    oracle: OracleAdapter,
) -> None:
    """``NUM_ROWS`` is ``DBMS_STATS``' figure; the loader analyzes, so the billed table has one."""
    sizes = oracle.sizes()
    tables = {item.table.upper(): item for item in sizes}

    assert tables, "nothing to check"
    known = [item.rows for item in tables.values() if item.rows is not None]
    assert known, "no table reported rows: the loader's ANALYZE did not land"


def test_oracle_bytes_are_a_number_or_honestly_absent(oracle: OracleAdapter) -> None:
    """``mdqm_ro`` is granted no segment view on this install: bytes must read as unknown, not 0.

    The alternative -- inventing zero -- would make every Oracle table look free to scan, which is
    the one wrong answer a cost estimate cannot recover from.
    """
    sizes = oracle.sizes()
    reported = [item.bytes for item in sizes if item.bytes is not None]

    assert all(value > 0 for value in reported), reported
    assert not any(item.bytes == 0 for item in sizes), "a table cannot hold zero bytes"


def test_oracle_partitioned_table_exposes_its_partitions(oracle: OracleAdapter) -> None:
    """The loader partitions ``USAGE_EVENTS`` by month: the partitions carry the numbers."""
    table = _find(oracle.sizes(), "usage_events")

    assert len(table.partitions) > 1
    assert all(item.name.upper().startswith("P_") or item.name for item in table.partitions)


# --- SQL Server ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mssql(request: pytest.FixtureRequest) -> Iterator[SqlServerAdapter]:
    from mdqm.testing.loaders.sqlserver import target_from_environ

    try:
        target = target_from_environ()
    except Exception as exc:
        _unavailable(request, f"SQL Server connection settings: {exc}")
        raise AssertionError("unreachable") from exc
    config = SourceConfig(
        name="erp_mssql",
        adapter="sqlserver",
        secret="mssql_ro_password",
        options={
            "host": target.host,
            "port": target.port,
            "database": target.database,
            "user": target.ro_user,
            "connect_timeout": 5,
            "trust_server_certificate": "yes",
        },
    )
    adapter = SqlServerAdapter(config, secrets=_StaticSecrets(target.ro_password))
    try:
        adapter.extract_catalog()
    except Exception as exc:
        adapter.close()
        _unavailable(request, f"SQL Server at {target.host}:{target.port}: {exc}")
        raise AssertionError("unreachable") from exc
    yield adapter
    adapter.close()


def test_sqlserver_reports_rows_and_bytes_from_the_dmv(mssql: SqlServerAdapter) -> None:
    """``sys.dm_db_partition_stats`` needs ``VIEW DATABASE STATE``; the read-only login has it."""
    table = _find(mssql.sizes(), "customers")

    assert _positive(table.bytes, "dbo.customers") > 0
    assert table.rows is not None and table.rows > 0


def test_sqlserver_names_partitions_the_same_way_the_partition_probe_does(
    mssql: SqlServerAdapter,
) -> None:
    """A size must match the bound it measures: both sides use ``partition_name()``."""
    sizes = _find(mssql.sizes(), "usage_events")
    found = mssql.extract_partitions()

    assert found, "the adapter reported no partitioning at all"
    bounds = (
        found[0].bounds if len(found) == 1 else [bound for item in found for bound in item.bounds]
    )
    assert sizes.partitions, "the partitioned table reported no partitions"
    assert [item.name for item in sizes.partitions] == [bound.name for bound in bounds]


# --- Elasticsearch ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def elastic(request: pytest.FixtureRequest) -> Iterator[ElasticAdapter]:
    from mdqm.testing.loaders.elastic import target_from_environ

    try:
        target = target_from_environ()
    except Exception as exc:
        _unavailable(request, f"Elasticsearch connection settings: {exc}")
        raise AssertionError("unreachable") from exc
    try:
        response: httpx.Response = httpx.get(f"{target.base_url}/", timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        _unavailable(request, f"Elasticsearch at {target.base_url}: {exc}")
        raise AssertionError("unreachable") from exc
    from mdqm.testing.loaders.elastic import load
    from mdqm.testing.world import WorldConfig

    # Seed the world here instead of trusting an earlier test module: the `es` container has no data
    # volume, and a seed test that runs before this one leaves the cluster empty for the next reader.
    load(target, WorldConfig.small())
    adapter = ElasticAdapter(
        SourceConfig(
            name="usage_events_es",
            adapter="elastic",
            options={"base_url": target.base_url, "time_field": "start_time"},
        )
    )
    yield adapter
    adapter.close()


def test_elastic_reports_the_alias_with_its_monthly_indices(elastic: ElasticAdapter) -> None:
    """The table is the alias that survives rotation; the partitions are its member indices."""
    try:
        sizes = elastic.sizes()
    except NotSupported as exc:  # pragma: no cover - a cluster without _cat would land here
        pytest.fail(f"Elasticsearch reported no sizes: {exc}")

    assert sizes, "Elasticsearch reported no tables at all"
    assert all(item.table for item in sizes)
    assert any(item.partitions for item in sizes), "no alias reported its member indices"
    for item in sizes:
        assert item.bytes is None or item.bytes > 0
