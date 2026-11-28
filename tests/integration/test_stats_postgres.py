"""PostgreSQL statistics collection and the bounded existence lookup.

Needs the ``pg`` profile (docs/dev-env.md). Without it the test skips; with
``--require-services`` a missing database is a failure.

    python scripts/dev.py up pg
    python -m pytest tests/integration/test_stats_postgres.py --require-services -q
"""

from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import pytest

from mdqm.stats.measure import ColumnRequest
from mdqm.stats.models import CollectionMethod
from mdqm.stats.postgres import exists, measure
from mdqm.testing.loaders.postgres import LoaderError, target_from_environ

WHEN = datetime(2026, 11, 28, tzinfo=UTC)


def _skip_or_fail(request: pytest.FixtureRequest, reason: str) -> None:
    if request.config.getoption("--require-services"):
        pytest.fail(f"required service unavailable: {reason}")
    pytest.skip(f"service unavailable: {reason}")


def test_postgres_catalog_or_sample_measures_the_planted_shares(
    request: pytest.FixtureRequest,
) -> None:
    """A temp table's NULL, zero, empty and whitespace shares match the rows that were inserted.

    The collector prefers ``pg_stats``. A brand-new temp table often has no statistics yet, and
    then the capped read is the fallback. Either way the shares are the planted ones and the
    method is recorded. The existence probe is the same statement the stale-stats path runs.
    """
    try:
        target = target_from_environ()
    except LoaderError as exc:
        _skip_or_fail(request, str(exc))
        return
    try:
        connection = psycopg.connect(target.dsn("crm"), connect_timeout=5)
    except psycopg.Error as exc:
        _skip_or_fail(request, str(exc))
        return
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE TEMP TABLE mdqm_stats_probe (status text, amount integer, note text)"
            )
            cursor.executemany(
                "INSERT INTO mdqm_stats_probe VALUES (%s, %s, %s)",
                [
                    ("Active", 10, "ok"),
                    ("Active", 0, ""),
                    ("Closed", None, "   "),
                    (None, 5, "ok"),
                ],
            )
            cursor.execute("ANALYZE mdqm_stats_probe")
        collected = measure(
            connection,
            source="crm",
            table="mdqm_stats_probe",
            columns=(
                ColumnRequest("status", text=True),
                ColumnRequest("amount", numeric=True),
                ColumnRequest("note", text=True),
            ),
            collected_at=WHEN,
        )
        by_name = {item.column: item for item in collected.quality}
        assert by_name["status"].null_share == pytest.approx(0.25)
        assert by_name["amount"].zero_share == pytest.approx(0.25)
        assert by_name["note"].empty_share == pytest.approx(0.25)
        assert by_name["note"].whitespace_share == pytest.approx(0.25)
        assert by_name["status"].method in {
            CollectionMethod.CATALOG,
            CollectionMethod.EXACT,
            CollectionMethod.SAMPLE,
            CollectionMethod.MIXED,
        }
        assert exists(connection, table="mdqm_stats_probe", column="status", value="Active") is True
        assert (
            exists(connection, table="mdqm_stats_probe", column="status", value="Missing") is False
        )
    connection.close()
