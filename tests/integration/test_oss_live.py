"""The open-source stack: MariaDB's read-only login and the OpenSearch seed.

Needs the ``oss`` profile. Without it the tests skip; with ``--require-services`` a missing
service is a failure.

    python scripts/dev.py up oss
    python -m pytest tests/integration/test_oss_live.py --require-services -q

MariaDB is reached through the container's own client. This project has no MySQL driver on
purpose (nothing in the server speaks that protocol); the password stays in the container
environment and is never placed on the host command line.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import httpx
import pytest

from mdqm.testing.loaders.postgres import find_dotenv, read_dotenv
from mdqm.testing.world import WorldConfig

pytestmark = pytest.mark.integration

MARIADB_CONTAINER = "mdqm-mariadb-1"
_MARIADB = (
    'mariadb --user=mdqm_ro --password="$MDQM_RO_PASSWORD" --database=mdqm '
    "--batch --skip-column-names"
)


def _unavailable(request: pytest.FixtureRequest, reason: str) -> None:
    if request.config.getoption("--require-services"):
        pytest.fail(f"required service unavailable: {reason}")
    pytest.skip(f"service unavailable: {reason}")


def _values() -> dict[str, str]:
    return read_dotenv(find_dotenv() or Path(".env"))


def _exec(script: str, *, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv; the password stays in the container
        ["docker", "exec", "-i", MARIADB_CONTAINER, "sh", "-c", script],  # noqa: S607
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_opensearch_serves_the_seeded_alias(request: pytest.FixtureRequest) -> None:
    """The alias, the monthly indices and a search all answer, and the document count matches."""
    port = int(_values().get("MDQM_OPENSEARCH_PORT", 59201))
    base = f"http://127.0.0.1:{port}"
    try:
        health = httpx.get(f"{base}/_cluster/health", timeout=5.0)
    except httpx.HTTPError as exc:
        _unavailable(request, f"OpenSearch at {base}: {exc}")
        return
    assert health.status_code == 200, health.text
    assert health.json()["status"] in {"yellow", "green"}

    alias = httpx.get(f"{base}/_alias/mdqm-usage-events", timeout=5.0)
    assert alias.status_code == 200, alias.text
    indices = sorted(alias.json())
    assert len(indices) >= 8, indices
    assert all(name.startswith("mdqm-usage-events-") for name in indices)

    counted = httpx.get(f"{base}/mdqm-usage-events/_count", timeout=10.0)
    assert counted.status_code == 200, counted.text
    assert counted.json()["count"] == WorldConfig.small().event_count

    found = httpx.post(
        f"{base}/mdqm-usage-events/_search",
        json={"size": 0, "track_total_hits": True},
        timeout=10.0,
    )
    assert found.status_code == 200, found.text


def test_mariadb_read_only_login_can_select_and_cannot_write(
    request: pytest.FixtureRequest,
) -> None:
    """``mdqm_ro`` reads the seeded world and MariaDB itself refuses a write (error 1142)."""
    ping = _exec('mariadb-admin ping -h127.0.0.1 -uroot -p"$MARIADB_ROOT_PASSWORD"')
    if ping.returncode != 0:
        _unavailable(
            request,
            f"MariaDB container {MARIADB_CONTAINER} is not answering: {ping.stderr.strip()}",
        )
        return

    selected = _exec(
        _MARIADB,
        stdin=(
            "SELECT COUNT(*) FROM billing_usage_events;\n"
            "SELECT COUNT(*) FROM crm_customers;\n"
            "SELECT COUNT(*) FROM erp_customers;\n"
        ),
    )
    assert selected.returncode == 0, selected.stderr
    counts = [int(line) for line in selected.stdout.splitlines() if line.strip().isdigit()]
    assert counts[0] == WorldConfig.small().event_count
    assert counts[1] == WorldConfig.small().customers
    assert counts[2] > counts[1], "the planted duplicate erp rows must survive the load"

    denied = _exec(
        'mariadb --user=mdqm_ro --password="$MDQM_RO_PASSWORD" --database=mdqm '
        '-e "CREATE TABLE mdqm_ro_must_fail (id INT)"'
    )
    assert denied.returncode != 0
    assert "1142" in denied.stderr or "denied" in denied.stderr.lower(), denied.stderr
