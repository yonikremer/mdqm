"""The published mock API (compose profile ``api``), not the in-process app.

The unit tests drive ``MockApi`` through ASGI and never open a port. This file is the proof that
the container on the documented host port is that same app: liveness is open, a collection needs
the key, a write method is not served, and a filtered read returns the planted rows.

    python scripts/dev.py up api
    python -m pytest tests/integration/test_mock_api_live.py --require-services -q
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from mdqm.testing.loaders.postgres import find_dotenv, read_dotenv
from mdqm.testing.mock_api import AUTH_HEADER

pytestmark = pytest.mark.integration

DEFAULT_PORT = 58000


def _unavailable(request: pytest.FixtureRequest, reason: str) -> None:
    if request.config.getoption("--require-services"):
        pytest.fail(f"required service unavailable: {reason}")
    pytest.skip(f"service unavailable: {reason}")


def _base_url() -> tuple[str, str]:
    values = read_dotenv(find_dotenv() or Path(".env"))
    port = int(values.get("MDQM_API_PORT", DEFAULT_PORT))
    key = values.get("MDQM_MOCK_API_KEY", "")
    return f"http://127.0.0.1:{port}", key


def test_published_mock_is_get_only_and_serves_the_planted_rates(
    request: pytest.FixtureRequest,
) -> None:
    """Health is open, ``/rates`` needs the key, POST is not a route, and USD rows come back."""
    base, key = _base_url()
    if not key:
        _unavailable(request, "MDQM_MOCK_API_KEY is empty (python scripts/dev.py env)")
        return
    try:
        health = httpx.get(f"{base}/health", timeout=5.0)
    except httpx.HTTPError as exc:
        _unavailable(request, f"mock API at {base}: {exc}")
        return
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    refused = httpx.get(f"{base}/rates", params={"currency": "USD"}, timeout=5.0)
    assert refused.status_code == 401

    posted = httpx.post(f"{base}/rates", headers={AUTH_HEADER: key}, timeout=5.0)
    assert posted.status_code == 405, posted.text

    page = httpx.get(
        f"{base}/rates",
        params={"currency": "USD", "limit": 100},
        headers={AUTH_HEADER: key},
        timeout=5.0,
    )
    assert page.status_code == 200, page.text
    body = page.json()
    assert body["total"] > 0
    assert {row["currency"] for row in body["items"]} == {"USD"}

    spec = httpx.get(f"{base}/openapi.json", timeout=5.0)
    assert spec.status_code == 200
    paths = spec.json()["paths"]
    assert "/rates" in paths
    assert set(paths["/rates"]) == {"get"}
