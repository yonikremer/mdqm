"""D04: the mock OpenAPI service — auth, filters, paging, rate limit, error shape, spec drift.

Two transports, one app:

* ``mock_client`` drives the app **in process** through ``httpx.ASGITransport``. No port, no
  socket, so it runs under ``tests/unit``'s ``pytest-socket`` block — which is also the proof that
  the mock cannot reach the internet.
* ``mock_api_served`` starts the same app on an ephemeral loopback port and tears it down. Only the
  one test that needs a socket opts in, and it allows ``127.0.0.1`` alone.

The API key is a throwaway literal here, never in ``src``: ``MockApiConfig(api_key=...)`` is the
only way in, and the default (empty) key authenticates nothing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_socket

from mdqm.testing.mock_api import (
    AUTH_HEADER,
    CURRENCIES,
    FIRST_DAY,
    RATE_DAYS,
    MockApi,
    MockApiConfig,
    _Budget,
    rate_rows,
)

SPEC_PATH = Path(__file__).resolve().parents[1] / "data" / "specs" / "fx.yaml"

#: A throwaway value for the fake API. It protects nothing; it must simply never live in ``src``.
KEY = "d04-throwaway-key"

#: The command the drift test tells you to run.
REGENERATE = "python -m mdqm.testing.mock_api --write-spec tests/data/specs/fx.yaml"

#: Windows builds the asyncio event loop from a loopback socket pair, so merely *creating* a loop
#: needs ``socket.socket`` **and** a connect to ``127.0.0.1``. Every test here is async, so the
#: marker has to be module-wide; the autouse fixture below then narrows the permission to loopback
#: and ``test_the_module_cannot_reach_the_internet`` proves nothing else is reachable.
pytestmark = pytest.mark.enable_socket


@pytest.fixture(autouse=True)
def loopback_only() -> Iterator[None]:
    """``127.0.0.1`` and nothing else: the loop's self-pipe, never a source or the internet."""
    pytest_socket.socket_allow_hosts(["127.0.0.1"])
    yield
    pytest_socket.socket_allow_hosts(["127.0.0.1"])


@pytest.fixture
def mock_api() -> MockApi:
    """The service itself: counters, budgets and the spec; the budget is out of the way."""
    return MockApi(MockApiConfig(api_key=KEY, rate_limit_budget=100, rate_limit_window_s=60.0))


@pytest.fixture
async def mock_client(mock_api: MockApi) -> AsyncIterator[httpx.AsyncClient]:
    """An authenticated client over the in-process ASGI app (no socket, no port)."""
    async with mock_api.client(headers={AUTH_HEADER: KEY}) as client:
        yield client


@pytest.fixture
def mock_api_served(mock_api: MockApi) -> Iterator[str]:
    """The same app on an ephemeral loopback port: the acceptance "starts on a port" fixture."""
    with mock_api.serve() as base_url:
        yield base_url


def rows_for(currency: str) -> list[dict[str, Any]]:
    """Ground truth straight from the module's dataset, never from the endpoint under test."""
    return [
        {"currency": row.currency, "date": row.date.isoformat()}
        for row in rate_rows()
        if row.currency == currency
    ]


# --- happy path and filters ---------------------------------------------------------------------


async def test_health_is_open_and_costs_nothing(mock_api: MockApi) -> None:
    async with mock_api.client() as client:  # no key configured on the client at all
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}
    stats = mock_api.stats()
    assert stats.total_calls == 0, "a liveness probe must not be counted as a data call"
    assert stats.budget_remaining == 100, "a liveness probe must not spend the budget"


async def test_rates_happy_path(mock_client: httpx.AsyncClient) -> None:
    response = await mock_client.get("/rates", params={"currency": "usd", "limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert [row["currency"] for row in body["items"]] == ["USD"] * 5
    assert body["total"] == RATE_DAYS
    assert body["next_cursor"] == "c5"
    assert response.headers["X-RateLimit-Limit"] == "100"
    assert response.headers["X-RateLimit-Remaining"] == "99"
    assert int(response.headers["X-RateLimit-Reset"]) >= 1


@pytest.mark.parametrize(
    ("params", "expected_currencies", "expected_days"),
    [
        ({}, tuple(sorted(CURRENCIES)), RATE_DAYS),
        ({"currency": "GBP"}, ("GBP",), RATE_DAYS),
        ({"currency": "gbp"}, ("GBP",), RATE_DAYS),
        ({"date_from": "2024-01-10", "date_to": "2024-01-12"}, tuple(sorted(CURRENCIES)), 3),
        ({"currency": "ILS", "date_from": "2024-01-30"}, ("ILS",), 1),
        ({"currency": "CHF"}, (), 0),
    ],
)
async def test_filters_shape_the_result(
    mock_client: httpx.AsyncClient,
    params: dict[str, str],
    expected_currencies: tuple[str, ...],
    expected_days: int,
) -> None:
    response = await mock_client.get("/rates", params={**params, "limit": 100})
    assert response.status_code == 200
    items = response.json()["items"]
    assert {row["currency"] for row in items} == set(expected_currencies)
    assert len({(row["currency"], row["date"]) for row in items}) == len(items), "no duplicates"
    assert response.json()["total"] == expected_days * len(expected_currencies)


async def test_pagination_covers_every_row_exactly_once(mock_client: httpx.AsyncClient) -> None:
    seen: list[dict[str, Any]] = []
    cursor: str | None = None
    pages = 0
    while True:
        params: dict[str, Any] = {"currency": "EUR", "limit": 7}
        if cursor is not None:
            params["cursor"] = cursor
        response = await mock_client.get("/rates", params=params)
        assert response.status_code == 200
        body = response.json()
        seen.extend(body["items"])
        pages += 1
        cursor = body["next_cursor"]
        if cursor is None:
            break
        assert pages < 20, "paging did not terminate"

    assert pages == 5, "30 rows at 7 per page is 5 pages"
    seen_keys = [(row["currency"], row["date"]) for row in seen]
    assert seen_keys == [(row["currency"], row["date"]) for row in rows_for("EUR")], (
        "every row exactly once, in order, and nothing else"
    )


async def test_customer_score_and_unknown_id(mock_client: httpx.AsyncClient) -> None:
    found = await mock_client.get("/customers/C-0007/score")
    assert found.status_code == 200
    assert found.json() == {"customer_id": "C-0007", "score": 57, "band": "medium"}

    missing = await mock_client.get("/customers/C-9999/score")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"
    assert "C-9999" in missing.json()["error"]["message"]


# --- the error envelope -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "label"),
    [
        ({}, "no header"),
        ({AUTH_HEADER: ""}, "empty key"),
        ({AUTH_HEADER: "not-the-key"}, "wrong key"),
    ],
)
async def test_missing_or_wrong_key_is_401(
    mock_api: MockApi, headers: dict[str, str], label: str
) -> None:
    async with mock_api.client(headers=headers) as client:
        response = await client.get("/rates")
    assert response.status_code == 401, label
    body = response.json()
    assert body["error"]["code"] == "unauthorized"
    assert body["error"]["details"]["header"] == AUTH_HEADER
    assert response.headers["X-RateLimit-Remaining"] == "0"
    assert "not-the-key" not in response.text, "the presented key must never be echoed"
    assert mock_api.stats().unauthorized == 1
    assert mock_api.stats().total_calls == 0, "a refused request is not a handled data call"


async def test_invalid_parameters_use_the_same_envelope(mock_client: httpx.AsyncClient) -> None:
    bad_cursor = await mock_client.get("/rates", params={"cursor": "zz-not-a-cursor"})
    assert bad_cursor.status_code == 400
    assert bad_cursor.json()["error"]["code"] == "invalid_request"

    bad_limit = await mock_client.get("/rates", params={"limit": 0})
    assert bad_limit.status_code == 400
    assert bad_limit.json()["error"]["code"] == "invalid_request"

    swapped = await mock_client.get(
        "/rates", params={"date_from": "2024-01-10", "date_to": "2024-01-02"}
    )
    assert swapped.status_code == 400
    assert swapped.json()["error"]["code"] == "invalid_request"
    assert "date_from" in swapped.json()["error"]["details"]["fix"]

    unknown_route = await mock_client.get("/nope")
    assert unknown_route.status_code == 404
    assert unknown_route.json()["error"]["code"] == "not_found"


# --- the rate-limit budget ----------------------------------------------------------------------


async def test_over_budget_is_429_with_retry_after() -> None:
    api = MockApi(MockApiConfig(api_key=KEY, rate_limit_budget=2, rate_limit_window_s=60.0))
    async with api.client(headers={AUTH_HEADER: KEY}) as client:
        assert (await client.get("/rates")).status_code == 200
        assert (await client.get("/rates")).status_code == 200
        refused = await client.get("/rates")

    assert refused.status_code == 429
    assert refused.json()["error"]["code"] == "rate_limited"
    assert refused.headers["X-RateLimit-Remaining"] == "0"
    retry_after = int(refused.headers["Retry-After"])
    assert 1 <= retry_after <= 60
    stats = api.stats()
    assert stats.rate_limited == 1
    assert stats.total_calls == 2, "the refused request never reached the handler"
    assert stats.budget_remaining == 0


async def test_a_new_window_admits_again() -> None:
    now = [500.0]
    api = MockApi(
        MockApiConfig(api_key=KEY, rate_limit_budget=1, rate_limit_window_s=30.0),
        clock=lambda: now[0],
    )
    async with api.client(headers={AUTH_HEADER: KEY}) as client:
        assert (await client.get("/rates")).status_code == 200
        refused = await client.get("/rates")
        assert refused.status_code == 429
        assert int(refused.headers["Retry-After"]) == 30

        now[0] += 30.5
        again = await client.get("/rates")
        assert again.status_code == 200
        assert again.headers["X-RateLimit-Remaining"] == "0"

        served = await client.get("/_stats")
    stats = served.json()
    assert stats["calls"] == {"/rates": 2}
    assert stats["rate_limited"] == 1
    assert stats["unauthorized"] == 0


@pytest.mark.parametrize(
    ("limit", "window", "times", "expected"),
    [
        (2, 10.0, [0.0], [(True, 1, 0.0)]),
        (2, 10.0, [0.0, 1.0], [(True, 1, 0.0), (True, 0, 0.0)]),
        (2, 10.0, [0.0, 1.0, 2.0], [(True, 1, 0.0), (True, 0, 0.0), (False, 0, 8.0)]),
        (1, 10.0, [100.0, 109.9, 110.0], [(True, 0, 0.0), (False, 0, 0.1), (True, 0, 0.0)]),
    ],
)
def test_budget_window_arithmetic(
    limit: int, window: float, times: list[float], expected: list[tuple[bool, int, float]]
) -> None:
    """The pure core of the limiter, so the 429 semantics are checked without any HTTP."""
    budget = _Budget(limit=limit, window_s=window)
    answers = []
    for at in times:
        allowed, remaining, retry_after = budget.take(at)
        answers.append((allowed, remaining, round(retry_after, 6)))
    assert answers == expected


async def test_budget_is_per_key_and_a_second_key_is_unaffected() -> None:
    api = MockApi(MockApiConfig(api_key=KEY, rate_limit_budget=1, rate_limit_window_s=60.0))
    async with api.client(headers={AUTH_HEADER: KEY}) as client:
        assert (await client.get("/rates")).status_code == 200
        assert (await client.get("/rates")).status_code == 429
    # A different (wrong) key is refused at the door, never metered against the real key's budget.
    async with api.client(headers={AUTH_HEADER: "another-key"}) as other:
        assert (await other.get("/rates")).status_code == 401
    assert api.stats().budget_remaining == 0


# --- counters, the slow endpoint, and configuration ---------------------------------------------


async def test_stats_count_handled_calls_only(mock_api: MockApi) -> None:
    async with mock_api.client(headers={AUTH_HEADER: KEY}) as client:
        await client.get("/rates", params={"limit": 1})
        await client.get("/rates", params={"limit": 1})
        await client.get("/customers/C-0001/score")
        first = await client.get("/_stats")
        second = await client.get("/_stats")

    assert first.json() == second.json(), "/_stats does not change what it reports"
    assert first.json()["calls"] == {
        "/customers/{customer_id}/score": 1,
        "/rates": 2,
    }
    assert first.json()["total_calls"] == 3
    assert first.json()["budget_remaining"] == 97


async def test_slow_endpoint_sleeps_and_caps(mock_api: MockApi) -> None:
    capped = MockApi(MockApiConfig(api_key=KEY, max_delay_ms=20))
    async with capped.client(headers={AUTH_HEADER: KEY}) as client:
        answer = await client.get("/slow", params={"delay_ms": 5000})
        plain = await client.get("/slow", params={"delay_ms": 0})

    assert answer.status_code == 200
    assert answer.json() == {"requested_ms": 5000, "slept_ms": 20, "capped": True}
    assert plain.json() == {"requested_ms": 0, "slept_ms": 0, "capped": False}
    assert capped.stats().calls["/slow"] == 2


@pytest.mark.parametrize(
    "config",
    [
        {"rate_limit_budget": 0},
        {"rate_limit_window_s": 0.0},
        {"max_delay_ms": -1},
    ],
)
def test_config_refuses_impossible_values(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MockApiConfig(**config)


async def test_an_unconfigured_key_authenticates_nothing() -> None:
    """Fail closed: a mock the operator forgot to configure must not answer anyone."""
    api = MockApi()
    assert api.config.api_key == ""
    async with api.client(headers={AUTH_HEADER: ""}) as client:
        assert (await client.get("/rates")).status_code == 401
    async with api.client(headers={AUTH_HEADER: "anything"}) as client:
        assert (await client.get("/rates")).status_code == 401


def test_rate_rows_are_deterministic_and_sorted() -> None:
    rows = rate_rows()
    assert rows == rate_rows()
    assert len(rows) == len(CURRENCIES) * RATE_DAYS
    assert rows[0].date == FIRST_DAY
    assert [row.currency for row in rows] == sorted(row.currency for row in rows)
    assert len({(row.currency, row.date) for row in rows}) == len(rows)


# --- the checked-in spec ------------------------------------------------------------------------


def test_checked_in_spec_matches_the_live_spec(mock_api: MockApi) -> None:
    """Drift test: the YAML in the repo is what the app serves today, byte for byte."""
    live = mock_api.spec_yaml()
    checked_in = SPEC_PATH.read_text(encoding="utf-8")
    assert checked_in == live, f"{SPEC_PATH} has drifted from the live spec: run `{REGENERATE}`"

    from ruamel.yaml import YAML

    assert YAML(typ="rt").load(checked_in) == mock_api.spec(), "and it parses to the same document"


def test_spec_documents_what_the_adapter_discovers(mock_api: MockApi) -> None:
    """What D4 reads: GET operations, filter params, paging, auth, 429 headers."""
    spec = mock_api.spec()
    assert spec["openapi"].startswith("3.")

    schemes = spec["components"]["securitySchemes"]
    assert schemes["ApiKeyAuth"] == {
        "type": "apiKey",
        "in": "header",
        "name": AUTH_HEADER,
        "description": schemes["ApiKeyAuth"]["description"],
    }

    operations = {path: item["get"] for path, item in spec["paths"].items()}
    assert set(operations) == {
        "/_stats",
        "/customers/{customer_id}/score",
        "/health",
        "/rates",
        "/slow",
    }
    assert all(set(item) == {"get"} for item in spec["paths"].values()), "GET only, by principle"

    rates = operations["/rates"]
    assert rates["security"] == [{"ApiKeyAuth": []}]
    params = {param["name"]: param for param in rates["parameters"]}
    assert set(params) == {"currency", "date_from", "date_to", "limit", "cursor"}
    assert all(param["in"] == "query" for param in params.values())
    assert params["date_from"]["schema"]["anyOf"][0] == {"type": "string", "format": "date"}
    assert params["limit"]["schema"]["type"] == "integer"
    assert params["cursor"]["schema"]["anyOf"][0]["pattern"] == "^c[0-9]+$", (
        "the cursor is opaque and its shape is documented"
    )

    ok = rates["responses"]["200"]
    assert ok["content"]["application/json"]["schema"]["$ref"].endswith("/RatePage")
    assert "X-RateLimit-Remaining" in ok["headers"]
    limited = rates["responses"]["429"]
    assert "Retry-After" in limited["headers"]
    assert limited["content"]["application/json"]["schema"]["$ref"].endswith("/ErrorBody")
    assert "401" in rates["responses"]
    assert "422" not in rates["responses"], "this API answers a bad parameter with 400"

    assert operations["/health"].get("security") is None, "health is open on purpose"
    assert set(spec["components"]["schemas"]["Health"]["properties"]) == {"status", "version"}


# --- the real-port path -------------------------------------------------------------------------


@pytest.mark.enable_socket
def test_serves_over_loopback_and_closes_the_port(mock_api: MockApi) -> None:
    """The acceptance fixture: ephemeral port, real roundtrip, teardown that closes it."""
    with mock_api.serve() as base_url:
        assert base_url.startswith("http://127.0.0.1:")
        port = int(base_url.rsplit(":", 1)[1])
        assert port > 0
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            assert client.get("/health").json()["status"] == "ok"
            assert client.get("/rates").status_code == 401
            page = client.get("/rates", params={"currency": "EUR"}, headers={AUTH_HEADER: KEY})
            assert page.status_code == 200
            assert len(page.json()["items"]) == 10

    client = httpx.Client(base_url=base_url, timeout=5.0)
    with client, pytest.raises(httpx.TransportError):
        client.get("/health")


def test_the_dataset_has_no_boundary_dependency_on_the_clock() -> None:
    """A real date type on the wire, so a caller cannot smuggle a datetime into a day filter."""
    row = rate_rows()[0]
    assert isinstance(row.date, date)
    assert row.date.isoformat() == "2024-01-01"


def test_the_module_cannot_reach_the_internet() -> None:
    """The guard that makes every other test's silence meaningful: only loopback is reachable.

    ``pytest-socket`` allows ``127.0.0.1`` for the event loop's self-pipe (see the module docstring)
    and refuses every other host, so a stray real request in this module would fail loudly here
    rather than quietly pass on a machine that has internet.
    """
    with pytest.raises(pytest_socket.SocketConnectBlockedError):
        httpx.get("http://203.0.113.10:9/", timeout=1.0)
