"""A06: the OpenAPI adapter against the real mock API, driven in process.

Two things are proved here that a stub could never prove: the catalog is read out of a *live*
OpenAPI document (``MockApi.spec()``), and every fetch is a real HTTP request answered by the
FastAPI app. There is no port and no socket: a small synchronous bridge drives
``httpx.ASGITransport``, so the tests also run under ``pytest-socket`` (the ``127.0.0.1`` allowance
below is for the event loop Windows builds from a loopback socket pair, never for a source).

The API key is a throwaway literal in this file, never in ``src``: the adapter reads it from a
secret *reference* through :class:`~mdqm.config.secrets.EnvSecretProvider`, and the tests assert the
value itself never reaches an error.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pyarrow as pa
import pytest
import pytest_socket

from mdqm.adapters.base import REQUIRED_METHODS, supports_method
from mdqm.adapters.openapi import (
    GetOnlyTransport,
    OpenApiAdapter,
    OpenApiError,
    OpenApiUnavailable,
    read_endpoints,
)
from mdqm.adapters.registry import AdapterRegistry
from mdqm.config.secrets import EnvSecretProvider, SecretProvider
from mdqm.config.settings import AuthSettings, SourceConfig
from mdqm.models import Limits, NormalizedType, TableKind
from mdqm.ratelimit import LimitPolicy, RateLimiter
from mdqm.testing.conformance import run_conformance
from mdqm.testing.mock_api import AUTH_HEADER, CURRENCIES, RATE_DAYS, MockApi, MockApiConfig

#: Where the in-process app answers. Any host works: the ASGI bridge never resolves it.
BASE_URL = "http://mock-api.invalid"

#: A throwaway key. It protects nothing; it must simply never live in ``src``.
KEY = "a06-throwaway-key"

#: The secret *reference* the source config names, and the env var the provider reads.
SECRET_REF = "rates_api_key"
SECRET_ENV = {"MDQM_SECRET_RATES_API_KEY": KEY}

pytestmark = pytest.mark.enable_socket


@pytest.fixture(autouse=True)
def loopback_only() -> Iterator[None]:
    """``127.0.0.1`` and nothing else: the event loop's self-pipe, never a source."""
    pytest_socket.socket_allow_hosts(["127.0.0.1"])
    yield
    pytest_socket.socket_allow_hosts(["127.0.0.1"])


class SyncAsgiTransport(httpx.BaseTransport):
    """Drive an ASGI app from the *synchronous* client the adapter uses.

    The adapter is blocking on purpose (``AGENTS.md`` section 3: the governor owns the thread pool),
    while ``httpx.ASGITransport`` is async. This bridge runs the request in a short-lived loop and
    hands back a fully-read response, so no port, no socket and no thread is involved.
    """

    def __init__(self, app: Any) -> None:
        self._app = app
        #: Every request the adapter actually issued, as ``(method, path)``.
        self.seen: list[tuple[str, str]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.seen.append((request.method, request.url.path))

        async def send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self._app)
            answer = await transport.handle_async_request(request)
            return httpx.Response(
                status_code=answer.status_code,
                headers=answer.headers,
                content=await answer.aread(),
            )

        return asyncio.run(send())


def config_for(*, base_url: str = BASE_URL) -> SourceConfig:
    """A source config naming the secret *reference*, never the value, and the base URL."""
    return SourceConfig(
        name="rates_api",
        adapter="openapi",
        auth=AuthSettings(header=AUTH_HEADER, secret=SECRET_REF),
        options={"base_url": base_url},
    )


def build(
    api: MockApi,
    *,
    secrets: SecretProvider | None = None,
    config: SourceConfig | None = None,
    limiter: RateLimiter | None = None,
    page_size: int = 100,
) -> tuple[OpenApiAdapter, SyncAsgiTransport]:
    """The adapter over ``api``, plus the transport so a test can count real requests."""
    bridge = SyncAsgiTransport(api.app)
    adapter = OpenApiAdapter(
        config if config is not None else config_for(),
        secrets if secrets is not None else EnvSecretProvider(environ=SECRET_ENV),
        spec=api.spec(),
        transport=bridge,
        limiter=limiter,
        page_size=page_size,
    )
    return adapter, bridge


@pytest.fixture
def api() -> MockApi:
    """The mock service; its budget is out of the way unless a test lowers it."""
    return MockApi(MockApiConfig(api_key=KEY, rate_limit_budget=1000, rate_limit_window_s=60.0))


def limits(*, max_rows: int = 100, timeout_s: float = 10.0) -> Limits:
    return Limits(timeout_s=timeout_s, max_rows=max_rows, max_bytes=1_000_000, max_partitions=8)


def rows_of(stream: Any) -> list[dict[str, Any]]:
    """Materialise an ``ArrowStream`` into rows, closing it as the governor would."""
    try:
        return pa.Table.from_batches(list(stream), schema=stream.schema).to_pylist()
    finally:
        stream.close()


# --- conformance: the only definition of "a real adapter" ---------------------------------------


def test_passes_the_conformance_kit(api: MockApi) -> None:
    """``run_conformance`` exercises every promise of the protocol; the kit is the judge."""
    adapter, _ = build(api)
    run_conformance(lambda: build(api)[0], fragment=adapter.fragment("rates", currency="USD"))


def test_declares_the_required_protocol_methods(api: MockApi) -> None:
    adapter, _ = build(api)
    missing = [name for name in REQUIRED_METHODS if not supports_method(adapter, name)]
    assert missing == [], f"the registry would reject this adapter: missing {missing}"
    assert adapter.kind == "openapi"
    assert adapter.dialect() == "dsl", "an API source carries a request, not SQL"


def test_registers_through_the_entry_point_group(api: MockApi, tmp_path: Any) -> None:
    """The factory the ``mdqm.adapters`` entry point names is the adapter the registry expects."""
    from mdqm.testing.mock_api import spec_yaml

    path = tmp_path / "api.yaml"
    path.write_text(spec_yaml(api.spec()), encoding="utf-8", newline="\n")
    registry = AdapterRegistry(secrets=EnvSecretProvider(environ=SECRET_ENV))
    registry.register("openapi", OpenApiAdapter)  # a class is its own factory
    created = registry.create(
        SourceConfig(
            name="rates_api",
            adapter="openapi",
            spec=str(path),
            auth=AuthSettings(header=AUTH_HEADER, secret=SECRET_REF),
            options={"base_url": BASE_URL},
        )
    )
    assert created.kind == "openapi", "registry key, kind and SourceConfig.adapter must agree"


def test_pyproject_declares_the_entry_point() -> None:
    """A plugin nobody can discover is not a plugin: the group entry must be in the manifest."""
    from pathlib import Path

    manifest = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    assert '[project.entry-points."mdqm.adapters"]' in manifest
    assert 'openapi = "mdqm.adapters.openapi:OpenApiAdapter"' in manifest


# --- catalog: the spec is the schema ------------------------------------------------------------


def test_catalog_matches_the_spec(api: MockApi) -> None:
    """One virtual table per GET collection, with the spec's columns and its query parameters."""
    adapter, _ = build(api)
    snapshot = adapter.extract_catalog()

    assert snapshot.source == "rates_api"
    assert [table.name for table in snapshot.tables] == ["rates"]
    table = snapshot.tables[0]
    assert table.kind is TableKind.ENDPOINT
    assert [column.name for column in table.columns] == ["currency", "date", "rate", "source"]
    assert {column.name: column.normalized_type for column in table.columns} == {
        "currency": NormalizedType.STRING,
        "date": NormalizedType.DATE,
        "rate": NormalizedType.FLOAT,
        "source": NormalizedType.STRING,
    }
    declared = api.spec()["paths"]["/rates"]["get"]["parameters"]
    assert [index.name for index in table.indexes] == sorted(param["name"] for param in declared), (
        "every declared query parameter is a filterable column, in a deterministic order"
    )


def test_the_probe_claims_nothing_it_cannot_serve(api: MockApi) -> None:
    adapter, _ = build(api)
    capabilities = adapter.probe()
    assert capabilities.adapter == "openapi"
    assert capabilities.source == "rates_api"
    assert capabilities.version == "1.0.0"
    assert capabilities.capabilities == {}, "no optional method is implemented, so none is claimed"


def test_only_collections_become_tables_and_the_rest_are_reported(api: MockApi) -> None:
    """A lookup endpoint and a non-array answer are not tables, and that is said out loud."""
    adapter, _ = build(api)
    ignored = set(adapter.ignored_operations)
    assert "GET /customers/{customer_id}/score" in ignored, (
        "a path parameter is a lookup, not a set"
    )
    assert "GET /health" in ignored, "an object response is one record, not rows"
    assert "GET /slow" in ignored
    assert "GET /_stats" in ignored


# --- fetching: predicates become query parameters, paging is followed ---------------------------


def test_a_filtered_fetch_returns_exactly_the_expected_rows_across_pages(api: MockApi) -> None:
    """``currency=USD`` for 30 days, three rows per page: 30 rows, 10 requests, no duplicates."""
    adapter, bridge = build(api, page_size=3)
    rows = rows_of(adapter.execute(adapter.fragment("rates", currency="USD"), limits()))

    assert len(rows) == RATE_DAYS
    assert {row["currency"] for row in rows} == {"USD"}
    assert [row["date"] for row in rows] == sorted(row["date"] for row in rows)
    assert len({row["date"] for row in rows}) == RATE_DAYS, "no row may repeat across pages"
    assert rows[0]["date"] == date(2024, 1, 1)
    assert isinstance(rows[0]["date"], date)
    assert len(bridge.seen) == RATE_DAYS // 3, "paging must be followed, not truncated"

    # Ground truth straight from the module's dataset, never from the endpoint under test.
    expected = [row for row in api.rows if row.currency == "USD"]
    assert [row["rate"] for row in rows] == [row.rate for row in expected]


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, RATE_DAYS * len(CURRENCIES)),
        ({"currency": "GBP"}, RATE_DAYS),
        ({"currency": "CHF"}, 0),
        ({"currency": "ILS", "date_from": "2024-01-10", "date_to": "2024-01-12"}, 3),
    ],
)
def test_declared_query_parameters_are_the_filters(
    api: MockApi, params: dict[str, Any], expected: int
) -> None:
    adapter, _ = build(api)
    rows = rows_of(adapter.execute(adapter.fragment("rates", **params), limits(max_rows=200)))
    assert len(rows) == expected


def test_an_undeclared_parameter_is_refused_before_any_request(api: MockApi) -> None:
    """Only declared filters may reach the source: an invented parameter is a spec error."""
    adapter, bridge = build(api)
    with pytest.raises(OpenApiError) as refused:
        adapter.execute(adapter.fragment("rates", evil="; DROP TABLE"), limits())
    assert refused.value.code == "INVALID_SPEC"
    assert "request.params.evil" in refused.value.response.problem
    assert "currency" in refused.value.response.fix
    assert bridge.seen == [], "the request must never be issued"


def test_an_unknown_table_is_refused_naming_the_known_ones(api: MockApi) -> None:
    adapter, _ = build(api)
    with pytest.raises(OpenApiError) as refused:
        adapter.execute(adapter.fragment("nope"), limits())
    assert refused.value.code == "UNKNOWN_TABLE"
    assert "rates" in refused.value.response.fix, "the fix names what may be asked for instead"


def test_the_row_cap_is_enforced_by_the_adapter(api: MockApi) -> None:
    adapter, _ = build(api, page_size=5)
    rows = rows_of(adapter.execute(adapter.fragment("rates"), limits(max_rows=7)))
    assert len(rows) == 7, "the cap is the adapter's promise, not the caller's"


def test_a_timeout_is_raised_as_a_timeout(api: MockApi) -> None:
    """The deadline is checked between pages, so a slow source raises instead of finishing."""
    ticks = iter([0.0] + [float(n) for n in range(1, 1000)])
    bridge = SyncAsgiTransport(api.app)
    adapter = OpenApiAdapter(
        config_for(),
        EnvSecretProvider(environ=SECRET_ENV),
        spec=api.spec(),
        transport=bridge,
        page_size=3,
        monotonic=lambda: next(ticks),
    )
    # timeout_s=0.5 with the clock jumping a whole second per read: the second page is too late.
    with pytest.raises(TimeoutError):
        adapter.execute(adapter.fragment("rates", currency="USD"), limits(timeout_s=0.5))
    assert len(bridge.seen) <= 1, "the deadline stops the fetch, it does not finish it"


def test_cancel_stops_the_stream_and_is_idempotent(api: MockApi) -> None:
    adapter, _ = build(api, page_size=5)
    stream = adapter.execute(adapter.fragment("rates"), limits())
    adapter.cancel(stream)
    adapter.cancel(stream)  # twice must be a no-op, not an error
    assert list(stream) == [], "a cancelled stream yields nothing"
    stream.close()


# --- the cache and single-flight: a repeated answer costs no request ----------------------------


def test_a_second_identical_request_makes_zero_http_calls(api: MockApi) -> None:
    """The acceptance, measured on the source's own counters: ``/_stats`` does not move."""
    adapter, bridge = build(api)
    fragment = adapter.fragment("rates", currency="USD")

    first = rows_of(adapter.execute(fragment, limits()))
    after_first = api.stats().calls.get("/rates", 0)
    second_stream = adapter.execute(fragment, limits())
    second = rows_of(second_stream)
    after_second = api.stats().calls.get("/rates", 0)

    assert after_first == 1
    assert after_second == after_first, "the cached answer must cost no source call"
    assert len(bridge.seen) == after_first
    assert second == first, "a cache hit answers the same question with the same rows"
    assert second_stream.cached is True
    assert second_stream.fetched_at == second_stream.fetched_at  # set, never the wall clock


def test_a_different_filter_is_a_different_cache_entry(api: MockApi) -> None:
    adapter, _ = build(api)
    rows_of(adapter.execute(adapter.fragment("rates", currency="USD"), limits()))
    rows_of(adapter.execute(adapter.fragment("rates", currency="GBP"), limits()))
    assert api.stats().calls.get("/rates", 0) == 2, "one key per distinct question"


def test_a_zero_ttl_disables_the_cache(api: MockApi) -> None:
    bridge = SyncAsgiTransport(api.app)
    adapter = OpenApiAdapter(
        config_for(),
        EnvSecretProvider(environ=SECRET_ENV),
        spec=api.spec(),
        transport=bridge,
        cache_ttl_s=0.0,
    )
    fragment = adapter.fragment("rates", currency="USD")
    rows_of(adapter.execute(fragment, limits()))
    rows_of(adapter.execute(fragment, limits()))
    assert api.stats().calls.get("/rates", 0) == 2


def test_an_expired_entry_is_fetched_again(api: MockApi) -> None:
    """The TTL is read from the injected clock, so this never sleeps."""
    now = [datetime(2024, 6, 1, tzinfo=UTC)]
    bridge = SyncAsgiTransport(api.app)
    adapter = OpenApiAdapter(
        config_for(),
        EnvSecretProvider(environ=SECRET_ENV),
        spec=api.spec(),
        transport=bridge,
        cache_ttl_s=60.0,
        clock=lambda: now[0],
    )
    fragment = adapter.fragment("rates", currency="USD")
    rows_of(adapter.execute(fragment, limits()))
    now[0] = now[0].replace(minute=2)  # two minutes later: inside a 60s TTL would be stale
    rows_of(adapter.execute(fragment, limits()))
    assert api.stats().calls.get("/rates", 0) == 2


# --- read-only: GET only, at the transport layer ------------------------------------------------


def test_every_request_the_adapter_issues_is_a_get(api: MockApi) -> None:
    adapter, bridge = build(api, page_size=7)
    rows_of(adapter.execute(adapter.fragment("rates", currency="USD"), limits()))
    assert bridge.seen, "the fetch really happened"
    assert {method for method, _ in bridge.seen} == {"GET"}


def test_the_transport_refuses_any_other_method() -> None:
    """The guard is the layer that proves read-only, so a future bug cannot leak a write."""
    inner = SyncAsgiTransport(MockApi(MockApiConfig(api_key=KEY)).app)
    guard = GetOnlyTransport(inner)
    with pytest.raises(OpenApiError) as refused:
        guard.handle_request(httpx.Request("POST", f"{BASE_URL}/rates", json={"x": 1}))
    assert refused.value.code == "WRITE_BLOCKED"
    assert inner.seen == [], "the request never reached the app"


def test_a_mutating_operation_in_the_spec_is_unreachable(api: MockApi) -> None:
    """A spec that offers POST/PUT exposes no table for them, and says that it skipped them."""
    document = api.spec()
    document["paths"]["/rates"]["post"] = {"operationId": "create_rate", "responses": {}}
    document["paths"]["/rates"]["put"] = {"operationId": "replace_rate", "responses": {}}
    bridge = SyncAsgiTransport(api.app)
    adapter = OpenApiAdapter(
        config_for(), EnvSecretProvider(environ=SECRET_ENV), spec=document, transport=bridge
    )

    assert [table.name for table in adapter.extract_catalog().tables] == ["rates"]
    assert {"POST /rates", "PUT /rates"} <= set(adapter.ignored_operations)

    # No table, so no fragment can name one.
    with pytest.raises(OpenApiError) as refused:
        adapter.execute(adapter.fragment("POST /rates"), limits())
    assert refused.value.code == "UNKNOWN_TABLE"
    assert bridge.seen == []


# --- error shape: an ErrorResponse, never a driver message -------------------------------------


def test_a_missing_secret_is_source_unavailable_naming_the_reference(api: MockApi) -> None:
    adapter, bridge = build(api, secrets=EnvSecretProvider(environ={}))
    with pytest.raises(OpenApiUnavailable) as refused:
        adapter.execute(adapter.fragment("rates"), limits())

    response = refused.value.response
    assert response.code == "SOURCE_UNAVAILABLE"
    assert SECRET_REF in response.fix, "the fix names the secret reference so it can be fixed"
    assert "MDQM_SECRET_RATES_API_KEY" not in response.fix, "the env var name is not the reference"
    assert bridge.seen == [], "nothing was sent"


def test_an_invalid_key_is_source_unavailable_and_never_leaks_the_value(api: MockApi) -> None:
    wrong = "a06-wrong-key"
    adapter, bridge = build(
        api, secrets=EnvSecretProvider(environ={"MDQM_SECRET_RATES_API_KEY": wrong})
    )
    with pytest.raises(OpenApiUnavailable) as refused:
        adapter.execute(adapter.fragment("rates"), limits())

    response = refused.value.response
    assert response.code == "SOURCE_UNAVAILABLE"
    assert SECRET_REF in response.fix
    assert wrong not in response.model_dump_json(), "the key value must never reach the agent"
    assert "401" not in response.problem, "the taxonomy names the cause, not the status line"


def test_a_spent_budget_is_quota_exceeded_not_a_crash() -> None:
    api = MockApi(MockApiConfig(api_key=KEY, rate_limit_budget=1, rate_limit_window_s=60.0))
    adapter, _ = build(api, page_size=3)
    with pytest.raises(OpenApiError) as refused:
        # Two pages: the second request is refused with 429.
        rows_of(adapter.execute(adapter.fragment("rates", currency="USD"), limits()))
    response = refused.value.response
    assert response.code == "QUOTA_EXCEEDED"
    assert "cache" in response.fix, "the fix points at what actually helps"


class _NoTime:
    """A virtual clock: acquiring never really sleeps, so a denial is instant."""

    def __init__(self) -> None:
        self._now = 0.0

    def now(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now += max(seconds, 0.0)


def test_a_denied_rate_limit_is_quota_exceeded_and_sends_nothing(api: MockApi) -> None:
    limiter = RateLimiter(
        {"shared": LimitPolicy(rps=1.0, burst=1, max_wait_s=0.01)},
        {"rates_api": ("shared",)},
        time_source=_NoTime(),
    )
    adapter, bridge = build(api, limiter=limiter)
    limiter.acquire("rates_api")  # spend the only token (virtually, no sleep)
    with pytest.raises(OpenApiError) as refused:
        adapter.execute(adapter.fragment("rates"), limits())
    assert refused.value.code == "QUOTA_EXCEEDED"
    assert bridge.seen == [], "a refusal must not spend a source request"


# --- the spec reader ----------------------------------------------------------------------------


def _toy_spec(parameters: list[dict[str, Any]]) -> dict[str, Any]:
    """A minimal document with one collection, for the paging styles the mock does not offer."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "toy", "version": "9.9"},
        "paths": {
            "/things": {
                "get": {
                    "parameters": parameters,
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "things": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {"id": {"type": "integer"}},
                                                },
                                            }
                                        },
                                    }
                                }
                            }
                        }
                    },
                }
            }
        },
    }


@pytest.mark.parametrize(
    ("parameters", "expected_style", "expected_param"),
    [
        ([{"name": "cursor", "in": "query"}], "cursor", "cursor"),
        ([{"name": "page_token", "in": "query"}], "cursor", "page_token"),
        ([{"name": "offset", "in": "query"}], "offset", "offset"),
        ([{"name": "page", "in": "query"}], "page", "page"),
        ([{"name": "limit", "in": "query"}], "none", None),
    ],
)
def test_the_paging_style_is_detected_from_the_parameters(
    parameters: list[dict[str, Any]], expected_style: str, expected_param: str | None
) -> None:
    """Cursor, offset and page are recognised, and a plain filter is not mistaken for paging."""
    endpoints, _ = read_endpoints(_toy_spec(parameters))
    assert len(endpoints) == 1
    assert endpoints[0].paging.style == expected_style
    assert endpoints[0].paging.param == expected_param


def test_a_spec_without_a_base_url_is_refused_with_a_fix() -> None:
    api = MockApi(MockApiConfig(api_key=KEY))
    with pytest.raises(ValueError, match="base URL"):
        OpenApiAdapter(
            SourceConfig(name="rates_api", adapter="openapi"),
            EnvSecretProvider(environ=SECRET_ENV),
            spec=api.spec(),
            transport=SyncAsgiTransport(api.app),
        )


def test_the_catalog_reads_a_document_from_a_file(tmp_path: Any) -> None:
    """A deployed source names a spec file; the same reader handles YAML and JSON."""
    from mdqm.testing.mock_api import spec_yaml

    api = MockApi(MockApiConfig(api_key=KEY))
    path = tmp_path / "api.yaml"
    path.write_text(spec_yaml(api.spec()), encoding="utf-8", newline="\n")

    adapter = OpenApiAdapter(
        SourceConfig(
            name="rates_api", adapter="openapi", spec=str(path), options={"base_url": BASE_URL}
        ),
        EnvSecretProvider(environ=SECRET_ENV),
        transport=SyncAsgiTransport(api.app),
    )
    assert [table.name for table in adapter.extract_catalog().tables] == ["rates"]
