"""A local, spec-driven fake internal HTTP API for the OpenAPI adapter (task D04).

The adapter lane needs something *real* to talk to: a served OpenAPI document plus endpoints that
behave like an internal API — an API-key header, query filters, cursor paging, a documented error
body, rate-limit headers with a ``429``, and a slow endpoint for timeout tests. Nothing here ever
contacts the internet: it is a Starlette app, so a test drives it **in process** through
``httpx.ASGITransport`` (no port, no socket, works under ``pytest-socket``), and
:meth:`MockApi.serve` starts the same app on a real *loopback* port for the cases that need a
socket, or for the ``api`` compose profile.

Shapes it deliberately imitates (``docs/DESIGN.md`` "D4 OpenAPI" discovers exactly these):

* ``GET`` collections as virtual tables: ``/rates`` with ``currency``/``date_from``/``date_to``
  filters and ``limit``/``cursor`` paging, so a filter is an "index" the planner can use;
* one resource endpoint with a path parameter: ``/customers/{customer_id}/score``;
* the key in a header, declared in the spec as an ``apiKey`` security scheme;
* ``X-RateLimit-Limit``/``-Remaining``/``-Reset`` on every protected answer, ``Retry-After`` on
  ``429`` (the shape :func:`mdqm.ratelimit.parse_retry_after` reads);
* one error envelope for every failure: ``{"error": {"code", "message", "details"}}``;
* ``/_stats`` for per-endpoint call counts, which is how a test proves that cache/coalescing
  removed the source calls rather than merely made them faster.

The API key is never written in a shipped file: :class:`MockApiConfig` takes it, and the container
entry point reads ``MDQM_MOCK_API_KEY``. An empty key (the default) authenticates *nothing*, so an
unconfigured mock fails closed instead of answering.

This module imports ``fastapi``, a dev-only extra. ``mdqm.testing`` is never imported by the server
path, and the container/real-port paths import ``uvicorn`` inside the function that needs it.

Usage::

    api = MockApi(MockApiConfig(api_key="dev-only-key"))

    async with api.client() as client:               # in process, no socket
        page = await client.get("/rates", params={"currency": "USD"})

    with api.serve() as base_url:                    # real loopback, ephemeral port
        httpx.get(f"{base_url}/health")

    python -m mdqm.testing.mock_api --write-spec tests/data/specs/fx.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import io
import math
import os
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Final

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Security
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field
from ruamel.yaml import YAML
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

__all__ = [
    "AUTH_HEADER",
    "MockApi",
    "MockApiConfig",
    "create_app",
    "main",
    "rate_rows",
    "spec_yaml",
]

#: The header the fake API authenticates with. One name, used by the guard, the spec and the tests.
AUTH_HEADER: Final = "X-API-Key"

#: Paths that are open on purpose: a real internal API liveness probe, the spec itself, and the
#: test-only counter endpoint. They also do not spend the rate-limit budget: counting a counter
#: would make a test's own bookkeeping change what it measures.
PUBLIC_PATHS: Final[frozenset[str]] = frozenset(
    {"/health", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc", "/_stats"}
)

#: Deterministic fake FX rows. Real organisations' numbers live in config, never in code; these
#: are invented values with no claim to reality, only to being the same on every machine.
CURRENCIES: Final[tuple[str, ...]] = ("EUR", "GBP", "ILS", "USD")
FIRST_DAY: Final = date(2024, 1, 1)
RATE_DAYS: Final = 30
CUSTOMER_IDS: Final[tuple[str, ...]] = tuple(f"C-{n:04d}" for n in range(1, 21))

_MICRO: Final = Decimal("0.000001")
_CURSOR_RE: Final = r"^c[0-9]+$"

#: ``date`` as a *type*, for a field that is also called ``date``: pydantic refuses a field whose
#: name is the type's own name, and the wire name is part of the fake API's contract.
DateOnly = date


class ErrorCode(StrEnum):
    """The ``error.code`` values this API can answer with."""

    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limited"
    INVALID_REQUEST = "invalid_request"
    NOT_FOUND = "not_found"
    INTERNAL = "internal"


#: Status -> code, so the error envelope and the spec cannot disagree about a status.
_CODE_BY_STATUS: Final[Mapping[int, ErrorCode]] = {
    400: ErrorCode.INVALID_REQUEST,
    401: ErrorCode.UNAUTHORIZED,
    404: ErrorCode.NOT_FOUND,
    429: ErrorCode.RATE_LIMITED,
}


# --- wire models -------------------------------------------------------------------------------


class _Model(BaseModel):
    """A response/request body of the *fake external* API."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ErrorDetail(_Model):
    code: str = Field(description="Stable machine-readable code, e.g. 'rate_limited'.")
    message: str = Field(description="Human-readable reason, safe to show to an operator.")
    details: dict[str, str] = Field(default_factory=dict, description="Extra key/value context.")


class ErrorBody(_Model):
    """The single error envelope: every non-2xx answer has this body."""

    error: ErrorDetail


class RateRow(_Model):
    currency: str = Field(description="ISO-4217 code, upper case.")
    date: DateOnly = Field(description="The day the rate applies to.")
    rate: float = Field(description="Units of the currency per one EUR.")
    source: str = Field(description="Which feed published the rate.")


class RatePage(_Model):
    items: list[RateRow] = Field(description="This page's rows.")
    next_cursor: str | None = Field(
        default=None, description="Pass back as `cursor` for the next page; null on the last page."
    )
    total: int = Field(description="Rows matching the filters, across all pages.")


class CustomerScore(_Model):
    customer_id: str = Field(description="The id from the path.")
    score: int = Field(description="Credit score, 0-100.")
    band: str = Field(description="'low', 'medium' or 'high'.")


class Health(_Model):
    status: str = Field(description="'ok' when the service answers.")
    version: str = Field(description="Service version, from the app's OpenAPI info.")


class SlowAnswer(_Model):
    requested_ms: int = Field(description="The delay the caller asked for.")
    slept_ms: int = Field(description="The delay actually slept, after capping.")
    capped: bool = Field(description="True when the server's cap shortened the delay.")


class Stats(_Model):
    """Per-endpoint call counts: the evidence that cache/coalescing avoided source calls."""

    calls: dict[str, int] = Field(description="Endpoint path -> successful handle count.")
    total_calls: int = Field(description="Sum of `calls`.")
    rate_limited: int = Field(description="Requests refused with 429.")
    unauthorized: int = Field(description="Requests refused with 401.")
    budget_limit: int = Field(description="Requests allowed per window, per key.")
    budget_remaining: int = Field(description="Requests left in the current window for the key.")


# --- configuration and state -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MockApiConfig:
    """How the fake API behaves. Everything a test wants to vary is here, nothing is a global.

    ``api_key`` has no "dev default" on purpose: an empty key authenticates nothing, so a mock the
    operator forgot to configure fails closed. Tests pass their own throwaway literal.
    """

    api_key: str = ""
    rate_limit_budget: int = 1000
    rate_limit_window_s: float = 60.0
    max_delay_ms: int = 2000

    def __post_init__(self) -> None:
        if self.rate_limit_budget < 1:
            raise ValueError("rate_limit_budget must be >= 1")
        if self.rate_limit_window_s <= 0:
            raise ValueError("rate_limit_window_s must be > 0")
        if self.max_delay_ms < 0:
            raise ValueError("max_delay_ms must be >= 0")


@dataclass(slots=True)
class _Budget:
    """One key's fixed window: ``limit`` requests per ``window_s`` seconds.

    Fixed window, not a token bucket, because the point is the *response shape*: a real API answers
    ``429`` with ``Retry-After`` when a window's budget is gone, and the header must be a whole
    number of seconds the client waits. Time is passed in, so tests never sleep.
    """

    limit: int
    window_s: float
    window_start: float = 0.0
    used: int = 0

    def take(self, now: float) -> tuple[bool, int, float]:
        """Spend one request. -> ``(allowed, remaining, retry_after_s)``."""
        if now - self.window_start >= self.window_s:
            self.window_start = now
            self.used = 0
        if self.used >= self.limit:
            return False, 0, max(self.window_start + self.window_s - now, 0.0)
        self.used += 1
        return True, self.limit - self.used, 0.0

    def remaining(self, now: float) -> int:
        """Requests left in the current window, without spending one."""
        if now - self.window_start >= self.window_s:
            return self.limit
        return max(self.limit - self.used, 0)


@dataclass(slots=True)
class _Counters:
    calls: Counter[str] = field(default_factory=Counter)
    rate_limited: int = 0
    unauthorized: int = 0

    def reset(self) -> None:
        self.calls.clear()
        self.rate_limited = 0
        self.unauthorized = 0


def rate_rows() -> tuple[RateRow, ...]:
    """The whole fake dataset: every currency for every day, sorted by (currency, date).

    Deterministic arithmetic on ``Decimal``, so the same rows appear on every OS and every run, and
    the numbers are exact rather than re-printed floats.
    """
    rows: list[RateRow] = []
    for currency_index, currency in enumerate(CURRENCIES):
        for day_index in range(RATE_DAYS):
            amount = Decimal(1) + Decimal(day_index) / Decimal(7) + Decimal(currency_index) / 4
            rows.append(
                RateRow(
                    currency=currency,
                    date=date.fromordinal(FIRST_DAY.toordinal() + day_index),
                    rate=float(amount.quantize(_MICRO)),
                    source="mock-feed",
                )
            )
    return tuple(rows)


# --- the service -------------------------------------------------------------------------------


class MockApi:
    """The fake API: one ASGI app plus the counters and budgets a test inspects.

    ``clock`` is injected so a rate-limit window can be crossed without waiting: pass a callable
    that returns monotonic seconds.
    """

    def __init__(
        self,
        config: MockApiConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config if config is not None else MockApiConfig()
        self._clock = clock
        self._lock = threading.Lock()
        self._budgets: dict[str, _Budget] = {}
        self._counters = _Counters()
        self.rows = rate_rows()
        self.app = _build_app(self)
        self.app.state.mock_api = self

    # -- what a test asks ---------------------------------------------------------------

    def spec(self) -> dict[str, Any]:
        """The live OpenAPI document (what ``/openapi.json`` serves)."""
        return self.app.openapi()

    def spec_yaml(self) -> str:
        """The live spec as the checked-in YAML text; ``--write-spec`` rewrites the copy."""
        return spec_yaml(self.spec())

    def stats(self) -> Stats:
        """A snapshot of the counters and the remaining budget."""
        with self._lock:
            calls = dict(sorted(self._counters.calls.items()))
            total = sum(calls.values())
            limited = self._counters.rate_limited
            unauthorized = self._counters.unauthorized
            remaining = self._budget_for(self.config.api_key).remaining(self._clock())
        return Stats(
            calls=calls,
            total_calls=total,
            rate_limited=limited,
            unauthorized=unauthorized,
            budget_limit=self.config.rate_limit_budget,
            budget_remaining=remaining,
        )

    def reset(self) -> None:
        """Forget the counters and the budgets (a fresh window); the data does not change."""
        with self._lock:
            self._counters.reset()
            self._budgets.clear()

    def client(self, **kwargs: Any) -> httpx.AsyncClient:
        """An async client bound to the in-process app: no port, no socket, no network."""
        kwargs.setdefault("base_url", "http://mock-api.invalid")
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), **kwargs)

    @contextmanager
    def serve(self, host: str = "127.0.0.1", port: int = 0) -> Iterator[str]:
        """Run the app on a real port (loopback, ephemeral by default) and yield its base URL.

        Only for the cases that need a socket: the compose ``api`` profile, and the one test that
        proves the same app answers over TCP. Tests that can use :meth:`client` should, because a
        loopback socket has to be opted into in ``tests/unit``.
        """
        import uvicorn

        config = uvicorn.Config(self.app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, name="mdqm-mock-api", daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 30.0
            while not server.started:
                if time.monotonic() > deadline:  # pragma: no cover - only on a wedged box
                    raise RuntimeError("the mock API did not start within 30s")
                time.sleep(0.01)
            bound = server.servers[0].sockets[0].getsockname()[1]
            yield f"http://{host}:{bound}"
        finally:
            server.should_exit = True
            thread.join(timeout=10)

    # -- used by the guard and the handlers ---------------------------------------------

    def record(self, endpoint: str) -> None:
        """Count one handled request: a refused request is not counted."""
        with self._lock:
            self._counters.calls[endpoint] += 1

    def count_refusal(self, status: int) -> None:
        """Count a refused request: 401 or 429, the two the guard produces."""
        with self._lock:
            if status == 401:
                self._counters.unauthorized += 1
            elif status == 429:
                self._counters.rate_limited += 1

    def admit(self, presented: str | None) -> tuple[int, int, float, float]:
        """Authorise and meter one request.

        -> ``(status, remaining, retry_after_s, reset_s)`` where ``status`` is 200 when admitted,
        401 for a missing/wrong key and 429 when the key's budget is spent. The presented key is
        compared in constant time and is never echoed anywhere.
        """
        expected = self.config.api_key
        if not expected or not presented or not hmac.compare_digest(presented, expected):
            self.count_refusal(401)
            return 401, 0, 0.0, 0.0
        now = self._clock()
        with self._lock:
            budget = self._budget_for(expected)
            allowed, remaining, retry_after = budget.take(now)
            reset_s = max(budget.window_start + budget.window_s - now, 0.0)
        if not allowed:
            self.count_refusal(429)
            return 429, 0, retry_after, reset_s
        return 200, remaining, 0.0, reset_s

    def _budget_for(self, key: str) -> _Budget:
        """The budget of ``key``, created on first use. Caller holds ``self._lock``."""
        budget = self._budgets.get(key)
        if budget is None:
            budget = _Budget(
                limit=self.config.rate_limit_budget, window_s=self.config.rate_limit_window_s
            )
            self._budgets[key] = budget
        return budget


def create_app(config: MockApiConfig | None = None) -> FastAPI:
    """The ASGI app alone, for the container (``uvicorn mdqm.testing.mock_api:create_app``)."""
    return MockApi(config).app


# --- the guard ---------------------------------------------------------------------------------


def _error_response(status: int, code: ErrorCode, message: str, **details: str) -> JSONResponse:
    """The one error envelope. Never includes the offending input: it could be a key."""
    body = ErrorBody(error=ErrorDetail(code=code.value, message=message, details=details))
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


class _Guard(BaseHTTPMiddleware):
    """Auth + rate-limit headers, in front of the router, like a real API gateway.

    The alternative — a dependency per route — would run *after* request validation, so a malformed
    request would leak past the key check and a refused request would already have cost parsing.
    """

    def __init__(self, app: ASGIApp, api: MockApi) -> None:
        super().__init__(app)
        self.api = api

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        status, remaining, retry_after, reset_s = self.api.admit(request.headers.get(AUTH_HEADER))
        if status == 401:
            denied = _error_response(
                401,
                ErrorCode.UNAUTHORIZED,
                f"missing or wrong {AUTH_HEADER} header",
                header=AUTH_HEADER,
            )
            denied.headers["X-RateLimit-Remaining"] = "0"
            return denied
        if status == 429:
            wait_s = str(max(math.ceil(retry_after), 1))
            denied = _error_response(
                429,
                ErrorCode.RATE_LIMITED,
                f"request budget spent: retry after {wait_s}s",
                retry_after_s=wait_s,
            )
            denied.headers["Retry-After"] = wait_s
            denied.headers["X-RateLimit-Limit"] = str(self.api.config.rate_limit_budget)
            denied.headers["X-RateLimit-Remaining"] = "0"
            denied.headers["X-RateLimit-Reset"] = str(max(math.ceil(reset_s), 1))
            return denied

        returned = await call_next(request)
        returned.headers["X-RateLimit-Limit"] = str(self.api.config.rate_limit_budget)
        returned.headers["X-RateLimit-Remaining"] = str(remaining)
        returned.headers["X-RateLimit-Reset"] = str(max(math.ceil(reset_s), 1))
        return returned


# --- the app -----------------------------------------------------------------------------------

_api_key_scheme = APIKeyHeader(
    name=AUTH_HEADER,
    scheme_name="ApiKeyAuth",
    auto_error=False,
    description="The API key issued to the server (a secret reference in mdqm config).",
)


async def _declared_auth(key: Annotated[str | None, Security(_api_key_scheme)] = None) -> None:
    """No runtime work: :class:`_Guard` already authorised the request.

    Declared as a dependency so FastAPI records the ``apiKey`` scheme and marks the operation as
    protected in the spec — that is what the adapter's D4 stage reads to learn the auth shape.
    """


class _MockFastAPI(FastAPI):
    """FastAPI whose spec carries the shared error envelope and the rate-limit headers.

    They are added to the generated document instead of each route, so the header set is declared
    once and cannot drift between endpoints.
    """

    def openapi(self) -> dict[str, Any]:
        return _decorate(super().openapi())


_RATE_LIMIT_HEADERS: Final[dict[str, Any]] = {
    "X-RateLimit-Limit": {
        "description": "Requests allowed per window, per key.",
        "schema": {"type": "integer"},
    },
    "X-RateLimit-Remaining": {
        "description": "Requests left in the current window.",
        "schema": {"type": "integer"},
    },
    "X-RateLimit-Reset": {
        "description": "Seconds until the window resets.",
        "schema": {"type": "integer"},
    },
}

_ERROR_REF: Final[dict[str, str]] = {"$ref": "#/components/schemas/ErrorBody"}


def _decorate(schema: dict[str, Any]) -> dict[str, Any]:
    """Fold the envelope, the protected-route responses and the rate-limit headers into ``schema``.

    Idempotent (fixed assignments and ``setdefault``) so FastAPI's cached document can be decorated
    on every read. The auto-generated ``422`` is dropped: this API answers a bad parameter with
    ``400`` and its own envelope, and a spec that promised ``422`` would be wrong.
    """
    components = schema.setdefault("components", {})
    schemes = components.setdefault("securitySchemes", {})
    schemes.setdefault(
        "ApiKeyAuth",
        {
            "type": "apiKey",
            "in": "header",
            "name": AUTH_HEADER,
            "description": "The API key issued to the server (a secret reference in mdqm config).",
        },
    )
    for path, operations in schema.get("paths", {}).items():
        if path in PUBLIC_PATHS:
            continue
        for operation in operations.values():
            if not isinstance(operation, dict) or "responses" not in operation:
                continue
            responses = operation["responses"]
            responses.pop("422", None)
            success = next((r for status, r in responses.items() if str(status)[0] == "2"), None)
            if isinstance(success, dict):
                headers = success.setdefault("headers", {})
                for name, description in _RATE_LIMIT_HEADERS.items():
                    headers[name] = dict(description)
            responses["401"] = {
                "description": f"Missing or wrong {AUTH_HEADER} header.",
                "content": {"application/json": {"schema": _ERROR_REF}},
            }
            responses["429"] = {
                "description": "The request budget for this key is spent.",
                "headers": {
                    "Retry-After": {
                        "description": "Seconds to wait.",
                        "schema": {"type": "integer"},
                    }
                },
                "content": {"application/json": {"schema": _ERROR_REF}},
            }
    # The two auto-generated validation schemas are unreferenced once 422 is gone.
    for unname in ("HTTPValidationError", "ValidationError"):
        components.get("schemas", {}).pop(unname, None)
    return schema


def _cursor_offset(cursor: str) -> int:
    """Decode the opaque cursor. The route ``pattern`` refused anything else with a 400 already."""
    return int(cursor[1:])


class _BadRequest(Exception):
    """A 400 the handler raises itself (not a validation error, so it can explain the fix)."""

    def __init__(self, message: str, *, fix: str) -> None:
        self.message = message
        self.fix = fix
        super().__init__(message)


def _build_app(api: MockApi) -> FastAPI:
    """Build the app: guard, handlers and the five documented endpoints."""
    app = _MockFastAPI(
        title="MDQM mock internal API",
        version="1.0.0",
        description=(
            "A local stand-in for an internal API with an OpenAPI spec: GET-only collections with "
            "filters and cursor paging, an API key header, rate-limit headers, one error envelope."
        ),
    )
    app.add_middleware(_Guard, api=api)

    async def http_error(request: Request, exc: Exception) -> Response:
        """Every HTTP error leaves in the documented envelope."""
        if isinstance(exc, StarletteHTTPException):
            status = exc.status_code
            code = _CODE_BY_STATUS.get(status, ErrorCode.INVALID_REQUEST)
            return _error_response(status, code, str(exc.detail))
        return _error_response(500, ErrorCode.INTERNAL, "internal error")

    async def bad_request(request: Request, exc: Exception) -> Response:
        if isinstance(exc, _BadRequest):
            return _error_response(400, ErrorCode.INVALID_REQUEST, exc.message, fix=exc.fix)
        return _error_response(500, ErrorCode.INTERNAL, "internal error")

    async def validation_error(request: Request, exc: Exception) -> Response:
        """Parameter problems are a 400 in this API's own envelope, not FastAPI's 422.

        Only the location and the reason are reported: FastAPI also records the offending input,
        and that could be a key-shaped string which must never be echoed.
        """
        if not isinstance(exc, RequestValidationError):
            return _error_response(500, ErrorCode.INTERNAL, "internal error")
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        return _error_response(
            400,
            ErrorCode.INVALID_REQUEST,
            problems or "invalid request",
            fix="check the parameter types",
        )

    app.add_exception_handler(StarletteHTTPException, http_error)
    app.add_exception_handler(_BadRequest, bad_request)
    app.add_exception_handler(RequestValidationError, validation_error)

    @app.get("/health", response_model=Health, summary="Liveness", tags=["meta"])
    async def health() -> Health:
        """Open on purpose and free of charge: a liveness probe must not spend the budget."""
        return Health(status="ok", version=app.version)

    @app.get(
        "/rates",
        response_model=RatePage,
        summary="Daily FX rates",
        tags=["rates"],
        dependencies=[Security(_declared_auth)],
    )
    async def rates(
        currency: Annotated[
            str | None, Query(description="ISO-4217 code, e.g. 'USD'. Case-insensitive.")
        ] = None,
        date_from: Annotated[
            date | None, Query(description="Inclusive lower bound on `date`.")
        ] = None,
        date_to: Annotated[
            date | None, Query(description="Inclusive upper bound on `date`.")
        ] = None,
        limit: Annotated[int, Query(ge=1, le=100, description="Rows per page.")] = 10,
        cursor: Annotated[
            str | None, Query(pattern=_CURSOR_RE, description="`next_cursor` of the previous page.")
        ] = None,
    ) -> RatePage:
        """Rows matching the filters, one page at a time: a filter is an index in the real API."""
        wanted = currency.upper() if currency else None
        if date_from is not None and date_to is not None and date_from > date_to:
            raise _BadRequest(
                "date_from is after date_to", fix="swap the bounds: date_from <= date_to"
            )
        rows = [
            row
            for row in api.rows
            if (wanted is None or row.currency == wanted)
            and (date_from is None or row.date >= date_from)
            and (date_to is None or row.date <= date_to)
        ]
        offset = _cursor_offset(cursor) if cursor else 0
        page = rows[offset : offset + limit]
        following = offset + limit
        api.record("/rates")
        return RatePage(
            items=page,
            next_cursor=f"c{following}" if following < len(rows) else None,
            total=len(rows),
        )

    @app.get(
        "/customers/{customer_id}/score",
        response_model=CustomerScore,
        summary="Credit score of one customer",
        tags=["customers"],
        dependencies=[Security(_declared_auth)],
    )
    async def customer_score(
        customer_id: Annotated[
            str, PathParam(pattern=r"^C-[0-9]{4}$", description="Customer id, e.g. 'C-0007'.")
        ],
    ) -> CustomerScore:
        """One resource behind a path parameter, so the adapter has a non-collection shape too."""
        if customer_id not in CUSTOMER_IDS:
            raise HTTPException(status_code=404, detail=f"no customer {customer_id}")
        number = int(customer_id[2:])
        score = (number * 37) % 101
        band = "low" if score < 34 else ("medium" if score < 67 else "high")
        api.record("/customers/{customer_id}/score")
        return CustomerScore(customer_id=customer_id, score=score, band=band)

    @app.get(
        "/slow",
        response_model=SlowAnswer,
        summary="Answers after a delay",
        tags=["meta"],
        dependencies=[Security(_declared_auth)],
    )
    async def slow(
        delay_ms: Annotated[
            int, Query(ge=0, description="How long to wait before answering.")
        ] = 100,
    ) -> SlowAnswer:
        """For client-timeout tests; the delay is capped by ``MockApiConfig.max_delay_ms``."""
        slept = min(delay_ms, api.config.max_delay_ms)
        if slept:
            await asyncio.sleep(slept / 1000)
        api.record("/slow")
        return SlowAnswer(requested_ms=delay_ms, slept_ms=slept, capped=slept != delay_ms)

    @app.get("/_stats", response_model=Stats, summary="Per-endpoint call counts", tags=["meta"])
    async def stats() -> Stats:
        """Test-only: how many times each endpoint was actually handled."""
        return api.stats()

    return app


# --- the checked-in spec -----------------------------------------------------------------------


def spec_yaml(spec: Mapping[str, Any]) -> str:
    """Render an OpenAPI document as the deterministically formatted YAML we check in.

    Round-trip dumper so key order survives (a diff then shows a real change, not a reshuffle), a
    wide width so long descriptions stay on one line, and a trailing newline (LF, this repo's rule).
    """
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096
    stream = io.StringIO()
    yaml.dump(dict(spec), stream)
    return stream.getvalue()


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m mdqm.testing.mock_api``: rewrite the spec, or serve on a real port."""
    parser = argparse.ArgumentParser(
        prog="python -m mdqm.testing.mock_api",
        description="Mock internal API for the OpenAPI adapter (task D04).",
    )
    parser.add_argument(
        "--write-spec",
        type=Path,
        metavar="PATH",
        help="write the live OpenAPI document to PATH as YAML, then exit (regenerates the copy)",
    )
    parser.add_argument("--serve", action="store_true", help="listen on a real port until stopped")
    parser.add_argument("--host", default=os.environ.get("MDQM_MOCK_API_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("MDQM_MOCK_API_PORT", "8000"))
    )
    args = parser.parse_args(argv)

    config = MockApiConfig(api_key=os.environ.get("MDQM_MOCK_API_KEY", ""))
    api = MockApi(config)
    if args.write_spec is not None:
        path: Path = args.write_spec
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(api.spec_yaml(), encoding="utf-8", newline="\n")
        return 0
    if not args.serve:
        parser.error("nothing to do: pass --write-spec PATH or --serve")
    import uvicorn

    uvicorn.run(api.app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
