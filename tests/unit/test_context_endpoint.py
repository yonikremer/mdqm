"""M06: ``POST /context`` -- the Context Pack the client hook injects, authenticated and budgeted.

Driven **in process** through ``httpx.ASGITransport`` over the real Starlette app the SDK builds
(``MCPServer.streamable_http_app``), so no port is bound and no request leaves the box. The loop
opt-in below is Windows' event loop, not a network path (see the ``pytestmark`` comment). The route,
the bearer check and the Context Pack are the same code the plugin's hook meets on the wire.

The promises proved here, in the order they matter:

* **It is authenticated like the tools** -- a missing or unknown token is ``401`` and reaches no
  handler.
* **The answer is the pack, or empty.** A data prompt (English and Hebrew) brings the semantic
  facts; a weather prompt and an empty prompt bring ``""``, which is a normal answer, not a 500.
* **It leaks nothing.** No token, no second field, no echo of the prompt, nothing but semantic text.
* **It is deterministic and inside the budget**, on every call, because the hook runs on every
  prompt and a reviewer must be able to diff what the agent was told.
* **It is fast**: a thousand sequential hook calls stay inside the 150 ms p95 target. The measured
  number is printed by the test run and recorded in ``progress/M06.md``.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_socket

from mdqm.config.settings import PathsSettings, Settings
from mdqm.context.index import ApprovedExample
from mdqm.context.pack import DEFAULT_PACK_BUDGET
from mdqm.results.summary import estimate_tokens
from mdqm.semantic import SemanticLayer
from mdqm.server.app import create_server
from mdqm.server.auth import StaticTokenVerifier, TokenEntry
from mdqm.server.context_endpoint import (
    MAX_PROMPT_CHARS,
    ContextPackSource,
    context_source_from_settings,
)
from mdqm.store import Store

TOKENS = (
    TokenEntry(token="tok-alice-secret-value", user_id="alice", display_name="Alice"),
    TokenEntry(token="tok-bob-secret-value", user_id="bob", display_name="Bob"),
)
ALICE = TOKENS[0].token

#: Windows builds the asyncio event loop from a loopback socket pair, so merely *creating* a loop
#: needs ``socket.socket`` and a connection to ``127.0.0.1``. Every test here is async, so the
#: marker is module-wide; the autouse fixture below narrows the permission to loopback. The
#: requests themselves go through ``http://mdqm.test`` in process and never open a socket.
pytestmark = pytest.mark.enable_socket


@pytest.fixture(autouse=True)
def loopback_only() -> Iterator[None]:
    """``127.0.0.1`` and nothing else: the loop's self-pipe, never a source or the internet."""
    pytest_socket.socket_allow_hosts(["127.0.0.1"])
    yield
    pytest_socket.socket_allow_hosts(["127.0.0.1"])

#: A layer with one measure stored three ways, a Hebrew-labelled status value map, one partitioned
#: event table and one approved join: enough that a real question has something to bring back.
SEMANTIC: dict[str, Any] = {
    "version": 3,
    "sources": {"crm": {"adapter": "postgresql"}, "billing": {"adapter": "postgresql"}},
    "value_maps": {
        "customer_status": {
            "canonical": ["active", "churned"],
            "synonyms": {"active": ["פעיל"], "churned": ["עזב"]},
            "sources": {"crm.customers.status": {"A": "active", "C": "churned"}},
        }
    },
    "measures": {
        "credit_limit_r5000": {
            "canonical": {"rounding": {"to": 5000, "mode": "half_up"}},
            "sources": {
                "crm.customers.credit_limit": {"normalize": "{col}"},
                "billing.accounts.credit_limit": {"normalize": "ROUND({col} / 5000) * 5000"},
            },
        }
    },
    "tables": {
        "billing.usage_events": {
            "key": ["event_id"],
            "partitioning": {"key": "start_time", "granularity": "month"},
            "time_columns": {
                "start_time": {"role": "event_time", "indexed": True, "use_by_default": True},
            },
        }
    },
    "approved_joins": [
        {
            "name": "usage_to_customers",
            "left": "billing.usage_events.client_no",
            "right": "crm.customers.cust_id",
            "on": "CAST({right} AS VARCHAR) = {left}",
            "cardinality": "many_to_one",
        }
    ],
}

LABELS: dict[str, tuple[str, ...]] = {
    "measure:credit_limit_r5000": ("credit limit", "מסגרת אשראי"),
    "table:crm.customers": ("customers", "לקוחות"),
}

EXAMPLES = (ApprovedExample(name="credit_limit_by_status", question="credit limit by status"),)

#: A question with real semantic hits, in each language the organisation asks in.
DATA_PROMPT = "average credit limit by customer status"
HEBREW_PROMPT = "מסגרת אשראי של לקוחות"
#: A question the layer knows nothing about: the hook must inject zero tokens.
OFF_TOPIC_PROMPT = "what is the weather in Tel Aviv"


def layer() -> SemanticLayer:
    """The test layer."""
    return SemanticLayer.model_validate(SEMANTIC)


@pytest.fixture()
def store(tmp_path: Path) -> Iterator[Store]:
    """A migrated metadata DB on a throwaway file (SQLite runs in process, no Docker)."""
    opened = Store(tmp_path / "meta.sqlite")
    opened.initialize()
    try:
        yield opened
    finally:
        opened.close()


@pytest.fixture()
def source(store: Store) -> ContextPackSource:
    """The indexed layer the endpoint answers from."""
    return ContextPackSource.build(
        layer(), store=store, labels=LABELS, examples=EXAMPLES, budget_tokens=DEFAULT_PACK_BUDGET
    )


@pytest.fixture()
async def client(source: ContextPackSource) -> AsyncIterator[httpx.AsyncClient]:
    """An in-process client over the app the server really serves (no socket)."""
    app = create_server(verifier=StaticTokenVerifier(TOKENS), context_source=source)
    transport = httpx.ASGITransport(app=app.streamable_http_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://mdqm.test") as http:
        yield http


@pytest.fixture()
async def client_without_source() -> AsyncIterator[httpx.AsyncClient]:
    """An in-process client on a server built with no semantic layer (a fresh installation)."""
    app = create_server(verifier=StaticTokenVerifier(TOKENS))
    transport = httpx.ASGITransport(app=app.streamable_http_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://mdqm.test") as http:
        yield http


def auth(token: str = ALICE) -> dict[str, str]:
    """The bearer header the plugin's hook sends."""
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------------------------------
# Authentication: like the tools, and before any work
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token",
    [None, "", "not-a-real-token", "tok-alice-secret-value-x"],
    ids=["missing", "empty", "unknown", "prefix-of-a-real-one"],
)
async def test_a_bad_token_is_401_and_reaches_no_handler(
    client: httpx.AsyncClient, token: str | None
) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = await client.post("/context", json={"prompt": DATA_PROMPT}, headers=headers)

    assert response.status_code == 401
    assert response.json()["error"] == "invalid_token"
    assert response.headers["www-authenticate"].startswith("Bearer")
    # Nothing was built, so nothing about the layer can be read out of a refusal.
    assert "additionalContext" not in response.text


async def test_a_malformed_authorization_header_is_401(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/context", json={"prompt": DATA_PROMPT}, headers={"Authorization": ALICE}
    )

    assert response.status_code == 401


@pytest.mark.parametrize("token", [TOKENS[0].token, TOKENS[1].token], ids=["alice", "bob"])
async def test_both_users_get_the_same_layer_context(
    client: httpx.AsyncClient, token: str
) -> None:
    # The block is semantic facts, not a person's data: who asks cannot change it.
    response = await client.post("/context", json={"prompt": DATA_PROMPT}, headers=auth(token))

    assert response.status_code == 200
    assert response.json()["additionalContext"]


# --------------------------------------------------------------------------------------------------
# The pack, or nothing
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("prompt", [DATA_PROMPT, HEBREW_PROMPT, "credit limit", "לקוחות"], ids=str)
async def test_a_data_prompt_gets_compact_context(
    client: httpx.AsyncClient, prompt: str
) -> None:
    response = await client.post("/context", json={"prompt": prompt}, headers=auth())

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"additionalContext"}, "the hook contract is one field"
    context = body["additionalContext"]
    assert context, f"{prompt!r} should have matched the semantic layer"
    assert context.splitlines()[0].startswith("mdqm context (semantic ")
    # One estimator in the project (C16), and the block stays inside the budget it was given.
    assert estimate_tokens(context) <= DEFAULT_PACK_BUDGET


async def test_the_data_prompt_brings_the_measure_and_the_value_map(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post("/context", json={"prompt": DATA_PROMPT}, headers=auth())

    context = response.json()["additionalContext"]
    assert "credit_limit_r5000" in context
    assert "round half_up to 5000" in context
    assert "customer_status" in context


@pytest.mark.parametrize(
    "prompt", [OFF_TOPIC_PROMPT, "", "   ", "write me a poem about the sea"], ids=repr
)
async def test_a_prompt_the_layer_cannot_help_with_is_empty_not_an_error(
    client: httpx.AsyncClient, prompt: str
) -> None:
    response = await client.post("/context", json={"prompt": prompt}, headers=auth())

    assert response.status_code == 200
    assert response.json() == {"additionalContext": ""}


async def test_the_answer_is_deterministic(client: httpx.AsyncClient) -> None:
    # The hook injects this on every prompt: two calls must be byte-identical, not merely equal.
    first = await client.post("/context", json={"prompt": DATA_PROMPT}, headers=auth())
    second = await client.post("/context", json={"prompt": DATA_PROMPT}, headers=auth())

    assert first.content == second.content


# --------------------------------------------------------------------------------------------------
# It leaks nothing
# --------------------------------------------------------------------------------------------------


async def test_the_response_carries_no_token_and_no_prompt_echo(
    client: httpx.AsyncClient,
) -> None:
    # A prompt that contains a credential-shaped string must not come back, and no token may leak.
    prompt = f"{DATA_PROMPT} password=hunter2 ak-{ALICE}"
    response = await client.post("/context", json={"prompt": prompt}, headers=auth())

    assert response.status_code == 200
    assert ALICE not in response.text
    assert "hunter2" not in response.text
    assert DATA_PROMPT not in response.text


# --------------------------------------------------------------------------------------------------
# Refusals are ErrorResponse-shaped
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        ({}, "context.prompt"),
        ({"prompt": 5}, "context.prompt"),
        ({"question": DATA_PROMPT}, "context.prompt"),
        ({"prompt": DATA_PROMPT, "limit": 10}, "context.limit"),
        ({"prompt": "x" * (MAX_PROMPT_CHARS + 1)}, "context.prompt"),
        ([1, 2, 3], "context"),
        ("not an object", "context"),
    ],
    ids=["empty-object", "not-a-string", "wrong-field", "unknown-field", "too-long", "list", "str"],
)
async def test_a_body_the_hook_must_not_send_is_a_400_refusal(
    client: httpx.AsyncClient, payload: object, needle: str
) -> None:
    response = await client.post("/context", json=payload, headers=auth())

    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "error"
    assert body["code"] == "INVALID_SPEC"
    assert needle in body["problem"]
    assert body["fix"]
    assert "additionalContext" not in body


async def test_a_body_that_is_not_json_is_a_400_refusal(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/context",
        content=b"{not json",
        headers={**auth(), "Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_SPEC"


async def test_a_server_with_no_semantic_layer_says_so_with_a_fix(
    client_without_source: httpx.AsyncClient,
) -> None:
    # A fresh installation has no approved facts yet. Saying "nothing matched" would be a lie.
    response = await client_without_source.post(
        "/context", json={"prompt": DATA_PROMPT}, headers=auth()
    )

    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "NOT_SUPPORTED"
    assert "context injection" in body["problem"]
    assert "without injected context" in body["fix"]
    assert "additionalContext" not in body


# --------------------------------------------------------------------------------------------------
# Wiring from settings: absence is normal, corruption is not
# --------------------------------------------------------------------------------------------------


def write_semantic(root: Path) -> Path:
    """A one-file semantic directory: what the bootstrap pipeline writes into ``generated/``."""
    (root / "generated").mkdir(parents=True)
    # JSON is valid YAML, so the fixture needs no YAML escaping.
    (root / "generated" / "semantic.yaml").write_text(json.dumps(SEMANTIC), encoding="utf-8")
    return root


def settings_for(tmp_path: Path, semantic_dir: Path) -> Settings:
    """Settings pointing at throwaway paths: no file in the repo, no env var."""
    return Settings(
        paths=PathsSettings(semantic_dir=semantic_dir, metadata_db=tmp_path / "meta.sqlite")
    )


def test_no_semantic_directory_means_no_source(tmp_path: Path) -> None:
    assert context_source_from_settings(settings_for(tmp_path, tmp_path / "absent")) is None


def test_a_semantic_directory_yields_a_working_source(tmp_path: Path) -> None:
    root = write_semantic(tmp_path / "semantic")

    built = context_source_from_settings(settings_for(tmp_path, root))

    assert built is not None
    assert built.layer.version_hash
    assert built.context_for(OFF_TOPIC_PROMPT).text == ""
    assert built.context_for(DATA_PROMPT).text


# --------------------------------------------------------------------------------------------------
# Latency: the hook runs on every prompt, so the budget is a product requirement
# --------------------------------------------------------------------------------------------------


@pytest.mark.slow
async def test_a_thousand_sequential_calls_stay_inside_the_p95_budget(
    client: httpx.AsyncClient,
) -> None:
    """1,000 hook calls through the real app; p95 must be under 150 ms (task M06 acceptance)."""
    timings: list[float] = []
    for _ in range(1_000):
        started = time.perf_counter()
        response = await client.post("/context", json={"prompt": DATA_PROMPT}, headers=auth())
        timings.append((time.perf_counter() - started) * 1_000)
        assert response.status_code == 200

    ordered = sorted(timings)
    p95 = ordered[int(len(ordered) * 0.95) - 1]
    print(
        f"\n/context latency over {len(timings)} sequential calls: "
        f"p50={statistics.median(ordered):.2f} ms p95={p95:.2f} ms max={ordered[-1]:.2f} ms"
    )
    assert p95 < 150.0, f"p95 was {p95:.1f} ms over 1,000 sequential calls"
