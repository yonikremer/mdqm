"""M02: ``find_data`` answers from the Context Pack, budgeted, in both languages, owner-aware.

Five claims carry this task, and each is tested as a behaviour:

* **The same question in two languages finds the same facts.** ``average credit limit`` and
  ``מסגרת אשראי ממוצעת`` return the same measure and the same tables, because the agent's language
  must not change what the server knows.
* **An unknown question is an empty pack with a hint, not an error**, and it is logged as an
  unanswered question (H04) so the data team learns the word the layer is missing.
* **The whole answer stays inside the pack budget**: the pack plus the proposed spec skeleton, for
  every budget, English and Hebrew alike. When nothing fits beside it the skeleton is dropped and
  the answer says why instead of silently shrinking the pack.
* **A refusal is an ``ErrorResponse``**: an unknown source name comes back with the valid sources,
  an exact ``fix``, and (when the server has an audit log) a row in it.
* **The identity the handler sees is the caller's**, proved once over a real MCP roundtrip on
  loopback (the one socket these unit tests opt in to).

Nothing here re-ranks the pack: that is H05's job and its own tests cover it.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx2
import pytest
import pytest_socket
import uvicorn
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

from mdqm.context.index import ApprovedExample, ContextIndex
from mdqm.context.pack import MIN_PACK_BUDGET, PackKind, build_pack
from mdqm.errors import ErrorCode
from mdqm.results.summary import estimate_tokens
from mdqm.semantic import SemanticLayer
from mdqm.server.app import create_server
from mdqm.server.auth import READ_SCOPE, StaticTokenVerifier, TokenEntry
from mdqm.server.tools.find_data import (
    EMPTY_HINT,
    NO_ROOM_HINT,
    FindDataServices,
    _render_items,
    configure_find_data,
    find_data,
    reset_find_data,
)
from mdqm.spec.validate import parse_spec
from mdqm.store import Store
from mdqm.store.audit import AuditLog, AuditOutcome

NOW: Final = datetime(2026, 3, 1, 9, 30, tzinfo=UTC)

#: The layer the dictionary is built from: one measure stored three ways, a categorical field with
#: Hebrew synonyms, a partitioned event table with both time columns, two approved joins and three
#: measures on the same business words, so the budget actually has something to choose between.
SEMANTIC: Final[dict[str, Any]] = {
    "version": 7,
    "sources": {
        "crm": {"adapter": "postgresql"},
        "billing": {"adapter": "postgresql"},
        "erp": {"adapter": "postgresql"},
    },
    "value_maps": {
        "customer_status": {
            "canonical": ["active", "churned", "suspended"],
            "synonyms": {
                "active": ["פעיל", "פָּעִיל"],
                "churned": ["עזב"],
                "suspended": ["מושהה"],
            },
            "sources": {
                "crm.customers.status": {"A": "active", "C": "churned", "S": "suspended"},
                "billing.accounts.state": {
                    "Active": "active",
                    "Closed": "churned",
                    "Hold": "suspended",
                },
            },
        },
        "payment_terms": {
            "canonical": ["net_30", "net_60"],
            "sources": {"billing.accounts.terms": {"30": "net_30", "60": "net_60"}},
        },
    },
    "measures": {
        "credit_limit_r5000": {
            "canonical": {"rounding": {"to": 5000, "mode": "half_up"}},
            "sources": {
                "crm.customers.credit_limit": {"normalize": "{col}"},
                "billing.accounts.credit_limit": {"normalize": "ROUND({col} / 5000) * 5000"},
                "erp.customers.limit_k": {"normalize": "{col} * 10"},
            },
        },
        "credit_limit_total": {
            "sources": {
                "crm.customers.credit_limit": {"normalize": "{col}"},
                "billing.accounts.credit_limit": {"normalize": "ROUND({col} / 5000) * 5000"},
                "erp.customers.limit_k": {"normalize": "{col} * 10"},
            }
        },
        "credit_limit_customer_count": {
            "sources": {"crm.customers.cust_id": {"normalize": "{col}"}}
        },
    },
    "tables": {
        "billing.usage_events": {
            "key": ["event_id"],
            "partitioning": {"key": "start_time", "granularity": "month"},
            "time_columns": {
                "start_time": {"role": "event_time", "indexed": True, "use_by_default": True},
                "insertion_time": {
                    "role": "ingestion_time",
                    "derives_bound_on": "start_time",
                    "lag": {"min": "0s", "max": "6h"},
                },
            },
            "mutability": {"class": "append_rarely", "status": "approved"},
        }
    },
    "approved_joins": [
        {
            "name": "usage_to_customers",
            "left": "billing.usage_events.client_no",
            "right": "crm.customers.cust_id",
            "on": "LPAD(CAST({right} AS VARCHAR), 10, '0') = {left}",
            "cardinality": "many_to_one",
            "usage": {"count_90d": 12, "users_90d": 3},
        },
        {
            "name": "usage_to_accounts",
            "left": "billing.usage_events.client_no",
            "right": "billing.accounts.client_no",
            "on": "CAST({right} AS VARCHAR) = {left}",
            "cardinality": "many_to_one",
            "usage": {"count_90d": 1, "users_90d": 1},
        },
    ],
}

#: The deployment's own words for things the layer names in one language only.
LABELS: Final[dict[str, tuple[str, ...]]] = {
    "measure:credit_limit_r5000": ("credit limit", "credit limits", "מסגרת אשראי"),
    "table:crm.customers": ("לקוחות", "customers"),
    "table:billing.accounts": ("חשבונות", "accounts"),
    "table:billing.usage_events": ("events", "אירועי שימוש"),
}

EXAMPLES: Final[tuple[ApprovedExample, ...]] = (
    ApprovedExample(
        name="avg_credit_limit_by_status_3mo",
        question="מהי מסגרת האשראי הממוצעת לפי סטטוס ב-3 החודשים האחרונים",
        ref="specs/avg_credit_limit_by_status_3mo",
    ),
    ApprovedExample(
        name="credit_limit_by_status",
        question="average credit limit by customer status",
        ref="specs/credit_limit_by_status",
    ),
)

#: The same question twice. The words are chosen so both reach the measure and its three tables:
#: "customer"/"status" would add the join, the value map and the event table on the English side
#: only, and those are tested separately.
EN_QUESTION: Final[str] = "average credit limit"
HE_QUESTION: Final[str] = "מסגרת אשראי ממוצעת"


def _bearer(user_id: str) -> str:
    """A throwaway dev token value: no test ever uses a real credential."""
    return f"dev-{user_id}"


TOKENS: Final = (
    TokenEntry(token=_bearer("alice"), user_id="alice", display_name="Alice"),
    TokenEntry(token=_bearer("bob"), user_id="bob", display_name="Bob"),
)


# --------------------------------------------------------------------------------------------------
# the dictionary, a throwaway database, and an authenticated caller
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    """A migrated metadata DB on a throwaway file (SQLite runs in-process, no Docker)."""
    opened = Store(tmp_path / "meta.sqlite")
    opened.initialize()
    try:
        yield opened
    finally:
        opened.close()


@pytest.fixture(autouse=True)
def no_leaked_dictionary() -> Iterator[None]:
    """The slot is process-wide, so no test may inherit another test's dictionary."""
    reset_find_data()
    try:
        yield
    finally:
        reset_find_data()


@pytest.fixture
def index(store: Store) -> ContextIndex:
    """The H04 index over :data:`SEMANTIC`, ready to answer questions."""
    built = ContextIndex(store, clock=lambda: NOW)
    built.build(_layer(), labels=LABELS, examples=EXAMPLES)
    return built


@pytest.fixture
def services(store: Store, index: ContextIndex) -> FindDataServices:
    """The configured dictionary, with an audit log so a refusal is provable."""
    configured = FindDataServices(
        layer=_layer(), index=index, examples=EXAMPLES, audit=AuditLog(store, clock=lambda: NOW)
    )
    configure_find_data(configured)
    return configured


def _layer() -> SemanticLayer:
    """The test layer, validated once per call (the models are frozen, so this is cheap)."""
    return SemanticLayer.model_validate(SEMANTIC)


def access_token(user_id: str) -> AccessToken:
    """The access token the SDK's middleware would have verified for ``user_id``."""
    return AccessToken(
        token=_bearer(user_id),
        client_id=f"mdqm:{user_id}",
        scopes=[READ_SCOPE],
        subject=user_id,
        claims={"display_name": user_id.title()},
    )


@contextmanager
def as_user(user_id: str) -> Iterator[None]:
    """Run the block as ``user_id``: what the SDK's auth middleware binds for one message."""
    reset = auth_context_var.set(AuthenticatedUser(access_token(user_id)))
    try:
        yield
    finally:
        auth_context_var.reset(reset)


def with_budget(services: FindDataServices, budget: int) -> FindDataServices:
    """The same dictionary with a different pack budget, as a server built for a smaller model."""
    return FindDataServices(
        layer=services.layer,
        index=services.index,
        examples=services.examples,
        audit=services.audit,
        budget_tokens=budget,
    )


def table_refs(answer: Any) -> set[str]:
    """The table refs the pack kept (the facts a spec must be written against)."""
    assert answer.pack is not None, "a successful find_data answer always carries a pack"
    return {item.ref for item in answer.pack.items_of_kind(PackKind.TABLE)}


def measure_refs(answer: Any) -> set[str]:
    """The measure refs the pack kept."""
    assert answer.pack is not None, "a successful find_data answer always carries a pack"
    return {item.ref for item in answer.pack.items_of_kind(PackKind.MEASURE)}


def skeleton_json(answer: Any) -> str:
    """The skeleton's spec as the agent would send it, or an empty string when there is none."""
    if answer.skeleton is None or answer.skeleton.spec is None:
        return ""
    dumped: str = answer.skeleton.spec.model_dump_json(by_alias=True, exclude_none=True)
    return dumped


# --------------------------------------------------------------------------------------------------
# the same facts in both languages
# --------------------------------------------------------------------------------------------------


def test_hebrew_and_english_ask_the_same_question_get_the_same_tables_and_measure(
    services: FindDataServices,
) -> None:
    """The acceptance criterion: the language changes the words, not the answer."""
    with as_user("alice"):
        english = find_data(EN_QUESTION)
        hebrew = find_data(HE_QUESTION)

    assert english.status == hebrew.status == "ok"
    assert (
        table_refs(english)
        == table_refs(hebrew)
        == {
            "crm.customers",
            "billing.accounts",
            "erp.customers",
        }
    )
    assert measure_refs(english) == measure_refs(hebrew)
    assert "credit_limit_r5000" in measure_refs(hebrew)
    assert hebrew.pack is not None
    # The per-source conversions are the whole point of the pack: Hebrew must carry them too.
    hebrew_measure = next(
        item
        for item in hebrew.pack.items_of_kind(PackKind.MEASURE)
        if item.ref == "credit_limit_r5000"
    )
    assert "erp.customers.limit_k={col} * 10" in hebrew_measure.text
    assert "round half_up to 5000" in hebrew_measure.text
    # And the proposed spec is the same shape in both languages.
    assert skeleton_json(english) == skeleton_json(hebrew)
    assert skeleton_json(hebrew), "the credit-limit measure must yield a skeleton"


def test_the_skeleton_is_a_valid_spec_and_names_the_matched_measure(
    services: FindDataServices,
) -> None:
    """The skeleton is a real spec (it validates) built from the measure and its grouped column."""
    with as_user("alice"):
        answer = find_data("average credit limit by customer status")

    assert answer.skeleton is not None
    assert answer.skeleton.based_on == "measure:credit_limit_r5000"
    spec = parse_spec(json.loads(skeleton_json(answer)))
    assert spec.from_table == "accounts"
    assert spec.group_by == ("state",)
    assert [item for item in spec.select if isinstance(item, str)] == ["state"]
    # No window and no aggregation is invented: both change what the answer means, and the
    # linter's own error carries the fix when one is needed.
    assert spec.time is None
    assert all(getattr(item, "agg", None) is None for item in spec.select)


# --------------------------------------------------------------------------------------------------
# nothing matched: an empty pack, a hint, and a logged question
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "what is the weather in Tel Aviv",
        "write me a poem about the sea",
        "מה מזג האוויר בתל אביב",
    ],
)
def test_an_unknown_question_is_an_empty_pack_with_a_short_hint(
    services: FindDataServices, question: str
) -> None:
    """Not an error: the agent is told to retry with better words (the acceptance criterion)."""
    with as_user("alice"):
        answer = find_data(question)

    assert answer.status == "ok"
    assert answer.error is None
    assert answer.pack is not None and answer.pack.is_empty
    assert answer.pack.text == ""
    assert answer.pack.items == ()
    assert answer.skeleton is None
    assert answer.tokens_total == 0
    assert answer.hint == EMPTY_HINT
    assert len(EMPTY_HINT) <= 300, "a hint is one line, not a report"


def test_an_empty_answer_is_logged_as_an_unanswered_question(
    services: FindDataServices,
) -> None:
    """The only cheap evidence that the layer lacks a word the organisation uses (H04)."""
    with as_user("bob"):
        find_data("what is the weather in Tel Aviv")

    unmatched = services.index.unmatched()
    assert len(unmatched) == 1
    assert unmatched[0].query == "what is the weather in tel aviv"
    assert unmatched[0].user_id == "bob"


def test_a_question_with_no_word_in_it_is_refused_with_a_fix(
    services: FindDataServices,
) -> None:
    """Whitespace is a caller bug, not a question about data: it is refused, not answered."""
    with as_user("alice"):
        answer = find_data("   ...   ")

    assert answer.status == "error"
    assert answer.error is not None
    assert answer.error.code == ErrorCode.INVALID_SPEC.value
    assert "question" in answer.error.fix
    assert answer.pack is None


# --------------------------------------------------------------------------------------------------
# the budget is a ceiling on the whole answer
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("question", [EN_QUESTION, HE_QUESTION, "usage events customers"])
@pytest.mark.parametrize("budget", [MIN_PACK_BUDGET, 800, 1_500, 4_000])
def test_the_answer_never_exceeds_the_pack_budget(
    services: FindDataServices, question: str, budget: int
) -> None:
    """Pack plus skeleton fit the budget, the estimate is honest, and something is returned."""
    configure_find_data(with_budget(services, budget))
    with as_user("alice"):
        answer = find_data(question)

    assert answer.pack is not None
    assert answer.pack.budget_tokens <= budget
    assert answer.pack.tokens_estimated <= budget
    assert answer.tokens_total <= budget
    assert answer.tokens_total == answer.pack.tokens_estimated + _skeleton_tokens_for(answer)


def _skeleton_tokens_for(answer: Any) -> int:
    """Re-measure the skeleton the way the tool does, so the reported total is checkable."""
    skeleton = answer.skeleton
    if skeleton is None:
        return 0
    parts = [skeleton.based_on, skeleton.note]
    if skeleton.spec is not None:
        parts.append(skeleton.spec.model_dump_json(by_alias=True, exclude_none=True))
    return estimate_tokens("\n".join(part for part in parts if part))


def test_a_tight_budget_drops_the_skeleton_and_says_why(
    services: FindDataServices,
) -> None:
    """When the pack needs the whole budget the answer explains the missing skeleton."""
    # This question fills the floor budget (nearly 400 tokens of pack), so the skeleton — about 60
    # more — cannot fit beside it and the answer has to say why.
    configure_find_data(with_budget(services, MIN_PACK_BUDGET))
    with as_user("alice"):
        answer = find_data("average credit limit by customer status")

    assert answer.pack is not None
    assert answer.pack.tokens_estimated <= MIN_PACK_BUDGET
    assert answer.skeleton is None
    assert answer.hint == NO_ROOM_HINT
    assert answer.tokens_total == answer.pack.tokens_estimated


# --------------------------------------------------------------------------------------------------
# the source scope
# --------------------------------------------------------------------------------------------------


def test_naming_one_source_drops_the_facts_of_the_others(
    services: FindDataServices,
) -> None:
    """A scoped question gets the facts of that source only, and what was dropped is reported."""
    with as_user("alice"):
        answer = find_data("average credit limit by customer status", sources=["crm"])

    assert answer.pack is not None
    assert answer.error is None
    assert table_refs(answer) and all(ref.startswith("crm.") for ref in table_refs(answer))
    assert answer.excluded, "the dropped refs must be reported, not silently removed"
    assert all(":" in ref for ref in answer.excluded)
    text = answer.pack.text
    assert "billing.accounts" not in text
    assert "erp.customers" not in text
    measure = next(
        item
        for item in answer.pack.items_of_kind(PackKind.MEASURE)
        if item.ref == "credit_limit_r5000"
    )
    assert "crm.customers.credit_limit={col}" in measure.text
    assert "erp.customers.limit_k" not in measure.text


def test_a_scope_that_drops_nothing_returns_the_pack_unchanged(
    services: FindDataServices,
) -> None:
    """Every source named: the answer is exactly H05's pack, byte for byte."""
    with as_user("alice"):
        scoped = find_data(EN_QUESTION, sources=["crm", "billing", "erp"])

    assert scoped.pack is not None
    direct = build_pack(EN_QUESTION, layer=_layer(), index=services.index, examples=EXAMPLES)
    assert scoped.excluded == ()
    assert scoped.pack.text == direct.text
    assert scoped.pack.items == direct.items


def test_the_space_the_renderer_reproduces_is_the_packs_own(services: FindDataServices) -> None:
    """Guard against layout drift: re-rendering every item equals the block H05 wrote."""
    pack = build_pack(EN_QUESTION, layer=_layer(), index=services.index, examples=EXAMPLES)
    assert pack.text
    assert (
        _render_items(
            pack.items, pack.omitted, budget=pack.budget_tokens, version=pack.semantic_version
        )
        == pack.text
    )


@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        (["nope"], ErrorCode.INVALID_SPEC),
        ([], ErrorCode.INVALID_SPEC),
        (["   "], ErrorCode.INVALID_SPEC),
    ],
)
def test_an_unusable_source_selection_is_an_error_response(
    services: FindDataServices, sources: list[str], expected: ErrorCode
) -> None:
    """A refusal is the structured error shape, with the valid sources, not an empty pack."""
    with as_user("alice"):
        answer = find_data(EN_QUESTION, sources=sources)

    assert answer.status == "error"
    assert answer.error is not None
    assert answer.error.code == expected.value
    assert "sources" in answer.error.fix
    assert set(answer.error.valid_values or ()) == {"billing", "crm", "erp"}
    assert answer.pack is None


def test_a_refusal_is_audited(services: FindDataServices) -> None:
    """A cross-checking caller leaves a row behind, like every other refusal (M05's rule)."""
    with as_user("alice"):
        find_data(EN_QUESTION, sources=["nope"])

    assert services.audit is not None
    blocked = [
        (record.user_id, record.tool, record.error_code)
        for record in services.audit.rows()
        if record.outcome is AuditOutcome.BLOCKED
    ]
    assert blocked == [("alice", "find_data", ErrorCode.INVALID_SPEC)]


def test_without_a_dictionary_the_tool_says_so() -> None:
    """A wiring mistake is reported as itself, never as an empty result."""
    reset_find_data()
    with as_user("alice"):
        answer = find_data(EN_QUESTION)

    assert answer.status == "not_implemented"
    assert answer.problem is not None and "semantic dictionary" in answer.problem
    assert answer.fix is not None and "bootstrap" in answer.fix


# --------------------------------------------------------------------------------------------------
# one real MCP roundtrip: the tool, the schema and the caller's identity
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def endpoint() -> Iterator[str]:
    """Run the real server on an ephemeral loopback port and yield its ``/mcp`` URL."""
    # The module is socket-blocked; this one test needs loopback and nothing else.
    pytest_socket.socket_allow_hosts(["127.0.0.1"])
    server = create_server(verifier=StaticTokenVerifier(TOKENS))
    config = uvicorn.Config(
        server.streamable_http_app(), host="127.0.0.1", port=0, log_level="warning"
    )
    running = uvicorn.Server(config)
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while not running.started:
            if time.monotonic() > deadline:  # pragma: no cover - only on a wedged box
                raise RuntimeError("the test server did not start within 30s")
            time.sleep(0.01)
        port = running.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        running.should_exit = True
        thread.join(timeout=10)


@asynccontextmanager
async def mcp_client(url: str, token: str) -> AsyncIterator[Client]:
    """An SDK client against ``url`` carrying the bearer token of one caller."""
    headers = {"Authorization": f"Bearer {token}"}
    http_client = httpx2.AsyncClient(headers=headers, timeout=30.0)
    try:
        async with Client(streamable_http_client(url, http_client=http_client)) as client:
            yield client
    finally:
        await http_client.aclose()


@pytest.mark.enable_socket
async def test_over_mcp_a_hebrew_question_returns_the_pack_and_the_caller(
    services: FindDataServices, endpoint: str
) -> None:
    """The tool the agent actually calls: Hebrew in, the pack and a skeleton out, as Bob."""
    async with mcp_client(endpoint, TOKENS[1].token) as bob:
        hebrew = await bob.call_tool("find_data", {"question": HE_QUESTION})
        unknown = await bob.call_tool("find_data", {"question": "what is the weather"})

    hebrew_body = hebrew.structured_content
    assert hebrew_body is not None
    assert hebrew_body["status"] == "ok"
    assert hebrew_body["user_id"] == "bob"
    pack = hebrew_body["pack"]
    assert pack["tokens_estimated"] <= pack["budget_tokens"]
    assert {item["ref"] for item in pack["items"] if item["kind"] == "measure"} == {
        "credit_limit_r5000",
        "credit_limit_total",
        "credit_limit_customer_count",
    }
    assert {item["ref"] for item in pack["items"] if item["kind"] == "table"} == {
        "crm.customers",
        "billing.accounts",
        "erp.customers",
    }
    assert hebrew_body["skeleton"]["spec"]["from"] == "accounts"
    assert hebrew_body["skeleton"]["based_on"].startswith("measure:")

    unknown_body = unknown.structured_content
    assert unknown_body is not None
    assert unknown_body["status"] == "ok"
    assert unknown_body["pack"]["text"] == ""
    assert unknown_body["pack"]["items"] == []
    assert unknown_body["hint"] == EMPTY_HINT

    # Bob's unanswered question is Bob's row in the log, not Alice's.
    unmatched = services.index.unmatched()
    assert [row.user_id for row in unmatched] == ["bob"]
