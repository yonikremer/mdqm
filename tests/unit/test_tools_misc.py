"""M05: the three lookup tools -- owner-scoped paging, your own history, the feedback queue.

Three claims carry this task, and each is tested as a *behaviour* rather than against helper names:

* **A handle is not a capability.** Paging a 10k-row result with the tool returns every row exactly
  once, and another user's handle answers with the same text as an id that never existed -- so the
  handle space cannot be probed to learn that someone else's result exists.
* **History is personal.** ``history`` lists and searches only the caller's own runs, and a re-run
  asked for with ``rerun_id`` is ``frozen`` (the stored ``exec_ts``) or ``relative`` (the clock's),
  exactly as :mod:`mdqm.store.history` defines it.
* **A rating is stored, never trusted.** ``feedback`` writes one row to the queue, refuses somebody
  else's run id like a missing one, and every refusal is an ``ErrorResponse`` that also lands in the
  audit log.

The tool bodies are called twice over: directly (fast, deterministic, clock injected) and once over
a real MCP roundtrip on loopback, because "the identity the handler sees is the caller's" is only
proved end to end. ``tests/unit`` is socket-blocked; that one test opts in to ``127.0.0.1`` only.
"""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import httpx2
import pyarrow as pa
import pytest
import pytest_socket
import uvicorn
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

from mdqm.errors import ErrorCode
from mdqm.governor.handles import ResultStore
from mdqm.server.app import create_server
from mdqm.server.auth import READ_SCOPE, StaticTokenVerifier, TokenEntry
from mdqm.server.tools import (
    ToolServices,
    configure_services,
    feedback,
    get_more,
    history,
    reset_services,
)
from mdqm.sql.exec_ts import ExecTsMode, rewrite_sql
from mdqm.store import Store
from mdqm.store.audit import AuditLog, AuditOutcome
from mdqm.store.feedback import FeedbackStore
from mdqm.store.history import HistoryEntry, HistoryStore

# --------------------------------------------------------------------------------------------------
# deterministic time, a throwaway database, and an authenticated caller
# --------------------------------------------------------------------------------------------------

BASE: Final = datetime(2026, 11, 25, 16, 0, tzinfo=UTC)

#: A weekly window: the only kind of run where ``frozen`` and ``relative`` can differ.
WEEKLY_ORIGINAL: Final = (
    "SELECT customer_status, AVG(credit_limit) FROM accounts "
    "WHERE start_time >= now() - INTERVAL '7 DAYS' GROUP BY customer_status"
)
WEEKLY_FROZEN: Final = rewrite_sql(WEEKLY_ORIGINAL, dialect="postgres", exec_ts=BASE).sql

#: A run with no now-like value, so a re-run could not move anything even if it wanted to.
HEBREW_SQL: Final = "SELECT cust_id, credit_limit FROM customers WHERE customer_status = 'פעיל'"


class ScriptedClock:
    """A clock that moves only when the test moves it: handles, history and feedback share it.

    Two shapes on purpose: the governor's :class:`~mdqm.governor.Clock` protocol wants ``now()`` and
    ``monotonic()``, while history and feedback take a plain callable.
    """

    def __init__(self, wall: datetime = BASE) -> None:
        self._wall = wall
        self._mono = 0.0

    def monotonic(self) -> float:
        return self._mono

    def now(self) -> datetime:
        return self._wall

    def __call__(self) -> datetime:
        return self._wall

    def advance(self, **kwargs: int) -> datetime:
        delta = timedelta(**kwargs)
        self._wall += delta
        self._mono += delta.total_seconds()
        return self._wall


@pytest.fixture
def clock() -> ScriptedClock:
    return ScriptedClock()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    """A migrated metadata DB on a throwaway file, closed afterwards."""
    opened = Store(tmp_path / "meta.sqlite")
    opened.initialize()
    try:
        yield opened
    finally:
        opened.close()


@pytest.fixture(autouse=True)
def no_leaked_services() -> Iterator[None]:
    """No test may inherit another test's stores: the slot is process-wide, so reset it twice."""
    reset_services()
    try:
        yield
    finally:
        reset_services()


@pytest.fixture
def services(store: Store, clock: ScriptedClock) -> ToolServices:
    """The four stores the tools read, all on the throwaway file and the scripted clock."""
    built = ToolServices(
        handles=ResultStore(clock=clock),
        history=HistoryStore(store, clock=clock),
        feedback=FeedbackStore(store, clock=clock),
        audit=AuditLog(store, clock=clock),
    )
    configure_services(built)
    return built


def access_token(user_id: str) -> AccessToken:
    """The access token the SDK's middleware would have verified for ``user_id``."""
    return AccessToken(
        token=f"tok-{user_id}",
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


def weekly_entry(user_id: str) -> HistoryEntry:
    """A representative run for ``user_id``: the weekly window, one source."""
    return HistoryEntry(
        user_id=user_id,
        exec_ts=BASE,
        mode=ExecTsMode.RELATIVE,
        original_sql=WEEKLY_ORIGINAL,
        frozen_sql=WEEKLY_FROZEN,
        dialect="postgres",
    )


def hebrew_entry(user_id: str) -> HistoryEntry:
    """A run whose question is in Hebrew, so search is exercised with real text, not ASCII."""
    return HistoryEntry(
        user_id=user_id,
        exec_ts=BASE,
        original_sql=HEBREW_SQL,
        frozen_sql=HEBREW_SQL,
        dialect="postgres",
    )


def numbered(rows: int) -> pa.Table:
    """A result with one readable column, so a page can be checked row by row."""
    return pa.table({"row_no": pa.array(range(rows), pa.int64())})


def refusals(services: ToolServices) -> list[tuple[str, str, ErrorCode | None]]:
    """Every audited refusal so far, as (user, tool, code): the proof that a refusal left a row."""
    return [
        (record.user_id, record.tool, record.error_code)
        for record in services.audit.rows()
        if record.outcome is AuditOutcome.BLOCKED
    ]


# --------------------------------------------------------------------------------------------------
# get_more: paging, and owner scoping
# --------------------------------------------------------------------------------------------------


def test_paging_a_10k_row_result_returns_every_row_exactly_once(
    services: ToolServices,
) -> None:
    handle = services.handles.put(owner="alice", source="crm", table=numbered(10_000))
    seen: list[int] = []
    pages = 0
    offset = 0
    with as_user("alice"):
        while True:
            answer = get_more(handle.handle, offset=offset, limit=1000)
            assert answer.status == "ok", answer.error
            assert answer.error is None and answer.page is not None
            page = answer.page
            seen.extend(int(row["row_no"]) for row in page.rows)
            pages += 1
            # The next offset is where this page actually ended, so no row can be skipped even
            # when the summariser hands back a shorter page than asked for.
            assert page.offset == offset
            if page.next_offset is None:
                assert page.truncated is False
                break
            assert page.next_offset == offset + len(page.rows)
            offset = page.next_offset
    assert pages >= 10, "a 10k result cannot arrive in fewer than ten 1000-row calls"
    assert seen == list(range(10_000)), "every row, once, in order"
    assert len(set(seen)) == 10_000


def test_another_users_handle_answers_exactly_like_a_missing_one(
    services: ToolServices,
) -> None:
    handle = services.handles.put(owner="alice", source="crm", table=numbered(3))
    with as_user("bob"):
        stolen = get_more(handle.handle, offset=0, limit=10)
        missing = get_more("h_0000000000000000", offset=0, limit=10)

    assert stolen.status == missing.status == "error"
    assert stolen.error is not None and missing.error is not None
    assert stolen.error.code == ErrorCode.NOT_FOUND.value
    # Byte-identical: "not yours" and "never existed" must be indistinguishable, or the handle
    # space becomes a way to learn that another user's result exists.
    assert stolen.error.model_dump_json() == missing.error.model_dump_json()
    assert handle.handle not in stolen.error.model_dump_json()
    assert stolen.page is None and stolen.user_id == "bob"
    assert refusals(services) == [
        ("bob", "get_more", ErrorCode.NOT_FOUND),
        ("bob", "get_more", ErrorCode.NOT_FOUND),
    ]


def test_the_owner_can_still_page_after_a_stranger_was_refused(
    services: ToolServices,
) -> None:
    handle = services.handles.put(owner="alice", source="crm", table=numbered(4))
    with as_user("bob"):
        assert get_more(handle.handle).status == "error"
    with as_user("alice"):
        answer = get_more(handle.handle)
    assert answer.page is not None
    assert [row["row_no"] for row in answer.page.rows] == [0, 1, 2, 3]


# --------------------------------------------------------------------------------------------------
# history: your own runs, search, and frozen vs relative re-run
# --------------------------------------------------------------------------------------------------


def test_history_lists_only_the_callers_own_runs(services: ToolServices) -> None:
    alice_first = services.history.record(weekly_entry("alice"))
    alice_second = services.history.record(hebrew_entry("alice"))
    bob_run = services.history.record(weekly_entry("bob"))

    with as_user("alice"):
        mine = history(limit=10)
    with as_user("bob"):
        theirs = history(limit=10)

    assert [item.id for item in mine.runs] == [alice_second.id, alice_first.id]
    assert [item.id for item in theirs.runs] == [bob_run.id]
    # The run's own shape is what makes it recognisable; ids alone would not be enough.
    assert mine.runs[1].spec is None and mine.runs[1].sql == WEEKLY_ORIGINAL
    assert mine.runs[1].mode == ExecTsMode.RELATIVE.value
    assert mine.runs[1].rerunnable is True


def test_history_search_is_scoped_to_the_caller_and_handles_hebrew(
    services: ToolServices,
) -> None:
    alice_run = services.history.record(hebrew_entry("alice"))
    services.history.record(weekly_entry("alice"))

    with as_user("alice"):
        found = history(text="פעיל")
    with as_user("bob"):
        not_mine = history(text="פעיל")

    assert [item.id for item in found.runs] == [alice_run.id]
    assert not_mine.runs == ()
    assert not_mine.status == "ok", "an empty result is an answer, not a refusal"


def test_history_search_with_no_word_is_a_refusal_with_a_fix(services: ToolServices) -> None:
    services.history.record(weekly_entry("alice"))
    with as_user("alice"):
        answer = history(text="   ")

    assert answer.status == "error" and answer.error is not None
    assert answer.error.code == ErrorCode.INVALID_SPEC.value
    assert '"text"' in answer.error.fix
    assert refusals(services) == [("alice", "history", ErrorCode.INVALID_SPEC)]


def test_rerun_frozen_keeps_the_stored_exec_ts_and_payload(
    services: ToolServices, clock: ScriptedClock
) -> None:
    run = services.history.record(weekly_entry("alice"))
    clock.advance(days=3)

    with as_user("alice"):
        answer = history(rerun_id=run.id, rerun_mode="frozen")

    assert answer.status == "ok", answer.error
    assert answer.rerun is not None
    assert answer.rerun.exec_ts == BASE, "a frozen re-run must not move with the clock"
    assert answer.rerun.sql == WEEKLY_FROZEN
    assert answer.rerun.frozen is True


def test_rerun_relative_moves_the_window_to_the_clock(
    services: ToolServices, clock: ScriptedClock
) -> None:
    run = services.history.record(weekly_entry("alice"))
    later = clock.advance(days=3)

    with as_user("alice"):
        answer = history(rerun_id=run.id, rerun_mode="relative")

    assert answer.rerun is not None
    assert answer.rerun.exec_ts == later, "a relative re-run takes a fresh exec_ts"
    assert answer.rerun.sql != WEEKLY_FROZEN, "the substituted timestamp must move with it"
    assert answer.rerun.frozen is False


def test_rerun_of_another_users_run_is_not_found(services: ToolServices) -> None:
    run = services.history.record(weekly_entry("alice"))

    with as_user("bob"):
        answer = history(rerun_id=run.id)

    assert answer.status == "error" and answer.error is not None
    assert answer.error.code == ErrorCode.NOT_FOUND.value
    assert answer.rerun is None
    assert refusals(services) == [("bob", "history", ErrorCode.NOT_FOUND)]


# --------------------------------------------------------------------------------------------------
# feedback: stored, owner-scoped, and never auto-trusted
# --------------------------------------------------------------------------------------------------


def test_feedback_is_stored_against_the_callers_own_run(services: ToolServices) -> None:
    run = services.history.record(weekly_entry("alice"))

    with as_user("alice"):
        answer = feedback(query_id=f"q_{run.id}", ok=False, corrected="status S means suspended")

    assert answer.status == "ok", answer.error
    assert answer.saved is not None
    assert (answer.saved.user_id, answer.saved.run_id, answer.saved.ok) == ("alice", run.id, False)
    # Stored, and readable by the queue that will consume it later.
    assert [
        (row.id, row.user_id, row.run_id, row.ok, row.corrected)
        for row in services.feedback.recent()
    ] == [(answer.saved.id, "alice", run.id, False, "status S means suspended")]
    assert services.feedback.count() == 1


def test_feedback_is_a_claim_not_a_change(services: ToolServices) -> None:
    # The queue records what the caller said; nothing here edits the stored run or the dictionary.
    run = services.history.record(weekly_entry("alice"))
    with as_user("alice"):
        feedback(query_id=str(run.id), ok=False, corrected="the rounding is wrong")
    stored = services.history.get(run.id)
    assert stored.frozen_sql == run.frozen_sql
    assert stored.applied_fixes == run.applied_fixes


def test_feedback_about_another_users_run_is_not_found_and_audited(
    services: ToolServices,
) -> None:
    run = services.history.record(weekly_entry("alice"))

    with as_user("bob"):
        theirs = feedback(query_id=str(run.id), ok=True)
        missing = feedback(query_id="999999", ok=True)

    assert theirs.status == missing.status == "error"
    assert theirs.error is not None and missing.error is not None
    assert theirs.error.code == ErrorCode.NOT_FOUND.value
    assert theirs.error.model_dump_json() == missing.error.model_dump_json()
    assert services.feedback.count() == 0, "a refused rating must not reach the queue"
    assert refusals(services) == [
        ("bob", "feedback", ErrorCode.NOT_FOUND),
        ("bob", "feedback", ErrorCode.NOT_FOUND),
    ]


def test_feedback_with_an_id_this_server_never_issued_is_not_found(
    services: ToolServices,
) -> None:
    with as_user("alice"):
        answer = feedback(query_id="h_8f21", ok=True)

    assert answer.status == "error" and answer.error is not None
    assert answer.error.code == ErrorCode.NOT_FOUND.value
    assert "history" in answer.error.fix
    assert refusals(services) == [("alice", "feedback", ErrorCode.NOT_FOUND)]


def test_feedback_text_is_scrubbed_before_it_is_stored(services: ToolServices) -> None:
    run = services.history.record(weekly_entry("alice"))

    with as_user("alice"):
        answer = feedback(
            query_id=str(run.id),
            ok=False,
            corrected="use password=hunter2hunter2 in the account instead",
        )

    assert answer.saved is not None
    assert answer.saved.corrected is not None
    assert "hunter2hunter2" not in answer.saved.corrected
    assert "[REDACTED]" in answer.saved.corrected


# --------------------------------------------------------------------------------------------------
# no stores configured: an honest answer, not an empty result
# --------------------------------------------------------------------------------------------------


def test_tools_say_not_implemented_when_the_server_was_built_without_stores() -> None:
    with as_user("alice"):
        answers = (
            get_more("h_1"),
            history(),
            feedback(query_id="1", ok=True),
        )

    for answer in answers:
        assert answer.status == "not_implemented"
        assert answer.user_id == "alice"
        assert answer.problem and answer.fix, "an unavailable tool must say what is missing"
        assert answer.error is None


# --------------------------------------------------------------------------------------------------
# one real MCP roundtrip: the identity the handler sees is the caller's
# --------------------------------------------------------------------------------------------------

TOKENS: Final = (
    TokenEntry(token="tok-alice", user_id="alice", display_name="Alice"),
    TokenEntry(token="tok-bob", user_id="bob", display_name="Bob"),
)


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
async def test_over_mcp_each_caller_sees_only_their_own_rows(
    services: ToolServices, endpoint: str
) -> None:
    handle = services.handles.put(owner="alice", source="crm", table=numbered(5))
    run = services.history.record(weekly_entry("alice"))

    async with mcp_client(endpoint, TOKENS[1].token) as bob:
        stolen = await bob.call_tool(
            "get_more", {"result_handle": handle.handle, "offset": 0, "limit": 10}
        )
        bob_history = await bob.call_tool("history", {"limit": 10})
        bob_feedback = await bob.call_tool("feedback", {"query_id": str(run.id), "ok": True})

    stolen_body = stolen.structured_content
    assert stolen_body is not None and stolen_body["status"] == "error"
    assert stolen_body["error"]["code"] == ErrorCode.NOT_FOUND.value
    assert stolen_body["user_id"] == "bob"
    assert bob_history.structured_content is not None
    assert bob_history.structured_content["runs"] == []
    assert bob_feedback.structured_content is not None
    assert bob_feedback.structured_content["status"] == "error"

    async with mcp_client(endpoint, TOKENS[0].token) as alice:
        page = await alice.call_tool(
            "get_more", {"result_handle": handle.handle, "offset": 0, "limit": 10}
        )
        alice_history = await alice.call_tool("history", {"limit": 10})
        alice_feedback = await alice.call_tool("feedback", {"query_id": f"q_{run.id}", "ok": True})

    page_body = page.structured_content
    assert page_body is not None and page_body["status"] == "ok"
    assert [row["row_no"] for row in page_body["page"]["rows"]] == [0, 1, 2, 3, 4]
    assert alice_history.structured_content is not None
    assert [item["id"] for item in alice_history.structured_content["runs"]] == [run.id]
    assert alice_feedback.structured_content is not None
    assert alice_feedback.structured_content["saved"]["run_id"] == run.id
    assert services.feedback.count() == 1
    assert refusals(services) == [
        ("bob", "get_more", ErrorCode.NOT_FOUND),
        ("bob", "feedback", ErrorCode.NOT_FOUND),
    ]
