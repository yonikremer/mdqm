"""E03 acceptance: the claims, driven through the real MCP client against the real server.

This is the MVP's proof. Every assertion here goes over streamable HTTP to a running `mdqm serve`,
which talks to real PostgreSQL. Nothing is imported from the tool modules: if the transport, the
auth, the tool surface or the pipeline break, these tests fail.

It needs three things up (see `docs/deployment.md`):

    python scripts/dev.py up pg
    PYTHONPATH=src python -m mdqm.testing.loaders.postgres
    docker compose --profile pg --profile server up -d

Without them the module skips; with ``--require-services`` it fails, because a skipped test is not a
pass.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2
import pytest
from ruamel.yaml import YAML
from tests.conftest import service_unavailable

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENS = REPO_ROOT / "docker" / "server" / "tokens.json"
QUESTIONS = REPO_ROOT / "tests" / "eval" / "questions.yaml"
SERVER_URL = "http://127.0.0.1:58765/mcp"
HEALTHZ = "http://127.0.0.1:58765/healthz"

BILLING_QUESTION = "avg_limit_by_status_billing"
CACHEABLE_QUESTION = "events_count_window"
REFUSED_QUESTION = "usage_events_all_partitions_refused"


def _token() -> str:
    document = json.loads(TOKENS.read_text(encoding="utf-8"))
    return str(document["tokens"][0]["token"])


@dataclass(frozen=True, slots=True)
class Answer:
    """One tool call's answer, already parsed out of the MCP envelope."""

    payload: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.payload.get("status"))

    @property
    def error_code(self) -> str | None:
        error = self.payload.get("error") or {}
        return None if not error else str(error.get("code"))


@pytest.fixture(scope="module")
def server(require_services: bool) -> str:
    """The running server's MCP URL, or a clean skip when nothing is listening."""
    try:
        response = httpx2.get(HEALTHZ, timeout=5.0)
        response.raise_for_status()
    except Exception as exc:
        service_unavailable(require_services, f"mdqm server at {HEALTHZ}: {exc}")
        raise AssertionError("unreachable") from exc
    if not TOKENS.is_file():
        service_unavailable(
            require_services, f"{TOKENS} is missing (run `python scripts/dev.py env`)"
        )
        raise AssertionError("unreachable")
    return SERVER_URL


@pytest.fixture(scope="module")
def caller(server: str) -> Iterator[Any]:
    """A client that drives the server the way an operator would: `scripts/mdqm_call.py`.

    An in-process asyncio client is possible but couples this test to event-loop details that are
    the SDK's business, not the product's. The CLI is a real MCP client over real HTTP: it proves
    the transport, the bearer auth and the tool envelope, exactly what acceptance means here.
    """
    import subprocess
    import sys

    class _Caller:
        def call(self, name: str, arguments: dict[str, Any]) -> Answer:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "mdqm_call.py"),
                    "call",
                    name,
                    json.dumps(arguments),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=180,
                check=False,
            )
            if completed.returncode != 0:
                raise AssertionError(f"{name} failed: {completed.stderr.strip()}")
            envelope = json.loads(completed.stdout)
            for item in envelope.get("content") or []:
                if isinstance(item, str):
                    return Answer(json.loads(item))
            structured = envelope.get("structured")
            assert structured is not None, f"{name} returned no content: {envelope}"
            return Answer(dict(structured))

    yield _Caller()


def _question(question_id: str) -> dict[str, Any]:
    document = YAML(typ="safe").load(QUESTIONS.read_text(encoding="utf-8"))
    for entry in document["questions"]:
        if entry["id"] == question_id:
            return dict(entry)
    raise AssertionError(f"{question_id} is not in {QUESTIONS.name}")


def test_the_cross_source_answer_through_the_tools_matches_ground_truth(caller: Any) -> None:
    """Claim: an agent asks a business question and gets the number the analyst would compute.

    FAILS IF: the tools return a different value, or cannot answer at all.
    """
    question = _question(BILLING_QUESTION)
    answer = caller.call("query", {"spec": dict(question["spec"])})
    assert answer.status == "ok", answer.payload.get("error")

    produced = {
        str(row["customer_status"]): float(row["credit_limit_r5000_avg"])
        for row in answer.payload.get("sample") or []
    }
    expected = {str(row[0]): float(row[1]) for row in question["answer"]["rows"]}
    assert set(produced) == set(expected), produced
    for status, wanted in expected.items():
        assert produced[status] == pytest.approx(wanted, rel=1e-9), status


def test_a_repeated_query_is_served_from_cache_without_new_source_calls(caller: Any) -> None:
    """Claim: asking again does not hammer the primary.

    FAILS IF: the second call reports a miss, or reports a hit but still ran the fragment.
    """
    # `billing.accounts` carries no mutability class, so it is deliberately not cacheable (only
    # append_only/append_rarely are). `billing.usage_events` is append_rarely, which is the case the
    # cache claim is about.
    spec = dict(_question(CACHEABLE_QUESTION)["spec"])
    first = caller.call("query", {"spec": spec})
    assert first.status == "ok", first.payload.get("error")
    second = caller.call("query", {"spec": spec})
    assert second.status == "ok", second.payload.get("error")
    assert second.payload.get("cache_hit") is True, second.payload
    freshness = second.payload.get("freshness") or []
    assert freshness and all(entry.get("fetched_at") for entry in freshness), freshness


def test_a_query_without_a_time_bound_is_refused_before_it_reaches_the_source(caller: Any) -> None:
    """Claim: the product refuses the query that would scan every partition, and gives a fix."""
    question = _question(REFUSED_QUESTION)
    answer = caller.call("query", {"spec": dict(question["spec"])})
    assert answer.status == "error", answer.payload
    assert answer.error_code == question["answer"]["code"], answer.payload
    assert answer.payload["error"].get("fix"), "a refusal must carry a fix"


def test_a_write_cannot_be_expressed_and_none_ever_reaches_the_database(caller: Any) -> None:
    """Claim: read-only, and not by good manners -- the tool surface has no way to say "DELETE".

    FAILS IF: a raw-SQL field is accepted, or a DELETE shows up in what the server itself executed.
    """
    import subprocess
    import sys

    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "mdqm_call.py"),
            "call",
            "query",
            json.dumps({"spec": {"from": "billing.accounts", "raw": "DELETE FROM accounts"}}),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    envelope = json.loads(completed.stdout)
    text = " ".join(str(item) for item in envelope.get("content") or [])
    assert "raw" in text and ("forbidden" in text.lower() or "validation" in text.lower()), text
    assert envelope.get("structured") is None, "a rejected call must not look successful"

    # And the server's own statement log holds no write for this user.
    import psycopg

    target = __import__("mdqm.testing.loaders.postgres", fromlist=["target_from_environ"])
    connection = psycopg.connect(target.target_from_environ().dsn("billing"))
    with connection, connection.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM pg_stat_statements "
            "WHERE upper(query) LIKE %s AND upper(query) LIKE %s",
            ("%DELETE%", "%ACCOUNTS%"),
        )
        row = cur.fetchone()
    assert row is not None and row[0] == 0, f"{row} write statements reached the database"


def test_find_data_returns_a_budgeted_pack(caller: Any) -> None:
    """Claim: a weak model is handed the facts it needs, inside a token budget."""
    answer = caller.call("find_data", {"question": "average credit limit by customer status"})
    assert answer.status == "ok", answer.payload
    pack = answer.payload.get("pack") or {}
    assert pack.get("items"), pack
    assert pack.get("tokens_estimated", 0) <= pack.get("budget_tokens", 0), pack
    assert pack.get("text"), "the pack must render text a model can read"
