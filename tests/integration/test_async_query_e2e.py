"""ASYNC-QUERY end to end: start a query, poll it, page its result -- over real HTTP.

This is the owner's ask driven the way an agent drives it: ``query`` with ``mode: "async"`` answers
with a ``query_id`` and the plan before any row exists, ``get_more(query_id=...)`` reports progress
while it runs and the answer when it is done, and the handle it hands back pages like any other.

It needs the same three things as ``test_acceptance_e2e.py``:

    python scripts/dev.py up pg
    PYTHONPATH=src python -m mdqm.testing.loaders.postgres
    docker compose --profile pg --profile server up -d --build

Without them the module skips; with ``--require-services`` it fails, because a skipped test is not a
pass. The server image must be built from this tree: a ``src/`` change needs a rebuild.
"""

from __future__ import annotations

import json
import time
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

#: The question the acceptance suite already proves runs: a windowed count on the partitioned
#: table, so the plan really has partitions to report and the answer really has rows.
QUESTION_ID = "events_count_window"

#: How long the poll may wait before the test calls it a failure. A query that runs longer than this
#: is a real problem on the fixture data, not a slow machine.
POLL_TIMEOUT_S = 120.0


def _spec() -> dict[str, Any]:
    """The fixture question's spec, read from the same file the acceptance suite uses."""
    document = YAML(typ="safe").load(QUESTIONS.read_text(encoding="utf-8"))
    for entry in document["questions"]:
        if entry["id"] == QUESTION_ID:
            return dict(entry["spec"])
    raise AssertionError(f"{QUESTION_ID} is not in {QUESTIONS.name}")


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
def caller(require_services: bool) -> Iterator[Any]:
    """A real MCP client: whether the server is listening decides skip or fail."""
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
                timeout=POLL_TIMEOUT_S,
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


def _poll(caller: Any, query_id: str) -> Answer:
    """Poll one job until it reaches a terminal state, or fail: no unbounded retrying."""
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        answer = caller.call("get_more", {"query_id": query_id})
        if answer.status not in {"queued", "running"}:
            return Answer(dict(answer.payload))
        time.sleep(1.0)
    raise AssertionError(f"{query_id} did not finish within {POLL_TIMEOUT_S:g}s")


def test_an_async_query_reports_the_plan_then_the_answer(caller: Any) -> None:
    """Claim: the caller learns the plan at once and the rows later, without asking twice.

    ``fresh`` is forced here on purpose. Without it the run can be answered from the segment cache,
    and a cache hit never asks the source for an EXPLAIN -- so no per-partition number exists to
    report and ``partitions_done`` stays ``None`` (honest, but it made this assertion depend on
    whether an earlier test had warmed the segment). The cache-hit shape has its own test below.
    """
    started = caller.call("query", {"spec": {**_spec(), "fresh": True}, "mode": "async"})
    assert started.status == "started", started.payload.get("error")
    query_id = started.payload.get("query_id")
    assert isinstance(query_id, str) and query_id
    plan = started.payload.get("plan") or {}
    assert plan.get("partitions_total"), plan
    assert started.payload.get("result_handle") is None

    finished = _poll(caller, query_id)
    assert finished.status == "done", finished.payload.get("error")
    handle = finished.payload.get("result_handle")
    assert isinstance(handle, str) and handle.startswith("h_")
    assert finished.payload.get("summary"), finished.payload
    progress = finished.payload.get("progress") or {}
    assert progress.get("state") == "done"
    assert progress.get("partitions_done") is not None, progress

    page = caller.call("get_more", {"result_handle": handle, "limit": 10})
    assert page.status == "ok", page.payload.get("error")
    assert page.payload.get("page", {}).get("rows") is not None


def test_the_blocking_shape_is_still_the_default(caller: Any) -> None:
    """Claim: nothing changes for a caller that does not ask for a job."""
    answer = caller.call("query", {"spec": _spec()})
    assert answer.status == "ok", answer.payload.get("error")
    assert answer.payload.get("result_handle")
    assert answer.payload.get("query_id") is None
