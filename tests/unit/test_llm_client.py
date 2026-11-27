"""Tests for the model client (S10): closed JSON output, retries, cache, probe, no key leaks.

Nothing here touches a socket: the endpoint is an ``httpx.MockTransport`` handler, time is a
:class:`FixedClock`, and sleeping is a recorder, so retry/backoff behaviour is asserted instead
of waited for. Table-driven where the shape repeats.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError

from mdqm.config import EnvSecretProvider, ModelSettings, PathsSettings, SecretNotFound, Settings
from mdqm.llm import (
    CACHE_FILE_NAME,
    MODEL_ENDPOINT,
    MODEL_NOT_CONFIGURED,
    MODEL_OUTPUT_INVALID,
    MODEL_TIMEOUT,
    MODEL_UNAVAILABLE,
    FakeLLM,
    ModelClient,
    ModelError,
    ResponseCache,
    StructuredLLM,
    context_window,
    probe_model_endpoint,
    structured_output_state,
)

# A throwaway value: the tests assert it never appears in a log line or an exception.
KEY = "throwaway-model-key-5d3e"
NOW = datetime(2026, 4, 1, 9, 30, tzinfo=UTC)
BASE_URL = "http://model.local/v1"
MODEL = "glm-test"


class TableList(BaseModel):
    """The closed output shape a pipeline stage would ask for."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    names: list[str]


class FixedClock:
    """Time under test control."""

    def __init__(self, now: datetime = NOW) -> None:
        self.current = now

    def now(self) -> datetime:
        return self.current


class Sleeps:
    """Records backoff delays instead of sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


Scripted = httpx.Response | BaseException | Callable[[httpx.Request], httpx.Response]


class Server:
    """A scripted OpenAI-compatible server: answers in order, records every request."""

    def __init__(self, *script: Scripted) -> None:
        self._script = list(script)
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(request.content) if request.content else {})
        if not self._script:
            raise AssertionError(
                f"the server got request {len(self.requests)}, which was not scripted"
            )
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(request)
        return item

    @property
    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]


def reply(content: str | None, *, status: int = 200) -> httpx.Response:
    """A chat-completions answer carrying ``content``."""
    return httpx.Response(
        status,
        json={"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]},
    )


def tool_reply(name: str = "probe_echo") -> httpx.Response:
    """A reply that called a tool, which is what the tool-calling probe looks for."""
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": name, "arguments": "{}"},
                            }
                        ],
                    },
                }
            ]
        },
    )


def error(status: int, message: str | None = None) -> httpx.Response:
    """A refusal in the OpenAI error shape."""
    body = {"error": {"message": message, "type": "invalid_request_error"}} if message else {}
    return httpx.Response(status, json=body)


def models_reply(**entry: Any) -> httpx.Response:
    return httpx.Response(200, json={"object": "list", "data": [{"id": MODEL, **entry}]})


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Configured endpoint; the API key is here as a throwaway dev value only."""
    return Settings(
        model=ModelSettings(
            base_url=BASE_URL,
            model=MODEL,
            api_key=SecretStr(KEY),
            timeout_s=7.5,
            max_retries=2,
        ),
        paths=PathsSettings(cache_dir=tmp_path / "cache"),
    )


def make_client(
    settings: Settings,
    *script: Scripted,
    clock: FixedClock | None = None,
    sleeps: Sleeps | None = None,
    **kwargs: Any,
) -> tuple[ModelClient, Server]:
    """A client wired to a scripted server: no socket, no sleeping, fixed time."""
    server = Server(*script)
    client = ModelClient(
        settings,
        transport=server.transport(),
        clock=clock or FixedClock(),
        sleep=sleeps or Sleeps(),
        **kwargs,
    )
    return client, server


# --- acceptance: invalid JSON, then valid JSON on retry 1 ------------------------------------


def test_invalid_json_then_valid_succeeds_on_retry_one(settings: Settings) -> None:
    sleeps = Sleeps()
    client, server = make_client(
        settings, reply("{not json"), reply('{"names": ["crm"]}'), sleeps=sleeps
    )

    result = client.structured("list the tables", TableList)

    assert result.value == TableList(names=["crm"])
    assert [attempt.outcome for attempt in result.attempts] == ["invalid_json", "ok"]
    assert [attempt.number for attempt in result.attempts] == [1, 2]
    assert result.attempts[0].retry_delay_s == 1.0
    assert result.attempts[1].retry_delay_s is None
    assert result.cached is False
    assert result.model == MODEL
    assert result.prompt_version == "1"
    assert result.fetched_at == NOW
    assert sleeps.delays == [1.0]
    assert len(server.requests) == 2

    # The validation error was sent back, with the model's own rejected text, not silently retried.
    messages = server.bodies[1]["messages"]
    assert messages[1] == {"role": "user", "content": "list the tables"}
    assert messages[-2] == {"role": "assistant", "content": "{not json"}
    assert "not valid JSON" in messages[-1]["content"]


def test_invalid_schema_then_valid_succeeds_and_names_the_field(settings: Settings) -> None:
    client, server = make_client(settings, reply('{"name": ["crm"]}'), reply('{"names": ["crm"]}'))

    result = client.structured("list the tables", TableList)

    assert result.value.names == ["crm"]
    assert [attempt.outcome for attempt in result.attempts] == ["invalid_schema", "ok"]
    first_problem = result.attempts[0].problem
    assert first_problem is not None and first_problem.startswith("names")
    assert "names" in server.bodies[1]["messages"][-1]["content"]


def test_code_fences_are_formatting_not_a_wrong_answer(settings: Settings) -> None:
    client, _ = make_client(settings, reply('```json\n{"names": ["erp"]}\n```'))

    result = client.structured("list the tables", TableList)

    assert result.value == TableList(names=["erp"])
    assert [attempt.outcome for attempt in result.attempts] == ["ok"]


@pytest.mark.parametrize(
    ("body", "outcome"),
    [
        ({"choices": []}, "empty_response"),
        ({"choices": [{"message": {"role": "assistant", "content": ""}}]}, "empty_response"),
        ({"object": "weird"}, "empty_response"),
    ],
)
def test_missing_reply_content_is_reported_then_retried(
    settings: Settings, body: dict[str, Any], outcome: str
) -> None:
    client, server = make_client(
        settings, httpx.Response(200, json=body), reply('{"names": ["crm"]}')
    )

    result = client.structured("list the tables", TableList)

    assert result.attempts[0].outcome == outcome
    assert len(server.requests) == 2


def test_exhausted_output_retries_raise_model_output_invalid(settings: Settings) -> None:
    sleeps = Sleeps()
    client, server = make_client(
        settings, reply("{bad"), reply("{bad"), reply("{bad"), sleeps=sleeps
    )

    with pytest.raises(ModelError) as caught:
        client.structured("list the tables", TableList)

    error_out = caught.value
    assert error_out.code == MODEL_OUTPUT_INVALID
    assert "3 attempt(s)" in error_out.problem
    assert [attempt.outcome for attempt in error_out.attempts] == ["invalid_json"] * 3
    assert sleeps.delays == [1.0, 2.0]
    assert len(server.requests) == 3
    assert error_out.to_error_response().code == MODEL_OUTPUT_INVALID


# --- acceptance: a cached call makes no second HTTP request ----------------------------------


def test_cached_call_makes_no_second_request(settings: Settings) -> None:
    client, server = make_client(settings, reply('{"names": ["crm"]}'))

    first = client.structured("list the tables", TableList)
    second = client.structured("list the tables", TableList)

    assert len(server.requests) == 1
    assert second.value == first.value
    assert second.cached is True
    assert second.attempts == ()
    assert second.fetched_at == first.fetched_at
    assert (settings.paths.cache_dir / CACHE_FILE_NAME).is_file()


def test_changed_input_or_prompt_version_is_a_miss(settings: Settings) -> None:
    client, server = make_client(settings, reply('{"names": ["crm"]}'), reply('{"names": ["crm"]}'))

    client.structured("list the tables", TableList)
    client.structured("list the tables", TableList, prompt_version="2")

    assert len(server.requests) == 2
    assert [body["messages"][1]["content"] for body in server.bodies] == ["list the tables"] * 2


def test_cache_key_is_deterministic_and_never_holds_the_key(settings: Settings) -> None:
    client, _ = make_client(settings)
    other = ModelClient(settings, transport=Server().transport())
    key = client.cache_key(prompt="p", schema=TableList, system="s")

    assert key == other.cache_key(prompt="p", schema=TableList, system="s")
    assert key != other.cache_key(prompt="p", schema=TableList, system="other")
    assert KEY not in key


def test_outdated_cache_row_is_refetched_and_never_served(settings: Settings) -> None:
    client, server = make_client(settings, reply('{"names": ["crm"]}'))
    key = client.cache_key(prompt="list the tables", schema=TableList)
    client.cache.put(
        key,
        content='{"names": ["crm"], "surprise": true}',  # extra=forbid: no longer valid
        model=MODEL,
        prompt_version="1",
        fetched_at=NOW,
    )

    result = client.structured("list the tables", TableList)

    assert len(server.requests) == 1
    assert result.cached is False
    assert result.value == TableList(names=["crm"])


def test_cache_row_with_a_broken_timestamp_reads_as_a_miss(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite3")
    cache.put("k", content="{}", model="m", prompt_version="1", fetched_at=NOW)
    import sqlite3

    with sqlite3.connect(cache.path) as connection:
        connection.execute("UPDATE llm_cache SET fetched_at = 'not a timestamp' WHERE key = 'k'")

    assert cache.get("k") is None


# --- acceptance: no API key in logs or exceptions --------------------------------------------


def test_api_key_never_reaches_logs_or_exceptions(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    # The server echoes the key back in its error text: the worst case, and still no leak.
    client, server = make_client(
        settings, error(401, f"invalid key {KEY}"), reply("{bad"), reply('{"names": ["crm"]}')
    )

    with pytest.raises(ModelError) as caught:
        client.structured("list the tables", TableList)

    failure = caught.value
    assert KEY in server.requests[0].headers["authorization"]  # it really was sent
    for text in (str(failure), repr(failure), repr(client), str(failure.to_error_response())):
        assert KEY not in text
    assert "[REDACTED]" in failure.problem
    assert failure.code == "MODEL_HTTP_ERROR"
    assert failure.status_code == 401
    assert failure.__cause__ is None and failure.__context__ is None

    # A successful call afterwards logs attempts, model names and outcomes, never the key.
    client2, _ = make_client(settings, reply("{bad"), reply('{"names": ["crm"]}'))
    result = client2.structured("list the tables", TableList)
    assert "names" in result.model_dump_json()
    logged = "\n".join(record.getMessage() + str(record.__dict__) for record in caplog.records)
    assert KEY not in logged
    assert "list the tables" not in logged  # prompts are not logging material either


def test_transport_failure_message_is_ours_and_carries_no_key(settings: Settings) -> None:
    sleeps = Sleeps()
    client, server = make_client(
        settings,
        httpx.ConnectError(f"tunnel refused for {KEY}"),
        httpx.ConnectError(f"tunnel refused for {KEY}"),
        httpx.ConnectError(f"tunnel refused for {KEY}"),
        sleeps=sleeps,
    )

    with pytest.raises(ModelError) as caught:
        client.structured("list the tables", TableList)

    failure = caught.value
    assert failure.code == MODEL_UNAVAILABLE
    assert KEY not in str(failure) and KEY not in repr(failure)
    assert failure.__cause__ is None and failure.__suppress_context__ is True
    assert len(server.requests) == 3
    assert sleeps.delays == [1.0, 2.0]


def test_missing_secret_is_a_configuration_error_naming_the_reference(settings: Settings) -> None:
    settings = settings.model_copy(
        update={"model": settings.model.model_copy(update={"api_key": None})}
    )
    client, server = make_client(
        settings, secret_provider=EnvSecretProvider(environ={}), secret_name="model_api_key"
    )

    with pytest.raises(ModelError) as caught:
        client.structured("list the tables", TableList)

    assert caught.value.code == MODEL_NOT_CONFIGURED
    assert "model_api_key" in caught.value.problem
    assert "MDQM_SECRET_MODEL_API_KEY" in caught.value.problem
    assert server.requests == []


def test_secret_provider_supplies_the_bearer_token(settings: Settings) -> None:
    settings = settings.model_copy(
        update={"model": settings.model.model_copy(update={"api_key": None})}
    )
    provider = EnvSecretProvider(environ={"MDQM_SECRET_MODEL_API_KEY": KEY})
    client, server = make_client(settings, reply('{"names": ["crm"]}'), secret_provider=provider)

    client.structured("list the tables", TableList)

    assert server.requests[0].headers["authorization"] == f"Bearer {KEY}"


def test_settings_secret_wins_over_the_provider(settings: Settings) -> None:
    provider = EnvSecretProvider(environ={"MDQM_SECRET_MODEL_API_KEY": "from-the-environment"})
    client, server = make_client(settings, reply('{"names": ["crm"]}'), secret_provider=provider)

    client.structured("list the tables", TableList)

    assert server.requests[0].headers["authorization"] == f"Bearer {KEY}"


# --- acceptance: the probe reports a model_endpoint Capabilities entry ------------------------


def test_probe_reports_a_capable_endpoint(settings: Settings) -> None:
    clock = FixedClock()
    client, server = make_client(
        settings,
        reply("ok"),  # reachable
        reply('{"ok": true}'),  # structured output honoured
        tool_reply(),  # tool calling works
        models_reply(context_length=8192),  # context window exposed
        clock=clock,
    )

    capabilities = probe_model_endpoint(client)

    assert capabilities.source == MODEL_ENDPOINT
    assert capabilities.retrieved_at == NOW
    assert capabilities.supports("reachable")
    assert capabilities.supports("structured_output")
    assert structured_output_state(capabilities) == "true"
    assert capabilities.supports("tool_calling")
    assert context_window(capabilities) == 8192
    assert client.structured_output is True
    assert server.paths == ["/v1/chat/completions"] * 3 + ["/v1/models"]


def test_probe_reports_a_refusing_endpoint(settings: Settings) -> None:
    client, _ = make_client(
        settings,
        reply("ok"),
        error(400, "unknown field: response_format"),
        error(400, "unknown field: tools"),
        error(404, "not found"),
    )

    capabilities = probe_model_endpoint(client)

    assert capabilities.supports("reachable")
    assert capabilities.get("structured_output").supported is False
    assert structured_output_state(capabilities) == "false"
    assert capabilities.get("tool_calling").supported is False
    assert capabilities.get("context_window").supported is False
    assert context_window(capabilities) is None
    assert client.structured_output is False


def test_probe_reports_an_unreachable_endpoint_without_firing_more_calls(
    settings: Settings,
) -> None:
    client, server = make_client(settings, httpx.ConnectError("no route"))

    capabilities = probe_model_endpoint(client)

    assert capabilities.get("reachable").supported is False
    assert capabilities.get("structured_output").supported is None
    assert structured_output_state(capabilities) == "unknown"
    assert capabilities.get("tool_calling").supported is None
    assert len(server.requests) == 1
    assert client.structured_output is None


@pytest.mark.parametrize(
    ("answer", "expected", "state"),
    [
        (reply('{"ok": true}'), True, "true"),
        (reply("I cannot do that"), None, "unknown"),  # accepted but ignored: not a claim
        (error(400, "response_format is not supported"), False, "false"),
        (error(401, "bad credential"), None, "unknown"),  # auth says nothing about the feature
        (error(500, "boom"), None, "unknown"),
        (httpx.ReadTimeout("slow"), None, "unknown"),
    ],
)
def test_probe_structured_output_verdicts(
    settings: Settings, answer: Scripted, expected: bool | None, state: str
) -> None:
    client, _ = make_client(settings, reply("ok"), answer, error(404, "no tools"), error(404, "no"))

    capabilities = probe_model_endpoint(client)

    assert capabilities.get("structured_output").supported is expected
    assert structured_output_state(capabilities) == state


def test_probe_context_window_from_alternative_field_names(settings: Settings) -> None:
    client, _ = make_client(
        settings,
        reply("ok"),
        reply('{"ok": true}'),
        tool_reply(),
        models_reply(max_model_len=32768),
    )

    assert context_window(probe_model_endpoint(client)) == 32768


# --- response_format: used only when it is known to work ------------------------------------


@pytest.mark.parametrize(
    ("verdict", "expects_format"), [(True, True), (False, False), (None, False)]
)
def test_response_format_is_sent_only_when_supported(
    settings: Settings, verdict: bool | None, expects_format: bool
) -> None:
    client, server = make_client(settings, reply('{"names": ["crm"]}'), structured_output=verdict)

    client.structured("list the tables", TableList)

    body = server.bodies[0]
    assert ("response_format" in body) is expects_format
    if expects_format:
        response_format = body["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["name"] == "TableList"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["schema"]["properties"]["names"]["type"] == "array"
    # The schema is spelled out in the system prompt either way: instruct is the fallback.
    system_prompt = body["messages"][0]["content"]
    assert body["messages"][0]["role"] == "system"
    assert '"names"' in system_prompt
    assert body["temperature"] == 0
    assert body["model"] == MODEL


def test_refused_response_format_downgrades_the_client_for_later_calls(settings: Settings) -> None:
    client, server = make_client(
        settings,
        error(400, "response_format is not supported by this model"),
        reply('{"names": ["crm"]}'),
        structured_output=True,
    )

    result = client.structured("list the tables", TableList)

    assert result.value == TableList(names=["crm"])
    assert [attempt.outcome for attempt in result.attempts] == ["response_format_rejected", "ok"]
    assert "response_format" in server.bodies[0]
    assert "response_format" not in server.bodies[1]
    assert client.structured_output is False


def test_system_prompt_keeps_the_caller_text_first(settings: Settings) -> None:
    client, server = make_client(settings, reply('{"names": ["crm"]}'))

    client.structured("list the tables", TableList, system="You are a careful data analyst.")

    system_prompt = server.bodies[0]["messages"][0]["content"]
    assert system_prompt.startswith("You are a careful data analyst.")
    assert "JSON" in system_prompt


# --- retries, backoff and timeouts -----------------------------------------------------------


@pytest.mark.parametrize(
    ("script", "max_retries", "backoff_s", "expected_delays", "code", "requests"),
    [
        (
            [httpx.ReadTimeout("t"), error(503, "upstream"), reply('{"names": []}')],
            2,
            1.0,
            [1.0, 2.0],
            None,
            3,
        ),
        (
            [httpx.ReadTimeout("t"), httpx.ReadTimeout("t"), httpx.ReadTimeout("t")],
            2,
            1.0,
            [1.0, 2.0],
            MODEL_TIMEOUT,
            3,
        ),
        (
            [error(503, "upstream"), error(401, "bad credential")],
            2,
            1.0,
            [1.0],
            "MODEL_HTTP_ERROR",
            2,
        ),
        ([httpx.ReadTimeout("t")], 0, 1.0, [], MODEL_TIMEOUT, 1),
        (
            [httpx.ReadTimeout("t"), httpx.ReadTimeout("t"), reply('{"names": []}')],
            2,
            20.0,
            [20.0, 30.0],  # doubling, capped at MAX_BACKOFF_S
            None,
            3,
        ),
        ([error(429, "slow down"), reply('{"names": []}')], 1, 0.0, [0.0], None, 2),
    ],
)
def test_retry_backoff_and_budget(
    settings: Settings,
    script: list[Scripted],
    max_retries: int,
    backoff_s: float,
    expected_delays: list[float],
    code: str | None,
    requests: int,
) -> None:
    settings = settings.model_copy(
        update={"model": settings.model.model_copy(update={"max_retries": max_retries})}
    )
    sleeps = Sleeps()
    client, server = make_client(settings, *script, sleeps=sleeps, backoff_s=backoff_s)

    if code is None:
        client.structured("list the tables", TableList)
    else:
        with pytest.raises(ModelError) as caught:
            client.structured("list the tables", TableList)
        assert caught.value.code == code
        assert len(caught.value.attempts) == requests

    assert sleeps.delays == expected_delays
    assert len(server.requests) == requests


def test_timeout_from_config_reaches_the_transport(settings: Settings) -> None:
    seen: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"]["read"])
        return reply('{"names": ["crm"]}')

    client = ModelClient(settings, transport=httpx.MockTransport(handler), clock=FixedClock())
    assert client.timeout.read == 7.5

    client.structured("list the tables", TableList)

    assert seen == [7.5]


@pytest.mark.parametrize(
    ("status", "fix_fragment"),
    [
        (401, "secret reference"),
        (403, "secret reference"),
        (429, "rate limiting"),
        (404, "OpenAI-compatible API root"),
        (502, "unhealthy"),
    ],
)
def test_status_codes_carry_an_exact_fix(
    settings: Settings, status: int, fix_fragment: str
) -> None:
    settings = settings.model_copy(
        update={"model": settings.model.model_copy(update={"max_retries": 0})}
    )
    client, _ = make_client(settings, error(status, "nope"))

    with pytest.raises(ModelError) as caught:
        client.structured("list the tables", TableList)

    assert caught.value.status_code == status
    assert fix_fragment in caught.value.fix


def test_non_json_success_body_is_reported_as_a_wrong_base_url(settings: Settings) -> None:
    client, _ = make_client(settings, httpx.Response(200, text="<html>proxy</html>"))

    with pytest.raises(ModelError) as caught:
        client.structured("list the tables", TableList)

    assert caught.value.code == "MODEL_HTTP_ERROR"
    assert "non-JSON" in caught.value.problem
    assert "API root" in caught.value.fix


@pytest.mark.parametrize("model_settings", [{}, {"base_url": BASE_URL}, {"model": MODEL}])
def test_unconfigured_endpoint_fails_before_any_request(
    tmp_path: Path, model_settings: dict[str, Any]
) -> None:
    settings = Settings(
        model=ModelSettings(**model_settings),
        paths=PathsSettings(cache_dir=tmp_path / "cache"),
    )
    server = Server()

    with pytest.raises(ModelError) as caught:
        ModelClient(settings, transport=server.transport())

    assert caught.value.code == MODEL_NOT_CONFIGURED
    assert server.requests == []


def test_client_is_a_context_manager_and_repr_hides_the_key(settings: Settings) -> None:
    server = Server()
    with ModelClient(settings, transport=server.transport()) as client:
        text = repr(client)
    assert KEY not in text
    assert MODEL in text


# --- FakeLLM ---------------------------------------------------------------------------------


def test_fake_llm_returns_scripted_instances_and_records_calls() -> None:
    llm = FakeLLM(TableList(names=["crm"]))
    typed: StructuredLLM = llm

    result = typed.structured("list the tables", TableList, system="be brief", prompt_version="7")

    assert result.value == TableList(names=["crm"])
    assert result.model == "fake-llm"
    assert result.prompt_version == "7"
    assert [call.schema_name for call in llm.calls] == ["TableList"]
    assert llm.calls[0].prompt == "list the tables"
    assert llm.calls[0].system == "be brief"


@pytest.mark.parametrize(
    ("scripted", "expected"),
    [
        ({"names": ["erp"]}, ["erp"]),
        ('{"names": ["billing"]}', ["billing"]),
    ],
)
def test_fake_llm_validates_mappings_and_json_text(scripted: Any, expected: list[str]) -> None:
    llm = FakeLLM(scripted)

    assert llm.structured("list", TableList).value.names == expected


def test_fake_llm_rejects_a_wrong_shape_and_a_scripted_exception() -> None:
    llm = FakeLLM({"names": 3})

    with pytest.raises(ValidationError):
        llm.structured("list", TableList)

    boom = ModelError("nope", "fix it", code="MODEL_UNAVAILABLE")
    llm = FakeLLM(boom)
    with pytest.raises(ModelError):
        llm.structured("list", TableList)


def test_fake_llm_fails_loudly_when_the_script_runs_out() -> None:
    llm = FakeLLM()

    with pytest.raises(AssertionError, match="no reply left"):
        llm.structured("list", TableList)


def test_fake_llm_can_compute_a_reply_from_the_prompt() -> None:
    llm = FakeLLM(lambda prompt: {"names": [prompt]})

    assert llm.structured("crm", TableList).value.names == ["crm"]


def test_fake_llm_secrets_are_not_in_repr() -> None:
    llm = FakeLLM(TableList(names=[]))

    assert KEY not in repr(llm)


def test_secret_not_found_is_not_leaked_by_the_provider_stub() -> None:
    with pytest.raises(SecretNotFound) as caught:
        EnvSecretProvider(environ={}).get("model_api_key")
    assert KEY not in str(caught.value)
