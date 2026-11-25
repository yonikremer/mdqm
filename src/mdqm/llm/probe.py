"""Probe an OpenAI-compatible endpoint: what does *this* server actually support?

The on-prem model is an unknown (AGENTS.md section 9), so every optional behaviour is measured
before a feature uses it, and an unmeasurable one reads as *unknown*, never as *supported*. The
result is a :class:`~mdqm.models.capabilities.Capabilities` entry whose ``source`` is
``model_endpoint``, so it travels with every other capability report instead of inventing a
second shape:

============================  =========================================================
capability                    how it is measured
============================  =========================================================
``reachable``                 a minimal chat call: any HTTP answer is reachable, a
                              transport error or a timeout is not
``structured_output``         a ``response_format``/``json_schema`` call: honoured JSON is
                              ``True``, an explicit refusal is ``False``, an ambiguous answer
                              (5xx, prose instead of JSON) stays ``None``
``tool_calling``              a forced ``tool_choice``: a matching ``tool_calls`` entry is
                              ``True``, a refusal is ``False``, silence is ``None``
``context_window``            ``GET /models`` when the server exposes it; the token count
                              lives in ``detail``, absent means unknown
============================  =========================================================

Every probe call is one attempt (``retries=0``): a probe must never turn into three slow calls.
Probe prompts are fixed, tiny and carry no organisation data.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Final

from mdqm.cache.freshness import Clock
from mdqm.llm.client import ModelClient
from mdqm.llm.errors import ModelError
from mdqm.models.capabilities import Capabilities, Capability

__all__ = [
    "MODEL_ENDPOINT",
    "PROBE_FUNCTION",
    "PROBE_SCHEMA",
    "context_window",
    "probe_model_endpoint",
    "structured_output_state",
]

_logger = logging.getLogger(__name__)

#: The ``Capabilities.source`` name of a probed model endpoint.
MODEL_ENDPOINT: Final = "model_endpoint"

#: The constant capability names this probe fills.
PROBE_FUNCTION: Final = "chat_completions"

#: Minimal closed schema for the structured-output probe (a server may reject unknown keywords).
PROBE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}

_TOOL: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": "probe_echo",
        "description": "Return the value you were given.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    },
}

#: Statuses that mean "I reject this request field", as opposed to "your credential is wrong".
_REFUSAL_STATUS: Final = frozenset({400, 404, 405, 415, 422})

#: Context-window keys seen in the wild. Order matters only for determinism.
_CONTEXT_KEYS: Final = (
    "context_length",
    "context_window",
    "max_model_len",
    "max_context_length",
    "n_ctx",
)


def structured_output_state(capabilities: Capabilities) -> str:
    """``"true"``, ``"false"`` or ``"unknown"``: the tri-state, spelled the way config does."""
    supported = capabilities.get("structured_output").supported
    if supported is None:
        return "unknown"
    return "true" if supported else "false"


def context_window(capabilities: Capabilities) -> int | None:
    """The measured context window in tokens, when the endpoint exposed one."""
    detail = capabilities.get("context_window").detail
    if not detail:
        return None
    try:
        return int(detail.split()[0])
    except (ValueError, IndexError):
        return None


def probe_model_endpoint(client: ModelClient, *, clock: Clock | None = None) -> Capabilities:
    """Measure the endpoint and record the verdicts (also on ``client.structured_output``).

    A probe never raises: an unreachable or refusing endpoint is a capability verdict, because
    that is what the caller has to act on. When the endpoint cannot be reached at all, the other
    features are reported unknown instead of firing three more doomed calls.
    """
    now = (clock or client.clock).now()
    reachable, reachable_detail = _probe_reachable(client)
    if reachable is False:
        skipped = Capability(detail="not probed: the endpoint is unreachable")
        client.structured_output = None
        return Capabilities(
            source=MODEL_ENDPOINT,
            adapter="openai_compatible",
            retrieved_at=now,
            capabilities={
                "reachable": Capability(supported=False, detail=reachable_detail),
                "structured_output": skipped,
                "tool_calling": skipped,
                "context_window": skipped,
            },
        )
    structured, structured_detail = _probe_structured_output(client)
    client.structured_output = structured
    tool_calling = _probe_tool_calling(client)
    context, context_detail = _probe_context_window(client)
    return Capabilities(
        source=MODEL_ENDPOINT,
        adapter="openai_compatible",
        retrieved_at=now,
        capabilities={
            "reachable": Capability(supported=True, detail=reachable_detail),
            "structured_output": Capability(supported=structured, detail=structured_detail),
            "tool_calling": tool_calling,
            "context_window": Capability(supported=context is not None, detail=context_detail),
        },
    )


def _call(client: ModelClient, **options: Any) -> dict[str, Any] | ModelError:
    """One probe call: the body, or the failure as a verdict input (never an exception)."""
    try:
        body: dict[str, Any] = client.chat(
            [{"role": "user", "content": "Reply with the single word: ok"}],
            options=options,
            retries=0,
        )
    except ModelError as exc:
        return exc
    return body


def _refused(result: dict[str, Any] | ModelError) -> bool:
    """True when the server explicitly refused this *feature*.

    Auth failures and rate limiting are excluded on purpose: they say nothing about whether the
    endpoint can do structured output, so they stay unknown.
    """
    return isinstance(result, ModelError) and result.status_code in _REFUSAL_STATUS


def _probe_reachable(client: ModelClient) -> tuple[bool | None, str]:
    result = _call(client, max_tokens=1)
    if isinstance(result, ModelError):
        if result.status_code is not None:
            return True, f"answered HTTP {result.status_code}"
        return False, f"no answer: {result.code}"
    return True, "answered HTTP 200"


def _probe_structured_output(client: ModelClient) -> tuple[bool | None, str]:
    result = _call(
        client,
        max_tokens=64,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "probe", "schema": PROBE_SCHEMA, "strict": True},
        },
    )
    if isinstance(result, ModelError):
        if _refused(result):
            return False, f"refused with HTTP {result.status_code}"
        return None, f"inconclusive: {result.code}"
    text = _reply_text(result)
    if text is None:
        return None, "inconclusive: the reply had no content"
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return None, "inconclusive: accepted response_format but replied with non-JSON"
    return True, "json_schema accepted and answered with JSON"


def _probe_tool_calling(client: ModelClient) -> Capability:
    result = _call(
        client,
        max_tokens=128,
        tools=[_TOOL],
        tool_choice={"type": "function", "function": {"name": "probe_echo"}},
    )
    if isinstance(result, ModelError):
        if _refused(result):
            return Capability(supported=False, detail=f"refused with HTTP {result.status_code}")
        return Capability(detail=f"inconclusive: {result.code}")
    if _called_tool(result, "probe_echo"):
        return Capability(supported=True, detail="forced tool_choice returned a tool_calls entry")
    return Capability(detail="inconclusive: no tool_calls entry in the reply")


def _probe_context_window(client: ModelClient) -> tuple[int | None, str]:
    try:
        body: Any = client.models()
    except ModelError as exc:
        return None, f"not exposed: {exc.code}"
    if not isinstance(body, dict):
        return None, "not exposed: /models did not answer with an object"
    for entry in _model_entries(body):
        for key in _CONTEXT_KEYS:
            value = entry.get(key)
            if isinstance(value, int) and value > 0:
                return value, f"{value} tokens (from /models field {key!r})"
    return None, "not exposed by /models"


def _model_entries(body: dict[str, Any]) -> list[dict[str, Any]]:
    data = body.get("data")
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    models = body.get("models")
    if isinstance(models, list):
        return [entry for entry in models if isinstance(entry, dict)]
    return [body]


def _reply_text(body: dict[str, Any]) -> str | None:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _called_tool(body: dict[str, Any], name: str) -> bool:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(calls, list):
        return False
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if isinstance(function, dict) and function.get("name") == name:
            return True
    return False
