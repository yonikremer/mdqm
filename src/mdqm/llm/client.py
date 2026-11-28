"""A minimal client for an on-prem, OpenAI-compatible ``/chat/completions`` endpoint.

No vendor SDK: ``httpx`` plus this module is the whole contract. The endpoint is unverified
(AGENTS.md section 9), so nothing about it is assumed beyond the six documented lines of the
chat-completions shape: ``POST {base_url}/chat/completions`` with ``model``, ``messages`` and
``temperature``, and a body whose ``choices[0].message.content`` holds the reply. Everything
optional is *probed* (:mod:`mdqm.llm.probe`) and falls back:

* ``response_format`` ``json_schema`` is used only when the probe says the server honours it;
  otherwise the schema is explained in the system prompt and validated in code all the same.
* A server that refuses ``response_format`` at call time downgrades the client to the fallback
  for the rest of its life, so the refusal costs one retry, not every call.
* Model name, context limits and timeouts come from config (:class:`mdqm.config.ModelSettings`),
  never from a constant in this file.

Determinism: ``temperature=0``; up to ``model.max_retries`` retries with exponential backoff for
transport errors, timeouts, 429/5xx and *invalid output* (the validation error is sent back to
the model, which is what makes a weak model recover); then a :class:`ModelError` carrying every
attempt. Answers are cached by input hash, so a re-run costs no call at all.

Secrets, prompts and completions never reach a log line or an exception (principle 11).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

import httpx
from pydantic import SecretStr, ValidationError

from mdqm.cache.freshness import Clock, SystemClock
from mdqm.config.secrets import SecretNotFound, SecretProvider
from mdqm.config.settings import Settings
from mdqm.llm.cache import CACHE_FILE_NAME, CacheEntry, ResponseCache
from mdqm.llm.errors import (
    MODEL_HTTP_ERROR,
    MODEL_NOT_CONFIGURED,
    MODEL_OUTPUT_INVALID,
    MODEL_TIMEOUT,
    MODEL_UNAVAILABLE,
    ModelError,
)
from mdqm.llm.result import (
    DEFAULT_PROMPT_VERSION,
    Attempt,
    AttemptOutcome,
    LLMResult,
    ModelT,
)

__all__ = [
    "CHAT_PATH",
    "DEFAULT_BACKOFF_S",
    "DEFAULT_SECRET_NAME",
    "MAX_BACKOFF_S",
    "MODELS_PATH",
    "ModelClient",
]

_logger = logging.getLogger(__name__)

CHAT_PATH: Final = "/chat/completions"
MODELS_PATH: Final = "/models"

#: Secret *reference* used when the key is not in settings (a name, never a value).
DEFAULT_SECRET_NAME: Final = "model_api_key"  # noqa: S105 - a reference name, not a value

DEFAULT_BACKOFF_S: Final = 1.0
MAX_BACKOFF_S: Final = 30.0

#: Statuses worth another call. 408/409/425/429 are transient; 5xx is the server's problem.
_TRANSIENT_STATUS: Final = frozenset({408, 409, 425, 429})
#: Statuses a server uses to say "I do not know this request field", e.g. ``response_format``.
_REFUSAL_STATUS: Final = frozenset({400, 404, 415, 422})

_FENCE = re.compile(r"^\s*```[A-Za-z0-9_-]*[ \t]*\r?\n(?P<body>.*?)\r?\n?[ \t]*```\s*$", re.DOTALL)

_SCHEMA_INSTRUCTION = (
    "Reply with one JSON object and nothing else: no prose, no markdown, no code fences. "
    "The object must validate against this JSON Schema:\n{schema}"
)


def _schema_instruction(schema: type[Any]) -> str:
    return _SCHEMA_INSTRUCTION.format(
        schema=json.dumps(schema.model_json_schema(), ensure_ascii=False, sort_keys=True)
    )


def _strip_fences(text: str) -> str:
    """A weak model wraps JSON in a markdown fence; that is formatting, not a wrong answer."""
    match = _FENCE.match(text)
    return match.group("body") if match else text


def _validation_problem(exc: ValidationError, *, limit: int = 3) -> str:
    """Field paths and messages only: pydantic also records the offending value, which is data."""
    parts = [
        f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
        for error in exc.errors()[:limit]
    ]
    if len(exc.errors()) > limit:
        parts.append(f"and {len(exc.errors()) - limit} more")
    return "; ".join(parts)[:400]


def _message_text(body: Mapping[str, Any]) -> str | None:
    """The reply text from ``choices[0].message``: ``content``, or ``parsed`` on some servers."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    message = first.get("message") if isinstance(first, Mapping) else None
    if not isinstance(message, Mapping):
        return None
    for key in ("content", "parsed"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, (dict, list)) and value:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return None


def _error_message(body: Any) -> str | None:
    """The server's own error text, when it sent JSON. Never a raw body (it may echo anything)."""
    if not isinstance(body, Mapping):
        return None
    error = body.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return " ".join(message.split())[:200]
    detail = body.get("detail")
    if isinstance(detail, str) and detail.strip():
        return " ".join(detail.split())[:200]
    return None


class ModelClient:
    """Synchronous client for one OpenAI-compatible chat endpoint.

    Build it once per process. It is safe to share between threads (``httpx.Client`` is), but
    ``structured_output`` is one flag on the instance: probe before the first call, from one
    place.

    Raises :class:`ModelError` (``MODEL_NOT_CONFIGURED``) when settings carry no ``base_url`` or
    no ``model``: a default model name would be a guess about an unverified server.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        cache: ResponseCache | None = None,
        clock: Clock | None = None,
        sleep: Callable[[float], None] | None = None,
        api_key: SecretStr | None = None,
        secret_provider: SecretProvider | None = None,
        secret_name: str = DEFAULT_SECRET_NAME,
        backoff_s: float = DEFAULT_BACKOFF_S,
        structured_output: bool | None = None,
    ) -> None:
        model = settings.model
        if not model.base_url or not model.model:
            raise ModelError(
                "the model endpoint is not configured: model.base_url and model.model are empty",
                "set MDQM_MODEL__BASE_URL and MDQM_MODEL__MODEL (or model.base_url / model.model "
                "in mdqm.yaml); there is no default endpoint or model name",
                code=MODEL_NOT_CONFIGURED,
            )
        self._model = model
        self._http = httpx.Client(transport=transport, timeout=model.timeout_s)
        self._cache = (
            cache
            if cache is not None
            else ResponseCache(settings.paths.cache_dir / CACHE_FILE_NAME)
        )
        self._clock: Clock = clock or SystemClock()
        self._sleep: Callable[[float], None] = sleep or time.sleep
        self._backoff_s = max(0.0, backoff_s)
        self._api_key: SecretStr | None = api_key if api_key is not None else model.api_key
        self._secret_provider = secret_provider
        self._secret_name = secret_name
        self._structured_output = structured_output

    # --- state the pipeline and the probe read ------------------------------------------------

    @property
    def clock(self) -> Clock:
        """The injected clock, so a caller timestamps with the same source as this client."""
        return self._clock

    @property
    def cache(self) -> ResponseCache:
        """The response cache in use (``settings.paths.cache_dir`` unless one was passed)."""
        return self._cache

    @property
    def timeout(self) -> httpx.Timeout:
        """The timeout handed to ``httpx``; it comes from ``model.timeout_s``."""
        return self._http.timeout

    @property
    def structured_output(self) -> bool | None:
        """Whether ``json_schema`` output is honoured: True, False or not yet probed."""
        return self._structured_output

    @structured_output.setter
    def structured_output(self, value: bool | None) -> None:
        self._structured_output = value

    @property
    def model_name(self) -> str:
        """The configured model id (sent in every request; never assumed)."""
        return self._model.model

    @property
    def base_url(self) -> str:
        """The configured API root, so a report can say which endpoint answers (never the key)."""
        return self._model.base_url

    # --- one raw call -------------------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: Mapping[str, Any] | None = None,
        retries: int | None = None,
    ) -> dict[str, Any]:
        """POST one chat completion and return the parsed body.

        ``options`` are merged into the request body (``max_tokens``, ``tools``,
        ``response_format``, ...). ``retries=None`` means ``model.max_retries``; the probe passes
        ``0`` so a slow feature costs one call, not three.
        """
        budget = self._model.max_retries if retries is None else max(0, retries)
        payload: dict[str, Any] = {
            "model": self._model.model,
            "messages": [dict(message) for message in messages],
            "temperature": 0,
        }
        if options:
            payload.update(options)
        attempts: list[Attempt] = []
        for number in range(1, budget + 2):
            try:
                return self._post_once(payload)
            except ModelError as exc:
                attempts.append(_attempt(number, exc))
                if not exc.retryable or number == budget + 1:
                    exc.attempts = tuple(attempts)
                    raise
                self._pause(number, attempts)
        raise AssertionError("the retry loop always returns or raises")

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> ModelClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"ModelClient(base_url={self._model.base_url!r}, model={self._model.model!r}, "
            f"structured_output={self._structured_output!r}, cache={self._cache.path!r})"
        )

    # --- the closed-JSON entry point ----------------------------------------------------------

    def structured(
        self,
        prompt: str,
        schema: type[ModelT],
        *,
        system: str | None = None,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> LLMResult[ModelT]:
        """Ask for one ``schema`` object; validate it in code; retry with the error; cache it.

        Serves a cached answer with ``cached=True`` and no HTTP call. Otherwise makes up to
        ``1 + model.max_retries`` calls at ``temperature 0``. On a rejected reply the invalid
        text and the sanitised validation problem go back to the model, because a weak model
        fixes its own JSON when told exactly what was wrong.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _system_prompt(system, schema)},
            {"role": "user", "content": prompt},
        ]
        key = self.cache_key(
            prompt=prompt, schema=schema, system=system, prompt_version=prompt_version
        )
        hit = self._cached(key, schema, prompt_version)
        if hit is not None:
            value, entry = hit
            return LLMResult(
                value=value,
                model=self._model.model,
                prompt_version=prompt_version,
                cached=True,
                fetched_at=entry.fetched_at,
            )

        budget = self._model.max_retries + 1
        attempts: list[Attempt] = []
        problem = ""
        outcome: AttemptOutcome = "invalid_json"
        for number in range(1, budget + 1):
            payload: dict[str, Any] = {
                "model": self._model.model,
                "messages": messages,
                "temperature": 0,
            }
            if self._structured_output is True:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema(),
                        "strict": True,
                    },
                }
            try:
                body = self._post_once(payload)
            except ModelError as exc:
                attempts.append(_attempt(number, exc))
                if not exc.retryable or number == budget:
                    exc.attempts = tuple(attempts)
                    raise
                # A refused response_format downgrades the client; the retry then instructs.
                self._pause(number, attempts)
                continue

            raw = _message_text(body)
            data: Any = None
            outcome = "empty_response"
            feedback = "Your reply was empty. Reply with only a JSON object matching the schema."
            if raw is not None:
                text = _strip_fences(raw)
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as exc:
                    outcome = "invalid_json"
                    problem = f"the reply is not valid JSON ({exc.msg} at column {exc.colno})"
                    feedback = (
                        f"The reply was rejected: {problem}. Reply again with only a JSON object "
                        "matching the schema, no prose and no code fences."
                    )
                else:
                    try:
                        value = schema.model_validate(data)
                    except ValidationError as exc:
                        outcome = "invalid_schema"
                        problem = _validation_problem(exc)
                        feedback = (
                            f"The reply was rejected because it does not match the schema: "
                            f"{problem}. Reply again with only a corrected JSON object."
                        )
                    else:
                        attempts.append(Attempt(number=number, outcome="ok"))
                        return self._remember(
                            key=key,
                            data=data,
                            value=value,
                            schema_name=schema.__name__,
                            prompt_version=prompt_version,
                            attempts=tuple(attempts),
                        )
            else:
                problem = "the reply carried no message content"

            attempts.append(Attempt(number=number, outcome=outcome, problem=problem))
            if number == budget:
                raise ModelError(
                    f"the model returned no valid {schema.__name__} after {budget} attempt(s): "
                    f"{problem}",
                    "check that model.structured_output matches what the endpoint really supports, "
                    "shorten the prompt, or raise model.max_retries",
                    code=MODEL_OUTPUT_INVALID,
                    outcome=outcome,
                    attempts=tuple(attempts),
                )
            messages.append({"role": "assistant", "content": raw or ""})
            messages.append({"role": "user", "content": feedback})
            self._pause(number, attempts)
        raise AssertionError("the retry loop always returns or raises")

    def cache_key(
        self,
        *,
        prompt: str,
        schema: type[Any],
        system: str | None = None,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> str:
        """The input hash for one call. Never includes the API key, so it is safe to log."""
        material = {
            "base_url": self._model.base_url,
            "model": self._model.model,
            "prompt_version": prompt_version,
            "system": system,
            "prompt": prompt,
            "schema": schema.model_json_schema(),
        }
        blob = json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    # --- internals ----------------------------------------------------------------------------

    def _remember(
        self,
        *,
        key: str,
        data: Any,
        value: ModelT,
        schema_name: str,
        prompt_version: str,
        attempts: tuple[Attempt, ...],
    ) -> LLMResult[ModelT]:
        fetched_at = self._clock.now()
        self._cache.put(
            key,
            content=json.dumps(data, ensure_ascii=False, sort_keys=True),
            model=self._model.model,
            prompt_version=prompt_version,
            fetched_at=fetched_at,
        )
        _logger.debug(
            "model answer stored",
            extra={"model": self._model.model, "schema": schema_name, "attempts": len(attempts)},
        )
        return LLMResult(
            value=value,
            attempts=attempts,
            model=self._model.model,
            prompt_version=prompt_version,
            cached=False,
            fetched_at=fetched_at,
        )

    def _cached(
        self, key: str, schema: type[ModelT], prompt_version: str
    ) -> tuple[ModelT, CacheEntry] | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        try:
            value = schema.model_validate(json.loads(entry.content))
        except (json.JSONDecodeError, ValidationError):
            # The schema moved on (or the row is damaged); validate, then refetch. Never serve it.
            _logger.warning(
                "cached model answer no longer validates; asking the model again",
                extra={"model": entry.model, "prompt_version": prompt_version},
            )
            return None
        _logger.debug(
            "model answer served from cache",
            extra={"model": entry.model, "prompt_version": prompt_version},
        )
        return value, entry

    def _pause(self, number: int, attempts: list[Attempt]) -> None:
        """Sleep the backoff for attempt ``number`` and record it on that attempt."""
        delay = min(self._backoff_s * (2.0 ** (number - 1)), MAX_BACKOFF_S)
        attempts[-1] = attempts[-1].model_copy(update={"retry_delay_s": delay})
        _logger.warning(
            "model call failed; retrying",
            extra={
                "model": self._model.model,
                "attempt": number,
                "outcome": attempts[-1].outcome,
                "delay_s": delay,
            },
        )
        self._sleep(delay)

    def _secret(self) -> str | None:
        if self._api_key is None and self._secret_provider is not None:
            try:
                self._api_key = SecretStr(self._secret_provider.get(self._secret_name))
            except SecretNotFound as exc:
                raise ModelError(
                    f"the model API key secret {self._secret_name!r} is not available: {exc.where}",
                    "create the secret (environment variable or secret file) or configure a "
                    "provider that has it; secret references are names, never values",
                    code=MODEL_NOT_CONFIGURED,
                ) from None
        return None if self._api_key is None else self._api_key.get_secret_value()

    def _headers(self) -> dict[str, str]:
        key = self._secret()
        return {} if key is None else {"Authorization": f"Bearer {key}"}

    def _redact(self, text: str) -> str:
        """A key must never travel in a message, even when a server echoes it back."""
        key = self._secret()
        return text if not key else text.replace(key, "[REDACTED]")

    def models(self) -> dict[str, Any]:
        """``GET {base_url}/models``: the endpoint's own model list, for the context-window probe.

        Not assumed to exist: a 404 is a capability verdict the probe records, not a crash.
        """
        return self._request("GET", MODELS_PATH)

    def _post_once(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._request("POST", CHAT_PATH, json_body=payload)

    def _request(
        self, method: str, path: str, *, json_body: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        url = self._model.base_url + path
        try:
            if json_body is None:
                response = self._http.request(method, url, headers=self._headers())
            else:
                response = self._http.request(
                    method, url, json=dict(json_body), headers=self._headers()
                )
        except httpx.TimeoutException:
            raise ModelError(
                f"the model endpoint did not answer {method} {path} within "
                f"{self._model.timeout_s:g}s",
                "raise model.timeout_s or lower the prompt size; the endpoint can be slow "
                "under load",
                code=MODEL_TIMEOUT,
                retryable=True,
                outcome="timeout",
            ) from None
        except httpx.HTTPError as exc:
            # Not `from exc`: the transport's message is ours, but its chain is not needed.
            raise ModelError(
                f"cannot reach the model endpoint at {self._model.base_url}{path}: "
                f"{type(exc).__name__}",
                "check model.base_url and that the endpoint is running; nothing outside the "
                "configured sources and model endpoint may be contacted at runtime",
                code=MODEL_UNAVAILABLE,
                retryable=True,
                outcome="transport_error",
            ) from None
        if response.status_code >= 400:
            self._raise_status(response)
        try:
            body = response.json()
        except ValueError:
            raise ModelError(
                f"the model endpoint answered HTTP {response.status_code} with a non-JSON body",
                "check that model.base_url points at the OpenAI-compatible API root "
                "(for example http://host:port/v1), not at a web page or a proxy",
                code=MODEL_HTTP_ERROR,
                status_code=response.status_code,
                outcome="non_json_body",
            ) from None
        if not isinstance(body, dict):
            raise ModelError(
                f"the model endpoint answered with a JSON {type(body).__name__}, not an object",
                "the endpoint is not OpenAI-compatible; point model.base_url at the chat API root",
                code=MODEL_HTTP_ERROR,
                status_code=response.status_code,
                outcome="non_json_body",
            )
        return body

    def _raise_status(self, response: httpx.Response) -> None:
        status = response.status_code
        try:
            detail = _error_message(response.json())
        except ValueError:
            detail = None
        detail = self._redact(detail) if detail else None
        outcome: AttemptOutcome = "http_error"
        retryable = status in _TRANSIENT_STATUS or status >= 500
        if status in _REFUSAL_STATUS and detail and "response_format" in detail.lower():
            # The probe said yes (or nobody probed); the endpoint just said no. Remember that.
            self._structured_output = False
            outcome = "response_format_rejected"
            retryable = True
        problem = f"the model endpoint returned HTTP {status}"
        if detail:
            problem = f"{problem}: {detail}"
        raise ModelError(
            problem,
            _status_fix(status),
            code=MODEL_HTTP_ERROR,
            retryable=retryable,
            status_code=status,
            outcome=outcome,
        )


def _attempt(number: int, exc: ModelError) -> Attempt:
    return Attempt(
        number=number,
        outcome=exc.outcome,
        problem=exc.problem[:400],
        status_code=exc.status_code,
        error_code=exc.code,
    )


def _system_prompt(system: str | None, schema: type[Any]) -> str:
    """Caller text plus the closed-JSON instruction; the schema is always spelled out."""
    parts = [part for part in (system, _schema_instruction(schema)) if part]
    return "\n\n".join(parts)


def _status_fix(status: int) -> str:
    if status in {401, 403}:
        return (
            "the endpoint rejected the credential: check the secret reference "
            "(MDQM_SECRET_MODEL_API_KEY or model.api_key) and that the key is valid there"
        )
    if status == 429:
        return "the endpoint is rate limiting: lower the call rate or raise model.max_retries"
    if status in {400, 404, 415, 422}:
        return (
            "the endpoint rejected the request: check model.model against the ids the endpoint "
            "serves, and model.base_url against its OpenAI-compatible API root"
        )
    if status >= 500:
        return (
            "the model endpoint is unhealthy: retry later; if it persists, report it to whoever "
            "runs the endpoint"
        )
    return "check model.base_url and model.model; the endpoint refused this request"
