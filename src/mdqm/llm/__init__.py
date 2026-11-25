"""The model lane: an OpenAI-compatible client with closed JSON output, its probe and a fake.

The endpoint is unverified (AGENTS.md section 9), so this package assumes nothing beyond
``POST /chat/completions``: capabilities are probed, unsupported features fall back to
"instruct and validate in code", timeouts and retries come from config, and the API key is a
secret reference resolved by a :class:`~mdqm.config.secrets.SecretProvider`.

``ModelClient.structured(prompt, schema)`` is the one entry point a pipeline uses; ``FakeLLM``
implements the same :class:`StructuredLLM` protocol for tests and for an offline deployment.
"""

from __future__ import annotations

from mdqm.llm.cache import CACHE_FILE_NAME, CacheEntry, ResponseCache
from mdqm.llm.client import DEFAULT_BACKOFF_S, DEFAULT_SECRET_NAME, ModelClient
from mdqm.llm.errors import (
    MODEL_HTTP_ERROR,
    MODEL_NOT_CONFIGURED,
    MODEL_OUTPUT_INVALID,
    MODEL_TIMEOUT,
    MODEL_UNAVAILABLE,
    ModelError,
)
from mdqm.llm.fake import FakeCall, FakeLLM
from mdqm.llm.probe import (
    MODEL_ENDPOINT,
    context_window,
    probe_model_endpoint,
    structured_output_state,
)
from mdqm.llm.result import (
    DEFAULT_PROMPT_VERSION,
    Attempt,
    AttemptOutcome,
    LLMResult,
    StructuredLLM,
)

__all__ = [
    "CACHE_FILE_NAME",
    "DEFAULT_BACKOFF_S",
    "DEFAULT_PROMPT_VERSION",
    "DEFAULT_SECRET_NAME",
    "MODEL_ENDPOINT",
    "MODEL_HTTP_ERROR",
    "MODEL_NOT_CONFIGURED",
    "MODEL_OUTPUT_INVALID",
    "MODEL_TIMEOUT",
    "MODEL_UNAVAILABLE",
    "Attempt",
    "AttemptOutcome",
    "CacheEntry",
    "FakeCall",
    "FakeLLM",
    "LLMResult",
    "ModelClient",
    "ModelError",
    "ResponseCache",
    "StructuredLLM",
    "context_window",
    "probe_model_endpoint",
    "structured_output_state",
]
