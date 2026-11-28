"""The one place that turns settings into a model client -- or, honestly, into none at all.

Nothing else in the tree builds a :class:`~mdqm.llm.client.ModelClient`: ``pipeline/core.py`` holds
no I/O, :class:`~mdqm.config.settings.Settings` never imports ``httpx``, and a stage must not know
how an endpoint is addressed. So the translation sits here, once::

    settings.model -> model_client(settings) -> PipelineContext.llm -> A1+A2

Three outcomes, and two of them are *no client* on purpose:

* **Nothing configured** (``model.base_url`` and ``model.model`` both empty): ``None``. A supported
  deployment -- every deterministic stage still runs, and A1+A2 says "no model client configured:
  nothing proposed" instead of pretending. This is the state of a fresh install and of every test
  that has no endpoint.
* **Half configured** (one of the two set): ``None``, plus one line naming which field is missing.
  A base URL with no model id is a typo, not a choice, and a default model name would be a guess
  about an unverified server (AGENTS.md section 9).
* **Configured**: a client. The API key is *not* read here -- :class:`ModelClient` resolves its
  secret reference lazily, on the first call, through the deployment's
  :class:`~mdqm.config.secrets.SecretProvider`. So a missing key costs one refused call with a fix,
  not a process that will not start, and no key value ever passes through this module or a log.

The endpoint itself stays unverified: this module assumes nothing about its context window, its
``response_format`` support or its tool calling. Those are probed
(:func:`~mdqm.llm.probe.probe_model_endpoint`) or simply not used.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from mdqm.cache.freshness import Clock
from mdqm.config.settings import Settings
from mdqm.llm.client import ModelClient
from mdqm.llm.errors import ModelError
from mdqm.llm.result import StructuredLLM

__all__ = ["describe_model", "model_client"]

_logger = logging.getLogger(__name__)

#: What a run with no endpoint prints. One string, so the CLI, the stage and the tests agree.
NO_MODEL_LINE = (
    "model endpoint not configured (model.base_url / model.model): A1+A2 will propose nothing"
)


def describe_model(llm: StructuredLLM | None) -> str:
    """One line an operator can read: which model answers, or that none will.

    Accepts anything shaped like :class:`~mdqm.llm.result.StructuredLLM`, so a context holding a
    ``FakeLLM`` (a test, a deployment with its own client) is described truthfully instead of
    claiming the real client's caching it does not have.
    """
    if llm is None:
        return NO_MODEL_LINE
    if isinstance(llm, ModelClient):
        return (
            f"model: {llm.model_name} at {llm.base_url} "
            "(temperature 0, answers cached by input hash)"
        )
    return f"model: {llm.model_name}"


def model_client(
    settings: Settings,
    *,
    clock: Clock | None = None,
    report: Callable[[str], None] | None = None,
) -> ModelClient | None:
    """Build the client this deployment's settings describe, or ``None`` with the reason said.

    ``report`` receives the reason a configured-looking endpoint produced no client (an
    operator reads it; the caller decides whether to print it). Never raises for a configuration
    problem: a bootstrap pass must still run its deterministic stages when the model is unreachable
    or badly configured, and the stage that needs a model already knows how to say it has none.
    """
    say = report or (lambda _message: None)
    model = settings.model
    if not model.base_url and not model.model:
        return None
    if not model.base_url or not model.model:
        missing = "model.base_url" if not model.base_url else "model.model"
        say(
            f"model endpoint is half configured: {missing} is empty, so no client was built and "
            "A1+A2 proposes nothing (set both, or neither)"
        )
        return None
    try:
        client = ModelClient(settings, clock=clock, secret_provider=settings.secret_provider())
    except ModelError as exc:
        # Not `from exc`: the key is a secret and a chained traceback is not a report.
        say(f"model client not built ({exc.code}): {exc.problem} -- {exc.fix}")
        return None
    _logger.info(
        "model client built",
        extra={"model": client.model_name, "base_url": client.base_url},
    )
    return client
