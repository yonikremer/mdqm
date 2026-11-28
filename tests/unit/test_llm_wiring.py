"""Settings -> model client -> ``PipelineContext.llm``: the wiring A1+A2 was missing.

Before this, ``context_for`` never set ``ctx.llm``, so ``mdqm bootstrap run`` answered "no model
client configured: nothing proposed" even when a deployment had an endpoint, and the semantic layer
could only be built by the deterministic stages plus hand-written labels.

What is pinned here, one class each:

* :class:`TestSettings` -- ``model.base_url`` / ``model`` / ``timeout_s`` / ``max_retries`` come from
  the environment and from ``mdqm.yaml``; an absent endpoint is the default; a malformed one names
  the key and never the value; a literal key in YAML is refused with the existing rule.
* :class:`TestBuildingTheClient` -- nothing configured means no client; a half-configured endpoint
  means no client *and* one line saying which field is empty; a configured one yields a client that
  is addressed by the settings and does not contact anything at construction time.
* :class:`TestContextWiring` -- :func:`~mdqm.pipeline.cli.context_for` puts that client on the
  context, an injected one wins untouched, and no endpoint leaves ``llm`` empty so the stage degrades
  exactly as it did before.
* :class:`TestA1A2ThroughTheContext` -- the stage asked over a wired context reaches the review queue
  with the model's words, and with no client it says so and queues nothing.

No test opens a socket: the client is only constructed (``httpx`` connects on the first request), and
the stage is driven by ``FakeLLM``. ``pytest-socket`` stays armed for everything else.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from mdqm.config.settings import ConfigError, ModelSettings, Settings
from mdqm.llm.client import ModelClient
from mdqm.llm.factory import NO_MODEL_LINE, describe_model, model_client
from mdqm.llm.fake import FakeLLM
from mdqm.models.catalog import CatalogSnapshot, Column, NormalizedType, Table
from mdqm.pipeline.cli import context_for
from mdqm.pipeline.core import run_pipeline
from mdqm.pipeline.stages.propose_names import Labels, ProposeNamesStage
from mdqm.review.queue import ProposalKind, ReviewQueue
from mdqm.store.db import Store

BASE_URL = "http://glm.internal:8080/v1"
MODEL = "glm-4-9b"
NOW_TEXT = "2026-03-01T12:00:00Z"


def yaml_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "mdqm.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """No ``MDQM_*`` variable and no working-directory YAML: the defaults only."""
    for name in (
        "MDQM_MODEL__BASE_URL",
        "MDQM_MODEL__MODEL",
        "MDQM_MODEL__API_KEY",
        "MDQM_MODEL__TIMEOUT_S",
        "MDQM_MODEL__MAX_RETRIES",
        "MDQM_PATHS__SEMANTIC_DIR",
        "MDQM_PATHS__CACHE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    yield


def settings_with(
    *,
    base_url: str = BASE_URL,
    model: str = MODEL,
    timeout_s: float = 120.0,
    max_retries: int = 2,
) -> Settings:
    return Settings(
        model=ModelSettings(
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
    )


# --- settings ---------------------------------------------------------------------------------


class TestSettings:
    @pytest.mark.parametrize(
        ("base_url", "model"),
        [("", ""), (BASE_URL, ""), ("", MODEL)],
    )
    def test_no_endpoint_is_the_default_state(
        self, clean_env: None, base_url: str, model: str
    ) -> None:
        """``""`` and ``""`` is "not configured"; a half endpoint is still not a client."""
        settings = Settings()

        assert settings.model.base_url == ""
        assert settings.model.model == ""
        assert settings.model.timeout_s > 0
        assert model_client(settings, report=lambda _line: pytest.fail("nothing to report")) is None

    def test_the_endpoint_comes_from_the_environment(self, clean_env: None) -> None:
        import os

        os.environ["MDQM_MODEL__BASE_URL"] = BASE_URL + "/"
        os.environ["MDQM_MODEL__MODEL"] = MODEL
        os.environ["MDQM_MODEL__TIMEOUT_S"] = "7.5"
        os.environ["MDQM_MODEL__MAX_RETRIES"] = "1"

        settings = Settings()

        assert settings.model.base_url == BASE_URL  # the trailing slash is normalised away
        assert settings.model.model == MODEL
        assert settings.model.timeout_s == 7.5
        assert settings.model.max_retries == 1

    def test_the_endpoint_comes_from_yaml(self, clean_env: None, tmp_path: Path) -> None:
        path = yaml_file(
            tmp_path,
            f"model:\n  base_url: {BASE_URL}\n  model: {MODEL}\n  timeout_s: 30\n",
        )

        settings = Settings.load(path)

        assert settings.model.base_url == BASE_URL
        assert settings.model.model == MODEL
        assert settings.model.timeout_s == 30

    def test_the_environment_wins_over_yaml(self, clean_env: None, tmp_path: Path) -> None:
        import os

        path = yaml_file(tmp_path, f"model:\n  base_url: http://from-file:1/v1\n  model: {MODEL}\n")
        os.environ["MDQM_MODEL__BASE_URL"] = BASE_URL

        assert Settings.load(path).model.base_url == BASE_URL

    @pytest.mark.parametrize(
        ("yaml_text", "key"),
        [
            ("model:\n  base_url: pg://not-http\n  model: glm\n", "model.base_url"),
            (
                "model:\n  base_url: http://h:1/v1\n  model: glm\n  timeout_s: 0\n",
                "model.timeout_s",
            ),
            (
                "model:\n  base_url: http://h:1/v1\n  model: glm\n  max_retries: 9\n",
                "model.max_retries",
            ),
            ("model:\n  port: 8080\n", "model.port"),
            ("model:\n  api_key: sk-live-123\n", "model.api_key"),
        ],
    )
    def test_a_malformed_model_section_names_the_key_and_not_the_value(
        self, clean_env: None, tmp_path: Path, yaml_text: str, key: str
    ) -> None:
        path = yaml_file(tmp_path, yaml_text)

        with pytest.raises(ConfigError) as caught:
            Settings.load(path)

        message = str(caught.value)
        assert key in message
        assert "sk-live-123" not in message
        assert str(path) in message


# --- building the client ----------------------------------------------------------------------


class TestBuildingTheClient:
    def test_nothing_configured_builds_nothing_and_says_nothing_alarming(
        self, clean_env: None
    ) -> None:
        lines: list[str] = []

        assert model_client(Settings(), report=lines.append) is None
        assert lines == []  # an absent endpoint is a choice, not a typo: the stage reports it
        assert describe_model(None) == NO_MODEL_LINE

    @pytest.mark.parametrize(
        ("base_url", "model", "missing"),
        [(BASE_URL, "", "model.model"), ("", MODEL, "model.base_url")],
    )
    def test_a_half_configured_endpoint_names_the_missing_field(
        self, clean_env: None, base_url: str, model: str, missing: str
    ) -> None:
        lines: list[str] = []

        assert (
            model_client(settings_with(base_url=base_url, model=model), report=lines.append) is None
        )
        assert len(lines) == 1
        assert missing in lines[0]
        assert "A1+A2 proposes nothing" in lines[0]

    def test_a_configured_endpoint_builds_a_client_addressed_by_the_settings(
        self, clean_env: None
    ) -> None:
        client = model_client(settings_with(timeout_s=7.5))

        assert isinstance(client, ModelClient)
        assert client.model_name == MODEL
        assert client.base_url == BASE_URL
        assert client.timeout.read == 7.5

    def test_building_a_client_reads_no_secret(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A configured secret name with no value must not fail the build: the call fails, later."""
        monkeypatch.setenv("MDQM_SECRET_MODEL_API_KEY", "")

        client = model_client(settings_with())

        assert isinstance(client, ModelClient)

    def test_the_description_names_the_model_and_never_a_key(self, clean_env: None) -> None:
        client = model_client(settings_with())
        assert client is not None

        line = describe_model(client)

        assert MODEL in line
        assert BASE_URL in line
        assert "api" not in line.lower().replace("api root", "")


# --- the context ------------------------------------------------------------------------------


class TestContextWiring:
    def test_no_endpoint_leaves_the_context_without_a_model(
        self, clean_env: None, tmp_path: Path
    ) -> None:
        opened = Store(tmp_path / "metadata.sqlite")
        try:
            ctx = context_for(Settings(), store=opened)
        finally:
            opened.close()

        assert ctx.llm is None

    def test_a_configured_endpoint_reaches_the_context(
        self, clean_env: None, tmp_path: Path
    ) -> None:
        opened = Store(tmp_path / "metadata.sqlite")
        try:
            ctx = context_for(settings_with(), store=opened)
        finally:
            opened.close()

        assert isinstance(ctx.llm, ModelClient)
        assert ctx.llm.model_name == MODEL

    def test_an_injected_model_wins_and_nothing_is_built(
        self, clean_env: None, tmp_path: Path
    ) -> None:
        fake = FakeLLM(model="scripted")
        opened = Store(tmp_path / "metadata.sqlite")
        try:
            ctx = context_for(settings_with(), store=opened, llm=fake)
        finally:
            opened.close()

        assert ctx.llm is fake
        assert describe_model(ctx.llm) == "model: scripted"

    def test_a_half_configured_endpoint_is_reported_through_the_caller(
        self, clean_env: None, tmp_path: Path
    ) -> None:
        lines: list[str] = []
        opened = Store(tmp_path / "metadata.sqlite")
        try:
            ctx = context_for(settings_with(model=""), store=opened, echo=lines.append)
        finally:
            opened.close()

        assert ctx.llm is None
        assert any("half configured" in line for line in lines)


# --- A1+A2 over a wired context ----------------------------------------------------------------


def catalog() -> CatalogSnapshot:
    from datetime import UTC, datetime

    return CatalogSnapshot(
        source="crm",
        captured_at=datetime(2026, 3, 1, tzinfo=UTC),
        tables=(
            Table(
                name="customers",
                columns=(
                    Column(
                        name="cust_id",
                        native_type="integer",
                        normalized_type=NormalizedType.INTEGER,
                    ),
                ),
                primary_key=("cust_id",),
            ),
        ),
    )


SCRIPTED = Labels(
    name_he="לקוחות",
    name_en="Customers",
    description_he="טבלת הלקוחות.",
    description_en="One row per customer.",
    synonyms_he=("אנשי קשר",),
    synonyms_en=("Clients",),
)

COLUMN = Labels(
    name_he="מזהה לקוח",
    name_en="Customer ID",
    synonyms_he=("מספר לקוח",),
    synonyms_en=("Customer Number",),
)


def scripted(prompt: str) -> Labels:
    """The reply for the target the prompt names: two targets, two different answers."""
    sheet = json.loads(prompt[prompt.index("{") :])
    return SCRIPTED if sheet["key"] == "crm.customers" else COLUMN


class TestA1A2ThroughTheContext:
    def test_a_wired_context_lets_the_stage_reach_the_queue(
        self, clean_env: None, tmp_path: Path
    ) -> None:
        opened = Store(tmp_path / "metadata.sqlite")
        try:
            ctx = context_for(settings_with(), store=opened, llm=FakeLLM(*([scripted] * 4)))
            stage = ProposeNamesStage({"crm": catalog()})
            result = run_pipeline([stage], ctx)
        finally:
            opened.close()

        report = result.reports[0]
        # one table and one column, each asked twice (the second run rotated)
        assert report.counts["model_calls"] == 4
        pending = ReviewQueue(opened, tmp_path).pending()
        keys = {item.key for item in pending}
        assert {"table:crm.customers", "column:crm.customers.cust_id"} <= keys
        assert {item.kind for item in pending} == {
            ProposalKind.LABEL,
            ProposalKind.NAME,
            ProposalKind.SYNONYM,
        }

    def test_no_client_means_no_proposal_and_a_reason(
        self, clean_env: None, tmp_path: Path
    ) -> None:
        opened = Store(tmp_path / "metadata.sqlite")
        try:
            ctx = context_for(Settings(), store=opened)
            result = run_pipeline([ProposeNamesStage({"crm": catalog()})], ctx)
        finally:
            opened.close()

        report = result.reports[0]
        assert ctx.llm is None
        assert report.counts["model_calls"] == 0
        assert any("no model client configured" in line for line in report.skipped)
        assert ReviewQueue(opened, tmp_path).pending() == ()

    def test_the_stage_is_deterministic_between_two_identical_runs(
        self, clean_env: None, tmp_path: Path
    ) -> None:
        """Seeded replies, injected clock: same script, same queue, byte for byte."""

        def propose(root: Path) -> list[tuple[str, str, str]]:
            opened = Store(root / "metadata.sqlite")
            try:
                ctx = context_for(
                    Settings(), store=opened, llm=FakeLLM(*([scripted] * 4)), now=None
                )
                ctx.semantic_root = root
                run_pipeline([ProposeNamesStage({"crm": catalog()})], ctx)
                return [
                    (item.id, item.key, json.dumps(item.patch, sort_keys=True, ensure_ascii=False))
                    for item in ReviewQueue(opened, root).pending()
                ]
            finally:
                opened.close()

        first = propose(tmp_path / "one")
        second = propose(tmp_path / "two")

        assert first == second
