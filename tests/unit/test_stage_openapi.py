"""Stage D4 (S03): an OpenAPI document becomes the semantic layer's table skeletons.

The task's two acceptance criteria are the first two groups, then the rules that make the skeleton
safe to trust:

* **a collection is a table** -- one entry per ``GET`` collection, its columns read from the item
  schema and its declared query parameters as filterable indexes, with the provenance that produced
  it. No model is asked and nothing is invented: identity key, mutability and time columns are
  absent because a spec says nothing about them;
* **the rest is reported** -- a write, a lookup behind a path parameter and a single-object answer
  are listed in the report's ``skipped`` *with a reason*, not dropped silently;
* **the file loads** -- the generated section is valid YAML that goes back through
  :func:`mdqm.semantic.loader.load` beside any other section, which is what a broken key would stop;
* **an unchanged document writes nothing** -- the S01 ledger skips the stage on the same input, and
  the file is byte-identical when it does run again.

There is no live internal API on this machine: the document comes from
:class:`~mdqm.testing.mock_api.MockApi` (a real FastAPI app), read in process, so no port, no socket
and no network are involved. No test sleeps and no test reads a clock -- ``exec_ts`` is injected.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from mdqm.adapters.openapi import OpenApiAdapter, read_endpoints
from mdqm.config.settings import AuthSettings, SourceConfig
from mdqm.pipeline.core import PipelineContext, PipelineResult, StageStatus, run_pipeline
from mdqm.pipeline.stages.openapi import OpenApiStage, read_specs
from mdqm.semantic.loader import load
from mdqm.semantic.models import SemanticLayer
from mdqm.store.db import Store
from mdqm.testing.mock_api import (
    AUTH_HEADER,
    MockApi,
    MockApiConfig,
    spec_yaml,
)

#: 2026-05-01 10:00 UTC. The stage never reads it, which is part of the point.
NOW = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)

#: A throwaway key for the in-process app. It protects nothing and never reaches ``src``.
KEY = "s03-throwaway-key"

SOURCE = "rates_api"

#: The rate-limit headers the document promises, as the skeleton must report them.
RATE_LIMIT_HEADERS = (
    "Retry-After",
    "X-RateLimit-Limit",
    "X-RateLimit-Remaining",
    "X-RateLimit-Reset",
)


@pytest.fixture
def api() -> MockApi:
    return MockApi(MockApiConfig(api_key=KEY))


@pytest.fixture
def config() -> SourceConfig:
    """The deployment facts a document cannot know: the auth header and the secret reference."""
    return SourceConfig(
        name=SOURCE,
        adapter="openapi",
        auth=AuthSettings(header=AUTH_HEADER, secret="rates_api_key"),
        options={"base_url": "https://mock-api.invalid"},
    )


class Kitchen:
    """A store, a semantic root and a runner call -- the three things every case needs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.store = Store(root / "meta.db")

    def context(self) -> PipelineContext:
        return PipelineContext(store=self.store, exec_ts=NOW, semantic_root=self.root)

    def run(self, stage: OpenApiStage, *, source: str | None = None) -> PipelineResult:
        """One pass. ``source`` goes to the runner: it owns ``ctx.source`` and resets it."""
        return run_pipeline([stage], self.context(), source=source)

    def section(self, source: str = SOURCE) -> Path:
        return self.root / "generated" / f"openapi_{source}.yaml"

    def layer(self) -> SemanticLayer:
        return load(self.root)


@pytest.fixture
def kitchen(tmp_path: Path) -> Kitchen:
    return Kitchen(tmp_path / "semantic")


# --- acceptance: GET collections become table skeletons -----------------------------------------


def test_every_get_collection_becomes_a_table_skeleton(
    api: MockApi, config: SourceConfig, kitchen: Kitchen
) -> None:
    """One entry per collection, its columns from the item schema, its filters as indexes."""
    result = kitchen.run(OpenApiStage({SOURCE: api.spec()}, configs={SOURCE: config}))
    report = result.reports[0]
    assert report.status is StageStatus.RAN, report.error
    assert report.counts == {
        "sources": 1,
        "tables": 1,
        "columns": 4,
        "indexes": 5,
        "skipped_operations": 4,
    }
    assert report.outputs == ("generated/openapi_rates_api.yaml",)

    table = kitchen.layer().tables[f"{SOURCE}.rates"]
    assert [column.name for column in table.columns] == ["currency", "date", "rate", "source"]
    assert [column.ordinal for column in table.columns] == [1, 2, 3, 4]
    assert [column.normalized_type.value for column in table.columns] == [
        "string",
        "date",
        "float",
        "string",
    ]
    assert table.columns[1].native_type == "date", "the source's own type text is kept"
    assert [index.name for index in table.indexes] == [
        "currency",
        "cursor",
        "date_from",
        "date_to",
        "limit",
    ]
    assert {index.index_type for index in table.indexes} == {"query_param"}
    assert table.provenance is not None
    assert table.provenance.detected_by == "stage D4 (openapi spec)"
    assert table.key == (), "an identity key is not knowable from a spec, so none is claimed"
    assert table.mutability is None, "how often an endpoint changes is measured (D6), never guessed"


def test_the_golden_yaml_matches(
    api: MockApi, config: SourceConfig, kitchen: Kitchen, golden: Any
) -> None:
    """The whole section as a reviewable artefact: a change to it must show up in the diff."""
    kitchen.run(OpenApiStage({SOURCE: api.spec()}, configs={SOURCE: config}))
    golden("openapi_rates_api.yaml", kitchen.section().read_text(encoding="utf-8"))


def test_pagination_and_the_envelope_are_in_the_provenance(
    api: MockApi, config: SourceConfig, kitchen: Kitchen
) -> None:
    """Cursor paging, the page-size parameter and the envelope key are stated, not guessed."""
    kitchen.run(OpenApiStage({SOURCE: api.spec()}, configs={SOURCE: config}))
    evidence = kitchen.layer().tables[f"{SOURCE}.rates"].provenance
    assert evidence is not None
    lines = list(evidence.evidence or ())
    assert "GET /rates" in lines
    assert "paging: cursor via 'cursor'" in lines
    assert "page size parameter: 'limit'" in lines
    assert "rows are the response's 'items' array" in lines
    assert "the next-page token is the response's 'next_cursor'" in lines


def test_the_auth_header_and_rate_limit_headers_are_reported(
    api: MockApi, config: SourceConfig, kitchen: Kitchen
) -> None:
    """The source entry says which header carries the key and which budgets the API publishes."""
    kitchen.run(OpenApiStage({SOURCE: api.spec()}, configs={SOURCE: config}))
    declaration = kitchen.layer().sources[SOURCE]
    assert declaration.adapter == "openapi"
    assert declaration.auth is not None
    assert declaration.auth.header == AUTH_HEADER
    assert declaration.auth.secret == "rates_api_key", "a reference, never the value"
    lines = declaration.provenance.evidence if declaration.provenance is not None else None
    assert isinstance(lines, tuple)
    evidence = " ".join(lines)
    assert f"apiKey header: {AUTH_HEADER}" in evidence
    for header in RATE_LIMIT_HEADERS:
        assert header in evidence
    assert KEY not in kitchen.section().read_text(encoding="utf-8"), "no secret reaches the layer"


def test_the_skeleton_agrees_with_the_adapter_catalog(
    api: MockApi, config: SourceConfig, kitchen: Kitchen
) -> None:
    """Two readers of one document must not disagree: the skeleton is the adapter's own reading."""
    kitchen.run(OpenApiStage({SOURCE: api.spec()}, configs={SOURCE: config}))
    adapter = OpenApiAdapter(config, spec=api.spec())
    try:
        catalog = adapter.extract_catalog()
    finally:
        adapter.close()
    table = kitchen.layer().tables[f"{SOURCE}.rates"]
    live = catalog.tables[0]
    assert live.name == "rates"
    assert [column.name for column in table.columns] == [column.name for column in live.columns]
    assert [column.normalized_type.value for column in table.columns] == [
        column.normalized_type.value for column in live.columns
    ]
    assert [index.name for index in table.indexes] == [index.name for index in live.indexes]


# --- acceptance: what is not a table is reported, with a reason ---------------------------------


@pytest.mark.parametrize(
    ("label", "reason_fragment"),
    [
        ("GET /customers/{customer_id}/score", "path parameter"),
        ("GET /health", "not a JSON array"),
        ("GET /slow", "not a JSON array"),
        ("GET /_stats", "not a JSON array"),
    ],
)
def test_a_non_collection_is_skipped_with_a_reason(
    api: MockApi, config: SourceConfig, kitchen: Kitchen, label: str, reason_fragment: str
) -> None:
    """A lookup and a single-object answer are not tables, and the report says why."""
    result = kitchen.run(OpenApiStage({SOURCE: api.spec()}, configs={SOURCE: config}))
    skipped = [line for line in result.reports[0].skipped if line.startswith(f"{label}:")]
    assert len(skipped) == 1, result.reports[0].skipped
    assert reason_fragment in skipped[0]


def test_a_mutating_operation_is_excluded_and_reported(
    api: MockApi, config: SourceConfig, kitchen: Kitchen
) -> None:
    """A spec offering POST/PUT exposes no table for them: only GET is ever read (principle 5)."""
    document = copy.deepcopy(api.spec())
    document["paths"]["/rates"]["post"] = {"operationId": "create_rate", "responses": {}}
    document["paths"]["/rates"]["put"] = {"operationId": "replace_rate", "responses": {}}
    result = kitchen.run(OpenApiStage({SOURCE: document}, configs={SOURCE: config}))

    assert result.reports[0].counts["tables"] == 1, "the collection is still one table"
    skipped = " | ".join(result.reports[0].skipped)
    assert "POST /rates: a POST operation: only GET is ever exposed" in skipped
    assert "PUT /rates: a PUT operation: only GET is ever exposed" in skipped
    assert sorted(kitchen.layer().tables) == [f"{SOURCE}.rates"]


def test_a_collection_without_properties_is_skipped_with_a_reason(
    api: MockApi, config: SourceConfig, kitchen: Kitchen
) -> None:
    """An array of opaque objects has no columns, so it is not pretended into a table."""
    document = copy.deepcopy(api.spec())
    document["paths"]["/opaque"] = {
        "get": {
            "operationId": "opaque",
            "responses": {
                "200": {
                    "content": {
                        "application/json": {
                            "schema": {"type": "array", "items": {"type": "object"}}
                        }
                    }
                }
            },
        }
    }
    result = kitchen.run(OpenApiStage({SOURCE: document}, configs={SOURCE: config}))
    skipped = " | ".join(result.reports[0].skipped)
    assert "GET /opaque: its array items declare no properties: no columns to filter on" in skipped
    assert sorted(kitchen.layer().tables) == [f"{SOURCE}.rates"]


# --- the layer loads, and only generated/ is written --------------------------------------------


def test_the_generated_section_loads_as_a_semantic_layer(
    api: MockApi, config: SourceConfig, kitchen: Kitchen
) -> None:
    """The point of writing only known keys: the file is one section of a loadable layer."""
    kitchen.run(OpenApiStage({SOURCE: api.spec()}, configs={SOURCE: config}))
    layer = kitchen.layer()
    assert isinstance(layer, SemanticLayer)
    assert sorted(layer.sources) == [SOURCE]
    assert sorted(layer.tables) == [f"{SOURCE}.rates"]
    assert layer.version_hash, "a loaded layer is hashed, which is what a cache key uses"


def test_nothing_is_written_under_curated(
    api: MockApi, config: SourceConfig, kitchen: Kitchen
) -> None:
    """Humans own ``curated/``: the pipeline has no write path there; this proves it stayed out."""
    kitchen.run(OpenApiStage({SOURCE: api.spec()}, configs={SOURCE: config}))
    curated = kitchen.root / "curated"
    assert not curated.exists() or list(curated.iterdir()) == []


def test_a_document_that_is_not_a_spec_fails_the_stage_and_writes_nothing(
    config: SourceConfig, kitchen: Kitchen
) -> None:
    """A broken document is a stage failure with the reason, not a half-written section."""
    result = kitchen.run(OpenApiStage({SOURCE: {"openapi": "3.1.0"}}, configs={SOURCE: config}))
    report = result.reports[0]
    assert report.status is StageStatus.FAILED
    assert report.error is not None and "paths" in report.error
    assert not kitchen.section().exists()


def test_a_config_naming_another_adapter_is_refused(api: MockApi, kitchen: Kitchen) -> None:
    """D4 cannot claim a source the deployment says is a database."""
    stage = OpenApiStage(
        {SOURCE: api.spec()},
        configs={SOURCE: SourceConfig(name=SOURCE, adapter="postgresql")},
    )
    result = kitchen.run(stage)
    assert result.reports[0].status is StageStatus.FAILED
    assert "postgresql" in (result.reports[0].error or "")


def test_a_config_without_a_document_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="no OpenAPI document"):
        OpenApiStage(
            {SOURCE: {"paths": {}}},
            configs={"other_api": SourceConfig(name="other_api", adapter="openapi")},
        )


def test_two_sources_write_one_section_each_and_merge(api: MockApi, kitchen: Kitchen) -> None:
    """Sections are merged by the loader, so two APIs are one layer with two sources."""
    document = api.spec()
    second = "fx_api"
    stage = OpenApiStage({SOURCE: document, second: copy.deepcopy(document)})
    result = kitchen.run(stage)

    assert result.reports[0].counts["sources"] == 2
    assert kitchen.section().exists() and kitchen.section(second).exists()
    layer = kitchen.layer()
    assert sorted(layer.sources) == sorted([SOURCE, second])
    assert sorted(layer.tables) == sorted([f"{SOURCE}.rates", f"{second}.rates"])


# --- idempotence: the same document writes nothing ----------------------------------------------


def test_rerunning_with_an_unchanged_document_writes_nothing(
    api: MockApi, config: SourceConfig, kitchen: Kitchen
) -> None:
    """S01's rule: an input hash match skips the stage, and the file is untouched when it runs."""
    stage = OpenApiStage({SOURCE: api.spec()}, configs={SOURCE: config})
    first = kitchen.run(stage)
    assert first.reports[0].status is StageStatus.RAN
    before = kitchen.section().read_bytes()

    second = kitchen.run(stage)
    report = second.reports[0]
    assert report.status is StageStatus.SKIPPED
    assert report.skipped == ("inputs unchanged since the last successful run",)
    assert report.outputs == (), "a skipped stage commits nothing"
    assert kitchen.section().read_bytes() == before


def test_a_changed_document_reruns(api: MockApi, config: SourceConfig, kitchen: Kitchen) -> None:
    """A spec that gained a field is a different input, so the skeleton is rebuilt."""
    stage = OpenApiStage({SOURCE: api.spec()}, configs={SOURCE: config})
    kitchen.run(stage)

    document = copy.deepcopy(api.spec())
    row = document["components"]["schemas"]["RateRow"]
    row["properties"]["note"] = {"type": "string"}
    changed = OpenApiStage({SOURCE: document}, configs={SOURCE: config})
    result = kitchen.run(changed)
    assert result.reports[0].status is StageStatus.RAN
    assert "note" in [column.name for column in kitchen.layer().tables[f"{SOURCE}.rates"].columns]


def test_source_filter_narrows_the_pass(api: MockApi, kitchen: Kitchen) -> None:
    """``--source`` reads one document and reports the other as skipped, without writing it."""
    stage = OpenApiStage({SOURCE: api.spec(), "fx_api": api.spec()})
    result = kitchen.run(stage, source=SOURCE)

    statuses = {report.stage: report.status for report in result.reports}
    assert statuses["D4"] is StageStatus.RAN
    assert kitchen.section().exists()
    assert not kitchen.section("fx_api").exists()


# --- the deployed shape: documents from files ---------------------------------------------------


def test_read_specs_reads_the_document_from_the_configured_path(
    api: MockApi, tmp_path: Path, config: SourceConfig
) -> None:
    """A deployed source names a spec file; the stage ends up with the same tables."""
    path = tmp_path / "api.yaml"
    path.write_text(spec_yaml(api.spec()), encoding="utf-8", newline="\n")
    from_file = read_specs([config.model_copy(update={"spec": str(path)})])
    fetched, _ = read_endpoints(from_file[SOURCE])
    in_memory, _ = read_endpoints(api.spec())

    assert [table.table for table in fetched] == [table.table for table in in_memory]
    assert [column.name for column in fetched[0].columns] == [
        column.name for column in in_memory[0].columns
    ]
    assert read_specs([SourceConfig(name="crm", adapter="postgresql")]) == {}, (
        "a source that is not an API is not D4's business"
    )


def test_the_stage_needs_a_document() -> None:
    with pytest.raises(ValueError, match="at least one OpenAPI document"):
        OpenApiStage({})
