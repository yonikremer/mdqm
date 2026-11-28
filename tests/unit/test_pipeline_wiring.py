"""The bootstrap wiring: which stages this deployment registers, in which order, and what it says.

The gap this pins: ``available_stages`` returned ``()`` on purpose and ``pipeline/cli.py`` never
asked the adapter registry to discover its entry points, so ``mdqm bootstrap run`` answered
``source crm: skipped (no adapter registered for 'postgresql' (registered kinds: none))`` followed by
``no bootstrap stage is registered in this build``. Both causes are covered here: the order
(:func:`~mdqm.pipeline.core.available_stages`), and the derivation
(:func:`~mdqm.pipeline.cli.build_stages`).

Nothing here touches a source: the adapters, catalogs, layer and OpenAPI documents are all fakes, so
a test asserts *composition* and a failure names the composition, not a database.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from mdqm.cli import app
from mdqm.config.settings import Settings
from mdqm.models import CatalogSnapshot, Column, NormalizedType, Table
from mdqm.pipeline import cli as pipeline_cli
from mdqm.pipeline.core import (
    STAGE_ORDER,
    PipelineContext,
    StageReport,
    StageStatus,
    available_stages,
)
from mdqm.pipeline.stages.conversions import ColumnPair, ConversionStage
from mdqm.semantic.loader import load
from mdqm.store.db import Store

NOW = datetime(2025, 11, 1, tzinfo=UTC)


class _Stage:
    """A stage that names itself and does nothing: the order is the only thing under test."""

    heavy = False
    measure: str | None = None

    def __init__(self, name: str, *, measure: str | None = None) -> None:
        self.name = name
        self.measure = measure

    def inputs_hash(self, ctx: PipelineContext) -> str:
        return name_of(self)

    def run(self, ctx: PipelineContext) -> StageReport:
        return StageReport(stage=self.name, status=StageStatus.SKIPPED)


def name_of(stage: _Stage) -> str:
    return stage.name


# --- the order --------------------------------------------------------------------------------


def test_every_known_stage_appears_once_in_the_declared_order() -> None:
    assert len(set(STAGE_ORDER)) == len(STAGE_ORDER), "a duplicated name would double-run a stage"
    given = [_Stage(name) for name in reversed(STAGE_ORDER)]

    ordered = [stage.name for stage in available_stages(given)]

    assert ordered == list(STAGE_ORDER)


def test_a_stage_the_order_does_not_list_runs_last_by_name() -> None:
    given = [_Stage("ZZ-new"), _Stage("D5"), _Stage("AA-new")]

    ordered = [stage.name for stage in available_stages(given)]

    assert ordered == ["D5", "AA-new", "ZZ-new"]


def test_two_instances_of_one_stage_keep_the_order_they_were_given() -> None:
    # D8 is built once per measure, and every instance carries the same stage name.
    first_measure, second_measure = "credit_limit_r5000", "usage_units"
    first = _Stage("D8", measure=first_measure)
    second = _Stage("D8", measure=second_measure)

    ordered = available_stages([first, second])

    assert [getattr(stage, "measure", None) for stage in ordered] == [first_measure, second_measure]


# --- the derivation ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Settings with three Postgres sources, read from the environment the way a deployment does."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "MDQM_SOURCES",
        json.dumps(
            [
                {
                    "name": name,
                    "adapter": "postgresql",
                    "secret": "pg_ro_password",
                    "options": {"host": "pg", "dbname": name, "user": "mdqm_ro"},
                }
                for name in ("crm", "billing", "erp")
            ]
        ),
    )
    monkeypatch.setenv("MDQM_PATHS__SEMANTIC_DIR", str(tmp_path / "semantic"))
    yield Settings()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    opened = Store(tmp_path / "metadata.sqlite")
    try:
        yield opened
    finally:
        opened.close()


def _catalog(source: str, *tables: tuple[str, str]) -> CatalogSnapshot:
    """One source's catalog: each argument is (table, primary-key column)."""
    return CatalogSnapshot(
        source=source,
        captured_at=NOW,
        tables=tuple(
            Table(
                name=table,
                columns=(
                    Column(name=key, native_type="integer", normalized_type=NormalizedType.INTEGER),
                ),
                primary_key=(key,),
            )
            for table, key in tables
        ),
    )


def _catalogs() -> dict[str, CatalogSnapshot]:
    return {
        "crm": _catalog("crm", ("customers", "cust_id")),
        "billing": _catalog("billing", ("accounts", "client_no")),
        "erp": _catalog("erp", ("customers", "cust_no")),
    }


def _layer(
    tmp_path: Path, *, normalizations: dict[str, str], keys: dict[str, str] | None = None
) -> Any:
    """A layer holding one measure with one source column per normalisation.

    ``keys`` adds ``tables: <table>: key: [<column>]``, which is how a table without a primary key
    still gets an entity key to align rows by.
    """
    root = tmp_path / "semantic"
    (root / "curated").mkdir(parents=True, exist_ok=True)
    sources = "\n".join(
        f"      {column}: {{ normalize: '{template}' }}"
        for column, template in normalizations.items()
    )
    tables = ""
    if keys:
        tables = "tables:\n" + "".join(
            f"  {table}:\n    key: [{column}]\n" for table, column in sorted(keys.items())
        )
    (root / "curated" / "measures.yaml").write_text(
        f"version: 1\n{tables}measures:\n  credit_limit_r5000:\n    sources:\n{sources}\n",
        encoding="utf-8",
    )
    return load(root)


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    *,
    adapters: dict[str, Any],
    catalogs: dict[str, CatalogSnapshot],
    layer: Any,
    specs: dict[str, Any] | None = None,
) -> list[str]:
    """Run ``build_stages`` over fakes and return the echoed lines."""
    echoed: list[str] = []
    monkeypatch.setattr(pipeline_cli, "_adapters", lambda _settings, *, echo: adapters)
    monkeypatch.setattr(pipeline_cli, "_catalogs", lambda _adapters, *, echo: catalogs)
    monkeypatch.setattr(pipeline_cli, "_layer", lambda _settings, *, echo: layer)
    monkeypatch.setattr(pipeline_cli, "_specs", lambda _settings, *, echo: specs or {})
    monkeypatch.setattr(pipeline_cli, "ReviewQueue", lambda *_args, **_kwargs: None)
    return echoed


def test_no_readable_source_names_the_stages_it_cannot_register(
    settings: Settings, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    echoed: list[str] = []
    monkeypatch.setattr(pipeline_cli, "_adapters", lambda _s, *, echo: {})
    monkeypatch.setattr(pipeline_cli, "_catalogs", lambda _a, *, echo: {})
    monkeypatch.setattr(pipeline_cli, "_layer", lambda _s, *, echo: None)
    monkeypatch.setattr(pipeline_cli, "_specs", lambda _s, *, echo: {})
    monkeypatch.setattr(pipeline_cli, "ReviewQueue", lambda *_a, **_k: None)

    assert pipeline_cli.build_stages(settings, store=store, echo=echoed.append) == ()
    assert any("D1+D2 not registered" in line for line in echoed)
    assert any("D4 not registered" in line for line in echoed)
    assert any("A1+A2 not registered" in line for line in echoed)


def test_the_stages_that_need_only_adapters_are_registered(
    settings: Settings, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapters = {name: object() for name in ("crm", "billing", "erp")}
    _wire(monkeypatch, adapters=adapters, catalogs=_catalogs(), layer=None)

    names = [stage.name for stage in pipeline_cli.build_stages(settings, store=store)]

    assert names == ["D1+D2", "D5", "D6", "D7", "D10", "A1+A2"]


def test_an_absent_openapi_document_leaves_d4_out(
    settings: Settings, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapters = {"crm": object()}
    _wire(monkeypatch, adapters=adapters, catalogs={"crm": _catalog("crm")}, layer=None)

    names = [stage.name for stage in pipeline_cli.build_stages(settings, store=store)]

    assert "D4" not in names


def test_one_conversion_stage_per_measure_and_it_compares_every_other_source(
    settings: Settings, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    layer = _layer(
        settings.paths.semantic_dir,
        normalizations={
            "crm.customers.credit_limit": "{col}",
            "billing.accounts.credit_limit": "ROUND({col} / 5000) * 5000",
            "erp.customers.limit_k": "{col} * 10",
        },
    )
    _wire(
        monkeypatch,
        adapters={name: object() for name in _catalogs()},
        catalogs=_catalogs(),
        layer=layer,
    )

    stages = pipeline_cli.build_stages(settings, store=store)

    conversion = [stage for stage in stages if isinstance(stage, ConversionStage)]
    assert len(conversion) == 1
    pairs: Sequence[ColumnPair] = conversion[0].pairs  # the stage's declared input, read-only
    assert {pair.reference.label for pair in pairs} == {"crm.customers.credit_limit"}
    assert {pair.other.label for pair in pairs} == {
        "billing.accounts.credit_limit",
        "erp.customers.limit_k",
    }


def test_a_measure_with_one_readable_source_is_not_a_pair(
    settings: Settings, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    layer = _layer(
        settings.paths.semantic_dir,
        normalizations={
            "crm.customers.credit_limit": "{col}",
            "erp.customers.limit_k": "{col} * 10",
        },
    )
    # erp is configured but its catalog could not be read, so its column cannot be compared.
    _wire(
        monkeypatch,
        adapters={name: object() for name in _catalogs()},
        catalogs={"crm": _catalog("crm", ("customers", "cust_id"))},
        layer=layer,
    )

    names = [stage.name for stage in pipeline_cli.build_stages(settings, store=store)]

    assert "D8" not in names


def test_a_missing_identity_source_is_reported_because_it_sets_the_direction(
    tmp_path: Path, settings: Settings, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    layer = _layer(
        settings.paths.semantic_dir,
        normalizations={
            "billing.accounts.credit_limit": "ROUND({col} / 5000) * 5000",
            "erp.customers.limit_k": "{col} * 10",
        },
    )
    echoed: list[str] = []
    monkeypatch.setattr(
        pipeline_cli, "_adapters", lambda _s, *, echo: {name: object() for name in _catalogs()}
    )
    monkeypatch.setattr(pipeline_cli, "_catalogs", lambda _a, *, echo: _catalogs())
    monkeypatch.setattr(pipeline_cli, "_layer", lambda _s, *, echo: layer)
    monkeypatch.setattr(pipeline_cli, "_specs", lambda _s, *, echo: {})
    monkeypatch.setattr(pipeline_cli, "ReviewQueue", lambda *_a, **_k: None)

    pipeline_cli.build_stages(settings, store=store, echo=echoed.append)

    assert any("no source declares the identity conversion" in line for line in echoed)


# --- the command ------------------------------------------------------------------------------


def test_cli_bootstrap_run_says_why_it_can_do_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MDQM_PATHS__METADATA_DB", str(tmp_path / "metadata.sqlite"))
    monkeypatch.delenv("MDQM_SOURCES", raising=False)

    result = CliRunner().invoke(app, ["bootstrap", "run"])

    assert result.exit_code == 0, result.output
    assert "no bootstrap stage can run here" in result.output


def test_a_table_without_a_primary_key_uses_the_key_the_layer_declares(
    settings: Settings, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The synthetic `erp.customers` has planted duplicates and so no primary key; the curated
    layer declares `cust_no` as its key. Dropping it would silently halve what D8 can check."""
    layer = _layer(
        settings.paths.semantic_dir,
        normalizations={
            "crm.customers.credit_limit": "{col}",
            "erp.customers.credit_limit": "{col} * 10",
        },
        keys={"erp.customers": "cust_no"},
    )
    catalogs = _catalogs()
    # the same catalog, with erp's primary key removed
    catalogs["erp"] = CatalogSnapshot(
        source="erp",
        captured_at=NOW,
        tables=(
            Table(
                name="customers",
                columns=(
                    Column(
                        name="credit_limit",
                        native_type="numeric",
                        normalized_type=NormalizedType.DECIMAL,
                    ),
                ),
            ),
        ),
    )
    _wire(
        monkeypatch, adapters={name: object() for name in catalogs}, catalogs=catalogs, layer=layer
    )

    echoed: list[str] = []
    monkeypatch.setattr(
        pipeline_cli, "_adapters", lambda _s, *, echo: {name: object() for name in catalogs}
    )
    monkeypatch.setattr(pipeline_cli, "_catalogs", lambda _a, *, echo: catalogs)
    monkeypatch.setattr(pipeline_cli, "_layer", lambda _s, *, echo: layer)
    monkeypatch.setattr(pipeline_cli, "_specs", lambda _s, *, echo: {})
    monkeypatch.setattr(pipeline_cli, "ReviewQueue", lambda *_a, **_k: None)
    stages = pipeline_cli.build_stages(settings, store=store, echo=echoed.append)

    conversions = [stage for stage in stages if isinstance(stage, ConversionStage)]
    assert len(conversions) == 1
    assert {pair.other.label for pair in conversions[0].pairs} == {
        "erp.customers.credit_limit",
    }
    assert conversions[0].pairs[0].other.key == "cust_no", "the declared key aligns the rows"


def test_a_measure_column_with_no_key_at_all_is_reported(
    settings: Settings, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silence would be a pair that quietly was not compared."""
    layer = _layer(
        settings.paths.semantic_dir,
        normalizations={
            "crm.customers.credit_limit": "{col}",
            "erp.customers.credit_limit": "{col} * 10",
        },
    )
    catalogs = _catalogs()
    catalogs["erp"] = CatalogSnapshot(
        source="erp",
        captured_at=NOW,
        tables=(
            Table(
                name="customers",
                columns=(
                    Column(
                        name="credit_limit",
                        native_type="numeric",
                        normalized_type=NormalizedType.DECIMAL,
                    ),
                ),
            ),
        ),
    )
    echoed: list[str] = []
    monkeypatch.setattr(
        pipeline_cli, "_adapters", lambda _s, *, echo: {name: object() for name in catalogs}
    )
    monkeypatch.setattr(pipeline_cli, "_catalogs", lambda _a, *, echo: catalogs)
    monkeypatch.setattr(pipeline_cli, "_layer", lambda _s, *, echo: layer)
    monkeypatch.setattr(pipeline_cli, "_specs", lambda _s, *, echo: {})
    monkeypatch.setattr(pipeline_cli, "ReviewQueue", lambda *_a, **_k: None)

    pipeline_cli.build_stages(settings, store=store, echo=echoed.append)

    assert any("no key to align rows with" in line for line in echoed)


# --- the sizes stage ----------------------------------------------------------------------


class _Sizing:
    """An adapter double that declares ``sizes()``: presence is derived from that, nothing else."""

    def sizes(self) -> list[Any]:
        return []


def test_a_source_that_reports_sizes_adds_the_sizes_stage_last(
    settings: Settings, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The profile is kept fresh by the pass that already runs nightly, not by a second scheduler."""
    adapters: dict[str, Any] = {name: _Sizing() for name in ("crm", "billing", "erp")}
    _wire(monkeypatch, adapters=adapters, catalogs=_catalogs(), layer=None)

    names = [stage.name for stage in pipeline_cli.build_stages(settings, store=store)]

    assert names[-1] == "sizes"
    assert names.count("sizes") == 1


def test_no_source_with_sizes_leaves_the_stage_out_and_names_it(
    settings: Settings, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    echoed: list[str] = []
    monkeypatch.setattr(
        pipeline_cli, "_adapters", lambda _s, *, echo: {name: object() for name in _catalogs()}
    )
    monkeypatch.setattr(pipeline_cli, "_catalogs", lambda _a, *, echo: _catalogs())
    monkeypatch.setattr(pipeline_cli, "_layer", lambda _s, *, echo: None)
    monkeypatch.setattr(pipeline_cli, "_specs", lambda _s, *, echo: {})
    monkeypatch.setattr(pipeline_cli, "ReviewQueue", lambda *_a, **_k: None)

    names = [
        stage.name for stage in pipeline_cli.build_stages(settings, store=store, echo=echoed.append)
    ]

    assert "sizes" not in names
    assert any("sizes not registered" in line for line in echoed)
