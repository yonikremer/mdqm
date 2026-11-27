"""Stage D1+D2 against the real database: the partitions it writes are the server's own.

The unit tests prove the shape; this proves the *numbers come from PostgreSQL*. It also checks the
one thing that would make the stage useless in practice: that what it writes loads back through the
semantic loader, because a generated file the loader rejects is a file nobody can use.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from ruamel.yaml import YAML
from tests.conftest import service_unavailable

from mdqm.pipeline.core import PipelineContext
from mdqm.pipeline.stages.catalog import SECTION, CatalogStage
from mdqm.semantic.loader import load
from mdqm.store import Store
from mdqm.testing.loaders.postgres import PgTarget, target_from_environ

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pg(require_services: bool) -> PgTarget:
    try:
        import psycopg

        target = target_from_environ()
        with psycopg.connect(target.dsn("billing"), connect_timeout=5):
            return target
    except Exception as exc:  # noqa: BLE001 - any failure means "no service here"
        service_unavailable(require_services, f"PostgreSQL: {exc}")
        raise AssertionError("unreachable: service_unavailable either fails or skips") from exc


@pytest.fixture(scope="module")
def written(pg: PgTarget, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, int]]:
    """Run the stage once against the live server and return the generated tree and its counts."""
    from mdqm.adapters.postgresql import PostgresAdapter
    from mdqm.config.settings import SourceConfig

    class _Secrets:
        def get(self, name: str) -> str:
            return pg.ro_password

    adapter = PostgresAdapter(
        SourceConfig(
            name="billing",
            adapter="postgresql",
            secret="billing_ro",
            options={
                "host": pg.host,
                "port": pg.port,
                "dbname": "billing",
                "user": pg.ro_user,
            },
        ),
        _Secrets(),
    )
    root = tmp_path_factory.mktemp("semantic")
    context = PipelineContext(
        store=Store(root / "metadata.sqlite"),
        exec_ts=datetime(2026, 1, 1, tzinfo=UTC),
        semantic_root=root,
    )
    try:
        stage = CatalogStage({"billing": adapter})
        stage.inputs_hash(context)
        report = stage.run(context)
        context.commit()
    finally:
        adapter.close()
    return root / "generated" / f"{SECTION}.yaml", dict(report.counts)


def test_the_live_partitions_are_written_for_usage_events(
    written: tuple[Path, dict[str, int]],
) -> None:
    path, counts = written
    document = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    events: dict[str, Any] = document["tables"]["billing.usage_events"]
    partitioning = events["partitioning"]
    assert partitioning["key"] == ["start_time"], "the range key is the event time"
    assert partitioning["granularity"] == "month"
    count = len(partitioning["bounds"])
    assert count >= 12, f"expected 12+ partitions, got {count}"
    assert counts["billing.partitions"] >= 12
    # A bound carries a real datetime range, not an opaque string: the planner needs the bounds.
    assert partitioning["bounds"][0]["lower"], partitioning["bounds"][0]


def test_the_written_layer_loads_back_through_the_loader(
    written: tuple[Path, dict[str, int]],
) -> None:
    path, _ = written
    layer = load(path)
    table = layer.tables["billing.usage_events"]
    assert table.partitioning is not None
    assert table.partitioning.key == ("start_time",)
    assert len(table.partitioning.bounds) >= 12
    assert table.mutability is None, "mutability is S05's conclusion, never assumed here"


def test_a_second_read_of_the_same_catalog_hashes_the_same(pg: PgTarget) -> None:
    """The runner skips on an equal hash, so an unchanged schema must hash equal."""
    from mdqm.adapters.postgresql import PostgresAdapter
    from mdqm.config.settings import SourceConfig

    class _Secrets:
        def get(self, name: str) -> str:
            return pg.ro_password

    def build() -> PostgresAdapter:
        return PostgresAdapter(
            SourceConfig(
                name="billing",
                adapter="postgresql",
                secret="billing_ro",
                options={
                    "host": pg.host,
                    "port": pg.port,
                    "dbname": "billing",
                    "user": pg.ro_user,
                },
            ),
            _Secrets(),
        )

    context = PipelineContext(
        store=Store(Path.cwd() / "unused-metadata.sqlite"),
        exec_ts=datetime(2026, 1, 1, tzinfo=UTC),
        semantic_root=Path.cwd(),
    )
    first_adapter, second_adapter = build(), build()
    try:
        first = CatalogStage({"billing": first_adapter}).inputs_hash(context)
        second = CatalogStage({"billing": second_adapter}).inputs_hash(context)
    finally:
        first_adapter.close()
        second_adapter.close()
    assert first == second
