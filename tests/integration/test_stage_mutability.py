"""Stage D6 against the real, loaded database (S05 acceptance).

Skips without Docker Compose services; **fails** with ``--require-services``, because a skipped
test is not a pass (``AGENTS.md`` section 6).

What it asserts is the whole point of the stage: the classes come out of counters that were read
from the server, the evidence travels with them, and the lag bound is a measurement with a sample
size behind it -- not a number in a YAML file.

The world it expects is the one D03 loads (``mdqm.testing.loaders.postgres``): ``usage_events``
partitioned monthly on event time with a planted ingestion lag, ``accounts_hist`` the frequently
updated table, ``status_lookup`` the static lookup table.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest

from mdqm.adapters.postgresql import PostgresAdapter
from mdqm.config.settings import SourceConfig
from mdqm.models.partitioning import PartitionStrategy
from mdqm.pipeline.core import PipelineContext, StageStatus
from mdqm.pipeline.stages.mutability import (
    BASELINE_PREFIX,
    MutabilityStage,
    read_lag_unverified,
)
from mdqm.store.db import Store
from mdqm.testing.loaders.postgres import PgTarget, target_from_environ

EXEC_TS = datetime.now(UTC)


class _StaticSecrets:
    """A secret provider returning the throwaway dev password for any reference."""

    def __init__(self, password: str) -> None:
        self._password = password

    def get(self, name: str) -> str:
        assert name, "a secret reference must be a name, never an empty string"
        return self._password


def _unavailable(request: pytest.FixtureRequest, reason: str) -> None:
    if request.config.getoption("--require-services"):
        pytest.fail(f"required service unavailable: {reason}")
    pytest.skip(f"service unavailable: {reason}")


def _config(name: str, database: str, target: PgTarget) -> SourceConfig:
    return SourceConfig(
        name=name,
        adapter="postgresql",
        secret="billing_ro",
        options={
            "host": target.host,
            "port": target.port,
            "dbname": database,
            "user": target.ro_user,
            "connect_timeout": 5,
        },
    )


@pytest.fixture(scope="module")
def target(request: pytest.FixtureRequest) -> PgTarget:
    try:
        found = target_from_environ()
    except Exception as exc:  # noqa: BLE001 - any settings failure means "not configured here"
        _unavailable(request, f"PostgreSQL connection settings: {exc}")
        raise AssertionError("unreachable") from exc
    try:
        with psycopg.connect(found.dsn("billing"), connect_timeout=5):
            pass
    except psycopg.Error as exc:
        _unavailable(request, f"PostgreSQL at {found.host}:{found.port}: {exc}")
        raise AssertionError("unreachable") from exc
    return found


@pytest.fixture
def billing(target: PgTarget) -> Iterator[PostgresAdapter]:
    adapter = PostgresAdapter(
        _config("billing-s05", "billing", target), _StaticSecrets(target.ro_password)
    )
    try:
        yield adapter
    finally:
        adapter.close()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    opened = Store(tmp_path / "mdqm.sqlite")
    opened.initialize()
    try:
        yield opened
    finally:
        opened.close()


@pytest.fixture
def semantic_root(tmp_path: Path) -> Path:
    """An empty semantic root: D6 writes ``generated/`` and never needs a curated file."""
    root = tmp_path / "semantic"
    (root / "generated").mkdir(parents=True)
    return root


def _run(stage: MutabilityStage, store: Store, root: Path):
    """Run one pass and commit it: the runner commits the buffered sections, a test must too."""
    ctx = PipelineContext(store=store, exec_ts=EXEC_TS, semantic_root=root, force=True)
    report = stage.run(ctx)
    ctx.commit()
    return report


def _written(root: Path) -> dict[str, object]:
    path = root / "generated" / "mutability_billing.yaml"
    assert path.is_file(), f"{path} was not written"
    from ruamel.yaml import YAML

    document = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def test_the_catalog_reports_the_loaded_world(billing: PostgresAdapter) -> None:
    """A precondition, stated: without these tables the classifications below mean nothing."""
    catalog = billing.extract_catalog()
    names = {table.name for table in catalog.tables}
    assert {"usage_events", "accounts_hist", "status_lookup"} <= names
    usage = catalog.table("usage_events")
    assert usage is not None
    partitions = billing.extract_partitions()
    usage_set = [
        item
        for item in partitions
        if item.key and all(bound.name.startswith("usage_events_") for bound in item.bounds)
    ]
    assert usage_set, "usage_events must report its monthly partitions"
    assert usage_set[0].key == ("start_time",)
    assert usage_set[0].strategy is PartitionStrategy.RANGE


def test_stage_classifies_the_live_database(
    billing: PostgresAdapter, store: Store, semantic_root: Path
) -> None:
    """The three acceptance classes, from counters this run read twice."""
    stage = MutabilityStage({"billing": billing}, old_row_sample=200)
    report = _run(stage, store, semantic_root)
    assert report.status is StageStatus.RAN
    assert report.counts["tables"] > 0
    assert report.counts["lags_measured"] >= 1, report.describe()

    written = _written(semantic_root)
    tables = written["tables"]
    assert isinstance(tables, dict)

    usage = tables["billing.usage_events"]["mutability"]
    assert usage["class"] == "append_rarely", usage
    history = tables["billing.accounts_hist"]["mutability"]
    assert history["class"] == "mutable_critical", history
    lookup = tables["billing.status_lookup"]["mutability"]
    assert lookup["class"] == "append_only", lookup

    # A measurement is not a human approval: nothing measured here may be cached yet.
    for key in ("billing.usage_events", "billing.accounts_hist", "billing.status_lookup"):
        assert tables[key]["mutability"]["status"] == "proposed", key


def test_the_classification_carries_its_counter_evidence(
    billing: PostgresAdapter, store: Store, semantic_root: Path
) -> None:
    """A class without the counters behind it is a guess, so the evidence must be in the file."""
    _run(MutabilityStage({"billing": billing}, old_row_sample=200), store, semantic_root)
    tables = _written(semantic_root)["tables"]
    assert isinstance(tables, dict)

    evidence = tables["billing.accounts_hist"]["provenance"]["evidence"]
    assert isinstance(evidence, list) and evidence
    joined = " ".join(str(line) for line in evidence)
    assert "lifetime counters" in joined, joined
    assert "delta over" in joined, joined
    # accounts_hist is the table the world updates in place: the counters must say so.
    updates = [
        line for line in evidence if "lifetime counters" in str(line) and "updated=" in str(line)
    ]
    assert updates, evidence
    # `hot_updated=0` is a different counter, so the in-place update count is matched exactly.
    inline = str(updates[0]).split("hot_updated")[0]
    assert re.search(r"updated=[1-9][0-9]*", inline), updates[0]

    # usage_events is partitioned, so its counters are its partitions' (its own are zero).
    usage_evidence = " ".join(
        str(line) for line in tables["billing.usage_events"]["provenance"]["evidence"]
    )
    assert "summed over" in usage_evidence, usage_evidence
    assert "partitions" in usage_evidence, usage_evidence


def test_measured_lag_is_positive_and_backed_by_a_sample(
    billing: PostgresAdapter, store: Store, semantic_root: Path
) -> None:
    """``max_lag`` may only be cleared by a measurement: positive, bounded and sampled."""
    report = _run(MutabilityStage({"billing": billing}, lag_sample=5_000), store, semantic_root)
    assert report.counts["lags_measured"] >= 1

    tables = _written(semantic_root)["tables"]
    assert isinstance(tables, dict)
    columns = tables["billing.usage_events"]["time_columns"]
    assert isinstance(columns, dict)
    ingestion = columns["insertion_time"]
    assert ingestion["role"] == "ingestion_time"
    assert ingestion["derives_bound_on"] == "start_time"
    assert ingestion["lag"]["verify"] == "nightly"
    written_max = str(ingestion["lag"]["max"])
    assert written_max.endswith("s"), written_max
    assert int(written_max[:-1]) > 0, "a lag of zero would make the derived bound meaningless"

    evidence = " ".join(
        str(line) for line in tables["billing.usage_events"]["provenance"]["evidence"]
    )
    assert "rows" in evidence and "max=" in evidence, evidence

    # The measurement is also recorded as the baseline the next pass compares against, so the
    # nightly re-measurement (and a violation) has something to compare with.
    raw = store.connect().execute(
        "SELECT value FROM mdqm_meta WHERE key = ?", (f"{BASELINE_PREFIX}.billing",)
    ).fetchone()
    assert raw is not None
    document = json.loads(str(raw["value"]))
    assert document["maxima"], document
    assert document["window_to"] >= document["window_from"]


def test_no_lag_violation_when_the_declared_bound_holds(
    billing: PostgresAdapter, store: Store, semantic_root: Path
) -> None:
    """The world plants nothing above ``max_lag``, so nothing may be flagged as violated."""
    report = _run(MutabilityStage({"billing": billing}, old_row_sample=200), store, semantic_root)
    assert report.counts["lags_violated"] == 0, report.describe()
    assert read_lag_unverified(store, "billing") == frozenset()


def test_a_partitioned_parent_is_never_classified_from_its_own_zero_counters(
    billing: PostgresAdapter,
) -> None:
    """The server keeps a partitioned parent's counters at zero: reading only them would lie."""
    stats = {_bare(item.table): item for item in billing.mutability_stats()}
    parent = stats["usage_events"]
    assert parent.rows_inserted == 0, "the parent's own counter is the trap this stage avoids"
    children = [value for name, value in stats.items() if name.startswith("usage_events_p")]
    assert children, "the partitions must be in the counter reading"
    assert sum(item.rows_inserted or 0 for item in children) > 0


def _bare(name: str) -> str:
    return name.rsplit(".", 1)[-1]
