"""``mdqm bootstrap run``: one pass over the bootstrap stages (task S01).

The command is thin on purpose: :func:`mdqm.pipeline.core.run_pipeline` owns every rule (skip on an
unchanged input hash, resume after a failure, refuse a heavy stage outside its quiet window), and
this module only wires the settings to a context, prints the report and sets the exit code.

This is also the one place that **builds the stages**, because it is the one place with settings,
secrets and a store. ``mdqm.pipeline.core.available_stages`` only orders what it is given: it must
not import the stages, since every stage imports it.

A stage whose inputs this deployment does not have is *left out and named* (principle 10): an
operator reading the run sees ``D4 not registered: no OpenAPI document is configured`` rather than
a silent gap.

Exit codes, because a nightly job reads them: ``0`` when every stage ran or was skipped, and also
when heavy work was **deferred** to the next quiet window (nothing is wrong, the night is not over);
``1`` when a stage failed; ``2`` (typer's own code) for an unknown ``--stage``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from ruamel.yaml import YAML

from mdqm.adapters.base import SourceAdapter
from mdqm.adapters.registry import AdapterRegistry
from mdqm.config.settings import ConfigError, Settings, SourceConfig
from mdqm.llm.factory import describe_model, model_client
from mdqm.llm.result import StructuredLLM
from mdqm.models import CatalogSnapshot, Partitioning
from mdqm.pipeline.core import PipelineContext, Stage, available_stages, run_pipeline
from mdqm.pipeline.stages.catalog import CatalogStage, partitions_by_table
from mdqm.pipeline.stages.conversions import ColumnPair, ColumnRef, ConversionStage
from mdqm.pipeline.stages.drift import DriftStage
from mdqm.pipeline.stages.joins import JoinStage
from mdqm.pipeline.stages.mutability import MutabilityStage
from mdqm.pipeline.stages.openapi import OpenApiStage
from mdqm.pipeline.stages.profiling import ProfilingStage
from mdqm.pipeline.stages.propose_names import ProposeNamesStage
from mdqm.review.queue import ReviewQueue
from mdqm.semantic.errors import SemanticError
from mdqm.semantic.loader import load
from mdqm.semantic.models import SemanticLayer
from mdqm.store.db import Store

__all__ = ["bootstrap_app", "build_stages", "context_for"]

logger = logging.getLogger(__name__)

bootstrap_app = typer.Typer(
    help="Bootstrap pipeline: build the semantic layer in idempotent, resumable stages.",
    no_args_is_help=True,
)

#: The identity template a measure source carries when that source needs no conversion.
_IDENTITY: str = "{col}"


def context_for(
    settings: Settings,
    *,
    store: Store,
    now: datetime | None = None,
    llm: StructuredLLM | None = None,
    echo: Callable[[str], None] | None = None,
) -> PipelineContext:
    """The context one pass runs in, derived from the settings (no silent defaults).

    ``llm`` is the model a stage may ask (A1+A2 is the only one that does). Left as ``None``, it is
    built from ``settings.model`` through :func:`~mdqm.llm.factory.model_client`: an absent or
    half-configured endpoint yields **no client** and one reported line, never an exception, so the
    deterministic stages of the same pass still run. A caller that wants a different model (a test
    with ``FakeLLM``, or a deployment with its own client) passes it and nothing here touches the
    network.

    The client is not closed here: it lives as long as the pass, and a pass ends with the process
    (``mdqm bootstrap run``). Its key is resolved lazily on the first call, so building a context
    never reads a secret.
    """
    client = llm if llm is not None else model_client(settings, report=echo)
    return PipelineContext(
        store=store,
        exec_ts=now or datetime.now(UTC),
        semantic_root=settings.paths.semantic_dir,
        quiet_windows=tuple(settings.quiet_windows),
        llm=client,
    )


def _adapters(settings: Settings, *, echo: Callable[[str], None]) -> dict[str, SourceAdapter]:
    """One adapter per configured source, after asking the registry to discover its plugins.

    Discovery is explicit because an entry point is what a real deployment reads (F04): without
    this call every source fails with "no adapter registered for 'postgresql' (registered kinds:
    none)" and the pipeline has nothing to run against. A source that cannot be built is reported
    and skipped -- the other sources still bootstrap (a deployment installs only the extras it
    serves).
    """
    registry = AdapterRegistry(secrets=settings.secret_provider())
    found = registry.discover()
    echo(f"adapter kinds discovered: {', '.join(found) or 'none'}")
    built: dict[str, SourceAdapter] = {}
    for source in settings.sources:
        try:
            built[source.name] = registry.create(source)
        except Exception as exc:
            echo(f"source {source.name}: skipped ({exc})")
    return built


def _catalogs(
    adapters: Mapping[str, SourceAdapter], *, echo: Callable[[str], None]
) -> dict[str, CatalogSnapshot]:
    """Each source's catalog, read once here for the stages that need it *before* they run.

    A metadata read, not a data read. A source whose catalog cannot be read is reported: D1 will
    report the same failure, and A1+A2 then has nothing to propose for it.
    """
    read: dict[str, CatalogSnapshot] = {}
    for name, adapter in adapters.items():
        try:
            read[name] = adapter.extract_catalog()
        except Exception as exc:
            echo(f"source {name}: catalog unreadable ({type(exc).__name__}: {exc})")
    return read


def _layer(settings: Settings, *, echo: Callable[[str], None]) -> SemanticLayer | None:
    """The layer this deployment already has, or ``None`` when it has none yet.

    ``None`` is normal on a fresh installation. A layer that exists but does not load is a problem
    for the caller to see, so it is reported and treated as absent -- the pipeline is the thing that
    *writes* the layer, and refusing to bootstrap over a broken curated file would hide the fix.
    """
    root = settings.paths.semantic_dir
    has_files = any(
        entry.is_file()
        for sub in ("generated", "curated")
        for pattern in ("*.yaml", "*.yml")
        for entry in (root / sub).glob(pattern)
    )
    if not has_files:
        return None
    try:
        return load(root)
    except SemanticError as exc:
        echo(f"semantic layer at {root} does not load, so the stages that read it skip it ({exc})")
        return None


def _specs(settings: Settings, *, echo: Callable[[str], None]) -> dict[str, dict[str, Any]]:
    """Every configured OpenAPI document, parsed, keyed by source name (what D4 reads)."""
    documents: dict[str, dict[str, Any]] = {}
    for source in settings.sources:
        if not source.spec:
            continue
        path = Path(source.spec)
        try:
            document = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            echo(f"source {source.name}: OpenAPI document {path} unreadable ({exc})")
            continue
        if not isinstance(document, dict):
            echo(f"source {source.name}: OpenAPI document {path} is not a mapping")
            continue
        documents[source.name] = document
    return documents


def _conversion_pairs(
    layer: SemanticLayer | None,
    catalogs: Mapping[str, CatalogSnapshot],
    adapters: Mapping[str, SourceAdapter],
    *,
    echo: Callable[[str], None],
) -> dict[str, tuple[ColumnPair, ...]]:
    """The measure declarations D8 can test, as pairs against the measure's reference column.

    The **declaration** is the layer's: a measure that lists two or more source columns says those
    columns hold the same business field, which is a human-approved fact (or a proposal that was
    approved). D8's job is then to *rediscover* how each source stores it, so this function never
    guesses a pair from the data -- :class:`~mdqm.pipeline.stages.conversions.ColumnPair` is
    explicitly a declaration, not a detection.

    The reference column is the one the layer stores with no conversion (``{col}``): the measure's
    units are defined there. With no identity column the alphabetically first wins, and the choice
    is reported, because it decides the direction of the conversion D8 will propose.
    """
    if layer is None:
        return {}
    declared: dict[str, tuple[ColumnPair, ...]] = {}
    for measure_key in sorted(layer.measures):
        measure = layer.measures[measure_key]
        candidates: list[tuple[str, ColumnRef]] = []
        for column in sorted(measure.sources):
            parts = column.split(".")
            if len(parts) != 3:
                continue
            source, table, name = parts
            adapter = adapters.get(source)
            snapshot = catalogs.get(source)
            if adapter is None or snapshot is None:
                continue
            entry = next((item for item in snapshot.tables if item.name == table), None)
            if entry is None:
                echo(f"measure {measure_key}: {column} is not comparable (no such table)")
                continue
            declared_table = layer.tables.get(f"{source}.{table}")
            # The alignment key: the catalog's primary key, else the key the layer declares. A table
            # with planted duplicate rows (the synthetic erp) has no primary key, and dropping it
            # would quietly halve what D8 can check.
            keys: tuple[str, ...] = tuple(entry.primary_key) or tuple(
                () if declared_table is None else declared_table.key
            )
            if not keys:
                echo(
                    f"measure {measure_key}: {column} is not comparable (no key to align rows with)"
                )
                continue
            candidates.append(
                (
                    column,
                    ColumnRef(source=source, table=table, column=name, key=keys[0]),
                )
            )
        if len(candidates) < 2:
            continue
        identity = [
            item for item in candidates if measure.sources[item[0]].normalize.strip() == _IDENTITY
        ]
        reference = identity[0] if identity else candidates[0]
        if not identity:
            echo(
                f"measure {measure_key}: no source declares the identity conversion, so "
                f"{reference[1].label} is the reference and the direction of a proposed "
                "conversion is this run's choice"
            )
        declared[measure_key] = tuple(
            ColumnPair(reference=reference[1], other=ref)
            for column, ref in candidates
            if column != reference[0]
        )
    return declared


def build_stages(
    settings: Settings,
    *,
    store: Store,
    echo: Callable[[str], None] = lambda _message: None,
) -> tuple[Stage, ...]:
    """The bootstrap stages this deployment can run, in run order.

    Everything is derived from the settings and the sources themselves -- nothing organisation
    specific is assumed. What cannot be derived is named instead of invented.
    """
    adapters = _adapters(settings, echo=echo)
    catalogs = _catalogs(adapters, echo=echo)
    layer = _layer(settings, echo=echo)
    specs = _specs(settings, echo=echo)
    queue = ReviewQueue(store, settings.paths.semantic_dir)
    specs_configs: dict[str, SourceConfig] = {
        source.name: source for source in settings.sources if source.name in specs
    }

    stages: list[Stage] = []
    if adapters:
        stages.append(CatalogStage(adapters))
    else:
        echo("D1+D2 not registered: no source could be read")
    if specs:
        stages.append(OpenApiStage(specs, configs=specs_configs))
    else:
        echo("D4 not registered: no OpenAPI document is configured (sources[].spec)")
    if adapters:
        stages.append(ProfilingStage(adapters))
        stages.append(MutabilityStage(adapters))
        stages.append(JoinStage(adapters, queue=queue))
        stages.append(
            DriftStage(
                extract=_catalog_reader(adapters),
                sources=tuple(sorted(adapters)),
                extract_partitions=_partition_reader(adapters, catalogs),
                queue=queue,
            )
        )
    pairs = _conversion_pairs(layer, catalogs, adapters, echo=echo)
    for measure_key, measure_pairs in pairs.items():
        stages.append(ConversionStage(adapters, measure_pairs, measure=measure_key))
    if layer is not None and not pairs:
        echo(
            "D8 not registered: no measure declares two or more readable source columns, so there "
            "is no pair to compare (a measure's sources are the declaration)"
        )
    if catalogs:
        stages.append(
            ProposeNamesStage(
                catalogs,
                profiles=None,
                value_maps=dict(layer.value_maps) if layer is not None else None,
                measures=dict(layer.measures) if layer is not None else None,
            )
        )
        if layer is None:
            echo(
                "A1+A2 runs without value maps or measures: this deployment has no semantic "
                "layer yet"
            )
    else:
        echo("A1+A2 not registered: no catalog was read")
    return available_stages(stages)


def _catalog_reader(adapters: Mapping[str, SourceAdapter]) -> Callable[[str], CatalogSnapshot]:
    """D10 asks for one source's catalog by name; this is that lookup over the built adapters."""

    def read(source: str) -> CatalogSnapshot:
        return adapters[source].extract_catalog()

    return read


def _partition_reader(
    adapters: Mapping[str, SourceAdapter], catalogs: Mapping[str, CatalogSnapshot]
) -> Callable[[str], Mapping[str, Any]]:
    """D10's partition lookup, keyed by table the way it asks, and empty where none is declared.

    The flat list a source returns is mapped to its table here, by the same rule D1 uses
    (:func:`~mdqm.pipeline.stages.catalog.partitions_by_table`): a bound whose owner is ambiguous is
    dropped rather than attached to a neighbour, so D10 cannot report drift on a guess.
    """
    from mdqm.adapters.base import NotSupported

    def read(source: str) -> Mapping[str, Any]:
        snapshot = catalogs.get(source)
        if snapshot is None:
            return {}
        read_partitions = getattr(adapters[source], "extract_partitions", None)
        if not callable(read_partitions):
            return {}
        reader = cast(Callable[[], Sequence[Partitioning]], read_partitions)
        try:
            partitions = list(reader())
        except NotSupported:
            return {}
        return partitions_by_table(partitions, snapshot, source)

    return read


@bootstrap_app.command("run")
def run(
    stage: Annotated[
        str | None, typer.Option("--stage", help="Run only this stage, e.g. --stage D5.")
    ] = None,
    source: Annotated[
        str | None, typer.Option("--source", help="Narrow every stage to one configured source.")
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Run even when the inputs are unchanged or the quiet window is shut."
        ),
    ] = False,
    root: Annotated[
        Path | None, typer.Option(help="Semantic root (default: settings.paths.semantic_dir).")
    ] = None,
) -> None:
    """Run the registered stages in order, skipping the ones whose inputs did not change."""
    try:
        settings = Settings()
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    opened = Store(settings.paths.metadata_db)
    try:
        # The model comes first: it decides whether A1+A2 can propose anything at all, and an
        # operator diagnosing an empty run must see that before a list of stages.
        ctx = context_for(settings, store=opened, echo=typer.echo)
        typer.echo(describe_model(ctx.llm))
        registry = list(build_stages(settings, store=opened, echo=typer.echo))
        if not registry:
            typer.echo(
                "no bootstrap stage can run here: no source was readable and no OpenAPI document "
                "is configured, so there is nothing to read"
            )
            return
        if root is not None:
            ctx.semantic_root = Path(root)
        result = run_pipeline(registry, ctx, stage=stage, source=source, force=force)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    finally:
        opened.close()

    typer.echo(result.describe())
    if result.failed is not None:
        raise typer.Exit(code=1)
    if result.deferred_until is not None:
        typer.echo("deferred: heavy stages wait for the next quiet window; nothing was written")
