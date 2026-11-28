"""``mdqm sizes``: measure how big the tables and partitions are, then read the numbers back.

Sizes are measured, never declared: a pass asks each source for its own rows and bytes, stores one
row per table and per partition, and the newest capture is what plans against those tables use. The
distribution (``p50``, ``p99``, spread) is what makes a plan's estimate honest -- a sum of partitions
cannot tell 34 even months from one month holding 20 times the median.

Exit codes, because a nightly job reads them: ``0`` when every source answered (or was reported
unavailable, which is a capability fact), ``1`` when a source failed.

No SQL is built here: the adapter runs the query, the parser turns rows into
:class:`~mdqm.models.TableSize`, and the store writes them through the chokepoint.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Annotated, cast

import typer

from mdqm.config.settings import ConfigError, Settings
from mdqm.models import TableSize
from mdqm.planner.sizes import SizeQuantiles
from mdqm.server.wiring import build_adapters
from mdqm.store.db import Store
from mdqm.store.sizes import SizeStore, as_json, describe

__all__ = ["sizes_app"]

sizes_app = typer.Typer(
    help="Measured table and partition sizes: measure them, show the distribution.",
    no_args_is_help=True,
)


def _store(settings: Settings) -> Store:
    """The metadata store, migrated: sizes live beside the ledger, not in the layer."""
    opened = Store(settings.paths.metadata_db)
    opened.initialize()
    return opened


@sizes_app.command("measure")
def measure(
    source: Annotated[
        str | None, typer.Option("--source", help="Measure only this configured source.")
    ] = None,
) -> None:
    """Ask every configured source for its table and partition sizes and store them."""
    settings = _settings()
    opened = _store(settings)
    try:
        adapters = build_adapters(settings)
        store = SizeStore(opened)
        measured = 0
        for name, adapter in sorted(adapters.items()):
            if source is not None and name != source:
                continue
            read_sizes = getattr(adapter, "sizes", None)
            if not callable(read_sizes):
                typer.echo(f"source {name}: skipped (this adapter reports no sizes)")
                continue
            measure_sizes = cast(Callable[[], list[TableSize]], read_sizes)
            try:
                sizes = measure_sizes()
            except Exception as exc:
                typer.echo(f"source {name}: failed ({type(exc).__name__}: {exc})", err=True)
                raise typer.Exit(code=1) from exc
            store.record(sizes)
            measured += len(sizes)
            typer.echo(describe(sizes))
    finally:
        opened.close()
    typer.echo(f"measured {measured} table(s); the newest capture answers every later plan")


@sizes_app.command("show")
def show(
    source: Annotated[str | None, typer.Option("--source", help="Only this source.")] = None,
    table: Annotated[
        str | None, typer.Option("--table", help="One table, by its bare name.")
    ] = None,
    as_json_output: Annotated[
        bool, typer.Option("--json", help="Print the numbers as JSON instead of lines.")
    ] = False,
) -> None:
    """Print the newest measured sizes, with the distribution per table."""
    settings = _settings()
    opened = _store(settings)
    try:
        store = SizeStore(opened)
        sizes: Sequence[TableSize]
        if table is not None:
            if source is None:
                typer.echo("error: --table needs --source (a table name is not unique)", err=True)
                raise typer.Exit(code=2)
            one = store.latest(source, table)
            sizes = () if one is None else (one,)
        else:
            sizes = tuple(
                size for size in store.latest_all() if source is None or size.source == source
            )
        captures = store.captures(limit=3)
    finally:
        opened.close()

    if not sizes:
        typer.echo("nothing measured yet: run `mdqm sizes measure`")
        return
    typer.echo(as_json(sizes) if as_json_output else describe(sizes))
    if captures:
        newest, samples = captures[0]
        typer.echo(f"newest capture {newest.isoformat()} ({samples} samples)")


@sizes_app.command("profile")
def profile(
    source: Annotated[str, typer.Argument(help="Source name, e.g. billing.")],
    table: Annotated[
        str, typer.Argument(help="Table name as the source reports it, e.g. usage_events.")
    ],
) -> None:
    """Print one table's distribution: the number a plan's estimate is built from."""
    settings = _settings()
    opened = _store(settings)
    try:
        quantiles = SizeStore(opened).profile(source, table)
    finally:
        opened.close()
    if quantiles is None:
        typer.echo(f"nothing measured for {source}.{table}: run `mdqm sizes measure`")
        raise typer.Exit(code=1)
    typer.echo(json.dumps(_render(quantiles), indent=2, sort_keys=True))


def _render(quantiles: SizeQuantiles) -> dict[str, object]:
    """The profile as plain JSON, with the derived ratios named."""
    return {
        "partitions": quantiles.partitions,
        "rows": quantiles.rows.model_dump(),
        "bytes": quantiles.bytes.model_dump(),
        "unbalanced": quantiles.unbalanced,
        "spread_ratio": round(quantiles.spread_ratio, 3),
    }


def _settings() -> Settings:
    """Settings, with the operator's file pinned by :mod:`mdqm.cli` when a command set it."""
    try:
        return Settings()
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
