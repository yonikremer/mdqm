"""``mdqm bootstrap run``: one pass over the bootstrap stages (task S01).

The command is thin on purpose: :func:`mdqm.pipeline.core.run_pipeline` owns every rule (skip on an
unchanged input hash, resume after a failure, refuse a heavy stage outside its quiet window), and
this module only wires the settings to a context, prints the report and sets the exit code.

Exit codes, because a nightly job reads them: ``0`` when every stage ran or was skipped, and also
when heavy work was **deferred** to the next quiet window (nothing is wrong, the night is not over);
``1`` when a stage failed; ``2`` (typer's own code) for an unknown ``--stage``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from mdqm.config.settings import ConfigError, Settings
from mdqm.pipeline.core import PipelineContext, available_stages, run_pipeline
from mdqm.store.db import Store

__all__ = ["bootstrap_app", "context_for"]

bootstrap_app = typer.Typer(
    help="Bootstrap pipeline: build the semantic layer in idempotent, resumable stages.",
    no_args_is_help=True,
)


def context_for(
    settings: Settings, *, store: Store, now: datetime | None = None
) -> PipelineContext:
    """The context one pass runs in, derived from the settings (no silent defaults)."""
    return PipelineContext(
        store=store,
        exec_ts=now or datetime.now(UTC),
        semantic_root=settings.paths.semantic_dir,
        quiet_windows=tuple(settings.quiet_windows),
    )


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

    registry = available_stages()
    if not registry:
        typer.echo(
            "no bootstrap stage is registered in this build; nothing to run "
            "(each D-stage task adds its own object to mdqm.pipeline.core.available_stages)"
        )
        return

    opened = Store(settings.paths.metadata_db)
    try:
        ctx = context_for(settings, store=opened)
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
