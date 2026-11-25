"""``mdqm probe``: refresh ``generated/capabilities.yaml`` and report what changed.

Read-only against sources: it calls ``probe()`` (and the metadata-store hooks) and nothing else.
Re-runnable at any time, so the nightly job and an operator use the same command:

* a source that cannot be built is named on stderr and the run continues (exit code 1, so a
  nightly job notices instead of over-reporting health);
* an admin override is merged after the probe and survives the rewrite;
* the previous report is read first, so the output says what actually moved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mdqm.adapters.base import SourceAdapter
from mdqm.adapters.registry import AdapterError, AdapterRegistry
from mdqm.cache.freshness import SystemClock
from mdqm.config.settings import ConfigError, Settings
from mdqm.probe.overrides import OVERRIDES_FILE, load_overrides
from mdqm.probe.report import PROBE_SECTION, read_report, write_report
from mdqm.probe.run import run_probe
from mdqm.probe.runner import DEFAULT_PROBE_TIMEOUT_S, ProbeRunner, describe_failure
from mdqm.semantic.loader import GENERATED_DIR

__all__ = ["probe"]


# `Annotated` options, not calls in the defaults: the call belongs to typer's registration step,
# and a default that is a function call trips flake8-bugbear (B008).
def probe(
    source: Annotated[
        str | None, typer.Option("--source", help="Probe only this configured source name.")
    ] = None,
    root: Annotated[
        Path | None, typer.Option(help="Semantic root (default: settings.paths.semantic_dir).")
    ] = None,
    overrides: Annotated[
        Path | None, typer.Option(help=f"Admin overrides file (default: <root>/{OVERRIDES_FILE}).")
    ] = None,
    timeout: Annotated[
        float, typer.Option(help="Per-feature timeout in seconds.")
    ] = DEFAULT_PROBE_TIMEOUT_S,
) -> None:
    """Probe every configured source and rewrite ``generated/capabilities.yaml``."""
    settings = Settings()
    semantic_root = settings.paths.semantic_dir if root is None else Path(root)
    overrides_path = Path(overrides) if overrides is not None else semantic_root / OVERRIDES_FILE
    report_path = semantic_root / GENERATED_DIR / f"{PROBE_SECTION}.yaml"

    try:
        pinned = load_overrides(overrides_path)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    previous = None
    try:
        previous = read_report(report_path)
    except ConfigError as exc:
        # Registered, not fatal: the file is machine written, so the fix is to regenerate it.
        typer.echo(f"warning: {exc}", err=True)

    selected = [config for config in settings.sources if source is None or config.name == source]
    if source is not None and not selected:
        typer.echo(f"error: no configured source named {source!r}", err=True)
        raise typer.Exit(code=1)

    adapters: dict[str, SourceAdapter] = {}
    failures = 0
    try:
        registry = AdapterRegistry(secrets=settings.secret_provider())
        registry.discover()
    except AdapterError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    runner = ProbeRunner(timeout_s=timeout)
    try:
        for config in selected:
            try:
                adapters[config.name] = registry.create(config)
            except Exception as exc:
                # A plugin constructor may raise anything (driver import, DSN parse). The run
                # continues; the failure is named, and the exit code carries the verdict.
                failures += 1
                named = describe_failure(str(config.name), exc)
                typer.echo(f"error: source {config.name}: {named}", err=True)
        outcome = run_probe(
            adapters,
            overrides=pinned,
            previous=previous,
            runner=runner,
            clock=SystemClock(),
        )
    finally:
        runner.close()

    written = write_report(semantic_root, outcome.report)
    for name in sorted(outcome.report.sources):
        entry = outcome.report.sources[name]
        supported = sum(1 for cap in entry.capabilities.values() if cap.supported is True)
        unsupported = sum(1 for cap in entry.capabilities.values() if cap.supported is False)
        unknown = sum(1 for cap in entry.capabilities.values() if cap.supported is None)
        typer.echo(
            f"{name}: {supported} supported, {unsupported} unsupported, {unknown} unknown, "
            f"{len(entry.fallbacks)} fallback(s)"
        )
    typer.echo(outcome.diff.describe())
    if outcome.report.unregistered_overrides:
        names = ", ".join(outcome.report.unregistered_overrides)
        typer.echo(
            f"warning: overrides name capabilities the catalogue does not know: {names}",
            err=True,
        )
    typer.echo(f"wrote {written}")
    if failures:
        raise typer.Exit(code=1)
