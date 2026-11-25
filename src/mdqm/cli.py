"""Command line entry point: ``mdqm serve``, ``mdqm audit`` and the version flag."""

import typer

from mdqm import __version__
from mdqm.config.settings import Settings
from mdqm.probe.cli import probe as probe_command
from mdqm.review.queue import ReviewError, ReviewQueue
from mdqm.server.app import AuthConfigError
from mdqm.server.app import serve as serve_mcp
from mdqm.store.audit import DEFAULT_RETENTION_DAYS, AuditLog
from mdqm.store.db import Store

app = typer.Typer(help="Multi DB Query MCP", no_args_is_help=True, add_completion=False)
audit_app = typer.Typer(
    help="Audit log: verify the hash chain, apply retention.", no_args_is_help=True
)
review_app = typer.Typer(
    help="Review queue: approve or reject proposed semantic facts (the human gate).",
    no_args_is_help=True,
)
app.add_typer(audit_app, name="audit")
# `mdqm probe` (task A01): refresh generated/capabilities.yaml. The command body lives in
# mdqm.probe.cli so the probe package owns its own surface.
app.command(name="probe")(probe_command)
app.add_typer(review_app, name="review")


def _review_queue(opened: Store) -> ReviewQueue:
    """The queue over one store, with the settings' semantic root."""
    settings = Settings()
    return ReviewQueue(opened, settings.paths.semantic_dir, audit=AuditLog(opened))


@app.callback(invoke_without_command=True)
def main(version: bool = typer.Option(False, "--version", help="Print version and exit.")) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def serve(
    host: str | None = typer.Option(
        None, help="Bind address (default: mdqm.yaml/env server.host)."
    ),
    port: int | None = typer.Option(None, help="Bind port (default: mdqm.yaml/env server.port)."),
) -> None:
    """Run the MCP server over streamable HTTP on /mcp, plus /healthz and /context."""
    bound = Settings().server
    try:
        serve_mcp(host=host or bound.host, port=port or bound.port)
    except AuthConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@audit_app.command("verify")
def audit_verify() -> None:
    """Verify the audit hash chain. Exit 0 when intact, 1 when a link does not hold."""
    opened = Store(Settings().paths.metadata_db)
    try:
        opened.initialize()  # idempotent: a fresh file is migrated, a current one is only read
        verdict = AuditLog(opened).verify_chain()
    finally:
        opened.close()
    if not verdict.ok:
        typer.echo(verdict.describe(), err=True)
        raise typer.Exit(code=1)
    typer.echo(verdict.describe())


@audit_app.command("purge")
def audit_purge(
    days: int | None = typer.Option(
        None, help="Retention window in days (default: 365, DESIGN section 13)."
    ),
) -> None:
    """Remove audit rows past retention as a whole prefix and checkpoint the chain."""
    opened = Store(Settings().paths.metadata_db)
    try:
        opened.initialize()
        log = AuditLog(
            opened, retention_days=DEFAULT_RETENTION_DAYS if days is None else days
        )
        report = log.purge()
    finally:
        opened.close()
    typer.echo(
        f"purged {report.purged_rows} row(s), kept {report.kept_rows}; "
        f"chain resumes after id {report.checkpoint_id}"
    )
@review_app.command("list")
def review_list(
    all_: bool = typer.Option(False, "--all", help="Include proposals a human already decided."),
) -> None:
    """Show what waits for a human, highest confidence first."""
    opened = Store(Settings().paths.metadata_db)
    try:
        queue = _review_queue(opened)
        rows = queue.decided() if all_ else queue.pending()
        proposals = [
            (row.id, row.kind.value, row.target, row.confidence, row.status.value)
            for row in rows
        ]
    finally:
        opened.close()
    if not proposals:
        typer.echo("nothing to review")
        return
    for proposal, kind, target, confidence, status in proposals:
        typer.echo(f"{proposal[:12]}  {status:<10} {kind:<12} {target}  ({confidence:.2f})")


@review_app.command("approve")
def review_approve(
    proposal: str = typer.Argument(..., help="Proposal id (or its unique prefix)."),
    by: str = typer.Option(..., help="Who approves it; recorded in curated YAML and the audit."),
) -> None:
    """Write an approved proposal into curated/*.yaml, with provenance."""
    opened = Store(Settings().paths.metadata_db)
    try:
        queue = _review_queue(opened)
        target = _resolve(queue, proposal)
        try:
            approved = queue.approve(target, by=by)
        except ReviewError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    finally:
        opened.close()
    typer.echo(f"approved {approved.id[:12]} into curated/{approved.section}.yaml")


@review_app.command("reject")
def review_reject(
    proposal: str = typer.Argument(..., help="Proposal id (or its unique prefix)."),
    by: str = typer.Option(..., help="Who rejects it; the rejection is remembered."),
) -> None:
    """Record a rejection. Curated YAML is untouched; the row stays as negative memory."""
    opened = Store(Settings().paths.metadata_db)
    try:
        queue = _review_queue(opened)
        target = _resolve(queue, proposal)
        queue.reject(target, by=by)
    finally:
        opened.close()
    typer.echo(f"rejected {target[:12]}")


@review_app.command("export-patch")
def review_export_patch(
    proposal: str = typer.Argument(..., help="Proposal id (or its unique prefix)."),
) -> None:
    """Print the approval as a unified diff, for the pull-request flow."""
    opened = Store(Settings().paths.metadata_db)
    try:
        queue = _review_queue(opened)
        try:
            patch = queue.export_patch(_resolve(queue, proposal))
        except ReviewError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    finally:
        opened.close()
    typer.echo(patch, nl=False)


def _resolve(queue: ReviewQueue, proposal: str) -> str:
    """Accept a unique id prefix, so a human can type the 12 characters the digest shows."""
    try:
        queue.get(proposal)
        return proposal
    except ReviewError:
        pass
    matches = [
        row.id for row in (*queue.pending(), *queue.decided()) if row.id.startswith(proposal)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        typer.echo(f"error: no proposal starting with {proposal!r}", err=True)
    else:
        typer.echo(
            f"error: {proposal!r} matches {len(matches)} proposals; use more characters", err=True
        )
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()