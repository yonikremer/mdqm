"""Command line entry point: ``mdqm serve``, ``mdqm audit`` and the version flag."""

import typer

from mdqm import __version__
from mdqm.config.settings import Settings
from mdqm.server.app import AuthConfigError
from mdqm.server.app import serve as serve_mcp
from mdqm.store.audit import DEFAULT_RETENTION_DAYS, AuditLog
from mdqm.store.db import Store

app = typer.Typer(help="Multi DB Query MCP", no_args_is_help=True, add_completion=False)
audit_app = typer.Typer(
    help="Audit log: verify the hash chain, apply retention.", no_args_is_help=True
)
app.add_typer(audit_app, name="audit")


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


if __name__ == "__main__":
    app()
