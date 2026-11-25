"""Command line entry point: ``mdqm serve`` and the version flag."""

import typer

from mdqm import __version__
from mdqm.config.settings import Settings
from mdqm.server.app import AuthConfigError
from mdqm.server.app import serve as serve_mcp

app = typer.Typer(help="Multi DB Query MCP", no_args_is_help=True, add_completion=False)


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


if __name__ == "__main__":
    app()
