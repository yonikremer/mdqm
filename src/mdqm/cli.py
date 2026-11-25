"""Command line entry point. Sub-commands are added by later tasks."""

import typer

from mdqm import __version__

app = typer.Typer(help="Multi DB Query MCP", no_args_is_help=True, add_completion=False)


@app.callback(invoke_without_command=True)
def main(version: bool = typer.Option(False, "--version", help="Print version and exit.")) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


if __name__ == "__main__":
    app()
