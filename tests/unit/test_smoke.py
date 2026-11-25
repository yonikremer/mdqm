import mdqm


def test_version() -> None:
    assert mdqm.__version__


def test_cli_version() -> None:
    from typer.testing import CliRunner

    from mdqm.cli import app

    res = CliRunner().invoke(app, ["--version"])
    assert res.exit_code == 0
    assert mdqm.__version__ in res.output


def test_mcp_sdk_is_v2() -> None:
    # Guards against accidentally resolving mcp 1.x (FastMCP API) in the PyPI mirror.
    from mcp.server.mcpserver import MCPServer  # noqa: F401
