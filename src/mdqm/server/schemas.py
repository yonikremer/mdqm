"""Export the tool schemas the server advertises, as one deterministic file for humans and reviews.

``docs/tool-schemas.json`` is generated, never hand-edited: regenerate it with
``python scripts/gen_tool_schemas.py``. A unit test compares the file with the live server, so a
changed signature or description cannot quietly drift out of the documentation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

__all__ = [
    "TOOL_SCHEMAS_PATH",
    "render_tool_schemas",
    "tool_schemas",
    "write_tool_schemas",
]

#: Repository-relative path of the generated document.
TOOL_SCHEMAS_PATH = Path("docs") / "tool-schemas.json"

GENERATED_BY = "scripts/gen_tool_schemas.py"


async def tool_schemas(server: MCPServer) -> dict[str, Any]:
    """The advertised schemas, sorted by tool name: input and output schema per tool."""
    tools = await server.list_tools()
    ordered = sorted(tools, key=lambda tool: tool.name)
    return {
        "generated_by": GENERATED_BY,
        "tools": [tool.model_dump(by_alias=True, exclude_none=True) for tool in ordered],
    }


def render_tool_schemas(document: dict[str, Any]) -> str:
    """Serialise the document: sorted keys, UTF-8, trailing newline, so a diff is a real change."""
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_tool_schemas(document: dict[str, Any], path: str | Path = TOOL_SCHEMAS_PATH) -> Path:
    """Write the document synchronously; the async caller awaits :func:`tool_schemas` first."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_tool_schemas(document), encoding="utf-8")
    return target
