"""A minimal MCP client for `mdqm`, used by the benchmark and by hand.

Why this file exists: a weak model's job in the benchmark is to *choose a tool and its arguments*.
Giving it a one-command client removes the transport work (JSON-RPC framing, session handling,
bearer auth) from the measurement, so what is being scored is the tool choice and the answer, not
the model's ability to hand-roll HTTP.

    python scripts/mdqm_call.py list
    python scripts/mdqm_call.py call find_data '{"question": "average credit limit by status"}'
    python scripts/mdqm_call.py call query '{"spec": {...}}'

The token is read from a JSON token file (the same file `mdqm serve` uses), never from argv, so it
does not land in a shell history or a process list.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_URL = "http://127.0.0.1:58765/mcp"
DEFAULT_TOKENS = Path("docker/server/tokens.json")


def read_token(path: Path) -> str:
    """First token in the JSON token file the server reads. Never printed."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"{path}: {exc.strerror or exc}") from None
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: not valid JSON at line {exc.lineno}: {exc.msg}") from None
    tokens = document.get("tokens") or []
    if not tokens:
        raise SystemExit(f"{path}: no tokens in the file")
    return str(tokens[0]["token"])


async def run(url: str, token: str, action: str, name: str | None, raw: str | None) -> int:
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    headers = {"Authorization": f"Bearer {token}"}
    http_client = httpx2.AsyncClient(headers=headers, timeout=120.0)
    async with http_client, streamable_http_client(url, http_client=http_client) as streams:
        read_stream, write_stream = streams[0], streams[1]
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            if action == "list":
                listing = await session.list_tools()
                for tool in listing.tools:
                    description = (tool.description or "").strip().splitlines()[0:1]
                    print(f"{tool.name}\t{description[0] if description else ''}")
                return 0

            if action == "schema":
                listing = await session.list_tools()
                for tool in listing.tools:
                    if name in (None, tool.name):
                        print(
                            json.dumps(
                                {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "input_schema": tool.input_schema,
                                },
                                indent=2,
                                ensure_ascii=False,
                            )
                        )
                return 0

            try:
                arguments: dict[str, Any] = json.loads(raw or "{}")
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"the tool arguments are not JSON ({exc.msg} at column {exc.colno})"
                ) from None
            result = await session.call_tool(name or "", arguments)
            payload: dict[str, Any] = {
                "is_error": getattr(result, "isError", None),
                "structured": getattr(result, "structuredContent", None),
            }
            if payload["structured"] is None:
                payload["content"] = [
                    getattr(item, "text", None) for item in getattr(result, "content", []) or []
                ]
            print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
            return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "schema", "call"])
    parser.add_argument("name", nargs="?", help="tool name for `call`/`schema`")
    parser.add_argument("arguments", nargs="?", help="JSON object of tool arguments")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    args = parser.parse_args(argv)

    if not args.tokens.is_file():
        raise SystemExit(
            f"{args.tokens}: no token file. `python scripts/dev.py env` writes one, or pass --tokens."
        )
    token = read_token(args.tokens)
    try:
        return asyncio.run(run(args.url, token, args.action, args.name, args.arguments))
    except Exception as exc:  # noqa: BLE001 - a CLI: report, do not traceback at the operator
        print(f"mdqm_call: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
