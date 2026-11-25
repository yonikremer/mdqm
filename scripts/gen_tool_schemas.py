"""Regenerate docs/tool-schemas.json from the tool schemas the server advertises.

    python scripts/gen_tool_schemas.py

Run after changing a tool signature, a docstring (the description) or a result model. A unit test
compares the file with a live server, so a stale file fails the build instead of misleading a reader.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    import anyio

    from mdqm.server.app import create_server
    from mdqm.server.auth import StaticTokenVerifier
    from mdqm.server.schemas import TOOL_SCHEMAS_PATH, tool_schemas, write_tool_schemas

    # Schemas do not depend on who is authenticated: an empty token file is enough.
    server = create_server(verifier=StaticTokenVerifier())
    document = anyio.run(tool_schemas, server)
    target = write_tool_schemas(document, root / TOOL_SCHEMAS_PATH)
    print(f"wrote {target} ({len(document['tools'])} tools)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
