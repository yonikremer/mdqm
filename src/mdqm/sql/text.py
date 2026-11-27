"""Marking text as SQL we authored.

psycopg types ``Cursor.execute``'s first argument as ``LiteralString``, which no string computed at
runtime can ever satisfy. Every adapter runs SQL that is genuinely ours -- compiled through the AST,
inspected by the guard, or a module constant -- so that type is too narrow for the only correct use.

Rather than scatter ``# pyright: ignore`` across the adapters, the conversion lives here once. It
returns ``Any`` on purpose: ``Any`` is assignable to ``LiteralString``, which lets both checkers
agree without an ignore comment per call site, and the docstring is the record of the laundering.

Nothing here weakens the guarantee that values are bound: a value never becomes a query.
"""

from __future__ import annotations

from typing import Any


def as_query(text: str) -> Any:  # noqa: ANN401 - the laundered type is the point, see the module docstring
    """Return ``text`` as something a driver's ``LiteralString`` parameter accepts.

    Callers pass SQL they built themselves (a compiled fragment, a module constant). Using this
    helper is a statement that the string is *code*, not data; user input must never reach it.
    """
    return text
