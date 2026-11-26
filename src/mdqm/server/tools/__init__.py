"""The five coarse tools the agent sees (``docs/DESIGN.md`` section 9).

``query`` does most of the work; the other four are lookups around it. Descriptions are written for
a weak model: what it does, when to call it, and one example.

This module is the package's front door: it holds the one tool whose body is still a stub
(``query`` in M03) plus the shared answer model, and re-exports the other four from
:mod:`mdqm.server.tools.misc` (M05) and :mod:`mdqm.server.tools.find_data` (M02). Each tool gets
its own module as its body lands; the name, description and wire schema stay put, because that is
the agent's contract.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from mdqm.models.base import MDQMModel
from mdqm.server.context import current_context
from mdqm.server.tools.find_data import (
    EMPTY_HINT,
    FindDataAnswer,
    FindDataServices,
    SpecSkeleton,
    configure_find_data,
    current_find_data,
    find_data,
    reset_find_data,
    use_find_data,
)
from mdqm.server.tools.misc import (
    Answer,
    FeedbackAnswer,
    GetMoreAnswer,
    HistoryAnswer,
    HistoryItem,
    ServicesUnavailable,
    ToolServices,
    configure_services,
    current_services,
    feedback,
    get_more,
    history,
    reset_services,
    use_services,
)
from mdqm.spec.models import QuerySpec

__all__ = [
    "EMPTY_HINT",
    "TOOL_NAMES",
    "Answer",
    "FeedbackAnswer",
    "FindDataAnswer",
    "FindDataServices",
    "GetMoreAnswer",
    "HistoryAnswer",
    "HistoryItem",
    "NotImplementedResult",
    "ServicesUnavailable",
    "SpecSkeleton",
    "ToolServices",
    "configure_find_data",
    "configure_services",
    "current_find_data",
    "current_services",
    "feedback",
    "find_data",
    "get_more",
    "history",
    "query",
    "register_tools",
    "reset_find_data",
    "reset_services",
    "use_find_data",
    "use_services",
]

#: The five tools, in the order the agent should learn them.
TOOL_NAMES: tuple[str, ...] = ("find_data", "query", "get_more", "history", "feedback")


class NotImplementedResult(MDQMModel):
    """What a stub tool answers: an honest, typed "not built yet", plus the authenticated caller.

    ``user_id`` is here so a misissued token shows up in the first call instead of in an empty
    result. Once the pipeline lands, the user id moves into the history and audit records.
    """

    status: Literal["not_implemented"] = "not_implemented"
    surface: str = Field(description="Which tool (or endpoint) answered.")
    user_id: str = Field(description="The authenticated caller the server saw.")
    problem: str = Field(description="What is missing.")
    fix: str = Field(description="What to do instead, right now.")


def _not_implemented(surface: str, problem: str, fix: str) -> NotImplementedResult:
    """Build the stub answer for the authenticated caller of this message."""
    return NotImplementedResult(
        surface=surface, user_id=current_context().user_id, problem=problem, fix=fix
    )


def query(
    spec: QuerySpec,
) -> NotImplementedResult:
    """Run one query spec and get a summary, a small sample and a handle. This is the main tool.

    Send a spec, never SQL. The server lints it, enforces time bounds and partition pruning, picks
    the source and the unit conversions, and answers with a summary plus a `handle` you page with
    `get_more`. Errors carry `code`, `problem` and an exact `fix`.
    Example: query({"spec": {"from": "customers", "select": [{"measure": "credit_limit_r5000",
    "agg": "avg"}], "group_by": ["customer_status"], "limit": 50}})
    """
    return _not_implemented(
        "query",
        f"no executor is wired yet, so the spec from {spec.from_table!r} cannot run",
        "this build serves the protocol only; run `mdqm probe` then `mdqm bootstrap` once the "
        "pipeline tasks have landed",
    )


#: The tool functions in :data:`TOOL_NAMES` order. Names and docstrings come from here.
TOOL_FUNCTIONS: tuple[Callable[..., MDQMModel], ...] = (
    find_data,
    query,
    get_more,
    history,
    feedback,
)


def register_tools(server: MCPServer) -> None:
    """Register the five tools. The name is the function name; the docstring is the description."""
    for function in TOOL_FUNCTIONS:
        server.add_tool(function)
