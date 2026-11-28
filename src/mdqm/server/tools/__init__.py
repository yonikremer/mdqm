"""The five coarse tools the agent sees (``docs/DESIGN.md`` section 9).

``query`` does most of the work; the other four are lookups around it. Descriptions are written for
a weak model: what it does, when to call it, and one example.

This module is the package's front door. Each tool's body lives in its own module -- ``query`` in
:mod:`mdqm.server.tools.query` (M03), ``find_data`` in :mod:`mdqm.server.tools.find_data` (M02), the
three lookups in :mod:`mdqm.server.tools.misc` (M05) -- because the name, description and wire
schema are the agent's contract while the body keeps growing. A server built without a pipeline
still answers, in the ``not_implemented`` shape, instead of failing.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from mdqm.models.base import MDQMModel
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
from mdqm.server.tools.query import (
    PlanInfo,
    QueryAnswer,
    QueryServices,
    QueryServicesUnavailable,
    SegmentFreshness,
    configure_query_services,
    current_query_services,
    query,
    reset_query_services,
    use_query_services,
)

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
    "PlanInfo",
    "QueryAnswer",
    "QueryServices",
    "QueryServicesUnavailable",
    "SegmentFreshness",
    "ServicesUnavailable",
    "SpecSkeleton",
    "ToolServices",
    "configure_find_data",
    "configure_query_services",
    "configure_services",
    "current_find_data",
    "current_query_services",
    "current_services",
    "feedback",
    "find_data",
    "get_more",
    "history",
    "query",
    "register_tools",
    "reset_find_data",
    "reset_query_services",
    "reset_services",
    "use_find_data",
    "use_query_services",
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


#: The tool functions in :data:`TOOL_NAMES` order. Names and docstrings come from here.
TOOL_FUNCTIONS: tuple[Callable[..., MDQMModel], ...] = (
    find_data,
    query,
    get_more,
    history,
    feedback,
)


def register_tools(server: MCPServer) -> None:
    """Register the five tools. The name is the function name; the docstring is the description.

    Python 3.13 dedents a function's ``__doc__`` itself; 3.11 and 3.12 keep the source indent.
    The advertised schema is compared to a single generated file, so the description has to be the
    dedented text on every version CI runs. ``cleandoc`` is idempotent on an already-clean docstring.
    """
    for function in TOOL_FUNCTIONS:
        doc = function.__doc__
        if doc:
            function.__doc__ = inspect.cleandoc(doc)
        server.add_tool(function)
