"""The five coarse tools the agent sees (``docs/DESIGN.md`` section 9).

``query`` does most of the work; the other four are lookups around it. Descriptions are written for
a weak model: what it does, when to call it, and one example. The bodies are stubs in M01 -- they
return a typed ``NOT_IMPLEMENTED`` answer and never invent data. The pipeline tasks replace each
body; the signature, the description and the wire schema stay.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from mdqm.models.base import MDQMModel
from mdqm.server.context import current_context
from mdqm.spec.models import QuerySpec

__all__ = [
    "TOOL_NAMES",
    "NotImplementedResult",
    "feedback",
    "find_data",
    "get_more",
    "history",
    "query",
    "register_tools",
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


def find_data(
    question: str,
    limit: Annotated[int, Field(ge=1, le=50, description="How many matches to return.")] = 10,
) -> NotImplementedResult:
    """Find tables, columns and values that match a question. Call this first when unsure of a name.

    Searches the semantic dictionary (Hebrew and English names, synonyms, categorical values) and
    returns what exists, so you never guess a table or column name.
    Example: find_data({"question": "credit limit by customer status"})
    """
    return _not_implemented(
        "find_data",
        f"the semantic dictionary is not loaded yet, so {question!r} cannot be searched",
        "search is not available in this build; call `query` with a spec from the examples, or ask "
        "the operator to run the bootstrap pipeline (`mdqm bootstrap`)",
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


def get_more(
    handle: str,
    offset: Annotated[
        int, Field(ge=0, description="Rows to skip; 0 for the next page in order.")
    ] = 0,
    limit: Annotated[int, Field(ge=1, le=1000, description="Rows to return.")] = 100,
) -> NotImplementedResult:
    """Page the rest of a result that `query` returned with a handle.

    Call this only with a `handle` from a previous answer, never to re-run a query: the rows come
    from the cached result, so a repeated call is cheap.
    Example: get_more({"handle": "h_8f21", "offset": 100, "limit": 100})
    """
    return _not_implemented(
        "get_more",
        f"no result handles exist yet, so {handle!r} cannot be paged",
        "re-run `query` once the executor lands; nothing is cached in this build",
    )


def history(
    limit: Annotated[int, Field(ge=1, le=50, description="How many recent queries to list.")] = 5,
) -> NotImplementedResult:
    """List your own recent queries with their ids, so you can ask `feedback` about one.

    Call this when the user refers to something asked before, or when you need a query id.
    Example: history({"limit": 5})
    """
    return _not_implemented(
        "history",
        f"the query history store is not wired yet, so the last {limit} queries cannot be listed",
        "nothing to do; ask the user to restate the question until history lands",
    )


def feedback(
    query_id: Annotated[
        str, Field(min_length=1, description="A `query_id` from a previous answer.")
    ],
    rating: Annotated[
        int, Field(ge=1, le=5, description="1 = useless, 5 = exactly what was needed.")
    ],
    comment: Annotated[str | None, Field(description="Optional one-line reason.")] = None,
) -> NotImplementedResult:
    """Rate a previous answer so the semantic layer learns which tables and measures work.

    Call this once per answered question when the user says the answer was right or wrong.
    Example: feedback({"query_id": "q_8f21", "rating": 5, "comment": "the 5000 rounding was right"})
    """
    return _not_implemented(
        "feedback",
        f"feedback for {query_id!r} cannot be stored yet (rating {rating})",
        "nothing to do; the rating is dropped in this build",
    )


#: The tool functions in :data:`TOOL_NAMES` order. Names and docstrings come from here.
TOOL_FUNCTIONS: tuple[Callable[..., NotImplementedResult], ...] = (
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
