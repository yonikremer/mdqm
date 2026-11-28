"""The one exception an agent can see from the governor.

Every refusal here -- a busy source, an exhausted quota, a blown timeout, an open circuit -- is an
:class:`~mdqm.models.ErrorResponse` the agent can act on (``docs/DESIGN.md`` "Error shape"), so the
exception carries the response and nothing else. The server layer turns it straight into the tool
result; no traceback, no free text, and the ``fix`` field already says what to change.

Internal failures do not travel this way: those are mapped by
:func:`mdqm.errors.from_exception` so they get a correlation id and stay out of the agent's view.
"""

from __future__ import annotations

from mdqm.models import ErrorResponse

__all__ = ["GovernorCancelled", "GovernorRefusal"]


class GovernorRefusal(Exception):
    """A refusal to run, carrying the agent-facing error.

    Args:
        response: the ``ErrorResponse`` to return; its ``code`` is the stable token the agent
            branches on.
    """

    def __init__(self, response: ErrorResponse) -> None:
        self.response = response
        super().__init__(response.code)

    @property
    def code(self) -> str:
        """The error code, for a caller that branches without unpacking the response."""
        return self.response.code


class GovernorCancelled(GovernorRefusal):
    """The *caller* stopped the run (code ``CANCELLED``), so it is not the source's failure.

    Separate from its parent for one reason: a cancelled run must not count against the source's
    circuit breaker. A slow source opens the breaker; a caller changing their mind must not.
    """
