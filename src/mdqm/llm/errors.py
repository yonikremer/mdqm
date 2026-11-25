"""Model-endpoint failures in the agent's error shape: ``code`` + ``problem`` + exact ``fix``.

The message and the :class:`~mdqm.models.errors.ErrorResponse` carry the endpoint, the model and
the status code. They never carry the API key, the prompt or the completion: the caller may put
user data in a prompt, and a header value is a credential (principle 11).

Codes are local to this lane; when the shared error-code registry lands they are the names to
register (``MODEL_NOT_CONFIGURED``, ``MODEL_TIMEOUT``, ``MODEL_UNAVAILABLE``,
``MODEL_HTTP_ERROR``, ``MODEL_OUTPUT_INVALID``).
"""

from __future__ import annotations

from mdqm.llm.result import Attempt, AttemptOutcome
from mdqm.models.errors import ErrorResponse

MODEL_NOT_CONFIGURED = "MODEL_NOT_CONFIGURED"
MODEL_TIMEOUT = "MODEL_TIMEOUT"
MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
MODEL_HTTP_ERROR = "MODEL_HTTP_ERROR"
MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"


class ModelError(RuntimeError):
    """A model call that could not return a validated object.

    ``retryable`` says whether another call may succeed (a timeout, a 503, a rejected
    ``response_format``); ``status_code`` lets the probe tell "the server refused this feature"
    apart from "the server is unreachable"; ``attempts`` is the audit trail of the failed call.
    """

    def __init__(
        self,
        problem: str,
        fix: str,
        *,
        code: str,
        retryable: bool = False,
        status_code: int | None = None,
        outcome: AttemptOutcome = "http_error",
        attempts: tuple[Attempt, ...] = (),
    ) -> None:
        self.problem = problem
        self.fix = fix
        self.code = code
        self.retryable = retryable
        self.status_code = status_code
        self.outcome = outcome
        self.attempts = attempts
        super().__init__(f"{code}: {problem}")

    def to_error_response(self) -> ErrorResponse:
        """The DESIGN section 5 error shape, safe to return to an agent verbatim."""
        return ErrorResponse(code=self.code, problem=self.problem, fix=self.fix)

    def __repr__(self) -> str:
        return (
            f"ModelError(code={self.code!r}, problem={self.problem!r}, "
            f"status_code={self.status_code!r}, attempts={len(self.attempts)})"
        )
