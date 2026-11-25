"""Agent-facing spec errors: exact ``path`` + ``problem`` + ``fix``, never a pydantic traceback.

``parse_spec`` is the only entry point a caller needs: it validates the JSON structurally (pydantic)
and then the cross-field rules (:func:`check_spec`), and raises :class:`SpecError` with every issue
it found. :meth:`SpecError.to_error_response` gives the ``ErrorResponse`` shape from DESIGN
section 5, so the server can return it verbatim.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import Field, ValidationError

from mdqm.models.base import MDQMModel
from mdqm.models.errors import ErrorResponse

#: At most this many issues are echoed back; a weak agent cannot act on fifty at once.
MAX_ISSUES = 5

#: Context keys worth quoting back to the agent (the bound or the allowed values).
_CONTEXT_KEYS = ("gt", "ge", "lt", "le", "expected")


class SpecIssue(MDQMModel):
    """One reason a spec was rejected: where it is, what is wrong, and what to write instead."""

    path: str = Field(min_length=1, description="Dotted location, e.g. 'filters.0.value'.")
    problem: str = Field(min_length=1)
    fix: str = Field(min_length=1)


class SpecError(ValueError):
    """A rejected spec. Holds every :class:`SpecIssue` found, in document order."""

    code = "INVALID_SPEC"

    def __init__(self, issues: tuple[SpecIssue, ...] | list[SpecIssue]) -> None:
        if not issues:
            raise ValueError("SpecError needs at least one issue")
        self.issues: tuple[SpecIssue, ...] = tuple(issues)
        super().__init__("; ".join(f"{i.path}: {i.problem}" for i in _capped(self.issues)))

    def to_error_response(self) -> ErrorResponse:
        """The DESIGN section 5 error shape: code, problem, fix (plus a count when truncated)."""
        shown = _capped(self.issues)
        problem = "; ".join(f"{issue.path}: {issue.problem}" for issue in shown)
        hidden = len(self.issues) - len(shown)
        if hidden > 0:
            problem = f"{problem}; and {hidden} more issue(s)"
        return ErrorResponse(
            code=self.code,
            problem=problem,
            fix="; ".join(issue.fix for issue in shown),
        )


def _capped(issues: tuple[SpecIssue, ...]) -> tuple[SpecIssue, ...]:
    return issues[:MAX_ISSUES]


#: Union branch labels pydantic puts in the location. They are not part of the agent's document:
#: a wrong value inside ``select`` belongs at ``select.0``, not at ``select.0.constrained-str``.
_PRIMITIVE_TAG = re.compile(r"^(constrained-)?(str|int|float|bool|bytes|datetime|date|null|none)$")
_COMPOSITE_TAG = re.compile(r"^(tuple|list|set|function-|is-instance|model)")
_MODEL_TAGS = frozenset({"MeasureSelect", "Duration"})


def path_of(location: tuple[int | str, ...]) -> str:
    """Render a pydantic error location as a dotted path: ``('filters', 0, 'value')``.

    Union branch labels are dropped, so ``select.0.constrained-str`` becomes ``select.0`` and
    ``select.0.MeasureSelect.agg`` becomes ``select.0.agg``: the agent must find the spot in *its*
    document.
    """
    kept = [part for part in location if not _is_branch_tag(part)]
    return ".".join(str(part) for part in kept) or "(root)"


def _is_branch_tag(part: int | str) -> bool:
    if isinstance(part, int):
        return False
    return (
        part in _MODEL_TAGS
        or _PRIMITIVE_TAG.match(part) is not None
        or _COMPOSITE_TAG.match(part) is not None
    )


#: Fixes for the structural failures a spec can hit, keyed by pydantic's error type.
_FIXES: dict[str, str] = {
    "missing": "add the required field {path} (spec_schema() lists every field and its values)",
    "extra_forbidden": (
        "remove '{path_last}': the spec has no such field "
        "(spec_schema() lists every accepted field at this level)"
    ),
    "string_type": "send {path} as a JSON string",
    "int_type": "send {path} as a whole number (no quotes, no decimals)",
    "int_parsing": "send {path} as a whole number (no quotes, no decimals)",
    "float_type": "send {path} as a number",
    "bool_type": "send {path} as true or false (no quotes)",
    "bool_parsing": "send {path} as true or false (no quotes)",
    "list_type": 'send {path} as a JSON list, e.g. ["value"]',
    "tuple_type": 'send {path} as a JSON list, e.g. ["value"]',
    "string_too_short": "send {path} as a non-empty string",
    "greater_than": "send {path} greater than {gt}",
    "greater_than_equal": "send {path} of at least {ge}",
    "less_than": "send {path} smaller than {lt}",
    "less_than_equal": "send {path} of at most {le}",
    "date_type": "send {path} as an ISO-8601 date, e.g. '2026-06-01'",
    "datetime_type": "send {path} as an ISO-8601 datetime, e.g. '2026-11-25T14:00:00'",
    "datetime_parsing": "send {path} as an ISO-8601 date or datetime, e.g. '2026-11-25T14:00:00'",
    "datetime_from_date_parsing": "send {path} as an ISO-8601 date or datetime",
    "json_invalid": "send one JSON object; spec_schema() describes its fields",
    "enum": "send {path} as one of {expected}",
    "literal_error": "send {path} as {expected}",
}

_DEFAULT_FIX = "send {path} in the shape spec_schema() describes"

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _fill(template: str, path: str, context: dict[str, Any]) -> str:
    """Format a fix template; fall back to the generic fix when a needed value is missing."""
    fill: dict[str, Any] = {"path": path, "path_last": path.rsplit(".", 1)[-1]}
    fill.update({key: context[key] for key in _CONTEXT_KEYS if key in context})
    if any(name not in fill for name in _PLACEHOLDER.findall(template)):
        template = _DEFAULT_FIX
    return template.format(**fill)


def issues_from_validation_error(exc: ValidationError) -> tuple[SpecIssue, ...]:
    """Translate pydantic failures into :class:`SpecIssue` values with an actionable fix."""
    issues: list[SpecIssue] = []
    for error in exc.errors():
        location = tuple(error["loc"])
        path = path_of(location)
        context: dict[str, Any] = dict(error.get("ctx") or {})
        message = str(error["msg"])
        if error["type"] == "value_error":
            # Raised by our own validators: the message already says what to send instead.
            problem = message.removeprefix("Value error, ")
            fix = problem
        else:
            problem = message
            fix = _fill(_FIXES.get(error["type"], _DEFAULT_FIX), path, context)
        issues.append(SpecIssue(path=path, problem=problem, fix=fix))
    if not issues:
        issues.append(
            SpecIssue(
                path="(root)",
                problem="the spec could not be validated",
                fix="send one JSON object shaped like spec_schema()",
            )
        )
    return _deepest(issues)


def _deepest(issues: list[SpecIssue]) -> tuple[SpecIssue, ...]:
    """Drop an issue whose path is a prefix of another's: one mistake, one message."""
    paths = [issue.path for issue in issues]
    kept = [
        issue
        for issue in issues
        if not any(other != issue.path and other.startswith(issue.path + ".") for other in paths)
    ]
    return tuple(kept) or (issues[0],)
