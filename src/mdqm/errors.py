"""The agent-facing error taxonomy: one closed set of stable codes, one builder per code.

Every failure an agent can see is an :class:`~mdqm.models.ErrorResponse` (``docs/DESIGN.md`` "Error
shape") whose ``code`` comes from :class:`ErrorCode`. A code is not a label for a stack trace: it
names the one thing the agent must change, and the builder that produces it says exactly where, in
spec terms (``"from"``, ``"select"``, ``"time"``, ``"limit"``, ``"joins"``, ``"sql"``). The set is
closed and the values are stable, so an agent, a stored history row and an audit record can branch
on ``code`` without chasing synonyms. Adding a code is a deliberate act: extend the enum, add the
builder, add the case to ``tests/unit/test_errors.py``.

Who raises what:

| Code | Raised by | The ``fix`` tells the agent to change |
|---|---|---|
| ``PARSE_ERROR`` | guard (C01), compiler | the statement / ``"from"`` + ``"select"`` |
| ``WRITE_BLOCKED`` | guard (C01) | the write, DDL, lock or command |
| ``MULTI_STATEMENT`` | guard (C01) | everything after the first ``;`` |
| ``INVALID_SPEC`` | spec validation (C03) | the named spec field |
| ``UNKNOWN_TABLE`` | linter (C04) | ``"from"`` |
| ``UNKNOWN_COLUMN`` | linter (C04) | the ``"select"``/``"filter"`` entry |
| ``AMBIGUOUS_VALUE`` | value-map resolver (C05) | the filter value, with the canonical set |
| ``UNKNOWN_VALUE`` | value-map resolver (C05) | the filter value, with the valid set |
| ``JOIN_NOT_APPROVED`` | linter (C04) | ``"joins"``, with the approved ON clause |
| ``INSERTION_TIME_NO_BOUND`` | time planner (C07) | ``"time"`` |
| ``PARTITION_SCAN_ALL`` | partition planner (C08) | ``"time"`` / a filter on the key |
| ``COST_TOO_HIGH`` | explain gate (C08) | ``"time"``, ``"limit"``, ``"group_by"`` |
| ``QUOTA_EXCEEDED`` | governor (C15) | nothing: wait until the reset time |
| ``TIMEOUT`` | governor (C15) | ``"time"`` / ``"group_by"`` |
| ``RESULT_TOO_LARGE`` | result summariser (C16) | ``"limit"`` / ``"group_by"`` |
| ``NOT_FOUND`` | the lookup tools (M05) | the id: use one from a previous answer |
| ``SOURCE_UNAVAILABLE`` | adapters (F04, A02) | nothing: check the source and its secret |
| ``NOT_SUPPORTED`` | adapters (``NotSupported``) | nothing: use the declared fallback |
| ``INTERNAL_ERROR`` | any unexpected exception | nothing: retry, then report the id |

Two rules hold for every response this module builds, and ``tests/unit/test_errors.py`` proves both:

* **No free text from a failure.** A builder interpolates only identifiers (table, column, value,
  spec field) and enum tokens into fixed templates. The exception text, its ``__cause__`` and its
  ``__context__`` never reach the agent: :func:`from_exception` logs them with a correlation id and
  returns :func:`internal_error` (or a narrower code when the exception type says what happened).
* **Every caller-supplied string is scrubbed** by :func:`_safe`: one line, bounded, no control
  characters, no stack trace, no credential pattern. A hostile value the agent itself pasted cannot
  make an error message carry a secret or a second line.

The module has no I/O and no global mutable state; :data:`BUILDERS` is read-only.
"""

from __future__ import annotations

import difflib
import logging
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final, Protocol

from pydantic import ValidationError

from mdqm.models import ErrorResponse

__all__ = [
    "BUILDERS",
    "ErrorCode",
    "GuardOutcome",
    "ambiguous_value",
    "cost_too_high",
    "from_exception",
    "from_guard",
    "from_validation_error",
    "insertion_time_no_bound",
    "internal_error",
    "invalid_spec",
    "join_not_approved",
    "multi_statement",
    "not_found",
    "not_supported",
    "parse_error",
    "partition_scan_all",
    "quota_exceeded",
    "result_too_large",
    "source_unavailable",
    "timeout",
    "unknown_column",
    "unknown_table",
    "unknown_value",
    "write_blocked",
]

_LOGGER = logging.getLogger(__name__)

#: Longest text any field may carry. Error text is read in a chat window, not a log file.
_MAX_TEXT: Final[int] = 512
#: Longest identifier (table, column, value, spec field) interpolated into a message.
_MAX_IDENT: Final[int] = 96
_TRUNCATION: Final[str] = "..."
_TRACEBACK_REDACTION: Final[str] = "[redacted: the text looked like a stack trace]"
_EMPTY: Final[str] = "[empty]"
#: How many candidates a ``did_you_mean``/``valid_values`` list may carry.
_MAX_CANDIDATES: Final[int] = 8

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_WHITESPACE = re.compile(r"\s+")
_TRACEBACK = re.compile(
    r"traceback \(most recent call last\)"
    r'|File "[^"]{0,260}", line \d+'
    r"|\b(?:stack trace|call stack)\b"
    r"|^\s+at [\w.$<>]{1,80} \(",
    re.IGNORECASE,
)

#: ``(pattern, replacement)`` pairs applied in order. Deliberately narrow: they catch the credential
#: shapes a pasted value or a driver message carries, and nothing that a legitimate message says.
_SECRETS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    # A bearer/basic credential first: the key/value rule below would otherwise eat the scheme word
    # and leave the token itself in place.
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=\-]{8,}"), r"\1 [redacted]"),
    (
        re.compile(
            r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key"
            r"|private[_-]?key|authorization)\b(\s*[:=]\s*)(\S+)"
        ),
        r"\1\2[redacted]",
    ),
    # scheme://user:password@host
    (
        re.compile(r"(?i)\b([a-z][a-z0-9+.\-]{0,20}://)[^/\s:@]{1,80}:[^/\s@]{1,80}@"),
        r"\1[redacted]@",
    ),
    # A JSON Web Token, with or without the word Bearer in front of it.
    (
        re.compile(r"\beyJ[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,}(?:\.[A-Za-z0-9_\-]{4,})?"),
        "[redacted]",
    ),
    (
        re.compile(
            r"-----BEGIN [A-Z0-9 ]{0,40}PRIVATE KEY-----"
            r".*?-----END [A-Z0-9 ]{0,40}PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[redacted]",
    ),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[redacted]"),
)


class ErrorCode(StrEnum):
    """The closed set of codes an agent may branch on. Values are stable; never rename one."""

    # Guard / compiler (C01, C09). The guard owns the first three; reuse them, never rename them.
    PARSE_ERROR = "PARSE_ERROR"
    WRITE_BLOCKED = "WRITE_BLOCKED"
    MULTI_STATEMENT = "MULTI_STATEMENT"
    # Spec and semantic layer (C03, C04, C05).
    INVALID_SPEC = "INVALID_SPEC"
    UNKNOWN_TABLE = "UNKNOWN_TABLE"
    UNKNOWN_COLUMN = "UNKNOWN_COLUMN"
    AMBIGUOUS_VALUE = "AMBIGUOUS_VALUE"
    UNKNOWN_VALUE = "UNKNOWN_VALUE"
    JOIN_NOT_APPROVED = "JOIN_NOT_APPROVED"
    # Planning (C07, C08).
    INSERTION_TIME_NO_BOUND = "INSERTION_TIME_NO_BOUND"
    PARTITION_SCAN_ALL = "PARTITION_SCAN_ALL"
    COST_TOO_HIGH = "COST_TOO_HIGH"
    # Governor and results (C15, C16).
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    TIMEOUT = "TIMEOUT"
    RESULT_TOO_LARGE = "RESULT_TOO_LARGE"
    # Lookups by id: a handle, a run id, a query id (M05).
    NOT_FOUND = "NOT_FOUND"
    # Sources (F04, adapters).
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    # Anything that is a bug on our side.
    INTERNAL_ERROR = "INTERNAL_ERROR"


class GuardOutcome(Protocol):
    """What :func:`mdqm.sql.guard.guard` returns, structurally.

    Duck-typed on purpose: this module is imported by nearly every lane, so it must not pull
    sqlglot in through the guard.
    """

    @property
    def ok(self) -> bool:
        """True when the statement passed the guard."""
        ...

    @property
    def error(self) -> ErrorResponse | None:
        """The rejection for a blocked statement, ``None`` when ``ok``."""
        ...


def _safe(text: str, limit: int = _MAX_TEXT) -> str:
    """Make caller-supplied text safe to show: one line, bounded, no credential, no stack trace."""
    value = text if isinstance(text, str) else str(text)
    if _TRACEBACK.search(value):
        return _TRACEBACK_REDACTION
    for pattern, replacement in _SECRETS:
        value = pattern.sub(replacement, value)
    value = _CONTROL_CHARS.sub("", value)
    value = _WHITESPACE.sub(" ", value).strip()
    if len(value) > limit:
        value = value[: limit - len(_TRUNCATION)].rstrip() + _TRUNCATION
    return value or _EMPTY


def _ident(value: str) -> str:
    """A scrubbed identifier (table, column, value, spec field) for interpolation."""
    return _safe(value, _MAX_IDENT)


def _candidates(values: Sequence[str], limit: int = _MAX_CANDIDATES) -> tuple[str, ...]:
    """Scrubbed, de-duplicated, order-preserving, bounded candidate list."""
    seen: dict[str, None] = {}
    for value in values:
        cleaned = _ident(value)
        seen.setdefault(cleaned, None)
    return tuple(seen)[:limit]


def _names(values: Sequence[str]) -> str:
    """Render candidates for a message: ``"a", "b" (+3 more)``."""
    shown = _candidates(values)
    if not shown:
        return "the values in the semantic layer"
    text = ", ".join(f'"{name}"' for name in shown)
    hidden = len(_candidates(values, limit=10_000)) - len(shown)
    return f"{text} (+{hidden} more)" if hidden > 0 else text


def _close(name: str, known: Sequence[str], *, limit: int = 3) -> tuple[str, ...]:
    """The nearest known names to ``name``, for ``did_you_mean``.

    Containment counts as a suggestion (``status`` -> ``customer_status``): difflib alone scores it
    below its cutoff, and the substring case is exactly the one a weak model hits.
    """
    folded = {candidate.casefold(): candidate for candidate in known if isinstance(candidate, str)}
    query = name.casefold()
    contained = [
        candidate
        for key, candidate in folded.items()
        if key != query and (query in key or key in query)
    ]
    rest = difflib.get_close_matches(
        query, [key for key in folded if folded[key] not in contained], n=limit, cutoff=0.6
    )
    matches = [*contained, *(folded[key] for key in rest)]
    return tuple(matches[:limit])


def _window(last: str) -> str:
    """The spec fragment that bounds a scan, as the agent must write it."""
    return '{"mode": "events_in_range", "last": "' + _ident(last) + '"}'


def _new_correlation_id() -> str:
    """A fresh id for one internal failure. It says nothing about the cause by design."""
    return uuid.uuid4().hex[:16]


def _build(
    code: ErrorCode,
    problem: str,
    fix: str,
    *,
    partitions_if_fixed: str | None = None,
    valid_values: Sequence[str] = (),
    did_you_mean: Sequence[str] = (),
) -> ErrorResponse:
    """The only constructor: both texts are scrubbed here even if a builder forgot."""
    return ErrorResponse(
        code=code.value,
        problem=_safe(problem),
        fix=_safe(fix),
        partitions_if_fixed=_safe(partitions_if_fixed) if partitions_if_fixed else None,
        valid_values=_candidates(valid_values) or None,
        did_you_mean=_candidates(did_you_mean) or None,
    )


# --- guard and compiler -----------------------------------------------------------------


def parse_error(*, dialect: str) -> ErrorResponse:
    """The statement does not parse in ``dialect`` (code ``PARSE_ERROR``)."""
    return _build(
        ErrorCode.PARSE_ERROR,
        f"The {_ident(dialect)} statement does not parse.",
        f'Send one read-only SELECT in the "sql" field, or a spec with "from" and "select"; the '
        f"{_ident(dialect)} dialect rejects this syntax.",
    )


def write_blocked(*, operation: str) -> ErrorResponse:
    """A write, DDL, lock, command or denied function was refused (code ``WRITE_BLOCKED``)."""
    name = _ident(operation)
    return _build(
        ErrorCode.WRITE_BLOCKED,
        f'The "{name}" operation is refused: this server can only read.',
        f'Replace "{name}" with a SELECT in the "sql" field; a write, schema change, lock or '
        "server command has no equivalent here.",
    )


def multi_statement(*, count: int) -> ErrorResponse:
    """More than one statement arrived (code ``MULTI_STATEMENT``)."""
    return _build(
        ErrorCode.MULTI_STATEMENT,
        f"The input holds {count} statements and exactly one runs per call.",
        "Keep the first statement and send everything after the first ';' as a separate call.",
    )


# --- spec and semantic layer ------------------------------------------------------------


def invalid_spec(*, field: str, reason: str, expected: Sequence[str] = ()) -> ErrorResponse:
    """A spec field failed validation (code ``INVALID_SPEC``).

    ``reason`` is a short machine-written sentence such as ``"it is required"``; it is never the
    offending value and never SQL, so a secret the agent pasted cannot travel through it.
    """
    name = _ident(field)
    if expected:
        fix = f'Set the spec field "{name}" to one of: {_names(expected)}.'
    else:
        fix = (
            f'Give the spec field "{name}" a valid value; the spec shape is in '
            "examples/query_specs.json."
        )
    return _build(
        ErrorCode.INVALID_SPEC,
        f'The spec field "{name}" is invalid: {_ident(reason)}.',
        fix,
        valid_values=expected,
    )


def unknown_table(*, name: str, known: Sequence[str]) -> ErrorResponse:
    """The spec names a table the semantic layer does not have (code ``UNKNOWN_TABLE``)."""
    return _build(
        ErrorCode.UNKNOWN_TABLE,
        f'The spec names table "{_ident(name)}", which the semantic layer does not have.',
        f'Set "from" to one of the known tables: {_names(known)}.',
        did_you_mean=_close(name, known),
    )


def unknown_column(*, table: str, name: str, known: Sequence[str]) -> ErrorResponse:
    """The spec names a column its table does not have (code ``UNKNOWN_COLUMN``)."""
    relation = _ident(table)
    return _build(
        ErrorCode.UNKNOWN_COLUMN,
        f'Table "{relation}" has no column "{_ident(name)}".',
        f'Change the "{_ident(name)}" entry in "select" or in a "filters" item to one of '
        f"{relation}'s columns: {_names(known)}.",
        did_you_mean=_close(name, known),
    )


def ambiguous_value(*, field: str, value: str, canonical: Sequence[str]) -> ErrorResponse:
    """One label maps to several canonical values (code ``AMBIGUOUS_VALUE``)."""
    name = _ident(field)
    return _build(
        ErrorCode.AMBIGUOUS_VALUE,
        f'The value "{_ident(value)}" in "{name}" matches several canonical values.',
        f'Pick exactly one canonical value for "{name}": {_names(canonical)}.',
        valid_values=canonical,
    )


def unknown_value(*, field: str, value: str, valid: Sequence[str]) -> ErrorResponse:
    """A label matches no source code and no canonical value (code ``UNKNOWN_VALUE``)."""
    name = _ident(field)
    return _build(
        ErrorCode.UNKNOWN_VALUE,
        f'The value "{_ident(value)}" in "{name}" is not a known source code or canonical value.',
        f'Replace "{_ident(value)}" in "{name}" with one of: {_names(valid)} '
        "(Hebrew and English labels both work).",
        valid_values=valid,
    )


def join_not_approved(*, left: str, right: str, approved_on: str | None = None) -> ErrorResponse:
    """The spec joins two tables no approved join covers (code ``JOIN_NOT_APPROVED``)."""
    pair = f'"{_ident(left)}" and "{_ident(right)}"'
    if approved_on is not None:
        fix = (
            f"Replace the join with the approved one for {pair}; its full ON clause is: "
            f"{_ident(approved_on)}."
        )
    else:
        fix = (
            f'Remove the join of {pair} from "joins", or ask an admin to approve an ON clause '
            "for it in curated/joins.yaml."
        )
    return _build(
        ErrorCode.JOIN_NOT_APPROVED,
        f"The join of {pair} is not approved in the semantic layer.",
        fix,
    )


# --- planning ---------------------------------------------------------------------------


def insertion_time_no_bound(
    *, table: str, event_column: str, max_lag: str | None = None
) -> ErrorResponse:
    """A filter on the ingestion time cannot be bounded (code ``INSERTION_TIME_NO_BOUND``)."""
    relation = _ident(table)
    column = _ident(event_column)
    if max_lag is not None:
        fix = (
            f'Use "time": {{"mode": "new_since", "since": "<ISO-8601>"}}; the server then adds '
            f'a bound on "{column}" (>= since - {_ident(max_lag)}) itself from the approved lag.'
        )
    else:
        fix = (
            f'No approved lag exists for "{relation}", so no bound can be derived: use "time": '
            f'{{"mode": "events_in_range", "from": "<date>", "to": "<date>"}} with a narrow window '
            f'on the indexed column "{column}".'
        )
    return _build(
        ErrorCode.INSERTION_TIME_NO_BOUND,
        f'"{relation}" is filtered on the ingestion time only, and that column is not indexed, '
        "so the scan would read the whole table.",
        fix,
    )


def partition_scan_all(
    *,
    table: str,
    key: str,
    granularity: str,
    default_window: str | None = None,
    partitions_if_fixed: str | None = None,
) -> ErrorResponse:
    """Nothing filters the partition key, so all partitions would be read.

    Code ``PARTITION_SCAN_ALL``.
    """
    relation = _ident(table)
    column = _ident(key)
    window = default_window if default_window is not None else "30d"
    return _build(
        ErrorCode.PARTITION_SCAN_ALL,
        f'"{relation}" is partitioned {_ident(granularity)} on "{column}" and the spec has no '
        "filter on it, so every partition would be scanned.",
        f'Add "time": {_window(window)} (or any filter on "{column}"): that is what limits the '
        "scan to a few partitions.",
        partitions_if_fixed=partitions_if_fixed,
    )


def cost_too_high(*, table: str, estimated: str, limit: str) -> ErrorResponse:
    """The EXPLAIN gate refuses an expensive read (code ``COST_TOO_HIGH``)."""
    relation = _ident(table)
    return _build(
        ErrorCode.COST_TOO_HIGH,
        f'Reading "{relation}" would touch about {_ident(estimated)}, above the gate\'s '
        f"{_ident(limit)}.",
        f'Narrow the filter, e.g. "time": {_window("7d")}, or add a "limit" or a "group_by"; the '
        "estimate comes from table statistics, not from a run.",
    )


# --- governor and results ---------------------------------------------------------------


def quota_exceeded(*, quota: str, limit: str, resets_at: str) -> ErrorResponse:
    """A quota (queries, rows, time budget) is used up (code ``QUOTA_EXCEEDED``)."""
    return _build(
        ErrorCode.QUOTA_EXCEEDED,
        f"The {_ident(quota)} quota of {_ident(limit)} is used up.",
        f"Wait until {_ident(resets_at)} and retry; meanwhile an identical repeated query is "
        'served from the cache (check "fetched_at").',
    )


def timeout(*, seconds: float | None = None, target: str | None = None) -> ErrorResponse:
    """A query ran past its wall-clock limit and was cancelled (code ``TIMEOUT``)."""
    where = f' on "{_ident(target)}"' if target else ""
    limit = f" of {seconds:g}s" if seconds is not None else ""
    return _build(
        ErrorCode.TIMEOUT,
        f"The query{where} hit the time limit{limit} and was cancelled.",
        f'Narrow the window in "time" (e.g. {_window("1d")}) and/or give the "group_by" fewer '
        "columns; the timeout is a server limit, not a spec field.",
    )


def result_too_large(*, max_rows: int, rows: int | None = None) -> ErrorResponse:
    """The result exceeds the response cap (code ``RESULT_TOO_LARGE``)."""
    size = f"{rows} rows" if rows is not None else "larger than the cap"
    return _build(
        ErrorCode.RESULT_TOO_LARGE,
        f"The result has {size}; the response cap is {max_rows} rows.",
        f'Add "limit": {max_rows}, or put a "group_by" on the columns you need and select the '
        "aggregates instead of single rows.",
    )


# --- lookups by id ----------------------------------------------------------------------


def not_found(*, what: str, fix: str) -> ErrorResponse:
    """Nothing is readable under the id the caller sent (code ``NOT_FOUND``).

    The ``problem`` text is built from ``what`` alone -- never from the id -- so "never existed",
    "expired" and "belongs to someone else" produce byte-identical answers. That is the whole
    point of an opaque handle: the id must not become a capability probe for another user's data.
    """
    return _build(
        ErrorCode.NOT_FOUND,
        f"No {_ident(what)} is available for this user, so nothing was returned.",
        fix,
    )


# --- sources ----------------------------------------------------------------------------


def source_unavailable(*, source: str, secret_name: str | None = None) -> ErrorResponse:
    """A source did not answer, so nothing ran (code ``SOURCE_UNAVAILABLE``).

    Names the *secret reference* when there is one; a secret value must never reach this message,
    which is why no free text is accepted here.
    """
    name = _ident(source)
    if secret_name is not None:
        fix = (
            f'Nothing in the spec can fix this: check that source "{name}" is reachable and that '
            f'the secret "{_ident(secret_name)}" it uses resolves (the value is never logged).'
        )
    else:
        fix = (
            f'Nothing in the spec can fix this: check that source "{name}" is reachable, then '
            "retry."
        )
    return _build(
        ErrorCode.SOURCE_UNAVAILABLE,
        f'Source "{name}" is not answering, so nothing ran.',
        fix,
    )


def not_supported(*, source: str, capability: str, fallback: str | None = None) -> ErrorResponse:
    """A source cannot provide a capability the request needs (code ``NOT_SUPPORTED``)."""
    name = _ident(source)
    feature = _ident(capability)
    if fallback is not None:
        fix = f'Use the declared fallback for "{feature}": {_ident(fallback)}.'
    else:
        fix = (
            f'Ask for the answer without "{feature}": this source cannot provide it and no '
            "fallback is declared."
        )
    return _build(
        ErrorCode.NOT_SUPPORTED,
        f'Source "{name}" does not support "{feature}".',
        fix,
    )


# --- our own bugs -----------------------------------------------------------------------


def internal_error(*, correlation_id: str) -> ErrorResponse:
    """An unexpected failure (code ``INTERNAL_ERROR``).

    Carries the correlation id and nothing else: the exception and its chain are in the log, keyed
    by that id. This is the only code whose text says nothing about what happened.
    """
    return _build(
        ErrorCode.INTERNAL_ERROR,
        "The server failed internally, so the request did not run.",
        f"Retry once; if it fails again, report correlation id {_ident(correlation_id)} to the "
        "administrator.",
    )


#: Every code and the builder that produces it. Read-only, so nobody can quietly swap a builder
#: for one that skips the scrub in :func:`_build`.
BUILDERS: Mapping[ErrorCode, Callable[..., ErrorResponse]] = MappingProxyType(
    {
        ErrorCode.PARSE_ERROR: parse_error,
        ErrorCode.WRITE_BLOCKED: write_blocked,
        ErrorCode.MULTI_STATEMENT: multi_statement,
        ErrorCode.INVALID_SPEC: invalid_spec,
        ErrorCode.UNKNOWN_TABLE: unknown_table,
        ErrorCode.UNKNOWN_COLUMN: unknown_column,
        ErrorCode.AMBIGUOUS_VALUE: ambiguous_value,
        ErrorCode.UNKNOWN_VALUE: unknown_value,
        ErrorCode.JOIN_NOT_APPROVED: join_not_approved,
        ErrorCode.INSERTION_TIME_NO_BOUND: insertion_time_no_bound,
        ErrorCode.PARTITION_SCAN_ALL: partition_scan_all,
        ErrorCode.COST_TOO_HIGH: cost_too_high,
        ErrorCode.QUOTA_EXCEEDED: quota_exceeded,
        ErrorCode.TIMEOUT: timeout,
        ErrorCode.RESULT_TOO_LARGE: result_too_large,
        ErrorCode.NOT_FOUND: not_found,
        ErrorCode.SOURCE_UNAVAILABLE: source_unavailable,
        ErrorCode.NOT_SUPPORTED: not_supported,
        ErrorCode.INTERNAL_ERROR: internal_error,
    }
)


# --- mapping real failures onto the taxonomy --------------------------------------------

#: Pydantic error type -> the sentence an agent reads. Only the *type* is used, never the value.
_VALIDATION_REASONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "missing": "it is required",
        "literal_error": "it is not one of the allowed values",
        "enum": "it is not one of the allowed values",
        "extra_forbidden": "it is not a field of the spec",
        "string_type": "it must be a string",
        "int_type": "it must be a whole number",
        "int_parsing": "it must be a whole number",
        "float_type": "it must be a number",
        "float_parsing": "it must be a number",
        "bool_type": "it must be true or false",
        "bool_parsing": "it must be true or false",
        "list_type": "it must be a list",
        "dict_type": "it must be an object",
        "model_type": "it must be an object",
        "model_attributes_type": "it must be an object",
        "greater_than": "it is below the minimum",
        "greater_than_equal": "it is below the minimum",
        "less_than": "it is above the maximum",
        "less_than_equal": "it is above the maximum",
        "string_too_long": "it is too long",
        "value_error": "it is not a valid value",
        "datetime_parsing": "it must be an ISO-8601 date or timestamp",
        "date_from_datetime_inexact": "it must be a whole date",
    }
)
_DEFAULT_REASON: Final[str] = "it is not a valid value"


def _field_path(location: Sequence[Any], root: str) -> str:
    """``("filters", 0, "value")`` -> ``"filters.0.value"``, the shape query_specs.json uses."""
    parts = [str(part) for part in location]
    return ".".join([root, *parts]) if root else ".".join(parts) or "spec"


def _expected_from(context: Any) -> tuple[str, ...]:
    """The allowed values pydantic put in ``ctx["expected"]``: ``"'a' or 'b'"`` -> ``a, b``."""
    if not isinstance(context, Mapping):
        return ()
    raw = context.get("expected")
    if not isinstance(raw, str):
        return ()
    return tuple(re.findall(r"'([^']{1,64})'", raw))


def from_validation_error(exc: ValidationError, *, root: str = "") -> ErrorResponse:
    """Turn the first field error of a spec validation into ``INVALID_SPEC``.

    Reads only the error type, the field path and the allowed values; the offending input, which is
    what pydantic stores in ``ctx``/``input`` and what an agent could have filled with a token,
    never leaves this function. ``root`` prefixes the field path when the spec sits inside a
    larger payload; the default renders the paths ``examples/query_specs.json`` uses
    (``time.mode``, ``filters.0.value``).
    """
    problems = exc.errors()
    if not problems:
        return _build(
            ErrorCode.INVALID_SPEC,
            "The spec is invalid.",
            "Check the spec shape in examples/query_specs.json and send it again.",
        )
    first = problems[0]
    return invalid_spec(
        field=_field_path(first.get("loc") or (), root),
        reason=_VALIDATION_REASONS.get(str(first.get("type")), _DEFAULT_REASON),
        expected=_expected_from(first.get("ctx")),
    )


def from_guard(result: GuardOutcome) -> ErrorResponse | None:
    """The guard's rejection as an ``ErrorResponse``, or ``None`` when it allowed the statement.

    The guard owns its codes and texts (``PARSE_ERROR``, ``WRITE_BLOCKED``, ``MULTI_STATEMENT``);
    this only re-sanitises them, so every agent-facing error takes the same path. A code the guard
    invented outside the taxonomy is a bug and becomes ``INTERNAL_ERROR``.
    """
    if result.ok:
        return None
    raw = result.error
    if raw is None:
        _LOGGER.error("the guard rejected a statement without an error", extra={"error_code": ""})
        return internal_error(correlation_id=_new_correlation_id())
    try:
        code = ErrorCode(raw.code)
    except ValueError:
        _LOGGER.error("the guard used an error code outside the taxonomy: %s", _safe(raw.code))
        return internal_error(correlation_id=_new_correlation_id())
    return _build(
        code,
        raw.problem,
        raw.fix,
        partitions_if_fixed=raw.partitions_if_fixed,
        valid_values=raw.valid_values or (),
        did_you_mean=raw.did_you_mean or (),
    )


def from_exception(
    exc: BaseException,
    *,
    source: str | None = None,
    timeout_s: float | None = None,
    correlation_id: str | None = None,
) -> ErrorResponse:
    """Map a raised exception onto a code, logging what the agent must not see.

    The exception text, its ``__cause__`` and its ``__context__`` go to the log with a correlation
    id and are never interpolated into the response. Well-known failures get a narrow code:

    * an adapter's ``NotSupported`` (anything carrying a string ``capability``) ->
      ``NOT_SUPPORTED``
    * ``TimeoutError`` (also ``asyncio`` and ``socket`` timeouts) -> ``TIMEOUT``
    * ``OSError`` when ``source`` is given (a driver refused the connection) ->
      ``SOURCE_UNAVAILABLE``
    * everything else, including a configuration error -> ``INTERNAL_ERROR``

    Args:
        exc: the caught exception; only its type and (for ``NotSupported``) its ``capability``
            attribute are read.
        source: the source being talked to, when the caller knows it.
        timeout_s: the limit that was exceeded, when the caller knows it.
        correlation_id: injected in tests; generated otherwise.
    """
    capability = getattr(exc, "capability", None)
    if isinstance(capability, str) and capability:
        return not_supported(source=source or "the source", capability=capability)
    if isinstance(exc, TimeoutError):
        return timeout(seconds=timeout_s, target=source)
    if source is not None and isinstance(exc, OSError):
        _LOGGER.warning(
            "a source refused the request",
            extra={
                "correlation_id": correlation_id or "",
                "source": source,
                "error_code": ErrorCode.SOURCE_UNAVAILABLE.value,
                "exception_type": type(exc).__name__,
            },
            exc_info=exc,
        )
        return source_unavailable(source=source)
    identifier = correlation_id or _new_correlation_id()
    _LOGGER.error(
        "unhandled internal error",
        extra={
            "correlation_id": identifier,
            "error_code": ErrorCode.INTERNAL_ERROR.value,
            "exception_type": type(exc).__name__,
        },
        exc_info=exc,
    )
    return internal_error(correlation_id=identifier)
