"""Cross-field rules and the one entry point: :func:`parse_spec`.

The pydantic models in :mod:`mdqm.spec.models` hold the structural rules (types, enums, ranges, what
a filter value may look like). Everything that needs two fields at once lives here, so each rule can
name the exact path it is about (``time.last``, ``group_by``, ``order_by.0.by``) and the fix in the
agent's own vocabulary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from mdqm.spec.errors import SpecError, SpecIssue, issues_from_validation_error, path_of
from mdqm.spec.models import QuerySpec, TimeMode


def parse_spec(data: str | bytes | Mapping[str, Any]) -> QuerySpec:
    """Validate an agent's spec (JSON text or a decoded mapping); raises :class:`SpecError`.

    Two stages, because they fail differently: the JSON shape first (pydantic), then the rules that
    span fields. Both report every problem they can see, so one round trip fixes the whole spec.
    """
    document = _decode(data)
    try:
        spec = QuerySpec.model_validate(document)
    except ValidationError as exc:
        raise SpecError(issues_from_validation_error(exc)) from None
    issues = check_spec(spec)
    if issues:
        raise SpecError(issues)
    return spec


def _decode(data: str | bytes | Mapping[str, Any]) -> Any:
    """Turn JSON text into a mapping, keeping decode failures inside the ``SpecError`` contract."""
    if isinstance(data, (str, bytes)):
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            raise SpecError(
                [
                    SpecIssue(
                        path="(root)",
                        problem=f"the spec is not valid JSON: {exc.msg} (line {exc.lineno})",
                        fix="send one JSON object; spec_schema() describes its fields",
                    )
                ]
            ) from None
    return data


def check_spec(spec: QuerySpec) -> tuple[SpecIssue, ...]:
    """Every cross-field rule, in document order. Pure: same spec in, same issues out."""
    issues: list[SpecIssue] = []
    issues.extend(_check_time(spec))
    issues.extend(_check_grouping(spec))
    issues.extend(_check_order_by(spec))
    issues.extend(_check_joins(spec))
    return tuple(issues)


def _check_time(spec: QuerySpec) -> list[SpecIssue]:
    time = spec.time
    if time is None:
        return []
    if time.mode is TimeMode.NEW_SINCE:
        issues = [
            SpecIssue(
                path=f"time.{name}",
                problem=(
                    f"mode 'new_since' is answered on the ingestion time and cannot also "
                    f"carry '{name}'"
                ),
                fix=f"remove time.{name}; 'new_since' needs only `since` (and `limit`)",
            )
            for name in ("last", "from", "to")
            if getattr(time, "from_time" if name == "from" else name) is not None
        ]
        if time.since is None:
            issues.append(
                SpecIssue(
                    path="time.since",
                    problem="mode 'new_since' needs `since`: the instant of your previous fetch",
                    fix='add time.since, e.g. "2026-11-25T14:00:00" (ISO-8601, UTC when no offset)',
                )
            )
        return issues

    issues = []
    has_window = time.last is not None
    has_absolute = time.from_time is not None or time.to is not None
    if time.since is not None:
        issues.append(
            SpecIssue(
                path="time.since",
                problem="mode 'events_in_range' takes `last` or `from`/`to`, not `since`",
                fix="use time.mode 'new_since' with time.since, or replace it with last/from/to",
            )
        )
    if has_window and has_absolute:
        issues.append(
            SpecIssue(
                path="time.last",
                problem="`last` and `from`/`to` both bound the window; one of them decides",
                fix="keep either time.last (e.g. '3mo') or time.from/time.to, not both",
            )
        )
    if not has_window and not has_absolute:
        issues.append(
            SpecIssue(
                path="time",
                problem="mode 'events_in_range' needs a window: `last`, or `from` and/or `to`",
                fix="add time.last (e.g. '3mo') or time.from and time.to (e.g. '2026-06-01')",
            )
        )
    if time.from_time is not None and time.to is not None and time.from_time >= time.to:
        issues.append(
            SpecIssue(
                path="time.to",
                problem=(
                    f"`to` ({time.to.isoformat()}) must be later than "
                    f"`from` ({time.from_time.isoformat()})"
                ),
                fix="send a later time.to, or drop it to get an open-ended window",
            )
        )
    return issues


def _check_grouping(spec: QuerySpec) -> list[SpecIssue]:
    issues: list[SpecIssue] = []
    seen: set[str] = set()
    for index, key in enumerate(spec.group_by):
        if not key.strip():
            issues.append(
                SpecIssue(
                    path=f"group_by.{index}",
                    problem="a group_by key cannot be blank",
                    fix="send a column name, or drop it from group_by",
                )
            )
        if key in seen:
            issues.append(
                SpecIssue(
                    path=f"group_by.{index}",
                    problem=f"'{key}' appears twice in group_by",
                    fix=f"remove the repeated '{key}'",
                )
            )
        seen.add(key)

    aggregated = tuple(i for i in spec.select if not isinstance(i, str))
    if not aggregated:
        return issues
    plain = tuple(i for i in spec.select if isinstance(i, str))
    for index, column in enumerate(plain):
        if column not in seen:
            issues.append(
                SpecIssue(
                    path=f"select.{index}",
                    problem=(
                        f"'{column}' is selected without an aggregate while other items are "
                        f"aggregated, and it is not in group_by"
                    ),
                    fix=(
                        f"add '{column}' to group_by, or select it as "
                        f"{{'measure': '{column}', 'agg': 'max'}}"
                    ),
                )
            )
    return issues


def _check_order_by(spec: QuerySpec) -> list[SpecIssue]:
    known = {*spec.selected_names(), *spec.group_by}
    issues: list[SpecIssue] = []
    for index, order in enumerate(spec.order_by):
        if order.by in known:
            continue
        issues.append(
            SpecIssue(
                path=f"order_by.{index}.by",
                problem=f"'{order.by}' is neither selected nor grouped by",
                fix=f"order by one of: {', '.join(sorted(known))}",
            )
        )
    return issues


def _check_joins(spec: QuerySpec) -> list[SpecIssue]:
    issues: list[SpecIssue] = []
    seen: set[str] = set()
    for index, name in enumerate(spec.joins):
        if not name.strip():
            issues.append(
                SpecIssue(
                    path=f"joins.{index}",
                    problem="a join name cannot be blank",
                    fix="send an approved join name, or drop it from joins",
                )
            )
        if name in seen:
            issues.append(
                SpecIssue(
                    path=f"joins.{index}",
                    problem=f"join '{name}' appears twice",
                    fix=f"remove the repeated '{name}'",
                )
            )
        seen.add(name)
    return issues


#: Re-exported so callers can format their own paths the same way.
__all__ = ["SpecError", "SpecIssue", "check_spec", "parse_spec", "path_of"]
