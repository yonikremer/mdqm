"""Linting: run the rules over one spec or one statement and report every change.

Two entry points, one contract. Each returns a :class:`LintResult`:

* ``applied_fixes`` lists one sentence per repair, in the order the rules applied it, so an answer
  can always show what was changed and why (principle 10: nothing silent);
* ``error`` is the first refusal, with the exact ``fix`` the agent must act on -- and then no
  repaired document is returned, because a half-fixed spec must never run;
* ``rules`` names the ids that fired, so the audit trail says which rule changed the request.

The rules themselves are pure and individually testable (``mdqm.linter.registry`` holds them). This
module only decides what runs, in what order, and what the caller sees.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from mdqm.linter.registry import REGISTRY, Registry
from mdqm.linter.rules import LintContext, LintState, Rule, Target
from mdqm.models import ErrorResponse
from mdqm.spec.errors import SpecError
from mdqm.spec.models import QuerySpec
from mdqm.spec.validate import parse_spec

__all__ = ["LintResult", "lint_spec", "lint_sql"]


@dataclass(frozen=True, slots=True)
class LintResult:
    """What the linter decided: the repaired document plus every reported change, or the refusal.

    ``request`` is the spec as it arrived, so a caller can compare "what the agent wrote" with "what
    runs". On a refusal ``spec`` and ``sql`` are ``None``: a half-fixed document must never run.
    """

    request: QuerySpec | None = None
    spec: QuerySpec | None = None
    sql: str | None = None
    applied_fixes: tuple[str, ...] = ()
    error: ErrorResponse | None = None
    rules: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when the input may run (nothing refused it)."""
        return self.error is None


def lint_spec(
    data: str | bytes | Mapping[str, Any] | QuerySpec,
    ctx: LintContext,
    *,
    registry: Registry = REGISTRY,
) -> LintResult:
    """Lint one spec (JSON text, a mapping or a parsed :class:`QuerySpec`).

    A spec that does not validate is not a lint problem: it comes back as the ``INVALID_SPEC``
    ``ErrorResponse`` the spec layer already builds, so the caller has one shape to report.
    """
    parsed = data if isinstance(data, QuerySpec) else _parse(data)
    if isinstance(parsed, ErrorResponse):
        return LintResult(error=parsed)
    return _run(
        LintState(spec=parsed),
        ctx,
        registry.for_target(Target.SPEC),
        request=parsed,
    )


def lint_sql(sql: str, ctx: LintContext, *, registry: Registry = REGISTRY) -> LintResult:
    """Lint one statement (the emergency exit): the guard first, then the repairs."""
    return _run(LintState(sql=sql), ctx, registry.for_target(Target.SQL))


def _parse(data: str | bytes | Mapping[str, Any]) -> QuerySpec | ErrorResponse:
    try:
        return parse_spec(data)
    except SpecError as error:
        return error.to_error_response()


def _run(
    state: LintState,
    ctx: LintContext,
    rules: tuple[Rule, ...],
    *,
    request: QuerySpec | None = None,
) -> LintResult:
    """Apply the rules in order, keeping every repair and stopping at the first refusal."""
    fixes: list[str] = []
    fired: list[str] = []
    for rule in rules:
        outcome = rule.run(state, ctx)
        if outcome is None:
            continue
        if outcome.error is not None:
            return LintResult(
                request=request,
                applied_fixes=tuple(fixes),
                error=outcome.error,
                rules=tuple(fired),
            )
        fired.append(rule.id)
        fixes.extend(outcome.fixes)
        state = replace(
            state,
            spec=outcome.spec if outcome.spec is not None else state.spec,
            sql=outcome.sql if outcome.sql is not None else state.sql,
        )
    return LintResult(
        request=request,
        spec=state.spec,
        sql=state.sql,
        applied_fixes=tuple(fixes),
        rules=tuple(fired),
    )
