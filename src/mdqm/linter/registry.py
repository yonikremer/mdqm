"""The rule registry: one ordered set of rules, and no rule without a test.

The registry is built at import, so a rule that cannot prove itself breaks the package instead of
reaching a review. It refuses:

* a duplicate or malformed id;
* a rule with no case, only firing cases, or only silent cases (both directions are the point);
* an auto-fix rule with no case showing the repaired document;
* an ``error``-severity rule with no case asserting its refusal code;
* a case that asks for a document of the wrong target, or expects nothing at all.

``tests/unit/test_linter.py`` replays every case, so the declarations here are the tests.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Final

from mdqm.linter.rules import Rule, RuleCase, Severity, Target
from mdqm.linter.spec_rules import SPEC_RULES
from mdqm.linter.sql_rules import SQL_RULES

__all__ = ["REGISTRY", "Registry"]

_ID: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


class Registry:
    """An ordered, validated set of rules. Read-only: rules are registered at import only."""

    def __init__(self, rules: Sequence[Rule]) -> None:
        for rule in rules:
            _validate(rule)
        ids = [rule.id for rule in rules]
        duplicates = sorted({rule_id for rule_id in ids if ids.count(rule_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate rule id(s): {duplicates}")
        self._rules: tuple[Rule, ...] = tuple(rules)

    def __iter__(self) -> Iterator[Rule]:
        return iter(self._rules)

    def __len__(self) -> int:
        return len(self._rules)

    def for_target(self, target: Target) -> tuple[Rule, ...]:
        """The rules that read one kind of input, in registration order."""
        return tuple(rule for rule in self._rules if rule.target is target)

    def ids(self) -> tuple[str, ...]:
        """Every rule id, in the order the rules run."""
        return tuple(rule.id for rule in self._rules)


def _validate(rule: Rule) -> None:
    """Refuse a rule that cannot be trusted, or that nobody would notice breaking."""
    if _ID.match(rule.id) is None:
        raise ValueError(f"rule id {rule.id!r} is not '<target>.<name>' in lower case")
    if not rule.id.startswith(f"{rule.target.value}."):
        raise ValueError(f"rule {rule.id!r} must be named after its target ({rule.target.value}.*)")
    if not rule.title.strip():
        raise ValueError(f"rule {rule.id!r} needs a one-line title")
    cases = rule.cases
    if not cases:
        raise ValueError(f"rule {rule.id!r} declares no case: a rule without a test is untested")
    if not any(case.fires for case in cases):
        raise ValueError(f"rule {rule.id!r} has no firing case: nothing proves it ever applies")
    if not any(not case.fires for case in cases):
        raise ValueError(f"rule {rule.id!r} has no silent case: nothing proves it can stay quiet")
    for case in cases:
        _validate_case(rule, case)
    if rule.autofix and not any(case.expect_spec is not None or case.expect_sql for case in cases):
        raise ValueError(f"rule {rule.id!r} claims an auto-fix and no case shows its output")
    if rule.severity is Severity.ERROR and not any(case.expect_code for case in cases):
        raise ValueError(f"rule {rule.id!r} can refuse and no case asserts the refusal code")


def _validate_case(rule: Rule, case: RuleCase) -> None:
    """A case must be one input and one claim, and the input must match the rule's target."""
    if not case.name.strip():
        raise ValueError(f"rule {rule.id!r} has a case without a name")
    if (case.spec is None) == (case.sql is None):
        raise ValueError(f"rule {rule.id!r}/{case.name}: send exactly one of spec and sql")
    if rule.target is Target.SPEC and case.spec is None:
        raise ValueError(f"rule {rule.id!r}/{case.name}: a spec rule takes a spec")
    if rule.target is Target.SQL and case.sql is None:
        raise ValueError(f"rule {rule.id!r}/{case.name}: a statement rule takes sql")
    claims = (
        case.expect_spec is not None,
        bool(case.expect_sql),
        case.expect_code is not None,
        bool(case.expect_fix),
        bool(case.expect_suggestions),
    )
    if not case.fires:
        if any(claims):
            raise ValueError(f"rule {rule.id!r}/{case.name}: a silent case cannot expect anything")
        return
    if not any(claims):
        raise ValueError(
            f"rule {rule.id!r}/{case.name}: a firing case must expect a fix, a document or a code"
        )
    if not case.expect_fix:
        raise ValueError(
            f"rule {rule.id!r}/{case.name}: assert the message with expect_fix: a rule whose "
            "wording nobody checks drifts away from what the agent can act on"
        )


#: The one registry the linter runs. ``spec`` rules first: the spec is the primary path.
REGISTRY: Final[Registry] = Registry((*SPEC_RULES, *SQL_RULES))


def _iter_cases() -> Iterator[tuple[Rule, RuleCase]]:
    """Every declared case, for the test suite that replays them."""
    for rule in REGISTRY:
        for case in rule.cases:
            yield rule, case
