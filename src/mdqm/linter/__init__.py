"""The linter: repair a spec where the meaning is preserved, refuse it where it might not be.

An agent's spec (or, on the emergency exit, its SQL) is checked against the semantic layer and the
catalog *before* anything is compiled or sent. Every rule either

* **auto-fixes** the input and reports the fix in ``applied_fixes`` (a missing cap, a Hebrew label,
  an ingestion filter's derived bound, ``now()`` -> ``exec_ts``, a partition key hidden behind a
  function), or
* **refuses** with the ``ErrorResponse`` whose ``fix`` names the exact change (unknown table or
  column with "did you mean", a join that is not approved, an unknown or ambiguous value, a scan
  that would read every partition, a write).

The registry (:mod:`mdqm.linter.registry`) will not hold a rule without a firing case and a silent
case, so ``tests/unit/test_linter.py`` cannot fall behind the rules.

Pure: no I/O, no clock read (``exec_ts`` is injected), no source call.
"""

from __future__ import annotations

from mdqm.linter.core import LintResult, lint_spec, lint_sql
from mdqm.linter.registry import REGISTRY, Registry
from mdqm.linter.rules import (
    LintContext,
    LintState,
    Rule,
    RuleCase,
    RuleOutcome,
    Severity,
    Target,
    duration_text,
    spec_dump,
    spec_duration,
)

__all__ = [
    "REGISTRY",
    "LintContext",
    "LintResult",
    "LintState",
    "Registry",
    "Rule",
    "RuleCase",
    "RuleOutcome",
    "Severity",
    "Target",
    "duration_text",
    "lint_spec",
    "lint_sql",
    "spec_dump",
    "spec_duration",
]
