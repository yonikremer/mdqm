"""Tests for the linter (task C11): the registry's own cases, the invariants, and the pipeline.

Four kinds of evidence:

* **every rule replays its declared cases** -- each rule ships a firing case and a silent case (the
  registry refuses it otherwise), so the table below is the whole rule set, both directions;
* **a fix that would change the meaning is never applied**: an ambiguous value and an unapproved
  join both refuse, and the caller gets no partially fixed document;
* **idempotence**: linting a fixed spec or a fixed statement changes nothing, so a retry loop
  converges and ``applied_fixes`` cannot grow on a re-run;
* **the fixed document still validates**: a repaired spec round-trips through the spec models, so a
  rule can never emit a spec the compiler would reject.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mdqm.errors import ErrorCode
from mdqm.linter import (
    REGISTRY,
    LintContext,
    LintState,
    Registry,
    Rule,
    RuleCase,
    RuleOutcome,
    Severity,
    Target,
    lint_spec,
    lint_sql,
    spec_dump,
)
from mdqm.linter.registry import _iter_cases
from mdqm.linter.rules import duration_text
from mdqm.semantic.loader import load
from mdqm.semantic.models import SemanticLayer
from mdqm.spec.errors import SpecError
from mdqm.spec.validate import parse_spec

DATA = Path(__file__).resolve().parents[1] / "data" / "semantic" / "compiler_semantic.yaml"

#: One execution timestamp for every lint in this module (C02: one value per execution).
EXEC_TS = datetime(2026, 11, 25, 15, 0, tzinfo=UTC)

#: The catalog as the adapters report it: the only spellings a rule may accept.
COLUMNS: Mapping[str, tuple[str, ...]] = {
    "crm.customers": ("cust_id", "status", "credit_limit", "region"),
    "billing.accounts": ("client_no", "state", "credit_limit", "region", "opened_at"),
    "billing.usage_events": ("event_id", "client_no", "units", "start_time", "insertion_time"),
    "erp.customers": ("cust_no", "status_code", "limit_k", "name"),
}

#: A layer whose value map spells a source code exactly like a *different* canonical value, so a
#: label really is ambiguous. Nothing may pick one: that would answer a different question.
AMBIGUOUS_LAYER: Mapping[str, Any] = {
    "version": 1,
    "value_maps": {
        "customer_status": {
            "canonical": ["active", "churned"],
            "sources": {"crm.customers.status": {"A": "active", "active": "churned"}},
        }
    },
    "tables": {"crm.customers": {"key": ["cust_id"]}},
}


@pytest.fixture(scope="module")
def layer() -> SemanticLayer:
    """The shared test semantic layer, loaded through the real loader (so its rules run too)."""
    return load(DATA)


@pytest.fixture(scope="module")
def context_for(layer: SemanticLayer) -> Any:
    """Build the lint context a case asks for: the shared layer, or its own inline one."""

    def build(case: RuleCase) -> LintContext:
        if case.layer is None:
            return LintContext(layer=layer, columns=COLUMNS, exec_ts=EXEC_TS)
        return LintContext(
            layer=SemanticLayer.model_validate(case.layer),
            columns=case.columns or {},
            exec_ts=EXEC_TS,
        )

    return build


CASES: Sequence[tuple[Rule, RuleCase]] = tuple(_iter_cases())


# --- every declared case ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule", "case"), CASES, ids=[f"{rule.id}:{case.name}" for rule, case in CASES]
)
def test_declared_case(rule: Rule, case: RuleCase, context_for: Any, layer: SemanticLayer) -> None:
    """Replay one declared before/after example of one rule."""
    ctx = context_for(case)
    state = (
        LintState(spec=parse_spec(case.spec)) if case.spec is not None else LintState(sql=case.sql)
    )
    outcome = rule.run(state, ctx)
    if not case.fires:
        assert outcome is None, f"{rule.id} fired on a case that must stay quiet"
        return
    assert outcome is not None, f"{rule.id} did not fire"
    if case.expect_code is not None:
        assert outcome.error is not None
        assert outcome.error.code == case.expect_code
        assert outcome.spec is None and outcome.sql is None
        assert case.expect_fix is not None and case.expect_fix in outcome.error.fix
        suggestions = set(outcome.error.did_you_mean or ())
        assert set(case.expect_suggestions) <= suggestions
        return
    assert outcome.error is None, f"{rule.id} refused where a fix was expected"
    if case.expect_fix is not None:
        assert any(case.expect_fix in fix for fix in outcome.fixes)
    if case.expect_spec is not None:
        assert outcome.spec is not None
        assert spec_dump(outcome.spec) == case.expect_spec
    if case.expect_sql:
        assert outcome.sql is not None
        for fragment in case.expect_sql:
            assert fragment in outcome.sql
    for fragment in case.expect_absent:
        assert outcome.sql is not None
        assert fragment not in outcome.sql


def test_every_rule_is_covered_by_at_least_two_cases() -> None:
    """The rule set is the size the task asks for, and every rule proves both directions."""
    assert len(REGISTRY) >= 12
    for rule in REGISTRY:
        assert len(rule.cases) >= 2, rule.id
        assert any(case.fires for case in rule.cases), rule.id
        assert any(not case.fires for case in rule.cases), rule.id


# --- the pipeline -----------------------------------------------------------------------


def spec_map(**overrides: Any) -> dict[str, Any]:
    """A minimal valid spec, with the given fields overridden."""
    base: dict[str, Any] = {"from": "crm.customers", "select": ["cust_id"], "limit": 5}
    base.update(overrides)
    return base


def context(layer: SemanticLayer, **overrides: Any) -> LintContext:
    return LintContext(layer=layer, columns=COLUMNS, exec_ts=EXEC_TS, **overrides)


def test_pipeline_reports_every_fix(layer: SemanticLayer) -> None:
    """A spec that needs two repairs gets two sentences, in rule order, plus the fixed spec."""
    result = lint_spec(
        {
            "from": "billing.accounts",
            "select": ["client_no"],
            "filters": [{"column": "state", "op": "eq", "value": "פעיל"}],
        },
        context(layer),
    )
    assert result.ok, result.error
    assert result.rules == ("spec.value", "spec.limit")
    assert len(result.applied_fixes) == 2
    assert result.spec is not None
    assert result.spec.limit == 1000
    assert result.spec.filters[0].value == "active"


def test_hebrew_label_is_rewritten_and_reported(layer: SemanticLayer) -> None:
    """Acceptance: a Hebrew label for 'active' becomes the canonical value, and is listed."""
    result = lint_spec(
        spec_map(
            **{
                "from": "billing.accounts",
                "select": ["client_no"],
                "filters": [{"column": "state", "op": "eq", "value": "פעיל"}],
            }
        ),
        context(layer),
    )
    assert result.spec is not None
    assert result.spec.filters[0].value == "active"
    assert any("פעיל" in fix and "active" in fix for fix in result.applied_fixes)


def test_ambiguous_value_is_never_fixed() -> None:
    """A label matching two canonical values is refused, with the candidates, and no document."""
    trap = SemanticLayer.model_validate(AMBIGUOUS_LAYER)
    result = lint_spec(
        spec_map(filters=[{"column": "status", "op": "eq", "value": "active"}]),
        LintContext(layer=trap, columns=COLUMNS, exec_ts=EXEC_TS),
    )
    assert result.error is not None
    assert result.error.code == ErrorCode.AMBIGUOUS_VALUE.value
    assert result.error.valid_values == ("active", "churned")
    assert result.spec is None, "a refused spec must never come back half fixed"
    assert result.applied_fixes == ()


def test_unapproved_join_is_never_fixed(layer: SemanticLayer) -> None:
    """A join nobody approved is refused with the approved ON, and nothing is applied."""
    result = lint_spec(spec_map(joins=["customers_to_usage"], select=["units"]), context(layer))
    assert result.error is not None
    assert result.error.code == ErrorCode.JOIN_NOT_APPROVED.value
    assert result.spec is None
    assert result.applied_fixes == ()

    reachable = lint_spec(spec_map(select=["units"]), context(layer))
    assert reachable.error is not None
    assert reachable.error.code == ErrorCode.JOIN_NOT_APPROVED.value
    assert "LPAD" in reachable.error.fix, "the approved ON clause must be in the fix"


def test_invalid_spec_comes_back_as_an_error_response(layer: SemanticLayer) -> None:
    """A spec that does not even validate is an ``INVALID_SPEC`` answer, not an exception."""
    result = lint_spec({"from": "crm.customers", "select": []}, context(layer))
    assert result.error is not None
    assert result.error.code == ErrorCode.INVALID_SPEC.value
    assert result.spec is None


def test_sql_pipeline_applies_the_guard_first(layer: SemanticLayer) -> None:
    """A write is refused before any repair rule looks at the statement."""
    result = lint_sql("DELETE FROM customers", context(layer))
    assert result.error is not None
    assert result.error.code == ErrorCode.WRITE_BLOCKED.value
    assert result.rules == (), "no repair rule may run on a statement that must not run"


def test_sql_pipeline_repairs_and_reports(layer: SemanticLayer) -> None:
    """Free SQL gets the same treatment: the rewrite is applied, reported and deterministic."""
    result = lint_sql(
        "SELECT client_no FROM accounts WHERE state = 'פעיל' AND opened_at > now()",
        context(layer),
    )
    assert result.ok, result.error
    assert result.sql is not None
    assert "2026-11-25" in result.sql
    assert "Active" in result.sql
    assert "LIMIT 1000" in result.sql
    assert result.rules == ("sql.now", "sql.limit", "sql.value")


# --- invariants -------------------------------------------------------------------------


LINTED_SPECS: Sequence[Mapping[str, Any]] = (
    spec_map(),
    {"from": "crm.customers", "select": ["cust_id"]},
    {
        "from": "billing.accounts",
        "select": ["client_no"],
        "filters": [{"column": "state", "op": "in", "value": ["פעיל", "Hold"]}],
    },
    {
        "from": "erp.customers",
        "select": ["cust_no", {"measure": "credit_limit_r5000", "agg": "max"}],
        "group_by": ["cust_no"],
        "filters": [{"column": "status_code", "op": "eq", "value": 2}],
        "limit": 50,
    },
    {
        "from": "billing.usage_events",
        "select": ["event_id"],
        "filters": [{"column": "insertion_time", "op": "gte", "value": "2026-11-25T15:00:00"}],
    },
)

LINTED_SQL: Sequence[str] = (
    "SELECT client_no FROM accounts",
    "SELECT client_no FROM accounts WHERE state = 'פעיל'",
    "SELECT event_id FROM usage_events WHERE start_time > now()",
    "SELECT event_id FROM usage_events WHERE CAST(start_time AS DATE) = '2026-11-05'",
    "WITH recent AS (SELECT cust_id FROM customers) SELECT cust_id FROM recent",
)


@pytest.mark.parametrize("document", LINTED_SPECS, ids=range(len(LINTED_SPECS)))
def test_lint_is_idempotent(layer: SemanticLayer, document: Mapping[str, Any]) -> None:
    """lint(lint(x)) == lint(x): a second pass has nothing left to do."""
    once = lint_spec(document, context(layer))
    assert once.ok, once.error
    assert once.spec is not None
    twice = lint_spec(once.spec, context(layer))
    assert twice.ok, twice.error
    assert twice.applied_fixes == ()
    assert twice.spec == once.spec


@pytest.mark.parametrize("statement", LINTED_SQL, ids=range(len(LINTED_SQL)))
def test_sql_lint_is_idempotent(layer: SemanticLayer, statement: str) -> None:
    """The same invariant for the emergency path: the fixed statement is a fixed point."""
    once = lint_sql(statement, context(layer))
    assert once.ok, once.error
    assert once.sql is not None
    twice = lint_sql(once.sql, context(layer))
    assert twice.ok, twice.error
    assert twice.applied_fixes == ()
    assert twice.sql == once.sql


@pytest.mark.parametrize("document", LINTED_SPECS, ids=range(len(LINTED_SPECS)))
def test_fixed_spec_still_validates(layer: SemanticLayer, document: Mapping[str, Any]) -> None:
    """A repaired spec round-trips through the spec models: no rule can emit an invalid spec."""
    fixed = lint_spec(document, context(layer)).spec
    assert fixed is not None
    reparsed = parse_spec(json.dumps(spec_dump(fixed)))
    assert reparsed == fixed


@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=0x5FF), min_size=1))
@settings(max_examples=40, deadline=None)
def test_value_rule_never_invents_a_value(label: str) -> None:
    """Whatever label arrives, the fix is one of the map's canonical values -- or a refusal."""
    layer = _shared_layer()
    ctx = LintContext(layer=layer, columns=COLUMNS, exec_ts=EXEC_TS)
    try:
        spec = parse_spec(
            {
                "from": "billing.accounts",
                "select": ["client_no"],
                "filters": [{"column": "state", "op": "eq", "value": label}],
                "limit": 5,
            }
        )
    except SpecError:
        return  # a value the spec layer refuses is not the linter's business
    outcome = next(rule for rule in REGISTRY if rule.id == "spec.value").run(
        LintState(spec=spec), ctx
    )
    if outcome is None or outcome.error is not None:
        return
    canonical = {str(condition.value) for condition in outcome.spec.filters}  # type: ignore[union-attr]
    assert canonical <= {"active", "churned", "suspended"}


_LAYER_CACHE: list[SemanticLayer] = []


def _shared_layer() -> SemanticLayer:
    """The module's layer for the property test (hypothesis cannot take a pytest fixture)."""
    if not _LAYER_CACHE:
        _LAYER_CACHE.append(load(DATA))
    return _LAYER_CACHE[0]


# --- the real SQL corpus ------------------------------------------------------------------

#: The corpus dialect per file (tests/data/queries, task C01/C02): real statements, real dialects.
CORPUS: Mapping[str, tuple[str, frozenset[str]]] = {
    "postgres.sql": ("postgres", frozenset()),
    "oracle.sql": ("oracle", frozenset({"dual"})),
    "tsql.sql": ("tsql", frozenset()),
}

#: Every table the corpus reads, so a refusal answers about scans and not about a missing catalog.
CORPUS_COLUMNS: Mapping[str, tuple[str, ...]] = {
    **COLUMNS,
    "crm.customers": (*COLUMNS["crm.customers"], "name", "updated_at", "deleted_at"),
    "billing.accounts": (*COLUMNS["billing.accounts"], "status"),
    "billing.usage_events": (*COLUMNS["billing.usage_events"], "cust_id", "bytes"),
    "billing.usage_by_load": ("load_id", "client_no", "bytes", "insertion_time"),
    "billing.accounts_hist": ("client_no", "status", "updated_at", "credit_limit"),
}


def corpus_statements(path: Path) -> list[tuple[Mapping[str, Any], str]]:
    """``(expect block, statement)`` for every statement of a golden SQL corpus file."""
    blocks: list[list[str]] = [[]]
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "-- ---":
            blocks.append([])
        else:
            blocks[-1].append(line)
    found: list[tuple[Mapping[str, Any], str]] = []
    for lines in blocks:
        meta: Mapping[str, Any] | None = None
        body: list[str] = []
        for line in lines:
            if line.startswith("-- expect:"):
                meta = json.loads(line[len("-- expect:") :])
            elif line.strip().startswith("--") or not line.strip():
                continue
            else:
                body.append(line)
        if meta is not None and body:
            found.append((meta, "\n".join(body)))
    return found


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_golden_sql_corpus_is_a_fixed_point(layer: SemanticLayer, name: str) -> None:
    """Real statements in three dialects: nothing raises, and a repaired one stays repaired."""
    dialect, extra = CORPUS[name]
    ctx = LintContext(
        layer=layer,
        columns=CORPUS_COLUMNS,
        exec_ts=EXEC_TS,
        dialect=dialect,
        extra_relations=extra,
    )
    statements = corpus_statements(Path(__file__).resolve().parents[1] / "data" / "queries" / name)
    assert statements, f"{name} has no statements"
    rewritten = 0
    for _, sql in statements:
        result = lint_sql(sql, ctx)
        if not result.ok:
            assert result.error is not None and result.error.fix
            continue
        assert result.sql is not None
        again = lint_sql(result.sql, ctx)
        assert again.ok and again.applied_fixes == () and again.sql == result.sql
        rewritten += 1 if "sql.now" in result.rules else 0
    assert rewritten > 0, f"no statement in {name} needed the exec_ts rewrite"


# --- the registry refuses an untested rule ----------------------------------------------


class _FakeRule(Rule):
    """A rule with knobs, so each registry refusal can be provoked on purpose."""

    id = "spec.fake"
    severity = Severity.NOTICE
    autofix = False
    target = Target.SPEC
    title = "fake"
    cases: ClassVar[tuple[RuleCase, ...]] = ()

    def run(self, state: LintState, ctx: LintContext) -> RuleOutcome | None:
        return None


def fake(**overrides: Any) -> Rule:
    """A fresh fake rule instance with the given class attributes replaced."""
    return type("Fake", (_FakeRule,), overrides)()


GOOD_CASES: tuple[RuleCase, ...] = (
    RuleCase(
        "fires",
        fires=True,
        spec=spec_map(),
        expect_code=ErrorCode.INVALID_SPEC.value,
        expect_fix="valid value",
    ),
    RuleCase("quiet", spec=spec_map()),
)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    (
        ({"cases": ()}, "no case"),
        ({"cases": (GOOD_CASES[0],)}, "no silent case"),
        ({"cases": (GOOD_CASES[1],)}, "no firing case"),
        (
            {"cases": (GOOD_CASES[0], RuleCase("x", spec=spec_map(), expect_fix="y"))},
            "silent case cannot expect",
        ),
        (
            {"cases": (RuleCase("x", fires=True, spec=spec_map()), GOOD_CASES[1])},
            "must expect",
        ),
        ({"autofix": True, "cases": GOOD_CASES}, "auto-fix"),
        (
            {"severity": Severity.ERROR, "cases": (RuleCase("x", fires=True, spec=spec_map()),)},
            "no silent case",
        ),
        ({"id": "not-dotted"}, "rule id"),
        ({"id": "sql.fake"}, "named after its target"),
        (
            {
                "cases": (
                    RuleCase("x", fires=True, sql="SELECT 1", expect_code="Y"),
                    RuleCase("quiet", sql="SELECT 1"),
                )
            },
            "takes a spec",
        ),
    ),
)
def test_registry_refuses_an_untested_rule(overrides: Mapping[str, Any], reason: str) -> None:
    """Every way of adding a rule without a test is rejected at registration."""
    with pytest.raises(ValueError, match=reason):
        Registry((fake(**overrides),))


def test_registry_refuses_a_duplicate_id() -> None:
    with pytest.raises(ValueError, match="duplicate rule id"):
        Registry((fake(cases=GOOD_CASES), fake(cases=GOOD_CASES)))


def test_registry_reports_the_rules_for_each_target() -> None:
    """Both entry points see only their own rules, in one fixed order."""
    assert REGISTRY.for_target(Target.SPEC) == tuple(
        rule for rule in REGISTRY if rule.target is Target.SPEC
    )
    assert len(REGISTRY.for_target(Target.SQL)) >= 5


def test_duration_text_matches_the_spec_spelling() -> None:
    """The window in a fix is written the way a spec writes it."""
    assert duration_text(timedelta(days=7)) == "1w"
    assert duration_text(timedelta(hours=6)) == "6h"
    assert duration_text(timedelta(days=30)) == "30d"
