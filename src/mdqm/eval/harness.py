"""Eval harness: the spec path against hand-written SQL, per E01 question.

The product's claim is not "it returns something" -- it is "the spec compiles to SQL that answers
the question the way an analyst's own SQL would". So every question is answered twice:

* **spec path** -- the question's curated `spec`, through the compiler and the live source;
* **oracle path** -- the question's `oracle_sql`, written by hand in E01, executed directly.

Both are compared with the ground truth in `tests/eval/questions.yaml`. A disagreement between the
three is a finding to report, never to smooth over: if the spec path and the oracle disagree, one of
them is wrong and the report says which leg produced which number.

The model leg belongs to :mod:`scripts.bench_glm`; this module deliberately does not call a model,
so a harness run is deterministic and needs no endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mdqm.semantic.loader import load as load_layer
from mdqm.sql.execution import run_local

REPO_ROOT = Path(__file__).resolve().parents[3]  # src/mdqm/eval/harness.py
QUESTIONS = Path("tests/eval/questions.yaml")
LAYER = Path("examples/semantic_example.yaml")


@dataclass(frozen=True, slots=True)
class Question:
    """One E01 question: its spec, its hand-written SQL and the answer derived without this code."""

    id: str
    spec: Mapping[str, Any] | None
    oracle_sql: str | None
    answer: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Verdict:
    """What the two paths produced for one question, and whether each matched the ground truth."""

    id: str
    kind: str
    spec_result: str
    oracle_result: str
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.spec_result == "PASS" and self.oracle_result in {"PASS", "N/A"}


@dataclass
class Report:
    """The whole run: per-question verdicts plus the counts a nightly job would read."""

    verdicts: list[Verdict] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for verdict in self.verdicts if verdict.passed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": len(self.verdicts),
            "passed": self.passed,
            "worst_first": [
                {
                    "id": v.id,
                    "kind": v.kind,
                    "spec": v.spec_result,
                    "oracle": v.oracle_result,
                    "detail": v.detail,
                }
                for v in sorted(self.verdicts, key=lambda v: (v.passed, v.id))
            ],
        }


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    """Every question, with the spec and the oracle SQL E01 wrote for it."""
    from ruamel.yaml import YAML

    document = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    return [
        Question(
            id=str(entry["id"]),
            spec=entry.get("spec"),
            oracle_sql=entry.get("oracle_sql"),
            answer=dict(entry["answer"]),
        )
        for entry in document["questions"]
    ]


def _rows_from_batches(batches: Sequence[Any]) -> list[dict[str, Any]]:
    import pyarrow as pa

    if not batches:
        return []
    return list(pa.Table.from_batches(list(batches)).to_pylist())


def _json_or_fail(text: str, what: str) -> Mapping[str, Any]:
    """Parse ``text`` or raise a message naming what could not be parsed and its first bytes.

    A harness that dies with `JSONDecodeError` on a non-JSON line tells the operator nothing; the
    first bytes of what actually came back usually do.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{what} is not JSON ({exc.msg} at {exc.colno}): {text[:120]!r}"
        ) from None
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{what} is JSON but not an object: {type(parsed).__name__}")
    return parsed


def run_spec_path(question: Question) -> Mapping[str, Any]:
    """Answer the question through the product, exactly as an agent does: the MCP `query` tool.

    Compiling and executing here was wrong, and running it proved why: the tool applies the
    value-map grouping and normalisations on the way out, so a hand-rolled compile+execute leg
    reported FAIL on questions the product answers correctly. The spec path now goes through
    `scripts/mdqm_call.py` -- the same client, transport and envelope an agent uses -- so the
    harness cannot drift from what the product actually does.
    """
    import subprocess
    import sys

    if question.spec is None:
        return {"kind": "skipped", "problem": "no spec in the question set"}
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "mdqm_call.py"),
            "call",
            "query",
            json.dumps({"spec": dict(question.spec)}),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        check=False,
        cwd=str(REPO_ROOT),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"mdqm_call failed: {completed.stderr.strip()[:200]}")
    envelope = _json_or_fail(completed.stdout, "mdqm_call stdout")
    for item in envelope.get("content") or []:
        if isinstance(item, str):
            payload = _json_or_fail(item, "query tool payload")
            if payload.get("status") == "error":
                error = payload.get("error") or {}
                return {"kind": "error", "error_code": error.get("code")}
            return {"kind": "rows", "rows": list(payload.get("sample") or [])}
    raise RuntimeError(f"query returned no content: {envelope}")


def oracle_duckdb(world: Any) -> Any:
    """A DuckDB holding the world as E01's oracle SQL expects it: one schema per source.

    This is the layout the oracle was written for -- ``billing.usage_events``, ``crm.customers`` --
    and its dialect is DuckDB's (integer division with ``//``). Running it against PostgreSQL, where
    each source is a separate database and every table is in ``public``, was never going to work:
    the harness reported that as an `ERROR` leg until this existed.
    """
    from dataclasses import asdict

    import duckdb
    import pyarrow as pa

    con = duckdb.connect(":memory:")
    # The world's timestamps are tz-aware UTC; E01's oracle SQL compares them against a *naive*
    # TIMESTAMP. Without this, DuckDB compares in the session time zone, which showed up as 3952
    # extra units in a SUM (and one extra row in a COUNT) -- a fixture bug, not a product one.
    con.execute("SET TimeZone='UTC'")
    datasets: dict[str, dict[str, Any]] = {
        "crm": {"customers": world.crm.customers},
        "billing": {
            "accounts": world.billing.accounts,
            "usage_events": world.billing.usage_events,
            "accounts_hist": world.billing.accounts_hist,
        },
        "erp": {"customers": world.erp.customers},
    }
    for schema, tables in datasets.items():
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
        run_local(con, f"CREATE SCHEMA IF NOT EXISTS {schema}")
        for name, rows in tables.items():
            table = pa.Table.from_pylist([asdict(row) for row in rows])
            con.register("_load", table)
            # Schema and table names come from the fixed dict above, never from a caller.
            # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
            # Names come from the fixed dict above; S608 cannot see that.
            run_local(
                con,
                f'CREATE OR REPLACE TABLE {schema}."{name}" AS SELECT * FROM _load',  # noqa: S608
            )
            con.unregister("_load")
    return con


def run_oracle_path(question: Question, *, world: Any) -> Mapping[str, Any]:
    """Run E01's hand-written SQL in DuckDB, over the same world the database was loaded from."""
    if not question.oracle_sql:
        return {"kind": "skipped", "problem": "no oracle SQL in the question set"}
    con = oracle_duckdb(world)
    try:
        cursor = con.execute(question.oracle_sql)
        names = [description[0] for description in cursor.description or ()]
        rows = [dict(zip(names, row, strict=False)) for row in cursor.fetchall()]
        return {"kind": "rows", "rows": rows}
    finally:
        con.close()


def compare(question: Question, actual: Mapping[str, Any], *, tolerance: float = 1e-9) -> str:
    """``PASS`` / ``FAIL`` / ``N/A`` for one leg against the ground truth."""
    from scripts.bench_glm import score_rows

    expected = question.answer
    if expected.get("kind") == "error":
        if actual.get("kind") == "error" and actual.get("error_code") == expected.get("code"):
            return "PASS"
        return "FAIL"
    if actual.get("kind") != "rows":
        return "FAIL"
    problem = score_rows({"rows": list(actual.get("rows") or [])}, dict(expected), tolerance)
    return "PASS" if problem is None else "FAIL"


def evaluate(*, target: Any, adapters: Mapping[str, Any], layer: Any, world: Any) -> Report:
    """Answer every question twice and report both legs against the ground truth."""
    report = Report()
    for question in load_questions():
        source = str(question.spec["from"]).split(".", 1)[0] if question.spec else ""
        adapter = adapters.get(source)
        spec_result, oracle_result, detail = "SKIP", "SKIP", ""
        if adapter is not None:
            try:
                spec_actual = run_spec_path(question)
                spec_result = compare(question, spec_actual)
            except Exception as exc:
                spec_result, detail = "ERROR", f"{type(exc).__name__}: {exc}"
        if question.answer.get("kind") == "error":
            # The oracle answers "what plain SQL would do"; for these questions the product's job is
            # to *refuse* that SQL, so an oracle run is not a comparison -- running it would report
            # FAIL for the one thing the product is supposed to do.
            oracle_result = "N/A"
        else:
            try:
                oracle_actual = run_oracle_path(question, world=world)
                oracle_result = compare(question, oracle_actual)
            except Exception as exc:
                oracle_result, detail = "ERROR", f"{type(exc).__name__}: {exc}"
        report.verdicts.append(
            Verdict(
                id=question.id,
                kind=str(question.answer.get("kind")),
                spec_result=spec_result,
                oracle_result=oracle_result,
                detail=detail,
            )
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m mdqm.eval.harness [--json]``: run the harness against the configured sources."""
    from mdqm.config.settings import Settings
    from mdqm.server.wiring import build_adapters
    from mdqm.testing.loaders.postgres import target_from_environ

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    args = parser.parse_args(argv)

    settings = Settings.load()
    from mdqm.testing.world import WorldConfig, build_world

    report = evaluate(
        target=target_from_environ(),
        adapters=build_adapters(settings),
        layer=load_layer(LAYER),
        world=build_world(WorldConfig()),
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, default=str))
    else:
        width = max(len(v.id) for v in report.verdicts) + 2
        print(f"{'question'.ljust(width)}{'kind':8}{'spec':7}oracle")
        for verdict in report.verdicts:
            detail = f"  {verdict.detail[:70]}" if verdict.detail else ""
            print(
                f"{verdict.id.ljust(width)}{verdict.kind:8}"
                f"{verdict.spec_result:7}{verdict.oracle_result}{detail}"
            )
        print(f"\npassed {report.passed}/{len(report.verdicts)}")
    return 0 if report.passed == len(report.verdicts) else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
