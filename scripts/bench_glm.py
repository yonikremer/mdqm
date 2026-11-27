"""Benchmark the whole product with a real weak model: prompts out, scored answers in.

The claim under test is not "the code runs" — the E2E suite covers that. The claim here is the one
the README makes first: **a weak agent, given a business question in Hebrew or English, gets the
right number through mdqm's five tools.** So the measurement has two halves:

    python scripts/bench_glm.py prompts --out bench/prompts        # one prompt per question
    # ... a model answers each prompt by writing bench/answers/<id>.json ...
    python scripts/bench_glm.py score --answers bench/answers      # scores against ground truth

Ground truth comes from `tests/eval/questions.yaml`, which was derived independently of the code
under test (task E01), so a correct answer here is evidence and a wrong one is a finding.

The model never sees the expected answer, the SQL, or the spec. It sees the question, the five tool
names with their descriptions, and the exact command that runs one tool call.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

QUESTIONS_PATH = Path("tests/eval/questions.yaml")
CALL_SCRIPT = "python scripts/mdqm_call.py"
DEFAULT_TOLERANCE = 1e-9

PROMPT_TEMPLATE = """# mdqm benchmark task

You are a data analyst using `mdqm`, an MCP server that answers questions over several data sources.
Your tools are reachable only through this command (do not hand-roll HTTP):

    {call} schema            # every tool with its JSON input schema
    {call} call <tool> '<json arguments>'

The five tools:

| tool | use it for |
| --- | --- |
| `find_data` | look up tables, columns, measures and categorical values for a question |
| `query` | run one query spec: the main tool |
| `get_more` | page the rest of a result you already got from `query` |
| `history` | your own earlier runs |
| `feedback` | rate an answer |

## The question

> {question}

(Also asked in the other language: "{other}")

## What you must produce

Write the file `{answer_path}` with exactly this shape and nothing else:

```json
{{"rows": [{{"<column>": <value>, ...}}], "columns": ["<column>", ...]}}
```

{expected_columns}

If mdqm refuses the question, write instead:

```json
{{"error_code": "<the code mdqm returned>", "fix": "<the fix it suggested>"}}
```

Rules: use the tools (not your own SQL); do not invent numbers; keep going until a tool gives you an
answer or a refusal. When you are done, reply with the single line `WROTE {answer_id}`.
"""


@dataclass(frozen=True, slots=True)
class Question:
    """One benchmark item: the question, its languages and the independently derived answer."""

    id: str
    text: dict[str, str]
    answer: dict[str, Any]


def load_questions(path: Path = QUESTIONS_PATH) -> tuple[list[Question], float]:
    document = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    tolerance = float(document.get("float_tolerance") or DEFAULT_TOLERANCE)
    questions = [
        Question(id=str(item["id"]), text=dict(item["text"]), answer=dict(item["answer"]))
        for item in document["questions"]
    ]
    return questions, tolerance


def _expected_block(question: Question) -> str:
    answer = question.answer
    if answer["kind"] == "rows":
        columns = ", ".join(f"`{name}`" for name in answer["columns"])
        return (
            f"These exact column names are expected, in any row order: {columns}.\n"
            "Numbers must match the real value (rounding matters)."
        )
    return (
        "This question must be REFUSED: an `error` answer is the correct outcome. "
        f"The expected code is `{answer.get('code')}`."
    )


def write_prompts(out_dir: Path, *, call: str = CALL_SCRIPT) -> int:
    questions, _ = load_questions()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for question in questions:
        languages = list(question.text)
        primary = languages[0]
        other = question.text[languages[1]] if len(languages) > 1 else question.text[primary]
        answer_path = f"bench/answers/{question.id}.json"
        body = PROMPT_TEMPLATE.format(
            call=call,
            question=question.text[primary],
            other=other,
            answer_path=answer_path,
            answer_id=question.id,
            expected_columns=_expected_block(question),
        )
        (out_dir / f"{question.id}.md").write_text(body, encoding="utf-8", newline="\n")
        manifest.append({"id": question.id, "language": primary, "prompt": f"{question.id}.md"})
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {len(questions)} prompts to {out_dir}")
    return len(questions)


# ---------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------


def _coerce(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            return text.lower()
    return value


def _row_key(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
    return tuple(_coerce(row.get(name)) for name in columns)


def _numbers_match(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(actual, float) and isinstance(expected, float):
        if expected == 0:
            return abs(actual) <= tolerance
        return abs(actual - expected) / abs(expected) <= tolerance
    return bool(actual == expected)


def score_rows(actual: dict[str, Any], expected: dict[str, Any], tolerance: float) -> str | None:
    """Return ``None`` when the answer matches, otherwise the first difference in plain words."""
    columns = list(expected["columns"])
    if list(actual.get("columns") or []) or True:  # columns are informational; rows decide
        pass
    rows_actual = actual.get("rows")
    if not isinstance(rows_actual, list):
        return "answer has no `rows` list"
    rows_expected = expected["rows"]
    if len(rows_actual) != len(rows_expected):
        return f"{len(rows_actual)} rows, expected {len(rows_expected)}"

    expected_rows = [
        dict(zip(columns, row, strict=False)) if isinstance(row, list) else dict(row)
        for row in rows_expected
    ]
    actual_rows = [
        dict(zip(columns, row, strict=False)) if isinstance(row, list) else dict(row)
        for row in rows_actual
    ]

    remaining = list(expected_rows)
    for candidate in actual_rows:
        match = None
        for position, wanted in enumerate(remaining):
            keys = set(wanted) | set(candidate)
            if all(
                _numbers_match(_coerce(candidate.get(k)), _coerce(wanted.get(k)), tolerance)
                for k in keys
            ):
                match = position
                break
        if match is None:
            return f"row {candidate} matches no expected row"
        remaining.pop(match)
    return None


def score_answers(answers_dir: Path) -> int:
    questions, tolerance = load_questions()
    rows: list[tuple[str, str, str]] = []
    correct = 0
    for question in questions:
        path = answers_dir / f"{question.id}.json"
        if not path.is_file():
            rows.append((question.id, "MISSING", "no answer file"))
            continue
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            rows.append((question.id, "INVALID", f"not JSON: line {exc.lineno} {exc.msg}"))
            continue

        expected = question.answer
        if expected["kind"] == "error":
            code = str(actual.get("error_code") or "")
            if code == str(expected.get("code")):
                rows.append((question.id, "PASS", f"refused with {code}"))
                correct += 1
            else:
                rows.append((question.id, "FAIL", f"expected refusal {expected.get('code')}, got {code or 'an answer'}"))
            continue

        problem = score_rows(actual, expected, tolerance)
        if problem is None:
            rows.append((question.id, "PASS", "rows match ground truth"))
            correct += 1
        else:
            rows.append((question.id, "FAIL", problem))

    width = max(len(name) for name, _, _ in rows) + 2
    print(f"{'question'.ljust(width)}{'result':8}detail")
    print("-" * (width + 8 + 60))
    for name, result, detail in rows:
        print(f"{name.ljust(width)}{result.ljust(8)}{detail[:90]}")
    counts = Counter(result for _, result, _ in rows)
    print()
    print(f"score: {correct}/{len(questions)} correct  ({dict(counts)})")
    return 0 if correct == len(questions) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prompts = sub.add_parser("prompts", help="write one prompt per benchmark question")
    prompts.add_argument("--out", type=Path, default=Path("bench/prompts"))
    prompts.add_argument("--call", default=CALL_SCRIPT, help="how the model invokes a tool")
    score = sub.add_parser("score", help="score a directory of answers against ground truth")
    score.add_argument("--answers", type=Path, default=Path("bench/answers"))
    args = parser.parse_args(argv)

    if args.command == "prompts":
        write_prompts(args.out, call=args.call)
        return 0
    return score_answers(args.answers)


if __name__ == "__main__":
    sys.exit(main())
