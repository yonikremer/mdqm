"""Regression: an ``after`` model validator must return ``self``, not a copy.

Pydantic warns (and does not support the copy) when a top-level ``@model_validator(mode="after")``
returns something other than ``self`` during ``__init__``. ``SemanticLayer`` used to return a
``model_copy`` to fill ``version_hash``, which made every construction warn. Scanning the source
catches any validator with that shape, not just the one this test was written for.
"""

from __future__ import annotations

import ast
import warnings
from collections.abc import Iterator
from pathlib import Path

from mdqm.semantic.models import SemanticLayer

SRC = Path(__file__).resolve().parents[2] / "src" / "mdqm"


def _is_after_model_validator(decorator: ast.expr) -> bool:
    return (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == "model_validator"
        and any(
            keyword.arg == "mode"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "after"
            for keyword in decorator.keywords
        )
    )


def _own_returns(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.Return]:
    """Return statements of this function only, never a nested function's."""
    stack = list(node.body)
    while stack:
        child = stack.pop()
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(child, ast.Return):
            yield child
            continue
        stack.extend(ast.iter_child_nodes(child))


def test_after_model_validators_return_self() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(_is_after_model_validator(dec) for dec in node.decorator_list):
                continue
            for statement in _own_returns(node):
                value = statement.value
                if not (isinstance(value, ast.Name) and value.id == "self"):
                    offenders.append(
                        f"{path.relative_to(SRC.parent.parent)}:{statement.lineno} "
                        f"{node.name} returns {ast.unparse(value) if value else 'None'}"
                    )
    assert not offenders, (
        "a top-level after model validator must return self; returning a copy is unsupported:\n"
        + "\n".join(offenders)
    )


def test_constructing_a_semantic_layer_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        layer = SemanticLayer()
    assert layer.version_hash
