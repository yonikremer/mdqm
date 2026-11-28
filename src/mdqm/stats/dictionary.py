"""Sentinels the dictionary marks as missing: ``''`` and ``0`` become NULL in SQL.

This is policy, not a measurement, and it is not part of the semantic layer. Putting it there
would change the layer's version hash every time an operator marked a sentinel. The file is a
small YAML mapping next to the layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

__all__ = ["load_sentinels"]


def load_sentinels(text: str, *, path: str = "sentinels.yaml") -> dict[str, tuple[str, ...]]:
    """Parse ``sentinels: {source.table.column: [values]}``.

    Raises:
        ValueError: the file is not the expected mapping. The message names ``path`` and, when
            the parser has one, the line.
    """
    yaml = YAML(typ="safe")
    try:
        loaded = yaml.load(text)
    except YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = f" line {mark.line + 1}" if mark is not None else ""
        raise ValueError(f"{path}{line}: cannot read sentinel YAML") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{path}: the file must be a mapping with a 'sentinels' key")
    section = loaded.get("sentinels", {})
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise ValueError(f"{path}: 'sentinels' must be a mapping of column to values")
    parsed: dict[str, tuple[str, ...]] = {}
    for key, values in section.items():
        name = str(key)
        if name.count(".") != 2:
            raise ValueError(
                f"{path}: sentinel key {name!r} must be source.table.column (two dots)"
            )
        if isinstance(values, str) or not isinstance(values, (list, tuple)):
            raise ValueError(f"{path}: sentinel values for {name!r} must be a list")
        parsed[name] = tuple(_sentinel(item, path=path, column=name) for item in values)
    return parsed


def _sentinel(item: Any, *, path: str, column: str) -> str:
    if isinstance(item, bool) or not isinstance(item, (str, int)):
        raise ValueError(f"{path}: sentinel for {column!r} must be a string or 0, got {item!r}")
    return str(item)
