"""Loading a semantic layer: read the YAML, merge ``generated/`` with ``curated/``, validate.

Precedence is the design promise of ``docs/DESIGN.md`` section 10: the pipeline rewrites
``generated/`` and never overwrites a human's work in ``curated/``, so curated wins per key. Merging
is recursive: a curated file may override only ``tables.usage_events.mutability`` and keep every
other generated fact about that table, and deleting that curated key brings the generated value
back.

Every failure is a :class:`~mdqm.semantic.errors.SemanticError` naming the file, the path inside the
document and the line, because a human fixes this file by hand.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.error import MarkedYAMLError, YAMLError

from mdqm.semantic.errors import SemanticError
from mdqm.semantic.models import SemanticLayer

__all__ = ["CURATED_DIR", "GENERATED_DIR", "load"]

#: Machine written facts. Rewritten by the pipeline on every run.
GENERATED_DIR = "generated"
#: Human written facts. Edited by people, never by the pipeline, and wins on conflict.
CURATED_DIR = "curated"

_YAML_SUFFIXES = (".yaml", ".yml")


def load(path: str | Path) -> SemanticLayer:
    """Load one semantic file, or a directory holding ``generated/`` and/or ``curated/``.

    A directory is merged in precedence order: every ``generated/*.yaml`` (file name order), then
    every ``curated/*.yaml``. The result is validated as one document, so a curated file may be a
    fragment that only makes sense together with the generated facts it overrides.
    """
    target = Path(path)
    if target.is_dir():
        files = _directory_files(target)
        label = str(target)
        if not files:
            raise SemanticError(
                label, ["no semantic files found: expected .yaml files in generated/ or curated/"]
            )
    elif target.is_file():
        files = [target]
        label = str(target)
    else:
        raise SemanticError(str(target), ["file or directory not found"])

    origins = _Origins()
    merged = CommentedMap()
    for file_path in files:
        document = _read_yaml(file_path)
        origins.record(document, file_path)
        _merge(merged, document, origins, file_path)
    return _validate(merged, label, origins, files)


def _directory_files(root: Path) -> list[Path]:
    """Return the files in merge order, so loading is deterministic on every OS."""
    files: list[Path] = []
    for section in (GENERATED_DIR, CURATED_DIR):
        directory = root / section
        if not directory.is_dir():
            continue
        files.extend(
            file
            for file in sorted(directory.iterdir())
            if file.is_file() and file.suffix.lower() in _YAML_SUFFIXES
        )
    return files


def _read_yaml(path: Path) -> CommentedMap:
    """Parse one file with the round-trip loader: it keeps comments and line numbers."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SemanticError(str(path), [f"cannot read the file: {exc.strerror or exc}"]) from None
    try:
        document = YAML(typ="rt").load(text)
    except MarkedYAMLError as exc:
        mark = exc.problem_mark
        where = f"line {mark.line + 1}, column {mark.column + 1}" if mark else "unknown position"
        # `from None`: ruamel echoes the offending source line, which is not needed to fix it and
        # would put file content into a message that may reach an agent.
        raise SemanticError(str(path), [f"invalid YAML at {where}: {exc.problem}"]) from None
    except YAMLError as exc:
        raise SemanticError(str(path), [f"invalid YAML: {type(exc).__name__}"]) from None
    if document is None:
        return CommentedMap()
    if not isinstance(document, CommentedMap):
        raise SemanticError(str(path), ["the top level of a semantic file must be a mapping"])
    return document


class _Origins:
    """Which file each merged node and each merged key came from.

    Merging happens in place, so a container can hold keys from two files; without this the error
    for a bad curated value would point at the generated file that supplied the surrounding block.
    """

    def __init__(self) -> None:
        self.nodes: dict[int, Path] = {}
        self.keys: dict[tuple[int, str], Path] = {}
        self.hints: dict[tuple[int, str], Any] = {}
        # id() is only stable while the object lives, and a node overridden by a later file is no
        # longer reachable from the merged document, so every recorded node is pinned here.
        self.pinned: list[Any] = []

    def record(self, node: Any, path: Path) -> None:
        if not isinstance(node, (Mapping, list)):
            return
        self.pinned.append(node)
        self.nodes[id(node)] = path
        if isinstance(node, Mapping):
            for key, child in node.items():
                self.keys[(id(node), str(key))] = path
                self.record(child, path)
        else:
            for child in node:
                self.record(child, path)

    def record_key(self, container: Any, key: Any, path: Path, node: Any = None) -> None:
        """Record where a key came from, and the node it came from so its line is right too."""
        self.pinned.append(container)
        self.keys[(id(container), str(key))] = path
        if node is not None:
            self.pinned.append(node)
            self.hints[(id(container), str(key))] = node

    def line_hint(self, container: Any, part: Any) -> int | None:
        """The line of the winning node for this key, so a curated value points at its own file."""
        if container is None:
            return None
        node = self.hints.get((id(container), str(part)))
        if node is None:
            return None
        return _key_line(node, part) or _line_of(node)

    def of(self, node: Any, parent: Any, part: Any) -> Path | None:
        if parent is not None:
            found = self.keys.get((id(parent), str(part)))
            if found is not None:
                return found
        return self.nodes.get(id(node)) or self.nodes.get(id(parent))


def _named_list(node: Any) -> bool:
    """True for a list of named items (``approved_joins``), the shape that merges per key."""
    return (
        isinstance(node, list)
        and bool(node)
        and all(isinstance(item, Mapping) and "name" in item for item in node)
    )


def _merge(
    lower: CommentedMap, higher: Mapping[Any, Any], origins: _Origins, path: Path
) -> CommentedMap:
    """Merge ``higher`` into ``lower`` in place: higher wins, mappings merge, lists replace.

    Merging into the lower document keeps its line numbers, and a node that ``higher`` overrides
    keeps its own, so both sides still report the line a human must edit.
    """
    for key, value in higher.items():
        current = lower.get(key)
        if value is None and current is not None:
            # An empty section in a curated file means "nothing to override here".
            continue
        if isinstance(current, CommentedMap) and isinstance(value, Mapping):
            _merge(current, value, origins, path)
            # The winning lines for these keys are written in `higher`, not in the merged node.
            for overridden in value:
                origins.record_key(current, overridden, path, value)
        elif _named_list(current) and _named_list(value):
            lower[key] = _merge_named(_list_of(current), _list_of(value), origins, path)
            origins.record_key(lower, key, path, higher)
        else:
            lower[key] = value
            origins.record_key(lower, key, path, higher)
            origins.record(value, path)
    return lower


def _list_of(node: Any) -> list[Any]:
    """The items of a list node; empty for anything else (the callers check the shape first)."""
    return list(node) if isinstance(node, list) else []


def _merge_named(
    lower: Sequence[Any], higher: Sequence[Any], origins: _Origins, path: Path
) -> list[Any]:
    """Merge two lists of named items per ``name``: higher wins per item, order is stable."""
    by_name: dict[str, Any] = {
        str(item["name"]): item for item in lower if isinstance(item, CommentedMap)
    }
    order = list(by_name)
    for item in higher:
        if not isinstance(item, Mapping) or "name" not in item:
            continue
        name = str(item["name"])
        existing = by_name.get(name)
        if isinstance(existing, CommentedMap):
            _merge(existing, item, origins, path)
            for overridden in item:
                origins.record_key(existing, overridden, path, item)
        else:
            order.append(name)
            by_name[name] = item
            origins.record(item, path)
    return [by_name[name] for name in order]


def _validate(
    document: CommentedMap, label: str, origins: _Origins, files: Sequence[Path]
) -> SemanticLayer:
    try:
        return SemanticLayer.model_validate(document)
    except ValidationError as exc:
        # One file: the source is already the label, so repeating it per problem is noise.
        with_origin = len(files) > 1
        raise SemanticError(
            label,
            [_problem(error, document, origins, files[0], with_origin) for error in exc.errors()],
        ) from None


def _problem(
    error: Mapping[str, Any],
    document: CommentedMap,
    origins: _Origins,
    fallback: Path,
    with_origin: bool,
) -> str:
    """Format one validation error as ``path: line N: message``, naming the file when merged."""
    location = tuple(error["loc"])
    node, parent = _find(document, location)
    part = location[-1] if location else None
    line = (
        origins.line_hint(parent, part)
        or _key_line(parent, part)
        or _line_of(node)
        or _line_of(parent)
    )
    where = f"line {line}: " if line is not None else ""
    origin = (origins.of(node, parent, part) or fallback).name
    prefix = f"{origin}: " if with_origin else ""
    return f"{prefix}{_format_path(location)}: {where}{_message(error, location, parent)}"


def _message(error: Mapping[str, Any], location: tuple[Any, ...], parent: Any) -> str:
    message = str(error["msg"])
    if error.get("type") == "extra_forbidden" and location:
        key = str(location[-1])
        near = [name for name in _known_names(parent) if name != key]
        close = difflib.get_close_matches(key, near, n=1, cutoff=0.7)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        return f"unknown field {key!r}{hint}: fix the spelling or remove the key"
    return message.replace("Value error, ", "", 1)


def _known_names(parent: Any) -> list[str]:
    """Candidate spellings for a misspelled key: its siblings first, then every model field."""
    siblings = [str(name) for name in parent] if isinstance(parent, Mapping) else []
    return siblings + [name for name in sorted(_field_names(SemanticLayer)) if name not in siblings]


@lru_cache(maxsize=1)
def _field_names(model: type[BaseModel]) -> frozenset[str]:
    """Every field name (and alias) reachable from ``model``, so a typo gets a real suggestion."""
    names: set[str] = set()
    seen: set[type[BaseModel]] = set()
    pending = [model]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for name, field in current.model_fields.items():
            names.add(name)
            names.add(field.alias or name)
            pending.extend(_nested_models(field.annotation))
    return frozenset(names)


def _nested_models(annotation: Any) -> list[type[BaseModel]]:
    """The model classes inside an annotation (``X``, ``X | None``, ``dict[str, X]``, ...)."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    found: list[type[BaseModel]] = []
    for argument in get_args(annotation) if get_origin(annotation) is not None else ():
        found.extend(_nested_models(argument))
    return found


def _find(document: Any, location: tuple[Any, ...]) -> tuple[Any, Any]:
    """Walk the parsed document along a validation location; return ``(node, parent)``."""
    node: Any = document
    parent: Any = None
    for part in location:
        child: Any = None
        if isinstance(node, Mapping):
            if part in node:
                child = node[part]
            elif str(part) in node:
                child = node[str(part)]
        elif (
            isinstance(node, Sequence)
            and not isinstance(node, str)
            and isinstance(part, int)
            and -len(node) <= part < len(node)
        ):
            child = node[part]
        if child is None:
            break
        parent, node = node, child
    return node, parent


def _key_line(node: Any, part: Any) -> int | None:
    """The line the key ``part`` is written on, which is where a human fixes a bad value."""
    key_line = getattr(getattr(node, "lc", None), "key", None)
    if key_line is None or not isinstance(part, str):
        return None
    try:
        return _to_line(key_line(part))
    except (KeyError, ValueError):
        return None


def _line_of(node: Any) -> int | None:
    """The 1-based line a node starts on, when ruamel recorded one."""
    return _to_line(getattr(getattr(node, "lc", None), "line", None))


def _to_line(value: Any) -> int | None:
    """ruamel reports ``(line, column)`` for flow-style nodes and a line for block nodes."""
    if isinstance(value, int):
        return value + 1
    if isinstance(value, tuple) and value and isinstance(value[0], int):
        return int(value[0]) + 1
    return None


def _format_path(location: tuple[Any, ...]) -> str:
    path = ""
    for part in location:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path = f"{path}.{part}" if path else str(part)
    return path or "<document>"
