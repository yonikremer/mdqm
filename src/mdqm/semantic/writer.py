"""Writing semantic files: generated facts are rewritten, curated comments survive.

The pipeline owns ``generated/`` and rewrites it wholesale; humans own ``curated/`` and must find
their comments exactly where they left them after a nightly regenerate touches the file around
them. One file is written atomically (temporary file, then replace), so a crash never leaves a
half-written layer that the next load would refuse.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.error import YAMLError

from mdqm.semantic.errors import SemanticError
from mdqm.semantic.loader import GENERATED_DIR

__all__ = ["update_curated", "write_generated"]

_SECTION_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")

_HEADER = "# Written by mdqm's bootstrap pipeline. Do not edit: every run replaces this file.\n"


def write_generated(root: Path, section: str, payload: Mapping[str, Any]) -> Path:
    """Write ``root/generated/<section>.yaml`` and never touch ``root/curated/``.

    ``section`` is a file stem, not a path: a name that could escape the directory is rejected
    instead of silently writing outside the semantic tree.
    """
    if not _SECTION_NAME.match(section):
        raise ValueError(f"{section!r} is not a file name: use letters, digits, '_', '-', '.'")
    path = root / GENERATED_DIR / f"{section}.yaml"
    _write_text(path, dump(payload, header=_HEADER))
    return path


def update_curated(path: Path, payload: Mapping[str, Any]) -> Path:
    """Merge ``payload`` into a curated file in place, keeping the comments around it.

    Values that ``payload`` sets are replaced; keys it does not mention keep their text, their
    order and their comments. A missing file is created.
    """
    if not path.is_file():
        _write_text(path, dump(payload))
        return path
    yaml = YAML(typ="rt")
    try:
        document = yaml.load(path.read_text(encoding="utf-8"))
    except YAMLError as exc:
        raise SemanticError(str(path), [f"invalid YAML: {type(exc).__name__}"]) from None
    if document is None:
        document = CommentedMap()
    if not isinstance(document, CommentedMap):
        raise SemanticError(str(path), ["the top level of a semantic file must be a mapping"])
    _update(document, payload)
    stream = io.StringIO()
    yaml.dump(document, stream)
    _write_text(path, stream.getvalue())
    return path


def dump(payload: Mapping[str, Any], header: str = "") -> str:
    """Render ``payload`` as block-style YAML with UTF-8 text (Hebrew stays readable)."""
    yaml = YAML(typ="rt")
    yaml.allow_unicode = True
    yaml.default_flow_style = False
    stream = io.StringIO()
    yaml.dump(CommentedMap(payload), stream)
    return header + stream.getvalue()


def _update(target: CommentedMap, payload: Mapping[str, Any]) -> None:
    """Recursively assign into the round-trip document, so untouched keys keep their comments."""
    for key, value in payload.items():
        current = target.get(key)
        if isinstance(current, CommentedMap) and isinstance(value, Mapping):
            _update(current, value)
        else:
            target[key] = value


def _write_text(path: Path, text: str) -> None:
    """Write through a temporary file in the same directory, then replace: no half-written layer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
