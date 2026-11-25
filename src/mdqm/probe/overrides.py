"""Admin overrides: the operator's verdict beats the probe, and says that it did.

Org facts live in config, never in code (principle 1). When a probe cannot see the truth — a
licence-gated feature, a driver that lies, a source behind a proxy — an operator pins the verdict
in ``overrides.yaml`` next to the generated capability map::

    # semantic/overrides.yaml (humans edit this; the pipeline never rewrites it)
    defaults:            # applied to every source, then overridden per source
      query_store: false
    sources:
      crm:
        explain_plan: true
        partition_metadata:
          supported: true
          detail: verified with the DBA, the probe's EXPLAIN is not readable

Each entry is ``true``/``false`` (pinned supported/unsupported), ``null`` (pinned *unknown*:
stop guessing) or a mapping with ``supported`` and an optional ``detail``. A pinned verdict
survives every re-probe because :func:`~mdqm.probe.merge.apply_overrides` runs after the probe and
marks the entry ``source: override`` (principle 10: nothing silent).

Errors name the file, the key path and the YAML line, and never echo a value, so the message is
safe to log and to show to an operator.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import Field
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from mdqm.config.settings import ConfigError
from mdqm.models.base import MDQMModel
from mdqm.probe.features import CAPABILITIES

__all__ = [
    "OVERRIDES_FILE",
    "CapabilityOverride",
    "OverrideSet",
    "load_overrides",
    "parse_overrides",
]

#: File name of the admin override file, relative to the semantic root (``docs/DESIGN.md`` 4).
OVERRIDES_FILE = "overrides.yaml"

#: The only keys an override document may carry at the top level.
_TOP_KEYS: tuple[str, ...] = ("defaults", "sources")

#: The only keys one capability entry may carry.
_ENTRY_KEYS: tuple[str, ...] = ("supported", "detail")

#: Any YAML tag. ``overrides.yaml`` holds only scalars, booleans, nulls and mappings, so a tag is
#: never legitimate here, and rejecting every tag makes object construction impossible regardless
#: of which ruamel constructor is in play (``docs/DESIGN.md`` section 4, principle 11).
_TAG = re.compile(r"!")


class _KeyMarker(Protocol):
    """ruamel's ``lc.key(key)``: the 0-based (line, column) a mapping key starts on."""

    def __call__(self, key: Any, /) -> tuple[int, int]: ...


class CapabilityOverride(MDQMModel):
    """One pinned verdict. ``supported`` is tri-state, ``None`` meaning "pin it as unknown"."""

    supported: bool | None = None
    detail: str | None = None


class OverrideSet(MDQMModel):
    """Everything ``overrides.yaml`` said, with ``defaults`` already folded under each source."""

    #: The file the overrides came from, or ``None`` when no file was configured.
    path: str | None = None
    defaults: dict[str, CapabilityOverride] = Field(default_factory=dict)
    sources: dict[str, dict[str, CapabilityOverride]] = Field(default_factory=dict)

    def for_source(self, name: str) -> dict[str, CapabilityOverride]:
        """The pinned verdicts for source ``name``: ``defaults`` first, then its own entries."""
        merged = dict(self.defaults)
        merged.update(self.sources.get(name, {}))
        return merged

    def capability_names(self) -> tuple[str, ...]:
        """Every capability name the file mentions, sorted (for typo reporting)."""
        names = set(self.defaults)
        for pinned in self.sources.values():
            names.update(pinned)
        return tuple(sorted(names))

    def unregistered(self) -> tuple[str, ...]:
        """Names the catalogue does not know: an override that silently does nothing is a bug."""
        known = set(CAPABILITIES)
        return tuple(name for name in self.capability_names() if name not in known)


def load_overrides(path: Path | None) -> OverrideSet:
    """Read ``path``; a missing or unconfigured file is an empty override set, not an error."""
    if path is None:
        return OverrideSet()
    if not path.is_file():
        return OverrideSet(path=str(path))
    text = path.read_text(encoding="utf-8")
    tag = _TAG.search(text)
    if tag is not None:
        raise ConfigError(
            str(path),
            [
                f"line {text[: tag.start()].count(chr(10)) + 1}: "
                "YAML tags are not allowed here; write a plain value"
            ],
        )
    # `typ="rt"` is the round-trip loader because only it records line numbers for the error
    # messages. Its constructor derives from SafeConstructor (verified: a `!!python/object/apply`
    # tag is not executed), and the tag guard above rejects every tag before this call, so no
    # tag-driven construction can happen at all.
    yaml = YAML(typ="rt")
    try:
        document = yaml.load(text)
    except YAMLError as exc:
        # `from None`: ruamel echoes the offending source line, which a broken file might make
        # carry a credential; the type and the file are enough to fix it.
        raise ConfigError(str(path), [f"invalid YAML: {type(exc).__name__}"]) from None
    return parse_overrides(document, path=path)


def parse_overrides(document: object, *, path: Path) -> OverrideSet:
    """Validate an already-parsed YAML document; every problem is reported in one error."""
    problems: list[str] = []
    if document is None:
        return OverrideSet(path=str(path))
    if not isinstance(document, Mapping):
        raise ConfigError(str(path), ["the top level must be a mapping of defaults/sources"])
    for key in document:
        if key not in _TOP_KEYS:
            problems.append(
                f"{_line(document, key)}unknown key {key!r}: expected 'defaults' or 'sources'"
            )
    defaults = _entries(
        document.get("defaults"),
        where="defaults",
        parent=document,
        parent_key="defaults",
        problems=problems,
    )
    raw_sources = document.get("sources")
    sources: dict[str, dict[str, CapabilityOverride]] = {}
    if raw_sources is not None:
        if not isinstance(raw_sources, Mapping):
            problems.append(
                f"{_line(document, 'sources')}'sources' must be a mapping of source names"
            )
        else:
            for name, entries in raw_sources.items():
                if not isinstance(name, str) or not name.strip():
                    problems.append(
                        f"{_line(raw_sources, name)}a source name must be a non-empty string"
                    )
                    continue
                sources[name] = _entries(
                    entries,
                    where=f"sources.{name}",
                    parent=raw_sources,
                    parent_key=name,
                    problems=problems,
                )
    if problems:
        raise ConfigError(str(path), problems)
    return OverrideSet(path=str(path), defaults=defaults, sources=sources)


def _entries(
    entries: object,
    *,
    where: str,
    parent: object,
    parent_key: str | None,
    problems: list[str],
) -> dict[str, CapabilityOverride]:
    """Parse one capability -> verdict mapping, appending a problem for each bad entry."""
    if entries is None:
        return {}
    if not isinstance(entries, Mapping):
        problems.append(f"{_line(parent, parent_key)}{where}: expected a mapping of capabilities")
        return {}
    parsed: dict[str, CapabilityOverride] = {}
    for name, value in entries.items():
        if not isinstance(name, str) or not name.strip():
            problems.append(
                f"{_line(entries, name)}{where}: a capability name must be a non-empty string"
            )
            continue
        verdict = _entry(value, where=f"{where}.{name}", node=entries, key=name, problems=problems)
        if verdict is not None:
            parsed[name] = verdict
    return parsed


def _entry(
    value: object,
    *,
    where: str,
    node: object,
    key: str,
    problems: list[str],
) -> CapabilityOverride | None:
    """One verdict: a bool, ``null``, or a ``{supported, detail}`` mapping."""
    where_line = _line(node, key)
    if value is None or isinstance(value, bool):
        return CapabilityOverride(supported=value)
    if isinstance(value, Mapping):
        for entry_key in value:
            if entry_key not in _ENTRY_KEYS:
                problems.append(
                    f"{_line(value, entry_key)}{where}: unknown key {entry_key!r}: "
                    "expected 'supported' or 'detail'"
                )
        supported = value.get("supported")
        if supported is not None and not isinstance(supported, bool):
            problems.append(
                f"{_line(value, 'supported')}{where}: 'supported' must be true, false or null"
            )
            return None
        detail = value.get("detail")
        if detail is not None and not isinstance(detail, str):
            problems.append(f"{_line(value, 'detail')}{where}: 'detail' must be text")
            return None
        return CapabilityOverride(supported=supported, detail=detail)
    problems.append(
        f"{where_line}{where}: expected true, false, null or a mapping with 'supported'"
    )
    return None


def _line(node: object, key: Any) -> str:
    """``"line 7: "`` when ruamel recorded where ``key`` starts, otherwise empty.

    A round-trip document carries line numbers; a plain dict (a caller passing Python objects)
    does not, and a message without a line is still a message.
    """
    locations = getattr(node, "lc", None)
    locate = getattr(locations, "key", None)
    if not callable(locate):
        return ""
    marker = cast(_KeyMarker, locate)
    try:
        line = marker(key)[0]
    except (KeyError, TypeError):
        return ""
    return f"line {line + 1}: " if isinstance(line, int) else ""
