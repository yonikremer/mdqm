"""The capability map artifact: ``generated/capabilities.yaml``, and the diff against the last run.

One file, machine written, keyed by source: what the probe found, what an operator pinned, and
which features therefore run a fallback (principle 10: fallbacks appear in the capability report).
Rendering is deterministic (sorted sources, sorted capability names, explicit ``null`` detail), so a
nightly re-probe produces a reviewable diff instead of noise.

The file is written through :func:`mdqm.semantic.writer.write_generated`, the same atomic writer the
bootstrap pipeline uses, and it parses back into :class:`ProbeReport`: that is what makes
:func:`diff_reports` able to compare against the previous run without a second format.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from mdqm.config.settings import ConfigError
from mdqm.models import Capabilities, Capability
from mdqm.models.base import MDQMModel, UtcDatetime
from mdqm.probe.features import FeatureFallback, fallbacks_for
from mdqm.probe.overrides import OverrideSet
from mdqm.semantic.writer import write_generated

__all__ = [
    "PROBE_SECTION",
    "ProbeDiff",
    "ProbeReport",
    "SourceReport",
    "build_report",
    "diff_reports",
    "read_report",
    "write_report",
]

#: File stem under ``generated/``: ``semantic/generated/capabilities.yaml`` (DESIGN section 3).
PROBE_SECTION = "capabilities"

#: What a capability reads as when the previous report did not mention it at all.
ABSENT = "absent"


class SourceReport(MDQMModel):
    """One source's entry in the map: what it reports, and what that degrades."""

    adapter: str
    version: str | None = None
    probed_at: UtcDatetime
    capabilities: dict[str, Capability] = Field(default_factory=dict)
    fallbacks: tuple[FeatureFallback, ...] = ()

    @classmethod
    def from_capabilities(cls, capabilities: Capabilities) -> SourceReport:
        """Build the entry, deriving the fallback list from the (already merged) verdicts."""
        return cls(
            adapter=capabilities.adapter,
            version=capabilities.version,
            probed_at=capabilities.retrieved_at,
            capabilities=capabilities.capabilities,
            fallbacks=fallbacks_for(capabilities.capabilities),
        )

    @property
    def overridden(self) -> tuple[str, ...]:
        """Capability names an operator pinned (their verdict beats the probe)."""
        return tuple(
            sorted(name for name, cap in self.capabilities.items() if cap.source == "override")
        )

    def get_capability(self, name: str) -> Capability:
        """One verdict; an unprobed name reads as unknown, never as an error."""
        return self.capabilities.get(name, Capability())

    def payload(self) -> dict[str, Any]:
        """YAML-ready mapping: sorted keys, explicit ``null``, no derived text."""
        return {
            "adapter": self.adapter,
            "version": self.version,
            "probed_at": self.probed_at.isoformat(),
            "capabilities": {
                name: self.capabilities[name].model_dump(mode="json")
                for name in sorted(self.capabilities)
            },
            "fallbacks": [_fallback_payload(item) for item in self.fallbacks],
        }


class ProbeReport(MDQMModel):
    """``generated/capabilities.yaml``: every probed source, plus what config had to say."""

    generated_at: UtcDatetime
    overrides_path: str | None = None
    unregistered_overrides: tuple[str, ...] = ()
    sources: dict[str, SourceReport] = Field(default_factory=dict)

    def source(self, name: str) -> SourceReport | None:
        """One source's entry, or ``None`` when this run did not probe it."""
        return self.sources.get(name)

    def payload(self) -> dict[str, Any]:
        """YAML-ready mapping, deterministic: sources and overrides sorted."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "overrides_path": self.overrides_path,
            "unregistered_overrides": list(self.unregistered_overrides),
            "sources": {name: self.sources[name].payload() for name in sorted(self.sources)},
        }


class ProbeDiff(MDQMModel):
    """What changed between two runs: the reviewable part of a re-probe.

    Each entry is one of ``"<source>.<capability>: <before> -> <after>"`` (states are
    ``supported``, ``unsupported``, ``unknown`` or ``absent``) or ``"<source>: source removed"``.
    """

    changed: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """True when nothing moved, which is the common case for a nightly re-probe."""
        return not self.changed

    def describe(self) -> str:
        """One line for the CLI: the count, or the changes themselves."""
        if self.is_empty:
            return "no capability changes"
        return f"changed: {len(self.changed)} ({'; '.join(self.changed)})"


def build_report(
    probed: Mapping[str, Capabilities],
    *,
    now: datetime,
    overrides: OverrideSet | None = None,
) -> ProbeReport:
    """Assemble the artifact from already-merged results (one entry per source name)."""
    return ProbeReport(
        generated_at=now,
        overrides_path=None if overrides is None else overrides.path,
        unregistered_overrides=() if overrides is None else overrides.unregistered(),
        sources={
            name: SourceReport.from_capabilities(probed[name]) for name in sorted(probed)
        },
    )


def diff_reports(previous: ProbeReport | None, current: ProbeReport) -> ProbeDiff:
    """Capabilities that changed state since ``previous``; a first run has nothing to compare."""
    if previous is None:
        return ProbeDiff()
    changed: list[str] = []
    for name in sorted(current.sources):
        before = previous.sources.get(name)
        after = current.sources[name]
        for capability in sorted(after.capabilities):
            was = ABSENT if before is None else before.capabilities.get(capability)
            state_before = ABSENT if was is None else Capability.model_validate(was).state
            state_after = after.capabilities[capability].state
            if state_before != state_after:
                changed.append(f"{name}.{capability}: {state_before} -> {state_after}")
        if before is not None:
            for capability in sorted(before.capabilities):
                if capability not in after.capabilities:
                    was = before.capabilities[capability].state
                    changed.append(f"{name}.{capability}: {was} -> {ABSENT}")
    for name in sorted(previous.sources):
        if name not in current.sources:
            changed.append(f"{name}: source removed")
    return ProbeDiff(changed=tuple(changed))


def write_report(root: Path, report: ProbeReport) -> Path:
    """Write ``root/generated/capabilities.yaml`` atomically and return its path."""
    return write_generated(root, PROBE_SECTION, report.payload())


def read_report(path: Path) -> ProbeReport | None:
    """Read a previous report, or ``None`` when the file does not exist yet.

    A corrupt file raises :class:`~mdqm.config.settings.ConfigError` naming it: it is machine
    written, so hand damage is a bug, and the caller decides whether to regenerate or stop.
    """
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    try:
        document = YAML(typ="safe").load(text)
    except YAMLError as exc:
        raise ConfigError(
            str(path), [f"invalid YAML: {type(exc).__name__}; delete it to regenerate"]
        ) from None
    if document is None:
        return None
    try:
        return ProbeReport.model_validate(document)
    except ValueError as exc:
        raise ConfigError(
            str(path),
            [f"not a capability report ({type(exc).__name__}); delete it to regenerate"],
        ) from None


def _fallback_payload(item: FeatureFallback) -> dict[str, Any]:
    return {
        "feature": item.feature,
        "missing": [
            {"capability": gap.capability, "state": gap.state} for gap in item.missing
        ],
        "fallback": item.fallback,
    }
