"""Capability detection: what a source can and cannot do, per source.

Every optional feature declares the capability it needs and has a fallback, so a capability is
tri-state: supported, unsupported, or unknown (not probed / unreadable). Unknown is not a failure.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from mdqm.models.base import MDQMModel, UtcDatetime


class CapabilitySource(StrEnum):
    """Where a verdict came from. An admin override in ``overrides.yaml`` beats a probe."""

    PROBED = "probed"
    OVERRIDE = "override"


class Capability(MDQMModel):
    """One capability verdict.

    ``supported`` is tri-state: ``True`` (probed, works), ``False`` (probed, refused or missing),
    ``None`` (not probed, or the probe could not tell). ``None`` never means failure: the caller
    takes the declared fallback.
    """

    supported: bool | None = None
    detail: str | None = None
    source: CapabilitySource = CapabilitySource.PROBED

    @property
    def state(self) -> str:
        """Human/agent-readable tri-state: ``supported``, ``unsupported`` or ``unknown``."""
        if self.supported is None:
            return "unknown"
        return "supported" if self.supported else "unsupported"


class Capabilities(MDQMModel):
    """The capability map of one source, keyed by capability name (e.g. ``explain_plan``)."""

    source: str
    adapter: str
    version: str | None = None
    retrieved_at: UtcDatetime
    capabilities: dict[str, Capability] = Field(default_factory=dict)

    def get(self, name: str) -> Capability:
        """Verdict for ``name``; an unprobed name reads as unknown, never as an error."""
        return self.capabilities.get(name, Capability())

    def supports(self, name: str) -> bool:
        """True only when the capability was positively detected."""
        return self.get(name).supported is True

    def is_unknown(self, name: str) -> bool:
        """True when nobody probed ``name`` (or the probe could not tell)."""
        return self.get(name).supported is None

    def unsupported(self) -> tuple[str, ...]:
        """Names probed as unavailable; the feature falls back."""
        return tuple(sorted(n for n, c in self.capabilities.items() if c.supported is False))

    def unknown(self) -> tuple[str, ...]:
        """Names with no usable verdict."""
        return tuple(sorted(n for n, c in self.capabilities.items() if c.supported is None))
