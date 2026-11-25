"""Merge probe results with admin overrides: the override wins, and says so.

Deterministic by construction: only the pinned names change, the result keeps the probe's source
metadata, and the merged map is keyed the way the report renders it (sorted). An override naming a
capability the probe never reported is still honoured — the operator may know something the
probe cannot see — and it reads ``source: override`` like every other pinned entry.
"""

from __future__ import annotations

from mdqm.models import Capabilities, Capability, CapabilitySource
from mdqm.probe.overrides import CapabilityOverride, OverrideSet

__all__ = ["apply_overrides"]


def apply_overrides(probed: Capabilities, overrides: OverrideSet | None) -> Capabilities:
    """Return ``probed`` with the admin's pins applied on top (``source: override``).

    The ``detail`` always carries the two facts an operator needs later: that this was an
    override, and what the probe had said before it was overridden.
    """
    if overrides is None:
        return probed
    pinned = overrides.for_source(probed.source)
    if not pinned:
        return probed
    merged = dict(probed.capabilities)
    for name in sorted(pinned):
        merged[name] = _override_capability(pinned[name], merged.get(name, Capability()))
    return probed.model_copy(update={"capabilities": merged})


def _override_capability(pin: CapabilityOverride, previous: Capability) -> Capability:
    """One pinned verdict, with the probe's own state quoted in the detail."""
    head = f"admin override: {pin.detail}" if pin.detail else "admin override"
    detail = f"{head} (probe said {previous.state})"
    return Capability(supported=pin.supported, detail=detail, source=CapabilitySource.OVERRIDE)
