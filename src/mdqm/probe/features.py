"""The capability catalogue: every optional feature, the capability it needs, its fallback.

Principle 1 ("generic via capability detection") is only real if the fallback is written down
next to the capability. This module is that one table, so a report can say *which* features are
running a fallback and *why* — instead of a caller silently guessing what a source can do.

A capability the probe never answered is ``unknown``, and ``unknown`` is not a failure: the
feature still takes its declared fallback (see :class:`~mdqm.models.capabilities.Capability`).
The report keeps the two states apart, so an operator sees the difference between "this source
cannot" and "nobody asked".

The table mirrors ``docs/DESIGN.md`` "Capabilities and fallbacks"; adding a feature there means
adding one row here, and ``tests/unit/test_probe.py`` proves every row is reachable.
"""

from __future__ import annotations

from collections.abc import Mapping

from mdqm.models import Capability
from mdqm.models.base import MDQMModel

__all__ = [
    "CAPABILITIES",
    "FEATURES",
    "CapabilityGap",
    "Feature",
    "FeatureFallback",
    "fallbacks_for",
    "is_registered",
]


class Feature(MDQMModel):
    """One optional feature: the capabilities it needs and what runs without them."""

    name: str
    needs: tuple[str, ...]
    fallback: str


class CapabilityGap(MDQMModel):
    """One capability that is not positively supported, with the state that made it a gap."""

    capability: str
    state: str


class FeatureFallback(MDQMModel):
    """A feature that must run its fallback, and the gates that are not open.

    Serialised into ``generated/capabilities.yaml`` as a list of mappings (see
    :mod:`mdqm.probe.report`), so the file answers "what is degraded here, and why".
    """

    feature: str
    missing: tuple[CapabilityGap, ...]
    fallback: str


#: Every optional feature of the product, sorted by name for deterministic output.
FEATURES: tuple[Feature, ...] = (
    Feature(
        name="context_search",
        needs=("fts5",),
        fallback="search the semantic layer by scanning with LIKE instead of using the FTS index",
    ),
    Feature(
        name="elastic_query_mode",
        needs=("query_mode",),
        fallback="send Query DSL, the mode every Elasticsearch serves",
    ),
    Feature(
        name="mutability_classification",
        needs=("mutability_counters",),
        fallback="checksum a sample; without that, classify the table as unknown (= mutable)",
    ),
    Feature(
        name="partition_planning",
        needs=("partition_metadata",),
        fallback="plan from indexes and the cost gate only, with no partition pruning",
    ),
    Feature(
        name="profiling",
        needs=("sampling",),
        fallback="profile from catalog statistics and a small, timed fetch",
    ),
    Feature(
        name="pruning_proof",
        needs=("explain_plan",),
        fallback="inject the filter without proof, warn, and cap the scan using statistics",
    ),
    Feature(
        name="query_mining",
        needs=("query_store",),
        fallback="mine the MCP's own query history only",
    ),
    Feature(
        name="structured_model_output",
        needs=("structured_output",),
        fallback="validate the model's JSON in code, up to two retries",
    ),
)

#: Capability names the catalogue knows about, sorted. Not a closed set for adapters (a vendor
#: may probe more), but an override naming none of these is a typo and gets reported as such.
CAPABILITIES: tuple[str, ...] = tuple(
    sorted({need for feature in FEATURES for need in feature.needs})
)


def is_registered(capability: str) -> bool:
    """True when the catalogue declares ``capability`` (so its fallbacks are known)."""
    return capability in CAPABILITIES


def fallbacks_for(capabilities: Mapping[str, Capability]) -> tuple[FeatureFallback, ...]:
    """Features that must run their fallback for one source, sorted by feature name.

    Both ``unsupported`` and ``unknown`` block a feature: a feature may only use a capability it
    positively detected. Anything else takes the declared fallback.
    """
    gaps: list[FeatureFallback] = []
    for feature in FEATURES:
        missing = tuple(
            CapabilityGap(capability=need, state=capabilities.get(need, Capability()).state)
            for need in feature.needs
            if capabilities.get(need, Capability()).supported is not True
        )
        if missing:
            gaps.append(
                FeatureFallback(
                    feature=feature.name, missing=missing, fallback=feature.fallback
                )
            )
    return tuple(gaps)
