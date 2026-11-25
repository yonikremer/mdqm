"""Capability probing: what each source can do, and which features fall back because it cannot.

``docs/DESIGN.md`` section 4, principle 1. The pieces:

* :mod:`mdqm.probe.features` — the one table: capability -> features that need it -> fallback.
* :mod:`mdqm.probe.runner` — guarded probe calls: a slow or failing feature is ``unknown``, never
  a crash.
* :mod:`mdqm.probe.overrides` — ``overrides.yaml``: the operator's pinned verdict.
* :mod:`mdqm.probe.merge` — the override wins over the probe, and the entry says so.
* :mod:`mdqm.probe.report` — ``generated/capabilities.yaml`` plus the diff against the last run.
* :mod:`mdqm.probe.run` — one run over every source.
"""

from __future__ import annotations

from mdqm.probe.features import (
    CAPABILITIES,
    FEATURES,
    CapabilityGap,
    Feature,
    FeatureFallback,
    fallbacks_for,
    is_registered,
)
from mdqm.probe.merge import apply_overrides
from mdqm.probe.overrides import (
    OVERRIDES_FILE,
    CapabilityOverride,
    OverrideSet,
    load_overrides,
    parse_overrides,
)
from mdqm.probe.report import (
    PROBE_SECTION,
    ProbeDiff,
    ProbeReport,
    SourceReport,
    build_report,
    diff_reports,
    read_report,
    write_report,
)
from mdqm.probe.run import (
    METADATA_STORE,
    ProbeOutcome,
    metadata_store_probes,
    metadata_store_version,
    run_probe,
)
from mdqm.probe.runner import DEFAULT_PROBE_TIMEOUT_S, ProbeRunner, describe_failure

__all__ = [
    "CAPABILITIES",
    "DEFAULT_PROBE_TIMEOUT_S",
    "FEATURES",
    "METADATA_STORE",
    "OVERRIDES_FILE",
    "PROBE_SECTION",
    "CapabilityGap",
    "CapabilityOverride",
    "Feature",
    "FeatureFallback",
    "OverrideSet",
    "ProbeDiff",
    "ProbeOutcome",
    "ProbeReport",
    "ProbeRunner",
    "SourceReport",
    "apply_overrides",
    "build_report",
    "describe_failure",
    "diff_reports",
    "fallbacks_for",
    "is_registered",
    "load_overrides",
    "metadata_store_probes",
    "metadata_store_version",
    "parse_overrides",
    "read_report",
    "run_probe",
    "write_report",
]
