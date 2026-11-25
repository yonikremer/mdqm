"""One probe run: adapters → guarded verdicts → admin overrides → report + diff.

This is the re-runnable entry point the CLI and the nightly job share. It is read-only against
sources (only ``probe()`` and the metadata-store hooks are called), it never raises for a source
that fails, and it returns the same :class:`ProbeOutcome` shape whether it ran over one fake
adapter in a test or over every configured source in production.

The metadata store is probed through callables, not imported behaviour, so H01 can plug the
richer Context Pack probes in later without this module learning about indexes: pass
``store_probes={...}`` and each entry gets its own timeout and its own verdict.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping

from mdqm.adapters.base import SourceAdapter
from mdqm.cache.freshness import Clock, SystemClock
from mdqm.models import Capabilities, Capability
from mdqm.models.base import MDQMModel
from mdqm.probe.merge import apply_overrides
from mdqm.probe.overrides import OverrideSet
from mdqm.probe.report import ProbeDiff, ProbeReport, build_report, diff_reports
from mdqm.probe.runner import ProbeRunner
from mdqm.store.db import check_fts5

__all__ = [
    "METADATA_STORE",
    "ProbeOutcome",
    "metadata_store_probes",
    "metadata_store_version",
    "run_probe",
]

#: Source name of the metadata store in the capability map: it is a source like any other.
METADATA_STORE = "metadata_store"

#: Adapter kind reported for the metadata store (the store is SQLite, always).
METADATA_STORE_ADAPTER = "sqlite"


class ProbeOutcome(MDQMModel):
    """What one run produced: the artifact to write and the reviewable difference."""

    report: ProbeReport
    diff: ProbeDiff


def metadata_store_probes() -> dict[str, Callable[[], Capability]]:
    """The metadata store's feature probes: FTS5 today, whatever H01 adds later.

    Each callable returns its own verdict instead of raising, so a missing module is a recorded
    capability (and :func:`~mdqm.probe.features.fallbacks_for` then names the degraded feature).
    """
    return {"fts5": lambda: check_fts5().capability()}


def metadata_store_version() -> str:
    """The SQLite version actually in use, discovered rather than assumed."""
    return f"sqlite-{sqlite3.sqlite_version}"


def run_probe(
    sources: Mapping[str, SourceAdapter],
    *,
    overrides: OverrideSet | None = None,
    previous: ProbeReport | None = None,
    store_probes: Mapping[str, Callable[[], Capability]] | None = None,
    runner: ProbeRunner | None = None,
    clock: Clock | None = None,
) -> ProbeOutcome:
    """Probe every source, merge the admin's pins, and diff against the previous run.

    ``sources`` maps the configured source name to an adapter (the name is authoritative). An
    adapter that raises or hangs is reported ``unknown`` for the capabilities it could not answer,
    never as an exception. ``store_probes`` defaults to :func:`metadata_store_probes`; pass ``{}``
    to leave the metadata store out of the map.
    """
    active_clock = clock or SystemClock()
    owned_runner = runner is None
    active = runner or ProbeRunner(clock=active_clock)
    try:
        probed: dict[str, Capabilities] = {}
        for name in sorted(sources):
            adapter = sources[name]
            capability_map = active.probe_adapter(adapter, name=name, kind=adapter.kind)
            probed[name] = apply_overrides(capability_map, overrides)
        hooks = metadata_store_probes() if store_probes is None else store_probes
        if hooks:
            store = active.probe_features(
                METADATA_STORE,
                hooks,
                adapter=METADATA_STORE_ADAPTER,
                version=metadata_store_version(),
            )
            probed[METADATA_STORE] = apply_overrides(store, overrides)
        report = build_report(probed, now=active_clock.now(), overrides=overrides)
        return ProbeOutcome(report=report, diff=diff_reports(previous, report))
    finally:
        if owned_runner:
            active.close()
