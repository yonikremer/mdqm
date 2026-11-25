"""Tests for the capability probe framework and ``generated/capabilities.yaml`` (task A01).

The framework keeps the product generic (principle 1), so the tests pin the three states a
capability can be in, the admin override that beats the probe, the fallbacks each gap triggers, and
the one behaviour a nightly job depends on: a feature that raises or hangs is ``unknown``, never a
crash.

Acceptance map:
  * "with FakeAdapters in 3 configurations the file matches golden YAML" ->
    :func:`test_golden_capability_map`
  * "a probe feature that raises or times out becomes ``unknown`` with detail, never crashes the
    run" -> :func:`test_raising_probe_is_unknown`, :func:`test_timed_out_feature_is_unknown`,
    :func:`test_a_dead_probe_does_not_stop_the_run`
  * "an override flips a feature and survives re-probe; the diff reports ``changed: [...]``"
    -> :func:`test_override_wins_and_survives_reprobe`,
    :func:`test_diff_reports_the_flipped_capability`
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast

import pytest
from ruamel.yaml import YAML
from typer.testing import CliRunner

from mdqm.adapters.base import OPTIONAL_METHODS, NotSupported, SourceAdapter
from mdqm.cli import app
from mdqm.config.settings import ConfigError
from mdqm.models import (
    Capabilities,
    Capability,
    CapabilitySource,
    PartitionBound,
    Partitioning,
    PartitionStrategy,
    RawQuery,
    TableStats,
)
from mdqm.probe import (
    CAPABILITIES,
    FEATURES,
    OVERRIDES_FILE,
    OverrideSet,
    ProbeDiff,
    ProbeOutcome,
    ProbeReport,
    ProbeRunner,
    apply_overrides,
    describe_failure,
    diff_reports,
    fallbacks_for,
    is_registered,
    load_overrides,
    metadata_store_probes,
    parse_overrides,
    read_report,
    run_probe,
    write_report,
)
from mdqm.probe.overrides import CapabilityOverride
from mdqm.probe.runner import ProbeTarget
from mdqm.semantic.loader import GENERATED_DIR
from mdqm.testing.fake_adapter import FakeAdapter

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

#: The DuckDB build string is the one environment-dependent value in the report: it names whatever
#: is installed, so a golden file must not pin it (AGENTS.md "never hard-code fast-changing
#: numbers"). Everything else in the artifact is fixed by the injected clock.
_DUCKDB_VERSION = re.compile(r"duckdb-\d+\.\d+\.\d+")


class FixedClock:
    """Time under test control: the report's timestamps are then part of the golden."""

    def __init__(self, now: datetime = NOW) -> None:
        self.current = now

    def now(self) -> datetime:
        return self.current


class Golden(Protocol):
    """The ``golden`` fixture from ``tests/conftest.py``."""

    def __call__(self, name: str, actual: str) -> None: ...


def _now() -> datetime:
    return NOW


def _probe(sources: Mapping[str, ProbeTarget], **kwargs: Any) -> ProbeOutcome:
    """``run_probe`` over test doubles.

    F05's ``FakeAdapter.kind`` is ``Final[str]`` while F04's ``SourceAdapter`` declares
    ``kind: str``, so a type checker rejects the fake as a ``SourceAdapter`` even though every
    adapter agrees at runtime (the registry accepts it, and conformance passes). One cast here
    keeps the production
    signature honest instead of widening it for tests.
    """
    return run_probe(cast("Mapping[str, SourceAdapter]", sources), **kwargs)


def _crm(**kwargs: object) -> FakeAdapter:
    """A FakeAdapter named ``crm`` with a frozen clock, so goldens are stable."""
    options: dict[str, object] = {"name": "crm", "clock": _now}
    options.update(kwargs)
    return FakeAdapter(**options)  # type: ignore[arg-type]


def _partitioning() -> Partitioning:
    return Partitioning(
        strategy=PartitionStrategy.RANGE,
        key=("start_time",),
        bounds=(
            PartitionBound(name="p1", row_estimate=10),
            PartitionBound(name="p2", row_estimate=10),
        ),
    )


def _normalised(text: str) -> str:
    """The artifact with the installed DuckDB build replaced by a placeholder."""
    return _DUCKDB_VERSION.sub("duckdb-<installed>", text)


def _write(tmp_path: Path, report: ProbeReport) -> str:
    return _normalised(write_report(tmp_path, report).read_text(encoding="utf-8"))


def _report(tmp_path: Path, sources: Mapping[str, ProbeTarget], **kwargs: object) -> ProbeReport:
    """Run one probe over ``sources`` and return the report (no metadata store, no diff)."""
    outcome = _probe(sources, store_probes={}, clock=FixedClock(), **kwargs)
    return outcome.report


# --- the catalogue ------------------------------------------------------------------------


def test_catalogue_is_sorted_and_unique() -> None:
    names = [feature.name for feature in FEATURES]
    assert names == sorted(names)
    assert len(names) == len(set(names))
    assert list(CAPABILITIES) == sorted(CAPABILITIES)


@pytest.mark.parametrize("capability", CAPABILITIES)
def test_every_capability_is_declared_by_a_feature(capability: str) -> None:
    assert is_registered(capability)
    assert any(capability in feature.needs for feature in FEATURES)


def test_unknown_capability_is_not_registered() -> None:
    assert not is_registered("quantum_join_engine")


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        pytest.param({}, ["unknown"] * len(FEATURES), id="nothing-probed"),
        pytest.param(
            {name: True for name in CAPABILITIES},
            [],
            id="everything-supported",
        ),
        pytest.param({"explain_plan": False}, ["unsupported"], id="one-unsupported"),
        pytest.param({"explain_plan": None}, ["unknown"], id="one-unknown"),
    ],
)
def test_fallbacks_for_the_three_states(
    verdicts: dict[str, bool | None], expected: list[str]
) -> None:
    """Every gap state is recorded; only a positive ``True`` opens the gate."""
    capabilities = {name: Capability(supported=value) for name, value in verdicts.items()}
    reported = fallbacks_for(capabilities)
    if not verdicts:
        assert {item.feature for item in reported} == {feature.name for feature in FEATURES}
        assert {gap.state for item in reported for gap in item.missing} == {"unknown"}
        return
    if not expected:
        assert reported == ()
        return
    # Only the capability under test is asserted on; the rest of the catalogue is unprobed.
    target = next(name for name in verdicts)
    affected = [item for item in reported if any(gap.capability == target for gap in item.missing)]
    assert affected, f"no feature declares {target}"
    states = {gap.state for item in affected for gap in item.missing}
    assert states == set(expected)


def test_unsupported_and_unknown_are_reported_apart() -> None:
    """``cannot`` and ``nobody asked`` are different facts, so the report keeps them apart."""
    reported = fallbacks_for(
        {"partition_metadata": Capability(supported=False), "explain_plan": Capability()}
    )
    by_feature = {item.feature: item for item in reported}
    assert by_feature["partition_planning"].missing[0].state == "unsupported"
    assert by_feature["pruning_proof"].missing[0].state == "unknown"


def test_no_fallback_when_every_gate_is_open() -> None:
    assert fallbacks_for({name: Capability(supported=True) for name in CAPABILITIES}) == ()


# --- overrides: parsing -------------------------------------------------------------------


def _write_overrides(tmp_path: Path, text: str) -> Path:
    path = tmp_path / OVERRIDES_FILE
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_overrides_defaults_apply_then_the_source_wins(tmp_path: Path) -> None:
    path = _write_overrides(
        tmp_path,
        "defaults:\n  query_store: false\nsources:\n  crm:\n    query_store: true\n",
    )
    loaded = load_overrides(path)
    assert loaded.for_source("crm")["query_store"].supported is True
    assert loaded.for_source("erp")["query_store"].supported is False
    assert loaded.path == str(path)


def test_overrides_accept_a_mapping_with_detail(tmp_path: Path) -> None:
    path = _write_overrides(
        tmp_path,
        "sources:\n  crm:\n    explain_plan:\n      supported: true\n      detail: DBA said so\n",
    )
    pinned = load_overrides(path).for_source("crm")["explain_plan"]
    assert pinned.supported is True
    assert pinned.detail == "DBA said so"


def test_a_null_override_pins_unknown(tmp_path: Path) -> None:
    path = _write_overrides(tmp_path, "sources:\n  crm:\n    sampling: null\n")
    pinned = load_overrides(path).for_source("crm")["sampling"]
    assert pinned.supported is None


def test_missing_overrides_file_is_empty_not_an_error(tmp_path: Path) -> None:
    loaded = load_overrides(tmp_path / "nope.yaml")
    assert loaded.for_source("crm") == {}
    assert load_overrides(None) == OverrideSet()


def test_unregistered_override_is_reported(tmp_path: Path) -> None:
    path = _write_overrides(tmp_path, "sources:\n  crm:\n    explain_planes: true\n")
    loaded = load_overrides(path)
    assert loaded.unregistered() == ("explain_planes",)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        pytest.param("sources:\n  crm:\n    explain_plan: 3\n", "expected true, false", id="int"),
        pytest.param("sources: 4\n", "must be a mapping", id="sources-not-map"),
        pytest.param("source:\n  crm: {}\n", "unknown key", id="typo-in-top-key"),
        pytest.param(
            "sources:\n  crm:\n    explain_plan:\n      supported: 'yes'\n",
            "must be true, false or null",
            id="supported-not-bool",
        ),
        pytest.param(
            "sources:\n  crm:\n    explain_plan:\n      support: true\n",
            "unknown key",
            id="typo-in-entry",
        ),
    ],
)
def test_override_errors_name_the_file_and_the_path(
    tmp_path: Path, text: str, problem: str
) -> None:
    path = _write_overrides(tmp_path, text)
    with pytest.raises(ConfigError) as caught:
        load_overrides(path)
    message = str(caught.value)
    assert str(path) in message
    assert problem in message


def test_override_error_carries_the_line_number(tmp_path: Path) -> None:
    path = _write_overrides(tmp_path, "sources:\n  crm:\n    explain_plan: 3\n")
    with pytest.raises(ConfigError) as caught:
        load_overrides(path)
    assert "line 3:" in str(caught.value)


def test_broken_yaml_names_the_file(tmp_path: Path) -> None:
    path = _write_overrides(tmp_path, "sources: [unclosed\n")
    with pytest.raises(ConfigError) as caught:
        load_overrides(path)
    assert "invalid YAML" in str(caught.value)


def test_a_yaml_tag_is_refused_before_it_is_constructed(tmp_path: Path) -> None:
    """No tag may reach a constructor, so a python tag cannot execute — it is refused by line."""
    path = _write_overrides(
        tmp_path,
        "sources:\n  crm:\n    x: !!python/object/apply:os.system ['echo hi']\n",
    )
    with pytest.raises(ConfigError) as caught:
        load_overrides(path)
    assert "YAML tags are not allowed" in str(caught.value)
    assert "line 3" in str(caught.value)


def test_parse_overrides_accepts_a_plain_mapping() -> None:
    parsed = parse_overrides(
        {"sources": {"crm": {"query_store": True}}}, path=Path("overrides.yaml")
    )
    assert parsed.for_source("crm")["query_store"].supported is True


# --- overrides: merging -------------------------------------------------------------------


def _probed(**verdicts: bool | None) -> Capabilities:
    return Capabilities(
        source="crm",
        adapter="fake",
        version="fake-1",
        retrieved_at=NOW,
        capabilities={name: Capability(supported=value) for name, value in verdicts.items()},
    )


def test_override_wins_and_says_so() -> None:
    overrides = OverrideSet(sources={"crm": {"explain_plan": CapabilityOverride(supported=True)}})
    merged = apply_overrides(_probed(explain_plan=False), overrides)
    entry = merged.get("explain_plan")
    assert entry.supported is True
    assert entry.source == CapabilitySource.OVERRIDE
    assert entry.detail is not None
    assert "admin override" in entry.detail
    assert "probe said unsupported" in entry.detail


def test_override_keeps_the_operator_detail() -> None:
    overrides = parse_overrides(
        {"sources": {"crm": {"explain_plan": {"supported": True, "detail": "DBA said so"}}}},
        path=Path("overrides.yaml"),
    )
    merged = apply_overrides(_probed(explain_plan=False), overrides)
    detail = merged.get("explain_plan").detail or ""
    assert detail.startswith("admin override: DBA said so")
    assert "probe said unsupported" in detail


def test_override_can_add_a_capability_the_probe_never_reported() -> None:
    overrides = parse_overrides(
        {"sources": {"crm": {"query_mode": True}}}, path=Path("overrides.yaml")
    )
    merged = apply_overrides(_probed(explain_plan=True), overrides)
    assert merged.get("query_mode").supported is True
    assert merged.get("query_mode").source == CapabilitySource.OVERRIDE


def test_override_can_pin_unknown_over_a_probe() -> None:
    overrides = parse_overrides(
        {"sources": {"crm": {"explain_plan": None}}}, path=Path("overrides.yaml")
    )
    merged = apply_overrides(_probed(explain_plan=True), overrides)
    assert merged.get("explain_plan").supported is None
    assert merged.get("explain_plan").source == CapabilitySource.OVERRIDE


def test_no_overrides_leaves_the_probe_untouched() -> None:
    probed = _probed(explain_plan=False)
    assert apply_overrides(probed, None) is probed
    assert apply_overrides(probed, OverrideSet()) is probed


def test_override_does_not_touch_other_sources() -> None:
    overrides = parse_overrides(
        {"sources": {"erp": {"explain_plan": True}}}, path=Path("overrides.yaml")
    )
    merged = apply_overrides(_probed(explain_plan=False), overrides)
    assert merged.get("explain_plan").source == CapabilitySource.PROBED


# --- golden artifact (acceptance 1) -------------------------------------------------------


def _configuration(name: str) -> FakeAdapter:
    if name == "minimal":
        return _crm()
    if name == "declared":
        return _crm(
            partitioning={"usage_events": _partitioning()},
            query_history=(RawQuery(text="SELECT 1", dialect="duckdb"),),
            mutability=(
                TableStats(source="crm", table="accounts", captured_at=NOW, rows_updated=3),
            ),
        )
    return _crm(
        explain_enabled=False,
        extra_capabilities={"query_mode": Capability(detail="not measured")},
    )


@pytest.mark.parametrize("configuration", ["minimal", "declared", "mixed"])
def test_golden_capability_map(configuration: str, tmp_path: Path, golden: Golden) -> None:
    """Three FakeAdapter configurations render one deterministic file each."""
    report = _report(tmp_path, {"crm": _configuration(configuration)})
    golden(f"capabilities_{configuration}.yaml", _write(tmp_path, report))


def test_golden_ordering_is_sorted(tmp_path: Path) -> None:
    report = _report(tmp_path, {"erp": _crm(), "crm": _crm()})
    document = YAML(typ="safe").load(_write(tmp_path, report))
    assert list(document["sources"]) == ["crm", "erp"]
    assert list(document["sources"]["crm"]["capabilities"]) == sorted(
        document["sources"]["crm"]["capabilities"]
    )


def test_rendering_is_stable_across_runs(tmp_path: Path) -> None:
    first = _report(tmp_path, {"crm": _crm()})
    second = _report(tmp_path, {"crm": _crm()})
    assert _write(tmp_path, first) == _write(tmp_path, second)


def test_the_artifact_round_trips(tmp_path: Path) -> None:
    report = _report(tmp_path, {"crm": _crm()})
    path = write_report(tmp_path, report)
    assert read_report(path) == report


def test_a_missing_report_reads_as_none(tmp_path: Path) -> None:
    assert read_report(tmp_path / GENERATED_DIR / "capabilities.yaml") is None


@pytest.mark.parametrize("text", ["sources: [\n", "generated_at: not-a-date\n"])
def test_a_corrupt_report_names_the_file(tmp_path: Path, text: str) -> None:
    path = tmp_path / GENERATED_DIR / "capabilities.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        read_report(path)
    assert str(path) in str(caught.value)


# --- guarded probes (acceptance 2) --------------------------------------------------------


class StubAdapter:
    """Minimal adapter-shaped object: the probe runner needs ``kind`` and ``probe()`` only."""

    kind: ClassVar[str] = "stub"

    def __init__(self, outcome: Capabilities | Exception) -> None:
        self._outcome = outcome

    def probe(self) -> Capabilities:
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def test_raising_probe_is_unknown() -> None:
    """A probe that raises becomes ``unknown`` for every optional capability, with detail."""
    adapter = StubAdapter(RuntimeError("the driver exploded"))
    with ProbeRunner(timeout_s=1.0, clock=FixedClock()) as runner:
        caps = runner.probe_adapter(adapter, name="crm", kind="stub")
    assert caps.source == "crm"
    assert caps.adapter == "stub"
    assert set(caps.capabilities) == set(OPTIONAL_METHODS.values())
    assert all(entry.supported is None for entry in caps.capabilities.values())
    assert all("RuntimeError" in (entry.detail or "") for entry in caps.capabilities.values())


def test_not_supported_from_probe_is_unsupported_with_detail() -> None:
    adapter = StubAdapter(NotSupported("explain_plan", "no PLAN_TABLE privileges"))
    with ProbeRunner(timeout_s=1.0, clock=FixedClock()) as runner:
        caps = runner.probe_adapter(adapter, name="crm", kind="stub")
    entry = caps.get("explain_plan")
    assert entry.supported is False
    assert entry.detail == "no PLAN_TABLE privileges"


def test_a_hanging_probe_times_out_instead_of_hanging_the_run() -> None:
    class SlowAdapter(StubAdapter):
        def probe(self) -> Capabilities:
            time.sleep(0.4)
            return super().probe()

    with ProbeRunner(timeout_s=0.05, clock=FixedClock()) as runner:
        slow = SlowAdapter(_probed(explain_plan=True))
        caps = runner.probe_adapter(slow, name="crm", kind="stub")
    assert all(entry.supported is None for entry in caps.capabilities.values())
    details = [entry.detail or "" for entry in caps.capabilities.values()]
    assert all("timed out after 0.05s" in detail for detail in details)


def test_timed_out_feature_is_unknown_and_the_others_still_report() -> None:
    """Per-feature timeouts: one slow hook costs that hook only."""

    def slow() -> Capability:
        time.sleep(0.4)
        return Capability(supported=True)

    probes = {
        "fts5": lambda: Capability(supported=True, detail="indexed"),
        "sampling": lambda: Capability(supported=False, detail="no sampling"),
        "query_mode": slow,
    }
    with ProbeRunner(timeout_s=0.05, clock=FixedClock()) as runner:
        caps = runner.probe_features("metadata_store", probes, adapter="sqlite")
    assert caps.get("fts5").supported is True
    assert caps.get("sampling").supported is False
    assert caps.get("query_mode").supported is None
    assert "timed out after 0.05s" in (caps.get("query_mode").detail or "")


def test_a_dead_probe_does_not_stop_the_run(tmp_path: Path) -> None:
    """The run completes and writes the artifact even when a source refuses to answer."""
    sources: dict[str, ProbeTarget] = {
        "crm": _crm(),
        "erp": StubAdapter(RuntimeError("gone")),
    }
    outcome = _probe(sources, store_probes={}, clock=FixedClock())
    assert set(outcome.report.sources) == {"crm", "erp"}
    assert outcome.report.sources["crm"].capabilities["explain_plan"].supported is True
    assert all(
        entry.supported is None for entry in outcome.report.sources["erp"].capabilities.values()
    )


def test_failure_detail_masks_credentials_and_clips() -> None:
    """A driver message can carry a DSN; it never reaches the report verbatim."""
    detail = describe_failure(
        "crm",
        RuntimeError("connect failed: postgresql://mdqm:s3cret@db.internal:5432/crm"),
    )
    assert "s3cret" not in detail
    assert "***@" in detail
    assert detail.startswith("crm raised RuntimeError")
    assert len(describe_failure("crm", RuntimeError("x" * 500))) <= 200


def test_failure_detail_masks_key_value_secrets() -> None:
    detail = describe_failure("crm", RuntimeError("login denied password=hunter2 token=abc"))
    assert "hunter2" not in detail
    assert "abc" not in detail


def test_metadata_store_is_probed_through_a_hook(tmp_path: Path) -> None:
    outcome = _probe(
        {"crm": _crm()},
        store_probes=metadata_store_probes(),
        clock=FixedClock(),
    )
    store = outcome.report.sources["metadata_store"]
    assert store.adapter == "sqlite"
    assert backticked_fts5(store.capabilities)


def backticked_fts5(capabilities: dict[str, Capability]) -> bool:
    """FTS5 is either available or not in this build; only "not asked" would be a bug."""
    entry = capabilities.get("fts5")
    return entry is not None and entry.supported in (True, False)


# --- overrides survive a re-probe, and the diff says what moved (acceptance 3) ------------


def test_override_wins_and_survives_reprobe(tmp_path: Path) -> None:
    overrides = parse_overrides(
        {"sources": {"crm": {"explain_plan": False}}}, path=Path("overrides.yaml")
    )
    first = _probe({"crm": _crm()}, overrides=overrides, store_probes={}, clock=FixedClock())
    assert first.report.sources["crm"].get_capability("explain_plan").supported is False

    second = _probe(
        {"crm": _crm()},
        overrides=overrides,
        previous=first.report,
        store_probes={},
        clock=FixedClock(),
    )
    entry = second.report.sources["crm"].get_capability("explain_plan")
    assert entry.supported is False
    assert entry.source == CapabilitySource.OVERRIDE
    assert entry.detail is not None and "admin override" in entry.detail
    assert second.diff.is_empty, "the same override must not look like a change on re-probe"


def test_diff_reports_the_flipped_capability(tmp_path: Path) -> None:
    """The probe says supported; the admin pins it off; the diff names the flip."""
    before = _probe({"crm": _crm()}, store_probes={}, clock=FixedClock()).report
    overrides = parse_overrides(
        {"sources": {"crm": {"explain_plan": False}}}, path=Path("overrides.yaml")
    )
    after = _probe(
        {"crm": _crm()}, overrides=overrides, previous=before, store_probes={}, clock=FixedClock()
    )
    assert "crm.explain_plan: supported -> unsupported" in after.diff.changed
    assert "changed: 1 (crm.explain_plan: supported -> unsupported)" in after.diff.describe()


def test_diff_records_a_resolved_capability() -> None:
    """The other direction: a pinned verdict removed shows the probe's answer coming back."""
    overrides = parse_overrides(
        {"sources": {"crm": {"explain_plan": False}}}, path=Path("overrides.yaml")
    )
    before = _probe(
        {"crm": _crm()}, overrides=overrides, store_probes={}, clock=FixedClock()
    ).report
    after = _probe({"crm": _crm()}, previous=before, store_probes={}, clock=FixedClock()).report
    assert "crm.explain_plan: unsupported -> supported" in diff_reports(before, after).changed


def test_diff_reports_an_overridden_unknown() -> None:
    overrides = parse_overrides(
        {"sources": {"crm": {"query_mode": None}}}, path=Path("overrides.yaml")
    )
    before = _probe({"crm": _crm()}, store_probes={}, clock=FixedClock()).report
    after = _probe(
        {"crm": _crm()}, overrides=overrides, previous=before, store_probes={}, clock=FixedClock()
    ).report
    assert "crm.query_mode: absent -> unknown" in diff_reports(before, after).changed


def test_first_run_has_no_diff() -> None:
    outcome = _probe({"crm": _crm()}, store_probes={}, clock=FixedClock())
    assert outcome.diff == ProbeDiff()
    assert outcome.diff.is_empty


def test_removed_source_is_reported() -> None:
    before = _probe({"crm": _crm(), "erp": _crm()}, store_probes={}, clock=FixedClock()).report
    after = _probe({"crm": _crm()}, store_probes={}, clock=FixedClock()).report
    assert "erp: source removed" in diff_reports(before, after).changed


def test_report_exposes_the_overridden_names(tmp_path: Path) -> None:
    overrides = parse_overrides(
        {"sources": {"crm": {"query_store": True}}}, path=Path("overrides.yaml")
    )
    report = _report(tmp_path, {"crm": _crm()}, overrides=overrides)
    assert report.sources["crm"].overridden == ("query_store",)
    assert report.overrides_path == "overrides.yaml"


def test_unregistered_override_reaches_the_report(tmp_path: Path) -> None:
    overrides = parse_overrides(
        {"sources": {"crm": {"explain_planes": True}}}, path=Path("overrides.yaml")
    )
    report = _report(tmp_path, {"crm": _crm()}, overrides=overrides)
    assert report.unregistered_overrides == ("explain_planes",)


# --- the CLI ------------------------------------------------------------------------------


def test_cli_writes_the_capability_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["probe", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    written = tmp_path / GENERATED_DIR / "capabilities.yaml"
    assert written.is_file()
    assert "metadata_store" in result.output
    assert "no capability changes" in result.output


def test_cli_applies_the_default_overrides_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_overrides(tmp_path, "sources:\n  metadata_store:\n    fts5: null\n")
    result = CliRunner().invoke(app, ["probe", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    report = read_report(tmp_path / GENERATED_DIR / "capabilities.yaml")
    assert report is not None
    entry = report.sources["metadata_store"].capabilities["fts5"]
    assert entry.supported is None
    assert entry.source == CapabilitySource.OVERRIDE


def test_cli_unknown_source_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["probe", "--source", "nope", "--root", str(tmp_path)])
    assert result.exit_code == 1
    assert "no configured source named" in result.output


def test_cli_reports_a_broken_overrides_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_overrides(tmp_path, "sources:\n  crm:\n    explain_plan: 3\n")
    result = CliRunner().invoke(app, ["probe", "--root", str(tmp_path)])
    assert result.exit_code == 1
    assert "error:" in result.output
