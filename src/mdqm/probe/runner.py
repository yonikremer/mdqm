"""Guarded probe calls: one slow, refusing or lying feature never aborts the run.

Two call shapes are guarded here:

* :meth:`ProbeRunner.probe_adapter` runs an adapter's ``probe()`` once, under a timeout. If it
  raises :class:`~mdqm.adapters.base.NotSupported`, that single capability becomes
  ``unsupported`` with the adapter's detail. If it raises anything else or overruns, every
  optional capability the protocol can carry becomes ``unknown`` with the failure in ``detail`` —
  never an exception out of the probe run (acceptance: "a probe feature that raises or times out
  becomes ``unknown`` with detail, never crashes the run").
* :meth:`ProbeRunner.probe_features` runs independent per-feature callables (the metadata store
  hook H01 plugs into), each with its own timeout, so a hanging feature costs that feature only.

Timeouts use a thread pool with ``future.result(timeout=...)``: the only approach that works the
same on Windows, Linux and macOS (``signal.SIGALRM`` does not exist on Windows and only works on
the main thread elsewhere). A timed-out call cannot be killed, only abandoned — see the
``shortcut:`` note on :meth:`ProbeRunner._guarded`.

Failure detail is deliberately narrow. An exception's message from a driver can carry a DSN with a
password, so credential-shaped text is masked, control characters are dropped and the line is
clipped before it ever reaches a log or the YAML report (principle 11).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, Protocol, TypeVar

from mdqm.adapters.base import OPTIONAL_METHODS, NotSupported
from mdqm.cache.freshness import Clock, SystemClock
from mdqm.models import Capabilities, Capability

__all__ = ["DEFAULT_PROBE_TIMEOUT_S", "ProbeRunner", "ProbeTarget", "describe_failure"]

#: Default per-call budget. Short on purpose: a probe runs on every re-probe and must never be the
#: reason a nightly job is slow. An operator raises it for a genuinely slow source.
DEFAULT_PROBE_TIMEOUT_S = 10.0

#: Upper bound on the failure text that reaches the report.
_MAX_DETAIL = 160

#: Credential-shaped text, masked before a driver message is stored or logged.
_SECRETS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s/@]*@"), "://***@"),
    (re.compile(r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key)\b\s*[=:]\s*\S+"), r"\1=***"),
)

_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")

_T = TypeVar("_T")


class ProbeTarget(Protocol):
    """What probing needs from a source: one ``probe()`` call, nothing more.

    Deliberately not :class:`~mdqm.adapters.base.SourceAdapter`: probing never touches
    catalog/execute, and the caller already knows the name and the adapter kind (it passes them
    in), so a test double with a single method is a valid argument.
    """

    def probe(self) -> Capabilities:
        """Detect what this source supports."""
        ...


@dataclass(frozen=True)
class _Outcome(Generic[_T]):
    """A guarded call: the value, or the capability verdict that replaces it."""

    value: _T | None = None
    failure: Capability | None = None


class ProbeRunner:
    """Runs probe calls under a timeout and turns every failure into a capability verdict."""

    def __init__(
        self,
        *,
        timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
        clock: Clock | None = None,
        max_workers: int = 4,
    ) -> None:
        """``max_workers`` bounds the abandoned threads a run can accumulate on repeated hangs."""
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        self.timeout_s = timeout_s
        self._clock = clock or SystemClock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mdqm-probe")
        self._closed = False

    # --- adapter probe ---------------------------------------------------------------------

    def probe_adapter(self, adapter: ProbeTarget, *, name: str, kind: str) -> Capabilities:
        """One adapter's capability map, or an all-unknown map naming why it failed.

        ``name`` is the configured source name and ``kind`` the adapter kind: both are supplied by
        the caller, because the report must use the name the operator configured, whatever the
        adapter calls itself.
        """
        started = self._clock.now()
        source = name

        def call() -> Capabilities:
            try:
                return adapter.probe()
            except NotSupported as exc:
                # A normal answer, not a failure: the adapter declares what it cannot do.
                return Capabilities(
                    source=source,
                    adapter=kind,
                    retrieved_at=started,
                    capabilities={exc.capability: Capability(supported=False, detail=exc.detail)},
                )

        outcome = self._guarded(f"{source} probe", call)
        if outcome.value is not None:
            probed = outcome.value
            return Capabilities(
                source=source,
                adapter=kind,
                version=probed.version,
                retrieved_at=started,
                capabilities=probed.capabilities,
            )
        detail = outcome.failure.detail if outcome.failure is not None else None
        return Capabilities(
            source=source,
            adapter=kind,
            retrieved_at=started,
            capabilities={
                capability: Capability(supported=None, detail=detail)
                for capability in sorted(set(OPTIONAL_METHODS.values()))
            },
        )

    # --- per-feature probes ----------------------------------------------------------------

    def probe_features(
        self,
        source: str,
        probes: Mapping[str, Callable[[], Capability]],
        *,
        adapter: str,
        version: str | None = None,
        timeout_s: float | None = None,
    ) -> Capabilities:
        """Run each feature probe under its own timeout, in sorted order, and record verdicts."""
        capabilities: dict[str, Capability] = {}
        for capability in sorted(probes):
            outcome = self._guarded(capability, probes[capability], timeout_s=timeout_s)
            if outcome.failure is not None:
                capabilities[capability] = outcome.failure
            elif outcome.value is not None:
                capabilities[capability] = outcome.value
            else:
                capabilities[capability] = Capability(detail="the probe returned no verdict")
        return Capabilities(
            source=source,
            adapter=adapter,
            version=version,
            retrieved_at=self._clock.now(),
            capabilities=capabilities,
        )

    def close(self) -> None:
        """Stop accepting work. A hung call is abandoned, not killed. Idempotent."""
        if not self._closed:
            self._closed = True
            # wait=False: a probe that ignored its timeout must not hold up the process.
            self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> ProbeRunner:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # --- the guard -------------------------------------------------------------------------

    def _guarded(
        self, label: str, call: Callable[[], _T], *, timeout_s: float | None = None
    ) -> _Outcome[_T]:
        """Call ``call`` under a timeout and never raise: the failure becomes a verdict."""
        limit = self.timeout_s if timeout_s is None else timeout_s
        future = self._pool.submit(call)
        try:
            return _Outcome(value=future.result(timeout=limit))
        except FuturesTimeout:
            # shortcut: a Python thread cannot be killed, so the call is abandoned and may keep
            # running until it returns. Bounded by max_workers; upgrade to a process pool (or the
            # driver's own cancel) if a source is known to hang.
            future.cancel()
            return _Outcome(failure=Capability(detail=f"probe timed out after {limit:g}s"))
        except Exception as exc:
            return _Outcome(failure=Capability(detail=describe_failure(label, exc)))


def describe_failure(label: str, exc: BaseException) -> str:
    """``"<label> raised <Type>: <masked, clipped message>"``; safe to log and to store."""
    text = str(exc)
    for pattern, replacement in _SECRETS:
        text = pattern.sub(replacement, text)
    text = _CONTROL.sub(" ", text).strip()
    if len(text) > _MAX_DETAIL:
        text = text[: _MAX_DETAIL - 1].rstrip() + "…"
    named = type(exc).__name__
    return f"{label} raised {named}: {text}" if text else f"{label} raised {named}"
