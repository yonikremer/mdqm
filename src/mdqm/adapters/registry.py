"""Adapter registry: the entry-point group ``mdqm.adapters`` plus explicit registration.

Core code asks the registry for an adapter by kind and gets something that satisfies
:class:`~mdqm.adapters.base.SourceAdapter`, or an error that names what is wrong. Third-party
adapters arrive through the entry-point group ``mdqm.adapters`` (one distribution per vendor, per
principle 2 "adapters are plugins"); tests register a factory directly with :meth:`register`.

Nothing here is global or mutable-by-import: build one registry per process (or per test).
"""

from __future__ import annotations

import importlib.metadata
import inspect
from collections.abc import Callable, Iterable
from typing import Protocol, cast

from mdqm.adapters.base import REQUIRED_METHODS, AdapterFactory, SourceAdapter, supports_method
from mdqm.config.secrets import SecretProvider
from mdqm.config.settings import SourceConfig

__all__ = [
    "ADAPTER_ENTRY_POINT_GROUP",
    "AdapterEntryPoint",
    "AdapterError",
    "AdapterNotRegistered",
    "AdapterRegistrationError",
    "AdapterRegistry",
]

#: Entry-point group every adapter distribution declares in its ``pyproject.toml``.
ADAPTER_ENTRY_POINT_GROUP = "mdqm.adapters"


class AdapterEntryPoint(Protocol):
    """The slice of ``importlib.metadata.EntryPoint`` the registry uses (test doubles are easy)."""

    name: str

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub; "..." is the unimplemented body
    def load(self) -> object:
        """Import and return the object the entry point points at."""
        ...


class AdapterError(Exception):
    """Base for adapter registry problems."""


class AdapterRegistrationError(AdapterError, ValueError):
    """An adapter cannot be registered or built: it does not implement the protocol."""


class AdapterNotRegistered(AdapterError, LookupError):
    """No adapter is registered under that kind."""


def _installed_entry_points(group: str) -> Iterable[AdapterEntryPoint]:
    """Every installed entry point in ``group`` (empty when no adapter package is installed)."""
    return list(importlib.metadata.entry_points(group=group))


def _require_methods(kind: str, candidate: object, origin: str) -> None:
    """Fail loudly, naming every missing method, so a broken plugin is a one-line fix."""
    missing = [name for name in REQUIRED_METHODS if not supports_method(candidate, name)]
    if missing:
        raise AdapterRegistrationError(
            f"adapter {kind!r} from {origin} does not implement the SourceAdapter protocol: "
            f"missing {', '.join(missing)}. "
            "Required: kind, dialect, probe, extract_catalog, execute, cancel "
            "(see mdqm.adapters.base.SourceAdapter); optional methods raise NotSupported."
        )


class AdapterRegistry:
    """Kind -> factory, filled from entry points and/or explicit registration."""

    def __init__(
        self,
        *,
        group: str = ADAPTER_ENTRY_POINT_GROUP,
        secrets: SecretProvider | None = None,
        entry_points: Callable[[], Iterable[AdapterEntryPoint]] | None = None,
    ) -> None:
        """``secrets`` is handed to each factory; ``entry_points`` is injectable for tests."""
        self._group = group
        self._secrets = secrets
        self._entry_points = entry_points or (lambda: _installed_entry_points(self._group))
        self._factories: dict[str, AdapterFactory] = {}

    # --- registration -------------------------------------------------------------------

    def register(self, kind: str, factory: AdapterFactory, *, origin: str | None = None) -> None:
        """Register ``factory`` under ``kind`` (the adapter's ``kind`` attribute).

        Validates the required methods when the factory is a class or an instance, the common
        case; a plain function is validated after it is called by :meth:`create`. Registering the
        same kind twice is an error: which plugin wins must never depend on import order.
        """
        if not kind.strip():
            raise AdapterRegistrationError("an adapter kind must be a non-empty name")
        where = origin or "explicit registration"
        if kind in self._factories:
            raise AdapterRegistrationError(
                f"adapter {kind!r} is already registered; unregister it or use another kind "
                f"(the duplicate comes from {where})"
            )
        if not callable(factory):
            raise AdapterRegistrationError(f"adapter {kind!r} from {where} is not callable")
        if not inspect.isroutine(factory):
            # A class or a factory object exposes its methods without being called. A plain
            # function/lambda does not, so it is checked by create() instead.
            _require_methods(kind, factory, where)
        self._factories[kind] = factory

    def discover(self) -> tuple[str, ...]:
        """Load every entry point in the group, register it, return all known kinds.

        Already-registered kinds are left alone, so ``discover`` can run again (on a version
        change, or after an explicit registration) without a duplicate-registration error.
        """
        for entry_point in self._entry_points():
            if entry_point.name in self._factories:
                continue
            origin = f"entry point {self._group}:{entry_point.name}"
            try:
                loaded = entry_point.load()
            except Exception as exc:  # a broken plugin must stop discovery, not vanish silently
                raise AdapterRegistrationError(
                    f"{origin} failed to load: {type(exc).__name__}: {exc}"
                ) from exc
            self.register(entry_point.name, cast(AdapterFactory, loaded), origin=origin)
        return self.names()

    # --- lookup -------------------------------------------------------------------------

    def names(self) -> tuple[str, ...]:
        """Registered kinds, sorted (stable output for logs and error messages)."""
        return tuple(sorted(self._factories))

    def __contains__(self, kind: object) -> bool:
        return kind in self._factories

    def factory(self, kind: str) -> AdapterFactory:
        """The factory for ``kind``; the error lists what is available."""
        try:
            return self._factories[kind]
        except KeyError:
            available = ", ".join(self.names()) or "none"
            raise AdapterNotRegistered(
                f"no adapter registered for {kind!r} (registered kinds: {available}). "
                f"Install the plugin that provides it (entry point group {self._group}) "
                "or register it explicitly."
            ) from None

    def create(self, config: SourceConfig) -> SourceAdapter:
        """Build the adapter for ``config`` and check it is what it claims to be."""
        kind = config.adapter
        factory = self.factory(kind)
        adapter = factory(config, self._secrets)
        _require_methods(kind, adapter, f"the factory for {kind!r}")
        created_kind = getattr(adapter, "kind", None)
        if created_kind != kind:
            raise AdapterRegistrationError(
                f"adapter registered as {kind!r} reports kind {created_kind!r}; "
                "the registry key, the adapter's kind and SourceConfig.adapter must agree"
            )
        return adapter
