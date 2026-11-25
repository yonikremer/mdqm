"""``mdqm.adapters``: the plugin seam between core and each data source.

Import the protocol and the registry from here; vendor code lives in separate distributions that
declare an entry point in the group ``mdqm.adapters``.
"""

from __future__ import annotations

from mdqm.adapters.base import (
    OPTIONAL_METHODS,
    REQUIRED_METHODS,
    AdapterFactory,
    ArrowStream,
    NotSupported,
    SourceAdapter,
    supports_method,
)
from mdqm.adapters.registry import (
    ADAPTER_ENTRY_POINT_GROUP,
    AdapterEntryPoint,
    AdapterError,
    AdapterNotRegistered,
    AdapterRegistrationError,
    AdapterRegistry,
)

__all__ = [
    "ADAPTER_ENTRY_POINT_GROUP",
    "OPTIONAL_METHODS",
    "REQUIRED_METHODS",
    "AdapterEntryPoint",
    "AdapterError",
    "AdapterFactory",
    "AdapterNotRegistered",
    "AdapterRegistrationError",
    "AdapterRegistry",
    "ArrowStream",
    "NotSupported",
    "SourceAdapter",
    "supports_method",
]
