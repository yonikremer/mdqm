"""Loaders that put the synthetic world into a real engine for integration tests.

One module per engine (``postgres`` and ``elastic`` today; Oracle and SQL Server are separate
tasks). Each exposes ``load(...)`` returning row counts per table (or per index, for
Elasticsearch), plus a ``python -m`` entry point. The world itself is engine-agnostic: see
:mod:`mdqm.testing.world`.
"""

from __future__ import annotations

import importlib
from typing import Any

#: The loaders this package provides, by module name (imported on first use, see below).
_LOADERS = ("elastic", "postgres")


def __getattr__(name: str) -> Any:
    """Import a loader on first use (PEP 562).

    Each loader needs its engine's driver, and a deployment installs only the extras it serves, so a
    package that imported all of them eagerly would fail to import for a site with no Elasticsearch.
    """
    if name in _LOADERS:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
