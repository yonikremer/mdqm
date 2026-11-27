"""Loaders that put the synthetic world into a real engine for integration tests.

One module per engine (``postgres`` and ``elastic`` today; Oracle and SQL Server are separate
tasks). Each exposes ``load(...)`` returning row counts per table (or per index, for
Elasticsearch), plus a ``python -m`` entry point. The world itself is engine-agnostic: see
:mod:`mdqm.testing.world`.
"""

from __future__ import annotations

__all__ = ["elastic", "postgres"]
