"""Loaders that put the synthetic world into a real engine for integration tests.

One module per engine (``postgres`` today; Oracle, SQL Server and Elasticsearch are separate
tasks). Each exposes ``load(...)`` returning row counts per table, plus a ``python -m`` entry
point. The world itself is engine-agnostic: see :mod:`mdqm.testing.world`.
"""

from __future__ import annotations

__all__ = ["postgres"]
