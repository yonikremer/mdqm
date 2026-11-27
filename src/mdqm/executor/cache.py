"""Whole-fragment result cache for the federated executor (task C17).

A **fragment** is cached, not a query and not a partition window: the key is the fragment's own SQL
plus its source and the semantic version it was compiled under, so an identical identical call
reuses it and a different window is a different entry. That is the smallest unit that makes
"a second identical call touches no source" true, and it is deliberately *not* a second copy of
:class:`~mdqm.cache.segments.SegmentCache`: C14 owns partition-aligned segments (a window split
along partition boundaries, a tail refresh, mixed freshness per piece), and re-implementing that
here would give the project two disagreeing cache rules.

What *is* shared with the rest of the cache lane is the rule and the clock:

* freshness comes from :func:`~mdqm.cache.freshness.is_fresh_at`, unmodified. A fragment has no
  window, so it gets the floor TTL (``min_ttl``) rather than a ratio TTL — the conservative
  reading, because a whole-table read has no span to be fresh against;
* ``mutable_critical`` and ``unknown`` are never served, exactly as for segments;
* the clock is injected, so a test drives staleness instead of sleeping.

Storage is the executor's own DuckDB connection: rows keep their real Arrow types and the metadata
lives in ``_mdqm_fragment_cache``. Built-in DuckDB only — no ``INSTALL``/``LOAD`` (principle 7).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, cast

import duckdb
import pyarrow as pa

from mdqm.cache.freshness import (
    DEFAULT_MAX_TTL,
    DEFAULT_MIN_TTL,
    Clock,
    Mutability,
    SystemClock,
    is_fresh_at,
)
from mdqm.sql.execution import run_local

__all__ = ["FragmentCache", "FragmentLookup", "fragment_key"]

#: Metadata table: one row per cached fragment. ``row_table`` names the fragment's own table.
_METADATA_DDL: Final = """
CREATE TABLE IF NOT EXISTS _mdqm_fragment_cache (
    key VARCHAR PRIMARY KEY,
    source VARCHAR NOT NULL,
    table_name VARCHAR NOT NULL,
    version_hash VARCHAR NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL,
    nrows BIGINT NOT NULL,
    nbytes BIGINT NOT NULL,
    row_table VARCHAR NOT NULL
)
"""

#: Separator for the key digest; ``\x1f`` (unit separator) cannot appear in an identifier or hash.
_SEPARATOR: Final = "\x1f"


@dataclass(frozen=True, slots=True)
class FragmentLookup:
    """The verdict on one fragment, with the provenance needed to report it.

    ``table`` is set only on a hit. ``reason`` always says why, so a refetch is explainable
    ("age 4m is past its ttl 30s") instead of mysterious (principle 10).
    """

    key: str
    hit: bool
    status: str
    reason: str
    fetched_at: datetime | None = None
    age_s: float | None = None
    table: pa.Table | None = None


def fragment_key(
    *, source: str, table: str, sql: str | None, version_hash: str, extra: str = ""
) -> str:
    """The cache identity of one fragment: everything that decides which rows it returns."""
    parts = (source, table, sql or "", version_hash, extra)
    for part in parts:
        if _SEPARATOR in part:
            raise ValueError(f"a cache key part must not contain {_SEPARATOR!r}")
    return hashlib.sha256(_SEPARATOR.join(parts).encode("utf-8")).hexdigest()


class FragmentCache:
    """Fragment results in the executor's DuckDB connection, judged by C12's freshness rule.

    The connection is *passed in* rather than opened here: the executor registers the cached
    tables as its own join inputs, so one connection is the honest way to share them. The caller
    owns the connection's lifetime; closing it closes the cache with it.
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        *,
        clock: Clock | None = None,
        min_ttl: timedelta = DEFAULT_MIN_TTL,
        max_ttl: timedelta = DEFAULT_MAX_TTL,
    ) -> None:
        self._con = con
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._min_ttl = min_ttl
        self._max_ttl = max_ttl
        self._run(_METADATA_DDL)

    def lookup(self, key: str, *, mutability: Mutability, exec_ts: datetime) -> FragmentLookup:
        """Hit, stale or miss for ``key``, without touching any source.

        Args:
            key: from :func:`fragment_key`.
            mutability: the table's measured class; a non-cacheable class can never hit.
            exec_ts: the execution's single timestamp (C02), used as the fragment's span: zero,
                because a whole-fragment read covers no declared window.
        """
        row = self._con.execute(
            "SELECT fetched_at, row_table FROM _mdqm_fragment_cache WHERE key = ?", [key]
        ).fetchall()
        if not row:
            return FragmentLookup(key=key, hit=False, status="miss", reason="not cached")
        fetched_at, row_table = row[0]
        now = self._clock.now()
        age = now - fetched_at
        age_s = max(0.0, age.total_seconds())
        if not is_fresh_at(
            fetched_at,
            self._clock,
            exec_ts,
            exec_ts,
            mutability=mutability,
            min_ttl=self._min_ttl,
            max_ttl=self._max_ttl,
        ):
            ttl = min(self._min_ttl, self._max_ttl)
            if mutability not in ("append_only", "append_rarely"):
                reason = f"'{mutability}' tables are never served from cache"
            else:
                reason = f"age {age_s:.0f}s is past its ttl {ttl.total_seconds():.0f}s"
            return FragmentLookup(
                key=key,
                hit=False,
                status="stale",
                reason=reason,
                fetched_at=fetched_at,
                age_s=age_s,
            )
        # row_table is this module's own digest-derived name, never caller text.
        table = cast(pa.Table, self._con.sql(f'SELECT * FROM "{row_table}"').to_arrow_table())  # noqa: S608
        return FragmentLookup(
            key=key,
            hit=True,
            status="hit",
            reason="fresh",
            fetched_at=fetched_at,
            age_s=age_s,
            table=table,
        )

    def store(
        self,
        key: str,
        table: pa.Table,
        *,
        source: str,
        table_name: str,
        version_hash: str,
    ) -> datetime:
        """Write ``table`` under ``key`` and return the ``fetched_at`` it was stamped with.

        The stamp is the injected clock's instant, not the source's: it is "when the server learned
        these rows", which is the only thing the freshness rule can honestly compare.
        """
        fetched_at = self._clock.now()
        row_table = _row_table_name(key)
        self._con.register("_mdqm_cache_load", table)
        try:
            # row_table is derived from a hex digest, so no caller text reaches this identifier.
            self._run(
                f'CREATE OR REPLACE TABLE "{row_table}" AS SELECT * FROM _mdqm_cache_load'  # noqa: S608
            )
        finally:
            self._con.unregister("_mdqm_cache_load")
        self._run(
            "INSERT OR REPLACE INTO _mdqm_fragment_cache "
            "(key, source, table_name, version_hash, fetched_at, nrows, nbytes, row_table) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                key,
                source,
                table_name,
                version_hash,
                fetched_at,
                table.num_rows,
                table.nbytes,
                row_table,
            ],
        )
        return fetched_at

    def _run(self, sql: str, params: list[object] | None = None) -> None:
        """Run a statement and drain its result.

        A DuckDB connection refuses a new statement while an earlier result is unfinished, so every
        write here is drained rather than left to the garbage collector.
        """
        run_local(self._con, sql, params).fetchall()

    def size(self) -> int:
        """How many fragments are held. A point-in-time count, for tests and the audit view."""
        rows = self._con.sql("SELECT COUNT(*) FROM _mdqm_fragment_cache").fetchall()
        return 0 if not rows else int(rows[0][0])


def _row_table_name(key: str) -> str:
    """The DuckDB table holding one fragment's rows: derived from the digest, so it is an id."""
    return f"_mdqm_frag_{key[:16]}"
