"""Segment cache: partition-aligned pieces of a result, stored in DuckDB (task C14).

The unit of caching is *not* a query and not an arbitrary range — it is a **segment**: one piece of
one fragment's result that covers exactly one partition (or the part of one the request asked for).
A request is answered by slicing the segments already held and fetching only what is missing, so
asking for a wider window reuses the months inside it (``docs/DESIGN.md`` section 8).

Three rules make an answer honest rather than merely fast:

* **Every segment carries its own** ``fetched_at``. An answer stitched from three months may have
  one month from last week and two from ten seconds ago; each piece reports which, so mixed
  freshness is visible instead of averaged into a lie (principle 10).
* **The tail is not trusted.** A segment that ends in the future (the current, still-filling
  partition) can gain rows at any moment, so it gets the floor TTL (:data:`DEFAULT_MIN_TTL`), not
  the ratio TTL its span would earn — a month-wide window would otherwise be trusted for hours.
  When it does go stale, an append table with a unique key is refreshed by fetching only
  ``insertion_time >= fetched_at`` and merging by key; everything else refetches the segment.
* **Some tables are never cached.** ``mutable_critical`` and ``unknown`` (which behaves the same
  until measured) never populate the store, so a row that changes under us can never be served
  from it.

Storage is DuckDB because the federated executor already speaks it: rows live in one table per
segment (real columns, real types) and the metadata lives in ``_mdqm_segments``. Only built-in
DuckDB features are used — no ``INSTALL``/``LOAD``, because an extension needs the internet and the
target network has none (principle 7). ``memory_limit`` and a spill directory are set at open, so a
large cache degrades to disk instead of to an out-of-memory kill.

Purity: the clock is injected (``Clock`` from :mod:`mdqm.cache.freshness`), the freshness policy is
that module's rule rather than a second copy, and the segment windows are partition boundaries from
:func:`mdqm.planner.partitions.align_to_partitions`. Nothing here reads the wall clock or sleeps.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import duckdb
import pyarrow as pa

from mdqm.cache.freshness import (
    CACHEABLE_MUTABILITY,
    DEFAULT_MAX_TTL,
    DEFAULT_MIN_TTL,
    DEFAULT_THETA,
    MUTABILITY_CLASSES,
    Clock,
    Mutability,
    SystemClock,
    is_fresh,
    ttl_for,
)
from mdqm.config.settings import Settings
from mdqm.models import PartitionGranularity
from mdqm.planner.partitions import Range, align_to_partitions
from mdqm.sql.execution import local_ident, local_idents, run_local, set_local

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MEMORY_LIMIT",
    "EvictionReport",
    "SegmentCache",
    "SegmentLookup",
    "SegmentPart",
    "SegmentSpec",
    "SegmentStatus",
    "TailFetch",
    "TailPolicy",
    "segment_key",
]

#: Where DuckDB spills a cache that outgrows memory. Its own spilling: no extension, no network.
DEFAULT_MEMORY_LIMIT: Final = "512MB"

#: How much cached row data may occupy before the oldest segments are dropped.
DEFAULT_MAX_BYTES: Final = 1 << 30

#: Metadata table. One row per segment; ``row_table`` names the segment's own DuckDB table.
_METADATA_DDL: Final = """
CREATE TABLE IF NOT EXISTS _mdqm_segments (
    key VARCHAR PRIMARY KEY,
    source VARCHAR NOT NULL,
    table_name VARCHAR NOT NULL,
    fragment_hash VARCHAR NOT NULL,
    version_hash VARCHAR NOT NULL,
    window_from TIMESTAMPTZ NOT NULL,
    window_to TIMESTAMPTZ NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL,
    nrows BIGINT NOT NULL,
    nbytes BIGINT NOT NULL,
    row_table VARCHAR NOT NULL
)
"""

#: A DuckDB size setting comes from config, so it is checked before it is inlined into a ``SET``.
_SIZE_PATTERN: Final = re.compile(r"[0-9]+(\.[0-9]+)?[KMGT]?B")

#: Separator for the key digest. ``\x1f`` (unit separator) cannot appear in an identifier or hash.
_SEPARATOR: Final = "\x1f"


class SegmentStatus(StrEnum):
    """What the store said about one segment before any fetcher ran."""

    HIT = "hit"
    """Fresh: served without touching the source."""
    STALE = "stale"
    """Held, but past its TTL: refreshed (a tail refresh when the table allows one)."""
    MISS = "miss"
    """Not held, not cacheable, or held for another semantic version."""


@dataclass(frozen=True, slots=True)
class SegmentLookup:
    """The verdict on one segment, with the provenance needed to explain it.

    ``fetched_at``, ``age`` and ``ttl`` are ``None`` only for a segment that was never stored.
    ``reason`` is short, agent-facing text: why a hit was not possible.
    """

    window: Range
    status: SegmentStatus
    fetched_at: datetime | None = None
    age: timedelta | None = None
    ttl: timedelta | None = None
    rows: int = 0
    reason: str = ""


@dataclass(frozen=True, slots=True)
class TailPolicy:
    """How to refresh a segment by fetching only what arrived after ``fetched_at``.

    Present only for append tables that have a unique key: without a key the new rows cannot be
    merged into the old ones without duplicating or losing them, so the segment is refetched whole.

    ``ingestion_column`` is when the row *landed* (not indexed, but the right question for "what is
    new"). ``event_column``/``max_lag`` give the derived bound on the indexed event column so the
    refresh still prunes: ``insertion_time >= X`` implies ``start_time >= X - max_lag``
    (:mod:`mdqm.planner.time` proves that bound loses no row).

    shortcut: a tail refresh only sees rows whose ingestion lag is within ``max_lag``; a correction
    inserted months after its event falls outside the derived bound, so an ``append_rarely`` table
    still needs the nightly checksum (DESIGN section 8). Widen ``max_lag`` to trade pruning for
    coverage, or drop the tail policy to refetch the whole segment.
    """

    ingestion_column: str
    unique_key: tuple[str, ...]
    event_column: str | None = None
    max_lag: timedelta | None = None

    def __post_init__(self) -> None:
        """Reject a policy that could not refresh safely: all identifiers are plain columns."""
        _require_identifier(self.ingestion_column, "TailPolicy.ingestion_column")
        if not self.unique_key:
            raise ValueError("TailPolicy.unique_key must name at least one column")
        for name in self.unique_key:
            _require_identifier(name, "TailPolicy.unique_key entry")
        if self.event_column is not None:
            _require_identifier(self.event_column, "TailPolicy.event_column")
        if self.max_lag is not None and self.max_lag < timedelta(0):
            raise ValueError(f"TailPolicy.max_lag must not be negative, got {self.max_lag}")


@dataclass(frozen=True, slots=True)
class TailFetch:
    """What a tail refresh asks the source for: the rows that arrived at or after ``since``.

    Handed to the caller's ``fetch_tail`` so the SQL is built by the compiler, not here. ``since``
    is our ``fetched_at``; re-fetching a row we already hold is harmless because the merge dedupes
    by key. ``event_bound`` is ``since - max_lag`` on the indexed event column, or ``None`` when no
    lag was declared (the caller then filters ingestion time only).
    """

    since: datetime
    event_bound: datetime | None = None
    ingestion_column: str = ""
    event_column: str | None = None


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    """The identity of one fragment's cache family, plus the policy that applies to it.

    ``fragment_hash`` must be the fragment *without* its time range: that is what lets two requests
    with different windows share the segments they overlap in. ``version_hash`` is the semantic
    layer version the result was compiled under, so a semantic change misses instead of serving an
    answer computed by the old rules.
    """

    source: str
    table: str
    fragment_hash: str
    version_hash: str
    mutability: Mutability = "append_only"
    granularity: PartitionGranularity = PartitionGranularity.MONTH
    tail: TailPolicy | None = None

    def __post_init__(self) -> None:
        """Reject a spec whose key could collide, or whose mutability is not a known class."""
        for name, value in (
            ("source", self.source),
            ("table", self.table),
            ("fragment_hash", self.fragment_hash),
            ("version_hash", self.version_hash),
        ):
            if not value:
                raise ValueError(f"SegmentSpec.{name} must be non-empty")
            if _SEPARATOR in value:
                raise ValueError(f"SegmentSpec.{name} must not contain {_SEPARATOR!r}")
        if self.mutability not in MUTABILITY_CLASSES:
            raise ValueError(
                f"unknown mutability class {self.mutability!r}: "
                f"use one of {', '.join(MUTABILITY_CLASSES)}"
            )

    @property
    def cacheable(self) -> bool:
        """True when this table's mutability class allows a segment to be stored."""
        return self.mutability in CACHEABLE_MUTABILITY


@dataclass(frozen=True, slots=True)
class SegmentPart:
    """One served piece of an answer, with its own provenance.

    ``status`` is what the store said *before* this call: ``HIT`` means no source call happened for
    this piece, ``STALE``/``MISS`` mean ``fetched`` is True. A tail refresh reports
    ``tail_refreshed=True`` and how many rows actually arrived, so the caller can say "today was
    rechecked and gained 3 rows" rather than implying a full refetch or hiding the call.
    """

    window: Range
    table: pa.Table
    fetched_at: datetime
    status: SegmentStatus
    fetched: bool
    tail_refreshed: bool = False
    rows_fetched: int = 0
    duplicates_dropped: int = 0

    @property
    def rows(self) -> int:
        """Rows this piece contributes to the answer."""
        # pyarrow ships no py.typed marker, so num_rows crosses as Any.
        return _as_int(self.table.num_rows, "a segment's row count")

    # The rows keep the order DuckDB produced when merging, which is not the source's ORDER BY
    # order. A caller that promises an order must sort the assembled answer itself.


@dataclass(frozen=True, slots=True)
class EvictionReport:
    """What eviction removed and freed. Reported, never silent (principle 10)."""

    expired: int = 0
    over_size: int = 0
    bytes_reclaimed: int = 0

    @property
    def removed(self) -> int:
        """Segments dropped in total."""
        return self.expired + self.over_size


def segment_key(
    *,
    source: str,
    table: str,
    fragment_hash: str,
    version_hash: str,
    window: Range,
) -> str:
    """The identity of one segment: everything that can change the rows it holds.

    The window is normalised to UTC before hashing, so the same instant written two ways is one
    segment. ``Range`` already refuses a naive or inverted window, so a key cannot be built from a
    bug.
    """
    parts = (
        source,
        table,
        fragment_hash,
        version_hash,
        window.lower.astimezone(UTC).isoformat(),
        window.upper.astimezone(UTC).isoformat(),
    )
    if any(_SEPARATOR in part for part in parts):
        raise ValueError(f"segment key fields must not contain {_SEPARATOR!r}")
    return hashlib.sha256(_SEPARATOR.join(parts).encode("utf-8")).hexdigest()


def _row_table_name(key: str) -> str:
    """The DuckDB table holding one segment's rows: a digest, so it is always a safe identifier."""
    return f"seg_{key[:32]}"


def _require_identifier(name: str, what: str) -> None:
    """Reject anything that is not a plain column name before it reaches an identifier slot."""
    if not name or not name.replace("_", "").isalnum() or name[0].isdigit():
        raise ValueError(f"{what} must be a plain column name, got {name!r}")


def _require_size(value: str, what: str) -> None:
    """Reject a DuckDB size setting that is not a plain ``<number><unit>`` before it is inlined."""
    if not _SIZE_PATTERN.fullmatch(value):
        raise ValueError(f"{what} must look like '512MB' or '4GB', got {value!r}")


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware (UTC internally), got naive {value!r}")


def _require_aligned(window: Range, granularity: PartitionGranularity) -> None:
    """Refuse a segment whose window straddles a partition boundary.

    A segment spanning two partitions cannot be judged fresh against either of them: one half may be
    sealed and the other still filling. So the alignment the planner produced is checked here rather
    than trusted, and the error says which call fixes it.
    """
    if window.is_empty:
        raise ValueError(
            f"segment window {window.lower.isoformat()}..{window.upper.isoformat()} is empty"
        )
    pieces = align_to_partitions(window, granularity)
    if len(pieces) != 1 or pieces[0] != window:
        raise ValueError(
            f"segment window {window.lower.isoformat()}..{window.upper.isoformat()} straddles a "
            f"{granularity.value} boundary; split it with align_to_partitions first"
        )


def _is_tail(window: Range, exec_ts: datetime, granularity: PartitionGranularity) -> bool:
    """True when ``window`` covers the partition ``exec_ts`` sits in, so it can still gain rows.

    Derived from the planner's own boundaries rather than a second copy of the calendar arithmetic:
    the last piece of ``[window.lower, exec_ts)`` starts at the partition ``exec_ts`` is in, and a
    window ending after that start is the current one.
    """
    if exec_ts <= window.lower:
        return True
    last = align_to_partitions(Range(window.lower, exec_ts), granularity)[-1]
    return window.upper > last.lower


def _as_int(value: Any, what: str) -> int:
    """A DuckDB count/sum as ``int``. A non-numeric value means corrupt metadata, so it is raised.

    ``int()`` is wrapped rather than trusted: the value crosses from DuckDB untyped, and a silently
    coerced ``0`` would understate a row count or a byte total.
    """
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{what} returned a non-numeric value {value!r}") from exc


def _human(value: timedelta) -> str:
    """A short, readable duration for a ``reason``: ``'7.2h'``, ``'30s'``, ``'3.5d'``."""
    seconds = value.total_seconds()
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


class SegmentCache:
    """A partition-aligned segment store over DuckDB (see the module docstring).

    One instance owns one DuckDB database: ``:memory:`` in tests, a file in production. Every public
    method takes the lock because the server calls the cache from a bounded thread pool and a
    :class:`duckdb.DuckDBPyConnection` is not safe for concurrent use.
    """

    def __init__(
        self,
        *,
        path: Path | None = None,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        spill_dir: Path | None = None,
        clock: Clock | None = None,
        theta: float = DEFAULT_THETA,
        min_ttl: timedelta = DEFAULT_MIN_TTL,
        max_ttl: timedelta = DEFAULT_MAX_TTL,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
    ) -> None:
        """Open the store.

        Args:
            path: the DuckDB file, or ``None`` for in-memory. Its parent directory is created.
            memory_limit: DuckDB's memory ceiling; a bigger cache spills instead of dying.
            spill_dir: DuckDB's temp directory for that spill. ``None`` means
                ``<paths.cache_dir>/spill`` from the settings, never DuckDB's default: that is
                ``.tmp`` relative to the process CWD, which is read-only in the image.
            clock: time source, injected so tests pin "the tail moved" instead of sleeping.
            theta: the ratio in the freshness rule (fraction of the window a TTL may be).
            min_ttl: the floor TTL, which is also what a still-filling segment gets.
            max_ttl: the ceiling TTL, which is also what a sealed segment gets.
            max_bytes: row bytes the store may hold before evicting the oldest segments.
        """
        _require_size(memory_limit, "memory_limit")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError(f"max_bytes must not be negative, got {max_bytes}")
        self._clock: Clock = clock or SystemClock()
        self._theta = theta
        self._min_ttl = min_ttl
        self._max_ttl = max_ttl
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._closed = False

        self._con: duckdb.DuckDBPyConnection
        if path is None:
            self._con = duckdb.connect(":memory:")
        else:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            self._con = duckdb.connect(str(target))
        # UTC everywhere: a stored segment's fetched_at must round-trip to the same instant.
        # `set_local` is the one place a setting reaches SQL text: it validates the name against a
        # fixed table and the value against that setting's pattern, because a driver cannot bind a
        # SET. `_require_size` has already restricted the size to <number><unit>.
        set_local(self._con, "TimeZone", "UTC")
        set_local(self._con, "memory_limit", memory_limit)
        # DuckDB's default temp_directory is `.tmp` relative to the process CWD (/app in the image,
        # read-only there), and TMPDIR does not move it, so this is always set explicitly.
        spill = Path(spill_dir) if spill_dir is not None else Settings().paths.spill_dir
        spill.mkdir(parents=True, exist_ok=True)
        set_local(self._con, "temp_directory", spill.as_posix())
        run_local(self._con, _METADATA_DDL)

    # --- lifecycle ------------------------------------------------------------------------

    def close(self) -> None:
        """Close the DuckDB database. Idempotent."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._con.close()

    def __enter__(self) -> SegmentCache:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    # --- lookup ---------------------------------------------------------------------------

    def lookup(self, spec: SegmentSpec, window: Range, *, exec_ts: datetime) -> SegmentLookup:
        """Decide hit / stale / miss for one segment, without any source call.

        ``exec_ts`` is the one execution timestamp: it decides whether the segment is still filling
        (and so gets only the floor TTL) or sealed (and so may live until ``max_ttl``).
        """
        _require_aware(exec_ts, "exec_ts")
        _require_aligned(window, spec.granularity)
        if not spec.cacheable:
            return SegmentLookup(
                window=window,
                status=SegmentStatus.MISS,
                reason=f"'{spec.mutability}' tables are never cached; the segment is refetched",
            )
        key = self._key(spec, window)
        with self._lock:
            row = self._metadata(key)
        if row is None:
            return SegmentLookup(
                window=window,
                status=SegmentStatus.MISS,
                reason="no segment is stored for this window and semantic version",
            )
        fetched_at, rows = row
        now = self._clock.now()
        _require_aware(now, "clock.now()")
        tail = _is_tail(window, exec_ts, spec.granularity)
        span = Range(window.lower, window.upper) if not tail else Range(window.lower, window.lower)
        effective_ttl = ttl_for(
            span.length, theta=self._theta, min_ttl=self._min_ttl, max_ttl=self._max_ttl
        )
        if fetched_at > now:
            return SegmentLookup(
                window=window,
                status=SegmentStatus.STALE,
                fetched_at=fetched_at,
                age=timedelta(0),
                ttl=effective_ttl,
                rows=rows,
                reason="fetched_at is in the future: clock skew or corrupt metadata",
            )
        age = now - fetched_at
        # The tail is judged against a zero-length span, so the ratio rule degenerates to min_ttl:
        # a still-filling partition is never trusted for more than the floor.
        fresh = is_fresh(
            age,
            span.lower,
            span.upper,
            theta=self._theta,
            min_ttl=self._min_ttl,
            max_ttl=self._max_ttl,
            mutability=spec.mutability,
            sealed=not tail,
        )
        return SegmentLookup(
            window=window,
            status=SegmentStatus.HIT if fresh else SegmentStatus.STALE,
            fetched_at=fetched_at,
            age=age,
            ttl=effective_ttl,
            rows=rows,
            reason=(
                ""
                if fresh
                else (
                    f"age {_human(age)} is past its {'tail' if tail else 'sealed'} ttl "
                    f"{_human(effective_ttl)}"
                )
            ),
        )

    # --- fetch ----------------------------------------------------------------------------

    def fetch(
        self,
        spec: SegmentSpec,
        windows: Sequence[Range],
        *,
        exec_ts: datetime,
        fetch_window: Callable[[Range], pa.Table],
        fetch_tail: Callable[[TailFetch], pa.Table] | None = None,
    ) -> tuple[SegmentPart, ...]:
        """Answer ``windows`` from cache where possible, fetching only what is missing.

        ``fetch_window`` fetches exactly one window (the caller builds the SQL); ``fetch_tail``
        fetches the rows that arrived since a given instant, for a tail refresh. Both are the
        caller's, so this module never builds SQL and every source call is attributable.

        Returns one :class:`SegmentPart` per window, in request order, each with its own
        ``fetched_at`` so a mixed-freshness answer shows which piece came from where.
        """
        _require_aware(exec_ts, "exec_ts")
        return tuple(
            self._resolve(
                spec,
                window,
                exec_ts=exec_ts,
                fetch_window=fetch_window,
                fetch_tail=fetch_tail,
            )
            for window in windows
        )

    def _resolve(
        self,
        spec: SegmentSpec,
        window: Range,
        *,
        exec_ts: datetime,
        fetch_window: Callable[[Range], pa.Table],
        fetch_tail: Callable[[TailFetch], pa.Table] | None,
    ) -> SegmentPart:
        """Serve one window from cache, by tail refresh, or by a full fetch."""
        look = self.lookup(spec, window, exec_ts=exec_ts)
        if look.status is SegmentStatus.HIT and look.fetched_at is not None:
            return SegmentPart(
                window=window,
                table=self._read_rows(spec, window),
                fetched_at=look.fetched_at,
                status=SegmentStatus.HIT,
                fetched=False,
            )
        if fetch_tail is not None and spec.tail is not None and look.fetched_at is not None:
            return self._tail_refresh(spec, window, look, fetch_tail)
        table = fetch_window(window)
        fetched_at = self._clock.now()
        if spec.cacheable:
            self._write(spec, window, table, fetched_at)
        return SegmentPart(
            window=window,
            table=table,
            fetched_at=fetched_at,
            status=look.status,
            fetched=True,
            rows_fetched=table.num_rows,
        )

    def _tail_refresh(
        self,
        spec: SegmentSpec,
        window: Range,
        look: SegmentLookup,
        fetch_tail: Callable[[TailFetch], pa.Table],
    ) -> SegmentPart:
        """Fetch only what arrived since ``fetched_at`` and merge it into the held segment.

        The merge dedupes by the declared unique key, keeping the newly fetched row, so a row that
        was inserted twice (a backdated correction, or a row that arrived at exactly ``fetched_at``
        and was already held) cannot appear twice. Duplicates dropped are counted, not hidden.
        """
        policy = spec.tail
        since = look.fetched_at
        if policy is None or since is None:  # pragma: no cover - guarded by _resolve
            raise RuntimeError("tail refresh requested without a policy or a stored segment")
        request = TailFetch(
            since=since,
            event_bound=None if policy.max_lag is None else since - policy.max_lag,
            ingestion_column=policy.ingestion_column,
            event_column=policy.event_column,
        )
        diff = fetch_tail(request)
        merged = self._merge(spec, window, diff)
        fetched_at = self._clock.now()
        self._write(spec, window, merged, fetched_at)
        dropped = look.rows + diff.num_rows - merged.num_rows
        return SegmentPart(
            window=window,
            table=merged,
            fetched_at=fetched_at,
            status=look.status,
            fetched=True,
            tail_refreshed=True,
            rows_fetched=diff.num_rows,
            duplicates_dropped=max(dropped, 0),
        )

    # --- rows -----------------------------------------------------------------------------

    def _merge(self, spec: SegmentSpec, window: Range, diff: pa.Table) -> pa.Table:
        """``held segment + new rows``, deduped by the unique key with the new row winning."""
        policy = spec.tail
        if policy is None:  # pragma: no cover - only reached from _tail_refresh
            raise RuntimeError("merge requested for a spec without a TailPolicy")
        if diff.num_rows == 0:
            # Nothing arrived. A source that returns no rows may hand back a table without any
            # columns, which DuckDB refuses to register, so the held rows are kept as they are.
            return self._read_rows(spec, window)
        ident = _row_table_name(self._key(spec, window))
        keys = local_idents(policy.unique_key)
        with self._lock:
            self._con.register("_mdqm_diff", diff)
            try:
                # _seq is the tie-break: fetched rows (1) win over held rows (0), which is exactly
                # what "the same key was re-inserted" means. It is a constant, so the result is
                # deterministic. BY NAME aligns the two sides when their projection order differs.
                # ident is 'seg_' + 32 hex digits of a sha256 and each key is a validated plain
                # column name, so the only things interpolated into this DML are safe identifiers
                # (DuckDB cannot bind an identifier as a parameter).
                # `ident` and every column in `keys` go through `local_ident`, which validates the
                # shape and quotes it: DuckDB cannot bind an identifier, so this is the one place the
                # name becomes text. Values are never interpolated here.
                run_local(
                    self._con,
                    f"CREATE OR REPLACE TABLE main.{local_ident(ident)} AS "  # noqa: S608
                    "SELECT * EXCLUDE (_seq, _rn) FROM ("
                    "SELECT *, ROW_NUMBER() OVER "
                    f"(PARTITION BY {keys} ORDER BY _seq DESC) AS _rn FROM ("
                    f"SELECT *, 0 AS _seq FROM main.{local_ident(ident)} "
                    "UNION ALL BY NAME "
                    "SELECT *, 1 AS _seq FROM _mdqm_diff"
                    ")) WHERE _rn = 1",
                )
                return self._con.sql(
                    f'SELECT * FROM main."{ident}"'  # noqa: S608
                ).to_arrow_table()
            finally:
                self._con.unregister("_mdqm_diff")

    def _write(
        self, spec: SegmentSpec, window: Range, table: pa.Table, fetched_at: datetime
    ) -> None:
        """Replace one segment's rows and metadata. The row table is keyed by the segment key."""
        key = self._key(spec, window)
        ident = _row_table_name(key)
        with self._lock:
            self._con.register("_mdqm_seg_load", table)
            try:
                run_local(
                    self._con,
                    f"CREATE OR REPLACE TABLE main.{local_ident(ident)} "  # noqa: S608
                    "AS SELECT * FROM _mdqm_seg_load",
                )
            finally:
                self._con.unregister("_mdqm_seg_load")
            self._con.execute(
                "INSERT OR REPLACE INTO _mdqm_segments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    key,
                    spec.source,
                    spec.table,
                    spec.fragment_hash,
                    spec.version_hash,
                    window.lower,
                    window.upper,
                    fetched_at,
                    table.num_rows,
                    table.nbytes,
                    ident,
                ],
            )

    def _key(self, spec: SegmentSpec, window: Range) -> str:
        """The segment key for ``spec`` and ``window`` (see :func:`segment_key`)."""
        return segment_key(
            source=spec.source,
            table=spec.table,
            fragment_hash=spec.fragment_hash,
            version_hash=spec.version_hash,
            window=window,
        )

    def _metadata(self, key: str) -> tuple[datetime, int] | None:
        """``(fetched_at, rows)`` for ``key``, or ``None`` when the segment is not stored."""
        # fetchall(), not fetchone(): a partially read result keeps the connection busy.
        rows = self._con.execute(
            "SELECT fetched_at, nrows FROM _mdqm_segments WHERE key = ?", [key]
        ).fetchall()
        return None if not rows else (rows[0][0], _as_int(rows[0][1], "a segment's row count"))

    def _read_rows(self, spec: SegmentSpec, window: Range) -> pa.Table:
        """Read a stored segment's rows back as Arrow, preserving the original column order."""
        ident = _row_table_name(self._key(spec, window))
        with self._lock:
            return self._con.sql(
                f'SELECT * FROM main."{ident}"'  # noqa: S608
            ).to_arrow_table()

    # --- eviction -------------------------------------------------------------------------

    def evict(self, *, now: datetime | None = None, max_bytes: int | None = None) -> EvictionReport:
        """Drop segments past ``max_ttl``, then the oldest until the store fits ``max_bytes``.

        Sealed segments live until ``max_ttl`` because a closed partition cannot change, which is
        the same ceiling the freshness rule applies: a segment is never evicted while it could
        still be served. Age is measured from ``fetched_at`` with the injected clock.
        """
        current = now if now is not None else self._clock.now()
        _require_aware(current, "now")
        limit = self._max_bytes if max_bytes is None else max_bytes
        with self._lock:
            rows = self._con.execute(
                "SELECT key, row_table, nbytes, fetched_at FROM _mdqm_segments ORDER BY fetched_at"
            ).fetchall()
        expired = [row for row in rows if current - row[3] > self._max_ttl]
        survivors = [row for row in rows if current - row[3] <= self._max_ttl]
        over: list[tuple[str, str, int, datetime]] = []
        if limit is not None:
            total = sum(row[2] for row in survivors)
            index = 0
            while total > limit and index < len(survivors):
                over.append(survivors[index])
                total -= survivors[index][2]
                index += 1
        self._drop(expired + over)
        return EvictionReport(
            expired=len(expired),
            over_size=len(over),
            bytes_reclaimed=sum(row[2] for row in expired + over),
        )

    def evict_version(self, *, source: str, table: str, version_hash: str) -> int:
        """Drop every segment of ``source.table`` compiled under another semantic version.

        Called when the semantic layer changes: the old segments can never be served again (their
        key carries the old version), so they are space, not cache. Returns how many were dropped.
        """
        with self._lock:
            rows = self._con.execute(
                "SELECT key, row_table FROM _mdqm_segments "
                "WHERE source = ? AND table_name = ? AND version_hash <> ?",
                [source, table, version_hash],
            ).fetchall()
        self._drop([(key, ident, 0, self._clock.now()) for key, ident in rows])
        return len(rows)

    def _drop(self, rows: Sequence[tuple[str, str, int, datetime]]) -> None:
        """Delete the segment row tables and their metadata, under one lock."""
        if not rows:
            return
        with self._lock:
            for key, ident, _bytes, _fetched in rows:
                # ident is 'seg_' + 32 hex digits of a sha256 (see _row_table_name).
                run_local(self._con, f"DROP TABLE IF EXISTS main.{local_ident(ident)}")
                self._con.execute("DELETE FROM _mdqm_segments WHERE key = ?", [key])

    # --- reporting ------------------------------------------------------------------------

    def segments(self) -> int:
        """How many segments are stored right now."""
        with self._lock:
            rows = self._con.execute("SELECT count(*) FROM _mdqm_segments").fetchall()
        return _as_int(rows[0][0], "the segment count") if rows else 0

    def total_bytes(self) -> int:
        """Row bytes the store currently holds."""
        with self._lock:
            rows = self._con.execute(
                "SELECT coalesce(sum(nbytes), 0) FROM _mdqm_segments"
            ).fetchall()
        return _as_int(rows[0][0], "the cached byte total") if rows else 0
