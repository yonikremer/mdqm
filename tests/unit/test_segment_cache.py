"""Tests for the partition-aligned segment cache (task C14).

The proof the task asks for is behavioural, not structural: what the **source** was asked to fetch.
So every test drives the cache through a real :class:`~mdqm.testing.fake_adapter.FakeAdapter` and
asserts on ``FakeAdapter.calls`` — a repeated question must add no call, a partially covered range
must add exactly one call for the missing segment, and a tail refresh must ask only for rows
inserted at or after the segment's own ``fetched_at``.

Time is injected everywhere (``_Clock``), so "the tail moved" is a two-line test instead of a sleep,
and every ``fetched_at`` assertion is exact. Nothing here touches the network or a container.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

from mdqm.cache.freshness import Mutability
from mdqm.cache.segments import (
    EvictionReport,
    SegmentCache,
    SegmentLookup,
    SegmentPart,
    SegmentSpec,
    SegmentStatus,
    TailFetch,
    TailPolicy,
    segment_key,
)
from mdqm.models import Fragment, Limits, PartitionGranularity
from mdqm.planner.partitions import Range, align_to_partitions
from mdqm.testing.fake_adapter import FakeAdapter

EXEC_TS = datetime(2026, 6, 20, tzinfo=UTC)


# --- fixtures and helpers ---------------------------------------------------------------


@dataclass
class _Clock:
    """A settable clock: the tests move time instead of sleeping."""

    moment: datetime
    reads: int = field(default=0)

    def now(self) -> datetime:
        self.reads += 1
        return self.moment

    def advance(self, delta: timedelta) -> None:
        self.moment += delta


def _ts(year: int, month: int, day: int) -> datetime:
    """Midnight UTC on a date, for terse fixtures."""
    return datetime(year, month, day, tzinfo=UTC)


def _month(year: int, month: int) -> Range:
    """A whole calendar month as a half-open window."""
    start = _ts(year, month, 1)
    end = _ts(year + 1, 1, 1) if month == 12 else _ts(year, month + 1, 1)
    return Range(start, end)


def _event(
    event_id: int,
    *,
    start: datetime,
    insertion: datetime,
    units: int,
    client_no: str = "C1",
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "client_no": client_no,
        "start_time": start,
        "insertion_time": insertion,
        "units": units,
        "corrects_event_id": None,
    }


def _table(rows: list[dict[str, object]]) -> pa.Table:
    """Build the usage-events table with explicit types, so timestamps stay timezone-aware."""
    return pa.table(
        {
            "event_id": pa.array([row["event_id"] for row in rows], pa.int64()),
            "client_no": pa.array([row["client_no"] for row in rows], pa.string()),
            "start_time": pa.array(
                [row["start_time"] for row in rows], pa.timestamp("us", tz="UTC")
            ),
            "insertion_time": pa.array(
                [row["insertion_time"] for row in rows], pa.timestamp("us", tz="UTC")
            ),
            "units": pa.array([row["units"] for row in rows], pa.int64()),
            "corrects_event_id": pa.array([row["corrects_event_id"] for row in rows], pa.int64()),
        }
    )


def _january_rows() -> list[dict[str, object]]:
    return [
        _event(1, start=_ts(2026, 1, 5), insertion=_ts(2026, 1, 5), units=10),
        _event(2, start=_ts(2026, 1, 19), insertion=_ts(2026, 1, 19), units=20),
    ]


def _february_rows() -> list[dict[str, object]]:
    return [
        _event(3, start=_ts(2026, 2, 3), insertion=_ts(2026, 2, 3), units=30),
        _event(4, start=_ts(2026, 2, 11), insertion=_ts(2026, 2, 11), units=40),
    ]


def _adapter(rows: list[dict[str, object]], *, exec_ts: datetime = EXEC_TS) -> FakeAdapter:
    return FakeAdapter(name="billing", tables={"usage_events": _table(rows)})


def _limits() -> Limits:
    return Limits(timeout_s=5.0, max_rows=10_000, max_bytes=10_000_000, max_partitions=8)


def _run(adapter: FakeAdapter, sql: str, *, exec_ts: datetime = EXEC_TS) -> pa.Table:
    """Run one fragment on the fake adapter and materialise its rows.

    ``sql`` is built by the helpers below from this module's own constants; no user input reaches
    it. The marker is for the taint scanner, which cannot see that across the call boundary.
    """
    fragment = Fragment(source="billing", dialect="duckdb", sql=sql, exec_ts=exec_ts)
    stream = adapter.execute(fragment, _limits())  # nosemgrep
    try:
        batches = list(stream)
    finally:
        stream.close()
    # Keep the stream's schema: an empty result must still have the source's columns.
    return pa.Table.from_batches(batches, schema=stream.schema)


def _litsql(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _window_fetcher(
    adapter: FakeAdapter, *, column: str = "start_time"
) -> Callable[[Range], pa.Table]:
    """A ``fetch_window`` that really queries the fake adapter and records the SQL."""

    def fetch(window: Range) -> pa.Table:
        low = _litsql(window.lower)
        high = _litsql(window.upper)
        # The column name and bounds here are this test's own literals, never user input.
        sql = (
            f'SELECT * FROM "usage_events" WHERE "{column}" >= TIMESTAMP \'{low}\' '
            f"AND \"{column}\" < TIMESTAMP '{high}' ORDER BY event_id"
        )  # nosemgrep
        return _run(adapter, sql)

    return fetch


def _tail_fetcher(adapter: FakeAdapter) -> Callable[[TailFetch], pa.Table]:
    """A ``fetch_tail`` that filters ingestion time and adds the derived event bound."""

    def fetch(request: TailFetch) -> pa.Table:
        # The column names and bounds here are this test's own values, never user input.
        sql = (
            'SELECT * FROM "usage_events" '
            f"WHERE \"insertion_time\" >= TIMESTAMP '{_litsql(request.since)}'"
        )
        if request.event_bound is not None and request.event_column is not None:
            sql += f" AND \"{request.event_column}\" >= TIMESTAMP '{_litsql(request.event_bound)}'"
        return _run(adapter, sql + " ORDER BY event_id")  # nosemgrep

    return fetch


def _spec(
    *,
    source: str = "billing",
    table: str = "usage_events",
    fragment_hash: str = "fh-annual-units",
    version_hash: str = "sem-v1",
    mutability: Mutability = "append_only",
    granularity: PartitionGranularity = PartitionGranularity.MONTH,
    tail: TailPolicy | None = None,
) -> SegmentSpec:
    """A spec with every component defaulted, so a test varies exactly the one it is about."""
    return SegmentSpec(
        source=source,
        table=table,
        fragment_hash=fragment_hash,
        version_hash=version_hash,
        mutability=mutability,
        granularity=granularity,
        tail=tail,
    )


def _tail_spec() -> SegmentSpec:
    """A spec whose table is append-only with a unique key, so a tail refresh is possible."""
    return _spec(
        tail=TailPolicy(
            ingestion_column="insertion_time",
            unique_key=("event_id",),
            event_column="start_time",
            max_lag=timedelta(minutes=5),
        )
    )


@pytest.fixture
def clock() -> _Clock:
    return _Clock(_ts(2026, 6, 20))


@pytest.fixture
def cache(clock: _Clock) -> Iterator[SegmentCache]:
    store = SegmentCache(clock=clock, min_ttl=timedelta(seconds=30))
    yield store
    store.close()


# --- acceptance: a repeated request costs nothing ----------------------------------------


def test_repeated_request_is_served_with_no_adapter_calls(
    cache: SegmentCache, clock: _Clock
) -> None:
    """The whole point: the same question twice reaches the source once."""
    adapter = _adapter(_january_rows())
    spec = _spec()
    windows = align_to_partitions(_month(2026, 1))

    first = cache.fetch(spec, windows, exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))
    calls_after_first = len(adapter.calls)
    clock.advance(timedelta(hours=1))

    second = cache.fetch(spec, windows, exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))

    assert calls_after_first == 1, "the first attempt must fetch the month once"
    assert len(adapter.calls) == calls_after_first, "a repeated request must fetch nothing"
    assert [part.status for part in first] == [SegmentStatus.MISS]
    assert [part.status for part in second] == [SegmentStatus.HIT]
    assert all(part.fetched is False for part in second)
    assert second[0].rows == 2


def test_partial_coverage_fetches_only_the_missing_segment(
    cache: SegmentCache, clock: _Clock
) -> None:
    """A wider request reuses the month it already holds and fetches only the new one."""
    adapter = _adapter(_january_rows() + _february_rows())
    spec = _spec()
    january = list(align_to_partitions(_month(2026, 1)))
    cache.fetch(spec, january, exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))
    assert len(adapter.calls) == 1

    clock.advance(timedelta(hours=1))
    wider = align_to_partitions(Range(_ts(2026, 1, 1), _ts(2026, 3, 1)))
    parts = cache.fetch(spec, wider, exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))

    assert len(adapter.calls) == 2, "only the missing month may be fetched"
    assert "2026-02-01" in adapter.calls[1], "the second fetch must be February"
    assert "2026-01-01" not in adapter.calls[1], "January must not be fetched again"
    assert [part.status for part in parts] == [SegmentStatus.HIT, SegmentStatus.MISS]
    assert [part.rows for part in parts] == [2, 2]


# --- acceptance: sealed months are hits, the tail is refreshed ----------------------------


def test_sealed_past_month_is_a_hit(cache: SegmentCache, clock: _Clock) -> None:
    """A closed partition cannot gain rows, so it is served long past the floor TTL."""
    adapter = _adapter(_january_rows())
    spec = _spec()
    windows = align_to_partitions(_month(2026, 1))
    cache.fetch(spec, windows, exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))

    clock.advance(timedelta(days=3))
    parts = cache.fetch(spec, windows, exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))

    assert len(adapter.calls) == 1
    assert parts[0].status is SegmentStatus.HIT


def test_tail_segment_is_refreshed_once_the_floor_ttl_passes(
    cache: SegmentCache, clock: _Clock
) -> None:
    """The current month is never trusted for the hours a month-wide ratio TTL would allow."""
    adapter = _adapter(_february_rows())
    spec = _tail_spec()
    window = Range(_ts(2026, 2, 1), _ts(2026, 3, 1))
    # The clock sits inside February, so exec_ts == now and February is the still-filling partition.
    clock.moment = _ts(2026, 2, 15)
    exec_ts = clock.moment
    cache.fetch(spec, [window], exec_ts=exec_ts, fetch_window=_window_fetcher(adapter))

    # One second after the fetch: still fresh, so no call.
    clock.advance(timedelta(seconds=1))
    hit = cache.fetch(
        spec,
        [window],
        exec_ts=exec_ts,
        fetch_window=_window_fetcher(adapter),
        fetch_tail=_tail_fetcher(adapter),
    )[0]
    assert hit.status is SegmentStatus.HIT
    assert len(adapter.calls) == 1

    # Past the floor TTL: the tail is rechecked, and only for rows inserted since fetched_at.
    clock.advance(timedelta(seconds=31))
    fetched_at = cache.lookup(spec, window, exec_ts=exec_ts).fetched_at
    assert fetched_at is not None
    refreshed = cache.fetch(
        spec,
        [window],
        exec_ts=exec_ts,
        fetch_window=_window_fetcher(adapter),
        fetch_tail=_tail_fetcher(adapter),
    )[0]

    assert refreshed.status is SegmentStatus.STALE
    assert refreshed.tail_refreshed is True
    assert len(adapter.calls) == 2, "the tail refresh is one call, not a full refetch"
    sql = adapter.calls[1]
    assert "insertion_time" in sql, "a tail refresh filters ingestion time"
    assert _litsql(fetched_at) in sql, "it asks for rows inserted at or after its own fetched_at"
    assert "2026-02-01 00:00:00" not in sql, "it must not refetch the whole month"
    assert _litsql(fetched_at - timedelta(minutes=5)) in sql, "the derived event bound is added"


def test_tail_refresh_merges_new_rows_and_dedupes_by_key(
    cache: SegmentCache, clock: _Clock
) -> None:
    """A re-inserted key updates the row instead of appearing twice."""
    adapter = _adapter(_february_rows())
    spec = _tail_spec()
    window = Range(_ts(2026, 2, 1), _ts(2026, 3, 1))
    # The clock sits inside February, so exec_ts == now and February is the still-filling partition.
    clock.moment = _ts(2026, 2, 15)
    exec_ts = clock.moment
    cache.fetch(spec, [window], exec_ts=exec_ts, fetch_window=_window_fetcher(adapter))

    # Two rows land one ingestion second after their event time (well inside max_lag): a re-insert
    # of key 4 with new values, and a brand new event 5.
    landed = clock.moment + timedelta(seconds=10)
    stamp = _litsql(landed)
    adapter.execute(
        Fragment(
            source="billing",
            dialect="duckdb",
            sql=(
                'INSERT INTO "usage_events" (event_id, client_no, start_time, insertion_time, '
                "units, corrects_event_id) VALUES "
                f"(4, 'C1', TIMESTAMP '{stamp}', TIMESTAMP '{stamp}', 400, NULL), "
                f"(5, 'C1', TIMESTAMP '{stamp}', TIMESTAMP '{stamp}', 50, NULL)"
            ),  # nosemgrep
            exec_ts=exec_ts,
        ),
        _limits(),
    ).close()
    adapter.calls.clear()

    clock.advance(timedelta(seconds=31))
    part = cache.fetch(
        spec,
        [window],
        exec_ts=exec_ts,
        fetch_window=_window_fetcher(adapter),
        fetch_tail=_tail_fetcher(adapter),
    )[0]

    rows = {row["event_id"]: row for row in part.table.to_pylist()}
    assert len(part.table) == 3, "events 3, 4 and 5 -- event 4 appears once"
    assert rows[4]["units"] == 400, "the newly inserted row wins over the held one"
    assert rows[5]["units"] == 50, "a new event is merged in"
    assert part.rows_fetched == 2
    assert part.duplicates_dropped == 1, "the dropped duplicate is counted, not hidden"


# --- acceptance: some tables are never cached --------------------------------------------


@pytest.mark.parametrize("mutability", ["mutable_critical", "unknown"])
def test_uncacheable_tables_never_populate_the_store(
    cache: SegmentCache, clock: _Clock, mutability: Mutability
) -> None:
    """A row that changes under us must never be served from a segment."""
    adapter = _adapter(_january_rows())
    spec = _spec(mutability=mutability)
    windows = align_to_partitions(_month(2026, 1))

    first = cache.fetch(spec, windows, exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))
    second = cache.fetch(spec, windows, exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))

    assert cache.segments() == 0, "nothing may be stored for this mutability class"
    assert len(adapter.calls) == 2, "every request is a real fetch"
    assert [part.status for part in first + second] == [SegmentStatus.MISS, SegmentStatus.MISS]
    assert "never cached" in cache.lookup(spec, windows[0], exec_ts=EXEC_TS).reason


def test_append_rarely_is_cached_like_append_only(cache: SegmentCache, clock: _Clock) -> None:
    """The cacheable classes are decided by freshness.py, not by a second list here."""
    adapter = _adapter(_january_rows())
    spec = _spec(mutability="append_rarely")
    windows = align_to_partitions(_month(2026, 1))

    cache.fetch(spec, windows, exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))
    clock.advance(timedelta(hours=1))
    parts = cache.fetch(spec, windows, exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))

    assert parts[0].status is SegmentStatus.HIT
    assert len(adapter.calls) == 1


# --- acceptance: a semantic version change invalidates -----------------------------------


def test_changing_the_version_hash_invalidates_affected_segments(
    cache: SegmentCache, clock: _Clock
) -> None:
    """A new semantic version must not be answered by a segment computed under the old one."""
    adapter = _adapter(_january_rows())
    windows = align_to_partitions(_month(2026, 1))
    cache.fetch(_spec(), windows, exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))
    assert len(adapter.calls) == 1

    clock.advance(timedelta(seconds=1))
    bumped = _spec(version_hash="sem-v2")
    parts = cache.fetch(bumped, windows, exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))

    assert parts[0].status is SegmentStatus.MISS, "the old version's segment must not answer"
    assert len(adapter.calls) == 2, "the bumped version is fetched"

    dropped = cache.evict_version(source="billing", table="usage_events", version_hash="sem-v2")
    assert dropped == 1, "the superseded segment is reclaimed"
    assert cache.segments() == 1, "only the current version remains"


# --- acceptance: every served segment reports its own fetched_at -------------------------


def test_a_mixed_answer_reports_each_segment_its_own_fetched_at(
    cache: SegmentCache, clock: _Clock
) -> None:
    """One old month plus one fresh month must not collapse into a single freshness."""
    adapter = _adapter(_january_rows() + _february_rows())
    spec = _spec()
    january, february = align_to_partitions(Range(_ts(2026, 1, 1), _ts(2026, 3, 1)))

    cache.fetch(spec, [january], exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))
    january_fetched_at = cache.lookup(spec, january, exec_ts=EXEC_TS).fetched_at
    clock.advance(timedelta(hours=2))

    parts = cache.fetch(
        spec, [january, february], exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter)
    )

    assert [part.status for part in parts] == [SegmentStatus.HIT, SegmentStatus.MISS]
    assert parts[0].fetched_at == january_fetched_at, "the hit reports when it was really fetched"
    assert parts[1].fetched_at == clock.moment, "the fetch reports its own time"
    assert parts[0].fetched_at != parts[1].fetched_at, "mixed freshness stays visible"


# --- segments must follow partitions -----------------------------------------------------


def test_a_window_straddling_a_partition_boundary_is_rejected(cache: SegmentCache) -> None:
    """A segment covering two partitions cannot be judged fresh against either of them."""
    straddling = Range(_ts(2026, 1, 15), _ts(2026, 2, 15))
    with pytest.raises(ValueError, match="straddles a month boundary"):
        cache.lookup(_spec(), straddling, exec_ts=EXEC_TS)


def test_an_unaligned_segment_cannot_be_written(cache: SegmentCache) -> None:
    adapter = _adapter(_january_rows())
    straddling = Range(_ts(2026, 1, 15), _ts(2026, 2, 15))
    with pytest.raises(ValueError, match="align_to_partitions"):
        cache.fetch(_spec(), [straddling], exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))
    assert adapter.calls == [], "a rejected segment must not reach the source"


def test_a_day_granularity_table_aligns_to_days(cache: SegmentCache, clock: _Clock) -> None:
    """The alignment uses the table's own granularity, not a hard-coded month."""
    adapter = _adapter(_january_rows())
    spec = _spec(granularity=PartitionGranularity.DAY)
    day = align_to_partitions(Range(_ts(2026, 1, 5), _ts(2026, 1, 6)), PartitionGranularity.DAY)
    parts = cache.fetch(spec, day, exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))
    assert parts[0].rows == 1


# --- eviction ----------------------------------------------------------------------------


def test_eviction_drops_segments_past_max_ttl(cache: SegmentCache, clock: _Clock) -> None:
    adapter = _adapter(_january_rows() + _february_rows())
    spec = _spec()
    january, february = align_to_partitions(Range(_ts(2026, 1, 1), _ts(2026, 3, 1)))
    cache.fetch(spec, [january], exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))
    clock.advance(timedelta(days=8))
    cache.fetch(spec, [february], exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))
    assert cache.segments() == 2

    report = cache.evict()

    assert isinstance(report, EvictionReport)
    assert report.expired == 1, "January is past max_ttl; February is not"
    assert cache.segments() == 1


def test_eviction_trims_the_oldest_segments_to_fit_the_byte_budget(
    cache: SegmentCache, clock: _Clock
) -> None:
    adapter = _adapter(_january_rows() + _february_rows())
    spec = _spec()
    january, february = align_to_partitions(Range(_ts(2026, 1, 1), _ts(2026, 3, 1)))
    cache.fetch(spec, [january], exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))
    clock.advance(timedelta(hours=1))
    cache.fetch(spec, [february], exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))
    held = cache.total_bytes()
    assert held > 0

    report = cache.evict(max_bytes=held - 1)

    assert report.over_size >= 1
    assert report.bytes_reclaimed > 0
    assert cache.total_bytes() <= held - 1


# --- durability and settings --------------------------------------------------------------


def test_a_file_backed_store_survives_a_reopen(tmp_path: Path) -> None:
    """Production is file-backed, so a restart must still find the segments it paid for."""
    path = tmp_path / "cache" / "segments.duckdb"
    clock = _Clock(EXEC_TS)
    adapter = _adapter(_january_rows())
    spec = _spec()
    windows = align_to_partitions(_month(2026, 1))

    with SegmentCache(path=path, clock=clock) as first:
        first.fetch(spec, windows, exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter))
        fetched_at = first.lookup(spec, windows[0], exec_ts=EXEC_TS).fetched_at
    assert path.exists()

    with SegmentCache(path=path, clock=clock) as reopened:
        look: SegmentLookup = reopened.lookup(spec, windows[0], exec_ts=EXEC_TS)
        assert look.status is SegmentStatus.HIT
        assert look.fetched_at == fetched_at


def test_duckdb_memory_limit_and_spill_dir_are_applied(tmp_path: Path) -> None:
    """The store must degrade to disk, and no extension may be needed to do it."""
    spill = tmp_path / "spill"
    store = SegmentCache(path=tmp_path / "segments.duckdb", memory_limit="64MB", spill_dir=spill)
    try:
        assert spill.is_dir(), "the spill directory is created"
        configured = store._con.execute(
            "SELECT current_setting('temp_directory')"
        ).fetchall()
        assert Path(configured[0][0]).resolve() == spill.resolve()
        loaded = store._con.execute(
            "SELECT extension_name, install_mode FROM duckdb_extensions() WHERE loaded"
        ).fetchall()
        assert loaded, "DuckDB's own core functions are linked in"
        assert all(mode == "STATICALLY_LINKED" for _name, mode in loaded), (
            "every loaded extension must be built into the wheel: a downloaded one needs the "
            f"internet, which the target network has not. loaded={loaded}"
        )
    finally:
        store.close()


def test_a_bad_memory_limit_is_rejected_before_it_reaches_duckdb() -> None:
    with pytest.raises(ValueError, match="must look like"):
        SegmentCache(memory_limit="512MB'; DROP TABLE x; --")


# --- construction guards -----------------------------------------------------------------


def _key(
    *,
    source: str = "billing",
    table: str = "usage_events",
    fragment_hash: str = "fh",
    version_hash: str = "v1",
    window: Range | None = None,
) -> str:
    """One segment key with every component defaulted, so a test varies exactly one of them."""
    return segment_key(
        source=source,
        table=table,
        fragment_hash=fragment_hash,
        version_hash=version_hash,
        window=window or _month(2026, 1),
    )


def test_segment_key_covers_everything_that_can_change_the_rows() -> None:
    key = _key()

    assert _key() == key, "the key is stable"
    assert _key(source="other") != key
    assert _key(table="other") != key
    assert _key(fragment_hash="other") != key
    assert _key(version_hash="other") != key
    assert _key(window=_month(2026, 2)) != key


def test_a_spec_rejects_an_empty_or_separator_bearing_field() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        _spec(source="")
    with pytest.raises(ValueError, match="must not contain"):
        _spec(table="usage\x1fevents")


def test_a_tail_policy_requires_a_unique_key_and_plain_column_names() -> None:
    with pytest.raises(ValueError, match="at least one column"):
        TailPolicy(ingestion_column="insertion_time", unique_key=())
    with pytest.raises(ValueError, match="plain column name"):
        TailPolicy(ingestion_column="insertion_time; DROP TABLE x", unique_key=("event_id",))


def test_a_naive_exec_ts_is_refused(cache: SegmentCache) -> None:
    """Every instant in the system is timezone-aware; a naive one is a bug, not agent input."""
    with pytest.raises(ValueError, match="timezone-aware"):
        cache.lookup(_spec(), _month(2026, 1), exec_ts=datetime(2026, 6, 20))


def test_a_segment_part_reports_its_row_count(cache: SegmentCache) -> None:
    adapter = _adapter(_january_rows())
    windows = align_to_partitions(_month(2026, 1))
    parts: tuple[SegmentPart, ...] = cache.fetch(
        _spec(), windows, exec_ts=EXEC_TS, fetch_window=_window_fetcher(adapter)
    )
    assert parts[0].rows == 2
    assert parts[0].rows_fetched == 2
    assert parts[0].tail_refreshed is False
