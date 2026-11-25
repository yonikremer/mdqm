"""Tests for the query history (H03): frozen versus relative re-run, search, and retention.

The claim under test is a semantic one, so the tests are written against the payloads a source
would receive, not against helper names: a **frozen** re-run must hand back the exact fragment that
ran (same window, same ``exec_ts``) even when the clock has moved, and a **relative** re-run must
move the window with the clock — changing the timestamps and nothing else.

Nothing sleeps and nothing reads the real clock: the clock is injected, so "run today, re-run
tomorrow" is one line. Every test uses ``tmp_path``, so no test depends on the working directory,
on FTS5 being compiled in, or on the OS.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mdqm.models import ErrorResponse, Fragment, PlanSummary
from mdqm.spec.models import QuerySpec
from mdqm.sql.exec_ts import ExecTsMode, rewrite_es_query, rewrite_sql
from mdqm.store import Store
from mdqm.store.history import (
    HISTORY_TABLE,
    CacheInfo,
    HistoryEntry,
    HistoryError,
    HistoryRecord,
    HistoryStore,
    RecordedFragment,
)

# ---------------------------------------------------------------------------
# fixtures: a real migrated file, a scripted clock
# ---------------------------------------------------------------------------

BASE = datetime(2026, 11, 25, 16, 0, tzinfo=UTC)


class FakeClock:
    """A clock the test advances by hand: no sleeping, no wall-clock coupling."""

    def __init__(self, start: datetime = BASE) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: int) -> datetime:
        self.now += timedelta(**kwargs)
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    """A migrated metadata DB on a throwaway file, closed afterwards."""
    opened = Store(tmp_path / "meta.sqlite")
    opened.initialize()
    try:
        yield opened
    finally:
        opened.close()


@pytest.fixture
def history(store: Store, clock: FakeClock) -> HistoryStore:
    return HistoryStore(store, clock=clock)


# ---------------------------------------------------------------------------
# the payloads under test: a weekly window, which is what makes the modes differ
# ---------------------------------------------------------------------------

WEEKLY_ORIGINAL = (
    "SELECT customer_status, AVG(credit_limit) FROM accounts "
    "WHERE start_time >= now() - INTERVAL '7 DAYS' GROUP BY customer_status"
)
WEEKLY_FROZEN = rewrite_sql(WEEKLY_ORIGINAL, dialect="postgres", exec_ts=BASE).sql

#: A payload with no now-like value: nothing a re-run could move.
STATIC_SQL = "SELECT customer_status, COUNT(*) FROM accounts GROUP BY customer_status"

_ES_ORIGINAL: dict[str, Any] = {"query": {"range": {"start_time": {"gte": "now-7d/d"}}}}
_ES_FROZEN, _ = rewrite_es_query(_ES_ORIGINAL, exec_ts=BASE)

#: An ISO timestamp, which is the only thing a relative re-run is allowed to change.
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?")


def masked(text: str) -> str:
    """``text`` with every timestamp replaced by a marker: what must be unchanged by a re-run."""
    return _TIMESTAMP.sub("<ts>", text)


def payload(fragment: RecordedFragment | Fragment) -> tuple[object, ...]:
    """The executed shape of a fragment (its ``exec_ts`` is asserted on the plan instead)."""
    return (
        fragment.source,
        fragment.dialect,
        fragment.sql,
        fragment.dsl,
        fragment.request,
        fragment.params,
        fragment.tables,
    )


def weekly_fragment(**overrides: object) -> RecordedFragment:
    fields: dict[str, object] = {
        "source": "billing",
        "dialect": "postgres",
        "sql": WEEKLY_FROZEN,
        "original_sql": WEEKLY_ORIGINAL,
        "tables": ("accounts",),
    }
    fields.update(overrides)
    return RecordedFragment(**fields)  # type: ignore[arg-type]


def weekly_entry(**overrides: object) -> HistoryEntry:
    """A representative run: the same weekly question, one source."""
    fields: dict[str, object] = {
        "user_id": "yoni",
        "exec_ts": BASE,
        "mode": ExecTsMode.RELATIVE,
        "original_sql": WEEKLY_ORIGINAL,
        "frozen_sql": WEEKLY_FROZEN,
        "dialect": "postgres",
        "fragments": (weekly_fragment(),),
        "duration_ms": 42,
    }
    fields.update(overrides)
    return HistoryEntry(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# frozen re-run: the original window and the original exec_ts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "moved_by",
    [
        {"days": 1},
        {"days": 400},
        {"hours": 1},
        {"minutes": 1},
        {"days": 0},
    ],
    ids=["+1d", "+400d", "+1h", "+1m", "same-moment"],
)
def test_frozen_rerun_reproduces_the_original_window_and_exec_ts(
    history: HistoryStore, clock: FakeClock, moved_by: dict[str, int]
) -> None:
    """Acceptance: even when "today" moved, a frozen re-run is the run that happened."""
    recorded = history.record(weekly_entry())
    clock.advance(**moved_by)  # "today" is now a different day

    plan = history.rerun(recorded.id, mode="frozen")

    assert plan.exec_ts == BASE, "frozen must keep the stored exec_ts, not read the clock"
    assert plan.frozen
    assert [payload(f) for f in plan.fragments] == [payload(f) for f in recorded.fragments]
    assert plan.fragments[0].exec_ts == recorded.exec_ts
    assert plan.fragments[0].sql == WEEKLY_FROZEN
    assert "2026-11-18 16:00:00" in (plan.fragments[0].sql or ""), (
        "the 7-day window start must not move"
    )


def test_frozen_rerun_carries_the_stored_spec_so_the_window_is_replannable(
    history: HistoryStore, clock: FakeClock
) -> None:
    """The spec travels with the plan: a frozen re-run re-plans against the same exec_ts."""
    spec = QuerySpec.model_validate(
        {
            "from": "accounts",
            "select": ["customer_status"],
            "time": {"mode": "events_in_range", "last": "7d"},
        }
    )
    recorded = history.record(weekly_entry(spec=spec))
    clock.advance(days=3)

    plan = history.rerun(recorded.id)

    assert plan.spec == spec
    assert plan.exec_ts == BASE
    assert plan.mode is ExecTsMode.FROZEN


# ---------------------------------------------------------------------------
# relative re-run: the window moves with the clock, and nothing else changes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "moved_by",
    [{"days": 1}, {"days": 7}, {"hours": 6}, {"days": 365}],
    ids=["+1d", "+7d", "+6h", "+365d"],
)
def test_relative_rerun_shifts_the_window_with_the_clock(
    history: HistoryStore, clock: FakeClock, moved_by: dict[str, int]
) -> None:
    """Acceptance: "last 7 days" is a different window tomorrow, and the payload says so."""
    recorded = history.record(weekly_entry())
    moved_to = clock.advance(**moved_by)

    plan = history.rerun(recorded.id, mode="relative")

    assert plan.exec_ts == moved_to
    assert not plan.frozen
    expected = rewrite_sql(WEEKLY_ORIGINAL, dialect="postgres", exec_ts=moved_to).sql
    assert plan.fragments[0].sql == expected
    assert plan.fragments[0].exec_ts == moved_to
    assert plan.sql == expected


def test_relative_rerun_differs_from_frozen_only_in_timestamps(
    history: HistoryStore, clock: FakeClock
) -> None:
    """The whole point: same statement, same tables, same shape — only the window bounds moved."""
    recorded = history.record(weekly_entry())
    clock.advance(days=1)

    frozen = history.rerun(recorded.id, mode="frozen").fragments[0]
    relative = history.rerun(recorded.id, mode="relative").fragments[0]

    assert frozen.sql != relative.sql, "a weekly window must move; an identical one would be a bug"
    assert masked(str(frozen.sql)) == masked(str(relative.sql))
    assert (frozen.source, frozen.dialect, frozen.tables, frozen.params) == (
        relative.source,
        relative.dialect,
        relative.tables,
        relative.params,
    )


def test_relative_rerun_leaves_a_time_independent_payload_alone(
    history: HistoryStore, clock: FakeClock
) -> None:
    """A fragment with no now-like value is the same query tomorrow, so only exec_ts changes."""
    fragment = RecordedFragment(source="crm", dialect="postgres", sql=STATIC_SQL)
    recorded = history.record(
        weekly_entry(fragments=(fragment,), frozen_sql=None, original_sql=None)
    )
    clock.advance(days=1)

    plan = history.rerun(recorded.id, mode="relative")

    assert plan.fragments[0].sql == STATIC_SQL
    assert plan.fragments[0].exec_ts == clock.now


def test_relative_rerun_moves_elasticsearch_date_math(
    history: HistoryStore, clock: FakeClock
) -> None:
    """ES date math is the same problem in a different dialect, so it gets the same treatment."""
    fragment = RecordedFragment(
        source="es",
        dialect="dsl",
        dsl=_ES_FROZEN,
        original_dsl=_ES_ORIGINAL,
    )
    recorded = history.record(
        weekly_entry(fragments=(fragment,), frozen_sql=None, original_sql=None)
    )
    moved_to = clock.advance(days=1)

    frozen = history.rerun(recorded.id, mode="frozen").fragments[0]
    relative = history.rerun(recorded.id, mode="relative").fragments[0]

    assert frozen.dsl == _ES_FROZEN
    assert relative.dsl == rewrite_es_query(_ES_ORIGINAL, exec_ts=moved_to)[0]
    assert relative.dsl != frozen.dsl


def test_rerun_of_an_unknown_id_is_refused(history: HistoryStore) -> None:
    with pytest.raises(HistoryError, match="no history row"):
        history.rerun(999)


# ---------------------------------------------------------------------------
# what is refused at write time: a run that could not be reproduced
# ---------------------------------------------------------------------------

_UNREPRODUCIBLE: list[tuple[str, dict[str, object]]] = [
    (
        "now-like SQL with no original to re-derive from",
        {"fragments": (RecordedFragment(source="crm", dialect="postgres", sql=WEEKLY_ORIGINAL),)},
    ),
    (
        "a stored payload that is not the substitution of its original",
        {"fragments": (weekly_fragment(sql=WEEKLY_FROZEN.replace("18", "17")),)},
    ),
    (
        "a dialect with no exec_ts substitution",
        {"fragments": (RecordedFragment(source="hive", dialect="hive", sql="SELECT 1"),)},
    ),
    (
        "a run-level payload that does not match its original",
        {"frozen_sql": WEEKLY_FROZEN.replace("2026-11-18", "2026-11-11")},
    ),
    (
        "Elasticsearch date math with no original",
        {"fragments": (RecordedFragment(source="es", dialect="dsl", dsl=_ES_ORIGINAL),)},
    ),
    (
        "SQL that does not parse",
        {"fragments": (RecordedFragment(source="crm", dialect="postgres", sql="SELECT FROM"),)},
    ),
]


@pytest.mark.parametrize(
    ("reason", "overrides"), _UNREPRODUCIBLE, ids=[case[0] for case in _UNREPRODUCIBLE]
)
def test_unreproducible_run_is_refused_and_not_stored(
    history: HistoryStore, reason: str, overrides: dict[str, object]
) -> None:
    """A row a re-run would answer wrongly is worse than no row, so it is refused, not stored."""
    with pytest.raises(HistoryError):
        history.record(weekly_entry(**overrides))
    assert history.count() == 0, f"nothing may be written when the refusal is {reason}"


def test_an_entry_with_nothing_to_ask_again_is_invalid() -> None:
    with pytest.raises(ValidationError, match="cannot be asked again"):
        HistoryEntry(user_id="yoni", exec_ts=BASE, dialect="postgres")


def test_run_level_sql_needs_its_dialect() -> None:
    with pytest.raises(ValidationError, match="needs 'dialect'"):
        HistoryEntry(user_id="yoni", exec_ts=BASE, frozen_sql=WEEKLY_FROZEN)


def test_fragment_carries_exactly_one_payload() -> None:
    with pytest.raises(ValidationError, match="exactly one of sql, dsl, request"):
        RecordedFragment(source="crm", dialect="postgres")
    with pytest.raises(ValidationError, match="exactly one of sql, dsl, request"):
        RecordedFragment(source="crm", dialect="postgres", sql="SELECT 1", request={"path": "/x"})


# ---------------------------------------------------------------------------
# secrets never reach a history row
# ---------------------------------------------------------------------------

_SECRETS = (
    "postgresql://yoni:s3cr3tpass@db.internal:5432/crm",
    "Bearer abcdef1234567890",
    "password=hunter2",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0",
)


def _file_bytes(directory: Path) -> str:
    """Every byte of every file the store wrote, including the WAL."""
    return "\n".join(
        path.read_bytes().decode("utf-8", errors="replace")
        for path in sorted(directory.iterdir())
        if path.is_file()
    )


def test_a_credential_never_reaches_the_file_or_the_index(
    history: HistoryStore, store: Store, tmp_path: Path
) -> None:
    """Acceptance (principle 11): no secret in any column, any index, or any file byte."""
    spec = QuerySpec.model_validate(
        {
            "from": "accounts",
            "select": ["customer_status"],
            "filters": [{"column": "customer_status", "op": "eq", "value": _SECRETS[0]}],
        }
    )
    error = ErrorResponse(
        code="SOURCE_UNAVAILABLE",
        problem=f"the source refused the connection {_SECRETS[1]}",
        fix=f"check the secret for {_SECRETS[3]}, then retry",
    )
    recorded = history.record(
        weekly_entry(
            spec=spec,
            error=error,
            applied_fixes=(f"used the connection string {_SECRETS[2]}",),
        )
    )

    assert recorded.redactions >= 4, "every credential shape must be counted, not just dropped"
    written = _file_bytes(tmp_path)
    for secret in _SECRETS:
        assert secret not in written, f"a credential reached the metadata file: {secret[:12]}..."
    for token in ("s3cr3tpass", "hunter2", "abcdef1234567890"):
        assert not history.search(token), "a redacted credential must not be searchable either"
    store.close()  # flush the WAL before the final byte scan
    for secret in _SECRETS:
        assert secret not in _file_bytes(tmp_path)


def test_a_redacted_row_is_refused_by_rerun_rather_than_returned_inexactly(
    history: HistoryStore,
) -> None:
    """A scrubbed fragment is not the fragment that ran, so re-running it would change meaning."""
    entry = weekly_entry(
        applied_fixes=("authenticated with Bearer abcdef1234567890",),
    )
    recorded = history.record(entry)
    assert recorded.redactions == 1
    assert recorded.rerunnable is False

    with pytest.raises(HistoryError, match="redacted"):
        history.rerun(recorded.id, mode="relative")


# ---------------------------------------------------------------------------
# search: Hebrew and English words from the spec
# ---------------------------------------------------------------------------

_HEBREW_SPEC = QuerySpec.model_validate(
    {
        "from": "accounts",
        "select": ["customer_status", "credit_limit"],
        "filters": [{"column": "customer_status", "op": "eq", "value": "פעיל"}],
        "time": {"mode": "events_in_range", "last": "7d"},
    }
)


def _seed(history: HistoryStore, clock: FakeClock) -> list[int]:
    """Three runs: a Hebrew-filtered spec, an English weekly query, and an ES one."""
    ids = [
        history.record(weekly_entry(spec=_HEBREW_SPEC)).id,
        history.record(
            weekly_entry(
                user_id="dana",
                fragments=(RecordedFragment(source="crm", dialect="postgres", sql=STATIC_SQL),),
                frozen_sql=None,
                original_sql=None,
            )
        ).id,
        history.record(
            weekly_entry(
                fragments=(RecordedFragment(source="es", dialect="dsl", dsl=_ES_FROZEN),),
                frozen_sql=None,
                original_sql=None,
            )
        ).id,
    ]
    clock.advance(minutes=1)
    return ids


_SEARCH_CASES: list[tuple[str, str, tuple[int, ...]]] = [
    ("a Hebrew filter value", "פעיל", (0,)),
    ("Hebrew with niqqud", "פָּעִיל", (0,)),
    ("English words from the spec", "credit limit", (0,)),
    ("an English table name", "accounts", (0, 1)),
    ("a word only one run has", "start_time", (0, 2)),
    ("an Elasticsearch index", "nothing-here", ()),
]


@pytest.mark.parametrize(
    ("description", "query", "expected"),
    _SEARCH_CASES,
    ids=[case[0] for case in _SEARCH_CASES],
)
def test_search_finds_a_past_query_by_its_words(
    history: HistoryStore,
    clock: FakeClock,
    description: str,
    query: str,
    expected: tuple[int, ...],
) -> None:
    """Acceptance: Hebrew or English words from the spec find the run."""
    ids = _seed(history, clock)
    found = tuple(ids[index] for index in expected)
    assert tuple(row.id for row in history.search(query)) == tuple(reversed(found))


def test_search_is_per_user_and_newest_first(history: HistoryStore, clock: FakeClock) -> None:
    ids = _seed(history, clock)
    history.record(weekly_entry(spec=_HEBREW_SPEC))

    dana = history.search("accounts", user_id="dana")
    assert tuple(row.id for row in dana) == (ids[1],)
    yoni = history.search("accounts", user_id="yoni")
    assert len(yoni) == 2
    assert yoni[0].id > yoni[1].id, "newest first"


def test_search_uses_the_like_fallback_when_fts5_is_missing(
    history: HistoryStore, store: Store, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Principle 1: a build without FTS5 loses speed, not the feature."""
    ids = _seed(history, clock)
    monkeypatch.setattr(Store, "fts5_available", lambda self: False)
    fallback = HistoryStore(store, clock=clock)

    assert tuple(row.id for row in fallback.search("credit limit")) == (ids[0],)
    assert tuple(row.id for row in fallback.search("פעיל")) == (ids[0],)
    assert fallback.search("start_time") != (), "start_time is in the first run's SQL"
    assert len(fallback.search("accounts")) == 2
    assert fallback.search("nothing-here") == ()


def test_search_needs_a_word(history: HistoryStore) -> None:
    with pytest.raises(HistoryError, match="at least one word"):
        history.search("   ")


# ---------------------------------------------------------------------------
# retention: only expired rows go
# ---------------------------------------------------------------------------

_PURGE_CASES: list[tuple[str, int, int, bool]] = [
    # how old the row is at purge time, retention, expected survivors, purged?
    ("yesterday", 1, 365, False),
    ("364 days old", 364, 365, False),
    ("exactly at the cutoff", 365, 365, False),
    ("366 days old", 366, 365, True),
    ("ten years old", 3650, 365, True),
]


@pytest.mark.parametrize(
    ("description", "age_days", "retention", "purged"),
    [(case[0], case[1], case[2], case[3]) for case in _PURGE_CASES],
    ids=[case[0] for case in _PURGE_CASES],
)
def test_purge_removes_only_expired_rows(
    store: Store,
    clock: FakeClock,
    description: str,
    age_days: int,
    retention: int,
    purged: bool,
) -> None:
    """Acceptance: a row inside the window is never touched, and the boundary is not off by one."""
    history = HistoryStore(store, clock=clock, retention_days=retention)
    clock.now = BASE - timedelta(days=age_days)
    expired_id = history.record(weekly_entry()).id
    clock.now = BASE
    kept_id = history.record(weekly_entry()).id

    report = history.purge(now=BASE)

    assert report.purged is purged
    assert report.purged_rows == (1 if purged else 0)
    survivors = {row.id for row in history.recent()}
    assert kept_id in survivors
    assert (expired_id not in survivors) is purged


def test_purge_removes_the_search_index_entry_too(history: HistoryStore, clock: FakeClock) -> None:
    clock.now = BASE - timedelta(days=400)
    history.record(weekly_entry(spec=_HEBREW_SPEC))
    clock.now = BASE
    history.record(weekly_entry())
    assert len(history.search("פעיל")) == 1

    history.purge(now=BASE)

    assert history.search("פעיל") == ()
    assert len(history.search("credit limit")) == 1


def test_a_second_purge_is_a_no_op(history: HistoryStore, clock: FakeClock) -> None:
    clock.now = BASE - timedelta(days=500)
    history.record(weekly_entry())
    clock.now = BASE

    first = history.purge(now=BASE)
    second = history.purge(now=BASE)

    assert (first.purged_rows, second.purged_rows) == (1, 0)
    assert second.kept_rows == 0


# ---------------------------------------------------------------------------
# the row itself: everything the caller handed over survives a round trip
# ---------------------------------------------------------------------------


def test_every_field_survives_the_round_trip(history: HistoryStore) -> None:
    spec = _HEBREW_SPEC
    error = ErrorResponse(code="COST_TOO_HIGH", problem="the scan is too wide", fix="add a filter")
    plan = PlanSummary(
        partitions_scanned=2,
        partitions_total=12,
        uses_index=True,
        indexes_used=("accounts_status_idx",),
        est_rows=120.0,
        pruning_proof="proven",
    )
    cache = CacheInfo(hit=True, fetched_at=BASE, fresh=False)
    recorded = history.record(
        weekly_entry(
            spec=spec,
            plan=plan,
            cache=cache,
            error=error,
            applied_fixes=("rounded credit_limit to 5000",),
        )
    )

    reloaded = history.get(recorded.id)

    assert reloaded == recorded
    assert reloaded.spec == spec
    assert reloaded.plan == plan
    assert reloaded.cache == cache
    assert reloaded.error == error
    assert reloaded.applied_fixes == ("rounded credit_limit to 5000",)
    assert reloaded.duration_ms == 42
    assert reloaded.created_at == BASE
    assert reloaded.mode is ExecTsMode.RELATIVE


def test_a_corrupt_row_is_reported_not_defaulted(history: HistoryStore) -> None:
    recorded = history.record(weekly_entry())
    history._store.connect().execute(  # noqa: SLF001 - the test tampers with the file on purpose
        f"UPDATE {HISTORY_TABLE} SET fragments = 'not json' WHERE id = ?", (recorded.id,)
    )

    with pytest.raises(HistoryError, match="corrupt"):
        history.get(recorded.id)


def test_history_keeps_runs_of_several_users_apart(history: HistoryStore) -> None:
    history.record(weekly_entry(user_id="yoni"))
    history.record(weekly_entry(user_id="dana"))

    assert [row.user_id for row in history.recent(user_id="yoni")] == ["yoni"]
    assert history.count() == 2


def test_recent_limit_must_be_positive(history: HistoryStore) -> None:
    with pytest.raises(HistoryError, match="limit"):
        history.recent(limit=0)


def test_the_stored_row_is_a_frozen_model(history: HistoryStore) -> None:
    recorded: HistoryRecord = history.record(weekly_entry())
    with pytest.raises(ValidationError):
        recorded.user_id = "someone-else"  # type: ignore[misc]
