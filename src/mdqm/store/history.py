"""Query history: what ran, so the same question can be asked again exactly (H03).

The one mistake this module exists to prevent: **a re-run that silently answers a different
question**. "Average credit limit over the last 7 days" means something different today than it
did when the report was first run, so the two re-run modes are not a formatting choice:

* ``frozen`` re-uses the stored ``exec_ts``. It must reproduce the original window and the original
  timestamp, byte for byte, even when "today" has moved.
* ``relative`` takes a fresh ``exec_ts`` from the injected clock. The window moves with it.

Reproducing both needs **two spellings of every payload**: the original, with its now-like
functions (``now() - interval '7 days'``, ES ``now-7d/d``), and the one that actually ran, with
``exec_ts`` already substituted (:mod:`mdqm.sql.exec_ts`). Storing only the executed payload would
make a relative re-run return the *old* window under a *new* timestamp — the exact silent change
this task is about. So :meth:`HistoryStore.record` refuses a row whose fragments cannot be
reproduced:

* a fragment that declares an original must equal that original re-substituted at the stored
  ``exec_ts`` (byte for byte), and
* a fragment that declares no original must contain no now-like value at all — verified by asking
  :func:`~mdqm.sql.exec_ts.rewrite_sql` / :func:`~mdqm.sql.exec_ts.rewrite_es_query` for their
  replacements, not by trusting the caller.
* the dialect must be one whose substitution is defined, so an unknown dialect is a refusal and
  never a guess.

The honest limit of that check: it reads the text. A caller that hands over an already-substituted
payload *and omits the original* is indistinguishable from a query that never had a now-like value,
and no re-read of the stored row can tell them apart — so the pipeline must always record both, and
every original that is recorded is verified here.

No result row is ever stored, and no credential: every text column is redacted of credential shapes
before it is written (the count lands in ``redactions``), the search index is built from the
redacted text, and :meth:`HistoryStore.rerun` **refuses** a redacted row instead of returning a
fragment that differs from what ran (AGENTS.md principle 11, principle 10).

Search is one text column and an FTS5 index created lazily, because a Python build whose SQLite
lacks FTS5 must still be able to migrate the metadata DB: the migration cannot create a virtual
table that a build may refuse. Without FTS5 the same query runs as a ``LIKE`` scan of the same
column, so the feature degrades instead of disappearing (AGENTS.md principle 1).

The window itself lives in the fragments' predicates, where the source reads it; this module never
re-derives it, so there is one owner of that rule (:mod:`mdqm.planner.time`).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Final, cast

from pydantic import Field, model_validator
from sqlglot.errors import SqlglotError

from mdqm.models import (
    ErrorResponse,
    Fragment,
    JsonObject,
    MDQMModel,
    PlanSummary,
    UtcDatetime,
)
from mdqm.semantic.text import normalize_label
from mdqm.spec.models import QuerySpec
from mdqm.sql.exec_ts import (
    SUPPORTED_DIALECTS,
    ExecTsMode,
    resolve_exec_ts,
    rewrite_es_query,
    rewrite_sql,
)
from mdqm.store.audit import _SECRET_SHAPES, DEFAULT_RETENTION_DAYS
from mdqm.store.db import Store, StoreError

__all__ = [
    "DEFAULT_RETENTION_DAYS",
    "HISTORY_FTS_TABLE",
    "HISTORY_TABLE",
    "CacheInfo",
    "Clock",
    "HistoryEntry",
    "HistoryError",
    "HistoryPurgeReport",
    "HistoryRecord",
    "HistoryStore",
    "RecordedFragment",
    "RerunPlan",
]

logger = logging.getLogger(__name__)

#: The history table (see ``0003_history.sql``). A constant because the SQL below names it.
HISTORY_TABLE: Final[str] = "query_history"
#: The lazily created FTS5 index over ``query_history.search_text``.
HISTORY_FTS_TABLE: Final[str] = "query_history_fts"
#: What a redacted credential becomes. Deliberately not empty: a reader must see that something
#: was removed, and :meth:`HistoryStore.rerun` uses the row's count to refuse an inexact re-run.
REDACTED: Final[str] = "[REDACTED]"

#: A clock, injected so "run today, re-run tomorrow" is a two-line test with no sleeping.
Clock = Callable[[], datetime]

_MAX_TOKEN: Final[int] = 200

# FTS5 index: `remove_diacritics 2` so Hebrew niqqud and Latin accents fold (the text is also
# normalised before it is stored, so the LIKE fallback matches the same way).
_FTS_CREATE: Final[str] = (
    f"CREATE VIRTUAL TABLE IF NOT EXISTS {HISTORY_FTS_TABLE} "  # noqa: S608 - module constant
    'USING fts5(search_text, tokenize="unicode61 remove_diacritics 2")'
)
# Backfill, so a row written while FTS5 was unavailable is still findable once it is available.
_FTS_BACKFILL: Final[str] = (
    f"INSERT INTO {HISTORY_FTS_TABLE} (rowid, search_text) "  # noqa: S608 - module constants
    f"SELECT id, search_text FROM {HISTORY_TABLE} "
    f"WHERE id NOT IN (SELECT rowid FROM {HISTORY_FTS_TABLE})"
)
_FTS_PRUNE: Final[str] = (
    f"DELETE FROM {HISTORY_FTS_TABLE} "  # noqa: S608 - module constants
    f"WHERE rowid NOT IN (SELECT id FROM {HISTORY_TABLE})"
)

#: FTS5's string delimiter: a word in a MATCH query is quoted so it is never read as an operator.
_FTS_QUOTE: Final[str] = '"'


#: Shapes where the audit log's own list only names the key: in free text the value must go too.
_VALUE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"(?i)\b(?:password|passwd|pwd|secret|token|access[_-]?key|api[_-]?key|authorization|auth)"
        r"\b\s*[:=]\s*[^\s\"',;]+"
    ),
    # A private key pasted into a spec value or an error message: the whole block, not its header.
    re.compile(r"-----BEGIN[^\n]*-----.*?-----END[^\n]*-----", re.DOTALL),
)


class HistoryError(StoreError, ValueError):
    """A run that must not be stored as-is, or a row that cannot be re-run exactly.

    Also a ``ValueError`` so a pydantic validator that calls into this module reports it as a normal
    field rejection instead of leaking an unexpected exception type.
    """


class RecordedFragment(MDQMModel):
    """One source's payload as it ran, plus the payload it was rendered from.

    ``sql``/``dsl``/``request`` is what the adapter was given (``exec_ts`` already substituted).
    ``original_sql``/``original_dsl`` is the same payload *before* substitution, and it is the only
    thing a relative re-run can be derived from: an original is not optional when the payload held
    a now-like value (:meth:`HistoryStore.record` enforces that).
    """

    source: str = Field(min_length=1)
    dialect: str = Field(min_length=1)
    sql: str | None = None
    dsl: JsonObject | None = None
    request: JsonObject | None = None
    params: JsonObject = Field(default_factory=dict)
    tables: tuple[str, ...] = ()
    original_sql: str | None = None
    original_dsl: JsonObject | None = None

    @model_validator(mode="after")
    def _check_single_payload(self) -> RecordedFragment:
        present = [
            name
            for name, value in (("sql", self.sql), ("dsl", self.dsl), ("request", self.request))
            if value is not None
        ]
        if len(present) != 1:
            raise ValueError(
                f"a recorded fragment carries exactly one of sql, dsl, request; got "
                f"{present or 'none'}."
            )
        if self.original_sql is not None and self.sql is None:
            raise ValueError("original_sql only means something next to a sql payload")
        if self.original_dsl is not None and self.dsl is None:
            raise ValueError("original_dsl only means something next to a dsl payload")
        return self

    @property
    def time_dependent(self) -> bool:
        """True when the payload was rendered from an original by ``exec_ts`` substitution."""
        return self.original_sql is not None or self.original_dsl is not None


class CacheInfo(MDQMModel):
    """What the answer came from: a cache hit shows its ``fetched_at``, never "now"."""

    hit: bool = False
    fetched_at: UtcDatetime | None = None
    fresh: bool | None = None


class HistoryEntry(MDQMModel):
    """One run to store: what the pipeline hands over when a query finishes (or fails).

    A run must carry at least one way of asking the question again: the validated ``spec``, the
    ``original_sql``, or per-source ``fragments``. A row with none of them would be a memory of
    nothing, so it is refused rather than written.
    """

    user_id: str = Field(min_length=1, max_length=_MAX_TOKEN)
    tool: str = Field(default="query", min_length=1, max_length=_MAX_TOKEN)
    exec_ts: UtcDatetime
    mode: ExecTsMode = ExecTsMode.RELATIVE
    spec: QuerySpec | None = None
    original_sql: str | None = None
    frozen_sql: str | None = None
    dialect: str | None = None
    fragments: tuple[RecordedFragment, ...] = ()
    applied_fixes: tuple[str, ...] = ()
    plan: PlanSummary | None = None
    cache: CacheInfo | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    error: ErrorResponse | None = None

    @model_validator(mode="after")
    def _check_replayable(self) -> HistoryEntry:
        nothing_to_replay = (
            self.spec is None and self.original_sql is None and self.frozen_sql is None
        )
        if nothing_to_replay and not self.fragments:
            raise ValueError(
                "a history entry needs 'spec', 'original_sql' or 'fragments': a run with none of "
                "them cannot be asked again"
            )
        if self.frozen_sql is not None and self.dialect is None:
            raise ValueError("'frozen_sql' needs 'dialect', the sqlglot dialect it was rendered in")
        if len(self.applied_fixes) > _MAX_TOKEN:
            raise ValueError(f"applied_fixes holds more than {_MAX_TOKEN} entries")
        return self


class HistoryRecord(HistoryEntry):
    """A stored run, as :meth:`HistoryStore.get` returns it."""

    id: int
    created_at: UtcDatetime
    redactions: int = Field(default=0, ge=0)

    @property
    def rerunnable(self) -> bool:
        """False when a credential was redacted out of the row: a re-run would not be exact."""
        return self.redactions == 0


class RerunPlan(MDQMModel):
    """What a re-run will execute: one ``exec_ts`` for the whole plan, and the payloads at it."""

    run_id: int
    mode: ExecTsMode
    exec_ts: UtcDatetime
    spec: QuerySpec | None = None
    sql: str | None = Field(
        default=None, description="The run's SQL rendered with this plan's exec_ts."
    )
    fragments: tuple[Fragment, ...] = ()

    @property
    def frozen(self) -> bool:
        """True when this plan re-uses the original ``exec_ts``."""
        return self.mode is ExecTsMode.FROZEN


class HistoryPurgeReport(MDQMModel):
    """What a retention purge removed, and what it kept."""

    purged_rows: int = Field(ge=0)
    kept_rows: int = Field(ge=0)

    @property
    def purged(self) -> bool:
        """True when this run removed anything; a second run removes nothing."""
        return self.purged_rows > 0


def _utc_now() -> datetime:
    """The default clock: timezone-aware UTC, never a naive local time."""
    return datetime.now(UTC)


class HistoryStore:
    """Query history over one :class:`~mdqm.store.db.Store`.

    Not a singleton: pass the store (and, in tests, the clock). Writes go through
    :meth:`Store.writing`, so the row and its search-index entry land in one transaction.
    """

    def __init__(
        self,
        store: Store,
        *,
        clock: Clock | None = None,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> None:
        if retention_days < 0:
            raise HistoryError("retention_days must not be negative")
        self._store = store
        self._clock = clock if clock is not None else _utc_now
        self.retention_days = retention_days
        self._fts: bool | None = None

    # -- writing -------------------------------------------------------------------------

    def record(self, entry: HistoryEntry) -> HistoryRecord:
        """Verify the run is reproducible, then store it and return the row.

        Raises:
            HistoryError: a fragment cannot be reproduced at this ``exec_ts``, its dialect has no
                defined substitution, or its payload holds a now-like value with no original to
                re-derive from. Storing such a row would make a re-run silently wrong, which is
                worse than refusing to store it.
        """
        created_at = self._now()
        self._verify(entry)
        redactions, columns = _scrubbed(entry)
        search_text = normalize_label(" ".join(value for value in columns.values() if value))
        with self._store.writing() as conn:
            indexed = self._ensure_fts(conn)
            cursor = conn.execute(
                f"INSERT INTO {HISTORY_TABLE} (created_at, user_id, tool, mode, exec_ts,"  # noqa: S608
                " spec_json, original_sql, frozen_sql, dialect, fragments, applied_fixes,"
                " plan_json, cache_json, duration_ms, error_json, redactions, search_text)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    created_at.isoformat(),
                    entry.user_id,
                    entry.tool,
                    entry.mode.value,
                    entry.exec_ts.isoformat(),
                    columns["spec_json"],
                    columns["original_sql"],
                    columns["frozen_sql"],
                    columns["dialect"],
                    columns["fragments"],
                    columns["applied_fixes"],
                    columns["plan_json"],
                    columns["cache_json"],
                    entry.duration_ms,
                    columns["error_json"],
                    redactions,
                    search_text,
                ),
            )
            run_id = int(cursor.lastrowid or 0)
            if indexed:
                conn.execute(
                    f"INSERT INTO {HISTORY_FTS_TABLE} (rowid, search_text) VALUES (?, ?)",  # noqa: S608
                    (run_id, search_text),
                )
        stored = self.get(run_id)
        logger.info(
            "history row recorded",
            extra={
                "run_id": run_id,
                "sources": [fragment.source for fragment in entry.fragments],
                "redactions": stored.redactions,
            },
        )
        return stored

    def _now(self) -> datetime:
        """One clock reading per write, normalised to UTC (the store holds UTC only)."""
        moment = self._clock()
        if moment.tzinfo is None:
            raise HistoryError("the clock returned a naive datetime; history stores UTC only")
        return moment.astimezone(UTC)

    def _verify(self, entry: HistoryEntry) -> None:
        """Refuse a run whose payloads a re-run could not reproduce."""
        if entry.frozen_sql is not None:
            _verify_sql(entry.frozen_sql, entry.original_sql, entry.dialect, entry.exec_ts, _RUN)
        for fragment in entry.fragments:
            _verify_fragment(fragment, entry.exec_ts)

    # -- reading -------------------------------------------------------------------------

    def get(self, run_id: int) -> HistoryRecord:
        """The stored run, by id.

        Raises:
            HistoryError: no such run, or the row no longer parses as a history record.
        """
        row = (
            self._store.connect()
            .execute(
                f"SELECT * FROM {HISTORY_TABLE} WHERE id = ?",  # noqa: S608 - module constant
                (run_id,),
            )
            .fetchone()
        )
        if row is None:
            raise HistoryError(f"no history row with id {run_id}")
        return _to_record(row)

    def recent(self, *, user_id: str | None = None, limit: int = 20) -> tuple[HistoryRecord, ...]:
        """The newest runs first, optionally only one user's."""
        sql = f"SELECT * FROM {HISTORY_TABLE}"  # noqa: S608 - module constant
        params: list[object] = []
        if user_id is not None:
            sql += " WHERE user_id = ?"
            params.append(user_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(_limit(limit))
        return tuple(_to_record(row) for row in self._store.connect().execute(sql, params))

    def search(
        self, text: str, *, user_id: str | None = None, limit: int = 20
    ) -> tuple[HistoryRecord, ...]:
        """Runs whose spec, SQL or fix list contains every word of ``text``, newest first.

        Words are normalised the same way the stored text was (:func:`normalize_label`), so Hebrew
        with niqqud, a gershayim variant or a different case still matches. Uses the FTS5 index when
        the build has one and an equivalent ``LIKE`` scan when it does not.
        """
        terms = normalize_label(text).split()
        if not terms:
            raise HistoryError("search needs at least one word to look for")
        if self._fts is None:
            # Creating and backfilling the index is a write, so it takes the store's writer lock.
            with self._store.writing() as writer:
                self._ensure_fts(writer)
        conn = self._store.connect()
        if self._fts:
            rows = conn.execute(
                f"SELECT * FROM {HISTORY_TABLE} WHERE id IN (SELECT rowid FROM "  # noqa: S608
                f"{HISTORY_FTS_TABLE} WHERE {HISTORY_FTS_TABLE} MATCH ?)"
                + _user_clause(user_id)
                + " ORDER BY id DESC LIMIT ?",
                [_match_query(terms), *_user_params(user_id), _limit(limit)],
            ).fetchall()
        else:
            clause = " AND ".join("search_text LIKE ? ESCAPE '\\'" for _ in terms)
            rows = conn.execute(
                f"SELECT * FROM {HISTORY_TABLE} WHERE "  # noqa: S608 - module constant
                + clause
                + _user_clause(user_id)
                + " ORDER BY id DESC LIMIT ?",
                [
                    *(f"%{_like_escape(term)}%" for term in terms),
                    *_user_params(user_id),
                    _limit(limit),
                ],
            ).fetchall()
        return tuple(_to_record(row) for row in rows)

    def count(self) -> int:
        """How many runs the history holds."""
        row = (
            self._store.connect()
            .execute(
                f"SELECT COUNT(*) FROM {HISTORY_TABLE}"  # noqa: S608 - module constant
            )
            .fetchone()
        )
        return int(row[0])

    # -- re-running ----------------------------------------------------------------------

    def rerun(
        self,
        run_id: int,
        *,
        mode: ExecTsMode | str = ExecTsMode.FROZEN,
        clock: Clock | None = None,
    ) -> RerunPlan:
        """The payloads of a past run at one ``exec_ts``: the same one, or a fresh one.

        ``frozen`` keeps the stored ``exec_ts``, so every substituted literal is what it was and
        the fragments are byte-identical to the run. ``relative`` asks the clock, so each
        substituted literal moves with it: the payloads differ from the original ones *only* in
        those timestamps, because they are re-derived from the same original by the same
        deterministic rewrite.

        Raises:
            HistoryError: the row does not exist, or a credential was redacted out of it, so the
                fragment that would come back is not the fragment that ran.
        """
        record = self.get(run_id)
        if not record.rerunnable:
            raise HistoryError(
                f"history row {run_id} had {record.redactions} credential(s) redacted out of it, "
                "so it cannot be re-run exactly; ask the question again instead of re-running it"
            )
        # Coerced, not compared as a string: ``resolve_exec_ts`` branches on identity, so a caller
        # passing 'frozen' instead of the enum must not silently get a relative re-run.
        selected = ExecTsMode(mode)
        exec_ts = resolve_exec_ts(
            selected, clock=clock if clock is not None else self._clock, stored=record.exec_ts
        )
        fragments = tuple(_refragment(fragment, exec_ts) for fragment in record.fragments)
        sql = _rendered_sql(record, exec_ts)
        return RerunPlan(
            run_id=record.id,
            mode=selected,
            exec_ts=exec_ts,
            spec=record.spec,
            sql=sql,
            fragments=fragments,
        )

    # -- retention -----------------------------------------------------------------------

    def purge(self, *, now: datetime | None = None) -> HistoryPurgeReport:
        """Remove the runs older than ``retention_days``, and nothing else.

        ``created_at`` decides, not ``exec_ts``: a frozen re-run of an old window stored today is
        retained for the same year as any other run.
        """
        moment = now if now is not None else self._clock()
        if moment.tzinfo is None:
            raise HistoryError("purge needs an aware 'now'; history stores UTC only")
        cutoff = moment.astimezone(UTC) - timedelta(days=self.retention_days)
        with self._store.writing() as conn:
            self._ensure_fts(conn)
            expired = [
                int(row["id"])
                for row in conn.execute(
                    f"SELECT id, created_at FROM {HISTORY_TABLE}"  # noqa: S608 - module constant
                ).fetchall()
                if _parse_ts(str(row["created_at"])) < cutoff
            ]
            for chunk in _chunks(expired):
                marks = ",".join("?" * len(chunk))
                conn.execute(
                    f"DELETE FROM {HISTORY_TABLE} WHERE id IN ({marks})",  # noqa: S608 - ints
                    chunk,
                )
            conn.execute(_FTS_PRUNE)
            kept = _count(conn)
        if expired:
            logger.info(
                "history purge removed %d run(s) past %d-day retention",
                len(expired),
                self.retention_days,
                extra={"purged_rows": len(expired), "kept_rows": kept},
            )
        return HistoryPurgeReport(purged_rows=len(expired), kept_rows=kept)

    # -- the search index ----------------------------------------------------------------

    def _ensure_fts(self, conn: sqlite3.Connection) -> bool:
        """Create and backfill the FTS5 index once, reporting whether search can use it.

        Never raises for a missing FTS5: the capability probe's verdict is remembered and
        :meth:`search` falls back to a ``LIKE`` scan of the same column.
        """
        if self._fts is None:
            if not self._store.fts5_available():
                logger.info("FTS5 unavailable: history search falls back to a LIKE scan")
                self._fts = False
            else:
                conn.execute(_FTS_CREATE)
                conn.execute(_FTS_BACKFILL)
                self._fts = True
        return self._fts


# ---------------------------------------------------------------------------------------------
# reproducibility: the check that makes a re-run exact
# ---------------------------------------------------------------------------------------------

#: How a payload is named in a refusal, when it is the run's own SQL rather than a fragment's.
_RUN: Final[str] = "the run's SQL"


def _verify_fragment(fragment: RecordedFragment, exec_ts: datetime) -> None:
    """Check one fragment can be reproduced at ``exec_ts`` (:meth:`HistoryStore.record`)."""
    if fragment.sql is not None:
        _verify_sql(fragment.sql, fragment.original_sql, fragment.dialect, exec_ts, fragment.source)
        return
    if fragment.dsl is not None:
        _verify_dsl(fragment.dsl, fragment.original_dsl, exec_ts, fragment.source)
        return
    # An API request carries values, not an engine clock: nothing substitutes a timestamp in it, so
    # a relative re-run of that fragment is the same request and no check is possible (or needed).


def _verify_sql(
    sql: str,
    original_sql: str | None,
    dialect: str | None,
    exec_ts: datetime,
    what: str,
) -> None:
    """A SQL payload must equal its original re-substituted, or hold nothing to substitute."""
    name = dialect if dialect is not None else ""
    if name not in SUPPORTED_DIALECTS:
        raise HistoryError(
            f"{what}: dialect {name!r} has no exec_ts substitution, so a re-run could not be "
            f"reproduced; record it in one of {sorted(SUPPORTED_DIALECTS)}"
        )
    if original_sql is not None:
        rendered = _rewrite(original_sql, name, exec_ts, what)
        if rendered != sql:
            raise HistoryError(
                f"{what}: the stored SQL is not 'original_sql' with this exec_ts substituted, so "
                "a re-run would not reproduce the run; store the SQL the substitution produced"
            )
        return
    if _replacements(sql, name, exec_ts, what):
        raise HistoryError(
            f"{what}: the SQL holds a now-like value but no 'original_sql', so a relative re-run "
            "could not move it; store the SQL before substitution as 'original_sql'"
        )


def _rewrite(sql: str, dialect: str, exec_ts: datetime, what: str) -> str:
    """``rewrite_sql`` with a parser failure reported as a history refusal, not a traceback."""
    try:
        return rewrite_sql(sql, dialect=dialect, exec_ts=exec_ts).sql
    except (ValueError, SqlglotError) as exc:
        raise HistoryError(f"{what}: {exc}") from exc


def _replacements(sql: str, dialect: str, exec_ts: datetime, what: str) -> int:
    """How many now-like values ``sql`` still holds (0 means the payload is time-independent)."""
    try:
        return len(rewrite_sql(sql, dialect=dialect, exec_ts=exec_ts).replacements)
    except (ValueError, SqlglotError) as exc:
        raise HistoryError(f"{what}: {exc}") from exc


def _verify_dsl(
    dsl: JsonObject,
    original_dsl: JsonObject | None,
    exec_ts: datetime,
    what: str,
) -> None:
    """The Elasticsearch equivalent of :func:`_verify_sql`: date math must be reproducible."""
    if original_dsl is not None:
        rewritten, _ = rewrite_es_query(original_dsl, exec_ts=exec_ts)
        if rewritten != dsl:
            raise HistoryError(
                f"{what}: the stored query is not 'original_dsl' with this exec_ts substituted, "
                "so a re-run would not reproduce the run"
            )
        return
    if rewrite_es_query(dsl, exec_ts=exec_ts)[1]:
        raise HistoryError(
            f"{what}: the query holds Elasticsearch date math but no 'original_dsl', so a "
            "relative re-run could not move it; store the query before substitution"
        )


def _refragment(fragment: RecordedFragment, exec_ts: datetime) -> Fragment:
    """One recorded fragment at ``exec_ts``: re-derived from the original when there is one."""
    sql = fragment.sql
    if sql is not None and fragment.original_sql is not None:
        sql = rewrite_sql(fragment.original_sql, dialect=fragment.dialect, exec_ts=exec_ts).sql
    dsl = fragment.dsl
    if dsl is not None and fragment.original_dsl is not None:
        dsl = rewrite_es_query(fragment.original_dsl, exec_ts=exec_ts)[0]
    return Fragment(
        source=fragment.source,
        dialect=fragment.dialect,
        sql=sql,
        dsl=dsl,
        request=fragment.request,
        params=fragment.params,
        tables=fragment.tables,
        exec_ts=exec_ts,
    )


def _rendered_sql(record: HistoryRecord, exec_ts: datetime) -> str | None:
    """The run's own SQL at ``exec_ts`` (``None`` when the run stored no run-level SQL)."""
    if record.frozen_sql is None:
        return None
    if record.original_sql is None or record.dialect is None:
        return record.frozen_sql
    return rewrite_sql(record.original_sql, dialect=record.dialect, exec_ts=exec_ts).sql


# ---------------------------------------------------------------------------------------------
# storage: column values, secret redaction, the search index
# ---------------------------------------------------------------------------------------------


def _scrubbed(entry: HistoryEntry) -> tuple[int, dict[str, str | None]]:
    """Every text column of ``entry`` as stored text, plus how many credentials were redacted.

    The count comes back beside the columns rather than inside them, so a caller cannot bind the
    text columns and forget that something was removed from them.
    """
    columns: dict[str, str | None] = {
        "spec_json": None if entry.spec is None else entry.spec.model_dump_json(),
        "original_sql": entry.original_sql,
        "frozen_sql": entry.frozen_sql,
        "dialect": entry.dialect,
        "fragments": json.dumps(
            [fragment.model_dump(mode="json") for fragment in entry.fragments],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "applied_fixes": json.dumps(list(entry.applied_fixes), ensure_ascii=False),
        "plan_json": None if entry.plan is None else entry.plan.model_dump_json(),
        "cache_json": None if entry.cache is None else entry.cache.model_dump_json(),
        "error_json": None if entry.error is None else entry.error.model_dump_json(),
    }
    redactions = 0
    for key, value in columns.items():
        if value is None:
            continue
        cleaned, found = _redact(value)
        columns[key] = cleaned
        redactions += found
    return redactions, columns


def _redact(text: str) -> tuple[str, int]:
    """Replace credential-shaped substrings with :data:`REDACTED`, counting them.

    The shapes are the audit log's list, imported rather than copied: one home for "what looks like
    a credential" means a new shape protects both files at once. It is *extended* here, because an
    audit field is refused outright while free text is kept: a key/value shape (``password=...``)
    and a PEM block must lose their **value** too, or the key alone would be redacted and the
    secret left standing. Those patterns run first for that reason.

    Over-redaction is accepted and reported (``redactions``, and a refusal in
    :meth:`HistoryStore.rerun`): a wrongly redacted row costs an exact re-run, a missed one puts a
    credential in the metadata file.
    """
    redacted = 0
    for pattern in _VALUE_PATTERNS + _SECRET_SHAPES:
        text, count = pattern.subn(REDACTED, text)
        redacted += count
    return text, redacted


def _to_record(row: sqlite3.Row) -> HistoryRecord:
    """Map one SQLite row onto the model; a row that no longer parses is an error, not a default."""
    try:
        raw_fragments = json.loads(str(row["fragments"]))
        run_id = int(row["id"])
        spec_json = row["spec_json"]
        plan_json = row["plan_json"]
        cache_json = row["cache_json"]
        error_json = row["error_json"]
        dialect = row["dialect"]
        original_sql = row["original_sql"]
        frozen_sql = row["frozen_sql"]
        duration_ms = row["duration_ms"]
        return HistoryRecord(
            id=run_id,
            created_at=_parse_ts(str(row["created_at"])),
            user_id=str(row["user_id"]),
            tool=str(row["tool"]),
            mode=ExecTsMode(str(row["mode"])),
            exec_ts=_parse_ts(str(row["exec_ts"])),
            spec=None if spec_json is None else QuerySpec.model_validate_json(str(spec_json)),
            original_sql=None if original_sql is None else str(original_sql),
            frozen_sql=None if frozen_sql is None else str(frozen_sql),
            dialect=None if dialect is None else str(dialect),
            fragments=tuple(
                RecordedFragment.model_validate(item) for item in cast("list[Any]", raw_fragments)
            ),
            applied_fixes=tuple(
                str(item) for item in cast("list[Any]", json.loads(str(row["applied_fixes"])))
            ),
            plan=None if plan_json is None else PlanSummary.model_validate_json(str(plan_json)),
            cache=None if cache_json is None else CacheInfo.model_validate_json(str(cache_json)),
            duration_ms=None if duration_ms is None else int(duration_ms),
            error=None
            if error_json is None
            else ErrorResponse.model_validate_json(str(error_json)),
            redactions=int(row["redactions"]),
        )
    except (ValueError, TypeError, KeyError, IndexError) as exc:  # ValidationError is a ValueError
        raise HistoryError(f"history row {row['id']} is corrupt: {exc}") from exc


def _parse_ts(text: str) -> datetime:
    """Parse a stored timestamp; an unparseable one is a broken row, not a default."""
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise HistoryError(f"a history row holds an unparseable timestamp: {exc}") from exc
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _count(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        f"SELECT COUNT(*) FROM {HISTORY_TABLE}"  # noqa: S608 - module constant
    ).fetchone()
    return int(row[0])


def _chunks(ids: Sequence[int], size: int = 500) -> Iterator[list[int]]:
    """Split ids into lists SQLite can bind in one statement."""
    for start in range(0, len(ids), size):
        yield list(ids[start : start + size])


def _limit(limit: int) -> int:
    if limit < 1:
        raise HistoryError(f"limit must be at least 1, got {limit}")
    return limit


def _user_clause(user_id: str | None) -> str:
    return " AND user_id = ?" if user_id is not None else ""


def _user_params(user_id: str | None) -> list[object]:
    return [] if user_id is None else [user_id]


def _match_query(terms: Sequence[str]) -> str:
    """FTS5 query: every word, quoted, so no word is read as an operator.

    A word is data here, never syntax: ``delete``, ``-x`` or ``a:b`` must not become a MATCH
    operator, so each one is a quoted string and the doubled quote is FTS5's escape.
    """
    quoted = [
        f"{_FTS_QUOTE}{term.replace(_FTS_QUOTE, _FTS_QUOTE * 2)}{_FTS_QUOTE}" for term in terms
    ]
    return " AND ".join(quoted)


def _like_escape(term: str) -> str:
    """Escape the LIKE wildcards so a word is matched literally (Hebrew included)."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
