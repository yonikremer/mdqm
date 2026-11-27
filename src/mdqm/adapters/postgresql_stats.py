"""PostgreSQL query-store mining and mutability counters (task A04).

Two server views, one job each, and neither of them writes anything:

* **``pg_stat_statements``** -- the statements the server already ran, with *cumulative* counters
  (``calls``, ``total_exec_time``, ``rows``) since the last ``pg_stat_statements_reset()``. Stage
  D9 sums what an adapter hands over, so this module reports **per-statement deltas** against the
  previous reading of the same adapter: handing the same totals twice would count one execution
  twice. The first reading of an adapter is therefore a *baseline* (the totals the view holds).
* **``pg_stat_user_tables``** -- ``n_tup_ins/upd/del/hot_upd``, ``n_mod_since_analyze``,
  ``last_analyze``, ``last_autovacuum``. Counters again: two readings give the change, which is
  what classifies a table as append-only or ``mutable_critical`` (stage S05).

What the view *cannot* answer, and this module therefore does not pretend to:

* **No per-statement timestamps** in ``pg_stat_statements`` 1.10 (PostgreSQL 14-16): no first/last
  seen, so :meth:`~mdqm.adapters.postgresql.PostgresAdapter.mine_queries` cannot filter by ``since``
  and :attr:`~mdqm.models.RawQuery.last_seen` stays ``None``. Only the view-wide ``stats_reset``
  from ``pg_stat_statements_info`` is known, and it is reported in the capability detail.
* **Another role's statement text is masked** (``<insufficient privilege>``) unless the reader is a
  superuser or holds ``pg_read_all_stats``. Masked rows are skipped: their text is not a statement.
  The dev seed grants that role to ``mdqm_ro`` so mining sees the whole database, not just our own
  reads -- without it the pipeline would mine nothing and look calm about it.

Read-only by construction: every statement here is a ``SELECT`` and the adapter's connections are
read-only transactions run as a ``SELECT``-only role.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final, NamedTuple

import psycopg
from psycopg import sql

from mdqm.adapters.base import OPTIONAL_METHODS, NotSupported
from mdqm.models import RawQuery, TableStats

__all__ = [
    "MASKED_QUERY",
    "MUTABILITY_SQL",
    "MUTABILITY_CAPABILITY",
    "QUERY_STORE_CAPABILITY",
    "QueryStoreState",
    "QueryStoreReader",
    "mutability_from_rows",
    "query_store_state",
]

_log = logging.getLogger(__name__)

QUERY_STORE_CAPABILITY: Final = OPTIONAL_METHODS["mine_queries"]
MUTABILITY_CAPABILITY: Final = OPTIONAL_METHODS["mutability_stats"]

MASKED_QUERY: Final = "<insufficient privilege>"
"""What the view shows for another role's text when we lack ``pg_read_all_stats``."""

#: ``pg_stat_user_tables`` counters, one row per user table, in a stable order.
MUTABILITY_SQL: Final = """
SELECT schemaname || '.' || relname AS table_name,
       n_live_tup, n_tup_ins, n_tup_upd, n_tup_del, n_tup_hot_upd,
       n_mod_since_analyze, last_analyze, last_autovacuum
FROM pg_stat_user_tables
ORDER BY 1
"""

_TOTAL_COLUMNS: Final = ("total_exec_time", "total_time")
"""Cumulative time column name: renamed in PostgreSQL 13, so it is detected, not assumed."""

_DECLARE_RE: Final = re.compile(
    r"\s*DECLARE\s+.*?\bCURSOR\b(?:\s+(?:BINARY|SCROLL|NO\s+SCROLL)\b)*"
    r"(?:\s+WITH(?:OUT)?\s+HOLD)?\s+FOR\s+",
    re.IGNORECASE | re.DOTALL,
)
"""The wrapper psycopg's server-side cursors add; mining wants the statement inside it."""

_PLUMBING: Final = frozenset(
    {
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "SET",
        "RESET",
        "SHOW",
        "FETCH",
        "MOVE",
        "CLOSE",
        "DEALLOCATE",
        "SAVEPOINT",
        "RELEASE",
        "DISCARD",
        "PREPARE",
        "EXECUTE",
        "LISTEN",
        "UNLISTEN",
    }
)
"""First words of statements that control a session and answer no question about the data."""

_MINE_SQL: Final = """
SELECT query,
       calls,
       {total} AS total_ms,
       rows,
       pg_catalog.pg_get_userbyid(userid) AS username
FROM pg_stat_statements
WHERE dbid = (SELECT oid FROM pg_catalog.pg_database WHERE datname = current_database())
  AND query NOT LIKE '%%pg_stat_statements%%'
ORDER BY 1, 5
"""


class Statement(NamedTuple):
    """One statement's counters, summed over the view's rows for it.

    ``pg_stat_statements`` groups by ``queryid`` *and* the raw text, so one statement can arrive as
    several rows: our server-side cursors wrap it in ``DECLARE "<cursor name>" CURSOR FOR ...``, and
    every cursor name is its own row. ``statements()`` sums them back into one statement, which is
    what a miner wants and what makes "run twice" equal ``calls == 2``.
    """

    text: str
    user: str | None
    calls: int
    total_ms: float
    rows: int


Counters = tuple[int, float, int]
"""``(calls, total_ms, rows)``: what a statement's counters read at one instant."""


class QueryStoreState(NamedTuple):
    """What ``pg_stat_statements`` offers *this* role right now: installed? readable? since when?"""

    present: bool
    readable: bool
    detail: str
    version: str | None = None
    stats_reset: datetime | None = None
    all_users: bool = False
    total_column: str = _TOTAL_COLUMNS[0]

    @property
    def supported(self) -> bool:
        """True only when the view exists *and* this role can select from it."""
        return self.present and self.readable


def query_store_state(cursor: psycopg.Cursor[Any]) -> QueryStoreState:
    """Probe the extension: installed, readable, its version, its reset instant, whose text we see.

    Never raises for a missing or unreadable extension: that is a capability verdict, not a
    failure, and the caller records it in :class:`~mdqm.models.Capabilities` instead of crashing.
    """
    cursor.execute("SELECT to_regclass('pg_stat_statements') IS NOT NULL")
    present_row = cursor.fetchone()
    present = bool(present_row and present_row[0])
    if not present:
        return QueryStoreState(
            present=False,
            readable=False,
            detail=(
                "pg_stat_statements is not installed in this database (or is not on this role's "
                "search_path); an operator must start the server with "
                "shared_preload_libraries=pg_stat_statements and run CREATE EXTENSION "
                "pg_stat_statements, then restart the connection"
            ),
        )
    try:
        cursor.execute("SELECT 1 FROM pg_stat_statements LIMIT 0")
    except psycopg.errors.InsufficientPrivilege as exc:
        return QueryStoreState(
            present=True,
            readable=False,
            detail=f"pg_stat_statements is installed but this role may not read it: {exc!s}",
        )
    cursor.execute("SELECT extversion::text FROM pg_extension WHERE extname = 'pg_stat_statements'")
    version_row = cursor.fetchone()
    version = None if version_row is None else str(version_row[0])
    stats_reset: datetime | None = None
    try:
        cursor.execute("SELECT stats_reset FROM pg_stat_statements_info")
        reset_row = cursor.fetchone()
        stats_reset = None if reset_row is None else reset_row[0]
    except psycopg.errors.UndefinedTable:
        # pg_stat_statements_info appeared in 1.8 (PostgreSQL 14): an older view has no reset.
        _log.debug("pg_stat_statements_info is missing on this server; stats_reset unknown")
    cursor.execute(
        """
        SELECT rolsuper OR pg_has_role(current_user, 'pg_read_all_stats', 'MEMBER')
        FROM pg_roles WHERE rolname = current_user
        """
    )
    all_row = cursor.fetchone()
    all_users = bool(all_row and all_row[0])
    total_column = _total_column(cursor)
    detail = f"pg_stat_statements {version or '?'}"
    if stats_reset is not None:
        detail += f", counters since {stats_reset.isoformat()}"
    detail += ", no per-statement first/last seen"
    if not all_users:
        detail += (
            "; other roles' statement text is masked (`<insufficient privilege>`) until "
            "`GRANT pg_read_all_stats TO <this role>`"
        )
    return QueryStoreState(
        present=True,
        readable=True,
        detail=detail,
        version=version,
        stats_reset=stats_reset,
        all_users=all_users,
        total_column=total_column,
    )


def _total_column(cursor: psycopg.Cursor[Any]) -> str:
    """``total_exec_time`` on PostgreSQL 13+, ``total_time`` before it: detected, never assumed."""
    cursor.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'pg_stat_statements'
          AND table_schema = ANY (current_schemas(false))
        """
    )
    names = {str(row[0]) for row in cursor.fetchall()}
    for candidate in _TOTAL_COLUMNS:
        if candidate in names:
            return candidate
    raise NotSupported(
        QUERY_STORE_CAPABILITY,
        "pg_stat_statements has neither total_exec_time nor total_time: unknown extension layout",
    )


def statements(cursor: psycopg.Cursor[Any], state: QueryStoreState) -> list[Statement]:
    """Statements in this database with their totals, or the plumbing it would be noise to mine.

    Three kinds of row are not a query: masked text, session control (``BEGIN``/``FETCH``/``SET``
    ... -- including our own cursor reads), and this module's own mining ``SELECT`` (excluded in
    SQL). A ``DECLARE ... CURSOR FOR <query>`` row *is* a query: the text is unwrapped to the
    statement inside and summed with every other row for the same statement.
    """
    total = sql.Identifier(state.total_column)
    # The statement is a module constant and the only interpolation is an identifier psycopg
    # quotes, so there is no value to bind; marked for the scanner, not hidden from the reader.
    # Compose first, execute second: the statement is a module constant with one psycopg-quoted
    # identifier, so the composition is not a place a value can enter.
    _q = sql.SQL(_MINE_SQL).format(total=total)
    # pi-lens-ignore: python-sql-injection
    cursor.execute(_q)  # nosemgrep
    totals: dict[str, Statement] = {}
    users: dict[str, set[str]] = {}
    dropped = 0
    for row in cursor.fetchall():
        text = str(row[0])
        if text == MASKED_QUERY:
            dropped += 1
            continue
        text = _unwrap_declare(text)
        if _is_plumbing(text):
            dropped += 1
            continue
        calls, total_ms, rows = int(row[1] or 0), float(row[2] or 0.0), int(row[3] or 0)
        if row[4] is not None:
            users.setdefault(text, set()).add(str(row[4]))
        seen = totals.get(text)
        totals[text] = Statement(
            text=text,
            user=None,
            calls=calls + (seen.calls if seen else 0),
            total_ms=total_ms + (seen.total_ms if seen else 0.0),
            rows=rows + (seen.rows if seen else 0),
        )
    if dropped:
        _log.debug("mine_queries skipped %d masked or plumbing statements", dropped)
    return [
        # Several roles running one statement is not one user's statement: say so with None.
        item._replace(user=next(iter(users[text])) if len(users.get(text, ())) == 1 else None)
        for text, item in sorted(totals.items())
    ]


def _unwrap_declare(text: str) -> str:
    """``DECLARE "c1" CURSOR FOR SELECT ...`` -> ``SELECT ...``: the cursor is transport."""
    match = _DECLARE_RE.match(text)
    return text[match.end() :].strip() if match else text


def _is_plumbing(text: str) -> bool:
    """True for a session-control statement, which is not a query a caller asked the data for."""
    if not text.strip():
        return True
    first = text.lstrip().split(None, 1)[0].upper()
    return first in _PLUMBING


def new_activity(rows: Sequence[Statement], previous: dict[str, Counters]) -> list[Statement]:
    """Per-statement deltas against ``previous``; updates ``previous`` to the reading just made.

    A counter that went *backwards* means the view was reset: the baseline is dropped and the
    current totals are reported, because a negative call count is not a statement's history.
    Statements whose call counter did not move are not activity and are not emitted.
    """
    current = {row.text: (row.calls, row.total_ms, row.rows) for row in rows}
    reset = any(
        calls < previous[key][0] for key, (calls, _, _) in current.items() if key in previous
    )
    if reset:
        _log.info("pg_stat_statements was reset since the last reading; restarted the baseline")
        previous.clear()
    deltas: list[Statement] = []
    for row in rows:
        key = row.text
        base_calls, base_total, base_rows = previous.get(key, (0, 0.0, 0))
        calls = row.calls - base_calls
        if calls <= 0:
            continue
        deltas.append(
            Statement(
                text=row.text,
                user=row.user,
                calls=calls,
                total_ms=max(0.0, row.total_ms - base_total),
                rows=max(0, row.rows - base_rows),
            )
        )
    previous.clear()
    previous.update(current)
    deltas.sort(key=lambda item: (-item.calls, item.text))
    return deltas


@dataclass
class QueryStoreReader:
    """Reads ``pg_stat_statements`` and turns cumulative counters into per-statement deltas.

    The watermark is the previous reading of this adapter instance: in-process, not persisted, so a
    restarted server process takes one fresh baseline. That is the safe direction (the first batch
    after a restart is larger, never double), and stage D9 dedupes batches by their digest anyway.
    """

    #: Statement text -> ``(calls, total_ms, rows)`` of the previous reading.
    previous: dict[str, Counters] = field(default_factory=dict)

    def read(
        self, cursor: psycopg.Cursor[Any], *, source: str, dialect: str, since: datetime
    ) -> list[RawQuery]:
        """Mined statements, newest activity first, or :class:`NotSupported` when unreadable.

        ``since`` cannot narrow anything on ``pg_stat_statements`` 1.10: the view keeps totals, not
        timestamps, so the only honest statements are "all the activity this adapter has not seen"
        and the first reading's totals. It is accepted for the interface and logged when it falls
        outside the counters' own window.
        """
        state = query_store_state(cursor)
        if not state.supported:
            raise NotSupported(QUERY_STORE_CAPABILITY, state.detail)
        if state.stats_reset is not None and since > state.stats_reset:
            _log.debug(
                "mine_queries(%s) is later than pg_stat_statements stats_reset %s: the view holds "
                "counters, so the window cannot be applied per statement",
                since.isoformat(),
                state.stats_reset.isoformat(),
            )
        deltas = new_activity(statements(cursor, state), self.previous)
        return [
            RawQuery(
                text=item.text,
                dialect=dialect,
                calls=item.calls,
                mean_time_ms=item.total_ms / item.calls,
                total_time_ms=item.total_ms,
                rows=item.rows,
                last_seen=None,  # the view does not carry it; see the module docstring
                user=item.user,
            )
            for item in deltas
        ]


def mutability_from_rows(
    source: str, rows: Sequence[Sequence[Any]], captured_at: datetime
) -> list[TableStats]:
    """Map ``MUTABILITY_SQL`` rows onto :class:`~mdqm.models.TableStats`, in the query's order."""
    stats: list[TableStats] = []
    for row in rows:
        stats.append(
            TableStats(
                source=source,
                table=str(row[0]),
                captured_at=captured_at,
                row_estimate=_as_int(row[1]),
                rows_inserted=_as_int(row[2]),
                rows_updated=_as_int(row[3]),
                rows_deleted=_as_int(row[4]),
                hot_updates=_as_int(row[5]),
                rows_modified_since_analyze=_as_int(row[6]),
                last_analyze=row[7],
                last_autovacuum=row[8],
            )
        )
    return stats


def _as_int(value: object) -> int | None:
    """``int`` or ``None``: these counters are NULL for a table nobody has touched yet."""
    if value is None or isinstance(value, bool):
        return None
    return int(value) if isinstance(value, int | float | str) else None
