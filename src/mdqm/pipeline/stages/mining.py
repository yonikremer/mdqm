"""Stage D9: deterministic mining of past queries. Code, not a model.

A source reports the statements it has run (:class:`~mdqm.models.query.RawQuery` from PostgreSQL
``pg_stat_statements``, Oracle ``V$SQL``, T-SQL Query Store, or the MCP's own history). This stage
turns that into ranked *candidates* for later stages and for the Context Pack:

* each statement is parsed, normalised and fingerprinted by :mod:`mdqm.sql.fingerprint` (a
  statement that does not parse is counted and reported, never fatal);
* statements sharing a fingerprint are merged: calls, distinct users, first and last seen;
* per-fingerprint counts are upserted into the metadata store, with a **per-source watermark** so
  the next pull asks only for what is new;
* candidates are ranked by ``a*log(count) + b*0.5**(age/half_life) + c*distinct_users`` with the
  constants in :class:`MiningWeights`;
* every conversion observed in history (``LPAD`` join keys, ``ROUND(x / 5000) * 5000``) is exposed
  as a typed :class:`~mdqm.sql.fingerprint.ConversionHint` carrying its evidence, for stages D7/D8
  to validate against data.

**Idempotent by input hash.** A batch of queries is identified by a sha256 over its contents -- the
same rule as DESIGN.md section 10, "stage inputs are hashed; unchanged input => skip". Re-delivering
a batch that was already mined writes nothing and reports ``skipped``: pulling the same window
twice never doubles a count. A source must therefore hand over statements it has not reported
before; a source with *cumulative* counters (``pg_statements`` totals since reset) is pulled as
deltas by its adapter, because adding overlapping totals would count one execution twice.

**Ranking approves nothing.** "Common" is not "correct" (DESIGN.md section 11): a rank orders
suggestions and the Context Pack, and a conversion hint raises a prior. Only validators plus human
approval turn any of it into truth.

Deterministic: no clock read (``exec_ts`` is injected), no ``hash()`` (sha256 over text), sorted
output, ties broken by fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

from mdqm.models.base import MDQMModel, UtcDatetime
from mdqm.models.query import RawQuery
from mdqm.sql.fingerprint import ConversionHint, HintEvidence, QueryFacts, analyze
from mdqm.store.db import Store

__all__ = [
    "DEFAULT_WEIGHTS",
    "INGEST_TABLE",
    "MINED_QUERY_TABLE",
    "WATERMARK_TABLE",
    "IngestReport",
    "MinedQuery",
    "MiningWeights",
    "ParseFailure",
    "batch_digest",
    "ingest",
    "mine",
    "rank",
    "read",
    "score",
    "watermark",
]

logger = logging.getLogger(__name__)

#: Per-fingerprint counters. One row per (source, fingerprint).
MINED_QUERY_TABLE: Final[str] = "mined_query"

#: The batches actually mined, so a repeated delivery of one batch writes nothing.
INGEST_TABLE: Final[str] = "mined_ingest"

#: Where each source's next pull starts.
WATERMARK_TABLE: Final[str] = "mining_watermark"


# The statements below are built once, from the module constants above; every value travels as a
# `?` parameter. Declared at module level so no call site builds SQL.
def _statement(*parts: str) -> str:
    """Assemble one SQL statement from module-level constants only (never from caller input)."""
    return "".join(parts)  # noqa: S608 - fragments are constants; values are bound parameters


_SELECT_WATERMARK: Final[str] = _statement(
    "SELECT watermark FROM ", WATERMARK_TABLE, " WHERE source = ?"
)
_SELECT_BATCH: Final[str] = _statement(
    "SELECT 1 FROM ", INGEST_TABLE, " WHERE source = ? AND batch_hash = ?"
)
_SELECT_ROW: Final[str] = _statement(
    "SELECT calls, statements, users, first_seen, last_seen FROM ",
    MINED_QUERY_TABLE,
    " WHERE source = ? AND fingerprint = ?",
)
_UPSERT_ROW: Final[str] = _statement(
    "INSERT INTO ",
    MINED_QUERY_TABLE,
    " (source, fingerprint, dialect, calls, statements, users, first_seen, last_seen,"
    " facts, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    " ON CONFLICT(source, fingerprint) DO UPDATE SET"
    " calls = excluded.calls, statements = excluded.statements, users = excluded.users,"
    " first_seen = excluded.first_seen, last_seen = excluded.last_seen,"
    " facts = excluded.facts, updated_at = excluded.updated_at",
)
_INSERT_BATCH: Final[str] = _statement(
    "INSERT INTO ",
    INGEST_TABLE,
    " (source, batch_hash, mined_at, statements, fingerprints, watermark)"
    " VALUES (?, ?, ?, ?, ?, ?)",
)
_UPSERT_WATERMARK: Final[str] = _statement(
    "INSERT INTO ",
    WATERMARK_TABLE,
    " (source, watermark, updated_at) VALUES (?, ?, ?) ON CONFLICT(source) DO UPDATE SET"
    " watermark = excluded.watermark, updated_at = excluded.updated_at",
)
_SELECT_ROWS: Final[str] = _statement(
    "SELECT source, fingerprint, dialect, calls, statements, users, first_seen, last_seen, facts"
    " FROM ",
    MINED_QUERY_TABLE,
)


def _stored_users(value: object) -> tuple[str, ...]:
    """The user list of a stored row, or an empty list when the column does not hold one.

    A metadata row that was edited by hand, or truncated, must not break a ranked read: mining
    reports fewer users, it never crashes the Context Pack.
    """
    try:
        decoded = json.loads(value) if isinstance(value, str) else None
    except ValueError:
        logger.debug("a mined_query row holds unreadable user JSON")
        return ()
    if not isinstance(decoded, list):
        return ()
    return tuple(sorted(str(user) for user in decoded))


class MiningWeights(MDQMModel):
    """The ranking constants of ``score = a*log(count) + b*0.5^(age/half_life) + c*users``.

    Configuration, not truth: an organisation that cares more about breadth than about volume
    raises ``users`` and lowers ``calls``. ``half_life_days`` is how long a statement takes to lose
    half of its recency weight.
    """

    calls: float = 1.0
    recency: float = 1.0
    users: float = 1.0
    half_life_days: float = 30.0


DEFAULT_WEIGHTS: Final[MiningWeights] = MiningWeights()


class MinedQuery(MDQMModel):
    """One fingerprint's counters plus the facts extracted from it: a *candidate*, not a fact.

    ``users`` is a sorted tuple of the identities that ran it (never a count of anonymous
    executions), so merging two batches is a set union rather than an unrecoverable sum.
    """

    source: str
    fingerprint: str
    dialect: str
    calls: int = 0
    statements: int = 0
    users: tuple[str, ...] = ()
    first_seen: UtcDatetime | None = None
    last_seen: UtcDatetime | None = None
    facts: QueryFacts
    score: float = 0.0

    @property
    def hints(self) -> tuple[ConversionHint, ...]:
        """The conversion hypotheses this query supports (evidence already attached)."""
        return self.facts.hints


class ParseFailure(MDQMModel):
    """A statement that did not parse, counted and named by the hash of its text.

    The text itself is not stored: it is the one thing here that may carry a literal with a name in
    it, and the hash is enough for a report or a human to find the statement at the source.
    """

    source: str
    dialect: str
    text_hash: str
    error: str
    count: int = 1


class IngestReport(MDQMModel):
    """What one mining run did: counters, the ranked candidates, and the parse failures."""

    source: str
    batch_hash: str
    skipped: bool = False
    """True when this exact batch had already been mined: nothing was written."""
    statements: int = 0
    parsed: int = 0
    failed: int = 0
    failures: tuple[ParseFailure, ...] = ()
    queries: tuple[MinedQuery, ...] = ()
    watermark_before: UtcDatetime | None = None
    watermark_after: UtcDatetime | None = None

    @property
    def hints(self) -> tuple[ConversionHint, ...]:
        """Every conversion hypothesis in this batch, deduplicated across statements."""
        unique: list[ConversionHint] = []
        for query in self.queries:
            for hint in query.hints:
                if hint not in unique:
                    unique.append(hint)
        return tuple(unique)


@dataclass
class _Group:
    """The running merge of every statement that shares one fingerprint."""

    facts: QueryFacts
    calls: int = 0
    statements: int = 0
    users: set[str] = field(default_factory=set)
    first_seen: datetime | None = None
    last_seen: datetime | None = None

    def add(self, query: RawQuery) -> None:
        self.statements += 1
        if query.calls:
            self.calls += query.calls
        if query.user:
            self.users.add(query.user)
        if query.last_seen is not None:
            if self.first_seen is None or query.last_seen < self.first_seen:
                self.first_seen = query.last_seen
            if self.last_seen is None or query.last_seen > self.last_seen:
                self.last_seen = query.last_seen


def text_hash(text: str) -> str:
    """sha256 of a statement's own text: how a failure is named without storing the text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def batch_digest(queries: Iterable[RawQuery]) -> str:
    """sha256 over a batch's contents, independent of the order the source listed them in.

    Two pulls that report the same statements therefore hash the same, which is what makes
    re-ingesting a batch a no-op instead of a doubled count.
    """
    rows = sorted(
        "\x1f".join(
            (
                query.dialect,
                text_hash(query.text),
                "" if query.calls is None else str(query.calls),
                query.user or "",
                "" if query.last_seen is None else query.last_seen.isoformat(),
            )
        )
        for query in queries
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def age_days(last_seen: datetime | None, exec_ts: datetime) -> float | None:
    """Days between a statement's last appearance and the execution time, or ``None`` if unknown."""
    if last_seen is None:
        return None
    return max(0.0, (exec_ts - last_seen).total_seconds() / 86_400.0)


def score(
    *,
    calls: int = 0,
    statements: int = 0,
    distinct_users: int = 0,
    age: float | None = None,
    weights: MiningWeights = DEFAULT_WEIGHTS,
) -> float:
    """The ranking score, rounded to 6 decimals so equal inputs give equal output bytes.

    ``count`` is the number of recorded calls, falling back to the number of statements when the
    source does not report calls, and to 1 so ``log`` stays defined. ``age`` is ``None`` when the
    statement's ``last_seen`` is unknown, and an unknown age earns **no** recency credit: not
    knowing when a query ran must not look like "just now".
    """
    count = max(calls, statements, 1)
    volume = weights.calls * math.log(count)
    recency = (
        weights.recency * 0.5 ** (age / weights.half_life_days)
        if age is not None and weights.half_life_days > 0
        else 0.0
    )
    breadth = weights.users * distinct_users
    return round(volume + recency + breadth, 6)


def rank(
    queries: Iterable[MinedQuery],
    *,
    exec_ts: datetime,
    weights: MiningWeights = DEFAULT_WEIGHTS,
) -> tuple[MinedQuery, ...]:
    """Recompute the score at ``exec_ts`` and order by it, ties broken by fingerprint.

    Scores are recomputed rather than trusted from storage, because recency moves with the clock and
    a stale score would rank an old statement above a fresh one.
    """
    rescored = [
        query.model_copy(
            update={
                "score": score(
                    calls=query.calls,
                    statements=query.statements,
                    distinct_users=len(query.users),
                    age=age_days(query.last_seen, exec_ts),
                    weights=weights,
                )
            }
        )
        for query in queries
    ]
    return tuple(sorted(rescored, key=lambda query: (-query.score, query.fingerprint)))


def _with_evidence(facts: QueryFacts, *, source: str, fingerprint: str, calls: int) -> QueryFacts:
    """Attach the provenance every hint must carry: which source, which query, how often."""
    if not facts.hints:
        return facts
    evidence = HintEvidence(source=source, fingerprint=fingerprint, calls=calls)
    return facts.model_copy(
        update={
            "hints": tuple(hint.model_copy(update={"evidence": evidence}) for hint in facts.hints)
        }
    )


def mine(
    queries: Sequence[RawQuery],
    *,
    source: str,
    exec_ts: datetime,
    weights: MiningWeights = DEFAULT_WEIGHTS,
) -> IngestReport:
    """Extract, merge and rank one batch of past queries. Pure: nothing is stored or fetched.

    Parse failures are collected per (dialect, statement) with a count; the rest of the batch is
    mined regardless, because one unparseable statement in a source's history is normal.
    """
    groups: dict[str, _Group] = {}
    failures: dict[tuple[str, str], ParseFailure] = {}
    for query in queries:
        facts = analyze(query.text, query.dialect)
        if facts.fingerprint is None:
            key = (query.dialect, text_hash(query.text))
            seen = failures.get(key)
            failures[key] = ParseFailure(
                source=source,
                dialect=query.dialect,
                text_hash=key[1],
                error=facts.error or "no fingerprint",
                count=1 if seen is None else seen.count + 1,
            )
            continue
        group = groups.get(facts.fingerprint)
        if group is None:
            group = _Group(facts=facts)
            groups[facts.fingerprint] = group
        group.add(query)
    mined = [
        MinedQuery(
            source=source,
            fingerprint=fingerprint,
            dialect=group.facts.dialect,
            calls=group.calls,
            statements=group.statements,
            users=tuple(sorted(group.users)),
            first_seen=group.first_seen,
            last_seen=group.last_seen,
            facts=_with_evidence(
                group.facts, source=source, fingerprint=fingerprint, calls=group.calls
            ),
        )
        for fingerprint, group in groups.items()
    ]
    ranked = rank(mined, exec_ts=exec_ts, weights=weights)
    return IngestReport(
        source=source,
        batch_hash=batch_digest(queries),
        statements=len(queries),
        parsed=sum(group.statements for group in groups.values()),
        failed=len(queries) - sum(group.statements for group in groups.values()),
        failures=tuple(failures[key] for key in sorted(failures)),
        queries=ranked,
    )


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat()


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    return datetime.fromisoformat(value)


def watermark(store: Store, source: str) -> datetime | None:
    """The next pull's starting point for ``source``: the newest ``last_seen`` ever mined."""
    row = store.connect().execute(_SELECT_WATERMARK, (source,)).fetchone()
    return None if row is None else _parse_iso(row["watermark"])


def _already_mined(conn: sqlite3.Connection, source: str, batch: str) -> bool:
    row = conn.execute(_SELECT_BATCH, (source, batch)).fetchone()
    return row is not None


def _merge_rows(
    existing: sqlite3.Row, query: MinedQuery
) -> tuple[int, int, str, str | None, str | None]:
    """Combine a stored row with a new batch: additive counters, union of users, widest window."""
    users = sorted({*_stored_users(existing["users"]), *query.users})
    first_seen = _earliest(_parse_iso(existing["first_seen"]), query.first_seen)
    last_seen = _latest(_parse_iso(existing["last_seen"]), query.last_seen)
    return (
        int(existing["calls"]) + query.calls,
        int(existing["statements"]) + query.statements,
        json.dumps(users),
        _iso(first_seen),
        _iso(last_seen),
    )


def _earliest(left: datetime | None, right: datetime | None) -> datetime | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _latest(left: datetime | None, right: datetime | None) -> datetime | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def ingest(
    store: Store,
    queries: Sequence[RawQuery],
    *,
    source: str,
    exec_ts: datetime,
    weights: MiningWeights = DEFAULT_WEIGHTS,
) -> IngestReport:
    """Mine a batch and upsert its counters, the batch ledger and the watermark in one transaction.

    Idempotent per batch: a batch already in the ledger writes nothing (``skipped=True``) and leaves
    the watermark alone. The watermark only ever moves forward, so an out-of-order pull cannot make
    the next pull re-read what was already mined.
    """
    report = mine(queries, source=source, exec_ts=exec_ts, weights=weights)
    seen = _seen_up_to(queries)
    with store.writing() as conn:
        before = watermark(store, source)
        if _already_mined(conn, source, report.batch_hash):
            logger.info(
                "mining batch already ingested",
                extra={"source": source, "batch": report.batch_hash[:12]},
            )
            return report.model_copy(
                update={"skipped": True, "watermark_before": before, "watermark_after": before}
            )
        after = _latest(before, seen)
        for query in report.queries:
            existing = conn.execute(_SELECT_ROW, (source, query.fingerprint)).fetchone()
            calls, statements, users, first_seen, last_seen = (
                query.calls,
                query.statements,
                (json.dumps(list(query.users))),
                _iso(query.first_seen),
                _iso(query.last_seen),
            )
            if existing is not None:
                calls, statements, users, first_seen, last_seen = _merge_rows(existing, query)
            conn.execute(
                _UPSERT_ROW,
                (
                    source,
                    query.fingerprint,
                    query.dialect,
                    calls,
                    statements,
                    users,
                    first_seen,
                    last_seen,
                    query.facts.model_dump_json(),
                    _iso(exec_ts),
                ),
            )
        conn.execute(
            _INSERT_BATCH,
            (
                source,
                report.batch_hash,
                _iso(exec_ts),
                report.statements,
                len(report.queries),
                _iso(after),
            ),
        )
        if after is not None:
            conn.execute(_UPSERT_WATERMARK, (source, _iso(after), _iso(exec_ts)))
    logger.info(
        "mined past queries",
        extra={
            "source": source,
            "statements": report.statements,
            "fingerprints": len(report.queries),
            "failed": report.failed,
        },
    )
    return report.model_copy(update={"watermark_before": before, "watermark_after": after})


def _seen_up_to(queries: Iterable[RawQuery]) -> datetime | None:
    """The newest ``last_seen`` in the batch, parsed or not: the watermark is what was *seen*.

    A statement we could not parse still came from the source, so the next pull must not fetch it
    again (and must not re-report the same failure forever).
    """
    seen = [query.last_seen for query in queries if query.last_seen is not None]
    return max(seen) if seen else None


def read(
    store: Store,
    *,
    exec_ts: datetime,
    source: str | None = None,
    weights: MiningWeights = DEFAULT_WEIGHTS,
    limit: int | None = None,
) -> tuple[MinedQuery, ...]:
    """The stored candidates, rescored at ``exec_ts`` and ranked: what the Context Pack reads.

    ``source`` narrows to one source; without it every source is ranked together, which is the only
    place a *cross-source* pattern can surface (the MCP's own history).
    """
    sql = _SELECT_ROWS
    params: tuple[object, ...] = ()
    if source is not None:
        sql = f"{_SELECT_ROWS} WHERE source = ?"
        params = (source,)
    rows = store.connect().execute(sql, params).fetchall()
    queries = [
        MinedQuery(
            source=str(row["source"]),
            fingerprint=str(row["fingerprint"]),
            dialect=str(row["dialect"]),
            calls=int(row["calls"]),
            statements=int(row["statements"]),
            users=_stored_users(row["users"]),
            first_seen=_parse_iso(row["first_seen"]),
            last_seen=_parse_iso(row["last_seen"]),
            facts=QueryFacts.model_validate_json(str(row["facts"])),
        )
        for row in rows
    ]
    ranked = rank(queries, exec_ts=exec_ts, weights=weights)
    return ranked if limit is None else ranked[:limit]
