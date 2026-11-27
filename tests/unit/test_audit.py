"""Tests for the hash-chained audit log (H02).

The three claims the MVP depends on get one test each, and each test tampers with a real SQLite file
rather than mocking the store: an edit is caught at the right id, a legal retention purge leaves a
verifiable chain, and a guard-refused ``DELETE`` leaves a row with the code the agent saw.

Nothing sleeps: the clock is injected, so ``ts`` and chain order are deterministic. Every test uses
``tmp_path``, so no test depends on the working directory or the OS.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import pytest

from mdqm.errors import ErrorCode
from mdqm.store import Store
from mdqm.store.audit import (
    AUDIT_CHECKPOINT_TABLE,
    AUDIT_LOG_TABLE,
    DEFAULT_RETENTION_DAYS,
    GENESIS_HASH,
    HEAD_KEY,
    AuditEntry,
    AuditError,
    AuditLog,
    AuditOutcome,
)

# ---------------------------------------------------------------------------
# fixtures: a real migrated file, a scripted clock
# ---------------------------------------------------------------------------


class FakeClock:
    """A clock the test advances by hand: no sleeping, no wall-clock coupling."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: int) -> datetime:
        self.now += timedelta(**kwargs)
        return self.now


BASE = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(BASE)


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
def audit(store: Store, clock: FakeClock) -> AuditLog:
    return AuditLog(store, clock=clock)


def ok_entry(**overrides: object) -> AuditEntry:
    """A representative success row; overrides let a test vary one field."""
    fields: dict[str, object] = {
        "user_id": "yoni",
        "tool": "run_query",
        "outcome": AuditOutcome.OK,
        "version_hash": "abc123",
        "sources": ("crm", "billing"),
        "row_count": 12,
    }
    fields.update(overrides)
    # `model_validate` takes the mapping as data, which is what this helper builds: a
    # splat would make pyright (rightly) ask which keyword arguments exist.
    return AuditEntry.model_validate(fields)


def fill(audit: AuditLog, count: int, clock: FakeClock, **overrides: object) -> None:
    """Append ``count`` success rows, advancing the clock one minute apart."""
    for index in range(count):
        audit.record(ok_entry(**{"row_count": index, **overrides}))
        clock.advance(minutes=1)


# ---------------------------------------------------------------------------
# the chain: genesis, linkage, verdict
# ---------------------------------------------------------------------------


def test_first_row_links_to_genesis(audit: AuditLog) -> None:
    record = audit.record(ok_entry())
    assert record.id == 1
    assert record.prev_hash == GENESIS_HASH
    assert len(record.hash) == 64
    assert record.hash != GENESIS_HASH


def test_each_row_links_to_the_previous(audit: AuditLog, clock: FakeClock) -> None:
    fill(audit, 4, clock)
    rows = audit.rows()
    assert [row.id for row in rows] == [1, 2, 3, 4]
    assert rows[0].prev_hash == GENESIS_HASH
    for previous, current in pairwise(rows):
        assert current.prev_hash == previous.hash


def test_verify_chain_accepts_a_clean_log(audit: AuditLog, clock: FakeClock) -> None:
    fill(audit, 5, clock)
    verdict = audit.verify_chain()
    assert verdict.ok, verdict.describe()
    assert verdict.checked == 5
    assert verdict.first_bad_id is None


def test_verify_chain_accepts_an_empty_log(audit: AuditLog) -> None:
    verdict = audit.verify_chain()
    assert verdict.ok
    assert verdict.checked == 0


def test_verify_chain_is_deterministic_and_repeatable(audit: AuditLog, clock: FakeClock) -> None:
    fill(audit, 3, clock)
    assert audit.verify_chain() == audit.verify_chain()


@pytest.mark.parametrize("count", [1, 2, 3, 7, 25])
def test_chain_verifies_at_every_length(audit: AuditLog, clock: FakeClock, count: int) -> None:
    fill(audit, count, clock)
    assert audit.verify_chain().ok


# ---------------------------------------------------------------------------
# tampering: an edit is caught at the right id
# ---------------------------------------------------------------------------


def tamper(store: Store, sql: str, params: tuple[object, ...] = ()) -> None:
    """Apply an out-of-band edit, dropping the append-only triggers the way an attacker would.

    The triggers live in the database, so a test that wants to prove the *chain* detects tampering
    has to get past them first; dropping them is exactly what someone with write access to the file
    can do, and it is why the chain, not the trigger, is the guarantee. They are restored afterwards
    so the store is left in its shipped shape and a later purge still finds its trigger.
    """
    conn = store.connect()
    guards = [
        str(row["sql"])
        for row in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger'"
            " AND name IN ('audit_log_no_update', 'audit_log_no_delete')"
        )
    ]
    conn.execute("DROP TRIGGER IF EXISTS audit_log_no_update")
    conn.execute("DROP TRIGGER IF EXISTS audit_log_no_delete")
    try:
        conn.execute(sql, params)
    finally:
        for ddl in guards:
            conn.execute(ddl)


def test_append_only_triggers_block_an_update(store: Store, audit: AuditLog) -> None:
    """The ordinary case is stopped before the chain has to catch it."""
    audit.record(ok_entry())
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.connect().execute(f"UPDATE {AUDIT_LOG_TABLE} SET row_count = 999 WHERE id = 1")


def test_append_only_triggers_block_a_delete(store: Store, audit: AuditLog) -> None:
    audit.record(ok_entry())
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.connect().execute(f"DELETE FROM {AUDIT_LOG_TABLE} WHERE id = 1")


@pytest.mark.parametrize(
    "column, value",
    [
        ("row_count", 999),
        ("blocked_attempts", 5),
        ("user_id", "someone-else"),
        ("tool", "delete_rows"),
        ("outcome", "error"),
        ("error_code", "TIMEOUT"),
        ("version_hash", "tampered"),
        ("sources", '["erp"]'),
        ("ts", "2020-01-01T00:00:00+00:00"),
    ],
)
def test_tampering_with_a_field_breaks_the_chain_at_that_id(
    store: Store, audit: AuditLog, clock: FakeClock, column: str, value: object
) -> None:
    """A single-column edit is caught, and the verdict names the edited row."""
    fill(audit, 5, clock)
    tamper(store, f"UPDATE {AUDIT_LOG_TABLE} SET {column} = ? WHERE id = 3", (value,))
    verdict = audit.verify_chain()
    assert not verdict.ok
    assert verdict.first_bad_id == 3
    assert "edited" in (verdict.problem or "")
    assert "3" in verdict.describe()


def test_deleting_a_middle_row_is_caught_as_a_gap(
    store: Store, audit: AuditLog, clock: FakeClock
) -> None:
    fill(audit, 5, clock)
    tamper(store, f"DELETE FROM {AUDIT_LOG_TABLE} WHERE id = 4")
    verdict = audit.verify_chain()
    assert not verdict.ok
    assert verdict.first_bad_id == 5
    assert "deleted" in (verdict.problem or "")


def test_rewriting_a_row_and_its_successor_still_breaks_later(
    store: Store, audit: AuditLog, clock: FakeClock
) -> None:
    """Recomputing one row's hash is not enough: the next row still points at the old hash."""
    fill(audit, 4, clock)
    rows = audit.rows()
    tamper(store, f"UPDATE {AUDIT_LOG_TABLE} SET row_count = 999 WHERE id = 2")
    # Re-hash row 2 consistently with its forged payload, the way a careful attacker would.
    forged = rows[1].model_copy(update={"row_count": 999})
    from mdqm.store.audit import _digest

    tamper(
        store,
        f"UPDATE {AUDIT_LOG_TABLE} SET hash = ? WHERE id = 2",
        (_digest(forged.payload()),),
    )
    verdict = audit.verify_chain()
    assert not verdict.ok
    assert verdict.first_bad_id == 3


def test_truncating_the_tail_is_caught_by_the_mirrored_head(
    store: Store, audit: AuditLog, clock: FakeClock
) -> None:
    """Removing the newest rows leaves a self-consistent chain, so the head mirror is the proof."""
    fill(audit, 4, clock)
    tamper(store, f"DELETE FROM {AUDIT_LOG_TABLE} WHERE id >= 4")
    verdict = audit.verify_chain()
    assert not verdict.ok
    assert "tail" in (verdict.problem or "")


def test_truncating_every_row_is_caught(store: Store, audit: AuditLog, clock: FakeClock) -> None:
    fill(audit, 3, clock)
    tamper(store, f"DELETE FROM {AUDIT_LOG_TABLE}")
    verdict = audit.verify_chain()
    assert not verdict.ok


def test_a_corrupted_stored_hash_is_caught(audit: AuditLog, clock: FakeClock) -> None:
    """A hash that does not match the row's fields is caught, even at full length."""
    fill(audit, 2, clock)
    conn = audit._store.connect()
    conn.execute("DROP TRIGGER IF EXISTS audit_log_no_update")
    conn.execute(f"UPDATE {AUDIT_LOG_TABLE} SET hash = ? WHERE id = 1", ("f" * 64,))
    verdict = audit.verify_chain()
    assert not verdict.ok
    assert verdict.first_bad_id == 1


def test_a_short_stored_hash_never_reaches_the_chain(audit: AuditLog, clock: FakeClock) -> None:
    """The table's own CHECK refuses a hash that is not 64 hex characters."""
    fill(audit, 2, clock)
    conn = audit._store.connect()
    conn.execute("DROP TRIGGER IF EXISTS audit_log_no_update")
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
        conn.execute(f"UPDATE {AUDIT_LOG_TABLE} SET hash = 'not-a-hash' WHERE id = 1")


# ---------------------------------------------------------------------------
# refusal auditing: the MVP acceptance item
# ---------------------------------------------------------------------------


def test_a_blocked_delete_is_audited_with_the_agent_visible_code(
    audit: AuditLog, clock: FakeClock
) -> None:
    """The guard refuses the DELETE, and that refusal is what lands in the log."""
    from mdqm.sql.guard import guard

    result = guard("DELETE FROM customers WHERE cust_id = 1", "postgres")
    assert not result.ok
    assert result.error is not None

    record = audit.record_refusal(
        user_id="yoni", tool="run_query", error=result.error, version_hash="abc123"
    )

    assert record.outcome is AuditOutcome.BLOCKED
    assert record.blocked_attempts == 1
    assert record.error_code is ErrorCode.WRITE_BLOCKED
    assert record.row_count == 0
    assert record.version_hash == "abc123"
    assert audit.verify_chain().ok


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("DELETE FROM customers", ErrorCode.WRITE_BLOCKED),
        ("UPDATE customers SET status = 'A'", ErrorCode.WRITE_BLOCKED),
        ("DROP TABLE customers", ErrorCode.WRITE_BLOCKED),
        ("INSERT INTO customers (cust_id) VALUES (1)", ErrorCode.WRITE_BLOCKED),
        ("SELECT 1; SELECT 2", ErrorCode.MULTI_STATEMENT),
        ("SELECT FROM WHERE", ErrorCode.PARSE_ERROR),
    ],
)
def test_every_guard_refusal_reaches_the_audit(
    audit: AuditLog, clock: FakeClock, sql: str, expected: ErrorCode
) -> None:
    """Table-driven: each refusal kind is auditable, not just the DELETE the MVP names."""
    from mdqm.sql.guard import guard

    result = guard(sql, "postgres")
    assert not result.ok, f"the guard allowed {sql!r}"
    assert result.error is not None
    record = audit.record_refusal(user_id="yoni", tool="run_query", error=result.error)
    assert record.error_code is expected
    assert record.outcome is AuditOutcome.BLOCKED


def test_refusals_and_successes_are_countable_apart(audit: AuditLog, clock: FakeClock) -> None:
    from mdqm.sql.guard import guard

    fill(audit, 3, clock)
    blocked = guard("DELETE FROM customers", "postgres").error
    assert blocked is not None
    audit.record_refusal(user_id="yoni", tool="run_query", error=blocked)

    rows = audit.rows()
    assert sum(1 for row in rows if row.outcome is AuditOutcome.OK) == 3
    assert sum(1 for row in rows if row.outcome is AuditOutcome.BLOCKED) == 1
    assert sum(row.blocked_attempts for row in rows) == 1


def test_a_blocked_row_must_carry_an_attempt_and_a_code() -> None:
    with pytest.raises(ValueError, match="blocked_attempts"):
        AuditEntry(user_id="yoni", tool="run_query", outcome=AuditOutcome.BLOCKED)
    with pytest.raises(ValueError, match="error_code"):
        AuditEntry(
            user_id="yoni",
            tool="run_query",
            outcome=AuditOutcome.BLOCKED,
            blocked_attempts=1,
        )


# ---------------------------------------------------------------------------
# no secret, no payload: what must never be storable
# ---------------------------------------------------------------------------


SECRET_VALUES: tuple[str, ...] = (
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ5b25pIn0.c2lnbmF0dXJl",
    "Bearer abcdefghijklmnop",
    "-----BEGIN-RSA-PRIVATE-KEY-----",
    "AKIAIOSFODNN7EXAMPLE",  # pragma: allowlist secret
    "password=hunter2",
    "token:abc123",
    "postgresql://yoni:hunter2@db.internal/crm",
    "api_key=secret",
    "sup3rs3cr3t value",
    "SELECT * FROM customers",
    "cust_id=1",
)


@pytest.mark.parametrize("secret", SECRET_VALUES)
@pytest.mark.parametrize("field", ["user_id", "tool", "version_hash"])
def test_a_secret_cannot_enter_a_text_field(field: str, secret: str) -> None:
    """Each credential-shaped value is refused before it can reach the file."""
    with pytest.raises(ValueError):
        AuditEntry(
            **{"user_id": "yoni", "tool": "run_query", "outcome": AuditOutcome.OK, field: secret}
        )


@pytest.mark.parametrize("secret", SECRET_VALUES)
def test_a_secret_cannot_enter_a_source_name(secret: str) -> None:
    with pytest.raises(ValueError):
        ok_entry(sources=(secret,))


def test_a_refused_secret_is_never_echoed_in_the_error() -> None:
    """The refusal names the field, never the value: quoting it would leak it into the log."""
    secret = "AKIAIOSFODNN7EXAMPLE"  # pragma: allowlist secret
    with pytest.raises(ValueError) as caught:
        ok_entry(tool=secret)
    assert secret not in str(caught.value)


def test_the_entry_model_has_no_payload_field() -> None:
    """A statement or a result row cannot be stored: there is no field to put one in."""
    for forbidden in ("sql", "statement", "rows", "result", "body", "payload", "query"):
        with pytest.raises(ValueError):
            AuditEntry(
                user_id="yoni",
                tool="run_query",
                outcome=AuditOutcome.OK,
                # the unknown key is the thing under test
                **{forbidden: "anything"},  # pyright: ignore[reportCallIssue]
            )


def test_the_stored_row_holds_no_payload_column(store: Store) -> None:
    """Belt and braces: the table itself has no column a payload could occupy."""
    columns = {
        str(row["name"]) for row in store.connect().execute(f"PRAGMA table_info({AUDIT_LOG_TABLE})")
    }
    assert columns == {
        "id",
        "ts",
        "user_id",
        "tool",
        "version_hash",
        "sources",
        "row_count",
        "blocked_attempts",
        "outcome",
        "error_code",
        "prev_hash",
        "hash",
    }


def test_an_over_long_field_is_refused() -> None:
    with pytest.raises(ValueError):
        ok_entry(tool="t" * 500)


# ---------------------------------------------------------------------------
# retention purge: whole prefixes, checkpoint, still verifiable
# ---------------------------------------------------------------------------


def test_purge_keeps_the_chain_verifiable(audit: AuditLog, clock: FakeClock) -> None:
    """The acceptance item: rows age out, the chain still verifies from the checkpoint."""
    fill(audit, 10, clock)
    clock.advance(days=DEFAULT_RETENTION_DAYS + 1)

    report = audit.purge()

    assert report.purged
    assert report.purged_rows == 10
    assert report.kept_rows == 0
    verdict = audit.verify_chain()
    assert verdict.ok, verdict.describe()
    assert verdict.checkpoint_id == report.checkpoint_id


def test_purge_removes_only_the_aged_prefix(audit: AuditLog, clock: FakeClock) -> None:
    """Recent rows survive: retention removes what is past it, not the whole log."""
    fill(audit, 3, clock)  # old rows, at BASE, BASE+1m, BASE+2m
    clock.advance(days=DEFAULT_RETENTION_DAYS + 1)  # strictly past the first three
    fill(audit, 2, clock)  # fresh rows

    report = audit.purge()

    assert report.purged_rows == 3
    assert [row.id for row in audit.rows()] == [4, 5]
    assert report.kept_rows == 2
    assert audit.verify_chain().ok


def test_purge_leaves_the_chain_verifiable_with_survivors(
    audit: AuditLog, clock: FakeClock
) -> None:
    """The interesting case: a checkpoint plus a surviving chain, verified end to end."""
    fill(audit, 5, clock)
    clock.advance(days=DEFAULT_RETENTION_DAYS + 1)
    audit.record(ok_entry(row_count=99))

    report = audit.purge()
    assert report.purged_rows == 5

    verdict = audit.verify_chain()
    assert verdict.ok, verdict.describe()
    assert verdict.checked == 1
    assert verdict.checkpoint_id == 5

    survivors = audit.rows()
    assert [row.id for row in survivors] == [6]
    assert survivors[0].prev_hash == report.checkpoint_hash


def test_the_chain_continues_after_a_full_purge(audit: AuditLog, clock: FakeClock) -> None:
    """A new row appends onto the checkpoint, not onto the genesis hash."""
    fill(audit, 3, clock)
    clock.advance(days=DEFAULT_RETENTION_DAYS + 1)
    report = audit.purge()
    audit.record(ok_entry(row_count=7))

    rows = audit.rows()
    assert len(rows) == 1
    assert rows[0].id == 4
    assert rows[0].prev_hash == report.checkpoint_hash
    assert audit.verify_chain().ok


def test_purge_is_idempotent(audit: AuditLog, clock: FakeClock) -> None:
    fill(audit, 4, clock)
    clock.advance(days=DEFAULT_RETENTION_DAYS + 1)
    first = audit.purge()
    second = audit.purge()
    assert first.purged
    assert not second.purged
    assert second.purged_rows == 0
    assert second.checkpoint_id == first.checkpoint_id


def test_purge_of_a_clean_recent_log_removes_nothing(audit: AuditLog, clock: FakeClock) -> None:
    fill(audit, 3, clock)
    report = audit.purge()
    assert not report.purged
    assert report.kept_rows == 3
    assert audit.verify_chain().ok


def test_purge_restores_the_append_only_trigger(
    store: Store, audit: AuditLog, clock: FakeClock
) -> None:
    """Append-only must hold again the moment the purge returns."""
    fill(audit, 3, clock)
    clock.advance(days=DEFAULT_RETENTION_DAYS + 1)
    audit.purge()
    audit.record(ok_entry(row_count=1))  # the trigger fires per row, so the table needs one
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.connect().execute(f"DELETE FROM {AUDIT_LOG_TABLE}")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.connect().execute(f"UPDATE {AUDIT_LOG_TABLE} SET row_count = 9")


def test_purge_records_a_checkpoint_row(store: Store, audit: AuditLog, clock: FakeClock) -> None:
    fill(audit, 2, clock)
    clock.advance(days=DEFAULT_RETENTION_DAYS + 1)
    report = audit.purge()
    row = (
        store.connect()
        .execute(
            # S608: constant table name.
            f"SELECT last_purged_id, last_purged_hash FROM {AUDIT_CHECKPOINT_TABLE} WHERE id = 1"
        )
        .fetchone()
    )
    assert row is not None
    assert int(row["last_purged_id"]) == report.checkpoint_id
    assert str(row["last_purged_hash"]) == report.checkpoint_hash


def test_a_custom_retention_window_is_honoured(
    audit: AuditLog, clock: FakeClock, store: Store
) -> None:
    short = AuditLog(store, clock=clock, retention_days=1)
    short.record(ok_entry(row_count=1))
    clock.advance(days=2)
    short.record(ok_entry(row_count=2))
    report = short.purge()
    assert report.purged_rows == 1
    assert short.verify_chain().ok


def test_a_negative_retention_is_refused(store: Store) -> None:
    with pytest.raises(AuditError):
        AuditLog(store, retention_days=-1)


def test_purge_after_tampering_does_not_hide_it(
    store: Store, audit: AuditLog, clock: FakeClock
) -> None:
    """Purging an aged prefix must not launder an edit that sits in a surviving row."""
    fill(audit, 3, clock)  # aged rows
    clock.advance(days=DEFAULT_RETENTION_DAYS + 1)
    fill(audit, 3, clock)  # fresh rows: ids 4, 5, 6
    tamper(store, f"UPDATE {AUDIT_LOG_TABLE} SET row_count = 999 WHERE id = 5")

    report = audit.purge()

    assert report.purged_rows == 3
    verdict = audit.verify_chain()
    assert not verdict.ok
    assert verdict.first_bad_id == 5


# ---------------------------------------------------------------------------
# determinism, limits of the design, and the CLI gate
# ---------------------------------------------------------------------------


def test_the_same_inputs_give_the_same_hashes(tmp_path: Path) -> None:
    """Two logs fed identical rows with identical clocks must agree, or a chain is not a chain."""
    hashes: list[list[str]] = []
    for name in ("a.sqlite", "b.sqlite"):
        opened = Store(tmp_path / name)
        opened.initialize()
        try:
            log = AuditLog(opened, clock=FakeClock(BASE))
            for index in range(3):
                log.record(ok_entry(row_count=index))
            hashes.append([row.hash for row in log.rows()])
        finally:
            opened.close()
    assert hashes[0] == hashes[1]


def test_ids_are_contiguous_across_a_purge(audit: AuditLog, clock: FakeClock) -> None:
    """A gap is how a deletion is detected, so a legal purge must not create one."""
    fill(audit, 4, clock)
    clock.advance(days=DEFAULT_RETENTION_DAYS + 1)
    audit.purge()
    fill(audit, 2, clock)
    ids = [row.id for row in audit.rows()]
    assert ids == list(range(ids[0], ids[0] + len(ids)))


def test_the_head_mirror_is_written(store: Store, audit: AuditLog, clock: FakeClock) -> None:
    fill(audit, 2, clock)
    row = (
        store.connect().execute("SELECT value FROM mdqm_meta WHERE key = ?", (HEAD_KEY,)).fetchone()
    )
    assert row is not None
    assert str(row["value"]) == audit.rows()[-1].hash


def test_a_naive_clock_is_refused(store: Store) -> None:
    """A naive datetime would silently mean local time; the audit stores UTC only."""
    log = AuditLog(store, clock=lambda: datetime(2024, 1, 1))
    with pytest.raises(AuditError):
        log.record(ok_entry())


def test_rows_can_be_read_incrementally(audit: AuditLog, clock: FakeClock) -> None:
    fill(audit, 5, clock)
    assert [row.id for row in audit.rows(since_id=3)] == [4, 5]
    assert [row.id for row in audit.rows(limit=2)] == [1, 2]


def test_count_matches_the_rows(audit: AuditLog, clock: FakeClock) -> None:
    fill(audit, 4, clock)
    assert audit.count() == 4 == len(audit.rows())


def test_cli_audit_verify_exits_zero_on_a_clean_log(tmp_path: Path) -> None:
    """Acceptance: `mdqm audit verify` exits 0 when the chain holds."""
    from typer.testing import CliRunner

    from mdqm.cli import app

    path = tmp_path / "meta.sqlite"
    opened = Store(path)
    opened.initialize()
    try:
        AuditLog(opened, clock=FakeClock(BASE)).record(ok_entry())
    finally:
        opened.close()

    result = CliRunner().invoke(
        app, ["audit", "verify"], env={"MDQM_PATHS__METADATA_DB": str(path)}
    )
    assert result.exit_code == 0, result.output
    assert "intact" in result.output


def test_cli_audit_verify_exits_one_on_a_tampered_log(tmp_path: Path) -> None:
    """Acceptance: and exits 1, naming the first bad id, when a row was edited."""
    from typer.testing import CliRunner

    from mdqm.cli import app

    path = tmp_path / "meta.sqlite"
    opened = Store(path)
    opened.initialize()
    try:
        log = AuditLog(opened, clock=FakeClock(BASE))
        for index in range(3):
            log.record(ok_entry(row_count=index))
        tamper(opened, f"UPDATE {AUDIT_LOG_TABLE} SET row_count = 999 WHERE id = 2")
    finally:
        opened.close()

    result = CliRunner().invoke(
        app, ["audit", "verify"], env={"MDQM_PATHS__METADATA_DB": str(path)}
    )
    assert result.exit_code == 1, result.output
    assert "2" in result.output
