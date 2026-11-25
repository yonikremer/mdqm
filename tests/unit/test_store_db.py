"""Tests for the metadata store (H01): migrations, writer discipline, WAL, and the FTS5 probe.

Table-driven where the behaviour is a table of cases (migration files, statement splitting, broken
migrations). Every test uses ``tmp_path`` so nothing depends on the OS or the working directory, and
nothing sleeps: the writer lock is exercised with real threads that must all succeed.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import cast

import pytest

from mdqm.store import (
    FTS5_PROBE_TABLE,
    Fts5UnavailableError,
    Migration,
    MigrationError,
    Store,
    StoreError,
    check_fts5,
    fts5_available,
    load_migrations,
    require_fts5,
    split_statements,
)

SHIPPED = load_migrations()
TARGET = max(m.version for m in SHIPPED)


def open_store(tmp_path: Path) -> Store:
    """A store on a throwaway file. Callers close it in ``finally`` so the file can be deleted."""
    return Store(tmp_path / "meta.sqlite")


def table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {str(row[0]) for row in rows}


def schema_dump(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """The DDL a DB actually holds, so "fresh" and "upgraded" can be compared as text."""
    rows = conn.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
    ).fetchall()
    return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]


def applied_rows(conn: sqlite3.Connection) -> list[tuple[int, str, str]]:
    rows = conn.execute(
        "SELECT version, name, applied_at FROM schema_migrations ORDER BY version"
    ).fetchall()
    return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]


# ---------------------------------------------------------------------------
# migrations: fresh DB, idempotent start, upgraded == fresh
# ---------------------------------------------------------------------------


def test_fresh_db_applies_every_migration(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    try:
        report = store.initialize()
        assert report.applied == tuple(m.version for m in SHIPPED)
        assert report.version == TARGET
        assert report.already_current is False
        assert store.applied_versions() == tuple(m.version for m in SHIPPED)
        assert store.schema_version() == TARGET
        assert "schema_migrations" in table_names(store.connect())
    finally:
        store.close()


def test_second_start_is_a_no_op(tmp_path: Path) -> None:
    first = open_store(tmp_path)
    try:
        assert first.initialize().already_current is False
        before = applied_rows(first.connect())
    finally:
        first.close()

    second = open_store(tmp_path)
    try:
        report = second.initialize()
        assert report.already_current is True
        assert report.applied == ()
        assert report.version == TARGET
        # Same rows *and* same timestamps: a re-run would have rewritten applied_at.
        assert applied_rows(second.connect()) == before
    finally:
        second.close()


def test_upgraded_db_has_the_same_schema_as_a_fresh_one(tmp_path: Path) -> None:
    extra_version = TARGET + 1
    extra = Migration.from_sql(
        extra_version, "extra_table", "CREATE TABLE extra_table (id INTEGER PRIMARY KEY);\n"
    )

    upgraded = Store(tmp_path / "upgraded.sqlite", migrations=SHIPPED)
    try:
        upgraded.initialize()
        assert upgraded.schema_version() == TARGET
    finally:
        upgraded.close()

    # The same file, opened by a build that knows one more migration: forward-only upgrade.
    step_by_step = Store(tmp_path / "upgraded.sqlite", migrations=(*SHIPPED, extra))
    try:
        report = step_by_step.initialize()
        assert report.applied == (extra_version,)
        assert report.version == extra_version
        upgraded_dump = schema_dump(step_by_step.connect())
    finally:
        step_by_step.close()

    fresh = Store(tmp_path / "fresh.sqlite", migrations=(*SHIPPED, extra))
    try:
        report = fresh.initialize()
        assert report.applied == (*[m.version for m in SHIPPED], extra_version)
        assert schema_dump(fresh.connect()) == upgraded_dump
    finally:
        fresh.close()


def test_a_newer_database_is_refused_instead_of_downgraded(tmp_path: Path) -> None:
    future_version = TARGET + 1
    future = Migration.from_sql(
        future_version, "from_the_future", "CREATE TABLE later (id INTEGER);\n"
    )

    newer = Store(tmp_path / "meta.sqlite", migrations=(*SHIPPED, future))
    try:
        newer.initialize()
    finally:
        newer.close()

    older = Store(tmp_path / "meta.sqlite", migrations=SHIPPED)
    try:
        with pytest.raises(StoreError, match="newer mdqm"):
            older.initialize()
        assert older.schema_version() == future_version
    finally:
        older.close()


# ---------------------------------------------------------------------------
# a failing migration rolls back and leaves the previous version intact
# ---------------------------------------------------------------------------

# Each script starts by creating a table that must NOT survive the rollback, then hits a real error.
BROKEN_SCRIPTS = [
    pytest.param(
        "CREATE TABLE half_done (id INTEGER PRIMARY KEY);\nINSERT INTO no_such_table VALUES (1);\n",
        id="insert-into-missing-table",
    ),
    pytest.param(
        "CREATE TABLE half_done (id INTEGER PRIMARY KEY);\nCREATE TABLE half_done (id INTEGER);\n",
        id="table-already-exists",
    ),
    pytest.param(
        "CREATE TABLE half_done (id INTEGER PRIMARY KEY);\n"
        "CREATE INDEX idx ON no_such_table (id);\n",
        id="index-on-missing-table",
    ),
]


@pytest.mark.parametrize("broken_sql", BROKEN_SCRIPTS)
def test_failed_migration_rolls_back(tmp_path: Path, broken_sql: str) -> None:
    broken_version = TARGET + 1
    base = Store(tmp_path / "meta.sqlite", migrations=SHIPPED)
    try:
        base.initialize()
    finally:
        base.close()

    broken = Migration.from_sql(broken_version, "broken", broken_sql)
    store = Store(tmp_path / "meta.sqlite", migrations=(*SHIPPED, broken))
    try:
        with pytest.raises(MigrationError) as raised:
            store.initialize()
        message = str(raised.value)
        assert f"{broken_version:04d}_broken" in message
        assert f"unchanged at version {TARGET}" in message

        # Previous version intact, nothing of the failed migration recorded or left behind.
        assert store.schema_version() == TARGET
        assert broken_version not in store.applied_versions()
        assert "half_done" not in table_names(store.connect())
    finally:
        store.close()

    # The DB is still usable: a fixed migration at the same version applies cleanly.
    fixed = Migration.from_sql(broken_version, "fixed", "CREATE TABLE fixed_up (id INTEGER);\n")
    repaired = Store(tmp_path / "meta.sqlite", migrations=(*SHIPPED, fixed))
    try:
        report = repaired.initialize()
        assert report.applied == (broken_version,)
        assert "fixed_up" in table_names(repaired.connect())
    finally:
        repaired.close()


def test_incomplete_migration_sql_is_rejected_before_it_runs() -> None:
    with pytest.raises(StoreError, match="incomplete statement"):
        Migration.from_sql(1, "truncated", "CREATE TABLE half_done (id INTEGER PRIMARY KEY)")


# ---------------------------------------------------------------------------
# loading and validating migration files
# ---------------------------------------------------------------------------

MIGRATION_BODIES = {
    "0001_first.sql": "CREATE TABLE first (id INTEGER);\n",
    "0002_second.sql": "CREATE TABLE second (id INTEGER);\n",
}


def write_migrations(directory: Path, files: dict[str, str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (directory / name).write_text(body, encoding="utf-8")
    return directory


LOAD_CASES = [
    pytest.param(
        {
            "0002_second.sql": MIGRATION_BODIES["0002_second.sql"],
            "0001_first.sql": MIGRATION_BODIES["0001_first.sql"],
        },
        (1, 2),
        None,
        id="sorted-by-version-not-by-filename-order",
    ),
    pytest.param({"0001_first.sql": MIGRATION_BODIES["0001_first.sql"]}, (1,), None, id="one-file"),
    pytest.param({"0001_bad-name.sql": "SELECT 1;\n"}, None, "NNNN_name.sql", id="hyphen-in-name"),
    pytest.param({"1_first.sql": "SELECT 1;\n"}, None, "NNNN_name.sql", id="too-few-digits"),
    pytest.param({"0001_First.sql": "SELECT 1;\n"}, None, "NNNN_name.sql", id="uppercase-name"),
    pytest.param(
        {"0001_first.sql": "SELECT 1;\n", "0001_other.sql": "SELECT 2;\n"},
        None,
        "two migrations claim version 0001",
        id="duplicate-version",
    ),
    pytest.param({"0001_first.txt": "SELECT 1;\n"}, (), None, id="non-sql-files-ignored"),
]


@pytest.mark.parametrize(("files", "expected_versions", "error"), LOAD_CASES)
def test_load_migrations(
    tmp_path: Path,
    files: dict[str, str],
    expected_versions: tuple[int, ...] | None,
    error: str | None,
) -> None:
    directory = write_migrations(tmp_path / "migrations", files)
    if error is not None:
        with pytest.raises(StoreError, match=error):
            load_migrations(directory)
        return
    assert tuple(m.version for m in load_migrations(directory)) == expected_versions


def test_shipped_migrations_load_in_order() -> None:
    versions = [m.version for m in SHIPPED]
    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions)
    assert all(m.statements for m in SHIPPED)


# ---------------------------------------------------------------------------
# statement splitting (comments, triggers, truncation)
# ---------------------------------------------------------------------------

SPLIT_CASES = [
    pytest.param("CREATE TABLE a (x INTEGER);\n", 1, None, id="one-statement"),
    pytest.param("-- a comment\nCREATE TABLE a (x INTEGER);\n", 1, None, id="leading-comment"),
    pytest.param("CREATE TABLE a (x INTEGER);\n-- trailing note\n", 1, None, id="trailing-comment"),
    pytest.param("-- nothing but a comment\n", 0, None, id="comment-only"),
    pytest.param("", 0, None, id="empty"),
    pytest.param(
        "CREATE TABLE a (x INTEGER);\nCREATE TABLE b (y INTEGER);\n", 2, None, id="two-statements"
    ),
    pytest.param(
        "CREATE TRIGGER t AFTER INSERT ON a BEGIN\n  DELETE FROM a;\nEND;\n",
        1,
        None,
        id="trigger-body",
    ),
    pytest.param("INSERT INTO a VALUES (';');\n", 1, None, id="semicolon-in-a-literal"),
    pytest.param(
        "CREATE TABLE a (x INTEGER)", None, "incomplete statement", id="missing-semicolon"
    ),
]


@pytest.mark.parametrize(("sql", "count", "error"), SPLIT_CASES)
def test_split_statements(sql: str, count: int | None, error: str | None) -> None:
    if error is not None:
        with pytest.raises(StoreError, match=error):
            split_statements(sql)
        return
    assert len(split_statements(sql)) == count


# ---------------------------------------------------------------------------
# WAL mode, one writer at a time, connection per thread
# ---------------------------------------------------------------------------


def test_wal_mode_and_foreign_keys(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    try:
        store.initialize()
        conn = store.connect()
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        store.close()


def test_writes_from_many_threads_all_land(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    threads = 8
    per_thread = 20
    failures: list[BaseException] = []

    def worker(worker_id: int) -> None:
        try:
            for index in range(per_thread):
                with store.writing() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO mdqm_meta (key, value, updated_at) "
                        "VALUES (?, ?, '2026-01-01T00:00:00+00:00')",
                        (f"k{worker_id}-{index}", "v"),
                    )
        except BaseException as exc:  # noqa: BLE001 - reported below, never swallowed
            failures.append(exc)

    try:
        store.initialize()
        workers = [threading.Thread(target=worker, args=(n,)) for n in range(threads)]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()
        assert failures == []
        rows = store.connect().execute("SELECT COUNT(*) FROM mdqm_meta").fetchone()[0]
        assert rows == threads * per_thread
    finally:
        store.close()


def test_writing_rolls_back_on_error(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    try:
        store.initialize()
        with pytest.raises(sqlite3.IntegrityError), store.writing() as conn:
            conn.execute("INSERT INTO mdqm_meta (key, value, updated_at) VALUES ('k', 'v', 'now')")
            conn.execute("INSERT INTO mdqm_meta (key, value, updated_at) VALUES ('k', 'v', 'now')")
        rows = store.connect().execute("SELECT COUNT(*) FROM mdqm_meta").fetchone()[0]
        assert rows == 0
    finally:
        store.close()


def test_store_creates_missing_directory(tmp_path: Path) -> None:
    store = Store(tmp_path / "deep" / "nested" / "meta.sqlite")
    try:
        store.initialize()
        assert (tmp_path / "deep" / "nested" / "meta.sqlite").is_file()
    finally:
        store.close()


# ---------------------------------------------------------------------------
# FTS5: probed, reported, never assumed
# ---------------------------------------------------------------------------


class NoFts5Connection:
    """Behaves like a Python build whose SQLite has no FTS5 module."""

    def execute(self, sql: str, parameters: object = ()) -> object:
        raise sqlite3.OperationalError("no such module: fts5")


def test_fts5_absent_is_reported_not_crashed() -> None:
    stub = cast(sqlite3.Connection, NoFts5Connection())
    status = check_fts5(stub)
    assert status.available is False
    assert "no such module: fts5" in status.detail
    assert "sqlite3" in status.detail  # names the build, so the reader can fix it
    assert status.capability().supported is False
    with pytest.raises(Fts5UnavailableError, match="FTS5 is unavailable"):
        require_fts5(stub)


def test_fts5_smoke_test_on_this_build() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        status = check_fts5(conn)
        assert status.detail
        if not status.available:
            # Honest on a build without FTS5: reported, not crashed, and the message says why.
            assert "FTS5 is unavailable" in status.detail
            assert fts5_available(conn) is False
            with pytest.raises(Fts5UnavailableError):
                require_fts5(conn)
            return

        assert status.capability().supported is True
        require_fts5(conn)  # does not raise
        # The probe cleans up after itself: nothing left in temp.
        left_overs = conn.execute(
            "SELECT name FROM sqlite_master WHERE name = ?", (FTS5_PROBE_TABLE,)
        ).fetchall()
        assert left_overs == []

        # Hebrew and English both index and match (AGENTS.md section 5).
        conn.execute("CREATE VIRTUAL TABLE docs USING fts5(body)")
        conn.executemany(
            "INSERT INTO docs (body) VALUES (?)",
            [("average credit limit by status",), ("מסגרת אשראי ממוצעת לפי סטטוס",)],
        )
        english = conn.execute("SELECT body FROM docs WHERE docs MATCH ?", ("credit",)).fetchall()
        hebrew = conn.execute("SELECT body FROM docs WHERE docs MATCH ?", ("אשראי",)).fetchall()
        assert len(english) == 1
        assert len(hebrew) == 1
    finally:
        conn.close()


def test_fts5_available_needs_no_connection() -> None:
    # The probe may ask before any source or store is opened.
    assert fts5_available() in (True, False)
    assert isinstance(check_fts5().detail, str)


def test_store_exposes_the_fts5_verdict(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    try:
        assert store.fts5_available() is fts5_available()
        if store.fts5_available():
            store.require_fts5()
    finally:
        store.close()


# ---------------------------------------------------------------------------
# secrets never live in the metadata DB (principle 11)
# ---------------------------------------------------------------------------

CREDENTIAL_WORDS = ("secret", "password", "passwd", "token", "api_key", "apikey", "credential")


def test_schema_has_no_credential_columns(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    try:
        store.initialize()
        conn = store.connect()
        offenders: list[str] = []
        for name in sorted(table_names(conn)):
            for column in conn.execute(f'PRAGMA table_info("{name}")').fetchall():
                column_name = str(column[1]).lower()
                if any(word in column_name for word in CREDENTIAL_WORDS):
                    offenders.append(f"{name}.{column_name}")
        assert offenders == []
    finally:
        store.close()
