"""The metadata store: one SQLite file for history, audit, proposals and the FTS index.

Why SQLite: the metadata DB is small, single-instance (DESIGN section 14) and must work air-gapped,
so stdlib ``sqlite3`` is the smallest thing that can hold it. This module owns only the plumbing:

* **Migrations, forward-only and idempotent.** Numbered ``NNNN_name.sql`` files under
  ``store/migrations/`` are applied in ascending order and recorded in ``schema_migrations``.
  Each migration is one transaction, so a failing migration leaves the previous version intact and
  the DB usable; the DB is never opened at a version newer than this build knows (a downgrade would
  silently lose columns), that is refused instead. A fresh DB and an upgraded DB end up with the
  same schema: the same files ran, only the number of runs differs.
* **One writer at a time** (``writing()``): SQLite has a single writer anyway, so serialising writes
  in-process turns lock contention into a queue instead of ``database is locked`` errors. One
  connection per thread (``connect()``) because a SQLite connection must not hop between threads
  while a statement is open.
* **WAL mode**, so a reader never blocks the writer.
* **FTS5 is probed, never assumed** (:func:`check_fts5`). Some Python builds ship a SQLite without
  the FTS5 module; the probe reports that as a capability instead of crashing, and a feature that
  needs FTS5 calls :func:`require_fts5` to fail once, with the build named in the message.

Secrets never go here: the store holds identifiers, timestamps and audit hashes, not credentials
(principle 11). Credentials live in the secret provider (``mdqm.config.secrets``).

Migration authoring rule: **one statement per line, each ending with ``;``**. Statements are split
with :func:`sqlite3.complete_statement`, which reads a line at a time; a line holding two statements
fails loudly at apply time (``execute`` rejects multi-statement strings) rather than half-applying.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import sys
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from mdqm.models.capabilities import Capability

__all__ = [
    "FTS5_PROBE_TABLE",
    "MIGRATIONS_DIR",
    "MIGRATION_TABLE",
    "Fts5Status",
    "Fts5UnavailableError",
    "Migration",
    "MigrationError",
    "MigrationReport",
    "Store",
    "StoreError",
    "check_fts5",
    "fts5_available",
    "load_migrations",
    "require_fts5",
    "split_statements",
]

logger = logging.getLogger(__name__)

#: Directory holding the numbered ``NNNN_name.sql`` migrations (this package, not the CWD).
MIGRATIONS_DIR: Final[Path] = Path(__file__).resolve().parent / "migrations"

#: The table that records which migrations ran. Created by this module, never by a migration.
MIGRATION_TABLE: Final[str] = "schema_migrations"

#: Name of the throwaway FTS5 table the probe creates in ``temp``.
FTS5_PROBE_TABLE: Final[str] = "mdqm_fts5_probe"

#: Busy timeout for cross-process writers; the in-process writer lock queues our own threads.
_BUSY_TIMEOUT_MS: Final[int] = 5000

#: A PRAGMA cannot bind a value, so its one substitution happens here, at import time.
_PRAGMA_BUSY_TIMEOUT: Final[str] = f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}"

_MIGRATION_FILE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<version>\d{4})_(?P<name>[a-z0-9_]+)\.sql$"
)

_CREATE_MIGRATION_TABLE: Final[str] = (
    "CREATE TABLE IF NOT EXISTS schema_migrations ("
    " version INTEGER PRIMARY KEY,"
    " name TEXT NOT NULL,"
    " applied_at TEXT NOT NULL)"
)

#: One row per applied migration, keyed by version; values are bound by the caller.
# S608: the only substitution is the module constant above, never caller input.
_INSERT_MIGRATION: Final[str] = (
    f"INSERT INTO {MIGRATION_TABLE} (version, name, applied_at) VALUES (?, ?, ?)"  # noqa: S608
)

# S608: same reason.
_SELECT_MIGRATION_VERSIONS: Final[str] = f"SELECT version FROM {MIGRATION_TABLE} ORDER BY version"  # noqa: S608


# FTS5 probe statements. The only substitution anywhere in this module is the probe table's own
# constant name (S608 is about caller input, and this module takes none into SQL).
def _fts5_sql(template: str) -> str:
    return template.format(table=FTS5_PROBE_TABLE)  # noqa: S608


_FTS5_CREATE: Final[str] = _fts5_sql("CREATE VIRTUAL TABLE temp.{table} USING fts5(body)")
_FTS5_INSERT: Final[str] = _fts5_sql("INSERT INTO temp.{table}(body) VALUES (?)")
_FTS5_SEARCH: Final[str] = _fts5_sql("SELECT body FROM temp.{table} WHERE {table} MATCH ?")
_FTS5_DROP: Final[str] = _fts5_sql("DROP TABLE IF EXISTS temp.{table}")

# Hebrew and English both matter (AGENTS.md section 5), so the probe indexes one of each: a build
# whose SQLite lacks the unicode61 tokenizer fails here instead of in the Context Pack.
_FTS5_ROWS: Final[tuple[str, str]] = ("credit limit", "מסגרת אשראי")


class StoreError(RuntimeError):
    """A metadata-store problem the caller must not ignore (bad migration, downgrade attempt)."""


class MigrationError(StoreError):
    """A migration failed and was rolled back; the schema is unchanged at the previous version."""


class Fts5UnavailableError(StoreError):
    """This Python build's SQLite has no usable FTS5 module."""


def _new_connection(target: str) -> sqlite3.Connection:
    """Open one connection with the pragmas this store relies on (WAL, FK, busy timeout)."""
    conn = sqlite3.connect(
        target,
        isolation_level=None,  # autocommit: transactions are explicit, see _transaction()
        timeout=_BUSY_TIMEOUT_MS / 1000,
    )
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # WAL on a file DB; ":memory:" stays "memory"
        conn.execute("PRAGMA synchronous=NORMAL")  # the pairing recommended for WAL
        conn.execute("PRAGMA foreign_keys=ON")
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
        # A PRAGMA cannot bind a value; the substituted value is a module constant.
        conn.execute(_PRAGMA_BUSY_TIMEOUT)
    except sqlite3.Error:
        conn.close()
        raise
    return conn


def _has_sql(text: str) -> bool:
    """True if ``text`` holds anything but whitespace and ``--`` comments."""
    lines = (line.strip() for line in text.splitlines())
    return any(line and not line.startswith("--") for line in lines)


def split_statements(sql: str) -> tuple[str, ...]:
    """Split a migration script into single statements, ready for ``execute()``.

    Uses :func:`sqlite3.complete_statement` so a ``BEGIN ... END`` trigger body or a ``;`` inside a
    string literal is not mistaken for a statement boundary. A trailing fragment that is not a
    complete statement is a bug in the migration (it would otherwise be skipped silently) and
    raises.
    """
    statements: list[str] = []
    buffer = ""
    for line in sql.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statements.append(buffer.strip())
            buffer = ""
    if _has_sql(buffer):
        raise StoreError(
            "migration SQL ends with an incomplete statement (a missing trailing ';'): "
            f"{buffer.strip()[:200]!r}"
        )
    return tuple(statements)


@dataclass(frozen=True)
class Migration:
    """One forward-only schema step: a version, a name, and the statements that apply it."""

    version: int
    name: str
    statements: tuple[str, ...]

    @classmethod
    def from_sql(cls, version: int, name: str, sql: str) -> Migration:
        """Build a migration from a script, splitting it into statements."""
        return cls(version=version, name=name, statements=split_statements(sql))

    @property
    def label(self) -> str:
        """``0001_initial``: how the migration is named in logs and errors."""
        return f"{self.version:04d}_{self.name}"


@dataclass(frozen=True)
class MigrationReport:
    """What :meth:`Store.initialize` did: which versions it applied, and the resulting version."""

    applied: tuple[int, ...]
    version: int

    @property
    def already_current(self) -> bool:
        """True when the DB was already at this build's version (the common startup case)."""
        return not self.applied


def load_migrations(directory: Path | None = None) -> tuple[Migration, ...]:
    """Read every ``NNNN_name.sql`` under ``directory`` (default :data:`MIGRATIONS_DIR`).

    A ``*.sql`` file that does not match the naming pattern raises instead of being skipped: a
    typo would otherwise mean a migration that silently never runs.
    """
    root = directory if directory is not None else MIGRATIONS_DIR
    if not root.is_dir():
        raise StoreError(f"migrations directory does not exist: {root}")
    migrations: list[Migration] = []
    malformed: list[str] = []
    for path in sorted(root.glob("*.sql")):
        match = _MIGRATION_FILE.match(path.name)
        if match is None:
            malformed.append(path.name)
            continue
        migrations.append(
            Migration.from_sql(
                version=int(match.group("version")),
                name=match.group("name"),
                sql=path.read_text(encoding="utf-8"),
            )
        )
    if malformed:
        raise StoreError(
            "migration files must be named NNNN_name.sql (four digits, lowercase name): "
            + ", ".join(malformed)
        )
    return _validated(migrations, source=str(root))


def _validated(migrations: Sequence[Migration], *, source: str) -> tuple[Migration, ...]:
    """Sort by version and reject duplicates: two migrations with one version cannot both run."""
    ordered = tuple(sorted(migrations, key=lambda m: m.version))
    seen: set[int] = set()
    for migration in ordered:
        if migration.version in seen:
            raise StoreError(
                f"{source}: two migrations claim version {migration.version:04d}; "
                "renumber one of them"
            )
        if migration.version <= 0:
            raise StoreError(
                f"{source}: migration versions must be positive, got {migration.version}"
            )
        seen.add(migration.version)
    return ordered


class Store:
    """The metadata SQLite file: connections per thread, migrations, one writer lock.

    Not a singleton and holds no global state: pass the path (``Settings.paths.metadata_db``) and,
    in tests, the migration list. Thread-safe: :meth:`connect` is thread-local and writes are
    serialised by :meth:`writing`.
    """

    def __init__(self, path: Path | str, *, migrations: Sequence[Migration] | None = None) -> None:
        self.path = Path(path)
        self._migrations = tuple(migrations) if migrations is not None else load_migrations()
        self._migrations = _validated(self._migrations, source=str(self.path))
        self._write_lock = threading.Lock()
        self._connections_lock = threading.Lock()
        self._connections: list[sqlite3.Connection] = []
        self._local = threading.local()

    @property
    def migrations(self) -> tuple[Migration, ...]:
        """The migrations this store knows, ascending by version."""
        return self._migrations

    @property
    def target_version(self) -> int:
        """The version a fully migrated DB has under this build (0 when there are no migrations)."""
        return max((m.version for m in self._migrations), default=0)

    def connect(self) -> sqlite3.Connection:
        """The calling thread's connection, opened on first use and configured with the pragmas."""
        cached: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if cached is not None:
            return cached
        self._ensure_parent_dir()
        conn = _new_connection(str(self.path))
        self._local.conn = conn
        with self._connections_lock:
            self._connections.append(conn)
        return conn

    def close(self) -> None:
        """Close every connection this store opened. Safe to call twice."""
        with self._connections_lock:
            connections, self._connections = self._connections, []
        self._local = threading.local()
        for conn in connections:
            try:
                conn.close()
            except sqlite3.Error:  # already closed: nothing left to release
                logger.debug("closing a metadata connection failed", exc_info=True)

    def _ensure_parent_dir(self) -> None:
        """Create the DB's directory; a missing parent is a startup problem, not an SQL error."""
        if str(self.path) == ":memory:":
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """An immediate transaction that rolls back on any exception (write lock already held)."""
        conn = self.connect()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:  # SQLite already unwound the transaction
                logger.debug("no transaction left to roll back", exc_info=True)
            raise
        else:
            conn.execute("COMMIT")

    @contextmanager
    def writing(self) -> Iterator[sqlite3.Connection]:
        """One writer at a time, in a transaction that commits on success and rolls back on error.

        Not re-entrant: a nested ``writing()`` in the same thread would deadlock on the writer lock.
        """
        with self._write_lock, self._transaction() as conn:
            yield conn

    def initialize(self) -> MigrationReport:
        """Bring the DB to this build's version and report what was applied.

        Idempotent: a DB already at the target version is only read (``already_current`` is True).
        Refuses a DB written by a newer build instead of running against a schema it does not know.
        """
        with self._write_lock:
            conn = self.connect()
            with self._transaction() as tx:
                tx.execute(_CREATE_MIGRATION_TABLE)
            applied = self.applied_versions()
            current = max(applied, default=0)
            if current > self.target_version:
                raise StoreError(
                    f"{self.path} is at schema version {current}, but this build knows only "
                    f"{self.target_version}: it was written by a newer mdqm. Upgrade mdqm instead "
                    "of opening it with an older one."
                )
            pending = [m for m in self._migrations if m.version > current]
            for migration in pending:
                self._apply(conn, migration)
                current = migration.version
            if pending:
                logger.info(
                    "metadata store migrated",
                    extra={"path": str(self.path), "version": current, "applied": len(pending)},
                )
            return MigrationReport(applied=tuple(m.version for m in pending), version=current)

    def _apply(self, conn: sqlite3.Connection, migration: Migration) -> None:
        """Run one migration in its own transaction, with the recorded row in the same commit."""
        try:
            with self._transaction() as tx:
                for statement in migration.statements:
                    # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
                    # The statements come from this package's own migration files, not a caller.
                    tx.execute(statement)
                tx.execute(
                    _INSERT_MIGRATION,
                    (migration.version, migration.name, datetime.now(UTC).isoformat()),
                )
        except sqlite3.Error as exc:
            raise MigrationError(
                f"migration {migration.label} failed and was rolled back ({exc}). "
                f"The schema is unchanged at version {self.schema_version()}."
            ) from exc

    def applied_versions(self) -> tuple[int, ...]:
        """Versions recorded as applied, ascending. Empty when the DB has never been migrated."""
        conn = self.connect()
        if not self._has_migration_table(conn):
            return ()
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
        # Module-constant SQL, values bound.
        rows = conn.execute(_SELECT_MIGRATION_VERSIONS).fetchall()
        return tuple(int(row["version"]) for row in rows)

    def schema_version(self) -> int:
        """The highest applied migration version, or 0 for a DB that has never been migrated."""
        return max(self.applied_versions(), default=0)

    def _has_migration_table(self, conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (MIGRATION_TABLE,)
        ).fetchone()
        return row is not None

    def fts5_available(self) -> bool:
        """Whether this build's SQLite can serve the FTS index (see :func:`check_fts5`)."""
        return check_fts5(self.connect()).available

    def require_fts5(self) -> None:
        """Raise :class:`Fts5UnavailableError` when the FTS index cannot be built here."""
        require_fts5(self.connect())


@dataclass(frozen=True)
class Fts5Status:
    """The probe result: is FTS5 usable, and what exactly did the build say."""

    available: bool
    detail: str

    def capability(self) -> Capability:
        """The verdict in the shape the capability map stores (DESIGN section 4)."""
        return Capability(supported=self.available, detail=self.detail)


def _build_description() -> str:
    """Which interpreter and SQLite answered, so a missing FTS5 can be fixed by the reader."""
    return f"sqlite3 {sqlite3.sqlite_version} in Python {sys.version.split()[0]} ({sys.executable})"


def _fts5_missing(exc: BaseException) -> Fts5Status:
    return Fts5Status(
        available=False,
        detail=(
            f"FTS5 is unavailable in this Python build: {exc} [{_build_description()}]. "
            "Install a Python whose SQLite is compiled with FTS5 (the official CPython "
            "installers are) or rebuild SQLite with --enable-fts5; until then features that need "
            "full-text search must use their declared fallback."
        ),
    )


def check_fts5(conn: sqlite3.Connection | None = None) -> Fts5Status:
    """Smoke-test FTS5: create a ``temp`` virtual table, index two rows, search them, drop it.

    Absence is *reported*, never raised, so the capability map can record it and the feature can
    fall back. The probe uses ``temp`` so nothing is written to the metadata file.
    """
    own = conn is None
    target = sqlite3.connect(":memory:") if conn is None else conn
    try:
        target.execute(_FTS5_CREATE)
        for row in _FTS5_ROWS:
            target.execute(_FTS5_INSERT, (row,))
        hits = target.execute(_FTS5_SEARCH, ("credit",)).fetchall()
        if len(hits) != 1:
            return _fts5_missing(
                RuntimeError(f"an FTS5 search for 'credit' returned {len(hits)} rows, expected 1")
            )
        return Fts5Status(available=True, detail=f"FTS5 works [{_build_description()}]")
    except sqlite3.Error as exc:
        return _fts5_missing(exc)
    finally:
        if not own:
            try:
                target.execute(_FTS5_DROP)
            except sqlite3.Error:  # the create may have failed: nothing to drop
                logger.debug("dropping the FTS5 probe table failed", exc_info=True)
        else:
            target.close()


def fts5_available(conn: sqlite3.Connection | None = None) -> bool:
    """True when this build's SQLite can create and query an FTS5 table."""
    return check_fts5(conn).available


def require_fts5(conn: sqlite3.Connection | None = None) -> None:
    """Fail once, clearly, where FTS5 is genuinely required (Context Pack, history search)."""
    status = check_fts5(conn)
    if not status.available:
        raise Fts5UnavailableError(status.detail)
