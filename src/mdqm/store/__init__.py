"""``mdqm.store``: the SQLite metadata database (history, audit, proposals, FTS index).

``db`` owns the file: connections per thread, forward-only numbered migrations, one writer lock, and
a probed FTS5 capability. Domain tables arrive through their own tasks and their own migrations.
"""

from __future__ import annotations

from mdqm.store.db import (
    FTS5_PROBE_TABLE,
    MIGRATION_TABLE,
    MIGRATIONS_DIR,
    Fts5Status,
    Fts5UnavailableError,
    Migration,
    MigrationError,
    MigrationReport,
    Store,
    StoreError,
    check_fts5,
    fts5_available,
    load_migrations,
    require_fts5,
    split_statements,
)

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
