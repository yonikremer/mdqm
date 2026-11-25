"""Content-addressed cache for model answers: the same input costs nothing the second time.

The key is a SHA-256 of the *input* (endpoint, model, prompt version, prompt, system, schema), so
a changed prompt or schema misses instead of returning a stale answer. Only the endpoint's raw
JSON text is stored; every read re-validates it against the caller's schema, so a schema change
can never smuggle an old shape into a new type.

Storage is stdlib ``sqlite3`` in one file: small, air-gapped, and safe for several threads and
processes (WAL). The metadata store (``mdqm.store``) is a different database with its own
migrations: this table belongs to the LLM lane only, so it is created here on first use.

Rows hold questions and answers, never credentials, so the key is not secret material.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from mdqm.models.base import MDQMModel, UtcDatetime

__all__ = ["CACHE_FILE_NAME", "CacheEntry", "ResponseCache"]

_logger = logging.getLogger(__name__)

#: Default file name under ``Settings.paths.cache_dir``.
CACHE_FILE_NAME = "llm_cache.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    fetched_at TEXT NOT NULL
)
"""


class CacheEntry(MDQMModel):
    """One stored answer: the raw JSON text the endpoint returned, and when."""

    content: str
    model: str
    prompt_version: str
    fetched_at: UtcDatetime


class ResponseCache:
    """A SQLite-backed store of model answers, keyed by the input hash."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """The SQLite file; its directory is created on first use."""
        return self._path

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(_SCHEMA)
        return connection

    def get(self, key: str) -> CacheEntry | None:
        """The stored answer for ``key``, or ``None``. A corrupt row reads as a miss."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT content, model, prompt_version, fetched_at FROM llm_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        try:
            return CacheEntry(
                content=row[0], model=row[1], prompt_version=row[2], fetched_at=row[3]
            )
        except ValidationError:
            # A truncated or hand-edited row must not fail a call; it is a miss and is reported.
            _logger.warning(
                "llm cache row is unreadable; treating it as a miss", extra={"key": key[:12]}
            )
            return None

    def put(
        self,
        key: str,
        *,
        content: str,
        model: str,
        prompt_version: str,
        fetched_at: datetime,
    ) -> None:
        """Store one answer under ``key`` (replacing an older answer for the same input)."""
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO llm_cache (key, content, model, prompt_version, fetched_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (key, content, model, prompt_version, fetched_at.isoformat()),
            )
            connection.commit()

    def __repr__(self) -> str:
        return f"ResponseCache(path={str(self._path)!r})"
