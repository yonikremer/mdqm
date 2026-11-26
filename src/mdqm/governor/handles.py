"""The result store: a large result becomes an opaque, owned, expiring handle.

The agent is weak and pays per token, so it never receives a whole result set (``docs/DESIGN.md``
section 9: "result = summary + sample + handle"). The full Arrow table stays here, keyed by an
**opaque** id, and :meth:`ResultStore.get_more` pages it on demand.

Three properties this module exists to guarantee:

* **Ownership.** A handle belongs to the user who created it. Another user's handle answers exactly
  like one that never existed -- same error, same text -- so the handle space cannot be probed to
  learn that someone else's result exists. That is the whole point of an opaque id.
* **Expiry.** A handle lives for ``ttl_s`` and then is gone. Nothing here is a cache: it is a
  short-lived hand-off, so a stale handle cannot hand back rows the source has since changed.
* **Bounded memory.** Handles hold Arrow tables, so the store caps the total bytes it keeps and
  evicts the oldest first. An eviction is counted, never silent: the caller sees
  :attr:`ResultStore.stats` and a missing handle is reported as exactly that.

The clock is injected, so TTL expiry is tested by driving time forward instead of sleeping. Row
conversion reuses :func:`mdqm.results.summary.summarise`, so a paged row is JSON-native in exactly
the same way the first page's sample was -- one conversion path, not two that can drift apart.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta

import pyarrow as pa
from pydantic import Field

from mdqm.governor.clock import Clock, SystemClock
from mdqm.models.base import JsonObject, MDQMModel
from mdqm.results.summary import summarise

__all__ = [
    "DEFAULT_MAX_HANDLES",
    "DEFAULT_TTL_S",
    "HandleNotFound",
    "Page",
    "ResultHandle",
    "ResultStore",
    "StoreStats",
]

#: How long a handle stays readable. Short on purpose: it is a hand-off, not a cache.
DEFAULT_TTL_S = 300.0
#: Handles kept at once before the oldest is evicted.
DEFAULT_MAX_HANDLES = 64
#: Rows a single handle may hold; a bigger result must be narrowed or aggregated by the caller.
DEFAULT_MAX_ROWS_PER_HANDLE = 1_000_000


class HandleNotFound(LookupError):
    """No readable handle with that id *for this user*.

    Deliberately identical for "does not exist", "expired" and "belongs to someone else": the
    distinction would tell an agent that another user's result exists. That is also why the message
    never carries the id -- two ids must produce byte-identical text. The id stays on ``handle`` for
    the server's own log.
    """

    def __init__(self, handle: str) -> None:
        self.handle = handle
        super().__init__(
            "no result handle is available for this user; it may have expired "
            "(handles are short-lived) or the id may be wrong"
        )


class ResultHandle(MDQMModel):
    """A stored result's identity and lifetime. What the agent is given and may quote back."""

    handle: str = Field(min_length=1, description="Opaque id. Never derived from the data.")
    owner: str = Field(min_length=1, description="The user this handle belongs to.")
    source: str = Field(min_length=1)
    rows: int = Field(ge=0)
    nbytes: int = Field(ge=0)
    created_at: datetime
    expires_at: datetime


class Page(MDQMModel):
    """One page of a stored result, for the owner only."""

    handle: str
    offset: int = Field(ge=0)
    rows: tuple[JsonObject, ...]
    rows_total: int = Field(ge=0)
    next_offset: int | None = Field(
        default=None, ge=0, description="Offset of the next page, or null when this is the last."
    )
    truncated: bool = Field(description="True when rows remain after this page.")
    expires_at: datetime


class StoreStats(MDQMModel):
    """What the store holds, and how much it has thrown away. Read it; never guess."""

    handles: int = Field(ge=0)
    bytes: int = Field(ge=0)
    max_handles: int = Field(ge=1)
    evicted: int = Field(ge=0, description="Handles dropped to stay within the caps.")
    expired: int = Field(ge=0, description="Handles dropped because their TTL passed.")


class _Entry:
    """One stored result. Private: only touched under the store lock."""

    __slots__ = ("handle", "table")

    def __init__(self, handle: ResultHandle, table: pa.Table) -> None:
        self.handle = handle
        self.table = table


class ResultStore:
    """Handles for large results; see the module docstring.

    Args:
        clock: injected time source; ``SystemClock`` in production.
        ttl_s: how long a handle stays readable.
        max_handles: how many handles are kept before the oldest is evicted.
        max_rows_per_handle: the largest result a handle will hold.

    Raises:
        ValueError: ``ttl_s`` is not positive, or a cap is below one.
    """

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        ttl_s: float = DEFAULT_TTL_S,
        max_handles: int = DEFAULT_MAX_HANDLES,
        max_rows_per_handle: int = DEFAULT_MAX_ROWS_PER_HANDLE,
    ) -> None:
        if ttl_s <= 0:
            raise ValueError(f"ttl_s must be positive, got {ttl_s}")
        if max_handles < 1:
            raise ValueError(f"max_handles must be at least 1, got {max_handles}")
        if max_rows_per_handle < 1:
            raise ValueError(f"max_rows_per_handle must be at least 1, got {max_rows_per_handle}")
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._ttl_s = ttl_s
        self._max_handles = max_handles
        self._max_rows = max_rows_per_handle
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        self._evicted = 0
        self._expired = 0

    @property
    def ttl_s(self) -> float:
        """How long a handle stays readable."""
        return self._ttl_s

    def put(self, *, owner: str, source: str, table: pa.Table) -> ResultHandle:
        """Store ``table`` for ``owner`` and return its handle.

        Raises:
            ValueError: the result is larger than ``max_rows_per_handle``. The caller must narrow
                the query or aggregate instead; silently storing it would defeat the cap.
        """
        if table.num_rows > self._max_rows:
            raise ValueError(
                f"result has {table.num_rows} rows, over the {self._max_rows}-row handle cap; "
                "narrow the query or aggregate it"
            )
        now = self._clock.now()
        handle = ResultHandle(
            handle=f"h_{uuid.uuid4().hex[:16]}",
            owner=owner,
            source=source,
            rows=table.num_rows,
            nbytes=table.nbytes,
            created_at=now,
            expires_at=now + timedelta(seconds=self._ttl_s),
        )
        with self._lock:
            self._sweep_locked()
            while len(self._entries) >= self._max_handles:
                self._drop_oldest_locked()
            self._entries[handle.handle] = _Entry(handle, table)
        return handle

    def meta(self, handle: str, *, owner: str) -> ResultHandle:
        """The handle's metadata, for its owner.

        Raises:
            HandleNotFound: the handle is unknown, expired, or another user's.
        """
        with self._lock:
            return self._lookup_locked(handle, owner).handle

    def table(self, handle: str, *, owner: str) -> pa.Table:
        """The whole stored table, for its owner. For an in-process caller, never for the wire.

        A tool returns summary + sample + handle; another lane (the federated executor) needs the
        rows themselves to join them locally, and paging them back through
        :meth:`get_more`'s JSON conversion would lose their types. Ownership and expiry are the
        same checks as :meth:`meta`, so this cannot be used to read someone else's result.

        Raises:
            HandleNotFound: the handle is unknown, expired, or another user's.
        """
        with self._lock:
            return self._lookup_locked(handle, owner).table

    def get_more(self, handle: str, *, owner: str, offset: int, limit: int) -> Page:
        """One page of the stored result, for its owner.

        Args:
            handle: the id the agent was given.
            owner: the user asking; must be the creator.
            offset: 0-based row offset.
            limit: how many rows this page may hold.

        Raises:
            HandleNotFound: the handle is unknown, expired, or another user's.
            ValueError: ``offset`` is negative or ``limit`` is not positive.
        """
        if offset < 0:
            raise ValueError(f"offset must not be negative, got {offset}")
        if limit < 1:
            raise ValueError(f"limit must be positive, got {limit}")
        with self._lock:
            entry = self._lookup_locked(handle, owner)
        total = entry.table.num_rows
        page_rows = entry.table.slice(offset, limit)
        # Reuse the summariser's row conversion so a paged row looks exactly like a sampled one.
        converted = summarise(page_rows, sample_rows=limit, page_rows=limit, handle=handle)
        rows = converted.sample
        consumed = offset + len(rows)
        remaining = total - consumed
        return Page(
            handle=handle,
            offset=offset,
            rows=rows,
            rows_total=total,
            next_offset=consumed if remaining > 0 else None,
            truncated=remaining > 0,
            expires_at=entry.handle.expires_at,
        )

    def drop(self, handle: str) -> None:
        """Forget one handle. Idempotent: dropping an unknown handle does nothing."""
        with self._lock:
            self._entries.pop(handle, None)

    def sweep(self) -> int:
        """Drop every expired handle and report how many went."""
        with self._lock:
            before = len(self._entries)
            self._sweep_locked()
            return before - len(self._entries)

    def stats(self) -> StoreStats:
        """What is held now, and what has been dropped."""
        with self._lock:
            return StoreStats(
                handles=len(self._entries),
                bytes=sum(entry.table.nbytes for entry in self._entries.values()),
                max_handles=self._max_handles,
                evicted=self._evicted,
                expired=self._expired,
            )

    def _lookup_locked(self, handle: str, owner: str) -> _Entry:
        """The entry, enforcing expiry and ownership. Caller holds the lock.

        A wrong owner raises the *same* error as a missing handle, so the store cannot be used to
        discover that another user's result exists.
        """
        self._sweep_locked()
        entry = self._entries.get(handle)
        if entry is None or entry.handle.owner != owner:
            raise HandleNotFound(handle)
        return entry

    def _sweep_locked(self) -> None:
        """Drop every expired entry. Caller holds the lock."""
        now = self._clock.now()
        for handle in [
            key for key, entry in self._entries.items() if entry.handle.expires_at <= now
        ]:
            del self._entries[handle]
            self._expired += 1

    def _drop_oldest_locked(self) -> None:
        """Evict the oldest handle. Caller holds the lock (and guarantees one exists)."""
        oldest = min(self._entries, key=lambda key: self._entries[key].handle.created_at)
        del self._entries[oldest]
        self._evicted += 1
