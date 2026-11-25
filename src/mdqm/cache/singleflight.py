"""Single-flight coalescing: identical concurrent calls share one source execution (task C13).

Why this exists (``docs/DESIGN.md`` section 8): primary databases and API budgets are the scarce
resource. An agent that fans out a hundred identical questions — retries, several users asking the
same thing at once, a client that does not wait — must cost one source call, not a hundred. The
second caller waits for the first call's outcome instead of issuing its own.

The contract:

* **One key, one execution.** Callers with an equal key share a single in-flight call. They all get
  the *same* return object, so they see the same ``fetched_at`` and the same freshness decision. Two
  callers with the same key cannot disagree about provenance.
* **Failures are broadcast, not cached.** Every waiter raises the leader's exception, and the key is
  dropped, so the next call retries. A failed call never poisons later ones.
* **No waiter owns the work.** The call runs in its own task behind :func:`asyncio.shield`, so
  cancelling one waiter (even the one that created the flight) leaves the work running for the rest.
* **No deadlock.** Different keys are independent. A function that re-enters its own key would wait
  for itself forever, so that is refused with :class:`RuntimeError` instead of hanging.

Not provided on purpose: TTL caching (that is :mod:`mdqm.cache.freshness` and the segment store) and
any quota or semaphore (that is the governor). This module only removes duplicate work.

One :class:`SingleFlight` belongs to one event loop: it holds :class:`asyncio.Task` objects, which
are loop-bound. The server has a single loop, so the default of one instance per server holds.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from functools import partial
from typing import Any, TypeVar, cast

__all__ = ["KEY_VERSION", "SingleFlight", "flight_key"]

T = TypeVar("T")

#: Prefix of every key. Bump it when :func:`flight_key` canonicalisation changes, so a key minted by
#: an older build can never be confused with a newer one.
KEY_VERSION = "v1"


def _canonical_value(value: object) -> object:
    """A JSON-representable, process-stable form of one parameter value.

    Deliberately closed: anything this function does not understand raises instead of falling back
    to ``repr``. ``repr`` of an arbitrary object contains a memory address, which would make the key
    differ per process — a silent cache miss at best, a wrong coalescing decision at worst.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        # str() is exact and lossless; a Decimal and the equal float stay distinct keys, which errs
        # on the side of a missed coalesce rather than a wrong one.
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                f"datetime parameters must be timezone-aware (UTC internally), got {value!r}"
            )
        return value.astimezone(UTC).isoformat()
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        return _canonical_mapping(value)
    if isinstance(value, Sequence):
        return [_canonical_value(item) for item in value]
    raise TypeError(
        f"cannot fingerprint a parameter of type {type(value).__name__}; pass JSON primitives, "
        "Decimal, datetime/date/time or a list/tuple of those"
    )


def _canonical_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    """``mapping`` with keys sorted and values canonicalised: insertion order stops mattering."""
    out: dict[str, object] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise TypeError(f"parameter names must be str, got {type(key).__name__} {key!r}")
        out[key] = _canonical_value(value)
    return dict(sorted(out.items()))


def flight_key(
    *,
    source: str,
    query: str,
    params: Mapping[str, object] | None = None,
    exec_ts: datetime | None = None,
) -> str:
    """The identity of one source call: same key means the same question to the same source.

    Everything that can change the answer is part of the key — the source, the query text, the
    parameters and the execution timestamp. Nothing else is, so the key says nothing about the
    caller: two users asking the identical question at the same instant share one call.

    ``exec_ts`` is normalised to UTC before hashing, so the same instant written in two time zones
    gives one key; a different instant gives a different key (the answer may differ). Parameters are
    canonicalised (sorted, type-preserving) so dictionary insertion order cannot change a key.

    The query is hashed **verbatim**: whitespace, case and formatting are part of the identity. Two
    spellings of the same SQL therefore miss the coalesce and each run once. That is the safe
    direction: a missed coalesce costs one source call, a wrong one would hand one caller another
    caller's answer. Normalising SQL belongs to the AST fingerprint (task C10), not here.
    """
    if not source:
        raise ValueError("flight_key needs a source name: '' would coalesce unrelated sources")
    if not query:
        raise ValueError("flight_key needs a non-empty query: '' would coalesce unrelated queries")
    if exec_ts is not None:
        if exec_ts.tzinfo is None or exec_ts.utcoffset() is None:
            raise ValueError(
                f"exec_ts must be timezone-aware (UTC internally), got naive {exec_ts!r}"
            )
        exec_ts = exec_ts.astimezone(UTC)
    envelope = {
        "source": source,
        "query": query,
        "params": _canonical_mapping(params or {}),
        "exec_ts": None if exec_ts is None else exec_ts.isoformat(),
    }
    payload = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{KEY_VERSION}:{digest}"


@dataclass(frozen=True, slots=True)
class _Flight:
    """The one execution a key is currently sharing."""

    task: asyncio.Task[Any]


async def _run(fn: Callable[[], Awaitable[T]]) -> T:
    """Run ``fn`` in the shared task, so a synchronous raise is shared by every waiter too."""
    return await fn()


class SingleFlight:
    """Coalesces concurrent calls that share a key into one execution (see the module docstring)."""

    def __init__(self) -> None:
        self._flights: dict[str, _Flight] = {}

    def __len__(self) -> int:
        """How many keys are in flight right now; ``0`` when every call has finished."""
        return len(self._flights)

    async def do(self, key: str, fn: Callable[[], Awaitable[T]]) -> T:
        """Return ``fn()``'s result, executing ``fn`` at most once per in-flight ``key``.

        The first caller for ``key`` starts ``fn`` in its own task; every caller that arrives while
        that task runs waits for the same outcome and receives the same object. When the call
        finishes — successfully or not — the key is dropped, so the next call starts fresh.

        Cancelling a caller propagates :class:`asyncio.CancelledError` to that caller only.
        """
        if not key:
            raise ValueError("single-flight needs a non-empty key: '' would coalesce everything")
        flight = self._flights.get(key)
        if flight is None:
            flight = _Flight(asyncio.create_task(_run(fn)))
            self._flights[key] = flight
            flight.task.add_done_callback(partial(self._finish, key, flight))
        elif flight.task is asyncio.current_task():
            # fn itself asked for its own key: it would wait for the task it is running in.
            raise RuntimeError(
                f"single-flight re-entry on key {key!r}: the running function asked for its own "
                "key, which can never finish. Compute the value once and pass it down instead."
            )
        # shield: cancelling this waiter must not cancel the call the other waiters depend on.
        # The cast restates what the key already promises — every caller of one key asked the same
        # question, so the erased task result is their T.
        return cast("T", await asyncio.shield(flight.task))

    def _finish(self, key: str, flight: _Flight, done: asyncio.Future[Any]) -> None:
        """Drop the key once its call is over, and consume the outcome nobody may be waiting for."""
        if self._flights.get(key) is flight:
            del self._flights[key]
        if not flight.task.cancelled():
            # Retrieve the exception so an abandoned failure never surfaces as
            # "Task exception was never retrieved" instead of as the callers' error.
            flight.task.exception()
