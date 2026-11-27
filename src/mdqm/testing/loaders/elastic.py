"""Load the synthetic world's usage events into Elasticsearch (task D07).

Layout the loader builds, against the `es` compose profile::

    index template  mdqm-usage-events          mappings + settings for mdqm-usage-events-*
    monthly indices mdqm-usage-events-YYYY-MM  one per event month (start_time), plus a
                                               12-month span so empty months exist too
    alias           mdqm-usage-events          all monthly indices; the newest one is the
                                               write index

Why monthly indices and not one index: the event month is the partition key (DESIGN section 6),
and an alias lets a query pick the months it needs by index pattern instead of scanning
everything. ``start_time`` is the event time and decides the index; ``insertion_time`` is the
ingestion time and stays a plain document field. A late-arriving row therefore lives in the
index of its *event* month while its ``insertion_time`` can fall in a later month — the world
plants such rows on purpose (see :func:`mismatched_events`).

Idempotency: the loader deletes every ``mdqm-usage-events-*`` index and reloads, and every
document carries the stable ``_id`` ``str(event_id)``. Both layers are there on purpose: the
delete makes a shrunk world shrink too (a stale month cannot survive), the stable id means that
a crash *between* delete and load, or a manual re-run of a single batch, overwrites instead of
duplicating.

Admin versus search (read-only contract): this module is the **admin** side — it creates
templates and indices and writes documents. The search adapter must never be able to do any of
that, so it gets a different object, :class:`EsSearchView`, whose only operation is ``search``
(``POST /{alias}/_search``). The loader reads ``MDQM_ES_ADMIN_USER``/``MDQM_ES_ADMIN_PASSWORD``;
the view reads ``MDQM_ES_SEARCH_USER``/``MDQM_ES_SEARCH_PASSWORD``. The dev compose service
runs with ``xpack.security.enabled=false``, so both are empty there and the two accounts are
identical; the separation is still real at the API surface (the adapter is handed no method
that can write) and in a secured cluster the search account gets ``read`` +
``view_index_metadata`` on the alias and nothing else. That distinction is why the adapter takes
an :class:`EsSearchView` and not this module's client.

HTTP is plain ``httpx``: the ``elasticsearch`` Python package is strictly coupled to one server
version, and this loader must work against 8.x and 9.x alike. No Elasticsearch dependent code,
no plugins.

Run from the repository root (the ``es`` compose profile must be up)::

    python scripts/dev.py up es
    python -m mdqm.testing.loaders.elastic --small

Nothing is hard-coded except the documented dev defaults: the host and port come from
``MDQM_ES_HOST``/``MDQM_ES_PORT`` in the environment or in the nearest ``.env``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx

from mdqm.testing.world import ANCHOR, UsageEvent, World, WorldConfig, build_world

INDEX_PREFIX: Final = "mdqm-usage-events"
"""One index per event month is ``{INDEX_PREFIX}-YYYY-MM``."""

ALIAS: Final = INDEX_PREFIX
"""The single name every reader uses; it points at every monthly index."""

INDEX_TEMPLATE: Final = INDEX_PREFIX
"""Name of the composable index template the loader PUTs."""

DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 59200
"""Host port published by the ``es`` compose profile (see .env.example)."""

INDEX_MONTHS: Final = 12
"""Months of monthly indices created, ending at the world anchor's month inclusive."""

BULK_BATCH: Final = 2000
"""Documents per ``_bulk`` request: one request per batch keeps each response small."""

DOTENV_NAME: Final = ".env"
"""Credentials file name, looked up by walking up from the working directory."""

USAGE_MAPPINGS: Final[dict[str, Any]] = {
    "dynamic": "strict",
    "properties": {
        # The event id is also the document id; kept in the body so a hit is self-describing.
        "event_id": {"type": "long"},
        # Both identifier spellings are keyword: they are compared and aggregated, never
        # full-text searched. `client_id` is the second source's name for the same key.
        "client_no": {"type": "keyword"},
        "client_id": {"type": "keyword"},
        # Event time (the partition key) and ingestion time (not indexed for pruning).
        "start_time": {"type": "date"},
        "insertion_time": {"type": "date"},
        "units": {"type": "integer"},
        # NULL for a normal row; set on a backdated correction.
        "corrects_event_id": {"type": "long"},
    },
}
"""Mapping for every ``mdqm-usage-events-*`` index. ``dynamic: strict`` rejects a field nobody
reviewed instead of inventing a type from one sample."""


class LoaderError(RuntimeError):
    """The loader could not do its job; the message names what and where."""


# --------------------------------------------------------------------------------------
# Target and credentials
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EsTarget:
    """Where Elasticsearch is and who may seed it.

    ``admin_user``/``admin_password`` are the seeding account (template + index + bulk). The
    search account is separate: see :meth:`search_view`.
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    scheme: str = "http"
    admin_user: str | None = None
    admin_password: str | None = None
    search_user: str | None = None
    search_password: str | None = None

    @property
    def base_url(self) -> str:
        """Origin every request goes to, without a trailing slash."""
        return f"{self.scheme}://{self.host}:{self.port}"

    @property
    def admin_auth(self) -> tuple[str, str] | None:
        """Basic auth for the admin client, or ``None`` when security is off (dev compose)."""
        if self.admin_user is None:
            return None
        return (self.admin_user, self.admin_password or "")

    @property
    def search_auth(self) -> tuple[str, str] | None:
        """Basic auth for the search-only view; a different account from :attr:`admin_auth`."""
        if self.search_user is None:
            return None
        return (self.search_user, self.search_password or "")

    def search_view(self) -> EsSearchView:
        """The read-only client the search adapter gets: one alias, one operation."""
        return EsSearchView(self.base_url, ALIAS, auth=self.search_auth)


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines of a dotenv file. Missing file yields an empty mapping.

    Deliberately tiny and local: importing the PostgreSQL loader's copy would make this module
    depend on ``psycopg`` (the ``pg`` extra) just to read two host/port values.
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def find_dotenv(start: Path | None = None) -> Path | None:
    """Nearest credentials file walking up from ``start`` (the working directory by default).

    Walking up rather than resolving the package location means a git worktree under
    ``.worktrees/<task>`` reuses the checkout's file instead of needing its own copy.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        path = candidate / DOTENV_NAME
        if path.is_file():
            return path
    return None


def _optional(value: str | None) -> str | None:
    return value if value else None


def target_from_environ(
    *, environ: dict[str, str] | None = None, dotenv_path: Path | None = None
) -> EsTarget:
    """Build an :class:`EsTarget` from the environment, falling back to the dotenv file.

    Nothing is required: the dev service runs with security off, so a missing credential means
    "no auth header", not "broken configuration". A port that is not an integer is an error.
    """
    path = dotenv_path if dotenv_path is not None else find_dotenv()
    values = read_dotenv(path) if path is not None else {}
    env: dict[str, str] = dict(os.environ) if environ is None else environ
    values.update({key: value for key, value in env.items() if key.startswith("MDQM_")})

    raw_port = values.get("MDQM_ES_PORT", str(DEFAULT_PORT))
    try:
        port = int(raw_port)
    except ValueError:
        raise LoaderError(f"MDQM_ES_PORT must be an integer, got {raw_port!r}") from None

    return EsTarget(
        host=values.get("MDQM_ES_HOST", DEFAULT_HOST),
        port=port,
        scheme=values.get("MDQM_ES_SCHEME", "http"),
        admin_user=_optional(values.get("MDQM_ES_ADMIN_USER")),
        admin_password=_optional(values.get("MDQM_ES_ADMIN_PASSWORD")),
        search_user=_optional(values.get("MDQM_ES_SEARCH_USER")),
        search_password=_optional(values.get("MDQM_ES_SEARCH_PASSWORD")),
    )


# --------------------------------------------------------------------------------------
# Pure body builders: the same world gives the same bytes
# --------------------------------------------------------------------------------------


def as_utc(moment: datetime) -> datetime:
    """``moment`` in UTC; a naive datetime is taken as UTC, never as the machine's local zone.

    Guessing the local zone would make the index name depend on the OS the loader runs on, and
    the same world would land in different months on a laptop in Jerusalem and a server in UTC.
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def index_for(moment: datetime) -> str:
    """Index name for an event: ``mdqm-usage-events-YYYY-MM`` from its UTC event time."""
    return f"{INDEX_PREFIX}-{as_utc(moment).strftime('%Y-%m')}"


def month_start(value: datetime) -> datetime:
    """First instant of the UTC month that contains ``value``."""
    return datetime(value.year, value.month, 1, tzinfo=UTC)


def shift_months(value: datetime, months: int) -> datetime:
    """``value`` moved by ``months`` calendar months, keeping the day and time."""
    total = value.year * 12 + (value.month - 1) + months
    return value.replace(year=total // 12, month=total % 12 + 1)


def index_months(
    events: Iterable[UsageEvent],
    anchor: datetime = ANCHOR,
    count: int = INDEX_MONTHS,
) -> tuple[str, ...]:
    """Every monthly index the loader creates, sorted.

    The union of two sets: the months that actually carry rows (so a world with a longer window
    never loses a document) and a contiguous ``count``-month span ending at the anchor's month,
    so the empty months that a month-ahead query would touch exist instead of 404-ing.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    names = {index_for(event.start_time) for event in events}
    start = shift_months(month_start(anchor), -(count - 1))
    for offset in range(count):
        names.add(index_for(shift_months(start, offset)))
    return tuple(sorted(names))


def index_template_body(seed: int) -> dict[str, Any]:
    """The composable index template: settings, mappings and the alias every index joins."""
    return {
        "index_patterns": [f"{INDEX_PREFIX}-*"],
        "priority": 200,
        "_meta": {
            "generator": "src/mdqm/testing/loaders/elastic.py",
            "source": "src/mdqm/testing/world.py",
            "seed": seed,
            "note": "one index per event month; query through the alias, never a bare index name",
        },
        "template": {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                # Single-node dev cluster: a replica would only ever be unassigned (yellow).
                "refresh_interval": "1s",
            },
            "mappings": USAGE_MAPPINGS,
            # A later month created by a rotation job joins the alias by itself; the loader then
            # moves `is_write_index` to the newest index.
            "aliases": {ALIAS: {"is_write_index": False}},
        },
    }


def document_for(event: UsageEvent) -> dict[str, Any]:
    """The document body for one world row. ``insertion_time`` is kept, never dropped.

    Its presence is what a freshness calculation needs: the event month decides the index, the
    ingestion month is data, and the two disagree for a late-arriving row.
    """
    return {
        "event_id": event.event_id,
        "client_no": event.client_no,
        "start_time": as_utc(event.start_time).isoformat(),
        "insertion_time": as_utc(event.insertion_time).isoformat(),
        "units": event.units,
        "corrects_event_id": event.corrects_event_id,
    }


def bulk_body(events: Iterable[UsageEvent]) -> str:
    """``_bulk`` NDJSON: action line then document line, ordered by index then event id.

    The ``_id`` is ``str(event_id)`` — stable across runs, so a re-sent document overwrites
    itself instead of becoming a duplicate. Ordering is a sort, not the input order, so the
    bytes do not depend on how the caller assembled the rows.
    """
    ordered = sorted(events, key=lambda row: (index_for(row.start_time), row.event_id))
    lines: list[str] = []
    for row in ordered:
        action = {"index": {"_index": index_for(row.start_time), "_id": str(row.event_id)}}
        lines.append(json.dumps(action, sort_keys=True, ensure_ascii=False))
        lines.append(json.dumps(document_for(row), sort_keys=True, ensure_ascii=False))
    return "\n".join(lines) + "\n" if lines else ""


def batches(
    events: Iterable[UsageEvent], size: int = BULK_BATCH
) -> Iterable[tuple[UsageEvent, ...]]:
    """Split the rows into ``size``-row chunks, in the generator's own order."""
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    chunk: list[UsageEvent] = []
    for event in events:
        chunk.append(event)
        if len(chunk) == size:
            yield tuple(chunk)
            chunk = []
    if chunk:
        yield tuple(chunk)


def mismatched_events(events: Iterable[UsageEvent]) -> tuple[UsageEvent, ...]:
    """Rows whose ``insertion_time`` month differs from their ``start_time`` month.

    These are the late arrivals the dataset exists to exercise: they are stored in the index of
    their *event* month, so a query that prunes by ``start_time`` still finds them and a query
    that filters on ``insertion_time`` has to look elsewhere.
    """
    return tuple(
        event
        for event in events
        if as_utc(event.insertion_time).strftime("%Y-%m")
        != as_utc(event.start_time).strftime("%Y-%m")
    )


# --------------------------------------------------------------------------------------
# Search-only view: the whole contract the adapter gets
# --------------------------------------------------------------------------------------


class EsSearchView:
    """Read-only Elasticsearch access: one alias, one operation.

    The search adapter is handed this object and nothing else, so "the adapter cannot write" is
    a property of the API it is given rather than a promise in a document. ``search`` issues
    ``POST /{alias}/_search`` and refuses any other path.
    """

    def __init__(
        self,
        base_url: str,
        alias: str = ALIAS,
        *,
        auth: tuple[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._alias = alias
        self._client = httpx.Client(
            base_url=base_url,
            auth=auth,
            transport=transport,
            timeout=timeout,
            headers={"content-type": "application/json"},
        )

    @property
    def url(self) -> str:
        """The exact endpoint this view can reach: ``{base}/{alias}/_search``."""
        return f"{str(self._client.base_url).rstrip('/')}/{self._alias}/_search"

    def search(
        self, body: dict[str, Any], *, request_timeout: float | None = None
    ) -> dict[str, Any]:
        """Run one search against the alias and return the parsed response."""
        response = self._client.post(
            f"/{self._alias}/_search",
            content=json.dumps(body, sort_keys=True),
            timeout=request_timeout or 30.0,
        )
        if response.status_code != httpx.codes.OK:
            raise LoaderError(f"search failed: HTTP {response.status_code}: {_excerpt(response)}")
        payload: dict[str, Any] = response.json()
        return payload

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EsSearchView:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------------------
# Admin client: template, indices, alias, bulk
# --------------------------------------------------------------------------------------


def _excerpt(response: httpx.Response, limit: int = 600) -> str:
    """A bounded slice of a response body, for an error message that names the problem."""
    text = response.text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


class EsAdmin:
    """The seeding client: everything that changes the cluster lives here."""

    def __init__(
        self,
        target: EsTarget,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._client = httpx.Client(
            base_url=target.base_url,
            auth=target.admin_auth,
            transport=transport,
            timeout=timeout,
            headers={"content-type": "application/json"},
        )

    # -- plumbing ---------------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EsAdmin:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        content: str | None = None,
    ) -> httpx.Response:
        response = self._client.request(method, path, params=params, content=content)
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise LoaderError(f"{method} {path}: HTTP {response.status_code}: {_excerpt(response)}")
        return response

    # -- reads ------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """``GET /``: cluster name and server version."""
        payload: dict[str, Any] = self._request("GET", "/").json()
        return payload

    def existing_indices(self, pattern: str) -> tuple[str, ...]:
        """Indices matching ``pattern``; a pattern with no match is an empty tuple, not an error."""
        response = self._client.get(
            f"/_cat/indices/{pattern}", params={"format": "json", "h": "index"}
        )
        if response.status_code == httpx.codes.NOT_FOUND:
            return ()
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise LoaderError(
                f"GET /_cat/indices/{pattern}: HTTP {response.status_code}: {_excerpt(response)}"
            )
        rows: list[dict[str, str]] = response.json()
        return tuple(sorted(row["index"] for row in rows))

    def index_template(self, name: str = INDEX_TEMPLATE) -> dict[str, Any] | None:
        """``GET /_index_template/{name}``; ``None`` when it does not exist."""
        response = self._client.get(f"/_index_template/{name}")
        if response.status_code == httpx.codes.NOT_FOUND:
            return None
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise LoaderError(
                f"GET /_index_template/{name}: HTTP {response.status_code}: {_excerpt(response)}"
            )
        payload: dict[str, Any] = response.json()
        return payload

    def alias_indices(self, alias: str = ALIAS) -> tuple[str, ...]:
        """Indices the alias points at, sorted; empty when the alias does not exist."""
        response = self._client.get(f"/_alias/{alias}")
        if response.status_code == httpx.codes.NOT_FOUND:
            return ()
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise LoaderError(
                f"GET /_alias/{alias}: HTTP {response.status_code}: {_excerpt(response)}"
            )
        payload: dict[str, Any] = response.json()
        return tuple(sorted(payload))

    def write_index(self, alias: str = ALIAS) -> str | None:
        """The index the alias writes to, or ``None`` when the alias has no index at all."""
        response = self._client.get(f"/_alias/{alias}")
        if response.status_code == httpx.codes.NOT_FOUND:
            return None
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise LoaderError(
                f"GET /_alias/{alias}: HTTP {response.status_code}: {_excerpt(response)}"
            )
        payload: dict[str, Any] = response.json()
        for name, meta in sorted(payload.items()):
            aliases = meta.get("aliases", {})
            if aliases.get(alias, {}).get("is_write_index") is True:
                return str(name)
        return None

    def count(self, index: str | None = None) -> int:
        """``_count`` for one index or for the alias/pattern."""
        target = index if index is not None else ALIAS
        payload: dict[str, Any] = self._request("GET", f"/{target}/_count").json()
        return int(payload["count"])

    def search(self, body: dict[str, Any], index: str | None = None) -> dict[str, Any]:
        """One ``_search`` on the alias (or an explicit index/pattern)."""
        target = index if index is not None else ALIAS
        payload: dict[str, Any] = self._request(
            "POST", f"/{target}/_search", content=json.dumps(body, sort_keys=True)
        ).json()
        return payload

    # -- writes -----------------------------------------------------------------

    def put_template(self, body: dict[str, Any], name: str = INDEX_TEMPLATE) -> None:
        """Install (or replace) the composable index template."""
        self._request("PUT", f"/_index_template/{name}", content=json.dumps(body, sort_keys=True))

    def delete_indices(self, names: Sequence[str]) -> None:
        """Delete the given indices, ignoring the ones that are already gone."""
        if not names:
            return
        response = self._client.delete("/" + ",".join(names), params={"ignore_unavailable": "true"})
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise LoaderError(
                f"DELETE {','.join(names)}: HTTP {response.status_code}: {_excerpt(response)}"
            )

    def create_index(self, name: str) -> None:
        """Create an empty index; the installed template fills in settings, mappings and alias."""
        self._request("PUT", f"/{name}")

    def update_alias(self, indices: Sequence[str], write_index: str) -> None:
        """Point the alias at exactly ``indices``, with ``write_index`` as the only write target."""
        actions: list[dict[str, Any]] = [
            {"add": {"index": name, "alias": ALIAS, "is_write_index": name == write_index}}
            for name in indices
        ]
        self._request("POST", "/_aliases", content=json.dumps({"actions": actions}, sort_keys=True))

    def bulk(self, ndjson: str) -> int:
        """POST one ``_bulk`` body; returns the item count and raises on any item error."""
        response = self._client.post(
            "/_bulk",
            content=ndjson.encode("utf-8"),
            headers={"content-type": "application/x-ndjson"},
        )
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise LoaderError(f"POST /_bulk: HTTP {response.status_code}: {_excerpt(response)}")
        payload: dict[str, Any] = response.json()
        items: list[dict[str, Any]] = payload.get("items", [])
        if payload.get("errors"):
            for item in items:
                for outcome in item.values():
                    if outcome.get("error"):
                        raise LoaderError(
                            f"_bulk rejected a document: {json.dumps(outcome['error'])[:600]}"
                        )
            raise LoaderError("_bulk reported errors but no item carried one")
        return len(items)

    def refresh(self, pattern: str = f"{INDEX_PREFIX}-*") -> None:
        """Force a refresh so counts are visible immediately instead of after the interval."""
        self._request("POST", f"/{pattern}/_refresh")


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EsLoadReport:
    """What one loader run left behind, per monthly index."""

    indices: tuple[str, ...]
    counts: dict[str, int] = field(default_factory=dict)
    alias: str = ALIAS
    write_index: str = ""
    template: str = INDEX_TEMPLATE
    deleted: tuple[str, ...] = ()
    mismatched_docs: int = 0

    @property
    def total(self) -> int:
        """Documents loaded across all monthly indices."""
        return sum(self.counts.values())


def load(
    target: EsTarget | None = None,
    config: WorldConfig | None = None,
    *,
    world: World | None = None,
    months: int = INDEX_MONTHS,
    transport: httpx.BaseTransport | None = None,
) -> EsLoadReport:
    """Install the template, rebuild the monthly indices and bulk-load the world's events.

    Idempotent by delete-and-reload with stable document ids (see the module docstring). The
    counts in the report are read back from the cluster after a refresh, so they describe what
    Elasticsearch stored and not what the loader intended.
    """
    resolved = target if target is not None else target_from_environ()
    built = world if world is not None else build_world(config)
    events = built.billing.usage_events
    desired = index_months(events, built.config.anchor, months)
    write_index = desired[-1]
    mismatched = mismatched_events(events)

    counts: dict[str, int] = {}
    with EsAdmin(resolved, transport=transport) as admin:
        admin.put_template(index_template_body(built.config.seed))
        deleted = admin.existing_indices(f"{INDEX_PREFIX}-*")
        admin.delete_indices(deleted)
        for name in desired:
            admin.create_index(name)
        admin.update_alias(desired, write_index)
        for chunk in batches(events):
            admin.bulk(bulk_body(chunk))
        admin.refresh()
        for name in desired:
            counts[name] = admin.count(name)

    return EsLoadReport(
        indices=desired,
        counts=counts,
        write_index=write_index,
        deleted=deleted,
        mismatched_docs=len(mismatched),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: load the world, print the per-index counts."""
    parser = argparse.ArgumentParser(
        prog="python -m mdqm.testing.loaders.elastic",
        description="Install the index template and load the world's usage events into "
        "monthly indices behind the alias.",
    )
    parser.add_argument("--small", action="store_true", help="200 customers / ~5k events")
    parser.add_argument("--host", default=None, help=f"Elasticsearch host (default {DEFAULT_HOST})")
    parser.add_argument(
        "--port", type=int, default=None, help=f"Elasticsearch port (default {DEFAULT_PORT})"
    )
    parser.add_argument("--quiet", action="store_true", help="print nothing but the counts")
    args = parser.parse_args(argv)

    base = target_from_environ()
    target = EsTarget(
        host=base.host if args.host is None else args.host,
        port=base.port if args.port is None else args.port,
        scheme=base.scheme,
        admin_user=base.admin_user,
        admin_password=base.admin_password,
        search_user=base.search_user,
        search_password=base.search_password,
    )
    config = WorldConfig.small() if args.small else WorldConfig()

    try:
        report = load(target, config)
    except (LoaderError, httpx.HTTPError) as error:
        sys.stderr.write(f"elastic loader: {error}\n")
        return 1

    if not args.quiet:
        sys.stdout.write(
            f"loaded seed={config.seed} events={config.event_count} "
            f"indices={len(report.indices)} alias={report.alias} "
            f"write_index={report.write_index} late_arrivals={report.mismatched_docs} "
            f"into {target.base_url}\n"
        )
    width = max((len(name) for name in report.counts), default=0)
    for name in report.indices:
        sys.stdout.write(f"{name:<{width}}  {report.counts.get(name, 0):>8}\n")
    sys.stdout.write(f"{'total':<{width}}  {report.total:>8}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
