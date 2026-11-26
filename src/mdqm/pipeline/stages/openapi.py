"""Stage D4: an OpenAPI document becomes the semantic layer's table skeletons (task S03).

``docs/DESIGN.md`` section 10 lists D4 as "GET endpoints, filter params, pagination, auth,
rate-limit headers; generates an adapter". Every one of those facts is *read out of the
document*, by the one reader the adapter already uses
(:func:`mdqm.adapters.openapi.read_endpoints`, task A06), so the skeleton cannot disagree with what
the adapter will actually execute. No model is asked, and nothing here contacts the API.

What a reader gets, written to ``generated/openapi_<source>.yaml`` and nowhere else:

* ``tables`` -- one entry per ``GET`` collection: its **columns** (the item schema's properties,
  with the source's own native type text and the portable normalised type), its **indexes** (the
  declared query parameters, which is what the planner treats as filterable) and the **provenance**
  that produced it, including the paging style the parameters imply and the envelope key the rows
  come from.
* ``sources`` -- the source's own declaration (``adapter: openapi``, its spec path, its auth header)
  with the rate-limit headers the document promises, so "the budget is spent, wait for the reset" is
  a fact rather than a guess.

Only keys the semantic layer already knows are written, so the file loads back through
:func:`mdqm.semantic.loader.load` beside every other section. The payload is validated *before*
it is buffered: an unknown key makes the whole layer unloadable (the rule stage D10 obeys too),
so a bad skeleton must fail here as a stage failure instead of as a broken semantic layer.

What is deliberately **not** written, and why:

* an operation that is not a table -- a write, a lookup behind a path parameter, a single-object
  answer -- is listed in the report's ``skipped`` with the reader's own reason, because "everything
  reported, nothing silent" (principle 10) and because a label alone does not explain the omission;
* a collection's identity key, mutability, time columns and default window are **not invented**. The
  layer's own defaults already read as "unknown" (``key: []``, no mutability means treat it as
  changing), and a spec says nothing about any of them: guessing is exactly the invented fact
  principle 3 forbids. Later stages measure them (D5, D6) and the layer wins on merge.

Deterministic: sources and tables in name order, indexes and evidence sorted, columns in the
document's own order with their ordinal, nothing reads a clock (``ctx.exec_ts`` is not used, so two
runs of one document are byte-identical), and ``generated/`` is the only write path.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from pydantic import ValidationError

from mdqm.adapters.openapi import SpecEndpoint, read_endpoints, read_spec
from mdqm.config.settings import SourceConfig
from mdqm.models.catalog import Column, Index
from mdqm.pipeline.core import PipelineContext, StageReport, StageStatus, hash_inputs
from mdqm.semantic.models import SemanticLayer

__all__ = ["STAGE_NAME", "STAGE_VERSION", "OpenApiStage", "read_specs"]

#: The stage's name in the pipeline ledger and in ``--stage``.
STAGE_NAME: Final[str] = "D4"

#: Bumped when the skeleton's *shape* changes, so an unchanged document is still rewritten.
STAGE_VERSION: Final[int] = 1

#: Header-name fragments that mark a rate-limit contract (``X-RateLimit-*``, ``RateLimit-*``,
#: ``Retry-After``). Compared against the name with ``_`` folded to ``-``, never case-sensitively.
_RATE_LIMIT_HINTS: Final[tuple[str, ...]] = ("ratelimit", "rate-limit", "retry-after")

#: What the layer records as this stage in a fact's provenance.
_DETECTED_BY: Final[str] = "stage D4 (openapi spec)"


def read_specs(configs: Sequence[SourceConfig]) -> dict[str, dict[str, Any]]:
    """Read every ``openapi`` source's document from the path its config names.

    The deployed shape of this stage: the documents live in files (YAML or JSON, one reader), and a
    source with another adapter is not D4's business.
    """
    return {config.name: read_spec(config) for config in configs if config.adapter == "openapi"}


class OpenApiStage:
    """Stage D4. Construct it with the documents (or their paths) and it writes one section each.

    ``specs`` maps a source name to an already-parsed OpenAPI document; ``configs`` is optional and
    supplies what a document cannot know about *this deployment* -- the spec path and the auth
    header the server sends -- so the source declaration in the layer is true of this install rather
    than of the API in general.
    """

    name: str = STAGE_NAME

    def __init__(
        self,
        specs: Mapping[str, Mapping[str, Any]],
        *,
        configs: Mapping[str, SourceConfig] | None = None,
    ) -> None:
        if not specs:
            raise ValueError("D4 needs at least one OpenAPI document to read")
        self._specs: dict[str, dict[str, Any]] = {
            str(name): dict(document) for name, document in specs.items()
        }
        self._configs: dict[str, SourceConfig] = dict(configs or {})
        stray = sorted(set(self._configs) - set(self._specs))
        if stray:
            raise ValueError(
                f"config(s) {', '.join(stray)} have no OpenAPI document; pass one for every source"
            )

    @property
    def sources(self) -> tuple[str, ...]:
        """The sources this stage reads, sorted (what the runner filters ``--source`` on)."""
        return tuple(sorted(self._specs))

    def inputs_hash(self, ctx: PipelineContext) -> str:
        """sha256 over every document, the deployment facts and the skeleton version.

        The documents themselves are the input: a spec that gained a field must re-run the stage,
        and one that did not must skip. ``ctx.exec_ts`` is deliberately absent -- a skeleton is not
        a fact about a day, so re-running it would only churn the diff.
        """
        parts: list[object] = [STAGE_NAME, STAGE_VERSION, ctx.source]
        for source in self.sources:
            config = self._configs.get(source)
            parts.append(
                {
                    "source": source,
                    "spec": self._specs[source],
                    "adapter": None if config is None else config.adapter,
                    "spec_path": None if config is None else config.spec,
                    "auth_header": None
                    if config is None or config.auth is None
                    else config.auth.header,
                    "secret_ref": None
                    if config is None or config.auth is None
                    else config.auth.secret,
                }
            )
        return hash_inputs(*parts)

    def run(self, ctx: PipelineContext) -> StageReport:
        """Read every selected document and buffer one ``generated/openapi_<source>.yaml`` each."""
        selected = (ctx.source,) if ctx.source is not None else self.sources
        missing = [source for source in selected if source not in self._specs]
        if missing:
            raise ValueError(f"no OpenAPI document for source(s) {', '.join(missing)}")
        counts: dict[str, int] = {
            "sources": 0,
            "tables": 0,
            "columns": 0,
            "indexes": 0,
            "skipped_operations": 0,
        }
        skipped: list[str] = []
        for source in selected:
            payload, source_skipped = self._skeleton(source, counts=counts)
            ctx.write_generated(_section(source), payload)
            counts["sources"] += 1
            skipped.extend(source_skipped)
        return StageReport(
            stage=self.name,
            status=StageStatus.RAN,
            counts=counts,
            skipped=tuple(skipped),
        )

    def _skeleton(
        self, source: str, *, counts: dict[str, int]
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        """One source's layer fragment, plus what it deliberately left out."""
        document = self._specs[source]
        config = self._configs.get(source)
        if config is not None and config.adapter != "openapi":
            raise ValueError(
                f"{source}: the config names adapter {config.adapter!r}, so D4 cannot claim it"
            )
        endpoints, skipped = read_endpoints(document)
        tables: dict[str, Any] = {}
        for endpoint in sorted(endpoints, key=lambda item: item.table):
            tables[f"{source}.{endpoint.table}"] = _table_entry(endpoint)
            counts["tables"] += 1
            counts["columns"] += len(endpoint.columns)
            counts["indexes"] += sum(1 for param in endpoint.params if param.location == "query")
        counts["skipped_operations"] += len(skipped)
        payload: dict[str, Any] = {
            "sources": {source: _source_entry(document, config)},
            "tables": tables,
        }
        _check_loads(payload, source=source)
        return payload, tuple(f"{item.label}: {item.reason}" for item in skipped)


# --- the layer fragment -------------------------------------------------------------------------


def _table_entry(endpoint: SpecEndpoint) -> dict[str, Any]:
    """One collection as a ``tables`` entry: columns, filterable indexes, provenance."""
    columns = tuple(
        Column(
            name=column.name,
            native_type=column.native_type,
            normalized_type=column.normalized,
            ordinal=position,
        )
        for position, column in enumerate(endpoint.columns, start=1)
    )
    indexes = tuple(
        Index(name=param.name, columns=(param.name,), index_type=f"{param.location}_param")
        for param in endpoint.params
        if param.location == "query"
    )
    return {
        # None-valued fields are dropped: a skeleton is facts, and "no comment" is not one.
        "columns": [column.model_dump(mode="json", exclude_none=True) for column in columns],
        "indexes": [
            index.model_dump(mode="json", exclude_none=True)
            for index in sorted(indexes, key=lambda item: item.name)
        ],
        "provenance": {
            "detected_by": _DETECTED_BY,
            "evidence": list(_evidence(endpoint)),
        },
    }


def _evidence(endpoint: SpecEndpoint) -> tuple[str, ...]:
    """What the document says about *how* to call this collection, as short facts.

    This is where pagination lives: the layer has no paging field because a database table has no
    pages, and inventing one for API tables alone would make the schema less uniform than the facts
    deserve. D4's evidence is read by the API compiler and by a human debugging a page loop.
    """
    paging = endpoint.paging
    lines = [f"GET {endpoint.path}"]
    if paging.style == "none":
        lines.append("no paging parameter is declared: one request answers the collection")
    else:
        lines.append(f"paging: {paging.style} via '{paging.param}'")
    if paging.limit_param is not None:
        lines.append(f"page size parameter: '{paging.limit_param}'")
    lines.append(
        "rows are the response array itself"
        if endpoint.items_key is None
        else f"rows are the response's '{endpoint.items_key}' array"
    )
    if paging.next_key is not None:
        lines.append(f"the next-page token is the response's '{paging.next_key}'")
    return tuple(lines)


def _source_entry(document: Mapping[str, Any], config: SourceConfig | None) -> dict[str, Any]:
    """One source as a ``sources`` entry: how to reach it, and what the document promises."""
    entry: dict[str, Any] = {"adapter": "openapi"}
    if config is not None and config.spec:
        entry["spec"] = config.spec
    if config is not None and config.auth is not None:
        entry["auth"] = {"header": config.auth.header, "secret": config.auth.secret}
    evidence: list[str] = []
    declared = _declared_api_key_headers(document)
    if declared:
        evidence.append("the document declares an apiKey header: " + ", ".join(declared))
    if config is not None and config.auth is not None:
        evidence.append(f"the source config sends its key in '{config.auth.header}'")
    rate_limits = _rate_limit_headers(document)
    if rate_limits:
        evidence.append("rate-limit headers: " + ", ".join(rate_limits))
    if evidence:
        entry["provenance"] = {"detected_by": _DETECTED_BY, "evidence": evidence}
    return entry


def _check_loads(payload: Mapping[str, Any], *, source: str) -> None:
    """Refuse a fragment the semantic layer would refuse: a bad section breaks the whole layer.

    Validating the fragment as a layer is the strongest check available before the commit: it runs
    the same models, the same frozen mapping and the same ``extra="forbid"`` the loader runs.
    """
    try:
        SemanticLayer.model_validate(payload)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
            for error in exc.errors()
        )
        raise ValueError(
            f"{source}: the skeleton would not load as a semantic layer ({problems})"
        ) from None


# --- reading the document's own promises --------------------------------------------------------


def _rate_limit_headers(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Response headers the document declares that belong to a rate-limit contract, sorted."""
    found: set[str] = set()
    for item in _mappings(document.get("paths")):
        for operation in _mappings(item):
            for response in _mappings(_field(operation, "responses")):
                for name in _fields(response, "headers"):
                    if _is_rate_limit_header(name):
                        found.add(name)
    return tuple(sorted(found))


def _declared_api_key_headers(document: Mapping[str, Any]) -> tuple[str, ...]:
    """The header names the document's own ``securitySchemes`` say carry the key, sorted."""
    schemes = _field(_field(document, "components"), "securitySchemes")
    names = {
        str(scheme["name"])
        for scheme in _mappings(schemes)
        if scheme.get("type") == "apiKey"
        and str(scheme.get("in", "")).lower() == "header"
        and isinstance(scheme.get("name"), str)
        and scheme["name"]
    }
    return tuple(sorted(names))


def _is_rate_limit_header(name: str) -> bool:
    folded = name.lower().replace("_", "-")
    return any(hint in folded for hint in _RATE_LIMIT_HINTS)


def _field(node: Any, key: str) -> Mapping[str, Any]:
    """One nested mapping of a document, or an empty one when the shape is not what we expect."""
    if not isinstance(node, Mapping):
        return {}
    value = node.get(key)
    return value if isinstance(value, Mapping) else {}


def _fields(node: Any, key: str) -> tuple[str, ...]:
    """The keys of one nested mapping (a response's ``headers``), empty when absent."""
    return tuple(str(name) for name in _field(node, key))


def _mappings(node: Any) -> tuple[Mapping[str, Any], ...]:
    """Every mapping inside a document node, skipping anything else (a spec may hold anything)."""
    if isinstance(node, Mapping):
        return tuple(value for value in node.values() if isinstance(value, Mapping))
    if isinstance(node, list):
        return tuple(value for value in node if isinstance(value, Mapping))
    return ()


def _section(source: str) -> str:
    """The generated section name for a source: its name, made safe for a file stem."""
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", source).strip(".") or "source"
    return f"openapi_{slug}"
