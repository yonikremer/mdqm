"""``find_data``: the dictionary lookup a weak agent calls before it writes a spec (task M02).

The agent does not know the organisation's table, column, measure or value names, and it cannot
read documentation (AGENTS.md principle 4). So this tool answers one question with the facts it
needs, taken from the Context Pack of task H05 and nothing else:

* **The pack itself.** :func:`~mdqm.context.pack.build_pack` does the ranking, the capping and the
  budget cut, and reports what it left out. This module does not re-rank and does not re-score: a
  second ranking would disagree with the pack the client hook injects, and the agent would be told
  two different things about the same question.
* **A spec skeleton.** The best matched measure, on the table it is stored in, grouped by a matched
  categorical column — the shape of the query, not the answer. No time window and no aggregation is
  invented: either one silently changes what the answer means, and the linter's own error carries
  the exact fix when one is needed.
* **The budget is a ceiling on the whole answer.** When the skeleton does not fit beside the pack,
  the pack is rebuilt smaller; if it still does not fit, the skeleton is dropped and the answer says
  so (``hint``), because "everything reported, nothing silent" (AGENTS.md principle 10).
* **Empty is an answer, not an error.** A question that matches nothing (``what is the weather``)
  returns an empty pack and one short hint. It is also appended to H04's unanswered-question log, so
  the data team learns which word the layer is missing. Only a question with no word at all, or a
  source name this deployment does not declare, is refused — as an ``ErrorResponse`` with a ``fix``.

Decisions that are not obvious from the code:

* **The stores are injected, not imported.** :class:`FindDataServices` is configured once when the
  server is built (the tool signature is the MCP wire schema, so a dependency cannot travel as an
  argument). It is a slot of its own rather than a field on ``misc.ToolServices`` because this
  module, ``misc`` and the ``/context`` endpoint are built by different tasks; one wiring task can
  merge the slots later.
* **``sources`` narrows the search, and the narrowing is visible.** H05's pack has no source scope,
  so the layer is scoped before the pack is built *and* any table line from another source is
  dropped afterwards with the dropped refs returned in ``excluded``. When ``sources`` is omitted
  none of that code runs and the pack is byte-identical to H05's.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Annotated, Final

from pydantic import Field

from mdqm.context.index import ApprovedExample, ContextIndex, ContextIndexError
from mdqm.context.pack import (
    DEFAULT_PACK_BUDGET,
    ContextPack,
    OmittedItem,
    PackItem,
    PackKind,
    build_pack,
    content_terms,
)
from mdqm.errors import invalid_spec
from mdqm.models import ErrorResponse
from mdqm.models.base import MDQMModel
from mdqm.results.summary import estimate_tokens
from mdqm.semantic.models import FrozenMapping, Measure, SemanticLayer
from mdqm.server.context import current_context
from mdqm.server.tools.misc import Answer
from mdqm.spec.models import MeasureSelect, QuerySpec, SelectItem
from mdqm.store.audit import AuditLog

__all__ = [
    "EMPTY_HINT",
    "FindDataAnswer",
    "FindDataServices",
    "NO_BASIS",
    "SpecSkeleton",
    "configure_find_data",
    "current_find_data",
    "find_data",
    "reset_find_data",
    "use_find_data",
]

logger = logging.getLogger(__name__)

#: What the agent reads when nothing matched. Short and imperative: retry with better words, or
#: send a spec from the examples. It names no table, because none is relevant.
EMPTY_HINT: Final[str] = (
    "Nothing in the semantic dictionary matches this question. Call `find_data` again with the "
    "business words the user used (Hebrew or English), or send `query` a spec adapted from an "
    "example."
)

#: What the agent reads when the skeleton was dropped to keep the pack inside its budget.
NO_ROOM_HINT: Final[str] = (
    "The pack used the whole token budget, so no spec skeleton is proposed: build the spec from "
    "the measure and table lines above."
)

#: ``based_on`` when neither a measure nor an approved example matched.
NO_BASIS: Final[str] = "none"


# --- what the tool is given -------------------------------------------------------------


@dataclass(frozen=True)
class FindDataServices:
    """The semantic dictionary the tool searches, built once by the server and injected.

    Args:
        layer: the semantic layer the keywords come from; the pack's facts are read out of it.
        index: the FTS index built from that same layer (task H04).
        examples: the approved examples the index holds, so the pack can name one.
        audit: where a refusal is appended, when the server has an audit log.
        budget_tokens: the Context Pack budget; the answer never exceeds it.
    """

    layer: SemanticLayer
    index: ContextIndex
    examples: tuple[ApprovedExample, ...] = ()
    audit: AuditLog | None = None
    budget_tokens: int = DEFAULT_PACK_BUDGET


_find_data_services: FindDataServices | None = None


def configure_find_data(services: FindDataServices | None) -> None:
    """Hand ``find_data`` its semantic layer and index. ``None`` clears the slot (tests)."""
    global _find_data_services
    _find_data_services = services


def reset_find_data() -> None:
    """Forget the configured dictionary. Tests use it; production configures once at startup."""
    configure_find_data(None)


def current_find_data() -> FindDataServices | None:
    """The configured dictionary, or ``None`` when the server was built without one."""
    return _find_data_services


@contextmanager
def use_find_data(services: FindDataServices) -> Iterator[FindDataServices]:
    """Configure the dictionary for the duration of a block (tests, and a future request scope)."""
    previous = _find_data_services
    configure_find_data(services)
    try:
        yield services
    finally:
        configure_find_data(previous)


# --- what the agent sees ----------------------------------------------------------------


class SpecSkeleton(MDQMModel):
    """A starting spec, and what it was derived from.

    ``spec`` is deliberately incomplete: it is the table, the measure and the categorical column
    that matched, so the agent edits rather than starts from nothing. It is ``None`` when no
    measure matched, and ``note`` says what to do instead.
    """

    spec: QuerySpec | None = Field(default=None, description="The proposed starting spec.")
    based_on: str = Field(
        description=f"The pack item it came from (``measure:<name>``, ``example:<name>``, or "
        f"``{NO_BASIS}``)."
    )
    note: str = Field(default="", description="What to change before sending it to `query`.")


class FindDataAnswer(Answer):
    """The dictionary lookup: the Context Pack, a spec skeleton, and what was left out."""

    pack: ContextPack | None = Field(
        default=None, description="The ranked context; empty means nothing relevant matched."
    )
    skeleton: SpecSkeleton | None = Field(default=None, description="A spec to start from.")
    hint: str | None = Field(
        default=None, description="A short instruction: an empty pack, or no skeleton."
    )
    excluded: tuple[str, ...] = Field(
        default=(), description="Refs dropped because they live in a source you excluded."
    )
    tokens_total: int = Field(
        default=0, ge=0, description="Estimated tokens of the pack plus the skeleton."
    )


# --- the tool ---------------------------------------------------------------------------


def find_data(
    question: Annotated[
        str,
        Field(
            min_length=1,
            max_length=1_000,
            description="The user's question, in Hebrew or English. Copy its business words.",
        ),
    ],
    sources: Annotated[
        list[str] | None,
        Field(description="Only these sources, e.g. ['crm']. Omit to search every source."),
    ] = None,
) -> FindDataAnswer:
    """Look up the tables, columns, values and measures a question is about. Call this first.

    Returns the Context Pack (what matched, ranked, with what was omitted) and a spec skeleton built
    from the best measure. Never guess a table, column or value name: ask here first. An empty pack
    with a `hint` means nothing matched, which is not an error.
    Example: find_data({"question": "credit limit by customer status"})
    """
    context = current_context()
    services = _find_data_services
    if services is None:
        return FindDataAnswer(
            status="not_implemented",
            surface="find_data",
            user_id=context.user_id,
            problem="no semantic dictionary is loaded on this server, so nothing can be searched",
            fix="Send `query` a spec adapted from examples/query_specs.json; ask the operator to "
            "load the semantic layer (`mdqm bootstrap`).",
        )
    if not _has_word(question):
        return _refused(
            services,
            context.user_id,
            invalid_spec(field="question", reason="it holds no word to search for"),
        )
    scope, refusal = _check_sources(sources, services.layer)
    if refusal is not None:
        return _refused(services, context.user_id, refusal)

    layer = services.layer if scope is None else _scoped_layer(services.layer, scope)
    pack = build_pack(
        question,
        layer=layer,
        index=services.index,
        examples=services.examples,
        budget_tokens=services.budget_tokens,
    )
    excluded: tuple[str, ...] = ()
    if scope is not None:
        pack, excluded = _scoped_pack(pack, scope)

    hint: str | None = None
    skeleton: SpecSkeleton | None = None
    if pack.is_empty:
        hint = EMPTY_HINT
        # Only an unscoped question that matched nothing is evidence the layer lacks a word.
        if scope is None and content_terms(question):
            _log_unanswered(services.index, question, context.user_id)
    else:
        skeleton = _skeleton(pack, layer)
        pack, skeleton, hint = _fit_to_budget(
            question, services, layer, pack, skeleton, scope=scope
        )

    return FindDataAnswer(
        surface="find_data",
        user_id=context.user_id,
        pack=pack,
        skeleton=skeleton,
        hint=hint,
        excluded=excluded,
        tokens_total=pack.tokens_estimated
        + (0 if skeleton is None else _skeleton_tokens(skeleton)),
    )


# --- refusals and the source scope ------------------------------------------------------


def _refused(services: FindDataServices, user_id: str, error: ErrorResponse) -> FindDataAnswer:
    """Audit the refusal (when the server has an audit log), then hand it to the agent."""
    if services.audit is not None:
        try:
            services.audit.record_refusal(user_id=user_id, tool="find_data", error=error)
        except Exception:  # noqa: BLE001 - an audit failure must not hide the refusal itself
            logger.exception(
                "could not audit a find_data refusal", extra={"error_code": error.code}
            )
    return FindDataAnswer(status="error", surface="find_data", user_id=user_id, error=error)


def _has_word(text: str) -> bool:
    """True when the text holds a letter or a digit: punctuation alone is not a question."""
    return any(character.isalnum() for character in text)


def _known_sources(layer: SemanticLayer) -> tuple[str, ...]:
    """Every source this deployment's layer talks about, sorted.

    The declared ``sources`` are authoritative, but a layer may name a table without declaring the
    source (a hand-written curated file), so the qualified names are read too rather than refusing a
    real source the layer uses.
    """
    names = set(layer.sources)
    names.update(_source_of(key) for key in layer.tables)
    for measure in layer.measures.values():
        names.update(_source_of(column) for column in measure.sources)
    for value_map in layer.value_maps.values():
        names.update(_source_of(column) for column in value_map.sources)
    for join in layer.approved_joins:
        names.update((_source_of(join.left), _source_of(join.right)))
    return tuple(sorted(names))


def _check_sources(
    sources: Sequence[str] | None, layer: SemanticLayer
) -> tuple[frozenset[str] | None, ErrorResponse | None]:
    """Validate ``sources``: the named sources, or the refusal that lists the valid ones."""
    if sources is None:
        return None, None
    known = _known_sources(layer)
    wanted = frozenset(name.strip() for name in sources if name.strip())
    if not wanted:
        return None, invalid_spec(field="sources", reason="it names no source", expected=known)
    if wanted - set(known):
        return None, invalid_spec(
            field="sources",
            reason="it names a source this deployment does not declare",
            expected=known,
        )
    return wanted, None


def _source_of(qualified: str) -> str:
    """``crm.customers.credit_limit`` -> ``crm``."""
    return qualified.split(".", 1)[0]


def _table_of(qualified: str) -> str:
    """``crm.customers.credit_limit`` -> ``crm.customers``."""
    return qualified.rsplit(".", 1)[0]


def _scoped_layer(layer: SemanticLayer, scope: frozenset[str]) -> SemanticLayer:
    """The layer with only the facts that live in ``scope``.

    A measure (or value map) stored in several sources is kept with the sources that remain, so
    the pack can still show its conversion for the source that was asked about. The version hash
    stays the full layer's: the subset came from that semantic version, and is not a version of
    its own.
    """
    measures: dict[str, Measure] = {}
    for name, measure in layer.measures.items():
        kept = {
            column: source
            for column, source in measure.sources.items()
            if _source_of(column) in scope
        }
        if kept:
            measures[name] = measure.model_copy(update={"sources": FrozenMapping(kept)})
    value_maps = {
        name: value_map.model_copy(
            update={
                "sources": FrozenMapping(
                    {
                        column: codes
                        for column, codes in value_map.sources.items()
                        if _source_of(column) in scope
                    }
                )
            }
        )
        for name, value_map in layer.value_maps.items()
        if any(_source_of(column) in scope for column in value_map.sources)
    }
    return layer.model_copy(
        update={
            "sources": FrozenMapping(
                {name: decl for name, decl in layer.sources.items() if name in scope}
            ),
            "measures": FrozenMapping(measures),
            "value_maps": FrozenMapping(value_maps),
            "tables": FrozenMapping(
                {
                    key: semantics
                    for key, semantics in layer.tables.items()
                    if _source_of(key) in scope
                }
            ),
            "approved_joins": tuple(
                join
                for join in layer.approved_joins
                if _source_of(join.left) in scope and _source_of(join.right) in scope
            ),
        }
    )


def _scoped_pack(pack: ContextPack, scope: frozenset[str]) -> tuple[ContextPack, tuple[str, ...]]:
    """Drop the table lines of sources outside ``scope``, and say which were dropped.

    H05 ranks every table a matched fact names, and the index holds all sources, so a scoped
    question can still surface another source's table. Rendering happens in the pack, so this
    mirrors that one layout (:func:`_render_items`); when nothing is dropped the original block is
    returned untouched, and ``test_the_space_the_renderer_reproduces_is_the_packs_own`` fails if
    the layout drifts.
    """
    dropped = tuple(
        f"{item.kind.value}:{item.ref}"
        for item in pack.items
        if item.kind is PackKind.TABLE and _source_of(item.ref) not in scope
    )
    if not dropped:
        return pack, ()
    kept = tuple(
        item
        for item in pack.items
        if item.kind is not PackKind.TABLE or _source_of(item.ref) in scope
    )
    omitted = tuple(
        item
        for item in pack.omitted
        if item.kind is not PackKind.TABLE or _source_of(item.ref) in scope
    )
    text = _render_items(kept, omitted, budget=pack.budget_tokens, version=pack.semantic_version)
    return (
        pack.model_copy(
            update={
                "items": kept,
                "omitted": omitted,
                "text": text,
                "tokens_estimated": estimate_tokens(text),
            }
        ),
        dropped,
    )


#: The section order the pack renders in; mirrored from :mod:`mdqm.context.pack`.
_KIND_ORDER: Final[dict[PackKind, int]] = {
    PackKind.MEASURE: 0,
    PackKind.VALUE: 1,
    PackKind.JOIN: 2,
    PackKind.TABLE: 3,
    PackKind.EXAMPLE: 4,
}


def _render_items(
    items: Sequence[PackItem],
    omitted: Sequence[OmittedItem],
    *,
    budget: int,
    version: str,
) -> str:
    """The injected block, in the pack's own layout (see :func:`_scoped_pack`)."""
    lines = [f"mdqm context (semantic {version[:8]})"]
    for kind in sorted(_KIND_ORDER, key=lambda item: _KIND_ORDER[item]):
        group = [item for item in items if item.kind is kind]
        if not group:
            continue
        lines.append(f"{kind.value}:")
        lines.extend(f"  {item.text}" for item in group)
    if omitted:
        lines.append(f"omitted: {len(omitted)} lower ranked fact(s) do not fit {budget} tokens")
    return "\n".join(lines) if items else ""


# --- the skeleton -----------------------------------------------------------------------


def _skeleton(pack: ContextPack, layer: SemanticLayer) -> SpecSkeleton:
    """A starting spec from the best matched measure, else the name of the best matched example."""
    for item in pack.items_of_kind(PackKind.MEASURE):
        measure = layer.measures.get(item.ref)
        if measure is None:
            continue
        table = _first_table(measure)
        if table is None:
            continue
        qualified, bare = table
        column = _group_column(pack, layer, qualified)
        select: list[SelectItem] = [MeasureSelect(measure=item.ref)]
        if column is not None:
            select.insert(0, column)
        return SpecSkeleton(
            spec=QuerySpec(
                from_table=bare,
                select=tuple(select),
                group_by=() if column is None else (column,),
            ),
            based_on=f"{PackKind.MEASURE.value}:{item.ref}",
            note='Add "time" and the aggregation the question asks for, then send it to `query`.',
        )
    examples = pack.items_of_kind(PackKind.EXAMPLE)
    if examples:
        return SpecSkeleton(
            based_on=f"{PackKind.EXAMPLE.value}:{examples[0].ref}",
            note="No measure matched, so no spec is proposed: adapt this approved example instead.",
        )
    return SpecSkeleton(
        based_on=NO_BASIS,
        note="No measure or approved example matched, so no spec is proposed: only the facts "
        "above are known to exist.",
    )


def _first_table(measure: Measure) -> tuple[str, str] | None:
    """The table a measure is stored in, as ``(source.table, table)``; sorted, so it is stable."""
    columns = sorted(measure.sources)
    if not columns:
        return None
    qualified = _table_of(columns[0])
    return qualified, qualified.rsplit(".", 1)[1]


def _group_column(pack: ContextPack, layer: SemanticLayer, qualified: str) -> str | None:
    """A matched categorical column on ``qualified``, in pack order; the natural ``group_by``."""
    for item in pack.items_of_kind(PackKind.VALUE):
        value_map = layer.value_maps.get(item.ref)
        if value_map is None:
            continue
        for column in sorted(value_map.sources):
            if _table_of(column) == qualified:
                return column.rsplit(".", 1)[1]
    return None


def _skeleton_tokens(skeleton: SpecSkeleton) -> int:
    """The skeleton's cost, measured the one way the project measures tokens."""
    parts = [skeleton.based_on, skeleton.note]
    if skeleton.spec is not None:
        parts.append(skeleton.spec.model_dump_json(by_alias=True, exclude_none=True))
    return estimate_tokens("\n".join(part for part in parts if part))


def _fit_to_budget(
    question: str,
    services: FindDataServices,
    layer: SemanticLayer,
    pack: ContextPack,
    skeleton: SpecSkeleton,
    *,
    scope: frozenset[str] | None,
) -> tuple[ContextPack, SpecSkeleton | None, str | None]:
    """Keep the whole answer inside the pack budget: shrink the pack, then drop the skeleton.

    The order matters. The pack is the load-bearing part, so it is rebuilt with the skeleton's cost
    subtracted from its budget; only if that cannot make room (the budget is at the floor) is the
    skeleton dropped, and then the agent is told why rather than left to wonder.
    """
    spent = _skeleton_tokens(skeleton)
    if pack.tokens_estimated + spent <= pack.budget_tokens:
        return pack, skeleton, None
    smaller = build_pack(
        question,
        layer=layer,
        index=services.index,
        examples=services.examples,
        budget_tokens=pack.budget_tokens - spent,
    )
    if scope is not None:
        smaller, _ = _scoped_pack(smaller, scope)
    if smaller.is_empty:
        return smaller, None, NO_ROOM_HINT
    if smaller.tokens_estimated + spent <= pack.budget_tokens:
        return smaller, skeleton, None
    return smaller, None, NO_ROOM_HINT


# --- the unanswered-question log --------------------------------------------------------


def _log_unanswered(index: ContextIndex, question: str, user_id: str) -> None:
    """Append the question to H04's unmatched log, through H04's own search path.

    H05 searches one word at a time and passes ``log_empty=False`` (one row per word would be
    noise), so the whole question is searched here, once, only when the pack came back empty.
    """
    try:
        index.search(question, limit=1, user_id=user_id, log_empty=True)
    except ContextIndexError:
        logger.info(
            "could not log an unanswered question",
            extra={"user_id": user_id, "reason": "the index cannot search it"},
        )
