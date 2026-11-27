"""The Context Pack: the few facts a weak agent gets, inside a token budget (task H05).

A weak agent cannot read documentation (AGENTS.md principle 4), so every answer, error and
``UserPromptSubmit`` hook carries a small context block instead: the measures that match the
question, the categorical value maps behind them, the approved join that crosses the sources, the
tables with their default time column and partition key, and one-to-three approved example specs.
This module builds that block from the semantic layer, ranked by the search index of task H04.

Decisions that are not obvious from the code:

* **Relevance is measured per question word, not per document score.** The index ANDs the words of
  one query, so a Hebrew question with one word the layer does not know (``הממוצעת``) would match
  nothing at all. Instead each content word is searched on its own and an item's relevance is *how
  many of the question's words found it*; only words that are in neither language's stopword list
  are searched, so ``what``/``much``/``של`` never drag in noise. A question whose words match
  nothing at all scores zero and the pack is EMPTY: "what is the weather" must add no tokens
  (DESIGN section 9).
* **Empty is a normal answer, not an error.** The pack is injected automatically, so a prompt that
  is not about data must produce nothing rather than an exception.
* **The budget is a hard ceiling, measured on the rendered text.** The estimator is
  :func:`mdqm.results.summary.estimate_tokens` (one estimator in the project, not two), and the pack
  is shrunk item by item from the *lowest* ranked end until the rendered block fits. What was
  dropped is reported with its reason (``budget`` or ``limit``), never silently.
* **One line per fact.** No prose: a line the agent can copy a name from. Item order inside a
  section is relevance order, so a tighter budget keeps the best facts.
* **Deterministic by construction.** No clock, no randomness, no set iteration in output: every
  collection is sorted before it is rendered, so the same question over the same indexed semantic
  version produces a byte-identical block. bm25 scores are kept on the items for diagnosis but are
  only a tie-break, because scores from separate one-word searches are not comparable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import Final

from pydantic import Field

from mdqm.context.index import ApprovedExample, ContextIndex, DocKind
from mdqm.models.base import MDQMModel
from mdqm.results.summary import estimate_tokens
from mdqm.semantic.models import (
    ApprovedJoin,
    Measure,
    SemanticLayer,
    TableSemantics,
    ValueMap,
    render_template,
)
from mdqm.semantic.text import normalize_label

__all__ = [
    "CONTEXT_TOKEN_RATIO",
    "DEFAULT_PACK_BUDGET",
    "MAX_EXAMPLES",
    "MAX_PACK_BUDGET",
    "MAX_TABLES",
    "MIN_PACK_BUDGET",
    "OmittedItem",
    "OmissionReason",
    "PackItem",
    "PackKind",
    "STOPWORDS",
    "ContextPack",
    "budget_for_context",
    "build_pack",
    "content_terms",
]

#: The default Context Pack budget (DESIGN section 9: 1,500 tokens).
DEFAULT_PACK_BUDGET: Final[int] = 1_500
#: Below this a pack cannot hold even one useful line; above it the pack stops being small.
MIN_PACK_BUDGET: Final[int] = 400
MAX_PACK_BUDGET: Final[int] = 4_000
#: The pack may take about 1.5% of a probed model context window: 1,500 of a 100K window.
CONTEXT_TOKEN_RATIO: Final[float] = 0.015
#: Approved examples are illustrations, not documentation; a handful is enough.
MAX_EXAMPLES: Final[int] = 3
#: A join bridges two sources, so it is the most expensive thing to get wrong — and the noisiest.
MAX_JOINS: Final[int] = 4
#: Tables are cheap lines but they are also the easiest way to blow the budget.
MAX_TABLES: Final[int] = 6
#: A long question is truncated to this many distinct words before it is searched.
MAX_TERMS: Final[int] = 12
#: Hits asked of the index per word; more than this only pushes noise down the ranking.
SEARCH_HITS: Final[int] = 60
#: Columns listed under one table line, so a wide table cannot eat the budget on its own.
MAX_TABLE_COLUMNS: Final[int] = 8

#: Words that carry no data meaning, in both languages the organisation asks in. A stopword is
#: never searched, so ``how``/``much``/``של`` cannot make a non-data prompt look relevant. Only
#: function words belong here: removing a word that could be a column name loses real hits.
STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        # English
        "a",
        "about",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "each",
        "for",
        "from",
        "get",
        "give",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "last",
        "list",
        "many",
        "me",
        "much",
        "my",
        "of",
        "on",
        "or",
        "our",
        "per",
        "please",
        "show",
        "some",
        "tell",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "this",
        "to",
        "top",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        # Hebrew (function words and the words a person wraps a question in)
        "אבל",
        "או",
        "אז",
        "איך",
        "אין",
        "אם",
        "אני",
        "אנחנו",
        "את",
        "אתה",
        "אתם",
        "ב",
        "גם",
        "ה",
        "הוא",
        "היא",
        "היה",
        "היו",
        "היכן",
        "הם",
        "הן",
        "יש",
        "כ",
        "כל",
        "כמה",
        "לא",
        "לאחר",
        "לה",
        "לו",
        "לי",
        "לכם",
        "לנו",
        "לפי",
        "מ",
        "מה",
        "מדוע",
        "מהי",
        "מהו",
        "מתי",
        "ן",
        "על",
        "עם",
        "צריך",
        "רוצה",
        "ש",
        "של",
        "תן",
        "תני",
        "תראה",
    }
)

#: The rule used for the first element of every ranking key: more matched words is better.
_MORE_IS_BETTER: Final[int] = -1


class PackKind(StrEnum):
    """What one pack line is about, and therefore which section it renders under."""

    MEASURE = "measure"
    VALUE = "value"
    JOIN = "join"
    TABLE = "table"
    EXAMPLE = "example"


#: Fixed section order, so the rendered block is structured the same way every time.
_KIND_ORDER: Final[dict[PackKind, int]] = {
    PackKind.MEASURE: 0,
    PackKind.VALUE: 1,
    PackKind.JOIN: 2,
    PackKind.TABLE: 3,
    PackKind.EXAMPLE: 4,
}


class OmissionReason(StrEnum):
    """Why a matching fact is not in the pack. Visible, so nothing is dropped silently."""

    BUDGET = "budget"
    LIMIT = "limit"


class PackItem(MDQMModel):
    """One fact in the pack: its kind, the semantic entity it names, and its one rendered line."""

    kind: PackKind
    ref: str = Field(min_length=1)
    text: str = Field(min_length=1)
    matched_terms: int = Field(ge=1, description="How many of the question's words found this.")
    score: float | None = Field(
        default=None, description="Best bm25 score seen, or null when the LIKE fallback ranked."
    )


class OmittedItem(MDQMModel):
    """A fact that matched but did not fit: reported with its reason instead of vanishing."""

    kind: PackKind
    ref: str = Field(min_length=1)
    reason: OmissionReason


class ContextPack(MDQMModel):
    """The token-budgeted context block for one question.

    ``text`` is what the client hook and every answer carry; ``items`` and ``omitted`` are the same
    content, structured, so a caller (or a test) can check what was included and what was not.
    """

    query: str
    semantic_version: str = Field(min_length=1)
    budget_tokens: int = Field(gt=0, description="Budget actually applied after clamping.")
    budget_clamped: bool = Field(description="True when the requested budget was not used as is.")
    tokens_estimated: int = Field(ge=0, description="Conservative estimate of ``text``.")
    items: tuple[PackItem, ...] = ()
    omitted: tuple[OmittedItem, ...] = ()
    text: str = Field(
        default="", description="The block to inject; empty means 'nothing relevant'."
    )

    @property
    def is_empty(self) -> bool:
        """True when nothing matched (or nothing fit): the caller injects no context at all."""
        return not self.items

    def items_of_kind(self, kind: PackKind) -> tuple[PackItem, ...]:
        """The included items of one kind, in pack order."""
        return tuple(item for item in self.items if item.kind is kind)

    def refs(self) -> tuple[str, ...]:
        """Every included ``kind:ref``, for a caller that logs what the agent was told."""
        return tuple(f"{item.kind.value}:{item.ref}" for item in self.items)


# --- the budget -------------------------------------------------------------------------


def budget_for_context(context_tokens: int | None) -> int:
    """The pack budget for a probed model context window, or the default when it is unknown.

    A deliberately small fraction: the pack is a nudge, not the conversation. An unknown window
    (the on-prem model is probed, see AGENTS.md section 9) gets :data:`DEFAULT_PACK_BUDGET`.
    """
    if context_tokens is None:
        return DEFAULT_PACK_BUDGET
    if context_tokens < 1:
        raise ValueError(f"context_tokens must be positive, got {context_tokens}")
    scaled = round(context_tokens * CONTEXT_TOKEN_RATIO)
    return min(max(scaled, MIN_PACK_BUDGET), MAX_PACK_BUDGET)


def _budget(requested: int) -> tuple[int, bool]:
    """Clamp ``requested`` into ``[MIN_PACK_BUDGET, MAX_PACK_BUDGET]`` and say whether it moved."""
    if requested < 1:
        raise ValueError(f"budget_tokens must be positive, got {requested}")
    clamped = min(max(requested, MIN_PACK_BUDGET), MAX_PACK_BUDGET)
    return clamped, clamped != requested


# --- question words ---------------------------------------------------------------------


def content_terms(question: str) -> tuple[str, ...]:
    """The distinct words of ``question`` worth searching, in the order they were asked.

    Folding is the index's own (:func:`~mdqm.semantic.text.normalize_label`), so a term matches what
    the index stored. Single characters and stopwords are dropped: they carry no business meaning
    and their prefix match would drag in half the layer.
    """
    terms = [
        word
        for word in normalize_label(question).split()
        if len(word) > 1 and any(character.isalnum() for character in word)
    ]
    return tuple(word for word in dict.fromkeys(terms) if word not in STOPWORDS)[:MAX_TERMS]


# --- reading the layer ------------------------------------------------------------------


def _table_of(qualified: str) -> str:
    """``crm.customers.credit_limit`` -> ``crm.customers``."""
    return qualified.rsplit(".", 1)[0]


def _span(duration: timedelta) -> str:
    """A duration as a person writes it: ``6h``, ``30m``, ``7d``, ``45s``."""
    seconds = duration // timedelta(seconds=1)
    for unit, size in (("d", 86_400), ("h", 3_600), ("m", 60)):
        if seconds and seconds % size == 0:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


def _measure_line(name: str, measure: Measure) -> str:
    """A measure: its definition, and the per-source expression onto the canonical scale."""
    parts: list[str] = []
    rounding = measure.canonical.rounding
    if rounding is not None:
        parts.append(f"round {rounding.mode.value} to {rounding.to}")
    if measure.sources:
        sources = sorted(measure.sources.items())
        parts.append("; ".join(f"{column}={source.normalize}" for column, source in sources))
    return f"{name}: " + " | ".join(parts) if parts else name


def _value_map_line(name: str, value_map: ValueMap) -> str:
    """A categorical field: each canonical value with its per-source code and its other labels."""
    rows: list[str] = []
    for value in sorted(value_map.canonical):
        codes = [
            (column, code)
            for column, mapping in sorted(value_map.sources.items())
            for code, target in sorted(mapping.items())
            if target == value
        ]
        row = value
        if codes:
            row += "=" + ", ".join(f"{column}:{code}" for column, code in codes)
        aliases = sorted(label for label in value_map.synonyms.get(value, ()) if label != value)
        if aliases:
            row += f" [{', '.join(aliases)}]"
        rows.append(row)
    return f"{name}: " + "; ".join(rows)


def _join_line(join: ApprovedJoin) -> str:
    """An approved join, with its conversion already rendered over the real columns."""
    condition = render_template(join.on, {"left": join.left, "right": join.right})
    line = f"{join.name}: {join.cardinality.value} | {condition}"
    if join.usage is not None and join.usage.count_90d:
        line += f" | used {join.usage.count_90d} (90d)"
    return line


def _table_line(key: str, semantics: TableSemantics | None, columns: Sequence[str]) -> str:
    """A table: its default time column, the partition key it must be pruned by, and its key."""
    parts: list[str] = []
    if semantics is not None:
        for column, time_column in sorted(semantics.time_columns.items()):
            note = time_column.role.value
            if time_column.use_by_default:
                note += ", use_by_default"
            if time_column.derives_bound_on is not None:
                note += f", derives bound on {time_column.derives_bound_on}"
            if time_column.lag is not None:
                note += f", max lag {_span(time_column.lag.maximum)}"
            parts.append(f"time {column} ({note})")
        partitioning = semantics.partitioning
        if partitioning is not None and partitioning.key:
            shape = partitioning.strategy.value
            if partitioning.granularity is not None:
                shape += f", {partitioning.granularity.value}"
            parts.append(f"partition key {', '.join(partitioning.key)} ({shape})")
        if semantics.key:
            parts.append(f"key {', '.join(semantics.key)}")
    else:
        parts.append("no approved semantics yet")
    if columns:
        parts.append(f"columns {', '.join(columns)}")
    return f"{key}: " + " | ".join(parts)


def _example_line(example: ApprovedExample) -> str:
    """An approved example: the question it answers and the spec a caller can run."""
    parts = [example.question] if example.question else []
    if example.ref is not None:
        parts.append(f"ref {example.ref}")
    return f"{example.name}: " + " | ".join(parts) if parts else example.name


# --- ranking ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Match:
    """How a searched document scored: how many question words found it, and the best bm25."""

    matched: int = 1
    score: float | None = None

    def combined(self, other: _Match) -> _Match:
        """The better of two sightings of the same fact (more words wins, then the better score)."""
        scores = [score for score in (self.score, other.score) if score is not None]
        return _Match(
            matched=max(self.matched, other.matched),
            score=min(scores) if scores else None,
        )


#: Matches per document kind, keyed by the semantic ref the pack renders.
_Matches = Mapping[DocKind, Mapping[str, _Match]]


@dataclass(frozen=True)
class _Candidate:
    """A rendered fact plus what decides its place in the pack."""

    kind: PackKind
    ref: str
    text: str
    match: _Match
    usage: int = 0

    @property
    def key(self) -> tuple[int, int, int, float, str]:
        """The deterministic ranking key: relevant, then cheap-to-trust, then stable."""
        return (
            _MORE_IS_BETTER * self.match.matched,
            _KIND_ORDER[self.kind],
            -self.usage,
            self.match.score if self.match.score is not None else 0.0,
            self.ref,
        )


def _search(index: ContextIndex, terms: Sequence[str]) -> dict[DocKind, dict[str, _Match]]:
    """One search per question word, folded per (kind, ref) — see the module docstring.

    Folding is keyed by the hit's ``ref``, not by its document id: the index's document ids carry
    the entity path (``value:customer_status:active``) while the ref is what the pack renders
    (``customer_status=active``), and only one of the two is the pack's business.
    """
    matched: dict[tuple[DocKind, str], int] = {}
    scores: dict[tuple[DocKind, str], list[float]] = {}
    for term in terms:
        for hit in index.search(term, limit=SEARCH_HITS, log_empty=False):
            key = (hit.kind, hit.ref)
            matched[key] = matched.get(key, 0) + 1
            if hit.score is not None:
                scores.setdefault(key, []).append(hit.score)
    found: dict[DocKind, dict[str, _Match]] = {}
    for (kind, ref), count in matched.items():
        seen = scores.get((kind, ref))
        found.setdefault(kind, {})[ref] = _Match(
            matched=count, score=min(seen) if seen is not None else None
        )
    return found


def _by_ref(matches: _Matches, kind: DocKind) -> Mapping[str, _Match]:
    """The matches of one document kind, keyed by the semantic ref the pack renders."""
    return matches.get(kind, {})


def _candidates(
    matches: _Matches,
    *,
    layer: SemanticLayer,
    examples: Sequence[ApprovedExample],
) -> tuple[list[_Candidate], dict[str, list[str]], dict[str, _Match]]:
    """Render every matched fact, and collect the tables the matched facts live in."""
    candidates: list[_Candidate] = []
    tables: dict[str, _Match] = {}
    columns: dict[str, list[str]] = {}

    def note_table(qualified: str, match: _Match) -> None:
        """Remember that a matched fact names this table, so its pruning facts come with it."""
        key = _table_of(qualified)
        existing = tables.get(key)
        tables[key] = match if existing is None else existing.combined(match)

    for name, match in _by_ref(matches, DocKind.MEASURE).items():
        measure = layer.measures.get(name)
        if measure is None:
            continue
        candidates.append(_Candidate(PackKind.MEASURE, name, _measure_line(name, measure), match))
        for column in measure.sources:
            note_table(column, match)

    values: dict[str, _Match] = {}
    for ref, match in _by_ref(matches, DocKind.VALUE).items():
        name = ref.split("=", 1)[0]
        existing = values.get(name)
        values[name] = match if existing is None else existing.combined(match)
    for name, match in values.items():
        value_map = layer.value_maps.get(name)
        if value_map is None:
            continue
        candidates.append(_Candidate(PackKind.VALUE, name, _value_map_line(name, value_map), match))
        for column in value_map.sources:
            note_table(column, match)

    for name, match in _by_ref(matches, DocKind.JOIN).items():
        join = next((item for item in layer.approved_joins if item.name == name), None)
        if join is None:
            continue
        usage = join.usage.count_90d if join.usage is not None else 0
        candidates.append(_Candidate(PackKind.JOIN, name, _join_line(join), match, usage=usage))
        note_table(join.left, match)
        note_table(join.right, match)

    for ref, match in _by_ref(matches, DocKind.COLUMN).items():
        note_table(ref, match)
        key, column = ref.rsplit(".", 1)
        columns.setdefault(key, []).append(column)

    for ref, match in _by_ref(matches, DocKind.TABLE).items():
        existing = tables.get(ref)
        tables[ref] = match if existing is None else existing.combined(match)

    examples_by_name = {example.name: example for example in examples}
    examples_by_ref = {example.ref: example for example in examples if example.ref is not None}
    seen: set[str] = set()
    for ref, match in _by_ref(matches, DocKind.EXAMPLE).items():
        example = examples_by_name.get(ref) or examples_by_ref.get(ref)
        if example is None or example.name in seen:
            continue
        seen.add(example.name)
        candidates.append(_Candidate(PackKind.EXAMPLE, example.name, _example_line(example), match))

    candidates.extend(_derived_joins(candidates, tables, layer))
    return candidates, columns, tables


def _derived_joins(
    candidates: Sequence[_Candidate], tables: Mapping[str, _Match], layer: SemanticLayer
) -> list[_Candidate]:
    """The joins that bridge a table the pack already names, even if no question word said so.

    A question about a credit limit reaches ``crm.customers``; the approved join to
    ``billing.usage_events`` is exactly the fact a weak agent cannot invent, and its own name need
    never appear in the question. Only joins with a named end qualify, so a prompt that matched
    nothing still gets nothing.
    """
    already = {item.ref for item in candidates if item.kind is PackKind.JOIN}
    derived: list[_Candidate] = []
    for join in layer.approved_joins:
        if join.name in already:
            continue
        match = max(
            (tables[_table_of(end)] for end in (join.left, join.right) if _table_of(end) in tables),
            key=lambda item: item.matched,
            default=None,
        )
        if match is None:
            continue
        usage = join.usage.count_90d if join.usage is not None else 0
        derived.append(_Candidate(PackKind.JOIN, join.name, _join_line(join), match, usage=usage))
    return derived


def _cap(
    candidates: Sequence[_Candidate], kind: PackKind, limit: int
) -> tuple[list[_Candidate], list[OmittedItem]]:
    """Keep the ``limit`` best ranked facts of one kind and report the rest as capped."""
    ranked = sorted((item for item in candidates if item.kind is kind), key=lambda i: i.key)
    return list(ranked[:limit]), [
        OmittedItem(kind=kind, ref=item.ref, reason=OmissionReason.LIMIT) for item in ranked[limit:]
    ]


def _table_candidates(
    tables: Mapping[str, _Match],
    columns: Mapping[str, list[str]],
    layer: SemanticLayer,
) -> tuple[list[_Candidate], list[OmittedItem]]:
    """Table lines in rank order, so the budget cut takes the least relevant table first."""
    ranked = sorted(
        tables,
        key=lambda key: _Candidate(PackKind.TABLE, key, "", tables[key]).key,
    )
    candidates: list[_Candidate] = []
    omitted: list[OmittedItem] = []
    for position, key in enumerate(ranked):
        matched = sorted(dict.fromkeys(columns.get(key, ())))[:MAX_TABLE_COLUMNS]
        text = _table_line(key, layer.tables.get(key), matched)
        candidate = _Candidate(PackKind.TABLE, key, text, tables[key])
        if position < MAX_TABLES:
            candidates.append(candidate)
        else:
            omitted.append(OmittedItem(kind=PackKind.TABLE, ref=key, reason=OmissionReason.LIMIT))
    return candidates, omitted


# --- rendering --------------------------------------------------------------------------


def _render(items: Sequence[_Candidate], omitted: int, *, budget: int, version: str) -> str:
    """The injected block: header, a section per kind, a line per fact, what was left out."""
    lines = [f"mdqm context (semantic {version[:8]})"]
    for kind in sorted(_KIND_ORDER, key=lambda item: _KIND_ORDER[item]):
        group = [item for item in items if item.kind is kind]
        if not group:
            continue
        lines.append(f"{kind.value}:")
        lines.extend(f"  {item.text}" for item in group)
    if omitted:
        lines.append(f"omitted: {omitted} lower ranked fact(s) do not fit {budget} tokens")
    return "\n".join(lines)


def _settle(
    ranked: Sequence[_Candidate], *, budget: int, version: str
) -> tuple[list[_Candidate], str, int]:
    """Drop the lowest ranked fact until the rendered block fits, and report the final block."""
    kept = list(ranked)
    while True:
        if not kept:
            return [], "", 0
        text = _render(kept, len(ranked) - len(kept), budget=budget, version=version)
        tokens = estimate_tokens(text)
        if tokens <= budget:
            return kept, text, tokens
        kept.pop()


def build_pack(
    question: str,
    *,
    layer: SemanticLayer,
    index: ContextIndex,
    examples: Sequence[ApprovedExample] = (),
    budget_tokens: int = DEFAULT_PACK_BUDGET,
) -> ContextPack:
    """Build the Context Pack for ``question`` over an indexed semantic layer.

    ``index`` must have been built from ``layer`` with the same ``examples`` (task H04's
    :meth:`~mdqm.context.index.ContextIndex.build`), so a hit names a fact this layer holds.

    Returns an EMPTY pack (``text == ""``, no items) when no content word of the question matches
    anything, and when even the best fact does not fit the budget. Every matched fact that was left
    out appears in ``omitted`` with its reason.

    Raises:
        ValueError: ``budget_tokens`` is not positive.
    """
    budget, clamped = _budget(budget_tokens)
    version = layer.version_hash
    empty = ContextPack(
        query=question,
        semantic_version=version,
        budget_tokens=budget,
        budget_clamped=clamped,
        tokens_estimated=0,
    )
    terms = content_terms(question)
    if not terms:
        return empty
    matches = _search(index, terms)
    if not matches:
        return empty

    candidates, columns, tables = _candidates(matches, layer=layer, examples=examples)
    table_candidates, limited = _table_candidates(tables, columns, layer)
    candidates.extend(table_candidates)

    # Two kinds are capped before the budget is considered: they illustrate, they do not answer.
    _, joins_over = _cap(candidates, PackKind.JOIN, MAX_JOINS)
    _, examples_over = _cap(candidates, PackKind.EXAMPLE, MAX_EXAMPLES)
    limited.extend([*joins_over, *examples_over])
    keep = {(item.kind, item.ref) for item in (*joins_over, *examples_over)}
    ranked = sorted(
        (item for item in candidates if (item.kind, item.ref) not in keep),
        key=lambda item: item.key,
    )
    kept, text, tokens = _settle(ranked, budget=budget, version=version)
    kept_keys = {(item.kind, item.ref) for item in kept}
    omitted = tuple(
        [
            OmittedItem(kind=item.kind, ref=item.ref, reason=OmissionReason.BUDGET)
            for item in ranked
            if (item.kind, item.ref) not in kept_keys
        ]
        + sorted(limited, key=lambda item: (item.kind.value, item.ref))
    )
    if not kept:
        return empty.model_copy(update={"omitted": omitted})
    return ContextPack(
        query=question,
        semantic_version=version,
        budget_tokens=budget,
        budget_clamped=clamped,
        tokens_estimated=tokens,
        items=tuple(
            PackItem(
                kind=item.kind,
                ref=item.ref,
                text=item.text,
                matched_terms=item.match.matched,
                score=item.match.score,
            )
            for item in kept
        ),
        omitted=omitted,
        text=text,
    )
