"""Context index: the searchable view of the semantic layer (task H04).

A weak agent asks "average credit limit by status" — in Hebrew or English, with or without niqqud —
and the server must hand it the credit-limit measure, the status value map and the approved join
instead of expecting it to read documentation (AGENTS.md principle 4). This module is the thing
that search hits: one document per **table, column, measure, categorical value, approved join and
approved example query**, four fields each, ranked by bm25.

Design decisions that are not obvious from the code:

* **Both languages are indexed, and so are synonyms.** A value map's ``canonical`` values, every
  ``synonyms`` label and the Hebrew names a deployment supplies for its measures and tables all
  land in one ``labels`` field, and the same label groups are stored so the **query** side expands
  too (:func:`expand_terms`). One spelling of a business word finds the same documents as another.
* **Index and query fold identically.** Every indexed field is passed through
  :func:`~mdqm.semantic.text.normalize_label` before it is stored, and so is the query text: NFKC,
  niqqud stripped, geresh/gershayim folded to ASCII, case folded. This is load-bearing rather than
  cosmetic — FTS5's ``remove_diacritics 2`` does **not** strip niqqud from Hebrew (it only handles
  Latin/Greek/Cyrillic), so ``מִסְגֶּרֶת`` only matches ``מסגרת`` because both sides are normalised
  here.
* **Hebrew has no offline stemmer, so the query side adds prefix variants.** The one-letter
  prefixes (``ה`` the, ``ו`` and, ``ב`` in, ``ל`` to, ``מ`` from, ``כ`` as, ``ש`` that) are stripped
  as an *additional* variant, never as a replacement: ``הלקוחות`` finds ``לקוחות`` without losing
  the original wording. A query term is one OR-group of its variants, and the groups are ANDed, so
  the query is widened, never narrowed.
* **FTS5 is probed, not assumed.** The capability probe lives in :mod:`mdqm.store.db` (H01) and is
  reused; when it reports FTS5 missing, the same documents are searched with a ``LIKE`` scan of the
  same normalised columns, so the feature degrades instead of disappearing (AGENTS.md principle 1)
  and the hits say so by carrying ``score=None``.
* **A rebuild is incremental by the layer's ``version_hash``.** An unchanged hash writes nothing;
  a changed one rewrites the documents, the synonym groups and the FTS index in one transaction.
* **A search that finds nothing is data.** It is appended to ``empty_searches`` — the only cheap
  evidence that the semantic layer is missing a word the organisation actually uses (DESIGN
  section 15).

No secrets, no result rows and no SQL text are stored: identifiers, labels and one-line semantic
facts only. Every ordering is a sort, so the same input produces the same rows on every OS.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

from pydantic import Field

from mdqm.models import CatalogSnapshot, MDQMModel, UtcDatetime
from mdqm.semantic.models import ApprovedJoin, Measure, SemanticLayer, TableSemantics, ValueMap
from mdqm.semantic.text import normalize_label
from mdqm.store.db import Store, StoreError, check_fts5

__all__ = [
    "DEFAULT_WEIGHTS",
    "DOC_TABLE",
    "EMPTY_SEARCH_TABLE",
    "FTS_TABLE",
    "HEBREW_PREFIXES",
    "STATE_TABLE",
    "SYNONYM_TABLE",
    "ApprovedExample",
    "Clock",
    "ContextHit",
    "ContextIndex",
    "ContextIndexError",
    "DocKind",
    "EmptySearch",
    "IndexReport",
    "expand_terms",
    "prefix_variants",
]

logger = logging.getLogger(__name__)

#: One row per indexed entity, always present (the ``LIKE`` fallback reads it).
DOC_TABLE: Final[str] = "context_document"
#: The FTS5 index over :data:`DOC_TABLE`'s four ranked fields, created lazily when FTS5 works.
FTS_TABLE: Final[str] = "context_document_fts"
#: The label groups both sides expand (see :func:`_synonym_groups`).
SYNONYM_TABLE: Final[str] = "context_synonym"
#: The one row that makes a rebuild incremental: the semantic ``version_hash`` it indexed.
STATE_TABLE: Final[str] = "context_index_state"
#: Questions that found nothing, for the data team.
EMPTY_SEARCH_TABLE: Final[str] = "empty_searches"

#: bm25 weights for (title, names, labels, body): a name is worth more than prose, and the title
#: more than the name it repeats, so "credit limit" ranks the measure above the table mentioning it.
DEFAULT_WEIGHTS: Final[tuple[float, float, float, float]] = (6.0, 3.0, 2.0, 1.0)

#: The one-letter Hebrew prefixes worth trying without: ה (the), ו (and), ב (in), ל (to),
#: מ (from), כ (as), ש (that). Two-letter combinations (``מהלקוחות``) are out of scope.
HEBREW_PREFIXES: Final[frozenset[str]] = frozenset("הובלמכש")
#: The shortest remainder worth searching, so ``של`` is not reduced to ``ל``.
MIN_VARIANT_LENGTH: Final[int] = 3
#: A question longer than this is truncated before it is logged; the log is not a data sink.
MAX_LOGGED_QUERY: Final[int] = 500

_QUOTE: Final[str] = '"'
_STAR: Final[str] = "*"

# `remove_diacritics 2` folds Latin accents; Hebrew niqqud are folded by normalize_label on both
# sides, because unicode61 does not remove them (see the module docstring).
_FTS_CREATE: Final[str] = (
    f"CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} "  # noqa: S608 - module constants
    'USING fts5(title, names, labels, body, tokenize="unicode61 remove_diacritics 2")'
)
#: A clock, injected so a logged search has a timestamp a test can predict without sleeping.
Clock = Callable[[], datetime]


class DocKind(StrEnum):
    """What one indexed document is about. The agent is told this, so it must be exact."""

    TABLE = "table"
    COLUMN = "column"
    MEASURE = "measure"
    VALUE = "value"
    JOIN = "join"
    EXAMPLE = "example"


class ContextIndexError(StoreError, ValueError):
    """The index cannot be used as asked: unknown label key, unusable query, missing tables.

    Also a ``ValueError`` so a pydantic validator that calls into this module reports a normal
    field rejection instead of leaking an unexpected exception type.
    """


class ContextHit(MDQMModel):
    """One search result: what matched, and how well.

    ``score`` is bm25's (more negative is better, so ascending order is best first). It is ``None``
    when the ``LIKE`` fallback answered, because that scan ranks nothing — and a caller that
    silently treated ``0.0`` as a real score would mis-order its Context Pack.
    """

    doc_id: str = Field(min_length=1)
    kind: DocKind
    ref: str = Field(min_length=1)
    score: float | None = None


class IndexReport(MDQMModel):
    """What one :meth:`ContextIndex.build` did, so "no work done" is visible, never assumed."""

    version_hash: str
    documents: int = Field(ge=0)
    written: int = Field(ge=0)
    rebuilt: bool

    @property
    def already_current(self) -> bool:
        """True when the semantic version was already indexed and nothing was written."""
        return not self.rebuilt


class EmptySearch(MDQMModel):
    """One question the index could not answer, as the data team reads it."""

    id: int = Field(ge=1)
    searched_at: UtcDatetime
    query: str
    user_id: str | None = None
    documents: int = Field(ge=0)


class ApprovedExample(MDQMModel):
    """An approved example query to index: what it is called and the question it answers.

    The semantic layer has no examples section yet (the Context Pack, task H05, owns their
    presentation), so the index takes them as explicit input instead of inventing a place in the
    layer for them.
    """

    name: str = Field(min_length=1)
    question: str = ""
    ref: str | None = None


# --- query building (pure, so it is testable without a database) ---------------------------


def prefix_variants(term: str) -> tuple[str, ...]:
    """The term, plus it without one common Hebrew prefix when that leaves a whole word.

    ``הלקוחות`` -> ``("הלקוחות", "לקוחות")``. The original is always kept: stripping is a widening,
    and Hebrew words that merely start with ``ה``/``ש`` are not changed, only added to.
    """
    if len(term) - 1 >= MIN_VARIANT_LENGTH and term[:1] in HEBREW_PREFIXES:
        return (term, term[1:])
    return (term,)


def expand_terms(
    terms: Sequence[str], groups: Sequence[Sequence[str]]
) -> tuple[tuple[str, ...], ...]:
    """One OR-group per query term: its Hebrew variants plus every label of a synonym it is in.

    The caller ANDs the groups, so a synonym or a variant can only ever add results. ``groups``
    are the already normalised label groups (:func:`_synonym_groups`); a term that is in none is
    just itself.
    """
    expanded: list[tuple[str, ...]] = []
    for term in terms:
        variants = list(prefix_variants(term))
        for group in groups:
            if term in group:
                variants.extend(label for label in group if label not in variants)
        expanded.append(tuple(variants))
    return tuple(expanded)


def _match_expression(groups: Sequence[Sequence[str]]) -> str:
    """The FTS5 MATCH text: each word quoted (data, never syntax) and prefix-matched, ANDed.

    Quoting matters: a question may contain ``NOT``, ``-x`` or ``a:b``, and an unquoted word would
    become a MATCH operator instead of a word to look for.
    """
    clauses = []
    for group in groups:
        quoted = [_prefix_query(variant) for variant in group]
        clauses.append("(" + " OR ".join(quoted) + ")" if len(quoted) > 1 else quoted[0])
    return " AND ".join(clauses)


def _prefix_query(term: str) -> str:
    """``"abc"*``: a quoted string followed by FTS5's prefix operator."""
    return f"{_QUOTE}{term.replace(_QUOTE, _QUOTE * 2)}{_QUOTE}{_STAR}"


def _has_word_char(term: str) -> bool:
    """True when a term holds a letter or a digit: punctuation alone matches no token."""
    return any(character.isalnum() for character in term)


def _like_escape(term: str) -> str:
    """Escape LIKE's wildcards so a word matches literally (Hebrew included)."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _utc_now() -> datetime:
    """The default clock: timezone-aware UTC, never a naive local time."""
    return datetime.now(UTC)


# --- documents ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Doc:
    """One document before it is stored: raw text, normalised once on the way in."""

    doc_id: str
    kind: DocKind
    ref: str
    title: str
    names: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    body: str = ""

    def fields(self) -> tuple[str, str, str, str]:
        """The four indexed fields, normalised exactly as a query is."""
        return (
            normalize_label(self.title),
            normalize_label(" ".join(self.names)),
            normalize_label(" ".join(self.labels)),
            normalize_label(self.body),
        )


def _phrase(value: object) -> str:
    """An enum or a literal as readable words (``append_rarely`` -> ``append rarely``).

    Some of these facts are enums (``TimeRole``) and some are literals (``Mutability`` is a
    ``Literal``), so the two are read the same way rather than assuming ``.value`` exists.
    """
    return str(value).replace("_", " ")


def _union(first: Iterable[str], second: Iterable[str]) -> tuple[str, ...]:
    """Order-preserving de-duplication: deterministic text, no set iteration order."""
    return tuple(dict.fromkeys((*first, *second)))


def _table_of(qualified: str) -> str:
    """``crm.customers.credit_limit`` -> ``crm.customers``."""
    source, table, _column = qualified.split(".", 2)
    return f"{source}.{table}"


def _column_of(qualified: str) -> str:
    """``crm.customers.credit_limit`` -> ``credit_limit``."""
    return qualified.rsplit(".", 1)[1]


def _table_columns(layer: SemanticLayer) -> dict[str, list[tuple[str, str]]]:
    """Every table the layer tells us about, with each column and why it is there."""
    facts: dict[str, list[tuple[str, str]]] = {}
    for name in sorted(layer.measures):
        for qualified in sorted(layer.measures[name].sources):
            facts.setdefault(_table_of(qualified), []).append(
                (_column_of(qualified), f"measure {name}")
            )
    for name in sorted(layer.value_maps):
        for qualified in sorted(layer.value_maps[name].sources):
            facts.setdefault(_table_of(qualified), []).append(
                (_column_of(qualified), f"value map {name}")
            )
    for key, semantics in layer.tables.items():
        facts.setdefault(key, [])
        for column in semantics.time_columns:
            facts[key].append((column, "time column"))
        if semantics.partitioning is not None:
            for column in semantics.partitioning.key:
                facts[key].append((column, "partition key"))
    for join in layer.approved_joins:
        facts.setdefault(_table_of(join.left), []).append(
            (_column_of(join.left), f"approved join {join.name}")
        )
        facts.setdefault(_table_of(join.right), []).append(
            (_column_of(join.right), f"approved join {join.name}")
        )
    return facts


def _documents(
    layer: SemanticLayer,
    catalogs: Sequence[CatalogSnapshot],
    labels: Mapping[str, tuple[str, ...]],
    examples: Sequence[ApprovedExample],
) -> list[_Doc]:
    """Every document the index holds, sorted by ``doc_id`` so rowids are deterministic."""
    facts = _table_columns(layer)
    docs: list[_Doc] = []
    measure_labels = {name: labels.get(f"measure:{name}", ()) for name in sorted(layer.measures)}
    labels_on_table: dict[str, list[str]] = {}
    for name, measure in layer.measures.items():
        for qualified in measure.sources:
            labels_on_table.setdefault(_table_of(qualified), []).extend(measure_labels[name])
    for key in facts:
        docs.append(
            _table_doc(
                key,
                facts[key],
                layer.tables.get(key),
                _union(labels.get(f"table:{key}", ()), labels_on_table.get(key, ())),
            )
        )
        for column, note in facts[key]:
            qualified = f"{key}.{column}"
            docs.append(
                _Doc(
                    doc_id=f"column:{qualified}",
                    kind=DocKind.COLUMN,
                    ref=qualified,
                    title=column,
                    names=(qualified,),
                    labels=labels.get(f"column:{qualified}", ()),
                    body=note,
                )
            )
    for snapshot in catalogs:
        docs.extend(_catalog_docs(snapshot, labels))
    for name, measure in layer.measures.items():
        docs.append(_measure_doc(name, measure, measure_labels[name]))
    for name, value_map in layer.value_maps.items():
        docs.extend(_value_docs(name, value_map))
    for join in layer.approved_joins:
        docs.append(_join_doc(join))
    for example in examples:
        docs.append(_example_doc(example))
    return _merged(docs)


def _table_doc(
    key: str,
    columns: Sequence[tuple[str, str]],
    semantics: TableSemantics | None,
    labels: tuple[str, ...],
) -> _Doc:
    """A table: its qualified and bare names, every known column, and the facts that prune it."""
    names = [key, key.rsplit(".", 1)[1]]
    names.extend(f"{key}.{column}" for column, _note in columns)
    names.extend(column for column, _note in columns)
    body = [f"source table {key}"]
    if semantics is not None:
        if semantics.key:
            body.append(f"key {' '.join(semantics.key)}")
        for column, time_column in semantics.time_columns.items():
            body.append(f"{_phrase(time_column.role)} {column}")
            if time_column.derives_bound_on is not None:
                body.append(f"derives bound on {time_column.derives_bound_on}")
        if semantics.partitioning is not None and semantics.partitioning.key:
            body.append(f"partition key {' '.join(semantics.partitioning.key)}")
        if semantics.mutability is not None:
            body.append(f"mutability {_phrase(semantics.mutability.class_)}")
        if semantics.cache is not None and not semantics.cache.enabled:
            body.append("never cached")
    return _Doc(
        doc_id=f"table:{key}",
        kind=DocKind.TABLE,
        ref=key,
        title=key,
        names=tuple(names),
        labels=labels,
        body=" ".join(body),
    )


def _catalog_docs(
    snapshot: CatalogSnapshot, labels: Mapping[str, tuple[str, ...]]
) -> Iterator[_Doc]:
    """The probed catalog adds the columns and comments the semantic layer does not carry."""
    for table in snapshot.tables:
        key = f"{snapshot.source}.{table.qualified_name}"
        columns = tuple(column.name for column in table.columns)
        yield _Doc(
            doc_id=f"table:{key}",
            kind=DocKind.TABLE,
            ref=key,
            title=f"{table.qualified_name}",
            names=(key, key.rsplit(".", 1)[1], *columns),
            labels=labels.get(f"table:{key}", ()),
            body=" ".join(
                part
                for part in (
                    f"{_phrase(table.kind)} in {snapshot.source}",
                    table.comment or "",
                    f"{len(columns)} columns",
                )
                if part
            ),
        )
        for column in table.columns:
            qualified = f"{key}.{column.name}"
            yield _Doc(
                doc_id=f"column:{qualified}",
                kind=DocKind.COLUMN,
                ref=qualified,
                title=column.name,
                names=(qualified,),
                labels=labels.get(f"column:{qualified}", ()),
                body=" ".join(
                    part
                    for part in (
                        f"{_phrase(column.normalized_type)} column of {table.qualified_name}",
                        column.native_type,
                        column.comment or "",
                    )
                    if part
                ),
            )


def _measure_doc(name: str, measure: Measure, labels: tuple[str, ...]) -> _Doc:
    """A measure: its name, the source columns it is stored in, and each normalisation."""
    names = [name, name.replace("_", " "), *sorted(measure.sources)]
    body = [f"measure {name}"]
    rounding = measure.canonical.rounding
    if rounding is not None:
        body.append(f"rounded to {rounding.to} {_phrase(rounding.mode)}")
    body.extend(
        f"{column} normalized {source.normalize}"
        for column, source in sorted(measure.sources.items())
    )
    return _Doc(
        doc_id=f"measure:{name}",
        kind=DocKind.MEASURE,
        ref=name,
        title=name,
        names=tuple(names),
        labels=labels,
        body=" ".join(body),
    )


def _value_docs(name: str, value_map: ValueMap) -> Iterator[_Doc]:
    """One document per canonical value, carrying every spelling and every source code."""
    columns = sorted(value_map.sources)
    for value in value_map.canonical:
        codes = [
            f"{column} {major}={code}"
            for column, major in sorted(value_map.sources.items())
            for code, target in sorted(major.items())
            if target == value
        ]
        yield _Doc(
            doc_id=f"value:{name}:{value}",
            kind=DocKind.VALUE,
            ref=f"{name}={value}",
            title=value,
            names=tuple([name, *columns, *codes]),
            labels=_union((value,), value_map.synonyms.get(value, ())),
            body=f"categorical value of {name}",
        )


def _join_doc(join: ApprovedJoin) -> _Doc:
    """An approved join: both ends, its cardinality and the conversion in its ``on`` clause."""
    names = [join.name, join.name.replace("_", " "), join.left, join.right]
    return _Doc(
        doc_id=f"join:{join.name}",
        kind=DocKind.JOIN,
        ref=join.name,
        title=join.name,
        names=tuple(names),
        body=f"{_phrase(join.cardinality)} join {join.left} {join.right} on {join.on}",
    )


def _example_doc(example: ApprovedExample) -> _Doc:
    """An approved example query: what it is called and the question it answers."""
    names = tuple(part for part in (example.name, example.ref or "") if part)
    return _Doc(
        doc_id=f"example:{example.name}",
        kind=DocKind.EXAMPLE,
        ref=example.ref if example.ref is not None else example.name,
        title=example.name,
        names=names,
        body=example.question,
    )


def _merged(docs: Iterable[_Doc]) -> list[_Doc]:
    """One document per ``doc_id`` (layer and catalog facts about one table merge), sorted."""
    by_id: dict[str, _Doc] = {}
    for doc in docs:
        existing = by_id.get(doc.doc_id)
        by_id[doc.doc_id] = doc if existing is None else _merge(existing, doc)
    return [by_id[doc_id] for doc_id in sorted(by_id)]


def _merge(first: _Doc, second: _Doc) -> _Doc:
    """Union two documents about the same entity: names and labels add, bodies concatenate."""
    if first.body and second.body and second.body != first.body:
        body = f"{first.body} | {second.body}"
    else:
        body = first.body or second.body
    return _Doc(
        doc_id=first.doc_id,
        kind=first.kind,
        ref=first.ref,
        title=first.title,
        names=_union(first.names, second.names),
        labels=_union(first.labels, second.labels),
        body=body,
    )


def _words(labels: Iterable[str]) -> tuple[str, ...]:
    """Every word of every label, normalised: FTS matches tokens, and a label may be a phrase."""
    return _union(
        (),
        (word for label in labels for word in normalize_label(label).split()),
    )


def _synonym_groups(
    layer: SemanticLayer, labels: Mapping[str, tuple[str, ...]]
) -> list[tuple[str, tuple[str, ...]]]:
    """The label groups both sides expand: value spellings, measure names, table names.

    A multiword label contributes each of its words, because a query is matched word by word. A
    group with one label expands nothing and is not stored.
    """
    groups: list[tuple[str, tuple[str, ...]]] = []
    for name in sorted(layer.value_maps):
        value_map = layer.value_maps[name]
        for value in value_map.canonical:
            groups.append(
                (
                    f"value:{name}:{value}",
                    _words(_union((value,), value_map.synonyms.get(value, ()))),
                )
            )
    for name in sorted(layer.measures):
        groups.append(
            (f"measure:{name}", _words(_union((name,), labels.get(f"measure:{name}", ()))))
        )
    for key in sorted(_table_columns(layer)):
        groups.append((f"table:{key}", _words(_union((key,), labels.get(f"table:{key}", ())))))
    return [(group, words) for group, words in groups if len(words) > 1]


def _refs(layer: SemanticLayer, catalogs: Sequence[CatalogSnapshot]) -> set[str]:
    """Every key a caller may attach extra labels to: a typo must fail, not be dropped."""
    refs = {f"table:{key}" for key in _table_columns(layer)}
    refs |= {f"measure:{name}" for name in layer.measures}
    for key, columns in _table_columns(layer).items():
        refs |= {f"column:{key}.{column}" for column, _note in columns}
    for snapshot in catalogs:
        for table in snapshot.tables:
            key = f"{snapshot.source}.{table.qualified_name}"
            refs.add(f"table:{key}")
            refs |= {f"column:{key}.{column.name}" for column in table.columns}
    return refs


# --- the index ------------------------------------------------------------------------------


class ContextIndex:
    """The context index over one :class:`~mdqm.store.db.Store`.

    Not a singleton: pass the store and, in tests, the clock and the FTS5 verdict. Writes go
    through :meth:`Store.writing`, so a rebuild's documents, synonym groups and FTS rows land in one
    transaction. :meth:`search` is a read except when it logs an empty search, so it must not be
    called from inside a write transaction.
    """

    def __init__(
        self,
        store: Store,
        *,
        clock: Clock | None = None,
        weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS,
        fts5_available: bool | None = None,
    ) -> None:
        self._store = store
        self._clock = clock if clock is not None else _utc_now
        self.weights = _weights(weights)
        self._fts5 = fts5_available

    # -- writing -------------------------------------------------------------------------

    def build(
        self,
        layer: SemanticLayer,
        *,
        catalogs: Sequence[CatalogSnapshot] = (),
        labels: Mapping[str, Sequence[str]] | None = None,
        examples: Sequence[ApprovedExample] = (),
    ) -> IndexReport:
        """Index the layer, or do nothing when its ``version_hash`` is already indexed.

        ``labels`` are the deployment's own names for things the layer does not name in both
        languages (``measure:credit_limit_r5000: ["מסגרת אשראי", "credit limit"]``). A key that
        names nothing raises :class:`ContextIndexError`: a silently ignored label is a search that
        mysteriously misses.
        """
        extra = _checked_labels(layer, catalogs, labels)
        documents = _documents(layer, catalogs, extra, examples)
        groups = _synonym_groups(layer, extra)
        with self._store.writing() as conn:
            indexed = self._ensure_fts(conn)
            if self._is_current(conn, layer.version_hash, len(documents), indexed):
                logger.info(
                    "context index is current; nothing rebuilt",
                    extra={"version_hash": layer.version_hash, "documents": len(documents)},
                )
                return IndexReport(
                    version_hash=layer.version_hash,
                    documents=len(documents),
                    written=0,
                    rebuilt=False,
                )
            self._rewrite(conn, documents, groups, layer.version_hash, indexed)
        logger.info(
            "context index rebuilt",
            extra={
                "version_hash": layer.version_hash,
                "documents": len(documents),
                "fts5": indexed,
            },
        )
        return IndexReport(
            version_hash=layer.version_hash,
            documents=len(documents),
            written=len(documents),
            rebuilt=True,
        )

    def _rewrite(
        self,
        conn: sqlite3.Connection,
        documents: Sequence[_Doc],
        groups: Sequence[tuple[str, tuple[str, ...]]],
        version_hash: str,
        indexed: bool,
    ) -> None:
        """Replace every document, synonym group and FTS row, then record the version."""
        conn.execute(f"DELETE FROM {DOC_TABLE}")  # noqa: S608 - module constants
        conn.execute(f"DELETE FROM {SYNONYM_TABLE}")  # noqa: S608 - module constants
        if indexed:
            conn.execute(f"DELETE FROM {FTS_TABLE}")  # noqa: S608 - module constants
        for doc in documents:
            fields = doc.fields()
            cursor = conn.execute(
                f"INSERT INTO {DOC_TABLE} "  # noqa: S608 - module constants
                "(doc_id, kind, ref, title, names, labels, body) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc.doc_id, doc.kind.value, doc.ref, *fields),
            )
            if indexed:
                conn.execute(
                    f"INSERT INTO {FTS_TABLE} "  # noqa: S608 - module constants
                    "(rowid, title, names, labels, body) VALUES (?, ?, ?, ?, ?)",
                    (cursor.lastrowid, *fields),
                )
        for group_id, labels in groups:
            conn.executemany(
                f"INSERT OR IGNORE INTO {SYNONYM_TABLE} (label, grp) VALUES (?, ?)",  # noqa: S608
                [(label, group_id) for label in labels],
            )
        conn.execute(
            f"INSERT OR REPLACE INTO {STATE_TABLE} "  # noqa: S608 - module constants
            "(id, version_hash, documents, fts5, indexed_at) VALUES (1, ?, ?, ?, ?)",
            (
                version_hash,
                len(documents),
                1 if indexed else 0,
                self._now().isoformat(),
            ),
        )

    def _now(self) -> datetime:
        """One clock reading, normalised to UTC (the store holds UTC only)."""
        moment = self._clock()
        if moment.tzinfo is None:
            raise ContextIndexError(
                "the clock returned a naive datetime; the index stores UTC only"
            )
        return moment.astimezone(UTC)

    def _ensure_fts(self, conn: sqlite3.Connection) -> bool:
        """Create the FTS index when it is usable here, and report which path search will take."""
        if self._fts5 is None:
            self._fts5 = check_fts5(conn).available
            if not self._fts5:
                logger.info("FTS5 unavailable: context search falls back to a LIKE scan")
        if self._fts5:
            conn.execute(_FTS_CREATE)
        return self._fts5

    def _is_current(
        self, conn: sqlite3.Connection, version_hash: str, documents: int, indexed: bool
    ) -> bool:
        """True when this exact semantic version, document count and search path were indexed."""
        row = conn.execute(
            f"SELECT version_hash, documents, fts5 FROM {STATE_TABLE} WHERE id = 1"  # noqa: S608
        ).fetchone()
        if row is None:
            return False
        return (
            str(row["version_hash"]) == version_hash
            and int(row["documents"]) == documents
            and bool(int(row["fts5"])) == indexed
        )

    # -- searching -----------------------------------------------------------------------

    def search(
        self,
        text: str,
        *,
        limit: int = 10,
        user_id: str | None = None,
        log_empty: bool = True,
    ) -> tuple[ContextHit, ...]:
        """The documents that match every word of ``text``, best first.

        Words are folded exactly like the indexed text, each word is prefix-matched, a Hebrew
        one-letter prefix is tried with and without, and a word in a synonym group brings the
        group's other labels with it. A search with no hits is logged to
        :data:`EMPTY_SEARCH_TABLE` unless ``log_empty`` is false.

        Raises:
            ContextIndexError: ``text`` holds no word, ``limit`` is below 1, or the index tables
                are missing (the store was never migrated).
        """
        terms = [term for term in normalize_label(text).split() if _has_word_char(term)]
        if not terms:
            raise ContextIndexError("search needs at least one word to look for")
        if limit < 1:
            raise ContextIndexError(f"limit must be at least 1, got {limit}")
        conn = self._store.connect()
        _require_tables(conn)
        groups = expand_terms(terms, self._groups_for(conn, terms))
        if self._use_fts(conn) and _has_table(conn, FTS_TABLE):
            hits = self._fts_hits(conn, groups, limit)
        else:
            hits = self._like_hits(conn, groups, limit)
        if not hits and log_empty:
            self._log_empty(text, user_id)
        return hits

    def _use_fts(self, conn: sqlite3.Connection) -> bool:
        """Whether this build can use FTS5, remembered after the first probe."""
        if self._fts5 is None:
            self._fts5 = check_fts5(conn).available
        return self._fts5

    def _groups_for(
        self, conn: sqlite3.Connection, terms: Sequence[str]
    ) -> tuple[tuple[str, ...], ...]:
        """The stored label groups any of ``terms`` belongs to, each one sorted."""
        found: list[tuple[str, ...]] = []
        for term in terms:
            rows = conn.execute(
                f"SELECT label FROM {SYNONYM_TABLE} WHERE grp IN "  # noqa: S608 - module constant
                f"(SELECT grp FROM {SYNONYM_TABLE} WHERE label = ?) ORDER BY label",
                (term,),
            ).fetchall()
            labels = tuple(str(row[0]) for row in rows)
            if labels:
                found.append(labels)
        return tuple(found)

    def _fts_hits(
        self,
        conn: sqlite3.Connection,
        groups: Sequence[Sequence[str]],
        limit: int,
    ) -> tuple[ContextHit, ...]:
        """bm25-ranked hits: ranked by the FTS table, joined to the documents for their identity."""
        sql = (
            f"SELECT d.doc_id, d.kind, d.ref, s.score FROM ("  # noqa: S608 - module constants
            f"SELECT rowid AS id, {self._bm25()} AS score FROM {FTS_TABLE} "
            f"WHERE {FTS_TABLE} MATCH ?) AS s "
            f"JOIN {DOC_TABLE} AS d ON d.id = s.id "
            "ORDER BY s.score, d.doc_id LIMIT ?"
        )
        rows = conn.execute(sql, (_match_expression(groups), limit)).fetchall()
        return tuple(
            ContextHit(
                doc_id=str(row["doc_id"]),
                kind=DocKind(str(row["kind"])),
                ref=str(row["ref"]),
                score=float(row["score"]),
            )
            for row in rows
        )

    def _like_hits(
        self,
        conn: sqlite3.Connection,
        groups: Sequence[Sequence[str]],
        limit: int,
    ) -> tuple[ContextHit, ...]:
        """The same query as a ``LIKE`` scan of the same normalised columns, unranked."""
        clauses: list[str] = []
        params: list[object] = []
        for group in groups:
            alternatives = []
            for variant in group:
                pattern = f"%{_like_escape(variant)}%"
                alternatives.append(
                    "(title LIKE ? ESCAPE '\\' OR names LIKE ? ESCAPE '\\'"
                    " OR labels LIKE ? ESCAPE '\\' OR body LIKE ? ESCAPE '\\')"
                )
                params.extend([pattern, pattern, pattern, pattern])
            clauses.append("(" + " OR ".join(alternatives) + ")")
        sql = (
            f"SELECT doc_id, kind, ref FROM {DOC_TABLE} "  # noqa: S608 - module constant
            "WHERE " + " AND ".join(clauses) + " ORDER BY doc_id LIMIT ?"
        )
        rows = conn.execute(sql, [*params, limit]).fetchall()
        return tuple(
            ContextHit(
                doc_id=str(row["doc_id"]), kind=DocKind(str(row["kind"])), ref=str(row["ref"])
            )
            for row in rows
        )

    def _bm25(self) -> str:
        """``bm25(table, w_title, w_names, w_labels, w_body)`` with validated literal weights.

        FTS5 requires the weights as constants in the statement (they cannot be bound), so they are
        rendered from already-validated finite numbers.
        """
        weights = ", ".join(repr(weight) for weight in self.weights)
        return f"bm25({FTS_TABLE}, {weights})"

    def _log_empty(self, text: str, user_id: str | None) -> None:
        """Append one unanswered question to the data team's queue."""
        query = normalize_label(text)[:MAX_LOGGED_QUERY]
        documents = self._document_count()
        with self._store.writing() as conn:
            conn.execute(
                f"INSERT INTO {EMPTY_SEARCH_TABLE} "  # noqa: S608 - module constant
                "(searched_at, query, user_id, documents) VALUES (?, ?, ?, ?)",
                (self._now().isoformat(), query, user_id, documents),
            )
        logger.info(
            "context search found nothing",
            extra={"query_length": len(query), "documents": documents},
        )

    def _document_count(self) -> int:
        conn = self._store.connect()
        if not _has_table(conn, DOC_TABLE):
            return 0
        row = conn.execute(
            f"SELECT COUNT(*) FROM {DOC_TABLE}"  # noqa: S608 - module constant
        ).fetchone()
        return int(row[0])

    # -- what the data team reads ---------------------------------------------------------

    def unmatched(self, *, limit: int = 20) -> tuple[EmptySearch, ...]:
        """The unanswered questions, newest first: what the semantic layer is missing words for."""
        if limit < 1:
            raise ContextIndexError(f"limit must be at least 1, got {limit}")
        conn = self._store.connect()
        rows = conn.execute(
            f"SELECT id, searched_at, query, user_id, documents "  # noqa: S608 - module constant
            f"FROM {EMPTY_SEARCH_TABLE} ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return tuple(
            EmptySearch(
                id=int(row["id"]),
                searched_at=_parse_ts(str(row["searched_at"])),
                query=str(row["query"]),
                user_id=None if row["user_id"] is None else str(row["user_id"]),
                documents=int(row["documents"]),
            )
            for row in rows
        )


# --- validation helpers ---------------------------------------------------------------------


def _weights(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Check the four bm25 weights: finite and non-negative, so no injected SQL can slip through."""
    numbers = tuple(float(value) for value in values)
    if len(numbers) != 4:
        raise ContextIndexError(
            f"weights must be (title, names, labels, body): 4 numbers, got {len(numbers)}"
        )
    if any(not math.isfinite(value) or value < 0 for value in numbers):
        raise ContextIndexError(f"weights must be finite and non-negative, got {numbers}")
    return (numbers[0], numbers[1], numbers[2], numbers[3])


def _checked_labels(
    layer: SemanticLayer,
    catalogs: Sequence[CatalogSnapshot],
    labels: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    """Validate the deployment's extra labels: a key must name something that exists."""
    if not labels:
        return {}
    known = _refs(layer, catalogs)
    unknown = sorted(key for key in labels if key not in known)
    if unknown:
        raise ContextIndexError(
            f"unknown label key(s) {unknown}: use 'table:<source.table>', 'measure:<name>' or "
            "'column:<source.table.column>' for something the layer or a catalog declares"
        )
    result: dict[str, tuple[str, ...]] = {}
    for key in sorted(labels):
        values = labels[key]
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ContextIndexError(f"labels for {key!r} must be non-empty strings")
        result[key] = tuple(values)
    return result


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _require_tables(conn: sqlite3.Connection) -> None:
    """Refuse to search a store that was never migrated, naming the fix."""
    if _has_table(conn, DOC_TABLE):
        return
    raise ContextIndexError(
        f"the context index tables are missing from this metadata DB: call Store.initialize() "
        f"before searching (expected {DOC_TABLE!r} and {EMPTY_SEARCH_TABLE!r})"
    )


def _parse_ts(text: str) -> datetime:
    """Parse a stored timestamp; an unparseable one is a broken row, not a default."""
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ContextIndexError(
            f"an empty-search row holds an unparseable timestamp: {exc}"
        ) from exc
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
