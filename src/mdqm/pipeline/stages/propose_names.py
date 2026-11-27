"""Stages A1+A2: the model proposes names, descriptions and synonyms; code and a human decide.

The one rule this module exists to keep (``docs/DESIGN.md`` section 10, AGENTS.md principle 3):
**a model proposes, deterministic code validates against the data, a human approves.** So the model
is asked one closed-JSON question per item and is never trusted beyond it:

* **A1 -- names and descriptions.** For one table, one column or one categorical value, the model
  gets a fact sheet built from stage D1 (catalog) and stage D5 (profile: observed values, null rate,
  distinct count) and answers with a Hebrew name, an English name, one description in each language,
  and synonyms.
* **A2 -- synonyms.** The same answer's synonyms become a separate proposal, because a human may
  want the name and refuse the synonym.

Every validator here is a pure function over strings the fact sheet already contains, so it runs
on a table of cases without a database and without a model:

* **language** -- a ``*_he`` field holds Hebrew letters, an ``*_en`` field holds neither Hebrew nor
  nothing;
* **normalisation** -- the repo's own :func:`~mdqm.semantic.text.normalize_label` must leave the
  label non-empty (a label of pure niqqud or zero-width characters normalises to nothing);
* **length** -- names and synonyms at most 64 characters, descriptions at most 200, 8 synonyms max;
* **column references** -- a field that names a column must name one that exists in that table (a
  backticked identifier, or a snake_case token, is how this project spells a column);
* **collision** -- no term may collide with another value's term in the same value map, or with
  another column's term in the same table. Cross-source repeats are *desirable*, so they pass;
* **personal data** -- a proposed term may not be an observed value that looks identifying (a digit,
  an ``@``, or a long string): that is a copied row, not a name.

Confidence comes from the validators and a **second run in a different order**, never from the model
saying so about itself: the fact sheet is presented twice, the second time with its lists rotated,
and the share of label fields the two runs agree on decides how much of the score is earned
(:func:`confidence`). Temperature 0, the prompt version and the model name travel with every
proposal, in its evidence and in the patch's ``provenance``.

Nothing is written to ``curated/`` here, and nothing to ``generated/`` either: the output is
:class:`~mdqm.review.queue.Proposal` rows in ``pending``, for the human gate (S12) to decide. A
refusal is reported with the field it came from and the reason, and is *not* queued -- a queue full
of facts that cannot be approved is worse than a short queue.

Deterministic and idempotent: targets come out sorted, the rotation is an index (never ``random``),
and the input hash covers the catalogs, the profiles, the value maps, the curated terms and the
model name, so the runner skips an unchanged rerun and a second run writes nothing.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Final

from pydantic import Field

from mdqm.llm.result import StructuredLLM
from mdqm.models.base import JsonObject, MDQMModel
from mdqm.models.catalog import CatalogSnapshot, Table
from mdqm.pipeline.core import PipelineContext, StageReport, StageStatus, hash_inputs
from mdqm.pipeline.stages.profiling import ColumnProfile, SourceProfile
from mdqm.review.queue import Proposal, ProposalKind, ReviewQueue, proposal_id
from mdqm.semantic.models import ValueMap
from mdqm.semantic.text import normalize_label

__all__ = [
    "AGREEMENT_FIELDS",
    "BASE_CONFIDENCE",
    "INSTRUCTIONS",
    "LABEL_LIMIT",
    "NAMES_SECTION",
    "PROMPT_VERSION",
    "STAGE_NAME",
    "SYSTEM_PROMPT",
    "TEMPERATURE",
    "VALUE_MAPS_SECTION",
    "FactSheet",
    "Labels",
    "NamesReport",
    "ProposeNamesStage",
    "Rejection",
    "RejectionCode",
    "Target",
    "TargetKind",
    "agreement",
    "collisions",
    "column_references",
    "confidence",
    "looks_personal",
    "matched_fields",
    "render_prompt",
    "rotated",
    "validate_labels",
]

logger = logging.getLogger(__name__)

STAGE_NAME: Final[str] = "A1+A2"

#: The prompt's identity. It is part of every proposal and of the input hash, so changing the ask
#: changes what a rerun does instead of silently reusing answers fetched with the old ask.
PROMPT_VERSION: Final[str] = "names-1"

#: Fixed, not configurable per call: a proposal must be reproducible from its evidence.
TEMPERATURE: Final[int] = 0

#: ``curated/names.yaml``. A table's or column's name and description live here and not in
#: ``tables``: ``TableSemantics`` is a measured skeleton with ``extra="forbid"``, where a display
#: name is a fact of its own rather than a field of the skeleton.
NAMES_SECTION: Final[str] = "names"

#: ``curated/value_maps.yaml``: a value's labels are that map's ``synonyms``, which the loader
#: already validates (a label that is a synonym of two values is a load error).
VALUE_MAPS_SECTION: Final[str] = "value_maps"

LABEL_LIMIT: Final[int] = 64
DESCRIPTION_LIMIT: Final[int] = 200
SYNONYM_LIMIT: Final[int] = 8

#: The confidence ladder, all four terms fixed so the same answers always score the same.
BASE_CONFIDENCE: Final[float] = 0.4
AGREEMENT_WEIGHT: Final[float] = 0.4
VALIDATOR_WEIGHT: Final[float] = 0.1
EVIDENCE_WEIGHT: Final[float] = 0.1

#: The fields the two runs are compared on. Descriptions are prose, so disagreement there says the
#: model was verbose, not that the meaning moved; names and synonyms are what an agent matches on.
AGREEMENT_FIELDS: Final[tuple[str, ...]] = ("name_he", "name_en", "synonyms_he", "synonyms_en")

SYSTEM_PROMPT: Final[str] = (
    "You name business data for a Hebrew and English analytics assistant. Propose; never decide: "
    "a human approves every name, description and synonym you suggest. Use only the fact sheet "
    "you are given. Never invent a column, table or value: when you must mention a column, copy "
    "it from the fact sheet verbatim, inside backticks."
)

INSTRUCTIONS: Final[str] = (
    "Propose for the single item in the fact sheet below: a Hebrew display name, an English "
    "display name, one one-sentence description in Hebrew and one in English, and up to eight "
    "synonyms in each language. Rules: the fields ending in _he contain Hebrew letters and no "
    "English; the fields ending in _en are English and contain no Hebrew; a name or a synonym is "
    "at most 64 characters and a description at most 200; a synonym must differ from the name and "
    "from every term in existing_terms; never copy an observed value into a name or a synonym; "
    "describe what the field means, do not restate its type. Reply with one JSON object only, "
    "matching the schema."
)

#: Hebrew block: one Hebrew letter is enough to say "this is the Hebrew field".
_HEBREW: Final[re.Pattern[str]] = re.compile(r"[\u0590-\u05ff]")
_LATIN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z]")
_IDENT: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BACKTICKED: Final[re.Pattern[str]] = re.compile(r"`([^`\n]+)`")
#: ``client_no``, ``start_time``: the shape this project gives a column, and not the shape of prose.
_SNAKE: Final[re.Pattern[str]] = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")
_DIGIT: Final[re.Pattern[str]] = re.compile(r"\d")

#: A value at least this long is treated as identifying even without a digit or an ``@``.
_PERSONAL_LENGTH: Final[int] = 20


# --- what the model answers ---------------------------------------------------------------


class Labels(MDQMModel):
    """The closed JSON A1+A2 ask for, and the only thing this stage reads from the model.

    Every field is a string or a tuple of strings: no nested structure, no free-form "confidence",
    no invented field names (``extra="forbid"``). An empty Hebrew name is accepted *here* and
    refused by the validators, because a stated reason for the refusal is worth more than a schema
    error the model only half understands.
    """

    name_he: str = ""
    name_en: str = ""
    description_he: str = ""
    description_en: str = ""
    synonyms_he: tuple[str, ...] = ()
    synonyms_en: tuple[str, ...] = ()


class TargetKind(StrEnum):
    """What one call is about. A categorical value is named through its value map's synonyms."""

    TABLE = "table"
    COLUMN = "column"
    VALUE = "value"


class Target(MDQMModel):
    """One thing to name: a table, a column, or one canonical value of one value map."""

    kind: TargetKind
    #: The semantic key: ``crm.customers`` / ``crm.customers.status_code`` / the value map's key.
    key: str = Field(min_length=1)
    source: str = Field(min_length=1)
    table: str | None = None
    column: str | None = None
    value: str | None = None

    @property
    def label(self) -> str:
        """How a report, an evidence line or a namespace owner names this target."""
        return self.key if self.value is None else f"{self.key}::{self.value}"


class FactSheet(MDQMModel):
    """Everything the model may use about one target -- and nothing else (D1 + D5, in words).

    ``rotation`` is the run's position in the presented order (0 and 1 today): the same facts with
    their lists rotated, so the second answer is an independent reading rather than a copy of the
    first prompt's layout. ``existing_terms`` is what the same namespace already answers to, so the
    model can avoid a collision instead of being told about one afterwards.
    """

    kind: TargetKind
    key: str
    rotation: int = Field(ge=0)
    source: str
    table: str | None = None
    column: str | None = None
    value: str | None = None
    native_type: str | None = None
    normalized_type: str | None = None
    nullable: bool | None = None
    comment: str | None = None
    columns: tuple[str, ...] = ()
    observed_values: tuple[str, ...] = ()
    null_rate: float | None = None
    distinct: int | None = None
    existing_terms: tuple[str, ...] = ()


# --- what a refused proposal says ----------------------------------------------------------


class RejectionCode(StrEnum):
    """Why one proposal did not reach the queue. A code, so a count can group them honestly."""

    NO_LABEL = "no_label"
    NOT_HEBREW = "not_hebrew"
    NOT_ENGLISH = "not_english"
    EMPTY_AFTER_NORMALISATION = "empty_after_normalisation"
    TOO_LONG = "too_long"
    MISSING_COLUMN = "missing_column"
    COLLISION = "collision"
    PERSONAL_DATA = "personal_data"


class Rejection(MDQMModel):
    """One refused proposal: which target, which field, why -- in words a human can act on."""

    target: str
    field: str
    code: RejectionCode
    detail: str


# --- pure helpers -------------------------------------------------------------------------


def rotated(values: Sequence[str], rotation: int) -> tuple[str, ...]:
    """Rotate ``values`` by ``rotation`` places: a different order over the same facts.

    A rotation and not a shuffle, because the input hash must cover what the model saw: the same
    facts in the same order have to hash the same, and ``random`` would make a rerun look changed.
    """
    if not values:
        return ()
    step = rotation % len(values)
    return tuple(values[step:]) + tuple(values[:step])


def looks_personal(value: str) -> bool:
    """True when a value looks like a row of data rather than a name an organisation would use.

    Deterministic and deliberately blunt: a digit, an ``@`` or a long string is enough. It is used
    to refuse a *proposed term* copied out of an observed value, never to filter the fact sheet.
    """
    text = normalize_label(value)
    return bool(_DIGIT.search(text)) or "@" in text or len(text) >= _PERSONAL_LENGTH


def column_references(text: str) -> tuple[str, ...]:
    """The identifiers in ``text`` read as column references, in order, deduplicated.

    Two shapes count: a backticked identifier (the form the system prompt asks for) and a snake_case
    token such as ``client_no``, which is how this project spells a column and not how prose reads.
    The rule is deliberately conservative -- a free-prose ``by_hand`` would be read as a column too
    -- and the prompt tells the model to backtick, so the false-refusal risk stays small.
    """
    references: list[str] = []
    for chunk in _BACKTICKED.findall(text):
        references.extend(_IDENT.findall(chunk))
    references.extend(_SNAKE.findall(_BACKTICKED.sub(" ", text)))
    return tuple(dict.fromkeys(references))


def _strings(labels: Labels, field: str) -> tuple[str, ...]:
    """One label field as strings: a name is a string, a synonym list is a tuple of strings."""
    value: object = getattr(labels, field)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    return ()


def _normalised_terms(labels: Labels, field: str) -> frozenset[str]:
    """One label field as lookup forms, so two runs are compared the way the linter compares."""
    return frozenset(normalize_label(term) for term in _strings(labels, field)) - {""}


def matched_fields(first: Labels, second: Labels) -> int:
    """How many of :data:`AGREEMENT_FIELDS` the two runs agree on.

    Agreement is compared on normalised terms, so ``פעיל`` and ``פָּעִיל`` agree: the question is
    whether the model found the same meaning twice, not whether it typed the same bytes.
    """
    return sum(
        1
        for field in AGREEMENT_FIELDS
        if _normalised_terms(first, field) == _normalised_terms(second, field)
    )


def agreement(first: Labels, second: Labels) -> float:
    """The share of :data:`AGREEMENT_FIELDS` two runs agree on, in ``[0, 1]``."""
    return matched_fields(first, second) / len(AGREEMENT_FIELDS)


def confidence(agree: float, *, evidenced: bool) -> float:
    """The score a proposal earns: the validators, then agreement, then the evidence behind it.

    Nothing here reads the model's own claim. A proposal that passed every validator, that two
    independently presented fact sheets agreed on completely, about a target D5 actually measured,
    scores ``BASE_CONFIDENCE + three weights = 1.0``; a full disagreement about an unmeasured target
    scores 0.5. Rounded to 3 decimals so equal inputs give equal bytes.
    """
    score = BASE_CONFIDENCE + AGREEMENT_WEIGHT * max(0.0, min(1.0, agree)) + VALIDATOR_WEIGHT
    if evidenced:
        score += EVIDENCE_WEIGHT
    return round(min(1.0, score), 3)


def _reject(target: Target, field: str, code: RejectionCode, detail: str) -> Rejection:
    return Rejection(target=target.label, field=field, code=code, detail=detail)


def _language_problem(field: str, text: str) -> RejectionCode | None:
    """The field's language, checked by the one thing that is not a matter of taste: its script."""
    if field.endswith("_he") and not _HEBREW.search(text):
        return RejectionCode.NOT_HEBREW
    if field.endswith("_en") and (_HEBREW.search(text) or not _LATIN.search(text)):
        return RejectionCode.NOT_ENGLISH
    return None


def validate_labels(
    target: Target,
    labels: Labels,
    *,
    known_columns: Iterable[str],
    observed: Sequence[str],
) -> tuple[Rejection, ...]:
    """Every per-target validator: language, normalisation, length, column references, data copies.

    Pure and total. It reads the labels, the names that exist in the table and the values that were
    observed, and returns the reasons this answer may not be queued. Collisions with *other*
    proposals need the whole pass, so they are :func:`collisions`' job.
    """
    rejections: list[Rejection] = []
    if not labels.name_en.strip() and not labels.name_he.strip():
        rejections.append(
            _reject(
                target,
                "name",
                RejectionCode.NO_LABEL,
                "both the Hebrew and the English name are empty",
            )
        )
    for field, limit in (
        ("name_he", LABEL_LIMIT),
        ("name_en", LABEL_LIMIT),
        ("description_he", DESCRIPTION_LIMIT),
        ("description_en", DESCRIPTION_LIMIT),
    ):
        text = str(getattr(labels, field))
        if not text.strip():
            continue
        if len(text) > limit:
            rejections.append(
                _reject(
                    target,
                    field,
                    RejectionCode.TOO_LONG,
                    f"{len(text)} characters, the limit is {limit}",
                )
            )
        problem = _language_problem(field, text)
        if problem is not None:
            rejections.append(
                _reject(
                    target,
                    field,
                    problem,
                    f"{text!r} is not written in the language of {field}",
                )
            )
        if normalize_label(text) == "":
            rejections.append(
                _reject(
                    target,
                    field,
                    RejectionCode.EMPTY_AFTER_NORMALISATION,
                    f"{text!r} has no letters left after NFKC and niqqud stripping",
                )
            )
    for field in ("synonyms_he", "synonyms_en"):
        synonyms = tuple(str(item) for item in getattr(labels, field))
        if len(synonyms) > SYNONYM_LIMIT:
            rejections.append(
                _reject(
                    target,
                    field,
                    RejectionCode.TOO_LONG,
                    f"{len(synonyms)} synonyms, the limit is {SYNONYM_LIMIT}",
                )
            )
        for synonym in synonyms:
            if len(synonym) > LABEL_LIMIT:
                rejections.append(
                    _reject(
                        target,
                        field,
                        RejectionCode.TOO_LONG,
                        f"{synonym!r} is {len(synonym)} characters, the limit is {LABEL_LIMIT}",
                    )
                )
            if normalize_label(synonym) == "":
                rejections.append(
                    _reject(
                        target,
                        field,
                        RejectionCode.EMPTY_AFTER_NORMALISATION,
                        f"synonym {synonym!r} normalises to nothing",
                    )
                )
                continue
            problem = _language_problem(field, synonym)
            if problem is not None:
                rejections.append(
                    _reject(
                        target,
                        field,
                        problem,
                        f"synonym {synonym!r} is not written in the language of {field}",
                    )
                )
    known = set(known_columns)
    for field in ("name_he", "name_en", "description_he", "description_en"):
        for reference in column_references(str(getattr(labels, field))):
            if reference not in known:
                rejections.append(
                    _reject(
                        target,
                        field,
                        RejectionCode.MISSING_COLUMN,
                        f"{field} names the column {reference!r}, which this table does not have "
                        f"(it has: {', '.join(sorted(known)) or 'no columns'})",
                    )
                )
    personal = [value for value in observed if looks_personal(value)]

    def _copied(term: str) -> str | None:
        folded = normalize_label(term)
        for value in personal:
            if normalize_label(value) == folded:
                return value
        return None

    for field in ("name_he", "name_en"):
        if _copied(str(getattr(labels, field))) is not None:
            rejections.append(
                _reject(
                    target,
                    field,
                    RejectionCode.PERSONAL_DATA,
                    f"{getattr(labels, field)!r} is a copied observed value, not a name",
                )
            )
    if target.kind is not TargetKind.VALUE:
        # A value's own labels are labels for a value we already read; only a table's or a column's
        # synonyms must never be a row of data.
        for field in ("synonyms_he", "synonyms_en"):
            for synonym in tuple(str(item) for item in getattr(labels, field)):
                if _copied(synonym) is not None:
                    rejections.append(
                        _reject(
                            target,
                            field,
                            RejectionCode.PERSONAL_DATA,
                            f"synonym {synonym!r} is a copied observed value",
                        )
                    )
    return tuple(rejections)


def collisions(
    target: Target, terms: Sequence[str], taken: Mapping[str, str]
) -> tuple[Rejection, ...]:
    """Reject a term that already means something else in the same namespace.

    ``taken`` maps a normalised term to the owner that already holds it, and the caller scopes it:
    one value map for a categorical value, one table (and its columns) for a table. A term owned by
    *this* target is not a collision -- it is the target's own name. Cross-source repeats are not
    collisions either: two sources calling the same business field "customer status" is the point of
    the semantic layer, not an ambiguity.
    """
    rejections: list[Rejection] = []
    for term in terms:
        folded = normalize_label(term)
        if not folded:
            continue
        owner = taken.get(folded)
        if owner is not None and owner != target.label:
            rejections.append(
                _reject(
                    target,
                    "synonym",
                    RejectionCode.COLLISION,
                    f"{term!r} already means {owner!r} in the same namespace",
                )
            )
    return tuple(rejections)


# --- the prompt -----------------------------------------------------------------------------


def render_prompt(sheet: FactSheet) -> str:
    """The user message: the rules, then the fact sheet as sorted JSON (Hebrew left unescaped)."""
    body = json.dumps(sheet.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
    return f"{INSTRUCTIONS}\n\n{body}"


# --- the stage ------------------------------------------------------------------------------


class NamesReport(MDQMModel):
    """What one pass proposed and refused: the queue's input, in one readable object."""

    model: str
    prompt_version: str = PROMPT_VERSION
    temperature: int = TEMPERATURE
    counts: dict[str, int] = Field(default_factory=dict)
    proposals: tuple[Proposal, ...] = ()
    rejections: tuple[Rejection, ...] = ()
    notes: tuple[str, ...] = ()
    fixes: tuple[str, ...] = ()

    @property
    def rejected(self) -> int:
        """How many refusals the validators reported."""
        return len(self.rejections)


class ProposeNamesStage:
    """Stage A1+A2 over the catalogs (D1), the profiles (D5) and the value maps of the layer.

    ``catalogs`` is what D1 extracted per source, ``profiles`` what D5 measured, ``value_maps`` the
    maps a human or A3 already has. Nothing here reads a source: the two model calls per target are
    the only work, so the stage is not ``heavy`` and needs no quiet window or source slot.

    ``known_terms`` carries terms that are already curated, keyed by namespace (``table:<key>`` or
    ``map:<key>``) and mapping a normalised term to its owner label, so a new proposal cannot
    quietly take a label a human already assigned.
    """

    name: str = STAGE_NAME

    def __init__(
        self,
        catalogs: Mapping[str, CatalogSnapshot],
        *,
        profiles: Mapping[str, SourceProfile] | None = None,
        value_maps: Mapping[str, ValueMap] | None = None,
        known_terms: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        self._catalogs = {source: catalogs[source] for source in sorted(catalogs)}
        self._profiles = dict(profiles or {})
        given_maps = dict(value_maps or {})
        self._value_maps = {key: given_maps[key] for key in sorted(given_maps)}
        self._known = {scope: dict(terms) for scope, terms in sorted((known_terms or {}).items())}
        self._base = self._registry()
        self._rejections: tuple[Rejection, ...] = ()

    # -- inputs ------------------------------------------------------------------------

    @property
    def rejections(self) -> tuple[Rejection, ...]:
        """The refusals of the last run, so a caller can print them without the report."""
        return self._rejections

    def targets(self) -> tuple[Target, ...]:
        """Every target of this pass, in a stable order: tables, then their columns, then values."""
        found: list[Target] = []
        for source, catalog in self._catalogs.items():
            for table in sorted(catalog.tables, key=lambda item: item.name):
                key = f"{source}.{table.name}"
                found.append(
                    Target(kind=TargetKind.TABLE, key=key, source=source, table=table.name)
                )
                for column in sorted(table.columns, key=lambda item: item.name):
                    found.append(
                        Target(
                            kind=TargetKind.COLUMN,
                            key=f"{key}.{column.name}",
                            source=source,
                            table=table.name,
                            column=column.name,
                        )
                    )
        for map_key, value_map in self._value_maps.items():
            source = _map_source(value_map)
            for value in value_map.canonical:
                found.append(Target(kind=TargetKind.VALUE, key=map_key, source=source, value=value))
        return tuple(found)

    def inputs_hash(self, ctx: PipelineContext) -> str:
        """sha256 over the facts the proposals depend on, including the model that will answer."""
        catalogs = [
            {
                "source": source,
                "tables": [
                    {
                        "name": table.name,
                        "comment": table.comment,
                        "columns": [
                            [column.name, column.native_type, column.normalized_type.value]
                            for column in sorted(table.columns, key=lambda item: item.name)
                        ],
                    }
                    for table in sorted(catalog.tables, key=lambda item: item.name)
                ],
            }
            for source, catalog in self._catalogs.items()
        ]
        profiles = [
            {
                "source": source,
                "tables": [
                    {
                        "table": table.table,
                        "columns": [
                            [
                                column.name,
                                "" if column.distinct is None else column.distinct,
                                "" if column.null_rate is None else column.null_rate,
                                [value.value for value in column.top_values],
                            ]
                            for column in table.columns
                        ],
                    }
                    for table in sorted(profile.tables, key=lambda item: item.table)
                ],
            }
            for source, profile in sorted(self._profiles.items())
        ]
        maps = [
            {
                "key": map_key,
                "canonical": list(value_map.canonical),
                "synonyms": {value: sorted(labels) for value, labels in value_map.synonyms.items()},
            }
            for map_key, value_map in self._value_maps.items()
        ]
        return hash_inputs(
            {
                "stage": self.name,
                "prompt_version": PROMPT_VERSION,
                "temperature": TEMPERATURE,
                "model": None if ctx.llm is None else ctx.llm.model_name,
                "catalogs": catalogs,
                "profiles": profiles,
                "value_maps": maps,
                "known_terms": {
                    scope: sorted(terms.items()) for scope, terms in self._known.items()
                },
            }
        )

    # -- one fact sheet ----------------------------------------------------------------

    def fact_sheet(self, target: Target, *, rotation: int = 0) -> FactSheet:
        """What the model may see about ``target``: catalog facts, D5 evidence, existing terms."""
        table = self._table(target)
        profile = self._column_profile(target)
        column = None if table is None or target.column is None else table.column(target.column)
        observed = () if profile is None else tuple(value.value for value in profile.top_values)
        if target.kind is TargetKind.VALUE:
            observed = (target.value or "",)
        return FactSheet(
            kind=target.kind,
            key=target.key,
            rotation=rotation,
            source=target.source,
            table=target.table,
            column=target.column,
            value=target.value,
            native_type=None if column is None else column.native_type,
            normalized_type=None if column is None else column.normalized_type.value,
            nullable=None if column is None else column.nullable,
            comment=None if column is None else column.comment,
            columns=()
            if table is None
            else rotated([entry.name for entry in table.columns], rotation),
            observed_values=rotated(observed, rotation),
            null_rate=None if profile is None else profile.null_rate,
            distinct=None if profile is None else profile.distinct,
            existing_terms=rotated(self._terms_of(target), rotation),
        )

    def _table(self, target: Target) -> Table | None:
        catalog = self._catalogs.get(target.source)
        if catalog is None or target.table is None:
            return None
        return next((table for table in catalog.tables if table.name == target.table), None)

    def _column_profile(self, target: Target) -> ColumnProfile | None:
        profile = self._profiles.get(target.source)
        if profile is None or target.table is None or target.column is None:
            return None
        return profile.column(target.table, target.column)

    def _namespace(self, target: Target) -> str:
        """The scope a term must be unique inside: one value map, or one table and its columns."""
        if target.kind is TargetKind.VALUE:
            return f"map:{target.key}"
        return f"table:{target.source}.{target.table}"

    def _terms_of(self, target: Target) -> tuple[str, ...]:
        """The terms this namespace already answers to, except the target's own."""
        namespace = self._base.get(self._namespace(target), {})
        return tuple(sorted(term for term, owner in namespace.items() if owner != target.label))

    def _own_terms(self, target: Target) -> tuple[str, ...]:
        """The terms that already are this target's: repeating them in a proposal is noise."""
        namespace = self._base.get(self._namespace(target), {})
        return tuple(sorted(term for term, owner in namespace.items() if owner == target.label))

    def _known_columns(self, target: Target) -> frozenset[str]:
        """Every name a field may reference: this table's columns, its name and its source."""
        names = {target.source} if target.table is None else {target.table, target.source}
        table = self._table(target)
        if table is not None:
            names.update(column.name for column in table.columns)
        return frozenset(names)

    def _registry(self) -> dict[str, dict[str, str]]:
        """Every term that already means something, per namespace, before this pass proposes one.

        Seeded from the caller's curated terms, the catalogs (a column's technical name) and the
        value maps (canonical values and their existing labels). ``setdefault`` lets the earliest
        seed win, so a curated term is never overwritten by a technical name that spells the same.
        """
        registry: dict[str, dict[str, str]] = {
            scope: dict(terms) for scope, terms in self._known.items()
        }
        for source, catalog in self._catalogs.items():
            for table in catalog.tables:
                bucket = registry.setdefault(f"table:{source}.{table.name}", {})
                for column in table.columns:
                    bucket.setdefault(
                        normalize_label(column.name), f"{source}.{table.name}.{column.name}"
                    )
        for map_key, value_map in self._value_maps.items():
            bucket = registry.setdefault(f"map:{map_key}", {})
            for value in value_map.canonical:
                bucket.setdefault(normalize_label(value), f"{map_key}::{value}")
                for label in value_map.synonyms.get(value, ()):
                    bucket.setdefault(normalize_label(label), f"{map_key}::{value}")
        return registry

    # -- the pass ----------------------------------------------------------------------

    def _ask(
        self, llm: StructuredLLM, target: Target, counts: dict[str, int]
    ) -> tuple[Labels, Labels]:
        """Ask twice about one target: the same facts, the second time rotated.

        Two calls and not one, because confidence must not be the model's opinion of itself
        (DESIGN.md section 10). The rotation is part of the prompt, so the two answers are two
        independent readings of one fact sheet rather than one answer served twice from the cache.
        """
        answers: list[Labels] = []
        for rotation in (0, 1):
            result = llm.structured(
                render_prompt(self.fact_sheet(target, rotation=rotation)),
                Labels,
                system=SYSTEM_PROMPT,
                prompt_version=PROMPT_VERSION,
            )
            counts["model_calls"] = counts.get("model_calls", 0) + 1
            answers.append(result.value)
        return answers[0], answers[1]

    def _clean(self, target: Target, terms: Sequence[str]) -> tuple[tuple[str, ...], int]:
        """Drop empty, repeated and already-own terms; return what is left and how many went."""
        own = frozenset(normalize_label(term) for term in self._own_terms(target))
        kept: list[str] = []
        seen: set[str] = set()
        dropped = 0
        for term in terms:
            folded = normalize_label(term)
            if not folded or folded in seen or folded in own:
                dropped += 1
                continue
            seen.add(folded)
            kept.append(term)
        return tuple(kept), dropped

    def _payloads(
        self,
        target: Target,
        labels: Labels,
        *,
        synonyms: Sequence[str],
        provenance: JsonObject,
    ) -> list[tuple[ProposalKind, JsonObject]]:
        """The payloads an accepted answer becomes, in a fixed order.

        A value has no name of its own -- the canonical value *is* the name, and every label for it
        is a synonym -- so a value's answer becomes one ``value_maps`` synonym proposal. A table or
        a column gets two: its name and description, and its synonyms separately, because a human
        may well want the one and refuse the other.
        """
        if target.kind is TargetKind.VALUE:
            return [
                (
                    ProposalKind.SYNONYM,
                    {
                        "synonyms": {str(target.value): list(synonyms)},
                        "provenance": provenance,
                    },
                )
            ]
        payloads: list[tuple[ProposalKind, JsonObject]] = []
        if labels.name_he.strip() or labels.name_en.strip() or labels.description_en.strip():
            payloads.append(
                (
                    ProposalKind.NAME,
                    {
                        "name": {"he": labels.name_he, "en": labels.name_en},
                        "description": {"he": labels.description_he, "en": labels.description_en},
                        "provenance": provenance,
                    },
                )
            )
        if synonyms:
            payloads.append(
                (
                    ProposalKind.SYNONYM,
                    {
                        "synonyms": {
                            "he": list(labels.synonyms_he),
                            "en": list(labels.synonyms_en),
                        },
                        "provenance": provenance,
                    },
                )
            )
        return payloads

    def _evidence(
        self,
        *,
        model: str,
        agreed: int,
        sheet: FactSheet,
    ) -> tuple[str, ...]:
        """What a human reads before deciding: the ask, the two runs, the data behind them."""
        observed = len(sheet.observed_values)
        lines = [
            f"model: {model}",
            f"prompt_version: {PROMPT_VERSION}",
            f"temperature: {TEMPERATURE}",
            f"two runs, the second with the fact sheet rotated: {agreed} of "
            f"{len(AGREEMENT_FIELDS)} label fields identical",
            f"fact sheet: {hash_inputs(sheet.model_dump(mode='json'))[:16]}",
        ]
        if sheet.kind is TargetKind.VALUE:
            lines.append(
                f"value map evidence: canonical value {sheet.value!r}, "
                f"{len(sheet.existing_terms)} other term(s) already in this map"
            )
        elif sheet.distinct is None and observed == 0:
            lines.append("d5 evidence: none (this target was not profiled)")
        else:
            detail = [f"{observed} observed value(s)"]
            if sheet.null_rate is not None:
                detail.append(f"null rate {sheet.null_rate}")
            if sheet.distinct is not None:
                detail.append(f"{sheet.distinct} distinct")
            lines.append("d5 evidence: " + ", ".join(detail))
        lines.append(
            "validators passed: language, normalisation, length, column reference, collision, "
            "personal data"
        )
        return tuple(lines)

    def propose(self, ctx: PipelineContext) -> NamesReport:
        """Ask, validate, queue. The only side effect is pending proposals in the store.

        Per target: two model calls, then every validator, then -- only if nothing was refused --
        one or two ``Proposal`` rows in ``pending``. A refusal is reported and counted, never queued
        and never written to ``curated/``: the human gate (S12) is the only writer there.
        """
        llm = ctx.llm
        targets = self.targets()
        counts = {
            "targets": len(targets),
            "tables": sum(1 for target in targets if target.kind is TargetKind.TABLE),
            "columns": sum(1 for target in targets if target.kind is TargetKind.COLUMN),
            "values": sum(1 for target in targets if target.kind is TargetKind.VALUE),
            "proposals": 0,
            "created": 0,
            "unchanged": 0,
            "resurfaced": 0,
            "suppressed": 0,
            "rejected": 0,
            "model_calls": 0,
        }
        if llm is None:
            self._rejections = ()
            return NamesReport(
                model="none",
                counts=counts,
                notes=("no model client configured: nothing proposed",),
            )
        queue = ReviewQueue(ctx.store, ctx.semantic_root, clock=ctx.clock)
        taken = {scope: dict(terms) for scope, terms in self._base.items()}
        proposals: list[Proposal] = []
        rejections: list[Rejection] = []
        notes: list[str] = []
        fixes: list[str] = []
        for target in targets:
            first, second = self._ask(llm, target, counts)
            sheet = self.fact_sheet(target)
            known_columns = self._known_columns(target)
            refused = validate_labels(
                target, first, known_columns=known_columns, observed=sheet.observed_values
            )
            if refused:
                rejections.extend(refused)
                counts["rejected"] += 1
                continue
            replicate = validate_labels(
                target, second, known_columns=known_columns, observed=sheet.observed_values
            )
            if replicate:
                notes.append(
                    f"{target.label}: the second, reordered run was refused "
                    f"({replicate[0].code.value}); agreement counted as 0"
                )
                matched = 0
            else:
                matched = matched_fields(first, second)
            score = confidence(
                matched / len(AGREEMENT_FIELDS), evidenced=sheet.distinct is not None
            )
            synonyms, dropped = self._clean(target, self._synonyms_of(target, first, second))
            if dropped:
                fixes.append(f"{target.label}: dropped {dropped} repeated or existing term(s)")
            named: tuple[str, ...] = (
                () if target.kind is TargetKind.VALUE else (first.name_he, first.name_en)
            )
            found = collisions(target, (*named, *synonyms), taken.get(self._namespace(target), {}))
            if found:
                rejections.extend(found)
                counts["rejected"] += 1
                continue
            evidence = self._evidence(model=llm.model_name, agreed=matched, sheet=sheet)
            provenance: JsonObject = {
                "detected_by": STAGE_NAME,
                "evidence": list(evidence),
                "match_rate": score,
            }
            for kind, patch in self._payloads(
                target, first, synonyms=synonyms, provenance=provenance
            ):
                result = queue.submit(
                    Proposal(
                        id=proposal_id(kind, self._section(target), target.key, "", patch),
                        kind=kind,
                        section=self._section(target),
                        key=target.key,
                        patch=patch,
                        evidence=evidence,
                        confidence=score,
                        proposed_by=STAGE_NAME,
                        proposed_at=ctx.clock.now(),
                    )
                )
                counts["proposals"] += 1
                counts[result.action.value] = counts.get(result.action.value, 0) + 1
                proposals.append(result.proposal)
            self._register(taken, target, first, synonyms)
        self._rejections = tuple(rejections)
        logger.info(
            "names proposed",
            extra={
                "model": llm.model_name,
                "prompt_version": PROMPT_VERSION,
                "created": counts["created"],
                "rejected": counts["rejected"],
            },
        )
        return NamesReport(
            model=llm.model_name,
            counts=counts,
            proposals=tuple(proposals),
            rejections=self._rejections,
            notes=tuple(notes),
            fixes=tuple(fixes),
        )

    def _synonyms_of(self, target: Target, first: Labels, second: Labels) -> tuple[str, ...]:
        """The terms to consider, in a fixed order.

        A table or a column proposes what its *first* answer said, because that answer is the
        proposal; the second run only measures agreement. A value proposes the union of both,
        because for a value a second word for the same value is a synonym, not a contradiction.
        """
        if target.kind is TargetKind.VALUE:
            return (
                *first.synonyms_he,
                first.name_he,
                *first.synonyms_en,
                first.name_en,
                *second.synonyms_he,
                second.name_he,
                *second.synonyms_en,
                second.name_en,
            )
        return (*first.synonyms_he, *first.synonyms_en)

    def _section(self, target: Target) -> str:
        """Which curated file a proposal belongs in: a value's labels live in the value maps."""
        return VALUE_MAPS_SECTION if target.kind is TargetKind.VALUE else NAMES_SECTION

    def _register(
        self,
        taken: dict[str, dict[str, str]],
        target: Target,
        labels: Labels,
        synonyms: Sequence[str],
    ) -> None:
        """Record the accepted terms, so a later target in the namespace cannot take them."""
        bucket = taken.setdefault(self._namespace(target), {})
        terms = list(synonyms)
        if target.kind is not TargetKind.VALUE:
            terms.extend([labels.name_he, labels.name_en])
        for term in terms:
            folded = normalize_label(term)
            if folded:
                bucket.setdefault(folded, target.label)

    def run(self, ctx: PipelineContext) -> StageReport:
        """Run the pass and report it: a refusal is named and counted, never silently dropped."""
        report = self.propose(ctx)
        skipped = tuple(
            f"rejected {item.target} ({item.field}, {item.code.value}): {item.detail}"
            for item in report.rejections
        )
        return StageReport(
            stage=self.name,
            status=StageStatus.RAN,
            counts=dict(report.counts),
            skipped=skipped + report.notes,
            applied_fixes=report.fixes,
        )


def _map_source(value_map: ValueMap) -> str:
    """The source a value map's columns belong to, from its ``source.table.column`` keys."""
    for column in sorted(value_map.sources):
        parts = column.split(".")
        if len(parts) >= 3:
            return parts[0]
    return "unknown"
