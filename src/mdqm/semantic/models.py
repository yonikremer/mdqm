"""The semantic layer: what the YAML in ``generated/`` and ``curated/`` means.

One immutable :class:`SemanticLayer` holds every business fact an agent needs instead of raw SQL:
which column is which measure, how each source stores it, which codes mean which category in
Hebrew and English, which time column answers which question, how often a table changes, and which
joins a human approved. The models are frozen and reject unknown fields: a typo in a hand-edited
curated file must fail loudly, not be dropped.

The layer is written by the bootstrap pipeline into ``generated/`` and by humans into ``curated/``.
Provenance travels with every item, because a proposal, a measured fact and an approved fact must
never look the same (``docs/DESIGN.md`` section 10).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from string import Formatter
from typing import Annotated, Any, TypeAlias, TypeVar

import sqlglot
from pydantic import AfterValidator, Field, field_validator, model_validator
from sqlglot.errors import ParseError

from mdqm.cache.freshness import (
    CACHEABLE_MUTABILITY,
    DEFAULT_MAX_TTL,
    DEFAULT_MIN_TTL,
    DEFAULT_THETA,
    MUTABILITY_CLASSES,
    Mutability,
)
from mdqm.config.quiet import QuietWindow
from mdqm.config.secrets import check_secret_name
from mdqm.config.settings import AuthSettings, LoadLimits, RateLimitSettings
from mdqm.models.base import MDQMModel, UtcDatetime
from mdqm.models.catalog import Column, Index
from mdqm.models.partitioning import Partitioning
from mdqm.semantic.durations import Duration
from mdqm.semantic.text import normalize_label

__all__ = [
    "ApprovalStatus",
    "ApprovedJoin",
    "CacheDefaults",
    "CacheOverride",
    "Defaults",
    "FrozenMapping",
    "JoinCardinality",
    "JoinUsage",
    "Lag",
    "Measure",
    "MeasureCanonical",
    "MeasureSource",
    "MutabilityDecl",
    "Provenance",
    "RoundingMode",
    "RoundingRule",
    "SemanticLayer",
    "SourceDecl",
    "TableSemantics",
    "TimeColumn",
    "TimeRole",
    "compute_version_hash",
    "render_template",
]

_K = TypeVar("_K")
_V = TypeVar("_V")
_T = TypeVar("_T")


class FrozenMapping(dict[_K, _V]):
    """A dict that refuses every write, so a loaded layer cannot be changed by accident.

    A plain ``dict`` inside a frozen pydantic model is still mutable, and one agent that writes
    into a shared layer would poison every later query. Reads and JSON serialisation behave
    exactly like a dict.
    """

    __slots__ = ()

    def _immutable(self, *args: object, **kwargs: object) -> Any:
        raise TypeError("the semantic layer is immutable: build a new layer instead")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    pop = _immutable
    popitem = _immutable
    clear = _immutable
    # `dict.update`/`dict.setdefault` declare `self` as a constrained protocol, so no override can
    # match their overloads; the assignment is deliberate and the call always raises.
    update = _immutable  # pyright: ignore[reportAssignmentType]
    setdefault = _immutable  # pyright: ignore[reportAssignmentType]


def _freeze(mapping: Mapping[str, _T]) -> dict[str, _T]:
    return FrozenMapping(mapping)


#: A string-keyed mapping that cannot be mutated after validation.
Frozen: TypeAlias = Annotated[dict[str, _T], AfterValidator(_freeze)]


def _qualified(name: str, what: str) -> str:
    """Check a ``source.table.column`` reference: an unqualified name is a guess, not a fact."""
    parts = name.split(".")
    if len(parts) < 3 or not all(parts):
        raise ValueError(
            f"{what} must be written as 'source.table.column', got {name!r} "
            "(a bare column name is ambiguous across sources)"
        )
    return name


def _text_keys(codes: Any) -> Any:
    """Stringify a code map's keys: a YAML ``1`` and a source's ``1`` compare as ``"1"``."""
    if not isinstance(codes, Mapping):
        return codes
    return {str(code): canonical for code, canonical in codes.items()}


def _template_fields(template: str) -> tuple[str, ...]:
    try:
        return tuple(field for _, field, _, _ in Formatter().parse(template) if field)
    except ValueError as exc:
        raise ValueError(f"{template!r} is not a valid template: {exc}") from None


def check_sql_template(
    template: str,
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    what: str,
) -> str:
    """Reject a template that names an unknown placeholder or does not parse as one expression.

    A normalize template and an approved join's ``on`` clause become SQL, so a broken template is
    caught here, at load time, with the file and the line — not in the middle of a query.
    """
    fields = _template_fields(template)
    unknown = sorted(set(fields) - allowed)
    if unknown:
        raise ValueError(
            f"{what} may only use {{{', '.join(sorted(allowed))}}}, "
            f"got the unknown placeholder(s) {', '.join(unknown)}"
        )
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"{what} must use the placeholder(s) {', '.join(missing)}")
    sql = render_template(template, dict.fromkeys(allowed, "x"))
    try:
        statements = sqlglot.parse(sql)
    except ParseError as exc:
        first = str(exc).splitlines()[0]
        raise ValueError(f"{what} is not valid SQL: {first}") from None
    if len([statement for statement in statements if statement is not None]) != 1:
        raise ValueError(
            f"{what} must be a single SQL expression, got {len(statements)} statements"
        )
    return template


def render_template(template: str, values: Mapping[str, str]) -> str:
    """Substitute ``{field}`` placeholders with already-validated SQL text.

    Only the declared placeholders are replaced, so a stray brace in the template cannot smuggle
    in a value; the caller (later the compiler) is responsible for what ``values`` contain.
    """
    rendered = template
    for field in _template_fields(template):
        if field not in values:
            raise ValueError(f"no value for the placeholder {{{field}}} in {template!r}")
        rendered = rendered.replace("{" + field + "}", values[field])
    return rendered


# --- provenance and classification -------------------------------------------------------


class ApprovalStatus(StrEnum):
    """Where a fact came from and whether a human accepted it."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"


class Provenance(MDQMModel):
    """Who claims this fact, on what evidence, and who accepted it.

    A deterministic stage sets ``detected_by``; a model proposal additionally has a ``match_rate``
    and needs ``approved_by`` before anything depends on it.
    """

    detected_by: str | None = None
    evidence: str | tuple[str, ...] | None = None
    match_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    approved_by: str | None = None
    approved_at: UtcDatetime | None = None


class MutabilityDecl(MDQMModel):
    """How often a table changes, and whether a human approved that claim.

    An unclassified table is ``unknown`` and behaves like ``mutable_critical``: never served from
    cache. ``may_cache`` is False for a class nobody approved, because caching a table that does
    change answers with stale rows and looks like success.
    """

    class_: Mutability = Field(alias="class")
    status: ApprovalStatus = ApprovalStatus.PROPOSED

    @field_validator("class_", mode="before")
    @classmethod
    def _known_class(cls, value: Any) -> Any:
        if isinstance(value, str) and value not in MUTABILITY_CLASSES:
            raise ValueError(
                f"unknown mutability class {value!r}: use one of {list(MUTABILITY_CLASSES)} "
                "('unknown' is the conservative choice until the class is measured)"
            )
        return value

    @property
    def may_cache(self) -> bool:
        """True only for an approved, provably append-only class."""
        return self.status is ApprovalStatus.APPROVED and self.class_ in CACHEABLE_MUTABILITY

    @property
    def behaves_as_mutable(self) -> bool:
        """True when the table must be treated as changing (the conservative reading)."""
        return not self.may_cache


class TimeRole(StrEnum):
    """What a timestamp column means, which decides which question may use it."""

    EVENT_TIME = "event_time"
    INGESTION_TIME = "ingestion_time"


class Lag(MDQMModel):
    """Measured ingestion lag: how far behind the event time a row can be inserted."""

    minimum: Duration = Field(alias="min")
    maximum: Duration = Field(alias="max")
    verify: str = Field(default="none", pattern="^(none|nightly)$")

    @model_validator(mode="after")
    def _ordered(self) -> Lag:
        if self.maximum < self.minimum:
            raise ValueError(
                f"lag.max ({self.maximum}) is smaller than lag.min ({self.minimum}); "
                "a negative lag is a measurement bug, re-measure instead of guessing"
            )
        return self


class TimeColumn(MDQMModel):
    """One timestamp column of a table and what it is good for.

    ``derives_bound_on`` names the indexed event-time column a filter on this (ingestion-time)
    column can narrow, using :attr:`Lag.maximum`. Deriving without a measured lag would silently
    drop late rows, so ``lag`` is required together with ``derives_bound_on``.
    """

    role: TimeRole
    indexed: bool = False
    use_by_default: bool = False
    derives_bound_on: str | None = None
    lag: Lag | None = None

    @model_validator(mode="after")
    def _derivation_needs_lag(self) -> TimeColumn:
        if self.derives_bound_on is not None:
            if self.role is not TimeRole.INGESTION_TIME:
                raise ValueError(
                    "derives_bound_on belongs on the ingestion-time column "
                    f"(role: ingestion_time), this one is role: {self.role.value}"
                )
            if self.lag is None:
                raise ValueError(
                    "derives_bound_on needs a measured lag (lag: {min: ..., max: ...}): "
                    "without max_lag the derived bound would drop late rows"
                )
        return self


class CacheOverride(MDQMModel):
    """A per-table cache exception. ``mode: none`` never caches, which is always safe."""

    mode: str = Field(default="default", pattern="^(default|none)$")
    staleness_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
    min_ttl: Duration | None = None
    max_ttl: Duration | None = None
    short_ttl: Duration | None = None

    @property
    def enabled(self) -> bool:
        """False when this table must never be served from cache."""
        return self.mode != "none"


class TableSemantics(MDQMModel):
    """Everything the planner, the linter and the cache need to know about one table.

    ``columns`` and ``indexes`` are the *skeleton* a deterministic stage writes: for a database the
    catalog, for an OpenAPI source one GET collection with its item properties as columns and its
    query parameters as filterable indexes (stage D4). They are carried through the catalog models
    rather than copied, so a skeleton and a probe cannot disagree about what a column is.
    """

    key: tuple[str, ...] = ()
    default_window: Duration | None = None
    partitioning: Partitioning | None = None
    time_columns: Frozen[TimeColumn] = Field(default_factory=lambda: FrozenMapping())
    mutability: MutabilityDecl | None = None
    cache: CacheOverride | None = None
    columns: tuple[Column, ...] = ()
    indexes: tuple[Index, ...] = ()
    provenance: Provenance | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, data: Any) -> Any:
        """Accept the file's shorter partitioning form: a bare ``key: col`` and no strategy."""
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        partitioning = values.get("partitioning")
        if isinstance(partitioning, Mapping):
            section = dict(partitioning)
            key = section.get("key")
            if isinstance(key, str):
                section["key"] = [key]
            if "strategy" not in section:
                section["strategy"] = (
                    "expression" if section.get("expression") else ("range" if key else "none")
                )
            values["partitioning"] = section
        return values

    @model_validator(mode="after")
    def _time_columns_consistent(self) -> TableSemantics:
        for name, column in self.time_columns.items():
            target = column.derives_bound_on
            if target is None:
                continue
            parent = self.time_columns.get(target)
            if parent is None:
                raise ValueError(
                    f"time column {name!r} derives its bound on {target!r}, "
                    "which is not listed under time_columns"
                )
            if parent.role is not TimeRole.EVENT_TIME:
                raise ValueError(
                    f"time column {name!r} derives its bound on {target!r}, "
                    f"but that column has role {parent.role.value}, not event_time"
                )
        return self

    @property
    def may_cache(self) -> bool:
        """True only when the table's cache is not disabled and its mutability was approved."""
        if self.cache is not None and not self.cache.enabled:
            return False
        return self.mutability is not None and self.mutability.may_cache


# --- value maps --------------------------------------------------------------------------


class ValueMap(MDQMModel):
    """One categorical field: its canonical values, both-language labels and per-source codes.

    Codes are stored as text even when the source stores numbers (``1`` in one ERP is written
    ``erp.customers.status_code: {1: active}``), because the value that arrives from a source is
    compared as text.
    """

    canonical: tuple[str, ...] = Field(min_length=1)
    synonyms: Frozen[tuple[str, ...]] = Field(default_factory=lambda: FrozenMapping())
    sources: Frozen[Frozen[str]] = Field(default_factory=lambda: FrozenMapping())
    provenance: Provenance | None = None

    @model_validator(mode="before")
    @classmethod
    def _text_codes(cls, data: Any) -> Any:
        """Source codes arrive as YAML scalars: ``1`` and ``A`` both compare as text."""
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        sources = values.get("sources")
        if isinstance(sources, Mapping):
            values["sources"] = {
                str(column): _text_keys(codes) for column, codes in sources.items()
            }
        return values

    @model_validator(mode="after")
    def _consistent(self) -> ValueMap:
        canonical = tuple(normalize_label(value) for value in self.canonical)
        if len(set(canonical)) != len(canonical):
            raise ValueError(
                f"canonical values must be distinct, got {list(self.canonical)} "
                "(two spellings of one value are separate values)"
            )
        if any(not value for value in canonical):
            raise ValueError("a canonical value cannot be empty")
        allowed = set(self.canonical)
        seen: dict[str, str] = {}
        for value, labels in self.synonyms.items():
            if value not in allowed:
                raise ValueError(
                    f"synonym group {value!r} is not one of the canonical values "
                    f"{list(self.canonical)}"
                )
            for label in labels:
                key = normalize_label(label)
                other = seen.get(key)
                if other is not None and other != value:
                    # Ambiguity would make the linter pick a value at random.
                    raise ValueError(
                        f"label {label!r} is a synonym of both {other!r} and {value!r}"
                    )
                seen[key] = value
        for column, codes in self.sources.items():
            _qualified(column, "a value map source column")
            stray = sorted({code for code in codes.values()} - allowed)
            if stray:
                raise ValueError(
                    f"source {column!r} maps to {stray}, which are not canonical values "
                    f"{list(self.canonical)}"
                )
        return self

    def resolve(self, label: str) -> str | None:
        """Return the canonical value for a label in Hebrew or English, or None if unknown."""
        key = normalize_label(label)
        for value in self.canonical:
            if normalize_label(value) == key:
                return value
        for value, labels in self.synonyms.items():
            if any(normalize_label(label_) == key for label_ in labels):
                return value
        return None

    def canonical_for(self, column: str, code: object) -> str | None:
        """Return the canonical value a source code stands for, or None if column/code is new."""
        codes = self.sources.get(column)
        return None if codes is None else codes.get(str(code))


# --- measures ----------------------------------------------------------------------------


class RoundingMode(StrEnum):
    """Rounding is a business decision, so the mode is stated, never inferred."""

    HALF_UP = "half_up"
    HALF_EVEN = "half_even"
    HALF_DOWN = "half_down"
    FLOOR = "floor"
    CEIL = "ceil"


class RoundingRule(MDQMModel):
    """Round to a multiple of ``to``: ``round_half_up(x / 5000) * 5000`` in the YAML."""

    to: int = Field(gt=0)
    mode: RoundingMode = RoundingMode.HALF_UP


class MeasureCanonical(MDQMModel):
    """The one true definition of the business field, before any source is involved."""

    rounding: RoundingRule | None = None


class MeasureSource(MDQMModel):
    """How one source stores the measure, as a template over ``{col}``."""

    normalize: str

    @model_validator(mode="after")
    def _template(self) -> MeasureSource:
        check_sql_template(
            self.normalize,
            allowed=frozenset({"col"}),
            required=frozenset({"col"}),
            what="a measure normalize template",
        )
        # Parsing is not enough: a subquery and an unknown function both parse. The allowlist is a
        # meaning rule, so it sits beside the rounding it protects. Imported here because
        # ``measures`` imports these models in turn.
        from mdqm.semantic.measures import check_normalize_template

        check_normalize_template(self.normalize, where="a measure normalize template")
        return self


class Measure(MDQMModel):
    """A business measure plus its per-source normalisation and its provenance."""

    canonical: MeasureCanonical = Field(default_factory=MeasureCanonical)
    sources: Frozen[MeasureSource] = Field(default_factory=lambda: FrozenMapping())
    provenance: Provenance | None = None

    @model_validator(mode="after")
    def _qualified_sources(self) -> Measure:
        for column in self.sources:
            _qualified(column, "a measure source column")
        return self

    def normalize_for(self, column: str, identifier: str) -> str | None:
        """The SQL expression for ``column`` in this measure, or None when it is not listed."""
        source = self.sources.get(column)
        if source is None:
            return None
        return render_template(source.normalize, {"col": identifier})

    @property
    def is_identity_everywhere(self) -> bool:
        """True when no source needs a conversion (the usual case)."""
        return all(source.normalize.strip() == "{col}" for source in self.sources.values())


# --- sources -----------------------------------------------------------------------------


class SourceDecl(MDQMModel):
    """A data source as the semantic layer declares it.

    ``secret`` is a *name* a SecretProvider resolves, never a value, and unknown fields are
    rejected, so a literal credential in this file is a validation error rather than a leak.
    """

    adapter: str = Field(min_length=1)
    secret: str | None = None
    spec: str | None = None
    load: LoadLimits = Field(default_factory=LoadLimits)
    auth: AuthSettings | None = None
    rate_limit: RateLimitSettings | None = None
    provenance: Provenance | None = None

    @model_validator(mode="after")
    def _secret_names(self) -> SourceDecl:
        if self.secret is not None:
            check_secret_name(self.secret)
        return self


# --- approved joins ----------------------------------------------------------------------


class JoinCardinality(StrEnum):
    """Cardinality of an approved join; ``many_to_one`` is the only shape DuckDB can count on."""

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class JoinUsage(MDQMModel):
    """How often this join was seen in past queries. It orders presentation, it approves nothing."""

    count_90d: int = Field(default=0, ge=0)
    users_90d: int = Field(default=0, ge=0)


class ApprovedJoin(MDQMModel):
    """A cross-source join a human approved, including its format conversion.

    The ``on`` template is the whole join condition over ``{left}`` and ``{right}``, so the
    zero-padding of an integer key lives with the join instead of being re-invented per query.
    """

    name: str = Field(min_length=1)
    left: str
    right: str
    on: str
    cardinality: JoinCardinality
    usage: JoinUsage | None = None
    provenance: Provenance | None = None

    @model_validator(mode="after")
    def _shape(self) -> ApprovedJoin:
        _qualified(self.left, "an approved join's left side")
        _qualified(self.right, "an approved join's right side")
        check_sql_template(
            self.on,
            allowed=frozenset({"left", "right"}),
            required=frozenset({"left", "right"}),
            what="an approved join 'on' clause",
        )
        return self


# --- defaults and the layer --------------------------------------------------------------


class CacheDefaults(MDQMModel):
    """The cache rule defaults (``docs/DESIGN.md`` section 8), overridable per source and table."""

    staleness_ratio: float = Field(default=DEFAULT_THETA, gt=0.0, le=1.0)
    min_ttl: Duration = DEFAULT_MIN_TTL
    max_ttl: Duration = DEFAULT_MAX_TTL

    @model_validator(mode="after")
    def _ordered(self) -> CacheDefaults:
        if self.max_ttl < self.min_ttl:
            raise ValueError(
                f"defaults.cache.max_ttl ({self.max_ttl}) is smaller than min_ttl ({self.min_ttl})"
            )
        return self


class Defaults(MDQMModel):
    """Layer-wide defaults: the cache rule and the hours in which heavy jobs may run."""

    cache: CacheDefaults = Field(default_factory=CacheDefaults)
    quiet_windows: tuple[QuietWindow, ...] = ()


class SemanticLayer(MDQMModel):
    """The merged semantic layer: immutable, with a content hash for cache keys and audit.

    ``version_hash`` covers every semantic fact in the layer (not comments, not layout), so a
    cached answer can never be reused after the meaning changed.
    """

    version: int = Field(default=0, ge=0)
    sources: Frozen[SourceDecl] = Field(default_factory=lambda: FrozenMapping())
    defaults: Defaults = Field(default_factory=Defaults)
    value_maps: Frozen[ValueMap] = Field(default_factory=lambda: FrozenMapping())
    measures: Frozen[Measure] = Field(default_factory=lambda: FrozenMapping())
    tables: Frozen[TableSemantics] = Field(default_factory=lambda: FrozenMapping())
    approved_joins: tuple[ApprovedJoin, ...] = ()
    version_hash: str = ""

    @model_validator(mode="after")
    def _unique_names(self) -> SemanticLayer:
        names = [join.name for join in self.approved_joins]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"approved_joins repeats the name(s) {duplicates}: one join has one definition"
            )
        return self

    @model_validator(mode="after")
    def _fill_version_hash(self) -> SemanticLayer:
        if self.version_hash:
            return self
        return self.model_copy(update={"version_hash": compute_version_hash(self)})


def compute_version_hash(layer: SemanticLayer) -> str:
    """SHA-256 over the layer's semantic content, as canonical JSON.

    Comments, ordering of keys and whitespace cannot reach this hash because it hashes the
    validated models, not the file text.
    """
    payload = layer.model_dump(mode="json", by_alias=True, exclude={"version_hash"})
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
