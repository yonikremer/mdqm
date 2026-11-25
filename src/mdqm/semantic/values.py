"""Resolve a value an agent supplied to a canonical value, and back to per-source literals.

The semantic layer names a categorical field once (``active``) and each source spells it its own
way (``A`` in the CRM, ``Active`` in billing, ``1`` in the ERP). This module is the one place that
turns a label a user or a weak model wrote — Hebrew or English, with niqqud, with a geresh or
gershayim, with stray spaces, in any case — into that canonical value, and the one place that
turns a canonical value back into the literal a source's driver expects.

Two rules decide the design (``docs/DESIGN.md`` sections 5 and 4):

* **A filter never silently matches nothing.** An input that matches no canonical value, or that
  matches more than one, comes back as a :class:`Resolution` carrying the valid canonical values
  and the nearest spellings, with the matching ``UNKNOWN_VALUE``/``AMBIGUOUS_VALUE`` error already
  built for the agent. ``None`` never means "I could not tell".
* **No guessing between spellings.** A label that is a canonical value and also a synonym of a
  different value is ambiguous, not a precedence puzzle; so is one source code that two sources
  spell for different values. Naming the column (``column=``) is how a caller states the context
  that removes the ambiguity.

The functions are pure: they read a :class:`~mdqm.semantic.models.ValueMap` and never change it,
never touch a source and never see a secret.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from mdqm.errors import ambiguous_value, unknown_value
from mdqm.models import ErrorResponse
from mdqm.semantic.models import ValueMap
from mdqm.semantic.text import normalize_label

__all__ = [
    "MAX_SUGGESTIONS",
    "Resolution",
    "ResolutionKind",
    "ReverseResult",
    "SourceLiteral",
    "resolve_value",
    "reverse_value",
]

#: How many ``did_you_mean`` spellings a near miss carries. A list of every value is already in
#: ``valid_values``; suggestions are the shortlist a weak model can actually pick from.
MAX_SUGGESTIONS: Final[int] = 3

#: The field name used in an error when the caller does not know the spec path.
DEFAULT_FIELD: Final[str] = "value"

#: A source code that is an integer, and only when a round trip through ``int`` is lossless: no
#: leading zeros (``"01"`` is a zero-padded identifier, not the number 1), no ``+``, no underscore.
#: Bounded to 18 digits so a long identifier never becomes a number silently.
_INTEGER_CODE: Final[re.Pattern[str]] = re.compile(r"-?(?:0|[1-9][0-9]{0,17})")

#: difflib's cut-off. Below 0.6 a "suggestion" is a different word and only adds noise.
_SUGGESTION_CUTOFF: Final[float] = 0.6


class ResolutionKind(StrEnum):
    """What a lookup found: exactly one canonical value, several, or none."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Resolution:
    """The outcome of resolving one user-supplied label.

    ``canonical`` is set only when ``kind`` is :attr:`ResolutionKind.RESOLVED`. ``candidates``
    carries what the agent may pick from: the matched values when ambiguous, every valid value when
    unknown. An empty ``candidates`` therefore never means "nothing to say".
    """

    label: str | int
    field: str
    kind: ResolutionKind
    canonical: str | None = None
    candidates: tuple[str, ...] = ()
    did_you_mean: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when exactly one canonical value was found."""
        return self.kind is ResolutionKind.RESOLVED

    @property
    def error(self) -> ErrorResponse | None:
        """The agent-facing rejection, or ``None`` when ``ok``.

        Built by the one builder per code in :mod:`mdqm.errors`, so the texts stay the taxonomy's;
        only ``did_you_mean`` is attached here, because the events an agent must fix are exactly
        the two this module produces and the suggestion comes from the map this module holds.
        """
        if self.kind is ResolutionKind.RESOLVED:
            return None
        if self.kind is ResolutionKind.AMBIGUOUS:
            return ambiguous_value(
                field=self.field, value=str(self.label), canonical=self.candidates
            )
        response = unknown_value(field=self.field, value=str(self.label), valid=self.candidates)
        if not self.did_you_mean:
            return response
        return response.model_copy(update={"did_you_mean": self.did_you_mean})


@dataclass(frozen=True, slots=True)
class SourceLiteral:
    """One source column's spelling of a canonical value, with the type its driver expects."""

    column: str
    literal: str | int

    @property
    def is_integer(self) -> bool:
        """True when the source stores this code as a number (``1``, not ``"1"``)."""
        return isinstance(self.literal, int)


@dataclass(frozen=True, slots=True)
class ReverseResult:
    """The per-source literals a canonical value compiles to, or the reason there are none.

    A canonical value that no source lists, or no source of the named column, comes back with
    ``literals == ()`` and no error: the value is valid, it simply is not stored there, and that
    is the caller's (the compiler's) decision, not a bad filter value.
    """

    label: str | int
    field: str
    canonical: str | None = None
    literals: tuple[SourceLiteral, ...] = ()
    error: ErrorResponse | None = None

    @property
    def ok(self) -> bool:
        """True when the label resolved to a canonical value."""
        return self.error is None


def _suggest(key: str, known: Sequence[str]) -> tuple[str, ...]:
    """The nearest known canonical values to a normalised label, spelling as the map spells them."""
    by_key = {normalize_label(name): name for name in known}
    close = difflib.get_close_matches(
        key, list(by_key), n=MAX_SUGGESTIONS, cutoff=_SUGGESTION_CUTOFF
    )
    return tuple(by_key[folded] for folded in close)


def _matched(value_map: ValueMap, key: str, column: str | None) -> tuple[str, ...]:
    """Every canonical value ``key`` is a label for, in the map's own order.

    Canonical spellings, both-language synonyms and the source codes of ``column`` (or of every
    listed column when ``column`` is ``None``) all count. Folding the codes too is deliberate: an
    agent that writes ``hold`` for billing's ``Hold`` must be understood, and a map that spells two
    different values ``A`` and ``a`` is reported as ambiguous instead of being guessed.
    """
    hits: set[str] = set()
    for value in value_map.canonical:
        if normalize_label(value) == key:
            hits.add(value)
    for value, labels in value_map.synonyms.items():
        if any(normalize_label(label) == key for label in labels):
            hits.add(value)
    for name, codes in sorted(value_map.sources.items()):
        if column is not None and name != column:
            continue
        hits.update(canonical for code, canonical in codes.items() if normalize_label(code) == key)
    return tuple(value for value in value_map.canonical if value in hits)


def resolve_value(
    value_map: ValueMap,
    label: str,
    *,
    field: str = DEFAULT_FIELD,
    column: str | None = None,
) -> Resolution:
    """Resolve ``label`` in Hebrew, English or a source code to the map's canonical value.

    Args:
        value_map: the field's map from the semantic layer.
        label: what the user or the agent wrote; normalised here, never trusted as an exact key.
        field: the spec path an error names, e.g. ``"filters.0.value"``.
        column: the ``source.table.column`` a filter will be compiled for. It scopes code
            matching, which is what tells two sources that spell one code differently apart.

    Returns:
        A :class:`Resolution`; ``.error`` is the ``UNKNOWN_VALUE`` or ``AMBIGUOUS_VALUE`` rejection
        with the valid canonical values, and ``.did_you_mean`` the nearest spellings.
    """
    key = normalize_label(str(label))
    matched = _matched(value_map, key, column)
    if len(matched) == 1:
        return Resolution(
            label=label,
            field=field,
            kind=ResolutionKind.RESOLVED,
            canonical=matched[0],
        )
    if matched:
        return Resolution(
            label=label,
            field=field,
            kind=ResolutionKind.AMBIGUOUS,
            candidates=matched,
        )
    return Resolution(
        label=label,
        field=field,
        kind=ResolutionKind.UNKNOWN,
        candidates=value_map.canonical,
        did_you_mean=_suggest(key, value_map.canonical),
    )


def _literal(code: str) -> str | int:
    """``"1"`` -> ``1``, ``"A"`` -> ``"A"``: the type the source column's driver expects."""
    return int(code) if _INTEGER_CODE.fullmatch(code) else code


def reverse_value(
    value_map: ValueMap,
    label: str,
    *,
    field: str = DEFAULT_FIELD,
    column: str | None = None,
) -> ReverseResult:
    """Map a label — canonical, synonym or code — to the literal each source stores.

    Used on the compile side: a filter the agent wrote must become ``'A'`` on the CRM,
    ``'Active'`` on billing and ``1`` on the ERP. Order is deterministic: columns sorted by name,
    and within a column the order the map lists its codes, so golden SQL never moves.

    Args:
        value_map: the field's map from the semantic layer.
        label: the filter value as supplied; resolved first, so any spelling works here too.
        field: the spec path an error names.
        column: restrict the literals to one ``source.table.column``.

    Returns:
        A :class:`ReverseResult` whose ``error`` is the forward resolution's rejection, so a bad
        value is reported once, in the words the agent must act on.
    """
    resolution = resolve_value(value_map, label, field=field, column=column)
    if not resolution.ok or resolution.canonical is None:
        return ReverseResult(label=label, field=field, error=resolution.error)
    canonical = resolution.canonical
    literals = tuple(
        SourceLiteral(column=name, literal=_literal(code))
        for name, codes in sorted(value_map.sources.items())
        if column is None or name == column
        for code, target in codes.items()
        if target == canonical
    )
    return ReverseResult(label=label, field=field, canonical=canonical, literals=literals)
