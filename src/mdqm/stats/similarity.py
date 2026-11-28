"""Closest categorical values: case, whitespace, Hebrew, transliteration, then edit distance.

Deterministic. The same query and the same counts always return the same suggestions in the
same order. Normalization is the semantic layer's own (:func:`mdqm.semantic.text.normalize_label`),
so a match here agrees with a match the value map would make. Transliteration is a small
letter map, not a dictionary: it connects ``שלום`` with ``shalom`` and nothing else. Business
synonyms (``פעיל`` / ``active``) come from the value map the caller passes in.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from mdqm.semantic.text import normalize_label

__all__ = [
    "MAX_SUGGESTIONS",
    "Suggestion",
    "edit_distance",
    "equivalent",
    "forms_of",
    "suggest",
]

#: How many alternatives a refusal carries. A weak agent picks from a short list.
MAX_SUGGESTIONS: Final[int] = 5

#: Edit distance at or below this is "close". Beyond it the suggestion is a different word.
_CLOSE: Final[int] = 2

#: Strings longer than this are compared on a prefix. The bound keeps the cost flat.
_MAX_COMPARE: Final[int] = 64

# One Latin reading per Hebrew letter, plus the two letters that are commonly read two ways.
# Final forms fold onto the same letter. Alef and ayin carry no consonant.
_LETTER: Final[dict[str, str]] = {
    "א": "",
    "ב": "b",
    "ג": "g",
    "ד": "d",
    "ה": "h",
    "ו": "v",
    "ז": "z",
    "ח": "ch",
    "ט": "t",
    "י": "y",
    "כ": "k",
    "ך": "k",
    "ל": "l",
    "מ": "m",
    "ם": "m",
    "נ": "n",
    "ן": "n",
    "ס": "s",
    "ע": "",
    "פ": "p",
    "ף": "p",
    "צ": "ts",
    "ץ": "ts",
    "ק": "k",
    "ר": "r",
    "ש": "sh",
    "ת": "t",
}
_ALTERNATE: Final[dict[str, str]] = {"ו": "o", "י": "i"}
# ו and י are often vowels (mater lectionis) and then carry no consonant at all.
_MATER: Final[frozenset[str]] = frozenset({"ו", "י"})
_LATIN_VOWELS: Final[frozenset[str]] = frozenset("aeiou")

#: Cap on transliteration variants. Two ambiguous letters are enough for names; more is noise.
_MAX_VARIANTS: Final[int] = 16


class Suggestion:
    """One real value, how far it is from what was typed, and how often it occurs."""

    __slots__ = ("count", "distance", "rows", "value")

    def __init__(self, value: str, distance: int, count: int, rows: int) -> None:
        self.value = value
        self.distance = distance
        self.count = count
        self.rows = rows

    def frequency(self) -> str:
        """``64 of 100, 64%``. The count is the sample; the percent is that count over the rows."""
        if self.rows <= 0:
            return str(self.count)
        percent = round(100 * self.count / self.rows)
        return f"{self.count} of {self.rows}, {percent}%"


def edit_distance(left: str, right: str) -> int:
    """Levenshtein distance. Compared on at most :data:`_MAX_COMPARE` characters."""
    left = left[:_MAX_COMPARE]
    right = right[:_MAX_COMPARE]
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for index, character in enumerate(left, start=1):
        current = [index]
        for column, other in enumerate(right, start=1):
            insert = current[column - 1] + 1
            delete = previous[column] + 1
            replace = previous[column - 1] + (character != other)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _skeleton(text: str) -> str:
    """Consonants only, so ``shalom`` and a vowelless transliteration of ``שלום`` can meet."""
    return "".join(character for character in text if character not in _LATIN_VOWELS)


def _remember(found: list[str], text: str) -> None:
    folded = normalize_label(text)
    if folded:
        found.append(folded)
    skeleton = _skeleton(folded)
    if skeleton and skeleton != folded:
        found.append(skeleton)


def _transliterations(text: str) -> tuple[str, ...]:
    """Folded readings of ``text``, including a consonant skeleton and Hebrew letter variants.

    A string with no Hebrew letter still contributes its skeleton (``shalom`` → ``shlm``), which
    is what makes a consonantal transliteration match an English spelling that kept the vowels.
    """
    if not any(character in _LETTER for character in text):
        found: list[str] = []
        _remember(found, text)
        return tuple(dict.fromkeys(found))
    variants: list[str] = [""]
    for character in text:
        if character not in _LETTER:
            variants = [item + character for item in variants][:_MAX_VARIANTS]
            continue
        options = [_LETTER[character]]
        alternate = _ALTERNATE.get(character)
        if alternate is not None:
            options.append(alternate)
        if character in _MATER:
            options.append("")
        grown: list[str] = []
        for item in variants:
            for option in options:
                grown.append(item + option)
                if len(grown) >= _MAX_VARIANTS:
                    break
            if len(grown) >= _MAX_VARIANTS:
                break
        variants = grown
    found = []
    for item in variants:
        _remember(found, item)
    return tuple(dict.fromkeys(found)) or (normalize_label(text),)


def _has_hebrew(text: str) -> bool:
    return any(character in _LETTER for character in text)


def forms_of(value: str) -> frozenset[str]:
    """Folded form, plus Hebrew transliterations. A Latin word does not lose its vowels here.

    Stripping vowels from every Latin word makes ``activ`` and ``active`` the same string
    (``ctv``). That collapse is only useful when one side is Hebrew and the other is the
    English spelling that kept the vowels, which :func:`equivalent` handles on its own.
    """
    folded = normalize_label(value)
    if not folded:
        return frozenset()
    if not _has_hebrew(folded):
        return frozenset({folded})
    return frozenset(_transliterations(folded))


def _skeletons(value: str) -> frozenset[str]:
    """Consonant skeletons of every reading, used only across the Hebrew/Latin boundary."""
    found: set[str] = set()
    for form in forms_of(value) or {normalize_label(value)}:
        skeleton = _skeleton(form)
        if skeleton:
            found.add(skeleton)
    return frozenset(found)


def equivalent(left: str, right: str) -> bool:
    """True when ``left`` and ``right`` are the same value after folding and transliteration.

    Two Latin spellings compare as folded text only, so a typo stays a typo. A Hebrew string
    and a Latin string also match when their consonants do (``שלום`` and ``shalom``).
    """
    if forms_of(left) & forms_of(right):
        return True
    left_hebrew = _has_hebrew(normalize_label(left))
    right_hebrew = _has_hebrew(normalize_label(right))
    if left_hebrew == right_hebrew:
        return False
    return bool(_skeletons(left) & _skeletons(right))


def suggest(
    query: str,
    observed: Sequence[tuple[str, int]],
    *,
    rows: int,
    limit: int = MAX_SUGGESTIONS,
    also: Sequence[str] = (),
) -> tuple[Suggestion, ...]:
    """The closest observed values to ``query``.

    ``observed`` is ``(stored value, count)``. ``also`` is extra spellings of the query (a
    canonical value, a synonym, a source code) that should count as what the agent meant.

    Order is distance, then count descending, then the value. A value whose distance is above
    the cutoff is omitted when something closer exists; when nothing is close, the most common
    values are returned so the agent still has a real value to retry with. Their distance is
    left as measured, and the caller can see it is not a near miss.
    """
    if limit < 1 or not observed:
        return ()
    labels = (query, *also)
    ranked: list[Suggestion] = []
    for value, count in observed:
        if any(equivalent(label, value) for label in labels):
            distance = 0
        else:
            distance = min(
                edit_distance(normalize_label(label), normalize_label(value)) for label in labels
            )
        ranked.append(Suggestion(value, distance, count, rows))
    close = [item for item in ranked if item.distance <= _CLOSE]
    pool = close or ranked
    pool.sort(key=lambda item: (item.distance, -item.count, item.value))
    return tuple(pool[:limit])
