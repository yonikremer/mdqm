"""Label normalisation: Hebrew and English are both first-class in this layer.

The same business value arrives written differently — ``פעיל``, ``פָּעִיל`` (with niqqud),
``פעיל״`` after a geresh/gershayim variant, ``Active`` with a capital, a full-width or ligature
compatibility form. Normalising both the index side and the query side is what makes a match.

Rules: NFKC first (so presentation forms expand into letter + mark), then niqqud and cantillation
marks are stripped, geresh/gershayim variants fold to ASCII ``'`` and ``"``, zero-width characters
are removed, case is folded and runs of whitespace collapse to one space.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize_label"]

# Hebrew points (niqqud) and cantillation marks, U+0591..U+05C7, minus the two punctuation marks
# U+05BE (maqaf) and U+05C0 (paseq) which are real characters rather than vowels.
_NIQQUD = re.compile(r"[\u0591-\u05bd\u05bf\u05c1\u05c2\u05c4-\u05c7]")

# Geresh and gershayim in their Hebrew and typographic spellings, plus the ASCII apostrophe.
_GERESH = re.compile("[\u05f3\u2018\u2019\u02bc`']")
_GERSHAYIM = re.compile('[\u05f4\u201c\u201d\u2033"]')

# Zero-width space/joiners and the soft hyphen: invisible, so they must not stop a match.
_INVISIBLE = re.compile(r"[\u00ad\u200b-\u200f\u2060\ufeff]")

_SPACES = re.compile(r"\s+")


def normalize_label(value: str) -> str:
    """Return the lookup form of a label (canonical value, synonym, Hebrew or English)."""
    text = unicodedata.normalize("NFKC", value)
    text = _NIQQUD.sub("", text)
    text = _GERESH.sub("'", text)
    text = _GERSHAYIM.sub('"', text)
    text = _INVISIBLE.sub("", text)
    text = text.casefold()
    return _SPACES.sub(" ", text).strip()
