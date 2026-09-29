"""Conservative, field-independent text normalization for retrieval."""

from __future__ import annotations

import unicodedata


def normalize_text(value: str | None) -> str:
    """NFKC, case-fold, separate punctuation, and collapse whitespace.

    Letters, numbers, and combining marks are preserved in their original
    scripts. No transliteration, suffix removal, or address parsing is used.
    """
    if not value:
        return ""
    folded = unicodedata.normalize("NFKC", value).casefold()
    chars = []
    for char in folded:
        category = unicodedata.category(char)
        chars.append(" " if category[0] in {"P", "S", "Z", "C"} else char)
    return " ".join("".join(chars).split())


def has_non_ascii(value: str | None) -> bool:
    return bool(value) and any(ord(char) > 127 for char in value)
