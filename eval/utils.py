from __future__ import annotations

import re
from typing import Iterable, Set


_WS_RE = re.compile(r"\s+")
_DASHES_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")
_SURROUNDING_QUOTES_RE = re.compile(r'^[\'"\u201c\u201d\u2018\u2019](.*)[\'"\u201c\u201d\u2018\u2019]$')


def normalize_identifier(value: str) -> str:
    """
    Normalize identifiers for matching:
    - trim
    - unicode dash -> '-'
    - collapse whitespace
    - uppercase
    """
    v = (value or "").strip()
    if not v:
        return ""
    m = _SURROUNDING_QUOTES_RE.match(v)
    if m:
        v = m.group(1).strip()
    v = _DASHES_RE.sub("-", v)
    v = _WS_RE.sub(" ", v).strip()
    return v.upper()


def _is_code_like(v: str) -> bool:
    # Heuristic: mostly codes/tokens like "ZLC-491", "DS18", "MK-7965"
    # Avoid generating variants for long natural-language strings.
    if len(v) > 64:
        return False
    return bool(re.fullmatch(r"[A-Z0-9][A-Z0-9 \-_/().;:+]*", v))


def match_keys(value: str) -> Set[str]:
    """
    Match keys used for 'exact identifier match' (post-normalization).
    We include a conservative "no separators" variant for code-like strings.
    """
    base = normalize_identifier(value)
    if not base:
        return set()

    keys = {base}

    # Variant: strip parenthetical suffix (common in CSV values like "X (Y)")
    if "(" in base and ")" in base:
        no_parens = re.sub(r"\s*\([^)]*\)\s*", " ", base)
        no_parens = _WS_RE.sub(" ", no_parens).strip()
        if no_parens:
            keys.add(no_parens)

    if _is_code_like(base):
        no_sep = re.sub(r"[\s\-_()/.;:+]", "", base)
        if no_sep and no_sep != base:
            keys.add(no_sep)

    return keys


def first_non_empty(values: Iterable[str | None]) -> str | None:
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


