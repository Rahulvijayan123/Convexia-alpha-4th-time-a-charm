from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, Set


_SURROUNDING_QUOTES_RE = re.compile(r'^[\'"\u201c\u201d\u2018\u2019](.*)[\'"\u201c\u201d\u2018\u2019]$')


# v1.4: single canonicalization function used everywhere (runtime + eval).
#
# We import from the runtime package; for convenience when running eval without installation,
# we also add the repo's ./src to sys.path on ImportError.
try:
    from drug_asset_discovery.utils.identifiers import canonicalize_identifier  # type: ignore
except Exception:  # pragma: no cover - best-effort local dev fallback
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from drug_asset_discovery.utils.identifiers import canonicalize_identifier  # type: ignore  # noqa: E402


def normalize_identifier(value: str) -> str:
    """
    v1.4: identifiers are matched canonically using the shared runtime canonicalization:
    - uppercase
    - remove whitespace and punctuation separators
    - keep only A-Z and 0-9
    """
    v = (value or "").strip()
    if not v:
        return ""
    m = _SURROUNDING_QUOTES_RE.match(v)
    if m:
        v = m.group(1).strip()
    return canonicalize_identifier(v)


def match_keys(value: str) -> Set[str]:
    """
    Match keys used for 'exact identifier match' (post-normalization).
    Explicit contract:
    - compare the canonicalized form (A-Z0-9 only)
    - also compare a variant with parenthetical content removed (common in CSV values like "X (Y)")
    """
    base = normalize_identifier(value)
    if not base:
        return set()

    keys = {base}

    raw = (value or "").strip()
    if raw:
        m = _SURROUNDING_QUOTES_RE.match(raw)
        if m:
            raw = m.group(1).strip()
        if "(" in raw and ")" in raw:
            no_parens_raw = re.sub(r"\([^)]*\)", "", raw).strip()
            nk = normalize_identifier(no_parens_raw)
            if nk:
                keys.add(nk)

    return keys


def first_non_empty(values: Iterable[str | None]) -> str | None:
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


