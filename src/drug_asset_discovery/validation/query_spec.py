from __future__ import annotations

import re

from drug_asset_discovery.models.domain import QuerySpec
from drug_asset_discovery.utils.identifiers import classify_identifier_type


_TARGET_CANDIDATE_RE = re.compile(r"\b[A-Za-z]{2,12}\d{1,2}(?:\s*[/\-]\s*\d{1,2})?\b")

_MODALITY_KEYWORDS = (
    "inhibitor",
    "inhibition",
    "degrader",
    "degradation",
    "protac",
    "molecular glue",
    "glue degrader",
    "antibody",
    "adc",
    "siRNA",
    "RNAi",
    "gene therapy",
    "cell therapy",
    "car-t",
    "car t",
    "vaccine",
)


def derive_query_spec(user_query: str) -> QuerySpec:
    """
    Best-effort, benchmark-blind extraction of a query intent spec from free-form user text.

    This is intentionally conservative; it does NOT call any LLMs.
    """
    uq = " ".join((user_query or "").strip().split())

    # Target terms: pull gene/target-like tokens (not drug codes) from the query.
    targets: list[str] = []
    seen_t: set[str] = set()
    for m in _TARGET_CANDIDATE_RE.finditer(uq):
        tok = m.group(0).strip()
        if not tok:
            continue
        if classify_identifier_type(tok) != "target_gene":
            continue
        key = tok.upper()
        if key in seen_t:
            continue
        seen_t.add(key)
        targets.append(tok)

    # Modality terms: simple keyword capture (preserve original canonical phrase where possible).
    uq_low = uq.lower()
    modalities: list[str] = []
    seen_m: set[str] = set()
    for kw in _MODALITY_KEYWORDS:
        if kw.lower() in uq_low:
            k = kw.lower()
            if k in seen_m:
                continue
            seen_m.add(k)
            modalities.append(kw)

    # Indication terms: only explicit patterns (avoid overfitting by guessing disease names).
    indications: list[str] = []
    m = re.search(r"\b(indication|disease)\s*:\s*([^;,\n]+)", uq, flags=re.IGNORECASE)
    if m:
        ind = m.group(2).strip()
        if ind:
            indications.append(ind)

    return QuerySpec(target_terms=targets, modality_terms=modalities, indication_terms=indications)

