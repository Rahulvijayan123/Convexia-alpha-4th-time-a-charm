from __future__ import annotations

import unicodedata
from typing import Any

from drug_asset_discovery.models.domain import DraftAsset, QuerySpec
from drug_asset_discovery.utils.identifiers import canonicalize_identifier


def _nfkc_casefold(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").casefold()


def _identifier_in_snippet(*, identifier_raw: str, identifier_canonical: str, snippet: str) -> bool:
    pid = (identifier_raw or "").strip()
    snip = (snippet or "").strip()
    if not pid or not snip:
        return False
    if pid in snip:
        return True
    pid_c = (identifier_canonical or "").strip() or canonicalize_identifier(pid)
    if not pid_c:
        return False
    return pid_c in canonicalize_identifier(snip)


def _term_in_text(*, term: str, text: str) -> bool:
    t = (term or "").strip()
    if not t:
        return False
    key = canonicalize_identifier(t)
    if key:
        return key in canonicalize_identifier(text or "")
    # Fallback for non A-Z0-9 terms
    return _nfkc_casefold(t) in _nfkc_casefold(text or "")


def _coerce_evidence_blocks(d: DraftAsset) -> list[dict[str, str]]:
    """
    Normalize DraftAsset evidence into a list of {url, snippet} blocks.
    Dedup by (url,snippet) pair.
    """
    blocks: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(url: str, snip: str) -> None:
        u = (url or "").strip()
        s = (snip or "").strip()
        if not u or not s:
            return
        k = f"{u}\n{s}"
        if k in seen:
            return
        seen.add(k)
        blocks.append({"url": u, "snippet": s})

    _add(d.evidence_url, d.evidence_snippet)

    for c in list(d.citations or []):
        if isinstance(c, dict):
            _add(str(c.get("url") or ""), str(c.get("snippet") or ""))
        else:
            # pydantic models
            url = getattr(c, "url", "")
            snip = getattr(c, "snippet", "")
            if isinstance(url, str) and isinstance(snip, str):
                _add(url, snip)
    return blocks


def _best_evidence_for_terms(
    *,
    blocks: list[dict[str, str]],
    anchor_url: str,
    identifier_raw: str,
    identifier_canonical: str,
    terms: list[str],
) -> tuple[float, dict[str, Any] | None]:
    """
    Evidence-based scoring policy (conservative):
    - 1.0 if a block contains BOTH identifier and a term.
    - 0.6 if a block from the anchor_url contains the term (same source section proxy).
    - 0.2 if the term appears somewhere else without identifier anchoring.
    - 0.0 otherwise.
    """
    if not terms:
        return 1.0, None
    anchor_url_s = (anchor_url or "").strip()

    best_score = 0.0
    best: dict[str, Any] | None = None

    for term in [t for t in terms if isinstance(t, str) and t.strip()]:
        for b in blocks:
            url = str(b.get("url") or "").strip()
            snip = str(b.get("snippet") or "").strip()
            if not url or not snip:
                continue

            term_present = _term_in_text(term=term, text=snip)
            if not term_present:
                continue

            anchored = _identifier_in_snippet(
                identifier_raw=identifier_raw, identifier_canonical=identifier_canonical, snippet=snip
            )
            if anchored:
                ev = {"term": term, "url": url, "snippet": snip[:600]}
                return 1.0, ev

            if anchor_url_s and url == anchor_url_s and best_score < 0.6:
                best_score = 0.6
                best = {"term": term, "url": url, "snippet": snip[:600]}
                continue

            if best_score < 0.2:
                best_score = 0.2
                best = {"term": term, "url": url, "snippet": snip[:600]}

    return float(best_score), best


def score_match_to_query(draft_asset: DraftAsset, query_spec: QuerySpec) -> dict[str, Any]:
    """
    v1.5 late filtering:
    Compute evidence-based match scores to the query spec.

    Returns a JSON-serializable dict including:
    - target_match_score
    - modality_match_score
    - indication_match_score
    - overall_match_score
    - evidence: {target, modality, indication} (optional per-dimension evidence block)
    """
    d = draft_asset
    qs = query_spec

    blocks = _coerce_evidence_blocks(d)

    # Default policy: if a dimension isn't specified in the query, it does not block promotion.
    target_required = bool(qs.target_terms)
    modality_required = bool(qs.modality_terms)
    indication_required = bool(qs.indication_terms)

    t_score, t_ev = _best_evidence_for_terms(
        blocks=blocks,
        anchor_url=d.evidence_url,
        identifier_raw=d.identifier_raw,
        identifier_canonical=d.identifier_canonical,
        terms=list(qs.target_terms or []),
    )
    m_score, m_ev = _best_evidence_for_terms(
        blocks=blocks,
        anchor_url=d.evidence_url,
        identifier_raw=d.identifier_raw,
        identifier_canonical=d.identifier_canonical,
        terms=list(qs.modality_terms or []),
    )
    i_score, i_ev = _best_evidence_for_terms(
        blocks=blocks,
        anchor_url=d.evidence_url,
        identifier_raw=d.identifier_raw,
        identifier_canonical=d.identifier_canonical,
        terms=list(qs.indication_terms or []),
    )

    required_scores: list[float] = []
    if target_required:
        required_scores.append(float(t_score))
    if modality_required:
        required_scores.append(float(m_score))
    if indication_required:
        required_scores.append(float(i_score))

    overall = float(sum(required_scores) / len(required_scores)) if required_scores else 0.0

    return {
        "target_match_score": float(t_score),
        "modality_match_score": float(m_score),
        "indication_match_score": float(i_score),
        "overall_match_score": float(overall),
        "required": {
            "target": target_required,
            "modality": modality_required,
            "indication": indication_required,
        },
        "evidence": {
            "target": t_ev,
            "modality": m_ev,
            "indication": i_ev,
        },
    }

