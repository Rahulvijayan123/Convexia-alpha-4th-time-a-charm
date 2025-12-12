from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from eval.types import PredictionItem
from eval.utils import first_non_empty


_DEFAULT_ID_KEYS = (
    # Common generic keys
    "identifier",
    "id",
    "asset_id",
    "asset_code",
    "code",
    "name",
    # This repo's schema (public.final_assets)
    "drug_name_code",
    # This repo's schema (public.candidates)
    "raw_identifier",
    "normalized_identifier",
)
_DEFAULT_EVIDENCE_KEYS = (
    "evidence_urls",
    "evidence_url",
    "sources",  # final_assets.sources (list[str])
    "evidence",
    "url",
    "source_url",
    "link",
)
_DEFAULT_CYCLE_KEYS = ("cycle", "iteration", "step", "round")


def _extract_predictions_from_dict(
    d: Dict[str, Any],
    *,
    id_key: Optional[str],
    evidence_key: Optional[str],
    cycle_key: Optional[str],
    inherited_cycle: Optional[int],
) -> List[PredictionItem]:
    candidates: List[PredictionItem] = []

    resolved_id_key = id_key or next((k for k in _DEFAULT_ID_KEYS if k in d), None)
    if resolved_id_key and resolved_id_key in d and isinstance(d.get(resolved_id_key), (str, int, float)):
        raw_id = str(d.get(resolved_id_key))
        ev: Any = None
        if evidence_key and evidence_key in d:
            ev = d.get(evidence_key)
        else:
            ev = first_non_empty(d.get(k) for k in _DEFAULT_EVIDENCE_KEYS if k in d)

        evidence_urls: list[str] = []
        if isinstance(ev, str):
            if ev.strip():
                evidence_urls = [ev.strip()]
        elif isinstance(ev, list):
            for item in ev:
                if isinstance(item, str) and item.strip():
                    evidence_urls.append(item.strip())
                elif isinstance(item, dict):
                    url = item.get("url") or item.get("source_url")
                    if isinstance(url, str) and url.strip():
                        evidence_urls.append(url.strip())
        elif isinstance(ev, dict):
            url = ev.get("url") or ev.get("source_url")
            if isinstance(url, str) and url.strip():
                evidence_urls = [url.strip()]
        cyc = None
        if cycle_key and cycle_key in d:
            cyc = d.get(cycle_key)
        else:
            cyc = first_non_empty(d.get(k) for k in _DEFAULT_CYCLE_KEYS if k in d)
        cycle = int(cyc) if cyc is not None and str(cyc).strip().isdigit() else int(inherited_cycle or 1)
        candidates.append(
            PredictionItem(
                raw_id=raw_id,
                evidence_urls=evidence_urls,
                cycle=cycle,
                raw=d,
            )
        )
        return candidates

    # Otherwise recurse into children.
    next_inherited = inherited_cycle
    if cycle_key and cycle_key in d:
        v = d.get(cycle_key)
        if v is not None and str(v).strip().isdigit():
            next_inherited = int(v)
    else:
        for k in _DEFAULT_CYCLE_KEYS:
            if k in d and d.get(k) is not None and str(d.get(k)).strip().isdigit():
                next_inherited = int(d.get(k))
                break

    for v in d.values():
        candidates.extend(
            extract_predictions(
                v,
                id_key=id_key,
                evidence_key=evidence_key,
                cycle_key=cycle_key,
                inherited_cycle=next_inherited,
            )
        )
    return candidates


def extract_predictions(
    obj: Any,
    *,
    id_key: Optional[str] = None,
    evidence_key: Optional[str] = None,
    cycle_key: Optional[str] = None,
    inherited_cycle: Optional[int] = None,
) -> List[PredictionItem]:
    """
    Best-effort extraction of prediction items from arbitrary JSON-like objects.
    Works with:
    - row-per-item tables (list[dict] each containing identifier)
    - blob outputs (dict containing nested lists)
    """
    if obj is None:
        return []

    if isinstance(obj, list):
        out: List[PredictionItem] = []
        for el in obj:
            out.extend(
                extract_predictions(
                    el,
                    id_key=id_key,
                    evidence_key=evidence_key,
                    cycle_key=cycle_key,
                    inherited_cycle=inherited_cycle,
                )
            )
        return out

    if isinstance(obj, dict):
        return _extract_predictions_from_dict(
            obj,
            id_key=id_key,
            evidence_key=evidence_key,
            cycle_key=cycle_key,
            inherited_cycle=inherited_cycle,
        )

    return []


def extract_predictions_from_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    id_key: Optional[str] = None,
    evidence_key: Optional[str] = None,
    cycle_key: Optional[str] = None,
) -> List[PredictionItem]:
    # If table is row-per-item, we'll find direct matches quickly.
    preds = extract_predictions(list(rows), id_key=id_key, evidence_key=evidence_key, cycle_key=cycle_key)
    return preds


