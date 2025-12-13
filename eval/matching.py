from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from eval.types import (
    BenchmarkItem,
    ExpectedOutcome,
    MatchDetail,
    PredictionItem,
    PredictionOutcome,
)
from eval.utils import match_keys, normalize_identifier


@dataclass(frozen=True)
class SeriesScores:
    # How much credit to assign for series-related matches.
    series_to_series: float = 1.0
    member_to_series: float = 1.0
    series_to_member: float = 0.5


@dataclass(frozen=True)
class SeriesIndex:
    series_ids: frozenset[str]
    member_to_series: Mapping[str, str]
    series_to_members: Mapping[str, Tuple[str, ...]]
    scores_by_series: Mapping[str, SeriesScores]

    def scores_for(self, series_id: str) -> SeriesScores:
        return self.scores_by_series.get(series_id, SeriesScores())


def load_synonyms(path: str | Path) -> Dict[str, str]:
    """
    Load alias/synonym rules from JSON.

    Supported formats:
    - {"aliases": {"ALIAS": "CANONICAL", ...}}
    - {"synonyms": [{"canonical": "X", "aliases": ["A","B"]}, ...]}

    Returns: dict[normalized_alias] -> normalized_canonical
    """
    p = Path(path)
    if not p.exists():
        return {}

    data = json.loads(p.read_text(encoding="utf-8"))

    aliases: Dict[str, str] = {}
    if isinstance(data, dict) and isinstance(data.get("aliases"), dict):
        for k, v in data["aliases"].items():
            nk = normalize_identifier(str(k))
            nv = normalize_identifier(str(v))
            if nk and nv:
                aliases[nk] = nv
        return aliases

    if isinstance(data, dict) and isinstance(data.get("synonyms"), list):
        for entry in data["synonyms"]:
            if not isinstance(entry, dict):
                continue
            canon = normalize_identifier(str(entry.get("canonical", "")))
            if not canon:
                continue
            for a in entry.get("aliases", []) or []:
                na = normalize_identifier(str(a))
                if na:
                    aliases[na] = canon
        return aliases

    raise ValueError(
        f"Unsupported synonyms JSON format in {p}. Expected keys: 'aliases' or 'synonyms'."
    )


def load_series_rules(path: str | Path) -> SeriesIndex:
    """
    Load series match rules with explicit scoring from JSON.

    Supported format:
    {
      "rules": [
        {
          "series_id": "PB-Series",
          "members": ["PB301", "PB316"],
          "scores": {
            "series_to_series": 1.0,
            "member_to_series": 1.0,
            "series_to_member": 0.5
          }
        }
      ]
    }
    """
    p = Path(path)
    if not p.exists():
        return SeriesIndex(
            series_ids=frozenset(),
            member_to_series={},
            series_to_members={},
            scores_by_series={},
        )

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        raise ValueError(f"Unsupported series rules JSON format in {p}. Expected top-level key: 'rules'.")

    series_ids: set[str] = set()
    member_to_series: Dict[str, str] = {}
    series_to_members: Dict[str, List[str]] = {}
    scores_by_series: Dict[str, SeriesScores] = {}

    for entry in data["rules"]:
        if not isinstance(entry, dict):
            continue
        series_id = normalize_identifier(str(entry.get("series_id", "")))
        if not series_id:
            continue
        members = entry.get("members", []) or []
        if not isinstance(members, list):
            raise ValueError(f"Series rule 'members' must be a list (series_id={series_id}).")

        norm_members: List[str] = []
        for m in members:
            nm = normalize_identifier(str(m))
            if not nm:
                continue
            norm_members.append(nm)
            # v1.3: preserve hyphens in the canonical member id, but also map a hyphenless form
            # so membership can be detected across formatting variants.
            for k in match_keys(nm):
                existing = member_to_series.get(k)
                if existing and existing != series_id:
                    raise ValueError(
                        f"Ambiguous series membership key '{k}': already mapped to series_id={existing}, "
                        f"cannot also map to series_id={series_id}."
                    )
                member_to_series[k] = series_id

        series_ids.add(series_id)
        series_to_members[series_id] = norm_members

        scores = entry.get("scores", {}) or {}
        if not isinstance(scores, dict):
            raise ValueError(f"Series rule 'scores' must be a dict (series_id={series_id}).")
        scores_by_series[series_id] = SeriesScores(
            series_to_series=float(scores.get("series_to_series", 1.0)),
            member_to_series=float(scores.get("member_to_series", 1.0)),
            series_to_member=float(scores.get("series_to_member", 0.5)),
        )

    return SeriesIndex(
        series_ids=frozenset(series_ids),
        member_to_series=member_to_series,
        series_to_members={k: tuple(v) for k, v in series_to_members.items()},
        scores_by_series=scores_by_series,
    )


def _best_match_is_better(
    new_score: float,
    new_type: str,
    new_cycle: int,
    old_score: float,
    old_type: str,
    old_cycle: int,
) -> bool:
    if new_score > old_score:
        return True
    if new_score < old_score:
        return False

    # Prefer "exact" > "synonym" > "series_*"
    rank = {
        "exact": 3,
        "exact_variant": 2,
        "synonym": 2,
        "series_to_series": 1,
        "member_to_series": 1,
        "series_to_member": 1,
    }
    if rank.get(new_type, 0) > rank.get(old_type, 0):
        return True
    if rank.get(new_type, 0) < rank.get(old_type, 0):
        return False

    # Prefer earlier cycle
    return new_cycle < old_cycle


def _build_expected_index(benchmark: Iterable[BenchmarkItem]) -> Tuple[Dict[str, BenchmarkItem], Dict[str, List[str]]]:
    expected_by_id: Dict[str, BenchmarkItem] = {}
    key_to_expected_ids: Dict[str, List[str]] = {}
    for item in benchmark:
        eid = normalize_identifier(item.canonical_id)
        if not eid:
            continue
        expected_by_id[eid] = item
        for k in match_keys(eid):
            key_to_expected_ids.setdefault(k, []).append(eid)
    return expected_by_id, key_to_expected_ids


def _apply_synonyms(norm_id: str, synonyms: Mapping[str, str]) -> str:
    return synonyms.get(norm_id, norm_id)


def evaluate(
    benchmark: List[BenchmarkItem],
    predictions: List[PredictionItem],
    *,
    synonyms: Mapping[str, str] | None = None,
    series: SeriesIndex | None = None,
) -> Dict[str, Any]:
    """
    Compute recall/precision and diff lists (TP/FP/FN) for a single run.

    IMPORTANT: This function is evaluation-only; it performs no model calls and does not generate queries.
    """
    synonyms = synonyms or {}
    series = series or SeriesIndex(frozenset(), {}, {}, {})

    expected_by_id, key_to_expected_ids = _build_expected_index(benchmark)
    expected_ids = list(expected_by_id.keys())
    expected_id_set = set(expected_ids)

    # Unify predictions by canonicalized identifier (post-synonym) to avoid double-counting aliases.
    pred_unique: Dict[str, PredictionItem] = {}
    pred_duplicates: List[PredictionItem] = []
    for p in predictions:
        raw_norm = normalize_identifier(p.raw_id)
        if not raw_norm:
            continue
        canon = _apply_synonyms(raw_norm, synonyms)
        existing = pred_unique.get(canon)
        if existing is None:
            pred_unique[canon] = PredictionItem(
                raw_id=p.raw_id,
                evidence_urls=list(p.evidence_urls or []),
                cycle=int(p.cycle or 1),
                raw=p.raw,
            )
        else:
            # Keep earliest cycle as the representative prediction for that canonical id.
            if int(p.cycle or 1) < int(existing.cycle or 1):
                pred_duplicates.append(existing)
                pred_unique[canon] = PredictionItem(
                    raw_id=p.raw_id,
                    evidence_urls=list(p.evidence_urls or []),
                    cycle=int(p.cycle or 1),
                    raw=p.raw,
                )
            else:
                pred_duplicates.append(p)

    # Track best match per expected id.
    best_expected_score: Dict[str, float] = {eid: 0.0 for eid in expected_ids}
    best_expected_match: Dict[str, Optional[MatchDetail]] = {eid: None for eid in expected_ids}

    # Track best match per prediction (for precision and FP list).
    prediction_outcomes: List[PredictionOutcome] = []

    for pred_canon_id, pred in pred_unique.items():
        pred_cycle = int(pred.cycle or 1)

        raw_norm = normalize_identifier(pred.raw_id)
        canon_norm = normalize_identifier(pred_canon_id)
        raw_keys = match_keys(raw_norm)
        canon_keys = match_keys(canon_norm)

        exact_matched_expected: set[str] = set()
        for k in (raw_keys | canon_keys):
            for eid in key_to_expected_ids.get(k, []):
                exact_matched_expected.add(eid)

        # Series-derived matches (may be partial score).
        series_candidates: List[Tuple[str, str, float]] = []  # (expected_id, match_type, score)
        pred_is_series = canon_norm in series.series_ids
        pred_series_id = canon_norm if pred_is_series else series.member_to_series.get(canon_norm)
        if not pred_is_series and not pred_series_id:
            # Try match-key variants (e.g., hyphenless) for member->series mapping.
            for k in match_keys(canon_norm):
                pred_series_id = series.member_to_series.get(k)
                if pred_series_id:
                    break

        if pred_is_series:
            scores = series.scores_for(canon_norm)
            # IMPORTANT (v1.3 contract): a benchmark row that represents a series is only a "hit"
            # when at least one explicit child member identifier is present in predictions.
            # Therefore we do NOT award series_to_series credit here.
            for member in series.series_to_members.get(canon_norm, ()):
                for mk in match_keys(member):
                    for eid in key_to_expected_ids.get(mk, []):
                        # Avoid accidentally matching series rows via series_to_member.
                        if eid in series.series_ids:
                            continue
                        series_candidates.append((eid, "series_to_member", scores.series_to_member))
        elif pred_series_id:
            scores = series.scores_for(pred_series_id)
            if pred_series_id in expected_id_set:
                series_candidates.append((pred_series_id, "member_to_series", scores.member_to_series))

        # Update expected best matches (independent per expected item).
        for eid in exact_matched_expected:
            # v1.3: do not count predicting the series identifier itself as a "hit" for a benchmark
            # series row. A series row is only hit via an explicit child member prediction.
            if eid in series.series_ids and canon_norm == eid:
                continue
            md = MatchDetail(
                expected_id=eid,
                predicted_id=canon_norm,
                predicted_raw_id=pred.raw_id,
                match_type=("synonym" if raw_norm != canon_norm else "exact"),
                score=1.0,
                cycle=pred_cycle,
                evidence_urls=list(pred.evidence_urls or []),
            )
            old = best_expected_match.get(eid)
            if old is None or _best_match_is_better(
                1.0,
                md.match_type,
                pred_cycle,
                best_expected_score.get(eid, 0.0),
                old.match_type,
                old.cycle,
            ):
                best_expected_score[eid] = 1.0
                best_expected_match[eid] = md

        for eid, mtype, score in series_candidates:
            # Only update if this expected wasn't already matched exactly, or if series score beats existing score.
            md = MatchDetail(
                expected_id=eid,
                predicted_id=canon_norm,
                predicted_raw_id=pred.raw_id,
                match_type=mtype,
                score=float(score),
                cycle=pred_cycle,
                evidence_urls=list(pred.evidence_urls or []),
            )
            old = best_expected_match.get(eid)
            old_score = best_expected_score.get(eid, 0.0)
            if old is None or _best_match_is_better(
                float(score),
                mtype,
                pred_cycle,
                float(old_score),
                old.match_type,
                old.cycle,
            ):
                best_expected_score[eid] = float(score)
                best_expected_match[eid] = md

        # Best match for this prediction (used for precision and FP).
        best_pred_match: Optional[MatchDetail] = None
        best_pred_score = 0.0

        # Prefer exact/synonym over series if score ties.
        for eid in sorted(exact_matched_expected):
            if eid in series.series_ids and canon_norm == eid:
                continue
            md = MatchDetail(
                expected_id=eid,
                predicted_id=canon_norm,
                predicted_raw_id=pred.raw_id,
                match_type=("synonym" if raw_norm != canon_norm else "exact"),
                score=1.0,
                cycle=pred_cycle,
                evidence_urls=list(pred.evidence_urls or []),
            )
            if best_pred_match is None or _best_match_is_better(
                md.score,
                md.match_type,
                pred_cycle,
                best_pred_score,
                best_pred_match.match_type if best_pred_match else "",
                best_pred_match.cycle if best_pred_match else 10**9,
            ):
                best_pred_match = md
                best_pred_score = 1.0

        for eid, mtype, score in sorted(series_candidates, key=lambda x: (-float(x[2]), x[0])):
            md = MatchDetail(
                expected_id=eid,
                predicted_id=canon_norm,
                predicted_raw_id=pred.raw_id,
                match_type=mtype,
                score=float(score),
                cycle=pred_cycle,
                evidence_urls=list(pred.evidence_urls or []),
            )
            if best_pred_match is None or _best_match_is_better(
                md.score,
                md.match_type,
                pred_cycle,
                best_pred_score,
                best_pred_match.match_type if best_pred_match else "",
                best_pred_match.cycle if best_pred_match else 10**9,
            ):
                best_pred_match = md
                best_pred_score = float(score)

        prediction_outcomes.append(
            PredictionOutcome(
                predicted_id=canon_norm,
                predicted_raw_id=pred.raw_id,
                score=float(best_pred_score),
                match=best_pred_match,
                cycle=pred_cycle,
                evidence_urls=list(pred.evidence_urls or []),
                raw=pred.raw,
            )
        )

    expected_outcomes: List[ExpectedOutcome] = []
    for eid in expected_ids:
        expected_outcomes.append(ExpectedOutcome(expected_id=eid, score=best_expected_score.get(eid, 0.0), match=best_expected_match.get(eid)))

    total_expected = len(expected_outcomes)
    total_pred = len(prediction_outcomes)
    recall = (sum(e.score for e in expected_outcomes) / total_expected) if total_expected else 0.0
    precision = (sum(p.score for p in prediction_outcomes) / total_pred) if total_pred else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    tp_expected_any = [e for e in expected_outcomes if e.score > 0.0]
    fn_expected = [e for e in expected_outcomes if e.score == 0.0]
    fp_predictions = [p for p in prediction_outcomes if p.score == 0.0]

    # v1.4 diagnostics: exact + canonical set matches + near-miss suggestions.
    v14 = v14_match_diagnostics(benchmark, predictions)

    return {
        "metrics": {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "expected_count": total_expected,
            "predicted_count": total_pred,
            "tp_expected_any_count": len(tp_expected_any),
            "fn_expected_count": len(fn_expected),
            "fp_predicted_count": len(fp_predictions),
            # v1.4 match reporting (set-based)
            "exact_tp": int(v14["exact"]["tp"]),
            "exact_fp": int(v14["exact"]["fp"]),
            "exact_fn": int(v14["exact"]["fn"]),
            "canonical_tp": int(v14["canonical"]["tp"]),
            "canonical_fp": int(v14["canonical"]["fp"]),
            "canonical_fn": int(v14["canonical"]["fn"]),
        },
        "expected_outcomes": expected_outcomes,
        "prediction_outcomes": prediction_outcomes,
        "v14_diagnostics": v14,
        "prediction_duplicates_dropped": [
            {"raw_id": d.raw_id, "cycle": int(getattr(d, "cycle", 1) or 1), "evidence_urls": list(getattr(d, "evidence_urls", []) or [])}
            for d in pred_duplicates
        ],
    }


def _strip_or_empty(v: Any) -> str:
    return str(v).strip() if v is not None else ""


def v14_match_diagnostics(benchmark: List[BenchmarkItem], predictions: List[PredictionItem]) -> Dict[str, Any]:
    """
    v1.4: contract-level matching diagnostics.
    - exact-match: raw string equality after strip()
    - canonical-match: equality after v1.4 canonicalization (eval.utils.normalize_identifier)
    - near-miss: for each unmatched benchmark canonical id, top-10 closest predicted canonical ids
    """
    bench_raw: list[str] = []
    bench_canon: list[str] = []
    for b in benchmark:
        r = _strip_or_empty(b.canonical_id)
        if not r:
            continue
        bench_raw.append(r)
        bench_canon.append(normalize_identifier(r))

    pred_raw: list[str] = []
    pred_canon: list[str] = []
    canon_to_raw: dict[str, str] = {}
    for p in predictions:
        r = _strip_or_empty(p.raw_id)
        if not r:
            continue
        c = normalize_identifier(r)
        pred_raw.append(r)
        pred_canon.append(c)
        if c and c not in canon_to_raw:
            canon_to_raw[c] = r

    bench_raw_set = set(bench_raw)
    pred_raw_set = set(pred_raw)
    exact_tp = len(bench_raw_set & pred_raw_set)
    exact_fp = len(pred_raw_set - bench_raw_set)
    exact_fn = len(bench_raw_set - pred_raw_set)

    bench_canon_set = {c for c in bench_canon if c}
    pred_canon_set = {c for c in pred_canon if c}
    canon_tp = len(bench_canon_set & pred_canon_set)
    canon_fp = len(pred_canon_set - bench_canon_set)
    canon_fn = len(bench_canon_set - pred_canon_set)

    # Near-miss suggestions (diagnostic only)
    unmatched_bench_canons = sorted(bench_canon_set - pred_canon_set)
    pred_canons_sorted = sorted(pred_canon_set)
    near_miss: list[dict[str, Any]] = []
    for bc in unmatched_bench_canons:
        scored: list[tuple[float, str]] = []
        for pc in pred_canons_sorted:
            if not pc:
                continue
            score = SequenceMatcher(None, bc, pc).ratio()
            scored.append((float(score), pc))
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:10]
        near_miss.append(
            {
                "benchmark_canonical": bc,
                "benchmark_raw": next((r for r in bench_raw if normalize_identifier(r) == bc), ""),
                "suggestions": [
                    {
                        "predicted_canonical": pc,
                        "predicted_raw": canon_to_raw.get(pc, ""),
                        "similarity": float(s),
                    }
                    for s, pc in top
                ],
            }
        )

    return {
        "exact": {"tp": exact_tp, "fp": exact_fp, "fn": exact_fn},
        "canonical": {"tp": canon_tp, "fp": canon_fp, "fn": canon_fn},
        "near_miss": near_miss,
    }


