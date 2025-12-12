from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.matching import evaluate
from eval.types import BenchmarkItem, PredictionItem


def _jsonable(obj: Any) -> Any:
    # Basic serializer for dataclasses and common types.
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")


def build_scorecard(run_id: str, version: str, result: Dict[str, Any]) -> Dict[str, Any]:
    m = result["metrics"]
    return {
        "run_id": run_id,
        "version": version,
        "prompt_version": result.get("run_meta", {}).get("prompt_version") if isinstance(result.get("run_meta"), dict) else None,
        "config_version": result.get("run_meta", {}).get("config_version") if isinstance(result.get("run_meta"), dict) else None,
        "recall": m["recall"],
        "precision": m["precision"],
        "f1": m["f1"],
        "expected_count": m["expected_count"],
        "predicted_count": m["predicted_count"],
        "tp_expected_any_count": m["tp_expected_any_count"],
        "fn_expected_count": m["fn_expected_count"],
        "fp_predicted_count": m["fp_predicted_count"],
    }


def build_details(result: Dict[str, Any]) -> Dict[str, Any]:
    expected_outcomes = result["expected_outcomes"]
    prediction_outcomes = result["prediction_outcomes"]
    return {
        "true_positives": [
            {
                "expected_id": e.expected_id,
                "score": e.score,
                "match": (e.match.__dict__ if e.match else None),
            }
            for e in expected_outcomes
            if e.score > 0.0
        ],
        "false_negatives": [
            {
                "expected_id": e.expected_id,
            }
            for e in expected_outcomes
            if e.score == 0.0
        ],
        "false_positives": [
            {
                "predicted_id": p.predicted_id,
                "predicted_raw_id": p.predicted_raw_id,
                "cycle": p.cycle,
                "evidence_urls": p.evidence_urls,
            }
            for p in prediction_outcomes
            if p.score == 0.0
        ],
        "prediction_duplicates_dropped": result.get("prediction_duplicates_dropped", []),
    }


def build_marginal_gains(
    run_id: str,
    version: str,
    benchmark: List[BenchmarkItem],
    predictions: List[PredictionItem],
    *,
    synonyms: dict[str, str] | None = None,
    series: Any = None,
) -> Dict[str, Any]:
    cycles = sorted({int(p.cycle or 1) for p in predictions} or {1})
    timeline: List[Dict[str, Any]] = []
    prev_recall: Optional[float] = None

    for c in cycles:
        filtered = [p for p in predictions if int(p.cycle or 1) <= c]
        res = evaluate(benchmark, filtered, synonyms=synonyms or {}, series=series)
        recall = float(res["metrics"]["recall"])
        precision = float(res["metrics"]["precision"])
        if prev_recall is None:
            gain = recall
        else:
            gain = recall - prev_recall
        prev_recall = recall
        timeline.append(
            {
                "cycle": c,
                "recall": recall,
                "precision": precision,
                "marginal_recall_gain": gain,
                "tp_expected_any_count": int(res["metrics"]["tp_expected_any_count"]),
                "fp_predicted_count": int(res["metrics"]["fp_predicted_count"]),
                "fn_expected_count": int(res["metrics"]["fn_expected_count"]),
                "predicted_count": int(res["metrics"]["predicted_count"]),
            }
        )

    return {"run_id": run_id, "version": version, "cycles": timeline}


def write_reports(
    *,
    out_dir: str | Path,
    run_id: str,
    version: str,
    benchmark: List[BenchmarkItem],
    predictions: List[PredictionItem],
    result: Dict[str, Any],
    synonyms: dict[str, str] | None = None,
    series: Any = None,
) -> Dict[str, Path]:
    out_base = Path(out_dir) / version / f"run_{run_id}"

    scorecard = build_scorecard(run_id, version, result)
    details = build_details(result)
    marginal = build_marginal_gains(run_id, version, benchmark, predictions, synonyms=synonyms, series=series)

    scorecard_path = out_base / "scorecard.json"
    details_path = out_base / "details.json"
    marginal_path = out_base / "marginal_gains.json"
    report_path = out_base / "report.json"

    write_json(scorecard_path, scorecard)
    write_json(details_path, details)
    write_json(marginal_path, marginal)
    write_json(
        report_path,
        {
            "scorecard": scorecard,
            "details": details,
            "marginal_gains": marginal,
        },
    )

    return {
        "scorecard": scorecard_path,
        "details": details_path,
        "marginal_gains": marginal_path,
        "report": report_path,
    }


