from __future__ import annotations

import json
from pathlib import Path

from eval.benchmark import load_benchmark_csv
from eval.matching import evaluate, load_series_rules, load_synonyms
from eval.sources.local_json import load_predictions_json


def main() -> int:
    base = Path(__file__).resolve().parent
    fixtures = base / "fixtures"
    baseline_path = base / "baselines" / "previous.json"

    benchmark = load_benchmark_csv(fixtures / "benchmark.csv", id_column="identifier")
    synonyms = load_synonyms(fixtures / "synonyms.json")
    series = load_series_rules(fixtures / "series_rules.json")

    run_files = [
        fixtures / "predictions.json",
        fixtures / "predictions_run2.json",
    ]
    recalls = []
    for f in run_files:
        predictions = load_predictions_json(f, id_key="identifier")
        result = evaluate(benchmark, predictions, synonyms=synonyms, series=series)
        recalls.append(float(result["metrics"]["recall"]))
    recall = sum(recalls) / len(recalls)

    payload = {
        "baseline_tag": "UNSET",
        "weighted_recall": recall,
        "max_drop": json.loads(baseline_path.read_text(encoding="utf-8")).get("max_drop", 0.01)
        if baseline_path.exists()
        else 0.01,
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote baseline: {baseline_path} (weighted_recall={recall:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


