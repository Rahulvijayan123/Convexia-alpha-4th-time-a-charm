from __future__ import annotations

import json
import unittest
from pathlib import Path

from eval.benchmark import load_benchmark_csv
from eval.matching import evaluate, load_series_rules, load_synonyms
from eval.sources.local_json import load_predictions_json


class TestRegression(unittest.TestCase):
    def test_weighted_recall_does_not_drop_beyond_threshold(self) -> None:
        base = Path(__file__).resolve().parent
        fixtures = base / "fixtures"
        baseline_path = base / "baselines" / "previous.json"

        benchmark = load_benchmark_csv(fixtures / "benchmark.csv", id_column="identifier")
        synonyms = load_synonyms(fixtures / "synonyms.json")
        series = load_series_rules(fixtures / "series_rules.json")

        # Fixed suite of "queries" / stored-output fixtures
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

        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_recall = float(baseline["weighted_recall"])
        max_drop = float(baseline.get("max_drop", 0.0))

        self.assertGreaterEqual(
            recall + max_drop,
            baseline_recall,
            msg=f"Recall regressed: current={recall:.6f} baseline={baseline_recall:.6f} max_drop={max_drop:.6f}",
        )


if __name__ == "__main__":
    unittest.main()


