from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Optional

from eval.benchmark import load_benchmark_csv
from eval.matching import evaluate, load_series_rules, load_synonyms
from eval.reporting import write_reports
from eval.sources.local_json import load_predictions_json
from eval.sources.supabase_rest import SupabaseConfig, SupabaseRestSource
from eval.utils import normalize_identifier


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _default_mapping_path(rel: str) -> Optional[Path]:
    p = (Path(__file__).resolve().parent / rel).resolve()
    return p if p.exists() else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval",
        description="Offline evaluation for web discovery outputs vs a local benchmark CSV (no LLM calls).",
    )
    p.add_argument("--run_id", required=True, help="Run id to evaluate (used to fetch stored outputs from Supabase).")
    p.add_argument(
        "--benchmark_csv",
        "--benchmark",
        required=True,
        dest="benchmark_csv",
        help="Path to benchmark CSV (local file). Offline only.",
    )
    p.add_argument("--version", required=True, help="Evaluation version label (e.g., v1.1, v1.2).")

    p.add_argument("--benchmark_id_column", default=None, help="CSV column containing the benchmark identifier.")
    p.add_argument(
        "--min_benchmark_rows",
        type=int,
        default=int(os.getenv("EVAL_MIN_BENCHMARK_ROWS") or "10"),
        help="Fail fast if loaded benchmark row count is below this threshold. Default: 10 (or env EVAL_MIN_BENCHMARK_ROWS).",
    )
    p.add_argument(
        "--explain",
        action="store_true",
        help="Explain-mode output: print match status per benchmark row and the predicted identifier that matched it.",
    )

    p.add_argument("--synonyms", default=None, help="Path to synonyms JSON mapping file.")
    p.add_argument("--series_rules", default=None, help="Path to series rules JSON mapping file.")

    p.add_argument("--out_dir", default=str(Path("eval") / "reports"), help="Output directory for JSON reports.")

    # For deterministic offline runs without Supabase (e.g., debugging or tests)
    p.add_argument("--predictions_json", default=None, help="Path to a JSON file containing stored outputs/predictions.")

    # Supabase config overrides (defaults read from env)
    p.add_argument("--table", default=None, help="Supabase table name holding stored outputs.")
    p.add_argument("--schema", default=None, help="Supabase schema (Accept-Profile header). Default: public.")
    p.add_argument("--run_id_column", default=None, help="Column name used to filter by run_id. Default: run_id.")

    # Output extraction hints (recommended for stable schemas)
    p.add_argument("--prediction_id_key", default=None, help="Explicit JSON key for predicted identifier.")
    p.add_argument("--prediction_evidence_key", default=None, help="Explicit JSON key for evidence URL.")
    p.add_argument("--prediction_cycle_key", default=None, help="Explicit JSON key for cycle/iteration number.")

    p.add_argument("--env_file", default=None, help="Optional path to an .env file for Supabase credentials.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Load env file(s) if provided, otherwise best-effort common defaults.
    if args.env_file:
        _load_env_file(Path(args.env_file))
    else:
        _load_env_file(Path(".env"))
        _load_env_file(Path("eval") / ".env")

    benchmark_path = Path(args.benchmark_csv).expanduser().resolve()
    if not benchmark_path.exists():
        raise SystemExit(f"Benchmark CSV not found: {benchmark_path}")
    benchmark_sha256 = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()

    benchmark = load_benchmark_csv(benchmark_path, id_column=args.benchmark_id_column)
    benchmark_rows = len(benchmark)

    print(f"benchmark_csv={benchmark_path}")
    print(f"benchmark_sha256={benchmark_sha256}")
    print(f"benchmark_rows={benchmark_rows}")
    if benchmark_rows < int(args.min_benchmark_rows):
        raise SystemExit(
            f"Benchmark CSV row count too small: loaded_rows={benchmark_rows} "
            f"min_required={int(args.min_benchmark_rows)} path={benchmark_path}"
        )

    synonyms_path = Path(args.synonyms) if args.synonyms else _default_mapping_path("mappings/synonyms.json")
    series_path = Path(args.series_rules) if args.series_rules else _default_mapping_path("mappings/series_rules.json")

    synonyms = load_synonyms(synonyms_path) if synonyms_path else {}
    series = load_series_rules(series_path) if series_path else None

    if args.predictions_json:
        predictions = load_predictions_json(
            args.predictions_json,
            id_key=args.prediction_id_key,
            evidence_key=args.prediction_evidence_key,
            cycle_key=args.prediction_cycle_key,
        )
        run_meta = None
    else:
        url = os.getenv("EVAL_SUPABASE_URL") or os.getenv("SUPABASE_URL") or ""
        key = (
            os.getenv("EVAL_SUPABASE_SERVICE_KEY")
            or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_SERVICE_KEY")
            or os.getenv("SUPABASE_KEY")
            or ""
        )
        table = args.table or os.getenv("EVAL_SUPABASE_TABLE") or "final_assets"
        schema = args.schema or os.getenv("EVAL_SUPABASE_SCHEMA") or "public"
        run_id_column = args.run_id_column or os.getenv("EVAL_SUPABASE_RUN_ID_COLUMN") or "run_id"

        if not url or not key:
            raise SystemExit(
                "Missing Supabase credentials. Set EVAL_SUPABASE_URL and EVAL_SUPABASE_SERVICE_KEY "
                "(or pass --predictions_json for a local run)."
            )

        src = SupabaseRestSource(
            SupabaseConfig(
                url=url,
                service_key=key,
                table=table,
                schema=schema,
                run_id_column=run_id_column,
                prediction_id_key=args.prediction_id_key or os.getenv("EVAL_PREDICTION_ID_KEY"),
                prediction_evidence_key=args.prediction_evidence_key or os.getenv("EVAL_PREDICTION_EVIDENCE_KEY"),
                prediction_cycle_key=args.prediction_cycle_key or os.getenv("EVAL_PREDICTION_CYCLE_KEY"),
            )
        )
        predictions = src.fetch_predictions_for_run(args.run_id)
        run_meta = src.fetch_run_metadata(args.run_id)

    result = evaluate(benchmark, predictions, synonyms=synonyms, series=series)
    result["run_meta"] = run_meta
    paths = write_reports(
        out_dir=args.out_dir,
        run_id=args.run_id,
        version=args.version,
        benchmark=benchmark,
        predictions=predictions,
        result=result,
        synonyms=dict(synonyms),
        series=series,
    )

    m = result["metrics"]
    print(f"run_id={args.run_id} version={args.version}")
    print(f"recall={m['recall']:.4f} precision={m['precision']:.4f} f1={m['f1']:.4f}")
    print(f"expected={m['expected_count']} predicted={m['predicted_count']}")
    print(f"tp_expected_any={m['tp_expected_any_count']} fn_expected={m['fn_expected_count']} fp_predicted={m['fp_predicted_count']}")
    print(f"exact_match tp={m.get('exact_tp', 0)} fp={m.get('exact_fp', 0)} fn={m.get('exact_fn', 0)}")
    print(
        f"canonical_match tp={m.get('canonical_tp', 0)} fp={m.get('canonical_fp', 0)} fn={m.get('canonical_fn', 0)}"
    )
    if args.explain:
        # Build a mapping back to original CSV row numbers for nicer explain output.
        by_norm = {normalize_identifier(b.canonical_id): b for b in benchmark}
        print("--- explain (per benchmark row) ---")
        for e in result["expected_outcomes"]:
            item = by_norm.get(e.expected_id)
            row_num = getattr(item, "row_number", None) if item else None
            expected_raw = getattr(item, "canonical_id", e.expected_id) if item else e.expected_id
            matched = e.score > 0.0
            pred = e.match.predicted_raw_id if e.match else ""
            print(
                f"row={row_num if row_num is not None else '?'} "
                f"expected={expected_raw} matched={str(matched).lower()} predicted={pred}"
            )
    print(f"report={paths['report']}")
    return 0


