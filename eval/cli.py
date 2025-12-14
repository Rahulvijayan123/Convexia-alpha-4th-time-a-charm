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
from eval.utils import match_keys, normalize_identifier


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
        "--mode",
        default="recall_final",
        choices=["recall_draft", "recall_final"],
        help="v1.5 eval mode: recall vs draft_assets or final_assets.",
    )
    p.add_argument(
        "--benchmark_csv",
        "--benchmark",
        required=False,
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


def _coerce_str(v: object) -> str | None:
    return v.strip() if isinstance(v, str) and v.strip() else None


def _coerce_str_list(v: object) -> list[str]:
    if not v:
        return []
    if isinstance(v, list):
        out: list[str] = []
        for x in v:
            s = _coerce_str(x)
            if s:
                out.append(s)
        return out
    s = _coerce_str(v)
    return [s] if s else []


def _dedup_keep_order(urls: list[str], *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        s = _coerce_str(u)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def _build_stage_url_maps(
    *,
    url: str,
    key: str,
    schema: str,
    run_id_column: str,
    run_id: str,
) -> dict[str, dict[str, list[str]]]:
    """
    Build per-stage maps:
      stage -> match_key (A-Z0-9) -> compact list of supporting URLs

    Matching is canonical-only (punctuation/case robust via match_keys) and does NOT require
    benchmark-specific alias lists.
    """

    def _add(stage_map: dict[str, set[str]], ident: str, urls: list[str]) -> None:
        for k in match_keys(ident):
            if not k:
                continue
            stage_map.setdefault(k, set()).update(urls)

    # ---- mentions
    mentions_src = SupabaseRestSource(
        SupabaseConfig(url=url, service_key=key, table="mentions", schema=schema, run_id_column=run_id_column)
    )
    mention_rows = mentions_src.fetch_rows_for_run_paginated(
        run_id,
        select="id,raw_text,normalized_text,canonical_text,source_url",
    )
    mentions_map: dict[str, set[str]] = {}
    for r in mention_rows:
        if not isinstance(r, dict):
            continue
        src_url = _coerce_str(r.get("source_url"))
        urls = [src_url] if src_url else []
        for ident in (r.get("raw_text"), r.get("canonical_text"), r.get("normalized_text")):
            s = _coerce_str(ident)
            if s:
                _add(mentions_map, s, urls)

    # ---- candidates (join to mentions for URL provenance)
    candidates_src = SupabaseRestSource(
        SupabaseConfig(url=url, service_key=key, table="candidates", schema=schema, run_id_column=run_id_column)
    )
    cand_rows = candidates_src.fetch_rows_for_run_paginated(
        run_id,
        select="id,raw_identifier,canonical_identifier,source_mention_id",
    )
    mention_ids: list[str] = []
    for r in cand_rows:
        if not isinstance(r, dict):
            continue
        mid = _coerce_str(r.get("source_mention_id"))
        if mid:
            mention_ids.append(mid)

    mention_id_to_url: dict[str, str] = {}
    if mention_ids:
        # NOTE: SupabaseRestSource exposes a private helper for batch ID fetches; ok for eval-only.
        mention_rows_by_id = mentions_src._fetch_rows_by_ids(  # type: ignore[attr-defined]
            table="mentions",
            ids=sorted(set(mention_ids)),
            select="id,source_url",
        )
        for r in mention_rows_by_id:
            if not isinstance(r, dict):
                continue
            mid = _coerce_str(r.get("id"))
            surl = _coerce_str(r.get("source_url"))
            if mid and surl:
                mention_id_to_url[mid] = surl

    candidates_map: dict[str, set[str]] = {}
    for r in cand_rows:
        if not isinstance(r, dict):
            continue
        urls = []
        mid = _coerce_str(r.get("source_mention_id"))
        if mid and mid in mention_id_to_url:
            urls = [mention_id_to_url[mid]]
        raw = _coerce_str(r.get("raw_identifier"))
        canon = _coerce_str(r.get("canonical_identifier"))
        if raw:
            _add(candidates_map, raw, urls)
        if canon:
            _add(candidates_map, canon, urls)

    # ---- found_assets (draft_assets)
    draft_src = SupabaseRestSource(
        SupabaseConfig(url=url, service_key=key, table="draft_assets", schema=schema, run_id_column=run_id_column)
    )
    draft_rows = draft_src.fetch_rows_for_run_paginated(
        run_id,
        select="identifier_raw,identifier_canonical,identifier_aliases_raw,evidence_url,citations",
    )
    found_map: dict[str, set[str]] = {}
    for r in draft_rows:
        if not isinstance(r, dict):
            continue
        urls = []
        urls.extend(_coerce_str_list(r.get("evidence_url")))
        for c in r.get("citations") or []:
            if isinstance(c, dict):
                urls.extend(_coerce_str_list(c.get("url") or c.get("source_url")))
        urls = _dedup_keep_order(urls, limit=12)
        raw = _coerce_str(r.get("identifier_raw"))
        canon = _coerce_str(r.get("identifier_canonical"))
        aliases = r.get("identifier_aliases_raw")
        if raw:
            _add(found_map, raw, urls)
        if canon:
            _add(found_map, canon, urls)
        if isinstance(aliases, list):
            for a in aliases:
                s = _coerce_str(a)
                if s:
                    _add(found_map, s, urls)

    # ---- validated_assets (final_assets)
    final_src = SupabaseRestSource(
        SupabaseConfig(url=url, service_key=key, table="final_assets", schema=schema, run_id_column=run_id_column)
    )
    final_rows = final_src.fetch_rows_for_run_paginated(
        run_id,
        select="drug_name_code,primary_identifier_raw,primary_identifier_canonical,identifier_aliases_raw,evidence_url,sources",
    )
    validated_map: dict[str, set[str]] = {}
    for r in final_rows:
        if not isinstance(r, dict):
            continue
        urls = []
        urls.extend(_coerce_str_list(r.get("evidence_url")))
        urls.extend(_coerce_str_list(r.get("sources")))
        urls = _dedup_keep_order(urls, limit=12)
        raw = _coerce_str(r.get("primary_identifier_raw")) or _coerce_str(r.get("drug_name_code"))
        canon = _coerce_str(r.get("primary_identifier_canonical"))
        aliases = r.get("identifier_aliases_raw")
        if raw:
            _add(validated_map, raw, urls)
        if canon:
            _add(validated_map, canon, urls)
        if isinstance(aliases, list):
            for a in aliases:
                s = _coerce_str(a)
                if s:
                    _add(validated_map, s, urls)

    def _finalize(m: dict[str, set[str]]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for k, urls in m.items():
            out[k] = _dedup_keep_order(sorted(urls), limit=8)
        return out

    return {
        "mentions": _finalize(mentions_map),
        "candidates": _finalize(candidates_map),
        "found_assets": _finalize(found_map),
        "validated_assets": _finalize(validated_map),
    }


def _build_stage_recall_and_pairing(
    *,
    benchmark,
    stage_url_maps: dict[str, dict[str, list[str]]],
) -> tuple[dict[str, object], dict[str, object]]:
    stages = ["mentions", "candidates", "found_assets", "validated_assets"]
    expected_count = len(benchmark)

    stage_key_sets: dict[str, set[str]] = {
        s: set((stage_url_maps.get(s) or {}).keys()) for s in stages
    }

    def _urls_for(stage: str, bench_keys: set[str]) -> list[str]:
        m = stage_url_maps.get(stage) or {}
        urls: list[str] = []
        for k in bench_keys:
            urls.extend(m.get(k) or [])
        return _dedup_keep_order(urls, limit=8)

    # Stage-wise recall counts
    stage_counts: dict[str, int] = {s: 0 for s in stages}
    first_stage_counts: dict[str, int] = {s: 0 for s in stages}
    missing_count = 0

    pairing_rows: list[dict[str, object]] = []
    for item in benchmark:
        bench_keys = match_keys(item.canonical_id)
        stage_present: dict[str, bool] = {s: bool(bench_keys & stage_key_sets.get(s, set())) for s in stages}
        stage_urls: dict[str, list[str]] = {s: _urls_for(s, bench_keys) for s in stages}

        first_stage = "missing"
        first_urls: list[str] = []
        for s in stages:
            if stage_present[s]:
                first_stage = s
                first_urls = stage_urls[s]
                break

        if first_stage == "missing":
            missing_count += 1
        else:
            first_stage_counts[first_stage] += 1

        for s in stages:
            if stage_present[s]:
                stage_counts[s] += 1

        pairing_rows.append(
            {
                "row_number": getattr(item, "row_number", None),
                "benchmark_id": item.canonical_id,
                "benchmark_id_normalized": normalize_identifier(item.canonical_id),
                "first_stage": first_stage,
                "supporting_urls": first_urls,
                "stage_present": stage_present,
                "stage_urls": stage_urls,
            }
        )

    stage_recall = {
        "expected_count": expected_count,
        "stages": [
            {
                "stage": s,
                "present_count": int(stage_counts[s]),
                "recall": (float(stage_counts[s]) / float(expected_count)) if expected_count else 0.0,
            }
            for s in stages
        ],
        "first_stage_counts": {**first_stage_counts, "missing": int(missing_count)},
        "matching": {
            "method": "canonical_match_keys",
            "notes": "Uses eval.utils.match_keys/normalize_identifier for punctuation/case robustness; no benchmark-specific alias lists required.",
        },
    }

    pairing_report = {
        "expected_count": expected_count,
        "matching": stage_recall["matching"],
        "rows": pairing_rows,
    }
    return stage_recall, pairing_report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Load env file(s) if provided, otherwise best-effort common defaults.
    if args.env_file:
        _load_env_file(Path(args.env_file))
    else:
        _load_env_file(Path(".env"))
        _load_env_file(Path("eval") / ".env")

    # v1.5 default benchmark: eval/private (offline-only, never used at runtime)
    if args.benchmark_csv:
        benchmark_path = Path(args.benchmark_csv).expanduser().resolve()
    else:
        default_private = (Path(__file__).resolve().parent / "private" / "cdk12_benchmark.csv").resolve()
        if not default_private.exists():
            raise SystemExit(
                "Missing --benchmark_csv and default private benchmark not found. "
                f"Expected: {default_private}. Provide --benchmark_csv explicitly."
            )
        benchmark_path = default_private
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
        table_label = "local_json"
        stage_recall = None
        pairing_report = None
    else:
        url = os.getenv("EVAL_SUPABASE_URL") or os.getenv("SUPABASE_URL") or ""
        key = (
            os.getenv("EVAL_SUPABASE_SERVICE_KEY")
            or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_SERVICE_KEY")
            or os.getenv("SUPABASE_KEY")
            or ""
        )
        table_default = "draft_assets" if args.mode == "recall_draft" else "final_assets"
        table = args.table or table_default
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
                prediction_id_key=(
                    args.prediction_id_key
                    or os.getenv("EVAL_PREDICTION_ID_KEY")
                    or ("identifier_raw" if args.mode == "recall_draft" else None)
                ),
                prediction_evidence_key=args.prediction_evidence_key or os.getenv("EVAL_PREDICTION_EVIDENCE_KEY"),
                prediction_cycle_key=args.prediction_cycle_key or os.getenv("EVAL_PREDICTION_CYCLE_KEY"),
            )
        )
        predictions = src.fetch_predictions_for_run(args.run_id)
        run_meta = src.fetch_run_metadata(args.run_id)
        table_label = table

        # Stage-wise recall + pairing report (Supabase runs only).
        stage_url_maps = _build_stage_url_maps(
            url=url,
            key=key,
            schema=schema,
            run_id_column=run_id_column,
            run_id=args.run_id,
        )
        stage_recall, pairing_report = _build_stage_recall_and_pairing(
            benchmark=benchmark,
            stage_url_maps=stage_url_maps,
        )

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
        stage_recall=stage_recall,
        pairing_report=pairing_report,
    )

    m = result["metrics"]
    print(f"run_id={args.run_id} version={args.version} mode={args.mode} table={table_label}")
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


