from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from eval.benchmark import load_benchmark_csv
from eval.sources.supabase_rest import SupabaseConfig, SupabaseRestSource
from eval.utils import normalize_identifier


def _openai_output_text(raw: Dict[str, Any]) -> str:
    # Mirrors drug_asset_discovery.openai_client.OpenAIResponse.output_text (best-effort).
    if isinstance(raw.get("output_text"), str):
        return str(raw.get("output_text") or "")
    texts: list[str] = []
    for item in raw.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if not isinstance(part, dict):
                continue
            t = part.get("text") or part.get("output_text") or part.get("content")
            if isinstance(t, str) and t.strip():
                texts.append(t)
    return "\n".join(texts).strip()


def _load_supabase_source(table: str) -> SupabaseRestSource:
    url = os.getenv("EVAL_SUPABASE_URL") or os.getenv("SUPABASE_URL") or ""
    key = (
        os.getenv("EVAL_SUPABASE_SERVICE_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_SERVICE_KEY")
        or os.getenv("SUPABASE_KEY")
        or ""
    )
    schema = os.getenv("EVAL_SUPABASE_SCHEMA") or "public"
    run_id_column = os.getenv("EVAL_SUPABASE_RUN_ID_COLUMN") or "run_id"
    if not url or not key:
        raise SystemExit(
            "Missing Supabase credentials. Set EVAL_SUPABASE_URL and EVAL_SUPABASE_SERVICE_KEY "
            "(or SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)."
        )
    return SupabaseRestSource(SupabaseConfig(url=url, service_key=key, table=table, schema=schema, run_id_column=run_id_column))


def _stage_sets_for_run(run_id: str) -> Tuple[str, set[str], set[str], set[str], set[str]]:
    """
    Returns:
    - retrieved_text_corpus_canon (string)
    - mention_canons (set)
    - candidate_canons (set)
    - final_asset_canons (set)
    - final_asset_raws (set) [for debug]
    """
    # results: retrieved snippets/text are embedded in response_json from web_search.
    results_src = _load_supabase_source("results")
    results_rows = results_src.fetch_rows_for_run(run_id)
    retrieved_texts: list[str] = []
    for r in results_rows:
        if not isinstance(r, dict):
            continue
        resp = r.get("response_json")
        if isinstance(resp, dict):
            txt = _openai_output_text(resp)
            if txt:
                retrieved_texts.append(txt)
    retrieved_corpus = "\n".join(retrieved_texts)
    retrieved_corpus_canon = normalize_identifier(retrieved_corpus)

    mentions_src = _load_supabase_source("mentions")
    mentions_rows = mentions_src.fetch_rows_for_run(run_id)
    mention_canons: set[str] = set()
    for r in mentions_rows:
        if not isinstance(r, dict):
            continue
        # Prefer stored canonical_text if available; otherwise compute from raw_text.
        c = r.get("canonical_text")
        if isinstance(c, str) and c.strip():
            mention_canons.add(c.strip())
            continue
        raw = r.get("raw_text")
        if isinstance(raw, str) and raw.strip():
            mention_canons.add(normalize_identifier(raw))

    candidates_src = _load_supabase_source("candidates")
    cand_rows = candidates_src.fetch_rows_for_run(run_id)
    candidate_canons: set[str] = set()
    for r in cand_rows:
        if not isinstance(r, dict):
            continue
        c = r.get("canonical_identifier")
        if isinstance(c, str) and c.strip():
            candidate_canons.add(c.strip())
            continue
        raw = r.get("raw_identifier")
        if isinstance(raw, str) and raw.strip():
            candidate_canons.add(normalize_identifier(raw))

    final_src = _load_supabase_source("final_assets")
    final_rows = final_src.fetch_rows_for_run(run_id)
    final_asset_canons: set[str] = set()
    final_asset_raws: set[str] = set()
    for r in final_rows:
        if not isinstance(r, dict):
            continue
        raw = None
        if isinstance(r.get("primary_identifier_raw"), str) and str(r.get("primary_identifier_raw")).strip():
            raw = str(r.get("primary_identifier_raw")).strip()
        elif isinstance(r.get("drug_name_code"), str) and str(r.get("drug_name_code")).strip():
            raw = str(r.get("drug_name_code")).strip()
        if raw:
            final_asset_raws.add(raw)
            final_asset_canons.add(normalize_identifier(raw))

    return retrieved_corpus_canon, mention_canons, candidate_canons, final_asset_canons, final_asset_raws


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="diagnose_pipeline_attribution",
        description="v1.4 evaluation diagnostic: pinpoint where benchmark identifiers are lost across the pipeline.",
    )
    p.add_argument("--run_id", required=True)
    p.add_argument(
        "--benchmark_csv",
        default=str(Path("eval") / "private" / "cdk12_benchmark.csv"),
        help="Path to benchmark CSV (offline). Default: eval/private/cdk12_benchmark.csv",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    benchmark_path = Path(args.benchmark_csv).expanduser().resolve()
    if not benchmark_path.exists():
        raise SystemExit(f"Benchmark CSV not found: {benchmark_path}")
    benchmark_sha256 = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()

    benchmark = load_benchmark_csv(benchmark_path)
    bench_rows = len(benchmark)
    print(f"benchmark_csv={benchmark_path}")
    print(f"benchmark_sha256={benchmark_sha256}")
    print(f"benchmark_rows={bench_rows}")

    bench_pairs: list[tuple[str, str]] = []
    bench_canons: set[str] = set()
    for b in benchmark:
        raw = str(b.canonical_id).strip()
        canon = normalize_identifier(raw)
        if not raw or not canon:
            continue
        bench_pairs.append((raw, canon))
        bench_canons.add(canon)

    retrieved_corpus_canon, mention_canons, candidate_canons, final_asset_canons, _final_raws = _stage_sets_for_run(
        args.run_id
    )

    # Per-benchmark-id stage booleans (unique IDs)
    rows_out: list[dict[str, Any]] = []
    retrieved_hits = 0
    mention_hits = 0
    candidate_hits = 0
    final_hits = 0

    for raw, canon in bench_pairs:
        retrieved = bool(canon and retrieved_corpus_canon and canon in retrieved_corpus_canon)
        mention = canon in mention_canons
        candidate = canon in candidate_canons
        final = canon in final_asset_canons
        rows_out.append(
            {
                "benchmark_raw": raw,
                "benchmark_canonical": canon,
                "retrieved_text_hit": str(retrieved).lower(),
                "mention_hit": str(mention).lower(),
                "candidate_hit": str(candidate).lower(),
                "final_asset_hit": str(final).lower(),
            }
        )
        if retrieved:
            retrieved_hits += 1
        if mention:
            mention_hits += 1
        if candidate:
            candidate_hits += 1
        if final:
            final_hits += 1

    print(f"retrieved_text_hit_count={retrieved_hits}")
    print(f"mention_hit_count={mention_hits}")
    print(f"candidate_hit_count={candidate_hits}")
    print(f"final_asset_hit_count={final_hits}")

    out_dir = Path("eval") / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run_id}_attribution.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "benchmark_raw",
                "benchmark_canonical",
                "retrieved_text_hit",
                "mention_hit",
                "candidate_hit",
                "final_asset_hit",
            ],
        )
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"attribution_csv={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


