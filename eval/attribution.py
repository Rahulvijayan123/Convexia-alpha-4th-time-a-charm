from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from eval.benchmark import load_benchmark_csv
from eval.matching import SeriesIndex, evaluate, load_series_rules, load_synonyms
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


def _keyspace(s: str) -> str:
    # Identifier-preserving normalization used for attribution presence checks:
    # - use eval's explicit normalization rules
    # - compare in a canonical A-Z0-9 keyspace
    return normalize_identifier(s)


def _extract_strings(obj: Any) -> List[str]:
    out: List[str] = []
    if obj is None:
        return out
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, (int, float, bool)):
        return out
    if isinstance(obj, list):
        for el in obj:
            out.extend(_extract_strings(el))
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_extract_strings(v))
        return out
    return out


def _fetch_all_rows_for_run(
    src: SupabaseRestSource,
    *,
    table: str,
    run_id: str,
    select: str,
    extra_filters: Optional[Dict[str, str]] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Fetch all rows for a given run_id with simple PostgREST pagination.
    """
    # NOTE: SupabaseRestSource is intentionally minimal; we re-use its internal request method.
    # pylint/ruff: this is an evaluation-only module; ok to call the private helper.
    rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        query: Dict[str, str] = {
            "select": select,
            "run_id": f"eq.{run_id}",
            "limit": str(limit),
            "offset": str(offset),
        }
        if extra_filters:
            query.update(extra_filters)
        chunk = src._fetch_rows(table=table, query=query)  # type: ignore[attr-defined]
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < limit:
            break
        offset += limit
    return rows


def _invert_synonyms(synonyms: Dict[str, str]) -> Dict[str, Set[str]]:
    inv: Dict[str, Set[str]] = {}
    for alias, canon in synonyms.items():
        inv.setdefault(canon, set()).add(alias)
    return inv


def _target_keyspaces_for_benchmark_id(
    canonical_id_norm: str,
    *,
    synonyms_inv: Dict[str, Set[str]],
    series: SeriesIndex,
) -> Set[str]:
    """
    Build the set of accepted identifier keyspaces that should count as "present" for this benchmark id.

    - If the benchmark id is a series id: require a child member (v1.3 contract), so we match on members (plus aliases).
    - Else: match on the canonical id itself plus any known aliases (synonyms) that map to it.
    """
    ids_to_check: Set[str] = set()

    if canonical_id_norm in series.series_ids:
        # Series benchmark row: presence requires an explicit member identifier.
        for member in series.series_to_members.get(canonical_id_norm, ()):
            ids_to_check.add(member)
            ids_to_check.update(synonyms_inv.get(member, set()))
    else:
        ids_to_check.add(canonical_id_norm)
        ids_to_check.update(synonyms_inv.get(canonical_id_norm, set()))
        # If this benchmark id is a known series member, also treat the parent series identifier as "present"
        # (consistent with the eval contract's optional series_to_member credit).
        parent_series: str | None = None
        for k in match_keys(canonical_id_norm):
            parent_series = series.member_to_series.get(k)
            if parent_series:
                break
        if parent_series:
            ids_to_check.add(parent_series)
            ids_to_check.update(synonyms_inv.get(parent_series, set()))

    out: Set[str] = set()
    for ident in ids_to_check:
        for k in match_keys(ident):
            out.add(k.replace("-", ""))  # hyphenless keyspace
    return {k for k in out if k}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval-attribution",
        description="Offline loss attribution report for a stored run (Supabase) vs a local benchmark CSV.",
    )
    p.add_argument("--run_id", required=True, help="Run id to attribute (UUID stored in Supabase).")
    p.add_argument(
        "--benchmark_csv",
        "--benchmark",
        required=True,
        dest="benchmark_csv",
        help="Path to benchmark CSV (local file). Offline only.",
    )
    p.add_argument("--benchmark_id_column", default=None, help="CSV column containing the benchmark identifier.")
    p.add_argument(
        "--min_benchmark_rows",
        type=int,
        default=int(os.getenv("EVAL_MIN_BENCHMARK_ROWS") or "10"),
        help="Fail fast if loaded benchmark row count is below this threshold. Default: 10 (or env EVAL_MIN_BENCHMARK_ROWS).",
    )

    p.add_argument("--synonyms", default=None, help="Path to synonyms JSON mapping file (optional).")
    p.add_argument("--series_rules", default=None, help="Path to series rules JSON mapping file (optional).")

    p.add_argument(
        "--out_csv",
        default=None,
        help="Output CSV path. Default: eval/reports/attribution/run_<run_id>/attribution.csv",
    )
    p.add_argument("--env_file", default=None, help="Optional path to an .env file for Supabase credentials.")
    return p


def _summary_table(rows: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    for r in rows:
        fm = str(r.get("failure_mode") or "unknown")
        counts[fm] = counts.get(fm, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def main(argv: List[str] | None = None) -> int:
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
    series = load_series_rules(series_path) if series_path else SeriesIndex(frozenset(), {}, {}, {})
    synonyms_inv = _invert_synonyms(dict(synonyms))

    url = os.getenv("EVAL_SUPABASE_URL") or os.getenv("SUPABASE_URL") or ""
    key = (
        os.getenv("EVAL_SUPABASE_SERVICE_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_SERVICE_KEY")
        or os.getenv("SUPABASE_KEY")
        or ""
    )
    if not url or not key:
        raise SystemExit(
            "Missing Supabase credentials. Set EVAL_SUPABASE_URL and EVAL_SUPABASE_SERVICE_KEY (or SUPABASE_* equivalents)."
        )

    src = SupabaseRestSource(SupabaseConfig(url=url, service_key=key, table="final_assets", schema="public"))

    # Stage 0: retrieval blobs (web_search response_json + urls)
    result_rows = _fetch_all_rows_for_run(
        src,
        table="results",
        run_id=args.run_id,
        select="id,tool_name,response_json,urls",
        extra_filters={"tool_name": "eq.web_search"},
    )
    retrieval_texts: List[str] = []
    for r in result_rows:
        retrieval_texts.extend(_extract_strings(r.get("response_json")))
        retrieval_texts.extend(_extract_strings(r.get("urls")))
    retrieval_blob_keyspace = _keyspace(" ".join(retrieval_texts))

    # Stage A: mentions (canonical set)
    mention_rows = _fetch_all_rows_for_run(
        src,
        table="mentions",
        run_id=args.run_id,
        select="mention_type,raw_text,normalized_text,canonical_text,source_url",
    )
    mention_keyspaces: Set[str] = set()
    for r in mention_rows:
        raw = r.get("raw_text")
        norm = r.get("normalized_text")
        canon = r.get("canonical_text")
        if isinstance(raw, str) and raw.strip():
            mention_keyspaces.add(_keyspace(raw))
        if isinstance(norm, str) and norm.strip():
            mention_keyspaces.add(_keyspace(norm))
        if isinstance(canon, str) and canon.strip():
            mention_keyspaces.add(_keyspace(canon))

    # Stage B (v1.5): draft_assets (canonical set)
    draft_rows = _fetch_all_rows_for_run(
        src,
        table="draft_assets",
        run_id=args.run_id,
        select="identifier_raw,identifier_canonical,identifier_aliases_raw",
    )
    draft_keyspaces: Set[str] = set()
    for r in draft_rows:
        raw = r.get("identifier_raw")
        canon = r.get("identifier_canonical")
        aliases = r.get("identifier_aliases_raw")
        if isinstance(raw, str) and raw.strip():
            draft_keyspaces.add(_keyspace(raw))
        if isinstance(canon, str) and canon.strip():
            draft_keyspaces.add(_keyspace(canon))
        if isinstance(aliases, list):
            for a in aliases:
                if isinstance(a, str) and a.strip():
                    draft_keyspaces.add(_keyspace(a))

    # Stage C: final assets (canonical set)
    final_rows = _fetch_all_rows_for_run(
        src,
        table="final_assets",
        run_id=args.run_id,
        select="drug_name_code,primary_identifier_raw,primary_identifier_canonical",
    )
    final_keyspaces: Set[str] = set()
    for r in final_rows:
        raw = r.get("primary_identifier_raw") or r.get("drug_name_code")
        canon = r.get("primary_identifier_canonical")
        if isinstance(raw, str) and raw.strip():
            final_keyspaces.add(_keyspace(raw))
        if isinstance(canon, str) and canon.strip():
            final_keyspaces.add(_keyspace(canon))

    # Official match outcomes (same contract as eval)
    predictions = src.fetch_predictions_for_run(args.run_id)
    eval_res = evaluate(benchmark, predictions, synonyms=synonyms, series=series)
    expected_outcomes = eval_res["expected_outcomes"]
    outcome_by_expected: Dict[str, Any] = {e.expected_id: e for e in expected_outcomes}

    report_rows: List[Dict[str, Any]] = []
    for item in benchmark:
        expected_norm = normalize_identifier(item.canonical_id)
        target_keyspaces = _target_keyspaces_for_benchmark_id(
            expected_norm, synonyms_inv=synonyms_inv, series=series
        )

        retrieval_present = any(k in retrieval_blob_keyspace for k in target_keyspaces)
        mentions_present = bool(target_keyspaces & mention_keyspaces)
        draft_assets_present = bool(target_keyspaces & draft_keyspaces)
        final_assets_present = bool(target_keyspaces & final_keyspaces)

        eo = outcome_by_expected.get(expected_norm)
        matched = False
        match_predicted = ""
        match_type = ""
        match_score = 0.0
        if eo is not None and getattr(eo, "score", 0.0) > 0.0:
            matched = True
            match_score = float(getattr(eo, "score", 0.0) or 0.0)
            md = getattr(eo, "match", None)
            if md is not None:
                match_predicted = getattr(md, "predicted_raw_id", "") or getattr(md, "predicted_id", "") or ""
                match_type = getattr(md, "match_type", "") or ""

        if matched:
            failure_mode = "hit"
        elif not retrieval_present:
            failure_mode = "retrieval_miss"
        elif not mentions_present:
            failure_mode = "extraction_miss"
        elif not draft_assets_present:
            failure_mode = "draft_asset_miss"
        elif not final_assets_present:
            failure_mode = "promotion_miss"
        else:
            # Should be rare: we saw it in final outputs but eval still didn't match.
            failure_mode = "matching_miss"

        report_rows.append(
            {
                "row_number": item.row_number,
                "benchmark_id": item.canonical_id,
                "benchmark_id_normalized": expected_norm,
                "matched": int(matched),
                "match_score": match_score,
                "matched_by_predicted_id": match_predicted,
                "match_type": match_type,
                "present_in_retrieval": int(retrieval_present),
                "present_in_stage_a_mentions": int(mentions_present),
                "present_in_draft_assets": int(draft_assets_present),
                "present_in_final_assets": int(final_assets_present),
                "failure_mode": failure_mode,
            }
        )

    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else (Path("eval") / "reports" / "attribution" / f"run_{args.run_id}" / "attribution.csv")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()) if report_rows else [])
        w.writeheader()
        for r in report_rows:
            w.writerow(r)

    print(f"out_csv={out_csv.resolve()}")
    print("--- summary (failure modes) ---")
    for name, count in _summary_table(report_rows):
        print(f"{name}\t{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


