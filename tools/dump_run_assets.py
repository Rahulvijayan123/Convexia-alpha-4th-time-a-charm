from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

from eval.sources.supabase_rest import SupabaseConfig, SupabaseRestSource
from eval.utils import normalize_identifier


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dump_run_assets",
        description="Debug tool: print predicted final_assets rows for a run (identifier + evidence).",
    )
    p.add_argument("--run_id", required=True)
    p.add_argument("--limit", type=int, default=200)
    return p


def _load_final_assets_rows(run_id: str) -> List[Dict[str, Any]]:
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
    src = SupabaseRestSource(
        SupabaseConfig(url=url, service_key=key, table="final_assets", schema=schema, run_id_column=run_id_column)
    )
    return src.fetch_rows_for_run(run_id)


def _one_line(s: str, limit: int = 140) -> str:
    t = " ".join((s or "").strip().split())
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _load_final_assets_rows(args.run_id)
    limit = max(0, int(args.limit))
    shown = 0

    for r in rows:
        if shown >= limit:
            break
        if not isinstance(r, dict):
            continue

        raw = (
            (r.get("primary_identifier_raw") if isinstance(r.get("primary_identifier_raw"), str) else None)
            or (r.get("drug_name_code") if isinstance(r.get("drug_name_code"), str) else None)
            or ""
        ).strip()
        canon = (
            (r.get("primary_identifier_canonical") if isinstance(r.get("primary_identifier_canonical"), str) else None)
            or normalize_identifier(raw)
        ).strip()
        url = (
            (r.get("evidence_url") if isinstance(r.get("evidence_url"), str) else None)
            or (r.get("sources")[0] if isinstance(r.get("sources"), list) and r.get("sources") else None)
            or ""
        )
        snip = (r.get("evidence_snippet") if isinstance(r.get("evidence_snippet"), str) else "") or ""

        print(f"{raw}\t{canon}\t{url}\t{_one_line(snip)}")
        shown += 1

    print(f"rows_total={len(rows)} rows_printed={shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


