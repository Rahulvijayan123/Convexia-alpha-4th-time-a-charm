from __future__ import annotations

import json
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from eval.sources.common import extract_predictions_from_rows
from eval.types import PredictionItem


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    service_key: str
    table: str
    schema: str = "public"
    run_id_column: str = "run_id"
    timeout_sec: int = 30

    # Optional explicit extraction hints (recommended if your stored output schema is fixed)
    prediction_id_key: Optional[str] = None
    prediction_evidence_key: Optional[str] = None
    prediction_cycle_key: Optional[str] = None


class SupabaseRestSource:
    """
    Minimal Supabase REST reader for evaluation.

    This is intentionally "read-only" and does not call any LLMs.
    """

    def __init__(self, cfg: SupabaseConfig) -> None:
        self._cfg = cfg

    def _fetch_rows(self, *, table: str, query: Dict[str, str]) -> List[Dict[str, Any]]:
        base = self._cfg.url.rstrip("/")
        if not base.startswith("http"):
            raise ValueError("Supabase URL must start with http(s).")

        url = f"{base}/rest/v1/{urllib.parse.quote(table)}?{urllib.parse.urlencode(query)}"

        headers = {
            "apikey": self._cfg.service_key,
            "Authorization": f"Bearer {self._cfg.service_key}",
            "Accept": "application/json",
        }
        if self._cfg.schema and self._cfg.schema != "public":
            headers["Accept-Profile"] = self._cfg.schema

        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.timeout_sec) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(
                f"Supabase request failed: HTTP {e.code}. "
                f"Table={table}. "
                f"Body={err_body[:500]}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Supabase request failed: {e}") from e

        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Supabase response was not valid JSON: {body[:500]}") from e

        if not isinstance(data, list):
            raise RuntimeError(f"Expected Supabase REST response to be a list, got {type(data).__name__}")
        return data  # type: ignore[return-value]

    def fetch_rows_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        # /rest/v1/<table>?select=*&run_id=eq.<run_id>
        query = {
            "select": "*",
            self._cfg.run_id_column: f"eq.{run_id}",
        }
        return self._fetch_rows(table=self._cfg.table, query=query)

    def fetch_run_metadata(self, run_id: str) -> Dict[str, Any] | None:
        """
        Fetches run metadata from the repo's schema (`public.runs`) if present.
        """
        query = {
            "select": "id,created_at,user_query,config_version,prompt_version,params,summary,status",
            "id": f"eq.{run_id}",
        }
        rows = self._fetch_rows(table="runs", query=query)
        if not rows:
            return None
        # Supabase REST returns a list
        row = rows[0]
        return row if isinstance(row, dict) else None

    def _fetch_rows_by_ids(
        self,
        *,
        table: str,
        ids: List[str],
        select: str,
        id_column: str = "id",
        chunk_size: int = 100,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not ids:
            return out
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i : i + chunk_size]
            query = {
                "select": select,
                id_column: f"in.({','.join(chunk)})",
            }
            out.extend(self._fetch_rows(table=table, query=query))
        return out

    def _derive_round_idx_by_candidate_id(self, candidate_ids: List[str]) -> Dict[str, int]:
        """
        Repo-specific enrichment:
        final_assets.candidate_id -> candidates.source_mention_id -> mentions.query_id -> queries.cycle_id -> cycles.round_idx
        """
        # candidates: id,source_mention_id
        cand_rows = self._fetch_rows_by_ids(
            table="candidates",
            ids=candidate_ids,
            select="id,source_mention_id",
        )
        cand_to_mention: Dict[str, str] = {}
        mention_ids: List[str] = []
        for r in cand_rows:
            cid = r.get("id")
            mid = r.get("source_mention_id")
            if isinstance(cid, str) and isinstance(mid, str) and cid and mid:
                cand_to_mention[cid] = mid
                mention_ids.append(mid)

        # mentions: id,query_id
        mention_rows = self._fetch_rows_by_ids(
            table="mentions",
            ids=mention_ids,
            select="id,query_id",
        )
        mention_to_query: Dict[str, str] = {}
        query_ids: List[str] = []
        for r in mention_rows:
            mid = r.get("id")
            qid = r.get("query_id")
            if isinstance(mid, str) and isinstance(qid, str) and mid and qid:
                mention_to_query[mid] = qid
                query_ids.append(qid)

        # queries: id,cycle_id
        query_rows = self._fetch_rows_by_ids(
            table="queries",
            ids=query_ids,
            select="id,cycle_id",
        )
        query_to_cycle: Dict[str, str] = {}
        cycle_ids: List[str] = []
        for r in query_rows:
            qid = r.get("id")
            cyid = r.get("cycle_id")
            if isinstance(qid, str) and isinstance(cyid, str) and qid and cyid:
                query_to_cycle[qid] = cyid
                cycle_ids.append(cyid)

        # cycles: id,round_idx
        cycle_rows = self._fetch_rows_by_ids(
            table="cycles",
            ids=cycle_ids,
            select="id,round_idx",
        )
        cycle_to_round: Dict[str, int] = {}
        for r in cycle_rows:
            cyid = r.get("id")
            ridx = r.get("round_idx")
            if isinstance(cyid, str) and cyid and isinstance(ridx, int):
                cycle_to_round[cyid] = ridx

        cand_to_round: Dict[str, int] = {}
        for cand_id, mention_id in cand_to_mention.items():
            qid = mention_to_query.get(mention_id)
            cyid = query_to_cycle.get(qid) if qid else None
            ridx = cycle_to_round.get(cyid) if cyid else None
            if ridx is not None:
                cand_to_round[cand_id] = int(ridx)
        return cand_to_round

    def fetch_predictions_for_run(self, run_id: str) -> List[PredictionItem]:
        rows = self.fetch_rows_for_run(run_id)
        preds = extract_predictions_from_rows(
            rows,
            id_key=self._cfg.prediction_id_key,
            evidence_key=self._cfg.prediction_evidence_key,
            cycle_key=self._cfg.prediction_cycle_key,
        )
        # If evaluating final_assets, enrich with cycle/round_idx using candidate_id links when available.
        if self._cfg.table == "final_assets" and not self._cfg.prediction_cycle_key:
            candidate_ids: List[str] = []
            for p in preds:
                if isinstance(p.raw, dict):
                    cid = p.raw.get("candidate_id")
                    if isinstance(cid, str) and cid:
                        candidate_ids.append(cid)
            cand_to_round = self._derive_round_idx_by_candidate_id(sorted(set(candidate_ids))) if candidate_ids else {}
            if cand_to_round:
                enriched: List[PredictionItem] = []
                for p in preds:
                    cycle = int(p.cycle or 1)
                    if isinstance(p.raw, dict):
                        cid = p.raw.get("candidate_id")
                        if isinstance(cid, str) and cid in cand_to_round:
                            cycle = int(cand_to_round[cid])
                    enriched.append(PredictionItem(raw_id=p.raw_id, evidence_urls=p.evidence_urls, cycle=cycle, raw=p.raw))
                return enriched
        return preds


