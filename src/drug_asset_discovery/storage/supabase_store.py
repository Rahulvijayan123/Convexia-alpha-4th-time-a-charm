from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from supabase import Client, create_client
from supabase.lib.client_options import SyncClientOptions

from drug_asset_discovery.models.domain import Candidate, Mention, ValidatedAsset
from drug_asset_discovery.utils.retry import RetryConfig, async_retry


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _data(obj: Any) -> Any:
    # supabase-py returns different response shapes across versions
    if hasattr(obj, "data"):
        return obj.data
    if isinstance(obj, dict) and "data" in obj:
        return obj["data"]
    return obj


@dataclass
class SupabaseStore:
    url: str
    service_role_key: str
    client: Client | None = None
    postgrest_timeout_s: int = 300
    max_concurrent_requests: int = 8
    _sem: asyncio.Semaphore | None = None

    def __post_init__(self) -> None:
        # Supabase client is sync; we call it via asyncio.to_thread().
        # Increase PostgREST timeout to tolerate bursts and add an async-side concurrency cap.
        self.client = create_client(
            self.url,
            self.service_role_key,
            options=SyncClientOptions(postgrest_client_timeout=httpx.Timeout(self.postgrest_timeout_s)),
        )
        self._sem = asyncio.Semaphore(self.max_concurrent_requests)

    @async_retry(RetryConfig(attempts=8, min_seconds=1.0, max_seconds=60.0))
    async def _to_thread(self, fn):
        # Prevent overloading Supabase/PostgREST with too many concurrent requests.
        assert self._sem is not None
        await self._sem.acquire()
        try:
            return await asyncio.to_thread(fn)
        finally:
            self._sem.release()

    async def health_check(self) -> bool:
        """
        Lightweight check to verify connectivity to the configured Supabase instance.
        Attempts a minimal query against the 'runs' table. Returns True if successful, False otherwise.
        """
        assert self.client is not None
        try:
            # Minimal read to verify connectivity; do not mutate state.
            res = await self._to_thread(
                lambda: self.client.table("runs").select("id").limit(1).execute()
            )
            _data(res)  # normalize shape; even if empty, no exception raised
            return True
        except Exception:
            return False

    async def create_run(
        self,
        *,
        user_query: str,
        config_version: str,
        prompt_version: str,
        params: dict[str, Any],
        idempotency_key: str | None,
    ) -> str:
        assert self.client is not None

        if idempotency_key:
            existing = await self._to_thread(
                lambda: self.client.table("runs")
                .select("id,status")
                .eq("idempotency_key", idempotency_key)
                .limit(1)
                .execute()
            )
            rows = _data(existing) or []
            if isinstance(rows, list) and rows:
                return rows[0]["id"]

        res = await self._to_thread(
            lambda: self.client.table("runs")
            .insert(
                {
                    "user_query": user_query,
                    "config_version": config_version,
                    "prompt_version": prompt_version,
                    "idempotency_key": idempotency_key,
                    "started_at": _utc_now_iso(),
                    "status": "running",
                    "params": params,
                }
            )
            .execute()
        )
        rows = _data(res)
        if not rows:
            raise RuntimeError("Failed to create run")
        return rows[0]["id"]

    async def finish_run(self, run_id: str, *, status: str, summary: dict[str, Any]) -> None:
        assert self.client is not None
        await self._to_thread(
            lambda: self.client.table("runs")
            .update({"status": status, "finished_at": _utc_now_iso(), "summary": summary})
            .eq("id", run_id)
            .execute()
        )

    async def insert_cycle(
        self,
        *,
        run_id: str,
        round_idx: int,
        worker_id: str,
        phase: str,
        planned_query: str | None,
        success: bool,
        metrics: dict[str, Any],
    ) -> str:
        assert self.client is not None
        res = await self._to_thread(
            lambda: self.client.table("cycles")
            .insert(
                {
                    "run_id": run_id,
                    "round_idx": round_idx,
                    "worker_id": worker_id,
                    "phase": phase,
                    "planned_query": planned_query,
                    "success": success,
                    "metrics": metrics,
                }
            )
            .execute()
        )
        rows = _data(res)
        return rows[0]["id"]

    async def insert_query(
        self,
        *,
        run_id: str,
        cycle_id: str | None,
        worker_id: str | None,
        phase: str,
        query_text: str,
        query_fingerprint: str,
    ) -> str:
        assert self.client is not None
        res = await self._to_thread(
            lambda: self.client.table("queries")
            .insert(
                {
                    "run_id": run_id,
                    "cycle_id": cycle_id,
                    "worker_id": worker_id,
                    "phase": phase,
                    "query_text": query_text,
                    "query_fingerprint": query_fingerprint,
                }
            )
            .execute()
        )
        rows = _data(res)
        return rows[0]["id"]

    async def insert_result(
        self,
        *,
        run_id: str,
        query_id: str,
        tool_name: str,
        response_json: dict[str, Any],
        urls: list[str],
    ) -> str:
        assert self.client is not None
        res = await self._to_thread(
            lambda: self.client.table("results")
            .insert(
                {
                    "run_id": run_id,
                    "query_id": query_id,
                    "tool_name": tool_name,
                    "response_json": response_json,
                    "urls": urls,
                }
            )
            .execute()
        )
        rows = _data(res)
        return rows[0]["id"]

    async def insert_mentions(
        self,
        *,
        run_id: str,
        query_id: str | None,
        result_id: str | None,
        mentions: list[Mention],
    ) -> list[str]:
        assert self.client is not None
        if not mentions:
            return []
        payload = [
            {
                "run_id": run_id,
                "query_id": query_id,
                "result_id": result_id,
                "mention_type": m.mention_type,
                "raw_text": m.raw_text,
                "normalized_text": m.normalized_text,
                "context": m.context,
                "source_url": m.source_url,
                "fingerprint": m.fingerprint,
            }
            for m in mentions
        ]
        res = await self._to_thread(lambda: self.client.table("mentions").insert(payload).execute())
        rows = _data(res) or []
        return [r["id"] for r in rows] if isinstance(rows, list) else []

    async def insert_candidate(
        self,
        *,
        run_id: str,
        candidate: Candidate,
        source_mention_id: str | None,
    ) -> str:
        assert self.client is not None
        res = await self._to_thread(
            lambda: self.client.table("candidates")
            .insert(
                {
                    "run_id": run_id,
                    "source_mention_id": source_mention_id,
                    "candidate_type": candidate.candidate_type,
                    "raw_identifier": candidate.raw_identifier,
                    "normalized_identifier": candidate.normalized_identifier,
                    "fingerprint": candidate.fingerprint,
                    "status": "pending",
                    "metadata": {},
                }
            )
            .execute()
        )
        rows = _data(res)
        return rows[0]["id"]

    async def update_candidate_status(self, candidate_id: str, status: str) -> None:
        assert self.client is not None
        await self._to_thread(
            lambda: self.client.table("candidates").update({"status": status}).eq("id", candidate_id).execute()
        )

    async def insert_validation(
        self,
        *,
        run_id: str,
        candidate_id: str,
        status: str,
        model_output: dict[str, Any] | None,
        evidence_urls: list[str],
        error: str | None,
    ) -> str:
        assert self.client is not None
        res = await self._to_thread(
            lambda: self.client.table("validations")
            .insert(
                {
                    "run_id": run_id,
                    "candidate_id": candidate_id,
                    "status": status,
                    "model_output": model_output,
                    "evidence_urls": evidence_urls,
                    "error": error,
                }
            )
            .execute()
        )
        rows = _data(res)
        return rows[0]["id"]

    async def insert_final_assets(
        self,
        *,
        run_id: str,
        candidate_id: str | None,
        assets: list[ValidatedAsset],
    ) -> list[str]:
        assert self.client is not None
        if not assets:
            return []
        payload = [
            {
                "run_id": run_id,
                "candidate_id": candidate_id,
                "drug_name_code": a.drug_name_code,
                "sponsor": a.sponsor,
                "target": a.target,
                "modality": a.modality,
                "indication": a.indication,
                "development_stage": a.development_stage,
                "geography": a.geography,
                "sources": a.sources,
                "fingerprint": a.fingerprint,
            }
            for a in assets
        ]
        res = await self._to_thread(lambda: self.client.table("final_assets").insert(payload).execute())
        rows = _data(res) or []
        return [r["id"] for r in rows] if isinstance(rows, list) else []

    async def insert_metric(self, *, run_id: str, round_idx: int | None, name: str, value: dict[str, Any]) -> None:
        assert self.client is not None
        await self._to_thread(
            lambda: self.client.table("metrics")
            .insert({"run_id": run_id, "round_idx": round_idx, "name": name, "value": value})
            .execute()
        )

    async def get_final_assets(self, run_id: str) -> list[dict[str, Any]]:
        assert self.client is not None
        res = await self._to_thread(
            lambda: self.client.table("final_assets")
            .select(
                "drug_name_code,sponsor,target,modality,indication,development_stage,geography,sources,fingerprint"
            )
            .eq("run_id", run_id)
            .execute()
        )
        rows = _data(res) or []
        return rows if isinstance(rows, list) else []


