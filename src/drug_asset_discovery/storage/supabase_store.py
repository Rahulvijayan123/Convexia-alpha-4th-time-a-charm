from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from postgrest.exceptions import APIError
from supabase import Client, create_client
from supabase.lib.client_options import SyncClientOptions

from drug_asset_discovery.models.domain import Candidate, Mention, ValidatedAsset
from drug_asset_discovery.utils.hashing import stable_sha256
from drug_asset_discovery.utils.retry import RetryConfig, async_retry

logger = logging.getLogger(__name__)


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
    _draft_assets_missing: bool = False

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

    def _insert_table_with_schema_fallback(self, table: str, payload: list[dict[str, Any]]):
        """
        Synchronous helper used inside thread executor to insert into `table`.
        If PostgREST complains about a missing column in the schema cache, remove that
        key from the payload and retry. This keeps the client tolerant to schema drift.
        """
        attempt_payload = [dict(item) for item in payload]
        while True:
            try:
                return self.client.table(table).insert(attempt_payload).execute()
            except Exception as e:
                msg = str(e)
                col = None
                try:
                    pattern = f"' column of '{table}'"
                    if "Could not find the '" in msg and pattern in msg:
                        start = msg.index("Could not find the '") + len("Could not find the '")
                        end = msg.index(pattern, start)
                        col = msg[start:end]
                except Exception:
                    col = None
                if not col:
                    # fallback: try common candidate keys
                    for candidate in ("canonical_identifier", "mention_class", "canonical_text", "primary_identifier_raw"):
                        if candidate in msg:
                            col = candidate
                            break
                if not col:
                    raise
                removed = False
                for item in attempt_payload:
                    if col in item:
                        item.pop(col, None)
                        removed = True
                if not removed:
                    raise

    async def health_check(self) -> bool:
        """
        Lightweight check to verify connectivity to the configured Supabase instance.
        Attempts a minimal query against the 'runs' table. Returns True if successful, False otherwise.
        """
        assert self.client is not None
        try:
            t0 = time.perf_counter()
            # Minimal read to verify connectivity; do not mutate state.
            res = await self._to_thread(
                lambda: self.client.table("runs").select("id").limit(1).execute()
            )
            _data(res)  # normalize shape; even if empty, no exception raised
            logger.debug("supabase health_check ok duration_ms=%s", int((time.perf_counter() - t0) * 1000))
            return True
        except Exception:
            logger.exception("supabase health_check failed")
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
        t0 = time.perf_counter()
        logger.info(
            "supabase create_run start config=%s prompt=%s idempotency=%s query_chars=%s",
            config_version,
            prompt_version,
            bool(idempotency_key),
            len(user_query or ""),
        )

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
                rid = rows[0]["id"]
                logger.info(
                    "supabase create_run idempotency_hit run_id=%s duration_ms=%s",
                    rid,
                    int((time.perf_counter() - t0) * 1000),
                )
                return rid

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
        rid = rows[0]["id"]
        logger.info("supabase create_run ok run_id=%s duration_ms=%s", rid, int((time.perf_counter() - t0) * 1000))
        return rid

    async def finish_run(self, run_id: str, *, status: str, summary: dict[str, Any]) -> None:
        assert self.client is not None
        t0 = time.perf_counter()
        await self._to_thread(
            lambda: self.client.table("runs")
            .update({"status": status, "finished_at": _utc_now_iso(), "summary": summary})
            .eq("id", run_id)
            .execute()
        )
        logger.info("supabase finish_run ok run_id=%s status=%s duration_ms=%s", run_id, status, int((time.perf_counter() - t0) * 1000))

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
        t0 = time.perf_counter()
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
        cid = rows[0]["id"]
        logger.debug(
            "supabase insert_cycle ok run_id=%s round=%s worker=%s phase=%s success=%s duration_ms=%s",
            run_id,
            round_idx,
            worker_id,
            phase,
            success,
            int((time.perf_counter() - t0) * 1000),
        )
        return cid

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
        t0 = time.perf_counter()
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
        qid = rows[0]["id"]
        logger.debug(
            "supabase insert_query ok run_id=%s phase=%s worker=%s query_chars=%s duration_ms=%s",
            run_id,
            phase,
            worker_id,
            len(query_text or ""),
            int((time.perf_counter() - t0) * 1000),
        )
        return qid

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
        t0 = time.perf_counter()
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
        rid = rows[0]["id"]
        logger.debug(
            "supabase insert_result ok run_id=%s tool=%s urls=%s duration_ms=%s",
            run_id,
            tool_name,
            len(urls or []),
            int((time.perf_counter() - t0) * 1000),
        )
        return rid

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
        t0 = time.perf_counter()
        payload = [
            {
                "run_id": run_id,
                "query_id": query_id,
                "result_id": result_id,
                "mention_type": m.mention_type,
                "raw_text": m.raw_text,
                "normalized_text": m.normalized_text,
                "canonical_text": m.canonical_text,
                "mention_class": m.mention_class,
                "context": m.context,
                "source_url": m.source_url,
                "fingerprint": m.fingerprint,
            }
            for m in mentions
        ]
        def _do_insert(p: list[dict[str, Any]]):
            # Try inserting; if PostgREST reports a missing column (schema cache mismatch),
            # remove that column from the payload and retry. Repeat until success or no removable keys left.
            attempt_payload = [dict(item) for item in p]
            while True:
                try:
                    return self.client.table("mentions").insert(attempt_payload).execute()
                except Exception as e:
                    msg = str(e)
                    # Look for pattern: Could not find the 'COLUMN' column of 'mentions'
                    col = None
                    try:
                        # naive parse for the quoted column name
                        if "Could not find the '" in msg and "' column of 'mentions'" in msg:
                            start = msg.index("Could not find the '") + len("Could not find the '")
                            end = msg.index("' column of 'mentions'", start)
                            col = msg[start:end]
                    except Exception:
                        col = None
                    if not col:
                        # fallback: try simple keywords
                        for candidate in ("canonical_text", "mention_class"):
                            if candidate in msg:
                                col = candidate
                                break
                    if not col:
                        # nothing we can handle programmatically; re-raise
                        raise
                    # remove the offending column from all items and retry
                    removed = False
                    for item in attempt_payload:
                        if col in item:
                            item.pop(col, None)
                            removed = True
                    if not removed:
                        # nothing removed; avoid infinite loop
                        raise

        res = await self._to_thread(lambda: _do_insert(payload))
        rows = _data(res) or []
        ids = [r["id"] for r in rows] if isinstance(rows, list) else []
        logger.debug(
            "supabase insert_mentions ok run_id=%s count=%s duration_ms=%s",
            run_id,
            len(ids),
            int((time.perf_counter() - t0) * 1000),
        )
        return ids

    async def insert_candidate(
        self,
        *,
        run_id: str,
        candidate: Candidate,
        source_mention_id: str | None,
    ) -> str:
        assert self.client is not None
        t0 = time.perf_counter()
        payload = {
            "run_id": run_id,
            "source_mention_id": source_mention_id,
            "candidate_type": candidate.candidate_type,
            "raw_identifier": candidate.raw_identifier,
            "normalized_identifier": candidate.normalized_identifier,
            "canonical_identifier": candidate.canonical_identifier,
            "mention_class": candidate.mention_class,
            "fingerprint": candidate.fingerprint,
            "status": "pending",
            "metadata": {},
        }
        res = await self._to_thread(lambda: self._insert_table_with_schema_fallback("candidates", [payload]))
        rows = _data(res)
        cid = rows[0]["id"]
        logger.debug(
            "supabase insert_candidate ok run_id=%s candidate_type=%s duration_ms=%s",
            run_id,
            candidate.candidate_type,
            int((time.perf_counter() - t0) * 1000),
        )
        return cid

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
        t0 = time.perf_counter()
        payload = [
            {
                "run_id": run_id,
                "candidate_id": candidate_id,
                # Back-compat: keep the legacy column populated with the evidence-anchored primary identifier.
                "drug_name_code": a.primary_identifier_raw,
                # v1.4 identifier contract
                "primary_identifier_raw": a.primary_identifier_raw,
                "primary_identifier_canonical": a.primary_identifier_canonical,
                "identifier_aliases_raw": a.identifier_aliases_raw,
                "evidence_snippet": a.evidence_snippet,
                "evidence_url": a.evidence_url,
                "evidence_source_type": a.evidence_source_type,
                # v1.5: allow partial enrichment in final_assets (nullable columns)
                "sponsor": (a.sponsor or None),
                "target": (a.target or None),
                "modality": (a.modality or None),
                "indication": (a.indication or None),
                "development_stage": (a.development_stage or None),
                "geography": (a.geography or None),
                "sources": a.sources,
                "fingerprint": a.fingerprint,
            }
            for a in assets
        ]
        res = await self._to_thread(lambda: self._insert_table_with_schema_fallback("final_assets", payload))
        rows = _data(res) or []
        ids = [r["id"] for r in rows] if isinstance(rows, list) else []
        logger.info(
            "supabase insert_final_assets ok run_id=%s count=%s duration_ms=%s",
            run_id,
            len(ids),
            int((time.perf_counter() - t0) * 1000),
        )
        return ids

    def _coerce_str_list(self, v: Any) -> list[str]:
        if not v:
            return []
        if isinstance(v, list):
            out: list[str] = []
            for x in v:
                if isinstance(x, str) and x.strip():
                    out.append(x.strip())
            return out
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    def _merge_str_lists(self, a: Any, b: Any) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for x in self._coerce_str_list(a) + self._coerce_str_list(b):
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    def _coerce_citations(self, v: Any) -> list[dict[str, str]]:
        """
        Coerce to list of {"url": str, "snippet": str} dicts.
        """
        if not v:
            return []
        if not isinstance(v, list):
            return []
        out: list[dict[str, str]] = []
        for item in v:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            snip = item.get("snippet")
            if not (isinstance(url, str) and url.strip()):
                continue
            if not (isinstance(snip, str) and snip.strip()):
                continue
            out.append({"url": url.strip(), "snippet": snip.strip()})
        return out

    def _merge_citations(self, a: Any, b: Any) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in self._coerce_citations(a) + self._coerce_citations(b):
            key = stable_sha256(f"{item.get('url','')}|{item.get('snippet','')}")
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _merge_enrichment_status(self, existing: Any, incoming: Any) -> str:
        """
        v1.5 status precedence:
        complete > partial > failed > pending
        """
        order = {"pending": 0, "failed": 1, "partial": 2, "complete": 3}
        e = str(existing or "pending").strip()
        i = str(incoming or "pending").strip()
        if e not in order:
            e = "pending"
        if i not in order:
            i = "pending"
        return e if order[e] >= order[i] else i

    def _mark_draft_assets_missing(self, err: Any) -> None:
        self._draft_assets_missing = True
        logger.warning(
            "Supabase schema missing public.draft_assets; skipping draft_assets persistence. "
            "Run supabase/schema.sql in Supabase SQL editor to fix. err=%s",
            err,
        )

    def _is_draft_assets_missing_error(self, err: Any) -> bool:
        # PostgREST shapes vary across versions; be defensive.
        if isinstance(err, dict):
            code = str(err.get("code") or "")
            msg = str(err.get("message") or "")
            hint = str(err.get("hint") or "")
            blob = " ".join([code, msg, hint]).lower()
            return ("draft_assets" in blob) and (code == "PGRST205" or "could not find the table" in blob)
        blob = str(err).lower()
        return ("draft_assets" in blob) and ("could not find the table" in blob or "pgrst205" in blob or "404" in blob)

    async def upsert_draft_asset(self, *, run_id: str, draft: dict[str, Any]) -> str:
        """
        Insert a v1.5 draft asset, deduped by (run_id, identifier_canonical).
        Merge identifier_aliases_raw and citations; never overwrite identifier_raw once set.
        """
        assert self.client is not None
        if self._draft_assets_missing:
            return ""
        t0 = time.perf_counter()
        canon = str(draft.get("identifier_canonical") or "").strip()
        if not canon:
            raise ValueError("draft.identifier_canonical is required for upsert")

        try:
            existing_res = await self._to_thread(
                lambda: self.client.table("draft_assets")
                .select("*")
                .eq("run_id", run_id)
                .eq("identifier_canonical", canon)
                .limit(1)
                .execute()
            )
        except APIError as e:
            err = e.args[0] if getattr(e, "args", None) else None
            if self._is_draft_assets_missing_error(err):
                self._mark_draft_assets_missing(err)
                return ""
            raise
        existing_rows = _data(existing_res) or []

        if not (isinstance(existing_rows, list) and existing_rows):
            # Fresh insert.
            payload = dict(draft)
            payload["run_id"] = run_id
            # Ensure aliases include identifier_raw if present.
            raw = str(payload.get("identifier_raw") or "").strip()
            aliases = self._merge_str_lists(payload.get("identifier_aliases_raw"), [raw] if raw else [])
            payload["identifier_aliases_raw"] = aliases
            # Ensure citations includes evidence_url/snippet.
            ev_url = str(payload.get("evidence_url") or "").strip()
            ev_snip = str(payload.get("evidence_snippet") or "").strip()
            base_cite = [{"url": ev_url, "snippet": ev_snip}] if (ev_url and ev_snip) else []
            payload["citations"] = self._merge_citations(payload.get("citations"), base_cite)
            res = await self._to_thread(lambda: self._insert_table_with_schema_fallback("draft_assets", [payload]))
            rows = _data(res)
            did = rows[0]["id"]
            logger.debug(
                "supabase upsert_draft_asset insert ok run_id=%s canon=%s duration_ms=%s",
                run_id,
                canon,
                int((time.perf_counter() - t0) * 1000),
            )
            return did

        row = existing_rows[0]
        draft_id = row.get("id")
        if not isinstance(draft_id, str) or not draft_id:
            raise RuntimeError("draft_assets upsert failed: missing id")

        # Merge without overwriting core discovery anchors.
        existing_raw = str(row.get("identifier_raw") or "").strip()
        incoming_raw = str(draft.get("identifier_raw") or "").strip()
        merged_aliases = self._merge_str_lists(row.get("identifier_aliases_raw"), draft.get("identifier_aliases_raw"))
        # Always preserve raw variants as aliases.
        merged_aliases = self._merge_str_lists(merged_aliases, [existing_raw, incoming_raw])

        merged_citations = self._merge_citations(row.get("citations"), draft.get("citations"))
        # Ensure citations include the (possibly existing) evidence_url/snippet.
        ev_url = str(row.get("evidence_url") or draft.get("evidence_url") or "").strip()
        ev_snip = str(row.get("evidence_snippet") or draft.get("evidence_snippet") or "").strip()
        merged_citations = self._merge_citations(merged_citations, [{"url": ev_url, "snippet": ev_snip}] if (ev_url and ev_snip) else [])

        update_payload: dict[str, Any] = {
            "identifier_aliases_raw": merged_aliases,
            "citations": merged_citations,
            "confidence_discovery": max(
                float(row.get("confidence_discovery") or 0.0),
                float(draft.get("confidence_discovery") or 0.0),
            )
            if (row.get("confidence_discovery") is not None or draft.get("confidence_discovery") is not None)
            else None,
            "enrichment_status": self._merge_enrichment_status(row.get("enrichment_status"), draft.get("enrichment_status")),
        }

        # v1.5 late-filtering fields (optional; safe to overwrite with new values)
        if draft.get("identifier_type") is not None and not str(row.get("identifier_type") or "").strip():
            update_payload["identifier_type"] = draft.get("identifier_type")
        if draft.get("match_scores") is not None:
            update_payload["match_scores"] = draft.get("match_scores")
        if draft.get("rejection_reason") is not None:
            update_payload["rejection_reason"] = draft.get("rejection_reason")
        if draft.get("extracted_context") is not None:
            existing_ctx = row.get("extracted_context")
            incoming_ctx = draft.get("extracted_context")
            if isinstance(existing_ctx, dict) and isinstance(incoming_ctx, dict):
                update_payload["extracted_context"] = {**existing_ctx, **incoming_ctx}
            else:
                update_payload["extracted_context"] = incoming_ctx

        # Fill optional enrichment fields if missing; never overwrite non-empty.
        for k in ("sponsor", "target", "modality", "indication", "stage", "geography"):
            cur = row.get(k)
            inc = draft.get(k)
            cur_s = str(cur).strip() if isinstance(cur, str) else ""
            inc_s = str(inc).strip() if isinstance(inc, str) else ""
            if not cur_s and inc_s:
                update_payload[k] = inc_s

        await self._to_thread(
            lambda: self.client.table("draft_assets").update(update_payload).eq("id", draft_id).execute()
        )
        logger.debug(
            "supabase upsert_draft_asset update ok run_id=%s canon=%s duration_ms=%s",
            run_id,
            canon,
            int((time.perf_counter() - t0) * 1000),
        )
        return draft_id

    async def get_draft_assets(self, run_id: str) -> list[dict[str, Any]]:
        assert self.client is not None
        if self._draft_assets_missing:
            return []
        try:
            res = await self._to_thread(
                lambda: self.client.table("draft_assets")
                .select("*")
                .eq("run_id", run_id)
                .execute()
            )
        except APIError as e:
            err = e.args[0] if getattr(e, "args", None) else None
            if self._is_draft_assets_missing_error(err):
                self._mark_draft_assets_missing(err)
                return []
            raise
        rows = _data(res) or []
        return rows if isinstance(rows, list) else []

    async def insert_metric(self, *, run_id: str, round_idx: int | None, name: str, value: dict[str, Any]) -> None:
        assert self.client is not None
        t0 = time.perf_counter()
        await self._to_thread(
            lambda: self.client.table("metrics")
            .insert({"run_id": run_id, "round_idx": round_idx, "name": name, "value": value})
            .execute()
        )
        logger.debug(
            "supabase insert_metric ok run_id=%s round_idx=%s name=%s duration_ms=%s",
            run_id,
            round_idx,
            name,
            int((time.perf_counter() - t0) * 1000),
        )

    async def get_final_assets(self, run_id: str) -> list[dict[str, Any]]:
        assert self.client is not None
        res = await self._to_thread(
            lambda: self.client.table("final_assets")
            .select(
                "drug_name_code,primary_identifier_raw,primary_identifier_canonical,identifier_aliases_raw,"
                "evidence_snippet,evidence_url,evidence_source_type,"
                "sponsor,target,modality,indication,development_stage,geography,sources,fingerprint"
            )
            .eq("run_id", run_id)
            .execute()
        )
        rows = _data(res) or []
        return rows if isinstance(rows, list) else []


