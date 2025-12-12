from __future__ import annotations

import asyncio
import logging
from collections import deque
from pathlib import Path
from typing import Any

from drug_asset_discovery.config import EnvSettings, PromptBundle, RunConfig, load_config, load_prompts
from drug_asset_discovery.models.domain import Candidate, Mention, ValidatedAsset
from drug_asset_discovery.openai_client import OpenAIResponsesClient
from drug_asset_discovery.storage.supabase_store import SupabaseStore
from drug_asset_discovery.utils.hashing import safe_normalize, stable_sha256
from drug_asset_discovery.utils.idempotency import idempotency_key
from drug_asset_discovery.validation.validator import validate_candidate
from drug_asset_discovery.worker.worker import DEFAULT_PROFILES, Worker

logger = logging.getLogger(__name__)


def _query_fingerprint(query: str) -> str:
    return stable_sha256(safe_normalize(query))


def _compact_summary(
    *,
    round_idx: int,
    unique_mentions_count: int,
    pending_candidates_count: int,
    validated_assets_count: int,
    validated_assets_sample: list[str],
    last_round_new_validated: int,
) -> dict[str, Any]:
    # IMPORTANT: compact summary only. No giant negative lists.
    return {
        "round_idx": round_idx,
        "unique_mentions_count": unique_mentions_count,
        "pending_candidates_count": pending_candidates_count,
        "validated_assets_count": validated_assets_count,
        "validated_assets_sample": validated_assets_sample[:10],
        "last_round_new_validated": last_round_new_validated,
    }


async def run_discovery(
    *,
    user_query: str,
    config_version: str,
    prompt_version: str,
    idempotency: str | None = None,
) -> dict[str, Any]:
    """
    Orchestrates a full run:
    - Loop Mode with 3–4 workers (multi-cycle)
    - Stage 1 recall harvesting (no filtering)
    - Stage 2 strict validation with citations/links
    - Persist everything to Supabase
    - Deterministic replay supported via stored results
    """
    env = EnvSettings()
    cfg: RunConfig = load_config(config_version)
    prompts: PromptBundle = load_prompts(prompt_version)

    if not env.supabase_url or not env.supabase_service_role_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

    store = SupabaseStore(url=env.supabase_url, service_role_key=env.supabase_service_role_key)
    run_id = await store.create_run(
        user_query=user_query,
        config_version=config_version,
        prompt_version=prompt_version,
        params={"workers": cfg.workers, "max_rounds": cfg.max_rounds},
        idempotency_key=idempotency,
    )

    cache_dir = Path(env.local_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    openai = OpenAIResponsesClient(
        api_key=env.openai_api_key,
        base_url=env.openai_base_url,
        timeout_s=cfg.timeouts.openai_seconds,
    )

    try:
        profiles = DEFAULT_PROFILES[: cfg.workers]
        workers: list[Worker] = []
        for i, profile in enumerate(profiles):
            workers.append(
                Worker(
                    worker_id=f"w{i+1}:{profile.name}",
                    profile=profile,
                    cfg=cfg,
                    prompts=prompts,
                    openai=openai,
                    cache_dir=cache_dir,
                    max_frontier_size=cfg.max_frontier_size,
                )
            )

        seen_mentions: set[str] = set()
        seen_candidates: set[str] = set()
        seen_assets: set[str] = set()

        candidate_queue: deque[tuple[str, Candidate]] = deque()  # (candidate_id, candidate)

        consecutive_no_new_validated = 0
        total_validations = 0

        validated_assets: list[ValidatedAsset] = []

        for round_idx in range(cfg.max_rounds):
            summary = _compact_summary(
                round_idx=round_idx,
                unique_mentions_count=len(seen_mentions),
                pending_candidates_count=len(candidate_queue),
                validated_assets_count=len(seen_assets),
                validated_assets_sample=[a.drug_name_code for a in validated_assets[-20:]],
                last_round_new_validated=0,
            )

            logger.info("run=%s round=%s starting workers=%s", run_id, round_idx, len(workers))
            cycle_outputs = await asyncio.gather(
                *[
                    w.run_cycle(run_id=run_id, user_query=user_query, global_summary=summary, round_idx=round_idx)
                    for w in workers
                ]
            )

            successful_workers = sum(1 for o in cycle_outputs if o.success)

            # Persist recall cycles, queries, results, mentions, candidates
            new_mentions_this_round = 0
            new_candidates_this_round = 0

            for o in cycle_outputs:
                cycle_id = await store.insert_cycle(
                    run_id=run_id,
                    round_idx=round_idx,
                    worker_id=o.worker_id,
                    phase="recall",
                    planned_query=o.query,
                    success=o.success,
                    metrics={
                        "planned_queries_count": len(o.planned_queries),
                        "mentions_count": len(o.mentions),
                        "urls_count": len(o.urls),
                    },
                )

                qid = await store.insert_query(
                    run_id=run_id,
                    cycle_id=cycle_id,
                    worker_id=o.worker_id,
                    phase="recall",
                    query_text=o.query,
                    query_fingerprint=_query_fingerprint(o.query),
                )

                # NOTE: we store only the text/urls here for now; full raw tool output is stored in the cache.
                rid = await store.insert_result(
                    run_id=run_id,
                    query_id=qid,
                    tool_name="web_search",
                    response_json=o.response_json,
                    urls=o.urls,
                )

                # Dedup mentions in code (no negative list passed to LLM)
                unique_mentions: list[Mention] = []
                for m in o.mentions:
                    if m.fingerprint in seen_mentions:
                        continue
                    seen_mentions.add(m.fingerprint)
                    unique_mentions.append(m)
                if unique_mentions:
                    await store.insert_mentions(run_id=run_id, query_id=qid, result_id=rid, mentions=unique_mentions)
                    new_mentions_this_round += len(unique_mentions)

                # Create broad candidates from mentions (Stage 1 recall; Stage 2 is strict)
                for m in unique_mentions:
                    c = Candidate.from_mention(m)
                    if c.fingerprint in seen_candidates:
                        continue
                    seen_candidates.add(c.fingerprint)
                    cid = await store.insert_candidate(run_id=run_id, candidate=c, source_mention_id=None)
                    candidate_queue.append((cid, c))
                    new_candidates_this_round += 1

            # Validation batch
            logger.info(
                "run=%s round=%s harvested new_mentions=%s new_candidates=%s queue=%s",
                run_id,
                round_idx,
                new_mentions_this_round,
                new_candidates_this_round,
                len(candidate_queue),
            )

            new_validated_assets = 0
            validation_budget = min(cfg.validation_batch_size_per_round, len(candidate_queue))
            validation_budget = min(validation_budget, max(0, cfg.max_total_validations - total_validations))

            sem = asyncio.Semaphore(2)

            async def _validate_one(cid: str, cand: Candidate) -> list[ValidatedAsset]:
                async with sem:
                    vr = await validate_candidate(
                        openai=openai,
                        cfg=cfg,
                        prompts=prompts,
                        cache_dir=cache_dir,
                        user_query=user_query,
                        candidate=cand,
                        idempotency_key=idempotency_key(
                            "validate.candidate",
                            {
                                "run_id": run_id,
                                "round_idx": round_idx,
                                "candidate_fp": cand.fingerprint,
                                "raw": cand.raw_identifier,
                            },
                        ),
                    )
                    await store.insert_validation(
                        run_id=run_id,
                        candidate_id=cid,
                        status=vr.status,
                        model_output=vr.model_output,
                        evidence_urls=vr.evidence_urls,
                        error=vr.rejected_reason if vr.status != "validated" else None,
                    )
                    if vr.status == "validated":
                        await store.update_candidate_status(cid, "validated")
                        await store.insert_final_assets(run_id=run_id, candidate_id=cid, assets=vr.assets)
                        return vr.assets
                    await store.update_candidate_status(cid, "rejected")
                    return []

            to_validate: list[tuple[str, Candidate]] = [candidate_queue.popleft() for _ in range(validation_budget)]
            tasks = [_validate_one(cid, cand) for cid, cand in to_validate]
            validated_batches = await asyncio.gather(*tasks) if tasks else []
            total_validations += validation_budget

            for batch in validated_batches:
                for asset in batch:
                    if asset.fingerprint in seen_assets:
                        continue
                    seen_assets.add(asset.fingerprint)
                    validated_assets.append(asset)
                    new_validated_assets += 1

            await store.insert_metric(
                run_id=run_id,
                round_idx=round_idx,
                name="round_summary",
                value={
                    "successful_workers": successful_workers,
                    "new_mentions": new_mentions_this_round,
                    "new_candidates": new_candidates_this_round,
                    "validated_new_assets": new_validated_assets,
                    "pending_candidates": len(candidate_queue),
                    "total_validations": total_validations,
                },
            )

            # Stopping condition: multiple successive high-quality rounds produce zero new validated assets
            high_quality = successful_workers >= cfg.min_successful_workers_per_round
            if high_quality and new_validated_assets == 0:
                consecutive_no_new_validated += 1
            elif new_validated_assets > 0:
                consecutive_no_new_validated = 0

            logger.info(
                "run=%s round=%s new_validated=%s streak=%s high_quality=%s",
                run_id,
                round_idx,
                new_validated_assets,
                consecutive_no_new_validated,
                high_quality,
            )

            if consecutive_no_new_validated >= cfg.stop_after_no_new_validated_rounds:
                break
            if total_validations >= cfg.max_total_validations:
                break

        summary = {
            "unique_mentions": len(seen_mentions),
            "unique_candidates": len(seen_candidates),
            "validated_assets": len(seen_assets),
            "total_validations": total_validations,
        }
        await store.finish_run(run_id, status="completed", summary=summary)

        return {
            "run_id": run_id,
            "status": "completed",
            "summary": summary,
            "assets": [a.model_dump() for a in validated_assets],
        }
    except Exception as e:
        await store.finish_run(run_id, status="failed", summary={"error": str(e)})
        raise
    finally:
        await openai.aclose()


