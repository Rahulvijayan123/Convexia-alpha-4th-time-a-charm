from __future__ import annotations

import asyncio
import logging
import random
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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


def _domain(url: str) -> str | None:
    try:
        host = urlparse(url).netloc
        host = host.lower().strip()
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None


def _compact_loop_summary(
    *,
    cycle_idx: int,
    unique_identifier_count: int,
    last_cycle_new_identifiers: int,
    sample_new_identifiers: list[str],
    top_domains: list[tuple[str, int]],
) -> dict[str, Any]:
    # IMPORTANT: compact summary only. No giant negative lists.
    return {
        "cycle_idx": cycle_idx,
        "unique_identifier_count": unique_identifier_count,
        "last_cycle_new_identifiers": last_cycle_new_identifiers,
        "sample_new_identifiers": sample_new_identifiers[:8],
        "top_domains": [{"domain": d, "count": c} for d, c in top_domains[:8]],
    }


async def loop_mode(
    *,
    user_query: str,
    config_version: str,
    prompt_version: str,
    idempotency: str | None = None,
) -> dict[str, Any]:
    """
    v1.2: ChatGPT Loop Mode
    - 4 independent workers (tabs) in parallel
    - 3 global cycles max
    - Each worker-cycle uses the Responses API with web_search (≤2 tool calls) and outputs JSON-only
    - Stage A: identifier-first harvesting (store raw strings with provenance)
    - Stage B: lightweight candidate-centric verification (implemented in a later step)
    """
    env = EnvSettings()
    cfg: RunConfig = load_config(config_version)
    prompts: PromptBundle = load_prompts(prompt_version)

    # Non-negotiable budget constraints for loop_mode v1.2
    if cfg.workers != 4:
        raise ValueError(f"loop_mode requires workers=4 (got {cfg.workers})")
    if cfg.max_rounds != 3:
        raise ValueError(f"loop_mode requires max_rounds=3 (got {cfg.max_rounds})")
    if cfg.max_web_search_calls_per_worker_cycle != 2:
        raise ValueError("loop_mode requires max_web_search_calls_per_worker_cycle=2")

    if not env.supabase_url or not env.supabase_service_role_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

    store = SupabaseStore(url=env.supabase_url, service_role_key=env.supabase_service_role_key)
    run_id = await store.create_run(
        user_query=user_query,
        config_version=config_version,
        prompt_version=prompt_version,
        params={
            "mode": "loop_mode",
            "workers": cfg.workers,
            "cycles": cfg.max_rounds,
            "min_new_identifiers_per_cycle": cfg.min_new_identifiers_per_cycle,
            "patience_cycles": cfg.patience_cycles,
            "verify_top_k": cfg.verify_top_k,
        },
        idempotency_key=idempotency,
    )

    cache_dir = Path(env.local_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    openai = OpenAIResponsesClient(
        api_key=env.openai_api_key,
        base_url=env.openai_base_url,
        timeout_s=cfg.timeouts.openai_seconds,
    )

    # Import here to keep module import-time stable even if loop_mode isn't used.
    from drug_asset_discovery.worker.loop_worker import DEFAULT_LOOP_PERSONAS, LoopWorker

    rng = random.Random(int(cfg.orchestration_seed))
    personas = list(DEFAULT_LOOP_PERSONAS)
    rng.shuffle(personas)
    personas = personas[:4]

    workers: list[LoopWorker] = []
    for i, persona in enumerate(personas):
        workers.append(
            LoopWorker(
                worker_id=f"lw{i+1}:{persona.name}",
                persona=persona,
                cfg=cfg,
                prompts=prompts,
                openai=openai,
                cache_dir=cache_dir,
            )
        )

    # Novelty accounting (do not pass the full negative list to the model)
    seen_identifier_keys: set[str] = set()

    # Domain contribution stats (for debug report)
    domain_counts: dict[str, int] = {}

    consecutive_low_novelty = 0
    last_cycle_new = 0
    last_cycle_sample_new: list[str] = []

    try:
        for cycle_idx in range(cfg.max_rounds):
            summary = _compact_loop_summary(
                cycle_idx=cycle_idx,
                unique_identifier_count=len(seen_identifier_keys),
                last_cycle_new_identifiers=last_cycle_new,
                sample_new_identifiers=last_cycle_sample_new,
                top_domains=sorted(domain_counts.items(), key=lambda kv: kv[1], reverse=True)[:8],
            )

            logger.info("run=%s loop_cycle=%s starting loop workers=%s", run_id, cycle_idx, len(workers))
            outputs = await asyncio.gather(
                *[
                    w.run_cycle(
                        run_id=run_id,
                        user_query=user_query,
                        global_summary=summary,
                        cycle_idx=cycle_idx,
                    )
                    for w in workers
                ]
            )

            # Stage A: union candidates, compute novelty, persist raw identifiers (with provenance via mentions)
            new_this_cycle_global = 0
            sample_new: list[str] = []
            per_worker_new: dict[str, int] = {}

            for o in outputs:
                per_worker_new[o.worker_id] = 0
                for c in o.candidates:
                    key = safe_normalize(c.raw_identifier)
                    if not key:
                        continue

                    # Update domain stats (even if duplicate)
                    d = _domain(c.source_url)
                    if d:
                        domain_counts[d] = domain_counts.get(d, 0) + 1

                    is_new = key not in seen_identifier_keys
                    if is_new:
                        seen_identifier_keys.add(key)
                        new_this_cycle_global += 1
                        per_worker_new[o.worker_id] += 1
                        if len(sample_new) < 12:
                            sample_new.append(c.raw_identifier)

                    # Persist mention + candidate (one row per occurrence; candidate links to mention for provenance)
                    # NOTE: We intentionally do not dedupe inserts here; Stage A must retain raw strings with provenance.
                    m = Mention.from_raw(
                        mention_type=c.mention_type,
                        raw_text=c.raw_identifier,
                        context=c.context_snippet,
                        source_url=c.source_url,
                    )
                    mention_ids = await store.insert_mentions(run_id=run_id, query_id=None, result_id=None, mentions=[m])
                    source_mention_id = mention_ids[0] if mention_ids else None
                    cand = Candidate.from_mention(m)
                    await store.insert_candidate(run_id=run_id, candidate=cand, source_mention_id=source_mention_id)

                logger.info(
                    "run=%s loop_cycle=%s worker=%s success=%s new_ids=%s executed_queries=%s",
                    run_id,
                    cycle_idx,
                    o.worker_id,
                    o.success,
                    per_worker_new.get(o.worker_id, 0),
                    len(o.executed_queries),
                )

            await store.insert_metric(
                run_id=run_id,
                round_idx=cycle_idx,
                name="loop_cycle_summary",
                value={
                    "cycle_idx": cycle_idx,
                    "new_identifiers_global": new_this_cycle_global,
                    "new_identifiers_by_worker": per_worker_new,
                    "top_domains": sorted(domain_counts.items(), key=lambda kv: kv[1], reverse=True)[:20],
                },
            )

            last_cycle_new = new_this_cycle_global
            last_cycle_sample_new = sample_new[:12]

            # Early stopping: novelty stalls
            if new_this_cycle_global < int(cfg.min_new_identifiers_per_cycle):
                consecutive_low_novelty += 1
            else:
                consecutive_low_novelty = 0

            logger.info(
                "run=%s loop_cycle=%s new_ids=%s streak_low_novelty=%s",
                run_id,
                cycle_idx,
                new_this_cycle_global,
                consecutive_low_novelty,
            )

            if consecutive_low_novelty >= int(cfg.patience_cycles):
                break

        # Stage B verifier is implemented separately; placeholder here keeps the v1.2 entrypoint stable.
        summary = {
            "unique_identifiers_stage_a": len(seen_identifier_keys),
            "verified_assets_stage_b": 0,
        }
        await store.finish_run(run_id, status="completed", summary=summary)
        return {"run_id": run_id, "status": "completed", "summary": summary, "verified_assets": []}
    except Exception as e:
        await store.finish_run(run_id, status="failed", summary={"error": str(e)})
        raise
    finally:
        await openai.aclose()


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
        # Allow cfg.workers to be larger than the number of distinct profiles by cycling profiles.
        workers: list[Worker] = []
        for i in range(cfg.workers):
            profile = DEFAULT_PROFILES[i % len(DEFAULT_PROFILES)]
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

            # Validation concurrency controlled by config
            sem = asyncio.Semaphore(cfg.validation_concurrency)

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


