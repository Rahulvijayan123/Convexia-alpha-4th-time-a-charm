from __future__ import annotations

import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from drug_asset_discovery.config import EnvSettings, PromptBundle, RunConfig, load_config, load_prompts
from drug_asset_discovery.models.domain import Candidate, DraftAsset, EvidenceSourceType, Mention, ValidatedAsset
from drug_asset_discovery.openai_client import OpenAIResponsesClient
from drug_asset_discovery.retrieval.web_search import run_validation_web_search
from drug_asset_discovery.storage.supabase_store import SupabaseStore
from drug_asset_discovery.utils.hashing import safe_normalize, stable_sha256
from drug_asset_discovery.utils.identifiers import canonicalize_identifier
from drug_asset_discovery.utils.idempotency import idempotency_key
from drug_asset_discovery.utils.json_extract import JSONExtractError, extract_first_json
from drug_asset_discovery.utils.manifest import build_run_manifest
from drug_asset_discovery.validation.validator import draft_asset_from_minimal_evidence, is_evidence_anchored_v15
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
    last_round_identifier_yield: int,
) -> dict[str, Any]:
    # IMPORTANT: compact summary only. No giant negative lists.
    return {
        "round_idx": round_idx,
        "unique_mentions_count": unique_mentions_count,
        "pending_candidates_count": pending_candidates_count,
        "validated_assets_count": validated_assets_count,
        "validated_assets_sample": validated_assets_sample[:10],
        "last_round_new_validated": last_round_new_validated,
        "last_round_identifier_yield": last_round_identifier_yield,
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


def _guess_evidence_source_type(url: str) -> EvidenceSourceType:
    """
    Best-effort mapping from a source URL to the v1.5 evidence_source_type enum.
    Keep intentionally generic (do not benchmark-tune).
    """
    u = (url or "").strip().lower()
    if not u:
        return "other"
    if "clinicaltrials.gov" in u or "isrctn.com" in u or "eudract" in u or "who.int/trialsearch" in u:
        return "trial"
    if "/patent" in u or "patentscope" in u or "espacenet" in u or "lens.org" in u:
        return "patent"
    if u.endswith(".pdf") or "filetype=pdf" in u:
        return "pipeline_pdf"
    if "pubmed" in u or "ncbi.nlm.nih.gov" in u or "doi.org" in u:
        return "paper"
    if "press" in u or "news" in u or "prnewswire" in u or "businesswire" in u:
        return "press_release"
    if "catalog" in u or "product" in u or "vendor" in u:
        return "vendor"
    return "other"


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
    manifest = build_run_manifest(cfg=cfg, config_version=config_version, prompt_version=prompt_version)
    logger.info("run_manifest=%s", manifest)

    # Budget guardrails: keep tool calls bounded and predictable.
    if cfg.max_web_search_calls_per_worker_cycle < 1:
        raise ValueError("loop_mode requires max_web_search_calls_per_worker_cycle >= 1")
    if cfg.workers < 1:
        raise ValueError("loop_mode requires workers >= 1")
    if cfg.max_rounds < 1:
        raise ValueError("loop_mode requires max_rounds >= 1")

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
            "manifest": manifest,
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
    if cfg.workers <= len(personas):
        personas = personas[: int(cfg.workers)]
    else:
        # Cycle personas if user requests more workers than distinct personas.
        personas = [personas[i % len(personas)] for i in range(int(cfg.workers))]

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

    def _guess_mention_type(raw_identifier: str) -> str:
        s = (raw_identifier or "").strip()
        u = s.upper()
        if u.startswith("NCT") and len(u) == 11 and u[3:].isdigit():
            return "trial_id"
        if u.startswith("ISRCTN") and len(u) == 14 and u[6:].isdigit():
            return "trial_id"
        if u.startswith("ACTRN") and len(u) >= 19:
            return "trial_id"
        if u.startswith("CTRI/"):
            return "trial_id"
        if u.startswith(("WO", "US", "EP")) and any(ch.isdigit() for ch in u):
            return "patent_id"
        # Default: treat as a drug/program identifier token
        return "drug_code"

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
                    # v1.4: identifier novelty is tracked on canonical form (formatting variants collapse).
                    key = canonicalize_identifier(c.raw_identifier)
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
                        mention_type=_guess_mention_type(c.raw_identifier),
                        raw_text=c.raw_identifier,
                        context=c.context_snippet,
                        source_url=c.source_url,
                    )
                    mention_ids = await store.insert_mentions(run_id=run_id, query_id=None, result_id=None, mentions=[m])
                    source_mention_id = mention_ids[0] if mention_ids else None
                    # v1.4: only drug_code_like and patent_id_like become candidates (others stay as mentions).
                    if m.mention_class in ("drug_code_like", "patent_id_like"):
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
        return {
            "run_id": run_id,
            "status": "completed",
            "manifest": manifest,
            "summary": summary,
            "verified_assets": [],
        }
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
    - Multi-worker recall harvesting (mentions + candidates, no strict field gating)
    - v1.5: evidence-anchored discovery -> `draft_assets` (minimal acceptance, never dropped for missing fields)
    - v1.5: optional, budgeted enrichment -> promote to `final_assets` when completeness threshold is met
    - Persist everything to Supabase
    """
    env = EnvSettings()
    cfg: RunConfig = load_config(config_version)
    prompts: PromptBundle = load_prompts(prompt_version)
    manifest = build_run_manifest(cfg=cfg, config_version=config_version, prompt_version=prompt_version)
    logger.info("run_manifest=%s", manifest)

    if not env.supabase_url or not env.supabase_service_role_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

    store = SupabaseStore(url=env.supabase_url, service_role_key=env.supabase_service_role_key)
    run_id = await store.create_run(
        user_query=user_query,
        config_version=config_version,
        prompt_version=prompt_version,
        params={"workers": cfg.workers, "max_rounds": cfg.max_rounds, "manifest": manifest},
        idempotency_key=idempotency,
    )

    cache_dir = Path(env.local_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    openai = OpenAIResponsesClient(
        api_key=env.openai_api_key,
        base_url=env.openai_base_url,
        timeout_s=cfg.timeouts.openai_seconds,
    )

    workers: list[Worker] = []
    try:
        # Allow cfg.workers to be larger than the number of distinct profiles by cycling profiles.
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

        t0 = time.monotonic()
        deadline = t0 + float(cfg.max_run_seconds or 0)

        def _time_budget_exceeded() -> bool:
            if not cfg.max_run_seconds:
                return False
            return time.monotonic() >= deadline

        seen_mentions: set[str] = set()
        seen_candidates: set[str] = set()
        seen_final_assets: set[str] = set()  # fingerprint dedup
        seen_identifier_tokens: set[str] = set()
        domain_counts: dict[str, int] = {}

        # v1.5: draft assets (evidence-anchored discovery, not dropped for missing enrichment fields)
        draft_by_canon: dict[str, DraftAsset] = {}
        dirty_draft_canons: set[str] = set()
        canon_to_candidate_id: dict[str, str] = {}

        # v1.5: promoted final assets (only after enrichment threshold)
        promoted_canons: set[str] = set()
        final_assets: list[ValidatedAsset] = []
        enrichment_attempts = 0
        consecutive_no_new_drafts = 0

        def _merge_drafts(a: DraftAsset, b: DraftAsset) -> DraftAsset:
            """
            Merge two DraftAsset objects with the same canonical id.
            - Keep a.identifier_raw/evidence_* as the stable anchor (never overwrite)
            - Union identifier_aliases_raw and citations
            - Take max confidence_discovery
            """
            if a.identifier_canonical != b.identifier_canonical:
                return a

            aliases: list[str] = []
            seen_alias: set[str] = set()
            for x in list(a.identifier_aliases_raw or []) + list(b.identifier_aliases_raw or []):
                if not isinstance(x, str):
                    continue
                s = x.strip()
                if not s or s in seen_alias:
                    continue
                seen_alias.add(s)
                aliases.append(s)
            if a.identifier_raw and a.identifier_raw not in seen_alias:
                aliases.insert(0, a.identifier_raw)

            cites: list[dict[str, str]] = []
            seen_cite: set[str] = set()
            for c in list(a.model_dump().get("citations") or []) + list(b.model_dump().get("citations") or []):
                if not isinstance(c, dict):
                    continue
                u = c.get("url")
                s = c.get("snippet")
                if not (isinstance(u, str) and u.strip() and isinstance(s, str) and s.strip()):
                    continue
                key = stable_sha256(f"{u.strip()}|{s.strip()}")
                if key in seen_cite:
                    continue
                seen_cite.add(key)
                cites.append({"url": u.strip(), "snippet": s.strip()})

            return a.model_copy(
                update={
                    "identifier_aliases_raw": aliases,
                    "citations": cites,
                    "confidence_discovery": max(float(a.confidence_discovery or 0.0), float(b.confidence_discovery or 0.0)),
                }
            )

        last_round_identifier_yield_total = 0
        last_round_identifier_yield_by_worker: dict[str, int] = {}
        last_round_top_queries_by_yield: list[dict[str, Any]] = []
        last_round_dedup_feedback_by_worker: dict[str, str] = {}

        for round_idx in range(cfg.max_rounds):
            if _time_budget_exceeded():
                logger.info("run=%s stopping: time_budget_exceeded", run_id)
                break
            summary = _compact_summary(
                round_idx=round_idx,
                unique_mentions_count=len(seen_mentions),
                pending_candidates_count=0,
                validated_assets_count=len(promoted_canons),
                validated_assets_sample=[a.primary_identifier_raw for a in final_assets[-20:]],
                last_round_new_validated=0,
                last_round_identifier_yield=last_round_identifier_yield_total,
            )
            # v1.3: marginal gain hints (compact only; no negative lists)
            summary["last_round_identifier_yield_by_worker"] = last_round_identifier_yield_by_worker
            summary["last_round_top_queries_by_identifier_yield"] = last_round_top_queries_by_yield[:5]
            summary["top_domains"] = sorted(domain_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
            summary["dedup_feedback_by_worker"] = last_round_dedup_feedback_by_worker
            summary["draft_assets_count"] = len(draft_by_canon)
            summary["final_assets_count"] = len(promoted_canons)
            summary["time_elapsed_seconds"] = round(time.monotonic() - t0, 2)

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
            new_identifier_tokens_this_round = 0
            new_draft_assets_this_round = 0
            dirty_draft_canons.clear()
            identifier_yield_by_worker: dict[str, int] = {}
            query_yields: list[tuple[str, int]] = []

            for o in cycle_outputs:
                # Dedup mentions in code (no negative list passed to LLM)
                unique_mentions: list[Mention] = []
                for m in o.mentions:
                    if m.fingerprint in seen_mentions:
                        continue
                    seen_mentions.add(m.fingerprint)
                    unique_mentions.append(m)

                identifier_yield = 0
                identifier_sample: list[str] = []
                for m in unique_mentions:
                    # v1.4: only eligible identifier-like mentions count towards identifier yield.
                    if m.mention_class not in ("drug_code_like", "patent_id_like"):
                        continue
                    k = m.canonical_text
                    if not k:
                        continue
                    if k in seen_identifier_tokens:
                        continue
                    seen_identifier_tokens.add(k)
                    identifier_yield += 1
                    if len(identifier_sample) < 6:
                        identifier_sample.append(m.raw_text)
                    if m.source_url:
                        d = _domain(m.source_url)
                        if d:
                            domain_counts[d] = domain_counts.get(d, 0) + 1
                new_identifier_tokens_this_round += identifier_yield
                identifier_yield_by_worker[o.worker_id] = identifier_yield
                query_yields.append((o.query, identifier_yield))
                logger.info(
                    "run=%s round=%s worker=%s identifier_yield=%s query=%s",
                    run_id,
                    round_idx,
                    o.worker_id,
                    identifier_yield,
                    o.query,
                )

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
                        "identifier_yield": identifier_yield,
                        "identifier_sample": identifier_sample,
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

                if unique_mentions:
                    mention_ids = await store.insert_mentions(
                        run_id=run_id, query_id=qid, result_id=rid, mentions=unique_mentions
                    )
                    new_mentions_this_round += len(unique_mentions)

                # v1.5: create candidates (for provenance) + draft_assets (for evidence-anchored discovery).
                for idx, m in enumerate(unique_mentions):
                    if m.mention_class not in ("drug_code_like", "patent_id_like"):
                        continue

                    # Candidates are still stored for traceability and eval attribution.
                    c = Candidate.from_mention(m)
                    if c.fingerprint not in seen_candidates:
                        seen_candidates.add(c.fingerprint)
                        source_mention_id = None
                        if unique_mentions and idx < len(mention_ids):
                            source_mention_id = mention_ids[idx]
                        cid = await store.insert_candidate(run_id=run_id, candidate=c, source_mention_id=source_mention_id)
                        new_candidates_this_round += 1
                        if c.candidate_type == "drug_asset" and c.canonical_identifier:
                            canon_to_candidate_id.setdefault(c.canonical_identifier, cid)

                    # Draft assets are only for drug/program identifiers (not patent-id candidates).
                    if m.mention_class != "drug_code_like":
                        continue

                    ev_url = (m.source_url or (o.urls[0] if o.urls else "")).strip()
                    ev_snip = (m.context or m.raw_text or "").strip()
                    ev_type: EvidenceSourceType = _guess_evidence_source_type(ev_url)
                    draft = draft_asset_from_minimal_evidence(
                        identifier_candidate=m.raw_text,
                        evidence_url=ev_url,
                        evidence_snippet=ev_snip,
                        evidence_source_type=ev_type,
                        discovered_by_worker_id=o.worker_id,
                        discovered_by_cycle_id=cycle_id,
                    )
                    if not draft:
                        continue

                    canon = draft.identifier_canonical
                    existing = draft_by_canon.get(canon)
                    if existing is None:
                        draft_by_canon[canon] = draft
                        dirty_draft_canons.add(canon)
                        new_draft_assets_this_round += 1
                    else:
                        merged = _merge_drafts(existing, draft)
                        if merged.model_dump() != existing.model_dump():
                            draft_by_canon[canon] = merged
                            dirty_draft_canons.add(canon)

            logger.info(
                "run=%s round=%s harvested new_mentions=%s new_candidates=%s new_identifier_tokens=%s new_draft_assets=%s dirty_drafts=%s",
                run_id,
                round_idx,
                new_mentions_this_round,
                new_candidates_this_round,
                new_identifier_tokens_this_round,
                new_draft_assets_this_round,
                len(dirty_draft_canons),
            )

            # Persist draft_assets discovered/updated this round (dedup by canonical, preserve aliases/citations).
            if dirty_draft_canons and not _time_budget_exceeded():
                await asyncio.gather(
                    *[
                        store.upsert_draft_asset(run_id=run_id, draft=draft_by_canon[c].model_dump())
                        for c in sorted(dirty_draft_canons)
                        if c in draft_by_canon
                    ]
                )
            persisted_drafts = len(dirty_draft_canons)
            dirty_draft_canons.clear()

            await store.insert_metric(
                run_id=run_id,
                round_idx=round_idx,
                name="round_summary",
                value={
                    "successful_workers": successful_workers,
                    "new_mentions": new_mentions_this_round,
                    "new_candidates": new_candidates_this_round,
                    "new_identifier_tokens": new_identifier_tokens_this_round,
                    "new_draft_assets": new_draft_assets_this_round,
                    "draft_assets_total": len(draft_by_canon),
                    "final_assets_total": len(promoted_canons),
                    "persisted_draft_assets": persisted_drafts,
                    "time_elapsed_seconds": round(time.monotonic() - t0, 2),
                },
            )
            last_round_identifier_yield_total = new_identifier_tokens_this_round
            last_round_identifier_yield_by_worker = dict(identifier_yield_by_worker)
            last_round_top_queries_by_yield = [
                {"query": q, "identifier_yield": y}
                for q, y in sorted(query_yields, key=lambda t: t[1], reverse=True)[:10]
            ]
            last_round_dedup_feedback_by_worker = {
                wid: "duplicate, continue, prioritize new identifiers"
                for wid, y in identifier_yield_by_worker.items()
                if int(y) == 0
            }

            # Stopping condition: multiple successive high-quality rounds produce zero new draft assets
            high_quality = successful_workers >= cfg.min_successful_workers_per_round
            if high_quality and new_draft_assets_this_round == 0:
                consecutive_no_new_drafts += 1
            elif new_draft_assets_this_round > 0:
                consecutive_no_new_drafts = 0

            logger.info(
                "run=%s round=%s new_draft_assets=%s streak_no_new_drafts=%s high_quality=%s",
                run_id,
                round_idx,
                new_draft_assets_this_round,
                consecutive_no_new_drafts,
                high_quality,
            )

            if consecutive_no_new_drafts >= cfg.stop_after_no_new_validated_rounds:
                break

        # ------------------------------------------------------------------
        # v1.5 enrichment stage (optional, budgeted)
        # ------------------------------------------------------------------
        if draft_by_canon and int(cfg.max_enrichment_validations or 0) > 0 and not _time_budget_exceeded():
            # Only enrich drafts that are below the promotion threshold.
            enrichable: list[DraftAsset] = [
                d for d in draft_by_canon.values() if int(d.completeness_score()) < 4 and d.enrichment_status != "complete"
            ]
            enrichable.sort(key=lambda d: float(d.confidence_discovery or 0.0), reverse=True)
            remaining = max(0, int(cfg.max_enrichment_validations) - int(enrichment_attempts))
            enrichable = enrichable[:remaining]

            logger.info(
                "run=%s enrichment_start candidates=%s remaining_budget=%s",
                run_id,
                len(enrichable),
                remaining,
            )

            # Keep concurrency modest; each enrichment uses web_search.
            sem_enrich = asyncio.Semaphore(max(1, min(int(cfg.workers or 4), 8)))

            def _coerce_str(v: Any) -> str | None:
                return v.strip() if isinstance(v, str) and v.strip() else None

            def _coerce_float_01(v: Any) -> float | None:
                try:
                    x = float(v)
                except Exception:
                    return None
                if x < 0.0:
                    x = 0.0
                if x > 1.0:
                    x = 1.0
                return float(x)

            def _coerce_citations(v: Any) -> list[dict[str, str]]:
                if not isinstance(v, list):
                    return []
                out: list[dict[str, str]] = []
                for item in v:
                    if not isinstance(item, dict):
                        continue
                    u = _coerce_str(item.get("url") or item.get("source_url"))
                    s = _coerce_str(item.get("snippet"))
                    if u and s:
                        out.append({"url": u, "snippet": s})
                return out

            def _merge_citations(existing: DraftAsset, incoming: list[dict[str, str]]) -> list[dict[str, str]]:
                merged: list[dict[str, str]] = []
                seen: set[str] = set()
                for c in list(existing.model_dump().get("citations") or []) + list(incoming or []):
                    if not isinstance(c, dict):
                        continue
                    u = _coerce_str(c.get("url"))
                    s = _coerce_str(c.get("snippet"))
                    if not u or not s:
                        continue
                    k = stable_sha256(f"{u}|{s}")
                    if k in seen:
                        continue
                    seen.add(k)
                    merged.append({"url": u, "snippet": s})
                return merged

            allowed_ev_types: set[str] = {"patent", "trial", "pipeline_pdf", "paper", "vendor", "press_release", "other"}

            async def _enrich_one(d: DraftAsset) -> None:
                nonlocal enrichment_attempts
                if _time_budget_exceeded():
                    return
                async with sem_enrich:
                    if _time_budget_exceeded():
                        return
                    enrichment_attempts += 1

                    builder = prompts.draft_asset_builder or prompts.validator
                    prompt = (
                        builder
                        + "\n\n"
                        + "Use web search as needed. Return ONLY JSON.\n"
                        + "Never overwrite identifier_raw.\n"
                        + "Keep evidence anchoring: provide evidence_snippet that contains identifier_raw (or canonical-equivalent).\n\n"
                        + f'identifier_candidate: "{d.identifier_raw}"\n'
                        + f'evidence_url: "{d.evidence_url}"\n'
                        + f'evidence_snippet: "{d.evidence_snippet}"\n'
                    )

                    resp = await run_validation_web_search(
                        openai=openai,
                        cfg=cfg,
                        cache_dir=cache_dir,
                        prompt=prompt,
                        idempotency_key=idempotency_key(
                            "enrich.draft_asset",
                            {
                                "run_id": run_id,
                                "identifier_canonical": d.identifier_canonical,
                                "attempt": enrichment_attempts,
                            },
                        ),
                    )

                    try:
                        obj = extract_first_json(resp.output_text)
                    except JSONExtractError:
                        # Mark failed attempt (do not drop)
                        failed = d.model_copy(update={"enrichment_status": "failed"})
                        draft_by_canon[d.identifier_canonical] = failed
                        await store.upsert_draft_asset(run_id=run_id, draft=failed.model_dump())
                        return

                    if not isinstance(obj, dict):
                        failed = d.model_copy(update={"enrichment_status": "failed"})
                        draft_by_canon[d.identifier_canonical] = failed
                        await store.upsert_draft_asset(run_id=run_id, draft=failed.model_dump())
                        return

                    out_ev_url = _coerce_str(obj.get("evidence_url")) or ""
                    out_ev_snip = _coerce_str(obj.get("evidence_snippet")) or ""
                    out_ev_type_raw = _coerce_str(obj.get("evidence_source_type")) or ""
                    out_ev_type: EvidenceSourceType = (
                        out_ev_type_raw if out_ev_type_raw in allowed_ev_types else _guess_evidence_source_type(out_ev_url)
                    )  # type: ignore[assignment]

                    out_conf = _coerce_float_01(obj.get("confidence_discovery"))

                    # Evidence anchoring for enrichment outputs (must mention the identifier, or canonical-equivalent).
                    anchored = bool(out_ev_url and out_ev_snip and is_evidence_anchored_v15(identifier_raw=d.identifier_raw, evidence_snippet=out_ev_snip))

                    out_citations = _coerce_citations(obj.get("citations"))
                    # Always include the model's evidence_url/snippet as a citation if present.
                    if out_ev_url and out_ev_snip:
                        out_citations.insert(0, {"url": out_ev_url, "snippet": out_ev_snip})

                    # Filter citations to anchored ones (keep conservative).
                    anchored_citations = [
                        c for c in out_citations if is_evidence_anchored_v15(identifier_raw=d.identifier_raw, evidence_snippet=str(c.get("snippet") or ""))
                    ]

                    if not anchored and not anchored_citations:
                        failed = d.model_copy(update={"enrichment_status": "failed"})
                        draft_by_canon[d.identifier_canonical] = failed
                        await store.upsert_draft_asset(run_id=run_id, draft=failed.model_dump())
                        return

                    # Fill only missing fields; never overwrite existing non-empty.
                    updates: dict[str, Any] = {}
                    for k in ("sponsor", "target", "modality", "indication", "stage", "geography"):
                        cur = getattr(d, k)
                        cur_ok = isinstance(cur, str) and cur.strip()
                        inc = _coerce_str(obj.get(k))
                        if not cur_ok and inc:
                            updates[k] = inc

                    # Preserve identifier_raw; merge any raw variant output as an alias if canonical matches.
                    out_id_raw = _coerce_str(obj.get("identifier_raw"))
                    out_id_canon = canonicalize_identifier(out_id_raw or "")
                    aliases = list(d.identifier_aliases_raw or [])
                    if out_id_raw and out_id_canon and out_id_canon == d.identifier_canonical and out_id_raw not in aliases:
                        aliases.append(out_id_raw)
                    if d.identifier_raw and d.identifier_raw not in aliases:
                        aliases.insert(0, d.identifier_raw)
                    updates["identifier_aliases_raw"] = aliases

                    # Merge citations
                    merged_cites = _merge_citations(d, anchored_citations or out_citations)
                    updates["citations"] = merged_cites

                    # Update confidence and status
                    updates["confidence_discovery"] = max(float(d.confidence_discovery or 0.0), float(out_conf or 0.0))
                    updates["evidence_source_type"] = d.evidence_source_type or out_ev_type

                    next_d = DraftAsset.model_validate({**d.model_dump(), **updates})
                    score = int(next_d.completeness_score())
                    if score >= 4:
                        next_d = next_d.model_copy(update={"enrichment_status": "complete"})
                    elif score > 0:
                        next_d = next_d.model_copy(update={"enrichment_status": "partial"})
                    else:
                        next_d = next_d.model_copy(update={"enrichment_status": "failed"})

                    draft_by_canon[next_d.identifier_canonical] = next_d
                    await store.upsert_draft_asset(run_id=run_id, draft=next_d.model_dump())

                    # Promotion rule: completeness >= 4 AND evidence anchoring holds (on the stored anchor evidence).
                    if (
                        int(next_d.completeness_score()) >= 4
                        and next_d.identifier_canonical not in promoted_canons
                        and is_evidence_anchored_v15(identifier_raw=next_d.identifier_raw, evidence_snippet=next_d.evidence_snippet)
                    ):
                        sources: list[str] = []
                        if next_d.evidence_url:
                            sources.append(next_d.evidence_url)
                        for c in next_d.model_dump().get("citations") or []:
                            if isinstance(c, dict):
                                u = _coerce_str(c.get("url"))
                                if u and u not in sources:
                                    sources.append(u)

                        fa = ValidatedAsset.from_fields(
                            primary_identifier_raw=next_d.identifier_raw,
                            identifier_aliases_raw=list(next_d.identifier_aliases_raw or []),
                            evidence_snippet=next_d.evidence_snippet,
                            evidence_url=next_d.evidence_url,
                            evidence_source_type=next_d.evidence_source_type,
                            sponsor=next_d.sponsor,
                            target=next_d.target,
                            modality=next_d.modality,
                            indication=next_d.indication,
                            development_stage=next_d.stage,
                            geography=next_d.geography,
                            sources=sources,
                        )
                        candidate_id = canon_to_candidate_id.get(next_d.identifier_canonical)
                        await store.insert_final_assets(run_id=run_id, candidate_id=candidate_id, assets=[fa])
                        if candidate_id:
                            await store.update_candidate_status(candidate_id, "validated")
                        if fa.fingerprint not in seen_final_assets:
                            seen_final_assets.add(fa.fingerprint)
                            final_assets.append(fa)
                        promoted_canons.add(next_d.identifier_canonical)

            await asyncio.gather(*[_enrich_one(d) for d in enrichable])

        stopped_reason = "time_budget_exceeded" if _time_budget_exceeded() else None
        summary = {
            "unique_mentions": len(seen_mentions),
            "unique_candidates": len(seen_candidates),
            "draft_assets": len(draft_by_canon),
            "final_assets": len(promoted_canons),
            "enrichment_attempts": int(enrichment_attempts),
            "time_elapsed_seconds": round(time.monotonic() - t0, 2),
            "stopped_reason": stopped_reason,
        }
        await store.finish_run(run_id, status="completed", summary=summary)

        return {
            "run_id": run_id,
            "status": "completed",
            "manifest": manifest,
            "summary": summary,
            "assets": [a.model_dump() for a in final_assets],
        }
    except Exception as e:
        await store.finish_run(run_id, status="failed", summary={"error": str(e)})
        raise
    finally:
        # Close any optional HTTP clients (e.g., fetchers) held by workers.
        if workers:
            await asyncio.gather(*[w.aclose() for w in workers], return_exceptions=True)
        await openai.aclose()


