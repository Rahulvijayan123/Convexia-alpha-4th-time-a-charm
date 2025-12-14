from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from drug_asset_discovery.config import EnvSettings, PromptBundle, RunConfig, load_config, load_prompts
from drug_asset_discovery.models.domain import Candidate, DraftAsset, EvidenceSourceType, Mention, ValidatedAsset
from drug_asset_discovery.openai_client import OpenAIResponsesClient
from drug_asset_discovery.retrieval.web_search import run_validation_web_search
from drug_asset_discovery.storage.supabase_store import SupabaseStore
from drug_asset_discovery.utils.hashing import safe_normalize, stable_sha256
from drug_asset_discovery.utils.identifiers import canonicalize_identifier, classify_identifier_type
from drug_asset_discovery.utils.idempotency import idempotency_key
from drug_asset_discovery.utils.json_extract import JSONExtractError, extract_first_json
from drug_asset_discovery.utils.manifest import build_run_manifest
from drug_asset_discovery.validation.validator import draft_asset_from_minimal_evidence, is_evidence_anchored_v15
from drug_asset_discovery.validation.match_scoring import score_match_to_query
from drug_asset_discovery.validation.query_spec import derive_query_spec
from drug_asset_discovery.worker.worker import DEFAULT_PROFILES, DEFAULT_V16_PROFILES, Worker

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
        return "registry"
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
    if not env.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

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
    on_run_id: Callable[[str], None] | None = None,
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
    if not env.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    store = SupabaseStore(url=env.supabase_url, service_role_key=env.supabase_service_role_key)
    query_spec = derive_query_spec(user_query)
    run_id = await store.create_run(
        user_query=user_query,
        config_version=config_version,
        prompt_version=prompt_version,
        params={
            "workers": cfg.workers,
            "max_rounds": cfg.max_rounds,
            "manifest": manifest,
            "query_spec": query_spec.model_dump(),
            "finalization_overall_match_threshold": float(cfg.finalization_overall_match_threshold),
        },
        idempotency_key=idempotency,
    )
    try:
        if on_run_id:
            on_run_id(run_id)
    except Exception:
        # Never allow UI hooks to break runs
        pass

    cache_dir = Path(env.local_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    openai = OpenAIResponsesClient(
        api_key=env.openai_api_key,
        base_url=env.openai_base_url,
        timeout_s=cfg.timeouts.openai_seconds,
    )

    workers: list[Worker] = []
    try:
        # v1.6: enforce diversity across 4 specialized worker tabs.
        profiles = DEFAULT_V16_PROFILES if str(cfg.version or "").startswith("v1.6") else DEFAULT_PROFILES
        # Allow cfg.workers to be larger than the number of distinct profiles by cycling profiles.
        for i in range(cfg.workers):
            profile = profiles[i % len(profiles)]
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
        seen_query_fps: set[str] = set()
        unique_queries: list[str] = []

        # v1.5: draft assets (evidence-anchored discovery, not dropped for missing enrichment fields)
        draft_by_canon: dict[str, DraftAsset] = {}
        dirty_draft_canons: set[str] = set()
        canon_to_candidate_id: dict[str, str] = {}

        # v1.5: promoted final assets (only after enrichment threshold)
        promoted_canons: set[str] = set()
        final_assets: list[ValidatedAsset] = []
        enrichment_attempts = 0
        consecutive_no_new_drafts = 0

        # v1.5 mandatory identifier pivot queue (runs before broad exploration continues)
        pivot_queue: deque[str] = deque()
        pivot_enqueued_canons: set[str] = set()

        new_draft_assets_per_cycle: list[int] = []
        new_final_assets_per_cycle: list[int] = []

        # v1.6 diagnostics: per-cycle worker gains + query-template yields.
        worker_marginal_gains_per_cycle: list[dict[str, Any]] = []
        query_template_yields: dict[str, int] = {}
        query_template_examples: dict[str, list[str]] = {}

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
                    # Preserve any already-computed typing/scoring fields when present.
                    "identifier_type": a.identifier_type or b.identifier_type,
                }
            )

        def _register_unique_query(q: str) -> bool:
            qq = " ".join((q or "").strip().split())
            if not qq:
                return False
            fp = _query_fingerprint(qq)
            if fp in seen_query_fps:
                return False
            seen_query_fps.add(fp)
            unique_queries.append(qq)
            return True

        _QUOTED_TOKEN_RE = re.compile(r'"[^"]+"')
        _NCT_RE = re.compile(r"\bNCT\d{8}\b", flags=re.IGNORECASE)
        _PATENT_RE = re.compile(r"\b(?:WO|US|EP)\d{4,}\b", flags=re.IGNORECASE)

        def _query_template(q: str) -> str:
            """
            Best-effort template extraction for yield aggregation.
            Keep it simple and benchmark-blind.
            """
            t = " ".join((q or "").strip().split())
            if not t:
                return ""
            t = _QUOTED_TOKEN_RE.sub('"{identifier}"', t)
            t = _NCT_RE.sub("{trial_id}", t)
            t = _PATENT_RE.sub("{patent_id}", t)
            return t

        def _enqueue_identifier_pivots(*, identifier_raw: str, identifier_canonical: str) -> None:
            """
            v1.5 mandatory pivoting:
            Whenever a NEW drug_code enters draft_assets, enqueue follow-up queries:
              - "{identifier} target"
              - "{identifier} Phase 1"
              - "{identifier} patent"
              - "{identifier} company pipeline pdf"
            """
            # v1.6: query generation is fully iterative and worker-driven; do not maintain a static pivot queue.
            if str(cfg.version or "").startswith("v1.6"):
                return
            canon = (identifier_canonical or "").strip()
            raw = (identifier_raw or "").strip()
            if not canon or not raw:
                return
            if canon in pivot_enqueued_canons:
                return
            pivot_enqueued_canons.add(canon)

            # Keep short and anchor on the exact token.
            base = f'"{raw}"'
            pivot_queue.append(f"{base} target")
            pivot_queue.append(f"{base} Phase 1")
            pivot_queue.append(f"{base} patent")
            pivot_queue.append(f"{base} company pipeline pdf")

        def _broad_queries_for_cycle(*, broad_cycle_idx: int) -> list[str]:
            """
            v1.5 tab-variance policy:
            - Use short templates (no giant OR queries)
            - Include multilingual variants for Japanese, Chinese, Korean, and European languages
            - Keep them as separate worker queries
            """
            # Primary anchor term for templates.
            target = query_spec.target_terms[0] if query_spec.target_terms else user_query
            target = " ".join((target or "").strip().split())
            patent_prefix = ("WO", "EP", "US")[int(broad_cycle_idx) % 3]
            eu_lang = "fr" if int(broad_cycle_idx) % 2 == 0 else "de"

            # Always include multilingual workers first (JP/CN/KR/EU).
            # Rotate templates across cycles to avoid repeating identical queries.
            if int(broad_cycle_idx) % 2 == 0:
                jp = f'{target} 臨床試験 NCT'
                cn = f'{target} PROTAC 蛋白降解 专利 {patent_prefix}'
                kr = f'{target} 첫 환자 투여 1상 보도자료'
            else:
                jp = f'{target} 阻害剤 パイプライン PDF filetype:pdf'
                cn = f'{target} 抑制剂 产品管线 PDF filetype:pdf'
                kr = f'{target} 임상시험 NCT'

            if eu_lang == "fr":
                eu = f'{target} communiqué de presse Phase 1 premier patient'
            else:
                eu = f'{target} Hemmer Pipeline PDF filetype:pdf'

            # English template workers (added if worker budget allows).
            en_pipeline = (
                f'{target} inhibitor pipeline PDF filetype:pdf'
                if int(broad_cycle_idx) % 2 == 0
                else f'{target} company pipeline pdf'
            )
            en_ic50 = (
                f'{target} compound IC50 PDF filetype:pdf'
                if int(broad_cycle_idx) % 2 == 0
                else f'{target} compound IC50 pdf'
            )

            # Respect the v1.5 run policy: 4–6 workers per cycle.
            base = [jp, cn, kr, eu]
            if int(cfg.workers) >= 5:
                base.append(en_pipeline)
            if int(cfg.workers) >= 6:
                base.append(en_ic50)
            return base[: int(cfg.workers)]

        def _score_and_label_draft(*, d: DraftAsset, cycle_idx: int) -> tuple[DraftAsset, bool]:
            """
            Apply v1.5 late filtering fields onto a DraftAsset and decide promotion eligibility.

            Eligibility:
            - evidence anchored (v1.5 rule)
            - identifier_type == drug_code
            - overall_match_score >= cfg.finalization_overall_match_threshold
            """
            id_type = d.identifier_type or classify_identifier_type(d.identifier_raw)
            scores = score_match_to_query(d, query_spec)
            thr = float(cfg.finalization_overall_match_threshold)
            min_dim = float(getattr(cfg, "validated_min_dimension_score", 0.6) or 0.6)

            required_fields_cfg = [
                str(x).strip()
                for x in (getattr(cfg, "validated_required_fields", None) or [])
                if isinstance(x, str) and x.strip()
            ]

            anchored = is_evidence_anchored_v15(identifier_raw=d.identifier_raw, evidence_snippet=d.evidence_snippet)

            eligible = False
            reason: str | None = None

            req = scores.get("required") if isinstance(scores.get("required"), dict) else {}
            t_req = bool(req.get("target")) if isinstance(req, dict) else False
            m_req = bool(req.get("modality")) if isinstance(req, dict) else False
            i_req = bool(req.get("indication")) if isinstance(req, dict) else False

            required_fields: list[str] = sorted(
                set(required_fields_cfg)
                | ({"target"} if t_req else set())
                | ({"modality"} if m_req else set())
                | ({"indication"} if i_req else set())
            )
            missing_required_fields: list[str] = []
            for k in required_fields:
                v = getattr(d, k, None)
                if not (isinstance(v, str) and v.strip()):
                    missing_required_fields.append(k)

            if not anchored:
                reason = "incomplete_evidence_anchor"
            elif id_type != "drug_code":
                reason = "identifier_type_not_drug_code"
            else:
                overall = float(scores.get("overall_match_score") or 0.0)
                t_score = float(scores.get("target_match_score") or 0.0)
                m_score = float(scores.get("modality_match_score") or 0.0)
                i_score = float(scores.get("indication_match_score") or 0.0)

                dims_ok = (
                    (not t_req or t_score >= min_dim)
                    and (not m_req or m_score >= min_dim)
                    and (not i_req or i_score >= min_dim)
                )
                fields_ok = not missing_required_fields

                eligible = bool(overall >= thr and dims_ok and fields_ok)
                if eligible:
                    reason = None
                else:
                    # Provide a specific, attributable reason when possible.
                    if missing_required_fields:
                        reason = "missing_required_fields"
                    elif t_req and t_score < min_dim:
                        reason = "insufficient_target_evidence"
                    elif m_req and m_score < min_dim:
                        reason = "insufficient_modality_evidence"
                    elif i_req and i_score < min_dim:
                        reason = "insufficient_indication_evidence"
                    else:
                        reason = "other"

            updated = d.model_copy(
                update={
                    "identifier_type": id_type,
                    "match_scores": scores,
                    "rejection_reason": reason,
                    "extracted_context": {
                        "scored_in_cycle": int(cycle_idx),
                        "finalization_overall_match_threshold": thr,
                        "validated_min_dimension_score": float(min_dim),
                        "validated_required_fields": list(required_fields),
                        "missing_required_fields": list(missing_required_fields),
                        "promotion_eligible": bool(eligible),
                    },
                }
            )
            return updated, eligible

        # ------------------------------------------------------------------
        # v1.6: enrichment runs DURING the loop (plus optional final sweep).
        # ------------------------------------------------------------------

        # Config-driven guarantee: do at least N enrichment attempts when any draft assets exist.
        min_enrichment_attempts_per_run = int(getattr(cfg, "min_enrichment_attempts_per_run", 0) or 0)
        if min_enrichment_attempts_per_run > 0 and int(cfg.max_enrichment_validations or 0) < min_enrichment_attempts_per_run:
            raise ValueError(
                "Config invalid: max_enrichment_validations must be >= min_enrichment_attempts_per_run "
                f"(got max_enrichment_validations={int(cfg.max_enrichment_validations or 0)} "
                f"min_enrichment_attempts_per_run={min_enrichment_attempts_per_run})"
            )

        enrichment_query_cursor = 0

        def _pick_enrichment_query(d: DraftAsset) -> str:
            """
            Enrichment micro-query policy (short, candidate-specific; no mega OR queries).
            Rotate queries to encourage source-type diversity.
            """
            nonlocal enrichment_query_cursor
            raw = str(d.identifier_raw or "").strip()
            if not raw:
                return ""
            base = f'"{raw}"'

            missing_target = not (isinstance(d.target, str) and d.target.strip())
            missing_modality = not (isinstance(d.modality, str) and d.modality.strip())
            missing_stage = not (isinstance(d.stage, str) and d.stage.strip())

            templates: list[str] = []
            if missing_target:
                templates.append(f"{base} target")
            if missing_stage:
                templates.append(f"{base} phase 1")
            if missing_modality:
                # Keep the query short and concrete; this helps classification without long OR clauses.
                templates.append(f"{base} PROTAC")

            # Always include evidence-source pivots (patents, pipelines/PDFs).
            templates.append(f"{base} patent")
            templates.append(f"{base} pipeline pdf")

            # Deterministic rotation.
            templates = [t for t in templates if t.strip()]
            q = templates[enrichment_query_cursor % len(templates)] if templates else f"{base} target"
            enrichment_query_cursor += 1
            return q

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

        allowed_ev_types: set[str] = {"patent", "trial", "pipeline_pdf", "paper", "vendor", "press_release", "other", "registry"}

        async def _enrich_some_drafts(
            *,
            enrichment_cycle_idx: int,
            max_to_enrich: int,
            force_at_least_one: bool,
        ) -> tuple[int, int]:
            """
            Enrich up to `max_to_enrich` draft assets, using short identifier-anchored micro-queries.
            Returns: (attempts_used, new_final_assets_promoted)
            """
            nonlocal enrichment_attempts

            if not draft_by_canon:
                return (0, 0)
            if int(cfg.max_enrichment_validations or 0) <= 0:
                return (0, 0)
            # If we already exceeded the time budget, still allow a single forced attempt
            # to satisfy the nonzero enrichment guarantee (when configured/enforced).
            if _time_budget_exceeded() and not (force_at_least_one and int(enrichment_attempts) == 0):
                return (0, 0)

            remaining_budget = max(0, int(cfg.max_enrichment_validations) - int(enrichment_attempts))
            if remaining_budget <= 0:
                return (0, 0)

            limit = max(0, min(int(max_to_enrich), int(remaining_budget)))
            if limit <= 0:
                return (0, 0)

            # Prefer incomplete, not-yet-promoted drug-code drafts.
            thr = float(cfg.finalization_overall_match_threshold)
            enrichable: list[DraftAsset] = []
            for d in draft_by_canon.values():
                id_type = d.identifier_type or classify_identifier_type(d.identifier_raw)
                if id_type != "drug_code":
                    continue
                if d.identifier_canonical in promoted_canons:
                    continue
                if d.enrichment_status == "complete":
                    continue
                ms = d.match_scores if isinstance(d.match_scores, dict) else {}
                overall = float(ms.get("overall_match_score") or 0.0)
                # Only enrich if still below promotion threshold OR still missing fields.
                if overall >= thr and int(d.completeness_score()) >= 4:
                    continue
                enrichable.append(d.model_copy(update={"identifier_type": id_type}))

            enrichable.sort(key=lambda d: float(d.confidence_discovery or 0.0), reverse=True)
            enrichable = enrichable[:limit]

            if force_at_least_one and not enrichable:
                # As a guarantee fallback, enrich the highest-confidence draft even if already "complete" or promoted.
                any_drafts = sorted(
                    list(draft_by_canon.values()),
                    key=lambda d: float(d.confidence_discovery or 0.0),
                    reverse=True,
                )
                if any_drafts:
                    enrichable = [any_drafts[0]]
                    limit = 1

            if not enrichable:
                return (0, 0)

            attempts_before = int(enrichment_attempts)
            enrichment_new_final_assets = 0
            promote_lock = asyncio.Lock()
            sem_enrich = asyncio.Semaphore(max(1, min(int(cfg.workers or 4), 8)))

            async def _enrich_one(d: DraftAsset) -> None:
                nonlocal enrichment_attempts, enrichment_new_final_assets
                if _time_budget_exceeded():
                    return
                async with sem_enrich:
                    if _time_budget_exceeded():
                        return
                    enrichment_attempts += 1
                    attempt_no = int(enrichment_attempts)

                    builder = prompts.draft_asset_builder or prompts.validator
                    q = _pick_enrichment_query(d)
                    prompt = (
                        builder
                        + "\n\n"
                        + "Use web search.\n"
                        + "STRICT: execute exactly ONE web_search tool call using THIS exact query (no mega OR queries):\n"
                        + f"{q}\n\n"
                        + "Return ONLY JSON.\n"
                        + "Never overwrite identifier_raw.\n"
                        + "Keep evidence anchoring: provide evidence_snippet that contains identifier_raw (or canonical-equivalent).\n\n"
                        + f'identifier_candidate: \"{d.identifier_raw}\"\\n'
                        + f'evidence_url: \"{d.evidence_url}\"\\n'
                        + f'evidence_snippet: \"{d.evidence_snippet}\"\\n'
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
                                "cycle_idx": int(enrichment_cycle_idx),
                                "attempt": attempt_no,
                                "q_fp": stable_sha256(safe_normalize(q)),
                            },
                        ),
                    )

                    try:
                        obj = extract_first_json(resp.output_text)
                    except JSONExtractError:
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

                    anchored = bool(
                        out_ev_url
                        and out_ev_snip
                        and is_evidence_anchored_v15(identifier_raw=d.identifier_raw, evidence_snippet=out_ev_snip)
                    )

                    out_citations = _coerce_citations(obj.get("citations"))
                    if out_ev_url and out_ev_snip:
                        out_citations.insert(0, {"url": out_ev_url, "snippet": out_ev_snip})

                    anchored_citations = [
                        c
                        for c in out_citations
                        if is_evidence_anchored_v15(identifier_raw=d.identifier_raw, evidence_snippet=str(c.get("snippet") or ""))
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

                    # Apply v1.5 gating fields (identifier typing + evidence-based scoring).
                    scored, eligible = _score_and_label_draft(d=next_d, cycle_idx=enrichment_cycle_idx)
                    next_d = scored

                    draft_by_canon[next_d.identifier_canonical] = next_d
                    await store.upsert_draft_asset(run_id=run_id, draft=next_d.model_dump())

                    if eligible:
                        async with promote_lock:
                            if next_d.identifier_canonical in promoted_canons:
                                return
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
                            enrichment_new_final_assets += 1

            await asyncio.gather(*[_enrich_one(d) for d in enrichable[:limit]])

            attempts_used = int(enrichment_attempts) - attempts_before
            return (attempts_used, int(enrichment_new_final_assets))

        last_round_identifier_yield_total = 0
        last_round_identifier_yield_by_worker: dict[str, int] = {}
        last_round_top_queries_by_yield: list[dict[str, Any]] = []
        last_round_dedup_feedback_by_worker: dict[str, str] = {}
        worker_flat_steps: dict[str, int] = {}
        last_round_new_identifiers_sample: list[str] = []
        source_type_counts: dict[str, int] = {}

        cycle_idx = 0
        broad_cycle_idx = 0
        while True:
            if _time_budget_exceeded():
                logger.info("run=%s stopping: time_budget_exceeded", run_id)
                break

            is_v16 = str(cfg.version or "").startswith("v1.6")
            is_pivot_cycle = (not is_v16) and bool(pivot_queue)
            policy_phase = "loop" if is_v16 else ("pivot" if is_pivot_cycle else "broad")

            # v1.6: cycles are purely iterative; stop after max_rounds.
            if is_v16:
                if cycle_idx >= int(cfg.max_rounds):
                    break
            else:
                if (not is_pivot_cycle) and broad_cycle_idx >= int(cfg.max_rounds):
                    break

            # Decide forced queries for this cycle, enforcing uniqueness across workers + cycles.
            forced_by_worker: dict[str, str] = {}
            if not is_v16:
                if is_pivot_cycle:
                    for w in workers:
                        q = ""
                        while pivot_queue and not q:
                            cand = pivot_queue.popleft()
                            if _register_unique_query(cand):
                                q = cand
                        if not q:
                            # If pivots drain early, fall back to a broad template query (rare).
                            fallback = _broad_queries_for_cycle(broad_cycle_idx=broad_cycle_idx)
                            q = fallback[0] if fallback else user_query
                            _register_unique_query(q)
                        forced_by_worker[w.worker_id] = q
                else:
                    broad_queries = _broad_queries_for_cycle(broad_cycle_idx=broad_cycle_idx)
                    # Ensure length matches worker count.
                    if len(broad_queries) < len(workers):
                        broad_queries = broad_queries + [user_query] * (len(workers) - len(broad_queries))
                    for w, q in zip(workers, broad_queries):
                        qq = q
                        if not _register_unique_query(qq):
                            # Tiny salt to avoid repeats while keeping the template short.
                            qq = f"{qq} {broad_cycle_idx}"
                            _register_unique_query(qq)
                        forced_by_worker[w.worker_id] = qq

            summary = _compact_summary(
                round_idx=cycle_idx,
                unique_mentions_count=len(seen_mentions),
                pending_candidates_count=0,
                validated_assets_count=len(promoted_canons),
                validated_assets_sample=[a.primary_identifier_raw for a in final_assets[-20:]],
                last_round_new_validated=0,
                last_round_identifier_yield=last_round_identifier_yield_total,
            )
            summary["policy_phase"] = policy_phase
            summary["unique_queries"] = len(unique_queries)
            summary["last_round_identifier_yield_by_worker"] = last_round_identifier_yield_by_worker
            summary["last_round_top_queries_by_identifier_yield"] = last_round_top_queries_by_yield[:5]
            summary["top_domains"] = sorted(domain_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
            summary["dedup_feedback_by_worker"] = last_round_dedup_feedback_by_worker
            summary["last_round_new_identifiers_sample"] = list(last_round_new_identifiers_sample[:12])
            summary["source_type_counts"] = [
                {"source_type": st, "count": c}
                for st, c in sorted(source_type_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
            ]
            core_types = ["registry", "patent", "pipeline_pdf", "paper"]
            core_counts = {t: int(source_type_counts.get(t, 0)) for t in core_types}
            min_core = min(core_counts.values()) if core_counts else 0
            summary["underrepresented_source_types"] = [t for t, c in core_counts.items() if c == min_core][:4]
            summary["draft_assets_count"] = len(draft_by_canon)
            summary["final_assets_count"] = len(promoted_canons)
            summary["time_elapsed_seconds"] = round(time.monotonic() - t0, 2)
            summary["workers_active"] = len(workers)

            logger.info(
                "run=%s cycle=%s phase=%s starting workers=%s",
                run_id,
                cycle_idx,
                policy_phase,
                len(workers),
            )
            cycle_outputs = await asyncio.gather(
                *[
                    w.run_cycle(
                        run_id=run_id,
                        user_query=user_query,
                        global_summary=summary,
                        round_idx=cycle_idx,
                        forced_query=forced_by_worker.get(w.worker_id),
                    )
                    for w in workers
                ]
            )

            successful_workers = sum(1 for o in cycle_outputs if o.success)

            # Persist recall cycles, queries, results, mentions, candidates
            new_mentions_this_round = 0
            new_candidates_this_round = 0
            new_identifier_tokens_this_round = 0
            new_draft_assets_this_round = 0
            new_final_assets_this_round = 0
            dirty_draft_canons.clear()
            identifier_yield_by_worker = {}
            query_yields: list[tuple[str, int]] = []
            new_identifiers_sample_this_round: list[str] = []
            new_identifiers_sample_seen: set[str] = set()
            cycle_worker_gains: list[dict[str, Any]] = []

            for o in cycle_outputs:
                # Dedup mentions in code (no negative list passed to LLM)
                unique_mentions: list[Mention] = []
                for m in o.mentions:
                    if m.fingerprint in seen_mentions:
                        continue
                    seen_mentions.add(m.fingerprint)
                    unique_mentions.append(m)

                # Track unique executed queries (global; do not pass the full list to the model).
                _register_unique_query(o.query)

                identifier_yield = 0
                identifier_sample: list[str] = []
                for m in unique_mentions:
                    # v1.4: only eligible identifier-like mentions count towards identifier yield.
                    if m.mention_class not in ("drug_code_like", "patent_id_like"):
                        continue
                    if m.source_url:
                        st = str(_guess_evidence_source_type(m.source_url))
                        source_type_counts[st] = source_type_counts.get(st, 0) + 1
                    k = m.canonical_text
                    if not k:
                        continue
                    if k in seen_identifier_tokens:
                        continue
                    seen_identifier_tokens.add(k)
                    identifier_yield += 1
                    if len(identifier_sample) < 6:
                        identifier_sample.append(m.raw_text)
                    if k not in new_identifiers_sample_seen and len(new_identifiers_sample_this_round) < 12:
                        new_identifiers_sample_seen.add(k)
                        new_identifiers_sample_this_round.append(m.raw_text)
                    if m.source_url:
                        dmn = _domain(m.source_url)
                        if dmn:
                            domain_counts[dmn] = domain_counts.get(dmn, 0) + 1
                new_identifier_tokens_this_round += identifier_yield
                identifier_yield_by_worker[o.worker_id] = identifier_yield
                query_yields.append((o.query, identifier_yield))
                cycle_worker_gains.append(
                    {
                        "worker_id": o.worker_id,
                        "query": o.query,
                        "identifier_yield": int(identifier_yield),
                        "success": bool(o.success),
                        "urls_count": int(len(o.urls or [])),
                    }
                )
                tmpl = _query_template(o.query)
                if tmpl:
                    query_template_yields[tmpl] = query_template_yields.get(tmpl, 0) + int(identifier_yield)
                    ex = query_template_examples.setdefault(tmpl, [])
                    if o.query not in ex and len(ex) < 3:
                        ex.append(o.query)
                logger.info(
                    "run=%s cycle=%s phase=%s worker=%s identifier_yield=%s query=%s",
                    run_id,
                    cycle_idx,
                    policy_phase,
                    o.worker_id,
                    identifier_yield,
                    o.query,
                )

                cycle_id = await store.insert_cycle(
                    run_id=run_id,
                    round_idx=cycle_idx,
                    worker_id=o.worker_id,
                    phase="recall",
                    planned_query=o.query,
                    success=o.success,
                    metrics={
                        "policy_phase": policy_phase,
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

                rid = await store.insert_result(
                    run_id=run_id,
                    query_id=qid,
                    tool_name="web_search",
                    response_json=o.response_json,
                    urls=o.urls,
                )

                mention_ids: list[str] = []
                if unique_mentions:
                    mention_ids = await store.insert_mentions(run_id=run_id, query_id=qid, result_id=rid, mentions=unique_mentions)
                    new_mentions_this_round += len(unique_mentions)

                # v1.5: create candidates (for provenance) + draft_assets (for evidence-anchored discovery).
                for idx, m in enumerate(unique_mentions):
                    if m.mention_class not in ("drug_code_like", "patent_id_like"):
                        continue

                    # Candidates are still stored for traceability and eval attribution.
                    c = Candidate.from_mention(m)
                    if c.fingerprint not in seen_candidates:
                        seen_candidates.add(c.fingerprint)
                        source_mention_id = mention_ids[idx] if (mention_ids and idx < len(mention_ids)) else None
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

                    # Identifier typing is computed early for pivoting + later gating.
                    id_type = classify_identifier_type(draft.identifier_raw)
                    draft = draft.model_copy(update={"identifier_type": id_type})

                    canon = draft.identifier_canonical
                    existing = draft_by_canon.get(canon)
                    if existing is None:
                        draft_by_canon[canon] = draft
                        dirty_draft_canons.add(canon)
                        new_draft_assets_this_round += 1
                        if id_type == "drug_code":
                            _enqueue_identifier_pivots(identifier_raw=draft.identifier_raw, identifier_canonical=draft.identifier_canonical)
                    else:
                        merged = _merge_drafts(existing, draft)
                        if merged.model_dump() != existing.model_dump():
                            draft_by_canon[canon] = merged
                            dirty_draft_canons.add(canon)

            # Apply late filtering fields + attempt promotion for drafts touched this cycle.
            promotable: list[DraftAsset] = []
            for canon in sorted(dirty_draft_canons):
                if canon not in draft_by_canon:
                    continue
                updated, eligible = _score_and_label_draft(d=draft_by_canon[canon], cycle_idx=cycle_idx)
                draft_by_canon[canon] = updated
                if eligible and canon not in promoted_canons:
                    promotable.append(updated)

            # Persist draft_assets discovered/updated this cycle (dedup by canonical).
            if dirty_draft_canons and not _time_budget_exceeded():
                await asyncio.gather(
                    *[
                        store.upsert_draft_asset(run_id=run_id, draft=draft_by_canon[c].model_dump())
                        for c in sorted(dirty_draft_canons)
                        if c in draft_by_canon
                    ]
                )
            persisted_drafts = len(dirty_draft_canons)

            # Promote eligible drafts to final_assets (after gates B+C).
            for d in promotable:
                canon = d.identifier_canonical
                if canon in promoted_canons:
                    continue
                sources: list[str] = []
                if d.evidence_url:
                    sources.append(d.evidence_url)
                for c in d.model_dump().get("citations") or []:
                    if isinstance(c, dict):
                        u = str(c.get("url") or "").strip()
                        if u and u not in sources:
                            sources.append(u)

                fa = ValidatedAsset.from_fields(
                    primary_identifier_raw=d.identifier_raw,
                    identifier_aliases_raw=list(d.identifier_aliases_raw or []),
                    evidence_snippet=d.evidence_snippet,
                    evidence_url=d.evidence_url,
                    evidence_source_type=d.evidence_source_type,
                    sponsor=d.sponsor,
                    target=d.target,
                    modality=d.modality,
                    indication=d.indication,
                    development_stage=d.stage,
                    geography=d.geography,
                    sources=sources,
                )
                candidate_id = canon_to_candidate_id.get(canon)
                await store.insert_final_assets(run_id=run_id, candidate_id=candidate_id, assets=[fa])
                if candidate_id:
                    await store.update_candidate_status(candidate_id, "validated")
                if fa.fingerprint not in seen_final_assets:
                    seen_final_assets.add(fa.fingerprint)
                    final_assets.append(fa)
                promoted_canons.add(canon)
                new_final_assets_this_round += 1

            # v1.6: start enrichment inside the main loop (top-K per cycle).
            enrichment_attempts_used = 0
            enrichment_new_final_assets = 0
            if draft_by_canon and not _time_budget_exceeded():
                top_k = int(getattr(cfg, "enrichment_top_k_per_cycle", 0) or 0)
                force_one = bool(draft_by_canon) and (min_enrichment_attempts_per_run > 0) and (
                    int(enrichment_attempts) < int(min_enrichment_attempts_per_run)
                )
                if force_one and top_k < 1:
                    top_k = 1
                if top_k > 0:
                    enrichment_attempts_used, enrichment_new_final_assets = await _enrich_some_drafts(
                        enrichment_cycle_idx=int(cycle_idx),
                        max_to_enrich=int(top_k),
                        force_at_least_one=bool(force_one),
                    )
                    if enrichment_new_final_assets:
                        new_final_assets_this_round += int(enrichment_new_final_assets)

            dirty_draft_canons.clear()
            new_draft_assets_per_cycle.append(int(new_draft_assets_this_round))
            new_final_assets_per_cycle.append(int(new_final_assets_this_round))

            logger.info(
                "run=%s cycle=%s phase=%s harvested new_mentions=%s new_candidates=%s new_identifier_tokens=%s new_draft_assets=%s new_final_assets=%s dirty_drafts=%s enrich_attempts=%s enrich_new_final=%s",
                run_id,
                cycle_idx,
                policy_phase,
                new_mentions_this_round,
                new_candidates_this_round,
                new_identifier_tokens_this_round,
                new_draft_assets_this_round,
                new_final_assets_this_round,
                persisted_drafts,
                enrichment_attempts_used,
                enrichment_new_final_assets,
            )

            await store.insert_metric(
                run_id=run_id,
                round_idx=cycle_idx,
                name="cycle_summary",
                value={
                    "policy_phase": policy_phase,
                    "successful_workers": successful_workers,
                    "new_mentions": new_mentions_this_round,
                    "new_candidates": new_candidates_this_round,
                    "new_identifier_tokens": new_identifier_tokens_this_round,
                    "new_draft_assets": new_draft_assets_this_round,
                    "new_final_assets": new_final_assets_this_round,
                    "enrichment_attempts": int(enrichment_attempts_used),
                    "enrichment_new_final_assets": int(enrichment_new_final_assets),
                    "draft_assets_total": len(draft_by_canon),
                    "final_assets_total": len(promoted_canons),
                    "persisted_draft_assets": persisted_drafts,
                    "unique_queries_total": len(unique_queries),
                    "time_elapsed_seconds": round(time.monotonic() - t0, 2),
                },
            )

            worker_marginal_gains_per_cycle.append(
                {
                    "cycle_idx": int(cycle_idx),
                    "policy_phase": policy_phase,
                    "workers": cycle_worker_gains,
                    "new_identifier_tokens": int(new_identifier_tokens_this_round),
                    "new_draft_assets": int(new_draft_assets_this_round),
                    "new_validated_assets": int(new_final_assets_this_round),
                    "enrichment_attempts": int(enrichment_attempts_used),
                    "enrichment_new_validated_assets": int(enrichment_new_final_assets),
                }
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
            last_round_new_identifiers_sample = list(new_identifiers_sample_this_round)

            # v1.6: stop workers early when marginal new identifiers is flat for N cycles.
            stop_after = int(getattr(cfg, "worker_stop_after_flat_steps", 0) or 0)
            if stop_after > 0 and len(workers) > 1:
                for w in workers:
                    y = int(identifier_yield_by_worker.get(w.worker_id, 0))
                    if y <= 0:
                        worker_flat_steps[w.worker_id] = worker_flat_steps.get(w.worker_id, 0) + 1
                    else:
                        worker_flat_steps[w.worker_id] = 0

                # Only consider currently-active workers for removal.
                removed_ids = [
                    w.worker_id
                    for w in workers
                    if int(worker_flat_steps.get(w.worker_id, 0)) >= stop_after
                ]
                if removed_ids:
                    removed_set = set(removed_ids)
                    before = len(workers)
                    survivors = [w for w in workers if w.worker_id not in removed_set]
                    # Always keep at least one worker alive.
                    if not survivors:
                        survivors = [workers[0]]
                    workers = survivors
                    after = len(workers)
                    logger.info(
                        "run=%s cycle=%s worker_early_stop removed=%s stop_after=%s workers_before=%s workers_after=%s",
                        run_id,
                        cycle_idx,
                        sorted(removed_ids),
                        stop_after,
                        before,
                        after,
                    )

            # Broad/loop exploration accounting + stopping condition.
            if policy_phase in ("broad", "loop"):
                if policy_phase == "broad":
                    broad_cycle_idx += 1
                high_quality = successful_workers >= cfg.min_successful_workers_per_round
                if high_quality and new_draft_assets_this_round == 0:
                    consecutive_no_new_drafts += 1
                elif new_draft_assets_this_round > 0:
                    consecutive_no_new_drafts = 0

                logger.info(
                    "run=%s cycle=%s phase=%s new_draft_assets=%s streak_no_new_drafts=%s high_quality=%s",
                    run_id,
                    cycle_idx,
                    policy_phase,
                    new_draft_assets_this_round,
                    consecutive_no_new_drafts,
                    high_quality,
                )

                if consecutive_no_new_drafts >= cfg.stop_after_no_new_validated_rounds:
                    break

            cycle_idx += 1

        # ------------------------------------------------------------------
        # v1.6 enrichment: primary enrichment happens DURING the loop (top-K/cycle).
        # Optional final sweep uses any remaining budget.
        # ------------------------------------------------------------------
        if draft_by_canon and int(cfg.max_enrichment_validations or 0) > 0 and not _time_budget_exceeded():
            remaining = max(0, int(cfg.max_enrichment_validations) - int(enrichment_attempts))
            if remaining > 0:
                await _enrich_some_drafts(
                    enrichment_cycle_idx=int(cycle_idx),
                    max_to_enrich=int(remaining),
                    force_at_least_one=False,
                )

        # Guarantee: if we found any draft assets, do at least one enrichment attempt (unless out of time/budget).
        if (
            draft_by_canon
            and int(cfg.max_enrichment_validations or 0) > 0
            and int(enrichment_attempts) == 0
        ):
            await _enrich_some_drafts(
                enrichment_cycle_idx=int(cycle_idx),
                max_to_enrich=1,
                force_at_least_one=True,
            )

        stopped_reason = "time_budget_exceeded" if _time_budget_exceeded() else None
        rejection_counts: dict[str, int] = {}
        for d in draft_by_canon.values():
            if d.identifier_canonical in promoted_canons:
                continue
            rr = d.rejection_reason if isinstance(d.rejection_reason, str) and d.rejection_reason else "other"
            rejection_counts[rr] = rejection_counts.get(rr, 0) + 1
        top_rejections = [
            {"reason": r, "count": c}
            for r, c in sorted(rejection_counts.items(), key=lambda kv: kv[1], reverse=True)[:12]
        ]

        # Run report (v1.6): worker gains, query templates by yield, rejection reasons by stage.
        top_query_templates_by_yield = [
            {
                "template": tmpl,
                "total_identifier_yield": int(y),
                "example_queries": list(query_template_examples.get(tmpl, [])[:3]),
            }
            for tmpl, y in sorted(query_template_yields.items(), key=lambda kv: kv[1], reverse=True)[:20]
            if tmpl
        ]
        enrichment_status_counts: dict[str, int] = {}
        for d in draft_by_canon.values():
            st = str(getattr(d, "enrichment_status", "") or "pending")
            enrichment_status_counts[st] = enrichment_status_counts.get(st, 0) + 1

        run_report_v1_6: dict[str, Any] = {
            "run_id": run_id,
            "config_version": config_version,
            "prompt_version": prompt_version,
            "per_worker_marginal_gains_per_cycle": list(worker_marginal_gains_per_cycle),
            "top_query_templates_by_identifier_yield": top_query_templates_by_yield,
            "rejection_reasons_by_stage": {
                "promotion_gate": dict(rejection_counts),
                "enrichment_status": dict(enrichment_status_counts),
            },
        }
        summary = {
            "unique_mentions": len(seen_mentions),
            "unique_candidates": len(seen_candidates),
            "draft_assets": len(draft_by_canon),
            "final_assets": len(promoted_canons),
            "enrichment_attempts": int(enrichment_attempts),
            "time_elapsed_seconds": round(time.monotonic() - t0, 2),
            "stopped_reason": stopped_reason,
            # v1.5 debug + eval support
            "unique_queries": list(unique_queries),
            "unique_domains": [{"domain": d, "count": c} for d, c in sorted(domain_counts.items(), key=lambda kv: kv[1], reverse=True)],
            "new_draft_assets_per_cycle": list(new_draft_assets_per_cycle),
            "new_final_assets_per_cycle": list(new_final_assets_per_cycle),
            "top_rejection_reasons": top_rejections,
            "run_report_v1_6": run_report_v1_6,
        }
        # Persist the full run report separately as a metric for easier querying.
        await store.insert_metric(run_id=run_id, round_idx=None, name="run_report_v1_6", value=run_report_v1_6)
        await store.finish_run(run_id, status="completed", summary=summary)

        # v1.6 product semantics:
        # - Found (unvalidated) == draft_assets
        # - Validated (evidence-complete) == final_assets
        found_assets = [draft_by_canon[k].model_dump() for k in sorted(draft_by_canon.keys())]
        validated_assets = [a.model_dump() for a in final_assets]

        return {
            "run_id": run_id,
            "status": "completed",
            "manifest": manifest,
            "summary": summary,
            # Back-compat: keep `assets` as the validated assets list.
            "assets": list(validated_assets),
            "found_assets": found_assets,
            "validated_assets": validated_assets,
            "run_report": run_report_v1_6,
        }
    except Exception as e:
        await store.finish_run(run_id, status="failed", summary={"error": str(e)})
        raise
    finally:
        # Close any optional HTTP clients (e.g., fetchers) held by workers.
        if workers:
            await asyncio.gather(*[w.aclose() for w in workers], return_exceptions=True)
        await openai.aclose()


