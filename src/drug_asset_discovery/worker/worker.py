from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drug_asset_discovery.config import PromptBundle, RunConfig
from drug_asset_discovery.extraction.llm_extractor import llm_extract_mentions
from drug_asset_discovery.extraction.regex_harvester import regex_harvest_mentions
from drug_asset_discovery.fetch.fetcher import Fetcher
from drug_asset_discovery.models.domain import Mention
from drug_asset_discovery.openai_client import OpenAIResponsesClient
from drug_asset_discovery.retrieval.web_search import run_recall_web_search
from drug_asset_discovery.utils.hashing import safe_normalize, stable_sha256
from drug_asset_discovery.utils.idempotency import idempotency_key
from drug_asset_discovery.utils.json_extract import JSONExtractError, extract_first_json


@dataclass(frozen=True)
class WorkerProfile:
    name: str
    focus: str
    query_seed_style: str


DEFAULT_PROFILES: list[WorkerProfile] = [
    WorkerProfile(
        name="clinical-trials",
        focus="Trial registries, congress abstracts, clinical-stage assets and trial IDs",
        query_seed_style="trial",
    ),
    WorkerProfile(
        name="pipelines",
        focus="Sponsor pipeline pages, press releases, investor decks, product candidates",
        query_seed_style="pipeline",
    ),
    WorkerProfile(
        name="patents",
        focus="Patent filings (WO/US/EP), assignees, inventions and drug code names",
        query_seed_style="patent",
    ),
    WorkerProfile(
        name="geography",
        focus="Non-US geographies (EU/JP/CN) and local sources; alternative spellings",
        query_seed_style="geo",
    ),
]


# v1.6: enforced diversity across 4 specialized workers (tabs)
DEFAULT_V16_PROFILES: list[WorkerProfile] = [
    WorkerProfile(
        name="trials-registries",
        focus="Trial registries and registries-like sources (ClinicalTrials.gov, EU/JP/CN registries), plus congress abstracts/posters with trial IDs",
        query_seed_style="trial",
    ),
    WorkerProfile(
        name="patents",
        focus="Patent filings (WO/US/EP), patent families, assignee pivots, and code-name harvesting from patent text/snippets",
        query_seed_style="patent",
    ),
    WorkerProfile(
        name="pipelines-investor-pdf",
        focus="Sponsor pipeline pages, investor decks, and pipeline PDFs (filetype:pdf), focusing on program code identifiers",
        query_seed_style="pipeline_pdf",
    ),
    WorkerProfile(
        name="literature-posters",
        focus="Literature, posters, abstracts, and preprints (PubMed/DOI/conference sites) to harvest code-like identifiers and trial links",
        query_seed_style="literature",
    ),
]


def _seed_queries(user_query: str, profile: WorkerProfile) -> list[str]:
    uq = user_query.strip()
    if profile.query_seed_style == "trial":
        return [
            f'{uq} site:clinicaltrials.gov',
            f'{uq} NCT trial',
            f'{uq} phase 1 phase 2 phase 3 drug code',
        ]
    if profile.query_seed_style == "pipeline":
        return [
            f'{uq} pipeline sponsor',
            f'{uq} "pipeline" PDF',
            f'{uq} press release trial',
        ]
    if profile.query_seed_style == "pipeline_pdf":
        return [
            f'{uq} pipeline pdf filetype:pdf',
            f'{uq} investor presentation pdf filetype:pdf',
            f'{uq} company pipeline filetype:pdf',
        ]
    if profile.query_seed_style == "patent":
        return [
            f'{uq} WO patent',
            f'{uq} "WO" "assignee" compound',
            f'{uq} patent drug code',
        ]
    if profile.query_seed_style == "literature":
        return [
            f'{uq} poster abstract',
            f'{uq} conference poster PDF filetype:pdf',
            f'{uq} site:pubmed.ncbi.nlm.nih.gov',
        ]
    # geo
    return [
        f'{uq} EU trial',
        f'{uq} Japan trial',
        f'{uq} China trial company pipeline',
    ]


@dataclass
class WorkerCycleOutput:
    worker_id: str
    round_idx: int
    query: str
    success: bool
    response_json: dict[str, Any]
    output_text: str
    urls: list[str]
    mentions: list[Mention]
    planned_queries: list[str]


class Worker:
    def __init__(
        self,
        *,
        worker_id: str,
        profile: WorkerProfile,
        cfg: RunConfig,
        prompts: PromptBundle,
        openai: OpenAIResponsesClient,
        cache_dir: Path,
        max_frontier_size: int,
    ) -> None:
        self.worker_id = worker_id
        self.profile = profile
        self.cfg = cfg
        self.prompts = prompts
        self.openai = openai
        self.cache_dir = cache_dir
        self.fetcher: Fetcher | None = (
            Fetcher(cache_dir=cache_dir, timeout_s=int(cfg.optional_fetch.timeout_seconds))
            if bool(cfg.optional_fetch.enabled)
            else None
        )
        from drug_asset_discovery.worker.frontier import Frontier

        self.frontier = Frontier(max_size=max_frontier_size)
        self._recent_queries: list[str] = []

    async def aclose(self) -> None:
        if self.fetcher is not None:
            await self.fetcher.aclose()

    def _remember_query(self, q: str) -> None:
        self._recent_queries.append(q)
        self._recent_queries = self._recent_queries[-12:]

    async def _plan_next_queries(self, *, user_query: str, global_summary: dict[str, Any], round_idx: int) -> list[str]:
        # IMPORTANT: Do not include long negative lists (unvalidated assets) here.
        prompt = (
            self.prompts.worker_planner
            + "\n\n"
            + f"WORKER PROFILE: {self.profile.name}\nFOCUS: {self.profile.focus}\n\n"
            + f"ORIGINAL USER QUERY:\n{user_query}\n\n"
            + f"ROUND: {round_idx}\n\n"
            + "RECENT QUERIES (this worker):\n"
            + "\n".join(f"- {q}" for q in self._recent_queries[-8:])
            + "\n\n"
            + "COMPACT PROGRESS SUMMARY:\n"
            + f"{global_summary}\n"
        )
        resp = await self.openai.create_response(
            model=self.cfg.model,
            input_text=prompt,
            reasoning_effort=self.cfg.reasoning_effort["worker_planning"],
            tools=None,
            idempotency_key=idempotency_key(
                "worker.plan",
                {
                    "worker_id": self.worker_id,
                    "profile": self.profile.name,
                    "round_idx": round_idx,
                    "recent": self._recent_queries[-3:],
                    "summary": {
                        "validated_assets": global_summary.get("validated_assets_count"),
                        "unique_mentions": global_summary.get("unique_mentions_count"),
                    },
                },
            ),
        )
        try:
            obj = extract_first_json(resp.output_text)
        except JSONExtractError:
            return []
        queries = obj.get("queries") if isinstance(obj, dict) else None
        if not isinstance(queries, list):
            return []
        out: list[str] = []
        for q in queries:
            if isinstance(q, str) and q.strip():
                out.append(" ".join(q.strip().split()))
        # Dedup preserving order
        seen = set()
        final: list[str] = []
        for q in out:
            k = safe_normalize(q)
            if k in seen:
                continue
            seen.add(k)
            final.append(q)
        return final[: self.cfg.max_planned_queries_per_cycle]

    async def run_cycle(
        self,
        *,
        run_id: str,
        user_query: str,
        global_summary: dict[str, Any],
        round_idx: int,
        forced_query: str | None = None,
    ) -> WorkerCycleOutput:
        is_v16 = str(self.cfg.version or "").startswith("v1.6")

        # Ensure we have something to do.
        # v1.6: avoid static seed query lists; seed via the query planner (micro-queries) using only compact summaries.
        if len(self.frontier) == 0 and not (isinstance(forced_query, str) and forced_query.strip()):
            if is_v16:
                planned_seed = await self._plan_next_queries(
                    user_query=user_query, global_summary=global_summary, round_idx=round_idx
                )
                for q in planned_seed:
                    self.frontier.add(q)
                # Fail-soft: if planning yields nothing, use a single profile-specific micro-query.
                if len(self.frontier) == 0:
                    style = str(self.profile.query_seed_style or "").strip()
                    fallback = user_query
                    if style == "trial":
                        fallback = f"{user_query} site:clinicaltrials.gov"
                    elif style == "patent":
                        fallback = f"{user_query} patent"
                    elif style == "pipeline_pdf":
                        fallback = f"{user_query} pipeline pdf filetype:pdf"
                    elif style == "literature":
                        fallback = f"{user_query} poster abstract"
                    self.frontier.add(fallback)
            else:
                # Legacy behavior for older versions.
                self.frontier.seed(_seed_queries(user_query, self.profile))

        query = " ".join((forced_query or "").strip().split()) if forced_query else ""
        if not query:
            query = self.frontier.pop() or user_query
        self._remember_query(query)

        # Recall-stage web search
        resp = await run_recall_web_search(
            openai=self.openai,
            cfg=self.cfg,
            cache_dir=self.cache_dir,
            query=query,
            idempotency_key=idempotency_key(
                "recall.web_search",
                {"run_id": run_id, "worker_id": self.worker_id, "round_idx": round_idx, "query": query},
            ),
        )

        text = resp.output_text

        # v1.3: prefer structured JSON output from web_search for provenance-friendly harvesting.
        sources: list[dict[str, Any]] = []
        try:
            obj = extract_first_json(text)
            cand = obj.get("sources") if isinstance(obj, dict) else None
            if isinstance(cand, list):
                sources = [s for s in cand if isinstance(s, dict)]
        except JSONExtractError:
            sources = []

        urls: list[str] = []
        source_blocks: list[tuple[str, str]] = []  # (url, combined_snippets_text)
        for s in sources:
            url = s.get("url") or s.get("source_url")
            if not isinstance(url, str) or not url.strip():
                continue
            snippets = s.get("snippets") or s.get("quotes") or []
            if not isinstance(snippets, list):
                snippets = []
            snips: list[str] = []
            for sn in snippets:
                if isinstance(sn, str) and sn.strip():
                    snips.append(sn.strip())
            combined = "\n".join(snips).strip()
            if not combined:
                continue
            urls.append(url.strip())
            source_blocks.append((url.strip(), combined[:4000]))  # keep extraction bounded

        # Fallback: if parsing failed, use best-effort URLs from annotations.
        if not urls:
            urls = resp.extracted_urls()

        success = bool(text)

        # Stage 1 extraction (reckless): regex + LLM
        regex_mentions: list[Mention] = []
        for url, block in source_blocks:
            regex_mentions.extend(regex_harvest_mentions(block, source_url=url))

        # Optional: fetch full documents and harvest regex mentions from fetched text (offline-safe, recall-first).
        if self.fetcher is not None and urls:
            for u in urls[: int(self.cfg.optional_fetch.max_docs_per_round)]:
                doc = await self.fetcher.fetch_text(u)
                if doc and doc.text:
                    regex_mentions.extend(regex_harvest_mentions(doc.text[:20000], source_url=u))

        # Build a provenance-friendly pack for the LLM extractor (URLs included per block).
        if source_blocks:
            pack_lines: list[str] = ["SOURCES:"]
            for i, (url, block) in enumerate(source_blocks[:8], start=1):
                pack_lines.append(f"[{i}] URL: {url}")
                pack_lines.append("SNIPPETS:")
                pack_lines.append(block)
                pack_lines.append("")
            extract_text = "\n".join(pack_lines)
        else:
            extract_text = text

        llm_mentions = await llm_extract_mentions(
            openai=self.openai,
            cfg=self.cfg,
            prompts=self.prompts,
            text=extract_text,
            idempotency_key=idempotency_key(
                "recall.extract_llm",
                {
                    "run_id": run_id,
                    "worker_id": self.worker_id,
                    "round_idx": round_idx,
                    "query_fp": stable_sha256(safe_normalize(query)),
                },
            ),
            source_url=(urls[0] if urls else None),
        )

        # Merge + dedup
        merged: list[Mention] = []
        seen = set()
        for m in (regex_mentions + llm_mentions):
            if m.fingerprint in seen:
                continue
            seen.add(m.fingerprint)
            merged.append(m)

        # v1.5 policy: query planning is orchestrator-driven (templates + pivoting).
        planned: list[str] = []
        if not str(self.cfg.version or "").startswith("v1.5"):
            planned = await self._plan_next_queries(user_query=user_query, global_summary=global_summary, round_idx=round_idx)
            for q in planned:
                self.frontier.add(q)

        return WorkerCycleOutput(
            worker_id=self.worker_id,
            round_idx=round_idx,
            query=query,
            success=success,
            response_json=resp.raw,
            output_text=text,
            urls=urls,
            mentions=merged,
            planned_queries=planned,
        )


