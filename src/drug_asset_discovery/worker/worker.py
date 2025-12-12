from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drug_asset_discovery.config import PromptBundle, RunConfig
from drug_asset_discovery.extraction.llm_extractor import llm_extract_mentions
from drug_asset_discovery.extraction.regex_harvester import regex_harvest_mentions
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
    if profile.query_seed_style == "patent":
        return [
            f'{uq} WO patent',
            f'{uq} "WO" "assignee" compound',
            f'{uq} patent drug code',
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
        from drug_asset_discovery.worker.frontier import Frontier

        self.frontier = Frontier(max_size=max_frontier_size)
        self._recent_queries: list[str] = []

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

    async def run_cycle(self, *, run_id: str, user_query: str, global_summary: dict[str, Any], round_idx: int) -> WorkerCycleOutput:
        # Ensure we have something to do
        if len(self.frontier) == 0:
            self.frontier.seed(_seed_queries(user_query, self.profile))

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
        urls = resp.extracted_urls()
        success = bool(text)

        # Stage 1 extraction (reckless): regex + LLM
        regex_mentions = regex_harvest_mentions(text, source_url=(urls[0] if urls else None))
        llm_mentions = await llm_extract_mentions(
            openai=self.openai,
            cfg=self.cfg,
            prompts=self.prompts,
            text=text,
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

        # Plan next queries
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


