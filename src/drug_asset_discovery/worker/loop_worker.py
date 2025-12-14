from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from drug_asset_discovery.config import PromptBundle, RunConfig
from drug_asset_discovery.openai_client import OpenAIResponsesClient
from drug_asset_discovery.utils.idempotency import idempotency_key
from drug_asset_discovery.utils.json_extract import JSONExtractError, extract_first_json


@dataclass(frozen=True)
class LoopPersona:
    name: str
    seed: str


DEFAULT_LOOP_PERSONAS: list[LoopPersona] = [
    LoopPersona(
        name="trial-registry-hunter",
        seed="Prioritize trial registries and congress abstracts; use site: filters; look for NCT/EudraCT/CTRI IDs and drug codes.",
    ),
    LoopPersona(
        name="pipeline-pdf-scavenger",
        seed="Prioritize sponsor pipeline pages and investor PDFs; use filetype:pdf and site: filters; look for program codes and series members.",
    ),
    LoopPersona(
        name="patent-miner",
        seed="Prioritize WO/US/EP patents and assignee searches; look for compound codes and patent IDs; include CPC/assignee terms.",
    ),
    LoopPersona(
        name="non-us-geo",
        seed="Prioritize EU/JP/CN sources and local-language queries when sensible; look for local trial IDs and local sponsor pipelines.",
    ),
    LoopPersona(
        name="vendor-catalogs",
        seed="Prioritize vendor catalogs and reagent listings; look for code-like tokens and program names; use site: and filetype where useful.",
    ),
]


SourceTypeGuess = Literal[
    "pipeline",
    "trial_registry",
    "patent",
    "conference",
    "vendor_catalog",
    "press_release",
    "paper",
    "other",
]


@dataclass(frozen=True)
class LoopCandidate:
    raw_identifier: str
    context_snippet: str
    source_url: str
    source_type_guess: SourceTypeGuess = "other"


@dataclass(frozen=True)
class LoopCycleOutput:
    worker_id: str
    cycle_idx: int
    success: bool
    executed_queries: list[str]
    candidates: list[LoopCandidate]
    next_cycle_query_ideas: list[str]
    response_json: dict[str, Any]
    output_text: str


def _web_search_tool(cfg: RunConfig) -> dict[str, Any]:
    tool: dict[str, Any] = {"type": "web_search"}
    tool["external_web_access"] = bool(cfg.retrieval.external_web_access)
    if cfg.retrieval.user_location:
        tool["user_location"] = cfg.retrieval.user_location
    return tool


class LoopWorker:
    def __init__(
        self,
        *,
        worker_id: str,
        persona: LoopPersona,
        cfg: RunConfig,
        prompts: PromptBundle,
        openai: OpenAIResponsesClient,
        cache_dir: Path,
    ) -> None:
        self.worker_id = worker_id
        self.persona = persona
        self.cfg = cfg
        self.prompts = prompts
        self.openai = openai
        self.cache_dir = cache_dir

    async def run_cycle(
        self,
        *,
        run_id: str,
        user_query: str,
        global_summary: dict[str, Any],
        cycle_idx: int,
        dedup_feedback: str | None = None,
    ) -> LoopCycleOutput:
        prompt = (
            self.prompts.loop_worker
            + "\n\n"
            + f"persona_seed: {self.persona.seed}\n\n"
            + "original_user_query:\n"
            + f"{user_query}\n\n"
            + "compact_progress_summary:\n"
            + f"{global_summary}\n\n"
        )
        if dedup_feedback:
            prompt += f"dedup_feedback:\n{dedup_feedback}\n\n"

        resp = await self.openai.create_response(
            model=self.cfg.model,
            input_text=prompt,
            reasoning_effort=self.cfg.reasoning_effort.get("loop_worker", "xhigh"),
            tools=[_web_search_tool(self.cfg)],
            idempotency_key=idempotency_key(
                "loop_worker.cycle",
                {
                    "run_id": run_id,
                    "worker_id": self.worker_id,
                    "persona": self.persona.name,
                    "cycle_idx": cycle_idx,
                },
            ),
        )

        out_text = resp.output_text
        success = bool(out_text)

        executed_queries: list[str] = []
        candidates: list[LoopCandidate] = []
        next_ideas: list[str] = []

        try:
            obj = extract_first_json(out_text)
        except JSONExtractError:
            obj = None

        if isinstance(obj, dict):
            eq = obj.get("executed_queries")
            if isinstance(eq, list):
                executed_queries = [str(x).strip() for x in eq if isinstance(x, str) and x.strip()]

            cand = obj.get("candidates")
            if isinstance(cand, list):
                for c in cand:
                    if not isinstance(c, dict):
                        continue
                    raw = c.get("raw_identifier")
                    ctx = c.get("context_snippet")
                    url = c.get("source_url")
                    st = c.get("source_type_guess") or "other"
                    if not (isinstance(raw, str) and raw.strip()):
                        continue
                    if not (isinstance(url, str) and url.strip()):
                        continue
                    if not (isinstance(ctx, str) and ctx.strip()):
                        ctx = raw.strip()
                    stg: SourceTypeGuess = st if st in (  # type: ignore[assignment]
                        "pipeline",
                        "trial_registry",
                        "patent",
                        "conference",
                        "vendor_catalog",
                        "press_release",
                        "paper",
                        "other",
                    ) else "other"
                    candidates.append(
                        LoopCandidate(
                            raw_identifier=raw.strip(),
                            context_snippet=ctx.strip(),
                            source_url=url.strip(),
                            source_type_guess=stg,
                        )
                    )

            ni = obj.get("next_cycle_query_ideas")
            if isinstance(ni, list):
                next_ideas = [str(x).strip() for x in ni if isinstance(x, str) and x.strip()]

        # Enforce hard budget in code (fail-soft: truncate).
        budget = int(self.cfg.max_web_search_calls_per_worker_cycle or 1)
        if budget > 0 and len(executed_queries) > budget:
            executed_queries = executed_queries[:budget]

        return LoopCycleOutput(
            worker_id=self.worker_id,
            cycle_idx=cycle_idx,
            success=success,
            executed_queries=executed_queries,
            candidates=candidates,
            next_cycle_query_ideas=next_ideas,
            response_json=resp.raw,
            output_text=out_text,
        )


