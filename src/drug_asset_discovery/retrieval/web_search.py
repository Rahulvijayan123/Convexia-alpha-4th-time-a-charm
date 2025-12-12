from __future__ import annotations

from pathlib import Path
from typing import Any

from drug_asset_discovery.config import RunConfig
from drug_asset_discovery.openai_client import OpenAIResponse, OpenAIResponsesClient
from drug_asset_discovery.retrieval.cache import DiskCache
from drug_asset_discovery.utils.hashing import stable_json_dumps, stable_sha256


def _web_search_tool(cfg: RunConfig) -> dict[str, Any]:
    tool: dict[str, Any] = {"type": "web_search"}
    # Per OpenAI docs: external_web_access can run cache-only when false.
    tool["external_web_access"] = bool(cfg.retrieval.external_web_access)
    if cfg.retrieval.user_location:
        tool["user_location"] = cfg.retrieval.user_location
    return tool


async def run_recall_web_search(
    *,
    openai: OpenAIResponsesClient,
    cfg: RunConfig,
    cache_dir: Path,
    query: str,
    idempotency_key: str,
) -> OpenAIResponse:
    """
    Stage 1 retrieval: maximize identifier yield. No relevance filtering.
    """
    cache = DiskCache(cache_dir)
    cache_key = stable_sha256(stable_json_dumps({"q": query, "tool": _web_search_tool(cfg), "v": "recall"}))
    cached = cache.get("web_search", cache_key)
    if cached:
        return OpenAIResponse(raw=cached)

    prompt = (
        "Use web search.\n"
        "Return a bulleted list of sources with:\n"
        "- Title\n"
        "- URL\n"
        "- 1–3 direct quote snippets that contain ANY identifiers (drug codes/names, trial IDs, patent IDs), targets, modalities, indications.\n"
        "Do NOT paraphrase identifiers; preserve tokens exactly.\n\n"
        f"Search query: {query}\n"
    )

    resp = await openai.create_response(
        model=cfg.model,
        input_text=prompt,
        reasoning_effort="medium",
        tools=[_web_search_tool(cfg)],
        idempotency_key=idempotency_key,
    )
    cache.set("web_search", cache_key, resp.raw)
    return resp


async def run_validation_web_search(
    *,
    openai: OpenAIResponsesClient,
    cfg: RunConfig,
    cache_dir: Path,
    prompt: str,
    idempotency_key: str,
) -> OpenAIResponse:
    """
    Stage 2 retrieval+reasoning: strict validation and structured output.
    """
    cache = DiskCache(cache_dir)
    cache_key = stable_sha256(stable_json_dumps({"prompt": prompt, "tool": _web_search_tool(cfg), "v": "validate"}))
    cached = cache.get("web_search_validate", cache_key)
    if cached:
        return OpenAIResponse(raw=cached)

    resp = await openai.create_response(
        model=cfg.model,
        input_text=prompt,
        reasoning_effort=cfg.reasoning_effort["validation"],
        tools=[_web_search_tool(cfg)],
        idempotency_key=idempotency_key,
    )
    cache.set("web_search_validate", cache_key, resp.raw)
    return resp


