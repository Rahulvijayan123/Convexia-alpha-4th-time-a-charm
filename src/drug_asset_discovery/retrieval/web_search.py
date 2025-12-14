from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from drug_asset_discovery.config import RunConfig
from drug_asset_discovery.openai_client import OpenAIResponse, OpenAIResponsesClient
from drug_asset_discovery.retrieval.cache import DiskCache
from drug_asset_discovery.utils.hashing import stable_json_dumps, stable_sha256

logger = logging.getLogger(__name__)


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
    cache_key = stable_sha256(stable_json_dumps({"q": query, "tool": _web_search_tool(cfg), "v": "recall_v1.3"}))
    cached = cache.get("web_search", cache_key)
    if cached:
        logger.debug("web_search recall cache_hit key=%s q_chars=%s", cache_key, len(query or ""))
        return OpenAIResponse(raw=cached)
    logger.debug("web_search recall cache_miss key=%s q_chars=%s", cache_key, len(query or ""))

    prompt = (
        "Use web search.\n"
        "Return ONLY valid JSON (no prose, no markdown) with this exact shape:\n"
        "{\n"
        '  "sources": [\n'
        "    {\n"
        '      "title": "string",\n'
        '      "url": "https://...",\n'
        '      "snippets": ["direct quote 1", "direct quote 2"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Keep sources to ~5–8.\n"
        "- Each snippet MUST be a short direct quote that contains ANY identifiers (drug codes/names, trial IDs, patent IDs).\n"
        "- IDENTIFIER PRESERVATION: do NOT paraphrase identifiers; copy tokens exactly.\n\n"
        f"Search query: {query}\n"
    )

    t0 = time.perf_counter()
    q_preview = (query or "").strip().replace("\n", " ")
    if len(q_preview) > 180:
        q_preview = q_preview[:180] + "…"
    logger.info("web_search recall start idempotency=%s query=%r", idempotency_key, q_preview)
    resp = await openai.create_response(
        model=cfg.model,
        input_text=prompt,
        reasoning_effort="medium",
        tools=[_web_search_tool(cfg)],
        idempotency_key=idempotency_key,
    )
    dt_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "web_search recall done duration_ms=%s output_text_chars=%s urls=%s cache_key=%s",
        dt_ms,
        len(resp.output_text or ""),
        len(resp.extracted_urls()),
        cache_key,
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
        logger.debug("web_search validate cache_hit key=%s prompt_chars=%s", cache_key, len(prompt or ""))
        return OpenAIResponse(raw=cached)
    logger.debug("web_search validate cache_miss key=%s prompt_chars=%s", cache_key, len(prompt or ""))

    t0 = time.perf_counter()
    p_preview = (prompt or "").strip().replace("\n", " ")
    if len(p_preview) > 180:
        p_preview = p_preview[:180] + "…"
    logger.info("web_search validate start idempotency=%s prompt_preview=%r", idempotency_key, p_preview)
    resp = await openai.create_response(
        model=cfg.model,
        input_text=prompt,
        reasoning_effort=cfg.reasoning_effort["validation"],
        tools=[_web_search_tool(cfg)],
        idempotency_key=idempotency_key,
    )
    dt_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "web_search validate done duration_ms=%s output_text_chars=%s urls=%s cache_key=%s",
        dt_ms,
        len(resp.output_text or ""),
        len(resp.extracted_urls()),
        cache_key,
    )
    cache.set("web_search_validate", cache_key, resp.raw)
    return resp


