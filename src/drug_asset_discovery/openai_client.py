from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from drug_asset_discovery.config import ReasoningEffort
from drug_asset_discovery.utils.retry import RetryConfig, async_retry


@dataclass(frozen=True)
class OpenAIResponse:
    raw: dict[str, Any]

    @property
    def output_text(self) -> str:
        # Best-effort extraction of text across SDK/format variants
        if isinstance(self.raw.get("output_text"), str):
            return self.raw["output_text"]

        texts: list[str] = []
        for item in self.raw.get("output", []) or []:
            if item.get("type") != "message":
                continue
            for part in item.get("content", []) or []:
                t = part.get("text") or part.get("output_text") or part.get("content")
                if isinstance(t, str) and t.strip():
                    texts.append(t)
        return "\n".join(texts).strip()

    def extracted_urls(self) -> list[str]:
        urls: list[str] = []
        # Attempt to parse url_citation annotations
        for item in self.raw.get("output", []) or []:
            if item.get("type") != "message":
                continue
            for part in item.get("content", []) or []:
                for ann in part.get("annotations", []) or []:
                    url = ann.get("url") or ann.get("source_url")
                    if isinstance(url, str):
                        urls.append(url)
                # Some variants may include a "sources" field
                for src in part.get("sources", []) or []:
                    url = src.get("url") if isinstance(src, dict) else None
                    if isinstance(url, str):
                        urls.append(url)
        # Some variants include top-level "sources"
        for src in self.raw.get("sources", []) or []:
            url = src.get("url") if isinstance(src, dict) else None
            if isinstance(url, str):
                urls.append(url)
        # Dedup preserving order
        seen = set()
        out: list[str] = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out


class OpenAIResponsesClient:
    """
    Minimal HTTP client for the OpenAI Responses API.

    Non-negotiable: call with model='gpt-5.2' only.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", timeout_s: int = 90):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_s),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @async_retry(RetryConfig())
    async def create_response(
        self,
        *,
        model: str,
        input_text: str,
        reasoning_effort: ReasoningEffort = "high",
        tools: list[dict[str, Any]] | None = None,
        idempotency_key: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> OpenAIResponse:
        if model != "gpt-5.2":
            raise ValueError(f"Non-negotiable: model must be 'gpt-5.2' (got {model!r})")

        payload: dict[str, Any] = {
            "model": model,
            "input": input_text,
            "reasoning": {"effort": reasoning_effort},
        }
        if tools:
            payload["tools"] = tools
        if extra:
            payload.update(extra)

        headers = {}
        headers["Idempotency-Key"] = idempotency_key or str(uuid.uuid4())

        resp = await self._client.post("/responses", json=payload, headers=headers)
        resp.raise_for_status()
        return OpenAIResponse(raw=resp.json())


