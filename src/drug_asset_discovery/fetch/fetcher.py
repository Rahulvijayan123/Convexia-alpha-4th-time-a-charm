from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from drug_asset_discovery.utils.hashing import stable_sha256


@dataclass(frozen=True)
class FetchedDocument:
    url: str
    content_type: str | None
    text: str


class Fetcher:
    """
    Optional full-document fetcher. Keep this conditional to control cost.
    This module does NOT perform search; it only fetches specific URLs.
    """

    def __init__(self, *, cache_dir: Path, timeout_s: int = 20):
        self.cache_dir = cache_dir
        self.timeout_s = timeout_s
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s), follow_redirects=True)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _cache_path(self, url: str) -> Path:
        key = stable_sha256(url)
        return self.cache_dir / "documents" / f"{key}.txt"

    async def fetch_text(self, url: str) -> FetchedDocument | None:
        p = self._cache_path(url)
        if p.exists():
            return FetchedDocument(url=url, content_type=None, text=p.read_text(encoding="utf-8", errors="ignore"))

        try:
            resp = await self._client.get(url, headers={"User-Agent": "drug-asset-discovery/0.1"})
            resp.raise_for_status()
        except Exception:
            return None

        ctype = resp.headers.get("content-type")
        text = resp.text
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", errors="ignore")
        return FetchedDocument(url=url, content_type=ctype, text=text)


