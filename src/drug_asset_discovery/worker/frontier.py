from __future__ import annotations

from collections import deque

from drug_asset_discovery.utils.hashing import safe_normalize


class Frontier:
    def __init__(self, max_size: int = 30):
        self._max_size = max_size
        self._q: deque[str] = deque()
        self._seen: set[str] = set()

    def __len__(self) -> int:
        return len(self._q)

    def add(self, query: str) -> bool:
        q = " ".join(query.strip().split())
        if not q:
            return False
        key = safe_normalize(q)
        if key in self._seen:
            return False
        if len(self._q) >= self._max_size:
            return False
        self._seen.add(key)
        self._q.append(q)
        return True

    def pop(self) -> str | None:
        if not self._q:
            return None
        return self._q.popleft()

    def seed(self, queries: list[str]) -> None:
        for q in queries:
            self.add(q)


