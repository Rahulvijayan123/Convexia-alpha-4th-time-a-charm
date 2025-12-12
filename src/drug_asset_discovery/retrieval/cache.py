from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DiskCache:
    root_dir: Path

    def _path(self, namespace: str, key: str) -> Path:
        return self.root_dir / namespace / f"{key}.json"

    def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        p = self._path(namespace, key)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def set(self, namespace: str, key: str, value: dict[str, Any]) -> None:
        p = self._path(namespace, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


