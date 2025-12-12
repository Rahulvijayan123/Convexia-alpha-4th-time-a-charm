from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any


def stable_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_normalize(text: str) -> str:
    """
    Conservative normalization for dedup keys:
    - Unicode NFKC
    - trim
    - collapse whitespace
    - casefold (does not remove tokens)

    IMPORTANT: We still store the raw string separately to preserve rare tokens.
    """
    t = unicodedata.normalize("NFKC", text)
    t = " ".join(t.strip().split())
    return t.casefold()


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


