from __future__ import annotations

import json
from typing import Any


class JSONExtractError(ValueError):
    pass


def extract_first_json(text: str) -> Any:
    """
    Best-effort extraction of the first JSON object/array from an LLM response.
    """
    s = text.strip()
    if not s:
        raise JSONExtractError("Empty text")

    # Fast path: whole string is JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # Search for first { or [
    starts = [i for i in (s.find("{"), s.find("[")) if i != -1]
    if not starts:
        raise JSONExtractError("No JSON start found")
    start = min(starts)
    candidate = s[start:]

    # Try progressively shrinking from the end
    for end in range(len(candidate), max(len(candidate) - 10000, 1), -1):
        chunk = candidate[:end].strip()
        try:
            return json.loads(chunk)
        except Exception:
            continue

    raise JSONExtractError("Failed to parse JSON")


