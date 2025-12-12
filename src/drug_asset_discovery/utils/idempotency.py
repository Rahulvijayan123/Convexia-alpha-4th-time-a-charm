from __future__ import annotations

from drug_asset_discovery.utils.hashing import stable_sha256, stable_json_dumps


def idempotency_key(namespace: str, payload: dict) -> str:
    """
    Stable idempotency key. Keep payload compact (do NOT include giant negative lists).
    """
    return stable_sha256(f"{namespace}:{stable_json_dumps(payload)}")


