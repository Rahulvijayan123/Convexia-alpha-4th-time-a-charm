from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from drug_asset_discovery.config import repo_root


def _secrets_path() -> Path:
    return repo_root() / ".cache" / "runtime_secrets.json"


def load_runtime_secrets() -> dict[str, str]:
    p = _secrets_path()
    if not p.exists():
        return {}
    try:
        obj: Any = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, str] = {}
    for k in ("OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()
    return out


def save_runtime_secrets(*, openai_api_key: str, supabase_url: str, supabase_service_role_key: str) -> None:
    p = _secrets_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "OPENAI_API_KEY": openai_api_key.strip(),
        "SUPABASE_URL": supabase_url.strip(),
        "SUPABASE_SERVICE_ROLE_KEY": supabase_service_role_key.strip(),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except Exception:
        pass


def apply_runtime_secrets() -> None:
    """
    Load secrets from .cache/runtime_secrets.json (if present) into process env.
    This makes `--reload` resilient without writing `.env`.
    """
    secrets = load_runtime_secrets()
    for k, v in secrets.items():
        if not os.environ.get(k):
            os.environ[k] = v

