from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from drug_asset_discovery.config import RunConfig, repo_root


def _first_env(*names: str) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v and v.strip():
            return v.strip()
    return None


def get_git_commit_hash() -> str:
    """
    Best-effort git commit hash for reproducibility.

    - Prefers common CI env vars when present
    - Falls back to `git rev-parse HEAD` when `.git/` is available
    - Returns "unknown" if neither is available
    """
    env_hash = _first_env(
        "GIT_COMMIT",
        "GITHUB_SHA",
        "VERCEL_GIT_COMMIT_SHA",
        "RAILWAY_GIT_COMMIT_SHA",
        "RENDER_GIT_COMMIT",
    )
    if env_hash:
        return env_hash

    root = repo_root()
    if not (root / ".git").exists():
        return "unknown"
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        s = out.decode("utf-8", errors="replace").strip()
        return s or "unknown"
    except Exception:
        return "unknown"


def build_run_manifest(*, cfg: RunConfig, config_version: str, prompt_version: str) -> dict[str, Any]:
    """
    Runtime manifest persisted to Supabase for reproducibility.
    """
    tool_call_budgets = {
        "max_web_search_calls_per_worker_cycle": int(cfg.max_web_search_calls_per_worker_cycle),
        "max_planned_queries_per_cycle": int(cfg.max_planned_queries_per_cycle),
        "optional_fetch": {
            "enabled": bool(cfg.optional_fetch.enabled),
            "max_docs_per_round": int(cfg.optional_fetch.max_docs_per_round),
        },
        "validation": {
            "validation_batch_size_per_round": int(cfg.validation_batch_size_per_round),
            "max_total_validations": int(cfg.max_total_validations),
            "validation_concurrency": int(cfg.validation_concurrency),
        },
        "loop_mode": {
            "min_new_identifiers_per_cycle": int(cfg.min_new_identifiers_per_cycle),
            "patience_cycles": int(cfg.patience_cycles),
            "verify_top_k": int(cfg.verify_top_k),
            "verification_concurrency": int(cfg.verification_concurrency),
        },
    }
    return {
        "git_commit_hash": get_git_commit_hash(),
        "prompt_version": prompt_version,
        "config_version": config_version,
        "model": cfg.model,
        "reasoning_effort": dict(cfg.reasoning_effort),
        "tool_call_budgets": tool_call_budgets,
    }


