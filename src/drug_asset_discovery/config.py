from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")

    supabase_url: str | None = Field(default=None, alias="SUPABASE_URL")
    supabase_service_role_key: str | None = Field(default=None, alias="SUPABASE_SERVICE_ROLE_KEY")

    local_cache_dir: str = Field(default=".cache", alias="LOCAL_CACHE_DIR")

    default_config_version: str = Field(default="v0.1", alias="DEFAULT_CONFIG_VERSION")
    default_prompt_version: str = Field(default="v0.1", alias="DEFAULT_PROMPT_VERSION")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]


class RetrievalConfig(BaseModel):
    tool: Literal["web_search"] = "web_search"
    external_web_access: bool = True
    # If provided, should match OpenAI docs: {country, city, region, timezone}
    user_location: dict[str, Any] | None = None


class OptionalFetchConfig(BaseModel):
    enabled: bool = False
    max_docs_per_round: int = 3
    timeout_seconds: int = 20


class TimeoutsConfig(BaseModel):
    openai_seconds: int = 90


class RunConfig(BaseModel):
    version: str
    model: str = "gpt-5.2"
    workers: int = 4
    max_rounds: int = 8
    stop_after_no_new_validated_rounds: int = 3
    min_successful_workers_per_round: int = 2
    max_planned_queries_per_cycle: int = 4
    max_frontier_size: int = 30
    validation_batch_size_per_round: int = 12
    max_total_validations: int = 80
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    optional_fetch: OptionalFetchConfig = Field(default_factory=OptionalFetchConfig)
    reasoning_effort: dict[str, ReasoningEffort] = Field(
        default_factory=lambda: {
            "worker_planning": "xhigh",
            "validation": "xhigh",
            "extraction": "low",
        }
    )
    timeouts: TimeoutsConfig = Field(default_factory=TimeoutsConfig)


@dataclass(frozen=True)
class PromptBundle:
    version: str
    worker_planner: str
    recall_llm_extractor: str
    validator: str


def repo_root() -> Path:
    # src/drug_asset_discovery/config.py -> repo root
    return Path(__file__).resolve().parents[2]


def load_config(version: str) -> RunConfig:
    cfg_path = repo_root() / "configs" / f"{version}.json"
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = RunConfig.model_validate(raw)
    if cfg.model != "gpt-5.2":
        raise ValueError(f"Non-negotiable: model must be 'gpt-5.2' (got {cfg.model!r})")
    if cfg.workers < 3 or cfg.workers > 4:
        raise ValueError("Non-negotiable: Loop Mode requires 3 to 4 workers")
    return cfg


def load_prompts(version: str) -> PromptBundle:
    base = repo_root() / "prompts" / version
    return PromptBundle(
        version=version,
        worker_planner=(base / "worker_planner.md").read_text(encoding="utf-8"),
        recall_llm_extractor=(base / "recall_llm_extractor.md").read_text(encoding="utf-8"),
        validator=(base / "validator.md").read_text(encoding="utf-8"),
    )


