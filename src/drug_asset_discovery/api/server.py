from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from drug_asset_discovery.config import EnvSettings
from drug_asset_discovery.logging import configure_logging
from drug_asset_discovery.orchestrator.orchestrator import run_discovery
from drug_asset_discovery.storage.replay import replay_run
from drug_asset_discovery.storage.supabase_store import SupabaseStore

logger = logging.getLogger(__name__)

app = FastAPI(title="Drug Asset Discovery", version="0.1.0")


class RunRequest(BaseModel):
    query: str = Field(min_length=3)
    config_version: str | None = None
    prompt_version: str | None = None
    replay: bool = False
    replay_run_id: str | None = None


class RunResponse(BaseModel):
    run_id: str
    status: str
    summary: dict[str, Any] | None = None
    assets: list[dict[str, Any]] = Field(default_factory=list)


@app.on_event("startup")
async def _startup() -> None:
    env = EnvSettings()
    configure_logging(env.log_level)


@app.post("/run", response_model=RunResponse)
async def post_run(
    body: RunRequest,
    x_idempotency_key: str | None = Header(default=None, alias="X-Idempotency-Key"),
) -> RunResponse:
    env = EnvSettings()
    config_version = body.config_version or env.default_config_version
    prompt_version = body.prompt_version or env.default_prompt_version

    if body.replay:
        if not body.replay_run_id:
            raise HTTPException(status_code=400, detail="replay_run_id is required when replay=true")
        if not env.supabase_url or not env.supabase_service_role_key:
            raise HTTPException(status_code=500, detail="Supabase env not configured")
        store = SupabaseStore(url=env.supabase_url, service_role_key=env.supabase_service_role_key)
        payload = await replay_run(store=store, run_id=body.replay_run_id)
        return RunResponse(run_id=payload["run_id"], status=payload["status"], summary=None, assets=payload["assets"])

    try:
        result = await run_discovery(
            user_query=body.query,
            config_version=config_version,
            prompt_version=prompt_version,
            idempotency=x_idempotency_key,
        )
        return RunResponse(
            run_id=result["run_id"],
            status=result["status"],
            summary=result.get("summary"),
            assets=result.get("assets", []),
        )
    except Exception as e:
        logger.exception("run failed")
        raise HTTPException(status_code=500, detail=str(e))


