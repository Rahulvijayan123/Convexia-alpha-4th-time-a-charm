from __future__ import annotations

from drug_asset_discovery.storage.supabase_store import SupabaseStore


async def replay_run(*, store: SupabaseStore, run_id: str) -> dict:
    assets = await store.get_final_assets(run_id)
    return {"run_id": run_id, "assets": assets, "status": "replay"}


