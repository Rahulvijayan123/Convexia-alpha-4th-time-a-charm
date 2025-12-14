from __future__ import annotations

from drug_asset_discovery.storage.supabase_store import SupabaseStore


async def replay_run(*, store: SupabaseStore, run_id: str) -> dict:
    found_assets = await store.get_draft_assets(run_id)
    validated_assets = await store.get_final_assets(run_id)
    return {
        "run_id": run_id,
        "status": "replay",
        # Back-compat
        "assets": validated_assets,
        # v1.6+ product semantics
        "found_assets": found_assets,
        "validated_assets": validated_assets,
    }


