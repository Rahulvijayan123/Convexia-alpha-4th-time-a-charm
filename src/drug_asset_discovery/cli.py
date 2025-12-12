from __future__ import annotations

import argparse
import asyncio
import json

from drug_asset_discovery.config import EnvSettings
from drug_asset_discovery.logging import configure_logging
from drug_asset_discovery.orchestrator.orchestrator import run_discovery
from drug_asset_discovery.storage.replay import replay_run
from drug_asset_discovery.storage.supabase_store import SupabaseStore


def _cmd_run(args: argparse.Namespace) -> None:
    env = EnvSettings()
    configure_logging(env.log_level)
    # Pre-flight health check: verify Supabase connectivity if credentials provided
    if env.supabase_url and env.supabase_service_role_key:
        from drug_asset_discovery.storage.supabase_store import SupabaseStore
        store = SupabaseStore(url=env.supabase_url, service_role_key=env.supabase_service_role_key)
        try:
            connected = asyncio.run(store.health_check())
        except Exception:
            connected = False
        if not connected:
            raise SystemExit("Failed to connect to Supabase with provided credentials. Aborting run.")
    result = asyncio.run(
        run_discovery(
            user_query=args.query,
            config_version=args.config_version or env.default_config_version,
            prompt_version=args.prompt_version or env.default_prompt_version,
            idempotency=args.idempotency_key,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_replay(args: argparse.Namespace) -> None:
    env = EnvSettings()
    configure_logging(env.log_level)
    if not env.supabase_url or not env.supabase_service_role_key:
        raise SystemExit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY required for replay")
    store = SupabaseStore(url=env.supabase_url, service_role_key=env.supabase_service_role_key)
    payload = asyncio.run(replay_run(store=store, run_id=args.run_id))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(prog="drug-asset-discovery")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run a new discovery job (stores to Supabase).")
    pr.add_argument("query", type=str)
    pr.add_argument("--config-version", type=str, default=None)
    pr.add_argument("--prompt-version", type=str, default=None)
    pr.add_argument("--idempotency-key", type=str, default=None)
    pr.set_defaults(func=_cmd_run)

    pp = sub.add_parser("replay", help="Deterministically replay a stored run from Supabase.")
    pp.add_argument("run_id", type=str)
    pp.set_defaults(func=_cmd_replay)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


