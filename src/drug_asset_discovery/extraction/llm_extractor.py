from __future__ import annotations

from typing import Any

from drug_asset_discovery.config import PromptBundle, RunConfig
from drug_asset_discovery.models.domain import Mention, MentionType
from drug_asset_discovery.openai_client import OpenAIResponsesClient
from drug_asset_discovery.utils.json_extract import JSONExtractError, extract_first_json


def _coerce_mention_type(t: str) -> MentionType:
    allowed: set[str] = {
        "drug_name",
        "drug_code",
        "sponsor",
        "target",
        "modality",
        "indication",
        "trial_id",
        "patent_id",
        "other",
    }
    tt = (t or "").strip()
    return tt if tt in allowed else "other"  # type: ignore[return-value]


async def llm_extract_mentions(
    *,
    openai: OpenAIResponsesClient,
    cfg: RunConfig,
    prompts: PromptBundle,
    text: str,
    idempotency_key: str,
    source_url: str | None = None,
) -> list[Mention]:
    # Keep extraction cheap.
    clipped = text[:12000]
    prompt = prompts.recall_llm_extractor + "\n\nINPUT TEXT:\n" + clipped

    resp = await openai.create_response(
        model=cfg.model,
        input_text=prompt,
        reasoning_effort=cfg.reasoning_effort["extraction"],
        tools=None,
        idempotency_key=idempotency_key,
    )

    try:
        obj = extract_first_json(resp.output_text)
    except JSONExtractError:
        return []

    mentions: list[Mention] = []
    items = obj.get("mentions") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []

    for it in items:
        if not isinstance(it, dict):
            continue
        raw = it.get("raw")
        if not isinstance(raw, str) or not raw.strip():
            continue
        mtype = _coerce_mention_type(str(it.get("type") or "other"))
        ctx = it.get("context")
        context = ctx if isinstance(ctx, str) else None
        su = it.get("source_url")
        resolved_url = su.strip() if isinstance(su, str) and su.strip() else source_url
        mentions.append(
            Mention.from_raw(
                mention_type=mtype,
                raw_text=raw.strip(),
                context=context,
                source_url=resolved_url,
            )
        )

    # Dedup by fingerprint preserving order
    seen = set()
    out: list[Mention] = []
    for m in mentions:
        if m.fingerprint in seen:
            continue
        seen.add(m.fingerprint)
        out.append(m)
    return out


