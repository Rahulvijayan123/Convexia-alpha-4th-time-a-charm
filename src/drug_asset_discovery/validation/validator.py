from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drug_asset_discovery.config import PromptBundle, RunConfig
from drug_asset_discovery.models.domain import Candidate, ValidatedAsset
from drug_asset_discovery.openai_client import OpenAIResponsesClient
from drug_asset_discovery.retrieval.web_search import run_validation_web_search
from drug_asset_discovery.utils.json_extract import JSONExtractError, extract_first_json


@dataclass(frozen=True)
class ValidationResult:
    status: str  # validated | rejected | error
    assets: list[ValidatedAsset]
    rejected_reason: str | None
    evidence_urls: list[str]
    model_output: dict[str, Any] | None


def _all_required_fields_present(a: dict[str, Any]) -> bool:
    required = [
        "drug_name_code",
        "sponsor",
        "target",
        "modality",
        "indication",
        "development_stage",
        "geography",
        "sources",
    ]
    for k in required:
        if k not in a:
            return False
    if not isinstance(a.get("sources"), list) or len(a.get("sources")) == 0:
        return False
    for k in required[:-1]:
        if not isinstance(a.get(k), str) or not a.get(k).strip():
            return False
    return True


async def validate_candidate(
    *,
    openai: OpenAIResponsesClient,
    cfg: RunConfig,
    prompts: PromptBundle,
    cache_dir: Path,
    user_query: str,
    candidate: Candidate,
    idempotency_key: str,
) -> ValidationResult:
    prompt = (
        prompts.validator
        + "\n\n"
        + f"ORIGINAL USER QUERY:\n{user_query}\n\n"
        + f"CANDIDATE IDENTIFIER:\n{candidate.raw_identifier}\n\n"
        + "Use web search to validate. Return ONLY the strict JSON described above.\n"
    )

    resp = await run_validation_web_search(
        openai=openai,
        cfg=cfg,
        cache_dir=cache_dir,
        prompt=prompt,
        idempotency_key=idempotency_key,
    )
    evidence_urls = resp.extracted_urls()

    try:
        obj = extract_first_json(resp.output_text)
    except JSONExtractError as e:
        return ValidationResult(
            status="error",
            assets=[],
            rejected_reason=f"JSON parse error: {e}",
            evidence_urls=evidence_urls,
            model_output=None,
        )

    if not isinstance(obj, dict):
        return ValidationResult(
            status="error",
            assets=[],
            rejected_reason="Model output was not a JSON object",
            evidence_urls=evidence_urls,
            model_output=None,
        )

    validated = bool(obj.get("validated"))
    assets_raw = obj.get("assets")
    rejected_reason = obj.get("rejected_reason")
    rr = rejected_reason if isinstance(rejected_reason, str) else None

    if not validated:
        return ValidationResult(
            status="rejected",
            assets=[],
            rejected_reason=rr or "Not validated",
            evidence_urls=evidence_urls,
            model_output=obj,
        )

    if not isinstance(assets_raw, list) or not assets_raw:
        return ValidationResult(
            status="rejected",
            assets=[],
            rejected_reason="validated=true but assets[] missing/empty",
            evidence_urls=evidence_urls,
            model_output=obj,
        )

    assets: list[ValidatedAsset] = []
    for a in assets_raw:
        if not isinstance(a, dict) or not _all_required_fields_present(a):
            continue
        sources = [s for s in a.get("sources", []) if isinstance(s, str)]
        assets.append(
            ValidatedAsset.from_fields(
                drug_name_code=str(a["drug_name_code"]).strip(),
                sponsor=str(a["sponsor"]).strip(),
                target=str(a["target"]).strip(),
                modality=str(a["modality"]).strip(),
                indication=str(a["indication"]).strip(),
                development_stage=str(a["development_stage"]).strip(),
                geography=str(a["geography"]).strip(),
                sources=sources,
            )
        )

    if not assets:
        return ValidationResult(
            status="rejected",
            assets=[],
            rejected_reason="validated=true but no assets had all required fields",
            evidence_urls=evidence_urls,
            model_output=obj,
        )

    return ValidationResult(
        status="validated",
        assets=assets,
        rejected_reason=None,
        evidence_urls=evidence_urls,
        model_output=obj,
    )


