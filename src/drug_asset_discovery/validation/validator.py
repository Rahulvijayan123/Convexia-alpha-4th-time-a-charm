from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drug_asset_discovery.config import PromptBundle, RunConfig
from drug_asset_discovery.models.domain import Candidate, DraftAsset, EvidenceSourceType, ValidatedAsset
from drug_asset_discovery.openai_client import OpenAIResponsesClient
from drug_asset_discovery.retrieval.web_search import run_validation_web_search
from drug_asset_discovery.utils.hashing import stable_sha256
from drug_asset_discovery.utils.identifiers import canonicalize_identifier
from drug_asset_discovery.utils.json_extract import JSONExtractError, extract_first_json


@dataclass(frozen=True)
class ValidationResult:
    status: str  # validated | rejected | error
    assets: list[ValidatedAsset]
    rejected_reason: str | None
    evidence_urls: list[str]
    model_output: dict[str, Any] | None


def _is_evidence_anchored(
    *,
    primary_identifier_raw: str,
    evidence_snippet: str,
    evidence_url: str,
    cache_dir: Path,
) -> bool:
    """
    v1.4 hard rule:
    primary_identifier_raw must appear verbatim in evidence_snippet OR
    in stored fetched document text for evidence_url (if present in the local cache).
    """
    pid = (primary_identifier_raw or "").strip()
    if not pid:
        return False
    snip = evidence_snippet or ""
    if snip and pid in snip:
        return True

    # If Stage A optional_fetch fetched this URL, it will exist in the local cache.
    # This is "stored fetched document text" for evidence_url.
    if evidence_url:
        p = cache_dir / "documents" / f"{stable_sha256(evidence_url)}.txt"
        if p.exists():
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                txt = ""
            if txt and pid in txt:
                return True
    return False


def is_evidence_anchored_v15(*, identifier_raw: str, evidence_snippet: str) -> bool:
    """
    v1.5 minimal acceptance rule for draft_assets:
    - identifier_raw present
    - evidence_snippet present
    - evidence_snippet must contain identifier_raw OR a canonical-equivalent string
      (using canonicalize_identifier for both)
    """
    pid = (identifier_raw or "").strip()
    snip = (evidence_snippet or "").strip()
    if not pid or not snip:
        return False
    if pid in snip:
        return True
    pid_c = canonicalize_identifier(pid)
    if not pid_c:
        return False
    snip_c = canonicalize_identifier(snip)
    return bool(snip_c and pid_c in snip_c)


def _coerce_evidence_source_type(v: Any) -> EvidenceSourceType | None:
    allowed: set[str] = {"patent", "trial", "registry", "pipeline_pdf", "paper", "vendor", "press_release", "other"}
    s = str(v or "").strip()
    return s if s in allowed else None  # type: ignore[return-value]


def _all_required_fields_present_v14(a: dict[str, Any]) -> bool:
    required_str = [
        "primary_identifier_raw",
        "evidence_snippet",
        "evidence_url",
        "sponsor",
        "target",
        "modality",
        "indication",
        "development_stage",
        "geography",
    ]
    for k in required_str:
        if not isinstance(a.get(k), str) or not str(a.get(k)).strip():
            return False
    if _coerce_evidence_source_type(a.get("evidence_source_type")) is None:
        return False
    # identifier_aliases_raw should be a list if present; allow missing (we'll default).
    if "identifier_aliases_raw" in a and not isinstance(a.get("identifier_aliases_raw"), list):
        return False
    # sources should be list[str] if present; allow missing (we'll set to [evidence_url]).
    if "sources" in a and not isinstance(a.get("sources"), list):
        return False
    return True


def parse_validated_assets_v14(*, assets_raw: Any, cache_dir: Path) -> list[ValidatedAsset]:
    """
    Parse and validate Stage B outputs under the v1.4 contract.
    This is intentionally pure (no network calls) to enable unit tests.
    """
    if not isinstance(assets_raw, list) or not assets_raw:
        return []

    assets: list[ValidatedAsset] = []
    for a in assets_raw:
        if not isinstance(a, dict) or not _all_required_fields_present_v14(a):
            continue
        pid_raw = str(a["primary_identifier_raw"]).strip()
        ev_snip = str(a["evidence_snippet"]).strip()
        ev_url = str(a["evidence_url"]).strip()
        ev_type = _coerce_evidence_source_type(a.get("evidence_source_type"))
        if ev_type is None:
            continue

        if not _is_evidence_anchored(
            primary_identifier_raw=pid_raw,
            evidence_snippet=ev_snip,
            evidence_url=ev_url,
            cache_dir=cache_dir,
        ):
            continue

        aliases_raw: list[str] = []
        raw_aliases = a.get("identifier_aliases_raw", []) or []
        if isinstance(raw_aliases, list):
            for x in raw_aliases:
                if isinstance(x, str) and x.strip():
                    aliases_raw.append(x.strip())
        if pid_raw and pid_raw not in aliases_raw:
            aliases_raw.insert(0, pid_raw)

        sources = [s for s in (a.get("sources") or []) if isinstance(s, str) and s.strip()]
        if ev_url and ev_url not in sources:
            sources.insert(0, ev_url)

        assets.append(
            ValidatedAsset.from_fields(
                primary_identifier_raw=pid_raw,
                identifier_aliases_raw=aliases_raw,
                evidence_snippet=ev_snip,
                evidence_url=ev_url,
                evidence_source_type=ev_type,
                sponsor=str(a["sponsor"]).strip(),
                target=str(a["target"]).strip(),
                modality=str(a["modality"]).strip(),
                indication=str(a["indication"]).strip(),
                development_stage=str(a["development_stage"]).strip(),
                geography=str(a["geography"]).strip(),
                sources=sources,
            )
        )
    return assets


def draft_asset_from_minimal_evidence(
    *,
    identifier_candidate: str,
    evidence_url: str,
    evidence_snippet: str,
    evidence_source_type: EvidenceSourceType,
    discovered_by_worker_id: str | None = None,
    discovered_by_cycle_id: str | None = None,
) -> DraftAsset | None:
    """
    v1.5 Discovery validator (cheap, anchored):
    Convert (candidate, url, snippet) into a DraftAsset if minimal acceptance passes.
    """
    raw = (identifier_candidate or "").strip()
    url = (evidence_url or "").strip()
    snip = (evidence_snippet or "").strip()
    if not raw or not url or not snip:
        return None
    if not is_evidence_anchored_v15(identifier_raw=raw, evidence_snippet=snip):
        return None
    canon = canonicalize_identifier(raw)
    conf = 0.9 if raw in snip else 0.6
    return DraftAsset.from_minimal(
        identifier_raw=raw,
        evidence_url=url,
        evidence_snippet=snip,
        evidence_source_type=evidence_source_type,
        discovered_by_worker_id=discovered_by_worker_id,
        discovered_by_cycle_id=discovered_by_cycle_id,
        confidence_discovery=conf,
        identifier_aliases_raw=[raw],
        citations=[{"url": url, "snippet": snip}],
    )


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

    assets = parse_validated_assets_v14(assets_raw=assets_raw, cache_dir=cache_dir)

    if not assets:
        return ValidationResult(
            status="rejected",
            assets=[],
            rejected_reason="validated=true but no assets satisfied the v1.4 required fields + evidence anchoring",
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


