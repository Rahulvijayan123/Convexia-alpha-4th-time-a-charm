from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from drug_asset_discovery.utils.hashing import safe_normalize, stable_sha256
from drug_asset_discovery.utils.identifiers import (
    IdentifierMentionClass,
    canonicalize_identifier,
    classify_identifier_mention,
)


MentionType = Literal[
    "drug_name",
    "drug_code",
    "sponsor",
    "target",
    "modality",
    "indication",
    "trial_id",
    "patent_id",
    "other",
]


class Mention(BaseModel):
    mention_type: MentionType
    raw_text: str
    normalized_text: str
    canonical_text: str
    mention_class: IdentifierMentionClass
    context: str | None = None
    source_url: str | None = None
    fingerprint: str

    @classmethod
    def from_raw(
        cls,
        *,
        mention_type: MentionType,
        raw_text: str,
        context: str | None = None,
        source_url: str | None = None,
    ) -> "Mention":
        normalized = safe_normalize(raw_text)
        canonical = canonicalize_identifier(raw_text)
        mclass = classify_identifier_mention(raw_text)
        fp = stable_sha256(f"{mention_type}|{normalized}")
        return cls(
            mention_type=mention_type,
            raw_text=raw_text,
            normalized_text=normalized,
            canonical_text=canonical,
            mention_class=mclass,
            context=context,
            source_url=source_url,
            fingerprint=fp,
        )


CandidateType = Literal["drug_asset", "trial_id", "patent_id", "other_identifier"]


class Candidate(BaseModel):
    candidate_type: CandidateType
    raw_identifier: str
    normalized_identifier: str
    canonical_identifier: str
    mention_class: IdentifierMentionClass
    fingerprint: str
    source_mention_fingerprint: str | None = None
    # Populated when stored
    id: str | None = None

    @classmethod
    def from_mention(cls, mention: Mention) -> "Candidate":
        # v1.4: candidate eligibility is controlled by mention typing (see orchestrator).
        if mention.mention_class == "drug_code_like":
            ctype: CandidateType = "drug_asset"
        elif mention.mention_class == "trial_id_like":
            ctype = "trial_id"
        elif mention.mention_class == "patent_id_like":
            ctype = "patent_id"
        else:
            ctype = "other_identifier"
        # v1.4: fingerprint dedupes formatting variants (e.g., CT-7439 vs CT7439).
        fp = stable_sha256(f"{ctype}|{mention.canonical_text}")
        return cls(
            candidate_type=ctype,
            raw_identifier=mention.raw_text,
            normalized_identifier=mention.normalized_text,
            canonical_identifier=mention.canonical_text,
            mention_class=mention.mention_class,
            fingerprint=fp,
            source_mention_fingerprint=mention.fingerprint,
        )


EvidenceSourceType = Literal[
    "patent",
    "trial",
    "pipeline_pdf",
    "paper",
    "vendor",
    "press_release",
    "other",
]

EnrichmentStatus = Literal["pending", "partial", "complete", "failed"]


class Citation(BaseModel):
    url: str
    snippet: str


class DraftAsset(BaseModel):
    """
    v1.5 draft asset:
    - Minimal, evidence-anchored discovery record (always keep; never drop for missing enrichment fields)
    - Deduped by identifier_canonical at the storage layer
    - Preserves raw variants as identifier_aliases_raw
    """

    identifier_raw: str
    identifier_canonical: str

    evidence_url: str
    evidence_snippet: str
    evidence_source_type: EvidenceSourceType

    discovered_by_worker_id: str | None = None
    discovered_by_cycle_id: str | None = None
    confidence_discovery: float = 0.0
    enrichment_status: EnrichmentStatus = "pending"

    sponsor: str | None = None
    target: str | None = None
    modality: str | None = None
    indication: str | None = None
    stage: str | None = None
    geography: str | None = None

    identifier_aliases_raw: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)

    @classmethod
    def from_minimal(
        cls,
        *,
        identifier_raw: str,
        evidence_url: str,
        evidence_snippet: str,
        evidence_source_type: EvidenceSourceType,
        discovered_by_worker_id: str | None = None,
        discovered_by_cycle_id: str | None = None,
        confidence_discovery: float = 0.0,
        identifier_aliases_raw: list[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> "DraftAsset":
        raw = (identifier_raw or "").strip()
        canon = canonicalize_identifier(raw)
        aliases = [a.strip() for a in (identifier_aliases_raw or []) if isinstance(a, str) and a.strip()]
        if raw and raw not in aliases:
            aliases.insert(0, raw)

        cites: list[Citation] = []
        if citations:
            for c in citations:
                if not isinstance(c, dict):
                    continue
                u = c.get("url")
                s = c.get("snippet")
                if isinstance(u, str) and u.strip() and isinstance(s, str) and s.strip():
                    cites.append(Citation(url=u.strip(), snippet=s.strip()))
        if evidence_url and evidence_snippet:
            # Always include the anchor evidence as a citation if present.
            cites.insert(0, Citation(url=str(evidence_url).strip(), snippet=str(evidence_snippet).strip()))

        return cls(
            identifier_raw=raw,
            identifier_canonical=canon,
            evidence_url=str(evidence_url).strip(),
            evidence_snippet=str(evidence_snippet).strip(),
            evidence_source_type=evidence_source_type,
            discovered_by_worker_id=discovered_by_worker_id,
            discovered_by_cycle_id=discovered_by_cycle_id,
            confidence_discovery=float(confidence_discovery or 0.0),
            enrichment_status="pending",
            sponsor=None,
            target=None,
            modality=None,
            indication=None,
            stage=None,
            geography=None,
            identifier_aliases_raw=aliases,
            citations=cites,
        )

    def completeness_score(self) -> int:
        fields = [self.sponsor, self.target, self.modality, self.indication, self.stage, self.geography]
        return sum(1 for v in fields if isinstance(v, str) and v.strip())


class ValidatedAsset(BaseModel):
    # v1.4 identifier contract (evidence-anchored)
    primary_identifier_raw: str
    primary_identifier_canonical: str
    identifier_aliases_raw: list[str] = Field(default_factory=list)
    evidence_snippet: str
    evidence_url: str
    evidence_source_type: EvidenceSourceType

    # v1.5: final assets are promoted at a completeness threshold, not "all fields required".
    sponsor: str | None = None
    target: str | None = None
    modality: str | None = None
    indication: str | None = None
    development_stage: str | None = None
    geography: str | None = None
    # Back-compat convenience for "source links"; should include evidence_url at minimum.
    sources: list[str] = Field(default_factory=list)
    fingerprint: str

    @classmethod
    def from_fields(
        cls,
        *,
        primary_identifier_raw: str,
        identifier_aliases_raw: list[str],
        evidence_snippet: str,
        evidence_url: str,
        evidence_source_type: EvidenceSourceType,
        sponsor: str | None,
        target: str | None,
        modality: str | None,
        indication: str | None,
        development_stage: str | None,
        geography: str | None,
        sources: list[str],
    ) -> "ValidatedAsset":
        # Preserve raw tokens; fingerprint only uses safe-normalized concatenation.
        primary_canon = canonicalize_identifier(primary_identifier_raw)
        sponsor_s = safe_normalize(str(sponsor or ""))
        target_s = safe_normalize(str(target or ""))
        modality_s = safe_normalize(str(modality or ""))
        indication_s = safe_normalize(str(indication or ""))
        stage_s = safe_normalize(str(development_stage or ""))
        geo_s = safe_normalize(str(geography or ""))
        fp = stable_sha256(
            "|".join(
                [
                    primary_canon,
                    sponsor_s,
                    target_s,
                    modality_s,
                    indication_s,
                    stage_s,
                    geo_s,
                ]
            )
        )
        return cls(
            primary_identifier_raw=primary_identifier_raw,
            primary_identifier_canonical=primary_canon,
            identifier_aliases_raw=identifier_aliases_raw,
            evidence_snippet=evidence_snippet,
            evidence_url=evidence_url,
            evidence_source_type=evidence_source_type,
            sponsor=(sponsor.strip() if isinstance(sponsor, str) and sponsor.strip() else None),
            target=(target.strip() if isinstance(target, str) and target.strip() else None),
            modality=(modality.strip() if isinstance(modality, str) and modality.strip() else None),
            indication=(indication.strip() if isinstance(indication, str) and indication.strip() else None),
            development_stage=(development_stage.strip() if isinstance(development_stage, str) and development_stage.strip() else None),
            geography=(geography.strip() if isinstance(geography, str) and geography.strip() else None),
            sources=sources,
            fingerprint=fp,
        )


