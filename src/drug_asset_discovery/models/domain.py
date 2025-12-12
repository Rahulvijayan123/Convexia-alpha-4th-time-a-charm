from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from drug_asset_discovery.utils.hashing import safe_normalize, stable_sha256


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
        fp = stable_sha256(f"{mention_type}|{normalized}")
        return cls(
            mention_type=mention_type,
            raw_text=raw_text,
            normalized_text=normalized,
            context=context,
            source_url=source_url,
            fingerprint=fp,
        )


CandidateType = Literal["drug_asset", "trial_id", "patent_id", "other_identifier"]


class Candidate(BaseModel):
    candidate_type: CandidateType
    raw_identifier: str
    normalized_identifier: str
    fingerprint: str
    source_mention_fingerprint: str | None = None
    # Populated when stored
    id: str | None = None

    @classmethod
    def from_mention(cls, mention: Mention) -> "Candidate":
        # Keep it broad; Stage 2 will be strict.
        if mention.mention_type in ("drug_name", "drug_code"):
            ctype: CandidateType = "drug_asset"
        elif mention.mention_type == "trial_id":
            ctype = "trial_id"
        elif mention.mention_type == "patent_id":
            ctype = "patent_id"
        else:
            ctype = "other_identifier"
        fp = stable_sha256(f"{ctype}|{mention.normalized_text}")
        return cls(
            candidate_type=ctype,
            raw_identifier=mention.raw_text,
            normalized_identifier=mention.normalized_text,
            fingerprint=fp,
            source_mention_fingerprint=mention.fingerprint,
        )


class ValidatedAsset(BaseModel):
    drug_name_code: str
    sponsor: str
    target: str
    modality: str
    indication: str
    development_stage: str
    geography: str
    sources: list[str] = Field(default_factory=list)
    fingerprint: str

    @classmethod
    def from_fields(
        cls,
        *,
        drug_name_code: str,
        sponsor: str,
        target: str,
        modality: str,
        indication: str,
        development_stage: str,
        geography: str,
        sources: list[str],
    ) -> "ValidatedAsset":
        # Preserve raw tokens; fingerprint only uses safe-normalized concatenation.
        fp = stable_sha256(
            "|".join(
                [
                    safe_normalize(drug_name_code),
                    safe_normalize(sponsor),
                    safe_normalize(target),
                    safe_normalize(modality),
                    safe_normalize(indication),
                    safe_normalize(development_stage),
                    safe_normalize(geography),
                ]
            )
        )
        return cls(
            drug_name_code=drug_name_code,
            sponsor=sponsor,
            target=target,
            modality=modality,
            indication=indication,
            development_stage=development_stage,
            geography=geography,
            sources=sources,
            fingerprint=fp,
        )


