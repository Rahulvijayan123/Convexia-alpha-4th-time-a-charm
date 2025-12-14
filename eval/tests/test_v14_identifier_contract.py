from __future__ import annotations

import sys
from pathlib import Path

# Allow running tests without requiring an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from drug_asset_discovery.utils.identifiers import canonicalize_identifier
from drug_asset_discovery.utils.identifiers import classify_identifier_mention, classify_identifier_type
from drug_asset_discovery.validation.validator import parse_validated_assets_v14


def test_canonicalize_identifier_zlc491_variants_match() -> None:
    assert canonicalize_identifier("ZLC-491") == canonicalize_identifier("ZLC491")


def test_canonicalize_identifier_ct7439_variants_match() -> None:
    assert canonicalize_identifier("CT-7439") == canonicalize_identifier("CT7439")


def test_evidence_anchoring_rejects_identifier_not_in_snippet(tmp_path) -> None:
    assets = parse_validated_assets_v14(
        assets_raw=[
            {
                "primary_identifier_raw": "CT-7439",
                "identifier_aliases_raw": [],
                "evidence_url": "https://example.com/evidence",
                # NOTE: missing hyphen, so the raw identifier does NOT appear verbatim.
                "evidence_snippet": "We evaluated CT7439 in preclinical models.",
                "evidence_source_type": "paper",
                "sponsor": "Example Pharma",
                "target": "Example Target",
                "modality": "small molecule",
                "indication": "Example indication",
                "development_stage": "preclinical",
                "geography": "global",
                "sources": ["https://example.com/evidence"],
            }
        ],
        cache_dir=tmp_path,
    )
    assert assets == []


def test_v15_identifier_typing_rejects_targets_and_stages() -> None:
    # Targets should never be treated as drug program identifiers.
    assert classify_identifier_type("CDK12") == "target_gene"
    assert classify_identifier_type("CDK12/13") == "target_gene"
    # Modality/stage phrases should never be treated as drug identifiers.
    assert classify_identifier_type("inhibitor") == "modality_phrase"
    assert classify_identifier_type("Phase 1") == "modality_phrase"


def test_v15_mention_classification_does_not_promote_gene_pairs_or_phase() -> None:
    assert classify_identifier_mention("CDK12/13") == "protein_gene_like"
    assert classify_identifier_mention("Phase 1") == "junk"


