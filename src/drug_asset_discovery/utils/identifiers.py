from __future__ import annotations

import re
from typing import Literal

# v1.4: typed identifier mentions used to control candidate eligibility and diagnostics.
IdentifierMentionClass = Literal[
    "drug_code_like",
    "patent_id_like",
    "trial_id_like",
    "protein_gene_like",
    "junk",
]

IdentifierType = Literal["drug_code", "target_gene", "modality_phrase", "trial_id", "company", "other"]


def canonicalize_identifier(value: str) -> str:
    """
    v1.4 canonicalization contract (used everywhere):
    - uppercase
    - remove whitespace
    - remove punctuation separators (- – _ / . , : ; ( ) [ ] { })
    - keep only A-Z and 0-9

    Implementation note: filtering to ASCII A-Z0-9 after uppercasing satisfies the above.
    """
    if value is None:
        return ""
    s = str(value).upper()
    # Keep only ASCII A-Z and 0-9.
    return "".join(ch for ch in s if ("A" <= ch <= "Z") or ("0" <= ch <= "9"))


# Trial registry identifiers (raw text variants)
_TRIAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bNCT\d{8}\b", re.IGNORECASE),
    re.compile(r"\bISRCTN\d{8}\b", re.IGNORECASE),
    re.compile(r"\bACTRN\d{14}\b", re.IGNORECASE),
    re.compile(r"\bCTRI/\d{4}/\d{2}/\d{6}\b", re.IGNORECASE),
    re.compile(r"\bEUCTR\d{4}-\d{6}-\d{2}\b", re.IGNORECASE),
    # EudraCT number without EUCTR prefix
    re.compile(r"\b\d{4}-\d{6}-\d{2}\b"),
)


def _is_trial_id_like(raw: str) -> bool:
    s = (raw or "").strip()
    if not s:
        return False
    return any(p.search(s) for p in _TRIAL_PATTERNS)


def _is_patent_id_like(raw: str) -> bool:
    c = canonicalize_identifier(raw)
    if not c:
        return False
    # Common publication prefixes and PCT formats (kept broad; Stage B still validates).
    if c.startswith("WO") and len(c) >= 10 and any(ch.isdigit() for ch in c[2:]):
        return True
    if c.startswith("EP") and len(c) >= 8 and any(ch.isdigit() for ch in c[2:]):
        return True
    if c.startswith("US") and len(c) >= 8 and any(ch.isdigit() for ch in c[2:]):
        return True
    if c.startswith("PCT") and any(ch.isdigit() for ch in c):
        return True
    return False


# Common gene/protein families where tokens like "CDK2"/"BRCA1" are almost never drug/program codes.
# Keep intentionally generic; do NOT benchmark-tune.
_GENE_PREFIX_DIGITS_1_2 = (
    "CDK",
    "BRCA",
    "ERBB",
    "HER",
    "IL",
    "TNF",
    "TGFB",
    "JAK",
    "STAT",
    "FGFR",
    "VEGF",
    "PDGF",
    "BRAF",
    "KRAS",
    "NRAS",
    "HRAS",
    "AKT",
    "MEK",
    "MAPK",
    "MTOR",
    "PARP",
    "ATR",
    "ATM",
    "KIT",
    "MET",
    "RET",
    "ALK",
    "NTRK",
    "CTLA",
    "PDCD",
    "PDL",
)
_GENE_PREFIX_RE = re.compile(rf"^({'|'.join(_GENE_PREFIX_DIGITS_1_2)})\d{{1,2}}$")

# A small stoplist of very common lab/regulatory tokens that match "letters+digits"
# but are not drug/program identifiers.
_COMMON_JUNK_TOKENS = {
    "IC50",
    "EC50",
    "ED50",
    "LD50",
    "KD",
    "KI",
    "PK",
    "PD",
    "ADME",
    "DMPK",
}

_COMMON_GENE_SYMBOLS_ALPHA_ONLY = {
    # Keep intentionally small and generic (common oncology/immunology targets).
    "EGFR",
    "KRAS",
    "NRAS",
    "HRAS",
    "BRAF",
    "ALK",
    "MET",
    "RET",
    "KIT",
    "MTOR",
    "PARP",
    "VEGF",
    "PDGF",
    "JAK",
    "STAT",
    "TNF",
}


def _is_protein_gene_like(raw: str) -> bool:
    c = canonicalize_identifier(raw)
    if not c:
        return False
    # Explicit common families with 1–2 digits (e.g., CDK2, BRCA1, IL6, JAK2)
    if _GENE_PREFIX_RE.match(c):
        return True
    # Common cluster-of-differentiation markers (e.g., CD19, CD3)
    if re.match(r"^CD\d{1,2}$", c):
        return True
    # Plain-letter common gene/protein symbols.
    if c.isalpha() and c in _COMMON_GENE_SYMBOLS_ALPHA_ONLY:
        return True
    return False


def _is_gene_pair_like(raw: str) -> bool:
    """
    Gene/target shorthand like "CDK12/13" or "CDK12-13" should never be treated as drug codes.
    Keep intentionally generic; do NOT benchmark-tune.
    """
    s = (raw or "").strip()
    if not s:
        return False
    if "/" not in s and "-" not in s and "\u2212" not in s:  # minus sign
        return False
    # Normalize common separators to "/"
    s2 = s.replace("\u2212", "-")
    # Pattern: PREFIX + d{1,2} + sep + d{1,2} (e.g., CDK12/13)
    m = re.match(r"^([A-Za-z]{2,12})(\d{1,2})\s*[/\-]\s*(\d{1,2})$", s2)
    if m:
        prefix = m.group(1)
        left = f"{prefix}{m.group(2)}"
        right = f"{prefix}{m.group(3)}"
        return _is_protein_gene_like(left) and _is_protein_gene_like(right)
    # Pattern: PREFIXd{1,2} sep PREFIXd{1,2} (e.g., JAK1/JAK2)
    m2 = re.match(r"^([A-Za-z]{2,12}\d{1,2})\s*[/\-]\s*([A-Za-z]{2,12}\d{1,2})$", s2)
    if m2:
        return _is_protein_gene_like(m2.group(1)) and _is_protein_gene_like(m2.group(2))
    return False


_MODALITY_OR_STAGE_HINTS = (
    "inhibitor",
    "agonist",
    "antagonist",
    "antibody",
    "mab",
    "adc",
    "vaccin",
    "cell therapy",
    "gene therapy",
    "sirna",
    "rnai",
    "oligonucleotide",
    "degrader",
    "glue degrader",
    "molecular glue",
    "protac",
    "phase",
    "first patient dosed",
)

_MODALITY_SINGLE_TOKENS = {
    "inhibitor",
    "agonist",
    "antagonist",
    "antibody",
    "degrader",
    "protac",
    "sirna",
    "rnai",
    "vaccine",
    "adc",
}


def _looks_like_modality_or_stage_phrase(raw: str) -> bool:
    s = (raw or "").strip()
    if not s:
        return False
    # Single-token "ADC-1234" etc should not be treated as modality phrases just due to substring matches.
    has_space = any(ch.isspace() for ch in s)
    s_low = s.lower()
    if has_space:
        return any(h in s_low for h in _MODALITY_OR_STAGE_HINTS)
    # Single-token modality labels.
    if s_low in _MODALITY_SINGLE_TOKENS:
        return True
    # Single-token stage strings like "Phase1" should be filtered too.
    c = canonicalize_identifier(s)
    if re.match(r"^PHASE\d{1,2}[A-Z]?$", c):
        return True
    return False


_COMPANY_HINTS = (
    "inc",
    "ltd",
    "llc",
    "gmbh",
    "plc",
    "ag",
    "sa",
    "s.a",
    "sas",
    "corp",
    "corporation",
    "pharma",
    "pharmaceutical",
    "therapeutics",
    "biotech",
    "bio-tech",
    "laboratories",
    "labs",
    "holdings",
)


def _looks_like_company(raw: str) -> bool:
    s = (raw or "").strip()
    if not s:
        return False
    if any(ch.isdigit() for ch in s):
        return False
    s_low = s.lower()
    # Corporate suffix/hint words.
    for h in _COMPANY_HINTS:
        if h in s_low:
            return True
    return False


def classify_identifier_type(text: str) -> IdentifierType:
    """
    v1.5 domain-agnostic identifier typing gate.

    Output categories:
    - drug_code: plausible program/compound identifier token
    - target_gene: gene/protein/target term (incl. gene pairs like CDK12/13)
    - modality_phrase: modality/stage phrase (e.g., "inhibitor", "glue degrader", "Phase 1")
    - trial_id: trial registry identifier (e.g., NCT########)
    - company: organization/company name
    - other: everything else

    IMPORTANT:
    - Target terms (CDK12, CDK12/13), modality labels ("inhibitor"), and stages ("Phase 1") must never be drug_code.
    """
    s = (text or "").strip()
    if not s:
        return "other"
    if _is_trial_id_like(s):
        return "trial_id"
    if _looks_like_modality_or_stage_phrase(s):
        return "modality_phrase"
    if _looks_like_company(s):
        return "company"
    if _is_patent_id_like(s):
        return "other"
    if _is_protein_gene_like(s) or _is_gene_pair_like(s):
        return "target_gene"
    c = canonicalize_identifier(s)
    if not c:
        return "other"
    # Explicit junk/stage-like tokens that look code-like (letters+digits) but are not identifiers.
    if c in _COMMON_JUNK_TOKENS or re.match(r"^PHASE\d{1,2}[A-Z]?$", c):
        return "other"
    # Drug/program code-like: alphanumeric, 4–20, contains at least one digit and one letter.
    if 4 <= len(c) <= 20 and any(ch.isdigit() for ch in c) and any(ch.isalpha() for ch in c):
        return "drug_code"
    return "other"


def classify_identifier_mention(raw: str) -> IdentifierMentionClass:
    """
    v1.4 mention typing.

    Only 'drug_code_like' and 'patent_id_like' are eligible to become candidates for Stage B.
    Everything else stays in mentions for traceability.
    """
    s = (raw or "").strip()
    if not s:
        return "junk"

    if _is_trial_id_like(s):
        return "trial_id_like"
    if _is_patent_id_like(s):
        return "patent_id_like"

    # Protein/gene terms should not consume validation budget.
    if _is_protein_gene_like(s) or _is_gene_pair_like(s):
        return "protein_gene_like"

    c = canonicalize_identifier(s)
    if not c:
        return "junk"

    if c in _COMMON_JUNK_TOKENS:
        return "junk"
    # Stage-like tokens (e.g., "Phase 1") must never be treated as drug identifiers.
    if re.match(r"^PHASE\d{1,2}[A-Z]?$", c):
        return "junk"

    # Modality/stage phrases that slipped through as short tokens.
    if _looks_like_modality_or_stage_phrase(s):
        return "junk"

    # Drug/program code-like: alphanumeric, 4–20, contains at least one digit and one letter.
    if 4 <= len(c) <= 20 and any(ch.isdigit() for ch in c) and any(ch.isalpha() for ch in c):
        return "drug_code_like"

    return "junk"


