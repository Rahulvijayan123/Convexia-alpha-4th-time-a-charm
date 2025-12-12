from __future__ import annotations

import re
from collections.abc import Iterable

from drug_asset_discovery.models.domain import Mention


_NCT = re.compile(r"\bNCT\d{8}\b")
_ISRCTN = re.compile(r"\bISRCTN\d{8}\b")
_EUDRACT = re.compile(r"\b\d{4}-\d{6}-\d{2}\b")
_ACTRN = re.compile(r"\bACTRN\d{14}\b")
_CTRI = re.compile(r"\bCTRI/\d{4}/\d{2}/\d{6}\b")

_WO = re.compile(r"\bWO\d{4}\d{6,}\w*\b")
_US_PAT = re.compile(r"\bUS\d{4,}\w*\b")
_EP = re.compile(r"\bEP\d{4,}\w*\b")

# Drug code patterns (intentionally broad; Stage 1 recall is reckless)
_CODE_DASH = re.compile(r"\b[A-Z]{1,6}-\d{2,7}[A-Z]?\b")
_CODE_ALNUM = re.compile(r"\b[A-Z]{2,6}\d{2,7}\b")
_CODE_MIXED = re.compile(r"\b[A-Z]{1,4}\d{2,5}[A-Z]{1,4}\d{0,5}\b")


def _context_window(text: str, start: int, end: int, window: int = 80) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right].replace("\n", " ").strip()


def regex_harvest_mentions(text: str, source_url: str | None = None) -> list[Mention]:
    mentions: list[Mention] = []

    def add(pattern: re.Pattern[str], mtype: str) -> None:
        for m in pattern.finditer(text):
            raw = m.group(0)
            ctx = _context_window(text, m.start(), m.end())
            mentions.append(Mention.from_raw(mention_type=mtype, raw_text=raw, context=ctx, source_url=source_url))

    add(_NCT, "trial_id")
    add(_ISRCTN, "trial_id")
    add(_EUDRACT, "trial_id")
    add(_ACTRN, "trial_id")
    add(_CTRI, "trial_id")

    add(_WO, "patent_id")
    add(_US_PAT, "patent_id")
    add(_EP, "patent_id")

    add(_CODE_DASH, "drug_code")
    add(_CODE_ALNUM, "drug_code")
    add(_CODE_MIXED, "drug_code")

    # Dedup by fingerprint preserving order
    seen = set()
    out: list[Mention] = []
    for ment in mentions:
        if ment.fingerprint in seen:
            continue
        seen.add(ment.fingerprint)
        out.append(ment)
    return out


