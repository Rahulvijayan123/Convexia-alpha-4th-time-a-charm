from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class BenchmarkItem:
    canonical_id: str
    row_number: int
    row: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionItem:
    raw_id: str
    evidence_urls: list[str] = field(default_factory=list)
    cycle: int = 1
    raw: Any = None


@dataclass(frozen=True)
class MatchDetail:
    expected_id: str
    predicted_id: str
    predicted_raw_id: str
    match_type: str
    score: float
    cycle: int
    evidence_urls: list[str]


@dataclass(frozen=True)
class ExpectedOutcome:
    expected_id: str
    score: float
    match: Optional[MatchDetail]


@dataclass(frozen=True)
class PredictionOutcome:
    predicted_id: str
    predicted_raw_id: str
    score: float
    match: Optional[MatchDetail]
    cycle: int
    evidence_urls: list[str]
    raw: Any


