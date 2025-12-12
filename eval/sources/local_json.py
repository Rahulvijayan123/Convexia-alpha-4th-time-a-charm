from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.sources.common import extract_predictions
from eval.types import PredictionItem


def load_predictions_json(
    path: str | Path,
    *,
    id_key: Optional[str] = None,
    evidence_key: Optional[str] = None,
    cycle_key: Optional[str] = None,
) -> List[PredictionItem]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Predictions JSON not found: {p}")

    data: Any = json.loads(p.read_text(encoding="utf-8"))
    preds = extract_predictions(data, id_key=id_key, evidence_key=evidence_key, cycle_key=cycle_key)
    return preds


