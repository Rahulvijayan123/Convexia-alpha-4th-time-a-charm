from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional

from eval.types import BenchmarkItem


def _norm_header(h: str) -> str:
    # Lowercase + remove non-alphanumerics to be tolerant to headers like "Asset Name / Code"
    return "".join(ch for ch in (h or "").strip().lower() if ch.isalnum())


_DEFAULT_ID_HEADER_CANDIDATES = (
    "identifier",
    "id",
    "assetid",
    "assetcode",
    "assetnamecode",
    "assetname",
    "asset",
    "code",
    "assetnamecode",  # duplicate ok
    "assetnamecode",  # keep simple
)


def _pick_id_column(fieldnames: Iterable[str]) -> Optional[str]:
    if not fieldnames:
        return None
    by_norm = {_norm_header(h): h for h in fieldnames if h}
    for cand in _DEFAULT_ID_HEADER_CANDIDATES:
        if cand in by_norm:
            return by_norm[cand]
    return None


def load_benchmark_csv(path: str | Path, id_column: str | None = None) -> List[BenchmarkItem]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {p}")

    items: List[BenchmarkItem] = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Benchmark CSV has no header row.")

        resolved_id_column = id_column or _pick_id_column(reader.fieldnames)
        if not resolved_id_column:
            raise ValueError(
                "Could not infer benchmark identifier column. "
                f"Headers: {reader.fieldnames}. "
                "Pass --benchmark_id_column explicitly."
            )

        for idx, row in enumerate(reader, start=2):  # header is line 1
            raw_id = (row.get(resolved_id_column) or "").strip()
            if not raw_id:
                continue
            items.append(BenchmarkItem(canonical_id=raw_id, row_number=idx, row=row))

    if not items:
        raise ValueError("Benchmark CSV yielded 0 benchmark items (after dropping empty identifier rows).")
    return items


