"""History-based UB (upper bound) for y[i,j].

This module loads past usage counts (or any precomputed UB) from a CSV and
provides lookup utilities.

Expected CSV columns (case-insensitive):
  - piece: piece/pattern name (e.g., "BACK WRAP SNAP")
  - leather: leather code (e.g., "L0696156")
  - ub: integer upper bound for y[piece, leather]

Notes:
- Names are normalized to upper-case and stripped.
- The solver still caps ub by demand.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


Key = Tuple[str, str]  # (PIECE, LEATHER)


@dataclass(frozen=True)
class HistoryUB:
    mapping: Dict[Key, int]

    def get(self, piece: str, leather: str) -> Optional[int]:
        k = (str(piece).strip().upper(), str(leather).strip().upper())
        return self.mapping.get(k)


def load_history_ub(csv_path: str | Path) -> HistoryUB:
    path = Path(csv_path)
    mapping: Dict[Key, int] = {}

    if not path.exists():
        return HistoryUB(mapping={})

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            piece = (row.get("piece") or row.get("Piece") or row.get("PIECE") or "").strip().upper()
            leather = (row.get("leather") or row.get("Leather") or row.get("LEATHER") or "").strip().upper()
            ub_raw = row.get("ub") or row.get("UB") or row.get("Ub") or row.get("qty") or row.get("QTY")

            if not piece or not leather or ub_raw is None:
                continue

            try:
                ub = int(float(str(ub_raw).strip()))
            except Exception:
                continue

            if ub < 0:
                ub = 0

            mapping[(piece, leather)] = ub

    return HistoryUB(mapping=mapping)
