"""Compute u (upper bounds) from dataset/report.html and write dataset/u_values.csv.

u = ceil(mean + STD_MULT * std)

  STD_MULT = 0  →  mean만 사용
  STD_MULT = 1  →  mean + 1×std
  STD_MULT = 2  →  mean + 2×std
"""

from __future__ import annotations

import csv
import math
import re
import statistics

from collections import defaultdict
from pathlib import Path
from typing import Dict

# =============================================
# 수정 가능한 상수
# =============================================

STD_MULT = 6.0  # u = ceil(mean + STD_MULT * std)

REPORT_PATH = Path("dataset/report.html")
OUT_PATH     = Path("dataset/u_values.csv")

from data.preprocessing import leathers, pieces


def _normalize_piece_name(raw: str) -> str:
    s = str(raw or "").strip()
    if "__" in s:
        s = s.split("__", 1)[1].strip()
    return s.upper()


def parse_report_counts(report_path: Path) -> Dict[str, Dict[str, int]]:
    text = report_path.read_text(encoding="utf-8", errors="ignore")

    # manual-<LEATHER_ID> sections
    sec_re = re.compile(r'<div\s+id="manual-([A-Za-z0-9]+)"', flags=re.I)
    starts = [(m.start(), m.group(1).strip()) for m in sec_re.finditer(text)]
    if not starts:
        return {}

    # One legend entry (pattern) inside a section
    # We use the *placements* fraction (per-leather placed count).
    # Order in HTML: part-name appears before part-placements.
    part_re = re.compile(
        r'<div class="shape-legend-part-box">.*?'
        r'<div class="shape-legend-part-name">\s*([^<]+?)\s*</div>.*?'
        r'<div class="shape-legend-part-placements">.*?'
        r'<div class="fraction-numerator">(\d+)</div>',
        flags=re.S,
    )

    valid_pieces = set(pieces.keys())

    counts: Dict[str, Dict[str, int]] = defaultdict(dict)

    for idx, (pos, leather_id) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(text)
        chunk = text[pos:end]

        for raw_name, num_s in part_re.findall(chunk):
            piece = _normalize_piece_name(raw_name)
            if piece not in valid_pieces:
                continue
            try:
                num = int(num_s)
            except Exception:
                continue
            counts[leather_id][piece] = num

    return dict(counts)


def main() -> None:
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"report not found: {REPORT_PATH}")

    counts = parse_report_counts(REPORT_PATH)
    if not counts:
        raise ValueError("Could not parse any manual-<ID> sections from report.html")

    leather_ids = sorted(counts.keys())
    piece_list = sorted(pieces.keys())

    piece_mean: Dict[str, float] = {}
    piece_std: Dict[str, float] = {}
    piece_u: Dict[str, int] = {}

    for p in piece_list:
        vals = [int(counts.get(lid, {}).get(p, 0)) for lid in leather_ids]
        mean = (sum(vals) / len(vals)) if vals else 0.0
        std = statistics.stdev(vals) if len(vals) >= 2 else 0.0

        piece_mean[p] = float(mean)
        piece_std[p] = float(std)
        piece_u[p] = max(0, int(math.ceil(mean + STD_MULT * std)))

    # 패턴별 평균/std/u 요약
    avg_out = Path("outputs/report_piece_avg_u.csv")
    avg_out.parent.mkdir(parents=True, exist_ok=True)
    with avg_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["piece", "mean", "std", "u", "std_mult"])
        for p in piece_list:
            w.writerow([p, f"{piece_mean[p]:.4f}", f"{piece_std[p]:.4f}", piece_u[p], STD_MULT])

    # u_values.csv — (패턴, 가죽) 쌍별 u 값
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["piece", "leather", "u"])
        for leather_code in sorted(leathers.keys()):
            for p in piece_list:
                w.writerow([p, leather_code, piece_u[p]])

    print(f"STD_MULT = {STD_MULT}  →  u = ceil(mean + {STD_MULT}×std)")
    print(f"Leathers in report : {len(leather_ids)}")
    print(f"Pieces             : {len(piece_list)}")
    print(f"Wrote: {OUT_PATH}")
    print(f"Wrote: {avg_out}")


if __name__ == "__main__":
    main()

