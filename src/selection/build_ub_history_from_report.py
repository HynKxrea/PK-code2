"""Build history-based UB from dataset/report.html.

Parses the nesting report and extracts, for each leather (manual-<ID> section),
how many of each pattern was nested.

Important:
- The per-leather placed count is taken from the first fraction numerator under
  `shape-legend-part-placements` (NOT the `shape-legend-part-remaining`, which
  typically represents global totals like placed/required across the whole job).

Then computes the mean count per pattern across leathers in the report and
writes dataset/ub_history.csv as a *per-leather* UB table for the current
solver leathers.

Usage:
  PYTHONPATH=src python3 -m selection.build_ub_history_from_report \
    --report dataset/report.html \
    --out dataset/ub_history.csv

By default, it will also write a small summary CSV under outputs/.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build ub_history.csv from dataset/report.html")
    ap.add_argument("--report", default="dataset/report.html", help="Path to report.html")
    ap.add_argument("--out", default="dataset/ub_history.csv", help="Output UB CSV path")
    ap.add_argument(
        "--matrix-out",
        default="outputs/report_piece_counts_by_leather.csv",
        help="Optional matrix CSV (leather rows, piece columns)",
    )
    ap.add_argument(
        "--avg-out",
        default="outputs/report_piece_avg_ub.csv",
        help="Optional per-piece mean UB CSV",
    )

    args = ap.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.exists():
        raise FileNotFoundError(f"report not found: {report_path}")

    counts = parse_report_counts(report_path)
    if not counts:
        raise ValueError("Could not parse any manual-<ID> sections from report.html")

    leather_ids = sorted(counts.keys())
    piece_list = sorted(pieces.keys())

    # Compute per-piece stats across leathers in the report
    piece_mean: Dict[str, float] = {}
    piece_std: Dict[str, float] = {}
    piece_ub: Dict[str, int] = {}

    for p in piece_list:
        vals = [int(counts.get(lid, {}).get(p, 0)) for lid in leather_ids]
        mean = (sum(vals) / len(vals)) if vals else 0.0
        # sample std dev (n-1)
        std = statistics.stdev(vals) if len(vals) >= 2 else 0.0

        piece_mean[p] = float(mean)
        piece_std[p] = float(std)

        # Use floor so we don't round back up.
        ub = int(math.floor(mean + 0.7 * std))  # default to demand if not found
        piece_ub[p] = max(0, ub)

    # Write matrix CSV (report leather IDs)
    matrix_out = Path(args.matrix_out)
    matrix_out.parent.mkdir(parents=True, exist_ok=True)
    with matrix_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["leather_id", *piece_list])
        for lid in leather_ids:
            row = [lid] + [counts.get(lid, {}).get(p, 0) for p in piece_list]
            w.writerow(row)

    # Write piece averages
    avg_out = Path(args.avg_out)
    avg_out.parent.mkdir(parents=True, exist_ok=True)
    with avg_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["piece", "mean_count_per_leather", "std_count_per_leather", "ub"])
        for p in piece_list:
            w.writerow([p, f"{piece_mean[p]:.6f}", f"{piece_std[p]:.6f}", piece_ub[p]])

    # Write ub_history.csv for *current solver leathers*.
    # This makes history UB usable even if the report leather IDs do not match our dataset.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["piece", "leather", "ub"])
        for leather_code in sorted(leathers.keys()):
            for p in piece_list:
                w.writerow([p, leather_code, piece_ub[p]])

    print(f"Parsed leathers in report: {len(leather_ids)}")
    print(f"Pieces (from pieces loader): {len(piece_list)}")
    print(f"Wrote matrix: {matrix_out}")
    print(f"Wrote per-piece mean UB: {avg_out}")
    print(f"Wrote ub_history: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

