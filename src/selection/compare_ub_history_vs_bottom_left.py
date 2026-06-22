"""Compare history UB vs bottom-left UB without solving the MIP.

This script:
- loads current pieces/leathers/demand via data.preprocessing
- computes required quality per piece from piece_catalog_area_rule.csv
- loads history UB from dataset/ub_history.csv (or a provided path)
- computes bottom-left UB via bottom_up.bottom_up_heuristic.max_pieces_for_pattern_on_leather
- prints summary counts and the most-tightened (piece, leather) pairs.

Usage:
  PYTHONPATH=src python3 -m selection.compare_ub_history_vs_bottom_left \
    --history dataset/ub_history.csv \
    --top 30

Notes:
- This may take some time because bottom-left UB rasterizes leather preview images.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

from bottom_up.bottom_up_heuristic import max_pieces_for_pattern_on_leather
from data.preprocessing import demand, leathers, merge_data, pieces
from selection.selection import _load_piece_required_quality_map
from selection.ub_history import load_history_ub


@dataclass(frozen=True)
class Row:
    piece: str
    leather: str
    ub_hist: int
    ub_bottom_left: int
    demand: int

    @property
    def delta(self) -> int:
        # positive means history is tighter (smaller)
        return int(self.ub_bottom_left) - int(self.ub_hist)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare history UB vs bottom-left UB")
    ap.add_argument("--history", default=os.path.join("dataset", "ub_history.csv"), help="History UB csv path")
    ap.add_argument("--top", type=int, default=30, help="Show top-N tightened pairs")
    ap.add_argument("--max-pieces", type=int, default=None, help="Limit number of pieces checked (debug)")
    ap.add_argument("--max-leathers", type=int, default=None, help="Limit number of leathers checked (debug)")
    args = ap.parse_args(argv)

    data = merge_data(pieces, demand)
    I = list(data.keys())
    J = list(leathers.keys())

    if args.max_pieces is not None:
        I = I[: max(0, int(args.max_pieces))]
    if args.max_leathers is not None:
        J = J[: max(0, int(args.max_leathers))]

    rq_map = _load_piece_required_quality_map(os.path.join("dataset", "pieces", "piece_catalog_area_rule.csv"))
    rq = {i: int(rq_map[i.upper()]) for i in I if i.upper() in rq_map}

    hist = load_history_ub(args.history)

    rows: List[Row] = []
    missing_hist = 0
    missing_rq = 0

    for i in I:
        if i not in rq:
            missing_rq += 1
            continue
        d_i = int(float(data[i]["demand"]))

        for j in J:
            ub_h = hist.get(i, j)
            if ub_h is None:
                missing_hist += 1
                continue

            ub_bl = max_pieces_for_pattern_on_leather(
                piece_geom_attrs=data[i].get("geom_attrs"),
                leather_code=j,
                required_grade=rq[i],
                dataset_leathers_dir=os.path.join("dataset", "leathers"),
            )

            rows.append(
                Row(
                    piece=str(i).upper(),
                    leather=str(j).upper(),
                    ub_hist=int(ub_h),
                    ub_bottom_left=int(ub_bl),
                    demand=d_i,
                )
            )

    if not rows:
        print("No comparable (piece, leather) pairs were produced.")
        print(f"missing required_quality for pieces: {missing_rq}")
        print(f"missing history rows: {missing_hist}")
        return 1

    tighter = sum(1 for r in rows if r.ub_hist < r.ub_bottom_left)
    equal = sum(1 for r in rows if r.ub_hist == r.ub_bottom_left)
    looser = sum(1 for r in rows if r.ub_hist > r.ub_bottom_left)

    binds_hist = sum(1 for r in rows if r.ub_hist < r.demand)
    binds_bl = sum(1 for r in rows if r.ub_bottom_left < r.demand)

    print(f"Compared pairs: {len(rows)}")
    print(f"History tighter (ub_hist < ub_bl): {tighter}")
    print(f"Equal: {equal}")
    print(f"History looser (ub_hist > ub_bl): {looser}")
    print(f"Pairs where history could bind vs demand (ub_hist < demand): {binds_hist}")
    print(f"Pairs where bottom-left could bind vs demand (ub_bl < demand): {binds_bl}")
    print()

    rows_sorted = sorted(rows, key=lambda r: (-(r.delta), r.piece, r.leather))
    topn = max(0, int(args.top))

    print(f"Top {topn} most tightened by history (largest ub_bl - ub_hist):")
    for r in rows_sorted[:topn]:
        if r.delta <= 0:
            break
        print(
            f"{r.piece:35s} {r.leather:10s} "
            f"ub_hist={r.ub_hist:6d} ub_bl={r.ub_bottom_left:6d} "
            f"delta={r.delta:6d} demand={r.demand:6d}"
        )

    print()
    print("Top most loosened by history (history > bottom-left):")
    rows_sorted2 = sorted(rows, key=lambda r: ((r.ub_hist - r.ub_bottom_left), r.piece, r.leather), reverse=True)
    for r in rows_sorted2[:topn]:
        diff = r.ub_hist - r.ub_bottom_left
        if diff <= 0:
            break
        print(
            f"{r.piece:35s} {r.leather:10s} "
            f"ub_hist={r.ub_hist:6d} ub_bl={r.ub_bottom_left:6d} "
            f"hist_minus_bl={diff:6d} demand={r.demand:6d}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
