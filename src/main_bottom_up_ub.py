"""Print bottom-up heuristic upper bounds ub_ij for all (piece i, leather j).

This script does NOT modify or solve the MIP. It only runs the bottom-up
rectangle placement heuristic used as the y[i,j] upper bound.

Usage:
    python3 src/main_bottom_up_ub.py
    python3 src/main_bottom_up_ub.py --piece "FLAP TOP"
    python3 src/main_bottom_up_ub.py --leather "L0696152"
    python3 src/main_bottom_up_ub.py --min-ub 1

"""

from __future__ import annotations

import argparse

from bottom_up.bottom_up_heuristic import (
    get_leather_rect_mm,
    get_piece_rect_mm,
    max_pieces_for_pattern_on_leather,
)
from data.preprocessing import assign_grade, demand, leathers, merge_data, pieces


def main() -> int:
    parser = argparse.ArgumentParser(description="Print bottom-up ub_ij upper bounds.")
    parser.add_argument("--piece", default=None, help="Filter by piece name (case-insensitive)")
    parser.add_argument("--leather", default=None, help="Filter by leather code (e.g., L0696152)")
    parser.add_argument("--min-ub", type=int, default=0, help="Only print rows with ub >= this")
    args = parser.parse_args()

    piece_filter = args.piece.upper() if args.piece else None
    leather_filter = args.leather.upper() if args.leather else None

    data = merge_data(pieces, demand)

    piece_names = sorted(data.keys())
    hides = sorted(leathers.keys())

    if piece_filter:
        piece_names = [p for p in piece_names if piece_filter in p.upper()]
    if leather_filter:
        hides = [h for h in hides if leather_filter in h.upper()]

    header = [
        "piece",
        "leather",
        "ub_ij",
        "piece_w_mm",
        "piece_h_mm",
        "leather_w_mm",
        "leather_h_mm",
    ]
    print("\t".join(header))

    for i in piece_names:
        geom = data[i].get("geom_attrs")
        piece_rect = get_piece_rect_mm(geom)
        pw, ph = piece_rect if piece_rect is not None else (None, None)

        for j in hides:
            leather_rect = get_leather_rect_mm(j)
            lw, lh = leather_rect if leather_rect is not None else (None, None)

            ub = max_pieces_for_pattern_on_leather(
                piece_geom_attrs=geom,
                leather_code=j,
                required_grade=assign_grade(i),
                dataset_leathers_dir="dataset/leathers",
            )

            if ub < args.min_ub:
                continue

            row = [
                i,
                j,
                str(int(ub)),
                "NA" if pw is None else f"{pw:.1f}",
                "NA" if ph is None else f"{ph:.1f}",
                "NA" if lw is None else f"{lw:.1f}",
                "NA" if lh is None else f"{lh:.1f}",
            ]
            print("\t".join(row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
