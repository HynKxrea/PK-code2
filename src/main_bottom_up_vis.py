"""Visualize bottom-up rectangle nesting.

This script visualizes the bottom-up heuristic placement (no rotation) used for
ub_ij, using rectangle approximations.

It saves a PNG under outputs/bottom_up_vis/.

Usage:
    python3 src/main_bottom_up_vis.py --piece "FLAP TOP" --leather "L0696152"
    python3 src/main_bottom_up_vis.py --piece "FLAP TOP" --leather "L0696152" --alpha-min 255 --max-side 1400

Notes:
- This is NOT true polygon nesting; it visualizes bounding-box placement only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from bottom_up.bottom_up_heuristic import (
    bottom_up_placements_for_pattern_on_leather,
    get_leather_rect_mm,
    get_piece_rect_mm,
)
from data.preprocessing import assign_grade, demand, leathers, merge_data, pieces


def _find_piece_key(data: dict, piece_query: str) -> str:
    pq = piece_query.strip().upper()
    # exact
    for k in data.keys():
        if k.upper() == pq:
            return k
    # substring
    for k in data.keys():
        if pq in k.upper():
            return k
    raise KeyError(f"Piece not found: {piece_query}")


def _find_leather_preview_png(leather_code: str, dataset_leathers_dir: str = "dataset/leathers") -> Path | None:
    base = Path(dataset_leathers_dir)
    if not base.exists():
        return None
    matches = list(base.glob(f"**/*{leather_code}*/leather_preview.png"))
    if matches:
        return matches[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize bottom-up ub_ij placement")
    parser.add_argument("--piece", required=True, help="Piece name (exact or substring, case-insensitive)")
    parser.add_argument("--leather", required=True, help="Leather code like L0696152")
    parser.add_argument("--max-side", type=int, default=1400, help="Max image side in pixels")
    parser.add_argument(
        "--use-preview",
        action="store_true",
        help="Overlay rectangles on dataset/leathers/**/leather_preview.png (recommended)",
    )
    parser.add_argument(
        "--required-grade",
        type=int,
        default=None,
        help="Override required grade (1..5). Default: assign_grade(piece_name).",
    )
    args = parser.parse_args()

    data = merge_data(pieces, demand)

    piece_key = _find_piece_key(data, args.piece)
    leather_code = args.leather.strip().upper()
    if leather_code not in leathers:
        raise KeyError(f"Leather not found: {leather_code}")

    piece_rect = get_piece_rect_mm(data[piece_key].get("geom_attrs"))
    leather_rect = get_leather_rect_mm(leather_code)
    if piece_rect is None:
        raise ValueError(f"Missing piece SIZE_X/SIZE_Y for piece: {piece_key}")
    if leather_rect is None:
        raise ValueError(f"Missing leather width/height for leather: {leather_code}")

    pw, ph = piece_rect
    lw, lh = leather_rect

    required_grade = assign_grade(piece_key) if args.required_grade is None else int(args.required_grade)
    placements = bottom_up_placements_for_pattern_on_leather(
        piece_geom_attrs=data[piece_key].get("geom_attrs"),
        leather_code=leather_code,
        required_grade=required_grade,
        dataset_leathers_dir="dataset/leathers",
        alpha_min=255,
    )

    # Base image: either leather_preview.png or a blank canvas
    offset_x = 0.0
    offset_y = 0.0

    if args.use_preview:
        preview_path = _find_leather_preview_png(leather_code)
        if preview_path is None:
            raise FileNotFoundError(
                f"leather_preview.png not found for {leather_code} under dataset/leathers (try without --use-preview)"
            )
        base_img = Image.open(preview_path).convert("RGBA")

        # optionally downscale for display
        s = min(args.max_side / base_img.size[0], args.max_side / base_img.size[1], 1.0)
        if s < 1.0:
            base_img = base_img.resize((int(base_img.size[0] * s), int(base_img.size[1] * s)))
        scale = s
        offset_x = 0.0
        offset_y = 0.0

        img = base_img
    else:
        # no preview: draw on a blank canvas in the same pixel coordinate system as preview would have
        # (use leather_rect_mm only to size a reasonable canvas)
        scale = min(args.max_side / lw, args.max_side / lh)
        W = max(1, int(lw * scale))
        H = max(1, int(lh * scale))
        img = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # border (leather bounding box)
    draw.rectangle([0, 0, img.size[0] - 1, img.size[1] - 1], outline=(0, 0, 0, 255), width=2)

    # optional font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, (x, y, w, h) in enumerate(placements, start=1):
        # placements are already in leather_preview pixel coords (top-left origin)
        x1 = int(offset_x + x * scale)
        y1 = int(offset_y + y * scale)
        x2 = int(offset_x + (x + w) * scale)
        y2 = int(offset_y + (y + h) * scale)

        draw.rectangle([x1, y1, x2, y2], outline=(30, 144, 255, 255), fill=(30, 144, 255, 40), width=2)
        if font and (x2 - x1) > 25 and (y2 - y1) > 14:
            draw.text((x1 + 2, y1 + 2), str(idx), fill=(0, 0, 0, 255), font=font)

    out_dir = Path("outputs") / "bottom_up_vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{piece_key.replace(' ', '_')}__{leather_code}.png"

    composed = Image.alpha_composite(img.convert("RGBA"), overlay)
    composed.save(out_path)

    print(f"piece: {piece_key}")
    print(f"leather: {leather_code}")
    print(f"piece_mm: ({pw:.1f}, {ph:.1f})")
    print(f"leather_mm: ({lw:.1f}, {lh:.1f})")
    print(f"required_grade: {required_grade}")
    print(f"placed: {len(placements)}")
    print(f"saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
