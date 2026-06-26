"""Pixel-level color counting (10-class nearest-color classifier).

Scans PNG files under a directory (default: dataset/N-code(png)) and classifies
EACH pixel by nearest RGB reference among 10 classes:
- base colors 5:  red, yellow, green, blue, cyan
- dark  colors 5: dark_red, dark_yellow, dark_green, dark_blue, dark_cyan

Classification:
- For each valid pixel, compute squared Euclidean distance in RGB to each of the
  10 reference colors and assign the pixel to the closest class.

Ignore (heuristics):
- alpha == 0
- near-black outline pixels
- optionally partially-transparent pixels to reduce boundary anti-aliasing

CLI:
    python3 src/calc_c.py
    python3 src/calc_c.py --dir "dataset/N-code(png)"

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

from PIL import Image

try:
    import numpy as np  # type: ignore

    _HAS_NUMPY = True
except Exception:
    np = None  # type: ignore
    _HAS_NUMPY = False


# 5 base reference colors
REFERENCE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "red": (255, 30, 3),
    "yellow": (245, 245, 30),
    "green": (0, 226, 43),
    "blue": (0, 95, 250),
    "cyan": (0, 250, 250),
}

# 5 dark reference colors (simple scaled versions; adjust if you have exact refs)
DARK_REFERENCE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "dark_red": (220, 25, 3),
    "dark_yellow": (220, 22, 25),
    "dark_green": (0, 204, 39),
    "dark_blue": (0, 86, 225),
    "dark_cyan": (0, 225, 225),
}

OUTPUT_CLASSES: Tuple[str, ...] = (
    "red",
    "dark_red",
    "yellow",
    "dark_yellow",
    "green",
    "dark_green",
    "blue",
    "dark_blue",
    "cyan",
    "dark_cyan",
)

BASE_COLORS: Tuple[str, ...] = ("red", "yellow", "green", "blue", "cyan")

# class -> RGB
ALL_CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    **REFERENCE_COLORS,
    **DARK_REFERENCE_COLORS,
}

# stable ordering for numpy argmin
CLASS_ORDER: Tuple[str, ...] = OUTPUT_CLASSES


@dataclass(frozen=True)
class ColorRule:
    # Ignore outlines / boundary anti-aliasing pixels (heuristics)
    ignore_near_black_max_channel: int = 25
    ignore_near_black_sum: int = 40

    # To reduce boundary anti-aliasing effects: ignore partially transparent pixels.
    # 255 means only fully-opaque pixels are counted.
    ignore_alpha_below: int = 255


def _resolve_default_dir(dir_path: Optional[str | Path]) -> Path:
    if dir_path is not None:
        return Path(dir_path)

    candidates = [Path("dataset/N-code(png)"), Path("dataset/N-code(PNG)")]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    dataset_dir = Path("dataset")
    if dataset_dir.exists():
        for p in dataset_dir.iterdir():
            if p.is_dir() and p.name.lower().startswith("n-code"):
                return p

    return Path("dataset/N-code(png)")


def iter_png_files(dir_path: str | Path) -> Iterator[Path]:
    p = Path(dir_path)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {p}")

    yield from sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".png"])


def _display_filename(png_name: str) -> str:
    """Return name starting from 'N' and without '.png'.

    Examples:
      '1. N0696153.png' -> 'N0696153'
      '14. N09169717.png' -> 'N09169717'
    """

    stem = Path(png_name).stem
    idx = stem.find("N")
    return stem[idx:] if idx >= 0 else stem


def _dist2_rgb(pr: int, pg: int, pb: int, ref: Tuple[int, int, int]) -> int:
    dr = pr - ref[0]
    dg = pg - ref[1]
    db = pb - ref[2]
    return dr * dr + dg * dg + db * db


def count_class_pixels_in_image(image_path: str | Path, rule: ColorRule = ColorRule()) -> Dict[str, int]:
    counts: Dict[str, int] = {k: 0 for k in OUTPUT_CLASSES}

    image_path = Path(image_path)
    with Image.open(image_path) as img:
        rgba = img.convert("RGBA")

        if _HAS_NUMPY:
            arr = np.asarray(rgba)
            r = arr[..., 0].astype(np.int32)
            g = arr[..., 1].astype(np.int32)
            b = arr[..., 2].astype(np.int32)
            a = arr[..., 3].astype(np.int32)

            s = r + g + b
            maxc = np.maximum(np.maximum(r, g), b)

            valid = (a >= int(rule.ignore_alpha_below)) & (s > 0)
            valid &= (maxc >= int(rule.ignore_near_black_max_channel))
            valid &= (s >= int(rule.ignore_near_black_sum))

            # Nearest-color assignment (argmin over 10 refs)
            best_dist = np.full(r.shape, 1_000_000_000, dtype=np.int32)
            best_idx = np.full(r.shape, -1, dtype=np.int16)

            for idx, cls in enumerate(CLASS_ORDER):
                ref = ALL_CLASS_COLORS[cls]
                dist = (r - int(ref[0])) * (r - int(ref[0])) + (g - int(ref[1])) * (g - int(ref[1])) + (b - int(ref[2])) * (b - int(ref[2]))
                better = dist < best_dist
                best_dist = np.where(better, dist, best_dist)
                best_idx = np.where(better, idx, best_idx)

            for idx, cls in enumerate(CLASS_ORDER):
                counts[cls] = int((valid & (best_idx == idx)).sum())

            return counts

        # No numpy fallback
        for pr, pg, pb, pa in rgba.getdata():
            if pa < rule.ignore_alpha_below:
                continue
            s = pr + pg + pb
            if s <= 0:
                continue
            if max(pr, pg, pb) < rule.ignore_near_black_max_channel:
                continue
            if s < rule.ignore_near_black_sum:
                continue

            best_cls = None
            best_d = None
            for cls in CLASS_ORDER:
                d = _dist2_rgb(pr, pg, pb, ALL_CLASS_COLORS[cls])
                if best_d is None or d < best_d:
                    best_d = d
                    best_cls = cls

            if best_cls is not None:
                counts[best_cls] += 1

        return counts


def count_class_pixels_in_dir(
    dir_path: Optional[str | Path] = None,
    rule: ColorRule = ColorRule(),
) -> Dict[str, Dict[str, int]]:
    target_dir = _resolve_default_dir(dir_path)
    return {png.name: count_class_pixels_in_image(png, rule=rule) for png in iter_png_files(target_dir)}


def main(argv: Optional[Iterable[str]] = None) -> int:
    import argparse
    import csv
    import json

    parser = argparse.ArgumentParser(description="Count pixels by nearest reference color (10 classes) and print")
    parser.add_argument(
        "--dir",
        default=None,
        help='PNG directory (default: tries dataset/N-code(png) or dataset/N-code(PNG))',
    )
    parser.add_argument(
        "--alpha-min",
        type=int,
        default=255,
        help="Ignore pixels with alpha < this value (default: 255, ignore boundary anti-aliasing)",
    )
    parser.add_argument("--csv", default=None, help="Optional output CSV path")
    parser.add_argument(
        "--render-out",
        default=None,
        help=(
            "Optional output directory to write classification-only PNGs. "
            "Valid pixels are recolored by predicted class; ignored pixels keep original RGBA."
        ),
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    rule = ColorRule(ignore_alpha_below=int(args.alpha_min))
    target_dir = _resolve_default_dir(args.dir)
    results = count_class_pixels_in_dir(target_dir, rule=rule)

    render_out_dir = Path(args.render_out) if args.render_out else None
    if render_out_dir is not None:
        render_out_dir.mkdir(parents=True, exist_ok=True)

    def dark_usage_percent(counts: Dict[str, int], base: str) -> Optional[float]:
        denom = int(counts.get(base, 0)) + int(counts.get(f"dark_{base}", 0))
        if denom <= 0:
            return None
        return 100.0 * int(counts.get(f"dark_{base}", 0)) / denom

    def fmt_pct(x: Optional[float]) -> str:
        return "NA" if x is None else f"{x:.2f}"

    usage_cols = [f"{c}_use(%)" for c in BASE_COLORS]
    q_cols = ["q1_use", "q2_use", "q3_use", "q4_use", "q5_use"]

    def cumulative_use_decimal(counts: Dict[str, int], bases: list[str]) -> Optional[float]:
        """Cumulative dark ratio as a decimal.

        Example (bases=['red','yellow']):
          (dark_red + dark_yellow) / (red + yellow + dark_red + dark_yellow)
        """

        dark = sum(int(counts.get(f"dark_{b}", 0)) for b in bases)
        normal = sum(int(counts.get(b, 0)) for b in bases)
        denom = dark + normal
        if denom <= 0:
            return None
        return dark / denom

    def fmt_ratio(x: Optional[float]) -> str:
        return "NA" if x is None else f"{x:.4f}"

    def _counts_for_q(counts: Dict[str, int]) -> Dict[str, int]:
        """Return a copy of counts with *dark_blue* forced to 0 (for q calc only).

        Pixel classification/counting and all non-q outputs are unchanged.
        Only q-values treat dark_blue as 0; cyan/dark_cyan stay as measured.
        """

        if not counts:
            return counts
        if "dark_blue" not in counts:
            return counts
        c2 = dict(counts)
        c2["dark_blue"] = 0
        return c2

    def render_classification_image(image_path: Path, out_path: Path) -> None:
        """Write a classification-only image.

        - Valid pixels: recolor to the nearest reference color class (CLASS_ORDER).
        - Ignored pixels (transparent/near-black/alpha<min): keep original RGBA.
        """

        with Image.open(image_path) as img:
            rgba = img.convert("RGBA")

            if _HAS_NUMPY:
                base = np.asarray(rgba)
                out = base.copy()

                r = base[..., 0].astype(np.int32)
                g = base[..., 1].astype(np.int32)
                b = base[..., 2].astype(np.int32)
                a = base[..., 3].astype(np.int32)

                s = r + g + b
                maxc = np.maximum(np.maximum(r, g), b)

                valid = (a >= int(rule.ignore_alpha_below)) & (s > 0)
                valid &= (maxc >= int(rule.ignore_near_black_max_channel))
                valid &= (s >= int(rule.ignore_near_black_sum))

                best_dist = np.full(r.shape, 1_000_000_000, dtype=np.int32)
                best_idx = np.full(r.shape, -1, dtype=np.int16)

                for idx, cls in enumerate(CLASS_ORDER):
                    ref = ALL_CLASS_COLORS[cls]
                    dist = (r - int(ref[0])) * (r - int(ref[0])) + (g - int(ref[1])) * (g - int(ref[1])) + (b - int(ref[2])) * (b - int(ref[2]))
                    better = dist < best_dist
                    best_dist = np.where(better, dist, best_dist)
                    best_idx = np.where(better, idx, best_idx)

                palette = np.asarray([ALL_CLASS_COLORS[cls] for cls in CLASS_ORDER], dtype=np.uint8)
                out[..., :3][valid] = palette[best_idx[valid]]

                Image.fromarray(out, mode="RGBA").save(out_path)
                return

            # No numpy fallback
            out_pixels = []
            for pr, pg, pb, pa in rgba.getdata():
                # Ignored pixels: keep original
                if pa < rule.ignore_alpha_below:
                    out_pixels.append((pr, pg, pb, pa))
                    continue
                s = pr + pg + pb
                if s <= 0:
                    out_pixels.append((pr, pg, pb, pa))
                    continue
                if max(pr, pg, pb) < rule.ignore_near_black_max_channel:
                    out_pixels.append((pr, pg, pb, pa))
                    continue
                if s < rule.ignore_near_black_sum:
                    out_pixels.append((pr, pg, pb, pa))
                    continue

                best_cls = None
                best_d = None
                for cls in CLASS_ORDER:
                    d = _dist2_rgb(pr, pg, pb, ALL_CLASS_COLORS[cls])
                    if best_d is None or d < best_d:
                        best_d = d
                        best_cls = cls

                if best_cls is None:
                    out_pixels.append((pr, pg, pb, pa))
                    continue

                rr, gg, bb = ALL_CLASS_COLORS[best_cls]
                out_pixels.append((int(rr), int(gg), int(bb), pa))

            out_img = Image.new("RGBA", rgba.size)
            out_img.putdata(out_pixels)
            out_img.save(out_path)

    def compute_meta(image_path: Path) -> Dict[str, int]:
        with Image.open(image_path) as img:
            rgba = img.convert("RGBA")
            w, h = rgba.size
            total_px = w * h

            if _HAS_NUMPY:
                arr = np.asarray(rgba)
                r = arr[..., 0].astype(np.int32)
                g = arr[..., 1].astype(np.int32)
                b = arr[..., 2].astype(np.int32)
                a = arr[..., 3].astype(np.int32)
                alpha_px = int((a > 0).sum())

                s = r + g + b
                maxc = np.maximum(np.maximum(r, g), b)
                valid_px = int(
                    ((a >= int(rule.ignore_alpha_below)) & (s > 0) & (maxc >= int(rule.ignore_near_black_max_channel)) & (s >= int(rule.ignore_near_black_sum))).sum()
                )
                return {"total_px": total_px, "alpha_px": alpha_px, "valid_px": valid_px}

            alpha_px = 0
            valid_px = 0
            for pr, pg, pb, pa in rgba.getdata():
                if pa == 0:
                    continue
                alpha_px += 1
                if pa < rule.ignore_alpha_below:
                    continue
                s = pr + pg + pb
                if s <= 0:
                    continue
                if max(pr, pg, pb) < rule.ignore_near_black_max_channel:
                    continue
                if s < rule.ignore_near_black_sum:
                    continue
                valid_px += 1

            return {"total_px": total_px, "alpha_px": alpha_px, "valid_px": valid_px}

    header = [
        "filename",
        *OUTPUT_CLASSES,
        "classified_total",
        "total_px",
        "alpha_px",
        "valid_px",
        "coverage_valid(%)",
        *usage_cols,
        *q_cols,
    ]

    print(f"dir: {target_dir}")
    print("\t".join(header))

    totals: Dict[str, int] = {k: 0 for k in OUTPUT_CLASSES}

    # Average q1..q5 over all leathers (per-image q values)
    q_sum = [0.0, 0.0, 0.0, 0.0, 0.0]
    q_count = [0, 0, 0, 0, 0]

    for name in sorted(results.keys()):
        c = results[name]
        for k in OUTPUT_CLASSES:
            totals[k] += int(c.get(k, 0))

        classified_total = sum(int(c.get(k, 0)) for k in OUTPUT_CLASSES)
        meta = compute_meta(target_dir / name)
        valid_px = int(meta["valid_px"])
        coverage_valid = None if valid_px <= 0 else (100.0 * classified_total / valid_px)

        usage = [fmt_pct(dark_usage_percent(c, base)) for base in BASE_COLORS]

        if render_out_dir is not None:
            render_classification_image(target_dir / name, render_out_dir / name)

        c_q = _counts_for_q(c)
        q1 = cumulative_use_decimal(c_q, ["red"])
        q2 = cumulative_use_decimal(c_q, ["red", "yellow"])
        q3 = cumulative_use_decimal(c_q, ["red", "yellow", "green"])
        q4 = cumulative_use_decimal(c_q, ["red", "yellow", "green", "cyan"])
        q5 = cumulative_use_decimal(c_q, ["red", "yellow", "green", "cyan", "blue"])
        q_vals = [q1, q2, q3, q4, q5]

        # accumulate for q*_use_all
        for i, v in enumerate(q_vals):
            if v is not None:
                q_sum[i] += float(v)
                q_count[i] += 1

        row = [
            _display_filename(name),
            *(str(int(c.get(k, 0))) for k in OUTPUT_CLASSES),
            str(classified_total),
            str(int(meta["total_px"])),
            str(int(meta["alpha_px"])),
            str(valid_px),
            fmt_pct(coverage_valid),
            *usage,
            *(fmt_ratio(x) for x in q_vals),
        ]
        print("\t".join(row))

    # Mean q1..q5 across all leathers
    q1_use_all = (q_sum[0] / q_count[0]) if q_count[0] > 0 else None
    q2_use_all = (q_sum[1] / q_count[1]) if q_count[1] > 0 else None
    q3_use_all = (q_sum[2] / q_count[2]) if q_count[2] > 0 else None
    q4_use_all = (q_sum[3] / q_count[3]) if q_count[3] > 0 else None
    q5_use_all = (q_sum[4] / q_count[4]) if q_count[4] > 0 else None

    # Persist q*_use_all to dataset/u_values.json (for selection.py to consume)
    # Do not change any other behavior; this only updates/creates the json keys.
    u_values_path = Path("dataset") / "c_values.json"
    u_values_path.parent.mkdir(parents=True, exist_ok=True)

    def _safe_float(x: Optional[float]) -> float:
        return 0.0 if x is None else float(x)

    q_payload = {
        "q1_use_all": _safe_float(q1_use_all),
        "q2_use_all": _safe_float(q2_use_all),
        "q3_use_all": _safe_float(q3_use_all),
        "q4_use_all": _safe_float(q4_use_all),
        "q5_use_all": _safe_float(q5_use_all),
    }

    try:
        if u_values_path.exists():
            with u_values_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {}
        else:
            existing = {}

        existing.update(q_payload)

        with u_values_path.open("w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
    except Exception:
        # keep terminal output stable; ignore persistence errors
        pass

    grand_total = sum(int(totals[k]) for k in OUTPUT_CLASSES)

    # Store per-color use values (percent) as separate variables
    red_use = dark_usage_percent(totals, "red")
    yellow_use = dark_usage_percent(totals, "yellow")
    green_use = dark_usage_percent(totals, "green")
    blue_use = dark_usage_percent(totals, "blue")
    cyan_use = dark_usage_percent(totals, "cyan")

    totals_q = _counts_for_q(totals)

    q1_use = cumulative_use_decimal(totals_q, ["red"])
    q2_use = cumulative_use_decimal(totals_q, ["red", "yellow"])
    q3_use = cumulative_use_decimal(totals_q, ["red", "yellow", "green"])
    q4_use = cumulative_use_decimal(totals_q, ["red", "yellow", "green", "cyan"])
    q5_use = cumulative_use_decimal(totals_q, ["red", "yellow", "green", "cyan", "blue"])

    total_usage = [
        fmt_pct(red_use),
        fmt_pct(yellow_use),
        fmt_pct(green_use),
        fmt_pct(blue_use),
        fmt_pct(cyan_use),
    ]

    total_q = [
        fmt_ratio(q1_use),
        fmt_ratio(q2_use),
        fmt_ratio(q3_use),
        fmt_ratio(q4_use),
        fmt_ratio(q5_use),
    ]

    total_row = [
        "__TOTAL__",
        *(str(int(totals[k])) for k in OUTPUT_CLASSES),
        str(grand_total),
        "-",
        "-",
        "-",
        "-",
        *total_usage,
        *total_q,
    ]
    print("\t".join(total_row))

    if args.csv:
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)

            for name in sorted(results.keys()):
                c = results[name]
                classified_total = sum(int(c.get(k, 0)) for k in OUTPUT_CLASSES)
                meta = compute_meta(target_dir / name)
                valid_px = int(meta["valid_px"])
                coverage_valid = None if valid_px <= 0 else (100.0 * classified_total / valid_px)

                c_q = _counts_for_q(c)
                w.writerow(
                    [
                        _display_filename(name),
                        *(int(c.get(k, 0)) for k in OUTPUT_CLASSES),
                        classified_total,
                        int(meta["total_px"]),
                        int(meta["alpha_px"]),
                        valid_px,
                        coverage_valid,
                        *(dark_usage_percent(c, base) for base in BASE_COLORS),
                        cumulative_use_decimal(c_q, ["red"]),
                        cumulative_use_decimal(c_q, ["red", "yellow"]),
                        cumulative_use_decimal(c_q, ["red", "yellow", "green"]),
                        cumulative_use_decimal(c_q, ["red", "yellow", "green", "cyan"]),
                        cumulative_use_decimal(c_q, ["red", "yellow", "green", "cyan", "blue"]),
                    ]
                )

            w.writerow(
                [
                    "__TOTAL__",
                    *(int(totals.get(k, 0)) for k in OUTPUT_CLASSES),
                    grand_total,
                    "-",
                    "-",
                    "-",
                    "-",
                    *(dark_usage_percent(totals, base) for base in BASE_COLORS),
                    q1_use,
                    q2_use,
                    q3_use,
                    q4_use,
                    q5_use,
                ]
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
