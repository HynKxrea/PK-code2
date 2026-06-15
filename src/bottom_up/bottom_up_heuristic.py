"""Bottom-up heuristic utilities.

This module provides an on-the-fly (no preprocessing/cache files) upper bound for
how many copies of one pattern (piece) can be placed on one leather, under:
- no rotation
- bottom-left sequential placement (row-by-row)
- stop when no more placements are possible

We approximate both the pattern and the leather by axis-aligned rectangles:
- pattern rectangle uses SIZE_X, SIZE_Y from the pattern's GEOM_INFO (cm)
- leather rectangle uses width/height from dataset/leathers/**/leather_svg.xml Metadata (mm)

This is intended as a *tight-ish* upper bound for MIP variable y[i,j].
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

try:
    import numpy as np  # type: ignore

    _HAS_NUMPY = True
except Exception:
    np = None  # type: ignore
    _HAS_NUMPY = False


def bottom_up_place_rectangles(
    bin_w_mm: float,
    bin_h_mm: float,
    rect_w_mm: float,
    rect_h_mm: float,
) -> List[Tuple[float, float, float, float]]:
    """Return placements for the rectangle-only bottom-up heuristic.

    Each placement is (x_mm, y_mm, w_mm, h_mm), where (0,0) is the bottom-left
    of the leather bounding rectangle.

    Note: This variant ignores the actual leather shape.
    """

    placements: List[Tuple[float, float, float, float]] = []

    if bin_w_mm <= 0 or bin_h_mm <= 0 or rect_w_mm <= 0 or rect_h_mm <= 0:
        return placements

    if rect_w_mm > bin_w_mm or rect_h_mm > bin_h_mm:
        return placements

    x = 0.0
    y = 0.0
    row_h = 0.0

    while True:
        if x + rect_w_mm <= bin_w_mm and y + rect_h_mm <= bin_h_mm:
            placements.append((x, y, rect_w_mm, rect_h_mm))
            x += rect_w_mm
            row_h = max(row_h, rect_h_mm)
            continue

        if row_h <= 0:
            break

        x = 0.0
        y += row_h
        row_h = 0.0

        if y + rect_h_mm > bin_h_mm:
            break

    return placements


def bottom_up_max_count(
    bin_w_mm: float,
    bin_h_mm: float,
    rect_w_mm: float,
    rect_h_mm: float,
) -> int:
    """Bottom-up (bottom-left) sequential placement for identical rectangles.

    Places rectangles left-to-right; when a rectangle no longer fits in the row,
    starts a new row above the current one. Stops when no more placements fit.

    Returns:
        Maximum number of rectangles placed.
    """

    return len(bottom_up_place_rectangles(bin_w_mm, bin_h_mm, rect_w_mm, rect_h_mm))


def _parse_mm(text: str) -> Optional[float]:
    if text is None:
        return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(text))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


@lru_cache(maxsize=512)
def _find_leather_svg_path(leather_code: str, dataset_leathers_dir: str) -> Optional[Path]:
    base = Path(dataset_leathers_dir)
    if not base.exists():
        return None

    # Typical folder name contains the leather code, e.g. *_L0696152_*
    # leather_svg.xml exists in each leather folder.
    # Use recursive glob; dataset size is small (~30 leathers).
    pattern = f"**/*{leather_code}*/leather_svg.xml"
    matches = list(base.glob(pattern))
    if matches:
        return matches[0]

    # Fallback: search by content in file name (rare)
    matches = list(base.glob("**/leather_svg.xml"))
    for p in matches:
        if leather_code in str(p.parent):
            return p

    return None


@lru_cache(maxsize=512)
def get_leather_rect_mm(leather_code: str, dataset_leathers_dir: str = "dataset/leathers") -> Optional[Tuple[float, float]]:
    """Return (width_mm, height_mm) of the leather bounding rectangle."""

    svg_path = _find_leather_svg_path(leather_code, dataset_leathers_dir)
    if svg_path is None or not svg_path.exists():
        return None

    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except Exception:
        return None

    # File structure:
    # <AureliaSVG>
    #   <Metadata>
    #     <width>2766.99mm</width>
    #     <height>1455.87mm</height>
    #   </Metadata>
    meta = root.find("Metadata")
    if meta is None:
        return None

    w_node = meta.find("width")
    h_node = meta.find("height")
    if w_node is None or h_node is None:
        return None

    w = _parse_mm(w_node.text or "")
    h = _parse_mm(h_node.text or "")
    if w is None or h is None:
        return None

    return (float(w), float(h))


def get_piece_rect_mm(piece_geom_attrs: Optional[Dict], cm_to_mm: float = 10.0) -> Optional[Tuple[float, float]]:
    """Return (width_mm, height_mm) from GEOM_INFO SIZE_X/SIZE_Y (cm)."""

    if not isinstance(piece_geom_attrs, dict):
        return None

    sx = piece_geom_attrs.get("SIZE_X")
    sy = piece_geom_attrs.get("SIZE_Y")
    try:
        w_cm = float(sx)
        h_cm = float(sy)
    except Exception:
        return None

    if w_cm <= 0 or h_cm <= 0:
        return None

    return (w_cm * cm_to_mm, h_cm * cm_to_mm)


# Quality reference colors used in leather_preview.png (dataset-specific)
# Map to Q levels: red=Q1, yellow=Q2, green=Q3, cyan=Q4, blue=Q5
_QUALITY_REFS: List[Tuple[int, int, int]] = [
    (255, 30, 3),  # Q1
    (245, 245, 30),  # Q2
    (0, 226, 43),  # Q3
    (0, 250, 250),  # Q4
    (0, 95, 250),  # Q5
]


def _find_leather_preview_path(leather_code: str, dataset_leathers_dir: str) -> Optional[Path]:
    base = Path(dataset_leathers_dir)
    if not base.exists():
        return None
    matches = list(base.glob(f"**/*{leather_code}*/leather_preview.png"))
    return matches[0] if matches else None


def _integral_image(mask: "np.ndarray") -> "np.ndarray":
    # mask: HxW bool or {0,1}
    integ = mask.astype(np.int32).cumsum(axis=0).cumsum(axis=1)
    # pad (H+1)x(W+1)
    out = np.zeros((integ.shape[0] + 1, integ.shape[1] + 1), dtype=np.int32)
    out[1:, 1:] = integ
    return out


def _sum_rect(integ: "np.ndarray", x: int, y: int, w: int, h: int) -> int:
    # integ is padded
    x2 = x + w
    y2 = y + h
    return int(integ[y2, x2] - integ[y, x2] - integ[y2, x] + integ[y, x])


@lru_cache(maxsize=256)
def _get_leather_allowed_integrals(
    leather_code: str,
    dataset_leathers_dir: str,
    alpha_min: int,
    ignore_near_black_max_channel: int,
    ignore_near_black_sum: int,
) -> Optional[Tuple[float, float, int, int, int, int, int, int, Tuple["np.ndarray", ...]]]:
    """Rasterize leather shape + quality into integral images.

    Returns:
        (leather_w_mm, leather_h_mm, crop_x0, crop_y0, crop_W_px, crop_H_px, full_W_px, full_H_px, allowed_integrals_by_grade)

    allowed_integrals_by_grade is a tuple of 5 integral images (for grades 1..5)
    where a pixel is 1 iff:
      - inside leather (alpha>=alpha_min)
      - not near-black outline
      - quality_level <= grade
    """

    if not _HAS_NUMPY:
        return None

    leather_rect = get_leather_rect_mm(leather_code, dataset_leathers_dir=dataset_leathers_dir)
    if leather_rect is None:
        return None
    lw_mm, lh_mm = leather_rect

    preview_path = _find_leather_preview_path(leather_code, dataset_leathers_dir)
    if preview_path is None or not preview_path.exists():
        return None

    img = Image.open(preview_path).convert("RGBA")
    arr = np.asarray(img)
    full_H, full_W = arr.shape[0], arr.shape[1]
    # Use int32 to avoid overflow in squared-distance computations
    r = arr[..., 0].astype(np.int32)
    g = arr[..., 1].astype(np.int32)
    b = arr[..., 2].astype(np.int32)
    a = arr[..., 3].astype(np.int32)

    # shape mask from alpha
    mask = a >= int(alpha_min)

    # remove near-black outline pixels
    s = r + g + b
    maxc = np.maximum(np.maximum(r, g), b)
    mask = mask & (maxc >= int(ignore_near_black_max_channel)) & (s >= int(ignore_near_black_sum))

    # crop to leather bbox in pixels (speed)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    # inclusive -> slice end +1
    mask_c = mask[y0 : y1 + 1, x0 : x1 + 1]
    r_c = r[y0 : y1 + 1, x0 : x1 + 1]
    g_c = g[y0 : y1 + 1, x0 : x1 + 1]
    b_c = b[y0 : y1 + 1, x0 : x1 + 1]

    H, W = mask_c.shape

    # quality classification by nearest ref among 5 base colors
    # compute dist^2 for each ref: (r-ref)^2+(g-ref)^2+(b-ref)^2
    dist_stack = []
    for (rr, gg, bb) in _QUALITY_REFS:
        dr = r_c - int(rr)
        dg = g_c - int(gg)
        db = b_c - int(bb)
        d = dr * dr + dg * dg + db * db
        dist_stack.append(d.astype(np.int32))
    dists = np.stack(dist_stack, axis=0)  # 5xHxW
    q_idx = np.argmin(dists, axis=0).astype(np.int8)  # 0..4
    q_level = (q_idx + 1).astype(np.int8)  # 1..5

    # Outside mask -> set to 99 so it's never allowed
    q_level = np.where(mask_c, q_level, np.int8(99))

    allowed_integrals = []
    for grade in range(1, 6):
        allowed = mask_c & (q_level <= grade)
        allowed_integrals.append(_integral_image(allowed))

    return (
        float(lw_mm),
        float(lh_mm),
        int(x0),
        int(y0),
        int(W),
        int(H),
        int(full_W),
        int(full_H),
        tuple(allowed_integrals),
    )


def _bottom_up_place_on_allowed(
    allowed_integ: "np.ndarray",
    W: int,
    H: int,
    rect_w: int,
    rect_h: int,
) -> List[Tuple[int, int, int, int]]:
    """Bottom-up placement on an allowed mask using integral image.

    Coordinates are in the *cropped* image pixel space, using top-left origin.

    Requirement-aligned greedy placement:
    - no rotation
    - scan y from bottom->top (below -> up)
    - scan x from left->right
    - place whenever possible
    - prevent overlaps (rectangles cannot overlap)

    Implementation notes:
    - Feasibility (inside leather + quality) is checked by integral image in O(1)
    - Overlap is prevented by a 1D segment tree over x that tracks the top-most
      placed rectangle y for each x-column span.

    Returns:
        list of (x_px, y_px, w_px, h_px)
    """

    if rect_w <= 0 or rect_h <= 0 or rect_w > W or rect_h > H:
        return []

    area = int(rect_w * rect_h)
    max_x = W - rect_w
    if max_x < 0:
        return []

    xs = np.arange(0, max_x + 1, dtype=np.int32)

    class _SegTreeMinAssign:
        # range assign + range min query
        def __init__(self, n: int, init_val: int):
            self.n0 = 1
            while self.n0 < n:
                self.n0 *= 2
            self.inf = 10**9
            self.minv = [init_val] * (2 * self.n0)
            self.lazy = [None] * (2 * self.n0)

        def _apply(self, idx: int, val: int):
            self.minv[idx] = val
            self.lazy[idx] = val

        def _push(self, idx: int):
            v = self.lazy[idx]
            if v is None:
                return
            self._apply(idx * 2, v)
            self._apply(idx * 2 + 1, v)
            self.lazy[idx] = None

        def range_assign(self, l: int, r: int, val: int, idx: int = 1, nl: int = 0, nr: Optional[int] = None):
            if nr is None:
                nr = self.n0
            if r <= nl or nr <= l:
                return
            if l <= nl and nr <= r:
                self._apply(idx, val)
                return
            self._push(idx)
            mid = (nl + nr) // 2
            self.range_assign(l, r, val, idx * 2, nl, mid)
            self.range_assign(l, r, val, idx * 2 + 1, mid, nr)
            self.minv[idx] = self.minv[idx * 2] if self.minv[idx * 2] < self.minv[idx * 2 + 1] else self.minv[idx * 2 + 1]

        def range_min(self, l: int, r: int, idx: int = 1, nl: int = 0, nr: Optional[int] = None) -> int:
            if nr is None:
                nr = self.n0
            if r <= nl or nr <= l:
                return self.inf
            if l <= nl and nr <= r:
                return self.minv[idx]
            self._push(idx)
            mid = (nl + nr) // 2
            a = self.range_min(l, r, idx * 2, nl, mid)
            b = self.range_min(l, r, idx * 2 + 1, mid, nr)
            return a if a < b else b

    # For each x-column, store the smallest y (top-most) of any placed rect covering it.
    # Start with "no rect below" => very large.
    skyline = _SegTreeMinAssign(W, init_val=10**8)

    placements: List[Tuple[int, int, int, int]] = []

    def feasible_xs_for_y(y: int) -> "np.ndarray":
        xw = xs + rect_w
        yh = y + rect_h
        sums = (
            allowed_integ[yh, xw]
            - allowed_integ[y, xw]
            - allowed_integ[yh, xs]
            + allowed_integ[y, xs]
        )
        return sums == area

    # y: top-left, bottom->top
    for y in range(H - rect_h, -1, -1):
        feas = feasible_xs_for_y(int(y))
        x_candidates = np.flatnonzero(feas)
        if x_candidates.size == 0:
            continue

        for x in x_candidates.tolist():
            # overlap check: if there is a rectangle below overlapping this x-span,
            # we need y+rect_h <= y_below
            y_below = skyline.range_min(int(x), int(x + rect_w))
            if (y + rect_h) <= y_below:
                placements.append((int(x), int(y), int(rect_w), int(rect_h)))
                skyline.range_assign(int(x), int(x + rect_w), int(y))

    return placements


def _bottom_up_max_count_on_allowed(
    allowed_integ: "np.ndarray",
    W: int,
    H: int,
    rect_w: int,
    rect_h: int,
) -> int:
    return len(_bottom_up_place_on_allowed(allowed_integ, W=W, H=H, rect_w=rect_w, rect_h=rect_h))


def bottom_up_placements_for_pattern_on_leather(
    piece_geom_attrs: Optional[Dict],
    leather_code: str,
    required_grade: int = 5,
    dataset_leathers_dir: str = "dataset/leathers",
    alpha_min: int = 255,
    ignore_near_black_max_channel: int = 25,
    ignore_near_black_sum: int = 40,
) -> List[Tuple[int, int, int, int]]:
    """Return placements for the same heuristic used to compute ub_ij.

    Returned placements are in *leather_preview.png pixel coordinates*
    (top-left origin): (x_px, y_px, w_px, h_px).

    Constraints:
    - inside leather mask (alpha)
    - not near-black outline
    - leather quality level <= required_grade
    - no rotation
    - bottom-up sequential placement (below->up, left->right)
    """

    required_grade = int(required_grade)
    if required_grade < 1:
        required_grade = 1
    if required_grade > 5:
        required_grade = 5

    if not _HAS_NUMPY:
        return []

    piece_rect = get_piece_rect_mm(piece_geom_attrs)
    if piece_rect is None:
        return []

    info = _get_leather_allowed_integrals(
        leather_code=leather_code,
        dataset_leathers_dir=dataset_leathers_dir,
        alpha_min=int(alpha_min),
        ignore_near_black_max_channel=int(ignore_near_black_max_channel),
        ignore_near_black_sum=int(ignore_near_black_sum),
    )
    if info is None:
        return []

    lw_mm, lh_mm, cx0, cy0, W, H, full_W, full_H, allowed_integrals = info
    pw_mm, ph_mm = piece_rect

    # Convert mm -> px using FULL preview dimensions (not cropped), to keep scale consistent.
    rect_w = max(1, int(round((pw_mm / lw_mm) * full_W)))
    rect_h = max(1, int(round((ph_mm / lh_mm) * full_H)))

    allowed_integ = allowed_integrals[required_grade - 1]
    placements_crop = _bottom_up_place_on_allowed(allowed_integ, W=W, H=H, rect_w=rect_w, rect_h=rect_h)

    # shift to original preview coordinates
    return [(x + cx0, y + cy0, w, h) for (x, y, w, h) in placements_crop]


def max_pieces_for_pattern_on_leather(
    piece_geom_attrs: Optional[Dict],
    leather_code: str,
    required_grade: int = 5,
    dataset_leathers_dir: str = "dataset/leathers",
    alpha_min: int = 255,
    ignore_near_black_max_channel: int = 25,
    ignore_near_black_sum: int = 40,
) -> int:
    """Compute max copies of one pattern that can fit on a leather.

    - Pattern is approximated as an axis-aligned rectangle (SIZE_X/SIZE_Y)
    - Leather is represented by its actual pixel mask from leather_preview.png
    - Quality constraint: every pixel under the rectangle must have leather quality
      level <= required_grade (Q1 best=1 ... Q5 worst=5)
    - Placement: bottom-up sequential rows, bottom-left preference, no rotation

    Returns:
        ub_ij (int)
    """

    required_grade = int(required_grade)
    if required_grade < 1:
        required_grade = 1
    if required_grade > 5:
        required_grade = 5

    piece_rect = get_piece_rect_mm(piece_geom_attrs)
    if piece_rect is None:
        return 0

    if not _HAS_NUMPY:
        # Fallback: old rectangle-only bound if numpy is unavailable
        leather_rect = get_leather_rect_mm(leather_code, dataset_leathers_dir=dataset_leathers_dir)
        if leather_rect is None:
            return 0
        pw, ph = piece_rect
        lw, lh = leather_rect
        return bottom_up_max_count(bin_w_mm=lw, bin_h_mm=lh, rect_w_mm=pw, rect_h_mm=ph)

    info = _get_leather_allowed_integrals(
        leather_code=leather_code,
        dataset_leathers_dir=dataset_leathers_dir,
        alpha_min=int(alpha_min),
        ignore_near_black_max_channel=int(ignore_near_black_max_channel),
        ignore_near_black_sum=int(ignore_near_black_sum),
    )
    if info is None:
        # fallback to rectangle-only
        leather_rect = get_leather_rect_mm(leather_code, dataset_leathers_dir=dataset_leathers_dir)
        if leather_rect is None:
            return 0
        pw, ph = piece_rect
        lw, lh = leather_rect
        return bottom_up_max_count(bin_w_mm=lw, bin_h_mm=lh, rect_w_mm=pw, rect_h_mm=ph)

    lw_mm, lh_mm, _cx0, _cy0, W, H, full_W, full_H, allowed_integrals = info

    pw_mm, ph_mm = piece_rect

    # Convert mm -> px using FULL preview dimensions (not cropped), to keep scale consistent.
    rect_w = max(1, int(round((pw_mm / lw_mm) * full_W)))
    rect_h = max(1, int(round((ph_mm / lh_mm) * full_H)))

    allowed_integ = allowed_integrals[required_grade - 1]
    return _bottom_up_max_count_on_allowed(allowed_integ, W=W, H=H, rect_w=rect_w, rect_h=rect_h)
