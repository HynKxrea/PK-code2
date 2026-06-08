"""python src/util/util.py

Utility functions for utilization extraction (red area) from nesting PNGs.
Provides: find_png_files, analyze_image, analyze_image_wrapper, process_all.
"""
from __future__ import annotations
import os
import glob
from typing import List, Dict, Tuple

from PIL import Image
import numpy as np


def find_png_files(dataset_dir: str) -> List[str]:
    pattern = os.path.join(dataset_dir, "*.png")
    pattern_upper = os.path.join(dataset_dir, "*.PNG")
    files = glob.glob(pattern) + glob.glob(pattern_upper)
    files = sorted(files)
    return files


def analyze_image(image_path: str,
                  alpha_threshold: int = 10,
                  red_hue_tol: int = 10,
                  red_sat_min: int = 50,
                  red_val_min: int = 50,
                  black_thresh: int = 50,
                  scale: int = 1,
                  min_area: int = 20,
                  max_area_ratio: float = 0.9,
                  morph_kernel: int = 3,
                  morph_iter: int = 1,
                  dilate_iter: int = 0) -> Tuple[int, int, int]:
    """Return (red_pixels, used_red_pixels, piece_count).

    Uses HSV-based red mask and flood-fill interior recovery (raster-based).
    Saves debug images under outputs/debug/<basename>/: red_mask.png, piece_mask.png, overlap.png
    """
    try:
        import cv2
    except Exception as e:
        raise RuntimeError("OpenCV (cv2) is required. Install with 'pip install opencv-python'.") from e

    im = Image.open(image_path)
    if scale and scale > 1:
        new_size = (im.width * scale, im.height * scale)
        im = im.resize(new_size, resample=Image.BILINEAR)

    rgba = im.convert("RGBA")
    arr = np.array(rgba)
    if arr.ndim != 3 or arr.shape[2] < 4:
        raise ValueError("Unexpected image shape")

    R = arr[:, :, 0].astype(np.uint8)
    G = arr[:, :, 1].astype(np.uint8)
    B = arr[:, :, 2].astype(np.uint8)
    A = arr[:, :, 3].astype(np.uint8)

    mask_alpha = A > alpha_threshold

    # HSV red mask
    bgr = np.dstack((B, G, R))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, red_sat_min, red_val_min], dtype=np.uint8)
    upper1 = np.array([red_hue_tol, 255, 255], dtype=np.uint8)
    lower2 = np.array([180 - red_hue_tol, red_sat_min, red_val_min], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    red_mask = cv2.bitwise_or(m1, m2)

    # small morphology to reduce AA noise
    small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, small_k, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, small_k, iterations=1)

    red_mask[~mask_alpha] = 0
    red_bool = red_mask.astype(bool)

    # New method per request: contour-based piece mask (no floodfill, no bbox fallback)
    gray = (0.299 * R.astype(np.float32) + 0.587 * G.astype(np.float32) + 0.114 * B.astype(np.float32)).astype(np.uint8)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # black by low V
    _, black_mask = cv2.threshold(gray_blur, black_thresh, 255, cv2.THRESH_BINARY_INV)

    # minimal closing to join thin lines
    k = 3
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    black_closed = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel3, iterations=1)

    # find contours (RETR_LIST to preserve internal structures)
    contours, _ = cv2.findContours(black_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    img_area = h * w
    min_area_pixels = max(1, int(img_area * 0.00005))
    max_area = int(img_area * float(max_area_ratio)) if max_area_ratio and max_area_ratio > 0 else img_area

    piece_mask = np.zeros((h, w), dtype=np.uint8)
    piece_pixels = 0
    accepted_count = 0
    for cnt in contours:
        area = int(cv2.contourArea(cnt))
        if area < min_area_pixels:
            continue
        if area > max_area:
            continue
        # fill contour interior
        cv2.drawContours(piece_mask, [cnt], -1, color=255, thickness=cv2.FILLED)
        piece_pixels += area
        accepted_count += 1

    piece_mask_final = piece_mask.astype(bool)

    red_pixels = int(np.count_nonzero(red_bool))
    used_red_pixels = int(np.count_nonzero(red_bool & piece_mask_final))

    piece_ratio = (piece_pixels / red_pixels) if red_pixels else 0.0
    warning = (piece_ratio < 0.20) or (piece_ratio > 0.95)

    # debug images
    try:
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'debug', base))
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, 'black_mask.png'), black_closed)
        cv2.imwrite(os.path.join(out_dir, 'red_mask.png'), red_mask)
        cv2.imwrite(os.path.join(out_dir, 'piece_mask.png'), (piece_mask_final.astype(np.uint8) * 255))
        overlap = np.zeros((h, w, 3), dtype=np.uint8)
        overlap[red_bool] = (0, 0, 255)
        overlap[piece_mask_final] = (255, 0, 0)
        cv2.imwrite(os.path.join(out_dir, 'overlap.png'), overlap)
    except Exception:
        pass

    return red_pixels, used_red_pixels, accepted_count, int(np.count_nonzero(black_closed)), piece_pixels, piece_ratio, warning


def analyze_image_wrapper(arg):
    """Wrapper for multiprocessing: accepts path or (path, scale) and returns (basename, dict)

    Dict keys: red_pixels, used_red_pixels, utilization
    """
    if isinstance(arg, (list, tuple)):
        image_path = arg[0]
        scale = int(arg[1]) if len(arg) > 1 else 1
    else:
        image_path = arg
        scale = 1
    try:
        res = analyze_image(image_path, scale=scale)
        # analyze_image may return expanded stats; handle accordingly
        if isinstance(res, (list, tuple)):
            if len(res) >= 3:
                red = int(res[0])
                used = int(res[1])
                pieces = int(res[2])
            else:
                raise ValueError("analyze_image returned insufficient values")
            # optional additional values
            piece_pixels = int(res[4]) if len(res) > 4 else None
            piece_ratio = float(res[5]) if len(res) > 5 else None
            warning = bool(res[6]) if len(res) > 6 else False
        else:
            raise ValueError("analyze_image returned unexpected type")

        util_pct = (used / red * 100.0) if red else 0.0
        out = {"red_pixels": red, "used_red_pixels": used, "utilization": float(util_pct)}
        if piece_pixels is not None:
            out.update({"piece_pixels": int(piece_pixels), "piece_ratio": float(piece_ratio), "warning": bool(warning)})
        return os.path.basename(image_path), out
    except Exception as e:
        return os.path.basename(image_path), {"red_pixels": 0, "used_red_pixels": 0, "utilization": 0.0, "error": str(e)}


def process_all(dataset_dir: str) -> Dict[str, Dict[str, float]]:
    """Process all PNG files in dataset_dir and return mapping filename -> stats dict"""
    files = find_png_files(dataset_dir)
    if not files:
        raise FileNotFoundError(f"No PNG files found in {dataset_dir}")
    results: Dict[str, Dict[str, float]] = {}
    for p in files:
        name, stats = analyze_image_wrapper(p)
        results[name] = stats
    return results
