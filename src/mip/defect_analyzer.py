"""Q 구역별 결함 팽창 기반 유효 면적 계산.

각 품질 구역 k에 대해, 하위 품질 픽셀을 DILATION_RADIUS_MM만큼 팽창시킨 후
배치 불가 영역을 제외한 유효 면적(Q'_kj)을 반환한다.

반환값: [Q1_eff, (Q1+Q2)_eff, (Q1+Q2+Q3)_eff, (Q1+Q2+Q3+Q4)_eff] (mm²)

캐시: dataset/effective_areas.json 에 저장되며, 이미 존재하면 재계산 없이 읽어옴.
      파라미터(DILATION_RADIUS_MM)가 바뀌면 캐시 파일을 삭제하고 재실행.
"""

import json
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image

# =============================================
# 파라미터 설정
# =============================================
DILATION_RADIUS_MM = 44.0   # 결함 팽창 반경 (mm) — 가장 작은 패턴 최소 폭 기준
COLOR_TOLERANCE    = 40     # 픽셀 색상 인식 허용 오차
CACHE_PATH         = os.path.join("dataset", "effective_areas.json")
# =============================================

_QUALITY_COLORS = {
    1: (255,  30,   3),
    2: (255, 255,   0),
    3: (  0, 226,  43),
    4: (  0, 250, 250),
    5: (  0,  95, 250),
}


def _color_mask(arr, rgb, tol=COLOR_TOLERANCE):
    r, g, b = rgb
    return (
        (np.abs(arr[:, :, 0].astype(int) - r) < tol) &
        (np.abs(arr[:, :, 1].astype(int) - g) < tol) &
        (np.abs(arr[:, :, 2].astype(int) - b) < tol)
    )


def _find_leather_folder(leathers_dir, leather_code):
    code_upper = leather_code.strip().upper()
    for name in os.listdir(leathers_dir):
        if code_upper in name.upper():
            full = os.path.join(leathers_dir, name)
            if os.path.isdir(full):
                return full
    return None


def _mm2_per_px2(stat_xml_path, img_shape):
    try:
        root = ET.parse(stat_xml_path).getroot()
        dims = root.find("HideInfo/Dimensions")
        w_mm = float(dims.find("Width").text.replace("mm", "").strip())
        h_mm = float(dims.find("Height").text.replace("mm", "").strip())
        h_px, w_px = img_shape[:2]
        return (w_mm / w_px) * (h_mm / h_px)
    except Exception:
        return None


def compute_effective_areas(leather_code, leathers_dir, dilation_radius_mm=DILATION_RADIUS_MM):
    """
    Returns:
        [Q1_eff, (Q1+Q2)_eff, (Q1+Q2+Q3)_eff, (Q1+Q2+Q3+Q4)_eff] in mm²
        None if PNG or stat.xml not found.
    """
    folder = _find_leather_folder(leathers_dir, leather_code)
    if folder is None:
        return None

    png_path  = os.path.join(folder, "leather_preview.png")
    stat_path = os.path.join(folder, "leather_stat.xml")
    if not os.path.exists(png_path) or not os.path.exists(stat_path):
        return None

    arr   = np.array(Image.open(png_path).convert("RGB"))
    scale = _mm2_per_px2(stat_path, arr.shape)
    if scale is None:
        return None

    r_px   = max(1, int(dilation_radius_mm / (scale ** 0.5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1))
    masks  = {k: _color_mask(arr, _QUALITY_COLORS[k]) for k in range(1, 6)}

    effective = []
    largest_blob_ratio = []
    for k in range(1, 5):
        usable = np.zeros(arr.shape[:2], dtype=bool)
        for q in range(1, k + 1):
            usable |= masks[q]
        defect = np.zeros(arr.shape[:2], dtype=bool)
        for q in range(k + 1, 6):
            defect |= masks[q]
        dilated  = cv2.dilate(defect.astype(np.uint8), kernel).astype(bool)
        eff_mask = usable & ~dilated
        eff_px   = float(eff_mask.sum())
        effective.append(eff_px * scale)

        # 파편화 비율: 최대 연속 블롭 면적 / 전체 유효 면적
        if eff_px > 0:
            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
                eff_mask.astype(np.uint8), connectivity=8
            )
            # label 0 = 배경 제외
            if n_labels > 1:
                largest = float(stats[1:, cv2.CC_STAT_AREA].max())
            else:
                largest = 0.0
            largest_blob_ratio.append(largest / eff_px)
        else:
            largest_blob_ratio.append(0.0)

    return {"effective": effective, "largest_blob_ratio": largest_blob_ratio}


def load_all_effective_areas(leather_codes, leathers_dir, dilation_radius_mm=DILATION_RADIUS_MM):
    """
    모든 가죽에 대해 유효 면적을 반환. 캐시가 있으면 바로 읽고, 없으면 계산 후 저장.

    Returns:
        {leather_code: [Q1_eff, ...] or None}
    """
    # 캐시 읽기
    if os.path.exists(CACHE_PATH):
        print(f"[defect_analyzer] 캐시 로드: {CACHE_PATH}")
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)

    # 캐시 없으면 계산
    print(f"[defect_analyzer] 유효 면적 계산 중 (가죽 {len(leather_codes)}개)...")
    result = {}
    for code in leather_codes:
        result[code] = compute_effective_areas(code, leathers_dir, dilation_radius_mm)

    # 저장
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[defect_analyzer] 캐시 저장 완료: {CACHE_PATH}")

    return result
