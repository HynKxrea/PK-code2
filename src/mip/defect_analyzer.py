"""Q 구역별 결함 팽창 기반 유효 면적 계산 (SVG 폴리곤 기반).

leather_svg.xml의 폴리곤 좌표를 Shapely로 처리하여
결함 팽창 후 유효 면적과 파편화 비율을 반환한다.

반환값: {"effective": [Q1_eff, Q1Q2_eff, Q1Q2Q3_eff, Q1Q2Q3Q4_eff],
         "largest_blob_ratio": [r1, r2, r3, r4]}  (단위: mm²)

캐시: dataset/effective_areas.json
      파라미터(DILATION_RADIUS_MM)가 바뀌면 캐시 삭제 후 재실행.
"""

import json
import os
import re
import xml.etree.ElementTree as ET

from shapely.geometry import Polygon
from shapely.ops import unary_union

# =============================================
# 파라미터 설정
# =============================================
DILATION_RADIUS_MM = 17.0
CACHE_PATH = os.path.join("dataset", "effective_areas.json")
# =============================================

_QUALITY_LEVELS = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4, 'Q5': 5}


def _extract_rings(path_data_str, scale_x, scale_y):
    """SVG PathData → [(points, is_ccw), ...] 링 리스트."""
    from shapely.geometry import LinearRing as LR
    normalized = re.sub(r'([MLZmlz])', r' \1 ', path_data_str)
    tokens = normalized.split()

    rings = []
    current = []
    cmd = None
    i = 0

    while i < len(tokens):
        t = tokens[i]
        if t in ('M', 'L', 'm', 'l'):
            cmd = t.upper()
            i += 1
        elif t in ('Z', 'z'):
            if len(current) >= 3:
                try:
                    lr = LR(current)
                    rings.append((current[:], lr.is_ccw))
                except Exception:
                    pass
            current = []
            cmd = None
            i += 1
        elif cmd in ('M', 'L') and i + 1 < len(tokens) and tokens[i + 1] not in ('M', 'L', 'Z', 'm', 'l', 'z'):
            try:
                x = float(t) * scale_x
                y = float(tokens[i + 1]) * scale_y
                current.append((x, y))
                i += 2
            except ValueError:
                i += 1
        else:
            i += 1

    if len(current) >= 3:
        try:
            from shapely.geometry import LinearRing as LR
            lr = LR(current)
            rings.append((current[:], lr.is_ccw))
        except Exception:
            pass

    return rings


def _parse_svg_paths(path_data_str, scale_x, scale_y):
    """SVG PathData → shapely Polygon 리스트 (hole 포함 처리)."""
    rings = _extract_rings(path_data_str, scale_x, scale_y)

    # is_ccw=True → 외곽(exterior), is_ccw=False → 구멍(hole)
    ext_rings  = [pts for pts, ccw in rings if ccw]
    hole_rings = [pts for pts, ccw in rings if not ccw]

    polygons = []
    for ext in ext_rings:
        try:
            ext_poly = Polygon(ext)
            # 이 외곽에 포함된 hole만 할당
            my_holes = [h for h in hole_rings
                        if ext_poly.contains(Polygon(h).centroid)]
            poly = Polygon(ext, my_holes)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_valid and poly.area > 0:
                polygons.append(poly)
        except Exception:
            pass

    return polygons


def _load_quality_polygons(svg_xml_path):
    """leather_svg.xml → {1: [Polygon,...], 2: [...], ...} (mm 좌표계)."""
    tree = ET.parse(svg_xml_path)
    root = tree.getroot()

    meta = root.find('Metadata')
    vb = meta.find('viewBox').text.strip().split()
    vb_w, vb_h = float(vb[2]), float(vb[3])
    w_mm = float(meta.find('width').text.replace('mm', '').strip())
    h_mm = float(meta.find('height').text.replace('mm', '').strip())
    scale_x = w_mm / vb_w
    scale_y = h_mm / vb_h

    quality_polys = {1: [], 2: [], 3: [], 4: [], 5: []}
    seen_levels = set()
    for path_el in root.iter('Path'):
        pid = path_el.get('id', '')
        if not pid.startswith('merged'):
            continue
        level = _QUALITY_LEVELS.get(path_el.get('qualityLevel', ''))
        if level is None:
            continue
        path_data = path_el.findtext('PathData', '')
        if path_data.strip():
            quality_polys[level].extend(_parse_svg_paths(path_data, scale_x, scale_y))

    return quality_polys


def _find_leather_folder(leathers_dir, leather_code):
    code_upper = leather_code.strip().upper()
    for name in os.listdir(leathers_dir):
        if code_upper in name.upper():
            full = os.path.join(leathers_dir, name)
            if os.path.isdir(full):
                return full
    return None


def compute_effective_areas(leather_code, leathers_dir, dilation_radius_mm=DILATION_RADIUS_MM):
    folder = _find_leather_folder(leathers_dir, leather_code)
    if folder is None:
        return None

    svg_path = os.path.join(folder, 'leather_svg.xml')
    if not os.path.exists(svg_path):
        return None

    try:
        quality_polys = _load_quality_polygons(svg_path)
    except Exception as e:
        print(f"[defect_analyzer] SVG 파싱 실패 {leather_code}: {e}")
        return None

    # Q별 union geometry
    quality_geoms = {}
    for k in range(1, 6):
        polys = quality_polys[k]
        if polys:
            try:
                quality_geoms[k] = unary_union(polys)
            except Exception:
                quality_geoms[k] = None
        else:
            quality_geoms[k] = None

    effective = []
    largest_blob_ratio = []

    for k in range(1, 5):
        # 사용 가능 영역: Q1~Qk 합집합
        usable_list = [quality_geoms[q] for q in range(1, k + 1) if quality_geoms.get(q)]
        usable = unary_union(usable_list) if usable_list else None

        # 결함 영역: Q(k+1)~Q5 합집합을 dilation_radius_mm만큼 팽창
        defect_list = [quality_geoms[q] for q in range(k + 1, 6) if quality_geoms.get(q)]
        dilated = unary_union(defect_list).buffer(dilation_radius_mm) if defect_list else None

        if usable is None:
            effective.append(0.0)
            largest_blob_ratio.append(0.0)
            continue

        eff_geom = usable.difference(dilated) if dilated else usable
        if not eff_geom.is_valid:
            eff_geom = eff_geom.buffer(0)

        eff_area = float(eff_geom.area)
        effective.append(eff_area)

        # 파편화 비율: 최대 연속 폴리곤 면적 / 전체 유효 면적
        if eff_area > 0:
            if eff_geom.geom_type == 'Polygon':
                largest_blob_ratio.append(1.0)
            elif eff_geom.geom_type == 'MultiPolygon':
                largest = max(p.area for p in eff_geom.geoms)
                largest_blob_ratio.append(float(largest / eff_area))
            else:
                largest_blob_ratio.append(0.0)
        else:
            largest_blob_ratio.append(0.0)

    return {"effective": effective, "largest_blob_ratio": largest_blob_ratio}


def load_all_effective_areas(leather_codes, leathers_dir, dilation_radius_mm=DILATION_RADIUS_MM):
    if os.path.exists(CACHE_PATH):
        print(f"[defect_analyzer] 캐시 로드: {CACHE_PATH}")
        with open(CACHE_PATH, encoding='utf-8') as f:
            return json.load(f)

    print(f"[defect_analyzer] SVG 기반 유효 면적 계산 중 (가죽 {len(leather_codes)}개)...")
    result = {}
    for code in leather_codes:
        result[code] = compute_effective_areas(code, leathers_dir, dilation_radius_mm)
        if not result[code]:
            print(f"  {code}: FAIL")

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"[defect_analyzer] 캐시 저장 완료: {CACHE_PATH}")

    return result
