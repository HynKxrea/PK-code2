import numpy as np
from collections import deque


def find_connected_components(mask):
    """
    mask: 2D boolean/uint8 array
    returns list of components: each is dict with keys 'pixels', 'area_px', 'bbox'=(minr,minc,maxr,maxc)
    """
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []

    for r in range(H):
        for c in range(W):
            if mask[r, c] and not visited[r, c]:
                # BFS/DFS
                q = deque()
                q.append((r, c))
                visited[r, c] = True
                pixels = []
                minr = r; maxr = r
                minc = c; maxc = c
                while q:
                    y, x = q.popleft()
                    pixels.append((y, x))
                    if y < minr: minr = y
                    if y > maxr: maxr = y
                    if x < minc: minc = x
                    if x > maxc: maxc = x
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny = y + dy; nx = x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
                comp = {
                    'area_px': len(pixels),
                    'bbox': (minr, minc, maxr, maxc)
                }
                comps.append(comp)
    return comps


def compute_ccs_info(quality_map, allowed_grades, mm_per_px_x, mm_per_px_y):
    """
    quality_map: 2D int array with quality levels (1..5, 0 for none)
    allowed_grades: iterable of grade integers (1..5)
    returns list of components sorted by area desc; each has area_mm2 and bbox_px=(h,w)
    """
    mask = np.isin(quality_map, allowed_grades)
    if mask.sum() == 0:
        return []

    comps = find_connected_components(mask.astype(np.uint8))

    out = []
    pixel_area = mm_per_px_x * mm_per_px_y
    for c in comps:
        area_mm2 = c['area_px'] * pixel_area
        minr, minc, maxr, maxc = c['bbox']
        h = maxr - minr + 1
        w = maxc - minc + 1
        out.append({'area_mm2': area_mm2, 'bbox_px': (h, w), 'area_px': c['area_px']})

    out.sort(key=lambda x: x['area_mm2'], reverse=True)
    return out


def piece_bbox_from_mask(piece_mask):
    """
    piece_mask: 2D uint8 array where piece pixels are 1
    returns bbox_px (h,w) and area_px
    """
    ys, xs = np.nonzero(piece_mask)
    if len(ys) == 0:
        return (0, 0), 0
    minr = ys.min(); maxr = ys.max()
    minc = xs.min(); maxc = xs.max()
    h = maxr - minr + 1
    w = maxc - minc + 1
    area_px = len(ys)
    return (h, w), area_px


def packing_efficiency_by_area(area_mm2):
    """
    Simple tiered packing efficiency based on piece area
    """
    if area_mm2 < 2000:
        return 0.85
    if area_mm2 < 10000:
        return 0.75
    return 0.65


def compute_fittable_counts_for_leather(quality_map, mm_per_px_x, mm_per_px_y,
                                        pieces_masks, pieces_meta, demand,
                                        grade_radius=1, decay=0.15,
                                        bbox_fit_threshold=1.0):
    """
    pieces_masks: dict piece_name -> mask (2D uint8) as loaded by snh_setting.load_pieces
    pieces_meta: dict piece_name -> meta dict containing area_mm2 (from loader)
    demand: dict piece_name -> remaining demand (int)

    Returns: dict piece_name -> fittable_count (int) for this leather considered alone
    """
    # mm per pixel area
    pixel_area = mm_per_px_x * mm_per_px_y

    # We'll compute for each unique preferred grade set; but simpler: for each piece, compute allowed grades
    fittable = {pname: 0 for pname in pieces_meta.keys() if pname in demand}

    # Precompute piece bbox and area_px
    piece_info = {}
    for pname, mask in pieces_masks.items():
        pname_up = pname.upper()
        if pname_up not in pieces_meta:
            continue
        (ph, pw), area_px = piece_bbox_from_mask(mask)
        area_mm2 = pieces_meta[pname_up]['area_mm2']
        piece_info[pname_up] = {'bbox_px': (ph, pw), 'area_mm2': area_mm2}

    # Precompute ccs per grade-window? Simpler: for each piece compute allowed grades and ccs
    H, W = quality_map.shape

    for pname, info in piece_info.items():
        if pname not in demand or demand[pname] <= 0:
            continue
        A = info['area_mm2']
        pref = None
        # try to get preferred grade from NAME heuristics: caller should provide assign_grade function if needed
        # Here we infer grade from piece_meta if available
        # Fallback: assume preferred grade 1
        pref = pieces_meta.get(pname, {}).get('preferred_grade', None) or pieces_meta.get(pname, {}).get('suggested_grade', None)
        if pref is None:
            # try to infer from name text in pieces_meta key
            try:
                name = pname.upper()
                if any(k in name for k in ["TOP", "FRONT", "BACK"]):
                    pref = 1
                elif any(k in name for k in ["GUSSET", "HANDLE"]):
                    pref = 2
                elif "LINING" in name:
                    pref = 3
                else:
                    pref = 4
            except Exception:
                pref = 3

        # allow only equal or better quality (no downgrading)
        # better quality has smaller numeric grade (Q1 best). We allow grades from (pref - radius) up to pref only.
        gmin = max(1, pref - grade_radius)  # better quality indices (smaller numbers)
        gmax = pref
        grades = list(range(gmin, gmax + 1))

        ccs = compute_ccs_info(quality_map, grades, mm_per_px_x, mm_per_px_y)
        if not ccs:
            fittable[pname] = 0
            continue

        eta = packing_efficiency_by_area(A)

        remaining_needed = demand[pname]
        total_fit = 0
        for idx, cc in enumerate(ccs):
            if remaining_needed <= 0:
                break
            cc_area = cc['area_mm2']
            # bbox check: convert piece bbox (ph,pw) in pixels to leather pixel units -- both are in pixels
            ph, pw = info['bbox_px']
            ch, cw = cc['bbox_px']
            # If bbox_fit_threshold == 1.0 require both dims >=, else allow partial
            fits_bbox = (ch >= ph and cw >= pw)
            size_factor = 1.0 if fits_bbox else bbox_fit_threshold
            # effective area with packing and fragmentation decay
            effective = cc_area * eta * (1.0 - decay * idx) * size_factor
            # number of pieces that can fit in this cc
            fit_count = int(effective // A) if A > 0 else 0
            if fit_count <= 0:
                # small CC might still accommodate if effective >= 0.5*A; allow 1 if so
                if effective >= 0.5 * A:
                    fit_count = 1
                else:
                    fit_count = 0
            use = min(fit_count, remaining_needed)
            total_fit += use
            remaining_needed -= use

        fittable[pname] = total_fit

    return fittable
