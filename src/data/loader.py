import xml.etree.ElementTree as ET
import pandas as pd
import os
import glob


# =========================
# Leather Loader
# =========================

def load_leathers(folder_path):
    """Load leather quality areas.

    Supports two layouts:
    1) Flat folder containing many *_stat.xml files.
    2) Dataset layout: dataset/leathers/<leather_dir>/leather_stat.xml (nested).

    Returns: dict hide_code -> [Q1..Q7] areas (mm^2)
    """
    leathers: dict[str, list[float]] = {}

    if not os.path.exists(folder_path):
        return leathers

    # Collect stat xml files (flat + recursive)
    stat_files: list[str] = []

    # flat
    try:
        for file in os.listdir(folder_path):
            if file.endswith("_stat.xml"):
                stat_files.append(os.path.join(folder_path, file))
    except Exception:
        pass

    # recursive (dataset layout)
    stat_files.extend(glob.glob(os.path.join(folder_path, "**", "*_stat.xml"), recursive=True))

    # de-duplicate
    stat_files = sorted(set(stat_files))

    for file_path in stat_files:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except Exception:
            continue

        # ex) hide_code attribute like "HIDE_CODE ABC123"
        hide_code_attr = root.attrib.get("hide_code", "")
        code = hide_code_attr.split()[-1].strip() if hide_code_attr else ""
        if not code:
            # fallback: derive from folder name e.g. 012_CY817_09_L0696168_Q1_71.45pct
            code = os.path.basename(os.path.dirname(file_path)).split("_")[-2] if "_" in os.path.basename(os.path.dirname(file_path)) else os.path.basename(os.path.dirname(file_path))

        Q = [0.0] * 7
        quality_levels = root.find("QualityLevels")
        if quality_levels is None:
            continue

        for level in quality_levels.findall("Level"):
            name = level.attrib.get("name", "")
            if not name.startswith("Q"):
                continue
            try:
                q = int(name[1])
            except Exception:
                continue
            if 1 <= q <= 7:
                area_node = level.find("Area_mm2")
                if area_node is not None and area_node.text is not None:
                    try:
                        Q[q - 1] = float(area_node.text)
                    except Exception:
                        Q[q - 1] = 0.0

        leathers[code] = Q

    return leathers


# =========================
# Piece Loader
# =========================

def load_pieces(folder_path):

    """Load piece definitions from pattern XML files.

    Returns a dict mapping piece_name (string) -> info dict with keys:
      - area_mm2: float
      - area_cm2: float
      - material: str or ''
      - unique: str or None
      - geom_attrs: dict of GEOM_INFO attributes

    This preserves more metadata so downstream code can distinguish pattern types and piece counts.
    """

    pieces = {}

    for file in os.listdir(folder_path):

        if not file.endswith(".xml"):
            continue

        file_path = os.path.join(folder_path, file)

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except Exception:
            continue

        for piece in root.findall("PIECE"):

            name_node = piece.find("NAME")

            if name_node is None or name_node.text is None:
                continue

            name = name_node.text.strip()
            # Normalize names to match demand keys: if pattern name contains '__', keep the suffix part.
            # Example: 'COH2709 A 17-21__FLAP TOP' -> 'FLAP TOP'
            if '__' in name:
                name = name.split('__', 1)[1].strip()

            # material (if present)
            mat_node = piece.find("MATERIAL")
            material = mat_node.text.strip() if mat_node is not None and mat_node.text is not None else ""

            unique_node = piece.find("UNIQUE")
            unique = unique_node.text.strip() if unique_node is not None and unique_node.text is not None else None

            size_node = piece.find("SIZE")

            if size_node is None:
                continue

            geom = size_node.find("GEOM_INFO")

            if geom is None:
                continue

            try:
                area_cm2 = float(geom.attrib.get("AREA", 0.0))
            except Exception:
                area_cm2 = 0.0

            area_mm2 = area_cm2 * 100.0

            geom_attrs = dict(geom.attrib)

            pieces[name.upper()] = {
                "area_cm2": area_cm2,
                "area_mm2": area_mm2,
                "material": material.upper(),
                "unique": unique,
                "geom_attrs": geom_attrs
            }

    return pieces


def load_piece_materials(folder_path):
    """Parse pattern XML files and return a set of piece names that use leather materials.
    Heuristic: consider MATERIAL child text; if it starts with 'LTR' or contains the word 'LEATHER',
    treat it as leather.
    Returns a set of piece names (uppercased) that are leather pieces.
    """
    leather_pieces = set()

    for file in os.listdir(folder_path):
        if not file.endswith(".xml"):
            continue

        file_path = os.path.join(folder_path, file)

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except Exception:
            continue

        for piece in root.findall("PIECE"):
            name_node = piece.find("NAME")
            mat_node = piece.find("MATERIAL")

            if name_node is None:
                continue

            raw_name = name_node.text.strip()
            if '__' in raw_name:
                raw_name = raw_name.split('__', 1)[1].strip()
            name = raw_name.upper()
            mat_text = "" if mat_node is None or mat_node.text is None else mat_node.text.strip().upper()

            if mat_text.startswith("LTR") or "LEATHER" in mat_text:
                leather_pieces.add(name)

    return leather_pieces


# =========================
# Demand Loader
# =========================

def _detect_columns(df):
    """Detect likely columns in a dataframe.
    Return (item_col, piece_col, qty_col) where:
      - item_col: column name that describes the material/item (e.g., 'ITEM NAME')
      - piece_col: column that gives the piece name (e.g., 'PIECE')
      - qty_col: column with quantity
    Any of them may be None if not detected.
    """
    cols = [c.upper().strip() for c in df.columns]
    col_map = {c.upper().strip(): c for c in df.columns}

    item_col = None
    piece_col = None
    qty_col = None

    # Heuristics for item/material column
    for candidate in ["ITEM NAME", "ITEM_NAME", "ITEM", "MATERIAL", "ITEM CODE", "ITEM_CODE"]:
        if candidate in cols:
            item_col = col_map[candidate]
            break

    # Heuristics for piece column
    for candidate in ["PIECE", "PIECE NAME", "PIECE_NAME", "NAME", "PIECE UNIQUE"]:
        if candidate in cols:
            piece_col = col_map[candidate]
            break

    # Heuristics for qty column
    for candidate in ["PIECE QTY", "QTY", "QUANTITY", "ITEM QTY", "PIECE_QTY"]:
        if candidate in cols:
            qty_col = col_map[candidate]
            break

    return item_col, piece_col, qty_col



def _extract_piece_name_from_item(item_name):
    """If item_name starts with 'LEATHER', strip that prefix and separators and return the remaining name.
    Otherwise return the original name.
    """
    if item_name is None:
        return None

    s = str(item_name).strip()
    s_up = s.upper()

    if s_up.startswith("LEATHER"):
        # remove the 'LEATHER' prefix and common separators
        rest = s[len("LEATHER"):]
        # strip common separators and whitespace
        rest = rest.lstrip(" -_:.")
        return rest.strip().upper()

    # otherwise return full name uppercased
    return s_up


def load_demand(folder_path, production_qty=1):

    demand = {}

    for file in os.listdir(folder_path):

        if not file.endswith(".xlsx"):
            continue

        file_path = os.path.join(folder_path, file)

        df = pd.read_excel(file_path)

        item_col, piece_col, qty_col = _detect_columns(df)

        if qty_col is None:
            # couldn't detect quantity column; skip this file
            continue

        # If piece_col is not detected but item_col is, we may try to extract piece name from item
        for _, row in df.iterrows():

            raw_item = None if item_col is None else row[item_col]
            raw_piece = None if piece_col is None else row[piece_col]
            raw_qty = row[qty_col]

            if pd.isna(raw_qty):
                continue

            # Determine if this row corresponds to leather material
            item_name_up = "" if pd.isna(raw_item) else str(raw_item).strip().upper()

            is_leather = False
            if "LEATHER" in item_name_up:
                is_leather = True

            # If item column not available, try to infer from item code (if present)
            # (we won't implement complex heuristics here; prefer explicit ITEM NAME)

            if not is_leather:
                continue

            # Determine piece name: prefer PIECE column if present, otherwise try to extract from item
            if raw_piece is not None and not pd.isna(raw_piece):
                piece_name = str(raw_piece).strip().upper()
            else:
                # fallback: try to extract from item name
                piece_name = _extract_piece_name_from_item(raw_item)

            try:
                qty = int(raw_qty)
            except Exception:
                try:
                    qty = int(float(raw_qty))
                except Exception:
                    continue

            if piece_name is None or piece_name == "":
                continue

            # Normalize piece name
            piece_name = piece_name.upper()

            demand[piece_name] = demand.get(piece_name, 0) + qty

    # 생산량 반영
    for key in list(demand.keys()):
        demand[key] *= production_qty

    return demand


# Note:
# - This loader now filters rows to only include items whose ITEM NAME starts with 'LEATHER'.
# - It also attempts to detect the name/quantity columns flexibly (handles different column names).
# - The resulting demand keys are uppercased piece names (with the leading 'LEATHER' prefix removed).
