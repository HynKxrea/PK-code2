import os
import csv
from .loader import (
    load_leathers,
    load_pieces,
    load_demand,
    load_piece_materials
)

# =========================
# Path
# =========================

# compute project root regardless of this file's depth
_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Use dataset folder for leathers and pieces
LEATHERS_DIR = os.path.join(_BASE, 'dataset', 'leathers')
PIECES_DIR = os.path.join(_BASE, 'dataset', 'pieces')

# For demand, prefer CZ402 XML in pieces dir if present, otherwise use pieces dir
import glob
cz_files = glob.glob(os.path.join(PIECES_DIR, 'CZ402*'))
if cz_files:
    DEMAND_PATH = cz_files[0]
else:
    DEMAND_PATH = PIECES_DIR

# For backward compatibility, set variables used elsewhere
LEATHER_PATH = LEATHERS_DIR
PATTERN_PATH = PIECES_DIR
pattern_dir = PIECES_DIR


# =========================
# Load Data
# =========================

# Set production quantity (number of bags to produce)
PRODUCTION_QTY = 20

leathers = load_leathers(LEATHER_PATH)

pieces = load_pieces(PATTERN_PATH)

# Load demand for the total production quantity (from CZ402 XML if present)
if os.path.isfile(DEMAND_PATH) and DEMAND_PATH.lower().endswith('.xml'):
    # Robust parsing of pattern XML using text-block extraction to reliably get NAME and QUANTITY
    import re
    from collections import defaultdict

    raw_counts = defaultdict(int)

    with open(DEMAND_PATH, 'r', encoding='utf-8') as fh:
        text = fh.read()

    # find all PIECE blocks
    blocks = re.findall(r'<PIECE>(.*?)</PIECE>', text, flags=re.S | re.I)

    for b in blocks:
        # extract NAME
        mname = re.search(r'<NAME>(.*?)</NAME>', b, flags=re.S | re.I)
        if not mname:
            continue
        raw_name = mname.group(1).strip()
        if '__' in raw_name:
            piece_name = raw_name.split('__', 1)[1].strip().upper()
        else:
            piece_name = raw_name.upper()

        # extract QUANTITY (try several tag forms)
        mqty = re.search(r'<QUANTITY>\s*(\d+)\s*</QUANTITY>', b, flags=re.I)
        if not mqty:
            mqty = re.search(r'<QTY>\s*(\d+)\s*</QTY>', b, flags=re.I)
        if not mqty:
            # try to find numeric after literal 'Quantity:' in any text
            mqty = re.search(r'Quantity\s*[:=]?\s*(\d+)', b, flags=re.I)
        qty = int(mqty.group(1)) if mqty else 1

        raw_counts[piece_name] += qty

    _demand_per_unit = dict(raw_counts)
    demand = {k: v * PRODUCTION_QTY for k, v in raw_counts.items()}
    _raw_pattern_types = len(_demand_per_unit)
    _raw_pieces_per_bag = sum(_demand_per_unit.values())

else:
    demand = load_demand(
        DEMAND_PATH,
        production_qty=PRODUCTION_QTY
    )

    # Also load demand for a single bag (production_qty=1) to check per-unit piece counts
    _demand_per_unit = load_demand(
        DEMAND_PATH,
        production_qty=1
    )

    # Capture raw counts from the demand file before filtering
    _raw_pattern_types = len(_demand_per_unit)
    _raw_pieces_per_bag = sum(_demand_per_unit.values())

# Treat all patterns as leather patterns (no material filtering)
# Previously we filtered demand to only pieces whose MATERIAL indicated leather.
# Per user's instruction, disable that filtering: use all patterns from the demand file.
# leather_piece_set = load_piece_materials(PATTERN_PATH)

# No filtering applied: demand and _demand_per_unit remain as parsed from the source file


# =========================
# Initial Data Summary (concise, English labels)
# =========================

# We'll print a short summary with the requested five lines:
# - Total Leathers
# - Pattern types per bag (distinct leather pattern types used per bag)
# - Pieces per bag (total leather pieces required per bag, sum of quantities)
# - Production quantity (number of bags)
# - Total pieces required (pieces per bag * production qty)


total_leathers = len(leathers)
pattern_types_per_bag = len(_demand_per_unit)
pieces_per_bag = sum(_demand_per_unit.values())
production_qty = PRODUCTION_QTY
total_pieces_required = sum(demand.values())  # already multiplied by production_qty

# Raw file-based counts (before material filtering)
raw_pattern_types = _raw_pattern_types
raw_pieces_per_bag = _raw_pieces_per_bag


def _load_piece_required_quality_map(csv_path: str) -> dict[str, int]:
    """Load required quality grade (Q1..Q5 -> 1..5) from piece_catalog_area_rule.csv.

    The catalog 'name' column may include pattern prefixes; we normalize to the
    suffix after '__' to match `load_pieces()` normalization.
    """

    mapping: dict[str, int] = {}

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_name = (row.get("name") or "").strip()
                if not raw_name:
                    continue

                if "__" in raw_name:
                    piece_name = raw_name.split("__", 1)[1].strip().upper()
                else:
                    piece_name = raw_name.strip().upper()

                q = (row.get("required_quality_area_rule") or "").strip().upper()
                if not q.startswith("Q"):
                    continue

                try:
                    grade = int(q[1:])
                except Exception:
                    continue

                if 1 <= grade <= 5:
                    mapping[piece_name] = grade

    except Exception:
        return {}

    return mapping


# Piece required quality mapping (used by visualization, etc.)
PIECE_CATALOG_PATH = os.path.join(PIECES_DIR, "piece_catalog_area_rule.csv")
REQUIRED_QUALITY_MAP: dict[str, int] = _load_piece_required_quality_map(PIECE_CATALOG_PATH)


def get_required_quality(piece_name: str) -> int | None:
    """Return required quality grade for a piece (1..5) if known."""

    if not piece_name:
        return None
    return REQUIRED_QUALITY_MAP.get(str(piece_name).strip().upper())


def required_grade_for_piece(piece_name: str) -> int:
    """Return required grade for piece.

    Priority:
    1) dataset/pieces/piece_catalog_area_rule.csv (REQUIRED_QUALITY_MAP)
    2) name-based heuristic (assign_grade)
    """

    rq = get_required_quality(piece_name)
    return int(rq) if rq is not None else int(assign_grade(piece_name))


def assign_grade(name):

    name = name.upper()

    if any(k in name for k in ["TOP", "FRONT", "BACK"]):
        return 1

    elif any(k in name for k in ["GUSSET", "HANDLE"]):
        return 2

    elif "LINING" in name:
        return 3

    else:
        return 4

def merge_data(pieces, demand):

    """Merge piece definitions (from load_pieces) with demand (per-piece quantities).

    pieces: dict of piece_name -> info dict (area_mm2, material, ...)
    demand: dict of piece_name -> total required qty

    Returns data: dict piece_name -> { 'area': area_mm2, 'demand': qty, 'material': ..., 'geom': ... }
    """

    data = {}

    for name, info in pieces.items():

        piece_name_up = name.upper()

        d = demand.get(piece_name_up, 0)

        if d == 0:
            continue

        area_mm2 = info.get('area_mm2') if isinstance(info, dict) else float(info)

        data[name] = {
            "area": area_mm2,
            "demand": d,
            "material": info.get('material') if isinstance(info, dict) else None,
            "unique": info.get('unique') if isinstance(info, dict) else None,
            "geom_attrs": info.get('geom_attrs') if isinstance(info, dict) else None
        }

    return data

def calculate_leather_score(leathers):

    """
    Higher score
    = higher quality leather
    = more valuable leather
    """

    # Q1 ~ Q5 weights
    quality_weights = [10, 7, 5, 2, 1]

    leather_scores = {}

    for leather_name, grades in leathers.items():

        score = 0

        # only use Q1 ~ Q5
        for idx in range(5):

            area = grades[idx]

            weight = quality_weights[idx]

            score += weight * area

        leather_scores[leather_name] = round(score, 2)

    return leather_scores
