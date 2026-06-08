"""Run selection using geometry-based scoring (v_j computed from geometry) and then run the same MIP selection.
Usage: python3 src/main_geometry.py [--scale 0.08] [--bbox-threshold 1.0]
"""
import argparse
import os
import json

from data.preprocessing import (
    pieces,
    demand,
    leathers,
    merge_data
)
import pandas as pd



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=float, default=0.08, help='scale used when loading SVG pieces/skins')
    parser.add_argument('--bbox-threshold', type=float, default=1.0, help='bbox fit threshold')
    args = parser.parse_args()

    SCALE = args.scale
    BBOX_THRESHOLD = args.bbox_threshold

    # Data summary (inline)
    leather_df = pd.DataFrame.from_dict(
        leathers,
        orient='index',
        columns=["Q1","Q2","Q3","Q4","Q5","Q6","Q7"]
    )
    leather_df.index.name = 'Leather'
    leather_df['Total'] = leather_df.sum(axis=1)
    for q in ["Q1","Q2","Q3","Q4","Q5"]:
        leather_df[f"{q}_ratio"] = (leather_df[q] * 100 / leather_df['Total']).round(1)

    print('\n' + '='*60)
    print('LEATHER SUMMARY (top 10 by Total area)')
    print('='*60)
    try:
        print(leather_df.sort_values('Total', ascending=False).head(10))
    except Exception:
        print('Total leathers:', len(leathers))

    # prepare inputs
    leather_folder = os.path.join('data', 'L-code(_stat.xml)')
    piece_xml = os.path.join('data', 'Pattern(.xml, .xlsx)', 'COH2709RGL000025002002.xml')
    pieces_meta = {k.upper(): v for k, v in pieces.items()}
    demand_up = {k.upper(): v for k, v in demand.items()}

    # Compute geometry-based leather scores (compatibility) per leather
    from snh.snh_setting import load_leather, load_pieces
    from selection.compatibility import compute_fittable_counts_for_leather
    import glob

    # load piece masks at same scale
    piece_masks_raw = load_pieces(piece_xml, scale=SCALE)
    piece_masks = {k.upper(): v['mask'] if isinstance(v, dict) and 'mask' in v else v for k, v in piece_masks_raw.items()}

    # compute total demand area
    total_demand_area = 0.0
    for pname, qty in demand_up.items():
        meta = pieces_meta.get(pname.upper())
        if meta is None:
            continue
        total_demand_area += qty * meta.get('area_mm2', 0.0)

    leather_scores_override = {}

    for code in sorted(list(leathers.keys())):
        # find svg for this leather
        matches = glob.glob(f"data/L-code(_stat.xml)/*{code}*_svg.xml")
        if not matches:
            leather_scores_override[code] = 0.0
            continue
        svg = matches[0]
        try:
            leather_info = load_leather(svg, scale=SCALE)
            qmap = leather_info['quality_map']
            mmx = leather_info.get('mm_per_pixel_x', 1.0)
            mmy = leather_info.get('mm_per_pixel_y', 1.0)
        except Exception:
            leather_scores_override[code] = 0.0
            continue

        fittable = compute_fittable_counts_for_leather(qmap, mmx, mmy, piece_masks, pieces_meta, demand_up,
                                                       grade_radius=1, decay=0.15, bbox_fit_threshold=BBOX_THRESHOLD)
        # compute covered area
        covered_area = 0.0
        for pname, cnt in fittable.items():
            meta = pieces_meta.get(pname.upper())
            if meta is None:
                continue
            area_mm2 = meta.get('area_mm2', 0.0)
            covered_area += min(cnt, demand_up.get(pname, 0)) * area_mm2

        compat = (covered_area / total_demand_area) if total_demand_area > 0 else 0.0
        leather_scores_override[code] = compat

    print('\nComputed geometry-based leather scores (compatibility) for', len(leather_scores_override), 'leathers')

    # Now call the same MIP selection but supply leather_scores_override
    from selection.selection import solve_selection_mip
    data = merge_data(pieces, demand)

    # Also prepare the scaled scores for printing (same mapping used by selection)
    from data.preprocessing import calculate_leather_score
    quality_scores = calculate_leather_score(leathers)
    q_vals = list(quality_scores.values())
    q_min = min(q_vals) if q_vals else 0.0
    q_max = max(q_vals) if q_vals else 1.0
    q_range = q_max - q_min if (q_max - q_min) != 0 else 1.0

    scaled_scores = {}
    for code, raw in leather_scores_override.items():
        r = float(raw)
        if r < 0.0: r = 0.0
        if r > 1.0: r = 1.0
        scaled_scores[code] = q_min + (1.0 - r) * q_range

    result = solve_selection_mip(data, leathers, leather_scores_override=leather_scores_override, score_direction_override='maximize')

    # Present results similarly to quality mode
    if result is None:
        print('\n❌ No feasible solution found using geometry-based scores')
        return

    scores = scaled_scores
    selected = result['selected_leathers']
    assignment_df = result['assignment_df']

    # Problem summary (concise) - same format as main_quality
    from data.preprocessing import PRODUCTION_QTY

    print('\n' + '=' * 70)
    print('PROBLEM SUMMARY (Geometry score)')
    print('=' * 70)

    total_leathers = len(leathers)
    pattern_types_per_bag = len(demand)
    pieces_per_bag = int(sum(demand.values()) / PRODUCTION_QTY) if PRODUCTION_QTY else sum(demand.values())
    production_qty = PRODUCTION_QTY
    total_pieces_required = sum(demand.values())

    print(f"Total Leathers: {total_leathers}")
    print(f"Pattern Types per Bag: {pattern_types_per_bag}")
    print(f"Pattern Pieces per Bag: {pieces_per_bag}")
    print()
    print(f"Bag Quantity: {production_qty}")
    print(f"Total Pieces: {total_pieces_required}")

    # Present results in the exact same format as main_quality
    print('\n' + '=' * 70)
    print('SELECTION RESULT')
    print('=' * 70)

    print(f"Objective Value : {result['objective_value']:.2f}\n")
    print(f"Selected Leathers: {len(selected)} / {len(leathers)}\n")

    for idx, leather in enumerate(selected, start=1):
        leather_rows = assignment_df[assignment_df['Leather'] == leather]
        used_area = 0
        for _, row in leather_rows.iterrows():
            piece = row['Piece']
            qty = row['Qty']
            used_area += data[piece]['area'] * qty

        total_capacity = sum(leathers[leather][:5])
        utilization = (used_area / total_capacity * 100) if total_capacity > 0 else 0

        print('\n' + '-' * 70)
        print(f"{idx}. Leather: {leather} | Score: {scores.get(leather,0):.4f} | Capacity(mm2): {total_capacity:,.2f} | Used(mm2): {used_area:,.2f} | Utilization: {utilization:.2f}%")
        print('Assigned Pieces:')

        if leather_rows.empty:
            print('  (no assigned pieces)')
        else:
            for _, row in leather_rows.sort_values('Piece').iterrows():
                piece = row['Piece']
                qty = row['Qty']
                area_each = data[piece]['area']
                print(f"  - {piece:<40} Qty: {qty:>6,.2f}  Area(each mm2): {area_each:>12,.2f}  Total area: {area_each*qty:>12,.2f}")

    # Note: SNH nesting is NOT run automatically here. To run nesting for the first selected leather,
    # use: python3 src/run_nesting.py  (this script runs SNH for the first selected leather from the MIP result)

    # save unified selection JSON with scores and assignment
    # enrich assignment records with required quality information
    from data.preprocessing import assign_grade

    enriched_assign = []
    for row in assignment_df.to_dict('records'):
        piece = row.get('Piece')
        req_q = None
        # prefer explicit metadata if available
        meta = pieces.get(piece) if piece in pieces else pieces.get(piece.upper())
        if isinstance(meta, dict):
            req_q = meta.get('preferred_grade') or meta.get('suggested_grade')
        if req_q is None:
            try:
                req_q = assign_grade(piece)
            except Exception:
                req_q = None
        new_row = dict(row)
        new_row['RequiredQuality'] = req_q
        enriched_assign.append(new_row)

    out = {
        'selected': selected,
        'assignment': enriched_assign,
        'objective_value': result.get('objective_value'),
        'scores': scores
    }
    os.makedirs('outputs', exist_ok=True)
    with open(os.path.join('outputs','selection.json'), 'w') as f:
        json.dump(out, f, indent=2)

    print('\nSaved summary to outputs/selection.json')


if __name__ == '__main__':
    main()
