"""Run selection using quality-based MIP (original flow).
Usage: python3 src/main_quality.py
"""
import os
import json
import time
import csv

from data.preprocessing import (
    leathers,
    pieces,
    demand,
    merge_data,
    calculate_leather_score,
    PRODUCTION_QTY
)

import pandas as pd

from selection.selection import solve_selection_mip


def main():
    t_start = time.perf_counter()

    # Data summary (inline, previously in utils.summary)
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
        # fallback: just print counts
        print('Total leathers:', len(leathers))

    # Build data
    t0 = time.perf_counter()
    data = merge_data(pieces, demand)
    t_build = time.perf_counter() - t0
    print(f"\n[TIMER] build_data: {t_build:.3f}s")

    # Solve selection MIP
    t0 = time.perf_counter()
    try:
        result = solve_selection_mip(data, leathers)
    except Exception as e:
        print('\n❌ Solver error or ortools not available:', e)
        result = None
    t_solve = time.perf_counter() - t0
    print(f"[TIMER] solve_selection_mip: {t_solve:.3f}s")
    print(f"[TIMER] elapsed_until_output: {time.perf_counter() - t_start:.3f}s")

    # Problem summary
    print('\n' + '=' * 70)
    print('PROBLEM SUMMARY (Quality score)')
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

    # Present results
    if result is None:
        print('\n❌ No Feasible Solution Found or solver failed')
        print(f"[TIMER] total_runtime: {time.perf_counter() - t_start:.3f}s")
        return

    scores = calculate_leather_score(leathers)

    # Optional: map hide_code (e.g., L0696156) -> 3-digit prefix (e.g., 001)
    # for display/final output only (solver internals remain unchanged).
    def _load_leather_prefix_map(manifest_path: str = os.path.join("dataset", "leather_manifest.csv")) -> dict:
        m: dict[str, str] = {}
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    hide_code = (row.get("hide_code") or "").strip().upper()
                    clean_folder = (row.get("clean_folder") or "").strip()
                    if not hide_code or not clean_folder:
                        continue
                    prefix = clean_folder.split("_", 1)[0]
                    if len(prefix) == 3 and prefix.isdigit():
                        m[hide_code] = prefix
        except Exception:
            # If manifest is missing, just display original leather codes.
            return {}
        return m

    leather_prefix = _load_leather_prefix_map()

    def _disp_leather(leather_code: str) -> str:
        return leather_prefix.get(str(leather_code).strip().upper(), str(leather_code))

    print('\n' + '=' * 70)
    print('SELECTION RESULT')
    print('=' * 70)

    print(f"Objective Value : {result['objective_value']:.2f}\n")

    selected = result['selected_leathers']
    assignment_df = result['assignment_df']

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
        leather_label = _disp_leather(leather)
        print(f"{idx}. Leather: {leather_label} | Score: {scores[leather]:,.2f} | Capacity(mm2): {total_capacity:,.2f} | Used(mm2): {used_area:,.2f} | Utilization: {utilization:.2f}%")
        print('Assigned Pieces:')

        if leather_rows.empty:
            print('  (no assigned pieces)')
        else:
            for _, row in leather_rows.sort_values('Piece').iterrows():
                piece = row['Piece']
                qty = row['Qty']
                area_each = data[piece]['area']
                print(f"  - {piece:<40} Qty: {qty:>6,.2f}  Area(each mm2): {area_each:>12,.2f}  Total area: {area_each*qty:>12,.2f}")

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

        # Display-only leather prefix (keep original code too)
        lcode = str(new_row.get('Leather') or '').strip().upper()
        new_row['LeatherCode'] = lcode
        new_row['Leather'] = _disp_leather(lcode)

        enriched_assign.append(new_row)

    selected_prefix = [_disp_leather(x) for x in selected]
    scores_prefix = {_disp_leather(k): v for k, v in scores.items()}

    out = {
        'selected': selected_prefix,
        'selected_codes': selected,
        'assignment': enriched_assign,
        'objective_value': result.get('objective_value'),
        'scores': scores_prefix,
        'scores_codes': scores,
    }
    os.makedirs('outputs', exist_ok=True)
    with open(os.path.join('outputs','selection.json'), 'w') as f:
        json.dump(out, f, indent=2)

    print('\nSaved summary to outputs/selection.json')
    print(f"[TIMER] total_runtime: {time.perf_counter() - t_start:.3f}s")


if __name__ == '__main__':
    main()
