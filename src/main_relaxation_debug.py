"""Debug whether one UB CSV case is truly a relaxation of another.

Usage (example):
  python3 src/main_relaxation_debug.py \
    --ub-std dataset/ub_mean_plus_std.csv \
    --ub-2std dataset/ub_mean_plus_2std.csv

This script:
  1) Solves both cases with solve_selection_mip(debug_audit=True)
  2) Compares model structure (var/constraint counts, y key set)
  3) Compares actual M_ij = min(demand_i, ub_used)
  4) Runs a feasibility check by fixing x from std in 2std

IMPORTANT:
- Your current selection MIP objective in src/selection/selection.py is NOT "min sum(x_j)".
  It uses leather_scores as coefficients.
  The relaxation check below is about feasible region / M_ij monotonicity.
"""

from __future__ import annotations

import argparse

from data.preprocessing import leathers, pieces, demand, merge_data
from selection.selection import solve_selection_mip
from selection.relaxation_debug import compare_audits


def main() -> int:
    parser = argparse.ArgumentParser(description="Relaxation debug for UB history CSV cases")
    parser.add_argument("--ub-std", required=True, help="CSV path for std case")
    parser.add_argument("--ub-2std", required=True, help="CSV path for 2std case")
    parser.add_argument("--solver", default="SCIP", help="OR-Tools backend name (default: SCIP)")
    args = parser.parse_args()

    data = merge_data(pieces, demand)

    print("\n--- Solving std case ---")
    res_std = solve_selection_mip(
        data,
        leathers,
        ub_method="history",
        ub_history_path=args.ub_std,
        debug_audit=True,
        debug_case_label="std",
    )
    if res_std is None:
        raise RuntimeError("std case returned no solution")

    print("\n--- Solving 2std case ---")
    res_2std = solve_selection_mip(
        data,
        leathers,
        ub_method="history",
        ub_history_path=args.ub_2std,
        debug_audit=True,
        debug_case_label="2std",
    )
    if res_2std is None:
        raise RuntimeError("2std case returned no solution")

    audit_std = res_std.get("audit")
    audit_2std = res_2std.get("audit")
    if audit_std is None or audit_2std is None:
        raise RuntimeError("Missing audit objects; ensure debug_audit=True")

    compare_audits(audit_std, audit_2std)

    print("\n--- Feasibility check: fix x from std in 2std ---")
    x_fix = audit_std.x_solution
    if x_fix is None:
        raise RuntimeError("std audit did not capture x_solution")

    res_fix = solve_selection_mip(
        data,
        leathers,
        ub_method="history",
        ub_history_path=args.ub_2std,
        fix_x=x_fix,
        feasibility_only=True,
        debug_audit=True,
        debug_case_label="2std_fixed_x",
    )

    if res_fix is None:
        print("\n[RESULT] INFEASIBLE when fixing x from std in 2std => 2std is NOT a relaxation of std.")
    else:
        print("\n[RESULT] FEASIBLE when fixing x from std in 2std => 2std must have optimum <= std (if objective identical).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
