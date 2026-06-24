from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

IJ = Tuple[Any, Any]


@dataclass
class SelectionAudit:
    """Captured artifacts to prove whether one case is a relaxation of another."""

    case: str

    # Structure
    num_variables: int
    num_constraints: int
    len_x: int
    len_y: int
    y_keys: Set[IJ]

    # Bounds actually used
    M_used: Dict[IJ, float]  # M_ij = min(demand_i, ub_ij_used)
    ub_used: Dict[IJ, float]  # ub_ij used after all fallback/default logic
    ub_raw: Dict[IJ, Optional[float]]  # raw ub from source (e.g., CSV), before fallback; None if missing/failed
    fallback_reason: Dict[IJ, Optional[str]]  # e.g., "missing", "<=0", "exception"

    # UB data hygiene
    missing_ub: Set[IJ]
    nonpositive_ub: Set[IJ]

    # Solution (optional)
    x_solution: Optional[Dict[Any, int]] = None
    objective_value: Optional[float] = None
    sum_x: Optional[float] = None


def compare_audits(std: SelectionAudit, two_std: SelectionAudit, *, max_show: int = 200) -> None:
    print(f"\n=== Relaxation check: '{two_std.case}' should relax '{std.case}' ===")

    print("\n[Structure counts]")
    print(f"{std.case}:  NumVariables={std.num_variables}, NumConstraints={std.num_constraints}, len(x)={std.len_x}, len(y)={std.len_y}")
    print(f"{two_std.case}: NumVariables={two_std.num_variables}, NumConstraints={two_std.num_constraints}, len(x)={two_std.len_x}, len(y)={two_std.len_y}")

    only_in_std = std.y_keys - two_std.y_keys
    only_in_two = two_std.y_keys - std.y_keys
    print("\n[y_ij key differences]")
    print(f"y keys only in {std.case}: {len(only_in_std)}")
    if only_in_std:
        print(" examples:", list(only_in_std)[:max_show])
    print(f"y keys only in {two_std.case}: {len(only_in_two)}")
    if only_in_two:
        print(" examples:", list(only_in_two)[:max_show])

    # Compare M decreases
    common = set(std.M_used.keys()) & set(two_std.M_used.keys())
    decreased: List[Tuple[IJ, float, float, float, float, Optional[float], Optional[float], Optional[str], Optional[str]]] = []
    for k in common:
        ms = std.M_used[k]
        m2 = two_std.M_used[k]
        if m2 + 1e-9 < ms:
            decreased.append(
                (
                    k,
                    ms,
                    m2,
                    std.ub_used.get(k, float("nan")),
                    two_std.ub_used.get(k, float("nan")),
                    std.ub_raw.get(k),
                    two_std.ub_raw.get(k),
                    std.fallback_reason.get(k),
                    two_std.fallback_reason.get(k),
                )
            )

    print("\n[M_ij decreases (PROOF not a relaxation if any exist)]")
    print(f"Count where M_ij({two_std.case}) < M_ij({std.case}): {len(decreased)}")
    for (k, ms, m2, ub_s, ub_2, raw_s, raw_2, fb_s, fb_2) in decreased[:max_show]:
        print(
            f"{k}: M_std={ms}, M_2std={m2} | ub_used_std={ub_s}, ub_used_2std={ub_2} | "
            f"ub_raw_std={raw_s}, ub_raw_2std={raw_2} | fallback_std={fb_s}, fallback_2std={fb_2}"
        )

    # Missing / nonpositive stats
    print("\n[UB data hygiene]")
    for a in (std, two_std):
        print(
            f"{a.case}: missing_ub={len(a.missing_ub)}, nonpositive_ub={len(a.nonpositive_ub)}"
        )

    # Solution info (if present)
    print("\n[Solution summary (if captured)]")
    print(f"{std.case}: obj={std.objective_value}, sum_x={std.sum_x}")
    print(f"{two_std.case}: obj={two_std.objective_value}, sum_x={two_std.sum_x}")
