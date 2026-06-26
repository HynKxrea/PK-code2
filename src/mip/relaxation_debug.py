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
    M_used: Dict[IJ, float]  # M_ij = min(demand_i, u_ij_used)
    u_used: Dict[IJ, float]  # u_ij used after all fallback/default logic
    u_raw: Dict[IJ, Optional[float]]  # raw u from source (e.g., CSV), before fallback; None if missing/failed
    fallback_reason: Dict[IJ, Optional[str]]  # e.g., "missing", "<=0", "exception"

    # u data hygiene
    missing_u: Set[IJ]
    nonpositive_u: Set[IJ]

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
                    std.u_used.get(k, float("nan")),
                    two_std.u_used.get(k, float("nan")),
                    std.u_raw.get(k),
                    two_std.u_raw.get(k),
                    std.fallback_reason.get(k),
                    two_std.fallback_reason.get(k),
                )
            )

    print("\n[M_ij decreases (PROOF not a relaxation if any exist)]")
    print(f"Count where M_ij({two_std.case}) < M_ij({std.case}): {len(decreased)}")
    for (k, ms, m2, u_s, u_2, raw_s, raw_2, fb_s, fb_2) in decreased[:max_show]:
        print(
            f"{k}: M_std={ms}, M_2std={m2} | u_used_std={u_s}, u_used_2std={u_2} | "
            f"u_raw_std={raw_s}, u_raw_2std={raw_2} | fallback_std={fb_s}, fallback_2std={fb_2}"
        )

    # Missing / nonpositive stats
    print("\n[u data hygiene]")
    for a in (std, two_std):
        print(
            f"{a.case}: missing_u={len(a.missing_u)}, nonpositive_u={len(a.nonpositive_u)}"
        )

    # Solution info (if present)
    print("\n[Solution summary (if captured)]")
    print(f"{std.case}: obj={std.objective_value}, sum_x={std.sum_x}")
    print(f"{two_std.case}: obj={two_std.objective_value}, sum_x={two_std.sum_x}")
