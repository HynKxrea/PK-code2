import json
import os
import csv
import pandas as pd
from ortools.linear_solver import pywraplp

from data.preprocessing import (
    assign_grade,
    calculate_leather_score
)

from selection.scorer_factory import create_scorer

from bottom_up.bottom_up_heuristic import max_pieces_for_pattern_on_leather
from selection.ub_history import load_history_ub


# =====================================
# Build Selection Model
# =====================================


def _load_piece_required_quality_map(csv_path: str) -> dict:
    """Load required quality grade (Q1..Q5 -> 1..5) from piece_catalog_area_rule.csv.

    The catalog 'name' column includes pattern prefixes; we normalize to the
    suffix after '__' to match piece keys used in `data`.
    """

    mapping: dict[str, int] = {}

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_name = (row.get("name") or "").strip()
                if not raw_name:
                    continue

                # match loader normalization: keep suffix after '__'
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
        # If loading fails, return empty and fall back to name-based assign_grade.
        return {}

    return mapping


def solve_selection_mip(
    data,
    leathers,
    leather_scores_override=None,
    score_direction_override=None,
    ub_method: str = "bottom_left",
    ub_history_path: str | None = None,
):

    # -----------------------------
    # Index
    # -----------------------------

    I = list(data.keys())
    J = list(leathers.keys())

    # -----------------------------
    # Required quality per piece (from piece_catalog_area_rule.csv)
    # -----------------------------

    catalog_path = os.path.join("dataset", "pieces", "piece_catalog_area_rule.csv")
    required_quality_map = _load_piece_required_quality_map(catalog_path)

    if not required_quality_map:
        raise ValueError(
            f"Could not load required quality mapping from {catalog_path}. "
            "Please ensure piece_catalog_area_rule.csv exists and has required_quality_area_rule values like Q1..Q5."
        )

    required_quality = {}
    missing_pieces = []

    for i in I:
        key = str(i).upper()
        if key not in required_quality_map:
            missing_pieces.append(i)
        else:
            required_quality[i] = int(required_quality_map[key])

    if missing_pieces:
        preview = ", ".join([str(x) for x in missing_pieces[:20]])
        more = "" if len(missing_pieces) <= 20 else f" ... (+{len(missing_pieces) - 20} more)"
        raise ValueError(
            "Missing required quality entries in piece_catalog_area_rule.csv for these pieces: "
            f"{preview}{more}"
        )

    # -----------------------------
    # UB strategy (y[i,j] upper bounds)
    # -----------------------------

    ub_method = str(ub_method or "bottom_left").strip().lower()
    if ub_method not in {"bottom_left", "history"}:
        raise ValueError(f"Unknown ub_method: {ub_method} (expected: bottom_left|history)")

    history_ub = None
    if ub_method == "history":
        if not ub_history_path:
            # default path (can be overridden by caller)
            ub_history_path = os.path.join("dataset", "ub_history.csv")
        history_ub = load_history_ub(ub_history_path)

    # -----------------------------
    # Leather Scores (via scorer) or override
    # -----------------------------

    leather_scores = {}

    if leather_scores_override is not None:
        # use provided override for scores (e.g., geometry-based compatibility)
        print('\nUsing leather_scores override (geometry-based).')

        # For override, we want high geometry-compatibility -> low cost (so minimization picks them).
        # We'll force minimization when override is used.
        score_direction = 'minimize'

        # Compute baseline quality score range to scale geometry-based compat values into same scale
        quality_scores = calculate_leather_score(leathers)
        q_vals = list(quality_scores.values())
        q_min = min(q_vals) if q_vals else 0.0
        q_max = max(q_vals) if q_vals else 1.0
        q_range = q_max - q_min if (q_max - q_min) != 0 else 1.0

        # leather_scores_override expected in [0,1] (compatibility). Map to cost so that higher compat => lower cost:
        # scaled = q_min + (1 - raw) * q_range
        for j in J:
            raw = float(leather_scores_override.get(j, 0.0))
            # clamp raw
            if raw < 0.0:
                raw = 0.0
            if raw > 1.0:
                raw = 1.0
            scaled = q_min + (1.0 - raw) * q_range
            leather_scores[j] = float(scaled)

    else:
        scoring_config_path = os.path.join('dataset', 'scoring_config.json')
        scorer, score_direction = create_scorer(scoring_config_path)

        for j in J:
            # leathers[j] expected to be sequence where first 5 entries are Q1..Q5 areas
            grades = leathers[j][:5]
            leather_scores[j] = scorer.score_leather(grades)

    # -----------------------------
    # Solver
    # -----------------------------

    solver = pywraplp.Solver.CreateSolver("SCIP")

    # -----------------------------
    # Variables
    # -----------------------------

    # leather selection
    x = {
        j: solver.IntVar(0, 1, f"x[{j}]")
        for j in J
    }

    # piece assignment (continuous)
    y = {}

    for i in I:
        for j in J:

            # y[i,j] is also bounded by demand via linking constraint; keep the tighter bound here.
            demand_i = float(data[i]["demand"])

            # Upper bound strategy for y[i,j]
            if ub_method == "bottom_left":
                # Upper bound by bottom-up heuristic (no rotation, bottom-left sequential placement)
                # This is computed on-the-fly without preprocessing.
                try:
                    ub_ij = int(
                        max_pieces_for_pattern_on_leather(
                            piece_geom_attrs=data[i].get("geom_attrs"),
                            leather_code=j,
                            required_grade=required_quality[i],
                            dataset_leathers_dir=os.path.join("dataset", "leathers"),
                        )
                    )
                except Exception:
                    # If we cannot compute, fall back to demand upper bound (do not over-restrict).
                    ub_ij = int(demand_i)

                # If heuristic returns 0 due to missing geometry, also fall back to demand.
                if ub_ij <= 0:
                    ub_ij = int(demand_i)

            elif ub_method == "history":
                # Upper bound from past observed usage (or any precomputed UB).
                # If missing OR if the UB is 0, fall back to demand to avoid over-restricting.
                assert history_ub is not None
                ub_hist = history_ub.get(i, j)
                ub_ij = int(demand_i) if ub_hist is None else int(ub_hist)
                if ub_ij <= 0:
                    ub_ij = int(demand_i)

            ub = min(demand_i, float(ub_ij))

            y[i, j] = solver.NumVar(
                0,
                ub,
                f"y[{i},{j}]"
            )

    # =====================================
    # Constraint 1
    # Quality Capacity with u_k multipliers (loaded from dataset/u_values.json or default 0.7)
    # =====================================

    # load u values from dataset/u_values.json if available
    # If the file does not exist, create it with a safe default.
    u_values_path = os.path.join('dataset', 'u_values.json')
    default_u_list = [0.6] * 5

    if not os.path.exists(u_values_path):
        try:
            os.makedirs(os.path.dirname(u_values_path), exist_ok=True)
            with open(u_values_path, 'w', encoding='utf-8') as f:
                json.dump({'u': default_u_list}, f, indent=2)
        except Exception:
            # If we can't write the file (permissions, etc.), fall back to defaults silently.
            pass

    try:
        with open(u_values_path, 'r', encoding='utf-8') as f:
            u_data = json.load(f)

            # Prefer q1_use_all..q5_use_all if present (computed from utilization step)
            q_keys = ["q1_use_all", "q2_use_all", "q3_use_all", "q4_use_all", "q5_use_all"]
            if isinstance(u_data, dict) and all(k in u_data for k in q_keys):
                u_list = [float(u_data[k]) for k in q_keys]
            else:
                u_list = u_data.get('u', default_u_list)

            # Safety: ensure length 5
            if not isinstance(u_list, list) or len(u_list) < 5:
                u_list = default_u_list
            else:
                u_list = [float(x) for x in u_list[:5]]

    except Exception:
        print(f"Warning: Could not load u values from {u_values_path}. Using default values: {default_u_list}")
        u_list = default_u_list

    for j in J:

        grades = leathers[j][:5]

        Q1, Q2, Q3, Q4, Q5 = grades

        # Q1
        solver.Add(

            solver.Sum(
                data[i]["area"] * y[i, j]
                for i in I
                if required_quality[i] <= 1
            )

            <= Q1 * x[j] * u_list[0]
        )

        # Q2
        solver.Add(

            solver.Sum(
                data[i]["area"] * y[i, j]
                for i in I
                if required_quality[i] <= 2
            )

            <= (Q1 + Q2) * x[j] * u_list[1]
        )

        # Q3
        solver.Add(

            solver.Sum(
                data[i]["area"] * y[i, j]
                for i in I
                if required_quality[i] <= 3
            )

            <= (Q1 + Q2 + Q3) * x[j] * u_list[2]
        )

        # Q4
        solver.Add(

            solver.Sum(
                data[i]["area"] * y[i, j]
                for i in I
                if required_quality[i] <= 4
            )

            <= (Q1 + Q2 + Q3 + Q4) * x[j] * u_list[3]
        )

        # Q5
        solver.Add(

            solver.Sum(
                data[i]["area"] * y[i, j]
                for i in I
                if required_quality[i] <= 5
            )

            <= (Q1 + Q2 + Q3 + Q4 + Q5) * x[j] * u_list[4]
        )

    # =====================================
    # Constraint 2
    # Demand Satisfaction
    # =====================================

    for i in I:

        demand = data[i]["demand"]

        solver.Add(

            solver.Sum(
                y[i, j]
                for j in J
            )

            >= demand
        )

    # =====================================
    # Constraint 3
    # Linking
    # =====================================

    for i in I:
        for j in J:

            solver.Add(

                y[i, j]

                <= data[i]["demand"] * x[j]
            )

    # =====================================
    # Objective
    # =====================================

    objective = solver.Objective()

    for j in J:
        objective.SetCoefficient(x[j], leather_scores[j])

    # respect scoring direction: 'maximize' means higher score is better
    if score_direction is not None and str(score_direction).lower() == 'maximize':
        objective.SetMaximization()
    else:
        objective.SetMinimization()

    # =====================================
    # Solve
    # =====================================

    status = solver.Solve()

    # =====================================
    # Result
    # =====================================

    if status != pywraplp.Solver.OPTIMAL:

        return None

    selected_leathers = [

        j for j in J
        if x[j].solution_value() > 0.5
    ]

    rows = []

    for i in I:
        for j in J:

            qty = y[i, j].solution_value()

            if qty > 1e-6:

                rows.append({
                    "Piece": i,
                    "Leather": j,
                    "Qty": round(qty, 2)
                })

    assignment_df = pd.DataFrame(rows)

    result = {

        "selected_leathers": selected_leathers,

        "assignment_df": assignment_df,

        "objective_value": objective.Value()
    }

    return result
