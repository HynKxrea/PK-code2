import json
import os
import pandas as pd
from ortools.linear_solver import pywraplp

from data.preprocessing import (
    assign_grade,
    calculate_leather_score
)

from selection.scorer_factory import create_scorer


# =====================================
# Build Selection Model
# =====================================

def solve_selection_mip(data, leathers, leather_scores_override=None, score_direction_override=None):

    # -----------------------------
    # Index
    # -----------------------------

    I = list(data.keys())
    J = list(leathers.keys())

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
        scoring_config_path = os.path.join('data', 'scoring_config.json')
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

            y[i, j] = solver.NumVar(
                0,
                solver.infinity(),
                f"y[{i},{j}]"
            )

    # =====================================
    # Constraint 1
    # Quality Capacity with u_k multipliers (loaded from data/u_values.json or default 0.7)
    # =====================================

    # load u values from data/u_values.json if available
    u_values_path = os.path.join('data', 'u_values.json')
    try:
        with open(u_values_path, 'r') as f:
            u_data = json.load(f)
            u_list = u_data.get('u', [0.7]*5)
    except Exception:
        u_list = [0.7] * 5

    # ensure length 5
    if len(u_list) < 5:
        u_list = list(u_list) + [0.7] * (5 - len(u_list))

    for j in J:

        grades = leathers[j][:5]

        Q1, Q2, Q3, Q4, Q5 = grades

        # Q1
        solver.Add(

            solver.Sum(
                data[i]["area"] * y[i, j]
                for i in I
                if assign_grade(i) <= 1
            )

            <= Q1 * x[j] * u_list[0]
        )

        # Q2
        solver.Add(

            solver.Sum(
                data[i]["area"] * y[i, j]
                for i in I
                if assign_grade(i) <= 2
            )

            <= (Q1 + Q2) * x[j] * u_list[1]
        )

        # Q3
        solver.Add(

            solver.Sum(
                data[i]["area"] * y[i, j]
                for i in I
                if assign_grade(i) <= 3
            )

            <= (Q1 + Q2 + Q3) * x[j] * u_list[2]
        )

        # Q4
        solver.Add(

            solver.Sum(
                data[i]["area"] * y[i, j]
                for i in I
                if assign_grade(i) <= 4
            )

            <= (Q1 + Q2 + Q3 + Q4) * x[j] * u_list[3]
        )

        # Q5
        solver.Add(

            solver.Sum(
                data[i]["area"] * y[i, j]
                for i in I
                if assign_grade(i) <= 5
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
