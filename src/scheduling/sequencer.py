import csv
import math
from collections import defaultdict

from ortools.linear_solver import pywraplp


def load_requirements(catalog_path: str) -> dict:
    """
    Returns:
        {pattern_code (str): quantity_per_bag (int)}
    """
    req = {}
    with open(catalog_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            piece_id = (row.get("piece_id") or "").strip()
            qty = int(row.get("quantity_per_bag") or 1)
            if piece_id:
                req[piece_id] = qty
    return req


def find_optimal_sequence(
    leather_pieces: dict,
    requirements: dict,
    total_sets: int = 20,
    lot_size: int = 2,
):
    """
    BIP로 최적 재단 순서 탐색.

    Returns:
        best_sequence: list of tuples (board_ids per batch)
        sum_cj: int
    """
    board_ids = list(leather_pieces.keys())
    N = len(board_ids)
    T = math.ceil(N / lot_size)
    S = list(range(1, total_sets + 1))
    P = list(requirements.keys())

    solver = pywraplp.Solver.CreateSolver("SCIP")

    # --------------------------------------------------
    # Variables
    # --------------------------------------------------

    # x[i, t]: 가죽 i를 배치 t에 배정
    x = {
        (i, t): solver.BoolVar(f"x[{i},{t}]")
        for i in range(N)
        for t in range(T)
    }

    # y[j, t]: 세트 j가 배치 t까지 완성됨
    y = {
        (j, t): solver.BoolVar(f"y[{j},{t}]")
        for j in S
        for t in range(T)
    }

    # --------------------------------------------------
    # Constraints
    # --------------------------------------------------

    # 각 가죽은 정확히 하나의 배치에 배정
    for i in range(N):
        solver.Add(solver.Sum(x[i, t] for t in range(T)) == 1)

    # 배치당 가죽 수 <= lot_size
    for t in range(T):
        solver.Add(solver.Sum(x[i, t] for i in range(N)) <= lot_size)

    # 완성 제약: 배치 t까지 누적 피스가 요구량 미달이면 y[j,t]=1 불가
    for j in S:
        for t in range(T):
            for p in P:
                rp = requirements[p]
                cumulative = solver.Sum(
                    leather_pieces[board_ids[i]].get(j, {}).get(p, 0)
                    * solver.Sum(x[i, tau] for tau in range(t + 1))
                    for i in range(N)
                )
                solver.Add(rp * y[j, t] <= cumulative)

    # --------------------------------------------------
    # Objective: max Σ y[j,t]  ≡  min Σ C_j
    # --------------------------------------------------

    objective = solver.Objective()
    for j in S:
        for t in range(T):
            objective.SetCoefficient(y[j, t], 1)
    objective.SetMaximization()

    # --------------------------------------------------
    # Solve
    # --------------------------------------------------

    status = solver.Solve()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None, None

    # --------------------------------------------------
    # 결과 복원
    # --------------------------------------------------

    batch_assignment = defaultdict(list)
    for i in range(N):
        for t in range(T):
            if x[i, t].solution_value() > 0.5:
                batch_assignment[t].append(board_ids[i])

    sequence = [tuple(batch_assignment[t]) for t in range(T)]

    # sum_cj 계산
    sum_cj = 0
    penalty = T + 1
    for j in S:
        cj = penalty
        for t in range(T):
            if y[j, t].solution_value() > 0.5:
                cj = t + 1
                break
        sum_cj += cj

    return sequence, sum_cj
