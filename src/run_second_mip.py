"""
2차 MIP: 1차 네스팅 후 남은 조각에 대한 추가 가죽 선택
실행: python src/run_second_mip.py
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from collections import defaultdict
from data.preprocessing import demand, pieces, leathers
from mip.solver import solve_selection_mip

# ─────────────────────────────────────────────────────────────────────
# [OPTION A] 남은 조각을 직접 지정 (rollout에서 자동 계산이 안 맞을 때)
# 형식: {"PIECE_NAME": 남은_수량, ...}  빈 dict {} 이면 자동 계산
# ─────────────────────────────────────────────────────────────────────
MANUAL_REMAINING: dict = {
    # "FLAP TOP":   3,
    # "GUSSET":     5,
}

# ─────────────────────────────────────────────────────────────────────
# [OPTION B] 1차에서 사용한 가죽 코드를 직접 지정 (자동 추출이 안 맞을 때)
# 빈 set() 이면 greedy_solution.json에서 자동 추출
# ─────────────────────────────────────────────────────────────────────
MANUAL_USED_LEATHERS: set = set()
# MANUAL_USED_LEATHERS = {"L0696156", "L0696166", ...}

# ─────────────────────────────────────────────────────────────────────
# 1. 1차에서 사용한 가죽 추출
# ─────────────────────────────────────────────────────────────────────
rollout_path = os.path.join("dataset", "rollout", "greedy_solution.json")
with open(rollout_path, encoding="utf-8") as f:
    rollout = json.load(f)

if MANUAL_USED_LEATHERS:
    used_leather_codes = MANUAL_USED_LEATHERS
else:
    used_leather_codes = set()
    for b in rollout["confirmed_boards"]:
        for part in b["board_id"].split("_"):
            if part.startswith("L") and len(part) >= 7:
                used_leather_codes.add(part)

print(f"1차 사용 가죽 ({len(used_leather_codes)}장): {sorted(used_leather_codes)}")

# ─────────────────────────────────────────────────────────────────────
# 2. 남은 가죽 풀
# ─────────────────────────────────────────────────────────────────────
remaining_leathers = {k: v for k, v in leathers.items() if k not in used_leather_codes}
print(f"남은 가죽 풀: {len(remaining_leathers)}장")

# ─────────────────────────────────────────────────────────────────────
# 3. 남은 demand 계산
# ─────────────────────────────────────────────────────────────────────
if MANUAL_REMAINING:
    remaining_demand = dict(MANUAL_REMAINING)
else:
    # uid(U00484) → piece_name(FLAP TOP) 역매핑
    uid_to_name = {v["unique"]: k for k, v in pieces.items() if "unique" in v}

    # rollout에서 배치된 수량 집계
    placed_count = defaultdict(int)
    for board in rollout["confirmed_boards"]:
        for p in board["placements"]:
            uid = p["piece_id"].split("__")[-1]
            name = uid_to_name.get(uid, uid)
            placed_count[name] += 1

    remaining_demand = {}
    for piece_name, total_qty in demand.items():
        placed = placed_count.get(piece_name, 0)
        left = int(total_qty) - placed
        if left > 0:
            remaining_demand[piece_name] = left

total_remaining = sum(remaining_demand.values())
print(f"남은 조각: {len(remaining_demand)}종, {total_remaining}개")

if total_remaining == 0:
    print("\n남은 조각 없음 — 1차 rollout에서 모두 배치됨.")
    print("남은 30조각 정보를 직접 입력하려면 위 MANUAL_REMAINING dict를 채우세요.")
    sys.exit(0)

for name, qty in remaining_demand.items():
    print(f"  {name}: {qty}개")

# ─────────────────────────────────────────────────────────────────────
# 4. MIP 입력 data 구성
# ─────────────────────────────────────────────────────────────────────
data = {}
for piece_name, qty in remaining_demand.items():
    if piece_name in pieces:
        data[piece_name] = {
            "area":   pieces[piece_name]["area_mm2"],
            "demand": qty,
        }
    else:
        print(f"  [경고] '{piece_name}' pieces에 없음 — 스킵")

if not data:
    print("유효한 조각 없음. 종료.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────
# 5. 2차 MIP 실행
# ─────────────────────────────────────────────────────────────────────
print("\n=== 2차 MIP 실행 ===")
result = solve_selection_mip(data, remaining_leathers)

if result is None:
    print("INFEASIBLE — 남은 가죽으로 해결 불가")
else:
    selected = result["selected_leathers"]
    print(f"\n선택 가죽 ({len(selected)}장): {selected}")
    print(f"Objective: {result['objective_value']:.2f}")

    os.makedirs("outputs", exist_ok=True)
    out = {
        "selected": selected,
        "objective": result["objective_value"],
        "remaining_demand": remaining_demand,
    }
    out_path = os.path.join("outputs", "second_mip_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"저장: {out_path}")
