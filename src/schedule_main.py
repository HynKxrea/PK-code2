"""재단 순서 최적화.

Usage: python3 src/schedule_main.py [nesting_result_file.json]
       (인자 없으면 dataset/nesting_result/ 내 모든 JSON 처리)
"""

# =============================================
# 파라미터 설정
# =============================================
SH_LOT_SIZE = 2   # 한 번에 재단하는 가죽 수
# =============================================
import os
import sys
import glob
import time

sys.path.insert(0, os.path.dirname(__file__))

from scheduling.parser import parse_nesting_result
from scheduling.sequencer import load_requirements, find_optimal_sequence


def short_id(board_id: str) -> str:
    return board_id.split("_")[0]


def main():
    catalog_path = os.path.join("dataset", "pieces", "piece_catalog_area_rule.csv")
    requirements = load_requirements(catalog_path)

    if len(sys.argv) > 1:
        json_files = [sys.argv[1]]
    else:
        nesting_dir = os.path.join("dataset", "nesting_result")
        json_files = sorted(glob.glob(os.path.join(nesting_dir, "*.json")))

    for json_path in json_files:
        print(f"\n{'='*60}")
        print(f"파일: {os.path.basename(json_path)}")
        print("=" * 60)

        leather_pieces = parse_nesting_result(json_path)
        board_ids = list(leather_pieces.keys())
        print(f"가죽 {len(board_ids)}개: {[short_id(b) for b in board_ids]}")

        t0 = time.perf_counter()
        sequence, sum_cj = find_optimal_sequence(leather_pieces, requirements, lot_size=SH_LOT_SIZE)
        elapsed = time.perf_counter() - t0

        print(f"\n최적 재단 순서 (ΣC_j = {sum_cj}, 계산 시간: {elapsed:.2f}s):")
        result_str = "".join(
            f"({', '.join(short_id(b) for b in pair)})" for pair in sequence
        )
        print(result_str)

        print("\n배치별 상세:")
        from collections import defaultdict
        cumulative = defaultdict(lambda: defaultdict(int))
        completed_total = set()
        for batch_num, pair in enumerate(sequence, start=1):
            for board_id in pair:
                for set_num, patterns in leather_pieces[board_id].items():
                    for pattern, cnt in patterns.items():
                        cumulative[set_num][pattern] += cnt

            newly = set()
            for s in range(1, 21):
                if s not in completed_total:
                    if all(cumulative[s].get(p, 0) >= q for p, q in requirements.items()):
                        newly.add(s)
            completed_total |= newly

            pair_str = f"({', '.join(short_id(b) for b in pair)})"
            print(f"  배치 {batch_num} {pair_str}: "
                  f"+{len(newly)}개 완성 → 누적 {len(completed_total)}/20")


if __name__ == "__main__":
    main()
