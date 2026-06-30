import json
import re
from collections import defaultdict


def parse_nesting_result(json_path: str) -> dict:
    """
    네스팅 결과 JSON을 파싱해 각 가죽별 피스 현황 반환.

    Returns:
        {
            board_id (str): {
                set_num (int): {pattern_code (str): count (int)}
            }
        }
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for board in data["confirmed_boards"]:
        board_id = board["board_id"]
        set_pieces = defaultdict(lambda: defaultdict(int))

        for placement in board["placements"]:
            piece_id = placement["piece_id"]
            parts = piece_id.split("__")
            if len(parts) < 2:
                continue

            m = re.match(r"set(\d+)", parts[0])
            if not m:
                continue

            set_num = int(m.group(1))
            pattern_code = parts[1]  # __q0/__q1 suffix 무시
            set_pieces[set_num][pattern_code] += 1

        result[board_id] = {k: dict(v) for k, v in set_pieces.items()}

    return result
