import json
from pathlib import Path
from typing import List, Dict


def load_json_file(json_path: Path):
    with open(json_path, "r") as f:
        json_data = json.load(f)
    return json_data


def get_max_id_from_seq(seq: List[Dict]) -> int:
    if not seq:
        return 0
    return max(item.get("id", 0) for item in seq)
