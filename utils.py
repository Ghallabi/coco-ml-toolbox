import json
from pathlib import Path


def load_json_file(json_path: Path):
    with open(json_path, "r") as f:
        json_data = json.load(f)
    return json_data


def get_max_id_from_seq(seq: list[dict]) -> int:
    if len(seq) == 0:
        return 0
    return max([elem["id"] for elem in seq])
