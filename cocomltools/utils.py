import json
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import List
import random
from cocomltools.models.base import Annotation
from collections import defaultdict


def check_is_json(file_path: str) -> bool:
    file_path = Path(file_path)
    return file_path.is_file() and file_path.suffix == ".json"


def load_json_file(json_path: Path):
    with open(json_path, "r") as f:
        json_data = json.load(f)
    return json_data


def get_max_id_from_seq(seq: list[dict]) -> int:
    if len(seq) == 0:
        return 0
    return max([elem["id"] for elem in seq])


def random_split(data: List, split_ratio: float = 0.2):
    random.shuffle(data)
    split_index = int(len(data) * split_ratio)
    set_A, set_B = data[split_index:], data[:split_index]
    return set_A, set_B


def stratified_split(data_dict, ratio=0.2):
    train_split = {}
    test_split = {}

    for key, annotations in data_dict.items():
        if len(annotations) > 1:
            # Splitting annotations with stratification
            annotations_train, annotations_test = train_test_split(
                annotations,
                test_size=ratio,
                stratify=[key] * len(annotations),
                random_state=42,
            )
            train_split[key] = annotations_train
            test_split[key] = annotations_test
        else:
            # For classes with only one annotation, add it to the training set
            train_split[key] = annotations
            test_split[key] = []

    return train_split, test_split


def create_split_priority_queue(
    annotations: List[Annotation], skip_categories: set[int] = None
) -> dict:
    categories_to_images_ids = defaultdict(list)
    for ann in annotations:
        categories_to_images_ids[ann.category_id].append(ann.image_id)

    categories_to_image_ids_stats = {}
    for category_id, image_list in categories_to_images_ids.items():
        image_counts = defaultdict(int)
        for image_name in image_list:
            image_counts[image_name] += 1

        stats_dict = {}
        stats_dict[category_id] = {}
        stats_dict[category_id]["images"] = dict(
            sorted(image_counts.items(), key=lambda x: x[1], reverse=True)
        )
        stats_dict[category_id]["total_images"] = len(image_counts)
        stats_dict[category_id]["total_instances"] = sum(
            [item for item in image_counts.values()]
        )
        categories_to_image_ids_stats[category_id] = stats_dict

    priority_queue = dict(
        sorted(
            categories_to_image_ids_stats.items(),
            key=lambda x: (x[1]["total_images"], x[1]["total_instances"]),
            reverse=True,
        )
    )

    return priority_queue
