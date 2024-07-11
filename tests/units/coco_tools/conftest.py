import pytest
import json


@pytest.fixture
def coco_merge_input_1():
    coco_1_file = "tests/mock/coco_merge_input_1.json"
    coco_json = {}
    with open(coco_1_file, "r") as f:
        coco_json = json.load(f)
    return coco_json

@pytest.fixture
def coco_merge_input_2():
    coco_2_file =  "tests/mock/coco_merge_input_2.json"
    coco_json = {}
    with open(coco_2_file, "r") as f:
        coco_json = json.load(f)
    return coco_json



@pytest.fixture
def coco_merge_input_3():
    coco_3_file =  "tests/mock/coco_merge_input_3.json"
    coco_json = {}
    with open(coco_3_file, "r") as f:
        coco_json = json.load(f)
    return coco_json


@pytest.fixture
def coco_split_random_input():
    coco_4_file =  "tests/mock/coco_split_random.json"
    coco_json = {}
    with open(coco_4_file, "r") as f:
        coco_json = json.load(f)
    return coco_json


@pytest.fixture
def info_merged_output_basic():
    return {
        "image_ids": set([1, 2, 3, 4]),
        "image_names": set(["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]),
        "category_names": set(["person", "bicycle", "car", "motorcycle", "airplane", "truck", "boat", "traffic light"]),
        "category_ids": set([1, 2, 3, 4, 5, 6, 7, 8])
    }


@pytest.fixture
def info_merged_output_duplicate():
    return {"image_ids": set([1, 2, 3]),
            "image_names": set(["image3.jpg", "image4.jpg", "image5.jpg"]),
            "category_names": set(['bicycle', 'boat', 'bus', 'car', 'stop sign', 'traffic light', 'truck']),
            "category_ids": set([1, 2, 3, 4, 5, 6, 7])}