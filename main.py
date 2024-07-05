from coco_utils.coco import COCO
import argparse


if __name__ == "__main__":
    coco_1 = COCO.from_json_file("./data/coco_train.json")
    coco_2 = COCO.from_json_file("./data/coco_val.json")
