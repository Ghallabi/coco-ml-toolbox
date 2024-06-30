from coco_utils.coco import COCO


class CocoOps:
    def __init__(self):
        pass

    def split(
        self, coco, split_ratio: float = 0.2, stratified: bool = False
    ) -> tuple[COCO, COCO]:
        coco_1 = COCO()
        coco_2 = COCO()
        return coco_1, coco_2

    def merge(self, coco_1: COCO, coco_2: COCO) -> dict:
        coco_merged = COCO()
        return coco_merged


if __name__ == "__main__":
    coco_ops = CocoOps()
