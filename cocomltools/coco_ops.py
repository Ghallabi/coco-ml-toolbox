from cocomltools.models.coco import COCO
from cocomltools.utils import (
    random_split,
    stratified_split,
    create_split_priority_queue,
)
from typing import List


class CocoOps:
    def __init__(self, coco: COCO, skip_categories: List[str] = None):
        self.coco = coco
        if skip_categories:
            self.skip_category_ids = set(
                self.coco.cat_names_to_ids[category_name]
                for category_name in skip_categories
            )

    def split(self, ratio: float = 0.2, mode="random"):
        if ratio > 0:
            return self._split(ratio=ratio, mode=mode)
        else:
            return (self.coco, COCO())

    def _split_random(self, ratio: float = 0.2):
        images_A, images_B = random_split(self.coco.images, split_ratio=ratio)
        images_A_ids = {elem.id for elem in images_A}
        images_B_ids = {elem.id for elem in images_B}

        # Separate annotations based on image ids
        annotations_A, annotations_B = [], []
        for elem in self.coco.annotations:
            if elem.image_id in images_A_ids:
                annotations_A.append(elem)
            elif elem.image_id in images_B_ids:
                annotations_B.append(elem)
        return (
            COCO(
                images=images_A,
                annotations=annotations_A,
                categories=self.coco.categories,
            ),
            COCO(
                images=images_B,
                annotations=annotations_B,
                categories=self.coco.categories,
            ),
        )

    def _split_strat_single_obj(self, ratio: float = 0.2):
        categ_to_ann_dict = {elem.id: [] for elem in self.coco.categories}
        for ann in self.coco.annotations:
            categ_to_ann_dict[ann.category_id].append(ann)

        ann_A_dict, ann_B_dict = stratified_split(categ_to_ann_dict, ratio=ratio)
        annotations_A, annotations_B = [], []
        image_ids_A, image_ids_B = set(), set()
        for ann_list in ann_A_dict.values():
            annotations_A.extend(ann_list)
            image_ids_A.update(ann.image_id for ann in ann_list)

        for ann_list in ann_B_dict.values():
            annotations_B.extend(ann_list)
            image_ids_B.update(ann.image_id for ann in ann_list)
        images_A = [elem for elem in self.coco.images if elem.id in image_ids_A]
        images_B = [elem for elem in self.coco.images if elem.id in image_ids_B]
        return COCO(
            images=images_A,
            annotations=annotations_A,
            categories=self.coco.categories,
        ), COCO(
            images=images_B,
            annotations=annotations_B,
            categories=self.coco.categories,
        )

    def _update_priority_queue(self, image_id: int):
        """
        Update the priority queue by deleting already selected
        images from other items in the queue and remove all classes
        that have total_images == 0
        """
        to_remove = []
        for other_class, other_stats in self.priority_queue.items():
            if image_id in other_stats["images"]:
                del other_stats["images"][image_id]
                other_stats["total_images"] -= 1
                if other_stats["total_images"] == 0:
                    to_remove.append(other_class)
        for class_to_remove in to_remove:
            del self.priority_queue[class_to_remove]

    def _split_strat_multi_obj(self, ratio: float = 0.2):
        self.priority_queue = create_split_priority_queue(
            self.coco.annotations, self.skip_category_ids
        )

        return COCO(), COCO()

    def _split(self, ratio: float = 0.2, mode="random"):
        if mode == "random":  # split at image level
            self._split_random(ratio=ratio)

        elif mode == "strat_single_obj":  # one object per image.
            self._split_strat_single_obj(ratio=ratio)
        elif mode == "strat_multi_obj":  # multiple objects per image
            return COCO(), COCO()
