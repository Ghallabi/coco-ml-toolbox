from cocomltools.models.coco import COCO
from cocomltools.models.base import Annotation
from cocomltools.utils import random_split, mlt_stratified_split
from cocomltools.logger import logger
from collections import defaultdict
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List
from cocomltools.utils import check_is_json, load_pil_image


class CocoOps:
    def __init__(self, coco: COCO):
        self.coco = coco

    def split(self, ratio: float = 0.2, mode: str = "random"):
        if ratio > 0:
            return self._split(ratio=ratio, mode=mode)
        else:
            return (self.coco, COCO())

    def filter(self, category_names: List[str] = None, image_names: List[str] = None):
        if category_names:
            for category_name in category_names:
                self.coco.remove_category_from_coco(category_name)
        if image_names:
            for image_name in image_names:
                self.coco.remove_image_from_coco(image_name)

        return COCO(
            images=self.coco.images,
            annotations=self.coco.annotations,
            categories=self.coco.categories,
        )

    def crop(
        self,
        images_dir: Path,
        output_dir: Path,
        max_workers: int = 1,
    ):
        if isinstance(images_dir, str):
            images_dir = Path(images_dir)

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            futures = [
                executor.submit(self._crop_one_image, elem, images_dir, output_dir)
                for elem in self.coco.images
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing annotation: {e}")
        elapsed_time = time.time() - start_time
        logger.info(f"Completed cropping dataset in {elapsed_time:.2f} seconds")

    def calculate_coco_stats(self) -> dict:

        stats = {}
        count_objs_per_image = defaultdict(int)
        count_objs_per_categ = defaultdict(int)
        scores_per_categ = defaultdict(float)

        ann_width_heights = []
        img_width_heights_dict = {}
        for elem in self.coco.images:
            img_width_heights_dict[elem.id] = [elem.width, elem.height]

        for elem in self.coco.annotations:
            count_objs_per_image[elem.image_id] += 1
            count_objs_per_categ[elem.category_id] += 1
            scores_per_categ[elem.category_id] += elem.score
            ann_width_heights.append(
                [
                    elem.bbox[2] / img_width_heights_dict[elem.image_id][0],
                    elem.bbox[3] / img_width_heights_dict[elem.image_id][1],
                    elem.category_id,
                ]
            )

        stats["avg_obj_per_image"] = (
            int(sum(count_objs_per_image.values()) / len(count_objs_per_image.values()))
            if len(count_objs_per_image.values()) > 0
            else 0
        )
        stats["min_obj_per_image"] = (
            min(count_objs_per_image.values()) if count_objs_per_image else 0
        )
        stats["max_obj_per_image"] = (
            max(count_objs_per_image.values()) if count_objs_per_image else 0
        )

        stats["class_scores"] = {
            cat_id: (
                scores_per_categ[cat_id] / count_objs_per_categ[cat_id]
                if count_objs_per_categ[cat_id] > 0
                else 0
            )
            for cat_id in scores_per_categ
        }
        stats["count_objs_per_categ"] = count_objs_per_categ
        stats["categories"] = [
            self.coco.cat_ids_to_names[cat_id] for cat_id in count_objs_per_categ.keys()
        ]
        stats["ann_width_heights"] = ann_width_heights
        stats["img_width_heights"] = img_width_heights_dict
        return stats

    def _split(self, ratio: float = 0.2, mode: str = "random"):
        if mode == "random":
            return self._random_split(ratio=ratio)
        elif mode == "strat":
            return self._stratified_split(ratio=ratio)
        else:
            raise NotImplementedError

    def _random_split(self, ratio: float = 0.2):
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

    def _stratified_split(self, ratio):
        images_to_categories = defaultdict(list)
        for ann in self.coco.annotations:
            images_to_categories[ann.image_id].append(ann.category_id)
        train_ids, test_ids = mlt_stratified_split(images_to_categories, ratio=ratio)
        annotations_A = []
        annotations_B = []
        for ann in self.coco.annotations:
            if ann.image_id in train_ids:
                annotations_A.append(ann)
            else:
                annotations_B.append(ann)
        images_A = [elem for elem in self.coco.images if elem.id in train_ids]
        images_B = [elem for elem in self.coco.images if elem.id in test_ids]
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

    def _crop_and_save_one_ann(
        self, image: Image.Image, ann: Annotation, output_dir: Path
    ):

        x1, y1, w, h = ann.bbox
        x2, y2 = x1 + w, y1 + h
        category_name = self.coco.cat_ids_to_names[ann.category_id]
        category_dir = Path(output_dir) / category_name
        category_dir.mkdir(exist_ok=True, parents=True)
        crop_out_file = category_dir / f"{ann.id}.jpg"
        crop = image.crop((x1, y1, x2, y2))
        crop.save(crop_out_file)

    def _crop_one_image(self, elem: Image, images_dir: Path, output_dir: Path):
        file_image = Path(images_dir) / elem.file_name
        annotations = self.coco.get_annotation_by_image_id(elem.id)
        if len(annotations) == 0:  # if no annotations, skip
            return
        image = load_pil_image(file_image)
        for ann in annotations:
            self._crop_and_save_one_ann(image, ann, output_dir)

    @staticmethod
    def merge(input_files: List[str]) -> COCO:
        if any(not check_is_json(file) for file in input_files):
            raise ValueError("One or more inputs have incorrect format")
        coco_base = COCO.from_json_file(input_files[0])
        for coco_file in input_files[1:]:
            coco = COCO.from_json_file(coco_file)
            coco_base.extend(coco)
        return coco_base

    @classmethod
    def from_dict(cls, coco_dict: dict) -> "CocoOps":
        coco = COCO.from_dict(coco_dict)
        return CocoOps(coco)

    @classmethod
    def from_json_file(cls, file: str) -> "CocoOps":
        coco = COCO.from_json_file(file)
        return CocoOps(coco)
