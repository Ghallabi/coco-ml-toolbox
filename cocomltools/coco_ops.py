from cocomltools.models.coco import COCO
from cocomltools.models.base import Annotation
from cocomltools.utils import random_split, mlt_stratified_split
from typing import List
from collections import defaultdict
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import time


class CocoOps:
    def __init__(self, coco: COCO):
        self.coco = coco

    def split(self, ratio: float = 0.2, mode: str = "random"):
        if ratio > 0:
            return self._split(ratio=ratio, mode=mode)
        else:
            return (self.coco, COCO())

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
        image = Image.open(file_image).convert("RGB")
        for ann in annotations:
            self._crop_and_save_one_ann(image, ann, output_dir)

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
        print(f"Completed cropping dataset in {elapsed_time:.2f} seconds")
