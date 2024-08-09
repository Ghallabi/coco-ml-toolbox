import json
import logging
from typing import List, Dict, Optional, Tuple
from cocomltools.utils import get_max_id_from_seq
from collections import defaultdict
from cocomltools.models.base import Image, Annotation, Category
from cocomltools.logger import logger


class COCO:
    def __init__(
        self,
        images: List[Image] = None,
        annotations: List[Annotation] = None,
        categories: List[Category] = None,
    ):

        self.images = images or []
        self.annotations = annotations or []
        self.categories = categories or []

        self.max_image_id = get_max_id_from_seq(
            [elem.model_dump() for elem in self.images]
        )
        self.max_ann_id = get_max_id_from_seq(
            [elem.model_dump() for elem in self.annotations]
        )
        self.max_cat_id = get_max_id_from_seq(
            [elem.model_dump() for elem in self.categories]
        )

        self.image_names_to_ids = {elem.file_name: elem.id for elem in self.images}

        self.image_ids_to_names = {elem.id: elem.file_name for elem in self.images}

        self.cat_names_to_ids = {elem.name: elem.id for elem in self.categories}

        self.cat_ids_to_names = {elem.id: elem.name for elem in self.categories}

        self.image_ids_to_anns = defaultdict(list)
        self.image_ids_to_ann_count = defaultdict(int)
        for ann in self.annotations:
            self.image_ids_to_anns[ann.image_id].append(ann)
            self.image_ids_to_ann_count[ann.image_id] += 1

    def remove_category_from_coco(self, category_name: str):
        if category_name not in self.cat_names_to_ids:
            logger.warning(f"No category found with {category_name} in coco - skipping")
            return
        categ_id = self.cat_names_to_ids[category_name]
        new_categories = [elem for elem in self.categories if elem.id != categ_id]
        new_annotations = []
        empty_images_names = set()
        for elem in self.annotations:
            if elem.category_id == categ_id:
                self.image_ids_to_ann_count[elem.image_id] -= 1
                if self.image_ids_to_ann_count[elem.image_id] == 0:
                    empty_images_names.add(self.image_ids_to_names[elem.image_id])
                continue
            new_annotations.append(elem)

        # Remove images with no annotations
        for image_name in empty_images_names:
            self.remove_image_from_coco(image_name)

        self.categories = new_categories
        self.annotations = new_annotations

    def add_image_to_coco(self, elem: Image) -> int:

        image_id = self._check_if_image_exists(elem.file_name)

        if image_id:
            return image_id

        new_id = self.max_image_id + 1
        elem.id = new_id
        self.max_image_id = new_id
        self.images.append(elem)
        self.image_names_to_ids = {elem.file_name: elem.id for elem in self.images}
        self.image_ids_to_names = {elem.id: elem.file_name for elem in self.images}
        return new_id

    def remove_image_from_coco(self, image_name: str):
        if image_name not in self.image_names_to_ids:
            logger.warning(f"No image found with {image_name} in coco - skipping")
            return
        image_id = self.image_names_to_ids[image_name]
        new_images = [elem for elem in self.images if elem.id != image_id]
        new_annotations = [
            elem for elem in self.annotations if elem.image_id != image_id
        ]
        self.images = new_images
        self.annotations = new_annotations

    def add_ann_to_coco(
        self, elem: Annotation, new_image_id: int, new_category_id: int
    ) -> int:
        new_id = self.max_ann_id + 1
        elem.id = new_id
        elem.image_id = new_image_id
        elem.category_id = new_category_id
        self.annotations.append(elem)
        self.image_ids_to_anns[elem.image_id].append(elem)
        self.max_ann_id = new_id
        return new_id

    def add_cat_to_coco(self, elem: Category) -> int:
        category_id = self._check_if_categ_exists(elem.name)

        if category_id:
            return category_id

        new_id = self.max_cat_id + 1
        elem.id = new_id
        self.categories.append(elem)
        self.cat_names_to_ids = {elem.name: elem.id for elem in self.categories}
        self.cat_ids_to_names = {elem.id: elem.name for elem in self.categories}
        self.max_cat_id = new_id
        return new_id

    def get_annotation_by_image_id(self, image_id: int) -> List[Dict]:
        return self.image_ids_to_anns[image_id]

    def get_annotations_by_image_name(self, image_name: str) -> List[Dict]:
        annotations = []
        try:
            image_id = self.image_names_to_ids[image_name]
            annotations = self.get_annotation_by_image_id(image_id)
        except KeyError:
            logging.error(f"Cannot find image by name {image_name}")

        return annotations

    def _check_if_image_exists(self, image_name: str) -> Optional[int]:
        return self.image_names_to_ids.get(image_name)

    def _check_if_categ_exists(self, category_name: str) -> Optional[int]:
        return self.cat_names_to_ids.get(category_name)

    def get_coco_dict(self) -> Dict:
        return {
            "images": [elem.model_dump() for elem in self.images],
            "annotations": [elem.model_dump() for elem in self.annotations],
            "categories": [elem.model_dump() for elem in self.categories],
        }

    def save_coco_dict(self, file_path: str):
        coco_dict = self.get_coco_dict()
        with open(file_path, "w") as f:
            json.dump(coco_dict, f, indent=4)

    def extend(self, coco: "COCO"):
        for image in coco.images:
            self.add_image_to_coco(image)

        for cat in coco.categories:
            self.add_cat_to_coco(cat)

        for ann in coco.annotations:
            new_image_id = self.image_names_to_ids[
                coco.image_ids_to_names[ann.image_id]
            ]
            new_category_id = self.cat_names_to_ids[
                coco.cat_ids_to_names[ann.category_id]
            ]
            self.add_ann_to_coco(ann, new_image_id, new_category_id)

    @classmethod
    def from_json_file(cls, json_file: str) -> "COCO":
        with open(json_file, "r") as f:
            coco_data = json.load(f)
        return cls._from_coco_data(coco_data)

    @classmethod
    def from_dict(cls, coco_data: Dict) -> "COCO":
        return cls._from_coco_data(coco_data)

    @classmethod
    def _from_coco_data(cls, coco_data: Dict) -> "COCO":
        images = [Image(**elem) for elem in coco_data.get("images", [])]
        annotations = [Annotation(**elem) for elem in coco_data.get("annotations", [])]
        categories = [Category(**elem) for elem in coco_data.get("categories", [])]
        return cls(images, annotations, categories)
