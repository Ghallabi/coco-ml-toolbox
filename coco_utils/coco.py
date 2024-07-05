import json
from utils import get_max_id_from_seq
import logging
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
import random


class Image(BaseModel):
    id: int = Field(default=0)
    file_name: str
    width: int
    height: int


class Annotation(BaseModel):
    id: int = Field(default=0)
    image_id: int
    category_id: int
    bbox: List[float]


class Category(BaseModel):
    id: int = Field(default=0)
    name: str


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
        self._update_attributes()

    def add_image_to_coco(self, elem: Image) -> int:

        image_id = self._check_if_categ_exists(elem.file_name)

        if image_id:
            return image_id

        new_id = self.max_image_id + 1
        elem.id = new_id
        self.max_image_id = new_id
        self.images.append(elem)
        self.image_names_to_ids = {elem.file_name: elem.id for elem in self.images}
        self.image_ids_to_names = {elem.id: elem.file_name for elem in self.images}
        return new_id

    def add_ann_to_coco(self, elem: Annotation) -> int:
        new_id = self.max_ann_id
        elem.id = new_id
        self.annotations.append(elem)
        self.max_ann_id = new_id
        return new_id

    def add_cat_to_coco(self, elem: Category) -> int:
        category_id = self._check_if_categ_exists(elem.name)

        if category_id:
            return category_id

        new_id = self.max_cat_id
        elem.id = new_id
        self.categories.append(elem)
        self.cat_names_to_ids = {elem.name: elem.id for elem in self.categories}
        self.max_cat_id = new_id
        return new_id

    def get_annotation_by_image_id(self, image_id: int) -> List[Dict]:
        return [ann for ann in self.annotations if ann.image_id == image_id]

    def get_annotations_by_image_name(self, image_name: str) -> List[Dict]:
        annotations = []
        try:
            image_id = self.image_names_to_ids[image_name]
            annotations = self.get_annotation_by_image_id(image_id)
        except KeyError:
            logging.error(f"Cannot find image by name {image_name}")

        return annotations

    def _update_attributes(self) -> None:
        self.max_image_id = get_max_id_from_seq([elem.dict() for elem in self.images])
        self.max_ann_id = get_max_id_from_seq(
            [elem.dict() for elem in self.annotations]
        )
        self.max_cat_id = get_max_id_from_seq([elem.dict() for elem in self.categories])

        self.image_names_to_ids = {elem.file_name: elem.id for elem in self.images}

        self.image_ids_to_names = {elem.id: elem.file_name for elem in self.images}

        self.cat_names_to_ids = {elem.name: elem.id for elem in self.categories}

    def _check_if_image_exists(self, image_name: str) -> Optional[int]:
        return self.image_names_to_ids.get(image_name)

    def _check_if_categ_exists(self, category_name: str) -> Optional[int]:
        return self.cat_names_to_ids.get(category_name)

    def get_coco_dict(self) -> Dict:
        return {
            "images": [elem.dict() for elem in self.images],
            "annotation": [elem.dict() for elem in self.annotations],
            "categories": [elem.dict() for elem in self.categories],
        }

    def save_coco_dict(self, file_path: str):
        coco_dict = self.get_coco_dict()
        with open(file_path, "w") as f:
            json.dump(coco_dict, f, indent=4)

    def extend(self, coco: "COCO"):
        for image in coco.images:
            self.add_image_to_coco(image)

        for ann in coco.annotations:
            self.add_ann_to_coco(ann)

        for cat in coco.categories:
            self.add_cat_to_coco(cat)

    def split(self, ratio: float = 0.2, mode="random") -> Tuple["COCO", "COCO"]:
        if mode == "random":  # split at image level
            # Split the images
            random.shuffle(self.images)
            split_index = int(len(self.images) * ratio)
            images_A, images_B = self.images[:split_index], self.images[split_index:]
            images_A_ids = {elem["id"] for elem in images_A}
            images_B_ids = {elem["id"] for elem in images_B}

            # Separate annotations based on image ids
            annotations_A, annotations_B = [], []
            for elem in self.annotations:
                if elem.image_id in images_A_ids:
                    annotations_A.append(elem)
                elif elem.image_id in images_B_ids:
                    annotations_B.append(elem)
            return (
                COCO(
                    images=images_A,
                    annotations=annotations_A,
                    categories=self.categories,
                ),
                COCO(
                    images=images_B,
                    annotations=annotations_B,
                    categories=self.categories,
                ),
            )

        elif mode == "stratified":
            # Use categories to split per class
            # Assumption is
            return COCO(), COCO()

    @classmethod
    def from_json_file(cls, coco_file: str) -> "COCO":
        with open(coco_file, "r") as f:
            coco_data = json.load(f)
        images = [Image(**elem) for elem in coco_data.get("images", [])]
        annotations = [Annotation(**elem) for elem in coco_data.get("annotations", [])]
        categories = [Category(**elem) for elem in coco_data.get("categories", [])]
        return cls(images, annotations, categories)
