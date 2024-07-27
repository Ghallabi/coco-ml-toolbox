import json
from models.utils import get_max_id_from_seq
import logging
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from models.utils import random_split, stratified_split
from pathlib import Path
from PIL import Image as PImage


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
    segmentation: List[float] | List[List[float]] = Field(default=[])
    area: int
    iscrowd: int = Field(default=0)


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

    def add_ann_to_coco(
        self, elem: Annotation, new_image_id: int, new_category_id: int
    ) -> int:
        new_id = self.max_ann_id + 1
        elem.id = new_id
        elem.image_id = new_image_id
        elem.category_id = new_category_id
        self.annotations.append(elem)
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

    def split(self, ratio: float = 0.2, mode="random") -> Tuple["COCO", "COCO"]:
        if mode == "random":  # split at image level
            images_A, images_B = random_split(self.images, split_ratio=ratio)
            images_A_ids = {elem.id for elem in images_A}
            images_B_ids = {elem.id for elem in images_B}

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

        elif mode == "strat_single_obj":  # one object per image.
            categ_to_ann_dict = {elem.id: [] for elem in self.categories}
            for ann in self.annotations:
                categ_to_ann_dict[ann.category_id].append(ann)

            ann_A_dict, ann_B_dict = stratified_split(categ_to_ann_dict)
            annotations_A, annotations_B = [], []
            image_ids_A, image_ids_B = set(), set()
            for ann_list in ann_A_dict.values():
                annotations_A.extend(ann_list)
                image_ids_A.update(ann.image_id for ann in ann_list)

            for ann_list in ann_B_dict.values():
                annotations_B.extend(ann_list)
                image_ids_B.update(ann.image_id for ann in ann_list)
            images_A = [elem for elem in self.images if elem.id in image_ids_A]
            images_B = [elem for elem in self.images if elem.id in image_ids_B]
            return COCO(
                images=images_A, annotations=annotations_A, categories=self.categories
            ), COCO(
                images=images_B, annotations=annotations_B, categories=self.categories
            )
        elif mode == "strat_multi_obj":  # multiple objects per image
            return COCO(), COCO()

    def crop(self, images_dir: str, output_dir: str, categories: List[str] = None):
        category_names_set = set(categories) if categories else None
        for elem in self.images:
            file_image = Path(images_dir) / elem.file_name
            image = PImage.open(file_image).convert("RGB")
            annotations = self.get_annotation_by_image_id(elem.id)
            for ann in annotations:
                category_name = self.cat_ids_to_names[ann.category_id]
                if category_names_set and category_name not in categories:
                    continue
                x1, y1, w, h = ann.bbox
                x2, y2 = x1 + w, y1 + h
                category_dir = Path(output_dir) / category_name
                category_dir.mkdir(exist_ok=True, parents=True)
                crop_out_file = category_dir / f"{ann.id}.jpg"
                crop = image.crop((x1, y1, x2, y2))
                crop.save(crop_out_file)

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