from pydantic import BaseModel, Field
from typing import List, Union


class Image(BaseModel):
    id: int = Field(default=0)
    file_name: str
    width: int
    height: int


class Annotation(BaseModel):
    id: int = Field(default=0)
    image_id: int
    category_id: int
    score: float = Field(default=1.0)
    bbox: List[float]
    segmentation: Union[List[float], List[List[float]]] = Field(default=[])
    area: float
    iscrowd: int = Field(default=0)


class Category(BaseModel):
    id: int = Field(default=0)
    name: str
