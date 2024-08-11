import pytest
from cocomltools.models.coco import COCO
from cocomltools.coco_ops import CocoOps


def test_filter_cmd(coco_delete_input):
    # ARANGE
    coco_ops = CocoOps.from_dict(coco_delete_input)

    # ACT - remove one image.
    images_to_remove = ["image1.jpg", "image5.jpg", "image10.jpg"]
    coco_ops.filter(image_names=images_to_remove)

    # Assert
    assert all(
        removed_image not in set(image.file_name for image in coco_ops.coco.images)
        for removed_image in images_to_remove
    )
    assert len(coco_ops.coco.annotations) == 16

    # ACT - remove category6
    coco_ops.filter(category_names=["category6"])

    # Assert
    assert len(coco_ops.coco.annotations) == 14
    assert "image11.jpg" not in set(
        image.file_name for image in coco_ops.coco.images
    ), "removing category6 should remove image11.jpg"
