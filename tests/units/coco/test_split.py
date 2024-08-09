from cocomltools.models.coco import COCO
from cocomltools.coco_ops import CocoOps


def test_split_random(coco_split_random_input):
    # ARRANGE
    coco = COCO.from_dict(coco_split_random_input)
    coco_ops = CocoOps(coco)
    coco_1, coco_2 = coco_ops.split(ratio=0.2, mode="random")

    ann_ids_coco_2 = set([elem.id for elem in coco_2.annotations])
    img_ids_coco_2 = set([elem.id for elem in coco_2.images])

    # ACT
    assert all(
        elem.id not in ann_ids_coco_2 for elem in coco_1.annotations
    ), "Ensure no annotations are shared between coco_1 and coco_2"
    assert all(
        elem.id not in img_ids_coco_2 for elem in coco_1.images
    ), "Ensure no images are shared between coco_1 and coco_2"

    # Ensure the split ratio is respected
    total_images = len(coco.images)
    expected_coco_2_images = int(0.2 * total_images)
    actual_coco_2_images = len(coco_2.images)
    assert (
        abs(actual_coco_2_images - expected_coco_2_images) <= 1
    ), "Split ratio not respected: #images(A) + #images(B) != total(images)"

    # Ensure annotations are correctly associated with the images in each split
    img_ids_coco_1 = set([elem.id for elem in coco_1.images])
    for annotation in coco_1.annotations:
        assert annotation.image_id in img_ids_coco_1

    for annotation in coco_2.annotations:
        assert annotation.image_id in img_ids_coco_2

    # Ensure the category lists are the same in both splits
    category_ids_coco_1 = set([elem.id for elem in coco_1.categories])
    category_ids_coco_2 = set([elem.id for elem in coco_2.categories])
    assert (
        category_ids_coco_1 == category_ids_coco_2
    ), "Ensure Category ids are the same in both splits"

    # Ensure the category lists are the same in both splits
    category_names_coco_1 = set([elem.name for elem in coco_1.categories])
    category_names_coco_2 = set([elem.name for elem in coco_2.categories])
    assert (
        category_names_coco_1 == category_names_coco_2
    ), "Ensure Category names are the same in both splits"


def test_split_strat():
    # TO BE IMPLEMENTED
    pass
