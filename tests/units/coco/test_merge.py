from cocomltools.models.coco import COCO
from cocomltools.coco_ops import CocoOps


def test_merge_cmd_basic(
    coco_merge_input_1, coco_merge_input_2, info_merged_output_basic
):
    # ARRANGE
    coco_1 = COCO.from_dict(coco_merge_input_1)
    coco_2 = COCO.from_dict(coco_merge_input_2)
    coco_1.extend(coco_2)
    image_names = set(elem.file_name for elem in coco_1.images)
    image_ids = set(elem.id for elem in coco_1.images)

    category_names = set(elem.name for elem in coco_1.categories)
    category_ids = set(elem.id for elem in coco_1.categories)

    # ACT
    assert (
        image_ids == info_merged_output_basic["image_ids"]
    ), "Check if the merged image IDs match the expected output"
    assert (
        image_names == info_merged_output_basic["image_names"]
    ), "Check if the merged image names match the expected output"
    assert (
        category_names == info_merged_output_basic["category_names"]
    ), "Check if the merged category names match the expected output"
    assert (
        category_ids == info_merged_output_basic["category_ids"]
    ), "Check if the merged category IDs match the expected output"
    assert len(set(ann.id for ann in coco_1.annotations)) == len(
        coco_1.annotations
    ), "Ensure that all annotation IDs are unique after merging"


def test_merge_with_duplicates(
    coco_merge_input_2, coco_merge_input_3, info_merged_output_duplicate
):
    # ARRANGE
    coco_1 = COCO.from_dict(coco_merge_input_2)
    coco_2 = COCO.from_dict(coco_merge_input_3)
    coco_1.extend(coco_2)
    image_names = set(elem.file_name for elem in coco_1.images)
    image_ids = set(elem.id for elem in coco_1.images)

    category_names = set(elem.name for elem in coco_1.categories)
    category_ids = set(elem.id for elem in coco_1.categories)

    # ACT
    assert (
        image_ids == info_merged_output_duplicate["image_ids"]
    ), "Check if the merged image IDs match the expected output with duplicates"
    assert (
        image_names == info_merged_output_duplicate["image_names"]
    ), "Check if the merged image names match the expected output with duplicates"
    assert (
        category_names == info_merged_output_duplicate["category_names"]
    ), "Check if the merged category names match the expected output with duplicates"
    assert (
        category_ids == info_merged_output_duplicate["category_ids"]
    ), "Check if the merged category IDs match the expected output with duplicates"
    assert (
        len(coco_1.images) == 3
    ), "Ensure the total number of images is as expected after merging with duplicates"
    assert (
        len(coco_1.categories) == 7
    ), "Ensure the total number of categories is as expected after merging with duplicates"
    assert (
        coco_1.cat_ids_to_names[coco_1.annotations[3].category_id] == "bus"
    ), "Verify that the category ID of the 4th annotation corresponds to 'bus'"
    assert (
        coco_1.cat_ids_to_names[coco_1.annotations[2].category_id] == "bicycle"
    ), "Verify that the category ID of the 3rd annotation corresponds to 'bicycle'"
