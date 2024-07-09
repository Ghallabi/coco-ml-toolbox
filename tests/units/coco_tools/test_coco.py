import pytest
from coco_tools.coco import COCO





def test_split_cmd(coco_1_input, coco_2_input):
    pass




def test_merge_cmd_basic(coco_1_input, coco_2_input, info_merged_output):
    # ARRANGE
    coco_1 = COCO.from_dict(coco_1_input)
    coco_2 = COCO.from_dict(coco_2_input)
    coco_1.extend(coco_2)
    image_names = set(elem.file_name for elem in coco_1.images)
    image_ids = set(elem.id for elem in coco_1.images)

    category_names = set(elem.name for elem in coco_1.categories)
    category_ids = set(elem.id for elem in coco_1.categories)
    #ACT
    
    assert image_ids == info_merged_output["image_ids"]
    assert image_names == info_merged_output['image_names']
    assert category_names == info_merged_output['category_names']
    assert category_ids == info_merged_output['category_ids']



