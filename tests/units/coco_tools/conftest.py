import pytest



@pytest.fixture
def coco_1_input():
    return {
    "info": {
        "description": "Sample COCO dataset 1",
        "year": 2024,
        "contributor": "OpenAI"
    },
    "licenses": [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ],
    "images": [
        {
            "id": 1,
            "file_name": "image1.jpg",
            "height": 800,
            "width": 600,
            "license": 1,
            "flickr_url": "http://example.com/image1.jpg",
            "coco_url": "http://example.com/image1.jpg",
            "date_captured": "2024-07-01 10:00:00"
        },
        {
            "id": 2,
            "file_name": "image2.jpg",
            "height": 800,
            "width": 600,
            "license": 1,
            "flickr_url": "http://example.com/image2.jpg",
            "coco_url": "http://example.com/image2.jpg",
            "date_captured": "2024-07-01 10:00:00"
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [
                100,
                200,
                50,
                80
            ],
            "area": 4000,
            "segmentation": [],
            "iscrowd": 0
        },
        {
            "id": 2,
            "image_id": 2,
            "category_id": 2,
            "bbox": [
                150,
                250,
                60,
                90
            ],
            "area": 5400,
            "segmentation": [],
            "iscrowd": 0
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "person",
            "supercategory": "person"
        },
        {
            "id": 2,
            "name": "bicycle",
            "supercategory": "vehicle"
        },
        {
            "id": 3,
            "name": "car",
            "supercategory": "vehicle"
        },
        {
            "id": 4,
            "name": "motorcycle",
            "supercategory": "vehicle"
        },
        {
            "id": 5,
            "name": "airplane",
            "supercategory": "vehicle"
        }
    ]
}

@pytest.fixture
def coco_2_input():
    return {
    "info": {
        "description": "Sample COCO dataset 2",
        "year": 2024,
        "contributor": "OpenAI"
    },
    "licenses": [
        {
            "id": 2,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ],
    "images": [
        {
            "id": 1,
            "file_name": "image3.jpg",
            "height": 1024,
            "width": 768,
            "license": 2,
            "flickr_url": "http://example.com/image3.jpg",
            "coco_url": "http://example.com/image3.jpg",
            "date_captured": "2024-07-01 12:00:00"
        },
        {
            "id": 2,
            "file_name": "image4.jpg",
            "height": 1024,
            "width": 768,
            "license": 2,
            "flickr_url": "http://example.com/image4.jpg",
            "coco_url": "http://example.com/image4.jpg",
            "date_captured": "2024-07-01 12:00:00"
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 2,
            "bbox": [
                120,
                220,
                70,
                100
            ],
            "area": 7000,
            "segmentation": [],
            "iscrowd": 0
        },
        {
            "id": 2,
            "image_id": 2,
            "category_id": 3,
            "bbox": [
                180,
                280,
                80,
                120
            ],
            "area": 9600,
            "segmentation": [],
            "iscrowd": 0
        }
    ],
    "categories": [
        {
            "id": 2,
            "name": "bicycle",
            "supercategory": "vehicle"
        },
        {
            "id": 3,
            "name": "car",
            "supercategory": "vehicle"
        },
        {
            "id": 6,
            "name": "truck",
            "supercategory": "vehicle"
        },
        {
            "id": 7,
            "name": "boat",
            "supercategory": "vehicle"
        },
        {
            "id": 8,
            "name": "traffic light",
            "supercategory": "outdoor"
        }
    ]
}



@pytest.fixture
def info_merged_output():
    return {
        "image_ids": set([1, 2, 3, 4]),
        "image_names": set(["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]),
        "category_names": set(["person", "bicycle", "car", "motorcycle", "airplane", "truck", "boat", "traffic light"]),
        "category_ids": set([1, 2, 3, 4, 5, 6, 7, 8])
    }