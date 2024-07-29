import streamlit as st
from typing import List


def check_valid_cocos_st(json_data_list: List[dict]):
    valid_keys = {"images", "annotations", "categories"}
    for json_data_path in json_data_list:
        json_data = json_data_path.getvalue().decode("utf-8")
        if not all(key in json_data for key in valid_keys):
            return False
    return True


def coco_file_uploader(container):

    uploaded_files = container.file_uploader(
        "Choose COCO files",
        type=["json"],
        accept_multiple_files=True,
    )
    files_ready = False
    if len(uploaded_files) > 0:
        if check_valid_cocos_st(uploaded_files):
            files_ready = True
        else:
            container.text("Invalid input json(s) coco format.")
    return files_ready, uploaded_files
