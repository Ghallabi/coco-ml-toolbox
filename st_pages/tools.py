import streamlit as st
from cocomltools.models.coco import COCO
from cocomltools.coco_ops import CocoOps
import json
import pandas as pd
from st_pages.utils import coco_file_uploader


class StApp:

    def __init__(self):
        self._setup_sidebar_ui()
        self._setup_main_ui()

    def _setup_sidebar_ui(self):

        if "merge_button_clicked" not in st.session_state:
            st.session_state.merge_button_clicked = False

        if "split_button_clicked" not in st.session_state:
            st.session_state.split_button_clicked = False

        if "results_ready" not in st.session_state:
            st.session_state.results_ready = False

        st.sidebar.subheader("Merge")
        cols = st.sidebar.columns([0.3, 0.4, 0.3])
        if cols[1].button("merge"):
            st.session_state.merge_button_clicked = True
            st.session_state.split_button_clicked = False

        st.sidebar.markdown("#")
        st.sidebar.subheader("Split")

        cols = st.sidebar.columns(2)

        if cols[0].button("stratified split"):
            st.session_state.split_button_clicked = True
            st.session_state.merge_button_clicked = False
            st.session_state.split_mode = "strat"

        if cols[1].button("random split"):
            st.session_state.split_button_clicked = True
            st.session_state.merge_button_clicked = False
            st.session_state.split_mode = "random"

        st.sidebar.markdown("#")
        st.sidebar.subheader("split params")

        self.split_ratios = []

        self.split_ratios.append(
            st.sidebar.slider("Split Train / test", min_value=0.0, max_value=1.0)
        )
        self.split_ratios.append(
            st.sidebar.slider("Split Train / Val", min_value=0.0, max_value=1.0)
        )

        st.sidebar.markdown("#")
        col1, col2, col3 = st.sidebar.columns([0.3, 0.4, 0.3])
        col1.markdown("  ")
        if col2.button("clear"):
            st.session_state.results_ready = False
            st.session_state.split_button_clicked = False
            st.session_state.merge_button_clicked = False
        col3.markdown("  ")

    def _setup_main_ui(self):

        st.session_state.files_ready, self.uploaded_files = coco_file_uploader(
            st, accept_multiple_files=True
        )
        self.process_files()

    def process_files(self):

        if st.session_state.split_button_clicked and st.session_state.files_ready:
            print(f"We will split the coco files {len(self.uploaded_files)}")
            if len(self.uploaded_files) > 1:
                print("Only one coco file is needed")
                # st.session_state.split_button_clicked = False
            else:
                coco_train, coco_val, coco_test = self.split_coco_file()

                col1, col2, col3 = st.columns(3)
                with col1:
                    self.download_coco_button(
                        coco_train.get_coco_dict(),
                        label="download coco train",
                        file_name="coco_train.json",
                    )

                with col2:
                    self.download_coco_button(
                        coco_val.get_coco_dict(),
                        label="download coco val",
                        file_name="coco_val.json",
                    )

                with col3:
                    self.download_coco_button(
                        coco_test.get_coco_dict(),
                        label="download coco test",
                        file_name="coco_test.json",
                    )
            st.session_state.processing_completed = True
        elif st.session_state.merge_button_clicked and st.session_state.files_ready:
            print(f"We will merge coco files {len(self.uploaded_files)}")

            if len(self.uploaded_files) <= 1:
                print("You need to upload more than one coco file to merge")
                st.session_state.merge_button_clicked = False
            else:
                coco_output = self.merge_coco_files()
                self.download_coco_button(
                    coco_output.get_coco_dict(),
                    label="download merged coco",
                    file_name="coco_merged.json",
                )

                st.session_state.results_ready = True

    def merge_coco_files(self) -> dict:
        coco_base = COCO.from_dict(
            json.loads(self.uploaded_files[0].getvalue().decode("utf-8"))
        )
        for uploaded_file in self.uploaded_files[1:]:
            coco_dict = json.loads(uploaded_file.getvalue().decode("utf-8"))
            coco_to_add = COCO.from_dict(coco_dict)
            coco_base.extend(coco_to_add)
        st.session_state.processing_completed = True
        st.session_state.merge_button_clicked = False
        return coco_base

    def split_coco_file(self) -> list[dict]:
        if self.split_ratios == [0.0, 0.0]:
            st.text("Both split ratios are zero! cannot split")
            return COCO(), COCO(), COCO()

        coco_base = COCO.from_dict(
            json.loads(self.uploaded_files[0].getvalue().decode("utf-8"))
        )
        train_coco, test_coco = CocoOps(coco_base).split(
            ratio=self.split_ratios[0], mode=st.session_state.split_mode
        )
        train_coco, val_coco = CocoOps(train_coco).split(
            ratio=self.split_ratios[1], mode=st.session_state.split_mode
        )

        return train_coco, val_coco, test_coco

    def download_coco_button(
        self,
        coco_dict,
        label: str = "Download COCO result",
        file_name: str = "coco_merged.json",
    ):
        json_string = json.dumps(coco_dict)
        st.download_button(
            label=label,
            file_name=file_name,
            mime="application/json",
            data=json_string,
            help=None,
        )


coco_tools_ui = StApp()
