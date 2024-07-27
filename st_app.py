import streamlit as st
from models.coco import COCO
import json
import pandas as pd


class StApp:

    def __init__(self):
        self._setup_sidebar_ui()
        self._setup_main_ui()

    def _setup_sidebar_ui(self):

        st.sidebar.markdown("# Coco ML Toolbox")

        if "merge_button_clicked" not in st.session_state:
            st.session_state.merge_button_clicked = False

        if "split_button_clicked" not in st.session_state:
            st.session_state.split_button_clicked = False

        if "results_ready" not in st.session_state:
            st.session_state.results_ready = False

        cols = st.sidebar.columns(2)

        if cols[0].button("merge"):
            st.session_state.merge_button_clicked = True
            st.session_state.split_button_clicked = False

        if cols[1].button("split"):
            st.session_state.split_button_clicked = True
            st.session_state.merge_button_clicked = False

        self.split_ratios = []
        # col1, col2 = st.sidebar.columns(2)

        self.split_ratios.append(
            st.sidebar.slider("Split Train / test", min_value=0.0, max_value=1.0)
        )
        self.split_ratios.append(
            st.sidebar.slider("Split Train / Val", min_value=0.0, max_value=1.0)
        )
        print(self.split_ratios)
        st.sidebar.markdown("#")
        col1, col2, col3 = st.sidebar.columns([0.3, 0.4, 0.3])
        col1.markdown("  ")
        if col2.button("clear"):
            st.session_state.results_ready = False
            st.session_state.split_button_clicked = False
            st.session_state.merge_button_clicked = False
        col3.markdown("  ")

    def _setup_logo_widget(self):
        col1, col2, col3 = st.sidebar.columns([0.2, 0.6, 0.2])
        with col1:
            st.markdown("")

        with col2:
            col2.image("./logo/nlb_logo.png", width=150)

        with col3:
            col3.markdown("")

    def _setup_main_ui(self):

        self.coco_file_uploader()
        self.process_files()

        print(len(self.uploaded_files))

    def coco_file_uploader(self):

        self.uploaded_files = st.file_uploader(
            "Choose COCO files", type=["json"], accept_multiple_files=True
        )
        if len(self.uploaded_files) > 0:
            st.session_state.files_ready = True
            print("We will perform processing here")
        else:
            st.session_state.files_ready = False

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
                        coco_train,
                        label="download coco train",
                        file_name="coco_train.json",
                    )

                with col2:
                    self.download_coco_button(
                        coco_val,
                        label="download coco val",
                        file_name="coco_val.json",
                    )

                with col3:
                    self.download_coco_button(
                        coco_test,
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
        coco_base = COCO(json.loads(self.uploaded_files[0].getvalue().decode("utf-8")))
        train_coco, val_coco, test_coco = coco_base.split_coco(
            val_fraction=self.split_ratios[1], test_fraction=self.split_ratios[0]
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