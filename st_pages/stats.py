import streamlit as st
from cocomltools.models.coco import COCO
from cocomltools.coco_ops import CocoOps
from st_pages.utils import coco_file_uploader
import json
from collections import defaultdict
import pandas as pd
import altair as alt


@st.cache_data
def stats_dict_to_dataframe(stats):
    counts = list(stats["count_objs_per_categ"].values())
    scores = list(stats["class_scores"].values())
    # Create a DataFrame for sorting
    df = pd.DataFrame(
        {"Category": stats["categories"], "Count": counts, "Scores": scores}
    )
    return df


@st.cache_data
def analyse_coco_input(input_file):
    coco_ops = CocoOps(
        COCO.from_dict(json.loads(input_file.getvalue().decode("utf-8")))
    )
    stats = coco_ops.calculate_coco_stats()
    df = stats_dict_to_dataframe(stats)
    return coco_ops, stats, df


class CocoAnalysis:
    def __init__(self):
        st.session_state.files_ready, self.uploaded_files = coco_file_uploader(
            st.sidebar
        )
        self._setup_tabs()
        self.top_k = st.sidebar.slider("Top k", min_value=1, max_value=100)

        if st.session_state.files_ready:

            self.coco_ops, self.stats, self.df = analyse_coco_input(
                self.uploaded_files[0]
            )
            self.display_stats_grid()

    def _setup_tabs(self):
        self.tab1, self.tab2, self.tab3 = st.tabs(
            ["Overview", "Distributions", "Per category stats"]
        )

    def display_stats_grid(self):
        with self.tab1:
            self.display_general_stats()

        with self.tab2:
            col1, col2 = st.columns(2)
            with col1:
                self.plot_bbox_distribution()
            with col2:
                self.plot_img_size_distribution()

        with self.tab3:
            col1, col2 = st.columns(2)
            with col1:
                self.plot_per_class_count(
                    top_n=self.top_k,
                    title=f"Top {self.top_k} Classes (Frequency)",
                    sort_key="Count",
                )
            with col2:
                self.plot_per_class_count(
                    bottom_n=self.top_k,
                    title=f"Bottom {self.top_k} Classes (Frequency)",
                    sort_key="Count",
                )

            col1, col2 = st.columns(2)
            with col1:
                self.plot_per_class_count(
                    top_n=self.top_k,
                    title=f"Top {self.top_k} Classes (scores)",
                    sort_key="Scores",
                )
            with col2:
                self.plot_per_class_count(
                    bottom_n=self.top_k,
                    title=f"Bottom {self.top_k} Classes (scores)",
                    sort_key="Scores",
                )

    def display_general_stats(self):
        # Collecting data in a more descriptive format
        stats = {
            "Metric": [
                "Number of Images",
                "Number of Annotations",
                "Number of Categories",
                "Average objets per image",
                "Min objects per image",
                "Max objects per image",
            ],
            "Value": [
                len(self.coco_ops.coco.images),
                len(self.coco_ops.coco.annotations),
                len(self.coco_ops.coco.categories),
                self.stats["avg_obj_per_image"],
                self.stats["min_obj_per_image"],
                self.stats["max_obj_per_image"],
            ],
        }

        # Creating a DataFrame with the collected data
        df = pd.DataFrame(stats)

        # Displaying the table with Streamlit
        # st.table(df)
        st.table(df.assign(hack="").set_index("hack"))

    def plot_img_size_distribution(self):

        df = pd.DataFrame(
            self.stats["img_width_heights"].values(),
            columns=["Img width", "Img height"],
        )

        chart = (
            alt.Chart(df)
            .mark_circle(size=60)
            .encode(
                x="Img width",
                y="Img height",
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    def plot_bbox_distribution(self):
        df = pd.DataFrame(
            self.stats["ann_width_heights"],
            columns=["Bbox width", "Bbox height", "Category"],
        )

        chart = (
            alt.Chart(df)
            .mark_circle(size=60)
            .encode(x="Bbox width", y="Bbox height")
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    def plot_per_class_count(
        self, top_n=None, bottom_n=None, title="", sort_key="Count"
    ):

        # Sort by count and select top_n or bottom_n
        if top_n:
            df = self.df.sort_values(by=sort_key, ascending=False).head(top_n)
        elif bottom_n:
            df = self.df.sort_values(by=sort_key, ascending=True).head(bottom_n)

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(f"{sort_key}:Q"),
                y=alt.Y("Category:N", sort="-x"),
                tooltip=["Category", sort_key],
            )
            .properties(title=title, width=500, height=400)
        )

        st.altair_chart(chart, use_container_width=True)


coco_analysis = CocoAnalysis()
