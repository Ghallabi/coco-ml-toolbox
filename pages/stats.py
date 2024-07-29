import streamlit as st
from models.coco import COCO
from pages.utils import coco_file_uploader
import json
from collections import defaultdict
import pandas as pd
import altair as alt


class CocoAnalysis:
    def __init__(self):
        st.session_state.files_ready, self.uploaded_files = coco_file_uploader(
            st.sidebar
        )
        self._setup_tabs()
        if st.session_state.files_ready:

            self.coco = COCO.from_dict(
                json.loads(self.uploaded_files[0].getvalue().decode("utf-8"))
            )
            self._compute_coco_stats()
            self.display_stats_grid()

    def _setup_tabs(self):
        self.tab1, self.tab2, self.tab3 = st.tabs(
            ["Overview", "Categories", "Performance"]
        )

    def _compute_coco_stats(self):
        self.num_images = len(self.coco.images)
        self.num_anns = len(self.coco.annotations)
        self.num_categs = len(self.coco.categories)
        self.class_count = defaultdict(int)
        class_scores = defaultdict(float)
        obj_per_image = defaultdict(int)
        for ann in self.coco.annotations:
            self.class_count[ann.category_id] += 1
            obj_per_image[ann.image_id] += 1
            class_scores[ann.category_id] += ann.score

        self.avg_obj_per_image = int(
            sum(obj_per_image.values()) / len(obj_per_image.values())
        )
        self.min_obj_per_image = min(obj_per_image.values())
        self.max_obj_per_image = max(obj_per_image.values())

        self.class_scores = {
            cat_id: class_scores[cat_id] / self.class_count[cat_id]
            for cat_id in class_scores
        }

        category_names = [
            self.coco.cat_ids_to_names[cat_id] for cat_id in self.class_count.keys()
        ]
        counts = list(self.class_count.values())
        scores = list(self.class_scores.values())
        # Create a DataFrame for sorting
        self.df = pd.DataFrame(
            {"Category": category_names, "Count": counts, "Scores": scores}
        )

    def display_stats_grid(self):
        with self.tab1:
            self.display_general_stats()

        with self.tab2:
            col1, col2 = st.columns(2)
            with col1:
                self.plot_per_class_count(
                    top_n=20, title="Top 20 Classes (frequency)", sort_key="Count"
                )
            with col2:
                self.plot_per_class_count(
                    bottom_n=20, title="Bottom 20 Classes by Count", sort_key="Count"
                )

        with self.tab3:
            col1, col2 = st.columns(2)
            with col1:
                self.plot_per_class_count(
                    top_n=20, title="Top 20 Classes (scores)", sort_key="Scores"
                )
            with col2:
                self.plot_per_class_count(
                    bottom_n=20, title="Bottom 20 Classes (scores)", sort_key="Scores"
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
                self.num_images,
                self.num_anns,
                self.num_categs,
                self.avg_obj_per_image,
                self.min_obj_per_image,
                self.max_obj_per_image,
            ],
        }

        # Creating a DataFrame with the collected data
        df = pd.DataFrame(stats)

        # Displaying the table with Streamlit
        # st.table(df)
        st.table(df.assign(hack="").set_index("hack"))

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
