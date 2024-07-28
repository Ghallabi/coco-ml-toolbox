import streamlit as st

# from src.main_ui import MainUI


st.set_page_config(
    page_title="COCO-ML-TOOLBOX",
    page_icon=":rocket",
)


col1, col2 = st.sidebar.columns([0.5, 0.5])
with col1:
    st.markdown("")

with col2:
    col2.markdown("")

st.write("# Welcome to COCO ML Toolbox! 👋")
st.markdown("###")
st.markdown(
    """

    ### Overview

    This app provides essential tools for managing COCO (Common Objects in Context) files during the ML lifecycle. It features two main tools: Split and Merge.

    ## :screwdriver: Tools

    #### Split Tool

    **Description**: Divide a COCO file into training, validation, and testing files.

    **How it Works**:
    1. **Upload COCO File**: Start by uploading your files.
    2. **Adjust Split Ratios**: Use the slidebar to set the ratios (e.g., 70% train, 20% val, 10% test).
    3. **Execute Split**: Generate and download three new files for training, validation, and testing.

    ### Merge Tool

    **Description**: Combine multiple COCO files into one.

    **How it Works**:
    1. **Upload COCO Files**: Select and upload multiple COCO files.
    2. **Merge Files**: The tool combines all uploaded coco files into a single COCO file.




    ## :mag: Analysis
    WIP
"""
)
