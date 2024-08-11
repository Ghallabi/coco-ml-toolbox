import streamlit as st

st.write("# Welcome to COCO ML Toolbox! 👋")
st.markdown("###")
st.markdown(
    """

    ### Overview

    This app provides essential tools for managing COCO (Common Objects in Context) files during the ML lifecycle. It features two main tools: Split and Merge.

    ## Tools :screwdriver:

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


    ##  Statistics :bar_chart:
    Upload a coco file to get insights about the following statistics:
    * Overal statistics such as number of images, number of annotations, number of categories. etc.
    * Per class statistics:
        * X Top / Bottom classes in terms of frequency
        * X Top / Bottom classes in term of class score (if you have coco output from inference job)
    * Distributions:
        * Width and height distribution for annotations (bounding boxes)
        * Width and height distribution for images.
        
"""
)
