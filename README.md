# COCO ML Toolbox

COCO ML Toolbox is a command-line interface (CLI) tool for managing COCO (Common Objects in Context) dataset files. This toolbox provides functionalities to split, merge, and crop COCO datasets, making it easier to manipulate and prepare datasets for machine learning tasks.

## Features

- **Split** a COCO dataset into training and testing datasets with a specified ratio.
- **Merge** multiple COCO dataset files into a single file.
- **Crop** images based on annotations in a COCO dataset.

## Streamlit App
You can use our streamlit app [here](https://coco-ml-toolbox.streamlit.app/). 
## Installation

To use the COCO ML Toolbox, clone the repository and install the required dependencies.

```bash
git clone https://github.com/Ghallabi/coco-ml-toolbox.git
cd coco-ml-toolbox
pip install -r requirements.txt
```

## Usage

The main script for the COCO ML Toolbox provides three commands: split, merge, and crop. Below are the details on how to use each command.

### Split

Splits a COCO dataset into training and testing datasets.

```bash
python main_cli.py split --coco-path /path/to/coco.json --output-dir /path/to/output --ratio 0.2 --mode random
```

* --coco-path: Path to the COCO file (JSON).
* --output-dir: (Optional) Path to save the split COCO files. Defaults to the directory of the input COCO file.
* --ratio: (Optional) Split ratio. Defaults to 0.2.
* --mode: (Optional) Split mode. Options are random, strat_single_obj, or strat_multi_obj. Defaults to random.


### Merge

Merges multiple COCO dataset files into a single file.

```bash
python main_cli.py merge --coco-paths /path/to/coco1.json,/path/to/coco2.json --output-dir /path/to/output
```

* --coco-paths: Comma-separated paths to the COCO files (JSON).
* --output-dir: (Optional) Path to save the merged COCO file. Defaults to the directory of the first input COCO file.

### Crop
Crops images based on annotations in a COCO dataset.

```bash
python main_cli.py crop --coco-path /path/to/coco.json --images-dir /path/to/images --output-dir /path/to/cropped_images
```

* --coco-path: Path to the COCO file (JSON).
* --images-dir: Path to the directory containing the COCO image files.
* --output-dir: (Optional) Path to save the cropped images. Defaults to a "cropped" directory within the parent directory of the images.
