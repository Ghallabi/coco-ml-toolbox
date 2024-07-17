# COCO ML Toolbox

COCO ML Toolbox is a command-line interface (CLI) tool for managing COCO (Common Objects in Context) dataset files. This toolbox provides functionalities to split, merge, and crop COCO datasets, making it easier to manipulate and prepare datasets for machine learning tasks.

## Features

- **Split** a COCO dataset into training and testing datasets with a specified ratio.
- **Merge** multiple COCO dataset files into a single file.
- **Crop** images based on annotations in a COCO dataset.

## Installation

To use the COCO ML Toolbox, clone the repository and install the required dependencies.

```bash
git clone https://github.com/your-username/coco-ml-toolbox.git
cd coco-ml-toolbox
pip install -r requirements.txt
```

## Usage

The main script for the COCO ML Toolbox provides three commands: split, merge, and crop. Below are the details on how to use each command.

### Split

Splits a COCO dataset into training and testing datasets.

```bash
python main.py split --coco-path /path/to/coco.json --output-dir /path/to/output --ratio 0.2 --mode random
```

* --coco-path: Path to the COCO file (JSON).
* --output-dir: (Optional) Path to save the split COCO files. Defaults to the directory of the input COCO file.
* --ratio: (Optional) Split ratio. Defaults to 0.2.
* --mode: (Optional) Split mode. Options are random, strat_single_obj, or strat_multi_obj. Defaults to random.