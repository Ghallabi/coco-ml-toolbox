from cocomltools.cmd import Cmd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        prog="cocomltools-cli", description="COCO tools for Machine Learning"
    )
    subparsers = parser.add_subparsers(dest="cmd")
    parser_split = subparsers.add_parser("split", help="Split coco file")
    parser_split.add_argument(
        "--coco-path", required=True, type=str, help="Path to coco file"
    )
    parser_split.add_argument(
        "--output-dir", required=False, type=str, help="Path to save split coco files"
    )
    parser_split.add_argument(
        "--ratio", type=int, default=0.2, help="Split ratio - default to 0.2"
    )
    parser_split.add_argument(
        "--mode",
        type=str,
        default="random",
        help="Should one of the following: random, strat_single_obj or strat_multi_obj",
    )
    parser_merge = subparsers.add_parser("merge", help="Merge coco files")
    parser_merge.add_argument(
        "--coco-paths", required=True, help="Path to coco files separated by comma"
    )
    parser_merge.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Path where to save merged coco file",
    )
    parser_crop = subparsers.add_parser(
        "crop", help="Crop images from annotations in coco file"
    )
    parser_crop.add_argument("--coco-path", required=True, help="Path to coco file")
    parser_crop.add_argument(
        "--images-dir", required=True, help="Path to coco image files"
    )
    parser_crop.add_argument("--output-dir", required=False, help="Path to output dir")
    parser_crop.add_argument(
        "--num-workers",
        required=False,
        default=1,
        type=int,
        help="number of workers to crop the dataset",
    )
    args = parser.parse_args()

    parser_filter = subparsers.add_parser("filter", help="Split coco file")
    parser_filter.add_argument(
        "--coco-path", required=True, type=str, help="Path to coco file"
    )
    parser_filter.add_argument(
        "--categories",
        required=True,
        type=str,
        help="comma separated string of categories to filter",
    )
    parser_filter.add_argument(
        "--output-dir", required=False, type=str, help="Path to save split coco files"
    )
    return args


def main():
    args = parse_args()
    Cmd(args)


if __name__ == "__main__":
    main()
