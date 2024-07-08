from coco_tools.coco import COCO
import argparse


def split_cmd(args):
    pass


def merge_cmd(args):
    pass


def crop_cmd(args):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        prog="coco-ml-cli", description="COCO tools for Machine Learning"
    )
    subparsers = parser.add_subparsers(dest="cmd")
    parser_split = subparsers.add_parser("split", help="Split coco file")
    parser_split.add_argument("--coco-path", required=True, help="Path to coco file")
    parser_merge = subparsers.add_parser("merge", help="Merge coco files")
    parser_merge.add_argument(
        "--coco-paths", required=True, help="Path to coco files separated by comma"
    )
    parser_crop = subparsers.add_parser(
        "crop", help="Crop images from annotations in coco file"
    )
    parser_crop.add_argument("--coco-path", required=True, help="Path to coco file")
    args = parser.parse_args()
    return args


def main(args):
    if args.cmd == "split":
        split_cmd(args)
    elif args.cmd == "merge":
        merge_cmd(args)
    elif args.cmd == "crop":
        crop_cmd(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
