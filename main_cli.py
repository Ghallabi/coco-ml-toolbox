from cocomltools.models.coco import COCO
from cocomltools.coco_ops import CocoOps
from cocomltools.utils import check_is_json
import argparse
from pathlib import Path


def split_cmd(args):

    coco_path = Path(args.coco_path)

    if not coco_path.is_file():
        raise FileNotFoundError(f"File not found: {args.coco_path}")
    if not args.coco_path.endswith(".json"):
        raise ValueError("Incorrect file format, provide JSON as input")

    coco = COCO.from_json_file(args.coco_path)
    coco_ops = CocoOps(coco)
    coco_1, coco_2 = coco_ops.split(ratio=args.ratio, mode=args.mode)
    if args.output_dir and Path(args.output_dir).is_dir():
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.coco_path).parent

    coco_1.save_coco_dict(output_dir / "coco_train.json")
    coco_2.save_coco_dict(output_dir / "coco_test.json")


def merge_cmd(args):
    if not all([check_is_json(file) for file in args.coco_paths.split(",")]):
        raise ValueError("One or more inputs have incorrect format")

    coco_files = args.coco_paths.split(",")
    coco_base = COCO.from_json_file(coco_files[0])
    for coco_file in coco_files[1:]:
        coco = COCO.from_json_file(coco_file)
        coco_base.extend(coco)

    if args.output_dir and Path(args.output_dir).is_dir():
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(coco_files[0]).parent

    coco_base.save_coco_dict(output_dir / "coco_merged.json")


def crop_cmd(args):

    coco_file = args.coco_path
    coco = COCO.from_json_file(coco_file)
    coco_ops = CocoOps(coco)
    images_dir = Path(args.images_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir else images_dir.parent / "cropped"
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    coco_ops.crop(images_dir, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="coco-ml-cli", description="COCO tools for Machine Learning"
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
