from pathlib import Path
from cocomltools.coco_ops import CocoOps
from cocomltools.utils import check_is_json


class Cmd:
    def __init__(self, args):
        if args.cmd == "split":
            self.split_cmd(args)
        elif args.cmd == "merge":
            self.merge_cmd(args)
        elif args.cmd == "crop":
            self.crop_cmd(args)
        elif args.cmd == "filter":
            self.filter_cmd(args)

    def split_cmd(self, args):

        coco_path = Path(args.coco_path)
        if not check_is_json(coco_path):
            raise ValueError("Missing / Incorrect file format, provide JSON as input")

        coco_ops = CocoOps.from_json_file(coco_path)
        coco_1, coco_2 = coco_ops.split(ratio=args.ratio, mode=args.mode)
        if args.output_dir and Path(args.output_dir).is_dir():
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(args.coco_path).parent

        coco_1.save_coco_dict(output_dir / "coco_train.json")
        coco_2.save_coco_dict(output_dir / "coco_test.json")

    def merge_cmd(self, args):
        input_files = args.coco_paths.split(",")

        coco_merged = CocoOps.merge(input_files)

        if args.output_dir and Path(args.output_dir).is_dir():
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(input_files[0]).parent

        coco_merged.save_coco_dict(output_dir / "coco_merged.json")

    def crop_cmd(self, args):

        coco_file = args.coco_path
        coco_ops = CocoOps.from_json_file(coco_file)
        images_dir = Path(args.images_dir)
        output_dir = (
            Path(args.output_dir) if args.output_dir else images_dir.parent / "cropped"
        )
        output_dir.mkdir(exist_ok=True, parents=True)
        coco_ops.crop(images_dir, output_dir, max_workers=args.num_workers)

    def filter_cmd(self, args):
        coco_file = args.coco_path
        coco_ops = CocoOps.from_json_file(coco_file)
        if args.output_dir and Path(args.output_dir).is_dir():
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(coco_file).parent
        coco_output = coco_ops.filter(args.categories.split(","))
        coco_output.save_coco_dict(output_dir / "coco_filtered.json")
