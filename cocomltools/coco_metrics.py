from cocomltools.models.coco import COCO
from cocomltools.models.base import Annotation
from collections import defaultdict


class COCOMetrics:

    def __init__(self, ground_truth_coco_file: str, predictions_coco_file: str):
        self.coco_ground_truth = COCO.from_json_file(ground_truth_coco_file)
        self.coco_predictions = COCO.from_json_file(predictions_coco_file)
        print(
            f"Number of annotations before removing {len(self.coco_ground_truth.annotations)}"
        )
        # self.coco_ground_truth.remove_category_from_coco("unknown")
        print(
            f"Number of annotations after removing {len(self.coco_ground_truth.annotations)}"
        )
        self.iou_threshold = 0.5

    def _iou(self, box1, box2) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])

        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def evaluate(self) -> dict:
        results = defaultdict(list)
        for image_id, preds in self.coco_predictions.image_ids_to_anns.items():
            file_name = self.coco_predictions.image_ids_to_names[image_id]
            gt_image_id = self.coco_ground_truth.image_names_to_ids[file_name]
            gt_anns = self.coco_ground_truth.image_ids_to_anns[gt_image_id]
            gt_matched = [False] * len(gt_anns)
            for pred in preds:
                best_iou = 0
                best_gt_idx = -1
                categ_name_pred = self.coco_predictions.cat_ids_to_names[
                    pred.category_id
                ]
                for idx, gt in enumerate(gt_anns):
                    categ_name_gt = self.coco_ground_truth.cat_ids_to_names[
                        gt.category_id
                    ]
                    if not gt_matched[idx] and categ_name_gt == categ_name_pred:
                        iou = self._iou(gt.bbox, pred.bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx

                if best_iou >= self.iou_threshold:
                    results["TP"].append(1)
                    gt_matched[best_gt_idx] = True
                else:
                    results["FP"].append(1)

            # Count unmatched ground truths as false negatives
            results["FN"].extend([1 for matched in gt_matched if not matched])

        # Calculate precision and recall
        TP = sum(results["TP"])
        FP = sum(results["FP"])
        FN = sum(results["FN"])

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0

        # Placeholder for mAP (mean Average Precision) implementation
        results["Precision"] = precision
        results["Recall"] = recall
        results["mAP"] = precision  # Simplified for single IoU threshold

        return results


if __name__ == "__main__":
    from pathlib import Path

    coco_gt_path = Path(
        "/Users/faroukneurolabs/dev-ws/coco-ml-toolbox/data/evals/coco_ground_truth.json"
    )
    coco_pred_path = Path(
        "/Users/faroukneurolabs/dev-ws/coco-ml-toolbox/data/evals/coco_predictions.json"
    )

    coco_metrics = COCOMetrics(
        predictions_coco_file=coco_pred_path, ground_truth_coco_file=coco_gt_path
    )
    results = coco_metrics.evaluate()
    print(
        f"Precision: {results['Precision']}, Recall: {results['Recall']}, mAP: {results['mAP']}"
    )
