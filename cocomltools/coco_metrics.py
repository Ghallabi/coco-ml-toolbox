from cocomltools.models.coco import COCO
from collections import defaultdict


class COCOMetrics:

    def __init__(self, ground_truth_coco_file: str, predictions_coco_file: str):
        self.coco_ground_truth = COCO.from_json_file(ground_truth_coco_file)
        self.coco_predictions = COCO.from_json_file(predictions_coco_file)

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

    def evaluate(self, iou_threshold: float = 0.5) -> dict:
        results = defaultdict(list)
        per_class_metrics = {
            self.coco_ground_truth.cat_ids_to_names[categ_id]: {
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "all": count,
            }
            for categ_id, count in self.coco_ground_truth.category_ids_to_ann_count.items()
        }
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

                if best_iou >= iou_threshold:
                    if categ_name_pred in per_class_metrics.keys():
                        per_class_metrics[categ_name_pred]["TP"] += 1
                    results["TP"].append(1)
                    gt_matched[best_gt_idx] = True
                else:
                    if categ_name_pred in per_class_metrics.keys():
                        per_class_metrics[categ_name_pred]["FP"] += 1
                    results["FP"].append(1)

            for idx, matched in enumerate(gt_matched):
                if not matched:
                    categ_name_gt = self.coco_ground_truth.cat_ids_to_names[
                        gt_anns[idx].category_id
                    ]
                    if categ_name_gt in per_class_metrics.keys():
                        per_class_metrics[categ_name_gt]["FN"] += 1
                    results["FN"].append(1)

        # per class metrics
        for stats in per_class_metrics.values():
            stats["Precision"] = (
                stats["TP"] / (stats["TP"] + stats["FP"])
                if (stats["TP"] + stats["FP"]) > 0
                else 0
            )
            stats["Recall"] = (
                stats["TP"] / (stats["TP"] + stats["FN"])
                if (stats["TP"] + stats["FN"]) > 0
                else 0
            )
            # stats['Recall'] = TP / (TP + FN)

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

        return results, per_class_metrics
