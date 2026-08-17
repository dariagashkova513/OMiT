from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .annotations import GTInstance, GTSample


@dataclass
class Prediction:
    bbox: tuple[float, float, float, float]
    class_id: int
    score: float
    mask: np.ndarray | None = None


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_predictions(gt_instances: list[GTInstance], predictions: list[Prediction], iou_thres=0.5):
    matched, true_positives, false_positives = set(), 0, 0
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        candidates = [(box_iou(gt.bbox, prediction.bbox), index) for index, gt in enumerate(gt_instances)
                      if index not in matched and gt.class_id == prediction.class_id]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_index >= 0 and best_iou >= iou_thres:
            matched.add(best_index)
            true_positives += 1
        else:
            false_positives += 1
    return true_positives, false_positives, len(gt_instances) - len(matched)


def precision_recall(tp, fp, fn):
    return (tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0)


def dataset_precision_recall(gt_samples: list[GTSample], predictions_by_image, iou_thres=0.5):
    tp = fp = fn = 0
    for sample in gt_samples:
        sample_tp, sample_fp, sample_fn = match_predictions(
            sample.instances, predictions_by_image.get(sample.image_id, []), iou_thres
        )
        tp, fp, fn = tp + sample_tp, fp + sample_fp, fn + sample_fn
    precision, recall = precision_recall(tp, fp, fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def per_class_precision_recall(gt_samples, predictions_by_image, class_names, iou_thres=0.5):
    report = {}
    for class_id, class_name in enumerate(class_names):
        filtered_samples = [GTSample(s.image_id, s.image_path, s.width, s.height,
                            [item for item in s.instances if item.class_id == class_id]) for s in gt_samples]
        filtered_predictions = {
            image_id: [item for item in items if item.class_id == class_id]
            for image_id, items in predictions_by_image.items()
        }
        report[class_name] = dataset_precision_recall(
            filtered_samples, filtered_predictions, iou_thres
        )
    return report


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} != {b.shape}")
    a, b = a.astype(bool), b.astype(bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def dice_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} != {b.shape}")
    a, b = a.astype(bool), b.astype(bool)
    denominator = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / denominator) if denominator else 0.0


def dataset_mask_metrics(gt_samples, predictions_by_image, iou_thres=0.5):
    """Class-aware instance matching based on mask IoU, ordered by confidence."""
    tp = fp = fn = 0
    matched_ious, matched_dice = [], []
    for sample in gt_samples:
        predictions = sorted(
            predictions_by_image.get(sample.image_id, []), key=lambda item: item.score, reverse=True
        )
        if any(instance.mask is None for instance in sample.instances):
            raise ValueError("mask metrics require masks for every ground-truth instance")
        if any(prediction.mask is None for prediction in predictions):
            raise ValueError("mask metrics require masks for every prediction")
        matched = set()
        for prediction in predictions:
            candidates = [
                (mask_iou(instance.mask, prediction.mask), index)
                for index, instance in enumerate(sample.instances)
                if index not in matched and instance.class_id == prediction.class_id
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_index >= 0 and best_iou >= iou_thres:
                matched.add(best_index)
                tp += 1
                matched_ious.append(best_iou)
                matched_dice.append(dice_coefficient(
                    sample.instances[best_index].mask, prediction.mask
                ))
            else:
                fp += 1
        fn += len(sample.instances) - len(matched)
    precision, recall = precision_recall(tp, fp, fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "match_iou_threshold": iou_thres,
        "mean_mask_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "mean_dice": float(np.mean(matched_dice)) if matched_dice else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }


def _encode_mask(mask, mask_utils):
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(encoded["counts"], bytes):
        encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def compute_coco_map(gt_samples, predictions_by_image, iou_type="bbox", max_detections=1000):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        from pycocotools import mask as mask_utils
    except ImportError as error:
        raise RuntimeError("pycocotools is required for the mAP metric") from error
    category_ids = sorted({instance.class_id for sample in gt_samples for instance in sample.instances})
    coco_gt, annotations, annotation_id = COCO(), [], 1
    for sample in gt_samples:
        for instance in sample.instances:
            x1, y1, x2, y2 = instance.bbox
            annotation = {"id": annotation_id, "image_id": sample.image_id,
                          "category_id": instance.class_id, "bbox": [x1, y1, x2-x1, y2-y1],
                          "area": (x2-x1)*(y2-y1), "iscrowd": 0}
            if iou_type == "segm":
                if instance.mask is None:
                    raise ValueError("segmentation mAP requires masks for every ground-truth instance")
                annotation["segmentation"] = _encode_mask(instance.mask, mask_utils)
                annotation["area"] = float(instance.mask.astype(bool).sum())
            annotations.append(annotation)
            annotation_id += 1
    coco_gt.dataset = {
        "images": [{"id": s.image_id, "width": s.width, "height": s.height,
                    "file_name": s.image_path} for s in gt_samples],
        "annotations": annotations,
        "categories": [{"id": category_id, "name": str(category_id)} for category_id in category_ids],
    }
    coco_gt.createIndex()
    detections = []
    for image_id, predictions in predictions_by_image.items():
        for prediction in predictions:
            x1, y1, x2, y2 = prediction.bbox
            detection = {"image_id": image_id, "category_id": prediction.class_id,
                         "bbox": [x1, y1, x2-x1, y2-y1], "score": prediction.score}
            if iou_type == "segm":
                if prediction.mask is None:
                    raise ValueError("segmentation mAP requires masks for every prediction")
                detection["segmentation"] = _encode_mask(prediction.mask, mask_utils)
            detections.append(detection)
    if not detections:
        return {"mAP_50_95": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}
    evaluator = COCOeval(
        coco_gt,
        coco_gt.loadRes(detections),
        iouType=iou_type)
    evaluator.params.maxDets = [1, 100, max_detections]
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    return {"mAP_50_95": float(evaluator.stats[0]), "mAP_50": float(evaluator.stats[1]),
            "mAP_75": float(evaluator.stats[2])}
