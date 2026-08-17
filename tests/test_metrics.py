import unittest

import numpy as np

from seg_eval.annotations import GTInstance, GTSample
from seg_eval.metrics import (
    Prediction, box_iou, dataset_mask_metrics, dataset_precision_recall,
    dice_coefficient, mask_iou, match_predictions,
)


class MetricTests(unittest.TestCase):
    def test_box_iou(self):
        self.assertEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)

    def test_duplicate_prediction_is_false_positive(self):
        ground_truth = [GTInstance((0, 0, 10, 10), 0)]
        predictions = [Prediction((0, 0, 10, 10), 0, 0.9), Prediction((0, 0, 10, 10), 0, 0.8)]
        self.assertEqual(match_predictions(ground_truth, predictions), (1, 1, 0))

    def test_dataset_precision_recall(self):
        samples = [
            GTSample(0, "first.jpg", 100, 100, [GTInstance((0, 0, 10, 10), 0)]),
            GTSample(1, "second.jpg", 100, 100,
                     [GTInstance((0, 0, 10, 10), 0), GTInstance((50, 50, 60, 60), 0)]),
        ]
        predictions = {0: [Prediction((0, 0, 10, 10), 0, 0.9)],
                       1: [Prediction((0, 0, 10, 10), 0, 0.9)]}
        result = dataset_precision_recall(samples, predictions)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 0, 1))
        self.assertEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 2 / 3)

    def test_mask_iou_and_dice_known_overlap(self):
        first = np.zeros((4, 4), dtype=np.uint8)
        second = np.zeros((4, 4), dtype=np.uint8)
        first[0:2, 0:2] = 1
        second[0:2, 1:3] = 1
        self.assertAlmostEqual(mask_iou(first, second), 1 / 3)
        self.assertAlmostEqual(dice_coefficient(first, second), 0.5)

    def test_dataset_mask_metrics_perfect_match(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        samples = [GTSample(0, "image.png", 10, 10, [GTInstance((2, 2, 6, 6), 0, mask)])]
        predictions = {0: [Prediction((2, 2, 6, 6), 0, 0.9, mask.copy())]}
        result = dataset_mask_metrics(samples, predictions)
        self.assertEqual(result["mean_mask_iou"], 1.0)
        self.assertEqual(result["mean_dice"], 1.0)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))
