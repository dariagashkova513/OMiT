from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from seg_eval.annotations import load_yolo_gt


class AnnotationTests(unittest.TestCase):
    def test_yolo_loader_includes_images_without_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, labels = root / "images", root / "labels"
            images.mkdir(); labels.mkdir()
            cv2.imwrite(str(images / "positive.jpg"), np.zeros((100, 200, 3), dtype=np.uint8))
            cv2.imwrite(str(images / "negative.jpg"), np.zeros((100, 200, 3), dtype=np.uint8))
            (labels / "positive.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")

            dataset = load_yolo_gt(labels, images, ["brick"])

            self.assertEqual(len(dataset), 2)
            by_name = {Path(sample.image_path).name: sample for sample in dataset.samples}
            self.assertEqual(by_name["negative.jpg"].instances, [])
            for actual, expected in zip(
                by_name["positive.jpg"].instances[0].bbox, (80.0, 30.0, 120.0, 70.0)
            ):
                self.assertAlmostEqual(actual, expected, places=5)
