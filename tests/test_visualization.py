import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from seg_eval.visualization import draw_predictions, save_result_image
from tests.common import SegmentationInstance, SegmentationResult


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((80, 100, 3), 255, dtype=np.uint8)
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[20:60, 25:75] = 1
        self.instance = SegmentationInstance(mask, (25, 20, 75, 60), 0.91, 0)

    def test_draw_predictions_returns_annotated_copy(self):
        rendered = draw_predictions(self.image, [self.instance], ["brick"])
        self.assertEqual(rendered.shape, self.image.shape)
        self.assertFalse(np.array_equal(rendered, self.image))
        self.assertTrue(np.all(self.image == 255))

    def test_rejects_wrong_mask_shape(self):
        self.instance.mask = np.zeros((10, 10), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "mask shape"):
            draw_predictions(self.image, [self.instance])

    def test_save_result_image(self):
        result = SegmentationResult([self.instance], self.image.shape[:2], "test")
        with tempfile.TemporaryDirectory() as directory:
            path = save_result_image(Path(directory) / "preview.jpg", self.image, result)
            saved = cv2.imread(str(path))
            self.assertIsNotNone(saved)
            self.assertEqual(saved.shape, self.image.shape)


if __name__ == "__main__":
    unittest.main()
