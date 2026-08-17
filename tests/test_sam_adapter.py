import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.adapters.sam import run_sam
from tests.main import load_registries, parse_args, sam_runtime_options


class _Value:
    def __init__(self, name):
        self.name = name


class _Encoder:
    def __init__(self):
        self.feed = None

    def get_inputs(self):
        return [_Value("image")]

    def get_outputs(self):
        return [_Value("image_embeddings")]

    def run(self, _, feed):
        self.feed = feed
        return [np.ones((1, 256, 64, 64), dtype=np.float32)]


class _Decoder:
    names = (
        "image_embeddings", "point_coords", "point_labels", "mask_input",
        "has_mask_input", "orig_im_size",
    )

    def __init__(self, score=0.95):
        self.score = score
        self.feeds = []

    def get_inputs(self):
        return [_Value(name) for name in self.names]

    def get_outputs(self):
        return [_Value("masks"), _Value("iou_predictions"), _Value("low_res_masks")]

    def run(self, _, feed):
        self.feeds.append(feed)
        height, width = feed["orig_im_size"].astype(int)
        masks = np.full((1, 3, height, width), -1.0, dtype=np.float32)
        masks[0, 1, 2:6, 3:8] = 1.0
        return [masks, np.asarray([[0.1, self.score, 0.2]], dtype=np.float32), masks]


class SamAdapterTests(unittest.TestCase):
    def test_explicit_prompt_produces_instance(self):
        encoder, decoder = _Encoder(), _Decoder()
        image = np.zeros((10, 20, 3), dtype=np.uint8)

        result = run_sam(
            encoder, image, decoder_session=decoder, input_size=100,
            points=[[10, 5]], min_mask_area=1,
        )

        self.assertEqual(len(result.instances), 1)
        self.assertEqual(result.instances[0].bbox, (3.0, 2.0, 8.0, 6.0))
        self.assertEqual(result.instances[0].class_id, 0)
        self.assertEqual(encoder.feed["image"].shape, (1, 3, 100, 100))
        np.testing.assert_allclose(decoder.feeds[0]["point_coords"], [[[50, 25]]])

    def test_low_quality_mask_is_filtered(self):
        result = run_sam(
            _Encoder(), np.zeros((10, 20, 3), dtype=np.uint8),
            decoder_session=_Decoder(score=0.2), points=[[1, 1]], min_mask_area=1,
        )
        self.assertEqual(len(result.instances), 0)

    def test_registry_resolves_decoder_path(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "models.yaml"
            registry.write_text(
                "sam:\n  family: sam\n  path: encoder.onnx\n  decoder_path: decoder.onnx\n",
                encoding="utf-8",
            )
            model = load_registries([registry])["sam"]
            self.assertEqual(model["path"], Path(directory) / "encoder.onnx")
            self.assertEqual(model["decoder_path"], Path(directory) / "decoder.onnx")

    def test_sam_cli_options(self):
        args = parse_args([
            "--model", "sam", "--image", "image.jpg",
            "--sam-point", "10", "20", "--sam-point", "30", "40",
            "--sam-score-threshold", "0.7", "--sam-min-mask-area", "25",
        ])
        self.assertEqual(sam_runtime_options(args, "sam"), {
            "points": [[10.0, 20.0], [30.0, 40.0]],
            "pred_iou_thres": 0.7,
            "min_mask_area": 25,
        })


if __name__ == "__main__":
    unittest.main()
