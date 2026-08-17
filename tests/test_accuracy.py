from pathlib import Path
import tempfile
import unittest

from tests.main import load_dataset_registries, load_registries


class RegistryTests(unittest.TestCase):
    def test_load_registry_resolves_relative_model_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "models.yaml"
            registry.write_text(
                "models:\n  demo:\n    path: models/demo.onnx\n    family: yolo_seg\n    input_size: [320, 320]\n",
                encoding="utf-8",
            )

            models = load_registries([registry])

            self.assertEqual(models["demo"]["path"], root / "models" / "demo.onnx")
            self.assertEqual(models["demo"]["input_size"], [320, 320])

    def test_load_registry_rejects_incomplete_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "invalid.yaml"
            registry.write_text("demo:\n  family: yolo_seg\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "requires 'path' and 'family'"):
                load_registries([registry])

    def test_load_dataset_registry_resolves_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "datasets.yaml"
            registry.write_text(
                "demo:\n  format: yolo\n  images: images\n  labels: labels\n  class_names: [brick]\n",
                encoding="utf-8",
            )
            datasets = load_dataset_registries([registry])
            self.assertEqual(datasets["demo"]["images"], root / "images")
            self.assertEqual(datasets["demo"]["labels"], root / "labels")


if __name__ == "__main__":
    unittest.main()
