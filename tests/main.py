from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import cv2
import yaml

from tests.adapters import run_maskrcnn, run_sam, run_yolo_seg
from seg_eval.annotations import load_dataset
from seg_eval.metrics import (
    Prediction, compute_coco_map, dataset_mask_metrics, dataset_precision_recall,
    per_class_precision_recall,
)
from seg_eval.visualization import save_result_image

FAMILY_RUNNERS = {"yolo_seg": run_yolo_seg, "maskrcnn": run_maskrcnn, "sam": run_sam}
DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "model_registry.yaml"


def load_registries(paths: list[Path]) -> dict[str, dict]:
    models: dict[str, dict] = {}
    for path in paths:
        registry_path = path.expanduser().resolve()
        with registry_path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        entries = document.get("models", document)
        if not isinstance(entries, dict):
            raise ValueError(f"{registry_path}: expected a YAML mapping")
        for name, raw_config in entries.items():
            if name in models:
                raise ValueError(f"duplicate model {name!r} in {registry_path}")
            if not isinstance(raw_config, dict) or "path" not in raw_config or "family" not in raw_config:
                raise ValueError(f"{registry_path}: model {name!r} requires 'path' and 'family'")
            config = dict(raw_config)
            model_path = Path(config["path"]).expanduser()
            config["path"] = model_path if model_path.is_absolute() else registry_path.parent / model_path
            if "decoder_path" in config:
                decoder_path = Path(config["decoder_path"]).expanduser()
                config["decoder_path"] = (
                    decoder_path if decoder_path.is_absolute() else registry_path.parent / decoder_path
                )
            models[name] = config
    return models


def load_dataset_registries(paths: list[Path]) -> dict[str, dict]:
    datasets = {}
    for path in paths:
        registry_path = path.expanduser().resolve()
        with registry_path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        entries = document.get("datasets", document)
        if not isinstance(entries, dict):
            raise ValueError(f"{registry_path}: expected a YAML mapping")
        for name, raw_config in entries.items():
            if name in datasets:
                raise ValueError(f"duplicate dataset {name!r} in {registry_path}")
            if not isinstance(raw_config, dict) or "format" not in raw_config or "images" not in raw_config:
                raise ValueError(f"{registry_path}: dataset {name!r} requires 'format' and 'images'")
            config = dict(raw_config)
            for key in ("images", "labels", "annotations"):
                if key in config:
                    value = Path(config[key]).expanduser()
                    config[key] = value if value.is_absolute() else registry_path.parent / value
            datasets[name] = config
    return datasets


def create_model_runner(name, models, providers=None, runtime_options=None):
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnxruntime is required to run model inference") from error

    config = models[name]
    family = config["family"]
    if family not in FAMILY_RUNNERS:
        raise ValueError(f"model {name!r} uses unsupported family {family!r}")
    session = ort.InferenceSession(str(config["path"]), providers=providers or ["CPUExecutionProvider"])
    options = dict(config.get("options", {}))
    options.update(runtime_options or {})
    if family == "sam":
        if "decoder_path" not in config:
            raise ValueError(f"SAM model {name!r} requires 'decoder_path'")
        options["decoder_session"] = ort.InferenceSession(
            str(config["decoder_path"]), providers=providers or ["CPUExecutionProvider"]
        )
    if "input_size" in config:
        options["input_size"] = tuple(config["input_size"])
    def inference(image_bgr):
        result = FAMILY_RUNNERS[family](session, image_bgr, **options)
        result.model_name = name
        return result
    return inference


def run_model(name, image_bgr, models, providers=None, runtime_options=None):
    return create_model_runner(name, models, providers, runtime_options)(image_bgr)


def evaluate_dataset(model_name, models, dataset_config, metrics, runtime_options=None,
                     visualization_dir=None, visualization_limit=2):
    dataset_config = dict(dataset_config)
    if "mask" in metrics and dataset_config["format"].lower() == "coco":
        dataset_config["decode_masks"] = True
    ground_truth = load_dataset(dataset_config)
    if not ground_truth.samples:
        raise ValueError("dataset contains no images")
    inference = create_model_runner(model_name, models, runtime_options=runtime_options)
    predictions, inference_times_ms = {}, []
    for index, sample in enumerate(ground_truth.samples, 1):
        image = cv2.imread(sample.image_path)
        if image is None:
            raise FileNotFoundError(f"could not read dataset image: {sample.image_path}")
        started = perf_counter()
        result = inference(image)
        inference_times_ms.append((perf_counter() - started) * 1000)
        predictions[sample.image_id] = [
            Prediction(tuple(instance.bbox), instance.class_id, instance.score, instance.mask)
            for instance in result.instances
        ]
        if visualization_dir is not None and index <= visualization_limit:
            output_path = Path(visualization_dir) / f"{Path(sample.image_path).stem}-predictions.jpg"
            save_result_image(output_path, image, result, ground_truth.class_names)
        print(f"[{index}/{len(ground_truth)}] {Path(sample.image_path).name}", flush=True)
    report = {
        "model": model_name,
        "images": len(ground_truth),
        "classes": ground_truth.class_names,
        "timing_ms": {
            "mean": float(np.mean(inference_times_ms)),
            "median": float(np.median(inference_times_ms)),
            "p95": float(np.percentile(inference_times_ms, 95)),
        },
    }
    if "precision_recall" in metrics:
        report["precision_recall"] = {
            "iou_50": dataset_precision_recall(ground_truth.samples, predictions, 0.50),
            "iou_75": dataset_precision_recall(ground_truth.samples, predictions, 0.75),
            "per_class_iou_50": per_class_precision_recall(
                ground_truth.samples, predictions, ground_truth.class_names, 0.50
            ),
        }
    if "map" in metrics:
        report["map"] = compute_coco_map(ground_truth.samples, predictions, iou_type="bbox")
    if "mask" in metrics:
        report["mask"] = {
            "quality_iou_50": dataset_mask_metrics(ground_truth.samples, predictions, 0.50),
            "quality_iou_75": dataset_mask_metrics(ground_truth.samples, predictions, 0.75),
            "map": compute_coco_map(ground_truth.samples, predictions, iou_type="segm"),
        }
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a registered ONNX segmentation model")
    parser.add_argument("--registry", action="append", type=Path, help="YAML registry; repeat to merge registries")
    parser.add_argument("--model", required=True, help="model name from the YAML registry")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="run inference on one image")
    source.add_argument("--dataset", help="evaluate a dataset from the dataset YAML registry")
    parser.add_argument("--dataset-registry", action="append", type=Path,
                        help="dataset YAML registry; required with --dataset and repeatable")
    parser.add_argument("--metric", action="append", choices=("precision_recall", "map", "mask"),
                        help="dataset metric; repeat to select several (default: precision_recall)")
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    parser.add_argument(
        "--visualize", type=Path,
        help="save an annotated image (single image) or annotated samples (dataset directory)",
    )
    parser.add_argument(
        "--visualize-limit", type=int, default=2,
        help="number of dataset samples to visualize (default: 2)",
    )
    sam = parser.add_argument_group("SAM options")
    sam.add_argument(
        "--sam-point", action="append", nargs=2, type=float, metavar=("X", "Y"),
        help="positive point prompt in original-image pixels; repeat for multiple prompts",
    )
    sam.add_argument(
        "--sam-points-per-side", type=int,
        help="automatic prompt-grid density (overrides the model registry)",
    )
    sam.add_argument(
        "--sam-score-threshold", type=float,
        help="minimum predicted mask IoU (overrides pred_iou_thres)",
    )
    sam.add_argument(
        "--sam-min-mask-area", type=int,
        help="minimum mask area in pixels",
    )
    return parser.parse_args(argv)


def sam_runtime_options(args, family):
    options = {}
    if args.sam_point is not None:
        options["points"] = args.sam_point
    if args.sam_points_per_side is not None:
        options["points_per_side"] = args.sam_points_per_side
    if args.sam_score_threshold is not None:
        options["pred_iou_thres"] = args.sam_score_threshold
    if args.sam_min_mask_area is not None:
        options["min_mask_area"] = args.sam_min_mask_area
    if options and family != "sam":
        raise SystemExit("--sam-* options can only be used with a model whose family is 'sam'")
    return options


def main(argv=None):
    args = parse_args(argv)
    if args.visualize_limit < 1:
        raise SystemExit("--visualize-limit must be at least 1")
    models = load_registries(args.registry or [DEFAULT_REGISTRY])
    if args.model not in models:
        raise SystemExit(f"unknown model {args.model!r}; choose from: {', '.join(sorted(models))}")
    runtime_options = sam_runtime_options(args, models[args.model]["family"])
    if args.dataset:
        if not args.dataset_registry:
            raise SystemExit("--dataset-registry is required with --dataset")
        datasets = load_dataset_registries(args.dataset_registry)
        if args.dataset not in datasets:
            raise SystemExit(f"unknown dataset {args.dataset!r}; choose from: {', '.join(sorted(datasets))}")
        report = evaluate_dataset(
            args.model, models, datasets[args.dataset],
            args.metric or ["precision_recall"], runtime_options,
            args.visualize, args.visualize_limit,
        )
    else:
        image = cv2.imread(str(args.image))
        if image is None:
            raise SystemExit(f"could not read image: {args.image}")
        result = run_model(args.model, image, models, runtime_options=runtime_options)
        if args.visualize:
            save_result_image(args.visualize, image, result)
        report = result.to_dict()
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
