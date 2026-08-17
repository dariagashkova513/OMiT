from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class GTInstance:
    bbox: tuple[float, float, float, float]
    class_id: int
    mask: np.ndarray | None = None


@dataclass
class GTSample:
    image_id: int
    image_path: str
    width: int
    height: int
    instances: list[GTInstance] = field(default_factory=list)


@dataclass
class GTDataset:
    samples: list[GTSample]
    class_names: list[str]

    def __len__(self):
        return len(self.samples)


def load_coco_gt(json_path, images_dir, decode_masks=False):
    images_dir = Path(images_dir)
    with Path(json_path).open(encoding="utf-8") as stream:
        coco = json.load(stream)
    categories = sorted(coco["categories"], key=lambda item: item["id"])
    category_ids = {category["id"]: index for index, category in enumerate(categories)}
    images = {image["id"]: image for image in coco["images"]}
    instances = {image_id: [] for image_id in images}
    for annotation in coco.get("annotations", []):
        x, y, width, height = annotation["bbox"]
        mask = None
        if decode_masks and annotation.get("segmentation"):
            mask = _decode_coco_segmentation(
                annotation["segmentation"],
                images[annotation["image_id"]]["height"],
                images[annotation["image_id"]]["width"],
            )
        instances[annotation["image_id"]].append(GTInstance(
            (x, y, x + width, y + height), category_ids[annotation["category_id"]], mask
        ))
    samples = [
        GTSample(image_id, str(images_dir / image["file_name"]), image["width"],
                 image["height"], instances[image_id])
        for image_id, image in sorted(images.items())
    ]
    return GTDataset(samples, [category["name"] for category in categories])


def _decode_coco_segmentation(segmentation, height, width):
    try:
        from pycocotools import mask as mask_utils
    except ImportError as error:
        raise RuntimeError("pycocotools is required to decode COCO masks") from error
    if isinstance(segmentation, list):
        segmentation = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    return mask_utils.decode(segmentation).astype(np.uint8)


def load_yolo_gt(labels_dir, images_dir, class_names, image_ext=".jpg", is_seg=False):
    labels_dir, images_dir = Path(labels_dir), Path(images_dir)
    samples = []
    image_paths = sorted(images_dir.glob(f"*{image_ext}"))
    if not image_paths:
        raise FileNotFoundError(f"no {image_ext} images found in {images_dir}")
    for image_id, image_path in enumerate(image_paths):
        label_path = labels_dir / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"could not read dataset image: {image_path}")
        height, width = image.shape[:2]
        instances = []
        label_text = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
        for line_number, line in enumerate(label_text.splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            class_id, coords = int(parts[0]), np.asarray(parts[1:], dtype=np.float32)
            if is_seg:
                if len(coords) < 6 or len(coords) % 2:
                    raise ValueError(f"{label_path}:{line_number}: invalid YOLO polygon")
                xs, ys = coords[0::2] * width, coords[1::2] * height
                bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [np.stack([xs, ys], axis=1).round().astype(np.int32)], 1)
            else:
                if len(coords) != 4:
                    raise ValueError(f"{label_path}:{line_number}: expected four box coordinates")
                xc, yc, box_width, box_height = coords * [width, height, width, height]
                bbox = (float(xc - box_width / 2), float(yc - box_height / 2),
                        float(xc + box_width / 2), float(yc + box_height / 2))
                mask = None
            instances.append(GTInstance(bbox, class_id, mask))
        samples.append(GTSample(image_id, str(image_path), width, height, instances))
    return GTDataset(samples, list(class_names))


def load_dataset(config):
    """Load a dataset from a validated, path-resolved YAML configuration."""
    dataset_format = config["format"].lower()
    if dataset_format == "coco":
        return load_coco_gt(config["annotations"], config["images"], config.get("decode_masks", False))
    if dataset_format in {"yolo", "yolo_seg"}:
        return load_yolo_gt(config["labels"], config["images"], config["class_names"],
                            config.get("image_ext", ".jpg"), dataset_format == "yolo_seg")
    raise ValueError(f"unsupported dataset format {dataset_format!r}")
