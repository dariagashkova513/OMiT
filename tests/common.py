from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SegmentationInstance:
    mask: np.ndarray
    bbox: tuple[float, float, float, float]
    score: float
    class_id: int | None
    id: int = -1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bbox": [float(value) for value in self.bbox],
            "score": float(self.score),
            "class_id": self.class_id,
            "mask_shape": list(self.mask.shape),
            "mask_sum": int(self.mask.sum()),
        }


@dataclass
class SegmentationResult:
    instances: list[SegmentationInstance]
    image_shape: tuple[int, int]
    model_name: str = "unknown"

    def __len__(self) -> int:
        return len(self.instances)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "image_shape": list(self.image_shape),
            "instances": [instance.to_dict() for instance in self.instances],
        }


def letterbox(image: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    height, width = image.shape[:2]
    ratio = min(new_shape[0] / height, new_shape[1] / width)
    resized_size = (int(round(width * ratio)), int(round(height * ratio)))
    dw = (new_shape[1] - resized_size[0]) / 2
    dh = (new_shape[0] - resized_size[1]) / 2
    resized = cv2.resize(image, resized_size, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, {"ratio": ratio, "pad": (left, top), "orig_shape": (height, width)}


def unletterbox_boxes(boxes: np.ndarray, params: dict) -> np.ndarray:
    left, top = params["pad"]
    boxes = boxes.copy().astype(np.float32)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / params["ratio"]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / params["ratio"]
    height, width = params["orig_shape"]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes


def crop_proto_mask_to_box(mask, box, params, input_size):
    proto_height, proto_width = mask.shape
    input_height, input_width = input_size
    ratio = params["ratio"]
    left, top = params["pad"]
    x1, y1, x2, y2 = box
    px1 = int(np.clip((x1 * ratio + left) * proto_width / input_width, 0, proto_width - 1))
    py1 = int(np.clip((y1 * ratio + top) * proto_height / input_height, 0, proto_height - 1))
    px2 = int(np.clip(np.ceil((x2 * ratio + left) * proto_width / input_width), px1 + 1, proto_width))
    py2 = int(np.clip(np.ceil((y2 * ratio + top) * proto_height / input_height), py1 + 1, proto_height))
    return mask[py1:py2, px1:px2]


def paste_mask_in_image(mask, box, image_shape, threshold=0.5):
    height, width = image_shape
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)
    canvas = np.zeros((height, width), dtype=np.uint8)
    if x2 <= x1 or y2 <= y1:
        return canvas
    resized = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    canvas[y1:y2, x1:x2] = (resized > threshold).astype(np.uint8)
    return canvas
