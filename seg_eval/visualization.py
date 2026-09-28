"""Render segmentation inference results on their source images."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


_PALETTE = (
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
)


def class_color(class_id: int | None) -> tuple[int, int, int]:
    """Return a stable BGR color for a class identifier."""
    return _PALETTE[(class_id or 0) % len(_PALETTE)]


def _label(instance, class_names: Sequence[str] | None) -> str:
    class_id = getattr(instance, "class_id", None)
    if class_id is None:
        name = "object"
    elif class_names is not None and 0 <= class_id < len(class_names):
        name = str(class_names[class_id])
    else:
        name = f"class {class_id}"
    score = getattr(instance, "score", None)
    return f"{name} {score:.2f}" if score is not None else name


def draw_predictions(
    image_bgr: np.ndarray,
    instances: Iterable,
    class_names: Sequence[str] | None = None,
    *,
    mask_alpha: float = 0.40,
    draw_masks: bool = True,
    draw_boxes: bool = True,
    draw_labels: bool = True,
) -> np.ndarray:
    """Return a copy of ``image_bgr`` with prediction instances overlaid.

    Instances may be ``SegmentationInstance`` or any object exposing ``bbox``,
    ``class_id``, ``score``, and optionally ``mask`` attributes.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be a non-empty HxWx3 image")
    if not 0.0 <= mask_alpha <= 1.0:
        raise ValueError("mask_alpha must be between 0 and 1")

    rendered = image_bgr.copy()
    instances = list(instances)
    if draw_masks:
        overlay = rendered.copy()
        has_mask = False
        for instance in instances:
            mask = getattr(instance, "mask", None)
            if mask is None:
                continue
            if mask.shape != rendered.shape[:2]:
                raise ValueError(
                    f"mask shape {mask.shape} does not match image shape {rendered.shape[:2]}"
                )
            overlay[mask.astype(bool)] = class_color(getattr(instance, "class_id", None))
            has_mask = True
        if has_mask:
            rendered = cv2.addWeighted(overlay, mask_alpha, rendered, 1.0 - mask_alpha, 0)

    height, width = rendered.shape[:2]
    thickness = max(1, round(min(height, width) / 400))
    font_scale = max(0.4, min(height, width) / 1000)
    for instance in instances:
        color = class_color(getattr(instance, "class_id", None))
        x1, y1, x2, y2 = (int(round(value)) for value in instance.bbox)
        x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
        if draw_boxes:
            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, thickness)
        if draw_labels:
            text = _label(instance, class_names)
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            label_top = max(0, y1 - text_height - baseline - 4)
            label_right = min(width - 1, x1 + text_width + 6)
            cv2.rectangle(rendered, (x1, label_top), (label_right, y1), color, -1)
            cv2.putText(
                rendered, text, (x1 + 3, max(text_height, y1 - baseline - 2)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness,
                cv2.LINE_AA,
            )
    return rendered


def draw_result(
    image_bgr: np.ndarray,
    result,
    class_names: Sequence[str] | None = None,
    **kwargs,
) -> np.ndarray:
    """Convenience wrapper for a result object with an ``instances`` field."""
    return draw_predictions(image_bgr, result.instances, class_names, **kwargs)


def save_result_image(
    output_path: str | Path,
    image_bgr: np.ndarray,
    result,
    class_names: Sequence[str] | None = None,
    **kwargs,
) -> Path:
    """Render a result and write it to disk, returning the output path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), draw_result(image_bgr, result, class_names, **kwargs)):
        raise OSError(f"could not write visualization: {output_path}")
    return output_path
