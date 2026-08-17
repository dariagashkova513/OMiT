"""ONNX adapter for the original Segment Anything Model (SAM).

The official ONNX export is split into an image encoder and a mask decoder.
This adapter runs the encoder once and uses a grid of positive point prompts to
turn promptable SAM into the instance-like output expected by the evaluator.
"""

from __future__ import annotations

import cv2
import numpy as np

from tests.common import SegmentationInstance, SegmentationResult


_PIXEL_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
_PIXEL_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def _encoder_blob(image_bgr, image_size):
    height, width = image_bgr.shape[:2]
    scale = image_size / max(height, width)
    resized_h, resized_w = int(height * scale + 0.5), int(width * scale + 0.5)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    normalized = (rgb.astype(np.float32) - _PIXEL_MEAN) / _PIXEL_STD
    padded = np.zeros((image_size, image_size, 3), dtype=np.float32)
    padded[:resized_h, :resized_w] = normalized
    return padded.transpose(2, 0, 1)[None], scale


def _named_output(session, outputs, preferred_names):
    names = [output.name for output in session.get_outputs()]
    for preferred in preferred_names:
        if preferred in names:
            return outputs[names.index(preferred)]
    return outputs[0]


def _bbox_from_mask(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _box_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-8)


def _nms(candidates, iou_threshold):
    order = np.argsort([-candidate[0] for candidate in candidates])
    kept = []
    while len(order):
        current = int(order[0])
        kept.append(candidates[current])
        if len(order) == 1:
            break
        rest = order[1:]
        boxes = np.stack([candidates[int(index)][1] for index in rest])
        order = rest[_box_iou(candidates[current][1], boxes) <= iou_threshold]
    return kept


def run_sam(
    encoder_session,
    image_bgr,
    *,
    decoder_session,
    input_size=1024,
    points_per_side=8,
    pred_iou_thres=0.88,
    mask_threshold=0.0,
    min_mask_area=100,
    nms_iou_thres=0.7,
    points=None,
):
    """Run split SAM ONNX models and return class-agnostic instances.

    ``points`` may contain explicit ``[x, y]`` image coordinates. When omitted,
    an evenly spaced grid is used for automatic mask generation.
    """
    if decoder_session is None:
        raise ValueError("SAM requires a decoder_session")
    orig_h, orig_w = image_bgr.shape[:2]
    blob, scale = _encoder_blob(image_bgr, int(input_size))
    encoder_outputs = encoder_session.run(
        None, {encoder_session.get_inputs()[0].name: blob.astype(np.float32)}
    )
    embedding = _named_output(
        encoder_session, encoder_outputs, ("image_embeddings", "image_embedding", "embeddings")
    )

    if points is None:
        count = int(points_per_side)
        if count < 1:
            raise ValueError("points_per_side must be at least 1")
        xs = (np.arange(count, dtype=np.float32) + 0.5) * orig_w / count
        ys = (np.arange(count, dtype=np.float32) + 0.5) * orig_h / count
        point_list = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    else:
        point_list = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    input_names = {value.name for value in decoder_session.get_inputs()}
    candidates = []
    for point in point_list:
        feed = {
            "image_embeddings": embedding,
            "point_coords": (point * scale).reshape(1, 1, 2).astype(np.float32),
            "point_labels": np.ones((1, 1), dtype=np.float32),
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros((1,), dtype=np.float32),
            "orig_im_size": np.asarray([orig_h, orig_w], dtype=np.float32),
        }
        missing = input_names - feed.keys()
        if missing:
            raise ValueError(f"unsupported SAM decoder inputs: {sorted(missing)}")
        outputs = decoder_session.run(None, {name: feed[name] for name in input_names})
        output_names = [value.name for value in decoder_session.get_outputs()]
        masks = _named_output(decoder_session, outputs, ("masks", "low_res_masks"))
        scores = None
        for name in ("iou_predictions", "iou_scores", "scores"):
            if name in output_names:
                scores = outputs[output_names.index(name)]
                break
        if scores is None:
            scores = outputs[1]
        masks = np.asarray(masks)[0]
        scores = np.asarray(scores).reshape(-1)
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score < pred_iou_thres:
            continue
        mask = masks[best]
        if mask.shape != (orig_h, orig_w):
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary = (mask > mask_threshold).astype(np.uint8)
        if int(binary.sum()) < min_mask_area:
            continue
        box = _bbox_from_mask(binary)
        if box is not None:
            candidates.append((score, box, binary))

    instances = [
        SegmentationInstance(mask=mask, bbox=tuple(box), score=score, class_id=1, id=index)
        for index, (score, box, mask) in enumerate(_nms(candidates, nms_iou_thres))
    ]
    return SegmentationResult(instances, (orig_h, orig_w), "sam")
