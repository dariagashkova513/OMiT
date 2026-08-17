import cv2
import numpy as np

from tests.common import SegmentationInstance, SegmentationResult, paste_mask_in_image


def run_maskrcnn(session, image_bgr, score_thres=0.5):
    orig_h, orig_w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (800, 800), interpolation=cv2.INTER_LINEAR)
    input_h, input_w = image_rgb.shape[:2]

    blob = image_rgb.astype(np.float32).transpose(2, 0, 1)
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    boxes, labels, masks, scores, _ = outputs

    keep = scores > score_thres
    instances = []

    for instance_id, (box, label, score, mask) in enumerate(
        zip(boxes[keep], labels[keep], scores[keep], masks[keep])
    ):
        box = box.astype(np.float32).copy()
        box[[0, 2]] *= orig_w / input_w
        box[[1, 3]] *= orig_h / input_h

        # Keep boxes inside the original image.
        box[[0, 2]] = np.clip(box[[0, 2]], 0, orig_w)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, orig_h)

        soft_mask = mask[0] if mask.ndim == 3 else mask

        instances.append(
            SegmentationInstance(
                id=instance_id,
                mask=paste_mask_in_image(soft_mask, box, (orig_h, orig_w)),
                bbox=tuple(box),
                score=float(score),
                class_id=int(label),
            )
        )

    return SegmentationResult(instances, (orig_h, orig_w), "maskrcnn")