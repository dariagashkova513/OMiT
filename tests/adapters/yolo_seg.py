import cv2
import numpy as np

from tests.common import (
    SegmentationInstance,
    SegmentationResult,
    crop_proto_mask_to_box,
    letterbox,
    paste_mask_in_image,
    unletterbox_boxes,
)


def run_yolo_seg(session, image_bgr, input_size=(640, 640), conf_thres=0.25, iou_thres=0.45):
    orig_shape = image_bgr.shape[:2]
    padded, params = letterbox(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), input_size)
    blob = padded.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    preds, proto = outputs[0][0].T, outputs[1][0]
    num_classes = preds.shape[1] - 4 - proto.shape[0]
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4 : 4 + num_classes]
    mask_coeffs = preds[:, 4 + num_classes :]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)
    keep = scores > conf_thres
    boxes_xywh, class_ids, scores, mask_coeffs = (
        values[keep] for values in (boxes_xywh, class_ids, scores, mask_coeffs)
    )
    boxes_xyxy = boxes_xywh.copy()
    boxes_xyxy[:, 0] -= boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] -= boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    nms_indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), conf_thres, iou_thres)
    nms_indices = np.asarray(nms_indices).reshape(-1) if len(nms_indices) else []
    instances = []
    proto_flat = proto.reshape(proto.shape[0], -1)
    for instance_id, index in enumerate(nms_indices):
        box = unletterbox_boxes(boxes_xyxy[index][None, :], params)[0]
        mask = (mask_coeffs[index] @ proto_flat).reshape(proto.shape[1:])
        mask = 1 / (1 + np.exp(-np.clip(mask, -60, 60)))
        mask = crop_proto_mask_to_box(mask, box, params, input_size)
        instances.append(SegmentationInstance(
            id=instance_id,
            mask=paste_mask_in_image(mask, box, orig_shape),
            bbox=tuple(box), score=float(scores[index]), class_id=int(class_ids[index]),
        ))
    return SegmentationResult(instances, orig_shape, "yolo_seg")
