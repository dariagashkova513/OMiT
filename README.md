# ONNX model evaluation suite

The runner supports single-image inference and repeatable dataset evaluation for
COCO, YOLO detection, and YOLO segmentation datasets. Model and dataset paths
live in YAML so an experiment can be reproduced without editing Python code.

## Setup

```bash
python -m pip install -r requirements.txt
```

For COCO mAP, install the optional dependency set instead:

```bash
python -m pip install -r requirements-map.txt
```

## Configure a dataset

Adjust `dataset_registry.yaml`
paths. Relative paths are resolved relative to the YAML file. Supported formats
are `coco`, `yolo`, and `yolo_seg`. Images without YOLO label files are included
as valid negative samples.

## Evaluate a model

```bash
python -m tests.main \
  --model yolov8l-seg-angl \
  --dataset bricks-yolo \
  --dataset-registry dataset_registry.yaml \
  --metric precision_recall \
  --metric map \
  --metric mask \
  --output results/yolov8l-bricks.json
```

The JSON report contains aggregate precision, recall, F1, TP/FP/FN at IoU 0.50
and 0.75, per-class results at IoU 0.50, box mAP, mask IoU, Dice, true COCO
segmentation mAP (when selected), and mean, median, and p95 inference time.
Mask evaluation requires `format: yolo_seg`, or COCO annotations containing
segmentation data. The ONNX session is created once per evaluation,
outside the timed per-image loop.

SAM's split ONNX export is also supported. Register the image encoder as
`path`, the prompt decoder as `decoder_path`, and use `family: sam`. By default
the adapter samples an 8x8 point grid and returns class-agnostic masks (class
0); thresholds and grid density can be changed under `options`:

```yaml
sam-vit-b:
  path: models/sam_vit_b_encoder.onnx
  decoder_path: models/sam_vit_b_decoder.onnx
  family: sam
  options:
    points_per_side: 8
    pred_iou_thres: 0.88
    min_mask_area: 100
```

For a quick inference without ground truth:

```bash
python -m tests.main --model yolov8l-seg-angl --image data/example.jpg
```

Run SAM with its automatic prompt grid:

```bash
python -m tests.main --model sam-vit-b --image data/1.jpg \
  --sam-points-per-side 8 --sam-score-threshold 0.88
```

Or test one or more explicit positive point prompts (image pixel coordinates):

```bash
python -m tests.main --model sam-vit-b --image data/1.jpg \
  --sam-point 320 240 --sam-point 500 300
```

Run the dependency-light unit tests with:

```bash
python -m unittest discover -s tests -v
```
