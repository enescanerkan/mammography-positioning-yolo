# Pre-trained YOLO Model Weights

This directory contains all YOLO pre-trained backbone weights used to
initialise training experiments.

## Selecting a Model

In any `configs/*.yaml` file, change the `model` key to the desired weight:

```yaml
# YOLOv8 large
model: "weights/yolov8l-pose.pt"

# YOLOv11
model: "weights/yolo11n.pt"

# YOLOv26 nano (faster, smaller)
model: "weights/yolo26n-pose.pt"

# YOLOv26 large (best accuracy — recommended)
model: "weights/yolo26l-pose.pt"
```

## Available Weights

| File | Version | Size | Notes |
|------|---------|------|-------|
| `yolov8l-pose.pt` | YOLOv8 large | – | – |
| `yolo11n.pt` | YOLOv11 nano | ~5 MB | – |
| `yolo26n-pose.pt` | YOLOv26 nano | ~7 MB | Lightweight v26 |
| `yolo26l-pose.pt` | YOLOv26 large | ~55 MB | **Best accuracy** |

## Trained Checkpoints

After training, Ultralytics saves the best checkpoint to:

```
experiments/runs/<view>/yolo_run/weights/best.pt
```

To use a trained checkpoint for evaluation, set `trained_model_path` in
your config YAML:

```yaml
trained_model_path: "experiments/runs/MLO/yolo_run/weights/best.pt"
```

## Notes

- These files are excluded from Git via `.gitignore` (binaries > 5 MB).
- Download from [Ultralytics GitHub Releases](https://github.com/ultralytics/ultralytics).
