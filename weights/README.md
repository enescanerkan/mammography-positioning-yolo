# Pre-trained YOLO Model Weights

This directory contains all YOLO pre-trained backbone weights used to
initialise training experiments.

## Selecting a Model

In any `configs/*.yaml` file, change the `model` key to the desired weight:

```yaml
# YOLOv8 — fast, lightweight baseline
model: "weights/yolov8n-pose.pt"

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
| `yolov8n-pose.pt` | YOLOv8 nano | ~6 MB | Good baseline |
| `yolo11n.pt` | YOLOv11 nano | ~5 MB | – |
| `yolo26n-pose.pt` | YOLOv26 nano | ~7 MB | Lightweight v26 |
| `yolo26l-pose.pt` | YOLOv26 large | ~55 MB | **Best accuracy** |

## Trained Models (Advanced Wavelet — Production)

These are the final trained models used by the API for inference:

| File | View | Strategy | Download |
|------|------|----------|----------|
| `mlo-yolo26-pose-advanced.pt` | MLO | wavelet | [Google Drive](https://drive.google.com/file/d/14OvSuC1XEvs_z5gsdgQ6I-P6JDlKb6l_/view) |
| `cc-yolo26-pose-advanced.pt` | CC | wavelet | [Google Drive](https://drive.google.com/file/d/1ZA3CY77hZupi9Nor9S18raPiikhVl5-s/view) |

**Auto-download:**

```bash
python download_models.py
```

This is also run automatically during `docker build`.

## Training Checkpoints

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
- Pre-trained backbones: [Ultralytics GitHub Releases](https://github.com/ultralytics/ultralytics).
- Trained models: [Google Drive folder](https://drive.google.com/drive/folders/1S3AHYaQlAk_e8HSav8dkwEgs1QaMpcom).
