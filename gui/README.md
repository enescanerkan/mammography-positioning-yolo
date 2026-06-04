# Mammography Positioning GUI

Desktop application for automated mammography positioning quality assessment using **YOLO pose estimation**.

## Quick Start

### 1. Install dependencies

```bash
# Recommended: use a Python 3.13 virtual environment
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r gui/requirements.txt
```

For GPU acceleration (recommended):

```bash
.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

### 2. Download model weights

> **Note (anonymous review):** Trained weight files are released together with the
> de-anonymized repository upon acceptance. For the review period, place your own
> trained `.pt` checkpoints in `gui/weights/` (filenames below), or train them with
> the parent pipeline using the wavelet config.

Place the following files in `gui/weights/`:

```
gui/weights/mlo-yolo26-pose-advanced.pt
gui/weights/cc-yolo26-pose-advanced.pt
```

### 3. Run the application

```bash
.venv\Scripts\python.exe gui/src/main.py
```

> **Note (Windows):** Import `torch` before `PyQt5` to avoid DLL conflicts. This is already handled in `main.py`.

## Features

### Image Viewer
- **Zoom** -- scroll wheel to zoom in/out on DICOM images
- **Pan** -- left-click drag on empty areas to pan the view
- **Reset** -- right-click or double-click to reset zoom to fit
- Full 640x640 resolution display (no downscaling)

### Analysis
1. **Select DICOM Pair** -- load one MLO and one CC DICOM file
2. **MLO Analysis** -- detects 3 keypoints (nipple, pectoral top/bottom), calculates PNL distance
3. **CC Analysis** -- detects nipple, measures distance to chest wall
4. **Compare** -- evaluates the 10 mm rule and shows positioning quality
5. **Save Results / Save Images** -- export findings

### Draggable Landmarks
After prediction, landmarks can be manually adjusted:

- **Drag** any landmark (Nipple, Pec Top, Pec Bottom) to reposition it
- Distance recalculates **live** as you drag
- Hover over a landmark to see the grab cursor
- Selected landmark shows a highlight halo
- Each landmark has a color-coded label:
  - Nipple (green), Pec Top (red), Pec Bottom (blue)

### Swap Views
Use the swap button between MLO and CC panels to switch image assignments if loaded incorrectly.

## Model Architecture

| View | Model | Keypoints | Output |
|------|-------|-----------|--------|
| MLO | YOLO26 pose | nipple, pectoral_top, pectoral_bottom | PNL distance (mm) |
| CC | YOLO26 pose | nipple | Chest-wall distance (mm) |

Models are trained on VinDr-Mammo dataset with the wavelet preprocessing strategy from the parent training pipeline.

## Weight Files

Place in `gui/weights/`:

| File | Availability |
|------|-------------|
| `mlo-yolo26-pose-advanced.pt` | Released with the de-anonymized repository upon acceptance |
| `cc-yolo26-pose-advanced.pt` | Released with the de-anonymized repository upon acceptance |

## Project Structure

```
gui/
  src/
    main.py                          # Application entry point
    gui/
      main_window.py                 # PyQt5 dark-themed UI
      interactive_canvas.py          # Zoom, pan, draggable landmarks
    analysis/
      mlo_analyzer.py                # MLO PNL distance calculation
      cc_analyzer.py                 # CC chest-wall distance
      comparison_engine.py           # 10mm rule evaluation
    models/model_manager.py          # YOLO pose model loading & inference
    preprocessing/                   # DICOM -> 640x640 pipeline
    data/data_manager.py             # DICOM loading & pixel spacing
    utils/weights_downloader.py      # Google Drive auto-download (gdown)
  weights/                           # .pt model files (gitignored)
  requirements.txt
```
