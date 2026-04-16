# Mammography Positioning GUI

Desktop application for automated mammography positioning quality assessment using **YOLO pose estimation**.

## Quick Start

### 1. Install dependencies

```bash
cd gui
pip install -r requirements.txt
```

### 2. Download model weights

**Option A -- Automatic (on first launch):**

The application will detect missing weights and offer to download them automatically from Google Drive.

**Option B -- Manual download:**

```bash
cd gui
python -c "from src.utils.weights_downloader import WeightsManager; WeightsManager().download_missing_weights(lambda f,p: print(f'{f}: {p*100:.0f}%'))"
```

**Option C -- Using gdown directly:**

```bash
mkdir -p gui/weights
gdown 14OvSuC1XEvs_z5gsdgQ6I-P6JDlKb6l_ -O gui/weights/mlo-yolo26-pose-advanced.pt
gdown 1ZA3CY77hZupi9Nor9S18raPiikhVl5-s -O gui/weights/cc-yolo26-pose-advanced.pt
```

### 3. Run the application

```bash
python gui/src/main.py
```

## Usage

1. **Select DICOM Pair** -- load one MLO and one CC DICOM file
2. **MLO Analysis** -- detects 3 keypoints (nipple, pectoral top/bottom), calculates PNL distance
3. **CC Analysis** -- detects nipple, measures distance to chest wall
4. **Compare** -- evaluates the 10 mm rule and shows positioning quality
5. **Save Results / Save Images** -- export findings

## Model Architecture

| View | Model | Keypoints | Output |
|------|-------|-----------|--------|
| MLO | YOLO26 pose | nipple, pectoral_top, pectoral_bottom | PNL distance (mm) |
| CC | YOLO26 pose | nipple | Chest-wall distance (mm) |

Models are trained on VinDr-Mammo dataset with the wavelet preprocessing strategy from the parent training pipeline.

## Weight Files

Place in `gui/weights/`:

| File | Google Drive |
|------|-------------|
| `mlo-yolo26-pose-advanced.pt` | [Download](https://drive.google.com/file/d/14OvSuC1XEvs_z5gsdgQ6I-P6JDlKb6l_/view) |
| `cc-yolo26-pose-advanced.pt` | [Download](https://drive.google.com/file/d/1ZA3CY77hZupi9Nor9S18raPiikhVl5-s/view) |

## Project Structure

```
gui/
  src/
    main.py                     # Application entry point
    gui/main_window.py          # PyQt5 dark-themed UI
    analysis/
      mlo_analyzer.py           # MLO PNL distance calculation
      cc_analyzer.py            # CC chest-wall distance
      comparison_engine.py      # 10mm rule evaluation
    models/model_manager.py     # YOLO pose model loading & inference
    preprocessing/              # DICOM -> 640x640 pipeline
    data/data_manager.py        # DICOM loading & pixel spacing
    utils/weights_downloader.py # Google Drive auto-download
  weights/                      # .pt model files (gitignored)
  requirements.txt
```
