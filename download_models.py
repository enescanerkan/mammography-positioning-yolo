"""
Download trained YOLO pose models from Google Drive.

This script is executed during Docker build (or manually) to fetch
the MLO and CC model weights into the weights/ directory.

Models:
    - mlo-yolo26-pose-advanced.pt  (MLO view, YOLO26 pose)
    - cc-yolo26-pose-advanced.pt   (CC view, YOLO26 pose)

Usage:
    python download_models.py
    python download_models.py --output_dir weights
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import gdown

MODELS = {
    "mlo-yolo26-pose-advanced.pt": {
        "file_id": "14OvSuC1XEvs_z5gsdgQ6I-P6JDlKb6l_",
        "description": "MLO view - YOLO26 pose (advanced wavelet)",
    },
    "cc-yolo26-pose-advanced.pt": {
        "file_id": "1ZA3CY77hZupi9Nor9S18raPiikhVl5-s",
        "description": "CC view - YOLO26 pose (advanced wavelet)",
    },
}


def download_models(output_dir: str = "weights") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for filename, info in MODELS.items():
        target = out / filename
        if target.exists():
            print(f"[SKIP] {filename} already exists at {target}")
            continue

        url = f"https://drive.google.com/uc?id={info['file_id']}"
        print(f"[DOWNLOAD] {info['description']}")
        print(f"           {url} -> {target}")

        gdown.download(url, str(target), quiet=False)

        if target.exists():
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"[OK] {filename} ({size_mb:.1f} MB)")
        else:
            print(f"[ERROR] Failed to download {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YOLO pose models from Google Drive")
    parser.add_argument("--output_dir", type=str, default="weights", help="Output directory")
    args = parser.parse_args()
    download_models(args.output_dir)
