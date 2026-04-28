"""
Singleton model manager that loads and caches YOLO pose models.

Handles MLO (3 keypoints) and CC (1 keypoint) models separately.
Models are loaded lazily on first request and kept in memory.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils.logger import get_logger

logger = get_logger("model_manager")


class ModelManager:
    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.mlo_model: Optional[YOLO] = None
        self.cc_model: Optional[YOLO] = None
        self.mlo_model_path: Optional[str] = None
        self.cc_model_path: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_mlo_model(self, model_path: str) -> bool:
        try:
            path = Path(model_path)
            if not path.exists():
                logger.error("MLO model not found: %s", path)
                return False
            logger.info("Loading MLO model: %s", path)
            self.mlo_model = YOLO(str(path))
            self.mlo_model_path = str(path)
            logger.info("MLO model loaded successfully")
            return True
        except Exception as exc:
            logger.error("Failed to load MLO model: %s", exc)
            return False

    def load_cc_model(self, model_path: str) -> bool:
        try:
            path = Path(model_path)
            if not path.exists():
                logger.error("CC model not found: %s", path)
                return False
            logger.info("Loading CC model: %s", path)
            self.cc_model = YOLO(str(path))
            self.cc_model_path = str(path)
            logger.info("CC model loaded successfully")
            return True
        except Exception as exc:
            logger.error("Failed to load CC model: %s", exc)
            return False

    def predict_keypoints(self, model: YOLO, image: np.ndarray) -> Optional[np.ndarray]:
        """Run inference and return keypoints as (N, 2) array."""
        if model is None:
            return None
        res = model(image, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints.xy) == 0:
            return None
        kp = res.keypoints.xy[0].cpu().numpy()
        return kp if len(kp) > 0 else None

    def predict_mlo(self, image: np.ndarray) -> Optional[np.ndarray]:
        return self.predict_keypoints(self.mlo_model, image)

    def predict_cc(self, image: np.ndarray) -> Optional[np.ndarray]:
        return self.predict_keypoints(self.cc_model, image)

    def load_dicom_as_image(self, dicom_path: str) -> Optional[np.ndarray]:
        """Load a DICOM file and return as BGR image for YOLO inference."""
        try:
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_voi_lut

            ds = pydicom.dcmread(dicom_path)
            arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)

            if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                arr = arr.max() - arr

            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            arr = (arr * 255).astype(np.uint8)

            if len(arr.shape) == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

            return arr
        except Exception as exc:
            logger.error("Failed to load DICOM %s: %s", dicom_path, exc)
            return None
