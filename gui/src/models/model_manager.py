"""
Model manager for loading and managing MLO and CC YOLO pose models.
"""

import os
import sys
import numpy as np
import cv2
import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from ultralytics import YOLO


class IModelManager(ABC):
    """Interface for model managers."""

    @abstractmethod
    def load_models(self) -> None:
        pass

    @abstractmethod
    def predict_landmarks(self, image: np.ndarray, model_type: str) -> Optional[np.ndarray]:
        pass


class ModelManager(IModelManager):
    """Manager for loading and using MLO and CC YOLO pose landmark detection models."""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = {}

    def load_models(self) -> None:
        self._load_mlo_model()
        self._load_cc_model()

    def _load_mlo_model(self) -> None:
        model_path = self._get_model_path("mlo-yolo26-pose-advanced.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MLO model not found: {model_path}")
        self.models['mlo'] = YOLO(model_path)

    def _load_cc_model(self) -> None:
        model_path = self._get_model_path("cc-yolo26-pose-advanced.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CC model not found: {model_path}")
        self.models['cc'] = YOLO(model_path)

    def _get_model_path(self, model_filename: str) -> str:
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.dirname(os.path.dirname(current_dir))
        return os.path.join(base_path, "weights", model_filename)

    def predict_landmarks(self, image: np.ndarray, model_type: str,
                         original_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """Predict landmarks using YOLO pose model.

        Args:
            image: Input image array (640x640 preprocessed, float32 0-1)
            model_type: 'mlo' or 'cc'
            original_shape: Not used, kept for compatibility

        Returns:
            Predicted landmarks as (N, 2) numpy array in 640x640 pixel space.
            MLO: [[nipple_x, nipple_y], [pec_top_x, pec_top_y], [pec_bottom_x, pec_bottom_y]]
            CC:  [[nipple_x, nipple_y]]
        """
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not available")

        try:
            model = self.models[model_type]

            if len(image.shape) == 3:
                image = image[0]

            img = image.copy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            results = model(img, verbose=False)[0]

            if results.keypoints is None or len(results.keypoints.xy) == 0:
                return None

            kp = results.keypoints.xy[0].cpu().numpy()
            return kp if len(kp) > 0 else None

        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def is_model_loaded(self, model_type: str) -> bool:
        return model_type in self.models

    def get_available_models(self) -> list:
        return list(self.models.keys())
