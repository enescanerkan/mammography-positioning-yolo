"""
Core abstract interfaces for the breast positioning pipeline.

Each interface follows the Interface Segregation Principle (ISP) — no class
is forced to implement methods it does not use.  Concrete implementations
live in their respective subpackages (preprocessing, training, evaluation).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class IEnhancementStrategy(ABC):
    """
    Strategy interface for pixel-level image enhancement.

    Implementations must be pure functions with respect to image geometry:
    they receive an 8-bit single-channel image and return an 8-bit
    single-channel image of the **exact same spatial dimensions**.
    Coordinate transforms are therefore completely unaffected.
    """

    @abstractmethod
    def run(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the enhancement to a single grayscale image.

        Args:
            image: 8-bit uint8 NumPy array with shape (H, W).

        Returns:
            Enhanced 8-bit uint8 array with the same shape (H, W).
        """


class IDataLoader(ABC):
    """Interface for loading raw imaging data from a source directory."""

    @abstractmethod
    def load(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load a single file and return the pixel array plus metadata.

        Args:
            file_path: Absolute path to the file to load.

        Returns:
            Tuple of:
                - pixel_array: Float32 NumPy array (H, W).
                - metadata: Dictionary of extracted header fields.
        """


class IPreprocessingPipeline(ABC):
    """Interface for end-to-end preprocessing pipelines."""

    @abstractmethod
    def run(self, view_mode: str) -> None:
        """
        Execute the full preprocessing pipeline for the specified view.

        Args:
            view_mode: Either ``'MLO'`` or ``'CC'``.
        """


class ITrainer(ABC):
    """Interface for model training wrappers."""

    @abstractmethod
    def train(self, data_yaml: str, project_dir: str) -> Any:
        """
        Launch training and return the framework result object.

        Args:
            data_yaml: Absolute path to the YOLO dataset YAML file.
            project_dir: Root directory where run artefacts are saved.

        Returns:
            Framework-specific result object (e.g., Ultralytics Results).
        """


class IEvaluator(ABC):
    """Interface for post-training evaluation modules."""

    @abstractmethod
    def evaluate(self) -> List[Dict[str, Any]]:
        """
        Run inference on the test split and compute per-image metrics.

        Returns:
            List of result dictionaries — one entry per processed image.
        """


class IAugmenter(ABC):
    """Interface for dataset augmentation strategies."""

    @abstractmethod
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply an image-level augmentation.

        Args:
            image: Input NumPy image array (H, W) or (H, W, C).

        Returns:
            Augmented NumPy array with the same dtype.
        """

    @abstractmethod
    def augment_label(self, label_line: str, image_width: int) -> str:
        """
        Transform a YOLO-format label string to match the augmented image.

        Args:
            label_line: YOLO pose label string.
            image_width: Pixel width of the (original) image.

        Returns:
            Transformed YOLO format label string.
        """
