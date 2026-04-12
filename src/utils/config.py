"""
Centralised configuration resolvers for the pipeline.

``ExperimentConfig`` is the single source of truth for an experiment.
It is loaded from a YAML config file and exposes all derived paths so
that every module can be initialised without duplicating path logic.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Legacy constants (kept for backward-compat with any existing imports)
# ---------------------------------------------------------------------------

DATA_DIR: str = "data/raw"
OUTPUT_DIR: str = "experiments/runs"
TARGET_SIZE: tuple = (640, 640)
LOG_LEVEL: int = logging.INFO


class Config:
    """
    Legacy static configuration class.

    Prefer ``ExperimentConfig`` for all new code.  This class is retained
    so that existing modules that import ``Config`` continue to work
    without modification.
    """

    DATA_DIR: str = DATA_DIR
    OUTPUT_DIR: str = OUTPUT_DIR
    LOG_LEVEL: int = LOG_LEVEL
    TARGET_SIZE: tuple = TARGET_SIZE
    YOLO_MODEL: str = "weights/yolo11n.pt"
    EPOCHS: int = 50

    @classmethod
    def setup_dirs(cls) -> None:
        """Create the default output directory if it does not exist."""
        Path(cls.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


class ExperimentConfig:
    """
    Typed configuration object built from a YAML experiment file.

    Every experiment lives in ``configs/<strategy>_<view>.yaml``.
    All paths are resolved relative to the project root so that the
    pipeline works regardless of the current working directory.

    Example
    -------
    >>> cfg = ExperimentConfig.from_yaml("configs/wavelet_mlo.yaml")
    >>> print(cfg.processed_data_dir)
    PosixPath('.../data/processed/wavelet/MLO')
    """

    def __init__(self, raw: Dict[str, Any], project_root: Path) -> None:
        self._raw = raw
        self._root = project_root

        # --- Experiment identity ---
        self.strategy: str = raw["strategy"]          # baseline | histeq | wavelet
        self.view: str = raw["view"].upper()           # MLO | CC

        # --- Model weight ---
        self.model: str = raw["model"]                 # e.g. weights/yolo26l-pose.pt

        # --- Raw data path ---
        self.data_path: Path = project_root / raw.get("data_path", "data/raw")

        # --- Training hyperparameters (forwarded verbatim to Ultralytics) ---
        self.device: Any = raw.get("device", 0)
        self.workers: int = int(raw.get("workers", 8))
        self.batch: int = int(raw.get("batch", 16))
        self.epochs: int = int(raw.get("epochs", 100))
        self.patience: int = int(raw.get("patience", 30))
        self.imgsz: int = int(raw.get("imgsz", 640))
        self.optimizer: str = raw.get("optimizer", "auto")

        # --- Optional override for the trained model checkpoint ---
        _trained: str = raw.get("trained_model_path", "")
        self.trained_model_path: Optional[Path] = (
            project_root / _trained if _trained else None
        )

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------

    @property
    def processed_data_dir(self) -> Path:
        """Root of the preprocessed dataset for this strategy + view."""
        return self._root / "data" / "processed" / self.strategy / self.view

    @property
    def dataset_yaml_path(self) -> Path:
        """YOLO dataset YAML auto-generated inside ``processed_data_dir``."""
        return self.processed_data_dir / "dataset.yaml"

    @property
    def experiments_dir(self) -> Path:
        """Directory where YOLO training artefacts (weights, plots) are saved."""
        return self._root / "experiments" / "runs" / self.view

    @property
    def default_trained_model(self) -> Path:
        """
        Resolved path to the best trained checkpoint.

        Precedence:
          1. ``trained_model_path`` from YAML (if explicitly set).
          2. Auto-resolved path inside ``experiments_dir``.
        """
        if self.trained_model_path and self.trained_model_path.exists():
            return self.trained_model_path
        return self.experiments_dir / "yolo_run" / "weights" / "best.pt"

    @property
    def evaluation_output_dir(self) -> Path:
        """Directory where evaluation results (JSON, plots) are written."""
        return self._root / "evaluation_results" / self.strategy / self.view

    @property
    def label_files(self) -> List[str]:
        """Absolute paths to both annotation CSV files."""
        labels = self._root / "data" / "labels"
        return [str(labels / "mlo_labels.csv"), str(labels / "cc_labels.csv")]

    @property
    def kpt_shape(self) -> List[int]:
        """
        YOLO pose keypoint shape descriptor.

        - MLO → [3, 3]: 3 keypoints (nipple, pectoral_top, pectoral_bottom),
                         each with (x, y, visibility).
        - CC  → [1, 3]: 1 keypoint (nipple) with (x, y, visibility).
        """
        return [3, 3] if self.view == "MLO" else [1, 3]

    @property
    def training_args(self) -> Dict[str, Any]:
        """Dictionary of Ultralytics ``model.train()`` keyword arguments."""
        return {
            "device": self.device,
            "workers": self.workers,
            "batch": self.batch,
            "epochs": self.epochs,
            "patience": self.patience,
            "imgsz": self.imgsz,
            "optimizer": self.optimizer,
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, config_path: str, project_root: Optional[Path] = None) -> "ExperimentConfig":
        """
        Build an ``ExperimentConfig`` from a YAML file path.

        Args:
            config_path: Path to the YAML file (absolute or relative to CWD).
            project_root: Override project root.  Defaults to the parent of
                          the directory containing the YAML file.

        Returns:
            Fully initialised ``ExperimentConfig`` instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            KeyError: If required keys are missing from the YAML.
        """
        path = Path(config_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        if project_root is None:
            # The config lives in  <project_root>/configs/<name>.yaml
            project_root = path.parent.parent

        return cls(raw, project_root)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ExperimentConfig("
            f"strategy={self.strategy!r}, "
            f"view={self.view!r}, "
            f"model={self.model!r})"
        )
