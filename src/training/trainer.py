"""
YOLO training wrapper for the breast positioning pipeline.

``YoloTrainer`` implements :class:`~src.core.interfaces.ITrainer`.
All hyperparameters are sourced exclusively from an
:class:`~src.utils.config.ExperimentConfig` instance, keeping training
reproducible and config-file-driven.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from ..core.interfaces import ITrainer
from ..utils.logger import get_logger

logger = get_logger(__name__)


class YoloTrainer(ITrainer):
    """
    Thin wrapper around the Ultralytics YOLO training API.

    Responsibilities
    ----------------
    - Load a YOLO model from the path specified in the experiment config.
    - Forward all hyperparameter overrides from the config to ``model.train()``.
    - Log training start/end with reproducible parameter snapshots.

    This class intentionally contains no hyperparameter logic — all values
    come from the caller (``ExperimentConfig.training_args``).
    """

    def __init__(self, model_path: str) -> None:
        """
        Load the YOLO model checkpoint.

        Args:
            model_path: Path to a YOLO weight file (``*.pt``), either a
                        pre-trained backbone (e.g. ``weights/yolo26l-pose.pt``)
                        or a previously trained checkpoint.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist on disk.
        """
        resolved = Path(model_path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"YOLO model weight not found: {resolved.resolve()}\n"
                "Make sure the .pt file is present in the weights/ directory."
            )

        logger.info("Loading YOLO model from: %s", resolved)
        self.model = YOLO(str(resolved))

    # ------------------------------------------------------------------
    # ITrainer implementation
    # ------------------------------------------------------------------

    def train(self, data_yaml: str, project_dir: str, **extra_kwargs: Any) -> Any:
        """
        Start YOLO pose-estimation training.

        Args:
            data_yaml: Path to the auto-generated ``dataset.yaml`` file.
            project_dir: Root directory for saving run artefacts
                         (weights, metrics, plots).
            **extra_kwargs: Additional keyword arguments forwarded verbatim
                            to ``model.train()`` (e.g. hyperparameters from
                            ``ExperimentConfig.training_args``).

        Returns:
            Ultralytics training result object.
        """
        train_kwargs = {
            "data": data_yaml,
            "project": project_dir,
            "name": "yolo_run",
            "exist_ok": True,
            **extra_kwargs,
        }

        logger.info("Starting training | model=%s", self.model.ckpt_path)
        logger.info("Training arguments: %s", train_kwargs)

        results = self.model.train(**train_kwargs)

        logger.info("Training complete.")
        return results
