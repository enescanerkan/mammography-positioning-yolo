"""
Main CLI entry point for the breast positioning YOLO pipeline.

Every experiment is fully described by a single YAML config file in
``configs/``.  No flags need to be changed between experiments — just
swap the config file.

Modes
-----
preprocess   Read raw DICOMs, apply the strategy from config, write
             640×640 PNG images + YOLO pose labels to
             ``data/processed/<strategy>/<view>/``.

augment      Apply horizontal flip augmentation to an already-preprocessed
             dataset split.

train        Build a YOLO dataset.yaml, then launch Ultralytics training
             using hyperparameters from the config.

full         preprocess → train  (end-to-end single command).

Usage examples
--------------
    python main.py --config configs/wavelet_mlo.yaml --mode preprocess
    python main.py --config configs/wavelet_mlo.yaml --mode preprocess --augment
    python main.py --config configs/wavelet_mlo.yaml --mode train
    python main.py --config configs/wavelet_mlo.yaml --mode full
    python main.py --config configs/baseline_mlo.yaml --mode full
    python main.py --config configs/histeq_cc.yaml   --mode preprocess
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``src.*`` imports work when
# the script is invoked from any working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ExperimentConfig
from src.utils.logger import get_logger
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.strategies import get_strategy
from src.training.trainer import YoloTrainer
from src.augmentation.flip_augmenter import FlipAugmenter

logger = get_logger("main")


# ---------------------------------------------------------------------------
# Dataset YAML generator
# ---------------------------------------------------------------------------

def create_dataset_yaml(cfg: ExperimentConfig) -> str:
    """
    Write (or overwrite) the ``dataset.yaml`` required by Ultralytics YOLO.

    The file is placed at ``data/processed/<strategy>/<view>/dataset.yaml``
    so that it sits alongside the ``images/`` and ``labels/`` directories.

    Args:
        cfg: Fully initialised experiment configuration.

    Returns:
        Absolute path string to the written YAML file.
    """
    data = {
        "path": str(cfg.processed_data_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "Breast"},
        "kpt_shape": cfg.kpt_shape,
    }

    yaml_path = cfg.dataset_yaml_path
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False)

    logger.info("Dataset YAML written: %s", yaml_path)
    return str(yaml_path)


# ---------------------------------------------------------------------------
# Pipeline actions
# ---------------------------------------------------------------------------

def run_preprocess(cfg: ExperimentConfig) -> None:
    """
    Execute the full preprocessing pipeline for the configured view and strategy.

    Args:
        cfg: Experiment configuration.
    """
    logger.info(
        "Starting preprocessing | view=%s | strategy=%s", cfg.view, cfg.strategy
    )

    strategy = get_strategy(cfg.strategy)
    pipeline = PreprocessingPipeline(
        data_dir=str(cfg.data_path),
        output_dir=str(cfg.processed_data_dir),
        labels_csv=cfg.label_files,
        strategy=strategy,
    )
    pipeline.run(view_mode=cfg.view)
    logger.info("Preprocessing complete.  Output: %s", cfg.processed_data_dir)


def run_augment(cfg: ExperimentConfig, splits: list[str]) -> None:
    """
    Apply horizontal flip augmentation to the preprocessed dataset.

    Args:
        cfg: Experiment configuration.
        splits: List of split names to augment (e.g. ``['train']``).
    """
    logger.info(
        "Starting flip augmentation | view=%s | splits=%s", cfg.view, splits
    )
    augmenter = FlipAugmenter(str(cfg.processed_data_dir), view_type=cfg.view)

    before = augmenter.get_statistics()
    logger.info("Dataset size before augmentation: %s", before)

    results = augmenter.run(splits=splits)
    logger.info("Augmentation complete: %s", results)

    after = augmenter.get_statistics()
    logger.info("Dataset size after  augmentation: %s", after)


def run_train(cfg: ExperimentConfig) -> None:
    """
    Create the dataset YAML and launch YOLO training.

    Args:
        cfg: Experiment configuration.
    """
    if not cfg.processed_data_dir.exists():
        logger.error(
            "Processed data directory not found: %s\n"
            "Run preprocessing first: "
            "python main.py --config %s --mode preprocess",
            cfg.processed_data_dir,
            "<your_config.yaml>",
        )
        sys.exit(1)

    logger.info(
        "Starting training | view=%s | strategy=%s | model=%s",
        cfg.view, cfg.strategy, cfg.model,
    )

    yaml_path = create_dataset_yaml(cfg)
    trainer = YoloTrainer(model_path=cfg.model)
    trainer.train(
        data_yaml=yaml_path,
        project_dir=str(cfg.experiments_dir),
        **cfg.training_args,
    )
    logger.info(
        "Training finished.  Best checkpoint: %s", cfg.default_trained_model
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Breast Positioning YOLO Pipeline\n\n"
            "All experiment settings are read from the config YAML.\n"
            "To switch between YOLOv8 / v11 / v26, edit the 'model' key\n"
            "inside the corresponding config file.\n\n"
            "Examples:\n"
            "  python main.py --config configs/wavelet_mlo.yaml --mode preprocess\n"
            "  python main.py --config configs/wavelet_mlo.yaml --mode train\n"
            "  python main.py --config configs/baseline_cc.yaml --mode full\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment YAML file (e.g. configs/wavelet_mlo.yaml).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["preprocess", "augment", "train", "full"],
        help=(
            "Pipeline mode:\n"
            "  preprocess – run image preprocessing only\n"
            "  augment    – apply flip augmentation to preprocessed data\n"
            "  train      – launch YOLO training only\n"
            "  full       – preprocess then train (end-to-end)"
        ),
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply flip augmentation immediately after preprocessing (preprocess/full mode).",
    )
    parser.add_argument(
        "--augment_splits",
        type=str,
        default="train",
        metavar="SPLITS",
        help="Comma-separated splits to augment (default: train). E.g. 'train,val'.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    logger.info("Loaded config: %r", cfg)

    splits = [s.strip() for s in args.augment_splits.split(",") if s.strip()]

    if args.mode in ("preprocess", "full"):
        run_preprocess(cfg)
        if args.augment:
            run_augment(cfg, splits)

    if args.mode == "augment":
        run_augment(cfg, splits)

    if args.mode in ("train", "full"):
        run_train(cfg)


if __name__ == "__main__":
    main()
