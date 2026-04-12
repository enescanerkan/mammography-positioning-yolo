"""
Evaluation CLI entry point for the breast positioning pipeline.

Usage
-----
Run evaluation for a single view (MLO **or** CC):

    python evaluate.py --config configs/wavelet_mlo.yaml
    python evaluate.py --config configs/wavelet_cc.yaml
    python evaluate.py --config configs/baseline_mlo.yaml

Run combined MLO + CC evaluation (positioning quality classification):

    python evaluate.py \\
        --mlo_config configs/wavelet_mlo.yaml \\
        --cc_config  configs/wavelet_cc.yaml  \\
        --threshold  10

The ``model`` key inside the YAML controls which YOLO version / checkpoint
is used for inference — nothing needs to be changed in this script.

Output
------
- ``evaluation_results/<strategy>/<view>/results.json``
- ``evaluation_results/<strategy>/combined/confusion_matrix.png``
- ``evaluation_results/<strategy>/combined/distance_correlation.png``
- ``evaluation_results/<strategy>/combined/difference_histogram.png``
- ``evaluation_results/<strategy>/combined/keypoint_errors_boxplot.png``
- ``evaluation_results/<strategy>/combined/paired_viz/pair_NNN_*.png``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow ``python evaluate.py`` to work from any subdirectory
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ExperimentConfig
from src.utils.logger import get_logger
from src.evaluation.evaluator import PoseEvaluator
from src.evaluation.visualizer import (
    CombinedQualityEvaluator,
    CombinedPairVisualiser,
    load_dicom_for_display,
    plot_confusion_matrix,
    plot_difference_histogram,
    plot_distance_correlation,
    plot_keypoint_error_boxplot,
)

import pandas as pd

logger = get_logger("evaluate")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_labels(cfg: ExperimentConfig) -> pd.DataFrame:
    """Load and merge all annotation CSV files defined by the config."""
    dfs = []
    for csv_path in cfg.label_files:
        p = Path(csv_path)
        if p.exists():
            dfs.append(pd.read_csv(p))
        else:
            logger.warning("Label file not found: %s", p)
    if not dfs:
        raise FileNotFoundError("No annotation CSV files found.")
    df = pd.concat(dfs, ignore_index=True)
    if "SOPInstanceUID" in df.columns:
        df["SOPInstanceUID"] = df["SOPInstanceUID"].astype(str)
    return df


def _print_metrics(metrics: dict) -> None:
    """Pretty-print classification metrics to stdout."""
    print("\n" + "=" * 55)
    print("  POSITIONING QUALITY CLASSIFICATION RESULTS")
    print("=" * 55)
    print(f"  Threshold   : {metrics['threshold_mm']:.0f} mm")
    print(f"  Total pairs : {metrics['total_pairs']}")
    print(f"  Correct     : {metrics['correct_predictions']}")
    print(f"  Accuracy    : {metrics['accuracy'] * 100:.2f} %")
    print(f"  Sensitivity : {metrics['sensitivity'] * 100:.2f} %")
    print(f"  Specificity : {metrics['specificity'] * 100:.2f} %")
    print(f"  Precision   : {metrics['precision'] * 100:.2f} %")
    print(f"  F1-Score    : {metrics['f1_score'] * 100:.2f} %")
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix (rows=GT, cols=Pred):")
    print(f"             Bad   Good")
    print(f"    Bad   [{cm[0][0]:5d} {cm[0][1]:5d}]")
    print(f"    Good  [{cm[1][0]:5d} {cm[1][1]:5d}]")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Single-view evaluation
# ---------------------------------------------------------------------------

def run_single(config_path: str) -> list:
    """
    Evaluate a single view (MLO or CC) and return the results list.

    Args:
        config_path: Path to the experiment YAML file.

    Returns:
        List of per-image result dictionaries.
    """
    cfg = ExperimentConfig.from_yaml(config_path)
    logger.info("=== Evaluating %s | strategy=%s ===", cfg.view, cfg.strategy)
    evaluator = PoseEvaluator(cfg)
    results = evaluator.evaluate()
    logger.info("Single-view evaluation complete.  Results: %d records.", len(results))
    return results


# ---------------------------------------------------------------------------
# Combined MLO + CC evaluation
# ---------------------------------------------------------------------------

def run_combined(
    mlo_config_path: str,
    cc_config_path: str,
    threshold_mm: float = 10.0,
    skip_viz: bool = False,
) -> None:
    """
    Run MLO and CC inference, match pairs, classify positioning quality,
    and generate all visualisation artefacts.

    Args:
        mlo_config_path: Path to the MLO experiment YAML.
        cc_config_path:  Path to the CC experiment YAML.
        threshold_mm:    Clinical decision threshold (default 10 mm).
        skip_viz:        If ``True``, skip paired DICOM visualisations
                         (useful when raw DICOMs are unavailable).
    """
    mlo_cfg = ExperimentConfig.from_yaml(mlo_config_path)
    cc_cfg = ExperimentConfig.from_yaml(cc_config_path)

    logger.info(
        "=== Combined evaluation | strategy=%s | threshold=%g mm ===",
        mlo_cfg.strategy, threshold_mm,
    )

    # --- Evaluate each view independently ---
    logger.info("--- Evaluating MLO view ---")
    mlo_evaluator = PoseEvaluator(mlo_cfg)
    mlo_results = mlo_evaluator.evaluate()

    logger.info("--- Evaluating CC view ---")
    cc_evaluator = PoseEvaluator(cc_cfg)
    cc_results = cc_evaluator.evaluate()

    # --- Match pairs and compute positioning quality metrics ---
    combined_eval = CombinedQualityEvaluator(mlo_results, cc_results, threshold_mm)
    combined_results = combined_eval.match_pairs()
    metrics = combined_eval.compute_classification_metrics()

    if metrics:
        _print_metrics(metrics)

    # --- Save combined results JSON ---
    out_dir = PROJECT_ROOT / "evaluation_results" / mlo_cfg.strategy / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_file = out_dir / "combined_results.json"
    with results_file.open("w", encoding="utf-8") as fh:
        # Convert numpy arrays in metrics before serialising
        if metrics:
            metrics_serialisable = {
                k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in metrics.items()
            }
        else:
            metrics_serialisable = {}
        json.dump(
            {"metrics": metrics_serialisable, "pairs": combined_results},
            fh,
            default=str,
            indent=2,
        )
    logger.info("Combined results saved: %s", results_file)

    # --- Generate plots ---
    if metrics is not None:
        import numpy as np
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]),
            out_dir / "confusion_matrix.png",
        )
    plot_distance_correlation(combined_results, out_dir / "distance_correlation.png")
    plot_difference_histogram(
        combined_results,
        out_dir / "difference_histogram.png",
        threshold_mm=threshold_mm,
    )
    plot_keypoint_error_boxplot(combined_results, out_dir / "keypoint_errors_boxplot.png")

    # --- Paired DICOM visualisations ---
    if not skip_viz:
        raw_dir = PROJECT_ROOT / "data" / "raw"
        if raw_dir.exists():
            labels_df = _load_labels(mlo_cfg)
            visualiser = CombinedPairVisualiser(
                combined_results,
                raw_dir,
                out_dir / "paired_viz",
                labels_df,
            )
            visualiser.render_all()
        else:
            logger.warning(
                "Raw DICOM directory not found (%s); skipping paired visualisations.", raw_dir
            )

    logger.info("Combined evaluation complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description=(
            "Breast Positioning YOLO Pose Evaluation\n\n"
            "Single-view:  python evaluate.py --config configs/wavelet_mlo.yaml\n"
            "Combined  :   python evaluate.py "
            "--mlo_config configs/wavelet_mlo.yaml "
            "--cc_config configs/wavelet_cc.yaml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Single-view mode
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a single experiment YAML (MLO or CC).",
    )

    # Combined mode
    parser.add_argument(
        "--mlo_config",
        type=str,
        default=None,
        help="MLO experiment YAML (required for combined evaluation).",
    )
    parser.add_argument(
        "--cc_config",
        type=str,
        default=None,
        help="CC experiment YAML (required for combined evaluation).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Clinical classification threshold in mm (default: 10).",
    )
    parser.add_argument(
        "--skip_viz",
        action="store_true",
        help="Skip paired DICOM visualisations (useful without raw DICOMs).",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.config:
        # --- Single-view mode ---
        run_single(args.config)

    elif args.mlo_config and args.cc_config:
        # --- Combined mode ---
        run_combined(
            mlo_config_path=args.mlo_config,
            cc_config_path=args.cc_config,
            threshold_mm=args.threshold,
            skip_viz=args.skip_viz,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
