"""
Visualisation helpers for the breast positioning evaluation pipeline.

Generates publication-quality plots from the results produced by
:class:`~src.evaluation.evaluator.PoseEvaluator`.  All plotting is
done with Matplotlib (non-interactive ``Agg`` backend).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ast
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import seaborn as sns
from pydicom.pixel_data_handlers.util import apply_voi_lut
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from ..utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# DICOM rendering helpers
# ---------------------------------------------------------------------------

def load_dicom_for_display(dicom_path: Path) -> np.ndarray:
    """
    Load a DICOM file and return a float32 image normalised to [0, 1].

    MONOCHROME1 images are inverted so that tissue appears bright on a
    dark background — consistent with the MONOCHROME2 convention used
    during training.

    Args:
        dicom_path: Absolute path to the DICOM file.

    Returns:
        Float32 NumPy array (H, W) in [0, 1].
    """
    ds = pydicom.dcmread(str(dicom_path))
    pixel_array = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array

    img_min, img_max = pixel_array.min(), pixel_array.max()
    pixel_array = (pixel_array - img_min) / (img_max - img_min + 1e-8)
    return pixel_array


def find_dicom_path(
    sop_uid: str,
    labels_df: pd.DataFrame,
    raw_dir: Path,
) -> Optional[Path]:
    """
    Locate a DICOM file by SOPInstanceUID.

    Tries the VinDr-Mammo directory structure
    ``<raw_dir>/<study_uid>/<sop_uid>.dicom`` first, then falls back to a
    flat scan of all sub-directories.

    Args:
        sop_uid: SOPInstanceUID of the image to find.
        labels_df: Label DataFrame that maps SOPInstanceUID to StudyInstanceUID.
        raw_dir: Root directory containing raw DICOM files.

    Returns:
        :class:`pathlib.Path` if found, ``None`` otherwise.
    """
    row = labels_df[labels_df["SOPInstanceUID"] == sop_uid]
    if not row.empty and "StudyInstanceUID" in row.columns:
        study_uid = str(row.iloc[0]["StudyInstanceUID"])
        candidate = raw_dir / study_uid / f"{sop_uid}.dicom"
        if candidate.exists():
            return candidate

    for study_dir in raw_dir.iterdir():
        if not study_dir.is_dir():
            continue
        candidate = study_dir / f"{sop_uid}.dicom"
        if candidate.exists():
            return candidate

    return None


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    title: str = "Quality Classification — Confusion Matrix",
) -> None:
    """
    Save a seaborn-styled confusion matrix figure.

    Args:
        cm: 2×2 confusion matrix (rows = GT, columns = Predicted).
        output_path: Target file path for the PNG.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Bad", "Good"],
        yticklabels=["Bad", "Good"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Ground Truth Label", fontsize=12)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", output_path)


def plot_distance_correlation(
    combined_results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Scatter plot of predicted MLO PNL distances vs CC chest-wall distances.

    Correctly classified pairs are shown in blue; misclassified in red.
    The identity line ``y = x`` is drawn for reference.

    Args:
        combined_results: Matched MLO-CC pair dicts from the combined evaluator.
        output_path: Target file path for the PNG.
    """
    if not combined_results:
        return

    mlo_d = [r["mlo_distance_mm"] for r in combined_results]
    cc_d = [r["cc_distance_mm"] for r in combined_results]
    colors = ["#e74c3c" if not r["correct_prediction"] else "#2980b9"
              for r in combined_results]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(mlo_d, cc_d, c=colors, alpha=0.65, edgecolors="white", linewidths=0.4)

    combined = mlo_d + cc_d
    lim_lo, lim_hi = min(combined) - 5, max(combined) + 5
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.45, label="y = x")

    ax.set_xlabel("MLO PNL Distance (mm)", fontsize=12)
    ax.set_ylabel("CC Chest-Wall Distance (mm)", fontsize=12)
    ax.set_title("MLO vs CC Distance Correlation", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Distance correlation plot saved: %s", output_path)


def plot_difference_histogram(
    combined_results: List[Dict[str, Any]],
    output_path: Path,
    threshold_mm: float = 10.0,
) -> None:
    """
    Histogram of |MLO PNL − CC chest-wall distance| with a threshold line.

    Args:
        combined_results: Matched MLO-CC pair dicts.
        output_path: Target file path for the PNG.
        threshold_mm: Classification threshold (vertical line).
    """
    if not combined_results:
        return

    diffs = [r["distance_diff_mm"] for r in combined_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(diffs, bins=30, alpha=0.75, color="#3498db", edgecolor="white")
    ax.axvline(
        x=threshold_mm,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.8,
        label=f"Threshold ({threshold_mm:.0f} mm)",
    )
    ax.set_xlabel("|MLO PNL − CC Chest-Wall| (mm)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distance Difference Histogram", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Difference histogram saved: %s", output_path)


def plot_keypoint_error_boxplot(
    combined_results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Box plot of per-keypoint localisation errors (mm) across all test pairs.

    Args:
        combined_results: Matched MLO-CC pair dicts.
        output_path: Target file path for the PNG.
    """
    error_fields = [
        ("MLO\nNipple",  "mlo_nipple_error_mm"),
        ("MLO\nPec Top", "mlo_pec1_error_mm"),
        ("MLO\nPec Bot", "mlo_pec2_error_mm"),
        ("MLO\nPNL",     "mlo_pnl_error_mm"),
        ("CC\nNipple",   "cc_nipple_error_mm"),
        ("CC\nCW Dist",  "cc_cw_error_mm"),
    ]

    box_data, tick_labels = [], []
    for label, key in error_fields:
        vals = [r[key] for r in combined_results if r.get(key) is not None]
        if vals:
            box_data.append(vals)
            tick_labels.append(label)

    if not box_data:
        logger.warning("No keypoint error data available for boxplot.")
        return

    palette = ["#FF9999", "#FF6666", "#CC3333", "#FF4444", "#66B2FF", "#3399FF"]
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(box_data, tick_labels=tick_labels, patch_artist=True, widths=0.55)
    for patch, colour in zip(bp["boxes"], palette[: len(box_data)]):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)

    ax.set_ylabel("Error (mm)", fontsize=12)
    ax.set_title("Per-Keypoint Localisation Errors", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Keypoint error boxplot saved: %s", output_path)


# ---------------------------------------------------------------------------
# Combined paired DICOM visualisation
# ---------------------------------------------------------------------------

class CombinedPairVisualiser:
    """
    Renders side-by-side MLO/CC visualisations on the original DICOM images.

    Each figure shows ground-truth keypoints (red tones) and predicted
    keypoints (green tones) overlaid on the full-resolution DICOM, with
    a colour-coded title indicating whether the positioning was correctly
    classified.

    Args:
        combined_results: List of dictionaries from :class:`CombinedQualityEvaluator`.
        raw_dir: Directory containing raw DICOM files.
        output_dir: Directory where PNG figures are saved.
        labels_df: Merged annotation DataFrame (used to resolve DICOM paths).
    """

    def __init__(
        self,
        combined_results: List[Dict[str, Any]],
        raw_dir: Path,
        output_dir: Path,
        labels_df: pd.DataFrame,
    ) -> None:
        self._results = combined_results
        self._raw_dir = raw_dir
        self._out_dir = output_dir
        self._labels_df = labels_df
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def render_all(self) -> None:
        """Render all matched pairs."""
        logger.info("Rendering %d paired visualisations...", len(self._results))
        for idx, record in enumerate(self._results):
            try:
                self._render_pair(idx, record)
            except Exception as exc:
                logger.warning("Pair %d visualisation failed: %s", idx, exc)

    def _render_pair(self, idx: int, record: Dict[str, Any]) -> None:
        mlo_path = find_dicom_path(record["mlo_sop"], self._labels_df, self._raw_dir)
        cc_path = find_dicom_path(record["cc_sop"], self._labels_df, self._raw_dir)
        if mlo_path is None or cc_path is None:
            return

        mlo_img = load_dicom_for_display(mlo_path)
        cc_img = load_dicom_for_display(cc_path)

        fig, (ax_mlo, ax_cc) = plt.subplots(1, 2, figsize=(26, 13))
        ax_mlo.imshow(mlo_img, cmap="gray")
        ax_cc.imshow(cc_img, cmap="gray")

        self._overlay_mlo(ax_mlo, record)
        self._overlay_cc(ax_cc, record)

        ax_mlo.axis("off")
        ax_cc.axis("off")

        title_colour = "#2ecc71" if record["correct_prediction"] else "#e74c3c"
        fig.suptitle(
            f"Pred: {record['predicted_quality'].upper()}  |  "
            f"GT: {record['gt_quality'].upper()}  |  "
            f"Δ = {record['distance_diff_mm']:.1f} mm",
            color=title_colour,
            fontsize=14,
            fontweight="bold",
        )

        out_name = (
            f"pair_{idx:03d}_{record['patient_id'][:8]}_{record['laterality']}.png"
        )
        fig.savefig(self._out_dir / out_name, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    @staticmethod
    def _overlay_mlo(ax: plt.Axes, record: Dict[str, Any]) -> None:
        """Draw GT (red) and Pred (green) MLO keypoints with PNL lines."""
        gt = record.get("mlo_gt_kpts")
        if gt:
            gn, gp1, gp2 = gt["nipple"], gt["pectoral_top"], gt["pectoral_bottom"]
            ax.plot([gp1[0], gp2[0]], [gp1[1], gp2[1]], "r-", lw=1.8, alpha=0.8)
            gt_int = record.get("mlo_gt_intersection")
            if gt_int:
                ax.plot([gn[0], gt_int[0]], [gn[1], gt_int[1]], "r--", lw=1.4, alpha=0.7)
            for pt in (gn, gp1, gp2):
                ax.plot(pt[0], pt[1], "rs", ms=7, alpha=0.9)

        pred = record.get("mlo_pred_kpts")
        if pred and len(pred) >= 3:
            ax.plot([pred[1][0], pred[2][0]], [pred[1][1], pred[2][1]], "g-", lw=1.8, alpha=0.9)
            pred_int = record.get("mlo_pred_intersection")
            if pred_int:
                ax.plot([pred[0][0], pred_int[0]], [pred[0][1], pred_int[1]], "g--", lw=1.4, alpha=0.8)
            for pt in pred[:3]:
                ax.plot(pt[0], pt[1], "go", ms=7, alpha=0.9)

        mlo_mm = record.get("mlo_distance_mm", 0)
        ax.set_title(f"MLO  PNL = {mlo_mm:.1f} mm", fontsize=11)

    @staticmethod
    def _overlay_cc(ax: plt.Axes, record: Dict[str, Any]) -> None:
        """Draw GT (red) and Pred (green) CC nipple points."""
        gt_nipple = record.get("cc_gt_nipple")
        if gt_nipple:
            ax.plot(gt_nipple[0], gt_nipple[1], "rs", ms=9, label="GT", alpha=0.9)

        pred_nipple = record.get("cc_pred_nipple")
        if pred_nipple:
            ax.plot(pred_nipple[0], pred_nipple[1], "go", ms=9, label="Pred", alpha=0.9)

        cc_mm = record.get("cc_distance_mm", 0)
        ax.set_title(f"CC  Chest-Wall = {cc_mm:.1f} mm", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")


# ---------------------------------------------------------------------------
# Combined quality evaluator (classification metrics + visualisations)
# ---------------------------------------------------------------------------

class CombinedQualityEvaluator:
    """
    Match MLO and CC results by study + laterality and compute classification metrics.

    One MLO image and one CC image from the same patient and laterality
    form a pair.  The |PNL_MLO − CW_CC| difference is compared against
    a threshold to predict positioning quality.

    Args:
        mlo_results: Output of :class:`PoseEvaluator` for the MLO view.
        cc_results: Output of :class:`PoseEvaluator` for the CC view.
        threshold_mm: Decision boundary (default 10 mm per clinical standard).
    """

    def __init__(
        self,
        mlo_results: List[Dict[str, Any]],
        cc_results: List[Dict[str, Any]],
        threshold_mm: float = 10.0,
    ) -> None:
        self._mlo = mlo_results
        self._cc = cc_results
        self.threshold_mm = threshold_mm
        self.combined_results: List[Dict[str, Any]] = []

    def match_pairs(self) -> List[Dict[str, Any]]:
        """
        Match MLO-CC pairs by ``(patient_id, laterality)`` key.

        Returns:
            List of combined dictionaries, one per matched pair.
        """
        logger.info("Matching MLO-CC pairs by StudyInstanceUID + laterality…")
        mlo_index = {
            f"{r['patient_id']}_{r['laterality']}": r for r in self._mlo
        }

        matched = []
        for cc in self._cc:
            key = f"{cc['patient_id']}_{cc['laterality']}"
            if key not in mlo_index:
                continue
            mlo = mlo_index[key]
            diff = abs(mlo["distance_mm"] - cc["distance_mm"])
            predicted = "good" if diff <= self.threshold_mm else "bad"
            gt = str(cc.get("gt_quality", "unknown")).lower()

            matched.append({
                "patient_id": cc["patient_id"],
                "laterality": cc["laterality"],
                "mlo_sop": mlo["sop_uid"],
                "cc_sop": cc["sop_uid"],
                "mlo_distance_mm": mlo["distance_mm"],
                "cc_distance_mm": cc["distance_mm"],
                "mlo_gt_distance_mm": mlo.get("gt_distance_mm"),
                "cc_gt_distance_mm": cc.get("gt_distance_mm"),
                "distance_diff_mm": diff,
                "predicted_quality": predicted,
                "gt_quality": gt,
                "correct_prediction": predicted == gt,
                # Keypoints for visualisation
                "mlo_pred_kpts": mlo.get("pred_kpts_orig"),
                "mlo_gt_kpts": mlo.get("gt_kpts"),
                "mlo_pred_intersection": mlo.get("pred_intersection"),
                "mlo_gt_intersection": mlo.get("gt_intersection"),
                "cc_pred_nipple": cc.get("pred_nipple_orig"),
                "cc_gt_nipple": cc.get("gt_nipple"),
                # Per-keypoint errors
                "mlo_nipple_error_mm": mlo.get("nipple_error_mm"),
                "mlo_pec1_error_mm": mlo.get("pec1_error_mm"),
                "mlo_pec2_error_mm": mlo.get("pec2_error_mm"),
                "mlo_pnl_error_mm": mlo.get("pnl_error_mm"),
                "cc_nipple_error_mm": cc.get("nipple_error_mm"),
                "cc_cw_error_mm": cc.get("cw_error_mm"),
            })

        self.combined_results = matched
        logger.info("Matched pairs: %d", len(matched))
        return matched

    def compute_classification_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Compute accuracy, sensitivity, specificity, and confusion matrix.

        Only pairs where the ground-truth quality is ``'good'`` or
        ``'bad'`` are included.

        Returns:
            Metrics dictionary, or ``None`` if no valid pairs exist.
        """
        if not self.combined_results:
            return None

        y_true, y_pred = [], []
        for r in self.combined_results:
            gt = str(r.get("gt_quality", "unknown")).lower()
            if gt not in ("good", "bad"):
                continue
            y_true.append(gt)
            y_pred.append(r["predicted_quality"])

        if not y_true:
            return None

        yt_enc = [1 if t == "good" else 0 for t in y_true]
        yp_enc = [1 if p == "good" else 0 for p in y_pred]

        acc = accuracy_score(yt_enc, yp_enc)
        prec, rec, f1, _ = precision_recall_fscore_support(
            yt_enc, yp_enc, average="weighted", zero_division=0
        )
        cm = confusion_matrix(yt_enc, yp_enc, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        return {
            "threshold_mm": self.threshold_mm,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "confusion_matrix": cm,
            "total_pairs": len(y_true),
            "correct_predictions": int(acc * len(y_true)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
