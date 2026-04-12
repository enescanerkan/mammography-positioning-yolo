"""
Unified YOLO pose evaluation engine for the breast positioning pipeline.

This module replaces the three separate ``evaluate_yolo_pose.py`` scripts
that previously lived in ``src/test/``, ``src/test_hist/``, and
``src/test_advanced/``.  A single :class:`PoseEvaluator` now handles all
preprocessing strategies by reading its configuration from an
``ExperimentConfig`` object.

Clinical classification rule
-----------------------------
Positioning quality is determined by the **10 mm rule**:

    |PNL_MLO − ChestWallDist_CC| ≤ threshold_mm  →  **Good**

Inverse coordinate transform (640×640 processed space → original DICOM pixels)
--------------------------------------------------------------------------------
    orig_x = (pred_x − pad_left) / scale + crop_x1
    orig_y = (pred_y − pad_top)  / scale + crop_y1
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from ..core.interfaces import IEvaluator
from ..utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure geometry helpers
# ---------------------------------------------------------------------------

def inverse_transform(
    kp_x: float,
    kp_y: float,
    pad_left: float,
    pad_top: float,
    scale: float,
    crop_x1: float,
    crop_y1: float,
) -> Tuple[float, float]:
    """
    Map a keypoint from the 640×640 processed image back to original DICOM space.

    Args:
        kp_x: Predicted x-coordinate in the 640×640 space (pixels).
        kp_y: Predicted y-coordinate in the 640×640 space (pixels).
        pad_left: Left padding applied during preprocessing (pixels).
        pad_top: Top padding applied during preprocessing (pixels).
        scale: Scale factor used during resize (target / max_side).
        crop_x1: Left boundary of the breast crop in original DICOM pixels.
        crop_y1: Top boundary of the breast crop in original DICOM pixels.

    Returns:
        Tuple ``(orig_x, orig_y)`` in original DICOM pixel coordinates.
    """
    orig_x = (kp_x - pad_left) / scale + crop_x1
    orig_y = (kp_y - pad_top) / scale + crop_y1
    return float(orig_x), float(orig_y)


def pnl_distance(
    nipple: Tuple[float, float],
    pec_top: Tuple[float, float],
    pec_bottom: Tuple[float, float],
) -> Tuple[float, List[float]]:
    """
    Compute the Posterior Nipple Line (PNL) perpendicular distance.

    The PNL is the shortest (perpendicular) distance from the nipple point
    to the infinite line defined by the two pectoral muscle landmarks.

    Args:
        nipple: (x, y) nipple coordinates in DICOM pixel space.
        pec_top: (x, y) pectoral top landmark.
        pec_bottom: (x, y) pectoral bottom landmark.

    Returns:
        Tuple of:
            - perpendicular distance in pixels (float).
            - intersection point [x, y] on the pectoral line.
    """
    n = np.array(nipple, dtype=np.float64)
    p1 = np.array(pec_top, dtype=np.float64)
    p2 = np.array(pec_bottom, dtype=np.float64)

    line_vec = p2 - p1
    line_len = float(np.linalg.norm(line_vec))

    if line_len < 1e-8:
        return float(np.linalg.norm(n - p1)), p1.tolist()

    line_unit = line_vec / line_len
    proj_len = float(np.dot(n - p1, line_unit))
    intersection = p1 + proj_len * line_unit
    perp_dist = float(np.linalg.norm(n - intersection))
    return perp_dist, intersection.tolist()


def chest_wall_distance(
    nipple_x: float,
    laterality: str,
    original_width: float,
) -> float:
    """
    Compute the CC chest-wall distance.

    For a **left** breast the chest wall is on the left edge (x = 0),
    so the distance equals ``nipple_x``.  For a **right** breast the
    chest wall is on the right edge, so the distance is
    ``original_width − nipple_x``.

    Args:
        nipple_x: Nipple x-coordinate in original DICOM pixel space.
        laterality: ``'L'`` or ``'R'``.
        original_width: Width of the original DICOM image in pixels.

    Returns:
        Distance in pixels.
    """
    if laterality == "L":
        return float(nipple_x)
    return float(abs(original_width - nipple_x))


def euclidean_error(
    pred: Tuple[float, float],
    gt: Tuple[float, float],
) -> float:
    """Return the Euclidean distance between two 2-D points."""
    return float(np.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2))


# ---------------------------------------------------------------------------
# Ground-truth label parsers
# ---------------------------------------------------------------------------

def parse_mlo_gt(sop_uid: str, labels_df: pd.DataFrame) -> Optional[Dict[str, Tuple]]:
    """
    Extract MLO ground-truth keypoints from the annotation DataFrame.

    Args:
        sop_uid: SOPInstanceUID of the target image.
        labels_df: Combined annotation DataFrame.

    Returns:
        Dictionary with keys ``'nipple'``, ``'pectoral_top'``,
        ``'pectoral_bottom'`` (each a ``(x, y)`` tuple in DICOM pixels),
        or ``None`` if not all three keypoints could be parsed.
    """
    rows = labels_df[labels_df["SOPInstanceUID"] == sop_uid]
    if rows.empty:
        return None

    keypoints: Dict[str, Tuple] = {}
    for _, row in rows.iterrows():
        label_name = row.get("labelName", "")
        data_str = row.get("data", "")
        if pd.isna(data_str):
            continue
        try:
            data = ast.literal_eval(str(data_str))
        except Exception:
            continue

        if label_name == "Nipple":
            cx = data["x"] + data["width"] / 2.0
            cy = data["y"] + data["height"] / 2.0
            keypoints["nipple"] = (cx, cy)
        elif label_name == "Pectoralis":
            pts = data["vertices"]
            if len(pts) >= 2:
                # Sort so that the topmost point (smallest y) is 'pectoral_top'
                if pts[0][1] < pts[1][1]:
                    keypoints["pectoral_top"] = (pts[0][0], pts[0][1])
                    keypoints["pectoral_bottom"] = (pts[1][0], pts[1][1])
                else:
                    keypoints["pectoral_top"] = (pts[1][0], pts[1][1])
                    keypoints["pectoral_bottom"] = (pts[0][0], pts[0][1])

    required = {"nipple", "pectoral_top", "pectoral_bottom"}
    if not required.issubset(keypoints):
        return None
    return keypoints


def parse_cc_gt(sop_uid: str, labels_df: pd.DataFrame) -> Optional[Dict[str, Tuple]]:
    """
    Extract CC ground-truth nipple keypoint from the annotation DataFrame.

    Args:
        sop_uid: SOPInstanceUID of the target image.
        labels_df: Combined annotation DataFrame.

    Returns:
        Dictionary ``{'nipple': (x, y)}`` or ``None`` if not found.
    """
    rows = labels_df[
        (labels_df["SOPInstanceUID"] == sop_uid)
        & (labels_df["labelName"] == "Nipple")
    ]
    if rows.empty:
        return None

    data_str = rows.iloc[0].get("data", "")
    if pd.isna(data_str):
        return None
    try:
        data = ast.literal_eval(str(data_str))
    except Exception:
        return None

    cx = data["x"] + data["width"] / 2.0
    cy = data["y"] + data["height"] / 2.0
    return {"nipple": (cx, cy)}


# ---------------------------------------------------------------------------
# YOLO predictor helpers
# ---------------------------------------------------------------------------

def _predict_keypoints(model: YOLO, image_path: Path) -> Optional[np.ndarray]:
    """
    Run YOLO pose inference on a single image.

    Args:
        model: Loaded Ultralytics YOLO model.
        image_path: Path to the preprocessed 640×640 PNG.

    Returns:
        NumPy array of shape (N_kpts, 2) in pixel coordinates,
        or ``None`` if inference fails or no detections are made.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("Could not read image: %s", image_path)
        return None

    results = model(img, verbose=False)[0]

    if results.keypoints is None or len(results.keypoints.xy[0]) == 0:
        return None

    return results.keypoints.xy[0].cpu().numpy()


# ---------------------------------------------------------------------------
# numpy JSON serialiser
# ---------------------------------------------------------------------------

def _numpy_serialiser(obj: Any) -> Any:
    """JSON serialiser that handles NumPy scalar types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Per-view evaluators
# ---------------------------------------------------------------------------

class _MLOViewEvaluator:
    """Internal class that runs inference and collects MLO metrics."""

    def __init__(self, model: YOLO, data_dir: Path, labels_df: pd.DataFrame) -> None:
        self._model = model
        self._data_dir = data_dir
        self._labels_df = labels_df

    def run(self) -> List[Dict[str, Any]]:
        """Evaluate the test split and return per-image result dictionaries."""
        meta_path = self._data_dir / "metadata.csv"
        if not meta_path.exists():
            logger.error("MLO metadata not found: %s", meta_path)
            return []

        meta_df = pd.read_csv(meta_path)
        test_df = meta_df[meta_df["split"] == "test"].copy()
        # Exclude flip-augmented images from test evaluation
        test_df = test_df[~test_df["sop_uid"].astype(str).str.contains("_flip", na=False)]
        logger.info("MLO test images: %d (augmented excluded)", len(test_df))

        results: List[Dict[str, Any]] = []

        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="MLO inference"):
            sop_uid = str(row["sop_uid"])
            img_path = self._data_dir / "images" / "test" / f"{sop_uid}.png"
            if not img_path.exists():
                continue

            kpts_640 = _predict_keypoints(self._model, img_path)
            if kpts_640 is None or len(kpts_640) < 3:
                continue

            # Inverse-transform all keypoints back to DICOM pixel space
            spacing = float(row.get("pixel_spacing", 0.085))
            kpts_orig = []
            for kp in kpts_640:
                ox, oy = inverse_transform(
                    kp[0], kp[1],
                    row["pad_left"], row["pad_top"],
                    row["scale"], row["crop_x1"], row["crop_y1"],
                )
                kpts_orig.append([ox, oy])

            pred_pnl_px, pred_intersection = pnl_distance(
                kpts_orig[0], kpts_orig[1], kpts_orig[2]
            )
            pred_pnl_mm = pred_pnl_px * spacing

            # Ground-truth comparison
            gt = parse_mlo_gt(sop_uid, self._labels_df)
            gt_pnl_mm = nipple_err = pec1_err = pec2_err = pnl_err = None
            gt_intersection = None

            if gt is not None:
                gt_pnl_px, gt_intersection = pnl_distance(
                    gt["nipple"], gt["pectoral_top"], gt["pectoral_bottom"]
                )
                gt_pnl_mm = gt_pnl_px * spacing
                nipple_err = euclidean_error(kpts_orig[0], gt["nipple"]) * spacing
                pec1_err = euclidean_error(kpts_orig[1], gt["pectoral_top"]) * spacing
                pec2_err = euclidean_error(kpts_orig[2], gt["pectoral_bottom"]) * spacing
                pnl_err = abs(pred_pnl_mm - gt_pnl_mm)

            # Resolve patient (study) ID
            patient_id = sop_uid
            lbl_row = self._labels_df[self._labels_df["SOPInstanceUID"] == sop_uid]
            if not lbl_row.empty and "StudyInstanceUID" in lbl_row.columns:
                patient_id = str(lbl_row.iloc[0]["StudyInstanceUID"])

            results.append({
                "patient_id": patient_id,
                "sop_uid": sop_uid,
                "view": "MLO",
                "laterality": row["laterality"],
                "pixel_spacing": spacing,
                "distance_mm": pred_pnl_mm,
                "distance_px": pred_pnl_px,
                "gt_distance_mm": gt_pnl_mm,
                "pred_kpts_orig": kpts_orig,
                "pred_intersection": pred_intersection,
                "gt_kpts": {k: list(v) for k, v in gt.items()} if gt else None,
                "gt_intersection": gt_intersection,
                "nipple_error_mm": nipple_err,
                "pec1_error_mm": pec1_err,
                "pec2_error_mm": pec2_err,
                "pnl_error_mm": pnl_err,
            })

        logger.info("MLO evaluated: %d / %d images.", len(results), len(test_df))
        return results


class _CCViewEvaluator:
    """Internal class that runs inference and collects CC metrics."""

    def __init__(self, model: YOLO, data_dir: Path, labels_df: pd.DataFrame) -> None:
        self._model = model
        self._data_dir = data_dir
        self._labels_df = labels_df

    def run(self) -> List[Dict[str, Any]]:
        """Evaluate the test split and return per-image result dictionaries."""
        meta_path = self._data_dir / "metadata.csv"
        if not meta_path.exists():
            logger.error("CC metadata not found: %s", meta_path)
            return []

        meta_df = pd.read_csv(meta_path)
        test_df = meta_df[meta_df["split"] == "test"].copy()
        test_df = test_df[~test_df["sop_uid"].astype(str).str.contains("_flip", na=False)]
        logger.info("CC test images: %d (augmented excluded)", len(test_df))

        results: List[Dict[str, Any]] = []

        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="CC inference"):
            sop_uid = str(row["sop_uid"])
            img_path = self._data_dir / "images" / "test" / f"{sop_uid}.png"
            if not img_path.exists():
                continue

            kpts_640 = _predict_keypoints(self._model, img_path)
            if kpts_640 is None or len(kpts_640) < 1:
                continue

            spacing = float(row.get("pixel_spacing", 0.085))
            orig_w = float(row["original_width"])

            pred_nipple_x, pred_nipple_y = inverse_transform(
                kpts_640[0][0], kpts_640[0][1],
                row["pad_left"], row["pad_top"],
                row["scale"], row["crop_x1"], row["crop_y1"],
            )
            pred_cw_px = chest_wall_distance(pred_nipple_x, row["laterality"], orig_w)
            pred_cw_mm = pred_cw_px * spacing

            gt = parse_cc_gt(sop_uid, self._labels_df)
            gt_cw_mm = gt_nipple = nipple_err = cw_err = None

            if gt is not None:
                gt_nipple = list(gt["nipple"])
                gt_cw_px = chest_wall_distance(gt_nipple[0], row["laterality"], orig_w)
                gt_cw_mm = gt_cw_px * spacing
                nipple_err = euclidean_error([pred_nipple_x, pred_nipple_y], gt_nipple) * spacing
                cw_err = abs(pred_cw_mm - gt_cw_mm)

            patient_id = sop_uid
            gt_quality = "unknown"
            lbl_row = self._labels_df[self._labels_df["SOPInstanceUID"] == sop_uid]
            if not lbl_row.empty:
                if "StudyInstanceUID" in lbl_row.columns:
                    patient_id = str(lbl_row.iloc[0]["StudyInstanceUID"])
                for q_col in ("qualitativeLabel", "quality"):
                    if q_col in lbl_row.columns:
                        gt_quality = str(lbl_row.iloc[0][q_col])
                        break

            results.append({
                "patient_id": patient_id,
                "sop_uid": sop_uid,
                "view": "CC",
                "laterality": row["laterality"],
                "pixel_spacing": spacing,
                "original_width": orig_w,
                "distance_mm": pred_cw_mm,
                "distance_px": pred_cw_px,
                "gt_distance_mm": gt_cw_mm,
                "pred_nipple_orig": [pred_nipple_x, pred_nipple_y],
                "gt_nipple": gt_nipple,
                "nipple_error_mm": nipple_err,
                "cw_error_mm": cw_err,
                "gt_quality": gt_quality,
            })

        logger.info("CC evaluated: %d / %d images.", len(results), len(test_df))
        return results


# ---------------------------------------------------------------------------
# Public evaluator
# ---------------------------------------------------------------------------

class PoseEvaluator(IEvaluator):
    """
    Strategy-agnostic pose evaluation engine.

    Reads preprocessed test images and metadata from the path determined
    by the experiment's ``strategy`` and ``view`` settings, loads the
    corresponding trained model checkpoint, and computes all clinical
    and keypoint-level metrics.

    Args:
        config: Fully initialised :class:`~src.utils.config.ExperimentConfig`.

    Example
    -------
    >>> from src.utils.config import ExperimentConfig
    >>> from src.evaluation.evaluator import PoseEvaluator
    >>>
    >>> cfg = ExperimentConfig.from_yaml("configs/wavelet_mlo.yaml")
    >>> evaluator = PoseEvaluator(cfg)
    >>> results = evaluator.evaluate()
    """

    def __init__(self, config: Any) -> None:
        self._cfg = config
        self._project_root = config._root

        # Validate paths up-front so failures are reported before inference
        issues = self._validate()
        if issues:
            for issue in issues:
                logger.error("Validation error: %s", issue)
            raise RuntimeError(
                "Evaluation cannot proceed due to missing paths. "
                "Run preprocessing and training first."
            )

    # ------------------------------------------------------------------
    # IEvaluator implementation
    # ------------------------------------------------------------------

    def evaluate(self) -> List[Dict[str, Any]]:
        """
        Run the full evaluation pipeline for the configured view.

        Returns:
            List of per-image result dictionaries with predicted
            distances, ground-truth distances, and keypoint errors.
        """
        labels_df = self._load_labels()

        model_path = self._cfg.default_trained_model
        logger.info("Loading trained model from: %s", model_path)
        model = YOLO(str(model_path))

        processed_dir = self._cfg.processed_data_dir

        if self._cfg.view == "MLO":
            evaluator = _MLOViewEvaluator(model, processed_dir, labels_df)
        else:
            evaluator = _CCViewEvaluator(model, processed_dir, labels_df)

        results = evaluator.run()
        self._save_results(results)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate(self) -> List[str]:
        """Return a list of human-readable validation error messages."""
        issues = []
        if not self._cfg.processed_data_dir.exists():
            issues.append(
                f"Processed data directory not found: {self._cfg.processed_data_dir}\n"
                "  → Run preprocessing first: "
                f"python main.py --config <your_config.yaml> --mode preprocess"
            )
        if not self._cfg.default_trained_model.exists():
            issues.append(
                f"Trained model not found: {self._cfg.default_trained_model}\n"
                "  → Run training first: "
                f"python main.py --config <your_config.yaml> --mode train\n"
                "  → Or set 'trained_model_path' in your config YAML."
            )
        return issues

    def _load_labels(self) -> pd.DataFrame:
        """Load and concatenate all annotation CSV files."""
        dfs = []
        for csv_path in self._cfg.label_files:
            p = Path(csv_path)
            if not p.exists():
                logger.warning("Label file not found: %s", p)
                continue
            dfs.append(pd.read_csv(p))

        if not dfs:
            raise FileNotFoundError("No label CSV files found.")

        df = pd.concat(dfs, ignore_index=True)
        if "SOPInstanceUID" in df.columns:
            df["SOPInstanceUID"] = df["SOPInstanceUID"].astype(str)
        return df

    def _save_results(self, results: List[Dict[str, Any]]) -> None:
        """Serialise results to JSON inside the evaluation output directory."""
        out_dir = self._cfg.evaluation_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / "results.json"
        with out_file.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, default=_numpy_serialiser, indent=2)

        logger.info("Results saved to: %s", out_file)
