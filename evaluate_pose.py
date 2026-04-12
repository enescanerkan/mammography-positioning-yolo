"""
Unified YOLO Pose Evaluation Script — Config-Driven

Replaces: src/test_based/, src/test_hist/, src/test_advanced/

All experiment paths and strategy names are read from a YAML
config file (same format used by main.py). No hardcoded paths.

Usage
-----
Single-view (MLO or CC):
    python evaluate_pose.py --config configs/baseline_mlo.yaml
    python evaluate_pose.py --config configs/histeq_mlo.yaml
    python evaluate_pose.py --config configs/wavelet_mlo.yaml

Combined MLO + CC (full positioning quality report):
    python evaluate_pose.py \\
        --mlo_config configs/baseline_mlo.yaml \\
        --cc_config  configs/baseline_cc.yaml  \\
        --threshold  10

Options
-------
--threshold FLOAT     Clinical decision threshold in mm (default: 10).
--dual_thresholds     Also run dual-threshold analysis (e.g. 9/11, 8/12 mm).
--skip_viz            Skip per-pair DICOM visualisations (faster, no raw DICOMs needed).
--output_dir PATH     Override output directory (default: evaluation_results/<strategy>/).
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    _PYDICOM_AVAILABLE = True
except ImportError:
    _PYDICOM_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ExperimentConfig
from src.utils.logger import get_logger

logger = get_logger("evaluate_pose")


# =============================================================================
# Geometry helpers
# =============================================================================

def inverse_transform(kp_x: float, kp_y: float, meta_row: pd.Series) -> Tuple[float, float]:
    """Map a keypoint from 640×640 processed space back to original DICOM pixel space."""
    ox = (kp_x - meta_row["pad_left"]) / meta_row["scale"] + meta_row["crop_x1"]
    oy = (kp_y - meta_row["pad_top"])  / meta_row["scale"] + meta_row["crop_y1"]
    return float(ox), float(oy)


def pnl_distance(nipple, p1, p2) -> Tuple[float, list]:
    """Perpendicular distance from nipple to pectoral line p1→p2."""
    n  = np.array(nipple, dtype=np.float64)
    a  = np.array(p1,     dtype=np.float64)
    b  = np.array(p2,     dtype=np.float64)
    v  = b - a
    vl = np.linalg.norm(v)
    if vl == 0:
        return float(np.linalg.norm(n - a)), a.tolist()
    proj = a + np.dot(n - a, v) / (vl * vl) * v
    return float(np.linalg.norm(n - proj)), proj.tolist()


def chest_wall_distance(nipple_x: float, laterality: str, original_width: float) -> float:
    """Horizontal distance from nipple to chest-wall edge (CC view)."""
    return float(nipple_x) if laterality == "L" else float(abs(original_width - nipple_x))


def kp_euclidean_error(pred, gt) -> float:
    return float(np.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2))


def stat_summary(values: list) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "count": 0}
    a = np.array(values)
    return {
        "mean":   float(np.mean(a)),
        "std":    float(np.std(a)),
        "median": float(np.median(a)),
        "count":  len(a),
    }


def numpy_serializer(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return str(obj)


# =============================================================================
# Label / DICOM helpers
# =============================================================================

def parse_gt_mlo_keypoints(sop_uid: str, labels_df: pd.DataFrame) -> Optional[dict]:
    rows = labels_df[labels_df["SOPInstanceUID"] == sop_uid]
    if rows.empty:
        return None
    kpts: dict = {}
    for _, row in rows.iterrows():
        label = row.get("labelName", "")
        raw   = row.get("data", "")
        if pd.isna(raw):
            continue
        try:
            data = ast.literal_eval(str(raw))
        except Exception:
            continue
        if label == "Nipple":
            kpts["nipple"] = (data["x"] + data["width"] / 2.0,
                              data["y"] + data["height"] / 2.0)
        elif label == "Pectoralis":
            pts = data.get("vertices", [])
            if len(pts) >= 2:
                p0, p1 = pts[0], pts[1]
                if p0[1] < p1[1]:
                    kpts["pectoral_top"], kpts["pectoral_bottom"] = p0, p1
                else:
                    kpts["pectoral_top"], kpts["pectoral_bottom"] = p1, p0
    if all(k in kpts for k in ("nipple", "pectoral_top", "pectoral_bottom")):
        return kpts
    return None


def parse_gt_cc_keypoints(sop_uid: str, labels_df: pd.DataFrame) -> Optional[dict]:
    rows = labels_df[
        (labels_df["SOPInstanceUID"] == sop_uid) & (labels_df["labelName"] == "Nipple")
    ]
    if rows.empty:
        return None
    raw = rows.iloc[0].get("data", "")
    if pd.isna(raw):
        return None
    try:
        data = ast.literal_eval(str(raw))
    except Exception:
        return None
    return {"nipple": (data["x"] + data["width"] / 2.0, data["y"] + data["height"] / 2.0)}


def load_dicom_image(dicom_path: Path) -> Optional[np.ndarray]:
    """Load DICOM → normalised 0-1 float32 array for visualisation."""
    if not _PYDICOM_AVAILABLE:
        return None
    try:
        ds  = pydicom.dcmread(str(dicom_path))
        arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            arr = arr.max() - arr
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr
    except Exception as exc:
        logger.warning("Could not read DICOM %s: %s", dicom_path, exc)
        return None


def find_dicom_path(sop_uid: str, labels_df: pd.DataFrame, raw_dir: Path) -> Optional[Path]:
    row = labels_df[labels_df["SOPInstanceUID"] == sop_uid]
    if not row.empty:
        candidate = raw_dir / str(row.iloc[0]["StudyInstanceUID"]) / f"{sop_uid}.dicom"
        if candidate.exists():
            return candidate
    for study_dir in raw_dir.iterdir():
        if not study_dir.is_dir():
            continue
        candidate = study_dir / f"{sop_uid}.dicom"
        if candidate.exists():
            return candidate
    return None


# =============================================================================
# Path configuration  (derived from ExperimentConfig + strategy name)
# =============================================================================

class EvalPathConfig:
    """
    Resolves all filesystem paths needed for evaluation from an ExperimentConfig.

    Strategy → processed data layout:
      baseline  : data/processed/baseline/{MLO,CC}/images/
      histeq    : archive/preprocessed-data/{MLO,CC}/images_histeq/
      wavelet   : data/processed/wavelet/{MLO,CC}/images/
      advanced  : data/processed/advanced/{MLO,CC}/images/
      (anything else follows the same pattern as baseline/wavelet)
    """

    # Image sub-directory name per strategy  (relative to the view data dir)
    _IMAGE_SUBDIR: Dict[str, str] = {
        "histeq": "images_histeq",
    }
    # Data root per strategy
    _DATA_ROOT: Dict[str, str] = {
        "histeq": "archive/preprocessed-data",
    }
    # Default model weight filename per strategy (inside the archive MLO/CC dirs)
    _MODEL_FILENAME: Dict[str, str] = {
        "histeq": "besthisteq.pt",
    }

    def __init__(self, cfg: ExperimentConfig, output_override: Optional[str] = None):
        self.strategy = cfg.strategy
        self.view     = cfg.view         # "MLO" | "CC"
        self.root     = PROJECT_ROOT

        # --- data root ---
        if self.strategy in self._DATA_ROOT:
            self.data_root = self.root / self._DATA_ROOT[self.strategy]
        else:
            self.data_root = self.root / "data" / "processed" / self.strategy

        # --- image sub-directory name ---
        self.image_subdir = self._IMAGE_SUBDIR.get(self.strategy, "images")

        # --- processed view dirs ---
        self.mlo_data_dir = self.data_root / "MLO"
        self.cc_data_dir  = self.data_root / "CC"

        # --- trained model weights ---
        #   Priority 1: trained_model_path from config
        #   Priority 2: experiments/<strategy>/<view>/yolo_run/weights/best.pt
        mlo_model_from_cfg = Path(cfg.trained_model_path) if cfg.trained_model_path else None
        if mlo_model_from_cfg and mlo_model_from_cfg.exists():
            self.mlo_model_path = mlo_model_from_cfg
            self.cc_model_path  = mlo_model_from_cfg         # same if single-view config
        else:
            weight_name = self._MODEL_FILENAME.get(self.strategy, "best.pt")
            if self.strategy == "histeq":
                self.mlo_model_path = (
                    self.root / "archive" / "MLO" / "yolo_run" / "weights" / weight_name
                )
                self.cc_model_path = (
                    self.root / "archive" / "CC"  / "yolo_run" / "weights" / weight_name
                )
            else:
                exp_base = self.root / "experiments" / self.strategy
                self.mlo_model_path = exp_base / "MLO" / "yolo_run" / "weights" / weight_name
                self.cc_model_path  = exp_base / "CC"  / "yolo_run" / "weights" / weight_name

        # --- raw DICOMs and labels ---
        self.raw_dir    = self.root / "data" / "raw"
        self.labels_dir = self.root / "data" / "labels"

        # --- output ---
        if output_override:
            self.output_dir = Path(output_override)
        else:
            self.output_dir = self.root / "evaluation_results" / self.strategy / self.view

        self.mlo_viz_dir      = self.output_dir.parent / "MLO" / "visualizations"
        self.cc_viz_dir       = self.output_dir.parent / "CC"  / "visualizations"
        self.combined_viz_dir = self.output_dir.parent / "combined" / "paired_viz"

    def setup_directories(self) -> None:
        for d in (self.output_dir, self.mlo_viz_dir, self.cc_viz_dir, self.combined_viz_dir):
            d.mkdir(parents=True, exist_ok=True)

    def validate(self) -> List[str]:
        issues = []
        for label, path in [
            ("MLO model",   self.mlo_model_path),
            ("CC  model",   self.cc_model_path),
            ("MLO data",    self.mlo_data_dir),
            ("CC  data",    self.cc_data_dir),
            ("Raw DICOMs",  self.raw_dir),
        ]:
            if not path.exists():
                issues.append(f"{label} not found: {path}")
        return issues


# =============================================================================
# YOLO predictor  (single class handles both views)
# =============================================================================

class YOLOPredictor:
    """Thin wrapper around an Ultralytics YOLO pose model."""

    def __init__(self, model_path: Path) -> None:
        self.model: Optional[YOLO] = None
        if model_path.exists():
            logger.info("Loading model: %s", model_path)
            self.model = YOLO(str(model_path))
        else:
            logger.warning("Model not found: %s", model_path)

    def predict_keypoints(self, image_path: Path) -> Optional[np.ndarray]:
        """Return keypoints as (N, 2) numpy array, or None on failure."""
        if self.model is None:
            return None
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        res = self.model(img, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints.xy) == 0:
            return None
        kp = res.keypoints.xy[0].cpu().numpy()
        return kp if len(kp) > 0 else None


# =============================================================================
# Per-view evaluation
# =============================================================================

def evaluate_mlo(
    paths: EvalPathConfig,
    labels_df: pd.DataFrame,
    mlo_model_path: Optional[Path] = None,
) -> List[dict]:
    logger.info("=" * 60)
    logger.info("EVALUATING MLO [%s]", paths.strategy.upper())
    logger.info("=" * 60)

    model_path = mlo_model_path or paths.mlo_model_path
    meta_path  = paths.mlo_data_dir / "metadata.csv"
    if not meta_path.exists():
        logger.error("MLO metadata not found: %s", meta_path)
        return []

    df      = pd.read_csv(meta_path)
    test_df = df[df["split"] == "test"].copy()
    test_df = test_df[~test_df["sop_uid"].astype(str).str.contains("_flip", na=False)]
    logger.info("MLO test images: %d (excluding flipped augmentations)", len(test_df))

    predictor = YOLOPredictor(model_path)
    if predictor.model is None:
        return []

    results: List[dict] = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="MLO prediction"):
        sop_uid  = str(row["sop_uid"])
        img_path = paths.mlo_data_dir / paths.image_subdir / "test" / f"{sop_uid}.png"
        if not img_path.exists():
            continue

        kpts_640 = predictor.predict_keypoints(img_path)
        if kpts_640 is None or len(kpts_640) < 3:
            continue

        kpts_orig = [list(inverse_transform(kp[0], kp[1], row)) for kp in kpts_640]
        spacing   = float(row.get("pixel_spacing", 0.085))

        pred_pnl_px, pred_inter = pnl_distance(kpts_orig[0], kpts_orig[1], kpts_orig[2])
        pred_pnl_mm = pred_pnl_px * spacing

        gt_kpts = parse_gt_mlo_keypoints(sop_uid, labels_df)
        gt_pnl_mm = gt_inter = None
        nipple_err = pec1_err = pec2_err = pnl_err = None

        if gt_kpts is not None:
            gt_pnl_px, gt_inter = pnl_distance(
                gt_kpts["nipple"], gt_kpts["pectoral_top"], gt_kpts["pectoral_bottom"]
            )
            gt_pnl_mm  = gt_pnl_px * spacing
            nipple_err = kp_euclidean_error(kpts_orig[0], gt_kpts["nipple"])         * spacing
            pec1_err   = kp_euclidean_error(kpts_orig[1], gt_kpts["pectoral_top"])   * spacing
            pec2_err   = kp_euclidean_error(kpts_orig[2], gt_kpts["pectoral_bottom"])* spacing
            pnl_err    = abs(pred_pnl_mm - gt_pnl_mm)

        l_row      = labels_df[labels_df["SOPInstanceUID"] == sop_uid]
        patient_id = str(l_row.iloc[0]["StudyInstanceUID"]) if not l_row.empty else sop_uid

        results.append({
            "patient_id":       patient_id,
            "sop_uid":          sop_uid,
            "laterality":       row["laterality"],
            "pixel_spacing":    spacing,
            "distance_mm":      pred_pnl_mm,
            "distance_px":      pred_pnl_px,
            "gt_distance_mm":   gt_pnl_mm,
            "pred_kpts_orig":   kpts_orig,
            "pred_intersection":pred_inter,
            "gt_kpts":          ({k: list(v) for k, v in gt_kpts.items()} if gt_kpts else None),
            "gt_intersection":  gt_inter,
            "nipple_error_mm":  nipple_err,
            "pec1_error_mm":    pec1_err,
            "pec2_error_mm":    pec2_err,
            "pnl_error_mm":     pnl_err,
        })

    logger.info("MLO evaluated: %d / %d", len(results), len(test_df))
    _log_error_summary("MLO", results,
                       ["nipple_error_mm", "pec1_error_mm", "pec2_error_mm", "pnl_error_mm"],
                       ["Nipple", "Pec Top", "Pec Bot", "PNL Dist"])
    return results


def evaluate_cc(
    paths: EvalPathConfig,
    labels_df: pd.DataFrame,
    cc_model_path: Optional[Path] = None,
) -> List[dict]:
    logger.info("=" * 60)
    logger.info("EVALUATING CC [%s]", paths.strategy.upper())
    logger.info("=" * 60)

    model_path = cc_model_path or paths.cc_model_path
    meta_path  = paths.cc_data_dir / "metadata.csv"
    if not meta_path.exists():
        logger.error("CC metadata not found: %s", meta_path)
        return []

    df      = pd.read_csv(meta_path)
    test_df = df[df["split"] == "test"].copy()
    test_df = test_df[~test_df["sop_uid"].astype(str).str.contains("_flip", na=False)]
    logger.info("CC test images: %d (excluding flipped augmentations)", len(test_df))

    predictor = YOLOPredictor(model_path)
    if predictor.model is None:
        return []

    results: List[dict] = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="CC prediction"):
        sop_uid  = str(row["sop_uid"])
        img_path = paths.cc_data_dir / paths.image_subdir / "test" / f"{sop_uid}.png"
        if not img_path.exists():
            continue

        kpts_640 = predictor.predict_keypoints(img_path)
        if kpts_640 is None or len(kpts_640) < 1:
            continue

        pred_nipple = list(inverse_transform(kpts_640[0][0], kpts_640[0][1], row))
        spacing     = float(row.get("pixel_spacing", 0.085))
        orig_w      = row["original_width"]
        pred_cw_px  = chest_wall_distance(pred_nipple[0], row["laterality"], orig_w)
        pred_cw_mm  = pred_cw_px * spacing

        gt_kpts = parse_gt_cc_keypoints(sop_uid, labels_df)
        gt_cw_mm = gt_nipple = nipple_err = cw_err = None

        if gt_kpts is not None:
            gt_nipple  = list(gt_kpts["nipple"])
            gt_cw_px   = chest_wall_distance(gt_nipple[0], row["laterality"], orig_w)
            gt_cw_mm   = gt_cw_px * spacing
            nipple_err = kp_euclidean_error(pred_nipple, gt_nipple) * spacing
            cw_err     = abs(pred_cw_mm - gt_cw_mm)

        l_row      = labels_df[labels_df["SOPInstanceUID"] == sop_uid]
        patient_id = str(l_row.iloc[0]["StudyInstanceUID"]) if not l_row.empty else sop_uid
        gt_quality = "unknown"
        if not l_row.empty:
            for q_col in ("qualitativeLabel", "quality"):
                if q_col in l_row.columns:
                    gt_quality = str(l_row.iloc[0][q_col])
                    break

        results.append({
            "patient_id":      patient_id,
            "sop_uid":         sop_uid,
            "laterality":      row["laterality"],
            "pixel_spacing":   spacing,
            "original_width":  orig_w,
            "distance_mm":     pred_cw_mm,
            "distance_px":     pred_cw_px,
            "gt_distance_mm":  gt_cw_mm,
            "pred_nipple_orig":pred_nipple,
            "gt_nipple":       gt_nipple,
            "nipple_error_mm": nipple_err,
            "cw_error_mm":     cw_err,
            "gt_quality":      gt_quality,
        })

    logger.info("CC evaluated: %d / %d", len(results), len(test_df))
    _log_error_summary("CC", results,
                       ["nipple_error_mm", "cw_error_mm"],
                       ["Nipple", "CW Dist"])
    return results


def _log_error_summary(view: str, results: list, keys: list, names: list) -> None:
    for key, name in zip(keys, names):
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            logger.info(
                "  %s %-12s: %.2f ± %.2f mm  (median: %.2f)",
                view, name, np.mean(vals), np.std(vals), np.median(vals),
            )


# =============================================================================
# Combined MLO + CC quality classifier
# =============================================================================

class CombinedQualityEvaluator:
    """Matches MLO/CC pairs by patient + laterality, classifies positioning quality."""

    def __init__(
        self,
        paths: EvalPathConfig,
        mlo_results: List[dict],
        cc_results:  List[dict],
        labels_df:   pd.DataFrame,
        threshold_mm: float = 10.0,
    ) -> None:
        self.paths           = paths
        self.mlo_results     = mlo_results
        self.cc_results      = cc_results
        self.labels_df       = labels_df
        self.combined_results: List[dict] = []
        self.threshold_mm    = threshold_mm

    # ---- pairing ------------------------------------------------------------

    def match_pairs(self) -> List[dict]:
        logger.info("Matching MLO-CC pairs by StudyInstanceUID + laterality …")
        mlo_dict = {f"{m['patient_id']}_{m['laterality']}": m for m in self.mlo_results}
        matched  = []
        for cc in self.cc_results:
            key = f"{cc['patient_id']}_{cc['laterality']}"
            if key not in mlo_dict:
                continue
            mlo  = mlo_dict[key]
            diff = abs(mlo["distance_mm"] - cc["distance_mm"])
            pred = "good" if diff <= self.threshold_mm else "bad"
            gt   = str(cc.get("gt_quality", "unknown")).lower()
            matched.append({
                "patient_id":          cc["patient_id"],
                "laterality":          cc["laterality"],
                "mlo_sop":             mlo["sop_uid"],
                "cc_sop":              cc["sop_uid"],
                "mlo_distance_mm":     mlo["distance_mm"],
                "cc_distance_mm":      cc["distance_mm"],
                "mlo_gt_distance_mm":  mlo.get("gt_distance_mm"),
                "cc_gt_distance_mm":   cc.get("gt_distance_mm"),
                "distance_diff_mm":    diff,
                "predicted_quality":   pred,
                "gt_quality":          gt,
                "correct_prediction":  pred == gt,
                "mlo_pred_kpts":       mlo.get("pred_kpts_orig"),
                "mlo_gt_kpts":         mlo.get("gt_kpts"),
                "mlo_pred_intersection":mlo.get("pred_intersection"),
                "mlo_gt_intersection": mlo.get("gt_intersection"),
                "cc_pred_nipple":      cc.get("pred_nipple_orig"),
                "cc_gt_nipple":        cc.get("gt_nipple"),
                "cc_laterality":       cc["laterality"],
                "cc_original_width":   cc.get("original_width"),
                "mlo_nipple_error_mm": mlo.get("nipple_error_mm"),
                "mlo_pec1_error_mm":   mlo.get("pec1_error_mm"),
                "mlo_pec2_error_mm":   mlo.get("pec2_error_mm"),
                "mlo_pnl_error_mm":    mlo.get("pnl_error_mm"),
                "cc_nipple_error_mm":  cc.get("nipple_error_mm"),
                "cc_cw_error_mm":      cc.get("cw_error_mm"),
                "mlo_pixel_spacing":   mlo.get("pixel_spacing"),
                "cc_pixel_spacing":    cc.get("pixel_spacing"),
            })
        self.combined_results = matched
        logger.info("Matched pairs: %d", len(matched))
        return matched

    # ---- classification metrics --------------------------------------------

    def calculate_metrics(self, threshold_mm: Optional[float] = None) -> Optional[dict]:
        th = threshold_mm or self.threshold_mm
        if not self.combined_results:
            return None
        y_true, y_pred = [], []
        for r in self.combined_results:
            gt = str(r.get("gt_quality", "unknown")).lower()
            if gt not in ("good", "bad"):
                continue
            y_true.append(gt)
            y_pred.append("good" if r["distance_diff_mm"] <= th else "bad")
        if not y_true:
            return None
        yp = [1 if p == "good" else 0 for p in y_pred]
        yt = [1 if t == "good" else 0 for t in y_true]
        acc          = accuracy_score(yt, yp)
        prec, rec, f1, _ = precision_recall_fscore_support(
            yt, yp, average="weighted", zero_division=0)
        cm           = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        return {
            "threshold_mm":        th,
            "accuracy":            acc,
            "precision":           prec,
            "recall":              rec,
            "f1_score":            f1,
            "sensitivity":         tp / (tp + fn) if (tp + fn) else 0,
            "specificity":         tn / (tn + fp) if (tn + fp) else 0,
            "confusion_matrix":    cm,
            "total_pairs":         len(y_true),
            "correct_predictions": sum(a == b for a, b in zip(y_pred, y_true)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }

    def calculate_dual_threshold_metrics(
        self, bad_threshold: float, good_threshold: float
    ) -> Optional[dict]:
        if not self.combined_results:
            return None
        gt_bad  = [r for r in self.combined_results if str(r.get("gt_quality","")).lower() == "bad"]
        gt_good = [r for r in self.combined_results if str(r.get("gt_quality","")).lower() == "good"]
        if not gt_bad and not gt_good:
            return None
        tn  = sum(1 for r in gt_bad  if r["distance_diff_mm"] >  bad_threshold)
        fp_ = sum(1 for r in gt_bad  if r["distance_diff_mm"] <= bad_threshold)
        fn_ = sum(1 for r in gt_good if r["distance_diff_mm"] >  good_threshold)
        tp  = sum(1 for r in gt_good if r["distance_diff_mm"] <= good_threshold)
        total   = tn + fp_ + fn_ + tp
        correct = tn + tp
        acc  = correct / total if total else 0
        sens = tp / (tp + fn_) if (tp + fn_) else 0
        spec = tn / (tn + fp_) if (tn + fp_) else 0
        pg   = tp / (tp + fp_) if (tp + fp_) else 0
        rg   = tp / (tp + fn_) if (tp + fn_) else 0
        f1g  = 2 * pg * rg / (pg + rg) if (pg + rg) else 0
        pb   = tn / (tn + fn_) if (tn + fn_) else 0
        rb   = tn / (tn + fp_) if (tn + fp_) else 0
        f1b  = 2 * pb * rb / (pb + rb) if (pb + rb) else 0
        n_g, n_b = len(gt_good), len(gt_bad)
        prec_w = (pg * n_g + pb * n_b) / total if total else 0
        rec_w  = (rg * n_g + rb * n_b) / total if total else 0
        f1_w   = (f1g * n_g + f1b * n_b) / total if total else 0
        return {
            "bad_threshold":  bad_threshold,
            "good_threshold": good_threshold,
            "label":          f"{int(bad_threshold)}-{int(good_threshold)}mm",
            "accuracy":       acc,
            "sensitivity":    sens,
            "specificity":    spec,
            "precision":      prec_w,
            "recall":         rec_w,
            "f1_score":       f1_w,
            "precision_good": pg,
            "recall_good":    rg,
            "f1_good":        f1g,
            "confusion_matrix": np.array([[tn, fp_], [fn_, tp]]),
            "total_pairs":    total,
            "correct_predictions": correct,
            "tp": int(tp), "fp": int(fp_), "tn": int(tn), "fn": int(fn_),
            "gt_bad_count":   n_b,
            "gt_good_count":  n_g,
        }

    def compute_error_summary(self) -> dict:
        keys = [
            "mlo_nipple_error_mm", "mlo_pec1_error_mm", "mlo_pec2_error_mm",
            "mlo_pnl_error_mm", "cc_nipple_error_mm", "cc_cw_error_mm",
        ]
        return {
            k.replace("_error_mm", ""): stat_summary(
                [r[k] for r in self.combined_results if r.get(k) is not None]
            )
            for k in keys
        }

    # ---- visualisations -------------------------------------------------

    def create_visualizations(self, metrics: Optional[dict], skip_pair_viz: bool = False) -> None:
        logger.info("Creating visualisations …")
        output = Path(str(self.paths.output_dir).replace(self.paths.view, "combined"))
        output.mkdir(parents=True, exist_ok=True)

        if metrics is not None:
            _plot_confusion_matrix(metrics["confusion_matrix"], output / "confusion_matrix.png")
        _plot_distance_correlation(self.combined_results, self.threshold_mm,
                                   output / "distance_correlation.png")
        _plot_difference_histogram(self.combined_results, self.threshold_mm,
                                   output / "difference_histogram.png")
        _plot_keypoint_error_boxplot(self.combined_results,
                                     output / "keypoint_errors_boxplot.png")
        if not skip_pair_viz:
            self._create_paired_viz(output / "paired_viz")

    def _create_paired_viz(self, viz_dir: Path) -> None:
        if not _PYDICOM_AVAILABLE:
            logger.warning("pydicom not available — skipping paired DICOM visualisations.")
            return
        viz_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Creating %d paired visualisations …", len(self.combined_results))
        for idx, r in enumerate(self.combined_results):
            try:
                self._draw_pair(idx, r, viz_dir)
            except Exception as exc:
                logger.warning("Viz error pair %d: %s", idx, exc)

    def _draw_pair(self, idx: int, r: dict, viz_dir: Path) -> None:
        mlo_dp = find_dicom_path(r["mlo_sop"], self.labels_df, self.paths.raw_dir)
        cc_dp  = find_dicom_path(r["cc_sop"],  self.labels_df, self.paths.raw_dir)
        if mlo_dp is None or cc_dp is None:
            return
        mlo_img = load_dicom_image(mlo_dp)
        cc_img  = load_dicom_image(cc_dp)
        if mlo_img is None or cc_img is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 13))
        ax1.imshow(mlo_img, cmap="gray"); _draw_mlo_kpts(ax1, r); ax1.axis("off")
        ax2.imshow(cc_img,  cmap="gray"); _draw_cc_kpts(ax2, r);  ax2.axis("off")
        col = "#2ecc71" if r["correct_prediction"] else "#e74c3c"
        fig.suptitle(
            f"Pred: {r['predicted_quality'].upper()} | GT: {r['gt_quality'].upper()} | "
            f"|MLO-CC| Diff: {r['distance_diff_mm']:.1f} mm",
            color=col, fontsize=14, fontweight="bold",
        )
        out = viz_dir / f"pair_{idx:03d}_{r['patient_id'][:8]}_{r['laterality']}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close()

    # ---- reports --------------------------------------------------------

    def save_json_report(self, metrics, error_summary, multi_threshold=None) -> None:
        report: dict = {
            "strategy":      self.paths.strategy,
            "threshold_mm":  self.threshold_mm,
            "total_pairs":   len(self.combined_results),
        }
        if metrics:
            report.update({k: v for k, v in metrics.items() if k != "confusion_matrix"})
            report["confusion_matrix"] = metrics["confusion_matrix"].tolist()
        report["keypoint_errors"] = error_summary
        if multi_threshold:
            report["dual_threshold_analysis"] = {}
            for m in multi_threshold.values():
                entry = {k: v for k, v in m.items() if k != "confusion_matrix"}
                entry["confusion_matrix"] = m["confusion_matrix"].tolist()
                report["dual_threshold_analysis"][m["label"]] = entry

        out_dir = Path(str(self.paths.output_dir).replace(self.paths.view, "combined"))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "evaluation_report.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, default=numpy_serializer)
        logger.info("Report saved: %s", path)

    def save_summary_csv(self, metrics, error_summary, multi_threshold=None) -> None:
        out_dir = Path(str(self.paths.output_dir).replace(self.paths.view, "combined"))
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Tuple[str, str]] = []
        if metrics:
            rows += [
                ("=== Single Threshold ===", ""),
                ("Threshold (mm)",        str(self.threshold_mm)),
                ("Accuracy",              f"{metrics['accuracy']*100:.2f}%"),
                ("Sensitivity",           f"{metrics['sensitivity']*100:.2f}%"),
                ("Specificity",           f"{metrics['specificity']*100:.2f}%"),
                ("Precision",             f"{metrics['precision']:.4f}"),
                ("Recall",                f"{metrics['recall']:.4f}"),
                ("F1 Score",              f"{metrics['f1_score']:.4f}"),
                ("TP/FP/TN/FN",
                 f"{metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}"),
                ("Total Pairs",           str(metrics["total_pairs"])),
                ("", ""),
            ]
        if multi_threshold:
            for m in sorted(multi_threshold.values(), key=lambda x: x["label"]):
                rows += [
                    (f"=== Dual Threshold: {m['label']} ===", ""),
                    ("Accuracy",  f"{m['accuracy']*100:.2f}%"),
                    ("Sensitivity", f"{m['sensitivity']*100:.2f}%"),
                    ("Specificity", f"{m['specificity']*100:.2f}%"),
                    ("F1 (Good)",   f"{m['f1_good']:.4f}"),
                    ("", ""),
                ]
        rows.append(("--- Keypoint Errors (mm) ---", ""))
        for key, label in [
            ("mlo_nipple", "MLO Nipple"), ("mlo_pec1", "MLO Pec Top"),
            ("mlo_pec2",  "MLO Pec Bot"), ("mlo_pnl",  "MLO PNL Dist"),
            ("cc_nipple", "CC Nipple"),   ("cc_cw",    "CC CW Dist"),
        ]:
            s = error_summary.get(key, {})
            if s.get("count", 0) > 0:
                rows.append((label, f"{s['mean']:.2f} ± {s['std']:.2f}  median={s['median']:.2f}"))

        pd.DataFrame(rows, columns=["Metric", "Value"]).to_csv(
            out_dir / "summary_metrics.csv", index=False
        )
        logger.info("Summary CSV saved: %s", out_dir / "summary_metrics.csv")


# =============================================================================
# Plotting helpers
# =============================================================================

def _plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
    plt.xlabel("Predicted"); plt.ylabel("Ground Truth")
    plt.title("Quality Classification — Confusion Matrix")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def _plot_distance_correlation(results: list, threshold_mm: float, path: Path) -> None:
    if not results:
        return
    mlo_d  = [r["mlo_distance_mm"] for r in results]
    cc_d   = [r["cc_distance_mm"]  for r in results]
    colors = ["red" if not r["correct_prediction"] else "steelblue" for r in results]
    plt.figure(figsize=(10, 8))
    plt.scatter(mlo_d, cc_d, c=colors, alpha=0.6)
    mn, mx = min(min(mlo_d), min(cc_d)), max(max(mlo_d), max(cc_d))
    plt.plot([mn, mx], [mn, mx], "k--", alpha=0.5)
    plt.xlabel("MLO PNL (mm)"); plt.ylabel("CC Chest-Wall (mm)")
    plt.title("MLO vs CC Distance Correlation")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def _plot_difference_histogram(results: list, threshold_mm: float, path: Path) -> None:
    if not results:
        return
    diffs = [r["distance_diff_mm"] for r in results]
    plt.figure(figsize=(10, 6))
    plt.hist(diffs, bins=30, alpha=0.7)
    plt.axvline(x=threshold_mm, color="red", linestyle="--",
                label=f"Threshold ({threshold_mm} mm)")
    plt.xlabel("|MLO PNL − CC CW| (mm)")
    plt.title("Distance Difference Histogram")
    plt.legend(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def _plot_keypoint_error_boxplot(results: list, path: Path) -> None:
    labels_data = [
        ("MLO\nNipple",   "mlo_nipple_error_mm"),
        ("MLO\nPec Top",  "mlo_pec1_error_mm"),
        ("MLO\nPec Bot",  "mlo_pec2_error_mm"),
        ("MLO\nPNL Dist", "mlo_pnl_error_mm"),
        ("CC\nNipple",    "cc_nipple_error_mm"),
        ("CC\nCW Dist",   "cc_cw_error_mm"),
    ]
    box_data, box_labels = [], []
    for lbl, key in labels_data:
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            box_data.append(vals); box_labels.append(lbl)
    if not box_data:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.6)
    palette = ["#FF9999", "#FF6666", "#CC3333", "#FF4444", "#66B2FF", "#3399FF"]
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Error (mm)"); ax.set_title("Per-Keypoint Localisation Errors")
    ax.grid(True, alpha=0.3)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def _draw_mlo_kpts(ax, r: dict) -> None:
    gt_kpts   = r.get("mlo_gt_kpts")
    pred_kpts = r.get("mlo_pred_kpts")
    if gt_kpts:
        gn, gp1, gp2 = gt_kpts["nipple"], gt_kpts["pectoral_top"], gt_kpts["pectoral_bottom"]
        ax.plot([gp1[0], gp2[0]], [gp1[1], gp2[1]], color="#FF4444", lw=2.0, alpha=0.85)
        gi = r.get("mlo_gt_intersection")
        if gi: ax.plot([gn[0], gi[0]], [gn[1], gi[1]], color="#FF4444", lw=1.5, ls="--", alpha=0.85)
        ax.plot(*gn,  "o", color="#FF4444", ms=11, mec="white", mew=1.5, label="GT Nipple",  zorder=5)
        ax.plot(*gp1, "s", color="#FF4444", ms=9,  mec="white", mew=1,   label="GT Pec Top", zorder=5)
        ax.plot(*gp2, "D", color="#FF4444", ms=9,  mec="white", mew=1,   label="GT Pec Bot", zorder=5)
    if pred_kpts and len(pred_kpts) >= 3:
        pn, pp1, pp2 = pred_kpts[0], pred_kpts[1], pred_kpts[2]
        ax.plot([pp1[0], pp2[0]], [pp1[1], pp2[1]], color="#00FF00", lw=2.0, alpha=0.85)
        pi = r.get("mlo_pred_intersection")
        if pi: ax.plot([pn[0], pi[0]], [pn[1], pi[1]], color="#00FF00", lw=1.5, ls="--", alpha=0.85)
        ax.plot(*pn,  "o", color="#00FF00", ms=11, mec="white", mew=1.5, label="Pred Nipple",  zorder=6)
        ax.plot(*pp1, "s", color="#00FF00", ms=9,  mec="white", mew=1,   label="Pred Pec Top", zorder=6)
        ax.plot(*pp2, "D", color="#00FF00", ms=9,  mec="white", mew=1,   label="Pred Pec Bot", zorder=6)
    err_parts = []
    for key, lbl in [("mlo_nipple_error_mm", "Nipple"), ("mlo_pec1_error_mm", "Pec1"),
                     ("mlo_pec2_error_mm", "Pec2"),     ("mlo_pnl_error_mm", "PNL")]:
        if r.get(key) is not None:
            err_parts.append(f"{lbl}: {r[key]:.1f}mm")
    if err_parts:
        ax.text(0.02, 0.02, " | ".join(err_parts), transform=ax.transAxes, fontsize=9,
                va="bottom", bbox=dict(boxstyle="round,pad=0.4", fc="black", alpha=0.75), color="white")
    pnl_pred = r.get("mlo_distance_mm", 0)
    pnl_gt   = r.get("mlo_gt_distance_mm")
    title = f"MLO ({r['laterality']})  PNL Pred: {pnl_pred:.1f}mm"
    if pnl_gt is not None: title += f" | GT: {pnl_gt:.1f}mm"
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)


def _draw_cc_kpts(ax, r: dict) -> None:
    pred_nip = r.get("cc_pred_nipple")
    gt_nip   = r.get("cc_gt_nipple")
    lat      = r.get("cc_laterality", r["laterality"])
    orig_w   = r.get("cc_original_width", 1)
    edge_x   = 0 if lat == "L" else orig_w
    if gt_nip:
        ax.plot(gt_nip[0], gt_nip[1], "o", color="#FF4444", ms=11, mec="white", mew=1.5,
                label="GT Nipple", zorder=5)
        ax.plot([gt_nip[0], edge_x], [gt_nip[1], gt_nip[1]],
                color="#FF4444", lw=1.5, ls="--", alpha=0.85)
    if pred_nip:
        ax.plot(pred_nip[0], pred_nip[1], "o", color="#00FF00", ms=11, mec="white", mew=1.5,
                label="Pred Nipple", zorder=6)
        ax.plot([pred_nip[0], edge_x], [pred_nip[1], pred_nip[1]],
                color="#00FF00", lw=1.5, ls="--", alpha=0.85)
    err_parts = []
    for key, lbl in [("cc_nipple_error_mm", "Nipple"), ("cc_cw_error_mm", "CW Dist")]:
        if r.get(key) is not None:
            err_parts.append(f"{lbl}: {r[key]:.1f}mm")
    if err_parts:
        ax.text(0.02, 0.02, " | ".join(err_parts), transform=ax.transAxes, fontsize=9,
                va="bottom", bbox=dict(boxstyle="round,pad=0.4", fc="black", alpha=0.75), color="white")
    cw_pred = r.get("cc_distance_mm", 0)
    cw_gt   = r.get("cc_gt_distance_mm")
    title = f"CC ({lat})  CW Pred: {cw_pred:.1f}mm"
    if cw_gt is not None: title += f" | GT: {cw_gt:.1f}mm"
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)


# =============================================================================
# CLI
# =============================================================================

def _print_metrics(metrics: dict, label: str = "") -> None:
    tag = f" [{label}]" if label else ""
    logger.info("=" * 60)
    logger.info("CLASSIFICATION RESULTS%s", tag)
    logger.info("  Threshold   : %.0f mm",  metrics["threshold_mm"])
    logger.info("  Total pairs : %d",        metrics["total_pairs"])
    logger.info("  Accuracy    : %.2f %%",   metrics["accuracy"]    * 100)
    logger.info("  Sensitivity : %.2f %%",   metrics["sensitivity"] * 100)
    logger.info("  Specificity : %.2f %%",   metrics["specificity"] * 100)
    logger.info("  Precision   : %.4f",      metrics["precision"])
    logger.info("  Recall      : %.4f",      metrics["recall"])
    logger.info("  F1 Score    : %.4f",      metrics["f1_score"])
    logger.info("  TP/FP/TN/FN : %d/%d/%d/%d",
                metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"])
    logger.info("=" * 60)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate_pose.py",
        description=(
            "Unified YOLO Pose Evaluation — config-driven\n\n"
            "Single view:\n"
            "  python evaluate_pose.py --config configs/baseline_mlo.yaml\n\n"
            "Combined MLO + CC:\n"
            "  python evaluate_pose.py \\\n"
            "      --mlo_config configs/baseline_mlo.yaml \\\n"
            "      --cc_config  configs/baseline_cc.yaml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",        type=str, default=None,
                   help="Single-view YAML config (MLO or CC).")
    p.add_argument("--mlo_config",    type=str, default=None,
                   help="MLO YAML config (combined mode).")
    p.add_argument("--cc_config",     type=str, default=None,
                   help="CC YAML config (combined mode).")
    p.add_argument("--threshold",     type=float, default=10.0,
                   help="Clinical threshold in mm (default: 10).")
    p.add_argument("--dual_thresholds", action="store_true",
                   help="Also compute dual-threshold analysis (9/11, 8/12 mm).")
    p.add_argument("--skip_viz",      action="store_true",
                   help="Skip per-pair DICOM visualisations.")
    p.add_argument("--output_dir",    type=str, default=None,
                   help="Override output directory.")
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if not args.config and not (args.mlo_config and args.cc_config):
        parser.print_help(); sys.exit(1)

    # ---- load labels -------------------------------------------------------
    labels_dir  = PROJECT_ROOT / "data" / "labels"
    mlo_lbl_csv = labels_dir / "mlo_labels.csv"
    cc_lbl_csv  = labels_dir / "cc_labels.csv"
    if not mlo_lbl_csv.exists() or not cc_lbl_csv.exists():
        logger.error("Label CSVs not found in %s", labels_dir); sys.exit(1)
    mlo_labels = pd.read_csv(mlo_lbl_csv)
    cc_labels  = pd.read_csv(cc_lbl_csv)
    mlo_labels["SOPInstanceUID"] = mlo_labels["SOPInstanceUID"].astype(str)
    cc_labels["SOPInstanceUID"]  = cc_labels["SOPInstanceUID"].astype(str)
    all_labels = pd.concat([mlo_labels, cc_labels], ignore_index=True)

    # ---- single-view mode --------------------------------------------------
    if args.config:
        cfg   = ExperimentConfig.from_yaml(args.config)
        paths = EvalPathConfig(cfg, args.output_dir)
        paths.setup_directories()
        issues = paths.validate()
        if issues:
            for iss in issues: logger.warning("Path issue: %s", iss)

        if cfg.view == "MLO":
            results = evaluate_mlo(paths, all_labels)
        else:
            results = evaluate_cc(paths, all_labels)

        out = paths.output_dir / "results.json"
        with out.open("w") as f:
            json.dump(results, f, indent=2, default=numpy_serializer)
        logger.info("Results saved: %s  (%d records)", out, len(results))
        return

    # ---- combined mode -----------------------------------------------------
    mlo_cfg   = ExperimentConfig.from_yaml(args.mlo_config)
    cc_cfg    = ExperimentConfig.from_yaml(args.cc_config)
    mlo_paths = EvalPathConfig(mlo_cfg, args.output_dir)
    cc_paths  = EvalPathConfig(cc_cfg,  args.output_dir)
    mlo_paths.setup_directories()
    cc_paths.setup_directories()

    for paths in (mlo_paths, cc_paths):
        for iss in paths.validate():
            logger.warning("Path issue: %s", iss)

    mlo_results = evaluate_mlo(mlo_paths, all_labels, mlo_paths.mlo_model_path)
    cc_results  = evaluate_cc(cc_paths,   all_labels, cc_paths.cc_model_path)

    # persist per-view JSONs
    out_base = PROJECT_ROOT / "evaluation_results" / mlo_cfg.strategy
    (out_base / "MLO").mkdir(parents=True, exist_ok=True)
    (out_base / "CC").mkdir(parents=True,  exist_ok=True)
    with (out_base / "MLO" / "results.json").open("w") as f:
        json.dump(mlo_results, f, indent=2, default=numpy_serializer)
    with (out_base / "CC" / "results.json").open("w") as f:
        json.dump(cc_results, f, indent=2, default=numpy_serializer)

    combined = CombinedQualityEvaluator(
        mlo_paths, mlo_results, cc_results, all_labels, args.threshold
    )
    combined.match_pairs()
    metrics       = combined.calculate_metrics()
    error_summary = combined.compute_error_summary()

    if metrics:
        _print_metrics(metrics)

    multi_threshold = {}
    if args.dual_thresholds:
        for bad_t, good_t in [(9, 11), (8, 12)]:
            m = combined.calculate_dual_threshold_metrics(bad_t, good_t)
            if m is not None:
                multi_threshold[(bad_t, good_t)] = m
                _print_metrics(m, label=m["label"])

    combined.create_visualizations(metrics, skip_pair_viz=args.skip_viz)
    combined.save_json_report(metrics, error_summary, multi_threshold or None)
    combined.save_summary_csv(metrics, error_summary, multi_threshold or None)

    with (out_base / "combined" / "combined_results.json"
          if (out_base / "combined").exists() else
          out_base / "combined_results.json").open("w", encoding="utf-8") as f:
        json.dump(combined.combined_results, f, indent=2, default=numpy_serializer)

    logger.info("Done. MLO: %d  CC: %d  Pairs: %d",
                len(mlo_results), len(cc_results), len(combined.combined_results))


if __name__ == "__main__":
    main()
