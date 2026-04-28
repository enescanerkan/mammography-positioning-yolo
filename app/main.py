"""
FastAPI application for the Mammography Positioning YOLO pipeline.

Endpoints:
    POST /predict   — MLO + CC DICOM paths → quality assessment
    POST /train     — Trigger training pipeline in background
    GET  /health    — Service & model health check
    GET  /status    — Current training job status
"""
from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.model_manager import ModelManager
from app.schemas import (
    CCResult,
    HealthResponse,
    KeypointResult,
    MLOResult,
    PredictRequest,
    PredictResponse,
    QualityLabel,
    TrainRequest,
    TrainResponse,
    TrainStatusResponse,
)
from src.utils.logger import get_logger

logger = get_logger("api")

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Mammography Positioning API",
    description="Breast positioning quality assessment via YOLO pose estimation",
    version="1.0.0",
)

manager = ModelManager.get_instance()


@app.get("/")
def root() -> RedirectResponse:
    """Root URL has no API; send users to interactive docs."""
    return RedirectResponse(url="/docs")


# Training state (in-process; single job at a time)
_train_state = {
    "is_training": False,
    "job_id": None,
    "config_path": None,
    "mode": None,
    "started_at": None,
    "error": None,
}
_train_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Startup: load models
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup_load_models() -> None:
    mlo_path = os.environ.get(
        "MLO_MODEL_PATH",
        str(PROJECT_ROOT / "weights" / "mlo-yolo26-pose-advanced.pt"),
    )
    cc_path = os.environ.get(
        "CC_MODEL_PATH",
        str(PROJECT_ROOT / "weights" / "cc-yolo26-pose-advanced.pt"),
    )

    if Path(mlo_path).exists():
        manager.load_mlo_model(mlo_path)
    else:
        logger.warning("MLO model not found at startup: %s", mlo_path)

    if Path(cc_path).exists():
        manager.load_cc_model(cc_path)
    else:
        logger.warning("CC model not found at startup: %s", cc_path)


# ---------------------------------------------------------------------------
# Geometry helpers (same logic as evaluate_pose.py)
# ---------------------------------------------------------------------------

def _pnl_distance(nipple, p1, p2) -> tuple[float, list]:
    n = np.array(nipple, dtype=np.float64)
    a = np.array(p1, dtype=np.float64)
    b = np.array(p2, dtype=np.float64)
    v = b - a
    vl = np.linalg.norm(v)
    if vl == 0:
        return float(np.linalg.norm(n - a)), a.tolist()
    proj = a + np.dot(n - a, v) / (vl * vl) * v
    return float(np.linalg.norm(n - proj)), proj.tolist()


def _chest_wall_distance(nipple_x: float, laterality: str, original_width: float) -> float:
    return float(nipple_x) if laterality == "L" else float(abs(original_width - nipple_x))


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if manager.mlo_model is None:
        raise HTTPException(status_code=503, detail="MLO model not loaded")
    if manager.cc_model is None:
        raise HTTPException(status_code=503, detail="CC model not loaded")

    if not Path(req.mlo_dicom_path).exists():
        raise HTTPException(status_code=404, detail=f"MLO DICOM not found: {req.mlo_dicom_path}")
    if not Path(req.cc_dicom_path).exists():
        raise HTTPException(status_code=404, detail=f"CC DICOM not found: {req.cc_dicom_path}")

    mlo_img = manager.load_dicom_as_image(req.mlo_dicom_path)
    cc_img = manager.load_dicom_as_image(req.cc_dicom_path)
    if mlo_img is None:
        raise HTTPException(status_code=422, detail="Failed to read MLO DICOM")
    if cc_img is None:
        raise HTTPException(status_code=422, detail="Failed to read CC DICOM")

    # MLO inference (3 keypoints: nipple, pec_top, pec_bottom)
    mlo_kpts = manager.predict_mlo(mlo_img)
    if mlo_kpts is None or len(mlo_kpts) < 3:
        raise HTTPException(status_code=422, detail="MLO model could not detect 3 keypoints")

    nipple_mlo = mlo_kpts[0].tolist()
    pec_top = mlo_kpts[1].tolist()
    pec_bottom = mlo_kpts[2].tolist()
    pnl_px, _ = _pnl_distance(nipple_mlo, pec_top, pec_bottom)
    pnl_mm = pnl_px * req.pixel_spacing

    # CC inference (1 keypoint: nipple)
    cc_kpts = manager.predict_cc(cc_img)
    if cc_kpts is None or len(cc_kpts) < 1:
        raise HTTPException(status_code=422, detail="CC model could not detect nipple keypoint")

    nipple_cc = cc_kpts[0].tolist()
    cc_width = float(cc_img.shape[1])
    cw_px = _chest_wall_distance(nipple_cc[0], req.laterality, cc_width)
    cw_mm = cw_px * req.pixel_spacing

    # Quality decision
    diff_mm = abs(pnl_mm - cw_mm)
    quality = QualityLabel.good if diff_mm <= req.threshold_mm else QualityLabel.bad

    return PredictResponse(
        quality=quality,
        mlo=MLOResult(
            nipple=KeypointResult(x=nipple_mlo[0], y=nipple_mlo[1]),
            pectoral_top=KeypointResult(x=pec_top[0], y=pec_top[1]),
            pectoral_bottom=KeypointResult(x=pec_bottom[0], y=pec_bottom[1]),
            pnl_distance_mm=round(pnl_mm, 2),
            pnl_distance_px=round(pnl_px, 2),
        ),
        cc=CCResult(
            nipple=KeypointResult(x=nipple_cc[0], y=nipple_cc[1]),
            chest_wall_distance_mm=round(cw_mm, 2),
            chest_wall_distance_px=round(cw_px, 2),
        ),
        distance_diff_mm=round(diff_mm, 2),
        threshold_mm=req.threshold_mm,
        laterality=req.laterality,
        message=f"Positioning quality: {quality.value.upper()} (|MLO PNL - CC CW| = {diff_mm:.1f} mm)",
    )


# ---------------------------------------------------------------------------
# POST /train
# ---------------------------------------------------------------------------

def _run_training(job_id: str, config_path: str, mode: str, augment: bool, augment_splits: str):
    """Background training thread."""
    try:
        from src.utils.config import ExperimentConfig
        from src.preprocessing.pipeline import PreprocessingPipeline
        from src.preprocessing.strategies import get_strategy
        from src.training.trainer import YoloTrainer
        from src.augmentation.flip_augmenter import FlipAugmenter
        import yaml

        cfg = ExperimentConfig.from_yaml(config_path)
        splits = [s.strip() for s in augment_splits.split(",") if s.strip()]

        if mode in ("preprocess", "full"):
            strategy = get_strategy(cfg.strategy)
            pipeline = PreprocessingPipeline(
                data_dir=str(cfg.data_path),
                output_dir=str(cfg.processed_data_dir),
                labels_csv=cfg.label_files,
                strategy=strategy,
            )
            pipeline.run(view_mode=cfg.view)
            if augment:
                aug = FlipAugmenter(str(cfg.processed_data_dir), view_type=cfg.view)
                aug.run(splits=splits)

        if mode == "augment":
            aug = FlipAugmenter(str(cfg.processed_data_dir), view_type=cfg.view)
            aug.run(splits=splits)

        if mode in ("train", "full"):
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

            trainer = YoloTrainer(model_path=cfg.model)
            trainer.train(
                data_yaml=str(yaml_path),
                project_dir=str(cfg.experiments_dir),
                **cfg.training_args,
            )

        logger.info("Training job %s completed successfully", job_id)
    except Exception as exc:
        logger.error("Training job %s failed: %s", job_id, exc)
        with _train_lock:
            _train_state["error"] = str(exc)
    finally:
        with _train_lock:
            _train_state["is_training"] = False


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    with _train_lock:
        if _train_state["is_training"]:
            raise HTTPException(
                status_code=409,
                detail=f"Training already in progress: job_id={_train_state['job_id']}",
            )

    config_path = str(PROJECT_ROOT / req.config_path)
    if not Path(config_path).exists():
        raise HTTPException(status_code=404, detail=f"Config not found: {req.config_path}")

    job_id = str(uuid.uuid4())[:8]

    with _train_lock:
        _train_state.update({
            "is_training": True,
            "job_id": job_id,
            "config_path": req.config_path,
            "mode": req.mode.value,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": None,
        })

    thread = threading.Thread(
        target=_run_training,
        args=(job_id, config_path, req.mode.value, req.augment, req.augment_splits),
        daemon=True,
    )
    thread.start()

    return TrainResponse(
        status="started",
        message=f"Training started in background (job_id: {job_id})",
        job_id=job_id,
        config_path=req.config_path,
        mode=req.mode.value,
    )


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------

@app.get("/status", response_model=TrainStatusResponse)
def training_status():
    with _train_lock:
        state = dict(_train_state)

    elapsed = None
    if state["started_at"] and state["is_training"]:
        started = time.mktime(time.strptime(state["started_at"], "%Y-%m-%d %H:%M:%S"))
        elapsed = round(time.time() - started, 1)

    msg = "Training in progress" if state["is_training"] else "No active training"
    if state.get("error"):
        msg = f"Last training failed: {state['error']}"

    return TrainStatusResponse(
        is_training=state["is_training"],
        job_id=state["job_id"],
        config_path=state["config_path"],
        mode=state["mode"],
        started_at=state["started_at"],
        elapsed_seconds=elapsed,
        message=msg,
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    gpu_available = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if gpu_available else "CPU"

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        mlo_model_loaded=manager.mlo_model is not None,
        cc_model_loaded=manager.cc_model is not None,
        gpu_available=gpu_available,
        device=device,
    )
