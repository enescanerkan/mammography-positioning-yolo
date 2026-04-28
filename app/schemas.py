from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PipelineMode(str, Enum):
    preprocess = "preprocess"
    augment = "augment"
    train = "train"
    full = "full"


class ViewType(str, Enum):
    MLO = "MLO"
    CC = "CC"


class QualityLabel(str, Enum):
    good = "good"
    bad = "bad"
    unknown = "unknown"


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    mlo_dicom_path: str = Field(..., description="MLO DICOM dosyasının sunucudaki tam yolu")
    cc_dicom_path: str = Field(..., description="CC DICOM dosyasının sunucudaki tam yolu")
    laterality: str = Field("L", description="Taraf: L veya R")
    pixel_spacing: float = Field(0.085, description="Piksel aralığı (mm)")
    threshold_mm: float = Field(10.0, description="Klinik karar eşiği (mm)")

    model_config = {"json_schema_extra": {
        "examples": [{
            "mlo_dicom_path": "/seas_data/mammography/dicoms/study1/mlo.dicom",
            "cc_dicom_path": "/seas_data/mammography/dicoms/study1/cc.dicom",
            "laterality": "L",
            "pixel_spacing": 0.085,
            "threshold_mm": 10.0,
        }]
    }}


class KeypointResult(BaseModel):
    x: float
    y: float
    confidence: Optional[float] = None


class MLOResult(BaseModel):
    nipple: KeypointResult
    pectoral_top: KeypointResult
    pectoral_bottom: KeypointResult
    pnl_distance_mm: float
    pnl_distance_px: float


class CCResult(BaseModel):
    nipple: KeypointResult
    chest_wall_distance_mm: float
    chest_wall_distance_px: float


class PredictResponse(BaseModel):
    quality: QualityLabel
    mlo: MLOResult
    cc: CCResult
    distance_diff_mm: float
    threshold_mm: float
    laterality: str
    message: str


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    config_path: str = Field(
        "configs/wavelet_mlo.yaml",
        description="Experiment YAML config dosyası yolu",
    )
    mode: PipelineMode = Field(PipelineMode.full, description="Pipeline modu")
    augment: bool = Field(False, description="Augmentation uygula")
    augment_splits: str = Field("train", description="Augment edilecek split'ler (virgülle ayır)")


class TrainResponse(BaseModel):
    status: str
    message: str
    job_id: str
    config_path: str
    mode: str


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class TrainStatusResponse(BaseModel):
    is_training: bool
    job_id: Optional[str] = None
    config_path: Optional[str] = None
    mode: Optional[str] = None
    started_at: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    mlo_model_loaded: bool
    cc_model_loaded: bool
    gpu_available: bool
    device: str
