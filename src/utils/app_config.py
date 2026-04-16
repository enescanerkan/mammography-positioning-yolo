"""
Application-level configuration with environment variable override support.

Loads ``configs/app_config.yaml`` as a base and allows any value to be
overridden via environment variables using a flat ``SECTION_KEY`` naming
convention (e.g. ``APP_ENV``, ``PATHS_DATA_ROOT``, ``TRAINING_DEVICE``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "app_config.yaml"

_ENV_MAP: Dict[str, tuple] = {
    "APP_ENV":                ("app", "env"),
    "APP_NAME":               ("app", "name"),
    "APP_VERSION":            ("app", "version"),
    "DATA_ROOT":              ("paths", "data_root"),
    "RAW_DATA_PATH":          ("paths", "raw_data"),
    "PROCESSED_DATA_PATH":    ("paths", "processed_data"),
    "LABELS_PATH":            ("paths", "labels"),
    "WEIGHTS_DIR":            ("paths", "weights_dir"),
    "EXPERIMENTS_DIR":        ("paths", "experiments_dir"),
    "EVALUATION_DIR":         ("paths", "evaluation_dir"),
    "LOG_LEVEL":              ("logging", "level"),
    "LOG_FORMAT":             ("logging", "format"),
    "LOG_FILE":               ("logging", "file"),
    "TRAINING_DEVICE":        ("training", "device"),
    "TRAINING_WORKERS":       ("training", "workers"),
    "TRAINING_BATCH":         ("training", "batch"),
    "TRAINING_IMGSZ":         ("training", "imgsz"),
    "TRAINING_DEFAULT_CONFIG":("training", "default_config"),
    "INFERENCE_CONF":         ("inference", "confidence_threshold"),
    "INFERENCE_NMS":          ("inference", "nms_threshold"),
    "INFERENCE_DEVICE":       ("inference", "device"),
    "EVAL_THRESHOLD_MM":      ("evaluation", "clinical_threshold_mm"),
    "EVAL_DUAL_THRESHOLDS":   ("evaluation", "dual_thresholds"),
    "EVAL_SKIP_VIZ":          ("evaluation", "skip_viz"),
}


def _cast_value(value: str, existing: Any) -> Any:
    """Attempt to cast an env-var string to the same type as the YAML default."""
    if isinstance(existing, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(existing, int):
        return int(value)
    if isinstance(existing, float):
        return float(value)
    return value


class AppConfig:
    """Singleton-like application config with env-var override support."""

    _instance: Optional["AppConfig"] = None

    def __init__(self, config_path: Optional[Path] = None) -> None:
        path = config_path or _DEFAULT_CONFIG_PATH
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                self._data: Dict[str, Any] = yaml.safe_load(fh) or {}
        else:
            self._data = {}

        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        for env_key, (section, key) in _ENV_MAP.items():
            env_val = os.environ.get(env_key)
            if env_val is None:
                continue
            if section not in self._data:
                self._data[section] = {}
            existing = self._data[section].get(key)
            self._data[section][key] = _cast_value(env_val, existing) if existing is not None else env_val

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._data.get(section, {}).get(key, default)

    @property
    def app_env(self) -> str:
        return self.get("app", "env", "development")

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def log_level(self) -> str:
        return self.get("logging", "level", "INFO")

    @property
    def data_root(self) -> Path:
        return Path(self.get("paths", "data_root", "data"))

    @property
    def weights_dir(self) -> Path:
        return Path(self.get("paths", "weights_dir", "weights"))

    @property
    def experiments_dir(self) -> Path:
        return Path(self.get("paths", "experiments_dir", "experiments"))

    @property
    def raw(self) -> Dict[str, Any]:
        return self._data

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "AppConfig":
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (useful for testing)."""
        cls._instance = None
