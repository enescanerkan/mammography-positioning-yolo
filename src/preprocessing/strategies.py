"""
Enhancement strategies for mammography preprocessing.

Each class implements :class:`~src.core.interfaces.IEnhancementStrategy`.
Strategies are pure image functions — they do not alter spatial dimensions,
so coordinate transforms in the pipeline are completely unaffected.

Available strategies
--------------------
- ``BaselineStrategy``       – No-op.  Returns the image unchanged.
- ``HistEqStrategy``         – Global histogram equalisation (cv2.equalizeHist).
- ``AdvancedWaveletStrategy``– NLM denoise → Daubechies wavelet edge boost → CLAHE.
"""
from __future__ import annotations

import cv2
import numpy as np
import pywt

from ..core.interfaces import IEnhancementStrategy
from ..utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Strategy map (used by the pipeline to select from a string key)
# ---------------------------------------------------------------------------

STRATEGY_MAP: dict[str, type] = {}  # populated by the registry decorator


def _register(name: str):
    """Class decorator that registers a strategy under a lookup key."""
    def decorator(cls):
        STRATEGY_MAP[name] = cls
        return cls
    return decorator


def get_strategy(name: str) -> IEnhancementStrategy:
    """
    Instantiate a registered strategy by its string key.

    Args:
        name: One of ``'baseline'``, ``'histeq'``, or ``'wavelet'``.

    Returns:
        Instantiated strategy object.

    Raises:
        ValueError: If the name is not a registered strategy.
    """
    if name not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {list(STRATEGY_MAP.keys())}"
        )
    return STRATEGY_MAP[name]()


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

@_register("baseline")
class BaselineStrategy(IEnhancementStrategy):
    """
    No-op enhancement strategy.

    Returns the normalised 8-bit image unchanged.  Used as a control
    baseline to isolate the effect of enhancement on model performance.
    """

    def run(self, image: np.ndarray) -> np.ndarray:
        """Return the image without any modification."""
        return image


@_register("histeq")
class HistEqStrategy(IEnhancementStrategy):
    """
    Global Histogram Equalisation strategy.

    Applies ``cv2.equalizeHist`` to redistribute pixel intensities
    across the full [0, 255] range.  Increases global contrast but can
    over-amplify noise in homogeneous regions.
    """

    def run(self, image: np.ndarray) -> np.ndarray:
        """Apply global histogram equalisation."""
        logger.debug("Applying Global Histogram Equalisation")
        return cv2.equalizeHist(image)


@_register("wavelet")
class AdvancedWaveletStrategy(IEnhancementStrategy):
    """
    Advanced three-stage enhancement optimised for mammography.

    Pipeline
    --------
    1. **NLM Spatial Denoising** – removes sensor noise while preserving
       anatomical edges (small window sizes for efficiency at 640 px).
    2. **Wavelet Edge Boosting** – Daubechies-4 two-level decomposition;
       detail coefficients are amplified to accentuate the pectoral
       muscle boundary and the nipple marker.
    3. **CLAHE** – local contrast equalisation on an 8×8 tile grid,
       preventing over-amplification (clip limit 2.5).

    This strategy was experimentally found to yield the best keypoint
    localisation accuracy on the VinDr-Mammo dataset.
    """

    def __init__(self) -> None:
        # Optimised CLAHE parameters for digital mammography contrast
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def run(self, image: np.ndarray) -> np.ndarray:
        """Apply NLM + Wavelet edge boost + CLAHE enhancement."""
        logger.debug("Applying Advanced Wavelet Strategy (NLM + Wavelet + CLAHE)")

        # Stage 1 — Non-Local Means denoising
        # Window size 5 keeps runtime acceptable at 640×640 resolution.
        denoised = cv2.fastNlMeansDenoising(
            image, None, h=8, templateWindowSize=5, searchWindowSize=11
        )

        # Stage 2 — Wavelet edge boosting (Daubechies-4, level 2)
        img_f = denoised.astype(np.float64)
        coeffs = pywt.wavedec2(img_f, "db4", level=2)
        boosted_coeffs = [coeffs[0]]  # approximation layer unchanged

        for level_idx, (cH, cV, cD) in enumerate(coeffs[1:], start=1):
            # pywt order: coeffs[1] is the coarsest (level 2), coeffs[2] the finest (level 1).
            # Coarser detail gets x1.3, finer detail x1.6.
            boost_factor = 1.3 if level_idx == 1 else 1.6
            boosted_coeffs.append((cH * boost_factor, cV * boost_factor, cD * boost_factor))

        reconstructed = pywt.waverec2(boosted_coeffs, "db4")

        # Trim any 1-pixel boundary artefact introduced by wavelet padding
        reconstructed = reconstructed[: image.shape[0], : image.shape[1]]
        enhanced = np.clip(reconstructed, 0, 255).astype(np.uint8)

        # Stage 3 — CLAHE for local contrast normalisation
        return self._clahe.apply(enhanced)
