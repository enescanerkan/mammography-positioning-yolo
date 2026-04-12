import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Union
from ..utils.logger import get_logger

logger = get_logger(__name__)

class DicomLoader:
    """
    Handles loading and basic normalization of DICOM files.
    """
    
    @staticmethod
    def load(file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Loads a DICOM file and returns the pixel array and metadata.
        
        Args:
            file_path: Path to the DICOM file.
            
        Returns:
            Tuple containing:
                - pixel_array: Numpy array of the image (float32).
                - metadata: Dictionary containing DICOM tags.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"DICOM file not found: {path}")
            
        try:
            ds = pydicom.dcmread(str(path))
            
            # Apply VOI LUT for proper windowing (same as test script)
            # This applies Window/Level settings from DICOM for better contrast
            pixel_array = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
            
            # Get PhotometricInterpretation - DO NOT invert here!
            # Inversion will be handled in pipeline.py to avoid double inversion
            # MONOCHROME1: high values = dark, low values = bright
            # MONOCHROME2: high values = bright, low values = dark (standard)
            photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
            
            # Handle Pixel Spacing
            pixel_spacing = [0.085, 0.085] # Default for VinDr-Mammo if missing
            if hasattr(ds, 'ImagerPixelSpacing'):
                pixel_spacing = [float(x) for x in ds.ImagerPixelSpacing]
            elif hasattr(ds, 'PixelSpacing'):
                pixel_spacing = [float(x) for x in ds.PixelSpacing]
            
            # Get laterality
            laterality = 'U'
            if hasattr(ds, 'ImageLaterality'):
                laterality = ds.ImageLaterality
            elif hasattr(ds, 'Laterality'):
                laterality = ds.Laterality
                
            metadata = {
                'SOPInstanceUID': getattr(ds, 'SOPInstanceUID', 'Unknown'),
                'SeriesDescription': getattr(ds, 'SeriesDescription', ''),
                'pixel_spacing': pixel_spacing,
                'photometric_interpretation': photometric,
                'ImageLaterality': laterality,
                'original_path': str(path)
            }
            
            return pixel_array, metadata
            
        except Exception as e:
            logger.error(f"Failed to read DICOM {path}: {e}")
            raise

    @staticmethod
    def normalize_to_8bit(image: np.ndarray) -> np.ndarray:
        """
        Normalizes a high-bit depth image to 8-bit using min-max scaling with outlier protection.
        Result: breast tissue = bright (high values), background = dark (low values ~0)
        """
        p1, p99 = np.percentile(image, [1, 99])
        img = np.clip(image, p1, p99)
        img = ((img - p1) / (p99 - p1) * 255).astype(np.uint8)
        return img
