"""
Data manager for handling images and pixel spacing data.
"""

import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

from preprocessing.preprocessing_utils import ImagePreprocessor


class IDataManager(ABC):
    """Interface for data managers."""
    
    @abstractmethod
    def load_image_pair(self, mlo_path: str, cc_path: str) -> None:
        """Load MLO and CC image pair."""
        pass
    
    @abstractmethod
    def load_pixel_spacing_data(self) -> None:
        """Load pixel spacing data."""
        pass
    
    @abstractmethod
    def get_pixel_spacing(self, image_type: str) -> float:
        """Get pixel spacing for image type."""
        pass


class DataManager(IDataManager):
    """Manager for image data and pixel spacing information."""
    
    def __init__(self):
        """Initialize the data manager."""
        self.current_mlo_image: Optional[np.ndarray] = None
        self.current_cc_image: Optional[np.ndarray] = None
        self.current_mlo_filename: Optional[str] = None
        self.current_cc_filename: Optional[str] = None
        self.pixel_spacing_data: Dict[str, Dict[str, Any]] = {}
        
        self.current_mlo_file_type: Optional[str] = None
        self.current_cc_file_type: Optional[str] = None
        self.current_mlo_original_pixel_spacing: Optional[tuple] = None
        self.current_cc_original_pixel_spacing: Optional[tuple] = None
        self.current_mlo_original_shape: Optional[tuple] = None
        self.current_cc_original_shape: Optional[tuple] = None
        
        self.current_mlo_transformation_info: Optional[Dict] = None
        self.current_cc_transformation_info: Optional[Dict] = None
        
        self.preprocessor = ImagePreprocessor()
        
    def load_image_pair(self, mlo_path: str, cc_path: str) -> None:
        """Load MLO and CC image pair from file paths."""
        if not os.path.exists(mlo_path):
            raise FileNotFoundError(f"MLO file not found: {mlo_path}")
        if not os.path.exists(cc_path):
            raise FileNotFoundError(f"CC file not found: {cc_path}")
        
        try:
            result = self._load_single_image(mlo_path)
            self.current_mlo_image = result[0]
            self.current_mlo_file_type = result[1]
            self.current_mlo_original_pixel_spacing = result[2]
            self.current_mlo_original_shape = result[3]
            self.current_mlo_transformation_info = result[4] if len(result) > 4 else None
            self.current_mlo_filename = os.path.basename(mlo_path).split('.')[0]
            
            result = self._load_single_image(cc_path)
            self.current_cc_image = result[0]
            self.current_cc_file_type = result[1]
            self.current_cc_original_pixel_spacing = result[2]
            self.current_cc_original_shape = result[3]
            self.current_cc_transformation_info = result[4] if len(result) > 4 else None
            self.current_cc_filename = os.path.basename(cc_path).split('.')[0]
            
            self._validate_image(self.current_mlo_image, "MLO")
            self._validate_image(self.current_cc_image, "CC")
            
        except Exception as e:
            self._clear_all_data()
            raise ValueError(f"Failed to load images: {e}")
    
    def _validate_image(self, image: np.ndarray, image_type: str) -> None:
        """Validate loaded image data."""
        if image is None:
            raise ValueError(f"{image_type} image is None")
        
        if not isinstance(image, np.ndarray):
            raise ValueError(f"{image_type} image is not a numpy array")
        
        if image.size == 0:
            raise ValueError(f"{image_type} image is empty")
        
        if len(image.shape) < 2:
            raise ValueError(f"{image_type} image has invalid dimensions: {image.shape}")
        
        if image.shape[-1] < 100 or image.shape[-2] < 100:
            raise ValueError(f"{image_type} image is too small: {image.shape}")
    
    def load_pixel_spacing_data(self) -> None:
        """Load pixel spacing data from CSV files."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            data_dir = os.path.join(project_root, "data")
            
            mlo_data_path = os.path.join(data_dir, "mlo_pixel_spacing.csv")
            if os.path.exists(mlo_data_path):
                self._load_pixel_spacing_file(mlo_data_path, "mlo")
            
            cc_data_path = os.path.join(data_dir, "cc_pixel_spacing.csv")
            if os.path.exists(cc_data_path):
                self._load_pixel_spacing_file(cc_data_path, "cc")
                
        except Exception as e:
            print(f"Warning: Failed to load pixel spacing data: {e}")
    
    def _load_pixel_spacing_file(self, file_path: str, image_type: str) -> None:
        """Load pixel spacing data from a specific CSV file."""
        try:
            df = pd.read_csv(file_path)
            
            for _, row in df.iterrows():
                filename = os.path.basename(row['filename']).replace('.dicom', '')
                key = f"{filename}_{image_type}"
                
                self.pixel_spacing_data[key] = {
                    'pixel_spacing_x': row['pixel_spacing_x_mm'],
                    'pixel_spacing_y': row['pixel_spacing_y_mm'],
                    'type': image_type.upper()
                }
                
        except Exception as e:
            print(f"Warning: Failed to load {image_type} pixel spacing data: {e}")
    
    def get_pixel_spacing(self, image_type: str) -> float:
        """Get pixel spacing for the current image of given type."""
        if image_type.lower() == 'mlo':
            if self.current_mlo_original_pixel_spacing:
                return self.current_mlo_original_pixel_spacing[0]
        elif image_type.lower() == 'cc':
            if self.current_cc_original_pixel_spacing:
                return self.current_cc_original_pixel_spacing[0]
        
        print(f"Warning: No pixel spacing found for {image_type}, using default 0.085mm/pixel")
        return 0.085
    
    def get_current_images(self) -> tuple:
        """Get currently loaded images."""
        return self.current_mlo_image, self.current_cc_image
    
    def get_current_filenames(self) -> tuple:
        """Get currently loaded filenames."""
        return self.current_mlo_filename, self.current_cc_filename
    
    def is_pair_loaded(self) -> bool:
        """Check if both MLO and CC images are loaded."""
        return (self.current_mlo_image is not None and 
                self.current_cc_image is not None)
    
    def _load_single_image(self, file_path: str) -> tuple:
        """Load a single DICOM image file."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension in ['.dicom', '.dcm']:
            if not DICOM_AVAILABLE:
                raise ImportError("pydicom is required. Install with: pip install pydicom")
            return self._load_dicom_file(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_extension}. Only DICOM (.dicom, .dcm) supported.")
    
    def _load_dicom_file(self, dicom_path: str) -> tuple:
        """Load DICOM file with preprocessing."""
        processed_image, original_shape, dicom_obj, transformation_info = self.preprocessor.process(dicom_path)
        pixel_spacing = self.preprocessor.extract_pixel_spacing(dicom_obj)
        
        return processed_image, 'dicom', pixel_spacing, original_shape, transformation_info
    
    def _clear_all_data(self) -> None:
        """Clear all current data."""
        self.current_mlo_image = None
        self.current_cc_image = None
        self.current_mlo_filename = None
        self.current_cc_filename = None
        self.current_mlo_file_type = None
        self.current_cc_file_type = None
        self.current_mlo_original_pixel_spacing = None
        self.current_cc_original_pixel_spacing = None
        self.current_mlo_original_shape = None
        self.current_cc_original_shape = None
        self.current_mlo_transformation_info = None
        self.current_cc_transformation_info = None

    def clear_images(self) -> None:
        """Clear currently loaded images."""
        self._clear_all_data()
    
    def get_image_info(self, image_type: str) -> Optional[Dict[str, Any]]:
        """Get information about loaded image."""
        if image_type.lower() == 'mlo':
            image = self.current_mlo_image
            filename = self.current_mlo_filename
        elif image_type.lower() == 'cc':
            image = self.current_cc_image
            filename = self.current_cc_filename
        else:
            return None
        
        if image is None:
            return None
        
        return {
            'filename': filename,
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min_value': float(image.min()),
            'max_value': float(image.max()),
            'mean_value': float(image.mean()),
            'pixel_spacing': self.get_pixel_spacing(image_type),
            'file_type': self.current_mlo_file_type if image_type.lower() == 'mlo' else self.current_cc_file_type,
            'original_shape': self.current_mlo_original_shape if image_type.lower() == 'mlo' else self.current_cc_original_shape
        }
