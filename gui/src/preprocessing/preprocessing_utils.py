"""
Central Image Preprocessing Module.
Standardized preprocessing pipeline for DICOM mammogram images.
"""

import numpy as np
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage import measure, morphology
from scipy import ndimage


class ImagePreprocessor:
    """
    Standard preprocessing class for converting DICOM files to model-expected format.
    Implements crop, pad, and resize operations with laterality-based padding.
    """
    TARGET_SIZE = (640, 640)

    def load_dicom(self, path: str) -> tuple:
        """Load DICOM file, apply VOI LUT, and normalize to 0-1 range.
        
        Args:
            path: Path to DICOM file
            
        Returns:
            Tuple of (normalized_image, dicom_object)
        """
        dicom = pydicom.dcmread(path)
        data = apply_voi_lut(dicom.pixel_array, dicom)
        data = data.astype(np.float32)
        
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.max(data) - data
        
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        
        return data, dicom

    def _find_largest_rectangle(self, img: np.ndarray) -> tuple:
        """Find the largest breast tissue region in the image."""
        thresh_val = img.mean()
        binary_image = img > thresh_val
        cleaned_image = morphology.opening(binary_image, morphology.disk(3))
        labeled_image, _ = ndimage.label(cleaned_image)
        regions = measure.regionprops(labeled_image, intensity_image=img)
        
        if not regions:
            return 0, img.shape[0]-1, 0, img.shape[1]-1
        
        largest_region = max(regions, key=lambda x: x.area)
        minr, minc, maxr, maxc = largest_region.bbox
        return minr, maxr, minc, maxc

    def _crop_image(self, img: np.ndarray) -> tuple:
        """Crop image based on largest region."""
        rmin, rmax, cmin, cmax = self._find_largest_rectangle(img)
        return img[rmin:rmax+1, cmin:cmax+1], (rmin, rmax, cmin, cmax)

    def _pad_and_resize(self, img: np.ndarray, series_description: str) -> tuple:
        """Add padding to maintain aspect ratio and resize to target size."""
        target_size = self.TARGET_SIZE
        height, width = img.shape[:2]
        max_side = max(height, width)
        
        pad_top = (max_side - height) // 2
        pad_bottom = max_side - height - pad_top
        total_lr_padding = max_side - width
        
        # Determine padding direction based on laterality
        if "L-MLO" in series_description or "L-CC" in series_description or "LCC" in series_description:
            pad_left = 0
            pad_right = total_lr_padding
        elif "R-MLO" in series_description or "R-CC" in series_description or "RCC" in series_description:
            pad_left = total_lr_padding
            pad_right = 0
        else:
            pad_left = total_lr_padding // 2
            pad_right = max_side - width - pad_left

        img_padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)
        img_resized = cv2.resize(img_padded, target_size, interpolation=cv2.INTER_LINEAR)
        
        return img_padded, img_resized, (pad_left, pad_right, pad_top, pad_bottom)

    def extract_pixel_spacing(self, dicom) -> tuple:
        """Extract pixel spacing from DICOM metadata."""
        try:
            if hasattr(dicom, 'ImagerPixelSpacing'):
                spacing = dicom.ImagerPixelSpacing
                return float(spacing[0]), float(spacing[1])
            elif hasattr(dicom, 'PixelSpacing'):
                spacing = dicom.PixelSpacing
                return float(spacing[0]), float(spacing[1])
            else:
                return 0.085, 0.085
        except (AttributeError, IndexError, ValueError):
            return 0.085, 0.085

    def process(self, dicom_path: str) -> tuple:
        """
        Process a DICOM file through complete preprocessing pipeline.
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            Tuple of (processed_image, original_shape, dicom_obj, transformation_info)
        """
        img, dicom_obj = self.load_dicom(dicom_path)
        original_shape = img.shape
        
        try:
            laterality = getattr(dicom_obj, 'ImageLaterality', 'L')
            view_position = getattr(dicom_obj, 'ViewPosition', 'MLO')
            series_description = f"{laterality}-{view_position}"
        except:
            series_description = 'L-MLO'
        
        cropped_img, crop_coords = self._crop_image(img)
        padded_img, resized_img, pad_coords = self._pad_and_resize(cropped_img, series_description)
        
        original_pixel_spacing = self.extract_pixel_spacing(dicom_obj)
        scale_x = resized_img.shape[1] / padded_img.shape[1]
        scale_y = resized_img.shape[0] / padded_img.shape[0]
        
        transformation_info = {
            'original_shape': original_shape,
            'crop_coords': crop_coords,
            'pad_coords': pad_coords,
            'series_description': series_description,
            'original_pixel_spacing': original_pixel_spacing,
            'scale_x': scale_x,
            'scale_y': scale_y
        }
        
        return resized_img, original_shape, dicom_obj, transformation_info

    def calculate_scaled_pixel_spacing(self, original_spacing: tuple, transformation_info: dict) -> tuple:
        """
        Calculate effective pixel spacing in 640x640 space.
        
        Args:
            original_spacing: Original pixel spacing (mm/pixel)
            transformation_info: Transformation info from process()
            
        Returns:
            Tuple of (scaled_pixel_spacing, scale_factor)
        """
        rmin, rmax, cmin, cmax = transformation_info['crop_coords']
        cropped_h = rmax - rmin + 1
        cropped_w = cmax - cmin + 1
        
        padded_size = max(cropped_h, cropped_w)
        scale_factor = padded_size / 640.0
        scaled_spacing = original_spacing[0] * scale_factor
        
        return scaled_spacing, scale_factor

    def transform_landmarks_to_original(self, landmarks_640: np.ndarray, transformation_info: dict) -> np.ndarray:
        """
        Transform landmarks from 640x640 space to original DICOM space.
        
        Args:
            landmarks_640: (N, 2) array of landmarks in 640x640 space
            transformation_info: Transformation info from process()
            
        Returns:
            (N, 2) array of landmarks in original space
        """
        rmin, rmax, cmin, cmax = transformation_info['crop_coords']
        pad_left, pad_right, pad_top, pad_bottom = transformation_info['pad_coords']
        
        cropped_height = rmax - rmin + 1
        cropped_width = cmax - cmin + 1
        padded_size = max(cropped_height, cropped_width)
        scale_factor = padded_size / 640.0
        
        original_coords = []
        for coord in landmarks_640:
            x_640, y_640 = coord[0], coord[1]
            
            x_padded = x_640 * scale_factor
            y_padded = y_640 * scale_factor
            
            x_cropped = x_padded - pad_left
            y_cropped = y_padded - pad_top
            
            x_original = x_cropped + cmin
            y_original = y_cropped + rmin
            
            original_coords.append([x_original, y_original])
        
        return np.array(original_coords)

    def detect_laterality_from_image(self, image: np.ndarray) -> str:
        """
        Detect breast laterality (L/R) from image content using variance analysis.
        
        This method works regardless of image polarity (dark/light background)
        by comparing the variance of left and right halves. The side with
        higher variance contains the breast tissue.
        
        Args:
            image: 2D numpy array (grayscale image, normalized 0-1)
            
        Returns:
            'L' if breast is on left side (LCC), 'R' if on right side (RCC)
        """
        if len(image.shape) == 3:
            image = image[0]
        
        height, width = image.shape
        mid_point = width // 2
        
        # Split image into left and right halves
        left_half = image[:, :mid_point]
        right_half = image[:, mid_point:]
        
        # Calculate variance for each half
        # Breast tissue has more texture/variation than uniform background
        left_variance = np.var(left_half)
        right_variance = np.var(right_half)
        
        # Alternative: use standard deviation of non-zero pixels
        # This helps when there's a lot of black padding
        left_nonzero = left_half[left_half > 0.05]
        right_nonzero = right_half[right_half > 0.05]
        
        left_std = np.std(left_nonzero) if len(left_nonzero) > 100 else 0
        right_std = np.std(right_nonzero) if len(right_nonzero) > 100 else 0
        
        # Also count non-background pixels (more robust)
        left_tissue_count = np.sum(left_half > 0.1)
        right_tissue_count = np.sum(right_half > 0.1)
        
        # Combine metrics: variance + tissue count
        left_score = left_variance + left_std + (left_tissue_count / (height * mid_point))
        right_score = right_variance + right_std + (right_tissue_count / (height * mid_point))
        
        # Higher score = more breast tissue = that side's laterality
        if left_score > right_score:
            return 'L'  # Left breast (LCC)
        else:
            return 'R'  # Right breast (RCC)
