"""
Central Image Preprocessing Module.
Standardized preprocessing pipeline for DICOM mammogram images.
"""

import numpy as np
import cv2
import pywt
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


class ImagePreprocessor:
    """
    Standard preprocessing class for converting DICOM files to model-expected format.
    Implements crop, pad, and resize operations with laterality-based padding.

    The checkpoints were trained on images that went through the training
    pipeline's breast crop and its NLM + wavelet + CLAHE enhancement, so the
    model input is built the same way here. What the user sees stays the plain
    rendering: ``process`` returns the display image and the model input
    separately, and they share one geometry, so landmarks predicted on the
    enhanced image land correctly on the displayed one.
    """
    TARGET_SIZE = (640, 640)

    # Training crop rule (src/preprocessing/segmenter.py): fixed threshold on the
    # 8-bit preview, morphological clean-up, bounding box of all foreground,
    # then a fixed margin. The mean-based largest-region rule used before gave a
    # different box and moved the predicted landmarks.
    MIN_THRESHOLD = 15
    CROP_PADDING = 30

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
        """Breast bounding box, matching the training pipeline's segmenter."""
        preview = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        binary = (preview > self.MIN_THRESHOLD).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

        height, width = preview.shape[:2]
        coords = cv2.findNonZero(binary)
        if coords is None:
            return 0, height - 1, 0, width - 1

        x, y, box_w, box_h = cv2.boundingRect(coords)
        if box_w < 100 or box_h < 100:
            return 0, height - 1, 0, width - 1

        pad = self.CROP_PADDING
        return (max(0, y - pad), min(height, y + box_h + pad) - 1,
                max(0, x - pad), min(width, x + box_w + pad) - 1)

    def enhance(self, img: np.ndarray) -> np.ndarray:
        """NLM + Daubechies-4 wavelet edge boost + CLAHE, as in training.

        Mirrors src/preprocessing/strategies.py AdvancedWaveletStrategy; the
        parameters are the ones the checkpoints were trained with.
        """
        image = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(image, None, h=8,
                                            templateWindowSize=5, searchWindowSize=11)

        coeffs = pywt.wavedec2(denoised.astype(np.float64), "db4", level=2)
        boosted = [coeffs[0]]
        for level_idx, (cH, cV, cD) in enumerate(coeffs[1:], start=1):
            # coeffs[1] is the coarsest level (x1.3), coeffs[2] the finest (x1.6)
            factor = 1.3 if level_idx == 1 else 1.6
            boosted.append((cH * factor, cV * factor, cD * factor))
        reconstructed = pywt.waverec2(boosted, "db4")[:image.shape[0], :image.shape[1]]
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(reconstructed).astype(np.float32) / 255.0

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
            Tuple of (display_image, model_input, original_shape, dicom_obj,
            transformation_info). Both images are 640x640 and share one geometry.
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

        # Model input: the same canvas with the breast region enhanced. Training
        # enhances the resized crop before padding, so the padding is excluded
        # here too - running the filters over the black border would alter the
        # CLAHE tiles near the chest wall.
        model_input = self._enhance_unpadded(resized_img, pad_coords, padded_img.shape[0])

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
        
        return resized_img, model_input, original_shape, dicom_obj, transformation_info

    def _enhance_unpadded(self, resized: np.ndarray, pad_coords: tuple,
                          padded_size: int) -> np.ndarray:
        """Enhance only the breast region of the 640 canvas, keeping the border."""
        pad_left, pad_right, pad_top, pad_bottom = pad_coords
        scale = self.TARGET_SIZE[0] / float(padded_size)
        left, right = int(round(pad_left * scale)), int(round(pad_right * scale))
        top, bottom = int(round(pad_top * scale)), int(round(pad_bottom * scale))

        y2, x2 = self.TARGET_SIZE[0] - bottom, self.TARGET_SIZE[1] - right
        content = resized[top:y2, left:x2]
        if content.size == 0:
            return self.enhance(resized)

        out = np.zeros_like(resized, dtype=np.float32)
        out[top:y2, left:x2] = self.enhance(content)
        return out

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
