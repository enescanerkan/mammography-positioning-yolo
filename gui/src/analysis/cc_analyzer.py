"""
CC analyzer for mammogram positioning analysis.
"""

import numpy as np
from typing import Dict, Any
from .base_analyzer import BaseAnalyzer


class CCAnalyzer(BaseAnalyzer):
    """Analyzer for CC (Craniocaudal) mammogram positioning."""
    
    def analyze(self) -> Dict[str, Any]:
        """Perform CC analysis.
        
        Returns:
            Dictionary containing analysis results
        """
        landmarks = self.model_manager.predict_landmarks(
            self.data_manager.current_cc_image, 
            'cc',
            None
        )
        
        if landmarks is None:
            raise ValueError("Failed to predict CC landmarks")
        
        original_pixel_spacing = self.data_manager.get_pixel_spacing('CC')
        
        transformation_info = self.data_manager.current_cc_transformation_info
        if transformation_info and hasattr(self.data_manager, 'preprocessor') and self.data_manager.preprocessor:
            scaled_pixel_spacing, scale_factor = self.data_manager.preprocessor.calculate_scaled_pixel_spacing(
                (original_pixel_spacing, original_pixel_spacing),
                transformation_info
            )
        else:
            scaled_pixel_spacing = original_pixel_spacing
            scale_factor = 1.0
        
        nipple = landmarks[0]
        image_width = 640
        
        # Detect which side of the image has the breast tissue
        breast_image_side = self._detect_breast_side(self.data_manager.current_cc_image)
        
        # Calculate distance to chest wall based on breast position
        direction, distance_pixels, edge_point = self._calculate_edge_distance_by_laterality(
            nipple, image_width, breast_image_side
        )
        
        distance_mm = distance_pixels * scaled_pixel_spacing
        
        return {
            'landmarks': landmarks,
            'pixel_spacing': original_pixel_spacing,
            'scaled_pixel_spacing': scaled_pixel_spacing,
            'scale_factor': scale_factor,
            'distance_pixels': distance_pixels,
            'distance_mm': distance_mm,
            'edge_point': edge_point,
            'direction': direction,
            'breast_side': breast_image_side,
            'nipple_position': nipple,
            'analysis_type': 'CC',
            'transformation_info': transformation_info
        }
    
    def _detect_breast_side(self, image: np.ndarray) -> str:
        """
        Detect which side of the IMAGE the breast tissue is on.
        
        Works regardless of image polarity (dark/light background).
        The side with higher variance contains the breast tissue.
        
        Args:
            image: 2D numpy array (grayscale image)
            
        Returns:
            'LEFT' if breast is on left side of image
            'RIGHT' if breast is on right side of image
        """
        if len(image.shape) == 3:
            image = image[0]
        
        height, width = image.shape
        mid_point = width // 2
        
        # Split image into left and right halves
        left_half = image[:, :mid_point]
        right_half = image[:, mid_point:]
        
        # Calculate variance for each half
        left_variance = np.var(left_half)
        right_variance = np.var(right_half)
        
        # Calculate standard deviation of non-background pixels
        left_nonzero = left_half[left_half > 0.05]
        right_nonzero = right_half[right_half > 0.05]
        
        left_std = np.std(left_nonzero) if len(left_nonzero) > 100 else 0
        right_std = np.std(right_nonzero) if len(right_nonzero) > 100 else 0
        
        # Count tissue pixels (more robust metric)
        left_tissue_count = np.sum(left_half > 0.1)
        right_tissue_count = np.sum(right_half > 0.1)
        
        # Combine metrics
        left_score = left_variance + left_std + (left_tissue_count / (height * mid_point))
        right_score = right_variance + right_std + (right_tissue_count / (height * mid_point))
        
        # Return which side of the IMAGE has the breast
        return 'LEFT' if left_score > right_score else 'RIGHT'
    
    def _calculate_edge_distance_by_laterality(self, nipple: np.ndarray, 
                                               image_width: int,
                                               breast_image_side: str) -> tuple:
        return CCAnalyzer.edge_distance(nipple, image_width, breast_image_side)

    @staticmethod
    def edge_distance(nipple: np.ndarray, image_width: int,
                      breast_image_side: str) -> tuple:
        """Calculate distance from nipple to chest wall edge.
        
        Can be called standalone without an analyzer instance.
        
        Returns:
            Tuple of (direction, distance_pixels, edge_point)
        """
        nipple_x = nipple[0]
        nipple_y = nipple[1]
        
        if breast_image_side == 'LEFT':
            distance_pixels = nipple_x
            edge_point = np.array([0, nipple_y])
            direction = "Left (Chest Wall)"
        else:
            distance_pixels = image_width - nipple_x
            edge_point = np.array([image_width, nipple_y])
            direction = "Right (Chest Wall)"
        
        return direction, distance_pixels, edge_point
    
    def validate_landmarks(self, landmarks: np.ndarray) -> bool:
        """Validate CC landmark predictions."""
        if landmarks is None:
            return False
        
        if landmarks.shape != (1, 2):
            return False
        
        nipple = landmarks[0]
        if not (0 <= nipple[0] <= 640 and 0 <= nipple[1] <= 640):
            return False
        
        return True
