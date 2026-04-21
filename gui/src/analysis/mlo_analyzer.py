"""
MLO analyzer for mammogram positioning analysis.
"""

import numpy as np
from typing import Dict, Any
from .base_analyzer import BaseAnalyzer


class MLOAnalyzer(BaseAnalyzer):
    """Analyzer for MLO (Mediolateral Oblique) mammogram positioning."""
    
    def analyze(self) -> Dict[str, Any]:
        """Perform MLO analysis.
        
        Returns:
            Dictionary containing analysis results
        """
        landmarks = self.model_manager.predict_landmarks(
            self.data_manager.current_mlo_image, 
            'mlo',
            None
        )
        
        if landmarks is None:
            raise ValueError("Failed to predict MLO landmarks")
        
        original_pixel_spacing = self.data_manager.get_pixel_spacing('MLO')
        
        transformation_info = self.data_manager.current_mlo_transformation_info
        if transformation_info and hasattr(self.data_manager, 'preprocessor') and self.data_manager.preprocessor:
            scaled_pixel_spacing, scale_factor = self.data_manager.preprocessor.calculate_scaled_pixel_spacing(
                (original_pixel_spacing, original_pixel_spacing),
                transformation_info
            )
        else:
            scaled_pixel_spacing = original_pixel_spacing
            scale_factor = 1.0
        
        nipple = landmarks[0]
        pectoral1 = landmarks[1]
        pectoral2 = landmarks[2]
        
        perp_distance_pixels, intersection = self._calculate_perpendicular_distance(
            pectoral1, pectoral2, nipple
        )
        
        perp_distance_mm = perp_distance_pixels * scaled_pixel_spacing
        
        return {
            'landmarks': landmarks,
            'pixel_spacing': original_pixel_spacing,
            'scaled_pixel_spacing': scaled_pixel_spacing,
            'scale_factor': scale_factor,
            'distance_pixels': perp_distance_pixels,
            'distance_mm': perp_distance_mm,
            'intersection': intersection,
            'pectoral_line': {
                'point1': pectoral1,
                'point2': pectoral2
            },
            'nipple_position': nipple,
            'analysis_type': 'MLO',
            'transformation_info': transformation_info
        }
    
    def _calculate_perpendicular_distance(self, point1: np.ndarray, point2: np.ndarray, 
                                        nipple: np.ndarray) -> tuple:
        return MLOAnalyzer.perpendicular_distance(point1, point2, nipple)

    @staticmethod
    def perpendicular_distance(point1: np.ndarray, point2: np.ndarray,
                               nipple: np.ndarray) -> tuple:
        """Calculate perpendicular distance from nipple to pectoral line.
        
        Can be called standalone without an analyzer instance.
        """
        line_vec = point2 - point1
        point_vec = nipple - point1
        
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return np.linalg.norm(point_vec), point1.copy()
        
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        proj = proj_length * line_unitvec
        
        perp_vec = point_vec - proj
        perp_distance = np.linalg.norm(perp_vec)
        intersection = point1 + proj
        
        return perp_distance, intersection
    
    def validate_landmarks(self, landmarks: np.ndarray) -> bool:
        """Validate MLO landmark predictions."""
        if landmarks is None:
            return False
        
        if landmarks.shape != (3, 2):
            return False
        
        for landmark in landmarks:
            if not (0 <= landmark[0] <= 640 and 0 <= landmark[1] <= 640):
                return False
        
        pectoral1, pectoral2 = landmarks[1], landmarks[2]
        if np.allclose(pectoral1, pectoral2, atol=1.0):
            return False
        
        return True
