"""
Comparison engine for analyzing MLO vs CC positioning results.
"""

from typing import Dict, Any
import pandas as pd


class ComparisonEngine:
    """Engine for comparing MLO and CC analysis results."""
    
    def __init__(self, threshold_mm: float = 10.0):
        """Initialize the comparison engine.
        
        Args:
            threshold_mm: Threshold for determining good positioning (default: 10mm)
        """
        self.threshold_mm = threshold_mm
    
    def compare(self, mlo_results: Dict[str, Any], 
                cc_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare MLO and CC analysis results.
        
        Args:
            mlo_results: MLO analysis results
            cc_results: CC analysis results
            
        Returns:
            Dictionary containing comparison results
        """
        if mlo_results is None or cc_results is None:
            raise ValueError("Both MLO and CC results are required for comparison")
        
        mlo_distance = mlo_results['distance_mm']
        cc_distance = cc_results['distance_mm']
        difference = abs(mlo_distance - cc_distance)
        
        quality_result, quality_color = self._evaluate_quality(difference)
        result_text = self._create_result_text(
            mlo_distance, cc_distance, difference, quality_result
        )
        
        return {
            'mlo_distance': mlo_distance,
            'cc_distance': cc_distance,
            'difference': difference,
            'threshold': self.threshold_mm,
            'quality_result': quality_result,
            'quality_color': quality_color,
            'result_text': result_text,
            'is_good_positioning': difference <= self.threshold_mm,
            'mlo_pixel_spacing': mlo_results['pixel_spacing'],
            'cc_pixel_spacing': cc_results['pixel_spacing'],
            'analysis_timestamp': self._get_timestamp()
        }
    
    def _evaluate_quality(self, difference: float) -> tuple:
        """Evaluate positioning quality based on difference.
        
        Args:
            difference: Difference between MLO and CC distances
            
        Returns:
            Tuple of (quality_result, quality_color)
        """
        if difference <= self.threshold_mm:
            return "✅ CORRECT POSITIONING", "green"
        elif difference <= self.threshold_mm * 1.5:
            return "⚠️ QUESTIONABLE POSITIONING", "orange"
        else:
            return "❌ INCORRECT POSITIONING", "red"
    
    def _create_result_text(self, mlo_distance: float, cc_distance: float,
                          difference: float, quality_result: str) -> str:
        """Create formatted result text for display."""
        return (
            f"MLO Distance: {mlo_distance:.2f} mm\n"
            f"CC Distance: {cc_distance:.2f} mm\n"
            f"Difference: {difference:.2f} mm\n\n"
            f"Result: {quality_result}\n"
            f"(Threshold: {self.threshold_mm:.0f}mm)"
        )
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for analysis."""
        return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def set_threshold(self, new_threshold: float) -> None:
        """Set new threshold for comparison.
        
        Args:
            new_threshold: New threshold value in mm
        """
        if new_threshold <= 0:
            raise ValueError("Threshold must be positive")
        self.threshold_mm = new_threshold
