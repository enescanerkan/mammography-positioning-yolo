"""
Analysis controller for coordinating mammogram positioning analyses.
"""

import pandas as pd
from typing import Optional, Dict, Any, Callable
from .mlo_analyzer import MLOAnalyzer
from .cc_analyzer import CCAnalyzer
from .comparison_engine import ComparisonEngine
from .result_saver import ResultSaver


class AnalysisController:
    """Controller for coordinating mammogram positioning analyses."""
    
    def __init__(self, data_manager, model_manager):
        """Initialize the analysis controller.
        
        Args:
            data_manager: Data manager instance
            model_manager: Model manager instance
        """
        self.data_manager = data_manager
        self.model_manager = model_manager
        
        self.mlo_analyzer = MLOAnalyzer(data_manager, model_manager)
        self.cc_analyzer = CCAnalyzer(data_manager, model_manager)
        self.comparison_engine = ComparisonEngine()
        self.result_saver = ResultSaver()
        
        self.mlo_results: Optional[Dict[str, Any]] = None
        self.cc_results: Optional[Dict[str, Any]] = None
        
        self.on_mlo_analysis_complete: Optional[Callable] = None
        self.on_cc_analysis_complete: Optional[Callable] = None
        self.on_comparison_complete: Optional[Callable] = None
    
    def analyze_mlo(self) -> Optional[Dict[str, Any]]:
        """Perform MLO analysis."""
        if not self._validate_mlo_prerequisites():
            return None
        
        try:
            self.mlo_results = self.mlo_analyzer.analyze()
            
            if self.on_mlo_analysis_complete and self.mlo_results:
                self.on_mlo_analysis_complete(self.mlo_results)
            
            return self.mlo_results
            
        except Exception as e:
            print(f"MLO analysis failed: {e}")
            return None
    
    def analyze_cc(self) -> Optional[Dict[str, Any]]:
        """Perform CC analysis."""
        if not self._validate_cc_prerequisites():
            return None
        
        try:
            self.cc_results = self.cc_analyzer.analyze()
            
            if self.on_cc_analysis_complete and self.cc_results:
                self.on_cc_analysis_complete(self.cc_results)
            
            return self.cc_results
            
        except Exception as e:
            print(f"CC analysis failed: {e}")
            return None
    
    def compare_results(self) -> Optional[Dict[str, Any]]:
        """Compare MLO and CC analysis results."""
        if not self._validate_comparison_prerequisites():
            return None
        
        try:
            comparison = self.comparison_engine.compare(self.mlo_results, self.cc_results)
            
            if self.on_comparison_complete and comparison:
                self.on_comparison_complete(comparison)
            
            return comparison
            
        except Exception as e:
            print(f"Comparison failed: {e}")
            return None
    
    def save_results(self) -> Optional[list]:
        """Save analysis results to files."""
        if not self.mlo_results and not self.cc_results:
            raise ValueError("No results to save! Please first perform MLO and/or CC analyses.")
        
        try:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            save_data = {
                'mlo_results': self.mlo_results,
                'cc_results': self.cc_results,
                'filenames': self.data_manager.get_current_filenames(),
                'timestamp': timestamp
            }
            
            saved_files = self.result_saver.save(save_data)
            return saved_files
            
        except Exception as e:
            print(f"Save results failed: {e}")
            return None
    
    def _validate_mlo_prerequisites(self) -> bool:
        """Validate prerequisites for MLO analysis."""
        if self.data_manager.current_mlo_image is None:
            raise ValueError("No MLO image loaded")
        
        if not self.model_manager.is_model_loaded('mlo'):
            raise ValueError("MLO model not loaded")
        
        return True
    
    def _validate_cc_prerequisites(self) -> bool:
        """Validate prerequisites for CC analysis."""
        if self.data_manager.current_cc_image is None:
            raise ValueError("No CC image loaded")
        
        if not self.model_manager.is_model_loaded('cc'):
            raise ValueError("CC model not loaded")
        
        return True
    
    def _validate_comparison_prerequisites(self) -> bool:
        """Validate prerequisites for comparison."""
        if self.mlo_results is None:
            raise ValueError("MLO analysis not yet performed!")
        
        if self.cc_results is None:
            raise ValueError("CC analysis not yet performed!")
        
        return True
    
    def get_mlo_results(self) -> Optional[Dict[str, Any]]:
        """Get MLO analysis results."""
        return self.mlo_results
    
    def get_cc_results(self) -> Optional[Dict[str, Any]]:
        """Get CC analysis results."""
        return self.cc_results
    
    def clear_results(self) -> None:
        """Clear all analysis results."""
        self.mlo_results = None
        self.cc_results = None
    
    def has_results(self) -> Dict[str, bool]:
        """Check which analyses have been completed."""
        return {
            'mlo': self.mlo_results is not None,
            'cc': self.cc_results is not None,
            'both': self.mlo_results is not None and self.cc_results is not None
        }
