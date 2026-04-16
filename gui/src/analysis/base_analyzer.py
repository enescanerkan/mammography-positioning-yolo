"""
Base analyzer class for mammogram positioning analysis.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAnalyzer(ABC):
    """Abstract base class for mammogram analyzers."""
    
    def __init__(self, data_manager, model_manager):
        """Initialize the base analyzer.
        
        Args:
            data_manager: Data manager instance
            model_manager: Model manager instance
        """
        self.data_manager = data_manager
        self.model_manager = model_manager
    
    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """Perform analysis.
        
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    def validate_prerequisites(self) -> bool:
        """Validate that prerequisites for analysis are met.
        
        Returns:
            True if prerequisites are met, False otherwise
        """
        if self.data_manager is None:
            return False
        
        if self.model_manager is None:
            return False
        
        return True
