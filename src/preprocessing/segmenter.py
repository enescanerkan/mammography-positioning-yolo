import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, TYPE_CHECKING
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1


class BreastSegmenter:
    """
    Handles the segmentation of the breast region from the background.
    Simple threshold-based approach for mammography images.
    
    Expects: breast tissue = bright (high values), background = dark (low values ~0)
    """
    
    def __init__(self):
        self.padding = 30
        self.min_threshold = 15  # Minimum pixel value to consider as breast tissue

    def find_breast_bbox(self, image: np.ndarray, laterality=None) -> BoundingBox:
        """
        Finds the bounding box of the breast region.
        
        Args:
            image: 8-bit input image (breast=bright, background=dark)
            laterality: Laterality enum (not used currently, for future improvements)
            
        Returns:
            BoundingBox of the breast region
        """
        h, w = image.shape[:2]
        
        # Simple threshold - anything above min_threshold is breast tissue
        binary = (image > self.min_threshold).astype(np.uint8) * 255
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Find bounding box of all non-zero pixels
        coords = cv2.findNonZero(binary)
        
        if coords is None:
            logger.warning("No breast region found, using full image")
            return BoundingBox(0, 0, w, h)
        
        x, y, bw, bh = cv2.boundingRect(coords)
        
        # Validate - if too small, use full image
        if bw < 100 or bh < 100:
            logger.warning("Detected region too small, using full image")
            return BoundingBox(0, 0, w, h)
        
        # Add padding
        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(w, x + bw + self.padding)
        y2 = min(h, y + bh + self.padding)
        
        return BoundingBox(x1, y1, x2, y2)

    def isolate_breast(self, image: np.ndarray) -> Tuple[np.ndarray, BoundingBox]:
        """
        Legacy method - detects the breast bbox and returns the cropped image.
        """
        bbox = self.find_breast_bbox(image)
        cropped = image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        return cropped, bbox
