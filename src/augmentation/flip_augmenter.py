"""
Flip Augmentation Module for Mammography Images.

This module provides horizontal flip augmentation for mammography datasets,
creating mirrored versions of images while properly transforming labels.

Key Features:
- LEFT breast images are flipped to create synthetic RIGHT breast images
- RIGHT breast images are flipped to create synthetic LEFT breast images
- YOLO Pose labels are properly transformed (x-coordinates mirrored)
- Supports both MLO (3 keypoints) and CC (1 keypoint) views
- Maintains train/val/test split integrity

Design Principles:
- Single Responsibility: Only handles flip augmentation
- Open/Closed: Extensible for other augmentation types via inheritance
- Dependency Inversion: Depends on abstractions (paths, configs)
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from tqdm import tqdm

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseAugmenter(ABC):
    """
    Abstract base class for image augmentation strategies.
    
    Provides common interface for all augmentation implementations,
    following the Strategy pattern for extensibility.
    """
    
    @abstractmethod
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to a single image.
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            
        Returns:
            Augmented image as numpy array
        """
        pass
    
    @abstractmethod
    def augment_label(self, label_line: str, image_width: int) -> str:
        """
        Transform label coordinates according to augmentation.
        
        Args:
            label_line: YOLO format label string
            image_width: Width of the image for coordinate transformation
            
        Returns:
            Transformed label string
        """
        pass


class FlipAugmenter(BaseAugmenter):
    """
    Horizontal flip augmentation for mammography images.
    
    Creates mirrored versions of breast images:
    - LEFT breast → flipped to appear as RIGHT breast
    - RIGHT breast → flipped to appear as LEFT breast
    
    This effectively doubles the dataset size while maintaining
    anatomical consistency (chest wall position).
    
    Attributes:
        data_dir: Path to preprocessed data directory (e.g., preprocessed-data/MLO)
        metadata_path: Path to metadata CSV containing laterality information
        view_type: View type ('MLO' or 'CC') for proper keypoint handling
        
    Example:
        >>> augmenter = FlipAugmenter(
        ...     data_dir="preprocessed-data/MLO",
        ...     view_type="MLO"
        ... )
        >>> augmenter.run()
    """
    
    # Suffix added to augmented file names
    AUGMENTED_SUFFIX = "_flipped"
    
    def __init__(self, data_dir: str, view_type: str = 'MLO'):
        """
        Initialize the FlipAugmenter.
        
        Args:
            data_dir: Path to preprocessed data directory containing
                      images/, labels/, and metadata.csv
            view_type: View type for keypoint handling ('MLO' or 'CC')
                      MLO has 3 keypoints, CC has 1 keypoint
        """
        self.data_dir = Path(data_dir)
        self.view_type = view_type.upper()
        self.metadata_path = self.data_dir / "metadata.csv"
        self.metadata: Optional[pd.DataFrame] = None
        
        self._validate_paths()
        self._load_metadata()
    
    def _validate_paths(self) -> None:
        """
        Validate that required directories and files exist.
        
        Raises:
            FileNotFoundError: If data directory or metadata file doesn't exist
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}. "
                "Run preprocessing first to generate metadata.csv"
            )
    
    def _load_metadata(self) -> None:
        """
        Load metadata CSV containing laterality information.
        
        The metadata CSV should contain at minimum:
        - sop_uid: Unique identifier for each image
        - laterality: 'L' for left breast, 'R' for right breast
        - split: 'train', 'val', or 'test'
        """
        self.metadata = pd.read_csv(self.metadata_path)
        self.metadata['sop_uid'] = self.metadata['sop_uid'].astype(str)
        
        logger.info(f"Loaded metadata with {len(self.metadata)} entries")
        
        # Log laterality distribution
        if 'laterality' in self.metadata.columns:
            lat_counts = self.metadata['laterality'].value_counts()
            logger.info(f"Laterality distribution: {lat_counts.to_dict()}")
    
    def get_laterality(self, sop_uid: str) -> str:
        """
        Get the laterality (L/R) for a given image.
        
        Args:
            sop_uid: SOPInstanceUID of the image
            
        Returns:
            'L' for left breast, 'R' for right breast, 'U' for unknown
        """
        if self.metadata is None:
            return 'U'
        
        row = self.metadata[self.metadata['sop_uid'] == sop_uid]
        if row.empty:
            return 'U'
        
        return row.iloc[0].get('laterality', 'U')
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply horizontal flip to image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Horizontally flipped image
        """
        return cv2.flip(image, 1)  # 1 = horizontal flip
    
    def augment_label(self, label_line: str, image_width: int = 640) -> str:
        """
        Transform YOLO Pose label for horizontal flip.
        
        For horizontal flip, x-coordinates are mirrored: new_x = 1.0 - old_x
        Y-coordinates remain unchanged.
        
        YOLO Pose format:
        <class> <cx> <cy> <w> <h> <kp1_x> <kp1_y> <v1> [<kp2_x> <kp2_y> <v2> ...]
        
        Args:
            label_line: Original YOLO format label string
            image_width: Image width (used for validation, coords are normalized)
            
        Returns:
            Transformed label string with mirrored x-coordinates
        """
        if not label_line.strip():
            return ""
        
        parts = label_line.strip().split()
        if len(parts) < 5:
            logger.warning(f"Invalid label format: {label_line}")
            return ""
        
        # Parse components
        class_id = parts[0]
        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        
        # Mirror center x-coordinate
        new_cx = 1.0 - cx
        
        # Build new label parts
        new_parts = [class_id, f"{new_cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
        
        # Transform keypoints (x, y, visibility triplets)
        keypoint_data = parts[5:]
        num_keypoints = len(keypoint_data) // 3
        
        for i in range(num_keypoints):
            idx = i * 3
            if idx + 2 < len(keypoint_data):
                kp_x = float(keypoint_data[idx])
                kp_y = float(keypoint_data[idx + 1])
                vis = keypoint_data[idx + 2]
                
                # Mirror x-coordinate if keypoint is visible
                if int(vis) > 0 and kp_x > 0:
                    new_kp_x = 1.0 - kp_x
                else:
                    new_kp_x = kp_x
                
                new_parts.extend([f"{new_kp_x:.6f}", f"{kp_y:.6f}", vis])
        
        return " ".join(new_parts)
    
    def _process_single_image(self, 
                              image_path: Path, 
                              label_path: Path,
                              output_image_path: Path,
                              output_label_path: Path) -> bool:
        """
        Process a single image-label pair for flip augmentation.
        
        Args:
            image_path: Path to source image
            label_path: Path to source label file
            output_image_path: Path for augmented image
            output_label_path: Path for augmented label
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read and flip image
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return False
            
            flipped_image = self.augment_image(image)
            
            # Read and transform label
            if label_path.exists():
                with open(label_path, 'r') as f:
                    label_line = f.read().strip()
                flipped_label = self.augment_label(label_line)
            else:
                flipped_label = ""
            
            # Save augmented image (PNG for lossless)
            cv2.imwrite(str(output_image_path), flipped_image)
            
            # Save augmented label
            with open(output_label_path, 'w') as f:
                f.write(flipped_label)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            return False
    
    def run(self, splits: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Execute flip augmentation on the dataset.
        
        Creates flipped versions of all images, effectively doubling
        the dataset size. Augmented files are saved with '_flipped' suffix.
        
        Args:
            splits: List of splits to augment. Default: ['train', 'val', 'test']
                   Use ['train'] to only augment training data.
                   
        Returns:
            Dictionary with counts: {'train': n, 'val': n, 'test': n, 'total': n}
        """
        if splits is None:
            splits = ['train', 'val', 'test']
        
        logger.info(f"Starting flip augmentation for {self.view_type} view")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Splits to augment: {splits}")
        
        results = {}
        total_augmented = 0
        
        for split in splits:
            images_dir = self.data_dir / "images" / split
            labels_dir = self.data_dir / "labels" / split
            
            if not images_dir.exists():
                logger.warning(f"Images directory not found: {images_dir}")
                results[split] = 0
                continue
            
            # Get all images in this split
            image_files = list(images_dir.glob("*.png"))
            
            if not image_files:
                logger.warning(f"No PNG images found in {images_dir}")
                results[split] = 0
                continue
            
            augmented_count = 0
            
            for image_path in tqdm(image_files, desc=f"Augmenting {split}"):
                sop_uid = image_path.stem
                
                # Skip already augmented files
                if sop_uid.endswith(self.AUGMENTED_SUFFIX):
                    continue
                
                # Create output paths with suffix
                output_image_name = f"{sop_uid}{self.AUGMENTED_SUFFIX}.png"
                output_label_name = f"{sop_uid}{self.AUGMENTED_SUFFIX}.txt"
                
                output_image_path = images_dir / output_image_name
                output_label_path = labels_dir / output_label_name
                
                # Skip if already exists
                if output_image_path.exists():
                    continue
                
                label_path = labels_dir / f"{sop_uid}.txt"
                
                if self._process_single_image(
                    image_path, label_path,
                    output_image_path, output_label_path
                ):
                    augmented_count += 1
            
            results[split] = augmented_count
            total_augmented += augmented_count
            logger.info(f"Augmented {augmented_count} images in {split} split")
        
        results['total'] = total_augmented
        logger.info(f"Total augmented images: {total_augmented}")
        
        return results
    
    def get_statistics(self) -> Dict[str, Dict[str, int]]:
        """
        Get dataset statistics before and after augmentation.
        
        Returns:
            Dictionary with original and augmented counts per split
        """
        stats = {}
        
        for split in ['train', 'val', 'test']:
            images_dir = self.data_dir / "images" / split
            
            if not images_dir.exists():
                stats[split] = {'original': 0, 'augmented': 0, 'total': 0}
                continue
            
            all_images = list(images_dir.glob("*.png"))
            augmented = [f for f in all_images if self.AUGMENTED_SUFFIX in f.stem]
            original = [f for f in all_images if self.AUGMENTED_SUFFIX not in f.stem]
            
            stats[split] = {
                'original': len(original),
                'augmented': len(augmented),
                'total': len(all_images)
            }
        
        return stats
