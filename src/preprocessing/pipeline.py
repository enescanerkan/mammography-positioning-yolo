"""
Mammography preprocessing pipeline.

Implements the ten-step processing order described in the project README.
Normalisation is deliberately applied **after** cropping so that the full
0–255 dynamic range maps to breast tissue contrast rather than the dark
background.

Processing order
----------------
1.  DICOM load with VOI LUT
2.  MONOCHROME1 inversion (if required)
3.  Quick 8-bit normalisation → segmentation preview
4.  Breast region bounding-box detection
5.  Keypoint-guided bounding-box expansion
6.  Crop raw float32 data
7.  Normalise cropped region to 8-bit
8.  Resize (aspect ratio preserved, longest side = 640 px)
9.  Apply enhancement strategy (Baseline / HistEq / Wavelet)
10. Smart laterality-aware black padding → 640×640
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple
from enum import Enum

from ..core.interfaces import IEnhancementStrategy
from ..preprocessing.strategies import BaselineStrategy
from ..utils.config import Config
from ..utils.logger import get_logger
from ..data.dicom_loader import DicomLoader
from ..data.label_manager import LabelManager
from .segmenter import BreastSegmenter, BoundingBox

logger = get_logger(__name__)


class Laterality(Enum):
    LEFT = 'L'
    RIGHT = 'R'
    UNKNOWN = 'U'


class PreprocessingPipeline:
    """
    Mammography image preprocessing pipeline.
    
    CORRECT PROCESSING ORDER:
    1. MONOCHROME1 correction (on raw data)
    2. Create preview (for Otsu/segmentation)
    3. Find crop bbox (on preview)
    4. Keypoint-guided bbox expansion
    5. Crop raw data
    6. Normalize cropped raw data
    7. Resize
    8. Padding (ALWAYS BLACK, smart positioning based on laterality)
    """
    
    def __init__(self, data_dir: str, output_dir: str, labels_csv: list, strategy: Optional[IEnhancementStrategy] = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.label_manager = LabelManager(labels_csv)
        self.segmenter = BreastSegmenter()
        self.strategy = strategy if strategy is not None else BaselineStrategy()
        self.metadata_list = []
        self.target_size = Config.TARGET_SIZE[0]  # 640
        
    def run(self, view_mode: str = 'MLO'):
        """Executes the pipeline for DICOMs matching the specified view."""
        self.setup_directories(self.output_dir)
        
        # Get valid SOPInstanceUIDs for this view from CSV
        valid_uids = self.label_manager.get_valid_sop_uids_for_view(view_mode)
        
        if not valid_uids:
            logger.error(f"No valid images found for {view_mode} view in CSV!")
            return
        
        df = self.label_manager.df
        mask = df['SeriesDescription'].str.contains(view_mode, case=False, na=False)
        valid_df = df[mask].drop_duplicates(subset=['SOPInstanceUID'])
        
        dicom_files = []
        for _, row in valid_df.iterrows():
            study_uid = row.get('StudyInstanceUID', '')
            sop_uid = row['SOPInstanceUID']
            
            # Direct path based on Vindr dataset structure (images/study_uid/sop_uid.dicom)
            d_path = self.data_dir / str(study_uid) / f"{sop_uid}.dicom"
            if d_path.exists():
                dicom_files.append(d_path)
            else:
                # Fallback flat
                d_path_flat = self.data_dir / f"{sop_uid}.dicom"
                if d_path_flat.exists():
                    dicom_files.append(d_path_flat)
                    
        # If absolutely no valid files found via direct path, fallback to rglob
        if not dicom_files:
            logger.info("Direct paths not found, falling back to scanning directory...")
            all_files = list(self.data_dir.rglob("*.dcm")) + list(self.data_dir.rglob("*.dicom"))
            dicom_files = [f for f in all_files if f.stem in valid_uids]
            
        logger.info(f"Targeting strictly {len(dicom_files)} specific DICOM files for {view_mode} view (ignoring the rest of the dataset).")
        
        success_count = 0
        skipped_count = 0
        self.metadata_list = []
        
        for dcm_path in tqdm(dicom_files, desc=f"Processing ONLY relevant {view_mode} files"):
            try:
                # Extract SOPInstanceUID from filename (filename without extension)
                sop_uid = dcm_path.stem
                
                self.process_single(dcm_path, view_mode)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to process {dcm_path.name}: {e}")
                
        # Save Metadata
        if self.metadata_list:
            df = pd.DataFrame(self.metadata_list)
            df.to_csv(self.output_dir / "metadata.csv", index=False)
            logger.info(f"Metadata saved to {self.output_dir / 'metadata.csv'}")
                
        logger.info(f"Preprocessing complete. Processed {success_count} {view_mode} images. Skipped {skipped_count} non-{view_mode} images.")

    def setup_directories(self, base_dir: Path):
        for split in ['train', 'val', 'test']:
            (base_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (base_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def process_single(self, dcm_path: Path, view_mode: str):
        # 1. Load raw DICOM data
        pixel_array, metadata = DicomLoader.load(dcm_path)
        sop_uid = metadata['SOPInstanceUID']
        photometric = metadata['photometric_interpretation']
        
        # Determine Split
        split = self.label_manager.get_split(sop_uid)
        
        # Get laterality
        laterality_str = metadata.get('ImageLaterality', 'U')
        laterality = Laterality.LEFT if laterality_str == 'L' else \
                     Laterality.RIGHT if laterality_str == 'R' else Laterality.UNKNOWN
        
        # Get keypoints for this image
        if view_mode == 'MLO':
            keypoints = self.label_manager.get_mlo_keypoints(sop_uid)
        else:
            keypoints = self.label_manager.get_cc_keypoints(sop_uid)
        
        # 2. Process image with correct order
        final_img, transform_info = self._process_image(
            pixel_array, photometric, laterality, keypoints
        )
        
        # 3. Handle Labels (Keypoints) - transform to final coordinates
        yolo_line = self.label_manager.format_yolo_pose(
            class_id=0,
            roi_bbox=BoundingBox(
                transform_info['crop_x1'],
                transform_info['crop_y1'],
                transform_info['crop_x2'],
                transform_info['crop_y2']
            ),
            keypoints=keypoints,
            transform_info=transform_info,
            final_shape=final_img.shape
        )
        
        # 4. Save (Split aware)
        out_name = f"{sop_uid}"
        images_dir = self.output_dir / "images" / split
        labels_dir = self.output_dir / "labels" / split
        
        # PNG for lossless compression (better for medical images)
        cv2.imwrite(str(images_dir / f"{out_name}.png"), final_img)
        
        with open(labels_dir / f"{out_name}.txt", "w") as f:
            f.write(yolo_line)
            
        # 5. Collect Metadata
        original_h, original_w = pixel_array.shape[:2]
        
        self.metadata_list.append({
            'sop_uid': sop_uid,
            'split': split,
            'laterality': laterality_str,
            'original_width': original_w,
            'original_height': original_h,
            'crop_x1': transform_info['crop_x1'],
            'crop_y1': transform_info['crop_y1'],
            'crop_x2': transform_info['crop_x2'],
            'crop_y2': transform_info['crop_y2'],
            'scale': transform_info['scale'],
            'pad_left': transform_info['pad_left'],
            'pad_top': transform_info['pad_top'],
            'pad_right': transform_info['pad_right'],
            'pad_bottom': transform_info['pad_bottom'],
            'pixel_spacing': metadata['pixel_spacing'][0] if metadata.get('pixel_spacing') else 0.085
        })

    def _process_image(self, pixel_array: np.ndarray, 
                       photometric: str,
                       laterality: Laterality,
                       keypoints: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Process image with CORRECT ORDER:
        1. MONOCHROME1 correction (raw data)
        2. Create preview for segmentation
        3. Find crop bbox
        4. Expand bbox for keypoints
        5. Crop raw data
        6. Normalize cropped data
        7. Resize
        8. Smart padding
        """
        orig_h, orig_w = pixel_array.shape[:2]
        
        # 1. MONOCHROME1 correction on RAW data
        # MONOCHROME1: high pixel values = dark areas, low values = bright areas
        # MONOCHROME2: high pixel values = bright areas, low values = dark areas
        # We want: high values = bright (breast tissue), low values = dark (background)
        # 
        # After correction: background should have LOW values (~0), breast should have HIGH values
        if photometric == "MONOCHROME1":
            pixel_array = pixel_array.max() - pixel_array
            logger.debug(f"Applied MONOCHROME1 inversion")
        
        # 2. Create preview for segmentation (quick 8-bit conversion)
        preview = self._quick_normalize(pixel_array)
        
        # 3. Find crop bbox using segmentation
        bbox = self.segmenter.find_breast_bbox(preview, laterality)
        
        # 4. Expand bbox to include all keypoints with margin
        if keypoints:
            bbox = self._expand_bbox_for_keypoints(bbox, keypoints, orig_w, orig_h, margin=80)
        
        # 5. Crop RAW data (not normalized yet!)
        pixel_array_cropped = pixel_array[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        
        # 6. Normalize ONLY the cropped region (better contrast for breast)
        img_normalized = self._normalize_to_8bit(pixel_array_cropped)
        orig_crop_h, orig_crop_w = img_normalized.shape[:2]
        
        # Calculate final mathematical scale from ORIGINAL crop -> 640 (Final YOLO)
        final_scale = self.target_size / max(orig_crop_h, orig_crop_w)
        
        # 7. Final Resize to target (max 640) maintaining aspect ratio
        final_w = int(orig_crop_w * final_scale)
        final_h = int(orig_crop_h * final_scale)
        img_resized = cv2.resize(img_normalized, (final_w, final_h), interpolation=cv2.INTER_LINEAR)
        
        # 7.5 Apply Enhancement Strategy 
        # Extremely fast execute at 640x640 resolution.
        img_enhanced = self.strategy.run(img_resized)
        
        # 8. Smart padding (BLACK, positioned based on laterality) using enhanced image
        img_final, pad_left, pad_top, pad_right, pad_bottom = self._apply_smart_padding(
            img_enhanced, final_w, final_h, laterality
        )
        
        transform_info = {
            'original_width': orig_w,
            'original_height': orig_h,
            'crop_x1': bbox.x1,
            'crop_y1': bbox.y1,
            'crop_x2': bbox.x2,
            'crop_y2': bbox.y2,
            'scale': final_scale,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'pad_right': pad_right,
            'pad_bottom': pad_bottom,
        }
        
        return img_final, transform_info

    def _quick_normalize(self, pixel_array: np.ndarray) -> np.ndarray:
        """Quick normalization for preview/segmentation only."""
        img_min = pixel_array.min()
        img_max = pixel_array.max()
        
        if img_max - img_min < 1e-8:
            return np.zeros_like(pixel_array, dtype=np.uint8)
        
        img = (pixel_array - img_min) / (img_max - img_min)
        img = (img * 255).astype(np.uint8)
        return img

    def _normalize_to_8bit(self, pixel_array: np.ndarray) -> np.ndarray:
        """
        Normalize to 8-bit using min-max scaling (same as test script).
        Result: breast tissue = bright (high values), background = dark (low values ~0)
        """
        # Simple min-max normalization like test script
        img_min = pixel_array.min()
        img_max = pixel_array.max()
        
        if img_max - img_min < 1e-8:
            return np.zeros_like(pixel_array, dtype=np.uint8)
        
        img = (pixel_array - img_min) / (img_max - img_min)
        img = (img * 255).astype(np.uint8)
        return img

    def _expand_bbox_for_keypoints(self, bbox: BoundingBox,
                                   keypoints: Optional[Dict[str, Tuple[float, float]]],
                                   img_w: int, img_h: int,
                                   margin: int = 80) -> BoundingBox:
        """
        Expand bbox to include all keypoints with margin.
        This ensures keypoints are never cropped out.
        """
        if not keypoints:
            return bbox
            
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        
        for name, (kp_x, kp_y) in keypoints.items():
            # Expand if keypoint is outside bbox
            if kp_x < x1:
                x1 = max(0, int(kp_x) - margin)
            if kp_x > x2:
                x2 = min(img_w, int(kp_x) + margin)
            if kp_y < y1:
                y1 = max(0, int(kp_y) - margin)
            if kp_y > y2:
                y2 = min(img_h, int(kp_y) + margin)
        
        return BoundingBox(x1, y1, x2, y2)

    def _apply_smart_padding(self, img: np.ndarray,
                             img_w: int, img_h: int,
                             laterality: Laterality) -> Tuple[np.ndarray, int, int, int, int]:
        """
        Apply BLACK padding with smart positioning based on laterality.
        
        - LEFT breast: padding on RIGHT side (chest wall on left)
        - RIGHT breast: padding on LEFT side (chest wall on right)
        - UNKNOWN: center padding
        """
        target = self.target_size
        pad_w = target - img_w
        pad_h = target - img_h
        
        # Vertical padding - centered
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        # Horizontal padding - based on laterality
        if laterality == Laterality.LEFT:
            # Left breast: chest wall on left, so padding goes on right
            pad_left = 0
            pad_right = pad_w
        elif laterality == Laterality.RIGHT:
            # Right breast: chest wall on right, so padding goes on left
            pad_left = pad_w
            pad_right = 0
        else:
            # Unknown: center
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
        
        # BLACK padding (value=0) - ALWAYS
        img_padded = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )
        
        return img_padded, pad_left, pad_top, pad_right, pad_bottom
