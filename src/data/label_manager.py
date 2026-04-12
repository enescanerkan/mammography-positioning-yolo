import pandas as pd
import ast
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from ..utils.logger import get_logger

logger = get_logger(__name__)

class LabelManager:
    """
    Handles parsing of annotation CSVs and conversion to YOLO Pose format.
    """
    
    def __init__(self, csv_paths: List[str]):
        # Load all CSVs, handle potential empty files or errors
        dfs = []
        for p in csv_paths:
            try:
                dfs.append(pd.read_csv(p))
            except Exception as e:
                logger.error(f"Error loading CSV {p}: {e}")
        
        self.df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        # Ensure SOPInstanceUID is string
        if not self.df.empty and 'SOPInstanceUID' in self.df.columns:
            self.df['SOPInstanceUID'] = self.df['SOPInstanceUID'].astype(str)
    
    def get_valid_sop_uids_for_view(self, view_mode: str) -> set:
        """
        Returns set of SOPInstanceUIDs that belong to the specified view (MLO or CC).
        Uses SeriesDescription column which contains L-MLO, R-MLO, L-CC, R-CC etc.
        """
        if self.df.empty or 'SeriesDescription' not in self.df.columns:
            return set()
        
        # Filter by view in SeriesDescription (e.g., 'L-MLO', 'R-MLO' for MLO view)
        mask = self.df['SeriesDescription'].str.contains(view_mode, case=False, na=False)
        valid_uids = set(self.df[mask]['SOPInstanceUID'].unique())
        logger.info(f"Found {len(valid_uids)} unique images for {view_mode} view")
        return valid_uids
            
    def get_split(self, sop_uid: str) -> str:
        """
        Returns the split (Train, Validation, Test) for a given image.
        Default to 'train' if unknown.
        """
        if self.df.empty:
            return 'train'
        
        row = self.df[self.df['SOPInstanceUID'] == sop_uid]
        if row.empty:
            return 'train'
            
        # Get first match's split
        split_val = row.iloc[0].get('Split', 'Train')
        
        # Normalize to lower case for folder naming
        if isinstance(split_val, str):
            split_lower = split_val.lower()
            if 'valid' in split_lower:
                return 'val' # YOLO usually expects 'val'
            return split_lower
        return 'train'
        
    def get_raw_annotations(self, sop_uid: str) -> List[Dict[str, Any]]:
        """
        Retrieves all raw rows for a given SOPInstanceUID.
        """
        if self.df.empty:
            return []
        rows = self.df[self.df['SOPInstanceUID'] == sop_uid]
        annotations = []
        
        for _, row in rows.iterrows():
            try:
                data_str = row['data']
                if pd.isna(data_str):
                    continue
                data = ast.literal_eval(data_str)
                annotations.append({
                    'labelName': row['labelName'],
                    'annotationMode': row.get('annotationMode', 'unknown'), # bbox or line
                    'data': data
                })
            except Exception as e:
                logger.error(f"Error parsing data for {sop_uid}: {e}")
                
        return annotations

    def get_mlo_keypoints(self, sop_uid: str) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Extracts MLO keypoints: nipple, pectoral_top, pectoral_bottom
        Returns: 
            Dict with keypoint name -> (x, y) in original image coordinates
            None if no keypoints found
        """
        anns = self.get_raw_annotations(sop_uid)
        
        if not anns:
            return None
            
        keypoints = {}
        
        for ann in anns:
            name = ann['labelName']
            data = ann['data']
            
            if name == 'Nipple':
                # BBox format: x, y, width, height (Top-Left)
                # Keypoint is Center
                cx = data['x'] + data['width'] / 2.0
                cy = data['y'] + data['height'] / 2.0
                keypoints['nipple'] = (cx, cy)
                
            elif name == 'Pectoralis':
                # Line format: vertices [[x1, y1], [x2, y2]]
                pts = data['vertices']
                if len(pts) >= 2:
                    # Sort by y-coordinate (top first)
                    if pts[0][1] < pts[1][1]:
                        keypoints['pectoral_top'] = (pts[0][0], pts[0][1])
                        keypoints['pectoral_bottom'] = (pts[1][0], pts[1][1])
                    else:
                        keypoints['pectoral_top'] = (pts[1][0], pts[1][1])
                        keypoints['pectoral_bottom'] = (pts[0][0], pts[0][1])
        
        return keypoints if keypoints else None

    def get_cc_keypoints(self, sop_uid: str) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Extracts CC keypoints: nipple only
        Returns: 
            Dict with keypoint name -> (x, y) in original image coordinates
            None if no keypoints found
        """
        anns = self.get_raw_annotations(sop_uid)
        
        if not anns:
            return None
            
        for ann in anns:
            if ann['labelName'] == 'Nipple':
                data = ann['data']
                cx = data['x'] + data['width'] / 2.0
                cy = data['y'] + data['height'] / 2.0
                return {'nipple': (cx, cy)}
        
        return None

    @staticmethod
    def format_yolo_pose(class_id: int, 
                         roi_bbox: Any, 
                         keypoints: Optional[Dict[str, Tuple[float, float]]], 
                         transform_info: Dict[str, Any],
                         final_shape: tuple) -> str:
        """
        Formats a single line for YOLO Pose.
        Line: <class> <cx> <cy> <w> <h> <p1x> <p1y> <v1> ...
        
        Args:
            class_id: Class ID (0 for breast)
            roi_bbox: BoundingBox object with crop coordinates
            keypoints: Dict of keypoint name -> (x, y) in original coordinates
            transform_info: Dict with crop, scale, and padding info
            final_shape: Final image shape (H, W)
            
        Returns:
            YOLO format string
        """
        if not keypoints:
            return ""
            
        img_h, img_w = final_shape[:2]
        
        scale = transform_info['scale']
        pad_x = transform_info['pad_left']
        pad_y = transform_info['pad_top']
        crop_x1 = transform_info['crop_x1']
        crop_y1 = transform_info['crop_y1']
        
        # Determine keypoint order based on available keypoints
        if 'pectoral_top' in keypoints:
            keypoint_order = ['nipple', 'pectoral_top', 'pectoral_bottom']
        else:
            keypoint_order = ['nipple']
        
        # Transform keypoints
        transformed_kpts = []
        valid_xs = []
        valid_ys = []
        
        for kp_name in keypoint_order:
            if kp_name in keypoints:
                ox, oy = keypoints[kp_name]
                
                # Transform: original -> crop -> scale -> pad
                # 1. Subtract crop offset
                kp_x = ox - crop_x1
                kp_y = oy - crop_y1
                
                # 2. Apply scale
                kp_x = kp_x * scale
                kp_y = kp_y * scale
                
                # 3. Add padding offset
                kp_x = kp_x + pad_x
                kp_y = kp_y + pad_y
                
                # Check visibility
                if kp_x < 0 or kp_x > img_w or kp_y < 0 or kp_y > img_h:
                    transformed_kpts.extend([0.0, 0.0, 0.0])
                else:
                    # Normalize to [0, 1]
                    kp_x_norm = kp_x / img_w
                    kp_y_norm = kp_y / img_h
                    
                    # Clamp to valid range
                    kp_x_norm = max(0, min(1, kp_x_norm))
                    kp_y_norm = max(0, min(1, kp_y_norm))
                    
                    transformed_kpts.extend([kp_x_norm, kp_y_norm, 2.0])  # visibility=2 (visible)
                    valid_xs.append(kp_x_norm)
                    valid_ys.append(kp_y_norm)
            else:
                transformed_kpts.extend([0.0, 0.0, 0.0])  # not visible
        
        if not valid_xs:
            return ""
        
        # Calculate bounding box from keypoints
        bbox_padding = 100 / img_w  # normalized padding
        x_min = max(0, min(valid_xs) - bbox_padding)
        x_max = min(1, max(valid_xs) + bbox_padding)
        y_min = max(0, min(valid_ys) - bbox_padding)
        y_max = min(1, max(valid_ys) + bbox_padding)
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        # Build YOLO line - visibility as integer (2), coordinates with 6 decimals
        kp_parts = []
        for i in range(0, len(transformed_kpts), 3):
            kp_x = transformed_kpts[i]
            kp_y = transformed_kpts[i + 1]
            vis = int(transformed_kpts[i + 2])  # visibility as integer
            kp_parts.append(f"{kp_x:.6f} {kp_y:.6f} {vis}")
        
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} " + " ".join(kp_parts)
        
        return line
