"""
Weights Downloader Module.

This module handles automatic downloading of model weights from Google Drive.
Uses gdown for reliable large-file downloads with confirmation handling.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass


@dataclass
class WeightFile:
    """Represents a model weight file configuration."""
    
    name: str
    drive_id: str
    filename: str
    size_mb: float = 0.0


class IDownloader(ABC):
    """Interface for file downloaders."""
    
    @abstractmethod
    def download(self, file_id: str, destination: str, 
                 progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        """Download a file to the specified destination."""
        pass


class GoogleDriveDownloader(IDownloader):
    """Google Drive file downloader using gdown for reliable large-file handling."""
    
    MIN_VALID_SIZE_BYTES = 1_000_000
    
    def download(self, file_id: str, destination: str,
                 progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        try:
            import gdown
            
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            url = f"https://drive.google.com/uc?id={file_id}"
            output = gdown.download(url, destination, quiet=False)
            
            if output is None:
                return False
            
            if os.path.getsize(destination) < self.MIN_VALID_SIZE_BYTES:
                os.remove(destination)
                print(f"Downloaded file too small, likely HTML error page")
                return False
            
            return True
            
        except Exception as e:
            print(f"Download error: {e}")
            return False


class WeightsManager:
    """
    Manages model weights downloading and verification.
    
    This class handles checking for existing weights and downloading
    missing ones from Google Drive.
    """
    
    # Default weight configurations
    DEFAULT_WEIGHTS: List[WeightFile] = [
        # Drive IDs redacted for anonymous review. Weights are released with the
        # de-anonymized repository upon acceptance. To run locally, set the IDs via
        # WeightsManager.set_drive_ids(...) or place the .pt files in the weights dir.
        WeightFile(
            name="MLO YOLO26 Pose Model",
            drive_id="",
            filename="mlo-yolo26-pose-advanced.pt",
            size_mb=120.0
        ),
        WeightFile(
            name="CC YOLO26 Pose Model",
            drive_id="",
            filename="cc-yolo26-pose-advanced.pt",
            size_mb=120.0
        ),
    ]
    
    def __init__(self, weights_dir: Optional[str] = None):
        """
        Initialize the weights manager.
        
        Args:
            weights_dir: Directory to store weights. Defaults to project weights folder.
        
        Supports both development mode and PyInstaller bundled executable.
        """
        import sys
        
        if weights_dir is None:
            # Check if running as PyInstaller bundle
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                base_path = os.path.dirname(os.path.dirname(current_dir))
            
            weights_dir = os.path.join(base_path, "weights")
        
        self.weights_dir = weights_dir
        self.downloader = GoogleDriveDownloader()
        self._weights_config = self.DEFAULT_WEIGHTS.copy()
    
    def set_drive_ids(self, mlo_id: str, cc_id: str) -> None:
        """
        Set Google Drive IDs for weight files.
        
        Args:
            mlo_id: Drive ID for MLO model weights
            cc_id: Drive ID for CC model weights
        """
        self._weights_config[0] = WeightFile(
            name="MLO YOLO26 Pose Model",
            drive_id=mlo_id,
            filename="mlo-yolo26-pose-advanced.pt",
            size_mb=120.0
        )
        self._weights_config[1] = WeightFile(
            name="CC YOLO26 Pose Model",
            drive_id=cc_id,
            filename="cc-yolo26-pose-advanced.pt",
            size_mb=120.0
        )
    
    def get_missing_weights(self) -> List[WeightFile]:
        """
        Get list of weight files that are missing or corrupted (too small).
        
        Returns:
            List of WeightFile objects that need to be downloaded
        """
        missing = []
        for weight in self._weights_config:
            path = os.path.join(self.weights_dir, weight.filename)
            if not os.path.exists(path) or os.path.getsize(path) < GoogleDriveDownloader.MIN_VALID_SIZE_BYTES:
                missing.append(weight)
        return missing
    
    def all_weights_exist(self) -> bool:
        """
        Check if all required weight files exist.
        
        Returns:
            True if all weights are present, False otherwise
        """
        return len(self.get_missing_weights()) == 0
    
    def download_missing_weights(self, 
                                  progress_callback: Optional[Callable[[str, float], None]] = None
                                  ) -> Dict[str, bool]:
        """
        Download all missing weight files.
        
        Args:
            progress_callback: Optional callback(filename, progress) for updates
            
        Returns:
            Dictionary mapping filename to download success status
        """
        results = {}
        missing = self.get_missing_weights()
        
        os.makedirs(self.weights_dir, exist_ok=True)
        
        for weight in missing:
            destination = os.path.join(self.weights_dir, weight.filename)
            
            def file_progress(progress: float):
                if progress_callback:
                    progress_callback(weight.filename, progress)
            
            print(f"Downloading {weight.name}...")
            success = self.downloader.download(
                weight.drive_id, 
                destination,
                file_progress
            )
            results[weight.filename] = success
            
            if success:
                print(f"✓ {weight.filename} downloaded successfully")
            else:
                print(f"✗ Failed to download {weight.filename}")
        
        return results
    
    def get_weights_status(self) -> Dict[str, bool]:
        """
        Get status of all weight files.
        
        Returns:
            Dictionary mapping filename to existence status
        """
        status = {}
        for weight in self._weights_config:
            path = os.path.join(self.weights_dir, weight.filename)
            status[weight.filename] = os.path.exists(path)
        return status
