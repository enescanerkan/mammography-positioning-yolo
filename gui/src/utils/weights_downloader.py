"""
Weights Downloader Module.

This module handles automatic downloading of model weights from Google Drive.
Implements a clean, reusable download system with progress tracking.
"""

import os
import requests
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
    """Google Drive file downloader implementation."""
    
    DRIVE_URL = "https://drive.google.com/uc?export=download"
    CHUNK_SIZE = 32768
    
    def download(self, file_id: str, destination: str,
                 progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        """
        Download a file from Google Drive.
        
        Args:
            file_id: Google Drive file ID
            destination: Local path to save the file
            progress_callback: Optional callback for progress updates (0.0 to 1.0)
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            session = requests.Session()
            response = session.get(self.DRIVE_URL, params={'id': file_id}, stream=True)
            
            # Handle large file confirmation
            token = self._get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(self.DRIVE_URL, params=params, stream=True)
            
            return self._save_response(response, destination, progress_callback)
            
        except Exception as e:
            print(f"Download error: {e}")
            return False
    
    def _get_confirm_token(self, response: requests.Response) -> Optional[str]:
        """Extract confirmation token for large files."""
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    
    def _save_response(self, response: requests.Response, destination: str,
                       progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        """Save response content to file with progress tracking."""
        try:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(self.CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress_callback(downloaded / total_size)
            
            return True
            
        except Exception as e:
            print(f"Save error: {e}")
            return False


class WeightsManager:
    """
    Manages model weights downloading and verification.
    
    This class handles checking for existing weights and downloading
    missing ones from Google Drive.
    """
    
    # Default weight configurations
    DEFAULT_WEIGHTS: List[WeightFile] = [
        WeightFile(
            name="MLO YOLO26 Pose Model",
            drive_id="14OvSuC1XEvs_z5gsdgQ6I-P6JDlKb6l_",
            filename="mlo-yolo26-pose-advanced.pt",
            size_mb=120.0
        ),
        WeightFile(
            name="CC YOLO26 Pose Model",
            drive_id="1ZA3CY77hZupi9Nor9S18raPiikhVl5-s",
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
        Get list of weight files that are missing.
        
        Returns:
            List of WeightFile objects that need to be downloaded
        """
        missing = []
        for weight in self._weights_config:
            path = os.path.join(self.weights_dir, weight.filename)
            if not os.path.exists(path):
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
