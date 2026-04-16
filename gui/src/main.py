#!/usr/bin/env python3
"""
Mammogram Positioning Analysis - Main Entry Point

Medical Imaging Application for analyzing mammogram positioning quality
using deep learning-based landmark detection.
"""

import sys
import os

src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# torch must be imported before PyQt5 to avoid DLL conflicts on Windows
import torch  # noqa: F401

from PyQt5.QtWidgets import QApplication, QMessageBox, QProgressDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from gui.main_window import MainApplicationWindow
from utils.logger import setup_logging
from utils.weights_downloader import WeightsManager


def check_and_download_weights(app: QApplication) -> bool:
    """
    Check for model weights and download if missing.
    
    Args:
        app: QApplication instance for dialog display
        
    Returns:
        True if weights are available, False if download failed
    """
    weights_manager = WeightsManager()
    
    if weights_manager.all_weights_exist():
        return True
    
    missing = weights_manager.get_missing_weights()
    missing_names = [w.name for w in missing]
    
    # Check if Drive IDs are configured
    has_valid_ids = all(
        "YOUR_" not in w.drive_id and w.drive_id != "" 
        for w in missing
    )
    
    if not has_valid_ids:
        QMessageBox.critical(
            None,
            "Missing Model Weights",
            f"The following model weights are missing:\n\n"
            f"• {chr(10).join(missing_names)}\n\n"
            f"Please download the weights manually and place them in the 'weights' folder.\n\n"
            f"Required files:\n"
            f"• mlo-yolo26-pose-advanced.pt\n"
            f"• cc-yolo26-pose-advanced.pt"
        )
        return False
    
    # Ask user to download
    reply = QMessageBox.question(
        None,
        "Download Model Weights",
        f"The following model weights need to be downloaded:\n\n"
        f"• {chr(10).join(missing_names)}\n\n"
        f"Download now? (This may take a few minutes)",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes
    )
    
    if reply != QMessageBox.Yes:
        return False
    
    # Show progress dialog
    progress = QProgressDialog("Downloading model weights...", "Cancel", 0, 100)
    progress.setWindowTitle("Downloading")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)
    
    current_file = [""]
    
    def update_progress(filename: str, file_progress: float):
        current_file[0] = filename
        progress.setLabelText(f"Downloading {filename}...\n{int(file_progress * 100)}%")
        progress.setValue(int(file_progress * 100))
        app.processEvents()
    
    results = weights_manager.download_missing_weights(update_progress)
    progress.close()
    
    # Check results
    failed = [f for f, success in results.items() if not success]
    
    if failed:
        QMessageBox.warning(
            None,
            "Download Failed",
            f"Failed to download:\n• {chr(10).join(failed)}\n\n"
            f"Please download manually and place in 'weights' folder."
        )
        return False
    
    QMessageBox.information(
        None,
        "Download Complete",
        "Model weights downloaded successfully!"
    )
    return True


def main():
    """Main entry point for the application."""
    setup_logging()

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Mammogram Positioning Analysis")
    app.setStyle('Fusion')

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Check and download weights if needed
    if not check_and_download_weights(app):
        sys.exit(1)

    window = MainApplicationWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
