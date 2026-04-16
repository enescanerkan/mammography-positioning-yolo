"""
Main application window -  Medical Imaging UI
Layout: Compact top bar + Large visualization area
"""

import sys
import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMessageBox, QApplication, QStatusBar,
    QGroupBox, QPushButton, QLabel, QTextEdit, QFileDialog,
    QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QTextCursor

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import cv2

from analysis.analysis_controller import AnalysisController
from models.model_manager import ModelManager
from data.data_manager import DataManager

# Dark theme stylesheet
STYLESHEET = """
QMainWindow { background-color: #1A1D23; }
QWidget { font-family: 'Segoe UI'; font-size: 12px; color: #F1F5F9; }
QGroupBox { background-color: #22262E; border: 1px solid #374151; border-radius: 8px; margin-top: 12px; padding: 12px; }
QGroupBox::title { color: #94A3B8; background-color: #22262E; }
QPushButton { background-color: #3B82F6; color: #FFFFFF; border: none; border-radius: 6px; padding: 8px 16px; font-weight: 500; }
QPushButton:hover { background-color: #60A5FA; }
QPushButton:pressed { background-color: #2563EB; }
QPushButton:disabled { background-color: #374151; color: #64748B; }
QPushButton[class="secondary"] { background-color: #2D323C; color: #F1F5F9; border: 1px solid #374151; }
QPushButton[class="secondary"]:hover { background-color: #374151; border-color: #3B82F6; }
QTextEdit { background-color: #1A1D23; border: 1px solid #374151; border-radius: 6px; padding: 8px; font-family: 'Consolas'; font-size: 11px; color: #F1F5F9; }
QTextEdit:focus { border-color: #3B82F6; }
QScrollBar:vertical { background: #22262E; width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background: #4B5563; border-radius: 4px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #3B82F6; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QStatusBar { background-color: #22262E; border-top: 1px solid #374151; color: #94A3B8; }
QToolTip { background-color: #2D323C; color: #F1F5F9; border: 1px solid #374151; border-radius: 4px; padding: 4px 8px; }
QMessageBox { background-color: #22262E; }
QMessageBox QLabel { color: #F1F5F9; }
"""

# Matplotlib dark style
matplotlib.rcParams.update({
    'figure.facecolor': '#22262E',
    'axes.facecolor': '#1A1D23',
    'text.color': '#F1F5F9',
    'axes.labelcolor': '#F1F5F9',
    'xtick.color': '#64748B',
    'ytick.color': '#64748B',
    'axes.edgecolor': '#374151',
})


class MainApplicationWindow(QMainWindow):
    """Professional Medical Imaging Application Window."""

    def __init__(self):
        super().__init__()
        self._setup_managers()
        self._setup_window()
        self._setup_ui()
        self._initialize_system()

    def _setup_managers(self):
        """Initialize managers."""
        self.data_manager = DataManager()
        self.model_manager = ModelManager()
        self.analysis_controller = AnalysisController(
            self.data_manager, self.model_manager
        )
        self.analysis_controller.on_mlo_analysis_complete = self._on_mlo_complete
        self.analysis_controller.on_cc_analysis_complete = self._on_cc_complete
        self.analysis_controller.on_comparison_complete = self._on_comparison_complete

    def _setup_window(self):
        """Setup window."""
        self.setWindowTitle("Mammogram Positioning Analysis")
        self.setMinimumSize(1400, 900)
        self.showMaximized()
        self.setStyleSheet(STYLESHEET)
        
        # Dark title bar for Windows
        self._set_dark_titlebar()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _set_dark_titlebar(self):
        """Set dark title bar on Windows."""
        try:
            import ctypes
            hwnd = int(self.winId())
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            value = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(value), ctypes.sizeof(value)
            )
        except Exception:
            pass

    def _setup_ui(self):
        """Setup UI - compact top, large visualization."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # === TOP BAR (compact) ===
        top_bar = QFrame()
        top_bar.setStyleSheet("background-color: #22262E; border-radius: 8px;")
        top_bar.setFixedHeight(70)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(16, 12, 16, 12)
        top_layout.setSpacing(16)

        # File selection
        self.select_btn = QPushButton("Select DICOM Pair")
        self.select_btn.setProperty("class", "secondary")
        self.select_btn.setMinimumWidth(140)
        self.select_btn.clicked.connect(self._select_files)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setProperty("class", "secondary")
        self.clear_btn.clicked.connect(self._clear_files)

        self.file_label = QLabel("No files selected")
        self.file_label.setStyleSheet("color: #94A3B8;")

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("background-color: #374151;")
        sep1.setFixedWidth(1)

        # Analysis buttons
        self.mlo_btn = QPushButton("MLO Analysis")
        self.mlo_btn.setProperty("class", "secondary")
        self.mlo_btn.clicked.connect(self._analyze_mlo)

        self.cc_btn = QPushButton("CC Analysis")
        self.cc_btn.setProperty("class", "secondary")
        self.cc_btn.clicked.connect(self._analyze_cc)

        self.compare_btn = QPushButton("Compare")
        self.compare_btn.setProperty("class", "secondary")
        self.compare_btn.clicked.connect(self._compare)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("background-color: #374151;")
        sep2.setFixedWidth(1)

        # Save buttons
        self.save_btn = QPushButton("Save Results")
        self.save_btn.setProperty("class", "secondary")
        self.save_btn.clicked.connect(self._save_results)

        self.save_img_btn = QPushButton("Save Images")
        self.save_img_btn.setProperty("class", "secondary")
        self.save_img_btn.clicked.connect(self._save_images)

        # System info (compact)
        self.sys_label = QLabel("Loading...")
        self.sys_label.setStyleSheet("""
            background-color: #2D323C;
            color: #94A3B8;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 11px;
        """)

        # Add to top bar
        top_layout.addWidget(self.select_btn)
        top_layout.addWidget(self.clear_btn)
        top_layout.addWidget(self.file_label, 1)
        top_layout.addWidget(sep1)
        top_layout.addWidget(self.mlo_btn)
        top_layout.addWidget(self.cc_btn)
        top_layout.addWidget(self.compare_btn)
        top_layout.addWidget(sep2)
        top_layout.addWidget(self.save_btn)
        top_layout.addWidget(self.save_img_btn)
        top_layout.addWidget(self.sys_label)

        main_layout.addWidget(top_bar)

        # === CONTENT AREA ===
        content = QHBoxLayout()
        content.setSpacing(10)

        # Left: Results log (fixed width)
        log_widget = QWidget()
        log_widget.setFixedWidth(320)
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)

        log_header = QLabel("ANALYSIS LOG")
        log_header.setStyleSheet("""
            color: #64748B;
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 1px;
            padding: 4px 0;
        """)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont('Cascadia Code', 10))

        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.setProperty("class", "secondary")
        clear_log_btn.clicked.connect(lambda: self.log_text.clear())

        log_layout.addWidget(log_header)
        log_layout.addWidget(self.log_text)
        log_layout.addWidget(clear_log_btn)

        # Right: Visualization (expanding)
        viz_widget = QWidget()
        viz_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        viz_layout = QHBoxLayout(viz_widget)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        viz_layout.setSpacing(10)

        # MLO Canvas
        mlo_container = QWidget()
        mlo_container.setStyleSheet("background-color: #22262E; border-radius: 8px;")
        mlo_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mlo_lay = QVBoxLayout(mlo_container)
        mlo_lay.setContentsMargins(8, 8, 8, 8)

        mlo_title = QLabel("MLO VIEW")
        mlo_title.setStyleSheet("color: #64748B; font-size: 10px; font-weight: 600; letter-spacing: 1px;")
        mlo_title.setAlignment(Qt.AlignCenter)

        self.mlo_fig, self.mlo_ax = plt.subplots(figsize=(10, 10))
        self.mlo_fig.patch.set_facecolor('#22262E')
        self.mlo_canvas = FigureCanvas(self.mlo_fig)
        self.mlo_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        mlo_lay.addWidget(mlo_title)
        mlo_lay.addWidget(self.mlo_canvas)

        # CC Canvas
        cc_container = QWidget()
        cc_container.setStyleSheet("background-color: #22262E; border-radius: 8px;")
        cc_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cc_lay = QVBoxLayout(cc_container)
        cc_lay.setContentsMargins(8, 8, 8, 8)

        cc_title = QLabel("CC VIEW")
        cc_title.setStyleSheet("color: #64748B; font-size: 10px; font-weight: 600; letter-spacing: 1px;")
        cc_title.setAlignment(Qt.AlignCenter)

        self.cc_fig, self.cc_ax = plt.subplots(figsize=(10, 10))
        self.cc_fig.patch.set_facecolor('#22262E')
        self.cc_canvas = FigureCanvas(self.cc_fig)
        self.cc_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        cc_lay.addWidget(cc_title)
        cc_lay.addWidget(self.cc_canvas)

        # Swap button between images
        swap_container = QWidget()
        swap_container.setFixedWidth(50)
        swap_layout = QVBoxLayout(swap_container)
        swap_layout.setContentsMargins(0, 0, 0, 0)
        swap_layout.addStretch()
        
        self.swap_btn = QPushButton("↔")
        self.swap_btn.setFixedSize(48, 48)
        self.swap_btn.setToolTip("Swap MLO and CC views")
        self.swap_btn.setStyleSheet("""
            QPushButton {
                background-color: #2D323C;
                color: #94A3B8;
                border: 2px solid #374151;
                border-radius: 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #374151;
                color: #F1F5F9;
                border-color: #3B82F6;
            }
            QPushButton:pressed {
                background-color: #3B82F6;
                color: #FFFFFF;
            }
        """)
        self.swap_btn.clicked.connect(self._swap_views)
        
        swap_layout.addWidget(self.swap_btn)
        swap_layout.addStretch()

        viz_layout.addWidget(mlo_container)
        viz_layout.addWidget(swap_container)
        viz_layout.addWidget(cc_container)

        content.addWidget(log_widget)
        content.addWidget(viz_widget, 1)

        main_layout.addLayout(content, 1)

        # Initialize empty plots
        self._show_empty(self.mlo_ax, "Select MLO DICOM")
        self._show_empty(self.cc_ax, "Select CC DICOM")
        self.mlo_canvas.draw()
        self.cc_canvas.draw()

        self._log("System initialized. Select DICOM pair to begin.")

    def _show_empty(self, ax, text):
        """Show empty plot."""
        ax.clear()
        ax.set_facecolor('#1A1D23')
        ax.text(0.5, 0.5, text, ha='center', va='center',
                transform=ax.transAxes, fontsize=16, color='#4B5563',
                weight='light')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _log(self, msg):
        """Add log message."""
        color = "#F1F5F9"
        if "=" in msg and len(msg) > 20:
            color = "#4B5563"
        self.log_text.append(f'<span style="color:{color}">{msg}</span>')
        self.log_text.moveCursor(QTextCursor.End)

    def _initialize_system(self):
        """Initialize system."""
        try:
            self.model_manager.load_models()
            self.data_manager.load_pixel_spacing_data()

            self.sys_label.setText(f"Ready | Device: {self.model_manager.device}")
            self.sys_label.setStyleSheet("""
                background-color: #2D323C;
                color: #10B981;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 11px;
            """)
            self._log("Models loaded successfully.")
            self.status_bar.showMessage("System ready")
        except Exception as e:
            self.sys_label.setText("Error")
            self._log(f"Init error: {e}")
            QMessageBox.critical(self, "Error", str(e))

    def _select_files(self):
        """Select DICOM files."""
        filt = "DICOM (*.dicom *.dcm);;All (*.*)"
        mlo, _ = QFileDialog.getOpenFileName(self, "Select MLO DICOM", "", filt)
        if not mlo:
            return
        cc, _ = QFileDialog.getOpenFileName(self, "Select CC DICOM", "", filt)
        if not cc:
            return

        try:
            self.analysis_controller.clear_results()
            self.data_manager.load_image_pair(mlo, cc)

            mlo_name = os.path.basename(mlo)
            cc_name = os.path.basename(cc)
            self.file_label.setText(f"MLO: {mlo_name} | CC: {cc_name}")
            self.file_label.setStyleSheet("color: #10B981; font-weight: 500;")

            self._log(f"\n{'='*40}")
            self._log(f"Loaded: {mlo_name}, {cc_name}")

            self._display_image(self.mlo_ax, self.mlo_canvas,
                                self.data_manager.current_mlo_image, "MLO")
            self._display_image(self.cc_ax, self.cc_canvas,
                                self.data_manager.current_cc_image, "CC")

            self._log("Ready for analysis")
            self.status_bar.showMessage(f"Loaded: {mlo_name}, {cc_name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _clear_files(self):
        """Clear all."""
        self.data_manager.clear_images()
        self.analysis_controller.clear_results()
        self._show_empty(self.mlo_ax, "Select MLO DICOM")
        self._show_empty(self.cc_ax, "Select CC DICOM")
        self.mlo_canvas.draw()
        self.cc_canvas.draw()
        self.file_label.setText("No files selected")
        self.file_label.setStyleSheet("color: #94A3B8;")
        self._log("Cleared")

    def _swap_views(self):
        """Swap MLO and CC image positions."""
        if self.data_manager.current_mlo_image is None or self.data_manager.current_cc_image is None:
            return
        
        # Swap images
        self.data_manager.current_mlo_image, self.data_manager.current_cc_image = \
            self.data_manager.current_cc_image, self.data_manager.current_mlo_image
        
        # Swap filenames
        self.data_manager.current_mlo_filename, self.data_manager.current_cc_filename = \
            self.data_manager.current_cc_filename, self.data_manager.current_mlo_filename
        
        # Swap pixel spacing
        self.data_manager.current_mlo_original_pixel_spacing, self.data_manager.current_cc_original_pixel_spacing = \
            self.data_manager.current_cc_original_pixel_spacing, self.data_manager.current_mlo_original_pixel_spacing
        
        # Swap original shapes
        self.data_manager.current_mlo_original_shape, self.data_manager.current_cc_original_shape = \
            self.data_manager.current_cc_original_shape, self.data_manager.current_mlo_original_shape
        
        # Swap transformation info
        self.data_manager.current_mlo_transformation_info, self.data_manager.current_cc_transformation_info = \
            self.data_manager.current_cc_transformation_info, self.data_manager.current_mlo_transformation_info
        
        # Clear analysis results since images swapped
        self.analysis_controller.clear_results()
        
        # Redisplay images
        self._display_image(self.mlo_ax, self.mlo_canvas,
                            self.data_manager.current_mlo_image, "MLO")
        self._display_image(self.cc_ax, self.cc_canvas,
                            self.data_manager.current_cc_image, "CC")
        
        self._log("Views swapped")
        self.status_bar.showMessage("Views swapped")

    def _prepare_image(self, img, size=800):
        """Prepare image for display."""
        disp = img.copy()
        if len(disp.shape) == 3:
            disp = disp[0]
        h, w = disp.shape
        if h > size or w > size:
            scale = min(size/h, size/w)
            nh, nw = int(h*scale), int(w*scale)
            disp = cv2.resize(disp, (nw, nh), interpolation=cv2.INTER_AREA)
            return disp, nw/w, nh/h
        return disp, 1.0, 1.0

    def _display_image(self, ax, canvas, img, title):
        """Display image."""
        if img is None:
            return
        ax.clear()
        disp, _, _ = self._prepare_image(img)
        ax.imshow(disp, cmap='gray', aspect='equal')
        ax.set_xlim(0, disp.shape[1])
        ax.set_ylim(disp.shape[0], 0)
        ax.set_title(title, color='#60A5FA', fontsize=11, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        for spine in ax.spines.values():
            spine.set_visible(False)
        canvas.draw()

    def _analyze_mlo(self):
        """MLO analysis."""
        if self.data_manager.current_mlo_image is None:
            QMessageBox.warning(self, "Warning", "Load MLO image first!")
            return
        self._log("\nMLO Analysis...")
        self.status_bar.showMessage("Analyzing MLO...")
        QApplication.processEvents()

        results = self.analysis_controller.analyze_mlo()
        if results:
            self._log_results("MLO", results)

    def _analyze_cc(self):
        """CC analysis."""
        if self.data_manager.current_cc_image is None:
            QMessageBox.warning(self, "Warning", "Load CC image first!")
            return
        self._log("\nCC Analysis...")
        self.status_bar.showMessage("Analyzing CC...")
        QApplication.processEvents()

        results = self.analysis_controller.analyze_cc()
        if results:
            self._log_cc_results(results)

    def _compare(self):
        """Compare results."""
        status = self.analysis_controller.has_results()
        if not status['mlo']:
            QMessageBox.warning(self, "Warning", "Run MLO analysis first!")
            return
        if not status['cc']:
            QMessageBox.warning(self, "Warning", "Run CC analysis first!")
            return

        comp = self.analysis_controller.compare_results()
        if comp:
            self._log(f"\n{'='*40}")
            self._log("COMPARISON")
            self._log(f"MLO: {comp['mlo_distance']:.2f} mm")
            self._log(f"CC: {comp['cc_distance']:.2f} mm")
            self._log(f"Diff: {comp['difference']:.2f} mm")
            self._log(f"{comp['quality_result']}")

    def _save_results(self):
        """Save results."""
        status = self.analysis_controller.has_results()
        if not status['mlo'] and not status['cc']:
            QMessageBox.warning(self, "Warning", "No results to save!")
            return
        saved = self.analysis_controller.save_results()
        if saved:
            self._log(f"Saved: {', '.join(saved)}")
            QMessageBox.information(self, "Saved", "\n".join(saved))

    def _save_images(self):
        """Save images."""
        status = self.analysis_controller.has_results()
        if not status['mlo'] and not status['cc']:
            QMessageBox.warning(self, "Warning", "No images to save!")
            return

        import pandas as pd
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        saved = []

        if status['mlo']:
            f = os.path.join(results_dir, f"mlo_{ts}.png")
            self.mlo_fig.savefig(f, dpi=150, bbox_inches='tight', facecolor='#22262E')
            saved.append(f)
        if status['cc']:
            f = os.path.join(results_dir, f"cc_{ts}.png")
            self.cc_fig.savefig(f, dpi=150, bbox_inches='tight', facecolor='#22262E')
            saved.append(f)

        self._log(f"Saved: {', '.join(saved)}")
        QMessageBox.information(self, "Saved", "\n".join(saved))

    def _log_results(self, view, r):
        """Log MLO results."""
        self._log(f"\n{'='*40}")
        self._log(f"{view} RESULTS")
        lm = r['landmarks']
        self._log(f"Nipple: [{lm[0][0]:.0f}, {lm[0][1]:.0f}]")
        self._log(f"Pectoral 1: [{lm[1][0]:.0f}, {lm[1][1]:.0f}]")
        self._log(f"Pectoral 2: [{lm[2][0]:.0f}, {lm[2][1]:.0f}]")
        self._log(f"Distance: {r['distance_mm']:.2f} mm")
        self.status_bar.showMessage(f"{view} complete: {r['distance_mm']:.2f} mm")

    def _log_cc_results(self, r):
        """Log CC results."""
        self._log(f"\n{'='*40}")
        self._log("CC RESULTS")
        lm = r['landmarks']
        self._log(f"Nipple: [{lm[0][0]:.0f}, {lm[0][1]:.0f}]")
        self._log(f"Distance: {r['distance_mm']:.2f} mm ({r['direction']})")
        self.status_bar.showMessage(f"CC complete: {r['distance_mm']:.2f} mm")

    def _on_mlo_complete(self, r):
        """MLO analysis complete callback."""
        self._display_mlo_analysis(r)

    def _on_cc_complete(self, r):
        """CC analysis complete callback."""
        self._display_cc_analysis(r)

    def _on_comparison_complete(self, c):
        """Comparison complete callback."""
        QMessageBox.information(self, "Assessment", c['result_text'])

    def _display_mlo_analysis(self, r):
        """Display MLO with analysis."""
        img = self.data_manager.current_mlo_image
        if img is None:
            return

        self.mlo_ax.clear()
        disp, sx, sy = self._prepare_image(img)
        self.mlo_ax.imshow(disp, cmap='gray')

        lm = r['landmarks']
        inter = r['intersection']
        dist = r['distance_mm']

        nip = [lm[0][0]*sx, lm[0][1]*sy]
        p1 = [lm[1][0]*sx, lm[1][1]*sy]
        p2 = [lm[2][0]*sx, lm[2][1]*sy]
        ins = [inter[0]*sx, inter[1]*sy]

        self.mlo_ax.plot(p1[0], p1[1], 'o', color='#EF4444', ms=9, mec='#1A1D23', mew=1)
        self.mlo_ax.plot(p2[0], p2[1], 'o', color='#3B82F6', ms=9, mec='#1A1D23', mew=1)
        self.mlo_ax.plot(nip[0], nip[1], 'o', color='#10B981', ms=9, mec='#1A1D23', mew=1)
        self.mlo_ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color='#3B82F6', lw=2)
        self.mlo_ax.plot([nip[0], ins[0]], [nip[1], ins[1]], '--', color='#EF4444', lw=2)

        mid = [(nip[0]+ins[0])/2, (nip[1]+ins[1])/2]
        self.mlo_ax.annotate(f"{dist:.1f} mm", xy=mid, xytext=(8, 8),
                             textcoords='offset points', fontsize=10, fontweight='bold',
                             color='#1A1D23',
                             bbox=dict(boxstyle='round,pad=0.3', fc='#F59E0B', ec='none'))

        self.mlo_ax.set_xlim(0, disp.shape[1])
        self.mlo_ax.set_ylim(disp.shape[0], 0)
        self.mlo_ax.set_title(f"MLO - {dist:.2f} mm", color='#10B981', fontsize=11, pad=8)
        self.mlo_ax.set_xticks([])
        self.mlo_ax.set_yticks([])
        self.mlo_ax.set_aspect('equal', adjustable='box')
        for spine in self.mlo_ax.spines.values():
            spine.set_visible(False)
        self.mlo_canvas.draw()

    def _display_cc_analysis(self, r):
        """Display CC with analysis."""
        img = self.data_manager.current_cc_image
        if img is None:
            return

        self.cc_ax.clear()
        disp, sx, sy = self._prepare_image(img)
        self.cc_ax.imshow(disp, cmap='gray')

        lm = r['landmarks']
        edge = r['edge_point']
        dist = r['distance_mm']

        nip = [lm[0][0]*sx, lm[0][1]*sy]
        ep = [edge[0]*sx, edge[1]*sy]

        self.cc_ax.plot(nip[0], nip[1], 'o', color='#10B981', ms=9, mec='#1A1D23', mew=1)
        self.cc_ax.plot([nip[0], ep[0]], [nip[1], ep[1]], '--', color='#EF4444', lw=2)

        mid = [(nip[0]+ep[0])/2, (nip[1]+ep[1])/2]
        self.cc_ax.annotate(f"{dist:.1f} mm", xy=mid, xytext=(8, 8),
                            textcoords='offset points', fontsize=10, fontweight='bold',
                            color='#1A1D23',
                            bbox=dict(boxstyle='round,pad=0.3', fc='#F59E0B', ec='none'))

        self.cc_ax.set_xlim(0, disp.shape[1])
        self.cc_ax.set_ylim(disp.shape[0], 0)
        self.cc_ax.set_title(f"CC - {dist:.2f} mm", color='#10B981', fontsize=11, pad=8)
        self.cc_ax.set_xticks([])
        self.cc_ax.set_yticks([])
        self.cc_ax.set_aspect('equal', adjustable='box')
        for spine in self.cc_ax.spines.values():
            spine.set_visible(False)
        self.cc_canvas.draw()

    def closeEvent(self, event):
        """Close event."""
        plt.close('all')
        event.accept()
