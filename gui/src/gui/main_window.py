"""
Main application window -  Medical Imaging UI
Layout: Compact top bar + Large visualization area
Features: Zoom/Pan, Draggable landmarks, Live recalculation
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

from gui.interactive_canvas import InteractiveCanvas
from analysis.analysis_controller import AnalysisController
from analysis.mlo_analyzer import MLOAnalyzer
from analysis.cc_analyzer import CCAnalyzer
from models.model_manager import ModelManager
from data.data_manager import DataManager

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
        self._mlo_results = None
        self._cc_results = None
        self._setup_managers()
        self._setup_window()
        self._setup_ui()
        self._initialize_system()

    def _setup_managers(self):
        self.data_manager = DataManager()
        self.model_manager = ModelManager()
        self.analysis_controller = AnalysisController(
            self.data_manager, self.model_manager
        )
        self.analysis_controller.on_mlo_analysis_complete = self._on_mlo_complete
        self.analysis_controller.on_cc_analysis_complete = self._on_cc_complete
        self.analysis_controller.on_comparison_complete = self._on_comparison_complete

    def _setup_window(self):
        self.setWindowTitle("Mammogram Positioning Analysis")
        self.setMinimumSize(1400, 900)
        self.showMaximized()
        self.setStyleSheet(STYLESHEET)
        self._set_dark_titlebar()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _set_dark_titlebar(self):
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
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # === TOP BAR ===
        top_bar = QFrame()
        top_bar.setStyleSheet("background-color: #22262E; border-radius: 8px;")
        top_bar.setFixedHeight(70)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(16, 12, 16, 12)
        top_layout.setSpacing(16)

        self.select_btn = QPushButton("Select DICOM Pair")
        self.select_btn.setProperty("class", "secondary")
        self.select_btn.setMinimumWidth(140)
        self.select_btn.clicked.connect(self._select_files)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setProperty("class", "secondary")
        self.clear_btn.clicked.connect(self._clear_files)

        self.file_label = QLabel("No files selected")
        self.file_label.setStyleSheet("color: #94A3B8;")

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("background-color: #374151;")
        sep1.setFixedWidth(1)

        self.mlo_btn = QPushButton("MLO Analysis")
        self.mlo_btn.setProperty("class", "secondary")
        self.mlo_btn.clicked.connect(self._analyze_mlo)

        self.cc_btn = QPushButton("CC Analysis")
        self.cc_btn.setProperty("class", "secondary")
        self.cc_btn.clicked.connect(self._analyze_cc)

        self.compare_btn = QPushButton("Compare")
        self.compare_btn.setProperty("class", "secondary")
        self.compare_btn.clicked.connect(self._compare)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("background-color: #374151;")
        sep2.setFixedWidth(1)

        self.save_btn = QPushButton("Save Results")
        self.save_btn.setProperty("class", "secondary")
        self.save_btn.clicked.connect(self._save_results)

        self.save_img_btn = QPushButton("Save Images")
        self.save_img_btn.setProperty("class", "secondary")
        self.save_img_btn.clicked.connect(self._save_images)

        self.sys_label = QLabel("Loading...")
        self.sys_label.setStyleSheet("""
            background-color: #2D323C;
            color: #94A3B8;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 11px;
        """)

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

        # Left: Results log
        log_widget = QWidget()
        log_widget.setFixedWidth(320)
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)

        log_header = QLabel("ANALYSIS LOG")
        log_header.setStyleSheet("""
            color: #64748B; font-size: 10px;
            font-weight: 600; letter-spacing: 1px; padding: 4px 0;
        """)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont('Cascadia Code', 10))

        # Zoom hint
        zoom_hint = QLabel("Scroll: Zoom | Right-click: Reset | Drag landmarks to adjust")
        zoom_hint.setStyleSheet("color: #4B5563; font-size: 9px; padding: 4px 0;")
        zoom_hint.setWordWrap(True)

        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.setProperty("class", "secondary")
        clear_log_btn.clicked.connect(lambda: self.log_text.clear())

        log_layout.addWidget(log_header)
        log_layout.addWidget(self.log_text)
        log_layout.addWidget(zoom_hint)
        log_layout.addWidget(clear_log_btn)

        # Right: Visualization with InteractiveCanvas
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

        self.mlo_canvas = InteractiveCanvas(parent=mlo_container)
        self.mlo_canvas.on_landmarks_moved = self._on_mlo_landmarks_moved

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

        self.cc_canvas = InteractiveCanvas(parent=cc_container)
        self.cc_canvas.on_landmarks_moved = self._on_cc_landmarks_moved

        cc_lay.addWidget(cc_title)
        cc_lay.addWidget(self.cc_canvas)

        # Swap button
        swap_container = QWidget()
        swap_container.setFixedWidth(50)
        swap_layout = QVBoxLayout(swap_container)
        swap_layout.setContentsMargins(0, 0, 0, 0)
        swap_layout.addStretch()

        self.swap_btn = QPushButton("\u2194")
        self.swap_btn.setFixedSize(48, 48)
        self.swap_btn.setToolTip("Swap MLO and CC views")
        self.swap_btn.setStyleSheet("""
            QPushButton {
                background-color: #2D323C; color: #94A3B8;
                border: 2px solid #374151; border-radius: 20px;
                font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #374151; color: #F1F5F9; border-color: #3B82F6; }
            QPushButton:pressed { background-color: #3B82F6; color: #FFFFFF; }
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
        self.mlo_canvas.show_empty("Select MLO DICOM")
        self.cc_canvas.show_empty("Select CC DICOM")

        self._log("System initialized. Select DICOM pair to begin.")

    def _log(self, msg):
        color = "#F1F5F9"
        if "=" in msg and len(msg) > 20:
            color = "#4B5563"
        self.log_text.append(f'<span style="color:{color}">{msg}</span>')
        self.log_text.moveCursor(QTextCursor.End)

    def _initialize_system(self):
        try:
            self.model_manager.load_models()
            self.data_manager.load_pixel_spacing_data()

            self.sys_label.setText(f"Ready | Device: {self.model_manager.device}")
            self.sys_label.setStyleSheet("""
                background-color: #2D323C; color: #10B981;
                padding: 8px 12px; border-radius: 6px; font-size: 11px;
            """)
            self._log("Models loaded successfully.")
            self.status_bar.showMessage("System ready")
        except Exception as e:
            self.sys_label.setText("Error")
            self._log(f"Init error: {e}")
            QMessageBox.critical(self, "Error", str(e))

    # ── File Operations ──

    def _select_files(self):
        filt = "DICOM (*.dicom *.dcm);;All (*.*)"
        mlo, _ = QFileDialog.getOpenFileName(self, "Select MLO DICOM", "", filt)
        if not mlo:
            return
        cc, _ = QFileDialog.getOpenFileName(self, "Select CC DICOM", "", filt)
        if not cc:
            return

        try:
            self.analysis_controller.clear_results()
            self._mlo_results = None
            self._cc_results = None
            self.data_manager.load_image_pair(mlo, cc)

            mlo_name = os.path.basename(mlo)
            cc_name = os.path.basename(cc)
            self.file_label.setText(f"MLO: {mlo_name} | CC: {cc_name}")
            self.file_label.setStyleSheet("color: #10B981; font-weight: 500;")

            self._log(f"\n{'='*40}")
            self._log(f"Loaded: {mlo_name}, {cc_name}")

            self.mlo_canvas.display_image(self.data_manager.current_mlo_image, "MLO")
            self.cc_canvas.display_image(self.data_manager.current_cc_image, "CC")

            self._log("Ready for analysis")
            self.status_bar.showMessage(f"Loaded: {mlo_name}, {cc_name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _clear_files(self):
        self.data_manager.clear_images()
        self.analysis_controller.clear_results()
        self._mlo_results = None
        self._cc_results = None
        self.mlo_canvas.show_empty("Select MLO DICOM")
        self.cc_canvas.show_empty("Select CC DICOM")
        self.file_label.setText("No files selected")
        self.file_label.setStyleSheet("color: #94A3B8;")
        self._log("Cleared")

    def _swap_views(self):
        if self.data_manager.current_mlo_image is None or self.data_manager.current_cc_image is None:
            return

        self.data_manager.current_mlo_image, self.data_manager.current_cc_image = \
            self.data_manager.current_cc_image, self.data_manager.current_mlo_image
        self.data_manager.current_mlo_filename, self.data_manager.current_cc_filename = \
            self.data_manager.current_cc_filename, self.data_manager.current_mlo_filename
        self.data_manager.current_mlo_original_pixel_spacing, self.data_manager.current_cc_original_pixel_spacing = \
            self.data_manager.current_cc_original_pixel_spacing, self.data_manager.current_mlo_original_pixel_spacing
        self.data_manager.current_mlo_original_shape, self.data_manager.current_cc_original_shape = \
            self.data_manager.current_cc_original_shape, self.data_manager.current_mlo_original_shape
        self.data_manager.current_mlo_transformation_info, self.data_manager.current_cc_transformation_info = \
            self.data_manager.current_cc_transformation_info, self.data_manager.current_mlo_transformation_info

        self.analysis_controller.clear_results()
        self._mlo_results = None
        self._cc_results = None

        self.mlo_canvas.display_image(self.data_manager.current_mlo_image, "MLO")
        self.cc_canvas.display_image(self.data_manager.current_cc_image, "CC")

        self._log("Views swapped")
        self.status_bar.showMessage("Views swapped")

    # ── Analysis ──

    def _analyze_mlo(self):
        if self.data_manager.current_mlo_image is None:
            QMessageBox.warning(self, "Warning", "Load MLO image first!")
            return
        self._log("\nMLO Analysis...")
        self.status_bar.showMessage("Analyzing MLO...")
        QApplication.processEvents()

        results = self.analysis_controller.analyze_mlo()
        if results:
            self._mlo_results = results
            self._log_mlo_results(results)

    def _analyze_cc(self):
        if self.data_manager.current_cc_image is None:
            QMessageBox.warning(self, "Warning", "Load CC image first!")
            return
        self._log("\nCC Analysis...")
        self.status_bar.showMessage("Analyzing CC...")
        QApplication.processEvents()

        results = self.analysis_controller.analyze_cc()
        if results:
            self._cc_results = results
            self._log_cc_results(results)

    def _compare(self):
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

    # ── Save ──

    def _save_results(self):
        status = self.analysis_controller.has_results()
        if not status['mlo'] and not status['cc']:
            QMessageBox.warning(self, "Warning", "No results to save!")
            return
        saved = self.analysis_controller.save_results()
        if saved:
            self._log(f"Saved: {', '.join(saved)}")
            QMessageBox.information(self, "Saved", "\n".join(saved))

    def _save_images(self):
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
            self.mlo_canvas.fig.savefig(f, dpi=150, bbox_inches='tight', facecolor='#22262E')
            saved.append(f)
        if status['cc']:
            f = os.path.join(results_dir, f"cc_{ts}.png")
            self.cc_canvas.fig.savefig(f, dpi=150, bbox_inches='tight', facecolor='#22262E')
            saved.append(f)

        self._log(f"Saved: {', '.join(saved)}")
        QMessageBox.information(self, "Saved", "\n".join(saved))

    # ── Logging ──

    def _log_mlo_results(self, r):
        self._log(f"\n{'='*40}")
        self._log("MLO RESULTS")
        lm = r['landmarks']
        self._log(f"Nipple: [{lm[0][0]:.0f}, {lm[0][1]:.0f}]")
        self._log(f"Pec Top: [{lm[1][0]:.0f}, {lm[1][1]:.0f}]")
        self._log(f"Pec Bottom: [{lm[2][0]:.0f}, {lm[2][1]:.0f}]")
        self._log(f"Distance: {r['distance_mm']:.2f} mm")
        self.status_bar.showMessage(f"MLO complete: {r['distance_mm']:.2f} mm")

    def _log_cc_results(self, r):
        self._log(f"\n{'='*40}")
        self._log("CC RESULTS")
        lm = r['landmarks']
        self._log(f"Nipple: [{lm[0][0]:.0f}, {lm[0][1]:.0f}]")
        self._log(f"Distance: {r['distance_mm']:.2f} mm ({r['direction']})")
        self.status_bar.showMessage(f"CC complete: {r['distance_mm']:.2f} mm")

    # ── Analysis Display Callbacks ──

    def _on_mlo_complete(self, r):
        self._display_mlo_analysis(r)

    def _on_cc_complete(self, r):
        self._display_cc_analysis(r)

    def _on_comparison_complete(self, c):
        QMessageBox.information(self, "Assessment", c['result_text'])

    # ── MLO Display + Overlay ──

    def _display_mlo_analysis(self, r):
        img = self.data_manager.current_mlo_image
        if img is None:
            return

        self.mlo_canvas.display_image(img, "MLO")

        lm = r['landmarks']
        self.mlo_canvas.set_landmarks(
            ['nipple', 'pec_top', 'pec_bottom'],
            [(lm[0][0], lm[0][1]), (lm[1][0], lm[1][1]), (lm[2][0], lm[2][1])]
        )

        self._draw_mlo_overlay(r)

    def _draw_mlo_overlay(self, r):
        self.mlo_canvas._clear_overlay()

        lm = r['landmarks']
        inter = r['intersection']
        dist = r['distance_mm']

        nip = lm[0]
        p1 = lm[1]
        p2 = lm[2]

        self.mlo_canvas.draw_line(p1[0], p1[1], p2[0], p2[1],
                                  color='#3B82F6', linewidth=2, linestyle='-')
        self.mlo_canvas.draw_line(nip[0], nip[1], inter[0], inter[1],
                                  color='#EF4444', linewidth=2, linestyle='--')

        mid_x = (nip[0] + inter[0]) / 2
        mid_y = (nip[1] + inter[1]) / 2
        self.mlo_canvas.draw_distance_label(mid_x, mid_y, f"{dist:.1f} mm")

        self.mlo_canvas.update_title(f"MLO - {dist:.2f} mm")
        self.mlo_canvas.draw_idle()

    # ── CC Display + Overlay ──

    def _display_cc_analysis(self, r):
        img = self.data_manager.current_cc_image
        if img is None:
            return

        self.cc_canvas.display_image(img, "CC")

        lm = r['landmarks']
        self.cc_canvas.set_landmarks(
            ['cc_nipple'],
            [(lm[0][0], lm[0][1])]
        )

        self._draw_cc_overlay(r)

    def _draw_cc_overlay(self, r):
        self.cc_canvas._clear_overlay()

        lm = r['landmarks']
        edge = r['edge_point']
        dist = r['distance_mm']

        nip = lm[0]

        self.cc_canvas.draw_line(nip[0], nip[1], edge[0], edge[1],
                                 color='#EF4444', linewidth=2, linestyle='--')

        mid_x = (nip[0] + edge[0]) / 2
        mid_y = (nip[1] + edge[1]) / 2
        self.cc_canvas.draw_distance_label(mid_x, mid_y, f"{dist:.1f} mm")

        self.cc_canvas.update_title(f"CC - {dist:.2f} mm")
        self.cc_canvas.draw_idle()

    # ── Landmark Drag Recalculation ──

    def _on_mlo_landmarks_moved(self):
        if self._mlo_results is None:
            return

        coords = self.mlo_canvas.get_landmark_coords()
        if 'nipple' not in coords or 'pec_top' not in coords or 'pec_bottom' not in coords:
            return

        nipple = np.array(coords['nipple'])
        pec_top = np.array(coords['pec_top'])
        pec_bottom = np.array(coords['pec_bottom'])

        perp_dist_px, intersection = MLOAnalyzer.perpendicular_distance(
            pec_top, pec_bottom, nipple
        )

        scaled_ps = self._mlo_results.get('scaled_pixel_spacing',
                                           self._mlo_results.get('pixel_spacing', 0.085))
        dist_mm = perp_dist_px * scaled_ps

        updated = dict(self._mlo_results)
        updated['landmarks'] = np.array([nipple, pec_top, pec_bottom])
        updated['intersection'] = intersection
        updated['distance_pixels'] = perp_dist_px
        updated['distance_mm'] = dist_mm
        self._mlo_results = updated

        self.analysis_controller.mlo_results = updated

        self._draw_mlo_overlay(updated)

        self._log(f"MLO adjusted: {dist_mm:.2f} mm")
        self.status_bar.showMessage(f"MLO adjusted: {dist_mm:.2f} mm")

    def _on_cc_landmarks_moved(self):
        if self._cc_results is None:
            return

        coords = self.cc_canvas.get_landmark_coords()
        if 'cc_nipple' not in coords:
            return

        nipple = np.array(coords['cc_nipple'])
        breast_side = self._cc_results.get('breast_side', 'LEFT')

        direction, dist_px, edge_point = CCAnalyzer.edge_distance(
            nipple, 640, breast_side
        )

        scaled_ps = self._cc_results.get('scaled_pixel_spacing',
                                          self._cc_results.get('pixel_spacing', 0.085))
        dist_mm = dist_px * scaled_ps

        updated = dict(self._cc_results)
        updated['landmarks'] = np.array([nipple])
        updated['edge_point'] = edge_point
        updated['direction'] = direction
        updated['distance_pixels'] = dist_px
        updated['distance_mm'] = dist_mm
        self._cc_results = updated

        self.analysis_controller.cc_results = updated

        self._draw_cc_overlay(updated)

        self._log(f"CC adjusted: {dist_mm:.2f} mm ({direction})")
        self.status_bar.showMessage(f"CC adjusted: {dist_mm:.2f} mm")

    # ── Cleanup ──

    def closeEvent(self, event):
        plt.close('all')
        event.accept()
