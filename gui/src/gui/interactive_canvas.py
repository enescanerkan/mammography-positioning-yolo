"""
Interactive matplotlib canvas with zoom, pan, and draggable landmarks.
"""

import numpy as np
from typing import Optional, List, Callable, Tuple

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.text import Annotation

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QSizePolicy


LANDMARK_COLORS = {
    'nipple': '#10B981',
    'pec_top': '#EF4444',
    'pec_bottom': '#3B82F6',
    'cc_nipple': '#10B981',
}

LANDMARK_LABELS = {
    'nipple': 'Nipple',
    'pec_top': 'Pec Top',
    'pec_bottom': 'Pec Bottom',
    'cc_nipple': 'Nipple',
}

HIT_RADIUS = 15.0
ZOOM_FACTOR = 1.15


class DraggableLandmark:
    """A single draggable landmark point on the canvas."""

    def __init__(self, ax, x: float, y: float, name: str, color: str, label: str):
        self.ax = ax
        self.x = x
        self.y = y
        self.name = name
        self.color = color
        self.selected = False

        self.marker, = ax.plot(
            x, y, 'o', color=color, markersize=10,
            markeredgecolor='#FFFFFF', markeredgewidth=1.5,
            zorder=10, picker=True
        )
        self.halo, = ax.plot(
            x, y, 'o', color=color, markersize=22,
            alpha=0.0, zorder=9
        )
        self.label_text = ax.annotate(
            label, xy=(x, y), xytext=(10, -10),
            textcoords='offset points', fontsize=9, fontweight='bold',
            color=color, zorder=11,
            bbox=dict(boxstyle='round,pad=0.2', fc='#1A1D23', ec=color, alpha=0.85)
        )

    def update_position(self, x: float, y: float):
        self.x = x
        self.y = y
        self.marker.set_data([x], [y])
        self.halo.set_data([x], [y])
        self.label_text.set_position((10, -10))
        self.label_text.xy = (x, y)

    def set_selected(self, selected: bool):
        self.selected = selected
        if selected:
            self.halo.set_alpha(0.3)
            self.marker.set_markersize(13)
            self.marker.set_markeredgewidth(2.5)
        else:
            self.halo.set_alpha(0.0)
            self.marker.set_markersize(10)
            self.marker.set_markeredgewidth(1.5)

    def distance_to(self, x: float, y: float) -> float:
        return np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def remove(self):
        self.marker.remove()
        self.halo.remove()
        self.label_text.remove()


class InteractiveCanvas(FigureCanvas):
    """Matplotlib canvas with zoom, pan, and draggable landmark support."""

    landmarks_changed = pyqtSignal()

    def __init__(self, parent=None, facecolor='#22262E'):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.patch.set_facecolor(facecolor)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._image_shape: Optional[Tuple[int, int]] = None
        self._base_xlim: Optional[Tuple[float, float]] = None
        self._base_ylim: Optional[Tuple[float, float]] = None

        self._panning = False
        self._pan_start = None

        self.landmarks: List[DraggableLandmark] = []
        self._dragging: Optional[DraggableLandmark] = None
        self._drag_active = False

        self._overlay_artists: list = []
        self._distance_annotation: Optional[Annotation] = None

        self._cid_press = self.mpl_connect('button_press_event', self._on_press)
        self._cid_release = self.mpl_connect('button_release_event', self._on_release)
        self._cid_motion = self.mpl_connect('motion_notify_event', self._on_motion)
        self._cid_scroll = self.mpl_connect('scroll_event', self._on_scroll)

        self.on_landmarks_moved: Optional[Callable] = None

    # ── Image Display ──

    def display_image(self, image: np.ndarray, title: str = ""):
        self.ax.clear()
        self._clear_landmarks()
        self._clear_overlay()

        disp = image.copy()
        if len(disp.shape) == 3:
            disp = disp[0]

        self.ax.imshow(disp, cmap='gray', aspect='equal', interpolation='bilinear')
        h, w = disp.shape
        self._image_shape = (h, w)
        self._base_xlim = (-0.5, w - 0.5)
        self._base_ylim = (h - 0.5, -0.5)
        self.ax.set_xlim(self._base_xlim)
        self.ax.set_ylim(self._base_ylim)
        self.ax.set_title(title, color='#60A5FA', fontsize=11, pad=8)
        self._style_axes()
        self.draw_idle()

    def show_empty(self, text: str):
        self.ax.clear()
        self._clear_landmarks()
        self._clear_overlay()
        self._image_shape = None
        self.ax.set_facecolor('#1A1D23')
        self.ax.text(0.5, 0.5, text, ha='center', va='center',
                     transform=self.ax.transAxes, fontsize=16, color='#4B5563',
                     weight='light')
        self._style_axes()
        self.draw_idle()

    def _style_axes(self):
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

    # ── Landmarks ──

    def set_landmarks(self, names: List[str], coords: List[Tuple[float, float]]):
        self._clear_landmarks()
        for name, (x, y) in zip(names, coords):
            color = LANDMARK_COLORS.get(name, '#F59E0B')
            label = LANDMARK_LABELS.get(name, name)
            lm = DraggableLandmark(self.ax, x, y, name, color, label)
            self.landmarks.append(lm)
        self.draw_idle()

    def get_landmark_coords(self) -> dict:
        return {lm.name: (lm.x, lm.y) for lm in self.landmarks}

    def _clear_landmarks(self):
        for lm in self.landmarks:
            try:
                lm.remove()
            except Exception:
                pass
        self.landmarks.clear()

    # ── Overlay (lines, annotations) ──

    def clear_overlay(self):
        self._clear_overlay()
        self.draw_idle()

    def _clear_overlay(self):
        for a in self._overlay_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._overlay_artists.clear()
        self._distance_annotation = None

    def draw_line(self, x1, y1, x2, y2, color='#3B82F6', linewidth=2, linestyle='-'):
        line, = self.ax.plot([x1, x2], [y1, y2], linestyle,
                             color=color, linewidth=linewidth, zorder=5)
        self._overlay_artists.append(line)
        return line

    def draw_distance_label(self, x, y, text, bg_color='#F59E0B'):
        ann = self.ax.annotate(
            text, xy=(x, y), xytext=(8, 8),
            textcoords='offset points', fontsize=10, fontweight='bold',
            color='#1A1D23', zorder=12,
            bbox=dict(boxstyle='round,pad=0.3', fc=bg_color, ec='none')
        )
        self._overlay_artists.append(ann)
        self._distance_annotation = ann
        return ann

    def update_title(self, title: str, color: str = '#10B981'):
        self.ax.set_title(title, color=color, fontsize=11, pad=8)
        self.draw_idle()

    # ── Zoom ──

    def _on_scroll(self, event):
        if event.inaxes != self.ax or self._image_shape is None:
            return

        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        if event.button == 'up':
            scale = 1.0 / ZOOM_FACTOR
        else:
            scale = ZOOM_FACTOR

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale
        new_height = (cur_ylim[0] - cur_ylim[1]) * scale

        relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rely = (ydata - cur_ylim[1]) / (cur_ylim[0] - cur_ylim[1])

        new_xlim = [xdata - new_width * relx, xdata + new_width * (1 - relx)]
        new_ylim = [ydata + new_height * (1 - rely), ydata - new_height * rely]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.draw_idle()

    def reset_zoom(self):
        if self._base_xlim and self._base_ylim:
            self.ax.set_xlim(self._base_xlim)
            self.ax.set_ylim(self._base_ylim)
            self.draw_idle()

    # ── Pan & Drag ──

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return

        # Right click = reset zoom
        if event.button == 3:
            self.reset_zoom()
            return

        # Double click = reset zoom
        if event.dblclick:
            self.reset_zoom()
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Left click: try to grab a landmark first
        if event.button == 1 and self.landmarks:
            closest = min(self.landmarks, key=lambda lm: lm.distance_to(x, y))
            pixel_dist = closest.distance_to(x, y)

            xlim = self.ax.get_xlim()
            view_width = abs(xlim[1] - xlim[0])
            hit_threshold = max(HIT_RADIUS, view_width * 0.025)

            if pixel_dist < hit_threshold:
                for lm in self.landmarks:
                    lm.set_selected(False)
                closest.set_selected(True)
                self._dragging = closest
                self._drag_active = False
                self.draw_idle()
                return

        # Middle click or left click on empty = pan
        if event.button == 2 or (event.button == 1 and self._dragging is None):
            self._panning = True
            self._pan_start = (event.x, event.y,
                               self.ax.get_xlim(), self.ax.get_ylim())

    def _on_motion(self, event):
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Dragging landmark
        if self._dragging is not None and event.button == 1:
            self._drag_active = True
            self._dragging.update_position(x, y)
            if self.on_landmarks_moved:
                self.on_landmarks_moved()
            self.draw_idle()
            return

        # Panning
        if self._panning and self._pan_start is not None:
            sx, sy, xlim, ylim = self._pan_start
            dx = event.x - sx
            dy = event.y - sy

            fig_w = self.fig.get_figwidth() * self.fig.dpi
            fig_h = self.fig.get_figheight() * self.fig.dpi

            x_range = xlim[1] - xlim[0]
            y_range = ylim[0] - ylim[1]

            dx_data = -dx / fig_w * x_range
            dy_data = dy / fig_h * y_range

            self.ax.set_xlim(xlim[0] + dx_data, xlim[1] + dx_data)
            self.ax.set_ylim(ylim[0] + dy_data, ylim[1] + dy_data)
            self.draw_idle()
            return

        # Hover cursor
        if self.landmarks:
            closest = min(self.landmarks, key=lambda lm: lm.distance_to(x, y))
            xlim = self.ax.get_xlim()
            view_width = abs(xlim[1] - xlim[0])
            hit_threshold = max(HIT_RADIUS, view_width * 0.025)

            if closest.distance_to(x, y) < hit_threshold:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def _on_release(self, event):
        if self._dragging is not None:
            self._dragging.set_selected(False)
            if self._drag_active and self.on_landmarks_moved:
                self.on_landmarks_moved()
            self._dragging = None
            self._drag_active = False
            self.draw_idle()

        self._panning = False
        self._pan_start = None
        self.setCursor(Qt.ArrowCursor)

    # ── Redraw helpers ──

    def refresh(self):
        self.draw_idle()
