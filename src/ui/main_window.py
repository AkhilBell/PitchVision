"""Main application window for PitchVision Tactical Studio.

Assembles the video canvas, radar minimap, and live heatmap panel into a
single PySide6 ``QMainWindow`` with toolbar and status-bar controls.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.ui.heatmap_widget import HeatmapWidget
from src.ui.radar_widget import RadarWidget
from src.ui.video_canvas import VideoCanvas

if TYPE_CHECKING:
    from PySide6.QtGui import QCloseEvent

    from src.core.packet import FramePacket

# Pipeline worker is imported lazily so the module can be loaded even
# when the pipeline package is not yet available.
try:
    from src.core.pipeline_worker import PipelineWorker  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    PipelineWorker = None  # type: ignore[assignment,misc]


class MainWindow(QMainWindow):
    """Top-level window for PitchVision.

    Layout::

        ┌──────────────────────────────────────────────────────────┐
        │ Toolbar: [Load Video] [Play/Pause] [Export MP4]          │
        ├────────────────────────────────┬─────────────────────────┤
        │                                │  RadarWidget            │
        │   VideoCanvas  (65 %)          ├─────────────────────────┤
        │                                │  HeatmapWidget          │
        ├────────────────────────────────┴─────────────────────────┤
        │ Status: ☑ Foot Markers ☑ Formation Lines ☑ Ball Trail FPS│
        └──────────────────────────────────────────────────────────┘
    """

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("PitchVision Tactical Studio")
        self.setMinimumSize(1400, 800)

        # ----- Pipeline worker (created on video load) -----
        self._worker: PipelineWorker | None = None  # type: ignore[assignment]
        self._paused: bool = False

        # ----- FPS tracking -----
        self._fps_timestamps: list[float] = []
        self._fps_window: int = 30  # rolling window of N frames

        # ----- Central widget & main layout -----
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left: video canvas (stretch factor = 65)
        self.video_canvas = VideoCanvas()
        main_layout.addWidget(self.video_canvas, 65)

        # Right: radar + heatmap (stretch factor = 35)
        right_panel = QVBoxLayout()
        self.radar_widget = RadarWidget()
        self.heatmap_widget = HeatmapWidget()
        right_panel.addWidget(self.radar_widget, 50)
        right_panel.addWidget(self.heatmap_widget, 50)
        main_layout.addLayout(right_panel, 35)

        # ----- Toolbar -----
        self._build_toolbar()

        # ----- Status bar -----
        self._build_status_bar()

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        """Create the top toolbar with Load / Play-Pause / Export buttons."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self._btn_load = QPushButton("Load Video")
        self._btn_load.clicked.connect(self._on_load_video)
        toolbar.addWidget(self._btn_load)

        self._btn_play_pause = QPushButton("Play/Pause")
        self._btn_play_pause.setCheckable(True)
        self._btn_play_pause.clicked.connect(self._on_play_pause)
        toolbar.addWidget(self._btn_play_pause)

        self._btn_export = QPushButton("Export MP4")
        self._btn_export.clicked.connect(self._on_export)
        toolbar.addWidget(self._btn_export)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _build_status_bar(self) -> None:
        """Create the bottom status bar with overlay toggles and FPS."""
        status = self.statusBar()

        self._chk_foot_markers = QCheckBox("Show Foot Markers")
        self._chk_foot_markers.setChecked(True)
        self._chk_foot_markers.stateChanged.connect(self._sync_overlay_flags)
        status.addPermanentWidget(self._chk_foot_markers)

        self._chk_formation_lines = QCheckBox("Show Formation Lines")
        self._chk_formation_lines.setChecked(True)
        self._chk_formation_lines.stateChanged.connect(self._sync_overlay_flags)
        status.addPermanentWidget(self._chk_formation_lines)

        self._chk_ball_trail = QCheckBox("Ball Trail")
        self._chk_ball_trail.setChecked(True)
        self._chk_ball_trail.stateChanged.connect(self._sync_overlay_flags)
        status.addPermanentWidget(self._chk_ball_trail)

        self._lbl_fps = QLabel("FPS: --")
        status.addPermanentWidget(self._lbl_fps)

    def _sync_overlay_flags(self) -> None:
        """Push checkbox state into the canvas widget."""
        self.video_canvas.show_ellipses = self._chk_foot_markers.isChecked()
        self.video_canvas.show_badges = self._chk_foot_markers.isChecked()
        self.video_canvas.show_ball = self._chk_ball_trail.isChecked()
        self.radar_widget.show_formations = self._chk_formation_lines.isChecked()

    # ------------------------------------------------------------------
    # Slot: Load Video
    # ------------------------------------------------------------------

    def _on_load_video(self) -> None:
        """Open a file dialog, then start the pipeline on the chosen video."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov)",
        )
        if not path:
            return

        if PipelineWorker is None:
            QMessageBox.critical(
                self,
                "Missing dependency",
                "PipelineWorker could not be imported.  "
                "Ensure the pipeline package is installed.",
            )
            return

        # Stop any existing worker.
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(2000)

        self._worker = PipelineWorker(path)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.error.connect(self._on_pipeline_error)
        self._worker.finished.connect(self._on_pipeline_finished)

        # Wire the heatmap engine into the widget.
        if hasattr(self._worker, "heatmap_engine") and self._worker.heatmap_engine is not None:
            self.heatmap_widget.set_engine(self._worker.heatmap_engine)

        self._worker.start()
        self._paused = False
        self._fps_timestamps.clear()
        self.statusBar().showMessage(f"Loaded: {path}")

    # ------------------------------------------------------------------
    # Slot: Frame Ready
    # ------------------------------------------------------------------

    def _on_frame_ready(self, packet: FramePacket) -> None:
        """Handle an incoming processed frame packet."""
        # Update child widgets.
        self.video_canvas.update_frame(packet)
        self.radar_widget.update_data(packet)

        # Heatmap widget: feed players and refresh.
        if hasattr(packet, "players"):
            self.heatmap_widget.update_players(packet.players)
        self.heatmap_widget.update_heatmap()

        # Rolling FPS calculation.
        now = time.monotonic()
        self._fps_timestamps.append(now)
        if len(self._fps_timestamps) > self._fps_window:
            self._fps_timestamps = self._fps_timestamps[-self._fps_window:]
        if len(self._fps_timestamps) >= 2:
            elapsed = self._fps_timestamps[-1] - self._fps_timestamps[0]
            if elapsed > 0:
                fps = (len(self._fps_timestamps) - 1) / elapsed
                self._lbl_fps.setText(f"FPS: {fps:.1f}")

    # ------------------------------------------------------------------
    # Slot: Play / Pause
    # ------------------------------------------------------------------

    def _on_play_pause(self) -> None:
        """Toggle the pipeline worker's paused state."""
        if self._worker is None:
            return
        self._paused = not self._paused
        self._worker.paused = self._paused

    # ------------------------------------------------------------------
    # Slot: Export
    # ------------------------------------------------------------------

    def _on_export(self) -> None:
        """Placeholder for MP4 export functionality."""
        QMessageBox.information(
            self,
            "Export",
            "MP4 export is not yet implemented.",
        )

    # ------------------------------------------------------------------
    # Slot: Pipeline error / finished
    # ------------------------------------------------------------------

    def _on_pipeline_error(self, message: str) -> None:
        """Show an error dialog when the pipeline fails."""
        QMessageBox.critical(self, "Pipeline Error", message)

    def _on_pipeline_finished(self) -> None:
        """Handle pipeline completion."""
        self.statusBar().showMessage("Pipeline finished.")

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        """Ensure the pipeline worker is stopped before closing."""
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(2000)
        super().closeEvent(event)
