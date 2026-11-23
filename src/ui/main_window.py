"""Main application window for PitchVision Tactical Studio.

Assembles the video canvas, radar minimap, and live heatmap panel into a
single PySide6 ``QMainWindow`` with toolbar and status-bar controls.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.core.packet import FramePacket
from src.core.pipeline_worker import PipelineWorker
from src.ui.heatmap_widget import HeatmapWidget
from src.ui.radar_widget import RadarWidget
from src.ui.video_canvas import VideoCanvas

if TYPE_CHECKING:
    from PySide6.QtGui import QCloseEvent


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
        self._worker: PipelineWorker | None = None
        self._paused: bool = False
        self._frame_count: int = 0
        self._fps_window_start: float = time.monotonic()
        self._fps_window_frames: int = 0

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

        self._chk_pitch_lines = QCheckBox("Pitch Lines")
        self._chk_pitch_lines.setChecked(False)
        self._chk_pitch_lines.setToolTip(
            "Debug: project detected pitch markings back onto the video"
        )
        self._chk_pitch_lines.stateChanged.connect(self._sync_overlay_flags)
        status.addPermanentWidget(self._chk_pitch_lines)

        self._progress = QProgressBar()
        self._progress.setFixedWidth(180)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        status.addPermanentWidget(self._progress)

        self._lbl_fps = QLabel("FPS: --")
        status.addPermanentWidget(self._lbl_fps)

    def _sync_overlay_flags(self) -> None:
        """Push checkbox state into the canvas widget."""
        self.video_canvas.show_ellipses = self._chk_foot_markers.isChecked()
        self.video_canvas.show_badges = self._chk_foot_markers.isChecked()
        self.video_canvas.show_ball = self._chk_ball_trail.isChecked()
        self.video_canvas.show_pitch_lines = self._chk_pitch_lines.isChecked()
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
            "Video Files (*.mp4 *.avi *.mov *.mkv)",
        )
        if not path:
            return

        # Stop any existing worker.
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None

        try:
            self._worker = PipelineWorker(path)
        except Exception as exc:  # noqa: BLE001 — surface model/load failures
            QMessageBox.critical(
                self,
                "Failed to start pipeline",
                f"Could not initialise the vision pipeline:\n{exc}",
            )
            self._worker = None
            return

        self.heatmap_widget.set_engine(self._worker.heatmap_engine)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._worker.start()

        self._paused = False
        self._btn_play_pause.setChecked(False)
        self._frame_count = 0
        self._fps_window_start = time.monotonic()
        self._fps_window_frames = 0
        # Left side of the status bar: which models are analysing the video,
        # so a missing weights file (generic fallback) is visible.
        self.statusBar().showMessage(self._worker.detector_description)

    # ------------------------------------------------------------------
    # Slot: Frame Ready
    # ------------------------------------------------------------------

    def _on_frame_ready(self, packet: FramePacket) -> None:
        """Handle an incoming processed frame packet."""
        self._frame_count += 1
        self._fps_window_frames += 1

        # Update child widgets.
        self.video_canvas.update_frame(packet)
        self.radar_widget.update_data(packet)

        # Heatmap widget: feed players and refresh (engine only accumulates
        # on calibrated frames, so this redraw freezes during close-ups).
        self.heatmap_widget.update_players(packet.players)
        self.heatmap_widget.update_heatmap()

        # Rolling-window FPS estimate.
        elapsed = time.monotonic() - self._fps_window_start
        if elapsed >= 1.0:
            fps = self._fps_window_frames / elapsed
            self._lbl_fps.setText(f"FPS: {fps:.1f}")
            self._fps_window_start = time.monotonic()
            self._fps_window_frames = 0

        # Status: advertise missing pitch view during close-ups / cuts.
        if self._worker is not None:
            base = self._worker.detector_description
            if packet.pitch_calibrated:
                self.statusBar().showMessage(base)
            else:
                self.statusBar().showMessage(f"No pitch view | {base}")

    def _on_progress(self, frame_index: int, total_frames: int) -> None:
        """Advance the progress bar; *frame_index* is zero-based."""
        if total_frames > 0:
            self._progress.setRange(0, total_frames)
            self._progress.setValue(min(frame_index + 1, total_frames))
        else:
            self._progress.setRange(0, 0)  # unknown length: busy indicator

    def _on_worker_error(self, message: str) -> None:
        """Show pipeline errors in a dialog."""
        QMessageBox.critical(self, "Pipeline error", message)

    def _on_worker_finished(self) -> None:
        """Reset pause UI when the worker completes."""
        self._paused = False
        self._btn_play_pause.setChecked(False)
        if self._progress.maximum() == 0:
            self._progress.setRange(0, 1)  # stop the busy indicator
            self._progress.setValue(1)
        # The live heatmap is throttled, so draw the final state once.
        self.heatmap_widget.update_heatmap(force=True)
        if self._worker is not None:
            self.statusBar().showMessage(
                f"Playback finished | {self._worker.detector_description}"
            )

    # ------------------------------------------------------------------
    # Slot: Play / Pause
    # ------------------------------------------------------------------

    def _on_play_pause(self) -> None:
        """Toggle the pipeline worker's paused state."""
        if self._worker is None:
            self._btn_play_pause.setChecked(False)
            return
        self._paused = self._btn_play_pause.isChecked()
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
    # Overrides
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        """Ensure the pipeline worker is stopped before closing."""
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(3000)
        super().closeEvent(event)
