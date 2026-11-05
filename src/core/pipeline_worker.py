"""Background QThread that drives the full PitchVision CV pipeline.

The worker iterates over every frame of a video, runs detection →
classification → homography → formation → heatmap, packages the
results into a :class:`~src.core.packet.FramePacket`, and emits it
via the ``frame_ready`` signal so the UI can update in real time.

Signals
-------
frame_ready(FramePacket)
    Emitted after each frame has been fully processed.
progress(int, int)
    ``(current_frame, total_frames)`` for progress-bar updates.
finished()
    Emitted when the video has been fully processed or the worker is stopped.
error(str)
    Emitted if an unrecoverable error occurs during processing.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QThread, Signal

from src.core.packet import (
    BallState,
    FramePacket,
    PitchState,
    PlayerState,
)
from src.core.video_io import frame_generator, get_video_properties

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for pipeline stages — gracefully degrade if modules are not
# yet implemented so that this file remains importable during early dev.
# ---------------------------------------------------------------------------
try:
    from src.vision.detector import Detector  # type: ignore[import-untyped]
except ImportError:
    Detector = None  # type: ignore[assignment,misc]

try:
    from src.vision.team_classifier import TeamClassifier  # type: ignore[import-untyped]
except ImportError:
    TeamClassifier = None  # type: ignore[assignment,misc]

try:
    from src.vision.homography import HomographyEstimator  # type: ignore[import-untyped]
except ImportError:
    HomographyEstimator = None  # type: ignore[assignment,misc]

try:
    from src.analytics.formation_engine import FormationEngine  # type: ignore[import-untyped]
except ImportError:
    FormationEngine = None  # type: ignore[assignment,misc]

try:
    from src.analytics.heatmap_engine import HeatmapEngine  # type: ignore[import-untyped]
except ImportError:
    HeatmapEngine = None  # type: ignore[assignment,misc]


class PipelineWorker(QThread):
    """Process a video file through the full PitchVision pipeline.

    Parameters
    ----------
    video_path : str
        Absolute or relative path to the input video file.
    parent : QObject | None
        Optional Qt parent.
    """

    # ── Qt signals ──────────────────────────────────────────────────────
    frame_ready = Signal(object)
    """Emits a :class:`FramePacket` for every processed frame."""

    progress = Signal(int, int)
    """Emits ``(current_frame_index, total_frames)``."""

    finished = Signal()
    """Emitted when all frames have been processed or the worker is stopped."""

    error = Signal(str)
    """Emits an error description string on failure."""

    def __init__(self, video_path: str, parent=None) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self._running: bool = True

        # ── Instantiate pipeline stages (skip if not available) ─────────
        self._detector = Detector() if Detector is not None else None
        self._team_classifier = TeamClassifier() if TeamClassifier is not None else None
        self._homography = HomographyEstimator() if HomographyEstimator is not None else None
        self._formation_engine = FormationEngine() if FormationEngine is not None else None
        self._heatmap_engine = HeatmapEngine() if HeatmapEngine is not None else None

    # ── Public API ──────────────────────────────────────────────────────

    def stop(self) -> None:
        """Request the worker to stop after the current frame."""
        self._running = False

    # ── QThread entry point ─────────────────────────────────────────────

    def run(self) -> None:  # noqa: C901 — sequential pipeline steps
        """Execute the pipeline over every frame of the video."""
        try:
            props = get_video_properties(self.video_path)
            total_frames: int = props["total_frames"]
            logger.info(
                "PipelineWorker started — %s (%d frames, %.1f fps)",
                self.video_path,
                total_frames,
                props["fps"],
            )

            for frame_index, timestamp_ms, frame in frame_generator(self.video_path):
                if not self._running:
                    logger.info("PipelineWorker stopped at frame %d", frame_index)
                    break

                # ── 1. Object detection ─────────────────────────────────
                players: list[PlayerState] = []
                ball: Optional[BallState] = None

                if self._detector is not None:
                    # TODO: Call self._detector.detect(frame) and unpack
                    #       player bounding boxes + ball position from the
                    #       raw detections.
                    pass
                else:
                    # Placeholder: no detections until detector is wired in
                    pass

                # ── 2. Team classification ──────────────────────────────
                if self._team_classifier is not None and players:
                    # TODO: Call self._team_classifier.classify(frame, players)
                    #       to assign team labels ("A", "B", "ref") to each
                    #       PlayerState based on jersey colour.
                    pass

                # ── 3. Homography → pitch coordinates ───────────────────
                homography_matrix = None
                if self._homography is not None:
                    # TODO: Compute or update the homography matrix and
                    #       project each player's foot_px → pitch_xy.
                    #       homography_matrix = self._homography.estimate(frame)
                    #       for p in players:
                    #           p.pitch_xy = self._homography.transform_point(p.foot_px)
                    #       if ball and ball.center_px:
                    #           ball.pitch_xy = self._homography.transform_point(ball.center_px)
                    pass

                # ── 4. Formation detection ──────────────────────────────
                pitch_state: Optional[PitchState] = None
                if self._formation_engine is not None:
                    # TODO: pitch_state = self._formation_engine.detect_formations(players)
                    pass

                # ── 5. Heatmap accumulation ─────────────────────────────
                if self._heatmap_engine is not None:
                    # TODO: self._heatmap_engine.update(players)
                    pass

                # ── 6. Package & emit ───────────────────────────────────
                packet = FramePacket(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    raw_frame=frame,
                    players=players,
                    ball=ball,
                    pitch_state=pitch_state,
                    homography_matrix=homography_matrix,
                )
                self.frame_ready.emit(packet)
                self.progress.emit(frame_index, total_frames)

        except FileNotFoundError as exc:
            self.error.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 — catch-all for the thread
            logger.exception("PipelineWorker encountered an error")
            self.error.emit(f"Pipeline error: {exc}")
        finally:
            self.finished.emit()
            logger.info("PipelineWorker finished")
