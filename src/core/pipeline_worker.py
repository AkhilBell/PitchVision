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
    from src.vision.detector import PlayerDetector  # type: ignore[import-untyped]
except ImportError:
    PlayerDetector = None  # type: ignore[assignment,misc]

try:
    from src.vision.ball_tracker import BallTracker  # type: ignore[import-untyped]
except ImportError:
    BallTracker = None  # type: ignore[assignment,misc]

try:
    from src.vision.team_classifier import TeamClassifier  # type: ignore[import-untyped]
except ImportError:
    TeamClassifier = None  # type: ignore[assignment,misc]

try:
    from src.vision.homography import HomographyTransformer  # type: ignore[import-untyped]
except ImportError:
    HomographyTransformer = None  # type: ignore[assignment,misc]

try:
    from src.analytics.formation_engine import FormationEngine  # type: ignore[import-untyped]
except ImportError:
    FormationEngine = None  # type: ignore[assignment,misc]

try:
    from src.analytics.heatmap_engine import HeatmapEngine  # type: ignore[import-untyped]
except ImportError:
    HeatmapEngine = None  # type: ignore[assignment,misc]

# Number of initial frames used to fit the team colour classifier.
_CLASSIFIER_FIT_FRAME = 30
# Re-fit the classifier every N frames to adapt to lighting changes.
_CLASSIFIER_REFIT_INTERVAL = 300


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
        self.paused: bool = False

        # ── Instantiate pipeline stages (skip if not available) ─────────
        self._detector = PlayerDetector() if PlayerDetector is not None else None
        self._ball_tracker = BallTracker() if BallTracker is not None else None
        self._team_classifier = TeamClassifier() if TeamClassifier is not None else None
        self._homography = HomographyTransformer() if HomographyTransformer is not None else None
        self._formation_engine = FormationEngine() if FormationEngine is not None else None
        self._heatmap_engine = HeatmapEngine() if HeatmapEngine is not None else None

        # Expose the heatmap engine so the UI can read from it.
        self.heatmap_engine = self._heatmap_engine

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
            frame_w: int = props["width"]
            frame_h: int = props["height"]
            logger.info(
                "PipelineWorker started — %s (%d frames, %.1f fps)",
                self.video_path,
                total_frames,
                props["fps"],
            )

            # Set a default homography from frame corners if no
            # calibration points are available.
            if self._homography is not None and not self._homography.is_calibrated:
                self._homography.set_default_transform(frame_w, frame_h)
                logger.info(
                    "Default homography set from %dx%d frame corners",
                    frame_w,
                    frame_h,
                )

            for frame_index, timestamp_ms, frame in frame_generator(self.video_path):
                if not self._running:
                    logger.info("PipelineWorker stopped at frame %d", frame_index)
                    break

                # Respect pause flag — spin-wait without burning CPU.
                while self.paused and self._running:
                    self.msleep(50)
                if not self._running:
                    break

                # ── 1. Object detection ─────────────────────────────────
                players: list[PlayerState] = []
                ball: Optional[BallState] = None

                if self._detector is not None:
                    player_dets, ball_dets = self._detector.detect(frame)

                    # Convert supervision Detections → PlayerState list.
                    for i in range(len(player_dets)):
                        bbox = player_dets.xyxy[i]
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        fx = cx
                        fy = y2  # bottom-centre = foot position

                        # ByteTrack assigns tracker_id; fall back to index.
                        tid = -1
                        if player_dets.tracker_id is not None:
                            tid = int(player_dets.tracker_id[i])
                        else:
                            tid = i

                        players.append(
                            PlayerState(
                                track_id=tid,
                                bbox=(x1, y1, x2, y2),
                                center_px=(cx, cy),
                                foot_px=(fx, fy),
                                pitch_xy=None,
                                team="unknown",
                            )
                        )

                    # Ball tracking with interpolation.
                    if self._ball_tracker is not None:
                        ball = self._ball_tracker.update(frame_index, ball_dets)
                    elif len(ball_dets) > 0:
                        bb = ball_dets.xyxy[0]
                        bcx = int((bb[0] + bb[2]) / 2)
                        bcy = int((bb[1] + bb[3]) / 2)
                        ball = BallState(center_px=(bcx, bcy))

                # ── 2. Team classification ──────────────────────────────
                if self._team_classifier is not None and players:
                    # Fit the classifier on an early frame (or periodically
                    # re-fit) so it adapts to the specific match jerseys.
                    if (
                        not self._team_classifier.is_fitted
                        and frame_index >= _CLASSIFIER_FIT_FRAME
                    ) or (
                        self._team_classifier.is_fitted
                        and frame_index % _CLASSIFIER_REFIT_INTERVAL == 0
                    ):
                        # Build a supervision-compatible Detections object
                        # from the current player list for fitting.
                        import numpy as _np
                        import supervision as _sv

                        fit_bboxes = _np.array(
                            [p.bbox for p in players], dtype=_np.float32
                        )
                        fit_dets = _sv.Detections(xyxy=fit_bboxes)
                        try:
                            self._team_classifier.fit(frame, fit_dets)
                            logger.info(
                                "TeamClassifier fitted at frame %d with %d players",
                                frame_index,
                                len(players),
                            )
                        except Exception as fit_exc:
                            logger.warning("TeamClassifier fit failed: %s", fit_exc)

                    # Classify each player.
                    for p in players:
                        p.team = self._team_classifier.classify(frame, p.bbox)

                # ── 3. Homography → pitch coordinates ───────────────────
                homography_matrix = None
                if self._homography is not None and self._homography.is_calibrated:
                    homography_matrix = self._homography._matrix
                    self._homography.transform_players(players)
                    if ball is not None:
                        self._homography.transform_ball(ball)

                # ── 4. Formation detection ──────────────────────────────
                pitch_state: Optional[PitchState] = None
                if self._formation_engine is not None and len(players) >= 6:
                    try:
                        pitch_state = self._formation_engine.detect_both_teams(players)
                    except Exception as form_exc:
                        logger.debug("Formation detection failed: %s", form_exc)

                # ── 5. Heatmap accumulation ─────────────────────────────
                if self._heatmap_engine is not None:
                    self._heatmap_engine.update(players)

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
