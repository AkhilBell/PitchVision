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
import time
from typing import Optional

import numpy as np
import supervision as sv
from PySide6.QtCore import QThread, Signal

from src import config
from src.analytics.formation_engine import FormationEngine
from src.analytics.heatmap_engine import HeatmapEngine
from src.core.packet import (
    BallState,
    FramePacket,
    PitchState,
    PlayerState,
)
from src.core.video_io import frame_generator, get_video_properties
from src.vision.ball_tracker import BallTracker
from src.vision.detector import PlayerDetector
from src.vision.homography import HomographyTransformer
from src.vision.team_classifier import TeamClassifier

logger = logging.getLogger(__name__)


def _detections_to_players(
    detections: sv.Detections, roles: list[str] | None = None
) -> list[PlayerState]:
    """Convert supervision detections into :class:`PlayerState` objects.

    *roles* gives the role of each detection, in the same order; when
    omitted everyone is a ``"player"``.
    """
    players: list[PlayerState] = []
    if len(detections) == 0:
        return players

    for i in range(len(detections)):
        x1, y1, x2, y2 = (int(v) for v in detections.xyxy[i])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        foot = (cx, y2)

        if detections.tracker_id is not None:
            track_id = int(detections.tracker_id[i])
            # ByteTrackTracker marks unmatched detections with tracker_id == -1.
            if track_id < 0:
                continue
        else:
            track_id = i

        players.append(
            PlayerState(
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                center_px=(cx, cy),
                foot_px=foot,
                role=roles[i] if roles is not None else "player",
            )
        )
    return players


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

    error = Signal(str)
    """Emits an error description string on failure."""

    def __init__(self, video_path: str, parent=None) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self._running: bool = True
        self._paused: bool = False

        self._detector = PlayerDetector(
            model_path=config.MODEL_PATH,
            confidence=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            device=config.DEVICE,
        )
        self._ball_tracker = BallTracker()
        self._team_classifier = TeamClassifier()
        self._homography = HomographyTransformer(
            pitch_length=config.PITCH_LENGTH,
            pitch_width=config.PITCH_WIDTH,
        )
        self._formation_engine = FormationEngine()
        self._heatmap_engine = HeatmapEngine(
            grid_shape=config.HEATMAP_GRID_RESOLUTION,
            sigma=config.HEATMAP_SIGMA,
        )

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def detector_description(self) -> str:
        """One-line summary of the detection models in use."""
        return self._detector.description

    @property
    def heatmap_engine(self) -> HeatmapEngine:
        """Shared heatmap accumulator used by the UI heatmap panel."""
        return self._heatmap_engine

    @property
    def paused(self) -> bool:
        """Whether the worker is currently paused between frames."""
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        self._paused = bool(value)

    def stop(self) -> None:
        """Request the worker to stop after the current frame."""
        self._paused = False
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

            # Playback pacing: never show frames faster than the source
            # frame rate (capped at MAX_FPS), but add no delay when the
            # pipeline is already slower than that.
            target_fps = min(float(config.MAX_FPS), props["fps"] or float(config.MAX_FPS))
            frame_interval = 1.0 / target_fps
            next_emit_at = 0.0  # earliest time.monotonic() the next frame may be shown

            for frame_index, timestamp_ms, frame in frame_generator(self.video_path):
                if not self._running:
                    logger.info("PipelineWorker stopped at frame %d", frame_index)
                    break

                while self._paused and self._running:
                    self.msleep(50)

                if not self._running:
                    break

                # ── 1. Object detection + ball tracking ─────────────────
                if self._ball_tracker.max_jump_px is None:
                    frame_h, frame_w = frame.shape[:2]
                    self._ball_tracker.max_jump_px = (
                        config.BALL_MAX_JUMP_FRACTION * float(np.hypot(frame_w, frame_h))
                    )

                player_dets, ball_dets = self._detector.detect(
                    frame, ball_hint=self._ball_tracker.search_center
                )
                roles = self._detector.roles(player_dets)
                players = _detections_to_players(player_dets, roles)
                ball: Optional[BallState] = self._ball_tracker.update(
                    frame_index, ball_dets
                )

                # ── 2. Team classification ──────────────────────────────
                # Roles reported by the detector take precedence over the
                # jersey-colour guess.
                if players:
                    if not self._team_classifier.is_fitted:
                        self._team_classifier.fit(frame, player_dets)

                    known_roles = self._detector.provides_roles
                    for player in players:
                        if player.role == "referee":
                            player.team = "ref"
                        elif player.role == "goalkeeper":
                            # A goalkeeper's kit matches neither team, so
                            # colour cannot say which side they are on.
                            player.team = "unknown"
                        else:
                            team = self._team_classifier.classify(frame, player.bbox)
                            if team == "ref" and known_roles:
                                # The detector says this is a player, so the
                                # referee colour match is wrong.
                                team = "unknown"
                            player.team = team

                # ── 3. Homography → pitch coordinates ───────────────────
                if not self._homography.is_calibrated:
                    h, w = frame.shape[:2]
                    self._homography.set_default_transform(w, h)

                players = self._homography.transform_players(players)
                if ball is not None:
                    ball = self._homography.transform_ball(ball)

                homography_matrix: Optional[np.ndarray] = self._homography.matrix

                # ── 4. Formation detection ──────────────────────────────
                pitch_state: PitchState = self._formation_engine.detect_both_teams(
                    players
                )

                # ── 5. Heatmap accumulation ─────────────────────────────
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
                # Hold the frame back only if the previous one was shown
                # less than one frame interval ago.
                delay = next_emit_at - time.monotonic()
                if delay > 0:
                    self.msleep(int(delay * 1000))
                self.frame_ready.emit(packet)
                self.progress.emit(frame_index, total_frames)
                next_emit_at = time.monotonic() + frame_interval

        except FileNotFoundError as exc:
            self.error.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 — catch-all for the thread
            logger.exception("PipelineWorker encountered an error")
            self.error.emit(f"Pipeline error: {exc}")
        finally:
            # QThread emits its built-in ``finished`` signal when run() returns.
            logger.info("PipelineWorker finished")
