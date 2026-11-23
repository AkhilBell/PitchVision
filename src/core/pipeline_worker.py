"""Background QThread that drives the full PitchVision CV pipeline.

The worker iterates over every frame of a video, runs detection →
classification → pitch calibration → formation → heatmap, packages the
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

import cv2
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
from src.vision.pitch_keypoints import PitchKeypointDetector
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


def _frame_histogram(frame: np.ndarray) -> np.ndarray:
    """Normalised grayscale histogram used for hard-cut detection."""
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    cv2.normalize(hist, hist)
    return hist


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
        self._pitch_detector = PitchKeypointDetector(device=config.DEVICE)
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
        self._prev_hist: np.ndarray | None = None
        self._smoothed_players: dict[int, tuple[float, float]] = {}
        self._smoothed_ball: tuple[float, float] | None = None

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def detector_description(self) -> str:
        """One-line summary of the detection models in use."""
        return f"{self._detector.description} | {self._pitch_detector.description}"

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

                # ── 0. Hard-cut detection ────────────────────────────────
                if self._is_scene_cut(frame):
                    self._on_scene_cut()

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

                # ── 3. Pitch keypoints → homography ─────────────────────
                pitch_calibrated = self._calibrate_pitch(frame)
                if pitch_calibrated:
                    players = self._homography.transform_players(players)
                    if ball is not None:
                        ball = self._homography.transform_ball(ball)
                    self._smooth_positions(players, ball)
                else:
                    # Close-up / cut / missing model: leave pitch_xy empty
                    # so radar and heatmap freeze for this frame.
                    for player in players:
                        player.pitch_xy = None
                    if ball is not None:
                        ball.pitch_xy = None

                homography_matrix: Optional[np.ndarray] = (
                    self._homography.matrix if pitch_calibrated else None
                )

                # ── 4. Formation detection ──────────────────────────────
                pitch_state: PitchState = self._formation_engine.detect_both_teams(
                    players
                )

                # ── 5. Heatmap accumulation (calibrated frames only) ────
                if pitch_calibrated:
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
                    pitch_calibrated=pitch_calibrated,
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

    # ── Pitch calibration helpers ───────────────────────────────────────

    def _calibrate_pitch(self, frame: np.ndarray) -> bool:
        """Estimate a per-frame homography from pitch landmarks.

        Returns
        -------
        bool
            ``True`` when a usable transform was set for this frame.
        """
        self._homography.clear()
        pixel_xy, pitch_xy, _indices = self._pitch_detector.detect(frame)
        if pixel_xy.shape[0] < config.PITCH_MIN_KEYPOINTS:
            return False
        try:
            self._homography.set_transform(pixel_xy, pitch_xy)
        except ValueError:
            logger.debug(
                "Homography failed with %d keypoints", pixel_xy.shape[0], exc_info=True
            )
            self._homography.clear()
            return False
        return self._homography.is_calibrated

    def _is_scene_cut(self, frame: np.ndarray) -> bool:
        """Return True when *frame* looks like a hard cut from the previous one."""
        hist = _frame_histogram(frame)
        cut = False
        if self._prev_hist is not None:
            corr = float(cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL))
            cut = corr < config.SCENE_CUT_HIST_CORR
        self._prev_hist = hist
        return cut

    def _on_scene_cut(self) -> None:
        """Reset trackers and pitch state after a hard cut."""
        logger.info("Scene cut detected — resetting tracker and pitch calibration")
        try:
            self._detector.tracker.reset()
        except Exception:
            logger.debug("Tracker reset failed", exc_info=True)
        self._homography.clear()
        self._smoothed_players.clear()
        self._smoothed_ball = None
        # Force a fresh ball search; IDs and last-ball position do not
        # carry across a cut / replay / angle change.
        self._ball_tracker = BallTracker(max_jump_px=self._ball_tracker.max_jump_px)

    def _smooth_positions(
        self,
        players: list[PlayerState],
        ball: Optional[BallState],
    ) -> None:
        """EMA-smooth pitch positions to reduce radar jitter."""
        alpha = float(config.POSITION_SMOOTHING_ALPHA)
        alpha = min(max(alpha, 0.0), 1.0)
        live_ids: set[int] = set()

        for player in players:
            if player.pitch_xy is None:
                continue
            live_ids.add(player.track_id)
            x, y = player.pitch_xy
            prev = self._smoothed_players.get(player.track_id)
            if prev is None or alpha >= 1.0:
                smoothed = (x, y)
            else:
                smoothed = (
                    alpha * x + (1.0 - alpha) * prev[0],
                    alpha * y + (1.0 - alpha) * prev[1],
                )
            self._smoothed_players[player.track_id] = smoothed
            player.pitch_xy = smoothed

        # Drop tracks that left the frame so a recycled ID starts fresh.
        for track_id in list(self._smoothed_players):
            if track_id not in live_ids:
                del self._smoothed_players[track_id]

        if ball is not None and ball.pitch_xy is not None:
            x, y = ball.pitch_xy
            if self._smoothed_ball is None or alpha >= 1.0:
                smoothed_ball = (x, y)
            else:
                px, py = self._smoothed_ball
                smoothed_ball = (
                    alpha * x + (1.0 - alpha) * px,
                    alpha * y + (1.0 - alpha) * py,
                )
            self._smoothed_ball = smoothed_ball
            ball.pitch_xy = smoothed_ball
        else:
            self._smoothed_ball = None
