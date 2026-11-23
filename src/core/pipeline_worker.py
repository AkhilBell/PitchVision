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
from collections import Counter
from dataclasses import replace
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

_VALID_ROLES = frozenset({"player", "goalkeeper", "referee"})


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
        self._role_votes: dict[int, Counter[str]] = {}
        self._role_settled: dict[int, str] = {}
        self._last_seen: dict[int, tuple[PlayerState, int]] = {}
        self._form_accepted: dict[str, str] = {"A": "—", "B": "—"}
        self._form_candidate: dict[str, str | None] = {"A": None, "B": None}
        self._cached_players: list[PlayerState] | None = None
        self._cached_ball: Optional[BallState] = None
        self._cached_calibrated: bool = False
        self._cached_matrix: Optional[np.ndarray] = None
        self._cached_pitch: Optional[PitchState] = None

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

            source_fps = float(props["fps"] or config.PLAYBACK_FPS)
            # Pace the picture at the file's frame rate, up to 30. Player
            # tracking runs most frames; pitch and ball run less often.
            target_fps = min(float(config.PLAYBACK_FPS), float(config.MAX_FPS), source_fps)
            frame_interval = 1.0 / target_fps
            next_emit_at = 0.0
            player_stride = max(1, int(config.PLAYER_STRIDE))
            pitch_stride = max(player_stride, int(config.PITCH_STRIDE))
            ball_stride = max(player_stride, int(config.BALL_STRIDE))
            hold_frames = max(1, int(config.OFFSCREEN_HOLD_S * source_fps))
            self._heatmap_engine.set_clip_frames(total_frames)
            # First inference on the GPU is several times slower than the
            # rest (shaders compile). Do it before the first frame is shown.
            self._warmup_models()

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

                run_players = (
                    self._cached_players is None or frame_index % player_stride == 0
                )
                run_pitch = run_players and (
                    frame_index % pitch_stride == 0 or not self._cached_calibrated
                )
                run_ball = run_players and frame_index % ball_stride == 0
                if run_players or run_pitch:
                    players, ball, pitch_calibrated = self._analyse_frame(
                        frame,
                        frame_index,
                        run_players=run_players,
                        run_pitch=run_pitch,
                        run_ball=run_ball,
                    )
                    self._cached_players = players
                    self._cached_ball = ball
                    self._cached_calibrated = pitch_calibrated
                    self._cached_matrix = (
                        self._homography.matrix if pitch_calibrated else None
                    )
                else:
                    players = self._cached_players or []
                    ball = self._cached_ball
                    pitch_calibrated = self._cached_calibrated

                # Remember players who just left the frame so the formation
                # can still count them, then drop anyone past 11 per team.
                roster = self._roster_with_offscreen(players, frame_index, hold_frames)
                roster = self._cap_roster(roster)
                # Formation clustering is much more expensive than drawing,
                # so it runs only on frames that tracked players again.
                if run_players or self._cached_pitch is None:
                    pitch_state = self._smoothed_formation(roster, fresh=True)
                    self._cached_pitch = pitch_state
                else:
                    pitch_state = self._cached_pitch

                # One count per video frame, on-screen players only, and
                # only while the camera is calibrated. Dividing by the
                # full clip length later makes this a time share.
                if pitch_calibrated:
                    self._heatmap_engine.update(
                        [p for p in players if not getattr(p, "offscreen", False)]
                    )

                packet = FramePacket(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    raw_frame=frame,
                    players=roster,
                    ball=ball,
                    pitch_state=pitch_state,
                    homography_matrix=self._cached_matrix if pitch_calibrated else None,
                    pitch_calibrated=pitch_calibrated,
                )
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
        self._role_votes.clear()
        self._role_settled.clear()
        self._team_classifier.clear_tracks()
        self._last_seen.clear()
        self._form_accepted = {"A": "—", "B": "—"}
        self._form_candidate = {"A": None, "B": None}
        self._cached_players = None
        self._cached_ball = None
        self._cached_calibrated = False
        self._cached_matrix = None
        self._cached_pitch = None
        # Force a fresh ball search; IDs and last-ball position do not
        # carry across a cut / replay / angle change.
        self._ball_tracker = BallTracker(max_jump_px=self._ball_tracker.max_jump_px)

    def _analyse_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        run_players: bool,
        run_pitch: bool,
        run_ball: bool,
    ) -> tuple[list[PlayerState], Optional[BallState], bool]:
        """Detect players, and optionally the ball and the pitch.

        Player tracking is what keeps identities stable, so it runs most
        often. The ball search and the pitch model are skipped on the
        frames in between and the previous result is reused.
        """
        if self._ball_tracker.max_jump_px is None:
            frame_h, frame_w = frame.shape[:2]
            self._ball_tracker.max_jump_px = (
                config.BALL_MAX_JUMP_FRACTION * float(np.hypot(frame_w, frame_h))
            )

        if run_players:
            player_dets, ball_dets = self._detector.detect(
                frame,
                ball_hint=self._ball_tracker.search_center,
                run_ball=run_ball,
            )
            roles = self._detector.roles(player_dets)
            players = _detections_to_players(player_dets, roles)
            if run_ball:
                ball = self._ball_tracker.update(frame_index, ball_dets)
            else:
                ball = self._cached_ball
        else:
            players = [replace(player, offscreen=False) for player in (self._cached_players or [])]
            ball = self._cached_ball

        if run_players and players:
            for player in players:
                player.role = self._observe_role(player.track_id, player.role)

            if not self._team_classifier.is_fitted:
                self._team_classifier.fit(frame, player_dets, roles)

            for player in players:
                if player.role == "referee":
                    player.team = "ref"
                elif player.role == "goalkeeper":
                    player.team = "unknown"
                else:
                    player.team = self._team_classifier.observe(
                        player.track_id, frame, player.bbox
                    )

        if run_pitch:
            pitch_calibrated = self._calibrate_pitch(frame)
        else:
            pitch_calibrated = self._homography.is_calibrated

        if pitch_calibrated:
            players = self._homography.transform_players(players)
            if ball is not None and run_ball:
                ball = self._homography.transform_ball(ball)
            elif ball is not None and ball.pitch_xy is None:
                ball = self._homography.transform_ball(ball)
            if run_players:
                self._smooth_positions(players, ball if run_ball else None)
        else:
            for player in players:
                player.pitch_xy = None
            if ball is not None:
                ball.pitch_xy = None

        if run_players and players:
            self._assign_goalkeeper_teams(players)
        return players, ball, pitch_calibrated

    def _roster_with_offscreen(
        self,
        players: list[PlayerState],
        frame_index: int,
        hold_frames: int,
    ) -> list[PlayerState]:
        """Visible players plus recently seen players who have left the frame."""
        visible_ids = {player.track_id for player in players}
        for player in players:
            player.offscreen = False
            if player.pitch_xy is None:
                continue
            self._last_seen[player.track_id] = (replace(player), frame_index)

        roster = list(players)
        for track_id, (snap, seen_at) in list(self._last_seen.items()):
            if track_id in visible_ids:
                continue
            if frame_index - seen_at > hold_frames:
                del self._last_seen[track_id]
                continue
            roster.append(replace(snap, offscreen=True))
        return roster

    def _cap_roster(self, roster: list[PlayerState]) -> list[PlayerState]:
        """Keep at most 11 icons per team and 22 players in total.

        People on screen beat remembered ones, and a lower track id beats
        a newly minted one. That stops a camera pan from leaving the old
        dot on the map and adding a second dot for the same player.
        Referees are not part of the 22.
        """

        def rank(player: PlayerState) -> tuple[bool, int]:
            return (bool(player.offscreen), player.track_id)

        refs = [
            player
            for player in roster
            if player.role == "referee" or player.team == "ref"
        ]
        ref_ids = {id(player) for player in refs}
        grouped: dict[str, list[PlayerState]] = {"A": [], "B": [], "other": []}
        for player in roster:
            if id(player) in ref_ids:
                continue
            key = player.team if player.team in ("A", "B") else "other"
            grouped[key].append(player)

        kept: list[PlayerState] = []
        for team in ("A", "B"):
            kept.extend(sorted(grouped[team], key=rank)[: config.MAX_PLAYERS_PER_TEAM])
        room = config.MAX_PLAYER_ICONS - len(kept)
        kept.extend(sorted(grouped["other"], key=rank)[: max(0, room)])
        if len(kept) > config.MAX_PLAYER_ICONS:
            kept = sorted(kept, key=rank)[: config.MAX_PLAYER_ICONS]
        return kept + refs

    def _smoothed_formation(
        self, players: list[PlayerState], fresh: bool
    ) -> PitchState:
        """Detect formations, and only change the label when it repeats.

        A single noisy frame would otherwise flip 4-4-2 to 4-3-3 and back.
        The label updates on analysis frames only, and only after the same
        shape has been seen on two analyses in a row.
        """
        state = self._formation_engine.detect_both_teams(players)
        if fresh:
            state.formation_a = self._latch_formation("A", state.formation_a)
            state.formation_b = self._latch_formation("B", state.formation_b)
        else:
            state.formation_a = self._form_accepted["A"]
            state.formation_b = self._form_accepted["B"]
        if state.formation_a == "—":
            state.formation_lines_a = []
        if state.formation_b == "—":
            state.formation_lines_b = []
        return state

    def _latch_formation(self, team: str, new_label: str) -> str:
        """Accept *new_label* once it agrees with the previous analysis."""
        accepted = self._form_accepted[team]
        if new_label == accepted:
            self._form_candidate[team] = None
            return accepted
        if new_label == self._form_candidate[team]:
            self._form_accepted[team] = new_label
            self._form_candidate[team] = None
            return new_label
        self._form_candidate[team] = new_label
        return accepted

    def _warmup_models(self) -> None:
        """Run one throwaway inference so the first real frame is not stalled."""
        blank = np.zeros((640, 360, 3), dtype=np.uint8)
        try:
            self._detector.detect(blank)
            self._pitch_detector.detect(blank)
        except Exception:
            logger.debug("Model warmup failed", exc_info=True)
        # Warmup detections must not leak into the match.
        try:
            self._detector.tracker.reset()
        except Exception:
            logger.debug("Tracker reset after warmup failed", exc_info=True)
        self._ball_tracker = BallTracker()

    def _observe_role(self, track_id: int, observed: str) -> str:
        """Majority-vote the role for *track_id* and lock it when stable."""
        settled = self._role_settled.get(track_id)
        if settled is not None:
            return settled

        role = observed if observed in _VALID_ROLES else "player"
        votes = self._role_votes.setdefault(track_id, Counter())
        votes[role] += 1
        best = votes.most_common(1)[0][0]
        if sum(votes.values()) >= config.ROLE_VOTE_SETTLE:
            self._role_settled[track_id] = best
        return best

    @staticmethod
    def _assign_goalkeeper_teams(players: list[PlayerState]) -> None:
        """Set each goalkeeper's team to the nearest outfield team's side."""
        outfield = [
            p
            for p in players
            if p.role == "player" and p.team in ("A", "B")
        ]
        goalkeepers = [p for p in players if p.role == "goalkeeper"]
        if not outfield or not goalkeepers:
            return

        def _xy(player: PlayerState) -> tuple[float, float] | None:
            if player.pitch_xy is not None:
                return float(player.pitch_xy[0]), float(player.pitch_xy[1])
            if player.foot_px is not None:
                return float(player.foot_px[0]), float(player.foot_px[1])
            return None

        centroids: dict[str, np.ndarray] = {}
        for team in ("A", "B"):
            pts = [_xy(p) for p in outfield if p.team == team]
            pts = [p for p in pts if p is not None]
            if pts:
                centroids[team] = np.mean(np.asarray(pts, dtype=np.float64), axis=0)

        if not centroids:
            return

        for gk in goalkeepers:
            gk_xy = _xy(gk)
            if gk_xy is None:
                continue
            best_team = min(
                centroids,
                key=lambda team: float(np.linalg.norm(centroids[team] - gk_xy)),
            )
            gk.team = best_team

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
