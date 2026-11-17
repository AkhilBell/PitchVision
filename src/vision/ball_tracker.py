"""Ball detection tracking and linear interpolation.

Maintains a rolling history of observed ball positions and fills short
detection gaps with linearly-interpolated estimates so that downstream
analytics receive a continuous ball trajectory.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from src.core.packet import BallState


class BallTracker:
    """Tracks a single ball across frames with gap interpolation.

    When the detector fails to find the ball for a short run of frames
    (up to *max_missing_frames*), the tracker interpolates between the
    last two known positions.  Beyond that threshold the ball is reported
    as lost (``center_px=None``).

    While the ball is being followed, detections that are implausibly
    far from its last position (another ball by the touchline, a white
    boot, a penalty spot) are ignored.  Once the ball has been lost it
    may be picked up again anywhere in the frame.

    Parameters
    ----------
    max_missing_frames : int
        Maximum number of consecutive frames without a detection before
        the ball is considered lost.
    max_jump_px : float | None
        Largest plausible ball movement per frame, in pixels.  *None*
        disables the check.
    """

    _HISTORY_LENGTH: int = 2  # only the last two detections are ever used

    def __init__(
        self,
        max_missing_frames: int = 10,
        max_jump_px: float | None = None,
    ) -> None:
        self.max_missing_frames: int = max_missing_frames
        self.max_jump_px: float | None = max_jump_px
        self._history: list[tuple[int, tuple[int, int]]] = []
        self._missing_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_tracking(self) -> bool:
        """Whether the ball has been detected within the last few frames."""
        return bool(self._history) and self._missing_count <= self.max_missing_frames

    @property
    def search_center(self) -> tuple[int, int] | None:
        """Where to look for the ball next: its last detected position.

        *None* when the ball is lost and the whole frame must be searched.
        """
        return self._history[-1][1] if self.is_tracking else None

    def update(
        self, frame_index: int, ball_detections: sv.Detections
    ) -> BallState:
        """Ingest detections for the current frame and return a ball state.

        Parameters
        ----------
        frame_index : int
            Zero-based index of the current frame.
        ball_detections : sv.Detections
            Ball detections (may be empty) produced by the detector for
            this frame.

        Returns
        -------
        BallState
            Updated ball state, potentially with an interpolated position.
        """
        center = self._select(frame_index, ball_detections)
        if center is not None:
            self._history.append((frame_index, center))
            del self._history[: -self._HISTORY_LENGTH]
            self._missing_count = 0

            return BallState(
                center_px=center,
                pitch_xy=None,
                is_interpolated=False,
            )

        # No usable detection this frame.
        self._missing_count += 1

        if self._missing_count <= self.max_missing_frames:
            interpolated = self._interpolate(frame_index)
            if interpolated is not None:
                return BallState(
                    center_px=interpolated,
                    pitch_xy=None,
                    is_interpolated=True,
                )

        # Ball is lost.
        return BallState(
            center_px=None,
            pitch_xy=None,
            is_interpolated=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select(
        self, frame_index: int, ball_detections: sv.Detections
    ) -> tuple[int, int] | None:
        """Pick the most confident plausible detection for this frame.

        Returns
        -------
        tuple[int, int] | None
            ``(cx, cy)`` of the chosen detection, or *None* if there is no
            detection or every one is too far from the ball's last position.
        """
        if len(ball_detections) == 0:
            return None

        boxes: np.ndarray = np.asarray(ball_detections.xyxy, dtype=float)
        centers: np.ndarray = np.column_stack(
            ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2)
        )
        confidences = ball_detections.confidence
        if confidences is None or len(confidences) != len(centers):
            confidences = np.ones(len(centers))
        confidences = np.asarray(confidences, dtype=float)

        if self.max_jump_px is not None and self.is_tracking:
            last_frame, (last_x, last_y) = self._history[-1]
            allowed: float = self.max_jump_px * max(1, frame_index - last_frame)
            distances = np.hypot(centers[:, 0] - last_x, centers[:, 1] - last_y)
            plausible = distances <= allowed
            if not plausible.any():
                return None
            centers, confidences = centers[plausible], confidences[plausible]

        best = int(np.argmax(confidences))
        return int(centers[best, 0]), int(centers[best, 1])

    def _interpolate(
        self, frame_index: int
    ) -> tuple[int, int] | None:
        """Linearly interpolate the ball position from the last two knowns.

        Parameters
        ----------
        frame_index : int
            The current frame index for which to predict.

        Returns
        -------
        tuple[int, int] | None
            Estimated ``(cx, cy)`` pixel coordinate or *None* if there
            are fewer than two history points.
        """
        if len(self._history) < 2:
            return None

        try:
            f1, (x1, y1) = self._history[-2]
            f2, (x2, y2) = self._history[-1]

            dt: int = f2 - f1
            if dt == 0:
                return (x2, y2)

            frames_ahead: int = frame_index - f2
            ratio: float = frames_ahead / dt

            cx: int = int(x2 + (x2 - x1) * ratio)
            cy: int = int(y2 + (y2 - y1) * ratio)
            return (cx, cy)
        except Exception:
            return None
