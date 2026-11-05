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

    Parameters
    ----------
    max_missing_frames : int
        Maximum number of consecutive frames without a detection before
        the ball is considered lost.
    """

    def __init__(self, max_missing_frames: int = 10) -> None:
        self.max_missing_frames: int = max_missing_frames
        self._history: list[tuple[int, tuple[int, int]]] = []
        self._missing_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        if len(ball_detections) > 0:
            # Pick the highest-confidence detection.
            try:
                confidences: np.ndarray = ball_detections.confidence
                if confidences is not None and len(confidences) > 0:
                    best_idx: int = int(np.argmax(confidences))
                else:
                    best_idx = 0
            except Exception:
                best_idx = 0

            bbox: np.ndarray = ball_detections.xyxy[best_idx]  # [x1, y1, x2, y2]
            cx: int = int((bbox[0] + bbox[2]) / 2)
            cy: int = int((bbox[1] + bbox[3]) / 2)

            self._history.append((frame_index, (cx, cy)))
            self._missing_count = 0

            return BallState(
                center_px=(cx, cy),
                pitch_xy=None,
                is_interpolated=False,
            )

        # No detection this frame.
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
