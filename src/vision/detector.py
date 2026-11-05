"""YOLO-based player and ball detector with ByteTrack tracking.

This module provides the primary detection pipeline for PitchVision,
using an Ultralytics YOLO model for object detection and the
supervision ByteTrack tracker for persistent player identification
across frames.
"""

from __future__ import annotations

import numpy as np
import supervision as sv
from ultralytics import YOLO

from src import config


class PlayerDetector:
    """Detects players and the ball in video frames using YOLO + ByteTrack.

    The detector distinguishes between *person* detections (COCO class 0)
    and *sports-ball* detections (COCO class 32).  Player detections are
    passed through a ByteTrack tracker so that each player receives a
    stable ``tracker_id`` across consecutive frames.

    Parameters
    ----------
    model_path : str | None
        Path to a YOLO weights file (e.g. ``yolov8x.pt``).  When *None*,
        the path is read from :pydata:`config.MODEL_PATH`.
    confidence : float
        Minimum confidence threshold for detections.
    iou : float
        IoU threshold used for non-maximum suppression.
    """

    # COCO class indices
    _PERSON_CLASS: int = 0
    _BALL_CLASS: int = 32

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float = 0.3,
        iou: float = 0.5,
    ) -> None:
        self.confidence: float = confidence
        self.iou: float = iou

        resolved_path: str = model_path if model_path is not None else config.MODEL_PATH
        try:
            self.model: YOLO = YOLO(resolved_path)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to load YOLO model from '{resolved_path}': {exc}"
            ) from exc

        self.tracker: sv.ByteTrack = sv.ByteTrack()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, frame: np.ndarray
    ) -> tuple[sv.Detections, sv.Detections]:
        """Run detection and tracking on a single video frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image array (H × W × 3).

        Returns
        -------
        player_detections : sv.Detections
            Detected people with ``tracker_id`` assigned by ByteTrack.
        ball_detections : sv.Detections
            Detected sports-ball instances (no tracking).
        """
        try:
            results = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou,
                verbose=False,
            )
        except Exception:
            # Return empty detections on inference failure.
            empty = sv.Detections.empty()
            return empty, empty

        all_detections: sv.Detections = self._to_supervision(results)

        # Split by class
        person_mask: np.ndarray = all_detections.class_id == self._PERSON_CLASS
        ball_mask: np.ndarray = all_detections.class_id == self._BALL_CLASS

        player_detections: sv.Detections = all_detections[person_mask]
        ball_detections: sv.Detections = all_detections[ball_mask]

        # Apply ByteTrack to players for persistent IDs
        try:
            player_detections = self.tracker.update_with_detections(player_detections)
        except Exception:
            pass  # Fall back to un-tracked detections.

        return player_detections, ball_detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_supervision(results) -> sv.Detections:
        """Convert Ultralytics *Results* objects to :class:`sv.Detections`.

        Parameters
        ----------
        results
            List of ``ultralytics.engine.results.Results`` returned by
            ``model(frame)``.

        Returns
        -------
        sv.Detections
            Merged detections from all result batches.
        """
        try:
            # results is typically a list with one element per image.
            detections = sv.Detections.from_ultralytics(results[0])
            return detections
        except Exception:
            return sv.Detections.empty()
