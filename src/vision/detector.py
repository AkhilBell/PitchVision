"""YOLO-based player and ball detector with ByteTrack tracking.

This module provides the primary detection pipeline for PitchVision,
using an Ultralytics YOLO model for object detection and the
supervision ByteTrack tracker for persistent player identification
across frames.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import supervision as sv
import torch
from trackers import ByteTrackTracker
from ultralytics import YOLO

from src import config

logger = logging.getLogger(__name__)


def _auto_device() -> str:
    """Return the fastest available torch device name."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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
    device : str | None
        Torch device for inference.  When *None*, the fastest available
        device is used (Apple GPU, then CUDA, then CPU).
    """

    # COCO class indices
    _PERSON_CLASS: int = 0
    _BALL_CLASS: int = 32

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float = 0.3,
        iou: float = 0.5,
        device: str | None = None,
    ) -> None:
        self.confidence: float = confidence
        self.iou: float = iou
        self.device: str = device if device is not None else _auto_device()

        configured = model_path if model_path is not None else config.MODEL_PATH
        # Prefer local weights when present; otherwise fall back to a small
        # Ultralytics model name that can be auto-downloaded for demos.
        if Path(configured).exists():
            resolved_path = configured
        else:
            resolved_path = "yolov8n.pt"

        try:
            self.model: YOLO = YOLO(resolved_path)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to load YOLO model from '{resolved_path}' "
                f"(configured='{configured}'): {exc}"
            ) from exc

        # supervision ≥ 0.31 moved ByteTrack into the external ``trackers`` package.
        self.tracker = ByteTrackTracker(
            track_activation_threshold=0.25,
            minimum_consecutive_frames=1,
        )

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
            results = self._infer(frame)
        except Exception:
            if self.device == "cpu":
                # Return empty detections on inference failure.
                empty = sv.Detections.empty()
                return empty, empty
            # GPU inference failed: switch to CPU for the rest of the run
            # rather than silently producing no detections.
            logger.warning(
                "Inference on '%s' failed; falling back to CPU", self.device, exc_info=True
            )
            self.device = "cpu"
            try:
                results = self._infer(frame)
            except Exception:
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
            player_detections = self.tracker.update(player_detections)
        except Exception:
            pass  # Fall back to un-tracked detections.

        return player_detections, ball_detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer(self, frame: np.ndarray):
        """Run the YOLO model on *frame* using the current device."""
        return self.model(
            frame,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

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
