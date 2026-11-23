"""Per-frame soccer pitch landmark detection for homography calibration.

Uses the soccer-trained YOLO pose model published with
`roboflow/sports <https://github.com/roboflow/sports>`_ to find up to 32
pitch landmarks (corners, box corners, centre circle, penalty spots).
Only the confident ones are returned so the caller can estimate a
homography when at least four are available.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
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


class PitchKeypointDetector:
    """Detect pitch landmarks on a video frame.

    Parameters
    ----------
    model_path : str | None
        Path to the YOLO pose weights.  When *None*, the path is read
        from :pydata:`config.PITCH_MODEL_PATH`.  Relative paths are
        resolved against the project root.  When the file is missing
        the detector is disabled and :meth:`detect` always returns
        empty arrays.
    confidence : float
        Minimum per-keypoint confidence to keep a landmark.
    device : str | None
        Torch device for inference.  When *None*, the fastest available
        device is used.
    """

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float | None = None,
        device: str | None = None,
    ) -> None:
        self.confidence: float = (
            float(confidence)
            if confidence is not None
            else float(config.PITCH_CONFIDENCE_THRESHOLD)
        )
        self.device: str = device if device is not None else _auto_device()
        self.model: YOLO | None = None
        self.model_name: str | None = None
        self.available: bool = False

        configured = (
            config.PITCH_MODEL_PATH if model_path is None else model_path
        )
        if not configured:
            logger.warning("Pitch keypoint model disabled in config")
            return

        weights = config.resolve_path(configured)
        if not weights.exists():
            logger.warning(
                "Pitch keypoint weights '%s' not found; radar will stay empty",
                configured,
            )
            return

        try:
            self.model = YOLO(str(weights))
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Could not load pitch keypoint model '%s': %s",
                weights,
                exc,
            )
            return

        self.model_name = weights.name
        self.available = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def description(self) -> str:
        """One-line summary of the pitch model, for the status bar."""
        if not self.available or self.model_name is None:
            return "Pitch: unavailable"
        return f"Pitch: {self.model_name} | Device: {self.device}"

    def detect(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return confident pitch landmarks for *frame*.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H × W × 3).

        Returns
        -------
        pixel_xy : np.ndarray
            Pixel coordinates of kept landmarks, shape ``(N, 2)``.
        pitch_xy : np.ndarray
            Corresponding pitch coordinates in metres, shape ``(N, 2)``.
        indices : np.ndarray
            Landmark indices (0–31) of the kept points, shape ``(N,)``.
        """
        empty = (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.int32),
        )
        if not self.available or self.model is None:
            return empty

        try:
            results = self.model(
                frame,
                conf=0.25,
                imgsz=config.PITCH_DETECTION_IMGSZ,
                device=self.device,
                verbose=False,
            )
        except Exception:
            if self.device == "cpu":
                return empty
            logger.warning(
                "Pitch inference on '%s' failed; falling back to CPU",
                self.device,
                exc_info=True,
            )
            self.device = "cpu"
            try:
                results = self.model(
                    frame,
                    conf=0.25,
                    imgsz=config.PITCH_DETECTION_IMGSZ,
                    device=self.device,
                    verbose=False,
                )
            except Exception:
                return empty

        if not results:
            return empty

        result = results[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            return empty

        xy = np.asarray(keypoints.xy[0].cpu().numpy(), dtype=np.float64)
        if keypoints.conf is None:
            conf = np.ones(len(xy), dtype=np.float64)
        else:
            conf = np.asarray(keypoints.conf[0].cpu().numpy(), dtype=np.float64)

        # Invisible / off-frame predictions sit near (0, 0); drop those too.
        keep = (
            (conf >= self.confidence)
            & (xy[:, 0] > 1.0)
            & (xy[:, 1] > 1.0)
        )
        indices = np.flatnonzero(keep).astype(np.int32)
        if indices.size == 0:
            return empty

        landmarks = np.asarray(config.PITCH_LANDMARKS_M, dtype=np.float64)
        return xy[indices], landmarks[indices], indices
