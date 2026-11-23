"""YOLO-based player and ball detector with ByteTrack tracking.

This module provides the primary detection pipeline for PitchVision.
A soccer-trained Ultralytics YOLO model finds players, goalkeepers and
referees; a ByteTrack tracker gives each of them a persistent ID; and a
dedicated ball model searches a small full-resolution window around the
ball's last known position, because the ball is too small to find
reliably in the downscaled frame the main model sees.

A generic COCO model also works: every person is then reported with the
role ``"player"`` and the ball comes from the COCO ``sports ball`` class.
"""

from __future__ import annotations

import logging

import numpy as np
import supervision as sv
import torch
from trackers import ByteTrackTracker
from ultralytics import YOLO

from src import config

logger = logging.getLogger(__name__)

_USE_CONFIG = "<config>"
"""Sentinel meaning "read the ball model path from the configuration"."""

# Detector class name → role stored on PlayerState.
_ROLE_BY_CLASS_NAME: dict[str, str] = {
    "player": "player",
    "person": "player",
    "goalkeeper": "goalkeeper",
    "referee": "referee",
}
_BALL_CLASS_NAMES: frozenset[str] = frozenset({"ball", "sports ball"})


def _auto_device() -> str:
    """Return the fastest available torch device name."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _round_up_to_stride(value: int, stride: int = 32) -> int:
    """Round *value* up to the next multiple of the YOLO stride."""
    return max(stride, ((int(value) + stride - 1) // stride) * stride)


class PlayerDetector:
    """Detects people and the ball in video frames using YOLO + ByteTrack.

    Classes are matched by *name*, so both the soccer-trained model
    (``ball``, ``goalkeeper``, ``player``, ``referee``) and generic COCO
    models (``person``, ``sports ball``) are supported.  People are passed
    through a ByteTrack tracker so that each one receives a stable
    ``tracker_id`` across consecutive frames.

    Parameters
    ----------
    model_path : str | None
        Path to the YOLO weights for people.  When *None*, the path is
        read from :pydata:`config.MODEL_PATH`.  Relative paths are
        resolved against the project root; if the file is missing,
        :pydata:`config.FALLBACK_MODEL_PATH` is used instead.
    confidence : float
        Minimum confidence threshold for detections.
    iou : float
        IoU threshold used for non-maximum suppression.
    device : str | None
        Torch device for inference.  When *None*, the fastest available
        device is used (Apple GPU, then CUDA, then CPU).
    ball_model_path : str | None
        Path to dedicated ball-detection weights.  By default the path is
        read from :pydata:`config.BALL_MODEL_PATH`.  When *None* or the
        file is missing, the ball is taken from the main model.
    """

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float = 0.3,
        iou: float = 0.5,
        device: str | None = None,
        ball_model_path: str | None = _USE_CONFIG,
    ) -> None:
        self.confidence: float = confidence
        self.iou: float = iou
        self.device: str = device if device is not None else _auto_device()

        # ── Main (people) model ─────────────────────────────────────────
        configured = model_path if model_path is not None else config.MODEL_PATH
        weights = config.resolve_path(configured)
        self.using_fallback_model: bool = not weights.exists()
        if self.using_fallback_model:
            # Ultralytics downloads standard models to this path if needed.
            weights = config.resolve_path(config.FALLBACK_MODEL_PATH)
            logger.warning(
                "Model weights '%s' not found; using generic model '%s'",
                configured,
                weights,
            )

        try:
            self.model: YOLO = YOLO(str(weights))
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to load YOLO model from '{weights}' "
                f"(configured='{configured}'): {exc}"
            ) from exc
        self.model_name: str = weights.name

        names: dict[int, str] = dict(self.model.names)
        self._role_by_class_id: dict[int, str] = {
            class_id: _ROLE_BY_CLASS_NAME[name]
            for class_id, name in names.items()
            if name in _ROLE_BY_CLASS_NAME
        }
        self._ball_class_ids: list[int] = [
            class_id for class_id, name in names.items() if name in _BALL_CLASS_NAMES
        ]
        if not self._role_by_class_id:
            raise RuntimeError(
                f"Model '{weights}' has no player or person class (classes: {names})"
            )
        self._wanted_class_ids: list[int] = sorted(
            [*self._role_by_class_id, *self._ball_class_ids]
        )
        self.provides_roles: bool = "referee" in self._role_by_class_id.values()
        """Whether the model tells referees and goalkeepers apart from players."""

        # ── Dedicated ball model (optional) ─────────────────────────────
        self.ball_model: YOLO | None = None
        self.ball_model_name: str | None = None
        ball_configured = (
            config.BALL_MODEL_PATH if ball_model_path == _USE_CONFIG else ball_model_path
        )
        if ball_configured:
            ball_weights = config.resolve_path(ball_configured)
            if ball_weights.exists():
                try:
                    self.ball_model = YOLO(str(ball_weights))
                    self.ball_model_name = ball_weights.name
                except Exception:
                    logger.warning(
                        "Could not load ball model '%s'; using the main model for the ball",
                        ball_weights,
                        exc_info=True,
                    )
            else:
                logger.warning(
                    "Ball model weights '%s' not found; using the main model for the ball",
                    ball_configured,
                )
        self._frames_since_full_ball_search: int = config.BALL_FULL_SEARCH_INTERVAL

        # supervision ≥ 0.31 moved ByteTrack into the external ``trackers`` package.
        self.tracker = ByteTrackTracker(
            track_activation_threshold=0.25,
            minimum_consecutive_frames=1,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def description(self) -> str:
        """One-line summary of the models in use, for display in the UI."""
        people = self.model_name
        if self.using_fallback_model:
            people += " (generic fallback)"
        ball = self.ball_model_name if self.ball_model is not None else "main model"
        return f"Detector: {people} | Ball: {ball} | Device: {self.device}"

    def detect(
        self,
        frame: np.ndarray,
        ball_hint: tuple[float, float] | None = None,
        run_ball: bool = True,
    ) -> tuple[sv.Detections, sv.Detections]:
        """Run detection and tracking on a single video frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image array (H × W × 3).
        ball_hint : tuple[float, float] | None
            Pixel position where the ball was last seen, or *None* when
            the ball is currently lost.  Used to search for the ball in a
            small window instead of the whole frame.
        run_ball : bool
            When ``False``, skip the dedicated ball model and return no
            ball detections. The caller keeps the previous ball position.

        Returns
        -------
        player_detections : sv.Detections
            Detected people with ``tracker_id`` assigned by ByteTrack.
            Use :meth:`roles` to get each one's role.
        ball_detections : sv.Detections
            Candidate ball detections (no tracking).
        """
        all_detections = self._run(
            self.model,
            frame,
            conf=self.confidence,
            classes=self._wanted_class_ids,
            imgsz=config.DETECTION_IMGSZ,
        )

        if len(all_detections) > 0:
            person_mask = np.isin(all_detections.class_id, list(self._role_by_class_id))
            ball_mask = np.isin(all_detections.class_id, self._ball_class_ids)
            player_detections: sv.Detections = all_detections[person_mask]
            ball_detections: sv.Detections = all_detections[ball_mask]
        else:
            player_detections = all_detections
            ball_detections = sv.Detections.empty()

        # Apply ByteTrack to people for persistent IDs
        try:
            player_detections = self.tracker.update(player_detections)
        except Exception:
            pass  # Fall back to un-tracked detections.

        if run_ball and self.ball_model is not None:
            dedicated = self._detect_ball(frame, ball_hint)
            # The main model's ball class is a free second opinion on
            # frames where the dedicated search finds nothing.
            if len(dedicated) > 0:
                ball_detections = dedicated

        return player_detections, ball_detections

    def roles(self, detections: sv.Detections) -> list[str]:
        """Return the role of each detection in *detections*.

        Roles are ``"player"``, ``"goalkeeper"`` or ``"referee"``.  They
        are read from the detections' own class IDs because the tracker
        does not keep detections in their original order.
        """
        if len(detections) == 0 or detections.class_id is None:
            return ["player"] * len(detections)
        return [
            self._role_by_class_id.get(int(class_id), "player")
            for class_id in detections.class_id
        ]

    # ------------------------------------------------------------------
    # Ball search
    # ------------------------------------------------------------------

    def _detect_ball(
        self, frame: np.ndarray, hint: tuple[float, float] | None
    ) -> sv.Detections:
        """Find ball candidates with the dedicated ball model.

        While the ball is being followed, only a small window around its
        last position is searched, at full resolution.  When that fails
        or the ball is lost, the whole frame is searched at high
        resolution, at most once every few frames because it is slow.
        """
        detections = sv.Detections.empty()
        if hint is not None:
            detections = self._search_ball_window(frame, hint)

        if (
            len(detections) == 0
            and self._frames_since_full_ball_search >= config.BALL_FULL_SEARCH_INTERVAL
        ):
            frame_h, frame_w = frame.shape[:2]
            imgsz = min(
                config.BALL_FULL_SEARCH_IMGSZ,
                _round_up_to_stride(max(frame_h, frame_w)),
            )
            detections = self._run(
                self.ball_model,
                frame,
                conf=config.BALL_CONFIDENCE_THRESHOLD,
                imgsz=imgsz,
            )
            self._frames_since_full_ball_search = 0
        else:
            self._frames_since_full_ball_search += 1

        return detections

    def _search_ball_window(
        self, frame: np.ndarray, center: tuple[float, float]
    ) -> sv.Detections:
        """Search for the ball in a full-resolution window around *center*."""
        frame_h, frame_w = frame.shape[:2]
        win_w = min(config.BALL_SEARCH_WINDOW[0], frame_w)
        win_h = min(config.BALL_SEARCH_WINDOW[1], frame_h)
        x0 = int(min(max(center[0] - win_w / 2, 0), frame_w - win_w))
        y0 = int(min(max(center[1] - win_h / 2, 0), frame_h - win_h))
        crop = frame[y0 : y0 + win_h, x0 : x0 + win_w]

        detections = self._run(
            self.ball_model,
            crop,
            conf=config.BALL_CONFIDENCE_THRESHOLD,
            imgsz=_round_up_to_stride(max(win_w, win_h)),
        )
        if len(detections) > 0:
            # Back to whole-frame coordinates.
            detections.xyxy = detections.xyxy + np.array(
                [x0, y0, x0, y0], dtype=detections.xyxy.dtype
            )
        return detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, model: YOLO, image: np.ndarray, **kwargs) -> sv.Detections:
        """Run *model* on *image* and return the detections.

        If inference fails on the GPU, the detector switches to the CPU
        for the rest of the run rather than silently producing nothing.
        """
        try:
            results = self._infer(model, image, **kwargs)
        except Exception:
            if self.device == "cpu":
                return sv.Detections.empty()
            logger.warning(
                "Inference on '%s' failed; falling back to CPU", self.device, exc_info=True
            )
            self.device = "cpu"
            try:
                results = self._infer(model, image, **kwargs)
            except Exception:
                return sv.Detections.empty()
        return self._to_supervision(results)

    def _infer(self, model: YOLO, image: np.ndarray, **kwargs):
        """Run a YOLO model on *image* using the current device."""
        return model(
            image,
            iou=self.iou,
            device=self.device,
            verbose=False,
            **kwargs,
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
            Detections for the (single) image that was processed.
        """
        try:
            # results is typically a list with one element per image.
            return sv.Detections.from_ultralytics(results[0])
        except Exception:
            return sv.Detections.empty()
