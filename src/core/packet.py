"""Frame-level data structures passed through the PitchVision pipeline.

Every processing stage reads from and writes to instances of these
dataclasses, keeping the pipeline loosely coupled.

Classes
-------
PlayerState
    Per-player tracking snapshot for a single frame.
BallState
    Ball position for a single frame (may be interpolated).
PitchState
    Tactical overlay: detected formations and formation-line coordinates.
FramePacket
    Top-level container aggregating all per-frame data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PlayerState:
    """Snapshot of a single tracked player within one video frame.

    Attributes
    ----------
    track_id : int
        Unique track identifier assigned by the tracker.
    bbox : tuple[int, int, int, int]
        Bounding box as *(x1, y1, x2, y2)* in pixel coordinates.
    center_px : tuple[int, int]
        Centre of the bounding box in pixel coordinates *(cx, cy)*.
    foot_px : tuple[int, int]
        Bottom-centre of the bounding box — best proxy for the player's
        position on the pitch surface *(fx, fy)*.
    pitch_xy : tuple[float, float] | None
        Position in real-world pitch coordinates (metres) after
        homography transformation, or ``None`` if not yet computed.
    team : str
        Team label: ``"A"``, ``"B"``, ``"ref"``, or ``"unknown"``.
    jersey_crop : np.ndarray | None
        Cropped torso image used for team-colour classification.
    """

    track_id: int
    bbox: tuple[int, int, int, int]
    center_px: tuple[int, int]
    foot_px: tuple[int, int]
    pitch_xy: Optional[tuple[float, float]] = None
    team: str = "unknown"
    jersey_crop: Optional[np.ndarray] = None


@dataclass
class BallState:
    """Ball position for a single frame.

    Attributes
    ----------
    center_px : tuple[int, int] | None
        Pixel centre of the detected ball, or ``None`` if not visible.
    pitch_xy : tuple[float, float] | None
        Real-world pitch coordinates, or ``None``.
    is_interpolated : bool
        ``True`` when the position was filled by interpolation rather
        than direct detection.
    """

    center_px: Optional[tuple[int, int]] = None
    pitch_xy: Optional[tuple[float, float]] = None
    is_interpolated: bool = False


@dataclass
class PitchState:
    """Tactical state derived from player positions on the pitch.

    Attributes
    ----------
    formation_a : str
        Detected formation string for Team A (e.g. ``"4-3-3"``).
    formation_b : str
        Detected formation string for Team B.
    formation_lines_a : list[tuple[tuple[float, float], ...]]
        Groups of pitch-coordinate tuples representing each detected
        formation line for Team A.
    formation_lines_b : list[tuple[tuple[float, float], ...]]
        Same as above for Team B.
    """

    formation_a: str = "unknown"
    formation_b: str = "unknown"
    formation_lines_a: list = field(default_factory=list)
    formation_lines_b: list = field(default_factory=list)


@dataclass
class FramePacket:
    """Top-level container carrying all data for a single video frame.

    Emitted by :class:`~src.core.pipeline_worker.PipelineWorker` and
    consumed by every UI widget and analytics engine.

    Attributes
    ----------
    frame_index : int
        Zero-based index of the frame within the video.
    timestamp_ms : float
        Presentation timestamp in milliseconds.
    raw_frame : np.ndarray
        Original BGR frame as read from the video file.
    players : list[PlayerState]
        All tracked players in this frame.
    ball : BallState | None
        Ball state, or ``None`` if detection was skipped.
    pitch_state : PitchState | None
        Formation / tactical data, or ``None`` if not yet computed.
    homography_matrix : np.ndarray | None
        3 × 3 perspective transform mapping pixel → pitch coordinates.
    """

    frame_index: int
    timestamp_ms: float
    raw_frame: np.ndarray
    players: list[PlayerState] = field(default_factory=list)
    ball: Optional[BallState] = None
    pitch_state: Optional[PitchState] = None
    homography_matrix: Optional[np.ndarray] = None
