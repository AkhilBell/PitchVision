"""Central configuration for the PitchVision pipeline and UI.

All constants are module-level for easy import:
    ``from src.config import PITCH_LENGTH, TEAM_A_COLOR``

Groups
------
Pitch geometry
    FIFA-standard pitch dimensions and line colours.
Team / object colours
    BGR tuples for OpenCV drawing **and** hex strings for the Qt/Matplotlib UI.
Model & detection
    YOLO weights path, confidence and IoU thresholds.
Heatmap
    Grid resolution (cells) and Gaussian blur sigma.
Formation detection
    Range of cluster counts to evaluate with K-Means.
Rendering
    FPS cap, export codec, annotation sizes.
"""

from __future__ import annotations

# ── Pitch geometry ──────────────────────────────────────────────────────────
PITCH_LENGTH: float = 105.0
"""Pitch length in metres (FIFA standard)."""

PITCH_WIDTH: float = 68.0
"""Pitch width in metres (FIFA standard)."""

PITCH_LINE_COLOR: tuple[int, int, int] = (255, 255, 255)
"""BGR colour for pitch boundary / marking lines."""

# ── Team & object colours (BGR for OpenCV) ──────────────────────────────────
TEAM_A_COLOR: tuple[int, int, int] = (0, 120, 255)
"""BGR colour for Team A annotations (orange-ish)."""

TEAM_B_COLOR: tuple[int, int, int] = (255, 50, 50)
"""BGR colour for Team B annotations (blue-ish)."""

REFEREE_COLOR: tuple[int, int, int] = (0, 255, 255)
"""BGR colour for referee annotations (yellow)."""

BALL_COLOR: tuple[int, int, int] = (0, 255, 0)
"""BGR colour for ball annotations (green)."""

# ── Team & object colours (hex for Qt / Matplotlib) ─────────────────────────
TEAM_A_COLOR_HEX: str = "#FF7800"
"""Hex colour string for Team A in the UI."""

TEAM_B_COLOR_HEX: str = "#3232FF"
"""Hex colour string for Team B in the UI."""

REFEREE_COLOR_HEX: str = "#FFFF00"
"""Hex colour string for referees in the UI."""

BALL_COLOR_HEX: str = "#00FF00"
"""Hex colour string for the ball in the UI."""

# ── Model & detection ───────────────────────────────────────────────────────
MODEL_PATH: str = "models/weights/yolov8x.pt"
"""Default path to YOLOv8 weights (relative to project root)."""

CONFIDENCE_THRESHOLD: float = 0.3
"""Minimum detection confidence for YOLO predictions."""

IOU_THRESHOLD: float = 0.5
"""IoU threshold for non-maximum suppression."""

# ── Heatmap ─────────────────────────────────────────────────────────────────
HEATMAP_GRID_RESOLUTION: tuple[int, int] = (105, 68)
"""(cols, rows) — one grid cell per metre of pitch surface."""

HEATMAP_SIGMA: float = 2.5
"""Standard deviation for the Gaussian blur applied to the heatmap."""

# ── Formation detection ─────────────────────────────────────────────────────
FORMATION_CLUSTERS_RANGE: range = range(2, 6)
"""Range of *k* values to evaluate when fitting K-Means for formation lines."""

# ── Rendering / export ──────────────────────────────────────────────────────
MAX_FPS: int = 30
"""Maximum frames-per-second for real-time playback."""

VIDEO_EXPORT_CODEC: str = "mp4v"
"""FourCC codec identifier used by :class:`cv2.VideoWriter`."""

ELLIPSE_HEIGHT: int = 10
"""Pixel height of the ellipse drawn at each player's feet."""

BADGE_FONT_SCALE: float = 0.5
"""Font scale for jersey-number / ID badges drawn on the overlay."""

BADGE_THICKNESS: int = 1
"""Line thickness for badge text drawn on the overlay."""
