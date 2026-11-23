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

from pathlib import Path

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

# ── Team & object colours (RGB for Qt) ──────────────────────────────────────
TEAM_A_COLOR_RGB: tuple[int, int, int] = (255, 120, 0)
"""RGB colour for Team A annotations in Qt widgets."""

TEAM_B_COLOR_RGB: tuple[int, int, int] = (50, 50, 255)
"""RGB colour for Team B annotations in Qt widgets."""

REFEREE_COLOR_RGB: tuple[int, int, int] = (255, 255, 0)
"""RGB colour for referee annotations in Qt widgets."""

BALL_COLOR_RGB: tuple[int, int, int] = (0, 255, 0)
"""RGB colour for ball annotations in Qt widgets."""

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
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
"""Repository root; relative model paths are resolved against it."""


def resolve_path(path: str) -> Path:
    """Return *path* as an absolute path, relative paths being taken from the project root."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


MODEL_PATH: str = "models/weights/football-player-detection.pt"
"""Soccer-trained YOLO weights with classes ball, goalkeeper, player, referee.

Relative paths are resolved against the project root.  If the file does
not exist, the detector falls back to ``FALLBACK_MODEL_PATH``.
"""

FALLBACK_MODEL_PATH: str = "models/weights/yolov8n.pt"
"""Generic COCO model used when ``MODEL_PATH`` is missing.

Ultralytics downloads it to this path automatically (6 MB).  It detects
every person in the frame (crowd and staff included) and cannot tell
referees or goalkeepers apart.
"""

CONFIDENCE_THRESHOLD: float = 0.3
"""Minimum detection confidence for YOLO predictions."""

DETECTION_IMGSZ: int = 640
"""Longest side, in pixels, of the image the people model runs on.

Without an explicit size the soccer model uses its training size (1280),
which is about three times slower and finds roughly the same people.
"""

BALL_MODEL_PATH: str | None = "models/weights/football-ball-detection.pt"
"""Dedicated ball-detection weights, or ``None`` to disable.

When missing or disabled, the ball is taken from the main model instead,
which finds it far less often.
"""

BALL_CONFIDENCE_THRESHOLD: float = 0.3
"""Minimum confidence for detections from the dedicated ball model."""

BALL_SEARCH_WINDOW: tuple[int, int] = (320, 192)
"""(width, height) in pixels of the full-resolution window searched around
the ball's last known position.  Both values should be multiples of 32."""

BALL_FULL_SEARCH_IMGSZ: int = 1280
"""Inference size for the whole-frame ball search used when the ball is lost."""

BALL_FULL_SEARCH_INTERVAL: int = 5
"""While the ball is lost, run the whole-frame search once every N frames."""

BALL_MAX_JUMP_FRACTION: float = 0.05
"""Largest plausible ball movement per frame, as a fraction of the frame
diagonal.  Detections further than this from the last position are ignored."""

PITCH_MODEL_PATH: str | None = "models/weights/football-pitch-detection.pt"
"""YOLO pose weights that find 32 pitch landmarks, or ``None`` to disable.

When missing or disabled the pipeline never calibrates, so radar, heatmap
and formation stay empty.  Relative paths are resolved against the
project root.  The weights come from the same [roboflow/sports](
https://github.com/roboflow/sports) release as the player and ball models.
"""

PITCH_CONFIDENCE_THRESHOLD: float = 0.5
"""Minimum per-keypoint confidence to keep a pitch landmark."""

PITCH_DETECTION_IMGSZ: int = 640
"""Inference size for the pitch keypoint model (its training size)."""

PITCH_MIN_KEYPOINTS: int = 4
"""Minimum confident landmarks needed to estimate a homography."""

# Pitch landmarks in metres on this app's 105 × 68 pitch.  They are the
# 32 vertices from roboflow/sports' SoccerPitchConfiguration (120 × 70 m,
# stored in cm) scaled by (105/120, 68/70).  Index order matches the
# football-pitch-detection.pt keypoint order.
PITCH_LANDMARKS_M: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 14.0857),
    (0.0, 25.1017),
    (0.0, 42.8983),
    (0.0, 53.9143),
    (0.0, 68.0),
    (4.8125, 25.1017),
    (4.8125, 42.8983),
    (9.625, 34.0),
    (17.6313, 14.0857),
    (17.6313, 25.1017),
    (17.6313, 42.8983),
    (17.6313, 53.9143),
    (52.5, 0.0),
    (52.5, 25.1114),
    (52.5, 42.8886),
    (52.5, 68.0),
    (87.3688, 14.0857),
    (87.3688, 25.1017),
    (87.3688, 42.8983),
    (87.3688, 53.9143),
    (95.375, 34.0),
    (100.1875, 25.1017),
    (100.1875, 42.8983),
    (105.0, 0.0),
    (105.0, 14.0857),
    (105.0, 25.1017),
    (105.0, 42.8983),
    (105.0, 53.9143),
    (105.0, 68.0),
    (44.4938, 34.0),
    (60.5063, 34.0),
)
"""Pitch landmark positions in metres, indexed like the pose model."""

# Edges connecting landmark indices (0-based), used to draw the projected
# pitch outline back onto the video for the debug toggle.  Same topology
# as roboflow/sports' SoccerPitchConfiguration.edges.
PITCH_LANDMARK_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7),
    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15),
    (15, 16), (17, 18), (18, 19), (19, 20), (22, 23),
    (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),
    (0, 13), (1, 9), (2, 6), (3, 7), (4, 12), (5, 16),
    (13, 24), (17, 25), (22, 26), (23, 27), (20, 28), (16, 29),
)
"""Pairs of landmark indices that form pitch marking segments."""

SCENE_CUT_HIST_CORR: float = 0.6
"""Histogram correlation below which two consecutive frames are treated
as a hard cut (replay, close-up jump, camera angle change)."""

POSITION_SMOOTHING_ALPHA: float = 0.4
"""EMA weight on the new pitch position each frame (0–1).

``smoothed = alpha * new + (1 - alpha) * previous``.  Lower values damp
radar jitter more aggressively; 1.0 disables smoothing.
"""

IOU_THRESHOLD: float = 0.5
"""IoU threshold for non-maximum suppression."""

DEVICE: str | None = None
"""Torch device for YOLO inference (``"mps"``, ``"cuda"`` or ``"cpu"``).

``None`` picks the fastest available: Apple GPU, then CUDA, then CPU.
"""

# ── Heatmap ─────────────────────────────────────────────────────────────────
HEATMAP_GRID_RESOLUTION: tuple[int, int] = (105, 68)
"""(cols, rows) — one grid cell per metre of pitch surface."""

HEATMAP_SIGMA: float = 2.5
"""Standard deviation for the Gaussian blur applied to the heatmap."""

HEATMAP_REDRAW_INTERVAL_S: float = 0.5
"""Minimum time between live heatmap redraws, in seconds."""

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
