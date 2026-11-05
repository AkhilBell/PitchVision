"""Video display widget with real-time tracking overlays.

Renders the raw video frame annotated with perspective foot-ellipses,
player ID badges, and ball markers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from PySide6.QtGui import QPaintEvent

    from src.models import FramePacket

# Attempt to pull team colours from the project config; fall back to
# sensible defaults so the module can be imported standalone.
try:
    from src import config  # type: ignore[import-untyped]

    TEAM_A_COLOR: tuple[int, int, int] = getattr(config, "TEAM_A_COLOR_BGR", (255, 50, 50))
    TEAM_B_COLOR: tuple[int, int, int] = getattr(config, "TEAM_B_COLOR_BGR", (50, 50, 255))
except ImportError:  # pragma: no cover
    TEAM_A_COLOR = (255, 50, 50)  # BGR red-ish
    TEAM_B_COLOR = (50, 50, 255)  # BGR blue-ish


class VideoCanvas(QWidget):
    """Widget that renders annotated video frames.

    Toggle overlay elements via the public boolean attributes
    ``show_ellipses``, ``show_badges``, and ``show_ball``.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._pixmap: QPixmap = QPixmap()

        # Overlay toggle flags (controlled by MainWindow status-bar).
        self.show_ellipses: bool = True
        self.show_badges: bool = True
        self.show_ball: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_frame(self, packet: FramePacket) -> None:
        """Draw overlays on *packet.raw_frame* and schedule a repaint.

        Parameters
        ----------
        packet : FramePacket
            The latest pipeline output containing the raw BGR frame and
            tracking results.
        """
        frame = packet.raw_frame
        if frame is None:
            return
        annotated = self._draw_overlays(frame, packet)
        self._pixmap = self._numpy_to_pixmap(annotated)
        self.update()  # schedule paintEvent

    # ------------------------------------------------------------------
    # Overlay drawing
    # ------------------------------------------------------------------

    def _draw_overlays(self, frame: np.ndarray, packet: FramePacket) -> np.ndarray:
        """Return a copy of *frame* with tracking annotations burned in.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H × W × 3).
        packet : FramePacket
            Tracking results for this frame.

        Returns
        -------
        np.ndarray
            Annotated BGR image.
        """
        vis = frame.copy()

        # -- Players --
        players: list[Any] = getattr(packet, "players", []) or []
        for player in players:
            foot_px = getattr(player, "foot_px", None)
            bbox = getattr(player, "bbox", None)
            track_id = getattr(player, "track_id", None)
            team = getattr(player, "team", None)

            color = TEAM_A_COLOR if team == "team_a" else TEAM_B_COLOR

            # Perspective ellipse at foot position.
            if self.show_ellipses and foot_px is not None:
                cx, cy = int(foot_px[0]), int(foot_px[1])
                axes = (18, 6)  # semi-major/minor in pixels
                cv2.ellipse(vis, (cx, cy), axes, 0, 0, 360, color, -1)
                cv2.ellipse(vis, (cx, cy), axes, 0, 0, 360, (255, 255, 255), 1)

            # ID badge above bounding box.
            if self.show_badges and bbox is not None and track_id is not None:
                x1, y1, x2, y2 = (int(v) for v in bbox[:4])
                label = str(track_id)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                badge_x = x1
                badge_y = max(y1 - th - 8, 0)
                cv2.rectangle(vis, (badge_x, badge_y), (badge_x + tw + 6, badge_y + th + 6), color, -1)
                cv2.putText(
                    vis,
                    label,
                    (badge_x + 3, badge_y + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # -- Ball --
        ball = getattr(packet, "ball", None)
        if self.show_ball and ball is not None:
            ball_px = getattr(ball, "pixel_xy", None) or getattr(ball, "foot_px", None)
            if ball_px is not None:
                bx, by = int(ball_px[0]), int(ball_px[1])
                # Small downward-pointing triangle marker.
                tri_pts = np.array(
                    [[bx, by - 14], [bx - 7, by - 22], [bx + 7, by - 22]],
                    dtype=np.int32,
                )
                cv2.fillPoly(vis, [tri_pts], (0, 255, 0))
                cv2.polylines(vis, [tri_pts], True, (255, 255, 255), 1, cv2.LINE_AA)

        return vis

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _numpy_to_pixmap(frame: np.ndarray) -> QPixmap:
        """Convert a BGR ``np.ndarray`` to a ``QPixmap``.

        Parameters
        ----------
        frame : np.ndarray
            BGR image array (H × W × 3, dtype uint8).

        Returns
        -------
        QPixmap
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_image)

    # ------------------------------------------------------------------
    # Qt paint
    # ------------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:  # type: ignore[override]
        """Draw the current pixmap scaled to fit the widget."""
        if self._pixmap.isNull():
            return
        painter = QPainter(self)
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()
