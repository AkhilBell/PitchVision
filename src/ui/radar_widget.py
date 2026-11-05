"""2D pitch minimap (radar) widget.

Renders a simplified top-down view of the pitch with player dots,
ball position, and optional formation overlays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from PySide6.QtGui import QPaintEvent

    from src.models import FramePacket, PitchState

# Team colours (RGB for Qt).
try:
    from src import config  # type: ignore[import-untyped]

    _TEAM_A_RGB: tuple[int, int, int] = getattr(config, "TEAM_A_COLOR_RGB", (220, 50, 50))
    _TEAM_B_RGB: tuple[int, int, int] = getattr(config, "TEAM_B_COLOR_RGB", (50, 80, 220))
except ImportError:  # pragma: no cover
    _TEAM_A_RGB = (220, 50, 50)
    _TEAM_B_RGB = (50, 80, 220)

# Pitch dimensions in metres (FIFA standard).
_PITCH_LENGTH = 105.0
_PITCH_WIDTH = 68.0

# Widget padding in pixels.
_PAD = 16


class RadarWidget(QWidget):
    """Top-down minimap showing player and ball positions plus optional
    formation annotations.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(300)

        # Data caches (updated every frame).
        self._players: list[Any] = []
        self._ball: Any | None = None
        self._pitch_state: PitchState | None = None  # type: ignore[assignment]

        # Overlay toggle.
        self.show_formations: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_data(self, packet: FramePacket) -> None:
        """Store tracking data from *packet* and trigger a repaint.

        Parameters
        ----------
        packet : FramePacket
            Latest pipeline output.
        """
        self._players = getattr(packet, "players", []) or []
        self._ball = getattr(packet, "ball", None)
        self._pitch_state = getattr(packet, "pitch_state", None)
        self.update()  # schedule paintEvent

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def _pitch_to_widget(self, x: float, y: float) -> tuple[int, int]:
        """Map pitch coordinates ``(x, y)`` to widget pixel coords.

        Parameters
        ----------
        x : float
            Along-pitch position in [0, 105].
        y : float
            Across-pitch position in [0, 68].

        Returns
        -------
        tuple[int, int]
            ``(px, py)`` in widget space.
        """
        draw_w = self.width() - 2 * _PAD
        draw_h = self.height() - 2 * _PAD - 30  # reserve space for labels
        px = int(_PAD + (x / _PITCH_LENGTH) * draw_w)
        py = int(_PAD + (y / _PITCH_WIDTH) * draw_h)
        return px, py

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:  # type: ignore[override]
        """Draw the minimap."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        draw_w = self.width() - 2 * _PAD
        draw_h = self.height() - 2 * _PAD - 30

        # 1. Pitch background.
        painter.setBrush(QBrush(QColor(34, 139, 34)))  # forest green
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(_PAD, _PAD, draw_w, draw_h)

        # 2. Pitch markings (white).
        pen_white = QPen(QColor(255, 255, 255), 1.5)
        painter.setPen(pen_white)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Outline.
        painter.drawRect(_PAD, _PAD, draw_w, draw_h)

        # Centre line.
        cx = _PAD + draw_w // 2
        painter.drawLine(cx, _PAD, cx, _PAD + draw_h)

        # Centre circle (radius ~9.15 m → fraction of pitch).
        r_px = int(9.15 / _PITCH_LENGTH * draw_w)
        cy = _PAD + draw_h // 2
        painter.drawEllipse(QPoint(cx, cy), r_px, int(r_px * draw_h / draw_w * _PITCH_LENGTH / _PITCH_WIDTH))

        # Penalty areas (16.5 m deep, 40.3 m wide).
        pa_depth = int(16.5 / _PITCH_LENGTH * draw_w)
        pa_half_w = int(20.15 / _PITCH_WIDTH * draw_h)
        # Left.
        painter.drawRect(_PAD, cy - pa_half_w, pa_depth, 2 * pa_half_w)
        # Right.
        painter.drawRect(_PAD + draw_w - pa_depth, cy - pa_half_w, pa_depth, 2 * pa_half_w)

        # Goal areas (5.5 m deep, 18.3 m wide).
        ga_depth = int(5.5 / _PITCH_LENGTH * draw_w)
        ga_half_w = int(9.15 / _PITCH_WIDTH * draw_h)
        painter.drawRect(_PAD, cy - ga_half_w, ga_depth, 2 * ga_half_w)
        painter.drawRect(_PAD + draw_w - ga_depth, cy - ga_half_w, ga_depth, 2 * ga_half_w)

        # 3. Players.
        tiny_font = QFont("Arial", 7)
        painter.setFont(tiny_font)

        for player in self._players:
            pitch_xy = getattr(player, "pitch_xy", None)
            if pitch_xy is None:
                continue
            px, py = self._pitch_to_widget(pitch_xy[0], pitch_xy[1])
            team = getattr(player, "team", None)
            if team == "team_a":
                color = QColor(*_TEAM_A_RGB)
            elif team == "team_b":
                color = QColor(*_TEAM_B_RGB)
            else:
                color = QColor(200, 200, 200)

            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255), 0.8))
            painter.drawEllipse(QPoint(px, py), 5, 5)

            # Track ID text.
            track_id = getattr(player, "track_id", None)
            if track_id is not None:
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.drawText(px + 7, py - 2, str(track_id))

        # 4. Ball.
        if self._ball is not None:
            ball_xy = getattr(self._ball, "pitch_xy", None)
            if ball_xy is not None:
                bx, by = self._pitch_to_widget(ball_xy[0], ball_xy[1])
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawEllipse(QPoint(bx, by), 4, 4)

        # 5. Formation labels & cluster lines.
        if self.show_formations and self._pitch_state is not None:
            label_font = QFont("Arial", 9)
            painter.setFont(label_font)
            painter.setPen(QPen(QColor(255, 255, 255)))

            form_a = getattr(self._pitch_state, "formation_a", "?")
            form_b = getattr(self._pitch_state, "formation_b", "?")
            label_y = _PAD + draw_h + 14
            painter.drawText(_PAD, label_y, f"Team A: {form_a}")
            painter.drawText(_PAD + draw_w // 2, label_y, f"Team B: {form_b}")

            # Draw lines connecting players within each formation line.
            self._draw_formation_lines(
                painter,
                getattr(self._pitch_state, "formation_lines_a", []),
                QColor(*_TEAM_A_RGB, 120),
            )
            self._draw_formation_lines(
                painter,
                getattr(self._pitch_state, "formation_lines_b", []),
                QColor(*_TEAM_B_RGB, 120),
            )

        painter.end()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _draw_formation_lines(
        self,
        painter: QPainter,
        formation_lines: list[list[tuple[float, float]]],
        color: QColor,
    ) -> None:
        """Connect players within each formation cluster line.

        Parameters
        ----------
        painter : QPainter
            Active painter.
        formation_lines : list[list[tuple[float, float]]]
            Groups of ``(x, y)`` pitch coords.
        color : QColor
            Line colour (may include alpha).
        """
        pen = QPen(color, 1.5)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for line_group in formation_lines:
            if len(line_group) < 2:
                continue
            widget_pts = [self._pitch_to_widget(x, y) for x, y in line_group]
            for i in range(len(widget_pts) - 1):
                painter.drawLine(
                    QPoint(*widget_pts[i]),
                    QPoint(*widget_pts[i + 1]),
                )
