"""Real-time player heatmap display widget.

Embeds a Matplotlib ``FigureCanvas`` inside a PySide6 layout and renders
a 2D positional heatmap for the currently-selected player, overlaid on
a simplified pitch outline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

try:
    import matplotlib

    matplotlib.use("QtAgg")  # Must be set before importing pyplot.
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False
    FigureCanvas = None  # type: ignore[assignment,misc]
    Figure = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from src.analytics.heatmap_engine import HeatmapEngine

# Pitch dimensions (metres).
_PITCH_LENGTH = 105.0
_PITCH_WIDTH = 68.0


class HeatmapWidget(QWidget):
    """Displays a live per-player heatmap using Matplotlib.

    The ``HeatmapEngine`` reference must be injected via
    :meth:`set_engine` before meaningful rendering can occur.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._engine: HeatmapEngine | None = None
        self._known_ids: set[int] = set()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Title label.
        self._label = QLabel("Live Player Heatmap")
        layout.addWidget(self._label)

        # Player selector combo box.
        self._combo = QComboBox()
        self._combo.currentIndexChanged.connect(self._on_player_changed)
        layout.addWidget(self._combo)

        # Matplotlib canvas.
        if _MPL_AVAILABLE:
            self._figure = Figure(figsize=(4, 3), dpi=80)
            self._ax = self._figure.add_subplot(111)
            self._canvas = FigureCanvas(self._figure)
            layout.addWidget(self._canvas, stretch=1)
        else:
            self._figure = None
            self._ax = None
            self._canvas = None
            layout.addWidget(QLabel("(matplotlib not available)"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_engine(self, engine: HeatmapEngine) -> None:
        """Inject a reference to the shared :class:`HeatmapEngine`.

        Parameters
        ----------
        engine : HeatmapEngine
            The engine accumulating player positions.
        """
        self._engine = engine

    def update_players(self, players: list[Any]) -> None:
        """Refresh the combo box with available track IDs.

        New IDs are appended without removing existing ones to prevent
        UI flickering.  If no player is currently selected the first
        entry is auto-selected.

        Parameters
        ----------
        players : list
            Objects with ``track_id: int``.
        """
        for p in players:
            tid = getattr(p, "track_id", None)
            if tid is not None and tid not in self._known_ids:
                self._known_ids.add(tid)
                self._combo.addItem(f"Player {tid}", userData=tid)

        if self._combo.currentIndex() < 0 and self._combo.count() > 0:
            self._combo.setCurrentIndex(0)

    def update_heatmap(self) -> None:
        """Re-render the heatmap for the currently selected player."""
        if self._engine is None or self._ax is None or self._canvas is None:
            return

        idx = self._combo.currentIndex()
        if idx < 0:
            return
        track_id: int = self._combo.itemData(idx)
        if track_id is None:
            return

        heatmap = self._engine.get_heatmap(track_id)

        self._ax.clear()

        # Draw a simplified pitch outline.
        self._draw_pitch_outline()

        if heatmap is not None:
            self._ax.imshow(
                heatmap.T,
                origin="lower",
                extent=[0, _PITCH_LENGTH, 0, _PITCH_WIDTH],
                cmap="hot",
                alpha=0.7,
                aspect="auto",
            )

        self._ax.set_xlim(0, _PITCH_LENGTH)
        self._ax.set_ylim(0, _PITCH_WIDTH)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._figure.tight_layout(pad=0.5)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_player_changed(self, index: int) -> None:
        """Handle combo-box selection change."""
        self.update_heatmap()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_pitch_outline(self) -> None:
        """Draw basic pitch markings on ``self._ax``."""
        if self._ax is None:
            return

        ax = self._ax
        lw = 0.8
        color = "white"

        # Background.
        ax.set_facecolor("#228B22")

        # Outline.
        ax.plot(
            [0, _PITCH_LENGTH, _PITCH_LENGTH, 0, 0],
            [0, 0, _PITCH_WIDTH, _PITCH_WIDTH, 0],
            color=color,
            linewidth=lw,
        )

        # Centre line.
        ax.plot(
            [_PITCH_LENGTH / 2, _PITCH_LENGTH / 2],
            [0, _PITCH_WIDTH],
            color=color,
            linewidth=lw,
        )

        # Centre circle.
        theta = np.linspace(0, 2 * np.pi, 60)
        r = 9.15
        cx, cy = _PITCH_LENGTH / 2, _PITCH_WIDTH / 2
        ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), color=color, linewidth=lw)

        # Penalty areas (16.5 m × 40.3 m).
        pa_d = 16.5
        pa_hw = 20.15
        mid = _PITCH_WIDTH / 2
        for x0 in [0, _PITCH_LENGTH - pa_d]:
            ax.plot(
                [x0, x0 + pa_d, x0 + pa_d, x0],
                [mid - pa_hw, mid - pa_hw, mid + pa_hw, mid + pa_hw],
                color=color,
                linewidth=lw,
            )

        # Goal areas (5.5 m × 18.3 m).
        ga_d = 5.5
        ga_hw = 9.15
        for x0 in [0, _PITCH_LENGTH - ga_d]:
            ax.plot(
                [x0, x0 + ga_d, x0 + ga_d, x0],
                [mid - ga_hw, mid - ga_hw, mid + ga_hw, mid + ga_hw],
                color=color,
                linewidth=lw,
            )
