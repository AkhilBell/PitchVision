"""Fast 2D heatmap accumulator for player positional data.

Maintains per-player accumulator grids and produces Gaussian-smoothed,
normalised heatmaps suitable for visualisation or analytics export.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


class HeatmapEngine:
    """Accumulates player positions on a discretised pitch grid and
    produces smoothed heatmaps on demand.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        Number of cells along the pitch length and width.
        Default ``(105, 68)`` mirrors a 105 × 68 m pitch at 1 m resolution.
    sigma : float
        Standard deviation (in grid cells) for the Gaussian blur applied
        when retrieving a heatmap.  Default ``2.5``.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (105, 68),
        sigma: float = 2.5,
    ) -> None:
        self.grid_shape: tuple[int, int] = grid_shape
        self.sigma: float = sigma
        self._grids: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, players: list[Any]) -> None:
        """Increment accumulator cells for every player with a valid
        ``pitch_xy`` attribute.

        Parameters
        ----------
        players : list
            Sequence of objects exposing ``track_id: int`` and
            ``pitch_xy: tuple[float, float] | None``.
        """
        rows, cols = self.grid_shape
        for player in players:
            pitch_xy = getattr(player, "pitch_xy", None)
            if pitch_xy is None:
                continue
            track_id: int = getattr(player, "track_id", -1)
            if track_id < 0:
                continue

            # Ensure an accumulator exists for this player.
            if track_id not in self._grids:
                self._grids[track_id] = np.zeros(self.grid_shape, dtype=np.float64)

            # Convert continuous pitch coords to integer grid indices.
            gx = int(np.clip(pitch_xy[0], 0, rows - 1))
            gy = int(np.clip(pitch_xy[1], 0, cols - 1))
            self._grids[track_id][gx, gy] += 1.0

    def get_heatmap(self, track_id: int) -> np.ndarray | None:
        """Return a Gaussian-smoothed, 0–1 normalised heatmap for
        *track_id*, or ``None`` if the id has never been seen.

        Parameters
        ----------
        track_id : int
            The player's tracking identifier.

        Returns
        -------
        np.ndarray | None
            2-D array of shape ``grid_shape`` with values in [0, 1],
            or ``None``.
        """
        grid = self._grids.get(track_id)
        if grid is None:
            return None

        smoothed = gaussian_filter(grid.astype(np.float64), sigma=self.sigma)
        peak = smoothed.max()
        if peak > 0:
            smoothed /= peak
        return smoothed

    def get_team_heatmap(self, players: list[Any], team: str) -> np.ndarray:
        """Aggregate smoothed heatmaps for every player belonging to
        *team*.

        Parameters
        ----------
        players : list
            Objects with ``track_id``, ``pitch_xy``, and ``team`` attrs.
        team : str
            Team identifier to filter on (e.g. ``"team_a"``).

        Returns
        -------
        np.ndarray
            Normalised aggregate heatmap (shape ``grid_shape``).
        """
        aggregate = np.zeros(self.grid_shape, dtype=np.float64)
        for player in players:
            if getattr(player, "team", None) != team:
                continue
            hm = self.get_heatmap(getattr(player, "track_id", -1))
            if hm is not None:
                aggregate += hm

        peak = aggregate.max()
        if peak > 0:
            aggregate /= peak
        return aggregate

    def get_all_track_ids(self) -> list[int]:
        """Return a sorted list of every track ID seen so far."""
        return sorted(self._grids.keys())

    def reset(self, track_id: int | None = None) -> None:
        """Clear accumulator data.

        Parameters
        ----------
        track_id : int | None
            If given, reset only that player's grid.  Otherwise clear
            everything.
        """
        if track_id is not None:
            if track_id in self._grids:
                self._grids[track_id] = np.zeros(self.grid_shape, dtype=np.float64)
        else:
            self._grids.clear()
