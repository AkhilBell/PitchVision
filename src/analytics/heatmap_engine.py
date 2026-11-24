"""Fast 2D heatmap accumulator for player positional data.

Maintains per-player accumulator grids and produces Gaussian-smoothed,
normalised heatmaps suitable for visualisation or analytics export.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import HEATMAP_RADIUS_M, PITCH_LENGTH, PITCH_WIDTH


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
        self._team_of: dict[int, str] = {}
        self._clip_frames: int = 1
        """Denominator for time-share heatmaps: frames in the whole clip."""
        self._kernel = _footprint_kernel(grid_shape, HEATMAP_RADIUS_M)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_clip_frames(self, total_frames: int) -> None:
        """Set the clip length used to turn counts into a time share.

        Parameters
        ----------
        total_frames : int
            Number of frames in the video being played.
        """
        self._clip_frames = max(1, int(total_frames))

    def update(self, players: list[Any], weight: float = 1.0) -> None:
        """Add *weight* video-frames of occupancy for each on-screen player.

        Parameters
        ----------
        players : list
            Sequence of objects exposing ``track_id: int`` and
            ``pitch_xy: tuple[float, float] | None``. Players with
            ``offscreen`` set are skipped.
        weight : float
            How many video frames this sample represents. One displayed
            frame should pass ``1`` so the total matches clip time.
        """
        length_cells, width_cells = self.grid_shape
        radius_x = self._kernel.shape[0] // 2
        radius_y = self._kernel.shape[1] // 2
        for player in players:
            if getattr(player, "offscreen", False):
                continue
            if getattr(player, "role", "") == "referee" or getattr(player, "team", "") == "ref":
                continue
            pitch_xy = getattr(player, "pitch_xy", None)
            if pitch_xy is None:
                continue
            track_id: int = getattr(player, "track_id", -1)
            if track_id < 0:
                continue

            if track_id not in self._grids:
                self._grids[track_id] = np.zeros(self.grid_shape, dtype=np.float64)

            team = getattr(player, "team", None)
            if team in ("A", "B"):
                self._team_of[track_id] = team

            gx = int(np.clip(round(pitch_xy[0] / PITCH_LENGTH * (length_cells - 1)), 0, length_cells - 1))
            gy = int(np.clip(round(pitch_xy[1] / PITCH_WIDTH * (width_cells - 1)), 0, width_cells - 1))
            self._stamp(self._grids[track_id], gx, gy, radius_x, radius_y, float(weight))

    def get_heatmap(self, track_id: int) -> np.ndarray | None:
        """Return this player's time-share heatmap.

        Values are ``smoothed frame-counts / frames in the clip``, so a
        cell of ``0.05`` means about 5% of the video was spent there.
        The result is not re-scaled to the player's own peak.

        Parameters
        ----------
        track_id : int
            The player's tracking identifier.

        Returns
        -------
        np.ndarray | None
            2-D array of shape ``grid_shape``, or ``None`` if unseen.
        """
        grid = self._grids.get(track_id)
        if grid is None:
            return None
        return grid / float(self._clip_frames)

    def get_team_heatmap(self, team: str, players: list[Any] | None = None) -> np.ndarray:
        """Aggregate time-share heatmap for every player last seen on *team*.

        Parameters
        ----------
        team : str
            ``"A"`` or ``"B"``.
        players : list | None
            Ignored. Kept so older callers that passed the live player
            list still work; membership comes from teams recorded in
            :meth:`update`.

        Returns
        -------
        np.ndarray
            Time-share heatmap (shape ``grid_shape``).
        """
        del players  # membership is stored per track, not taken from one frame
        aggregate = np.zeros(self.grid_shape, dtype=np.float64)
        for track_id, grid in self._grids.items():
            if self._team_of.get(track_id) != team:
                continue
            aggregate += grid
        return aggregate / float(self._clip_frames)

    def _stamp(
        self,
        grid: np.ndarray,
        gx: int,
        gy: int,
        radius_x: int,
        radius_y: int,
        weight: float,
    ) -> None:
        """Add the 3 ft footprint around one player."""
        x0 = max(0, gx - radius_x)
        x1 = min(grid.shape[0], gx + radius_x + 1)
        y0 = max(0, gy - radius_y)
        y1 = min(grid.shape[1], gy + radius_y + 1)
        kx0 = x0 - (gx - radius_x)
        ky0 = y0 - (gy - radius_y)
        kernel = self._kernel[kx0 : kx0 + (x1 - x0), ky0 : ky0 + (y1 - y0)]
        grid[x0:x1, y0:y1] += kernel * weight

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
            self._team_of.clear()


def _footprint_kernel(
    grid_shape: tuple[int, int], radius_m: float
) -> np.ndarray:
    """Soft disc of *radius_m* metres, 1 at the player and 0 at the edge."""
    length_cells, width_cells = grid_shape
    cell_x = PITCH_LENGTH / max(1, length_cells - 1)
    cell_y = PITCH_WIDTH / max(1, width_cells - 1)
    radius_x = max(1, int(np.ceil(radius_m / cell_x)))
    radius_y = max(1, int(np.ceil(radius_m / cell_y)))
    xs, ys = np.mgrid[-radius_x : radius_x + 1, -radius_y : radius_y + 1]
    distance = np.hypot(xs * cell_x, ys * cell_y)
    kernel = np.clip(1.0 - distance / radius_m, 0.0, 1.0)
    return kernel.astype(np.float64)
