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
        self._team_of: dict[int, str] = {}
        self._clip_frames: int = 1
        """Denominator for time-share heatmaps: frames in the whole clip."""

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
        rows, cols = self.grid_shape
        for player in players:
            if getattr(player, "offscreen", False):
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

            gx = int(np.clip(pitch_xy[0], 0, rows - 1))
            gy = int(np.clip(pitch_xy[1], 0, cols - 1))
            self._grids[track_id][gx, gy] += float(weight)

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
        smoothed = gaussian_filter(grid.astype(np.float64), sigma=self.sigma)
        return smoothed / float(self._clip_frames)

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
        smoothed = gaussian_filter(aggregate, sigma=self.sigma)
        return smoothed / float(self._clip_frames)

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
