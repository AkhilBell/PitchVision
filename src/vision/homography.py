"""Perspective homography transformation between pixel and pitch coordinates.

Provides calibration-based and fallback (corner-mapping) methods to
project pixel locations from broadcast footage onto a FIFA-standard
pitch coordinate system.
"""

from __future__ import annotations

import cv2
import numpy as np


class HomographyTransformer:
    """Transforms coordinates between the video frame and the pitch plane.

    All pitch coordinates follow the FIFA convention:

    * Origin ``(0, 0)`` is the top-left corner of the pitch.
    * *x* runs along the length (0 → ``pitch_length``).
    * *y* runs along the width  (0 → ``pitch_width``).

    Parameters
    ----------
    pitch_length : float
        Pitch length in metres (default ``105.0``).
    pitch_width : float
        Pitch width in metres (default ``68.0``).
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ) -> None:
        self.pitch_length: float = pitch_length
        self.pitch_width: float = pitch_width
        self._matrix: np.ndarray | None = None
        self._inverse: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def set_transform(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> None:
        """Compute and store the homography from ≥ 4 point correspondences.

        Parameters
        ----------
        src_points : np.ndarray
            Pixel coordinates of known pitch landmarks, shape ``(N, 2)``
            with ``N >= 4``.
        dst_points : np.ndarray
            Corresponding pitch coordinates in metres, shape ``(N, 2)``.

        Raises
        ------
        ValueError
            If fewer than 4 point correspondences are provided or if
            ``cv2.findHomography`` fails.
        """
        src = np.asarray(src_points, dtype=np.float64)
        dst = np.asarray(dst_points, dtype=np.float64)

        if src.shape[0] < 4 or dst.shape[0] < 4:
            raise ValueError(
                "At least 4 point correspondences are required for homography."
            )

        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if matrix is None:
            raise ValueError("cv2.findHomography failed — check point quality.")

        self._matrix = matrix

        # Compute inverse for pitch → pixel mapping.
        inverse, inv_mask = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
        self._inverse = inverse

    def set_default_transform(
        self, frame_width: int, frame_height: int
    ) -> None:
        """Set a rough homography by mapping frame corners to pitch corners.

        This provides a *good-enough* transform for demos and testing
        without requiring real calibration points.

        Parameters
        ----------
        frame_width : int
            Width of the video frame in pixels.
        frame_height : int
            Height of the video frame in pixels.
        """
        src_pts = np.array(
            [
                [0, 0],
                [frame_width, 0],
                [frame_width, frame_height],
                [0, frame_height],
            ],
            dtype=np.float64,
        )
        dst_pts = np.array(
            [
                [0, 0],
                [self.pitch_length, 0],
                [self.pitch_length, self.pitch_width],
                [0, self.pitch_width],
            ],
            dtype=np.float64,
        )
        self.set_transform(src_pts, dst_pts)

    @property
    def is_calibrated(self) -> bool:
        """Whether a homography matrix has been set."""
        return self._matrix is not None

    # ------------------------------------------------------------------
    # Point transformations
    # ------------------------------------------------------------------

    def pixel_to_pitch(
        self, point_px: tuple[int, int]
    ) -> tuple[float, float] | None:
        """Transform a pixel coordinate to pitch metres.

        Parameters
        ----------
        point_px : tuple[int, int]
            ``(x, y)`` in pixel space.

        Returns
        -------
        tuple[float, float] | None
            ``(x_m, y_m)`` in pitch metres, or *None* if the transformer
            has not been calibrated.
        """
        if self._matrix is None:
            return None

        try:
            pt = np.array([[[point_px[0], point_px[1]]]], dtype=np.float64)
            transformed = cv2.perspectiveTransform(pt, self._matrix)
            x: float = float(transformed[0, 0, 0])
            y: float = float(transformed[0, 0, 1])
            return (x, y)
        except Exception:
            return None

    def pitch_to_pixel(
        self, point_pitch: tuple[float, float]
    ) -> tuple[int, int] | None:
        """Transform a pitch coordinate back to pixel space.

        Parameters
        ----------
        point_pitch : tuple[float, float]
            ``(x_m, y_m)`` in pitch metres.

        Returns
        -------
        tuple[int, int] | None
            ``(x, y)`` in pixel space, or *None* if no inverse matrix
            is available.
        """
        if self._inverse is None:
            return None

        try:
            pt = np.array(
                [[[point_pitch[0], point_pitch[1]]]], dtype=np.float64
            )
            transformed = cv2.perspectiveTransform(pt, self._inverse)
            x: int = int(round(transformed[0, 0, 0]))
            y: int = int(round(transformed[0, 0, 1]))
            return (x, y)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Batch transformations
    # ------------------------------------------------------------------

    def transform_players(self, players: list) -> list:
        """Set ``pitch_xy`` on each :class:`PlayerState` from its ``foot_px``.

        Parameters
        ----------
        players : list
            List of ``PlayerState`` objects, each having a ``foot_px``
            attribute and a mutable ``pitch_xy`` attribute.

        Returns
        -------
        list
            The same list with ``pitch_xy`` populated where possible.
        """
        for player in players:
            try:
                foot = player.foot_px
                if foot is not None:
                    player.pitch_xy = self.pixel_to_pitch(foot)
            except Exception:
                continue
        return players

    def transform_ball(self, ball: object) -> object:
        """Set ``pitch_xy`` on a :class:`BallState` from its ``center_px``.

        Parameters
        ----------
        ball : object
            A ``BallState``-like object with ``center_px`` and ``pitch_xy``
            attributes.

        Returns
        -------
        object
            The same ball with ``pitch_xy`` populated if possible.
        """
        try:
            center = ball.center_px  # type: ignore[union-attr]
            if center is not None:
                ball.pitch_xy = self.pixel_to_pitch(center)  # type: ignore[union-attr]
        except Exception:
            pass
        return ball
