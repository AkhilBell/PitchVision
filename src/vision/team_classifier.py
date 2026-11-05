"""Jersey-colour K-Means team classifier.

Assigns each detected player to *Team A*, *Team B*, or *Referee* by
clustering the dominant jersey colour extracted from the upper-middle
region of their bounding box.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans


class TeamClassifier:
    """Cluster players into two teams and a referee group using jersey colour.

    The classifier operates in two phases:

    1. **fit** — collect dominant colours from all detections in a
       representative frame and cluster them into *n_clusters* groups.
       The two largest groups become Team A / B and the smallest becomes
       the referee cluster.
    2. **classify** — assign an individual bounding box to a team label
       (``"A"``, ``"B"``, or ``"ref"``) using the fitted model.

    Parameters
    ----------
    n_clusters : int
        Number of colour clusters (default ``3``: two teams + referee).
    """

    def __init__(self, n_clusters: int = 3) -> None:
        self.n_clusters: int = n_clusters
        self._kmeans: KMeans | None = None
        self._cluster_to_team: dict[int, str] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, frame: np.ndarray, detections: sv.Detections) -> None:
        """Fit the colour model from a set of player detections.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H × W × 3).
        detections : sv.Detections
            Player detections whose bounding boxes will be used.
        """
        dominant_colours: list[np.ndarray] = []

        for bbox in detections.xyxy:
            try:
                crop = self._extract_jersey_crop(frame, tuple(bbox.astype(int)))
                if crop.size == 0:
                    continue
                colour = self._get_dominant_color(crop)
                dominant_colours.append(colour)
            except Exception:
                continue

        if len(dominant_colours) < self.n_clusters:
            # Not enough data to fit — leave unfitted.
            return

        colour_array: np.ndarray = np.array(dominant_colours)

        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
        )
        labels: np.ndarray = self._kmeans.fit_predict(colour_array)

        # Count members of each cluster and map to team labels.
        cluster_sizes: dict[int, int] = {}
        for label in labels:
            cluster_sizes[label] = cluster_sizes.get(int(label), 0) + 1

        # Sort clusters by size (descending).
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)

        self._cluster_to_team = {}
        for rank, (cluster_id, _) in enumerate(sorted_clusters):
            if rank == 0:
                self._cluster_to_team[cluster_id] = "A"
            elif rank == 1:
                self._cluster_to_team[cluster_id] = "B"
            else:
                self._cluster_to_team[cluster_id] = "ref"

        self._fitted = True

    def classify(self, frame: np.ndarray, bbox: tuple) -> str:
        """Classify a single player as Team A, Team B, or referee.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H × W × 3).
        bbox : tuple
            Bounding box ``(x1, y1, x2, y2)`` in pixel coordinates.

        Returns
        -------
        str
            ``"A"``, ``"B"``, ``"ref"``, or ``"unknown"`` if the model
            has not been fitted.
        """
        if not self._fitted or self._kmeans is None:
            return "unknown"

        try:
            crop = self._extract_jersey_crop(frame, bbox)
            if crop.size == 0:
                return "unknown"
            colour = self._get_dominant_color(crop)
            cluster: int = int(self._kmeans.predict(colour.reshape(1, -1))[0])
            return self._cluster_to_team.get(cluster, "unknown")
        except Exception:
            return "unknown"

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called successfully."""
        return self._fitted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_jersey_crop(frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """Extract the jersey region from a bounding box.

        The jersey region is defined as the upper 40 % of the bounding
        box height and the middle 60 % of the width, which is where
        the jersey colour is most consistent (avoiding head / shorts).

        Parameters
        ----------
        frame : np.ndarray
            BGR image.
        bbox : tuple
            ``(x1, y1, x2, y2)`` integer pixel coordinates.

        Returns
        -------
        np.ndarray
            Cropped image in CIE-LAB colour space.
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        h: int = y2 - y1
        w: int = x2 - x1

        if h <= 0 or w <= 0:
            return np.array([])

        # Upper 40 % height
        crop_y1: int = y1
        crop_y2: int = y1 + int(h * 0.4)

        # Middle 60 % width
        margin: int = int(w * 0.2)
        crop_x1: int = x1 + margin
        crop_x2: int = x2 - margin

        # Clamp to frame bounds
        fh, fw = frame.shape[:2]
        crop_y1 = max(0, crop_y1)
        crop_y2 = min(fh, crop_y2)
        crop_x1 = max(0, crop_x1)
        crop_x2 = min(fw, crop_x2)

        if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
            return np.array([])

        crop_bgr: np.ndarray = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_lab: np.ndarray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        return crop_lab

    @staticmethod
    def _get_dominant_color(crop: np.ndarray) -> np.ndarray:
        """Determine the dominant colour of a crop via 2-cluster KMeans.

        Parameters
        ----------
        crop : np.ndarray
            Image crop (H × W × 3) in any colour space.

        Returns
        -------
        np.ndarray
            1-D array of length 3 representing the dominant colour
            (the centre of the largest cluster).
        """
        pixels: np.ndarray = crop.reshape(-1, 3).astype(np.float32)

        km = KMeans(n_clusters=2, random_state=0, n_init=5)
        km.fit(pixels)

        # Pick the cluster with more members.
        labels: np.ndarray = km.labels_
        count_0: int = int(np.sum(labels == 0))
        count_1: int = int(np.sum(labels == 1))

        dominant_idx: int = 0 if count_0 >= count_1 else 1
        return km.cluster_centers_[dominant_idx].astype(np.float64)
