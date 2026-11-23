"""Jersey-colour team classifier with per-track majority voting.

Fits a two-cluster colour model on outfield players only (referees and
goalkeepers are assigned elsewhere), using a tight torso crop and
grass-pixel filtering so the dominant colour is the shirt, not the pitch.
Once fitted, each track ID accumulates colour votes and settles on Team A
or Team B after a few observations.
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans

from src import config


class TeamClassifier:
    """Assign outfield players to Team A or Team B by jersey colour.

    Workflow
    --------
    1. :meth:`fit` — once, on a frame with enough outfield players.
    2. :meth:`observe` — each frame, for each outfield track; returns the
       current majority team and locks it after
       :pydata:`config.TEAM_VOTE_SETTLE` votes.
    3. :meth:`clear_tracks` — drop per-track votes after a hard cut (the
       colour model itself is kept).
    """

    def __init__(self) -> None:
        self._kmeans: KMeans | None = None
        self._cluster_to_team: dict[int, str] = {}
        self._fitted: bool = False
        self._votes: dict[int, Counter[str]] = {}
        self._settled: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        roles: list[str] | None = None,
    ) -> bool:
        """Fit a 2-team colour model from outfield player detections.

        Goalkeepers and referees are skipped.  Returns ``True`` when the
        model was fitted successfully.
        """
        features: list[np.ndarray] = []
        for i, bbox in enumerate(detections.xyxy):
            if roles is not None and roles[i] != "player":
                continue
            feature = self._jersey_feature(frame, tuple(int(v) for v in bbox))
            if feature is not None:
                features.append(feature)

        if len(features) < config.TEAM_FIT_MIN_PLAYERS:
            return False

        colour_array = np.asarray(features, dtype=np.float64)
        self._kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = self._kmeans.fit_predict(colour_array)

        # Larger cluster → Team A (stable label across runs; size is only
        # a tie-break, not a semantic "home" assignment).
        counts = Counter(int(label) for label in labels)
        ordered = [cid for cid, _ in counts.most_common()]
        self._cluster_to_team = {
            ordered[0]: "A",
            ordered[1]: "B",
        }
        self._fitted = True
        return True

    def observe(self, track_id: int, frame: np.ndarray, bbox: tuple) -> str:
        """Classify *bbox* and fold the result into this track's vote.

        After enough votes the team is locked and colour extraction is
        skipped on later frames (cheaper and stable).
        """
        settled = self._settled.get(track_id)
        if settled is not None:
            return settled

        label = self.classify(frame, bbox)
        if label not in ("A", "B"):
            votes = self._votes.get(track_id)
            return votes.most_common(1)[0][0] if votes else "unknown"

        votes = self._votes.setdefault(track_id, Counter())
        votes[label] += 1
        team = votes.most_common(1)[0][0]
        if sum(votes.values()) >= config.TEAM_VOTE_SETTLE:
            self._settled[track_id] = team
        return team

    def classify(self, frame: np.ndarray, bbox: tuple) -> str:
        """One-shot colour classification without updating track votes."""
        if not self._fitted or self._kmeans is None:
            return "unknown"
        feature = self._jersey_feature(frame, bbox)
        if feature is None:
            return "unknown"
        try:
            cluster = int(self._kmeans.predict(feature.reshape(1, -1))[0])
        except Exception:
            return "unknown"
        return self._cluster_to_team.get(cluster, "unknown")

    def clear_tracks(self) -> None:
        """Drop per-track votes after a scene cut (IDs are reused)."""
        self._votes.clear()
        self._settled.clear()

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has succeeded."""
        return self._fitted

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _jersey_feature(
        self, frame: np.ndarray, bbox: tuple
    ) -> np.ndarray | None:
        """Return a LAB colour feature for the shirt, or ``None``."""
        crop = self._extract_jersey_crop(frame, bbox)
        if crop is None:
            return None
        return self._colour_feature(crop)

    @staticmethod
    def _extract_jersey_crop(
        frame: np.ndarray, bbox: tuple
    ) -> np.ndarray | None:
        """Tight torso crop: mid-height band, middle half of the box width.

        The old "upper 40 %" crop pulled in grass around the shoulders at
        broadcast resolution; this band sits on the shirt body instead.
        """
        x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        h = y2 - y1
        w = x2 - x1
        if h < 8 or w < 8:
            return None

        crop_y1 = y1 + int(h * 0.15)
        crop_y2 = y1 + int(h * 0.55)
        margin = int(w * 0.25)
        crop_x1 = x1 + margin
        crop_x2 = x2 - margin

        fh, fw = frame.shape[:2]
        crop_y1 = max(0, crop_y1)
        crop_y2 = min(fh, crop_y2)
        crop_x1 = max(0, crop_x1)
        crop_x2 = min(fw, crop_x2)
        if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
            return None
        return frame[crop_y1:crop_y2, crop_x1:crop_x2]

    @staticmethod
    def _colour_feature(crop_bgr: np.ndarray) -> np.ndarray | None:
        """Median LAB of non-grass pixels in *crop_bgr*.

        Grass is detected in HSV (green hue, moderate saturation).  If too
        few non-grass pixels remain, the median of the whole crop is used.
        """
        if crop_bgr.size == 0:
            return None

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        grass = (h >= 35) & (h <= 95) & (s >= 40) & (v >= 40) & (v <= 220)

        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        mask = (~grass).reshape(-1)
        kept = lab[mask]
        if len(kept) < max(10, int(0.15 * len(lab))):
            kept = lab
        if len(kept) == 0:
            return None
        return np.median(kept, axis=0).astype(np.float64)
