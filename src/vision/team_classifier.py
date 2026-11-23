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
        self._centers: np.ndarray | None = None
        self._cluster_to_team: dict[int, str] = {}
        self._fitted: bool = False
        self._samples: list[np.ndarray] = []
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
        """Add this frame's jersey colours and fit once the kits separate.

        Samples accumulate across frames so a shot of mostly one team does
        not define both colours. The fit is kept only when the two colours
        are far apart and both clusters are large enough (a white kit and
        a blue kit, for example, rather than two shades of the same shirt).
        """
        if self._fitted:
            return True

        for i, bbox in enumerate(detections.xyxy):
            if roles is not None and roles[i] != "player":
                continue
            feature = self._jersey_feature(frame, tuple(int(v) for v in bbox))
            if feature is not None:
                self._samples.append(feature)

        if len(self._samples) > 80:
            self._samples = self._samples[-80:]
        if len(self._samples) < config.TEAM_FIT_MIN_PLAYERS:
            return False

        colour_array = np.asarray(self._samples, dtype=np.float64)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(colour_array)
        counts = Counter(int(label) for label in labels)
        smaller = min(counts.values())
        if smaller / len(labels) < config.TEAM_MIN_CLUSTER_FRACTION:
            return False
        centers = kmeans.cluster_centers_
        if float(np.linalg.norm(centers[0] - centers[1])) < config.TEAM_MIN_CENTER_DISTANCE:
            return False

        ordered = [cid for cid, _ in counts.most_common()]
        self._kmeans = kmeans
        self._centers = centers
        self._cluster_to_team = {ordered[0]: "A", ordered[1]: "B"}
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
        if feature is None or self._centers is None:
            return "unknown"
        distances = np.linalg.norm(self._centers - feature, axis=1)
        order = np.argsort(distances)
        nearest, second = int(order[0]), int(order[1])
        # A white shirt with a blue number, or the reverse, lands between
        # the two kits. Skip it instead of voting for the wrong team.
        if distances[second] < distances[nearest] * 1.35:
            return "unknown"
        return self._cluster_to_team.get(nearest, "unknown")

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
        # Green only. A wider band used to swallow blue and teal shirts,
        # so a blue kit and a white kit both collapsed to skin tone.
        grass = (h >= 35) & (h <= 85) & (s >= 40) & (v >= 40)
        # White kit: bright and nearly unsaturated. Scaled so it can
        # separate white from blue, which are close in raw LAB.
        white = (s <= 45) & (v >= 150)

        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        flat_white = white.reshape(-1)
        mask = (~grass).reshape(-1)
        kept = lab[mask]
        kept_white = flat_white[mask]
        if len(kept) < max(10, int(0.15 * len(lab))):
            kept = lab
            kept_white = flat_white
        if len(kept) == 0:
            return None
        median = np.median(kept, axis=0)
        white_score = float(kept_white.mean()) * 100.0
        return np.array([median[0], median[1], median[2], white_score], dtype=np.float64)
