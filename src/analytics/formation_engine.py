"""Formation detection via spatial clustering.

Uses KMeans on the X-coordinate of player pitch positions to identify
defensive/midfield/attack lines and matches cluster sizes against a
library of known formation templates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src import config


class FormationEngine:
    """Detect team formations from player pitch coordinates.

    Known formation templates are stored as a mapping from a tuple of
    cluster-line sizes (sorted goal-to-goal) to a human-readable string.
    """

    def __init__(self) -> None:
        self._templates: dict[tuple[int, ...], str] = {
            (4, 4, 2): "4-4-2",
            (4, 3, 3): "4-3-3",
            (3, 5, 2): "3-5-2",
            (4, 2, 3, 1): "4-2-3-1",
            (4, 1, 4, 1): "4-1-4-1",
            (3, 4, 3): "3-4-3",
            (5, 3, 2): "5-3-2",
            (5, 4, 1): "5-4-1",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        team_players: list[Any],
    ) -> tuple[str, list[list[tuple[float, float]]]]:
        """Detect the formation for a single team.

        Parameters
        ----------
        team_players : list
            Objects with ``pitch_xy: tuple[float, float] | None`` and
            ``track_id: int``.

        Returns
        -------
        tuple[str, list[list[tuple[float, float]]]]
            ``(formation_string, formation_lines)`` where
            *formation_lines* groups ``(x, y)`` pitch coordinates by
            cluster line (sorted from defensive to attacking).
        """
        # Outfield players only. The goalkeeper is identified by role and
        # used solely to decide which goal this team defends.
        outfield: list[tuple[float, float]] = []
        gk_x: list[float] = []
        for p in team_players:
            xy = getattr(p, "pitch_xy", None)
            if xy is None:
                continue
            role = getattr(p, "role", "player")
            if role == "goalkeeper":
                gk_x.append(float(xy[0]))
            elif role != "referee":
                outfield.append((float(xy[0]), float(xy[1])))

        if not gk_x and outfield:
            # The detector often misses the goalkeeper. Fall back to the
            # player closest to either goal line and take them out of the
            # outfield count so they are not a formation line.
            nearest = min(
                range(len(outfield)),
                key=lambda i: min(outfield[i][0], config.PITCH_LENGTH - outfield[i][0]),
            )
            gk_x = [outfield[nearest][0]]
            del outfield[nearest]

        if len(outfield) < config.FORMATION_MIN_OUTFIELD or not gk_x:
            return ("—", [])

        # The goalkeeper stands in front of their own goal. x = 0 is the
        # left goal line on this pitch.
        defend_left = float(np.mean(gk_x)) < (config.PITCH_LENGTH / 2.0)

        outfield_coords = np.array(outfield, dtype=np.float64)
        x_vals = outfield_coords[:, 0].reshape(-1, 1)

        # Try k in 2..5 and keep the clustering with best silhouette.
        best_score: float = -1.0
        best_labels: np.ndarray | None = None
        best_k: int = 2

        for k in range(2, min(6, len(outfield_coords))):
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            labels = km.fit_predict(x_vals)
            if len(set(labels)) < 2:
                continue
            score = float(silhouette_score(x_vals, labels))
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k

        if best_labels is None:
            return ("—", [])

        # Sort clusters from this team's own goal toward the other goal.
        cluster_means: list[tuple[int, float]] = []
        for cid in range(best_k):
            mask = best_labels == cid
            cluster_means.append((cid, float(outfield_coords[mask, 0].mean())))
        cluster_means.sort(key=lambda t: t[1], reverse=not defend_left)

        cluster_sizes: list[int] = []
        formation_lines: list[list[tuple[float, float]]] = []
        for cid, _ in cluster_means:
            mask = best_labels == cid
            points = outfield_coords[mask].tolist()
            formation_lines.append([(p[0], p[1]) for p in points])
            cluster_sizes.append(int(mask.sum()))

        formation_str = self._match_template(tuple(cluster_sizes))
        return formation_str, formation_lines

    def detect_both_teams(self, players: list[Any]) -> Any:
        """Separate *players* by team, detect formations, and return a
        ``PitchState`` dataclass.

        Parameters
        ----------
        players : list
            All tracked players (must expose ``team`` attribute).

        Returns
        -------
        PitchState
            Populated with ``formation_a``, ``formation_b``,
            ``formation_lines_a`` and ``formation_lines_b``.
        """
        from src.core.packet import PitchState

        team_a = [p for p in players if getattr(p, "team", None) == "A"]
        team_b = [p for p in players if getattr(p, "team", None) == "B"]

        form_a, lines_a = self.detect(team_a)
        form_b, lines_b = self.detect(team_b)

        return PitchState(
            formation_a=form_a,
            formation_b=form_b,
            formation_lines_a=lines_a,
            formation_lines_b=lines_b,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match_template(self, cluster_sizes: tuple[int, ...]) -> str:
        """Match *cluster_sizes* to a known formation template.

        Parameters
        ----------
        cluster_sizes : tuple[int, ...]
            Number of players in each cluster line, ordered from
            defensive to attacking.

        Returns
        -------
        str
            The matched formation string, e.g. ``"4-3-3"``, or the
            sizes joined by dashes if no exact match is found.
        """
        if cluster_sizes in self._templates:
            return self._templates[cluster_sizes]
        # Fallback: join the numbers.
        return "-".join(str(s) for s in cluster_sizes)
