"""Standard pitch markings, shared by the radar and the heatmap.

Coordinates are metres: x runs from the left goal line to the right goal
line (0 to the pitch length), y runs from one touchline to the other
(0 to the pitch width). Each marking is a polyline so both ends of the
pitch get the same lines, including the penalty-area line that sits
16.5 m out and the arc of the penalty "D".
"""

from __future__ import annotations

import math

import numpy as np

from src.config import PITCH_LENGTH, PITCH_WIDTH

_PENALTY_DEPTH = 16.5
_PENALTY_HALF_WIDTH = 20.16
_GOAL_DEPTH = 5.5
_GOAL_HALF_WIDTH = 9.16
_SPOT_DISTANCE = 11.0
_ARC_RADIUS = 9.15
_CENTRE_RADIUS = 9.15
_CORNER_RADIUS = 1.0


_MARKINGS: list[np.ndarray] | None = None


def pitch_markings() -> list[np.ndarray]:
    """Return every pitch marking as an ``(N, 2)`` polyline in metres."""
    global _MARKINGS
    if _MARKINGS is None:
        _MARKINGS = _build_markings()
    return _MARKINGS


def _build_markings() -> list[np.ndarray]:
    length = float(PITCH_LENGTH)
    width = float(PITCH_WIDTH)
    mid = width / 2.0
    lines: list[np.ndarray] = [
        _closed((0.0, 0.0), (length, 0.0), (length, width), (0.0, width)),
        np.array([[length / 2.0, 0.0], [length / 2.0, width]], dtype=np.float64),
        _circle(length / 2.0, mid, _CENTRE_RADIUS),
        _circle(length / 2.0, mid, 0.35, samples=12),
    ]
    for goal_x, direction in ((0.0, 1.0), (length, -1.0)):
        lines.append(_box(goal_x, direction, _PENALTY_DEPTH, _PENALTY_HALF_WIDTH, mid))
        lines.append(_box(goal_x, direction, _GOAL_DEPTH, _GOAL_HALF_WIDTH, mid))
        spot_x = goal_x + direction * _SPOT_DISTANCE
        lines.append(_circle(spot_x, mid, 0.3, samples=12))
        lines.append(_penalty_arc(spot_x, mid, direction))
    lines.extend(_corner_arcs(length, width))
    return lines


def _closed(*points: tuple[float, float]) -> np.ndarray:
    """A polyline that returns to its first point, so every side is drawn."""
    coords = list(points) + [points[0]]
    return np.asarray(coords, dtype=np.float64)


def _box(
    goal_x: float, direction: float, depth: float, half_width: float, mid: float
) -> np.ndarray:
    """Penalty or goal area. *direction* points from the goal into the pitch."""
    outer = goal_x + direction * depth
    return _closed(
        (goal_x, mid - half_width),
        (outer, mid - half_width),
        (outer, mid + half_width),
        (goal_x, mid + half_width),
    )


def _circle(cx: float, cy: float, radius: float, samples: int = 64) -> np.ndarray:
    """A full circle as a closed polyline."""
    theta = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=True)
    return np.column_stack((cx + radius * np.cos(theta), cy + radius * np.sin(theta)))


def _penalty_arc(spot_x: float, mid: float, direction: float) -> np.ndarray:
    """The part of the 9.15 m circle that sits outside the penalty area.

    That arc is the "D". It meets the penalty line (5.5 m beyond the
    penalty spot) and bows toward the halfway line.
    """
    edge = _PENALTY_DEPTH - _SPOT_DISTANCE
    half_angle = math.acos(edge / _ARC_RADIUS)
    theta = np.linspace(-half_angle, half_angle, 32)
    return np.column_stack(
        (
            spot_x + direction * _ARC_RADIUS * np.cos(theta),
            mid + _ARC_RADIUS * np.sin(theta),
        )
    )


def _corner_arcs(length: float, width: float) -> list[np.ndarray]:
    """Quarter-circles of the corner arcs, inside the pitch."""
    radius = _CORNER_RADIUS
    quarters = (
        (0.0, 0.0, 0.0, math.pi / 2.0),
        (0.0, width, -math.pi / 2.0, 0.0),
        (length, 0.0, math.pi / 2.0, math.pi),
        (length, width, math.pi, 3.0 * math.pi / 2.0),
    )
    arcs: list[np.ndarray] = []
    for cx, cy, start, end in quarters:
        theta = np.linspace(start, end, 12)
        arcs.append(
            np.column_stack((cx + radius * np.cos(theta), cy + radius * np.sin(theta)))
        )
    return arcs
