"""Keep one identity per player on the field.

ByteTrack starts a new number whenever someone leaves the frame and comes
back. That splits one person's heatmap across many ids. This roster
remembers up to 11 players a side and reattaches a returning player to
the shirt pattern and the place they were last seen.

Jersey numbers are not read as digits. At broadcast size the shirt is
only a few dozen pixels tall, so the cue is the pattern of dark marks on
the shirt (a number, a sponsor, a stripe) together with where that player
was standing.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.config import MAX_PLAYERS_PER_TEAM

# A returning player farther than this from their last spot is treated as
# someone else, unless the shirt pattern is a very close match.
_FAR_M = 18.0
# Cosine distance below which two shirt patterns are the same player.
_SAME_SHIRT = 0.22


@dataclass
class _Slot:
    stable_id: int
    team: str
    appearance: np.ndarray | None = None
    pitch_xy: tuple[float, float] | None = None
    on_screen: bool = False


class PlayerRoster:
    """Map short-lived tracker ids onto at most 22 stable player ids."""

    def __init__(self) -> None:
        self._slots: list[_Slot] = []
        self._next_number: dict[str, int] = {"A": 1, "B": 1}
        self._bindings: dict[int, _Slot] = {}

    def begin_segment(self) -> None:
        """Forget tracker numbers after a cut. The 22 players stay."""
        self._bindings.clear()
        for slot in self._slots:
            slot.on_screen = False

    def assign(self, players: list) -> None:
        """Rewrite ``track_id`` on *players* to a stable roster id.

        Referees are left unchanged. A detection that cannot be one of
        the 22 gets ``track_id`` -1 and should be left out of the heatmap.
        """
        claimed: set[int] = set()
        for player in players:
            if _is_referee(player) or getattr(player, "offscreen", False):
                continue
            raw_id = int(player.track_id)
            slot = self._bindings.get(raw_id)
            if slot is None or id(slot) in claimed:
                slot = self._match_or_create(player)
                if slot is None:
                    player.track_id = -1
                    continue
                self._bindings[raw_id] = slot
            self._touch(slot, player)
            claimed.add(id(slot))
            player.track_id = slot.stable_id

        for slot in self._slots:
            if id(slot) not in claimed:
                slot.on_screen = False

        for player in players:
            if not getattr(player, "offscreen", False) or _is_referee(player):
                continue
            slot = self._bindings.get(int(player.track_id))
            if slot is not None:
                player.track_id = slot.stable_id

    def _match_or_create(self, player) -> _Slot | None:
        team = getattr(player, "team", "unknown")
        if team not in ("A", "B"):
            team = self._guess_team(player)
        if team not in ("A", "B"):
            return None

        feature = _unit(getattr(player, "appearance", None))
        best: _Slot | None = None
        best_score = 1e9
        for slot in self._slots:
            if slot.on_screen or slot.team != team:
                continue
            score = _score(feature, player, slot)
            if score < best_score:
                best, best_score = slot, score

        count = sum(1 for slot in self._slots if slot.team == team)
        if best is not None and (best_score <= _SAME_SHIRT or count >= MAX_PLAYERS_PER_TEAM):
            return best
        if count >= MAX_PLAYERS_PER_TEAM:
            return best
        return self._create(team, feature, getattr(player, "pitch_xy", None))

    def _guess_team(self, player) -> str | None:
        """Use the nearest remembered teammate when this frame has no team yet."""
        here = getattr(player, "pitch_xy", None)
        feature = _unit(getattr(player, "appearance", None))
        if here is None and feature is None:
            return None
        best_team: str | None = None
        best_score = 1e9
        for slot in self._slots:
            score = _score(feature, player, slot)
            if score < best_score:
                best_team, best_score = slot.team, score
        if best_team is not None and best_score <= _SAME_SHIRT:
            return best_team
        return None

    def _create(
        self,
        team: str,
        feature: np.ndarray | None,
        pitch_xy: tuple[float, float] | None,
    ) -> _Slot:
        number = self._next_number[team]
        self._next_number[team] += 1
        stable_id = number if team == "A" else 11 + number
        slot = _Slot(
            stable_id=stable_id,
            team=team,
            appearance=None if feature is None else feature.copy(),
            pitch_xy=pitch_xy,
            on_screen=True,
        )
        self._slots.append(slot)
        return slot

    def _touch(self, slot: _Slot, player) -> None:
        slot.on_screen = True
        feature = _unit(getattr(player, "appearance", None))
        if feature is not None:
            if slot.appearance is None:
                slot.appearance = feature.copy()
            else:
                mixed = 0.85 * slot.appearance + 0.15 * feature
                slot.appearance = _unit(mixed)
        pitch_xy = getattr(player, "pitch_xy", None)
        if pitch_xy is not None:
            slot.pitch_xy = (float(pitch_xy[0]), float(pitch_xy[1]))
        team = getattr(player, "team", None)
        if team in ("A", "B"):
            slot.team = team


def jersey_appearance(frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
    """Shirt pattern used to recognise a player who has left the frame.

    The vector is the colour of a 4 × 4 grid on the torso, plus how dark
    each cell is. A printed number moves those dark cells. The result is
    unit length so two crops can be compared with a cosine distance.
    """
    crop = _torso_crop(frame, bbox)
    if crop is None:
        return None
    small = cv2.resize(crop, (16, 16), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    parts: list[np.ndarray] = []
    darkness: list[float] = []
    for row in range(4):
        for col in range(4):
            block = lab[row * 4 : (row + 1) * 4, col * 4 : (col + 1) * 4]
            parts.append(block.reshape(-1, 3).mean(axis=0))
            darkness.append(float((block[:, :, 0] < 90).mean()))
    vector = np.concatenate([*parts, np.asarray(darkness, dtype=np.float32)])
    return _unit(vector)


def _torso_crop(frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
    x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    height, width = y2 - y1, x2 - x1
    if height < 8 or width < 8:
        return None
    crop_y1 = y1 + int(height * 0.15)
    crop_y2 = y1 + int(height * 0.55)
    margin = int(width * 0.25)
    crop_x1 = x1 + margin
    crop_x2 = x2 - margin
    frame_h, frame_w = frame.shape[:2]
    crop_y1 = max(0, min(frame_h - 1, crop_y1))
    crop_y2 = max(0, min(frame_h, crop_y2))
    crop_x1 = max(0, min(frame_w - 1, crop_x1))
    crop_x2 = max(0, min(frame_w, crop_x2))
    if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
        return None
    return frame[crop_y1:crop_y2, crop_x1:crop_x2]


def _unit(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm < 1e-6:
        return None
    return array / norm


def _score(feature: np.ndarray | None, player, slot: _Slot) -> float:
    if feature is None or slot.appearance is None:
        appearance = 0.35
    else:
        appearance = float(1.0 - np.dot(feature, slot.appearance))
    here = getattr(player, "pitch_xy", None)
    if here is not None and slot.pitch_xy is not None:
        distance = float(np.hypot(here[0] - slot.pitch_xy[0], here[1] - slot.pitch_xy[1]))
        if distance > _FAR_M:
            appearance += 0.45
        elif distance > 8.0:
            appearance += 0.12
    return appearance


def _is_referee(player) -> bool:
    return getattr(player, "role", "") == "referee" or getattr(player, "team", "") == "ref"
