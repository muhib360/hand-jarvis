"""Gesture detection using calibrated thresholds."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

from config_store import AppConfig, GestureCalibration

# Landmark visibility — below this, finger is treated as occluded / off-camera
VISIBILITY_MIN = 0.45

FINGER_CHAINS = (
    (8, 6),   # index
    (12, 10),  # middle
    (16, 14),  # ring
    (20, 18),  # pinky
)


class Gesture(Enum):
    NONE = auto()
    OPEN_PALM = auto()
    PINCH = auto()
    FIST = auto()
    POINT = auto()
    SWIPE_LEFT = auto()
    SWIPE_RIGHT = auto()


def _dist(a, b) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def hand_scale(landmarks) -> float:
    """Palm size proxy — wrist to middle MCP (distance from camera)."""
    return _dist(landmarks[0], landmarks[9])


def landmark_visible(lm) -> bool:
    if lm.visibility is not None:
        return lm.visibility >= VISIBILITY_MIN
    if lm.presence is not None:
        return lm.presence >= VISIBILITY_MIN
    return True


def finger_pair_visible(landmarks, tip: int, pip: int) -> bool:
    return landmark_visible(landmarks[tip]) and landmark_visible(landmarks[pip])


def finger_extended(landmarks, tip: int, pip: int, margin: float) -> bool:
    if not finger_pair_visible(landmarks, tip, pip):
        return False
    return landmarks[tip].y < landmarks[pip].y - margin


def is_pinch(landmarks, g: GestureCalibration) -> bool:
    return _dist(landmarks[4], landmarks[8]) < g.pinch_threshold


def is_index_extended(landmarks, g: GestureCalibration) -> bool:
    return finger_extended(landmarks, 8, 6, g.index_extend_margin)


def count_visible_other_fingers(landmarks) -> int:
    """Middle, ring, pinky — how many are visible to the camera."""
    return sum(
        1
        for tip, pip in FINGER_CHAINS[1:]
        if finger_pair_visible(landmarks, tip, pip)
    )


def count_extended_other_fingers(landmarks, g: GestureCalibration) -> int:
    n = 0
    for tip, pip in FINGER_CHAINS[1:]:
        if finger_extended(landmarks, tip, pip, g.index_extend_margin):
            n += 1
    return n


def is_open_palm(landmarks, g: GestureCalibration) -> bool:
    """All four fingers visible and extended — not when only index shows."""
    if is_pinch(landmarks, g):
        return False
    visible = 0
    extended = 0
    for tip, pip in FINGER_CHAINS:
        if finger_pair_visible(landmarks, tip, pip):
            visible += 1
            if finger_extended(landmarks, tip, pip, g.index_extend_margin):
                extended += 1
    return visible >= 4 and extended >= 4


def is_clutch_pose(landmarks, g: GestureCalibration) -> bool:
    """Thumb tucked while index extended — freeze cursor to reposition hand."""
    if not is_index_extended(landmarks, g):
        return False
    if is_pinch(landmarks, g):
        return False
    thumb_tip, thumb_ip = landmarks[4], landmarks[3]
    thumb_tucked = thumb_tip.y > thumb_ip.y + 0.02
    if thumb_tucked:
        return True
    return _dist(thumb_tip, landmarks[0]) < 0.11


def is_precision_pose(landmarks, g: GestureCalibration) -> bool:
    """Pinky extended while pointing — slow cursor gain."""
    if not is_index_extended(landmarks, g):
        return False
    return finger_extended(landmarks, 20, 18, g.index_extend_margin)


def is_teleport_pose(landmarks, g: GestureCalibration) -> bool:
    """Index + middle extended — jump cursor to aim position."""
    if not is_index_extended(landmarks, g):
        return False
    return finger_extended(landmarks, 12, 10, g.index_extend_margin)


def is_fist(landmarks, g: GestureCalibration) -> bool:
    if is_pinch(landmarks, g) or is_clutch_pose(landmarks, g):
        return False
    if is_index_extended(landmarks, g):
        return False
    wrist = landmarks[0]
    visible_tips = [
        landmarks[i]
        for i in (8, 12, 16, 20)
        if landmark_visible(landmarks[i]) and landmark_visible(wrist)
    ]
    if len(visible_tips) < 3:
        return False
    fist_avg = float(np.mean([_dist(t, wrist) for t in visible_tips]))
    return fist_avg < g.fist_tip_wrist_max


def is_pointing(landmarks, g: GestureCalibration) -> bool:
    """
    Mouse / index control pose.

    Works when only the index is visible (other fingers blocked by camera angle).
    Uses strict curled-finger check only when middle/ring/pinky are actually visible.
    """
    if not is_index_extended(landmarks, g):
        return False
    if is_pinch(landmarks, g):
        return False
    if is_open_palm(landmarks, g):
        return False
    if is_fist(landmarks, g):
        return False

    visible_others = count_visible_other_fingers(landmarks)
    extended_others = count_extended_other_fingers(landmarks, g)

    # Index-only: other fingers off-camera or occluded — trust extended index
    if visible_others == 0:
        return True
    if visible_others == 1 and extended_others == 0:
        return True

    # Some fingers visible: reject only if a visible finger is clearly extended (open hand)
    if extended_others >= 2:
        return False
    if extended_others == 1 and visible_others >= 2:
        return False

    # Visible others must not be extended (curled / hidden fist)
    for tip, pip in FINGER_CHAINS[1:]:
        if finger_pair_visible(landmarks, tip, pip):
            if finger_extended(landmarks, tip, pip, g.index_extend_margin):
                return False
    return True


# Back-compat alias
is_pointing_for_mouse = is_pointing


def detect_raw_gesture(landmarks, config: AppConfig) -> Gesture:
    g = config.gestures

    if is_pinch(landmarks, g):
        return Gesture.PINCH
    if is_fist(landmarks, g):
        return Gesture.FIST
    if is_open_palm(landmarks, g):
        return Gesture.OPEN_PALM
    if is_pointing(landmarks, g):
        return Gesture.POINT
    return Gesture.NONE
