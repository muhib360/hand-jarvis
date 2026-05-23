"""Gesture hysteresis and brief dropout tolerance."""

from __future__ import annotations

from dataclasses import dataclass

from config_store import PointerSettings
from gestures import Gesture


@dataclass
class GestureStateMachine:
    settings: PointerSettings
    effective: Gesture = Gesture.NONE
    _point_streak: int = 0
    _release_streak: int = 0
    _dropout_left: int = 0
    _last_landmarks: list | None = None

    def update(self, raw: Gesture, landmarks_present: bool, landmarks=None) -> Gesture:
        if landmarks_present and landmarks is not None:
            self._last_landmarks = landmarks
            self._dropout_left = self.settings.dropout_frames

        if raw == Gesture.POINT:
            self._point_streak += 1
            self._release_streak = 0
        else:
            self._point_streak = 0
            self._release_streak += 1

        if self._point_streak >= self.settings.gesture_point_hold_frames:
            self.effective = Gesture.POINT
            return self.effective

        if self.effective == Gesture.POINT:
            if raw == Gesture.POINT:
                return self.effective
            if not landmarks_present and self._dropout_left > 0:
                self._dropout_left -= 1
                return self.effective
            if self._release_streak < self.settings.gesture_point_release_frames:
                return self.effective
            self.effective = raw if landmarks_present else Gesture.NONE
            return self.effective

        self.effective = raw if landmarks_present else Gesture.NONE
        return self.effective

    def landmarks_for_control(self, landmarks_present: bool, landmarks):
        if landmarks_present:
            return landmarks
        if self.effective == Gesture.POINT and self._dropout_left > 0 and self._last_landmarks:
            return self._last_landmarks
        return None

    def reset(self) -> None:
        self.effective = Gesture.NONE
        self._point_streak = 0
        self._release_streak = 0
        self._dropout_left = 0
        self._last_landmarks = None
