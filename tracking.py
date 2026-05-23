"""MediaPipe hand tracking wrapper with landmark smoothing."""

from __future__ import annotations

import os
import types
import urllib.request
from collections import deque
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS


def ensure_model() -> Path:
    if MODEL_PATH.is_file():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")
    return MODEL_PATH


def flip_frame(frame):
    return cv2.flip(frame, 1)


def _clone_landmark(lm):
    return types.SimpleNamespace(
        x=lm.x,
        y=lm.y,
        z=lm.z,
        visibility=getattr(lm, "visibility", None),
        presence=getattr(lm, "presence", None),
    )


class LandmarkSmoother:
    """Rolling median filter per landmark coordinate."""

    def __init__(self, window: int = 5) -> None:
        self.window = max(1, window)
        self._buf: deque[list] = deque(maxlen=self.window)

    def apply(self, landmarks) -> list:
        self._buf.append(landmarks)
        if len(self._buf) < 2:
            return landmarks
        n = len(landmarks)
        out = []
        for i in range(n):
            xs = [frame[i].x for frame in self._buf]
            ys = [frame[i].y for frame in self._buf]
            zs = [frame[i].z for frame in self._buf]
            base = landmarks[i]
            out.append(
                types.SimpleNamespace(
                    x=float(np.median(xs)),
                    y=float(np.median(ys)),
                    z=float(np.median(zs)),
                    visibility=getattr(base, "visibility", None),
                    presence=getattr(base, "presence", None),
                )
            )
        return out

    def clear(self) -> None:
        self._buf.clear()


class HandTracker:
    def __init__(self, smooth_window: int = 5) -> None:
        model_path = ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._frame_idx = 0
        self._fps = 30.0
        self._smoother = LandmarkSmoother(smooth_window)
        self._last_raw: list | None = None

    def set_fps(self, fps: float) -> None:
        self._fps = fps or 30.0

    def set_smooth_window(self, window: int) -> None:
        self._smoother = LandmarkSmoother(window)

    def detect(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((self._frame_idx / self._fps) * 1000)
        self._frame_idx += 1
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.hand_landmarks:
            raw = [_clone_landmark(lm) for lm in result.hand_landmarks[0]]
            self._last_raw = raw
            return self._smoother.apply(raw)
        return None

    def draw_skeleton(self, frame, landmarks) -> None:
        mp_drawing.draw_landmarks(frame, landmarks, HAND_CONNECTIONS)

    def clear_smooth(self) -> None:
        self._smoother.clear()
        self._last_raw = None

    def close(self) -> None:
        self._landmarker.close()
